"""
Memory forensics: detailed actual vs predicted breakdown.

Snapshots GPU memory at each phase (static -> fwd -> loss -> bwd) and
prints a side-by-side comparison with calculator predictions, broken
down by component (params, grads, optimizer, per-layer activations,
non-layer activations, transient). Reports the gap and expresses it
in units of B*S*H tensors for intuition.

Usage:
    python bench_memory_debug.py --type baseline  --hidden 512 --batch 16 --seq 2048
    python bench_memory_debug.py --type efficient --hidden 1024 --batch 32 --seq 4096
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gc
import argparse

import torch
from torch.autograd.graph import saved_tensors_hooks

from config import TransformerConfig
from model.transformer import BaselineTransformer
from efficient_model.transformer import EfficientTransformer
from efficient_optimizer.ademamix import AdEMAMix
from calculators import (
    ModelConfig,
    TrainingConfig,
    GPUSpec,
    BaselineCalculator,
    EfficientCalculator,
)

DTYPE = torch.bfloat16
DEVICE = torch.device("cuda")


def mb(x):
    return f"{x / 1e6:>8.0f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seq", type=int, default=2048)
    parser.add_argument("--type", choices=["baseline", "efficient"], default="baseline")
    args = parser.parse_args()

    # Override for your hardware.
    gpu = GPUSpec(name="RTX 3090", memory_bandwidth_gbps=842, flops_bf16_tflops=66, interconnect_bandwidth_gbps=32)

    config = TransformerConfig(
        hidden_dim=args.hidden,
        num_heads=args.heads,
        max_seq_len=args.seq * 2,
        dropout=0.0,
    )
    mc = ModelConfig(
        vocab_size=config.vocab_size,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        intermediate_dim=config.intermediate_dim,
        max_seq_len=config.max_seq_len,
    )
    tc = TrainingConfig(batch_size=args.batch, seq_len=args.seq, num_gpus=1)

    is_baseline = args.type == "baseline"
    ModelClass = BaselineTransformer if is_baseline else EfficientTransformer
    CalcClass = BaselineCalculator if is_baseline else EfficientCalculator

    model = ModelClass(config).to(device=DEVICE, dtype=DTYPE)
    optimizer = AdEMAMix(model.parameters(), lr=1e-4, betas=(0.9, 0.999, 0.9999))
    calc = CalcClass(mc, tc, gpu)

    input_ids = torch.randint(0, config.vocab_size, (args.batch, args.seq), device=DEVICE)
    labels = input_ids.clone()

    # --- Warmup (compile, allocator settle) ---
    if is_baseline:
        loss = model.compute_loss(model(input_ids), labels)
    else:
        loss = model(input_ids, labels=labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    del loss
    gc.collect()
    torch.cuda.empty_cache()

    # --- 1. Measure saved tensors ---
    exclude = {p.data.data_ptr() for p in model.parameters()}
    exclude |= {b.data.data_ptr() for b in model.buffers()}
    seen, total_saved = set(), 0

    def pack(t):
        nonlocal total_saved
        ptr = t.data.data_ptr()
        if ptr not in exclude and ptr not in seen:
            seen.add(ptr)
            total_saved += t.numel() * t.element_size()
        return t

    optimizer.zero_grad(set_to_none=True)
    with saved_tensors_hooks(pack, lambda t: t):
        if is_baseline:
            loss = model.compute_loss(model(input_ids), labels)
        else:
            loss = model(input_ids, labels=labels)
        loss.backward()
    del loss
    gc.collect()
    torch.cuda.empty_cache()

    # --- 2. Measure peak with phase snapshots ---
    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    static = torch.cuda.memory_allocated()

    if is_baseline:
        logits = model(input_ids)
        after_fwd = torch.cuda.memory_allocated()
        peak_fwd = torch.cuda.max_memory_allocated()
        loss = model.compute_loss(logits, labels)
        after_loss = torch.cuda.memory_allocated()
        peak_loss = torch.cuda.max_memory_allocated()
    else:
        loss = model(input_ids, labels=labels)
        after_fwd = torch.cuda.memory_allocated()
        peak_fwd = torch.cuda.max_memory_allocated()
        after_loss, peak_loss = after_fwd, peak_fwd

    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    after_bwd = torch.cuda.memory_allocated()
    peak_bwd = torch.cuda.max_memory_allocated()

    # --- Print ---
    V, I, N = config.vocab_size, config.intermediate_dim, config.num_layers
    print(f"\n{args.type}  H={args.hidden} B={args.batch} S={args.seq} V={V} I={I} N={N}")

    print(f"\nActual:")
    print(f"  static:         {mb(static)} MB")
    if is_baseline:
        print(f"  after_fwd:      {mb(after_fwd)} MB  peak={mb(peak_fwd)}  fwd_delta={mb(after_fwd - static)}")
        print(f"  after_loss:     {mb(after_loss)} MB  peak={mb(peak_loss)}  loss_delta={mb(after_loss - after_fwd)}")
    else:
        print(f"  after_fwd+loss: {mb(after_fwd)} MB  peak={mb(peak_fwd)}  fwd_delta={mb(after_fwd - static)}")
    print(f"  after_bwd:      {mb(after_bwd)} MB  peak={mb(peak_bwd)}  bwd_transient={mb(peak_bwd - after_loss)}")
    print(f"  saved_tensors:  {mb(total_saved)} MB")
    print(f"  actual_peak:    {mb(peak_bwd)} MB")

    print(f"\nPredicted:")
    p = calc.param_memory()
    g = calc.gradient_memory()
    o = calc.optimizer_memory()
    a = calc.activation_memory()
    t = calc._peak_transient_bytes()
    per_layer = calc._attn_saved_bytes() + calc._mlp_saved_bytes() + 2 * calc._norm_saved_bytes()
    non_layer = calc._non_layer_saved_bytes()

    print(f"  params:         {mb(p)} MB")
    print(f"  grads:          {mb(g)} MB")
    print(f"  optim:          {mb(o)} MB")
    print(f"  activations:    {mb(a)} MB")
    print(f"    per_layer:    {mb(per_layer)} MB  x{N} = {mb(per_layer * N)}")
    print(f"      attn:       {mb(calc._attn_saved_bytes())} MB")
    print(f"      mlp:        {mb(calc._mlp_saved_bytes())} MB")
    print(f"      norm x2:    {mb(2 * calc._norm_saved_bytes())} MB")
    print(f"    non_layer:    {mb(non_layer)} MB")
    print(f"  transient:      {mb(t)} MB")
    print(f"  pred_peak:      {mb(calc.peak_memory())} MB")

    gap = peak_bwd - calc.peak_memory()
    bsh_bytes = args.batch * args.seq * args.hidden * 2
    print(f"\n  gap:            {mb(gap)} MB")
    print(f"  B*S*H*bf16:    {mb(bsh_bytes)} MB")
    print(f"  gap / B*S*H:   {gap / bsh_bytes:>8.1f} tensors")


if __name__ == "__main__":
    main()
