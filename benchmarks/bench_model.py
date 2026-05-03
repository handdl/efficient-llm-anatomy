"""
Full-model benchmark: predicted vs actual for time, memory, FLOPs.

Runs a single (model_type, config) combination. Use bench_sweep.py to
run across multiple configs.

Supports single-GPU and distributed (DDP for baseline, FSDP for efficient).

Usage:
    python bench_model.py --type baseline  --hidden 512 --heads 8 --batch 16 --seq 512
    python bench_model.py --type efficient --hidden 512 --heads 8 --batch 16 --seq 512

    torchrun --nproc_per_node=2 bench_model.py --type baseline  --hidden 512 --heads 8 --batch 16 --seq 512
    torchrun --nproc_per_node=2 bench_model.py --type efficient --hidden 512 --heads 8 --batch 16 --seq 512
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import time
import argparse

import torch
import torch.distributed as dist
from torch.autograd.graph import saved_tensors_hooks
from torch.utils.flop_counter import FlopCounterMode

from config import TransformerConfig
from model.transformer import BaselineTransformer
from efficient_model.transformer import EfficientTransformer, TransformerBlock
from efficient_optimizer.ademamix import AdEMAMix
from calculators import (
    ModelConfig, TrainingConfig, GPUSpec,
    BaselineCalculator, EfficientCalculator,
)

DTYPE = torch.bfloat16


def bench_time_ms(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / iters


def measure_saved_bytes(fn, exclude_tensors=None):
    exclude_ptrs = {t.data.data_ptr() for t in (exclude_tensors or [])}
    seen, total = set(), 0

    def pack(t):
        nonlocal total
        ptr = t.data.data_ptr()
        if ptr not in exclude_ptrs and ptr not in seen:
            seen.add(ptr)
            total += t.numel() * t.element_size()
        return t

    with saved_tensors_hooks(pack, lambda t: t):
        fn()
    return total


def measure_peak_memory(fn):
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()


def setup_distributed():
    if "RANK" not in os.environ:
        return 0, 1, 0
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def wrap_distributed(model, world_size, local_rank, model_type):
    if world_size == 1:
        return model
    if model_type == "baseline":
        from torch.nn.parallel import DistributedDataParallel as DDP
        return DDP(model, device_ids=[local_rank])
    else:
        from torch.distributed.fsdp import fully_shard
        for layer in model.layers:
            fully_shard(layer)
        fully_shard(model)
        return model


def calc_fwd_flops(calc):
    """Sum per-layer + non-layer forward FLOPs from breakdown methods."""
    f = 0
    for _ in range(calc.N):
        f += calc._attn_breakdown().flops
        f += calc._mlp_breakdown().flops
        f += 2 * calc._norm_breakdown().flops
    f += calc._embedding_breakdown().flops
    f += calc._lm_head_breakdown().flops
    f += calc._loss_breakdown().flops
    return f


def calc_fwd_bwd_flops(calc):
    """Forward+backward FLOPs. Loss is not recomputed, everything else x3."""
    fwd = calc_fwd_flops(calc)
    loss = calc._loss_breakdown().flops
    return (fwd - loss) * 3 + loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--type", choices=["baseline", "efficient"], required=True)
    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    is_master = rank == 0
    is_distributed = world_size > 1

    # Override for your hardware.
    gpu = GPUSpec(name="RTX 3090", memory_bandwidth_gbps=842,
                  flops_bf16_tflops=66, interconnect_bandwidth_gbps=32)

    config = TransformerConfig(
        hidden_dim=args.hidden, num_heads=args.heads,
        max_seq_len=args.seq * 2, dropout=0.0,
    )
    mc = ModelConfig(
        vocab_size=config.vocab_size, hidden_dim=config.hidden_dim,
        num_heads=config.num_heads, num_layers=config.num_layers,
        intermediate_dim=config.intermediate_dim, max_seq_len=config.max_seq_len,
    )
    tc = TrainingConfig(batch_size=args.batch, seq_len=args.seq, num_gpus=world_size)

    ModelClass = BaselineTransformer if args.type == "baseline" else EfficientTransformer
    CalcClass = BaselineCalculator if args.type == "baseline" else EfficientCalculator

    model = ModelClass(config).to(device=device, dtype=DTYPE)
    model = wrap_distributed(model, world_size, local_rank, args.type)
    raw = model.module if hasattr(model, "module") else model
    optimizer = AdEMAMix(model.parameters(), lr=1e-4, betas=(0.9, 0.999, 0.9999))

    input_ids = torch.randint(0, config.vocab_size, (args.batch, args.seq), device=device)
    labels = input_ids.clone()

    if args.type == "baseline":
        def fwd():
            return raw.compute_loss(raw(input_ids), labels)

        def fwd_bwd():
            raw.compute_loss(raw(input_ids), labels).backward()

        def full_step():
            optimizer.zero_grad(set_to_none=True)
            raw.compute_loss(raw(input_ids), labels).backward()
            optimizer.step()
    else:
        def fwd():
            return raw(input_ids, labels=labels)

        def fwd_bwd():
            raw(input_ids, labels=labels).backward()

        def full_step():
            optimizer.zero_grad(set_to_none=True)
            raw(input_ids, labels=labels).backward()
            optimizer.step()

    calc = CalcClass(mc, tc, gpu)

    # FlopCounterMode inflates peak memory, so measure peak first.
    peak = measure_peak_memory(full_step)
    pred_peak = calc.peak_memory()

    # Single-GPU: also measure saved tensors and FLOPs.
    if not is_distributed:
        exclude = list(model.parameters()) + list(model.buffers())
        saved = measure_saved_bytes(fwd_bwd, exclude_tensors=exclude)
        pred_saved = calc.activation_memory()

        with FlopCounterMode(display=False) as fc:
            fwd()
        torch_fwd_flops = fc.get_total_flops()

        with FlopCounterMode(display=False) as fc:
            fwd_bwd()
        torch_fwd_bwd_flops = fc.get_total_flops()

        pred_fwd_flops = calc_fwd_flops(calc)
        pred_fwd_bwd_flops = calc_fwd_bwd_flops(calc)

    if is_distributed:
        dist.barrier()

    fwd_time = bench_time_ms(fwd)
    fwd_bwd_time = bench_time_ms(fwd_bwd)
    step_time = bench_time_ms(full_step)

    if is_master:
        mode = ("ddp" if args.type == "baseline" else "fsdp") if is_distributed else "single"

        header = (
            f"{'pass':<5} {'type':<10} {'mode':<6} "
            f"{'ms':>7} {'pred':>7} "
            f"{'save':>6} {'pred':>6} "
            f"{'peak':>6} {'pred':>6} "
            f"{'GF':>7} {'pred':>7} "
            f"{'TF/s':>6}"
        )
        print(f"\nG={world_size} H={args.hidden} B={args.batch} S={args.seq}")
        print(header)
        print("-" * len(header))

        na = "---"

        if not is_distributed:
            fwd_tfs = torch_fwd_flops / (fwd_time / 1000) / 1e12
            bwd_tfs = torch_fwd_bwd_flops / (fwd_bwd_time / 1000) / 1e12

            print(
                f"{'fwd':<5} {args.type:<10} {mode:<6} "
                f"{fwd_time:>7.1f} {calc.time_forward_ms():>7.1f} "
                f"{saved / 1e6:>6.0f} {pred_saved / 1e6:>6.0f} "
                f"{na:>6} {na:>6} "
                f"{torch_fwd_flops / 1e9:>7.0f} {pred_fwd_flops / 1e9:>7.0f} "
                f"{fwd_tfs:>6.1f}"
            )
            print(
                f"{'fb':<5} {args.type:<10} {mode:<6} "
                f"{fwd_bwd_time:>7.1f} {calc.time_forward_backward_ms():>7.1f} "
                f"{na:>6} {na:>6} "
                f"{peak / 1e6:>6.0f} {pred_peak / 1e6:>6.0f} "
                f"{torch_fwd_bwd_flops / 1e9:>7.0f} {pred_fwd_bwd_flops / 1e9:>7.0f} "
                f"{bwd_tfs:>6.1f}"
            )
            print(
                f"{'step':<5} {args.type:<10} {mode:<6} "
                f"{step_time:>7.1f} {na:>7} "
                f"{na:>6} {na:>6} {na:>6} {na:>6} {na:>7} {na:>7} {na:>6}"
            )
        else:
            pred_step = calc.time_total_step_ms()
            pred_comm = calc.time_communication_ms()
            print(
                f"{'step':<5} {args.type:<10} {mode:<6} "
                f"{step_time:>7.1f} {pred_step:>7.1f} "
                f"{na:>6} {na:>6} "
                f"{peak / 1e6:>6.0f} {pred_peak / 1e6:>6.0f} "
                f"{na:>7} {na:>7} {na:>6}"
            )
            print(f"comm_pred={pred_comm:.2f} ms (theoretical lower bound)\n")

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()