"""
Per-component benchmark: Attention, RMSNorm, SwiGLU, LM Head + Loss.

Compares baseline vs efficient implementations across time, saved-tensor
memory, FLOPs (torch counter vs calculator prediction), and throughput.

Usage:
    python bench_layers.py
    python bench_layers.py --hidden 1024 --heads 16 --batch 64 --seq 1024
"""

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.graph import saved_tensors_hooks
from torch.utils.flop_counter import FlopCounterMode

from config import TransformerConfig
from model import attention as base_attn, norm as base_norm, swiglu as base_swiglu
from model.loss import cross_entropy_loss
from efficient_model import attention as eff_attn, norm as eff_norm, swiglu as eff_swiglu
from efficient_model.loss import CrossEntropyLoss as FusedCrossEntropyLoss
from calculators import (
    ModelConfig,
    TrainingConfig,
    GPUSpec,
    BaselineCalculator,
    EfficientCalculator,
    RTX_3090,
)

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------


def bench_time_ms(fn, warmup=3, iters=10):
    """Wall-clock time per call (ms), averaged over iters after warmup."""
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
    """Total unique bytes saved for backward, excluding params/buffers."""
    exclude_ptrs = {t.data_ptr() for t in (exclude_tensors or [])}
    seen, total = set(), 0

    def pack(t):
        nonlocal total
        ptr = t.data_ptr()
        if ptr not in exclude_ptrs and ptr not in seen:
            seen.add(ptr)
            total += t.numel() * t.element_size()
        return t

    with saved_tensors_hooks(pack, lambda t: t):
        fn()
    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def make_configs(args):
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
    return config, mc, tc


def bench_layers(config, mc, tc, gpu):
    """Run all per-component benchmarks and print results."""

    # (name, baseline_module, efficient_module,
    #  calc_time_method, calc_saved_method, calc_breakdown_method)
    components = [
        (
            "Attention",
            base_attn.MultiHeadAttention(config),
            eff_attn.MultiHeadAttention(config),
            "time_attention_ms",
            "_attn_saved_bytes",
            "_attn_breakdown",
        ),
        (
            "RMSNorm",
            base_norm.RMSNorm(config.hidden_dim),
            eff_norm.RMSNorm(config.hidden_dim),
            "time_rms_norm_ms",
            "_norm_saved_bytes",
            "_norm_breakdown",
        ),
        (
            "SwiGLU",
            base_swiglu.SwiGLUFeedForward(config.hidden_dim, config.intermediate_dim),
            eff_swiglu.SwiGLUFeedForward(config.hidden_dim, config.intermediate_dim),
            "time_mlp_ms",
            "_mlp_saved_bytes",
            "_mlp_breakdown",
        ),
    ]

    header = (
        f"{'Component':<12} {'Type':<12} {'Time ms':>10} {'Pred ms':>10} "
        f"{'Mem MB':>10} {'Pred MB':>10} {'Torch GF':>10} {'Calc GF':>10} {'TFLOP/s':>10}"
    )
    print(f"\n{'=' * len(header)}")
    print(f"  Per-Layer  B={tc.batch_size}  S={tc.seq_len}  H={config.hidden_dim}")
    print(f"{'=' * len(header)}")
    print(header)
    print("-" * len(header))

    for name, base_mod, eff_mod, time_m, saved_m, bd_m in components:
        for label, mod, CalcCls in [
            ("baseline", base_mod, BaselineCalculator),
            ("efficient", eff_mod, EfficientCalculator),
        ]:
            mod = mod.to(device=DEVICE, dtype=DTYPE)
            x = torch.randn(
                tc.batch_size,
                tc.seq_len,
                config.hidden_dim,
                device=DEVICE,
                dtype=DTYPE,
                requires_grad=True,
            )
            params = list(mod.parameters())

            actual_time = bench_time_ms(lambda: mod(x))
            actual_mem = (
                measure_saved_bytes(
                    lambda: mod(x).sum().backward(),
                    exclude_tensors=params,
                )
                / 1e6
            )

            with FlopCounterMode(display=False) as fc:
                mod(x)
            torch_gf = fc.get_total_flops() / 1e9

            calc = CalcCls(mc, tc, gpu)
            # time_attention_ms -> roofline_time_ms(*_attn_breakdown())
            # but we call the named method for consistency
            pred_time = getattr(calc, time_m)()
            pred_mem = getattr(calc, saved_m)() / 1e6
            pred_flops = getattr(calc, bd_m)().flops
            throughput = pred_flops / (actual_time / 1000) / 1e12 if actual_time > 0 else 0

            print(
                f"{name:<12} {label:<12} {actual_time:>10.2f} {pred_time:>10.2f} "
                f"{actual_mem:>10.1f} {pred_mem:>10.1f} "
                f"{torch_gf:>10.2f} {pred_flops / 1e9:>10.2f} {throughput:>10.2f}"
            )
        print("-" * len(header))


def bench_lm_head_loss(config, mc, tc, gpu):
    """
    Benchmark LM Head + Loss as a combined operation.

    Baseline: logits = hidden @ weight.T, then F.cross_entropy.
    Efficient: Liger fused linear CE (chunked, never materializes logits).

    This doesn't fit the generic component loop because the baseline is
    two separate ops and the efficient version fuses them into one.
    """
    B, S, H, V = tc.batch_size, tc.seq_len, config.hidden_dim, config.vocab_size

    header = (
        f"{'Component':<12} {'Type':<12} {'Time ms':>10} {'Pred ms':>10} "
        f"{'Mem MB':>10} {'Pred MB':>10} {'Torch GF':>10} {'Calc GF':>10} {'TFLOP/s':>10}"
    )
    print(f"\n{'=' * len(header)}")
    print(f"  LM Head + Loss  B={B}  S={S}  H={H}  V={V}")
    print(f"{'=' * len(header)}")
    print(header)
    print("-" * len(header))

    # Shared inputs
    lm_weight = torch.randn(V, H, device=DEVICE, dtype=DTYPE, requires_grad=True)
    labels = torch.randint(0, V, (B, S), device=DEVICE)

    # --- Baseline: separate lm_head + cross_entropy ---
    def baseline_fwd(hidden):
        logits = hidden @ lm_weight.T
        return cross_entropy_loss(logits, labels)

    def baseline_fwd_bwd():
        h = torch.randn(B, S, H, device=DEVICE, dtype=DTYPE, requires_grad=True)
        loss = baseline_fwd(h)
        loss.backward()

    h_base = torch.randn(B, S, H, device=DEVICE, dtype=DTYPE, requires_grad=True)

    base_time = bench_time_ms(lambda: baseline_fwd(h_base))
    base_mem = measure_saved_bytes(baseline_fwd_bwd, exclude_tensors=[lm_weight]) / 1e6

    with FlopCounterMode(display=False) as fc:
        baseline_fwd(h_base)
    base_torch_gf = fc.get_total_flops() / 1e9

    base_calc = BaselineCalculator(mc, tc, gpu)
    base_bd = base_calc._lm_head_breakdown()
    base_loss_bd = base_calc._loss_breakdown()
    base_pred_flops = base_bd.flops + base_loss_bd.flops
    base_pred_time = base_calc.roofline_time_ms(
        base_pred_flops,
        base_bd.mem_bytes + base_loss_bd.mem_bytes,
    )
    base_pred_mem = (base_calc._non_layer_saved_bytes()) / 1e6  # logits + softmax dominate
    base_tp = base_pred_flops / (base_time / 1000) / 1e12 if base_time > 0 else 0

    print(
        f"{'LMHead+Loss':<12} {'baseline':<12} {base_time:>10.2f} {base_pred_time:>10.2f} "
        f"{base_mem:>10.1f} {base_pred_mem:>10.1f} "
        f"{base_torch_gf:>10.2f} {base_pred_flops / 1e9:>10.2f} {base_tp:>10.2f}"
    )

    # --- Efficient: fused linear cross-entropy ---
    fused_ce = FusedCrossEntropyLoss()

    def efficient_fwd(hidden):
        return fused_ce(hidden, lm_weight, labels)

    def efficient_fwd_bwd():
        h = torch.randn(B, S, H, device=DEVICE, dtype=DTYPE, requires_grad=True)
        loss = efficient_fwd(h)
        loss.backward()

    h_eff = torch.randn(B, S, H, device=DEVICE, dtype=DTYPE, requires_grad=True)

    eff_time = bench_time_ms(lambda: efficient_fwd(h_eff))
    eff_mem = measure_saved_bytes(efficient_fwd_bwd, exclude_tensors=[lm_weight]) / 1e6

    with FlopCounterMode(display=False) as fc:
        efficient_fwd(h_eff)
    eff_torch_gf = fc.get_total_flops() / 1e9

    eff_calc = EfficientCalculator(mc, tc, gpu)
    eff_bd = eff_calc._lm_head_breakdown()
    eff_loss_bd = eff_calc._loss_breakdown()
    eff_pred_flops = eff_bd.flops + eff_loss_bd.flops
    eff_pred_time = eff_calc.roofline_time_ms(
        eff_pred_flops,
        eff_bd.mem_bytes + eff_loss_bd.mem_bytes,
    )
    eff_pred_mem = (eff_calc._non_layer_saved_bytes()) / 1e6
    eff_tp = eff_pred_flops / (eff_time / 1000) / 1e12 if eff_time > 0 else 0

    print(
        f"{'LMHead+Loss':<12} {'efficient':<12} {eff_time:>10.2f} {eff_pred_time:>10.2f} "
        f"{eff_mem:>10.1f} {eff_pred_mem:>10.1f} "
        f"{eff_torch_gf:>10.2f} {eff_pred_flops / 1e9:>10.2f} {eff_tp:>10.2f}"
    )
    print("-" * len(header))


def main():
    parser = argparse.ArgumentParser(description="Per-component benchmark")
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--seq", type=int, default=512)
    args = parser.parse_args()

    config, mc, tc = make_configs(args)

    # Override GPU spec here for your hardware.
    gpu = GPUSpec(name="RTX 3090", memory_bandwidth_gbps=936, flops_bf16_tflops=50, interconnect_bandwidth_gbps=32)

    bench_layers(config, mc, tc, gpu)
    bench_lm_head_loss(config, mc, tc, gpu)


if __name__ == "__main__":
    main()
