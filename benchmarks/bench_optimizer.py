"""
Optimizer benchmarks: compilation check and step time comparison.

Usage:
    python bench_optimizer.py
    python bench_optimizer.py --check-kernel
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import gc
import time

import torch
import torch.nn as nn

from efficient_optimizer.ademamix import AdEMAMix


def check_compiled_kernel():
    """
    Print compiled kernel code for manual inspection.
    
    You should see a single fused kernel for the foreach path,
    not one kernel per param. Not automatable -- output format
    changes between torch versions.
    """
    torch._dynamo.reset()
    torch._logging.set_logs(output_code=True)

    model = nn.Sequential(
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
    ).cuda()

    optimizer = AdEMAMix(model.parameters(), use_foreach_map=False)
    x = torch.randn(4, 256, device="cuda")
    loss = model(x).sum()
    loss.backward()
    optimizer.step()

    torch._logging.set_logs(output_code=False)
    print("\n^^^ Check the output above for a single fused kernel ^^^")


def build_model(hidden=512, layers=24, device="cuda", dtype=torch.float32):
    modules = []
    for _ in range(layers):
        modules.append(nn.Linear(hidden, hidden, bias=True))
        modules.append(nn.ReLU())
    return nn.Sequential(*modules).to(device=device, dtype=dtype)


def fake_grads(model):
    for p in model.parameters():
        p.grad = torch.randn_like(p)


def benchmark_step(opt, model, warmup_steps=20, bench_steps=100):
    """Returns ms/step and peak memory delta in MB."""
    fake_grads(model)
    for _ in range(warmup_steps):
        opt.step()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()

    start = time.perf_counter()
    for _ in range(bench_steps):
        opt.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    mem_peak = torch.cuda.max_memory_allocated()
    mem_delta_mb = (mem_peak - mem_before) / 1024 / 1024
    ms_per_step = (elapsed / bench_steps) * 1000
    return ms_per_step, mem_delta_mb


def run_comparison():
    device = "cuda"
    dtype = torch.bfloat16

    model_sizes = [
        (4096, 128),
        (4096, 32),
        (256, 32),
        (256, 128),
        (512, 32),
        (512, 128),
    ]

    from optimizer.ademamix import AdEMAMix

    optimizers = [
        "AdamW",
        "AdamW foreach",
        "AdamW fused",
        "AdEMAMix foreach",
        "AdEMAMix foreach_map",
    ]

    print(f"\n{'hidden':>6}  {'layers':>6}  {'params':>10}  {'optimizer':30s}  {'ms/step':>10}  {'peak MB':>10}")

    for hidden, layers in model_sizes:
        for name in optimizers:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch._dynamo.reset()

            model = build_model(hidden, layers, device, dtype)
            num_params = sum(p.numel() for p in model.parameters())
            fake_grads(model)

            if name == "AdamW":
                opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1,
                                        foreach=False, fused=False)
            elif name == "AdamW foreach":
                opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1,
                                        foreach=True, fused=False)
            elif name == "AdamW fused":
                opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1,
                                        foreach=False, fused=True)
            elif name == "AdEMAMix foreach":
                opt = AdEMAMix(model.parameters(), lr=1e-3, weight_decay=0.1,
                                      alpha_warmup=51, beta3_warmup=51,
                                      use_foreach_map=False)
            elif name == "AdEMAMix foreach_map":
                opt = AdEMAMix(model.parameters(), lr=1e-3, weight_decay=0.1,
                               alpha_warmup=51, beta3_warmup=51,
                               use_foreach_map=True)

            ms, mem = benchmark_step(opt, model)
            print(f"{hidden:>6}  {layers:>6}  {num_params:>10,}  {name:30s}  {ms:>10.3f}  {mem:>10.1f}")
            del model, opt

        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-kernel", action="store_true",
                        help="Only print compiled kernel code for manual inspection")
    args = parser.parse_args()

    if args.check_kernel:
        check_compiled_kernel()
    else:
        run_comparison()


if __name__ == "__main__":
    main()