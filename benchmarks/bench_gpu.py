"""
GPU microbenchmark: measure actual memory bandwidth and compute throughput.

Measures real numbers instead of relying on spec sheet values.
Use the output to create a GPUSpec with achievable (not peak) performance.

Usage:
    python bench_gpu.py
    python bench_gpu.py --size 256  # MB for bandwidth test
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time

import torch


def measure_bandwidth_gbps(size_mb=256, dtype=torch.bfloat16, iters=100, warmup=10):
    """
    Measure sustained HBM bandwidth via vector copy.

    Allocates two tensors and copies one to another repeatedly.
    This is a pure memory-bound operation -- no compute.
    Returns read+write bandwidth in GB/s.
    """
    elem_bytes = torch.finfo(dtype).bits // 8
    numel = (size_mb * 1024 * 1024) // elem_bytes
    a = torch.randn(numel, device="cuda", dtype=dtype)
    b = torch.empty_like(a)

    for _ in range(warmup):
        b.copy_(a)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        b.copy_(a)
    torch.cuda.synchronize()
    elapsed = start = time.perf_counter() - start

    bytes_per_iter = numel * elem_bytes * 2  # read a + write b
    total_bytes = bytes_per_iter * iters
    return total_bytes / elapsed / 1e9


def measure_flops_tflops(dtype=torch.bfloat16, m=4096, n=4096, k=4096, iters=100, warmup=10):
    """
    Measure sustained matmul throughput.

    Runs C = A @ B with large square matrices. This is a pure
    compute-bound operation at this size.
    Returns throughput in TFLOP/s.
    """
    a = torch.randn(m, k, device="cuda", dtype=dtype)
    b = torch.randn(k, n, device="cuda", dtype=dtype)

    for _ in range(warmup):
        torch.mm(a, b)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        torch.mm(a, b)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    flops_per_iter = 2 * m * n * k
    total_flops = flops_per_iter * iters
    return total_flops / elapsed / 1e12


def measure_elementwise_flops(dtype=torch.bfloat16, size_mb=256, iters=100, warmup=10):
    """
    Measure throughput on a memory-bound elementwise op (sigmoid).

    This gives the effective FLOP/s for memory-bound kernels like
    layernorm, softmax, activations -- where bandwidth limits throughput,
    not ALU.
    """
    elem_bytes = torch.finfo(dtype).bits // 8
    numel = (size_mb * 1024 * 1024) // elem_bytes
    a = torch.randn(numel, device="cuda", dtype=dtype)

    for _ in range(warmup):
        torch.sigmoid(a)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        torch.sigmoid(a)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    flops_per_iter = numel * 4  # sigmoid ~ 4 ops (exp, add, div, neg)
    total_flops = flops_per_iter * iters
    return total_flops / elapsed / 1e12


def main():
    parser = argparse.ArgumentParser(description="GPU microbenchmark")
    parser.add_argument("--size", type=int, default=256, help="MB for bandwidth test")
    parser.add_argument("--matmul-dim", type=int, default=4096, help="Matrix dim for compute test")
    args = parser.parse_args()

    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}")
    print(f"Memory: {props.total_memory / 1e9:.1f} GB")
    print()

    bw = measure_bandwidth_gbps(size_mb=args.size)
    flops_compute = measure_flops_tflops(m=args.matmul_dim, n=args.matmul_dim, k=args.matmul_dim)
    flops_membound = measure_elementwise_flops(size_mb=args.size)

    print(f"Sustained HBM bandwidth:     {bw:.0f} GB/s")
    print(f"Sustained matmul throughput:  {flops_compute:.1f} TFLOP/s (bf16)")
    print(f"Elementwise throughput:       {flops_membound:.2f} TFLOP/s (memory-bound)")
    print()

    arithmetic_intensity = flops_compute * 1e12 / (bw * 1e9)
    print(f"Arithmetic intensity cutoff:  {arithmetic_intensity:.0f} FLOP/byte")
    print(f"  (ops with lower intensity are memory-bound)")
    print()

    print("Suggested GPUSpec:")
    print(f'  GPUSpec(')
    print(f'      name="{props.name} (measured)",')
    print(f'      memory_bandwidth_gbps={bw:.0f},')
    print(f'      flops_bf16_tflops={flops_compute:.1f},')
    print(f'      interconnect_bandwidth_gbps=???,  # measure with NCCL tests')
    print(f'  )')


if __name__ == "__main__":
    main()