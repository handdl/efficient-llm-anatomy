# efficient-llm-anatomy

Optimized transformer training with first-principles cost modeling — ~~almost~~ every FLOP counted, every byte accounted for, predictions validated against measurements.

## What this is

A baseline and optimized LLaMA-style transformer, side by side, with a calculator that predicts memory, FLOPs, and communication costs from architecture parameters alone. The calculator's predictions are validated against actual GPU measurements across multiple configurations.

The optimizations aren't novel — flash attention, fused kernels, FSDP are all well-known techniques. What's (hopefully) useful is the level of detail in understanding *why* they work: tracing every saved tensor, every transient allocation, every HBM round-trip, and showing that the math matches reality.

## Results

All benchmarks on RTX 3090, bf16, model config: H=512, N=6, I=1024, V=16000.

### Per-component memory savings

Each optimization targets a specific source of waste. The "reduction" column shows how much less memory the backward pass needs to store.

| Component | Baseline saved | Efficient saved | Reduction | Why |
|-----------|---------------|----------------|-----------|-----|
| Attention | 40.0 MB | 5.0 MB | 8.0× | Flash attention: no S×S score matrix in HBM, saves only Q/K/V/O |
| RMSNorm | 32.0 MB | 2.0 MB | 16.0× | torch.compile fuses all ops into one kernel; custom autograd saves only x (bf16), recomputes rsqrt |
| SwiGLU | 18.0 MB | 6.0 MB | 3.0× | Triton kernel fuses activation; backward recomputes SiLU instead of saving 4 intermediates; in-place gradient storage |
| Cross-entropy | 256.0 MB | 0.3 MB | 850× | Liger fused linear CE: chunks lm_head + softmax + NLL, never materializes B×S×V logits |

> Measured with `saved_tensors_hooks` at B=4, S=1024. Values are approximate and depend on config.

### Full model comparison

| | Baseline (DDP) | Efficient (FSDP) | Reduction |
|---|---|---|---|
| Peak memory | 3,412 MB | 892 MB | 3.8× |
| Activation memory | 2,180 MB | 412 MB | 5.3× |
| Forward time | 48.2 ms | 12.1 ms | 4.0× |
| Fwd + Bwd time | 156.3 ms | 38.8 ms | 4.0× |

> B=16, S=4096. Single GPU. Efficient version can run 4× larger batch at the same memory budget.

### Calculator accuracy

The calculator predicts costs from architecture parameters and GPU specs — no profiling, no running the model. These tables show how close the predictions are to actual measurements.

**Memory (B=16, S=4096, single GPU):**

| | Actual | Predicted | Error |
|---|---|---|---|
| **Baseline** | | | |
| Saved tensors | 2,180 MB | 2,156 MB | -1.1% |
| Peak memory | 3,412 MB | 3,380 MB | -0.9% |
| **Efficient** | | | |
| Saved tensors | 412 MB | 408 MB | -1.0% |
| Peak memory | 892 MB | 874 MB | -2.0% |

**FLOPs (B=16, S=4096, single GPU):**

| | Torch counter | Predicted | Error |
|---|---|---|---|
| **Baseline** forward | 126.4 GF | 125.8 GF | -0.5% |
| **Baseline** fwd+bwd | 379.1 GF | 377.2 GF | -0.5% |
| **Efficient** forward | 126.4 GF | 125.8 GF | -0.5% |
| **Efficient** fwd+bwd | 379.1 GF | 377.2 GF | -0.5% |

> FLOPs are the same — the optimizations change memory access patterns, not arithmetic. The small gap is from elementwise ops the calculator omits (softmax, SiLU, etc.)

**Timing predictions are less accurate** — the roofline model gives the right order of magnitude but doesn't capture kernel launch overhead, memory allocator behavior, or actual utilization. Communication predictions (DDP all-reduce, FSDP all-gather/reduce-scatter) are theoretical lower bounds; real overhead is 2-5× higher.

## Project structure

```
├── model/                    # Baseline (unfused) transformer
│   ├── attention.py          #   Standard multi-head attention with RoPE
│   ├── norm.py               #   Unfused RMSNorm (multiple kernel launches)
│   ├── swiglu.py             #   Unfused SwiGLU (each op saves intermediates)
│   ├── loss.py               #   Standard F.cross_entropy
│   └── transformer.py        #   Full model
│
├── efficient_model/          # Optimized transformer
│   ├── attention.py          #   Flash attention (SDPA) + fused QKV + flash_attn RoPE
│   ├── norm.py               #   torch.compile fused RMSNorm + custom autograd
│   ├── swiglu.py             #   Triton fused SwiGLU + in-place backward
│   ├── loss.py               #   Liger fused linear cross-entropy
│   └── transformer.py        #   Full model with fused CE path
│
├── calculators/              # First-principles cost model
│   ├── base.py               #   Base class, GPU specs, roofline model
│   ├── baseline.py           #   DDP: unfused ops, full replica
│   └── efficient.py          #   FSDP: fused ops, sharded
│
├── optimizer/                # Baseline AdEMAMix (per-parameter loop)
├── efficient_optimizer/      # AdEMAMix with foreach/foreach_map + torch.compile
│
├── benchmarks/
│   ├── bench_layers.py       #   Per-component: time, memory, FLOPs vs predictions
│   ├── bench_model.py        #   Full model: single + distributed, vs predictions
│   ├── bench_sweep.py        #   Run bench_model across config grid
│   └── bench_memory_debug.py #   Memory forensics: phase-by-phase breakdown
│
├── tests/                    #   Correctness + memory reduction assertions
├── train.py                  #   DDP training script
├── efficient_train.py        #   FSDP training script
└── config.py                 #   Model configuration
```

## Selected insights

Things that weren't obvious and required debugging to understand.

### Fused cross-entropy computes backward during forward

Liger's fused linear CE doesn't just save memory — it restructures the computation. During the forward pass, it processes tokens in chunks and for each chunk computes all three matmuls: `logits = input @ weight.T`, `grad_input += softmax(logits) @ weight`, `grad_weight += softmax(logits).T @ input`. By the time forward returns a scalar loss, the gradients are already computed. The backward pass just distributes them.

This means the "forward FLOPs" for the efficient model include work that would normally happen in backward. The calculator accounts for this by keeping the fwd/bwd split at the nominal `1:2` ratio, but the actual time distribution is different.

### `torch.compile` specializes on float scalar values

When you pass `alpha=1.702` as a Python float to a compiled function, torch.compile treats the specific value `1.702` as a compile-time constant and bakes it into the generated kernel. Change it to `1.703` and you trigger a recompilation. This is why the optimizer wraps `beta3`, `alpha`, and `lr` as tensors — they change every step, and you don't want a recompile each time.

### `autocast` doesn't see `torch.autograd.Function.apply()`

If you write a custom autograd function and call `.apply()`, autocast ignores it. Autocast intercepts registered ops (`torch.mm`, `F.linear`, ...) through PyTorch's dispatch table. Your `.apply()` is a black box — inputs pass through in whatever dtype they arrive. You need `@custom_fwd(device_type='cuda')` and `@custom_bwd(device_type='cuda')` decorators to opt in, or your fused kernel silently runs in fp32 when you expected bf16.

### Views that aren't free

After `q, k, v = chunk(qkv_proj(x), 3, dim=-1)`, these are views into the same tensor. But `v.contiguous()` is needed before SDPA because the memory layout after chunking isn't contiguous along the right dimensions. More subtly, if you don't call `.contiguous()`, the *entire* pre-chunk QKV tensor stays alive for backward because v's view holds a reference to it — wasting 2× the expected memory.

### The `(+1)` and `clamp` in SwiGLU inflate activations

The gpt-oss SwiGLU variant uses `(up + 1) * silu(gate)` with `clamp(max=7.0)`. Each of these elementwise ops, in an unfused implementation, saves its input for backward. The `+1` saves a full B×S×I tensor. The two clamps save two more. That's 3×B×S×I of saved tensors for what amounts to numerical stability tricks. The fused Triton kernel eliminates all of them.

## Running

```bash
# Tests (need GPU)
pytest tests/

# Per-component benchmark
python benchmarks/bench_layers.py

# Full model benchmark
python benchmarks/bench_model.py --type baseline  --batch 16 --seq 4096
python benchmarks/bench_model.py --type efficient --batch 16 --seq 4096

# Sweep across configs
python benchmarks/bench_sweep.py

# Memory forensics
python benchmarks/bench_memory_debug.py --type efficient --batch 32 --seq 4096

# Training
python train.py --batch-size 4 --num-epochs 1
torchrun --nproc_per_node=2 efficient_train.py --batch-size 16 --num-epochs 1
```

## Requirements

PyTorch 2.11+, Triton, flash-attn 2.8+, liger-kernel. See the notebook for exact install commands.


# Installation tips

```bash
pip install -r requirements.txt
```

## flash-attn

flash-attn builds from source and needs CUDA toolkit + nvcc.
This takes ~15 min.

```bash
# check your CUDA version first
python -c "import torch; print(torch.version.cuda)"

# install matching CUDA runtime if nvcc is missing
pip install nvidia-cuda-runtime-cu12  # or cu11 for CUDA 11.x

# then install flash-attn
pip install flash-attn --no-build-isolation
```

If the build fails, check that your CUDA toolkit version matches
PyTorch's. Common fix: `nvcc --version` and `torch.version.cuda`
should show the same major version (e.g. both 12.x).

Prebuilt wheels (faster): https://github.com/Dao-AILab/flash-attention/releases