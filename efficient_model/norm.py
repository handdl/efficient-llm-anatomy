"""
Fused RMSNorm with torch.compile and custom autograd.

Compiled forward fuses square/mean/rsqrt/scale into a single kernel.
Custom backward saves x (bf16) and rsqrt (fp32, B*S*1 -- negligible),
recomputing x_normalized in backward instead of saving it.
Reduces saved-tensor memory from 2*B*S*H*4 (fp32) to B*S*H*2 (bf16) -- ~4x reduction.
"""

import torch
import torch.nn as nn


@torch.compile
def rmsnorm_forward(x, weight, eps):
    input_dtype = x.dtype
    x_fp32 = x.float()
    mean_sq = (x_fp32 * x_fp32).mean(dim=-1, keepdim=True)
    rsqrt = torch.rsqrt(mean_sq + eps)
    normalized = x_fp32 * rsqrt
    scale = 1.0 + weight.float()
    output = (normalized * scale).to(input_dtype)
    return output, rsqrt, x


@torch.compile
def rmsnorm_backward(grad_output, rsqrt, x, weight):
    x_fp32 = x.float()
    grad_fp32 = grad_output.float()
    scale = 1.0 + weight.float()
    x_norm = x_fp32 * rsqrt
    scaled_grad = grad_fp32 * scale
    inner = (scaled_grad * x_norm).mean(dim=-1, keepdim=True)
    x_grad = rsqrt * (scaled_grad - x_norm * inner)
    all_dims_but_last = tuple(range(grad_output.dim() - 1))
    weight_grad = (scaled_grad * x_norm).sum(dim=all_dims_but_last, keepdim=False)
    return x_grad.to(x.dtype), weight_grad


class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        output, rsqrt, x_orig = rmsnorm_forward(x, weight, eps)
        ctx.save_for_backward(rsqrt, x_orig, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        rsqrt, x_orig, weight = ctx.saved_tensors
        x_grad, weight_grad = rmsnorm_backward(grad_output, rsqrt, x_orig, weight)
        return x_grad, weight_grad, None


class RMSNorm(nn.Module):
    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return RMSNormFunction.apply(x, self.weight, self.eps)
