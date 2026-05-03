"""
gpt-oss style SwiGLU Feed-Forward Network with fusion on triton and optimized checkpointing

Reference SwiGLU implementation:
https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/swiglu.py
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl

from liger_kernel.ops.utils import calculate_settings, ensure_contiguous

from torch.amp import custom_fwd, custom_bwd


@triton.jit
def silu(x, alpha):
    return x * tl.sigmoid(x * alpha)


@triton.jit
def _swiglu_forward_kernel(
    a_ptr, b_ptr, c_ptr, stride, alpha: float, limit: float, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    # a = gate, b = up
    program_id = tl.program_id(0).to(tl.int64)

    a_ptr += program_id * stride
    b_ptr += program_id * stride
    c_ptr += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # sigmoid requires fp32
    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)

    a_row = tl.minimum(a_row, limit)
    b_row = tl.clamp(b_row, min=-limit, max=limit)

    c_row = silu(a_row, alpha).cast(b_row.dtype) * (b_row + 1)
    tl.store(c_ptr + col_offsets, c_row, mask=mask)


@triton.jit
def _swiglu_backward_kernel(
    dc_ptr, a_ptr, b_ptr, stride, alpha: float, limit: float, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    program_id = tl.program_id(0).to(tl.int64)

    dc_ptr += program_id * stride
    a_ptr += program_id * stride
    b_ptr += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dc_row = tl.load(dc_ptr + col_offsets, mask=mask, other=0)

    # sigmoid requires fp32
    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)

    a_row = tl.minimum(a_row, limit)
    b_row = tl.clamp(b_row, min=-limit, max=limit)

    # for c=[sigmoid(a*alpha)*a]*(b+1) we have dL/da=dL/dc*[alpha*dsigmoid/da+sigmoid]*(b+1)
    sig_a = tl.sigmoid(a_row * alpha)
    silu_a = a_row * sig_a
    db_row = dc_row * silu_a
    da_row = (b_row + 1) * dc_row * (alpha * silu_a * (1 - sig_a) + sig_a)

    # clamp derivatives
    da_row = tl.where(a_row > limit, 0.0, da_row)
    db_row = tl.where((b_row < -limit) | (b_row > limit), 0.0, db_row)

    # downcast gate, up only in the end to preserve precision
    tl.store(a_ptr + col_offsets, da_row.cast(b_row.dtype), mask=mask)
    tl.store(b_ptr + col_offsets, db_row.cast(b_row.dtype), mask=mask)

    # compute swiglu in bf16 to match forward and grad of activation_out with activation_out itself
    tl.store(dc_ptr + col_offsets, (silu_a.cast(b_row.dtype) * (b_row + 1)), mask=mask)


def swiglu_forward(a, b, alpha, limit):
    ori_shape = a.shape
    n_cols = ori_shape[-1]

    # we work with 1D vectors
    a = a.view(-1, n_cols)
    b = b.view(-1, n_cols)
    c = torch.empty_like(a)
    n_rows = a.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _swiglu_forward_kernel[(n_rows,)](
        a,
        b,
        c,
        c.stride(-2),
        alpha=alpha,
        limit=limit,
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return a, b, c.view(*ori_shape)


def swiglu_backward(a, b, dc, alpha, limit):
    ori_shape = dc.shape
    n_cols = ori_shape[-1]
    dc = dc.view(-1, n_cols)
    n_rows = dc.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _swiglu_backward_kernel[(n_rows,)](
        dc,
        a,
        b,
        dc.stride(-2),
        alpha=alpha,
        limit=limit,
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return a.view(*ori_shape), b.view(*ori_shape)


class MemoryEfficientSwiGLUMLP(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(ctx, x, w_gate, w_up, w_down, alpha, limit):
        gate = x @ w_gate.T
        up = x @ w_up.T
        gate, up, activation_out = swiglu_forward(gate, up, alpha, limit)
        out = activation_out @ w_down.T

        ctx.save_for_backward(x, gate, up, w_gate, w_up, w_down)
        ctx.alpha = alpha
        ctx.limit = limit

        return out

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, out_grad):
        """
        Memory-efficient backward: reuse storage from forward (gate, up, x)
        and overwrite activation_out_grad with activation_out via the Triton kernel.

        Here we use gradient identities for Y = X @ W.T (G = dL/dY):
        dL/dX = G @ W, dL/dW = G.T @ X.
        """
        x, gate, up, w_gate, w_up, w_down = ctx.saved_tensors

        # first, for swiglu_backward we need the following gradient:
        activation_out_grad = out_grad @ w_down

        # second, we run backward for swiglu. Note, it overwrites every arguments!
        gate_grad, up_grad = swiglu_backward(gate, up, activation_out_grad, ctx.alpha, ctx.limit)

        # third, since backward just overwritten activation_out_grad with activation_out,
        # we can calculate all remaining gradients just as linear layers:
        # a) out = activation_out @ w_down.T
        # b) gate = x @ w_gate.T
        # c) up = x @ w_up.T
        activation_out = activation_out_grad
        w_down_grad = out_grad.reshape(-1, out_grad.shape[-1]).T @ activation_out.reshape(-1, activation_out.shape[-1])
        w_gate_grad = gate_grad.reshape(-1, gate_grad.shape[-1]).T @ x.reshape(-1, x.shape[-1])
        w_up_grad = up_grad.reshape(-1, up_grad.shape[-1]).T @ x.reshape(-1, x.shape[-1])

        # reuse x storage for x_grad!
        x_grad = x
        x_grad.copy_(gate_grad @ w_gate)
        x_grad += up_grad @ w_up
        return x_grad, w_gate_grad, w_up_grad, w_down_grad, None, None


class SwiGLUFeedForward(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.alpha = 1.702
        self.limit = 7.0
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return MemoryEfficientSwiGLUMLP.apply(
            x, self.gate_proj.weight, self.up_proj.weight, self.down_proj.weight, self.alpha, self.limit
        )
