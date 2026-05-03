"""
AdEMAMix optimizer with two compiled backends.

Optimizations applied:
- torch.compile(fullgraph=True) on both step functions to avoid Python overhead
  and let torch fuse elementwise ops into fewer kernels.
- Scheduled values (beta3, alpha, lr, bias_corrections) are passed as tensors,
  not Python floats. Floats would trigger recompilation every step because
  torch.compile specializes on float values. Tensors are traced symbolically.
- foreach_ backend (default): uses torch._foreach_* ops which launch one kernel
  per operation across all params (vs one kernel per param per op in naive loop).
"""

import math

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer
from torch._higher_order_ops.foreach_map import foreach_map


def linear_warmup_scheduler(step, alpha_end, alpha_start=0, warmup=1):
    if step < warmup:
        a = step / float(warmup)
        return (1.0 - a) * alpha_start + a * alpha_end
    return alpha_end


def linear_hl_warmup_scheduler(step, beta_end, beta_start=0, warmup=1):
    def f(beta, eps=1e-8):
        return math.log(0.5) / math.log(beta + eps) - 1

    def f_inv(t):
        return math.pow(0.5, 1 / (t + 1))

    if step < warmup:
        a = step / float(warmup)
        return f_inv((1.0 - a) * f(beta_start) + a * f(beta_end))
    return beta_end


def _ademamix_single(
    param,
    exp_avg_fast,
    exp_avg_slow,
    exp_avg_sq,
    grad,
    beta1,
    beta2,
    beta3,
    one_minus_beta3,
    alpha,
    bias_correction1,
    bias_correction2,
    lr,
    eps,
    weight_decay,
):
    new_exp_avg_fast = exp_avg_fast * beta1 + grad * (1 - beta1)
    new_exp_avg_sq = exp_avg_sq * beta2 + grad * grad * (1 - beta2)
    new_exp_avg_slow = exp_avg_slow * beta3 + grad * one_minus_beta3

    denom = (new_exp_avg_sq / bias_correction2).sqrt() + eps
    update = (new_exp_avg_fast / bias_correction1 + alpha * new_exp_avg_slow) / denom
    update = update + param * weight_decay
    new_param = param + update * (-lr)

    return new_param, new_exp_avg_fast, new_exp_avg_slow, new_exp_avg_sq


@torch.compile(fullgraph=True)
def ademamix_foreach_map_fn(
    params,
    grads,
    exp_avg_fasts,
    exp_avg_slows,
    exp_avg_sqs,
    beta1: float,
    beta2: float,
    beta3: Tensor,
    one_minus_beta3: Tensor,
    alpha: Tensor,
    bias_correction1: Tensor,
    bias_correction2: Tensor,
    lr: float,
    eps: float,
    lmbda: float,
):
    result = foreach_map(
        _ademamix_single,
        params,
        exp_avg_fasts,
        exp_avg_slows,
        exp_avg_sqs,
        grads,
        beta1,
        beta2,
        beta3,
        one_minus_beta3,
        alpha,
        bias_correction1,
        bias_correction2,
        lr,
        eps,
        lmbda,
    )
    new_params, new_fasts, new_slows, new_sqs = zip(*result)
    torch._foreach_copy_(params, list(new_params))
    torch._foreach_copy_(exp_avg_fasts, list(new_fasts))
    torch._foreach_copy_(exp_avg_slows, list(new_slows))
    torch._foreach_copy_(exp_avg_sqs, list(new_sqs))


@torch.compile(fullgraph=True)
def ademamix_foreach_fn(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avg_fasts: list[Tensor],
    exp_avg_slows: list[Tensor],
    exp_avg_sqs: list[Tensor],
    beta1: float,
    beta2: float,
    beta3: Tensor,
    one_minus_beta3: Tensor,
    alpha: Tensor,
    bias_correction1: Tensor,
    bias_correction2: Tensor,
    lr: Tensor,
    eps: float,
    lmbda: float,
):
    """
    Workaround: _foreach_add_(tensors, other, alpha=scalar) requires
    alpha to be a Python float, but we have to keep them as tensor
    to avoid constant recompilation because of schedule. So for tensor-valued
    alpha/beta3/lr we pre-multiply with _foreach_mul and add the result instead.
    """
    torch._foreach_mul_(exp_avg_fasts, beta1)
    torch._foreach_mul_(exp_avg_sqs, beta2)
    torch._foreach_mul_(exp_avg_slows, beta3)

    torch._foreach_add_(exp_avg_fasts, grads, alpha=1 - beta1)
    torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1 - beta2)

    torch._foreach_add_(exp_avg_slows, torch._foreach_mul(grads, one_minus_beta3))

    m = torch._foreach_div(exp_avg_fasts, bias_correction1)
    v = torch._foreach_div(exp_avg_sqs, bias_correction2)
    torch._foreach_sqrt_(v)
    torch._foreach_add_(v, eps)

    torch._foreach_add_(m, torch._foreach_mul(exp_avg_slows, alpha))

    torch._foreach_div_(m, v)
    torch._foreach_add_(m, params, alpha=lmbda)

    torch._foreach_add_(params, torch._foreach_mul(m, -lr))


class AdEMAMix(Optimizer):
    r"""Implements the AdEMAMix algorithm.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999, 0.9999))
            corresponding to beta_1, beta_2, beta_3 in AdEMAMix
        alpha (float): AdEMAMix alpha coeficient mixing the slow and fast EMAs (default: 2)
        beta3_warmup (int, optional): number of warmup steps used to increase beta3 (default: None)
        alpha_warmup: (int, optional): number of warmup steps used to increase alpha (default: None)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay as in AdamW (default: 0)
        use_foreach_map (bool, optional): use foreach_map (fused kernel) instead of foreach ops (default: False)
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999, 0.9999),
        alpha=2.0,
        beta3_warmup=None,
        alpha_warmup=None,
        eps=1e-8,
        weight_decay=0,
        use_foreach_map=False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError("Invalid beta parameter at index 2: {}".format(betas[2]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            alpha=alpha,
            beta3_warmup=beta3_warmup,
            alpha_warmup=alpha_warmup,
            weight_decay=weight_decay,
        )
        super(AdEMAMix, self).__init__(params, defaults)
        self.use_foreach_map = use_foreach_map

    def __setstate__(self, state):
        super(AdEMAMix, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            lmbda = group["weight_decay"]
            eps = group["eps"]
            beta1, beta2, beta3_final = group["betas"]
            beta3_warmup = group["beta3_warmup"]
            alpha_final = group["alpha"]
            alpha_warmup = group["alpha_warmup"]

            params: list[Tensor] = []
            grads: list[Tensor] = []
            exp_avg_fasts: list[Tensor] = []
            exp_avg_slows: list[Tensor] = []
            exp_avg_sqs: list[Tensor] = []

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg_slow"] = torch.zeros_like(p)
                    state["exp_avg_fast"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                params.append(p)
                grads.append(p.grad)
                exp_avg_fasts.append(state["exp_avg_fast"])
                exp_avg_slows.append(state["exp_avg_slow"])
                exp_avg_sqs.append(state["exp_avg_sq"])

            if not params:
                continue

            group["step"] = group.get("step", 0) + 1

            bias_correction1 = 1 - beta1 ** group["step"]
            bias_correction2 = 1 - beta2 ** group["step"]

            if alpha_warmup is not None:
                alpha = linear_warmup_scheduler(
                    group["step"], alpha_end=alpha_final, alpha_start=0, warmup=alpha_warmup
                )
            else:
                alpha = alpha_final

            if beta3_warmup is not None:
                beta3 = linear_hl_warmup_scheduler(
                    group["step"], beta_end=beta3_final, beta_start=beta1, warmup=beta3_warmup
                )
            else:
                beta3 = beta3_final

            device = params[0].device
            bias_correction1 = torch.tensor(bias_correction1, device=device)
            bias_correction2 = torch.tensor(bias_correction2, device=device)
            one_minus_beta3 = torch.tensor(1 - beta3, device=device)
            alpha = torch.tensor(alpha, device=device)
            beta3 = torch.tensor(beta3, device=device)
            lr = torch.tensor(lr, device=device)

            step_fn = ademamix_foreach_map_fn if self.use_foreach_map else ademamix_foreach_fn
            step_fn(
                params=params,
                grads=grads,
                exp_avg_fasts=exp_avg_fasts,
                exp_avg_slows=exp_avg_slows,
                exp_avg_sqs=exp_avg_sqs,
                beta1=beta1,
                beta2=beta2,
                beta3=beta3,
                one_minus_beta3=one_minus_beta3,
                alpha=alpha,
                bias_correction1=bias_correction1,
                bias_correction2=bias_correction2,
                lr=lr,
                eps=eps,
                lmbda=lmbda,
            )

        return loss
