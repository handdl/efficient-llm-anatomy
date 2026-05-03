"""
Fused linear cross-entropy for causal LM.

Uses Liger's chunked implementation: processes lm_head projection
and CE loss together without materializing the full B*S*V logits tensor.
"""

import torch
import torch.nn as nn

from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss


class CrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index
        self.fused_lce = LigerFusedLinearCrossEntropyLoss(reduction="mean", ignore_index=ignore_index)

    def forward(self, hidden_states, weight, labels) -> torch.Tensor:
        hidden_states = hidden_states[:, :-1].contiguous()
        labels = labels[:, 1:].contiguous()
        B, S, H = hidden_states.shape
        return self.fused_lce(weight, hidden_states.view(B * S, H), labels.view(B * S))
