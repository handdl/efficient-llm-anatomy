"""
Multi-head attention with flash attention implementation and RoPE.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import TransformerConfig

torch.backends.cuda.enable_mem_efficient_sdp(True)

from flash_attn.layers.rotary import apply_rotary_emb


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))  # (S, head_dim//2)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        positions = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(positions, self.inv_freq)  # (S, head_dim//2)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary positional embedding to q and k.

        Args:
            q: (B, num_heads, S, head_dim)
            k: (B, num_heads, S, head_dim)
            seq_len: sequence length (must be <= max_seq_len)

        Returns:
            q_rotated, k_rotated with same shapes
        """
        cos = self.cos[:seq_len]  # (S, head_dim//2)
        sin = self.sin[:seq_len]
        q_rotated = apply_rotary_emb(q.transpose(1, 2), cos, sin).transpose(1, 2)
        k_rotated = apply_rotary_emb(k.transpose(1, 2), cos, sin).transpose(1, 2)
        return q_rotated, k_rotated


class MultiHeadAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.qkv_proj = nn.Linear(config.hidden_dim, 3 * config.hidden_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.rope = RotaryPositionalEmbedding(
            head_dim=self.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, S, H = x.shape
        q, k, v = torch.chunk(self.qkv_proj(x), 3, dim=-1)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k, S)
        v = v.contiguous()  # detach it to free qkv_proj(x)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout.p if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().view(B, S, H)
        out = self.out_proj(out)
        return out
