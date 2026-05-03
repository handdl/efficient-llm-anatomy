"""
Base classes for first-principles transformer training calculators.

Conventions throughout this module:
    - Parameters counted as scalar count (not bytes).
    - Memory always in bytes.
    - Time always in milliseconds.
    - FLOPs use the multiply-accumulate = 2 ops convention.
    - All breakdown methods return OpBreakdown(flops, mem_bytes).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple


class OpBreakdown(NamedTuple):
    """Return type for every _*_breakdown() method."""

    flops: int
    mem_bytes: int


@dataclass
class GPUSpec:
    """Hardware parameters needed for roofline estimates."""

    name: str
    memory_bandwidth_gbps: float  # sustained HBM bandwidth, GB/s
    flops_bf16_tflops: float  # sustained BF16 throughput, TFLOP/s
    interconnect_bandwidth_gbps: float  # per-GPU per-direction, GB/s


@dataclass
class ModelConfig:
    """Architecture parameters (LLaMA-style transformer)."""

    vocab_size: int  # V
    hidden_dim: int  # H
    num_heads: int  # h
    num_layers: int  # N
    intermediate_dim: int  # I  (SwiGLU intermediate)
    max_seq_len: int


@dataclass
class TrainingConfig:
    """Training-time parameters."""

    batch_size: int  # B
    seq_len: int  # S
    num_gpus: int  # G
    dtype_bytes: int = 2  # bf16 = 2


H100_SXM = GPUSpec(
    name="H100 SXM",
    memory_bandwidth_gbps=2800,
    flops_bf16_tflops=800,
    interconnect_bandwidth_gbps=400,
)

RTX_3090 = GPUSpec(
    name="RTX 3090",
    memory_bandwidth_gbps=936,
    flops_bf16_tflops=142,
    interconnect_bandwidth_gbps=32,
)

T4 = GPUSpec(
    name="T4",
    memory_bandwidth_gbps=320,
    flops_bf16_tflops=8.1,
    interconnect_bandwidth_gbps=32,
)


class BaseCalculator(ABC):
    """
    Abstract base for per-GPU training cost estimation.

    Subclasses define how each operation is implemented (fused vs naive,
    flash vs standard attention, sharded vs replicated) by overriding
    the breakdown and memory methods.
    """

    def __init__(
        self,
        model: ModelConfig,
        training: TrainingConfig,
        gpu: GPUSpec,
    ):
        self.model = model
        self.training = training
        self.gpu = gpu

        # Shortcuts used everywhere in formulas.
        self.B = training.batch_size
        self.S = training.seq_len
        self.H = model.hidden_dim
        self.I = model.intermediate_dim
        self.N = model.num_layers
        self.V = model.vocab_size
        self.G = training.num_gpus
        self.h = model.num_heads
        self.d = self.H // self.h  # head dimension

        # Ring all-reduce fraction: each GPU sends (G-1)/G of the data.
        self.k = (self.G - 1) / self.G

    def roofline_time_ms(self, flops: int, mem_bytes: int) -> float:
        """
        Roofline model: time = max(compute_time, memory_time).

        GPU pipelines memory loads with ALU work, so the slower side
        dominates while the faster one hides behind it.
        """
        compute_s = flops / (self.gpu.flops_bf16_tflops * 1e12)
        memory_s = mem_bytes / (self.gpu.memory_bandwidth_gbps * 1e9)
        return max(compute_s, memory_s) * 1000

    def total_params(self) -> int:
        """
        Total scalar parameters for a LLaMA-style transformer.

        Per layer:
            attention: Wq + Wk + Wv + Wo              = 4*H*H
            MLP (SwiGLU): W_gate + W_up + W_down       = 3*H*I
            norms: 2x RMSNorm scale (no bias)           = 2*H

        Non-layer:
            token embedding                             = V*H
            LM head (unshared)                          = V*H
            final RMSNorm                               = H
        """
        H, I, N, V = self.H, self.I, self.N, self.V

        per_layer = (4 * H * H) + (3 * H * I) + (2 * H)
        non_layer = 2 * V * H + H

        return N * per_layer + non_layer

    @abstractmethod
    def param_memory(self) -> int:
        """Parameter memory per GPU, bytes."""

    @abstractmethod
    def gradient_memory(self) -> int:
        """Gradient memory per GPU, bytes."""

    @abstractmethod
    def optimizer_memory(self) -> int:
        """Optimizer state memory per GPU, bytes."""

    @abstractmethod
    def _attn_saved_bytes(self) -> int:
        """Tensors saved for backward by the attention block."""

    @abstractmethod
    def _mlp_saved_bytes(self) -> int:
        """Tensors saved for backward by the MLP block."""

    @abstractmethod
    def _norm_saved_bytes(self) -> int:
        """Tensors saved for backward by one RMSNorm."""

    @abstractmethod
    def _non_layer_saved_bytes(self) -> int:
        """Saved tensors outside the layer stack (embedding, CE, etc.)."""

    def activation_memory(self) -> int:
        """
        Total saved-tensor memory across all layers.

        Each layer saves: attention + MLP + 2x norm.
        Residual connections are views, not new allocations.
        """
        per_layer = self._attn_saved_bytes() + self._mlp_saved_bytes() + 2 * self._norm_saved_bytes()
        return self.N * per_layer + self._non_layer_saved_bytes()

    @abstractmethod
    def _peak_transient_bytes(self) -> int:
        """
        Largest temporary allocation during fwd+bwd that isn't in
        saved tensors, params, grads, or optimizer states.

        Candidates: attention score matrix, logits, CE softmax, etc.
        Only the max matters because they don't coexist.
        """

    def peak_memory(self) -> int:
        """Peak GPU memory during a full training step."""
        return (
            self.param_memory()
            + self.gradient_memory()
            + self.optimizer_memory()
            + self.activation_memory()
            + self._peak_transient_bytes()
        )

    def _embedding_breakdown(self) -> OpBreakdown:
        """
        Embedding lookup: pure gather, no FLOPs.

        Memory: read indices (B*S int64) + table (V*H) + write output (B*S*H).
        """
        B, S, H, V = self.B, self.S, self.H, self.V
        d = self.training.dtype_bytes

        mem = B * S * 8 + V * H * d + B * S * H * d
        return OpBreakdown(flops=0, mem_bytes=mem)

    @abstractmethod
    def _attn_breakdown(self) -> OpBreakdown:
        """Attention FLOPs and HBM traffic. Subclass decides flash vs standard."""

    @abstractmethod
    def _mlp_breakdown(self) -> OpBreakdown:
        """SwiGLU MLP FLOPs and HBM traffic. Subclass decides fusion."""

    @abstractmethod
    def _norm_breakdown(self) -> OpBreakdown:
        """RMSNorm FLOPs and HBM traffic. Subclass decides fusion."""

    @abstractmethod
    def _lm_head_breakdown(self) -> OpBreakdown:
        """LM head projection. Subclass decides chunking."""

    @abstractmethod
    def _loss_breakdown(self) -> OpBreakdown:
        """Cross-entropy loss. Subclass decides fusion with LM head."""

    def time_forward_ms(self) -> float:
        """Forward pass time on a single GPU (no communication)."""
        total = self.roofline_time_ms(*self._embedding_breakdown())
        for _ in range(self.N):
            total += self.roofline_time_ms(*self._norm_breakdown())
            total += self.roofline_time_ms(*self._attn_breakdown())
            total += self.roofline_time_ms(*self._norm_breakdown())
            total += self.roofline_time_ms(*self._mlp_breakdown())
        total += self.roofline_time_ms(*self._norm_breakdown())  # final norm
        total += self.roofline_time_ms(*self._lm_head_breakdown())
        total += self.roofline_time_ms(*self._loss_breakdown())
        return total

    def time_backward_ms(self) -> float:
        """Backward ~ 2x forward (same ops, but both input and weight grads)."""
        return 2.0 * self.time_forward_ms()

    def time_forward_backward_ms(self) -> float:
        return self.time_forward_ms() + self.time_backward_ms()

    # ------------------------------------------------------------------
    # Communication (theoretical lower bound)
    #
    # These estimates assume ideal bandwidth utilization with no
    # software overhead, no kernel launch gaps, and no contention.
    # Honestly, I didn't have time to make it close with real time.
    # ------------------------------------------------------------------

    @abstractmethod
    def communication_volume(self) -> int:
        """Total bytes through each GPU's link per step."""

    @abstractmethod
    def time_communication_ms(self) -> float:
        """Theoretical communication time (lower bound)."""

    @abstractmethod
    def overlap_efficiency(self) -> float:
        """Fraction of communication overlapped with compute (0-1)."""

    @abstractmethod
    def time_total_step_ms(self) -> float:
        """Total step = compute + (1 - overlap) * comm."""
