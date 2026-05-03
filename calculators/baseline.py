"""
Baseline calculator: DDP with standard (unfused) operations.

Assumes:
    - Full model replica on each GPU (no parameter sharding).
    - Standard attention (materializes B*h*S*S score matrix in HBM).
    - Unfused RMSNorm (multiple kernel launches, fp32 intermediates).
    - Unfused SwiGLU (each elementwise op saves its own intermediates).
    - Standard cross-entropy (materializes full B*S*V logits).
    - AdEMAMix optimizer: 3 states (m_fast, m_slow, v) in model dtype,
      no fp32 master weights.
    - Communication: DDP all-reduce on gradients during backward.
"""

from calculators.base import BaseCalculator, OpBreakdown


class BaselineCalculator(BaseCalculator):
    def param_memory(self) -> int:
        """Full params on each GPU in model dtype (bf16)."""
        return self.total_params() * self.training.dtype_bytes

    def gradient_memory(self) -> int:
        """Full gradients on each GPU in model dtype."""
        return self.total_params() * self.training.dtype_bytes

    def optimizer_memory(self) -> int:
        """
        AdEMAMix: 3 states per parameter in model dtype, unsharded.

        No fp32 master copy -- optimizer runs in bf16.
        Total: 3 * P * dtype_bytes.
        """
        return 3 * self.total_params() * self.training.dtype_bytes

    def _attn_saved_bytes(self) -> int:
        """
        Standard attention saves intermediates at every step:

            1. qkv_proj(x): input x                              B*S*H
            2. rope(q, k): saves q and k (cos/sin negligible)    2*B*S*H
            3. q * scale: scalar only                             --
            4. matmul(q_scaled, k.T): saves q_scaled              B*S*H
            5. masked_fill_: in-place                             --
            6. softmax: saves output probabilities                B*h*S*S
            7. dropout: bool mask only                            --
            8. matmul(attn_weights, v): softmax already saved     --
            9. out_proj(out): saves input                         B*S*H

        Total: 5*B*S*H + B*h*S*S  (all in model dtype).
        """
        B, S, H, h = self.B, self.S, self.H, self.h
        d = self.training.dtype_bytes
        return (5 * B * S * H + B * h * S * S) * d

    def _mlp_saved_bytes(self) -> int:
        """
        Unfused SwiGLU -- each op saves its own inputs/outputs:

            1. gate_proj(x): input x                              B*S*H
            2. up_proj(x): shares x pointer                       --
            3. gate.clamp(): pre-clamp gate                       B*S*I
            4. up.clamp(): pre-clamp up                           B*S*I
            5. gate * alpha: shares gate pointer                  --
            6. sigmoid(gate*alpha): sigmoid output                B*S*I
            7. gate * sigmoid: shares gate, sigmoid               B*S*I
            8. (up+1) * glu: saves (up+1) and glu                2*B*S*I
            9. down_proj(intermediate): input                     B*S*I

        Total: B*S*H + 7*B*S*I  (all in model dtype).
        """
        B, S, H, I = self.B, self.S, self.H, self.I
        d = self.training.dtype_bytes
        return (B * S * H + 7 * B * S * I) * d

    def _norm_saved_bytes(self) -> int:
        """
        Unfused RMSNorm upcasts to fp32 and keeps two intermediates:

            - x (original input, for backward)
            - x_normalized (after rsqrt scaling)

        Both stored in fp32: 2*B*S*H * 4 bytes.
        """
        B, S, H = self.B, self.S, self.H
        return 2 * B * S * H * 4  # fp32

    def _non_layer_saved_bytes(self) -> int:
        """
        Tensors saved outside the layer stack:

            - Embedding indices                    B*S * 8  (int64)
            - Final norm input                     B*S*H * dtype
            - LM head input                        B*S*H * dtype
            - CE softmax output (fp32 cast)        B*S*V * 4
        """
        B, S, H, V = self.B, self.S, self.H, self.V
        d = self.training.dtype_bytes
        return (
            B * S * 8  # embedding indices
            + B * S * H * d  # final norm input
            + B * S * H * d  # lm_head input
            + B * S * V * 4  # CE softmax in fp32
        )

    def _peak_transient_bytes(self) -> int:
        """
        Largest temporary allocation not captured in saved tensors.

        Three candidates (only the max coexists with saved tensors):

        1. Attention: score matrix + attn_weights + weight/activation grads
           = 2*B*h*S^2*d + 3*H^2*d + 2*B*S*H*d

        2. Cross-entropy: fp32 softmax probs + fp32 logit grads
           = 2*B*S*V * 4

        3. LM head backward: weight grad + input/output grads
           = B*V*d + B*S*H*d + B*S*V * 4

        Additionally, the full logits tensor (B*S*V fp32) is materialized
        during forward and persists until CE backward finishes, so it's
        added on top of the worst candidate.
        """
        B, S, H, V, h = self.B, self.S, self.H, self.V, self.h
        d = self.training.dtype_bytes

        attn_peak = (
            2 * B * h * S * S * d  # scores + attn_weights
            + 3 * H * H * d  # weight grads (one proj at a time)
            + 2 * B * S * H * d  # input + output activation grads
        )

        ce_peak = 2 * B * S * V * 4  # fp32 softmax + fp32 logit grads

        lm_head_peak = (
            B * V * d + B * S * H * d + B * S * V * 4  # weight grad accumulator  # input grad  # output grad in fp32
        )

        # Logits (B*S*V fp32) live from forward until CE backward.
        logits_persistent = B * S * V * 4

        return logits_persistent + max(attn_peak, ce_peak, lm_head_peak)

    def _attn_breakdown(self) -> OpBreakdown:
        """
        Standard multi-head attention (materializes S*S in HBM).

        FLOPs:
            QKV projections:  3 * 2*B*S*H^2
            Output projection:    2*B*S*H^2
            Score (Q @ K.T):      2*B*S^2*H
            Softmax:            ~5*B*h*S^2   (max, shift, exp, sum, div)
            Context (P @ V):      2*B*S^2*H

        Memory (HBM traffic -- read inputs + write outputs per op):
            QKV:     3 * (B*S*H + H^2 + B*S*H)*d
            Score:   (2*B*S*H + B*h*S^2)*d
            Softmax: 3*B*h*S^2*d               (reads twice: max pass + normalize)
            Context: (B*h*S^2 + B*S*H + B*S*H)*d
            Out:     (B*S*H + H^2 + B*S*H)*d
        """
        B, S, H, h = self.B, self.S, self.H, self.h
        d = self.training.dtype_bytes

        flops = (
            4 * 2 * B * S * H * H  # Q, K, V, O projections
            + 2 * 2 * B * S * S * H  # Q@K.T and P@V
            + 5 * B * h * S * S  # softmax
        )

        mem = (
            3 * (B * S * H + H * H + B * S * H) * d  # QKV projections
            + (2 * B * S * H + B * h * S * S) * d  # score matmul
            + 3 * B * h * S * S * d  # softmax (reads twice)
            + (B * h * S * S + 2 * B * S * H) * d  # context matmul
            + (B * S * H + H * H + B * S * H) * d  # output projection
        )

        return OpBreakdown(flops, mem)

    def _mlp_breakdown(self) -> OpBreakdown:
        """
        Unfused SwiGLU MLP.

        FLOPs:
            3 matmuls: gate, up, down = 3 * 2*B*S*H*I = 6*B*S*H*I
            Elementwise (sigmoid, muls, add, clamp) negligible vs matmuls.

        Memory:
            Each intermediate is read/written separately because nothing
            is fused. Every clamp, sigmoid, multiply is a separate kernel
            that round-trips through HBM.
        """
        B, S, H, I = self.B, self.S, self.H, self.I
        d = self.training.dtype_bytes

        flops = 6 * B * S * H * I

        mem = (
            (2 * B * S * H + 2 * H * I + 2 * B * S * I)  # gate & up projections
            + 4 * B * S * I  # clamp gate, clamp up
            + 2 * B * S * I  # gate * alpha
            + 2 * B * S * I  # sigmoid
            + 3 * B * S * I  # gate * sigmoid
            + 2 * B * S * I  # up + 1
            + 3 * B * S * I  # (up+1) * glu
            + (B * S * I + H * I + B * S * H)  # down_proj
        ) * d

        return OpBreakdown(flops, mem)

    def _norm_breakdown(self) -> OpBreakdown:
        """
        Unfused RMSNorm -- multiple kernel launches, fp32 intermediates.

        FLOPs: ~4*B*S*H (square, reduce, rsqrt, scale).

        Memory: each step is a separate kernel doing a full read+write
        pass through B*S*H elements, with fp32 upcasting in between.

            read bf16 input -> write fp32 copy
            read fp32 (square) -> write x^2
            read x^2 (mean) -> write mean
            read fp32 (normalize) -> write normalized
            read normalized (scale) -> write scaled fp32
            write bf16 output (downcast)
        """
        B, S, H = self.B, self.S, self.H
        d = self.training.dtype_bytes

        mem = (
            B * S * H * d  # read bf16 input
            + B * S * H * 4  # write fp32 copy
            + B * S * H * 4  # read fp32 for squaring
            + B * S * H * 4  # write x^2
            + B * S * H * 4  # read x^2 for mean
            + B * S * H * 4  # read fp32 for normalize
            + B * S * H * 4  # write normalized
            + B * S * H * 4  # read for scaling
            + B * S * H * 4  # write scaled fp32
            + B * S * H * d  # write bf16 output
        )

        return OpBreakdown(flops=4 * B * S * H, mem_bytes=mem)

    def _lm_head_breakdown(self) -> OpBreakdown:
        """
        Standard linear projection: hidden_states @ W.T -> logits.

        FLOPs: 2*B*S*H*V.
        Memory: read input (B*S*H) + weight (H*V), write output (B*S*V).
        """
        B, S, H, V = self.B, self.S, self.H, self.V
        d = self.training.dtype_bytes

        return OpBreakdown(
            flops=2 * B * S * H * V,
            mem_bytes=(B * S * H + H * V + B * S * V) * d,
        )

    def _loss_breakdown(self) -> OpBreakdown:
        """
        Standard cross-entropy over pre-materialized logits.

        FLOPs: ~5*B*S*V (max, shift, exp, sum, div).
        Memory: ~4 passes over B*S*V (read logits twice for softmax,
                write probs, read probs for NLL).
        """
        B, S, V = self.B, self.S, self.V
        d = self.training.dtype_bytes

        return OpBreakdown(
            flops=5 * B * S * V,
            mem_bytes=4 * B * S * V * d,
        )

    def communication_volume(self) -> int:
        """
        DDP all-reduce gradient volume per GPU.

        Ring all-reduce = reduce-scatter + all-gather, each transferring
        (G-1)/G of the gradient buffer.
        Total per GPU: 2 * (G-1)/G * grad_bytes.
        """
        grad_bytes = self.total_params() * self.training.dtype_bytes
        return int(2 * self.k * grad_bytes)

    def time_communication_ms(self) -> float:
        """Theoretical lower bound: volume / link bandwidth."""
        volume = self.communication_volume()
        return volume / (self.gpu.interconnect_bandwidth_gbps * 1e9) * 1000

    def overlap_efficiency(self) -> float:
        """
        How much DDP communication hides behind backward compute.

        DDP overlaps gradient all-reduce with backward. If backward
        compute time exceeds memory + communication time, everything
        is hidden (efficiency = 1.0). Otherwise, the fraction of
        communication that fits inside the compute-memory gap.
        """
        attn = self._attn_breakdown()
        mlp = self._mlp_breakdown()
        norm = self._norm_breakdown()
        emb = self._embedding_breakdown()
        lm = self._lm_head_breakdown()
        loss = self._loss_breakdown()

        # Backward does ~2x the work of forward
        bwd_flops = 2 * (
            self.N * (attn.flops + mlp.flops + 2 * norm.flops) + emb.flops + norm.flops + lm.flops + loss.flops
        )
        bwd_mem = 2 * (
            self.N * (attn.mem_bytes + mlp.mem_bytes + 2 * norm.mem_bytes)
            + emb.mem_bytes
            + norm.mem_bytes
            + lm.mem_bytes
            + loss.mem_bytes
        )

        compute_ms = bwd_flops / (self.gpu.flops_bf16_tflops * 1e12) * 1000
        mem_ms = bwd_mem / (self.gpu.memory_bandwidth_gbps * 1e9) * 1000
        comm_ms = self.time_communication_ms()

        if compute_ms >= mem_ms + comm_ms:
            return 1.0
        return max(0.0, (compute_ms - mem_ms) / comm_ms)

    def time_total_step_ms(self) -> float:
        """fwd + bwd + exposed communication."""
        fwd = self.time_forward_ms()
        bwd = self.time_backward_ms()
        comm = self.time_communication_ms()
        exposed = (1 - self.overlap_efficiency()) * comm
        return fwd + bwd + exposed
