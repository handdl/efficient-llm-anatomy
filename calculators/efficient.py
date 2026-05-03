"""
Efficient calculator: FSDP with fused/optimized operations.

Assumes:
    - FSDP: params, gradients, optimizer states all sharded across G GPUs.
    - FlashAttention (via SDPA): no S*S matrix in HBM, saves Q/K/V/O only.
    - Fused SwiGLU (Triton kernel): single kernel for activation, saves
      x/gate/up only -- backward recomputes the activation function.
    - Compiled RMSNorm (torch.compile): single fused kernel, saves only
      the bf16 input -- rsqrt is recomputed in backward.
    - Fused linear cross-entropy (Liger): processes lm_head + CE in chunks,
      never materializes the full B*S*V logits tensor. Forward computes
      all three matmuls (logits, grad_input, grad_weight) -- backward is
      essentially free for the CE part.
    - AdEMAMix: 3 states in model dtype, sharded with FSDP.

Communication:
    Forward:  all-gather params per layer (prefetch next while computing current).
    Backward: all-gather params again + reduce-scatter gradients per layer.
    Total: 3 * (G-1)/G * params * dtype  per step.

    These are theoretical lower bounds -- see base.py docstring for caveats.
"""

from calculators.base import BaseCalculator, OpBreakdown


def _liger_chunk_size(batch_tokens: int, vocab_size: int, hidden_dim: int) -> int:
    """
    Reproduce Liger's chunk size selection for fused linear CE.

    Liger splits B*(S-1) tokens into chunks, each processing chunk_size
    tokens against the full vocab. The chunk size is the smallest power
    of two that fits ceil(V / H) chunks into B*(S-1) tokens.
    """
    num_chunks = -(-vocab_size // hidden_dim)  # ceil(V / H)
    tokens_per_chunk = -(-batch_tokens // num_chunks)  # ceil(BT / num_chunks)
    return 1 << (tokens_per_chunk - 1).bit_length()  # next power of two


class EfficientCalculator(BaseCalculator):
    def param_memory(self) -> int:
        """FSDP: each GPU holds 1/G of parameters in bf16."""
        return self.total_params() * self.training.dtype_bytes // self.G

    def gradient_memory(self) -> int:
        """After reduce-scatter, each GPU holds 1/G of gradients."""
        return self.total_params() * self.training.dtype_bytes // self.G

    def optimizer_memory(self) -> int:
        """
        AdEMAMix: 3 states in model dtype, sharded.

        Total per GPU: 3 * P * dtype / G.
        """
        return 3 * self.total_params() * self.training.dtype_bytes // self.G

    def _attn_saved_bytes(self) -> int:
        """
        FlashAttention saves only the inputs and output -- no S*S matrix:

            - x (input to QKV projection)    B*S*H
            - Q, K, V (after projection)     3*B*S*H
            - O (attention output)           B*S*H

        Log-sum-exp (B*h*S) is negligible and omitted.
        Total: 5*B*S*H * dtype.
        """
        B, S, H = self.B, self.S, self.H
        return 5 * B * S * H * self.training.dtype_bytes

    def _mlp_saved_bytes(self) -> int:
        """
        Fused SwiGLU saves only what can't be cheaply recomputed:

            - x (MLP input, needed for weight grads)   B*S*H
            - gate (pre-activation, for backward)      B*S*I
            - up (pre-activation, for backward)        B*S*I

        The activation function (sigmoid, multiply, +1) is recomputed
        in the backward kernel -- costs FLOPs but saves 4*B*S*I memory.
        Total: (B*S*H + 2*B*S*I) * dtype.
        """
        B, S, H, I = self.B, self.S, self.H, self.I
        return (B * S * H + 2 * B * S * I) * self.training.dtype_bytes

    def _norm_saved_bytes(self) -> int:
        """
        Compiled RMSNorm saves x (bf16) and rsqrt (fp32, B*S*1 -- negligible).
        Recomputes x_normalized in backward instead of saving it.
        Reduces saved-tensor memory from 2*B*S*H*4 (fp32) to B*S*H*2 (bf16) -- ~4x reduction.
        """
        B, S, H = self.B, self.S, self.H
        return B * S * H * self.training.dtype_bytes

    def _non_layer_saved_bytes(self) -> int:
        """
        Non-layer saved tensors with fused CE:

            - Embedding indices              B*S * 8   (int64)
            - Final norm input               B*S*H * dtype
            - Fused CE hidden states input   B*S*H * dtype
            - Fused CE labels                B*S * 8   (int64)

        No logits tensor -- fused CE never materializes B*S*V.
        """
        B, S, H = self.B, self.S, self.H
        d = self.training.dtype_bytes
        return (
            B * S * 8  # embedding indices
            + B * S * H * d  # final norm input
            + B * S * H * d  # CE hidden states
            + B * S * 8  # CE labels
        )

    def _peak_transient_bytes(self) -> int:
        """
        Two candidates for peak transient, plus FSDP buffer overhead:

        1. Fused chunked CE: processes chunk_size tokens at a time.
           Temporary: chunk_size * V logits + B*S*H input grad + V*H weight grad.

        2. FlashAttention backward: needs grad buffers for O and Q/K/V.
           Temporary: 4*B*S*H.

        FSDP adds:
            - Unsharded root units (embedding + lm_head) always live
              because fused CE needs both: 2*V*H * dtype.
            - When G > 1: current layer unsharded + its gradients +
              prefetched next layer = (2*largest_unit + layer_params) * dtype.
        """
        B, S, H, V, I = self.B, self.S, self.H, self.V, self.I
        d = self.training.dtype_bytes

        # Candidate 1: fused chunked CE ---
        batch_tokens = B * (S - 1)  # after label shift
        chunk = _liger_chunk_size(batch_tokens, V, H)
        ce_peak = (
            chunk * V * d  # chunk logits
            + B * S * H * d  # input grad (accumulated)
            + V * H * d  # weight grad (accumulated)
        )

        # Candidate 2: flash attention backward ---
        flash_peak = 4 * B * S * H * d

        # FSDP overhead: embedding + lm_head must be unsharded for fused CE
        root_unsharded = 2 * V * H * d

        if self.G > 1:
            layer_params = 4 * H * H + 3 * H * I + 2 * H
            embed_params = V * H
            lm_head_params = V * H
            largest = max(layer_params, embed_params, lm_head_params)
            # Current unit unsharded + grads + prefetched next unit
            fsdp_buffers = (2 * largest + layer_params) * d
        else:
            fsdp_buffers = 0

        return max(ce_peak, flash_peak) + root_unsharded + fsdp_buffers

    def _attn_breakdown(self) -> OpBreakdown:
        """
        FlashAttention: same FLOPs as standard, but radically less HBM traffic.

        FLOPs (identical to standard attention):
            QKV + O projections: 4 * 2*B*S*H^2
            Q@K.T and P@V:       2 * 2*B*S^2*H

        Memory (the key difference):
            FlashAttention tiles Q@K.T and softmax in SRAM, so the S*S
            score matrix never hits HBM. Only Q, K, V, O are read/written.
            HBM traffic ~ 4*B*S*H * dtype.

            This is a simplification -- actual flash attention traffic
            depends on tile sizes and is slightly higher. But the S*S
            term vanishes, which is what matters for long sequences.
        """
        B, S, H = self.B, self.S, self.H
        d = self.training.dtype_bytes

        flops = 4 * 2 * B * S * H * H + 2 * 2 * B * S * S * H  # QKV + O projections  # Q@K.T + P@V (same math, tiled)
        mem = 4 * B * S * H * d  # Q, K, V read + O write

        return OpBreakdown(flops, mem)

    def _mlp_breakdown(self) -> OpBreakdown:
        """
        Fused SwiGLU MLP.

        FLOPs: 3 matmuls = 6*B*S*H*I. Activation FLOPs negligible.

        Memory: fusion eliminates HBM round-trips for intermediates.
            gate + up:    read x twice (B*S*H) + weights (2*H*I), write gate+up (2*B*S*I)
            fused SwiGLU: read gate+up (2*B*S*I), write activation (B*S*I)
            down:         read activation (B*S*I) + weight (I*H), write output (B*S*H)
        """
        B, S, H, I = self.B, self.S, self.H, self.I
        d = self.training.dtype_bytes

        flops = 6 * B * S * H * I

        mem = (
            2 * B * S * H
            + 2 * H * I
            + 2 * B * S * I  # gate & up projections
            + 2 * B * S * I
            + B * S * I  # fused activation
            + B * S * I
            + I * H
            + B * S * H  # down projection
        ) * d

        return OpBreakdown(flops, mem)

    def _norm_breakdown(self) -> OpBreakdown:
        """
        Compiled RMSNorm: single fused kernel, one read + one write.

        FLOPs: ~4*B*S*H (square, reduce, rsqrt, scale) + 2*B*S (reduction).
        Memory: read input + write output = 2*B*S*H * dtype.
        """
        B, S, H = self.B, self.S, self.H
        d = self.training.dtype_bytes

        return OpBreakdown(
            flops=4 * B * S * H + 2 * B * S,
            mem_bytes=2 * B * S * H * d,
        )

    def _lm_head_breakdown(self) -> OpBreakdown:
        """
        LM head fused with cross-entropy (chunked over vocab).

        The full B*S*V logits tensor is never materialized. Instead,
        Liger processes chunks of tokens, computing logits + loss + grads
        per chunk. Each chunk reads the input slice and the full weight
        matrix.

        FLOPs: 2*B*S*H*V (same matmul, just chunked).

        Memory:
            Input read num_chunks times: num_chunks * B*S*H  (but only
            chunk_size rows per read, so effectively 2*B*S*H total).
            Weight read num_chunks times: num_chunks * H*V.
            Weight grad written once: H*V.

        Note: Liger actually computes all three matmuls (logits,
        grad_input, grad_weight) in the forward pass, so backward
        for this component is essentially free. We don't account for
        that here to keep the "backward ~ 2x forward" heuristic.
        """
        B, S, H, V = self.B, self.S, self.H, self.V
        d = self.training.dtype_bytes

        batch_tokens = B * (S - 1)
        chunk = _liger_chunk_size(batch_tokens, V, H)
        num_chunks = B * S // chunk

        flops = 2 * B * S * H * V

        mem = (
            2 * B * S * H  # input read twice (fwd + grad accumulation)
            + 2 * num_chunks * H * V  # weight read per chunk (fwd + grad)
            + H * V  # weight grad write
        ) * d

        return OpBreakdown(flops, mem)

    def _loss_breakdown(self) -> OpBreakdown:
        """
        Loss is computed inline with lm_head chunks (fused CE).

        FLOPs: ~5*B*S*V for the softmax/NLL part (max, exp, sum, sub, div).
        Memory: effectively zero additional HBM -- folded into lm_head chunks.
        """
        B, S, V = self.B, self.S, self.V
        return OpBreakdown(flops=5 * B * S * V, mem_bytes=0)

    def communication_volume(self) -> int:
        """
        FSDP per-step communication volume per GPU:

            Forward:  all-gather params     = (G-1)/G * P * dtype
            Backward: all-gather params     = (G-1)/G * P * dtype
                      reduce-scatter grads  = (G-1)/G * P * dtype
            Total: 3 * (G-1)/G * P * dtype.
        """
        param_bytes = self.total_params() * self.training.dtype_bytes
        return 3 * int(self.k * param_bytes)

    def time_communication_ms(self) -> float:
        """Theoretical lower bound: total volume / link bandwidth."""
        return self.communication_volume() / (self.gpu.interconnect_bandwidth_gbps * 1e9) * 1000

    def overlap_efficiency(self) -> float:
        """
        FSDP overlap model.

        Forward: all-gather of layer N+1 overlaps compute of layer N.
        Backward: all-gather of layer N-1 + reduce-scatter of layer N
                  both overlap backward compute of layer N.

        Since backward ~ 2x forward in both compute and communication,
        the overlap ratio is the same -- we compute it for one direction.
        """
        attn = self._attn_breakdown()
        mlp = self._mlp_breakdown()
        norm = self._norm_breakdown()
        emb = self._embedding_breakdown()
        lm = self._lm_head_breakdown()
        loss = self._loss_breakdown()

        fwd_flops = self.N * (attn.flops + mlp.flops + 2 * norm.flops) + emb.flops + norm.flops + lm.flops + loss.flops
        fwd_mem = (
            self.N * (attn.mem_bytes + mlp.mem_bytes + 2 * norm.mem_bytes)
            + emb.mem_bytes
            + norm.mem_bytes
            + lm.mem_bytes
            + loss.mem_bytes
        )

        # Per-direction communication (total / 3 since fwd uses 1 of 3 transfers)
        per_dir_bytes = self.communication_volume() / 3

        compute_ms = fwd_flops / (self.gpu.flops_bf16_tflops * 1e12) * 1000
        mem_ms = fwd_mem / (self.gpu.memory_bandwidth_gbps * 1e9) * 1000
        comm_ms = per_dir_bytes / (self.gpu.interconnect_bandwidth_gbps * 1e9) * 1000

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
