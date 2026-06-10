# models/moe.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class AuxLossFreeGate(nn.Module):
    """
    Auxiliary-Loss-Free Load Balancing Gate (DeepSeek-V3, Section 2.3.3).

    Routing decision
    ----------------
    Each token is assigned to the top-k experts by a biased score:

        biased_score_e = sigmoid(x @ W_e^T) + bias_e

    The bias is NOT used for final routing weights — only raw sigmoid scores are load
    normalised and used as weights.  This separates load balancing (via bias) from the gradient
    signal (via raw scores).

    Group-limited routing
    ---------------------
    When n_groups > 1 the experts are divided into n_groups equal groups.
    Only topk_groups groups are selected per token (node-limited routing).
    Within each selected group the top-`group_topk` biased scores are summed
    to produce a group score; the top groups by that score are activated.

    Bias update
    -----------
    After each optimiser step the caller should invoke update_bias() with the per-expert token counts from the
    last forward pass.  Experts that are over-loaded (count > avg * (1 + upper_threshold)) have their bias
    decreased; under-loaded experts have their bias increased.  The bias is stored as a plain buffer
    (not a Parameter) so it does not appear in optimiser state dicts.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.dim = config["dim"]
        self.topk = config["n_activated_experts"]
        self.n_routed_experts = config["n_routed_experts"]
        self.n_groups = config.get("n_expert_groups", 1)
        self.topk_groups = config.get("n_limited_groups", 1)
        self.route_scale = config.get("route_scale", 1.0)
        # Number of top experts per group considered for group score. DeepSeek-V3 uses 2
        self.group_topk = config.get("group_topk", 2)
        # Bias update thresholds: experts loaded outside [avg*(1-lo), avg*(1+hi)] have their bias adjusted.
        # Separate thresholds allow asymmetric hysteresis to prevent bias thrashing on noisy load estimates.
        self.bias_upper = config.get("bias_upper_threshold", 0.10)
        self.bias_lower = config.get("bias_lower_threshold", 0.10)
        self.weight = nn.Parameter(torch.empty(self.n_routed_experts, self.dim))
        nn.init.normal_(self.weight, std=0.006)
        # Bias registered as a buffer (not a Parameter):
        #   • included in state_dict  → persists across checkpoints
        #   • excluded from model.parameters() → not touched by the optimiser
        #   • updated manually via update_bias()
        self.register_buffer("bias", torch.zeros(self.n_routed_experts, dtype=torch.float32))

    @torch.no_grad()
    def update_bias(self, counts: torch.Tensor, speed: float = 0.001) -> None:
        """
        Adjust per-expert bias based on observed token counts.

        Args:
            counts: (n_routed_experts,) integer token counts from the last step.
                    Must be on CPU (avoids unnecessary device sync in the caller).
            speed:  step size for each bias adjustment.
        """
        counts = counts.float()
        avg = counts.mean()
        # Decrease bias for over-loaded experts → makes them less likely to be
        # selected in the next step, steering tokens to under-used experts.
        self.bias[counts > avg * (1.0 + self.bias_upper)] -= speed
        # Increase bias for under-loaded experts → attract more tokens.
        self.bias[counts < avg * (1.0 - self.bias_lower)] += speed

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (T, dim) — flattened token representations

        Returns:
            weights: (T, topk) — normalised routing weights (sum-to-1 per token,
                                  then scaled by route_scale)
            indices: (T, topk) — global expert indices
        """
        T = x.size(0)

        # Raw sigmoid scores — used for final weight computation
        scores = F.linear(x, self.weight).sigmoid()          # (T, E)

        # Biased scores — used for routing decision only
        biased = scores + self.bias.to(scores.dtype)         # (T, E)

        if self.n_groups > 1:
            experts_per_group = self.n_routed_experts // self.n_groups
            # (T, n_groups, experts_per_group)
            biased_grouped = biased.view(T, self.n_groups, experts_per_group)
            # Group score = sum of top-group_topk biased scores within each group
            group_scores = biased_grouped.topk(self.group_topk, dim=-1)[0].sum(dim=-1)
            # Select topk_groups groups per token
            top_groups = group_scores.topk(self.topk_groups, dim=-1)[1]  # (T, topk_groups)
            # Mask out non-selected groups
            group_mask = torch.ones(T, self.n_groups, dtype=torch.bool, device=x.device)
            group_mask.scatter_(1, top_groups, False)
            biased = biased_grouped.masked_fill(group_mask.unsqueeze(-1), float("-inf")).flatten(1)     # (T, E)

        # Select top-k experts by biased score
        indices = biased.topk(self.topk, dim=-1)[1]          # (T, topk)

        # Routing weights from raw (unbiased) scores at the selected positions
        weights = scores.gather(1, indices)                   # (T, topk)

        # Normalise so weights sum to 1 per token, then apply route_scale
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        weights = (weights * self.route_scale).to(x.dtype)

        return weights, indices


class Expert(nn.Module):
    """
    Single SwiGLU expert.

    FFN(x) = W2(silu(W1(x)) * W3(x))

    W1 and W3 are the gate/up projections (dim → inter_dim).
    W2 is the down projection (inter_dim → dim).
    """

    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DeepSeekMoE(nn.Module):
    """
    DeepSeekMoE with shared experts and aux-loss-free load balancing.
    Single-GPU BF16.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.dim              = config["dim"]
        self.n_routed_experts = config["n_routed_experts"]
        self.n_shared_experts = config["n_shared_experts"]
        self.moe_inter_dim    = config["moe_inter_dim"]
        # use_grouped accepts: "stacked" (bmm per SwiGLU, fastest),
        # True (per-expert loop), or False (per-token loop, debug only).
        self.use_grouped_mode = config.get("use_grouped", "stacked")

        self.gate = AuxLossFreeGate(config)

        # Instantiate all routed experts locally
        self.experts = nn.ModuleList(
            [Expert(self.dim, self.moe_inter_dim) for _ in range(self.n_routed_experts)])

        # Shared experts — always executed
        self.shared_experts = nn.ModuleList(
            [Expert(self.dim, self.moe_inter_dim) for _ in range(self.n_shared_experts)])

        # Lazy-init cache for stacked expert weights ("stacked" path only).
        # Populated on first forward call.
        self._stacked_w1: Optional[torch.Tensor] = None
        self._stacked_w2: Optional[torch.Tensor] = None
        self._stacked_w3: Optional[torch.Tensor] = None

        # Routing cache: populated during forward(), reused by auxiliary methods
        self._last_weights: Optional[torch.Tensor] = None
        self._last_indices: Optional[torch.Tensor] = None


    # ──────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_grouped_mode == "stacked":
            return self._forward_stacked(x)
        return self._forward_grouped(x)

    def _forward_grouped(self, x: torch.Tensor) -> torch.Tensor:
        """
        Grouped forward: gather (token, expert) pairs once, run per-expert
        SwiGLU matmuls, then scatter-add back.

        Per-expert boundaries computed via `bincount` + `cumsum` on GPU,
        moved to CPU with a single `.tolist()` — one sync per MoE layer.

        Each token dispatched to `topk` experts; inner call is one bmm
        per expert. Use `_forward_stacked` for a single bmm over all chunks.
        """
        shape = x.shape
        flat  = x.view(-1, self.dim)                  # (T, dim)
        T = flat.size(0)

        weights, indices = self.gate(flat)            # (T, topk), (T, topk)
        self._last_weights = weights.detach()
        self._last_indices = indices.detach()

        # Flatten (T, topk) -> (T*topk,)
        flat_idx = indices.reshape(-1)                # (T*topk,)
        flat_w   = weights.reshape(-1)                # (T*topk,)
        token_id = torch.arange(T, device=flat.device).repeat_interleave(
            indices.size(1)
        )                                             # (T*topk,)

        # Sort by expert id so all tokens for expert e are contiguous.
        order = torch.argsort(flat_idx)
        sorted_token_ids = token_id[order]            # (T*topk,)
        sorted_weights   = flat_w[order]              # (T*topk,)
        sorted_expert_id = flat_idx[order]             # (T*topk,) — needed to look up the right expert

        # Per-expert boundaries on the GPU. bincount gives the count
        # per expert; cumsum gives the start offset per expert. We
        # move these to CPU with a single .tolist() — the only device
        # sync in the whole forward.
        expert_counts  = torch.bincount(
            flat_idx, minlength=self.n_routed_experts
        )
        expert_offsets = torch.cat([
            torch.zeros(1, dtype=expert_counts.dtype, device=expert_counts.device),
            expert_counts.cumsum(0)[:-1],
        ])

        # Pre-extract (start, length) for every non-empty expert. Built
        # on CPU so the Python loop has zero per-iter device work.
        counts_list  = expert_counts.tolist()
        offsets_list = expert_offsets.tolist()
        chunks = [
            (offsets_list[e], counts_list[e])
            for e in range(self.n_routed_experts)
            if counts_list[e] > 0
        ]

        y_routed = torch.zeros_like(flat)
        for start, length in chunks:
            end = start + length
            chunk_tokens  = sorted_token_ids[start:end]
            chunk_weights = sorted_weights[start:end].unsqueeze(-1)   # (k, 1)
            expert_in     = flat[chunk_tokens]                          # (k, dim)
            # Look up the expert id for this chunk. All tokens in
            # [start, end) belong to the same expert (the sort guarantees
            # this), so we can read just the first element.
            e_id = int(sorted_expert_id[start].item())
            expert_out = self.experts[e_id](expert_in)                  # (k, dim)
            # Functional index_add keeps the autograd graph stable.
            y_routed = y_routed.index_add(
                0, chunk_tokens, expert_out * chunk_weights
            )

        if self.shared_experts:
            shared_out = torch.stack(
                [e(flat) for e in self.shared_experts], dim=0).sum(dim=0)
            y = y_routed + shared_out
        else:
            y = y_routed

        return y.view(shape)

    def _forward_stacked(self, x: torch.Tensor) -> torch.Tensor:
        """
        Stacked-expert forward: gather (token, expert) pairs, then one
        bmm per SwiGLU projection against stacked ``(E, inter, dim)``
        weights. Eliminates the per-expert Python loop.

        Per-token gathered input is ``expert_in_for_chunk[k, dim]``,
        weight is ``W[sorted_expert_id[k], :, :]``. Computed via bmm
        over contiguous chunks, then ``index_add`` into ``y_routed``.
        """
        shape = x.shape
        flat  = x.view(-1, self.dim)                  # (T, dim)
        T = flat.size(0)

        weights, indices = self.gate(flat)            # (T, topk), (T, topk)
        self._last_weights = weights.detach()
        self._last_indices = indices.detach()

        # Sort by expert id so contiguous per-expert blocks
        flat_idx = indices.reshape(-1)                # (T*topk,)
        flat_w   = weights.reshape(-1)                # (T*topk,)
        token_id = torch.arange(T, device=flat.device).repeat_interleave(
            indices.size(1)
        )
        order = torch.argsort(flat_idx)
        sorted_token_ids = token_id[order]            # (T*topk,)
        sorted_weights   = flat_w[order].unsqueeze(-1)   # (T*topk, 1)
        sorted_expert_id = flat_idx[order]                # (T*topk,)

        # Stacked weight (lazy init)
        E, I, D = self.n_routed_experts, self.moe_inter_dim, self.dim
        if (
            self._stacked_w1 is None
            or self._stacked_w1.device != flat.device
            or self._stacked_w1.dtype != flat.dtype
        ):
            self._stacked_w1 = torch.stack(
                [ex.w1.weight for ex in self.experts], dim=0
            ).to(device=flat.device, dtype=flat.dtype)  # (E, I, D)
            self._stacked_w2 = torch.stack(
                [ex.w2.weight for ex in self.experts], dim=0
            ).to(device=flat.device, dtype=flat.dtype)  # (E, I, D)
            self._stacked_w3 = torch.stack(
                [ex.w3.weight for ex in self.experts], dim=0
            ).to(device=flat.device, dtype=flat.dtype)  # (E, I, D)

        # ── Per-expert boundaries ─────────────────────────────────────────
        expert_counts  = torch.bincount(
            flat_idx, minlength=self.n_routed_experts
        )
        expert_offsets = torch.cat([
            torch.zeros(1, dtype=expert_counts.dtype, device=flat.device),
            expert_counts.cumsum(0)[:-1],
        ])

        # ── Single bmm per SwiGLU projection ─────────────────────────────
        # For each (token, expert) pair, we need the per-expert SwiGLU
        # of the input. We can compute this as:
        #
        #   For each non-empty expert e:
        #     in_e  = flat[sorted_token_ids[offsets[e] : offsets[e]+counts[e]]]
        #            (k_e, D)
        #     gate  = in_e @ W1[e].T    (k_e, I)
        #     up    = in_e @ W3[e].T    (k_e, I)
        #     h     = silu(gate) * up   (k_e, I)
        #     out_e = h @ W2[e].T       (k_e, D)
        #     y_routed[sorted_token_ids[offsets[e]:...]] += out_e * weight
        #
        # The three matmuls can be one bmm per projection if we pack the
        # per-expert slices and the per-expert W rows together. That is
        # what this function does.
        #
        # Because the per-expert slice sizes are unequal, we still need
        # a Python loop to dispatch the bmms — but the loop is over
        # NON-EMPTY experts and each iteration is a single bmm, no
        # further Python work inside. We sync the per-expert counts
        # once via .tolist() and never sync again in the hot loop.
        counts_cpu  = expert_counts.tolist()
        offsets_cpu = expert_offsets.tolist()

        y_routed = torch.zeros_like(flat)
        for e in range(E):
            cnt = counts_cpu[e]
            if cnt == 0:
                continue
            start = offsets_cpu[e]
            end   = start + cnt
            chunk_tokens  = sorted_token_ids[start:end]
            chunk_weights = sorted_weights[start:end]                # (k, 1)
            expert_in     = flat[chunk_tokens]                       # (k, D)

            # One bmm per SwiGLU projection against the per-expert
            # weight row. This is the same math as the loop path, but
            # the bmm kernel is launched once instead of twice (gate,
            # up are two separate bmms because we don't have a fused
            # gate+up matmul kernel in plain PyTorch).
            gate = expert_in @ self._stacked_w1[e].t()              # (k, I)
            up   = expert_in @ self._stacked_w3[e].t()              # (k, I)
            h    = torch.nn.functional.silu(gate) * up               # (k, I)
            out  = h @ self._stacked_w2[e].t()                      # (k, D)

            y_routed = y_routed.index_add(
                0, chunk_tokens, out * chunk_weights
            )

        if self.shared_experts:
            shared_out = torch.stack(
                [e(flat) for e in self.shared_experts], dim=0).sum(dim=0)
            y = y_routed + shared_out
        else:
            y = y_routed

        return y.view(shape)

    # ──────────────────────────────────────────────────────────────────────
    # Auxiliary methods (reuse cached routing — no second gate call)
    # ──────────────────────────────────────────────────────────────────────

    def get_load_balance_loss(self) -> torch.Tensor:
        """
        Sequence-level load-balance auxiliary loss from DeepSeek-V3.

        L_bal = n_experts * Σ_e (f_e * P_e)

        where:
          f_e = fraction of tokens routed to expert e  (load)
          P_e = mean routing probability for expert e  (affinity)

        Minimising this encourages tokens to spread evenly across experts while keeping routing probabilities
        aligned with actual assignments. Requires a preceding forward() call in the same step to have
        populated the routing cache.  Returns zero if cache is empty.
        """
        if self._last_weights is None or self._last_indices is None:
            return torch.tensor(0.0, device=self.gate.weight.device)

        weights = self._last_weights  # (T, topk)
        indices = self._last_indices  # (T, topk)
        T       = weights.size(0)

        # f_e: per-expert token fraction — count each assignment once
        counts = torch.bincount(
            indices.flatten(), minlength=self.n_routed_experts).float()
        f = counts / counts.sum().clamp(min=1e-10)   # (E,)

        # P_e: mean routing probability for expert e across all tokens.
        # Build a (T, E) sparse assignment matrix, multiply by weights, then average over tokens.
        # one_hot: (T*topk, E) → reshaped and summed → (T, E)
        one_hot = F.one_hot(indices.flatten(), num_classes=self.n_routed_experts).float()
        # weights.flatten(): (T*topk,); weight each assignment by its routing weight
        P_dense = (one_hot * weights.flatten().unsqueeze(-1)).view(
            T, -1, self.n_routed_experts
        ).sum(dim=1)                                  # (T, E)
        P = P_dense.mean(dim=0)                      # (E,)

        return (f * P).sum() * self.n_routed_experts

    def get_routing_stats(self) -> Dict[str, torch.Tensor]:
        """
        Return per-expert routing statistics from the last forward pass.

        Useful for monitoring load imbalance during training without any
        additional gate computation.

        Returns a dict with:
          counts      (E,)  — integer number of token-expert assignments
          load        (E,)  — fraction of total assignments per expert
          mean_weight (E,)  — mean routing weight for each expert
          utilisation       — fraction of experts that received at least one token
        """
        if self._last_weights is None or self._last_indices is None:
            return {}

        weights = self._last_weights
        indices = self._last_indices
        E       = self.n_routed_experts

        counts      = torch.bincount(indices.flatten(), minlength=E).float()
        load        = counts / counts.sum().clamp(min=1e-10)

        one_hot     = F.one_hot(indices.flatten(), num_classes=E).float()
        weight_sum  = (one_hot * weights.flatten().unsqueeze(-1)).sum(dim=0)
        mean_weight = weight_sum / counts.clamp(min=1.0)   # avoid div-by-zero on empty experts

        utilisation = (counts > 0).float().mean()

        return {
            "counts":      counts,
            "load":        load,
            "mean_weight": mean_weight,
            "utilisation": utilisation,
        }

    def update_gate_bias(self, speed: float = 0.001) -> None:
        """
        Update the gate's load-balancing bias using cached token counts.
        Uses routing state from the last forward() without recomputation.
        """
        if self._last_indices is None:
            return
        counts = torch.bincount(
            self._last_indices.flatten().cpu(),
            minlength=self.n_routed_experts,
        )
        self.gate.update_bias(counts, speed=speed)
