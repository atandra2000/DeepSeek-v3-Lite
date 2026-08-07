import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class AuxLossFreeGate(nn.Module):
    """Select routed experts with sigmoid scores and an auxiliary-loss-free bias."""
    def __init__(self, config: dict):
        super().__init__()
        self.dim = config["dim"]
        self.topk = config["n_activated_experts"]
        self.n_routed_experts = config["n_routed_experts"]
        self.route_scale = config.get("route_scale", 1.0)
        self.bias_upper = config.get("bias_upper_threshold", 0.10)
        self.bias_lower = config.get("bias_lower_threshold", 0.10)
        self.weight = nn.Parameter(torch.empty(self.n_routed_experts, self.dim))
        nn.init.normal_(self.weight, std=0.006)
        self.register_buffer("bias", torch.zeros(self.n_routed_experts, dtype=torch.float32))

    @torch.no_grad()
    def update_bias(self, counts: torch.Tensor, speed: float = 0.001) -> None:
        """Nudge expert biases away from over- or under-utilized experts."""
        counts = counts.float()
        avg = counts.mean()
        self.bias[counts > avg * (1.0 + self.bias_upper)] -= speed
        self.bias[counts < avg * (1.0 - self.bias_lower)] += speed

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return normalized top-k weights and their expert indices."""
        T = x.size(0)
        scores = F.linear(x, self.weight).sigmoid()
        biased = scores + self.bias.to(scores.dtype)
        indices = biased.topk(self.topk, dim=-1)[1]
        weights = scores.gather(1, indices)
        weights = (weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-10) * self.route_scale).to(x.dtype)
        return weights, indices


class Expert(nn.Module):
    """Single SwiGLU expert: ``W2(silu(W1(x)) * W3(x))``."""
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the expert's gated feed-forward transformation."""
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DeepSeekMoE(nn.Module):
    """DeepSeekMoE with routed experts, shared experts, and bias balancing."""
    def __init__(self, config: dict):
        super().__init__()
        self.dim = config["dim"]
        self.n_routed_experts = config["n_routed_experts"]
        self.n_shared_experts = config["n_shared_experts"]
        self.moe_inter_dim = config["moe_inter_dim"]
        # The grouped kernel is limited by register pressure at larger widths.
        self.moe_dispatch = config.get("moe_dispatch", "stacked")
        self.gate = AuxLossFreeGate(config)
        self.experts = nn.ModuleList([Expert(self.dim, self.moe_inter_dim) for _ in range(self.n_routed_experts)])
        self.shared_experts = nn.ModuleList([Expert(self.dim, self.moe_inter_dim) for _ in range(self.n_shared_experts)])
        self._stacked_w1: Optional[torch.Tensor] = None
        self._stacked_w2: Optional[torch.Tensor] = None
        self._stacked_w3: Optional[torch.Tensor] = None
        self._shared_w1: Optional[torch.Tensor] = None
        self._shared_w2: Optional[torch.Tensor] = None
        self._shared_w3: Optional[torch.Tensor] = None
        self._last_weights: Optional[torch.Tensor] = None
        self._last_indices: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route flattened tokens through routed and shared experts."""
        shape = x.shape
        flat = x.view(-1, self.dim)
        T = flat.size(0)
        weights, indices = self.gate(flat)
        self._last_weights = weights.detach()
        self._last_indices = indices.detach()
        E, I, D = self.n_routed_experts, self.moe_inter_dim, self.dim
        # Rebuild views after optimizer steps; cached copies would become stale.
        self._stacked_w1 = torch.stack([ex.w1.weight for ex in self.experts], dim=0).to(device=flat.device, dtype=flat.dtype)
        self._stacked_w2 = torch.stack([ex.w2.weight for ex in self.experts], dim=0).to(device=flat.device, dtype=flat.dtype)
        self._stacked_w3 = torch.stack([ex.w3.weight for ex in self.experts], dim=0).to(device=flat.device, dtype=flat.dtype)
        dispatch = self.moe_dispatch
        if dispatch == "triton_grouped":
            try:
                y_routed = self._routed_forward_triton(flat, indices, weights)
            except (ImportError, ValueError) as exc:
                # Keep the fallback quiet after the first backend warning.
                if not getattr(self, "_triton_fallback_warned", False):
                    print(f"[moe] triton_grouped unavailable ({type(exc).__name__}: {exc}); "
                          f"falling back to 'stacked' for this model.")
                    self._triton_fallback_warned = True
                y_routed = self._routed_forward_stacked(flat, indices, weights)
        else:
            y_routed = self._routed_forward_stacked(flat, indices, weights)
        y = y_routed
        if self.shared_experts:
            y = y + self._shared_forward(flat)
        return y.view(shape)

    def _routed_forward_stacked(self, flat: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Dispatch tokens through the portable per-expert implementation."""
        T = flat.size(0)
        flat_idx = indices.reshape(-1)
        flat_w = weights.reshape(-1)
        token_id = torch.arange(T, device=flat.device).repeat_interleave(indices.size(1))
        order = torch.argsort(flat_idx)
        sorted_token_ids = token_id[order]
        sorted_weights = flat_w[order]
        expert_counts = torch.bincount(flat_idx, minlength=self.n_routed_experts)
        expert_offsets = torch.cat([torch.zeros(1, dtype=expert_counts.dtype, device=flat.device), expert_counts.cumsum(0)[:-1]])
        counts_cpu = expert_counts.tolist()
        offsets_cpu = expert_offsets.tolist()
        y_routed = torch.zeros_like(flat)
        for e in range(self.n_routed_experts):
            cnt = counts_cpu[e]
            if cnt == 0:
                continue
            start = offsets_cpu[e]
            end = start + cnt
            chunk_tokens = sorted_token_ids[start:end]
            chunk_weights = sorted_weights[start:end]
            expert_in = flat[chunk_tokens]
            gate = expert_in @ self._stacked_w1[e].t()
            up = expert_in @ self._stacked_w3[e].t()
            h = torch.nn.functional.silu(gate) * up
            out = h @ self._stacked_w2[e].t()
            y_routed = y_routed.index_add(0, chunk_tokens, out * chunk_weights.unsqueeze(-1))
        return y_routed

    def _routed_forward_triton(self, flat: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Dispatch sorted tokens through the optional grouped-GEMM kernel."""
        from .moe_triton import triton_grouped_moe_dispatch
        T = flat.size(0)
        flat_idx = indices.reshape(-1)
        flat_w = weights.reshape(-1)
        token_id = torch.arange(T, device=flat.device).repeat_interleave(indices.size(1))
        order = torch.argsort(flat_idx)
        sorted_token_ids = token_id[order]
        sorted_weights_1d = flat_w[order]
        expert_counts = torch.bincount(flat_idx, minlength=self.n_routed_experts)
        expert_offsets = torch.cat(
            [torch.zeros(1, dtype=expert_counts.dtype, device=flat.device), expert_counts.cumsum(0)[:-1]]
        )

        x_sorted = flat[sorted_token_ids].contiguous()

        # Keep weights in the input dtype; the kernel casts gradient buffers back.
        y_sorted = triton_grouped_moe_dispatch(
            x_sorted=x_sorted,
            w1=self._stacked_w1,
            w2=self._stacked_w2,
            w3=self._stacked_w3,
            sorted_weights=sorted_weights_1d,
            expert_offsets=expert_offsets,
        )
        # Restore the original token order after grouped dispatch.
        y_routed = torch.zeros_like(flat)
        y_routed.index_add_(0, sorted_token_ids, y_sorted)
        return y_routed

    def _shared_forward(self, flat: torch.Tensor) -> torch.Tensor:
        """Run all shared experts with one batched matrix multiply per projection."""
        if self.n_shared_experts == 0:
            return torch.zeros_like(flat)
        # Rebuild stacked weights so optimizer updates are visible immediately.
        self._shared_w1 = torch.stack([e.w1.weight for e in self.shared_experts], dim=0).to(device=flat.device, dtype=flat.dtype)
        self._shared_w2 = torch.stack([e.w2.weight for e in self.shared_experts], dim=0).to(device=flat.device, dtype=flat.dtype)
        self._shared_w3 = torch.stack([e.w3.weight for e in self.shared_experts], dim=0).to(device=flat.device, dtype=flat.dtype)
        E = self.n_shared_experts
        gate = torch.bmm(flat.unsqueeze(0).expand(E, -1, -1), self._shared_w1.transpose(-1, -2))
        up = torch.bmm(flat.unsqueeze(0).expand(E, -1, -1), self._shared_w3.transpose(-1, -2))
        h = torch.nn.functional.silu(gate) * up
        out = torch.bmm(h, self._shared_w2.transpose(-1, -2))
        return out.sum(dim=0)

    def get_load_balance_loss(self) -> torch.Tensor:
        """Return the diagnostic load-balance loss from the latest forward."""
        if self._last_weights is None or self._last_indices is None:
            return torch.tensor(0.0, device=self.gate.weight.device)
        weights = self._last_weights
        indices = self._last_indices
        T = weights.size(0)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).float()
        f = counts / counts.sum().clamp(min=1e-10)
        one_hot = F.one_hot(indices.flatten(), num_classes=self.n_routed_experts).float()
        P = (one_hot * weights.flatten().unsqueeze(-1)).view(T, -1, self.n_routed_experts).sum(dim=1).mean(dim=0)
        return (f * P).sum() * self.n_routed_experts

    def update_gate_bias(self, speed: float = 0.001) -> None:
        """Update routing bias using counts captured by the latest forward."""
        if self._last_indices is None:
            return
        # Keep counts on the bias's device: boolean indexing in update_bias
        # requires the mask and self.bias to share a device.
        counts = torch.bincount(self._last_indices.flatten(), minlength=self.n_routed_experts)
        self.gate.update_bias(counts, speed=speed)
