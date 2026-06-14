import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class AuxLossFreeGate(nn.Module):
    """Auxiliary-Loss-Free Load Balancing Gate (DeepSeek-V3 §2.3.3).
    Routing via sigmoid + bias; bias updated separately (not a Parameter)."""
    def __init__(self, config: dict):
        super().__init__()
        self.dim = config["dim"]
        self.topk = config["n_activated_experts"]
        self.n_routed_experts = config["n_routed_experts"]
        self.n_groups = config.get("n_expert_groups", 1)
        self.topk_groups = config.get("n_limited_groups", 1)
        self.route_scale = config.get("route_scale", 1.0)
        self.group_topk = config.get("group_topk", 2)
        self.bias_upper = config.get("bias_upper_threshold", 0.10)
        self.bias_lower = config.get("bias_lower_threshold", 0.10)
        self.weight = nn.Parameter(torch.empty(self.n_routed_experts, self.dim))
        nn.init.normal_(self.weight, std=0.006)
        self.register_buffer("bias", torch.zeros(self.n_routed_experts, dtype=torch.float32))

    @torch.no_grad()
    def update_bias(self, counts: torch.Tensor, speed: float = 0.001) -> None:
        counts = counts.float()
        avg = counts.mean()
        self.bias[counts > avg * (1.0 + self.bias_upper)] -= speed
        self.bias[counts < avg * (1.0 - self.bias_lower)] += speed

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        T = x.size(0)
        scores = F.linear(x, self.weight).sigmoid()
        biased = scores + self.bias.to(scores.dtype)
        if self.n_groups > 1:
            experts_per_group = self.n_routed_experts // self.n_groups
            biased_grouped = biased.view(T, self.n_groups, experts_per_group)
            group_scores = biased_grouped.topk(self.group_topk, dim=-1)[0].sum(dim=-1)
            top_groups = group_scores.topk(self.topk_groups, dim=-1)[1]
            group_mask = torch.ones(T, self.n_groups, dtype=torch.bool, device=x.device)
            group_mask.scatter_(1, top_groups, False)
            biased = biased_grouped.masked_fill(group_mask.unsqueeze(-1), float("-inf")).flatten(1)
        indices = biased.topk(self.topk, dim=-1)[1]
        weights = scores.gather(1, indices)
        weights = (weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-10) * self.route_scale).to(x.dtype)
        return weights, indices


class Expert(nn.Module):
    """Single SwiGLU expert: W2(silu(W1(x)) * W3(x))."""
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DeepSeekMoE(nn.Module):
    """DeepSeekMoE with shared experts and aux-loss-free load balancing. Single-GPU BF16."""
    def __init__(self, config: dict):
        super().__init__()
        self.dim = config["dim"]
        self.n_routed_experts = config["n_routed_experts"]
        self.n_shared_experts = config["n_shared_experts"]
        self.moe_inter_dim = config["moe_inter_dim"]
        self.use_grouped_mode = config.get("use_grouped", "stacked")
        self.gate = AuxLossFreeGate(config)
        self.experts = nn.ModuleList([Expert(self.dim, self.moe_inter_dim) for _ in range(self.n_routed_experts)])
        self.shared_experts = nn.ModuleList([Expert(self.dim, self.moe_inter_dim) for _ in range(self.n_shared_experts)])
        self._stacked_w1: Optional[torch.Tensor] = None
        self._stacked_w2: Optional[torch.Tensor] = None
        self._stacked_w3: Optional[torch.Tensor] = None
        self._last_weights: Optional[torch.Tensor] = None
        self._last_indices: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_stacked(x) if self.use_grouped_mode == "stacked" else self._forward_grouped(x)

    def _forward_grouped(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.view(-1, self.dim)
        T = flat.size(0)
        weights, indices = self.gate(flat)
        self._last_weights = weights.detach()
        self._last_indices = indices.detach()
        flat_idx = indices.reshape(-1)
        flat_w = weights.reshape(-1)
        token_id = torch.arange(T, device=flat.device).repeat_interleave(indices.size(1))
        order = torch.argsort(flat_idx)
        sorted_token_ids = token_id[order]
        sorted_weights = flat_w[order]
        sorted_expert_id = flat_idx[order]
        expert_counts = torch.bincount(flat_idx, minlength=self.n_routed_experts)
        expert_offsets = torch.cat([torch.zeros(1, dtype=expert_counts.dtype, device=expert_counts.device), expert_counts.cumsum(0)[:-1]])
        counts_list = expert_counts.tolist()
        offsets_list = expert_offsets.tolist()
        chunks = [(offsets_list[e], counts_list[e]) for e in range(self.n_routed_experts) if counts_list[e] > 0]
        y_routed = torch.zeros_like(flat)
        for start, length in chunks:
            end = start + length
            chunk_tokens = sorted_token_ids[start:end]
            chunk_weights = sorted_weights[start:end].unsqueeze(-1)
            expert_in = flat[chunk_tokens]
            e_id = int(sorted_expert_id[start].item())
            y_routed = y_routed.index_add(0, chunk_tokens, self.experts[e_id](expert_in) * chunk_weights)
        y = y_routed
        if self.shared_experts:
            y = y + torch.stack([e(flat) for e in self.shared_experts], dim=0).sum(dim=0)
        return y.view(shape)

    def _forward_stacked(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.view(-1, self.dim)
        T = flat.size(0)
        weights, indices = self.gate(flat)
        self._last_weights = weights.detach()
        self._last_indices = indices.detach()
        flat_idx = indices.reshape(-1)
        flat_w = weights.reshape(-1)
        token_id = torch.arange(T, device=flat.device).repeat_interleave(indices.size(1))
        order = torch.argsort(flat_idx)
        sorted_token_ids = token_id[order]
        sorted_weights = flat_w[order].unsqueeze(-1)
        sorted_expert_id = flat_idx[order]
        E, I, D = self.n_routed_experts, self.moe_inter_dim, self.dim
        if self._stacked_w1 is None or self._stacked_w1.device != flat.device or self._stacked_w1.dtype != flat.dtype:
            self._stacked_w1 = torch.stack([ex.w1.weight for ex in self.experts], dim=0).to(device=flat.device, dtype=flat.dtype)
            self._stacked_w2 = torch.stack([ex.w2.weight for ex in self.experts], dim=0).to(device=flat.device, dtype=flat.dtype)
            self._stacked_w3 = torch.stack([ex.w3.weight for ex in self.experts], dim=0).to(device=flat.device, dtype=flat.dtype)
        expert_counts = torch.bincount(flat_idx, minlength=self.n_routed_experts)
        expert_offsets = torch.cat([torch.zeros(1, dtype=expert_counts.dtype, device=expert_counts.device), expert_counts.cumsum(0)[:-1]])
        counts_cpu = expert_counts.tolist()
        offsets_cpu = expert_offsets.tolist()
        y_routed = torch.zeros_like(flat)
        for e in range(E):
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
            y_routed = y_routed.index_add(0, chunk_tokens, out * chunk_weights)
        y = y_routed
        if self.shared_experts:
            y = y + torch.stack([e(flat) for e in self.shared_experts], dim=0).sum(dim=0)
        return y.view(shape)

    def get_load_balance_loss(self) -> torch.Tensor:
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

    def get_routing_stats(self) -> Dict[str, torch.Tensor]:
        if self._last_weights is None or self._last_indices is None:
            return {}
        weights = self._last_weights
        indices = self._last_indices
        E = self.n_routed_experts
        counts = torch.bincount(indices.flatten(), minlength=E).float()
        load = counts / counts.sum().clamp(min=1e-10)
        one_hot = F.one_hot(indices.flatten(), num_classes=E).float()
        mean_weight = (one_hot * weights.flatten().unsqueeze(-1)).sum(dim=0) / counts.clamp(min=1.0)
        return {"counts": counts, "load": load, "mean_weight": mean_weight, "utilisation": (counts > 0).float().mean()}

    def update_gate_bias(self, speed: float = 0.001) -> None:
        if self._last_indices is None:
            return
        counts = torch.bincount(self._last_indices.flatten().cpu(), minlength=self.n_routed_experts)
        self.gate.update_bias(counts, speed=speed)
