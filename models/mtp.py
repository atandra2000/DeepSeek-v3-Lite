from torch.nn.modules.module import Module
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class MTPBlock(nn.Module):
    """Single MTP block: independent norms → fusion → causal self-attention → SwiGLU FFN."""
    def __init__(self, config: dict):
        super().__init__()
        self.dim = config["dim"]
        n_heads = config["n_heads"]
        inter_dim = config["inter_dim"]
        self.norm_h = nn.RMSNorm(self.dim, eps=1e-6)
        self.norm_e = nn.RMSNorm(self.dim, eps=1e-6)
        self.proj = nn.Linear(self.dim * 2, self.dim, bias=False)
        self.norm_attn = nn.RMSNorm(self.dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(self.dim, n_heads, batch_first=True, bias=False)
        self.norm_ffn = nn.RMSNorm(self.dim, eps=1e-6)
        self.w1 = nn.Linear(self.dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, self.dim, bias=False)
        self.w3 = nn.Linear(self.dim, inter_dim, bias=False)
        self._causal_mask_size: int = 0
        self.register_buffer("_causal_mask", torch.empty(0, 0), persistent=False)

    def _get_causal_mask(self, seqlen: int, device: torch.device) -> torch.Tensor:
        if seqlen > self._causal_mask_size or self._causal_mask.device != device:
            mask = torch.triu(torch.full((seqlen, seqlen), float("-inf"), device=device), diagonal=1)
            self._causal_mask = mask
            self._causal_mask_size = seqlen
        return self._causal_mask[:seqlen, :seqlen]

    def forward(self, prev_hidden: torch.Tensor, target_emb: torch.Tensor) -> torch.Tensor:
        fused = self.proj(torch.cat([self.norm_h(prev_hidden), self.norm_e(target_emb)], dim=-1))
        seqlen = fused.size(1)
        attn_in = self.norm_attn(fused)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, attn_mask=self._get_causal_mask(seqlen, fused.device), is_causal=False)
        fused = fused + attn_out
        ffn_in = self.norm_ffn(fused)
        return fused + self.w2(F.silu(self.w1(ffn_in)) * self.w3(ffn_in))


class MTPModule(nn.Module):
    """MTP prediction head for depth d. Uses shared output_head set by MultiTokenPrediction."""
    def __init__(self, config: dict, depth: int = 1):
        super().__init__()
        self.depth = depth
        self.dim = config["dim"]
        self.vocab_size = config["vocab_size"]
        self.block = MTPBlock(config)
        self.norm = nn.RMSNorm(self.dim, eps=1e-6)
        self.output_head: Optional[nn.Linear] = None

    def set_output_head(self, head: nn.Linear) -> None:
        self.output_head = head

    def forward(self, prev_hidden: torch.Tensor, target_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.output_head is None:
            raise RuntimeError(f"MTPModule(depth={self.depth}): output_head not set.")
        if prev_hidden.shape != target_emb.shape:
            raise ValueError(f"Shape mismatch: {prev_hidden.shape} vs {target_emb.shape}")
        h = self.block(prev_hidden, target_emb)
        h_norm = self.norm(h)
        return self.output_head(h_norm), h_norm


class MultiTokenPrediction(nn.Module):
    """Wraps a Transformer with D MTP prediction heads. Shared head & embedding with main model."""
    def __init__(self, config: dict, main_model: nn.Module):
        super().__init__()
        self.main_model = main_model
        if "mtp_depth" in config:
            self.depth = config["mtp_depth"]
            self.mtp_weight = config.get("mtp_loss_weight", 0.3)
        else:
            mtp_cfg = config.get("mtp", {})
            self.depth = mtp_cfg.get("depth", 1)
            self.mtp_weight = mtp_cfg.get("weight", 0.3)
        model_cfg = config.get("model", config)
        self.mtp_modules = nn.ModuleList([MTPModule(model_cfg, d + 1) for d in range(self.depth)])
        self.add_module("embed", main_model.embed)
        shared_head = main_model.head
        for mtp in self.mtp_modules:
            mtp.set_output_head(shared_head)

    def forward(self, tokens: torch.Tensor, start_pos: int = 0) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Returns (main_logits, mtp_pairs) where each pair is (logits, targets) already length-aligned."""
        if tokens.dim() < 2:
            raise ValueError(f"Expected (bsz, seq) tokens, got {tokens.shape}")
        seq_len = tokens.size(1)
        main_logits, prev_h = self.main_model.forward_with_hidden(tokens)
        mtp_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for d, mtp in enumerate[Module](self.mtp_modules):
            usable = seq_len - d - 2
            if usable <= 0:
                break
            h_in = prev_h[:, :usable]
            emb_in = self.embed(tokens[:, d + 1: d + 1 + usable])
            tgt = tokens[:, d + 2: d + 2 + usable]
            logits, hidden = mtp(h_in, emb_in)
            mtp_pairs.append((logits, tgt))
            prev_h = hidden
        return main_logits, mtp_pairs

    def compute_loss(self, main_logits: torch.Tensor, targets: torch.Tensor,
                     mtp_pairs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (total_loss, main_loss, mtp_loss). MTP loss is mean across depths."""
        main_loss = F.cross_entropy(main_logits.reshape(-1, main_logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        if not mtp_pairs:
            return main_loss, main_loss, main_loss.new_zeros(())
        depth_losses: List[torch.Tensor] = []
        for logits, tgt in mtp_pairs:
            if tgt.numel() == 0:
                continue
            depth_losses.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-100))
        mtp_loss = torch.stack(depth_losses).mean() if depth_losses else main_loss.new_zeros(())
        return main_loss + self.mtp_weight * mtp_loss, main_loss, mtp_loss
