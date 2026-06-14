import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MultiHeadLatentAttention(nn.Module):
    """Multi-Head Latent Attention (MLA) from DeepSeek-V3.
    Low-rank KV compression, decoupled RoPE, absorption trick."""
    def __init__(self, config: dict, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_impl = config.get("attn_impl", "sdpa")
        self.dim = config["dim"]
        self.n_heads = config["n_heads"]
        self.q_lora_rank = config["q_lora_rank"]
        self.kv_lora_rank = config["kv_lora_rank"]
        self.qk_nope_head_dim = config["qk_nope_head_dim"]
        self.qk_rope_head_dim = config["qk_rope_head_dim"]
        self.v_head_dim = config["v_head_dim"]
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.max_seq_len = config["max_seq_len"]
        self.n_local_heads = self.n_heads
        self.rope_theta = config["rope_theta"]
        self.rope_factor = config.get("rope_factor", 1.0)
        mscale_raw = config.get("mscale", 1.0)
        self.mscale = 0.1 * mscale_raw * math.log(self.rope_factor) + 1.0 if self.rope_factor > 1.0 else mscale_raw
        self.softmax_scale = self.qk_head_dim ** -0.5
        if self.max_seq_len > 4096 and self.mscale != 1.0:
            self.softmax_scale *= self.mscale ** 2
        if self.q_lora_rank > 0:
            self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False)
            self.q_norm = nn.RMSNorm(self.q_lora_rank, eps=1e-6)
            self.wq_b = nn.Linear(self.q_lora_rank, self.n_local_heads * self.qk_head_dim, bias=False)
        else:
            self.wq = nn.Linear(self.dim, self.n_local_heads * self.qk_head_dim, bias=False)
        self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.kv_lora_rank, eps=1e-6)
        self.wkv_b = nn.Linear(self.kv_lora_rank, self.n_local_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False)
        self.wo = nn.Linear(self.n_local_heads * self.v_head_dim, self.dim, bias=False)
        self._cache_batch: int = 0
        self.kv_cache: Optional[torch.Tensor] = None
        self.pe_cache: Optional[torch.Tensor] = None
        self._rope_seq_len: int = 0
        self.register_buffer("freqs_cis", torch.empty(0, self.qk_rope_head_dim // 2, dtype=torch.complex64), persistent=False)

    def _extend_rope(self, seq_len: int, device: torch.device) -> None:
        if seq_len <= self._rope_seq_len:
            return
        dim = self.qk_rope_head_dim
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        if self.rope_factor > 1.0:
            inv_freq = inv_freq / self.rope_factor
        grow_to = max(seq_len, self._rope_seq_len * 2, 64)
        grow_to = min(grow_to, self.max_seq_len)
        t = torch.arange(grow_to, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        self.freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        self._rope_seq_len = grow_to

    def _apply_rope(self, x: torch.Tensor, start_pos: int, seqlen: int) -> torch.Tensor:
        dtype = x.dtype
        x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs = self.freqs_cis[start_pos: start_pos + seqlen].view(1, seqlen, 1, -1)
        return torch.view_as_real(x_c * freqs).flatten(-2).to(dtype)

    def _per_batch_bmm(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        bsz, seqlen_q, h, d_k = q.shape
        seqlen_k = k.size(1)
        out = torch.empty(bsz, h, seqlen_q, seqlen_k, dtype=q.dtype, device=q.device)
        for b in range(bsz):
            q_h_b = q[b].permute(1, 0, 2).contiguous()
            k_b = k[b]
            out[b] = torch.bmm(q_h_b, k_b.t().unsqueeze(0).expand(h, -1, -1))
        return out

    def _ensure_cache(self, bsz: int, device: torch.device, dtype: torch.dtype) -> None:
        need_alloc = self.kv_cache is None or bsz > self._cache_batch or self.kv_cache.device != device or self.kv_cache.dtype != dtype
        if not need_alloc:
            return
        new_bsz = max(bsz, self._cache_batch * 2, 16)
        self.kv_cache = torch.zeros(new_bsz, self.max_seq_len, self.kv_lora_rank, device=device, dtype=dtype)
        self.pe_cache = torch.zeros(new_bsz, self.max_seq_len, self.qk_rope_head_dim, device=device, dtype=dtype)
        self._cache_batch = new_bsz

    def reset_cache(self) -> None:
        self.kv_cache = None
        self.pe_cache = None
        self._cache_batch = 0

    def prefill_cache(self, kv_latent: torch.Tensor, k_pe: torch.Tensor, start_pos: int) -> None:
        bsz, seqlen, _ = kv_latent.shape
        end_pos = start_pos + seqlen
        if end_pos > self.max_seq_len:
            raise ValueError(f"prefill_cache: end_pos {end_pos} > max_seq_len {self.max_seq_len}")
        self._extend_rope(end_pos, kv_latent.device)
        self._ensure_cache(bsz, kv_latent.device, kv_latent.dtype)
        self.kv_cache[:bsz, start_pos:end_pos] = kv_latent
        self.pe_cache[:bsz, start_pos:end_pos] = k_pe

    def forward(self, x: torch.Tensor, start_pos: int = 0, mask: Optional[torch.Tensor] = None, use_cache: bool = True) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        end_pos = start_pos + seqlen
        if end_pos > self.max_seq_len:
            raise RuntimeError(f"Layer {self.layer_idx}: end_pos {end_pos} exceeds max_seq_len {self.max_seq_len}")
        self._extend_rope(end_pos, x.device)
        if use_cache:
            self._ensure_cache(bsz, x.device, x.dtype)

        if self.q_lora_rank > 0:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        else:
            q = self.wq(x)
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = self._apply_rope(q_pe, start_pos, seqlen)

        kv_a = self.wkv_a(x)
        kv_latent, k_pe_raw = kv_a.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_normed = self.kv_norm(kv_latent)
        k_pe = self._apply_rope(k_pe_raw.unsqueeze(2), start_pos, seqlen).squeeze(2)

        if use_cache:
            self.kv_cache[:bsz, start_pos:end_pos] = kv_normed.detach()
            self.pe_cache[:bsz, start_pos:end_pos] = k_pe.detach()
            ctx_kv = self.kv_cache[:bsz, :end_pos]
            ctx_pe = self.pe_cache[:bsz, :end_pos]
        else:
            ctx_kv = kv_normed
            ctx_pe = k_pe

        wkv_b_full = self.wkv_b.weight.view(self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank)
        wkv_b_k = wkv_b_full[:, :self.qk_nope_head_dim]
        wkv_b_v = wkv_b_full[:, self.qk_nope_head_dim:]

        bsz, seqlen_q, h, d = q_nope.shape
        q_nope_h = q_nope.permute(2, 0, 1, 3).reshape(h, bsz * seqlen_q, d)
        q_nope_proj_h = torch.bmm(q_nope_h, wkv_b_k)
        q_nope_proj = q_nope_proj_h.reshape(h, bsz, seqlen_q, self.kv_lora_rank).permute(1, 2, 0, 3).contiguous()

        if self.attn_impl == "sdpa":
            seqlen_k = ctx_kv.size(1)
            ctx_kv_bmm = ctx_kv.reshape(bsz * seqlen_k, self.kv_lora_rank).unsqueeze(0).expand(h, -1, -1)
            wkv_b_k_t = wkv_b_k.transpose(-1, -2)
            K_nope_h = torch.bmm(ctx_kv_bmm, wkv_b_k_t)
            K_nope = K_nope_h.reshape(h, bsz, seqlen_k, self.qk_nope_head_dim).permute(1, 0, 2, 3).contiguous()
            wkv_b_v_t = wkv_b_v.transpose(-1, -2)
            V_h = torch.bmm(ctx_kv_bmm, wkv_b_v_t)
            V = V_h.reshape(h, bsz, seqlen_k, self.v_head_dim).permute(1, 0, 2, 3).contiguous()
            Q_nope = q_nope.transpose(1, 2)
            Q_rope = q_pe.transpose(1, 2)
            K_rope = ctx_pe.unsqueeze(1).expand(-1, h, -1, -1)
            attn_mask = mask.expand(bsz, h, seqlen_q, -1) if mask is not None else None
            attn = F.scaled_dot_product_attention(
                torch.cat([Q_nope, Q_rope], dim=-1), torch.cat([K_nope, K_rope], dim=-1), V,
                attn_mask=attn_mask, scale=self.softmax_scale)
            return self.wo(attn.transpose(1, 2).contiguous().flatten(2))

        scores_content = self._per_batch_bmm(q_nope_proj, ctx_kv)
        scores_rope = self._per_batch_bmm(q_pe, ctx_pe)
        scores = (scores_content + scores_rope) * self.softmax_scale
        if mask is not None:
            scores = scores + mask.expand(bsz, h, seqlen_q, -1)
        attn = scores.softmax(dim=-1, dtype=torch.float32).to(x.dtype)
        out_latent = torch.empty(bsz, h, seqlen_q, self.kv_lora_rank, dtype=x.dtype, device=x.device)
        for b in range(bsz):
            out_latent[b] = torch.bmm(attn[b], ctx_kv[b].unsqueeze(0).expand(h, -1, -1))
        out_h = out_latent.permute(1, 0, 2, 3).reshape(h, bsz * seqlen_q, self.kv_lora_rank)
        out_v = torch.bmm(out_h, wkv_b_v.transpose(-1, -2))
        out = out_v.reshape(h, bsz, seqlen_q, self.v_head_dim).permute(1, 2, 0, 3).contiguous()
        return self.wo(out.flatten(2))
