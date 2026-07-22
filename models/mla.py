import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MultiHeadLatentAttention(nn.Module):
    """Multi-Head Latent Attention (MLA) from DeepSeek-V3."""
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
        # mscale: DeepSeek-V3 / YaRN-style softplus on the raw mscale, gated
        # by the YaRN rope_factor. When rope_factor > 1.0 the long-context
        # schedule multiplies the raw mscale by 0.1 * log(rope_factor) so
        # the attention logits grow as the context stretches; otherwise
        # the raw mscale is the multiplier. softmax_scale is then
        # `qk_head_dim**-0.5` scaled by `mscale**2` for long-context
        # (max_seq_len > 4096) and left at `qk_head_dim**-0.5` otherwise.
        mscale_raw = config.get("mscale", 1.0)
        if self.rope_factor > 1.0:
            self.mscale = 0.1 * mscale_raw * math.log(self.rope_factor)
        else:
            self.mscale = mscale_raw
        if self.max_seq_len > 4096:
            self.softmax_scale = (self.qk_head_dim ** -0.5) * (self.mscale ** 2)
        else:
            self.softmax_scale = self.qk_head_dim ** -0.5
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
        return torch.matmul(q.transpose(1, 2), k.unsqueeze(1).transpose(2, 3))

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

        if self.attn_impl == "triton":
            # Fused MLA materialise+RoPE+attn via models/mla_triton.py.
            # See AGENTS.md rule #1: this is one of the two sanctioned
            # Triton paths. Falls back to the SDPA path with a one-time
            # warning if the kernel is unavailable or a dim exceeds the
            # 256 cap.
            try:
                return self._forward_triton(
                    q_nope, q_pe, ctx_kv, ctx_pe, wkv_b_k, wkv_b_v,
                    bsz, seqlen, h, mask,
                )
            except (ImportError, ValueError) as exc:
                if not getattr(self, "_triton_fallback_warned", False):
                    print(f"[mla] triton attn_impl unavailable "
                          f"({type(exc).__name__}: {exc}); "
                          f"falling back to 'sdpa' for this model.")
                    self._triton_fallback_warned = True
                # Fall through to SDPA below
                pass
        if self.attn_impl == "sdpa":
            seqlen_k = ctx_kv.size(1)
            ctx_kv_bmm = ctx_kv.reshape(bsz * seqlen_k, self.kv_lora_rank).unsqueeze(0).expand(h, -1, -1)
            # ponytail: fuse the two bmm's over the same ctx_kv (one for K_nope, one for V) into a single matmul.
            wkv_b_kv = torch.cat([wkv_b_k, wkv_b_v], dim=1)
            KV_nope_h = torch.bmm(ctx_kv_bmm, wkv_b_kv.transpose(-1, -2))
            K_nope_h, V_h = KV_nope_h.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            K_nope = K_nope_h.reshape(h, bsz, seqlen_k, self.qk_nope_head_dim).permute(1, 0, 2, 3).contiguous()
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

    def _forward_triton(
        self,
        q_nope: torch.Tensor,        # (B, S_q, H, D_nope)
        q_pe: torch.Tensor,          # (B, S_q, H, D_rope)  RoPE-rotated
        ctx_kv: torch.Tensor,        # (B, S_kv, R)
        ctx_pe: torch.Tensor,        # (B, S_kv, D_rope)
        wkv_b_k: torch.Tensor,       # (H, D_nope, R)
        wkv_b_v: torch.Tensor,       # (H, D_v, R)
        bsz: int,
        seqlen: int,
        h: int,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Fused MLA materialise+RoPE+attn path. See models/mla_triton.py.

        Re-arranges the q_* tensors to (B, H, S_q, D) for the kernel,
        calls `triton_mla_attention`, then runs the `wo` output
        projection outside the kernel.
        """
        from .mla_triton import triton_mla_attention
        # Kernel layout: q_nope (B, H, S_q, D_nope), q_pe (B, H, S_q, D_rope)
        q_nope_k = q_nope.permute(0, 2, 1, 3).contiguous()
        q_pe_k = q_pe.permute(0, 2, 1, 3).contiguous()
        # Causal mask: when training (use_cache=False or prefill),
        # S_q == S_kv. The kernel applies the mask internally.
        is_causal = (mask is not None) and (seqlen == ctx_kv.size(1))
        out = triton_mla_attention(
            q_nope=q_nope_k,
            q_pe=q_pe_k,
            ctx_kv=ctx_kv,
            ctx_pe=ctx_pe,
            wkv_b_k=wkv_b_k,
            wkv_b_v=wkv_b_v,
            softmax_scale=self.softmax_scale,
            is_causal=is_causal,
        )
        # out: (B, S_q, H, D_v) — flatten to (B, S_q, H*D_v) for wo.
        return self.wo(out.flatten(2))
