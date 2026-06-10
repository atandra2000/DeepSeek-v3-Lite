# models/mla.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) from DeepSeek-V3.

    Key ideas
    ---------
    • Low-rank KV compression: the KV cache stores the normalised latent
      c_KV ∈ R^{kv_lora_rank} instead of full per-head K/V tensors, giving
      a (n_heads * (qk_nope_head_dim + v_head_dim)) / kv_lora_rank ≈ 10–20×
      reduction in KV-cache memory.

    • Decoupled RoPE: positional encodings are applied only to the
      qk_rope_head_dim slice of Q/K.  The nope slice carries content-based
      similarity; the rope slice carries positional similarity.  The two
      score contributions are summed before softmax.

    • Absorption trick: instead of expanding c_KV → (K_nope, V) at every
      step, wkv_b is absorbed into q_nope at query time, so attention scores
      are computed directly against the cached latent.  V is also recovered
      from the latent post-softmax.  This avoids materialising full K/V
      tensors during decode.

    • YaRN-compatible softmax scaling: for long contexts, the softmax scale is
      multiplied by mscale > 1 to prevent underflow.
    """

    def __init__(
        self,
        config: dict,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_impl = config.get("attn_impl", "sdpa")  # "sdpa" or "manual"

        # ── Dimensions ────────────────────────────────────────────────────

        self.dim              = config["dim"]                    # Model hidden size (2048)
        self.n_heads          = config["n_heads"]                # Total attention heads (16)
        self.q_lora_rank      = config["q_lora_rank"]            # 0 means no LoRA compression fot Q
        self.kv_lora_rank     = config["kv_lora_rank"]           # 512 - the key compression dimention
        self.qk_nope_head_dim = config["qk_nope_head_dim"]       # 128 - content based Q/K dimention
        self.qk_rope_head_dim = config["qk_rope_head_dim"]       # 64 - positional Q/K dimention
        self.v_head_dim       = config["v_head_dim"]             # 128 - value head dimention
        self.qk_head_dim      = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.max_seq_len      = config["max_seq_len"]

        self.n_local_heads = self.n_heads

        # ── RoPE config ───────────────────────────────────────────────────

        self.rope_theta  = config["rope_theta"]
        self.rope_factor = config.get("rope_factor", 1.0)

        mscale_raw  = config.get("mscale", 1.0)
        self.mscale = (
            0.1 * mscale_raw * math.log(self.rope_factor) + 1.0
            if self.rope_factor > 1.0
            else mscale_raw
        )

        # ── Softmax scale ─────────────────────────────────────────────────

        # Base: 1/sqrt(qk_head_dim). YaRN corrects for extended contexts by multiplying by mscale^2

        self.softmax_scale = self.qk_head_dim ** -0.5
        if self.max_seq_len > 4096 and self.mscale != 1.0:
            self.softmax_scale *= self.mscale ** 2

        # ── Query projections ──────────────────────────────────────────────

        # When q_lora_rank > 0 the query is produced via a low-rank
        # bottleneck: x → (wq_a) → q_lora_rank → (RMSNorm) → (wq_b) → Q.
        # wq_b output is sized for local heads (tensor parallelism ready).

        if self.q_lora_rank > 0:
            self.wq_a   = nn.Linear(self.dim, self.q_lora_rank, bias=False)
            self.q_norm = nn.RMSNorm(self.q_lora_rank, eps=1e-6)
            self.wq_b   = nn.Linear(
                self.q_lora_rank,
                self.n_local_heads * self.qk_head_dim,
                bias=False,
            )
        else:
            self.wq = nn.Linear(
                self.dim,
                self.n_local_heads * self.qk_head_dim,
                bias=False,
            )

        # ── KV projections with latent compression ─────────────────────────

        # wkv_a projects x → (latent ‖ k_rope).  Replicated across all ranks
        # because the latent must be written to the KV cache in full — sharding
        # it would require a gather before every cache write.

        self.wkv_a  = nn.Linear(
            self.dim,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
        )
        self.kv_norm = nn.RMSNorm(self.kv_lora_rank, eps=1e-6)

        # wkv_b expands latent → (K_nope ‖ V) for local heads only.
        self.wkv_b = nn.Linear(
            self.kv_lora_rank,
            self.n_local_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        # Output projection: (n_local_heads * v_head_dim) → dim.
        # Row-parallel in tensor-parallel setups; plain linear here.

        self.wo = nn.Linear(self.n_local_heads * self.v_head_dim, self.dim, bias=False)

        # ── KV cache ──────────────────────────────────────────────────────

        # Allocated lazily on first forward call; grown as needed.
        # reset_cache() releases memory between independent sessions.

        self._cache_batch: int              = 0
        self.kv_cache: Optional[torch.Tensor] = None
        self.pe_cache: Optional[torch.Tensor] = None

        # ── RoPE frequency table ───────────────────────────────────────────

        # Extended lazily up to the maximum position seen so far.
        self._rope_seq_len: int = 0
        self.register_buffer(
            "freqs_cis",
            torch.empty(0, self.qk_rope_head_dim // 2, dtype=torch.complex64),
            persistent=False,
        )

    # ──────────────────────────────────────────────────────────────────────
    # RoPE helpers
    # ──────────────────────────────────────────────────────────────────────

    def _extend_rope(self, seq_len: int, device: torch.device) -> None:
        """
        Extend precomputed RoPE table to cover at least `seq_len` positions.
        No-op when already large enough.
        """
        if seq_len <= self._rope_seq_len:
            return

        dim      = self.qk_rope_head_dim
        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        if self.rope_factor > 1.0:
            inv_freq = inv_freq / self.rope_factor

        # Grow to at least 2× the current length to amortise reallocation cost
        # during autoregressive generation where seq_len increments by 1 each step.

        grow_to = max(seq_len, self._rope_seq_len * 2, 64)
        grow_to = min(grow_to, self.max_seq_len)
        t = torch.arange(grow_to, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)                          # (grow_to, dim//2)
        self.freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        self._rope_seq_len = grow_to

    def _apply_rope(
        self,
        x: torch.Tensor,
        start_pos: int,
        seqlen: int,
    ) -> torch.Tensor:
        """
        Apply rotary embeddings.

        Args:
            x:         (bsz, seqlen, n_heads, rope_dim)
            start_pos: absolute position of the first token
            seqlen:    number of tokens

        Returns:
            Tensor of same shape and dtype as x.
        """
        dtype = x.dtype
        x_c   = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

        # freqs: (seqlen, rope_dim//2) → broadcast (1, seqlen, 1, rope_dim//2)
        freqs = self.freqs_cis[start_pos : start_pos + seqlen].view(1, seqlen, 1, -1)
        return torch.view_as_real(x_c * freqs).flatten(-2).to(dtype)

    def _per_batch_bmm(
        self,
        q: torch.Tensor,    # (bsz, seqlen_q, h, d_k)
        k: torch.Tensor,    # (bsz, seqlen_k, d_k)
    ) -> torch.Tensor:
        """
        Per-batch bmm of ``q @ k^T`` returning ``(bsz, h, seqlen_q, seqlen_k)``.
        Used by manual attention path to avoid n_heads != bsz broadcast issues.
        """
        bsz, seqlen_q, h, d_k = q.shape
        seqlen_k = k.size(1)
        out = torch.empty(bsz, h, seqlen_q, seqlen_k, dtype=q.dtype, device=q.device)
        for b in range(bsz):
            q_h_b = q[b].permute(1, 0, 2).contiguous()                    # (h, seqlen_q, d_k)
            k_b   = k[b]                                                   # (seqlen_k, d_k)
            out[b] = torch.bmm(q_h_b, k_b.t().unsqueeze(0).expand(h, -1, -1))
        return out

    # ──────────────────────────────────────────────────────────────────────
    # Cache management
    # ──────────────────────────────────────────────────────────────────────

    def _ensure_cache(self, bsz: int, device: torch.device, dtype: torch.dtype) -> None:
        """
        Ensure kv_cache / pe_cache can hold at least `bsz` sequences.
        Doubles capacity on growth (floor 16) to amortise reallocation.
        """
        need_alloc = (
            self.kv_cache is None
            or bsz > self._cache_batch
            or self.kv_cache.device != device
            or self.kv_cache.dtype  != dtype
        )
        if not need_alloc:
            return

        new_bsz = max(bsz, self._cache_batch * 2, 16)
        self.kv_cache     = torch.zeros(
            new_bsz, self.max_seq_len, self.kv_lora_rank,
            device=device, dtype=dtype,
        )
        self.pe_cache     = torch.zeros(
            new_bsz, self.max_seq_len, self.qk_rope_head_dim,
            device=device, dtype=dtype,
        )
        self._cache_batch = new_bsz

    def reset_cache(self) -> None:
        """Release KV-cache memory and reset tracking state."""
        self.kv_cache     = None
        self.pe_cache     = None
        self._cache_batch = 0

    def prefill_cache(
        self,
        kv_latent: torch.Tensor,
        k_pe: torch.Tensor,
        start_pos: int,
    ) -> None:
        """
        Write pre-computed KV latents into the cache at an offset.
        Useful for prefix/prompt caching (shared prompt prefix).

        Args:
            kv_latent: (bsz, seqlen, kv_lora_rank) — already kv_norm'd
            k_pe:      (bsz, seqlen, rope_dim)      — already RoPE-rotated
            start_pos: token offset to begin writing at
        """
        bsz, seqlen, _ = kv_latent.shape
        end_pos = start_pos + seqlen
        if end_pos > self.max_seq_len:
            raise ValueError(
                f"prefill_cache: end_pos {end_pos} > max_seq_len {self.max_seq_len}"
            )
        self._extend_rope(end_pos, kv_latent.device)
        self._ensure_cache(bsz, kv_latent.device, kv_latent.dtype)
        self.kv_cache[:bsz, start_pos:end_pos] = kv_latent
        self.pe_cache[:bsz, start_pos:end_pos] = k_pe

    # ──────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            x:         (bsz, seqlen, dim)
            start_pos: First absolute token index. 0 for prefill/training;
                       decode step position during generation.
            mask:      Additive causal mask (1, 1, seqlen_q, seqlen_k).
                       None for single-token decode.
            use_cache: Write KV latents to persistent cache and read full
                       context. False for training (current sequence only).

        Returns:
            (bsz, seqlen, dim)
        """
        bsz, seqlen, _ = x.shape
        end_pos = start_pos + seqlen

        if end_pos > self.max_seq_len:
            raise RuntimeError(
                f"Layer {self.layer_idx}: end_pos {end_pos} exceeds "
                f"max_seq_len {self.max_seq_len}"
            )

        # Extend the RoPE table on demand before any positional encoding
        self._extend_rope(end_pos, x.device)

        if use_cache:
            self._ensure_cache(bsz, x.device, x.dtype)

        # ── Queries ────────────────────────────────────────────────────────

        if self.q_lora_rank > 0:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        else:
            q = self.wq(x)
        # (bsz, seqlen, n_local_heads, qk_head_dim)
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = self._apply_rope(q_pe, start_pos, seqlen)

        # ── KV latent compression ──────────────────────────────────────────

        kv_a                 = self.wkv_a(x)
        kv_latent, k_pe_raw  = kv_a.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_normed            = self.kv_norm(kv_latent)

        # k_pe_raw: (bsz, seqlen, rope_dim) — temporarily add head dim for _apply_rope
        k_pe = self._apply_rope(k_pe_raw.unsqueeze(2), start_pos, seqlen).squeeze(2)
        # k_pe: (bsz, seqlen, rope_dim) — shared across all heads

        if use_cache:
            # Detach before writing into the persistent cache. The cache must
            # not carry autograd references from the current forward — it
            # would otherwise hold the graph alive across multiple forwards.
            self.kv_cache[:bsz, start_pos:end_pos] = kv_normed.detach()
            self.pe_cache[:bsz, start_pos:end_pos] = k_pe.detach()
            # … then read back the full context window (prefix + current tokens)
            ctx_kv = self.kv_cache[:bsz, :end_pos]   # (bsz, end_pos, kv_lora_rank)
            ctx_pe = self.pe_cache[:bsz, :end_pos]    # (bsz, end_pos, rope_dim)
        else:
            # Training / no-cache path: context is the current sequence only.
            ctx_kv = kv_normed   # (bsz, seqlen, kv_lora_rank)
            ctx_pe = k_pe        # (bsz, seqlen, rope_dim)

        # ── Absorption trick ───────────────────────────────────────────────

        # wkv_b weight: (n_local_heads * (qk_nope + v_head), kv_lora_rank)
        # Split into two matrices along the output dim so we can matmul
        # against the latent without einsum.
        wkv_b_full = self.wkv_b.weight.view(
            self.n_local_heads,
            self.qk_nope_head_dim + self.v_head_dim,
            self.kv_lora_rank,
        )
        wkv_b_k = wkv_b_full[:, : self.qk_nope_head_dim]   # (h, qk_nope, kv_lora)
        wkv_b_v = wkv_b_full[:, self.qk_nope_head_dim:]    # (h, v_head, kv_lora)

        # Project q_nope into latent space so scores are computed directly
        # against ctx_kv — no need to materialise full K.
        # q_nope: (bsz, seqlen, h, qk_nope)
        # Permute so head is the batch dim, then bmm against the head's W.
        bsz, seqlen_q, h, d = q_nope.shape
        q_nope_h = q_nope.permute(2, 0, 1, 3).reshape(h, bsz * seqlen_q, d)
        # bmm: (h, bsz*s, d) @ (h, d, kv_lora) → (h, bsz*s, kv_lora)
        q_nope_proj_h = torch.bmm(q_nope_h, wkv_b_k)
        # Restore (bsz, seqlen, h, kv_lora)
        q_nope_proj = q_nope_proj_h.reshape(h, bsz, seqlen_q, self.kv_lora_rank).permute(1, 2, 0, 3).contiguous()

        if self.attn_impl == "sdpa":
            # ── SDPA path (default, fastest on CUDA GPUs) ────────────────────
            # Materialise K_nope and V once via wkv_b, then run a single
            # scaled_dot_product_attention call. K_rope is concatenated to
            # K_nope so a single attention call handles both the content
            # and positional score terms.
            #
            #   K = [K_nope | K_rope]   (per head)
            #   Q = [Q_nope | Q_rope]
            #   V = V
            #
            # K_rope is shared across heads; we just repeat it.
            seqlen_k = ctx_kv.size(1)

            # ctx_kv: (bsz, seqlen_k, kv_lora)
            # Per-head K_nope: ctx_kv @ wkv_b_k^T → (bsz, h, seqlen_k, qk_nope)
            # We need to lay out the data as (B*Sk, kv_lora) with rows in
            # (b, s) order matching the reshape below. The natural
            # layout of ctx_kv is (B, Sk, kv_lora) which is already what
            # we want — no permute needed; just flatten the first two.
            ctx_kv_bmm = ctx_kv.reshape(bsz * seqlen_k, self.kv_lora_rank) \
                              .unsqueeze(0).expand(h, -1, -1)               # (h, bsz*seqlen_k, kv_lora)
            wkv_b_k_t = wkv_b_k.transpose(-1, -2)                            # (h, kv_lora, qk_nope)
            K_nope_h = torch.bmm(ctx_kv_bmm, wkv_b_k_t)                      # (h, bsz*seqlen_k, qk_nope)
            K_nope = K_nope_h.reshape(h, bsz, seqlen_k, self.qk_nope_head_dim) \
                          .permute(1, 0, 2, 3).contiguous()                  # (bsz, h, seqlen_k, qk_nope)

            wkv_b_v_t = wkv_b_v.transpose(-1, -2)                            # (h, kv_lora, v_head)
            V_h = torch.bmm(ctx_kv_bmm, wkv_b_v_t)                           # (h, bsz*seqlen_k, v_head)
            V = V_h.reshape(h, bsz, seqlen_k, self.v_head_dim) \
                    .permute(1, 0, 2, 3).contiguous()                        # (bsz, h, seqlen_k, v_head)

            # Q_nope: (bsz, h, seqlen_q, qk_nope)
            Q_nope = q_nope.transpose(1, 2)
            # Q_rope: (bsz, h, seqlen_q, rope_dim)
            Q_rope = q_pe.transpose(1, 2)

            # K_rope: (bsz, h, seqlen_k, rope_dim) — head-shared, expanded.
            K_rope = ctx_pe.unsqueeze(1).expand(-1, h, -1, -1)

            # Concat along the head_dim axis
            Q_full = torch.cat([Q_nope, Q_rope], dim=-1)
            K_full = torch.cat([K_nope, K_rope], dim=-1)

            attn_mask = None
            if mask is not None:
                attn_mask = mask.expand(bsz, h, seqlen_q, -1)

            attn = F.scaled_dot_product_attention(
                Q_full, K_full, V,
                attn_mask=attn_mask,
                scale=self.softmax_scale,
            )   # (bsz, h, seqlen_q, v_head_dim)
            out = attn.transpose(1, 2).contiguous().flatten(2)
            return self.wo(out)

        # ── Manual path (fallback / debugging) ──────────────────────────
        # True absorption trick: project q into latent space, then score
        # directly against the latent. Returns scores of shape
        # (bsz, h, seqlen_q, seqlen_k). The two score contributions
        # are computed by per-batch bmm (one bmm per (bsz, h) pair),
        # then summed.
        #
        # shapes: q_nope_proj: (bsz, seqlen_q, h, kv_lora)
        #         ctx_kv:      (bsz, seqlen_k, kv_lora)
        #         q_pe:        (bsz, seqlen_q, h, rope_dim)
        #         ctx_pe:      (bsz, seqlen_k, rope_dim)
        scores_content = self._per_batch_bmm(q_nope_proj, ctx_kv)         # (bsz, h, seqlen_q, seqlen_k)
        scores_rope    = self._per_batch_bmm(q_pe, ctx_pe)                # (bsz, h, seqlen_q, seqlen_k)
        scores = (scores_content + scores_rope) * self.softmax_scale

        if mask is not None:
            # mask is (1, 1, seqlen_q, seqlen_k); expand to (bsz, h, seqlen_q, seqlen_k)
            scores = scores + mask.expand(bsz, h, seqlen_q, -1)

        attn = scores.softmax(dim=-1, dtype=torch.float32).to(x.dtype)   # (bsz, h, seqlen_q, seqlen_k)
        # Weighted sum of latent KV: ctx_kv is (bsz, seqlen_k, kv_lora),
        # attn is (bsz, h, seqlen_q, seqlen_k). The matmul is over the
        # last two dims; h and bsz must match for broadcasting, so we
        # do a per-batch bmm loop. Cost: bsz × h bmm calls of size
        # (seqlen_q, seqlen_k) @ (seqlen_k, kv_lora).
        out_latent = torch.empty(
            bsz, h, seqlen_q, self.kv_lora_rank, dtype=x.dtype, device=x.device
        )
        for b in range(bsz):
            # attn[b]: (h, seqlen_q, seqlen_k)
            a_b = attn[b]
            k_b = ctx_kv[b]                                                # (seqlen_k, kv_lora)
            out_latent[b] = torch.bmm(a_b, k_b.unsqueeze(0).expand(h, -1, -1))
        # Project to v_head_dim via wkv_b_v: wkv_b_v is (h, v_head, kv_lora),
        # we need (h, kv_lora, v_head) for the bmm.
        out_h = out_latent.permute(1, 0, 2, 3).reshape(h, bsz * seqlen_q, self.kv_lora_rank)
        wkv_b_v_t = wkv_b_v.transpose(-1, -2)                              # (h, kv_lora, v_head)
        out_v = torch.bmm(out_h, wkv_b_v_t)
        out = out_v.reshape(h, bsz, seqlen_q, self.v_head_dim).permute(1, 2, 0, 3).contiguous()
        return self.wo(out.flatten(2))
