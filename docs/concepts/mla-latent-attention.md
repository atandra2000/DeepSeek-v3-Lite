# DeepSeek-v3-Lite — MLA: Latent Attention in One Read

> **Audience:** intermediate. Assumes you know what attention and a KV cache are; everything else is built here.
>
> **Read this if** you want the one-page mental model of *why* the KV cache shrinks and where the up-projections went. **Skip if** you want the full derivations and shape traces → [MLA & Mixed Precision](attention-and-precision.md) and [R3 — MLA API](../references/R3_mla_api.md).

**Depends on:** nothing beyond standard attention · **Read next:** [MLA & Mixed Precision](attention-and-precision.md) (full treatment), [Aux-Loss-Free MoE Balance](aux-loss-free-moe-balance.md)

---

## 1. The problem: the cache grows with heads

Standard multi-head attention caches, per token and per layer, a key and a value vector **for every head**: `2 · H · d_head` floats. At the canonical 422 M config that is `2 · 12 · 64 = 1,536` floats. During decode, the whole cache is re-read from HBM every step, so this number — not FLOPs — sets decode latency at long context.

## 2. The idea: cache a rank-192 summary instead

MLA factorizes K and V through a **low-rank latent**. One projection (`models/mla.py:MultiHeadLatentAttention` — `wkv_a`) maps the hidden state to a shared latent of size `kv_lora_rank = 192`, plus a small 24-float positional key that lives outside the latent (`qk_rope_head_dim = 24`):

```python
# illustrative — trimmed from models/mla.py:MultiHeadLatentAttention.forward; shapes at the 422M config
kv_a = self.wkv_a(x)                                  # (B, S, 768) -> (B, S, 216)
kv_latent, k_pe_raw = kv_a.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
kv_normed = self.kv_norm(kv_latent)                   # latent is RMSNorm'd *before* caching
```

**Only the latent (192 floats) and the RoPE key (24 floats) are cached** — `216` floats per token per layer, head-shared in the MQA sense. That is the entire cache. The per-head keys and values are **reconstructed on demand** by a second projection (`wkv_b`, `192 → 12 × (48 + 64)`) whenever attention runs.

## 3. The up-projections, and where they went

`wkv_b` produces per-head K (the 48-dim nope slice) and V (the 64-dim slice) from the shared latent. Because K = `W_k · c` for a shared `c`, the score `qᵀK` can be folded:

- **Absorbed (manual path):** compute `(W_kᵀ q)ᵀ c` — the query is projected *into latent space* and attention is scored against the cached latents directly; the value up-projection is absorbed into the output side, so V is never materialized either.
- **Materialized (sdpa path, the default):** the repo instead reconstructs K and V for the whole context with one batched matmul — `torch.bmm(ctx_kv_bmm, wkv_b_kv.transpose(-1, -2))` in `models/mla.py:MultiHeadLatentAttention.forward` — then calls `F.scaled_dot_product_attention` as usual. SDPA gets its optimized kernel; the latent cache still keeps memory small.

## 4. The RoPE exception

RoPE rotates K and Q by position, and a position-dependent rotation cannot be folded into a linear up-projection (`RoPE(W c, pos) ≠ W · RoPE(c, pos)`). So MLA **decouples** RoPE: a separate 24-dim per-head-slice key `k_pe` is projected alongside the latent, rotated by RoPE, and cached next to it (`models/mla.py:MultiHeadLatentAttention._extend_rope` builds the table, `_apply_rope` applies it). Queries are `48` nope dims + `24` rope dims; scores are the sum of a content term (latent space) and a positional term. RoPE keys are shared across heads, so they cost 24 floats — not 12 × 24.

This repo trains with `q_lora_rank = 0` (full-rank Q projection — no query compression) and `rope_factor = 1.0` (YaRN scale inactive at train time; the `mscale` branch in `__init__` only activates when context is extended).

## 5. The cache contract

Two buffers per layer, allocated lazily by `models/mla.py:MultiHeadLatentAttention._ensure_cache` and dropped by `reset_cache`:

| Buffer | Shape (per layer) | Contents |
|---|---|---|
| `kv_cache` | `(B, max_seq_len, 192)` | RMSNorm'd latents, `.detach()`-ed on write |
| `pe_cache` | `(B, max_seq_len, 24)` | RoPE'd shared keys |

**Derived arithmetic, not measured** [INFERENCE]: at `B=1, max_seq_len=2048, 18 layers`, that is `2048 · 216 · 18 ≈ 8.0 M` floats ≈ **16 MB in BF16**; the MHA baseline at `2·12·64` would be ≈ **113 MB** — the same **~7.1×** per-token ratio (216 vs 1,536 floats) that [attention-and-precision.md](attention-and-precision.md) derives for `B=8` (127 MB vs 906 MB). No GPU run has executed in this repo; both figures are arithmetic.

## 6. Backends

Three interchangeable backends behind one `attn_impl` key: `sdpa` (default), `manual` (the true-absorption latent-space path), and `triton` (opt-in fused kernel, `models/mla_triton.py:triton_mla_attention`, tested against `models/mla_triton.py:mla_attention_reference`; see [kernels-and-ops.md](kernels-and-ops.md) for the double-opt-in guard). The Triton path re-computes in latent space rather than materializing per-head K/V — it is the absorption trick in kernel form.

## Check yourself

1. Why can the value up-projection be deferred to the output side of attention? (Because `softmax(qᵀK)·V` is linear in V, and `V = W_v · c` — fold `W_v` past the softmax.)
2. Why does RoPE need a separate cached key? (Position-dependent rotation breaks the linear fold.)
3. What breaks if you cache the *un-normalized* latent? (`kv_norm` runs before the cache write; downstream reconstruction assumes the normed latent.)

<!-- docs:verified 2026-09-21 · d9a5de4 -->
