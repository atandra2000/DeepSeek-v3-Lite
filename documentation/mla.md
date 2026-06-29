# MLA — Multi-Head Latent Attention

> For the full MLA theory (KV-cache problem, absorption-trick algebra,
> decoupled RoPE derivation, dimension breakdown, comparison against
> MHA/GQA/MQA), see the authoritative **[`../MLA.md`](../MLA.md)** (643
> lines). This file records only project-specific notes from
> `models/mla.py`.

## Dimensions (422M canonical config)

| Constant | Value | Source |
|----------|-------|--------|
| `kv_lora_rank` | 192 | KV compression latent dim |
| `qk_nope_head_dim` | 48 | Content Q/K per head (no RoPE) |
| `qk_rope_head_dim` | 24 | Positional Q/K per head (RoPE) |
| `qk_head_dim` | 72 | `qk_nope_head_dim + qk_rope_head_dim` |
| `v_head_dim` | 64 | Value head dim |
| `q_lora_rank` | 0 | No query compression (422M simplification) |
| KV cache per token | 216 | `kv_lora_rank + qk_rope_head_dim = 192 + 24` |

The 82M smoke-test config (`tests/conftest.py::cfg`) uses
`kv_lora_rank=128`, `qk_rope_head_dim=16` (cache = 144 floats/token).

## Two forward paths

- **`attn_impl="sdpa"` (default)** — materialises full `K_nope` and `V`
  from the latent via `bmm` against `wkv_b_k` / `wkv_b_v`, then calls
  `F.scaled_dot_product_attention` (FlashAttention-2 fused). The
  materialisation cost is offset by the FA2 kernel on GPU. This is **not**
  the true absorption trick — it is the production path.
- **`attn_impl="manual"`** — the true absorption trick: content scores
  computed in latent space via `_per_batch_bmm(q_nope_proj, ctx_kv)` (the
  query is pre-projected into the latent by `wkv_b_kᵀ`), positional scores
  via MQA-style `_per_batch_bmm(q_pe, ctx_pe)`, softmax in latent space,
  then V recovered once via `wkv_b_v`. ~4× FLOPs vs MHA but lowest
  memory-bandwidth. Reference/debug path; `test_sdpa_and_manual_agree`
  verifies the two paths agree.

## Decoupled RoPE

`qk_head_dim` is split into a content half (`qk_nope_head_dim`, no RoPE,
absorbable) and a positional half (`qk_rope_head_dim`, RoPE, single shared
key head — MQA-style). The shared RoPE key `k_pe` is produced once per
layer and expanded to all query heads via `ctx_pe.unsqueeze(1).expand(-1, h,
-1, -1)`.

## KV cache

- `kv_cache`: `(batch, max_seq_len, kv_lora_rank)` — compressed latents.
- `pe_cache`: `(batch, max_seq_len, qk_rope_head_dim)` — RoPE keys.
- `_ensure_cache` lazily allocates and **doubles** capacity (min 16) to
  amortise reallocation.
- Cache writes use `.detach()` to prevent cross-forward autograd leaks
  during training.
- `prefill_cache` writes pre-computed latents at an arbitrary offset
  (useful for shared prompt prefixes / CUDA graphs).
- `reset_cache()` between independent generation requests.

## YaRN scaling

`rope_factor > 1.0` divides `inv_freq` by the factor and adds an
`mscale = 0.1 * mscale_raw * log(rope_factor) + 1.0` term. The softmax
scale is multiplied by `mscale**2` when `max_seq_len > 4096` to prevent
attention-score underflow at extended context. With the canonical training
config (`rope_factor=1.0`, `max_seq_len=2048`) all of this is bypassed.

## `_extend_rope`

Grows the precomputed `freqs_cis` table up to `max_seq_len` in 2× steps to
amortise during autoregressive generation. Uses
`torch.view_as_complex` / `torch.polar` for the rotation.