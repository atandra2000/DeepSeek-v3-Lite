# DeepSeek-v3-Lite — R3 MLA API Reference

> **What this is:** the complete symbol-level reference for `MultiHeadLatentAttention`, the DeepSeek-V3-style multi-head latent attention layer used by every transformer block in this repo. Every entry is a signature, a shape contract, a default, and a caller list — no tutorial prose (that lives in [MLA & Mixed Precision](../concepts/attention-and-precision.md) and [Foundations & Architecture](../concepts/foundations.md) §5–7).
>
> **Scope:** one class, eight public methods, three attention implementations (`sdpa` / `manual` / `triton`), one KV-cache contract, one RoPE table. All shapes below use the canonical config (`configs/pretrain_a100_422m.yaml`): $D{=}768$, $H{=}12$, $R{=}192$, $D_{\text{nope}}{=}48$, $D_{\text{rope}}{=}24$, $D_v{=}64$, $D_{\text{qk}}{=}72$, $S_{\max}{=}2048$.
>
> **Companion files:** R1 Config Schema (every YAML key read here), R2 Transformer API (the caller), R6 Triton API (the kernel this layer dispatches to). The math and the worked example are in [MLA & Mixed Precision](../concepts/attention-and-precision.md).

---

## 1. 60-second summary

MLA replaces the per-head K/V cache of MHA with a single per-token latent $c \in \mathbb{R}^{R}$ plus a small rotation-only key $k^{pe} \in \mathbb{R}^{D_{\text{rope}}}$ — $192 + 24 = 216$ floats per token instead of $12 \times (72 + 64) = 1\\,632$ (a $\sim$7.6$\times$ cache cut). The per-head keys and values are *reconstructed* from the latent on the fly by the up-projection `wkv_b`; the absorption trick folds that up-projection into the query and output projections so the reconstruction only happens as a fused batched matmul (`sdpa` path) or never at all (`manual` path scores directly in latent space).

The layer has one job per forward: take hidden states $(B, S_q, D)$, produce attention output $(B, S_q, D)$, and — when `use_cache=True` — write the normalized latent and rotated rope key of the current positions into a lazily-allocated, batch-growable, length-fixed cache so later decode steps can attend to them.

## 2. Why it exists

Standard MHA caches per-head K and V; at 12 heads × 2048 positions that is a large, mostly-redundant tensor (the paper's argument: per-head K/V are up-projections of a shared low-rank representation). MLA caches the small shared representation instead and pays a slightly larger attention-time matmul. For this repo the win is architectural demonstration plus long-context decode memory; at the canonical 2,048 context the absolute cache is small, but the *contract* (what is cached, where it is written, when it is dropped) is load-bearing for `generate`, chunked prefill, and speculative decoding.

## 3. Intuition

Think of each token's `wkv_b` up-projection as a **shared codebook**: instead of storing 12 decoded keys and 12 decoded values, store the 192-dim code. At attention time, either decode the code into the 12 keys/values you need (SDPA path — one fused bmm does all heads at once), or keep the code folded into the query and score directly in code space (manual path — the absorption trick). RoPE breaks this algebra, so positions get a separate 24-dim rotation-only key shared across heads. The cache is just a scratchpad holding every past token's code and rotation key.

## 4. Class and configuration

### `models/mla.py:MultiHeadLatentAttention`

```python
class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config: dict, layer_idx: int = 0):
```

One instance per transformer block. `config` is the `model:` section of the YAML (or an equivalent dict); every key below is read at construction and the resolved values become attributes. All canonical values from `configs/pretrain_a100_422m.yaml`; for the complete key list and defaults see R1 §Config keys.

| Config key | Default | Attribute | Canonical | Notes |
|---|---|---|---|---|
| `dim` | — | `dim` | 768 | hidden size; `config["dim"]` (required) |
| `n_heads` | — | `n_heads`, `n_local_heads` | 12 | query heads; no GQA split, so `n_local_heads == n_heads` |
| `q_lora_rank` | — | `q_lora_rank` | 0 | 0 → direct `wq`; >0 → `wq_a` + `q_norm` + `wq_b` LoRA path |
| `kv_lora_rank` | — | `kv_lora_rank` | 192 | latent width $R$; the KV-cache contraction dim |
| `qk_nope_head_dim` | — | `qk_nope_head_dim` | 48 | content part of each query/key head |
| `qk_rope_head_dim` | — | `qk_rope_head_dim` | 24 | rotation part of each query/key head |
| `v_head_dim` | — | `v_head_dim` | 64 | value head width |
| `max_seq_len` | — | `max_seq_len` | 2048 | hard cache bound; also switches `softmax_scale` at >4096 |
| `rope_theta` | — | `rope_theta` | 10000 | RoPE base frequency |
| `rope_factor` | `1.0` | `rope_factor` | 1.0 | YaRN stretch; >1.0 enables mscale correction |
| `mscale` | `1.0` | `mscale` | 1.0 | YaRN attention-scale factor |
| `attn_impl` | `"sdpa"` | `attn_impl` | `"sdpa"` | `"sdpa"` \| `"manual"` \| `"triton"`; see §6 |

Derived attributes set in `__init__`:

- `qk_head_dim = qk_nope_head_dim + qk_rope_head_dim` (72) — the concatenated query/key width used by `wq`/`wq_b`.
- `mscale` and `softmax_scale` — see §5.
- `_cache_batch: int = 0`, `kv_cache: Optional[Tensor] = None`, `pe_cache: Optional[Tensor] = None` — cache state, see §7.
- `_rope_seq_len: int = 0` — length of the currently-materialized RoPE table.
- `freqs_cis` — non-persistent buffer, shape `(0, qk_rope_head_dim // 2)`, `torch.complex64`, grown lazily by `_extend_rope` (see §8).

### Sub-modules created

| Branch | Sub-module | Weight shape | Notes |
|---|---|---|---|
| `q_lora_rank > 0` | `wq_a = Linear(dim, q_lora_rank, bias=False)` | $(q_{\text{lora}}, D)$ | — |
| | `q_norm = RMSNorm(q_lora_rank, eps=1e-6)` | — | — |
| | `wq_b = Linear(q_lora_rank, n_heads * qk_head_dim, bias=False)` | $(H \cdot D_{\text{qk}}, q_{\text{lora}})$ | — |
| `q_lora_rank == 0` (canonical) | `wq = Linear(dim, n_heads * qk_head_dim, bias=False)` | $(864, 768)$ | direct query |
| always | `wkv_a = Linear(dim, kv_lora_rank + qk_rope_head_dim, bias=False)` | $(216, 768)$ | latent + raw rope key |
| always | `kv_norm = RMSNorm(kv_lora_rank, eps=1e-6)` | — | normalizes the latent **before** caching |
| always | `wkv_b = Linear(kv_lora_rank, n_heads * (qk_nope_head_dim + v_head_dim), bias=False)` | $(12 \cdot 112, 192)$ | the shared up-projection; viewed as `(H, D_nope + D_v, R)` in forward |
| always | `wo = Linear(n_heads * v_head_dim, dim, bias=False)` | $(768, 768)$ | output projection |

The bias-free `Linear` layers are the repo's default; all parameters are float32 at construction (BF16 comes from autocast at training time).

## 5. The mscale / softmax_scale computation

Source: `models/mla.py:MultiHeadLatentAttention.__init__`.

```python
mscale_raw = config.get("mscale", 1.0)
if self.rope_factor > 1.0:
    self.mscale = 0.1 * mscale_raw * math.log(self.rope_factor) + 1.0
else:
    self.mscale = mscale_raw
if self.max_seq_len > 4096:
    self.softmax_scale = (self.qk_head_dim ** -0.5) * (self.mscale ** 2)
else:
    self.softmax_scale = self.qk_head_dim ** -0.5
```

- **`mscale`**: the YaRN attention-scale correction. When `rope_factor > 1.0` (context stretching), `mscale = 0.1 \cdot mscale_{\text{raw}} \cdot \ln(\text{rope\_factor}) + 1.0`; the logarithm compensates for the density of the stretched frequency grid. When `rope_factor == 1.0` (canonical config), `mscale = mscale_raw` — the whole branch is dormant, guarded by `config.get("mscale", 1.0)`.
- **`softmax_scale`**: passed verbatim as the `scale=` argument of `F.scaled_dot_product_attention` and into the Triton kernel. For `max_seq_len <= 4096` it is the plain $\sqrt{D_{\text{qk}}}^{-1}$; for `max_seq_len > 4096` it is multiplied by `mscale ** 2` to counteract the attention entropy loss of stretched RoPE.
- **Canonical value:** $D_{\text{qk}} = 48 + 24 = 72$, so `softmax_scale = 72^{-1/2} = 0.11785...` and `mscale = 1.0`.

## 6. Attention implementation modes

`attn_impl` selects among three forward strategies; all three consume the *same* cache contract (§7) and produce numerically-equivalent outputs (guarded by `tests/test_models.py:TestMLA.test_sdpa_and_manual_agree`).

| Mode | Path in forward | What it computes | Used for |
|---|---|---|---|
| `"sdpa"` (default) | SDPA branch | reconstructs $K^{\text{nope}}, V$ from the cached latent with one fused bmm, then `F.scaled_dot_product_attention` | training + inference; the canonical mode |
| `"manual"` | fall-through branch | scores directly in latent space via `_per_batch_bmm` (true absorption); softmax in fp32; per-batch output bmm | reference / debug; `tests/test_models.py:TestMLA` |
| `"triton"` | Triton branch | delegates to `triton_mla_attention` via `_forward_triton` (fused materialize+RoPE+attention) | opt-in fast path; needs `ENABLE_TRITON_KERNELS=1` + Triton installed |

**Fallback semantics (`triton` mode only).** The Triton branch is wrapped in `try/except (ImportError, ValueError)` inside `models/mla.py:MultiHeadLatentAttention.forward`. On the first failure it prints one warning (`[mla] triton attn_impl unavailable (...); falling back to 'sdpa' for this model.`), sets `self.attn_impl = "sdpa"` (persistent for this layer), and completes the forward via the SDPA branch. Later failures are silent — the layer is permanently on SDPA until the module is rebuilt. Any `ImportError` (Triton missing) or `ValueError` (kernel dimension limits, e.g. `D_nope > 256`) triggers the fallback; other exceptions propagate. A `RuntimeError` from `_forward_triton` does **not** trigger it — this matches the MoE layers' behavior (see R6 §dispatch and R4).

## 7. KV-cache contract

Two tensors per layer, both allocated lazily and **per-layer** (18 layers × the shapes below at the canonical config; there is no sharing between layers):

```
kv_cache: Tensor(new_bsz, max_seq_len, kv_lora_rank)   # (B_alloc, S_max, 192)  — normalized latent, pre-up-projection
pe_cache: Tensor(new_bsz, max_seq_len, qk_rope_head_dim)  # (B_alloc, S_max, 24)  — RoPE-rotated key, shared across heads
```

- **Batch axis is allocated, sequence axis is fixed.** `new_bsz = max(bsz, _cache_batch * 2, 16)`: the first `use_cache=True` forward allocates at least 16 rows, and any later forward whose batch exceeds the allocation reallocates at double the previous capacity (amortized growth). The sequence axis is always `max_seq_len` — pre-allocated, never grown. See `models/mla.py:MultiHeadLatentAttention._ensure_cache`.
- **Reallocation drops contents.** The trigger is `bsz > _cache_batch` **or** a device/dtype change; the new allocation is `torch.zeros`, so the previous prefix is lost. Change batch shape only via `reset_cache()`.
- **Write is positional, read is a prefix.** Forward writes `kv_cache[:bsz, start_pos:end_pos]` and `pe_cache[:bsz, start_pos:end_pos]`, then reads `[:bsz, :end_pos]` as the attended context. `start_pos` is the global (not per-chunk) offset — this is what makes chunked prefill and decode work (guarded by `tests/test_models.py:TestTransformer.test_chunked_prefill_matches_full_forward` and the Triton twin in `tests/test_mla_triton.py`).
- **Sequence bound is a hard error.** `end_pos = start_pos + seqlen > max_seq_len` raises `RuntimeError(f"Layer {self.layer_idx}: end_pos {end_pos} exceeds max_seq_len {self.max_seq_len}")` — never a silent truncation.
- **Writes are detached.** `kv_cache[...] = kv_normed.detach()`, `pe_cache[...] = k_pe.detach()`: cache entries carry no autograd graph, so decode-step forwards never build a graph through past positions. With `use_cache=False` (training) the cache is never touched.
- **State machine:** `UNINIT` (both `None`, `_cache_batch = 0`) → `READY` (first `use_cache=True` forward) → `RESET` (back to `UNINIT` via `reset_cache()`). `Transformer.generate` calls `Transformer.reset_cache` (which fans out to every layer) before every generation; bare `forward(use_cache=True)` calls in a serving loop must reset by hand or they inherit the previous prompt's prefix.

## 8. Method reference

### `models/mla.py:MultiHeadLatentAttention._extend_rope`

```python
def _extend_rope(self, seq_len: int, device: torch.device) -> None:
```

Lazily materializes the complex RoPE table `freqs_cis` to cover `seq_len` positions; no-op if `seq_len <= _rope_seq_len`.

- **Growth:** `grow_to = min(max(seq_len, _rope_seq_len * 2, 64), max_seq_len)` — doubles from the previous size (floor 64) to amortize table rebuilds across decode steps.
- **Construction:** `inv_freq = 1.0 / (rope_theta ** (arange(0, dim, 2, dtype=float32, device) / dim))` — note the exponent `arange / dim` (full `qk_rope_head_dim`, not half); then `freqs_cis = polar(ones_like(freqs), outer(t, inv_freq))` stored as `torch.complex64` on the device of the first forward.
- **Invariants:** table length never exceeds `max_seq_len`; `_rope_seq_len` tracks the table length. With `rope_factor > 1.0` the base frequency would be divided by the factor (YaRN); at the canonical `rope_factor = 1.0` the path is plain RoPE.
- **Caller:** `forward` (once per forward, with `end_pos`).

### `models/mla.py:MultiHeadLatentAttention._apply_rope`

```python
def _apply_rope(self, x: torch.Tensor, start_pos: int, seqlen: int) -> torch.Tensor:
```

Rotates the last dim of `x` by the absolute positions `[start_pos, start_pos + seqlen)`.

- **Contract:** `x` has last dim `qk_rope_head_dim` (any leading shape; the rope dim must be even) → returns same shape, same dtype.
- **Mechanics:** `x.float()` → `view_as_complex(x.reshape(*x.shape[:-1], -1, 2))` → multiply by `freqs_cis[start_pos:start_pos+seqlen].view(1, seqlen, 1, -1)` → `view_as_real(...).flatten(-2)` → cast back to `x.dtype`. One complex multiply per element; the table row index is the **absolute** position, which is what makes the score depend only on relative offset (rotation-composition property, see [MLA & Mixed Precision](../concepts/attention-and-precision.md) §6).
- **dtype note:** the rotation happens in fp32 (via `.float()`), the result is cast back to the input dtype — no precision loss from BF16 table arithmetic.
- **Callers:** `forward` (twice: query rope key `q_pe` with shape `(B, S_q, H, D_rope)`; key rope key `k_pe` from `(B, S_q, 1, D_rope)`, squeezed back to `(B, S_q, D_rope)`).
- **Requires** `freqs_cis` to already cover `start_pos + seqlen` (guaranteed by `_extend_rope` earlier in `forward`).

### `models/mla.py:MultiHeadLatentAttention._per_batch_bmm`

```python
def _per_batch_bmm(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    return torch.matmul(q.transpose(1, 2), k.unsqueeze(1).transpose(2, 3))
```

The absorption-trick score primitive: `q` of shape `(B, S_q, H, D_k)` and `k` of shape `(B, S_k, D_k)` → scores `(B, H, S_q, S_k)`. `k` is unsqueezed to `(B, 1, S_k, D_k)` so the head axis broadcasts; the matmul runs per batch element. Used only by the `manual` path (`q_nope_proj` in latent space, `q_pe` against `ctx_pe`). Guarded for equivalence by `tests/test_models.py:TestMLA.test_sdpa_and_manual_agree`.

### `models/mla.py:MultiHeadLatentAttention._ensure_cache`

```python
def _ensure_cache(self, bsz: int, device: torch.device, dtype: torch.dtype) -> None:
```

Allocates (or grows) `kv_cache` and `pe_cache` per §7. Reallocation trigger: `kv_cache is None or bsz > _cache_batch or kv_cache.device != device or kv_cache.dtype != dtype`. Growth formula: `new_bsz = max(bsz, _cache_batch * 2, 16)`; both tensors are `torch.zeros` on `(device, dtype)` — the dtype is captured from the first cached forward's input, so under autocast a later forward in a different dtype would trigger a reallocation (content loss). Sets `_cache_batch = new_bsz`. Caller: `forward` (only when `use_cache=True`).

### `models/mla.py:MultiHeadLatentAttention.reset_cache`

```python
def reset_cache(self) -> None:
    self.kv_cache = None
    self.pe_cache = None
    self._cache_batch = 0
```

Full teardown of this layer's cache: next `use_cache=True` forward allocates from scratch. No partial clear exists. Callers: `models/transformer.py:Transformer.reset_cache` (loops `if hasattr(layer.attn, "reset_cache")` over all 18 layers), `models/transformer.py:Transformer.generate` (via the fan-out, before every generation), and tests (`tests/test_models.py:TestMLA.test_reset_cache`).

### `models/mla.py:MultiHeadLatentAttention.forward`

```python
def forward(self, x: torch.Tensor, start_pos: int = 0,
            mask: Optional[torch.Tensor] = None, use_cache: bool = True) -> torch.Tensor:
```

The one public entry point. **Inputs → outputs contract:**

- `x`: `(B, S_q, D)` hidden states (the caller passes `attn_norm`'s output — the layer itself applies no norm to `x`).
- `start_pos`: global KV-cache offset of the first query position; `end_pos = start_pos + seqlen` must be `<= max_seq_len` (else `RuntimeError`).
- `mask`: additive causal mask `(1, 1, S_q, S_kv)`, produced by `models/transformer.py:Transformer._build_causal_mask`; expanded inside the layer to `(B, H, S_q, S_kv)`. `None` → no masking (only safe for full-length self-attention already handled upstream).
- `use_cache`: `True` writes positions `[start_pos, end_pos)` into the cache and attends over the cache prefix `[0, end_pos)`; `False` attends over the current step's own latents only (training mode — the cache is never allocated).
- Returns `(B, S_q, D)`.

**Step-by-step (all intermediate shapes canonical):**

1. Guards + rope: bounds check; `_extend_rope(end_pos, x.device)`; `_ensure_cache(bsz, x.device, x.dtype)` if `use_cache`.
2. Query: `q = wq(x)` (or `wq_b(q_norm(wq_a(x)))` when `q_lora_rank > 0`) → `(B, S_q, H, 72)`; split → `q_nope (B, S_q, H, 48)`, `q_pe (B, S_q, H, 24)`; `q_pe = _apply_rope(q_pe, start_pos, seqlen)`.
3. Key/value latent: `kv_a(x)` → split into `kv_latent (B, S_q, 192)` and `k_pe_raw (B, S_q, 24)`; `kv_normed = kv_norm(kv_latent)`; `k_pe = _apply_rope(k_pe_raw.unsqueeze(2), ...).squeeze(2)` → `(B, S_q, 24)`.
4. Cache write/read (if `use_cache`): `kv_cache[:bsz, start_pos:end_pos] = kv_normed.detach()`, `pe_cache[:bsz, start_pos:end_pos] = k_pe.detach()`; context = `kv_cache[:bsz, :end_pos]` (`(B, S_kv, 192)`) and `pe_cache[:bsz, :end_pos]` (`(B, S_kv, 24)`). Else context = current-step `kv_normed` / `k_pe`.
5. Up-projection view: `wkv_b.weight.view(H, D_nope + D_v, R)` → `wkv_b_k = wkv_b_full[:, :48]` (`(12, 48, 192)`), `wkv_b_v = wkv_b_full[:, 48:]` (`(12, 64, 192)`).
6. Absorbed query: `q_nope_h = q_nope.permute(2, 0, 1, 3).reshape(H, B*S_q, D_nope)`; `q_nope_proj_h = bmm(q_nope_h, wkv_b_k)` → `(12, B*S_q, 192)` → reshaped back to `q_nope_proj (B, S_q, H, 192)`.
7. Dispatch to `attn_impl` (§6), all paths ending in `self.wo(out.flatten(2))` with `out` of shape `(B, S_q, H, D_v)`:
   - **SDPA:** `ctx_kv_bmm = ctx_kv.reshape(B*S_kv, R).unsqueeze(0).expand(H, -1, -1)`; fused `bmm(ctx_kv_bmm, cat([wkv_b_k, wkv_b_v], dim=1).T)` → split into `K_nope (B, H, S_kv, 48)` and `V (B, H, S_kv, 64)`; `K_rope = ctx_pe.unsqueeze(1).expand(B, H, S_kv, 24)`; `attn_mask = mask.expand(B, H, S_q, -1)`; `F.scaled_dot_product_attention(cat([Q_nope, Q_rope], -1), cat([K_nope, K_rope], -1), V, attn_mask=attn_mask, scale=self.softmax_scale)` → `(B, H, S_q, 64)` → transpose + flatten.
   - **Manual:** `scores = (_per_batch_bmm(q_nope_proj, ctx_kv) + _per_batch_bmm(q_pe, ctx_pe)) * softmax_scale` → `(B, H, S_q, S_kv)`; add `mask.expand(bsz, h, seqlen_q, -1)`; `scores.softmax(dim=-1, dtype=torch.float32).to(x.dtype)`; per-batch `bmm(attn[b], ctx_kv[b].expand(H, -1, -1))` into `out_latent (B, H, S_q, 192)`; `out_v = bmm(out_latent.permute(1,0,2,3).reshape(H, B*S_q, R), wkv_b_v.T)`.
   - **Triton:** `_forward_triton(...)` (§below); on `(ImportError, ValueError)` fall back to SDPA with a one-time warning (§6).

**dtype notes.** The softmax in the manual path runs in fp32 (`dtype=torch.float32`) and is cast back to `x.dtype`; the SDPA path leaves numerics to `scaled_dot_product_attention`'s internal fp32 accumulation. `q_nope_proj` is `.contiguous()` after the permute/reshape so the manual path's bmm gets a dense layout. Under autocast, the layer runs in the input dtype (BF16 in training); the RoPE table is complex64 and the rotation itself is fp32.

**Callers.** `models/transformer.py:TransformerBlock.forward` (`self.attn(self.attn_norm(x), start_pos, mask, use_cache)`); transitively `Transformer.forward` / `Transformer.forward_with_hidden` / `Transformer.generate` and the speculative decoder (`inference/speculative.py:SpeculativeDecoder.generate`, via the main model). Tests: `tests/test_models.py:TestMLA`, `tests/test_mla_triton.py` (chunked-prefill equivalence at the Triton boundary).

### `models/mla.py:MultiHeadLatentAttention._forward_triton`

```python
def _forward_triton(self, q_nope, q_pe, ctx_kv, ctx_pe, wkv_b_k, wkv_b_v,
                    bsz, seqlen, h, mask, start_pos, use_cache) -> torch.Tensor:
```

The host wrapper for the fused Triton kernel — it imports `triton_mla_attention` from `models/mla_triton.py` (lazily, so the import cost and the `ImportError` happen here, inside the `try` in `forward`), re-arranges tensors to the kernel's layout, and calls it. Never anchor the JIT kernel itself (`_mla_flash_fwd_kernel`) — it is an implementation detail; the documented public boundary is `models/mla_triton.py:triton_mla_attention` (see R6).

- **Input layout (python-level):** `q_nope (B, S_q, H, D_nope)`, `q_pe (B, S_q, H, D_rope)` (already rotated), `ctx_kv (B, S_kv, R)`, `ctx_pe (B, S_kv, D_rope)`, `wkv_b_k (H, D_nope, R)`, `wkv_b_v (H, D_v, R)`.
- **Kernel layout:** permutes the queries to `(B, H, S_q, D_*)` and `.contiguous()`; passes `softmax_scale=self.softmax_scale`.
- **Causality translation:** `is_causal = mask is not None`; `q_start = start_pos if (is_causal and use_cache) else 0`. The kernel builds its own causal mask in SRAM keyed on **global** position (`s_q_off + q_start >= k_off`); `q_start` is the cache offset of the query block, so a cached mid-sequence prefill cannot attend its own future. A wrong `q_start` (e.g. 0 during cached decode) silently leaks future keys — see [MLA & Mixed Precision](../concepts/attention-and-precision.md) Appendix C.
- **Output:** kernel returns `(B, S_q, H, D_v)`; the wrapper flattens to `(B, S_q, H*D_v)` and returns `self.wo(...)` — identical epilogue to the other paths.
- **Caller:** `forward` (Triton branch only).

## 9. Pitfalls

1. **Forgetting `reset_cache()` between requests.** `Transformer.generate` resets internally; a bare `forward(use_cache=True)` does not — the next forward reads the previous prompt's positions `[0, end_pos)` and silently attends to stale content.
2. **Batch growth drops the prefix.** `_ensure_cache` reallocates when `bsz > _cache_batch`; the new `torch.zeros` is empty. Prefill batch 1 then decode batch 8 without a reset → the batch-1 prefix is gone.
3. **`start_pos` is global, not per-chunk.** Chunked prefill calls `forward(chunk, start_pos=chunk_start, use_cache=True)`; the write and the mask both use absolute positions. Off-by-one here is the classic cause of post-first-token gibberish.
4. **Sequence overflow is a hard error by design** (`RuntimeError`), never a silent truncation — but only if the caller respects the bound; `generate` breaks at `max_seq_len`, a manual decode loop must too.
5. **`attn_impl="triton"` silently degrades** to SDPA after the first failure (one warning). If you need to know which path ran, check `self.attn_impl` after construction/forward — production logs will not tell you per step.
6. **The cache dtype is captured from the first forward.** A device/dtype change reallocates (and clears) the cache.
7. **`use_cache=True` during training** fills the cache with detached latents — harmless but wasteful; the training loop always passes `use_cache=False`.

## 10. Check your understanding

1. **Q:** Why does the SDPA path materialize `K_nope`/`V` while the manual path does not? **A:** SDPA must materialize (it takes explicit K/V tensors); the fused bmm `ctx_kv_bmm @ cat([wkv_b_k, wkv_b_v]).T` reconstructs all heads' K and V in one operation, avoiding the per-head loop and the four-tensor `cat` of the naive reference. The manual path keeps scores in latent space (`q_nope_proj @ ctx_kvᵀ`) and is ~4× the FLOPs of MHA — it exists as the references/teaching path and the equivalence anchor.
2. **Q:** What is in the cache, exactly, and why is the write detached? **A:** The *normalized* latent `kv_normed` (192 floats) and the *rotated* rope key `k_pe` (24 floats), per token per layer. Detach keeps cache entries off the autograd graph so a decode forward never back-propagates through past positions.
3. **Q:** When does `q_start` differ from `start_pos` in `_forward_triton`? **A:** When the call is causal and cached (`mask is not None and use_cache`), `q_start = start_pos`; otherwise 0. For a cache-free full-sequence forward the block-local and global positions coincide, so 0 is correct.
4. **Q:** At `max_seq_len = 8192, mscale = 1.1, rope_factor = 1.0`, what is `softmax_scale`? **A:** `mscale = 1.1` (factor is 1.0, so no log correction), and since 8192 > 4096, `softmax_scale = 72^{-1/2} \cdot 1.1^2 \approx 0.1426` — guarded by `tests/test_models.py:TestMLAAdditional.test_softmax_scale_with_mscale`.

---

## References

- [MLA & Mixed Precision](../concepts/attention-and-precision.md) — the math and worked examples
- [R1 — Config Schema](../references/R1_config_schema.md) - every YAML key read here
- [R2 — Transformer API](../references/R2_transformer_api.md) - the caller
- [R6 — Triton API](../references/R6_triton_api.md) - the kernel this layer dispatches to
