# R2 — Transformer API Reference (`models/transformer.py`)

> **60-second summary.** `models/transformer.py` is the model body of DeepSeek-V3-Lite: an
> embedding, a stack of pre-norm blocks (MLA attention + SwiGLU/MoE FFN), a final RMSNorm and
> output head, plus the top-level entry points — causal-mask construction, training forward,
> KV-cached autoregressive `generate`, the static sampling helper, and a deduplicated parameter
> counter. Everything else (MLA internals, MoE gating, MTP) lives in sibling modules; this file
> only wires them together. It is a plain `nn.Module` — no wrappers, no compilation, no data
> loading.

**Why it exists.** One file must own the *composition* contract: how blocks stack, how the
causal mask is built and cached, how the KV cache is offset by `start_pos`, and how
training (`training/pretrain.py`) and inference (`inference/`) share one forward path. The
design goal is that every other subsystem talks to exactly five surfaces: construction,
`forward`, `forward_with_hidden`, `generate`, and `moe_layers`.

**Intuition.** The residual stream `h` flows `(bsz, seqlen, dim)` through 18 blocks; each block
adds a pre-normed attention sublayer then a pre-normed FFN sublayer (residuals are post-add, so
streams never wait on norms). The head is a bias-free linear back to vocab size, tied to the
embedding by default so one weight tensor serves both tables — `count_parameters` exists because
that tie makes naive `sum(p.numel())` overcount by 7.1M… the embedding head, precisely.

---

## 1. Module map

| Public symbol | Anchor | Role |
|---|---|---|
| `SwiGLUFFN` | `models/transformer.py:SwiGLUFFN` | Dense SwiGLU FFN (first `n_dense_layers` blocks) |
| `TransformerBlock` | `models/transformer.py:TransformerBlock` | One pre-norm MLA + FFN block |
| `Transformer` | `models/transformer.py:Transformer` | Full model: embed, blocks, norm, head |
| `count_parameters` | `models/transformer.py:count_parameters` | Deduped `(total, trainable)` count |

Private helpers are part of the API contract because training/inference reach into them
indirectly: `_build_causal_mask` defines the mask geometry the Triton kernel consumes (see
`../12_Triton_Kernels.md`), and `_sample` is called cross-module by `inference/speculative.py`
as a static method.

---

## 2. Config keys consumed

`Transformer.__init__` reads `config.get("model", config)` first, so both a flat dict and a
`{"model": {...}}` nesting are accepted (`tests/test_models.py` covers both). Keys consumed
here; everything else is forwarded untouched to `MultiHeadLatentAttention` / `DeepSeekMoE`
(see `R3_mla_api.md`, `R4_moe_api.md`).

| Key | Required / default | Consumed at | Notes |
|---|---|---|---|
| `max_seq_len` | required | `Transformer.__init__`, `generate` | Context bound; `generate` stops when `output.size(1) >= max_seq_len` |
| `vocab_size` | required | `__init__` | `embed` and `head` widths (canonical 100,018) |
| `dim` | required | `__init__`, `TransformerBlock` | Hidden width (canonical 768) |
| `n_layers` | required | `__init__` | Block count (canonical 18) |
| `n_dense_layers` | required | `TransformerBlock.__init__` | First N blocks get dense `SwiGLUFFN`; the rest get `DeepSeekMoE` (canonical 2) |
| `inter_dim` | required for dense blocks | `SwiGLUFFN.__init__` | Dense FFN hidden width (canonical 1536) |
| `weight_tying` | `False` via `.get` | `__init__` | Ties `head.weight` storage to `embed.weight`; canonical config sets `true` |
| `attn_impl`, `moe_dispatch` | dispatch keys | `__init__` (guard) | Rewritten to `sdpa`/`stacked` by `enforce_triton_env_var` unless `ENABLE_TRITON_KERNELS=1` |

The guard `models/_triton_dispatch.py:enforce_triton_env_var` runs inside
`Transformer.__init__` (not only in `training/pretrain.py`), so model construction from
inference scripts or tests cannot silently take a Triton path without the env var.

---

## 3. `SwiGLUFFN`

Dense two-gate FFN: $W_2\big(\text{silu}(W_1 x) \odot W_3 x\big)$. Purpose: the
`n_dense_layers` early blocks get a deterministic dense FFN instead of MoE routing.

### `models/transformer.py:SwiGLUFFN.__init__`

```python
def __init__(self, dim: int, inter_dim: int):
```

- `dim` — hidden width in/out. `inter_dim` — FFN hidden width.
- Creates three bias-free linears: `w1` `(inter_dim, dim)`, `w2` `(dim, inter_dim)`,
  `w3` `(inter_dim, dim)` — i.e. gate and up projections share the input, the down
  projection is `silu(gate) * up`.
- All `bias=False` (no FFN bias anywhere in the model).

### `models/transformer.py:SwiGLUFFN.forward`

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
```

- Shape contract: `(..., dim) -> (..., dim)`; `nn.Linear` handles any leading batch dims.
- Body (verbatim): `return self.w2(F.silu(self.w1(x)) * self.w3(x))` — the SwiGLU gate is
  `silu` (Sigmoid Linear Unit), not `gelu`; $W_3$ is not the gating sigmoid of a gated
  linear unit in the classic sense, it is the multiplicative gate input.
- Callers: `TransformerBlock.__init__` for `layer_id < n_dense_layers`; no module outside
  the block stack instantiates it directly.

---

## 4. `TransformerBlock`

One pre-norm residual block. Purpose: the smallest stackable unit — attention sublayer plus
FFN sublayer, both with post-add residuals.

### `models/transformer.py:TransformerBlock.__init__`

```python
def __init__(self, layer_id: int, config: dict):
```

- `layer_id` — 0-based index; decides dense vs MoE: `layer_id < config["n_dense_layers"]`.
- Builds, in order: `attn_norm` = `nn.RMSNorm(dim, eps=1e-6)`, `attn` =
  `MultiHeadLatentAttention(config, layer_id)`, `ffn_norm` = `nn.RMSNorm(dim, eps=1e-6)`,
  `ffn` = `SwiGLUFFN(dim, config["inter_dim"])` or `DeepSeekMoE(config)`.
- Stores `layer_id`, `dim`, `n_dense_layers` as attributes.

### `models/transformer.py:TransformerBlock.forward`

```python
def forward(self, x: torch.Tensor, start_pos: int = 0, mask: Optional[torch.Tensor] = None, use_cache: bool = True) -> torch.Tensor:
```

- Shape contract: `(bsz, seqlen, dim) -> (bsz, seqlen, dim)`.
- Body (verbatim):
  ```python
  x = x + self.attn(self.attn_norm(x), start_pos, mask, use_cache)
  x = x + self.ffn(self.ffn_norm(x))
  return x
  ```
- Pre-norm: the residual is added *after* the normed sublayer, so the stream itself is
  never normalized — only the branch inputs are. `start_pos`/`mask`/`use_cache` pass
  straight through to MLA (see `R3_mla_api.md` for what the attention layer does with
  them). The FFN never sees them.
- Callers: `Transformer._run_layers`; also the unit of gradient checkpointing there.

---

## 5. `Transformer`

The full model. Purpose: instantiate the DeepSeek-V3-style stack (MLA + MoE + optional MTP
wiring) and expose the training/inference entry points.

### `models/transformer.py:Transformer.__init__`

```python
def __init__(self, config: dict, use_checkpoint: bool = False):
```

- `config` — flat model dict or `{"model": {...}}`; `use_checkpoint` — enable per-block
  activation checkpointing (only active while `self.training`).
- Sequence of work (verbatim essentials):
  1. `model_cfg = config.get("model", config)`; run
     `enforce_triton_env_var(model_cfg, print)`.
  2. `self.embed = nn.Embedding(vocab_size, dim)` initialized with
     `nn.init.normal_(self.embed.weight, std=0.006)` — the whole model's init scheme is
     this one std (see `../01_Foundations.md`).
  3. `self.layers = nn.ModuleList([TransformerBlock(i, model_cfg) for i in range(n_layers)])`.
  4. `self.norm = nn.RMSNorm(dim, eps=1e-6)` — final pre-head norm (pre-norm is applied
     once more before the head).
  5. `self.head = nn.Linear(dim, vocab_size, bias=False)`; if `weight_tying`:
     `self.head.weight = self.embed.weight` — **aliases the same `Parameter` object**, so
     `state_dict` has both keys pointing at one storage (`tests/test_utils.py` asserts
     equal `data_ptr`). This is why `count_parameters` reports 411.6M deduped, not ~419M
     naive.
  6. Mask memoization fields `_mask_cache` / `_mask_key` (see §5.1).
- Effect on parameter counts (canonical config, verified): 411,632,256 total/trainable
  base (411.6M); `MultiTokenPrediction` adds ~7.1M → 418,713,984 (418.7M) with MTP.
  Active params per token ≈ 185M. The file `configs/pretrain_a100_422m.yaml` is the
  canonical config *filename*; "422M" is historical nominal, never a measured count.
- Callers: `training/pretrain.py` (`Transformer(config.model_config,
  use_checkpoint=config.grad_checkpoint)`), `inference/generate.py`
  (`Transformer(model_cfg)`), and every model-level test.

### 5.1 `models/transformer.py:Transformer._build_causal_mask`

```python
def _build_causal_mask(self, seqlen: int, kv_len: int, start_pos: int, device: torch.device) -> torch.Tensor:
```

- Purpose (docstring verbatim): *"Additive causal mask (1,1,S_q,S_kv), causal by global position."*
- **Mask geometry contract** — the exact semantics every consumer (SDPA, the Triton MLA
  kernel, chunked prefill) relies on:
  - Shape `(1, 1, seqlen, kv_len)` — broadcast over batch and heads.
  - Query global positions are `start_pos … start_pos + seqlen − 1`; key positions are
    `0 … kv_len − 1`. Element `[0,0,i,j] = 0` when `start_pos + i >= j`, else `-inf`.
  - It is an **additive** mask (0 / −∞), not multiplicative; attention adds it to the
    scores (`attn_scores + mask`).
  - Causality is by *global* position: with a KV cache spanning the past
    (`kv_len = end_pos`), a mid-sequence prefill chunk cannot attend its own future
    because its query positions are offset by `start_pos`.
- **Memoization**: cached as `self._mask_cache` keyed by the exact tuple
  `(seqlen, kv_len, start_pos, device)`; recomputed only when the key changes. `device` is
  part of the key, so moving the model to another device rebuilds automatically. Note the
  cache key holds `torch.device` — masks are never shared across devices.
- Callers: `Transformer.forward` and `forward_with_hidden` (never called directly by
  external code).

### 5.2 `models/transformer.py:Transformer._run_layers`

```python
def _run_layers(self, h: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor], use_cache: bool) -> torch.Tensor:
```

- Purpose: run the block stack, optionally under activation checkpointing.
- Body (verbatim):
  ```python
  for layer in self.layers:
      if self.use_checkpoint and self.training:
          h = torch.utils.checkpoint.checkpoint(
              layer, h, start_pos, mask, use_cache, use_reentrant=False,
          )
      else:
          h = layer(h, start_pos, mask, use_cache)
  return h
  ```
- Each `TransformerBlock` (both sublayers, its norms, and any MoE routing) is **one**
  checkpointed segment. Checkpointing engages only while `self.training` — eval/inference
  always runs the plain path regardless of `use_checkpoint`.
- Callers: `forward`, `forward_with_hidden`.

### 5.3 `models/transformer.py:Transformer.reset_cache`

```python
def reset_cache(self) -> None:
```

- Purpose: drop every per-layer KV/PE cache so the next forward starts from an empty
  context.
- Behavior: iterates `self.layers` and calls `layer.attn.reset_cache()` when present
  (`hasattr` guard). `MultiHeadLatentAttention.reset_cache` (see `R3_mla_api.md`) clears
  `kv_cache`, `pe_cache`, and the cached batch size.
- **Mandatory between sequences**: a stale cache corrupts attention because MLA attends to
  `cache[:, :end_pos]` — leftover positions from a longer previous sequence would leak in.
- Callers: `Transformer.generate` (before prefill), `inference/speculative.py`
  `SpeculativeDecoder.generate`, and chunked-prefill tests (`tests/test_mla_triton.py`).

### 5.4 `models/transformer.py:Transformer.moe_layers`

```python
def moe_layers(self):
```

- Purpose: yield the `DeepSeekMoE` FFN of every MoE block, in layer order. A generator —
  cheap to iterate; returns nothing material.
- Body (verbatim): `yield layer.ffn` for each layer whose `ffn` is an instance of
  `DeepSeekMoE`. Dense `SwiGLUFFN` blocks are skipped, so it yields exactly
  `n_layers − n_dense_layers` modules (16 at canonical config).
- Callers: `training/pretrain.py` `_update_moe_bias` (gate-bias updates) and
  `_moe_balance_metric` (load-balance loss aggregation); MoE training tests.

### 5.5 `models/transformer.py:Transformer.forward`

```python
def forward(self, tokens: torch.Tensor, start_pos: int = 0, use_cache: bool = True) -> torch.Tensor:
```

- Purpose (docstring verbatim): *"(bsz, seqlen) -> (bsz, seqlen, vocab_size). start_pos: KV-cache offset."*
- Shape contract:
  | In | Out | Notes |
  |---|---|---|
  | `tokens: (bsz, seqlen)` int | `(bsz, seqlen, vocab_size)` | logits over the full vocab |
- Behavior (verbatim essentials):
  1. **Dtype boundary**: `if tokens.dtype != torch.long: tokens = tokens.to(torch.long)` —
     accepts `uint32` (the mmap'd shard dtype in `training/pretrain.py`) and any other
     int dtype; `nn.Embedding` requires `int64`.
  2. `h = self.embed(tokens)` → `(bsz, seqlen, dim)`; `end_pos = start_pos + seqlen`.
  3. Mask selection: if `seqlen > 1`, `kv_len = end_pos if use_cache else seqlen` and
     `mask = self._build_causal_mask(seqlen, kv_len, start_pos if use_cache else 0, tokens.device)`.
     If `seqlen == 1` (single-token decode) `mask = None` — a lone query at the last
     position needs no constraint, and skipping the mask avoids rebuilding cache entries.
  4. `h = self._run_layers(h, start_pos, mask, use_cache)`; return
     `self.head(self.norm(h))`.
- Semantics of the three knobs:
  - `use_cache=True` — each MLA layer writes `(kv_normed, k_pe)` into `cache[:, start_pos:end_pos]`
    and attends to `cache[:, :end_pos]` (cache *grows* per call; see `R3_mla_api.md`).
    `start_pos` also offsets RoPE. Bounds: MLA raises `RuntimeError` when
    `start_pos + seqlen > max_seq_len`.
  - `use_cache=False` — no cache writes; attention is over the current chunk only
    (`kv_len = seqlen`, mask offset 0). This is the training path
    (`start_pos=0, use_cache=False` from `train_step`).
  - Corner combination `use_cache=False, start_pos > 0`: the mask is purely causal over the
    chunk (offset 0), but `start_pos` still reaches the layers (RoPE offset). Only sensible
    for standalone chunk computation; training never does this.
- Callers: `training/pretrain.py` `train_step` (non-MTP path: `self.model(tokens,
  start_pos=0, use_cache=False)`), `Transformer.generate` (prefill + decode),
  `inference/speculative.py` (prefill), all model tests. When MTP is enabled, `train_step`
  calls `self.model(tokens)` — that is the `MultiTokenPrediction` wrapper (see
  `R5_mtp_api.md`), which internally calls this `forward`.

### 5.6 `models/transformer.py:Transformer.forward_with_hidden`

```python
def forward_with_hidden(self, tokens: torch.Tensor, start_pos: int = 0, use_cache: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
```

- Purpose (docstring verbatim): *"Returns (logits, h). h is the pre-norm trunk hidden (V3
  feeds this to MTP blocks, which apply their own norms); logits use the normed h."*
- Identical computation to `forward` except it returns the **pre-norm** trunk hidden `h`
  alongside logits: `return self.head(self.norm(h)), h`.
- Shape contract: `(logits: (bsz, seqlen, vocab_size), h: (bsz, seqlen, dim))`.
- Defaults differ from `forward`: `use_cache=False` (training/verification posture; the
  speculative decoder passes `use_cache=True` explicitly).
- Callers: `inference/speculative.py` `SpeculativeDecoder.generate_step` — one call feeds
  both the main-token logits and the hidden state that the MTP draft block consumes
  (`mtp(hidden_last, token_main_emb)`). Tests: `tests/test_models.py`
  (`forward_with_hidden` shape).

### 5.7 `models/transformer.py:Transformer.generate`

```python
@torch.inference_mode()
def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 512, temperature: float = 1.0,
             top_p: float = 0.9, top_k: int = 0, eos_token_id: Optional[int] = None) -> torch.Tensor:
```

- Purpose (docstring verbatim): *"Autoregressive generation with KV-cache, top-p and top-k sampling."*
- Shape contract: `input_ids: (bsz, prompt_len)` → `(bsz, prompt_len + n_generated)` where
  `n_generated <= max_new_tokens`. `bsz` must be 1 for a single `eos_token_id` comparison
  to be meaningful across the batch (the `finished` mask is per-row, so batch > 1 works,
  stopping only when all rows finish).
- Guard: raises `ValueError` if `temperature < 0.0`.
- Loop invariants (verbatim essentials):
  1. Saves `was_training = self.training`, calls `reset_cache()`, `self.eval()`; restores
     `self.train()` at the end if it was training. `@torch.inference_mode()` suppresses
     autograd graph building for the whole call.
  2. **Prefill**: `self.forward(output, start_pos=0, use_cache=True)` over the whole
     prompt; `next_logits = prefill_logits[:, -1, :]`. The full prompt is cached in one
     forward; decode then appends one token per step.
  3. Per step: `next_token = self._sample(next_logits, temperature, top_p, top_k)` →
     `(bsz, 1)`; `output = torch.cat([output, next_token], dim=1)`.
  4. Termination, in order: (a) if `eos_token_id` is not `None`, `finished |= (token ==
     eos)` and break when `finished.all()`; (b) break when `output.size(1) >=
     self.max_seq_len` (context cap); (c) the `for` loop bound.
  5. Decode: `self.forward(next_token, start_pos=prompt_len + step, use_cache=True)` —
     single-token forward (`seqlen == 1` → `mask=None`), cache offset advances by one
     each step; `next_logits = decode_logits[:, -1, :]`.
- Callers: `inference/generate.py` `generate_interactive` (the non-speculative branch),
  `tests/test_inference.py`. The speculative path deliberately does **not** use
  `generate` — it drives `forward`/`forward_with_hidden` directly
  (`inference/speculative.py`).

### 5.8 `models/transformer.py:Transformer._sample`

```python
@staticmethod
def _sample(logits: torch.Tensor, temperature: float, top_p: float, top_k: int) -> torch.Tensor:
```

- Purpose (docstring verbatim): *"Temperature + top-k + top-p sampling. Temperature==0 -> argmax."*
- Shape contract: `(..., vocab_size) -> (..., 1)` — leading dims preserved, last dim
  collapsed to one sampled index. Returns the sampled token ids, `keepdim=True`.
- Algorithm (verbatim essentials):
  1. `temperature == 0.0` → `logits.argmax(dim=-1, keepdim=True)` (deterministic greedy).
  2. `logits = logits / temperature`; if `top_k > 0`, keep only the top
     `min(top_k, vocab)` logits by masking the rest with `-inf`
     (`kth_vals = logits.topk(...)[0][:, -1:]`; `masked_fill(logits < kth_vals, -inf)`).
  3. `probs = softmax(logits, -1)`; if `top_p < 1.0`, sort descending, drop the tail whose
     `(cumulative − sorted_probs) > top_p` (keeps the smallest set whose *prefix* crosses
     top_p — the classic mass-based filter, equivalent to *cumulative − p* > top_p),
     renormalize with `sum.clamp(min=1e-10)`, and sample by `torch.multinomial` over the
     **sorted** index order, gathering back the original indices.
  4. `top_p >= 1.0` → plain `torch.multinomial(probs, num_samples=1)`.
- Notes: negative temperature is not guarded here — `generate` rejects it before calling;
  `SpeculativeDecoder` calls `_sample` directly with `temperature >= 0`
  (`Transformer._sample(main_logits[:, -1, :], temperature, top_p=1.0, top_k=0)`). With
  `top_p=1.0, top_k=0` the method reduces to temperature sampling / argmax.
- Callers: `Transformer.generate`, `inference/speculative.py`, sampling tests. See
  `../10_Inference_and_Serving.md` for the theory.

---

## 6. `models/transformer.py:count_parameters`

```python
def count_parameters(model: nn.Module) -> Tuple[int, int]:
```

- Purpose (docstring verbatim): *"(total, trainable) — deduplicated by tensor id (shared weights counted once)."*
- Algorithm: iterate `model.parameters()`, skip any parameter whose `id(p)` was already
  seen, accumulate `p.numel()` into `total` and (only if `p.requires_grad`) into
  `trainable`. Returns `(total, trainable)`.
- **Why dedup matters**: with `weight_tying=True`, `embed.weight` and `head.weight` are the
  same `Parameter` object and appear twice in `parameters()`; dedup by id counts the shared
  storage once. Without dedup the base model would overcount by 100,018 × 768 ≈ 76.8M.
- Verified canonical numbers (from `training/pretrain.py` logging): 411,632,256 total =
  trainable for the base Transformer (411.6M); `MultiTokenPrediction(config, raw_model)`
  → 418,713,984 (418.7M), MTP delta ≈ 7.1M. The μP LR is then
  $6.0\mathrm{e}{-4}\times\sqrt{757\,226\,496/N}$ → **8.14e-4** base / **8.07e-4** with MTP
  (exact: 8.138e-4 / 8.069e-4); see `../08_Training_Pipeline.md`.
- Callers: `training/pretrain.py` (base + MTP totals feed the μP LR computation and the
  param log), `tests/test_training.py` (LR formula), memory estimator tests.

---

## 7. Caller map

| Caller | Uses | Contract relied on |
|---|---|---|
| `training/pretrain.py` `Pretrainer.__init__` | `Transformer(config.model_config, use_checkpoint=config.grad_checkpoint)`, `count_parameters` | flat/nested config, deduped totals |
| `training/pretrain.py` `_update_moe_bias`, `_moe_balance_metric` | `raw_model.moe_layers()` | yields only MoE FFNs, layer order |
| `training/pretrain.py` `train_step` | `self.model(tokens, start_pos=0, use_cache=False)` (non-MTP); `self.model(tokens)` (MTP wrapper) | training forward: no cache, no mask offset |
| `inference/generate.py` | `Transformer(model_cfg)`, `model.generate(...)` | `generate` loop + sampling |
| `inference/speculative.py` | `Transformer._sample` (static), `forward_with_hidden`, `main_model.embed`, `reset_cache`, prefill `forward` | hidden-return contract, static sampler |
| `tests/test_models.py`, `test_inference.py`, `test_mla_triton.py`, `test_training.py`, `test_utils.py` | construction, weight-tying pointer identity, chunked-prefill equality, cache reset | mask geometry, storage aliasing, cache lifecycle |

---

## 8. Pitfalls

- **Stale KV cache between sequences.** `forward(..., use_cache=True)` never clears the
  cache; MLA attends to `cache[:, :end_pos]`, so a shorter next sequence reads leftover
  positions. `generate` and `SpeculativeDecoder.generate` call `reset_cache()`
  internally; any hand-rolled decode loop must do the same.
- **`max_seq_len` overflow raises, not clips.** `start_pos + seqlen > max_seq_len` raises
  `RuntimeError` inside the MLA layer. `generate` avoids it by breaking at
  `output.size(1) >= max_seq_len`, but a manual forward past the bound is a hard error.
- **Mask cache is keyed by device.** The memoized `(1,1,S_q,S_kv)` mask is per
  `(seqlen, kv_len, start_pos, device)` — correct across device moves, but be aware the
  cache holds one mask per distinct key per model instance.
- **`use_checkpoint` is training-only.** `_run_layers` gates checkpointing on
  `self.training`; eval always runs the eager path. Expect no memory savings at inference.
- **`forward_with_hidden` returns the *pre-norm* hidden.** MTP blocks apply their own
  norms; do not reuse the returned `h` as if it were `norm(h)` — logits come from the
  normed stream, the hidden does not.
- **`_sample` does not validate temperature.** Only `generate` rejects negatives;
  `SpeculativeDecoder` calls it directly with non-negative values by construction. A
  negative temperature silently inverts the logit sign.
- **Weight tying is storage aliasing.** `head.weight` and `embed.weight` share one
  `Parameter`; `state_dict` contains both keys (same storage), and loading must preserve
  the aliasing (`tests/test_utils.py` guards this). Optimizer parameter lists also see the
  tensor twice — AdamW updates it twice per step unless deduplicated (the trainer's
  optimizer setup handles this; see `R7_training_api.md`).
- **Chunked prefill depends on mask offset.** With `use_cache=True` and `start_pos > 0`,
  the mask is causal by global position — a mid-sequence chunk cannot attend its own
  future. Passing `start_pos=0` for a cached chunk would leak future tokens.

---

## 9. Check your understanding

1. **Q:** With `use_cache=True`, `start_pos=8`, and a chunk of 8 tokens, what `kv_len`
   does `forward` request and what does element `[0,0,3,5]` of the mask equal?
   **A:** `kv_len = end_pos = 16`; the query at global position `8+3=11 >= 5`, so the mask
   value is `0` (attend). Query global positions run 8…15, keys 0…15.
2. **Q:** Why does `count_parameters` report 411.6M while summing `p.numel()` over
   `model.parameters()` naively gives more?
   **A:** With `weight_tying=True` the tied head is the same tensor as the embedding and
   appears twice; `id(p)` dedup counts it once.
3. **Q:** `generate` with `top_k=0, top_p=1.0, temperature=0.0` — what does each step
   produce, and why is the loop still required?
   **A:** Deterministic argmax tokens; the loop is still needed because each step depends
   on the previously sampled token (autoregressive dependence) and the KV cache.
4. **Q:** When is `mask=None` in `forward`, and why is that safe?
   **A:** When `seqlen == 1` — a single query at the (last) position has no future to
   hide, and skipping the mask avoids churning the mask cache during decode steps.

---

**Related docs:** `R3_mla_api.md` (attention + cache contract), `R4_moe_api.md` (MoE
gate/bias), `R5_mtp_api.md` (MTP wrapper over this model), `R7_training_api.md` (trainer
callsites), `R9_inference_api.md` (generation entry points),
`../02_Model_Architecture.md` (topology walkthrough), `../08_Training_Pipeline.md` (μP LR
derivation), `../10_Inference_and_Serving.md` (sampling theory), `../12_Triton_Kernels.md`
(mask consumption in kernels).

<!-- docs:verified 2026-08-04 · 59aeef3 -->
