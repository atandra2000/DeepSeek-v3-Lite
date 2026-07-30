# Transformer — Top-Level Wiring

## A Comprehensive Technical Reference

> **Prerequisites:** [foundations.md](foundations.md), [MLA.md](MLA.md), [moe.md](moe.md).

> **Covers**: `Transformer`, `TransformerBlock`, `SwiGLUFFN` in `models/transformer.py` — how the full DeepSeek-V3 stack is assembled and executed.

> **Read this if** you need the end-to-end forward/generate wiring. **Read first** if new to the repo: [architecture.md](architecture.md) → [MLA.md](MLA.md) / [moe.md](moe.md).

---

## Table of Contents

1. [Abstract](#abstract)
2. [Pre-Norm Residual Architecture](#pre-norm-residual-architecture)
3. [Layer Topology (422M)](#layer-topology-422m)
4. [SwiGLUFFN](#swiglu-ffn)
5. [TransformerBlock](#transformerblock)
6. [Transformer Class](#transformer-class)
7. [Forward Contracts](#forward-contracts)
8. [Causal Mask Cache](#causal-mask-cache)
9. [Weight Tying](#weight-tying)
10. [Generation API](#generation-api)
11. [Gradient Checkpointing](#gradient-checkpointing)
12. [Config Shape](#config-shape)
13. [Appendix A — Tensor shape trace](#appendix-a--tensor-shape-trace)
14. [Appendix B — FAQ](#appendix-b--faq)
15. [Appendix C — Glossary](#appendix-c--glossary)
16. [Load-Bearing Invariants](#load-bearing-invariants)
17. [References](#references)

---

## Abstract

The `Transformer` class is the root `nn.Module` for DeepSeek-V3-Lite. It stacks 18 `TransformerBlock` layers (2 dense SwiGLU + 16 MoE), each consisting of pre-norm MLA attention and pre-norm FFN. It exposes three critical interfaces: `forward` (training/inference logits), `forward_with_hidden` (MTP training), and `generate` (autoregressive decode with KV cache).



---

## Line-by-Line Construction Walkthrough

### `Transformer.__init__` (`models/transformer.py`)

```python
model_cfg = config.get("model", config)
enforce_triton_env_var(model_cfg, print)  # AGENTS rule #7
```

**Step 1 — Config unwrap:** Tests pass flat dicts; YAML passes nested `{"model": {...}}`. Single unwrap point prevents double-nesting bugs.

**Step 2 — Triton guard:** If `attn_impl=triton` or `moe_dispatch=triton_grouped` without `ENABLE_TRITON_KERNELS=1`, force-back to `sdpa`/`stacked` with one warning.

**Step 3 — Embedding:** `nn.Embedding(vocab_size, dim)` with `N(0, 0.006²)` init.

**Step 4 — Layers:** `ModuleList([TransformerBlock(i, model_cfg) for i in range(n_layers)])`. Layer index determines dense vs MoE.

**Step 5 — Head + tying:** `Linear(dim, vocab_size, bias=False)`. If `weight_tying`: `head.weight = embed.weight` (same storage).

### `TransformerBlock.forward`

```python
x = x + self.attn(self.attn_norm(x), start_pos, mask, use_cache)
x = x + self.ffn(self.ffn_norm(x))
```

**Residual highway:** $\mathbf{x}$ shape $(B, S, d)$ unchanged across both sub-layers. Gradients flow through `+` directly to earlier layers (pre-norm benefit).

**MoE note:** `DeepSeekMoE.forward(x)` flattens to $(B \cdot S, d)$ internally for routing — position-agnostic expert assignment.

### `count_parameters` — deduplication logic

```python
seen = set()
for p in model.parameters():
    if id(p) not in seen:
        seen.add(id(p))
        total += p.numel()
```

Weight tying means `embed.weight` and `head.weight` share `id` — counted once. Critical for μP LR denominator.

---

## FLOP Breakdown Per Layer (422M)

| Layer type | Attention FLOPs/token | FFN FLOPs/token |
|---|---|---|
| Dense (0-1) | $\approx 4 S d^2$ (MLA compressed) | $6 d \cdot I = 6 \times 768 \times 1536$ |
| MoE (2-17) | same | $6 d \cdot I_{\text{moe}} \times k = 6 \times 768 \times 384 \times 4$ routed + shared |

MoE **executes** 5 experts but **stores** 21 — memory bound, not compute bound at inference.

---

## Comparison with GPT-2 / LLaMA Block

| Feature | GPT-2 | LLaMA | DeepSeek-V3-Lite |
|---|---|---|---|
| Norm | LayerNorm | RMSNorm | RMSNorm |
| FFN | GELU MLP | SwiGLU | SwiGLU / MoE |
| Attention | MHA | GQA | MLA |
| Position | Learned | RoPE | RoPE (decoupled in MLA) |
| MoE | No | No (dense) | Yes (layers 2-17) |

This repo follows DeepSeek-V3 family conventions, not GPT-2 — do not port GPT-2 hyperparameters blindly.

---

## `enforce_triton_env_var` Integration

Called at `Transformer.__init__` and mirrored in `Pretrainer`. Ensures default-config runs never silently enable Triton (AGENTS.md hard rule). Tests in `test_force_back.py` lock this behaviour.


---

## Pre-Norm Residual Architecture

DeepSeek-V3 uses **pre-normalisation** (RMSNorm before each sub-layer):

```
x = x + Attention(RMSNorm(x))
x = x + FFN(RMSNorm(x))
```

Compared to post-norm (norm after residual), pre-norm:

- Stabilises training at depth 18+ without extra tricks
- Keeps residual stream magnitude bounded
- Matches LLaMA / DeepSeek-V3 convention

RMSNorm (no mean centering, scale-only):

```
RMSNorm(x) = x / RMS(x) · γ        where RMS(x) = sqrt(mean(x²) + ε)
```

`eps = 1e-6` throughout this repo.

---

## Layer Topology (422M)

| Layer ID | FFN type | FFN width | Attention |
|---|---|---|---|
| 0, 1 | `SwiGLUFFN` | `inter_dim=1536` | MLA |
| 2–17 | `DeepSeekMoE` | `moe_inter_dim=384` × 21 experts | MLA |

Selection logic in `TransformerBlock.__init__`:

```python
self.ffn = SwiGLUFFN(dim, inter_dim) if layer_id < n_dense_layers else DeepSeekMoE(config)
```

With `n_dense_layers=2`, layers 0–1 are dense, 2–17 are MoE.

---

## SwiGLUFFN

Dense FFN used in layers 0–1 and as the building block inside each MoE expert:

```
SwiGLU(x) = W₂ · silu(W₁x) ⊙ W₃x
```

```python
class SwiGLUFFN(nn.Module):
    w1: Linear(dim → inter_dim)
    w2: Linear(inter_dim → dim)
    w3: Linear(dim → inter_dim)
```

**Parameter count (dense layer):** `3 × dim × inter_dim = 3 × 768 × 1536 ≈ 3.5M`.

**FLOPs per token:** `≈ 6 × dim × inter_dim` (three matmuls, gate+up share input).

---

### The residual stream — central design object

Anthropic's "residual stream" framing applies directly: each layer reads from and writes to the same tensor $\mathbf{x} \in \mathbb{R}^{B \times S \times d}$. Attention and FFN are **updates** to this stream, not replacements.

$$
\mathbf{x}_{\ell+1} = \mathbf{x}_\ell + \mathrm{Attn}_\ell(\mathrm{RMSNorm}(\mathbf{x}_\ell)) + \mathrm{FFN}_\ell(\mathrm{RMSNorm}(\mathbf{x}_\ell'))
$$

At 422M: $d=768$, $L=18$. The stream width never changes — only the information content grows deeper in the stack.

---

## TransformerBlock

```python
class TransformerBlock(nn.Module):
    attn_norm: RMSNorm(dim)
    attn: MultiHeadLatentAttention(config, layer_id)
    ffn_norm: RMSNorm(dim)
    ffn: SwiGLUFFN | DeepSeekMoE
```

Forward:

```python
def forward(x, start_pos, mask, use_cache):
    x = x + self.attn(self.attn_norm(x), start_pos, mask, use_cache)
    x = x + self.ffn(self.ffn_norm(x))
    return x
```

**Note:** MoE `DeepSeekMoE.forward` only takes `x` — no `start_pos` or cache. Routing is position-agnostic (token-level, not sequence-level).

---

## Transformer Class

### Construction

```python
Transformer(config, use_checkpoint=False)
```

1. Unwrap nested config: `model_cfg = config.get("model", config)`
2. `enforce_triton_env_var(model_cfg)` — force-back Triton keys if env var unset
3. `nn.Embedding(vocab_size, dim)` — init `N(0, 0.006²)`
4. `ModuleList` of 18 `TransformerBlock`s
5. Final `RMSNorm(dim)`
6. `Linear(dim, vocab_size)` — LM head
7. Optional weight tying: `head.weight = embed.weight`

### `moe_layers()`

Generator yielding `DeepSeekMoE` instances from MoE layers. Used by `Pretrainer` for bias updates and balance metrics.

### `reset_cache()`

Calls `attn.reset_cache()` on every MLA layer. **Required** before each new generation session.

---

## Forward Contracts

### `forward(tokens, start_pos=0, use_cache=True) → (B, S, V)`

Primary interface. Returns vocabulary logits.

- Casts non-Long tokens to `int64` at boundary (uint32 shards).
- Builds causal mask when `seqlen > 1`; `None` for single-token decode.
- Runs all layers via `_run_layers`.
- Returns `head(norm(h))`.

**Training:** `use_cache=False` (set in `train_step`). No KV cache writes.

**Inference prefill:** `use_cache=True`, `start_pos=0`, full prompt length.

**Inference decode:** `use_cache=True`, `start_pos=prompt_len+step`, `seqlen=1`.

### `forward_with_hidden(tokens, start_pos=0, use_cache=False) → (logits, h)`

Returns `(head(norm(h)), h)` where `h` is the **pre-norm** trunk hidden.

- Used by `MultiTokenPrediction` to feed MTP heads.
- `use_cache=False` during training.
- `use_cache=True` in `SpeculativeDecoder.generate_step` (cache grows).

### Shape reference (422M, B=2, S=128)

| Tensor | Shape |
|---|---|
| `tokens` | (2, 128) |
| `h` (embedding) | (2, 128, 768) |
| per-layer `x` | (2, 128, 768) |
| `logits` | (2, 128, 100018) |

---

## Causal Mask Cache

```python
def _build_causal_mask(seqlen, device):
    mask = triu(-inf, diagonal=1)    # (S, S)
    return mask.unsqueeze(0).unsqueeze(0)   # (1, 1, S, S)
```

Cached in `_mask_cache` keyed by `(seqlen, device)`.

**Skipped when `seqlen == 1`:** During autoregressive decode, each step passes a single token. A causal mask is unnecessary (one query, many keys from cache). Skipping saves allocation and SDPA overhead.

---

## Weight Tying

```python
if weight_tying:
    self.head.weight = self.embed.weight
```

**Effect:** `head.weight` and `embed.weight` are the **same storage**. Saves ~77M parameters (`100018 × 768`).

**Checkpoint:** `save_checkpoint` drops `head.weight` from state dict (redundant). Load restores via `embed.weight` with `strict=False`.

**Parameter counting:** `count_parameters` deduplicates by `id(p)`:

```python
seen = set()
for p in model.parameters():
    if id(p) not in seen:
        seen.add(id(p))
        total += p.numel()
```

---

## Generation API

```python
@torch.inference_mode()
def generate(input_ids, max_new_tokens=512, temperature=1.0,
             top_p=0.9, top_k=0, eos_token_id=None) -> Tensor
```

### Flow

1. Save/restore `self.training` mode
2. `reset_cache()`
3. `eval()`
4. **Prefill:** `forward(prompt, start_pos=0, use_cache=True)` → logits for all prompt positions
5. **Decode loop:** sample from last logits, append token, `forward(single_token, start_pos=prompt_len+step)`
6. Stop on EOS or `max_seq_len`

### `_sample(logits, temperature, top_p, top_k)`

| Setting | Behaviour |
|---|---|
| `temperature=0` | Greedy argmax |
| `temperature>0` | Scale logits, then sample |
| `top_k>0` | Mask logits below k-th largest |
| `top_p<1` | Nucleus sampling on sorted probabilities |

---

## Gradient Checkpointing

```python
Transformer(config, use_checkpoint=True)
```

When `use_checkpoint and self.training`:

```python
h = torch.utils.checkpoint.checkpoint(
    layer, h, start_pos, mask, use_cache, use_reentrant=False
)
```

**Trade-off:** ~3× less activation memory, ~33% more backward FLOPs (recompute forward during backward).

422M config: `grad_checkpoint: true`. 1650 smoke config: `false` (model fits without it).

---

## Config Shape

`Transformer.__init__` accepts:

```python
# Flat (tests, direct construction)
cfg = {"vocab_size": 100018, "dim": 768, ...}

# Nested (YAML from pretrain)
cfg = {"model": {"vocab_size": 100018, ...}, "training": {...}}
```

Unwrap: `model_cfg = config.get("model", config)`.

`Pretrainer` passes the full YAML dict; `Transformer` unwraps internally. MLA/MoE receive the unwrapped flat dict via `TransformerBlock`.

---


## `generate()` and `_sample()` — Full Walkthrough

See `models/transformer.py:110-158`. Generation is the only production code path that sets `use_cache=True` on every layer.

**Training forward:** `mask` built when `seqlen>1`; writes no cache when `use_cache=False`.
**Prefill:** `seqlen=prompt_len`, full causal mask, fills MLA `kv_cache`/`pe_cache` slots `[0:prompt_len)`.
**Decode:** `seqlen=1`, `mask=None` (fast path), writes one slot at `start_pos`.

### Why `reset_cache()` is mandatory

Without reset, a second `generate()` call appends to stale cache tensors — logits reference wrong positions. `generate()` always calls `reset_cache()` at entry.

### Batch generation caveat

All rows in batch `B>1` share the same loop length; EOS only stops a row logically — implementation checks `finished.all()` but still runs decode steps until all finish or `max_new_tokens` exhausted.

---

## Parameter Budget by Component (422M)

Approximate counts from `Pretrainer._log_per_component_params` categories:

| Component | ~Params | % of total |
|---|---|---|
| MoE routed experts | 298M | 70% |
| MLA attention | 52M | 12% |
| Embeddings + tied head | 77M | 18% (counted once) |
| Dense SwiGLU (layers 0–1) | 7M | 2% |
| MoE shared experts | 14M | 3% |
| MTP (depth=1) | 3M | <1% |
| RMSNorm + gates | <1M | <1% |

**μP note:** `count_parameters` deduplicates tied embed/head — do not sum embedding and lm_head rows separately.

---

## `forward_with_hidden` — MTP Contract

```python
h = embed(tokens)
h = _run_layers(h, start_pos, mask, use_cache=False)
return head(norm(h)), h   # logits use normed h; MTP consumes raw h
```

MTP blocks apply **their own** `RMSNorm` before fusion — the returned `h` is the pre-final-norm trunk state, matching DeepSeek-V3's "main hidden before head norm" convention in this repo.

---

## Causal Mask Cache

`_build_causal_mask(seqlen, device)` caches `(1,1,S,S)` upper-triangular `-inf` masks keyed by `(seqlen, device)`.

| Call context | seqlen | mask |
|---|---|---|
| Training | 2048 | full causal |
| Prefill | prompt_len | full causal |
| Decode | 1 | `None` (no self-attention mask needed) |

Changing `max_seq_len` in config without clearing `_mask_cache` is safe — cache miss triggers rebuild when `seqlen` differs.

---

## Extension Points (Safe vs Unsafe)

| Change | Safe? | Notes |
|---|---|---|
| `n_dense_layers` | Yes | Test with `small_cfg` first |
| `weight_tying: false` | Yes | Doubles embedding+head params; update μP count |
| `n_layers` | Yes | Watch VRAM linear in depth |
| Remove `enforce_triton_env_var` | **No** | Violates AGENTS hard rule |
| Post-norm residuals | **No** | Untested; breaks training stability |
| `use_cache=True` in training | **No** | Breaks gradients / leaks stale state |

---

## Worked Tensor Trace — Decode Step

Config: `dim=768`, `vocab=100018`, `B=1`, one new token at `start_pos=10`:

```
input_ids:     (1, 1)
embed:         (1, 1, 768)
layer 0 MLA:   (1, 1, 768)  attends to cache[0:10] + current
layer 0 FFN:   (1, 1, 768)
...
norm:          (1, 1, 768)
head:          (1, 1, 100018)
```

MoE FFN internally views `(1,1,768)` as `(1,768)` for routing — expert indices shape `(1,)`.



## Appendix A — Tensor shape trace

Single forward, `B=1, S=4`, 422M:

```
input_ids:     (1, 4)  int64
embed:         (1, 4, 768)
  layer 0 MLA: (1, 4, 768)  + residual
  layer 0 FFN: (1, 4, 768)  SwiGLU 1536
  ...
  layer 17:    (1, 4, 768)  MoE
norm:          (1, 4, 768)
head:          (1, 4, 100018)
```

Decode step 1 after prefill (prompt len=4):

```
input_ids:     (1, 1)  single new token
embed:         (1, 1, 768)
mask:          None  (seqlen==1)
KV cache:      positions 0..4 populated
start_pos:     4
output logits: (1, 1, 100018)
```

---

## Appendix B — FAQ

**Q: Why 2 dense layers before MoE?** DeepSeek-V3 uses early dense layers for stable low-level feature extraction before sparse routing. This matches the paper's layer schedule.

**Q: Can I make all layers MoE?** Set `n_dense_layers=0`. Untested at 422M scale but architecturally valid.

**Q: Does `generate` support batch size > 1?** Yes, but all sequences in the batch share the same generation length (no per-sequence early stopping except EOS).

---

## Appendix C — Glossary

| Term | Meaning |
|---|---|
| `n_dense_layers` | Count of initial dense SwiGLU layers |
| `inter_dim` | Dense FFN hidden width (1536) |
| `weight_tying` | Share embed and LM head weights |
| `use_cache` | Enable MLA KV cache read/write |
| `start_pos` | Cache offset for current forward slice |
| `use_checkpoint` | Enable gradient checkpointing |

---

## Load-Bearing Invariants

1. **Weight tying** — `head.weight = embed.weight`. Removing breaks generation.
2. **Mask skip at seqlen=1** — required for efficient decode.
3. **`use_cache=False` in training** — prevents detached cache pollution.
4. **`reset_cache()` before generate** — prevents cross-request cache bleed.
5. **Config unwrap** — always `config.get("model", config)` at Transformer boundary.
6. **`enforce_triton_env_var`** — called at construction, not per-forward.

---

## References

- [transformer.md](transformer.md) → [MLA.md](MLA.md), [moe.md](moe.md)
- `models/transformer.py` — authoritative source
- [architecture.md](architecture.md) — system-level overview

## `generate()` — Full Source Walkthrough

`models/transformer.py:generate` (inference-only):

```python
@torch.inference_mode()
def generate(self, input_ids, max_new_tokens=512, temperature=1.0,
             top_p=0.9, top_k=0, eos_token_id=None):
    was_training = self.training
    self.reset_cache()
    self.eval()
    bsz, prompt_len = input_ids.shape
    output = input_ids.clone()
    prefill_logits = self.forward(output, start_pos=0, use_cache=True)
    next_logits = prefill_logits[:, -1, :]
    finished = torch.zeros(bsz, dtype=torch.bool, device=input_ids.device)
    for step in range(max_new_tokens):
        next_token = self._sample(next_logits, temperature, top_p, top_k)
        output = torch.cat([output, next_token], dim=1)
        if eos_token_id is not None:
            finished = finished | (next_token.squeeze(-1) == eos_token_id)
            if finished.all():
                break
        if output.size(1) >= self.max_seq_len:
            break
        decode_logits = self.forward(next_token, start_pos=prompt_len + step, use_cache=True)
        next_logits = decode_logits[:, -1, :]
    if was_training:
        self.train()
    return output
```

**Batch semantics:** All sequences in batch share `max_new_tokens`; per-sequence early stop only when **all** rows hit EOS (`finished.all()`).

---

## Parameter Budget — 422M Breakdown

From `Pretrainer._log_per_component_params` (typical run):

| Component | ~Params | ~% |
|---|---|---|
| MoE routed experts | ~210M | 50% |
| Embedding (tied head) | ~77M | 18% |
| MLA attention | ~45M | 11% |
| MoE shared experts | ~25M | 6% |
| Dense SwiGLU (layers 0-1) | ~7M | 2% |
| MTP modules | ~3M | 1% |
| Gates + norms | remainder | |

MoE **stores** 21 experts per layer but **executes** 5 per token.

---

## `_build_causal_mask` Cache

```python
def _build_causal_mask(self, seqlen, device):
    if cache miss or device change:
        mask = triu(-inf, diagonal=1)  # (S, S)
        cache (1, 1, S, S)
    return cache
```

**Invalidation:** New `seqlen` or device change rebuilds. Decode with `seqlen=1` skips mask entirely (`mask=None`).

---

## Extension Points

| Goal | Where to change |
|---|---|
| Add layer type | `TransformerBlock.__init__` branch on `layer_id` |
| New position encoding | `models/mla.py` RoPE section |
| Batched speculative decode | `inference/speculative.py` (not implemented) |
| Multi-GPU | New `training/pretrain_distributed.py` (out of scope) |
| Different vocab | YAML `vocab_size` + re-tokenize data |

<!-- docs:verified 2026-07-31 · 5a880d2 -->
