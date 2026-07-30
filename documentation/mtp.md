# Multi-Token Prediction (MTP) + Speculative Decoding

## A Comprehensive Technical Reference

> **Prerequisites:** [foundations.md](foundations.md) §9, [transformer.md](transformer.md).

> **Covers**: DeepSeek-V3 MTP auxiliary heads, training loss coupling, and MTP-based speculative decoding in this repo (`models/mtp.py`, `inference/speculative.py`).

> **Read this if** you're working on MTP loss alignment or speculative decoding. **Skip if** you only need the standard train loop → [training.md](training.md).

---

## Table of Contents

1. [Abstract](#abstract)
2. [Motivation — Denser Training Signal](#motivation--denser-training-signal)
3. [DeepSeek-V3 MTP in the Paper](#deepseek-v3-mtp-in-the-paper)
4. [Mathematical Formulation](#mathematical-formulation)
5. [MTPBlock — Internal Architecture](#mtpblock--internal-architecture)
6. [MTPModule — Single Depth Head](#mtpmodule--single-depth-head)
7. [MultiTokenPrediction — Training Wrapper](#multitokenprediction--training-wrapper)
8. [Training Loss](#training-loss)
9. [Speculative Decoding](#speculative-decoding)
10. [KV Cache Sharing](#kv-cache-sharing)
11. [Checkpoint Format](#checkpoint-format)
12. [Comparison: MTP vs Standard Decode](#comparison-mtp-vs-standard-decode)
13. [Appendix A — Worked alignment example](#appendix-a--worked-alignment-example)
14. [Appendix B — Gradient flow](#appendix-b--gradient-flow)
15. [Appendix C — FAQ](#appendix-c--faq)
16. [Appendix D — Glossary](#appendix-d--glossary)
17. [Load-Bearing Invariants](#load-bearing-invariants)
18. [Implementation Checklist](#implementation-checklist)
19. [References](#references)

---

## Abstract

**Multi-Token Prediction (MTP)** trains auxiliary heads to predict tokens *beyond* the immediate next-token objective. In DeepSeek-V3, MTP modules predict `t+2`, `t+3`, … in parallel with the main head's `t+1` prediction. This densifies the training signal and, critically, enables **speculative decoding** at inference: a lightweight draft head proposes a future token that the main model verifies in one forward pass.

This repo implements `mtp_depth=1` (one auxiliary head predicting `t+2`) with `mtp_loss_weight=0.3`. The MTP block uses standard `nn.MultiheadAttention` (not MLA) and shares the main model's embedding and LM head.

---

## Motivation — Denser Training Signal

Standard causal LM training optimises one target per position:

```
∀ position t:  predict token_{t+1} from hidden_t
```

MTP adds auxiliary targets:

```
∀ position t:  predict token_{t+2} from (hidden_t, embed(token_{t+1}))
```

**Benefits:**

1. **More gradient per forward** — each token position supervises multiple future predictions.
2. **Representation quality** — the trunk hidden state must encode enough information to skip one step ahead.
3. **Inference acceleration** — the MTP head becomes a draft model for speculative decoding without training a separate model.

**Cost:** Extra parameters (~3M for depth=1 at 422M scale) and ~5% extra compute per training step.

---

## DeepSeek-V3 MTP in the Paper

DeepSeek-V3 (arXiv:2412.19437) stacks MTP modules with increasing depth:

- Depth 1: predict `t+2` given trunk hidden at `t` and embedding of `t+1`
- Depth 2: predict `t+3` given depth-1 hidden and embedding of `t+2`
- …

Each depth module has the same block structure (norm → fuse → attention → FFN) but **independent weights**. The output head is shared across main model and all MTP depths.

This repo implements depth=1 only (`mtp_depth: 1` in YAML). The `ModuleList` in `MultiTokenPrediction` is structured to support `depth > 1` if extended.

---

## Mathematical Formulation

Let `h_t ∈ ℝ^d` be the trunk hidden state at position `t` (pre-final-RMSNorm). Let `e_t = Embed(x_t)`.

### Main head (standard LM)

```
logits_t^{(0)} = W_head · RMSNorm(h_t)        predicts x_{t+1}
```

### MTP depth d (d = 1, 2, …)

```
h_t^{(d)} = MTPBlock_d( h_t^{(d-1)}, e_{t+d} )     where h_t^{(0)} = h_t
logits_t^{(d)} = W_head · RMSNorm(h_t^{(d)})       predicts x_{t+d+1}
```

For depth=1:

```
h_t^{(1)} = MTPBlock_1( h_t, e_{t+1} )
logits_t^{(1)} predicts x_{t+2}
```

### Training loss

```
L = L_main + λ · mean_d( L_MTP^{(d)} )

L_main = CE(logits^{(0)}, targets)
L_MTP^{(d)} = CE(logits^{(d)}, targets shifted by d+1)
λ = mtp_loss_weight = 0.3
```

---

## MTPBlock — Internal Architecture

```python
class MTPBlock(nn.Module):
```

**Data flow:**

```
prev_hidden (B, S, D)          target_emb (B, S, D)
       │                              │
       ▼                              ▼
  RMSNorm (norm_h)              RMSNorm (norm_e)
       │                              │
       └──────────┬───────────────────┘
                  ▼
         cat → Linear(2D → D)    # fusion projection
                  │
                  ▼
         RMSNorm → MultiheadAttention (causal mask)
                  │  + residual
                  ▼
         RMSNorm → SwiGLU FFN
                  │  + residual
                  ▼
            output (B, S, D)
```

**Why independent norms on hidden and embedding?** The trunk hidden and token embedding live in different semantic spaces (post-17-layers vs lookup table). Separate RMSNorm before fusion prevents scale mismatch.

**Why `nn.MultiheadAttention` instead of MLA?** MTP operates on short aligned windows during training (not full-context decode with KV cache). Standard SDPA is simpler and sufficient. MTP has its own `_causal_mask` buffer — it does **not** share MLA's KV cache.

---

## MTPModule — Single Depth Head

```python
class MTPModule(nn.Module):
    def forward(prev_hidden, target_emb) -> (logits, h_norm)
```

- `output_head` is set externally via `set_output_head(main_model.head)` — **shared storage** with the main LM head.
- Raises `RuntimeError` if `output_head` not set before forward.
- Raises `ValueError` on shape mismatch between `prev_hidden` and `target_emb`.
- Returns `(logits, h_norm)` where `h_norm` is the post-block, post-RMSNorm hidden (fed to the next depth if `depth > 1`).

---

## MultiTokenPrediction — Training Wrapper

```python
class MultiTokenPrediction(nn.Module):
    def forward(tokens) -> (main_logits, mtp_pairs)
```

**Step-by-step:**

1. `main_logits, prev_h = main_model.forward_with_hidden(tokens)`
   - `prev_h` is the **pre-norm** trunk hidden `(B, S, D)`.
   - `main_logits` uses `head(norm(h))`.

2. For each depth `d` (0-indexed, depth value = d+1):
   ```python
   usable = seq_len - d - 2
   h_in  = prev_h[:, :usable]
   emb   = embed(tokens[:, d+1 : d+1+usable])
   tgt   = tokens[:, d+2 : d+2+usable]
   logits, hidden = mtp_modules[d](h_in, emb)
   mtp_pairs.append((logits, tgt))
   prev_h = hidden   # chain for depth > 1
   ```

3. If `usable <= 0` (sequence too short), skip that depth.

**Embedding sharing:** `self.add_module("embed", main_model.embed)` — same `nn.Embedding` tensor, counted once in optimiser dedup.

---

## Training Loss

```python
def compute_loss(main_logits, targets, mtp_pairs):
    main_loss = CE(main_logits, targets)
    if not mtp_pairs:
        return main_loss, main_loss, zeros
    mtp_loss = mean(CE(logits, tgt) for logits, tgt in mtp_pairs)
    return main_loss + 0.3 * mtp_loss, main_loss, mtp_loss
```

**Empty pairs case:** When `seq_len < d + 2` for all depths, `mtp_pairs` is empty. All three return values equal `main_loss` (MTP slot is zero tensor).

**Gradient accumulation:** Only the **total** loss is divided by `gradient_accumulation_steps` in `train_step`. Main and MTP components are not separately scaled.

---

## Speculative Decoding

`SpeculativeDecoder` in `inference/speculative.py` uses the depth-1 MTP head as a draft model.

### Algorithm (per generation step)

```
Input: last_token, start_pos (current cache position)

1. main_logits = main_model(last_token, start_pos, use_cache=True)
2. t1 = argmax(softmax(main_logits[:, -1]))
3. hidden = forward_with_hidden(t1, start_pos=t1_pos, use_cache=True)[1][:, -1]
4. draft_logits = mtp(hidden, embed(t1))
5. t2 = argmax(softmax(draft_logits[:, -1]))
6. p_main  = softmax(main_logits)[t2]
7. p_draft = softmax(draft_logits)[t2]
8. accept = (p_main >= threshold * p_draft)   # default threshold=0.8
9. Emit t1; if accept, also emit t2
```

### Acceptance interpretation

This is a **simplified greedy verifier**, not full rejection sampling:

- Compares argmax draft token's probability under main vs draft distributions.
- If the main model agrees the draft token is likely enough, accept both tokens in one "step".
- Otherwise fall back to one token (standard decode speed).

Measured acceptance ≈ **0.8** on smoke tests → expected **1.5–2× throughput** vs naive single-token decode.

### `generate()` loop

```python
reset_cache()
prefill(prompt)                    # fills KV cache for all prompt positions
while n_generated < max_new_tokens:
    token_main, token_draft, accepted = generate_step(last_token, start_pos)
    emit token_main
    if accepted: emit token_draft
    check EOS
```

---

## KV Cache Sharing

| Component | KV cache? | Notes |
|---|---|---|
| Main model MLA | yes | Grows during prefill + decode |
| MTP MultiheadAttention | **no** | Operates on length-1 windows at decode |
| SpeculativeDecoder | reuses main cache | No separate MTP cache |

**Load-bearing:** `generate_step` calls `forward_with_hidden(..., use_cache=True)`, so the main model's MLA cache grows by one position per accepted main token. The MTP head sees only the latest hidden state — it never needs historical KV.

**Pitfall:** Forgetting `reset_cache()` between conversations leaves stale cache entries from the previous prompt.

---

## Checkpoint Format

MTP weights are saved with `mtp.` prefix:

```
mtp.mtp_modules.0.block.norm_h.weight
mtp.mtp_modules.0.block.proj.weight
...
```

On load (`inference/generate.py`):

```python
mtp_state = {k.removeprefix("mtp."): v for k, v in state.items() if k.startswith("mtp.")}
mtp_module.load_state_dict(mtp_state, strict=False)
```

Optimizer state for MTP is **not** saved separately — MTP params are included in the main optimizer state dict via `MultiTokenPrediction.parameters()`.

---

## Comparison: MTP vs Standard Decode

| Aspect | Standard `generate()` | `SpeculativeDecoder` |
|---|---|---|
| Tokens per step | 1 | 1 or 2 |
| Forward passes per token | 1 | ~0.55 (at 0.8 accept) |
| Requires MTP weights | no | yes |
| Sampling | temperature, top-k, top-p | greedy only |
| KV cache | main MLA | main MLA only |
| Training overhead | none (if mtp_depth=0) | +30% loss weight |

---

## Appendix A — Worked alignment example

`seq_len=6`, tokens `[a, b, c, d, e, f]`, depth=1:

```
Main head at position t predicts token t+1:
  pos 0→b, 1→c, 2→d, 3→e, 4→f

MTP depth 1:
  usable = 6 - 0 - 2 = 4
  h_in  = hidden[:, 0:4]     positions 0,1,2,3
  emb   = embed([b,c,d,e])   tokens at t+1
  tgt   = [c,d,e,f]          tokens at t+2

Alignment:
  h[0] + emb(b) → predict c
  h[1] + emb(c) → predict d
  h[2] + emb(d) → predict e
  h[3] + emb(e) → predict f
```

---

## Appendix B — Gradient flow

```
total_loss = main_loss + 0.3 * mtp_loss
     │              │
     ▼              ▼
main_logits    mtp_logits
     │              │
     ▼              ▼
head.weight ◄── SHARED ──► head.weight
     │              │
     ▼              ▼
norm(h)         MTPBlock → norm → ...
     │              │
     ▼              ▼
trunk hidden    embed(tokens) ──► embed.weight (shared)
     │
     ▼
18 × TransformerBlock (all receive gradients from both paths)
```

The shared head and embed mean MTP training **also improves the main head** — the auxiliary loss is not isolated.

---

## Appendix C — FAQ

**Q: Why not use MLA in MTPBlock?** MTP windows are short and MTP doesn't use KV cache at inference. MHA is simpler and matches the "lightweight draft" role.

**Q: Can I disable MTP?** Set `mtp_depth: 0` or `mtp_loss_weight: 0.0` in config. `Pretrainer` skips the wrapper.

**Q: Does speculative decoding work without trained MTP weights?** It runs but acceptance rate collapses — the draft head is random.

**Q: Why greedy only in speculative decode?** Rejection sampling with temperature requires matching draft and target distributions exactly. The current implementation uses argmax for simplicity.

---

## Appendix D — Glossary

| Term | Meaning |
|---|---|
| `mtp_depth` | Number of MTP heads (canonical: 1) |
| `mtp_loss_weight` | λ in `L = L_main + λ·L_mtp` (0.3) |
| `usable` | `seq_len - d - 2`, valid MTP window length |
| `mtp_pairs` | List of `(logits, targets)` per depth |
| `prev_h` | Pre-norm trunk hidden from `forward_with_hidden` |
| `acceptance_threshold` | Min probability ratio to accept draft (0.8) |

---


## Historical Context — Multi-Token Prediction

MTP in DeepSeek-V3 (arXiv:2412.19437) extends the **multi-token prediction** idea explored in:

- **Medusa** (Cai et al., 2024) — multiple heads on frozen backbone
- **EAGLE** (Li et al., 2024) — feature-based draft models
- **DeepSeek MTP** — integrated training with shared embedding/head

**Key difference from Medusa:** MTP modules are trained **end-to-end** with the main model from scratch, not bolted on post-hoc. Gradients flow through shared trunk.

---

## Gradient Flow Analysis

### Main loss path

$$
\frac{\partial \mathcal{L}_{\text{main}}}{\partial \theta_{\text{trunk}}} \neq 0
$$

Standard backprop through all 18 layers.

### MTP loss path

$$
\frac{\partial \mathcal{L}_{\text{MTP}}}{\partial \theta_{\text{trunk}}} = \frac{\partial \mathcal{L}_{\text{MTP}}}{\partial h} \cdot \frac{\partial h}{\partial \theta_{\text{trunk}}}
$$

MTP loss backprops through:
1. Shared `output_head` (tied with embed)
2. `MTPBlock` weights (independent)
3. Trunk hidden $h$ via `forward_with_hidden`

**Effect:** Trunk representations must encode enough information to predict $t+2$ given $t+1$ — denser training signal than CE alone.

### Shared head gradient

Main and MTP losses both differentiate through `head.weight`:

$$
\nabla_{W_{\text{head}}} = \nabla \mathcal{L}_{\text{main}} + \lambda \nabla \mathcal{L}_{\text{MTP}}
$$

No gradient conflict resolution — simple sum. $\lambda = 0.3$ balances magnitudes empirically.

---

## Alignment Mathematics (depth=1)

For sequence length $S$, MTP depth $d=1$ loses $d+1 = 2$ positions at the tail:

```
tokens:  [x0, x1, x2, ..., x_{S-1}]
main:    predicts x1..x_S     (S targets)
MTP:     uses h[0:S-2], embed(x1:S-1), predicts x2..x_S  (S-2 targets)
```

**Usable length:** $S - (d+1) = S - 2$ for depth 1.

Implementation in `MultiTokenPrediction.forward`:

```python
usable = seqlen - (depth + 1)
h_in = hidden[:, :usable]
emb = embed(tokens[:, d+1 : d+1+usable])
tgt = tokens[:, d+2 : d+2+usable]
```

---

## Speculative Decoding Connection

At inference, MTP head becomes a **draft model** without separate training:

| Training | Inference |
|---|---|
| Predict $x_{t+2}$ from $(h_t, e_{t+1})$ | Draft next token after main generates $t_1$ |
| CE loss on draft | Accept/reject vs main distribution |
| Shared head weights | Same `output_head` |

**Acceptance rate** depends on:
1. MTP training quality (λ, depth)
2. `acceptance_threshold` (default 0.8)
3. Distribution shift between greedy decode and training teacher forcing

---

## Parameter Count (422M, depth=1)

Per `MTPModule`:
- `MTPBlock`: 2× RMSNorm + fusion Linear + MHA + SwiGLU
- Roughly ~3M params at dim=768, inter_dim=1536, 12 heads

Negligible vs 298M MoE params but adds ~5% forward compute during training.

---



## MTPBlock Internals — Layer-by-Layer

`MTPBlock` (`models/mtp.py`) mirrors a slim Transformer block:

```
x = RMSNorm(hidden)
y = RMSNorm(embed(next_token))
fused = Linear(concat(x, y))          # fusion projection
h = fused + MHA(fused)                # nn.MultiheadAttention (NOT MLA)
h = h + SwiGLUFFN(h)
return h
```

**Why MHA instead of MLA for MTP?** Draft head runs on **single-token** slices during speculative decode — KV cache complexity of MLA buys nothing; standard MHA is simpler and sufficient.

---

## `compute_loss` — Alignment Detail

For depth $d=1$, usable length $U = S - (d+1) = S - 2$:

| Tensor | Slice | Role |
|---|---|---|
| `hidden[:, :U]` | trunk states | Predict $t+2$ from position $t$ |
| `tokens[:, d+1:d+1+U]` | $t+1$ tokens | Embedded as conditioning |
| `targets[:, d+2:d+2+U]` | $t+2$ tokens | CE labels |

Loss:

$$
\mathcal{L}_{total} = \mathcal{L}_{CE}^{main} + \lambda \mathcal{L}_{CE}^{mtp}
$$

with $\lambda =$ `mtp_loss_weight` (0.3 at 422M).

---

## Speculative Decoding — Acceptance Math (Simplified)

Let $q(x)$ be draft distribution, $p(x)$ main distribution. This repo accepts draft token $x$ when:

$$
p(x) \geq 	au \cdot q(x), \quad 	au = 0.8
$$

**Not exact sampling** from the target distribution (unlike full speculative sampling with rejection correction) — but empirically fast when MTP is co-trained.

**When acceptance fails:** only `token_main` is emitted; draft is discarded; next iteration recomputes from new last token.

---

## Depth > 1 Extension Sketch

`MultiTokenPrediction` uses `ModuleList` of `MTPModule` length `mtp_depth`. Each depth adds another $(h, e_{t+d})$ fusion predicting $t+d+1$. This repo ships `mtp_depth=1` only; extending requires:

1. YAML `mtp_depth: K`
2. Loop depths in `forward` with shifting embed slices
3. Sum MTP losses with optional per-depth weights
4. Speculative decode chain for multi-token drafts

Untested beyond depth=1 in this codebase.

---

## MTP vs Separate Draft Model

| Approach | Pros | Cons |
|---|---|---|
| MTP (this repo) | Shared trunk/head; no extra training run | Draft quality tied to λ |
| Separate small LM | Independent draft tuning | 2× training cost, weight sync |
| EAGLE / Medusa | Multi-draft trees | More complex acceptance |

---

## Training Ablations (Suggested Experiments)

1. `mtp_loss_weight ∈ {0, 0.1, 0.3, 0.5}` — measure speculative acceptance after 10k steps
2. `mtp_depth=0` — disable wrapper entirely for CE-only baseline
3. Freeze trunk, train MTP only — isolates auxiliary head capacity

Document results in your experiment log; no automated benchmark ships with this repo.



## Load-Bearing Invariants

1. **Shared output head** — `mtp.set_output_head(main_model.head)`. Never create a separate LM head for MTP.
2. **Shared embedding** — `self.add_module("embed", main_model.embed)`.
3. **`forward_with_hidden` returns pre-norm `h`** — MTP applies its own `RMSNorm` before the shared head.
4. **`use_cache=False` during training** — MTP training path does not use MLA KV cache.
5. **MTP checkpoint prefix** — `mtp.` keys in safetensors; strip on load.

---

## Implementation Checklist

- [ ] `pytest tests/test_models.py -k MTP` passes
- [ ] `pytest tests/test_inference.py -k Speculative` passes
- [ ] `test_shared_head_mtp` — head storage shared
- [ ] `test_forward_short_sequence` — empty pairs handled
- [ ] `test_save_load_with_mtp` — checkpoint roundtrip

---

## References

- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — MTP module design
- [Speculative Decoding (Leviathan et al.)](https://arxiv.org/abs/2211.17192)
- `models/mtp.py`, `inference/speculative.py` — authoritative implementation

## MTPBlock — Internal Walkthrough

`models/mtp.py:MTPBlock.forward`:

```python
fused = proj(cat([norm_h(prev_hidden), norm_e(target_emb)], dim=-1))
attn_out = MHA(norm_attn(fused), causal_mask)
fused = fused + attn_out
return fused + SwiGLU(norm_ffn(fused))
```

**Design choice:** MTP uses standard `nn.MultiheadAttention` (not MLA) — draft head is shallow; MLA compression is unnecessary for length-1 windows at inference.

### Causal mask in MTPBlock

`_get_causal_mask(seqlen)` builds upper-triangular $-\infty$ mask. At training, `usable` positions may be $< S$; mask size matches fused sequence length.

---

## `MultiTokenPrediction.forward` — Alignment Table

For `depth=1`, `seq_len=S`:

| Tensor | Slice | Shape |
|---|---|---|
| `main_logits` | full | $(B, S, V)$ |
| `h_in` | `[:, :S-2]` | $(B, S-2, d)$ |
| `emb_in` | `tokens[:, 2:S]` | $(B, S-2, d)$ |
| `tgt` | `tokens[:, 3:S+1]` | $(B, S-2,)$ |

**Lost positions:** Last 2 tokens have no MTP target at depth 1 (need $t$, $t+1$, $t+2$).

---

## `compute_loss` — Three Return Values

```python
total_loss, main_loss, mtp_loss = mtp_wrapper.compute_loss(main_logits, targets, mtp_pairs)
```

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{main}} + \lambda \cdot \text{mean}(\mathcal{L}_{\text{MTP}}^{(d)})
$$

Only `total_loss / grad_accum` is backpropped. `main_loss` and `mtp_loss` are detached for logging.

---

## Speculative Decoding Connection (Inference)

| Training | Inference |
|---|---|
| Predict $x_{t+2}$ from $(h_t, e_{t+1})$ | Draft after main commits $t_1$ |
| CE loss on draft | Accept/reject vs main distribution |
| Shared `output_head` | Same `head.weight` pointer |
| Teacher forcing on $e_{t+1}$ | Greedy $e_{t_1}$ from main argmax |

**Acceptance rate** correlates with MTP loss during training — monitor `mtp_loss` in training logs.

---

## Depth > 1 (Not Enabled at 422M)

`ModuleList([MTPModule(config, d+1) for d in range(depth)])` supports stacking. Depth 2 would predict $t+3$ from depth-1 hidden. DeepSeek-V3 paper uses multiple depths; this repo sets `mtp_depth: 1` for simplicity.

<!-- docs:verified 2026-07-31 · 5a880d2 -->
