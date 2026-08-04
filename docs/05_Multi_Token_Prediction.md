# Multi-Token Prediction (MTP) + Speculative Decoding

> **Prerequisites:** [[Docs/02_Model_Architecture|Model Architecture]], [[Docs/03_Multi_Head_Latent_Attention|MLA]].

> **Read this if** you're working on MTP loss alignment or speculative decoding. **Skip if** you only need the standard train loop → [[Docs/08_Training_Pipeline|Training]].

**Depends on:** [[Docs/02_Model_Architecture|Model Architecture]], [[Docs/03_Multi_Head_Latent_Attention|MLA]] · **Read next:** [[Docs/10_Inference_and_Serving|Inference]]

---

## Table of Contents

1. [Abstract](#abstract)
2. [Motivation — Denser Training Signal](#motivation--denser-training-signal)
3. [DeepSeek-V3 MTP in the Paper](#deepseek-v3-mtp-in-the-paper)
4. [Mathematical Formulation](#mathematical-formulation)
5. [MTPBlock — Internal Architecture](#mtpblock--internal-architecture)
6. [MTPModule — Single Depth Head](#mtpmodule--single-depth-head)
7. [MultiTokenPrediction — Training Wrapper](#multitokenprediction--training-wrapper)
8. [The MTP Loss — Derivation and Semantics](#the-mtp-loss--derivation-and-semantics)
9. [Length-Alignment Algebra — Deriving `usable`](#length-alignment-algebra--deriving-usable)
10. [Shared-Head Mechanics — `set_output_head`](#shared-head-mechanics--set_output_head)
11. [Speculative Decoding Theory](#speculative-decoding-theory)
12. [The Repo's Decoder — Code Walkthrough](#the-repos-decoder--code-walkthrough)
13. [Cache-Consistency Walkthrough](#cache-consistency-walkthrough)
14. [Temperature Handling](#temperature-handling)
15. [KV Cache Sharing](#kv-cache-sharing)
16. [Checkpoint Format](#checkpoint-format)
17. [Comparison: MTP vs Standard Decode](#comparison-mtp-vs-standard-decode)
18. [Historical Context — Multi-Token Prediction](#historical-context--multi-token-prediction)
19. [Gradient Flow Analysis](#gradient-flow-analysis)
20. [Depth > 1 Extension Sketch](#depth--1-extension-sketch)
21. [MTP vs Separate Draft Model](#mtp-vs-separate-draft-model)
22. [Training Ablations (Suggested Experiments)](#training-ablations-suggested-experiments)
23. [Appendix A — Worked Alignment Example](#appendix-a--worked-alignment-example)
24. [Appendix B — Gradient Flow](#appendix-b--gradient-flow)
25. [Appendix C — FAQ](#appendix-c--faq)
26. [Appendix D — Glossary](#appendix-d--glossary)
27. [Load-Bearing Invariants](#load-bearing-invariants)
28. [Implementation Checklist](#implementation-checklist)
29. [References](#references)

---

## Abstract

**Multi-Token Prediction (MTP)** trains auxiliary heads to predict tokens *beyond* the immediate next-token objective. In DeepSeek-V3, MTP modules predict `t+2`, `t+3`, … in parallel with the main head's `t+1` prediction. This densifies the training signal and, critically, enables **speculative decoding** at inference: a lightweight draft head proposes a future token that the main model verifies in one forward pass.

This repo implements `mtp_depth=1` (one auxiliary head predicting `t+2`) with `mtp_loss_weight=0.3`. The MTP block uses standard `nn.MultiheadAttention` (not MLA) and shares the main model's embedding and LM head. The MTP module adds **7,081,728 parameters** — exactly the difference between the 418.7M with-MTP and 411.6M base deduped counts — and roughly 5% extra FLOPs per training step (an estimate; no GPU run has measured it).

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

**Cost:** Extra parameters (~7.1M for depth=1 at the canonical config — 1.7% of the 418.7M total) and roughly 5% extra compute per training step (estimate — the MTP block is about one extra transformer block on a slightly shorter sequence, against 18 trunk layers; no GPU run has measured the real overhead).

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

### MTP depth D (D = 1, 2, …)

```
h_t^{(D)} = MTPBlock_D( h_t^{(D-1)}, e_{t+D} )     where h_t^{(0)} = h_t
logits_t^{(D)} = W_head · RMSNorm(h_t^{(D)})       predicts x_{t+D+1}
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

Section [8](#the-mtp-loss--derivation-and-semantics) derives this loss in full, including the exact slicing that makes every depth's targets length-aligned.

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

The forward is short enough to quote in full (`models/mtp.py:MTPBlock.forward`):

```python
def forward(self, prev_hidden: torch.Tensor, target_emb: torch.Tensor) -> torch.Tensor:
    fused = self.proj(torch.cat([self.norm_h(prev_hidden), self.norm_e(target_emb)], dim=-1))
    seqlen = fused.size(1)
    attn_in = self.norm_attn(fused)
    attn_out, _ = self.attn(attn_in, attn_in, attn_in, attn_mask=self._get_causal_mask(seqlen, fused.device), is_causal=False)
    fused = fused + attn_out
    ffn_in = self.norm_ffn(fused)
    return fused + self.w2(F.silu(self.w1(ffn_in)) * self.w3(ffn_in))
```

**Why independent norms on hidden and embedding?** The trunk hidden and token embedding live in different semantic spaces (post-17-layers vs lookup table). Separate RMSNorm before fusion prevents scale mismatch.

**Why `nn.MultiheadAttention` instead of MLA?** MTP operates on short aligned windows during training (not full-context decode with KV cache). Standard SDPA is simpler and sufficient. MTP has its own `_causal_mask` buffer — it does **not** share MLA's KV cache.

**Causal mask.** `models/mtp.py:MTPBlock._get_causal_mask` builds an upper-triangular `-inf` mask once and caches it, growing the buffer when a longer sequence arrives:

```python
def _get_causal_mask(self, seqlen: int, device: torch.device) -> torch.Tensor:
    if seqlen > self._causal_mask_size or self._causal_mask.device != device:
        mask = torch.triu(torch.full((seqlen, seqlen), float("-inf"), device=device), diagonal=1)
        self._causal_mask = mask
        self._causal_mask_size = seqlen
    return self._causal_mask[:seqlen, :seqlen]
```

The mask is passed with `is_causal=False` because the additive mask already encodes causality. Note the buffer is registered `persistent=False`, so it never appears in checkpoints.

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

The forward contract (`models/mtp.py:MTPModule.forward`):

```python
def forward(self, prev_hidden: torch.Tensor, target_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if self.output_head is None:
        raise RuntimeError(f"MTPModule(depth={self.depth}): output_head not set.")
    if prev_hidden.shape != target_emb.shape:
        raise ValueError(f"Shape mismatch: {prev_hidden.shape} vs {target_emb.shape}")
    h = self.block(prev_hidden, target_emb)
    h_norm = self.norm(h)
    return self.output_head(h_norm), h_norm
```

The shape check is load-bearing: the training wrapper must hand this module two tensors of identical length, which is exactly what the alignment algebra in section [9](#length-alignment-algebra--deriving-usable) guarantees.

---

## MultiTokenPrediction — Training Wrapper

```python
class MultiTokenPrediction(nn.Module):
    def forward(tokens) -> (main_logits, mtp_pairs)
```

**Step-by-step** (`models/mtp.py:MultiTokenPrediction.forward`):

1. `main_logits, prev_h = main_model.forward_with_hidden(tokens)`
   - `prev_h` is the **pre-norm** trunk hidden `(B, S, D)`.
   - `main_logits` uses `head(norm(h))`.

2. For each depth `d` (0-indexed, depth value = d+1):
   ```python
   usable = seq_len - d - 2
   h_in  = prev_h[:, :usable]
   emb   = embed(tokens[:, d + 1 : d + 1 + usable])
   tgt   = tokens[:, d + 2 : d + 2 + usable]
   logits, hidden = mtp_modules[d](h_in, emb)
   mtp_pairs.append((logits, tgt))
   prev_h = hidden   # chain for depth > 1
   ```

3. If `usable <= 0` (sequence too short), skip that depth.

**Embedding sharing:** `self.add_module("embed", main_model.embed)` — same `nn.Embedding` tensor, counted once in optimiser dedup.

The wrapper calls `forward_with_hidden` with its default `use_cache=False` — the training path never touches the MLA KV cache (see [Load-Bearing Invariants](#load-bearing-invariants)).

---

## The MTP Loss — Derivation and Semantics

### 8.1 Setup and notation

Let $S$ be the sequence length, $V$ the vocabulary size, $D$ the hidden dimension, and $B$ the batch size. Write $x_t$ for the token at position $t$ ($0 \le t < S$), $h_t$ for the pre-norm trunk hidden state at $t$ (the output of `models/transformer.py:Transformer.forward_with_hidden`), and $e_t = \mathrm{Embed}(x_t) \in \mathbb{R}^D$.

The main head produces, at every position,

$$
\text{logits}_t^{(0)} = W_{\text{head}}\, \mathrm{RMSNorm}(h_t),
$$

which is trained to predict $x_{t+1}$ — the standard next-token objective. The dataset supplies the shifted targets; `ignore_index=-100` in the cross-entropy lets the trainer mask padding or truncated positions.

### 8.2 Depth-$D$ targets

For depth $D$ (1-indexed; the code's 0-indexed `d` equals $D-1$), the module consumes the previous depth's hidden state and the embedding of the token $D$ steps ahead of the current position:

$$
h_t^{(D)} = \mathrm{MTPBlock}_D\!\left(h_t^{(D-1)},\, e_{t+D}\right), \qquad
\text{logits}_t^{(D)} = W_{\text{head}}\, \mathrm{RMSNorm}\!\left(h_t^{(D)}\right),
$$

trained to predict $x_{t+D+1}$. The conditioning token $x_{t+D}$ is the *ground-truth* token during training (teacher forcing) — the same trick that makes the draft head useful at inference, where the conditioning token is the main model's own output.

**Which positions are valid?** Position $t$ has a depth-$D$ target iff $t + D + 1 \le S - 1$, i.e. $t \le S - D - 2$. The number of valid positions is

$$
S - D - 1 = S - (d+1) - 1 = S - d - 2 = \texttt{usable}.
$$

So depth $D$ "loses" $D+1$ positions at the tail of the sequence: the last $D+1$ tokens can never serve as the *input* hidden of a depth-$D$ prediction because there is no token after them to predict. Section [9](#length-alignment-algebra--deriving-usable) works through the slicing in detail.

### 8.3 The loss

Let $\mathrm{CE}(\text{logits}, \text{targets})$ denote the token-wise cross-entropy averaged over all valid (non-ignored) positions. The three losses are:

$$
\mathcal{L}_{\text{main}} = \mathrm{CE}\!\left(\text{logits}^{(0)},\, x_{1:}\right),
\qquad
\mathcal{L}_d = \mathrm{CE}\!\left(\text{logits}^{(d)},\, x_{d+2:}\right),
$$

where the depth-$d$ targets are the length-`usable` slice $x_{d+2}, \dots, x_{S-1}$. The MTP loss is the **mean across depths**,

$$
\mathcal{L}_{\text{mtp}} = \frac{1}{D_{\max}} \sum_{d=0}^{D_{\max}-1} \mathcal{L}_d,
$$

and the total is

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{main}} + \lambda\, \mathcal{L}_{\text{mtp}}, \qquad \lambda = \texttt{mtp\_loss\_weight} = 0.3.
$$

With `mtp_depth=1` the mean is trivial ($\mathcal{L}_{\text{mtp}} = \mathcal{L}_0$), but the code path is written for the general case.

### 8.4 The code

`models/mtp.py:MultiTokenPrediction.compute_loss` implements exactly this:

```python
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
```

Three details worth noting:

- **`ignore_index=-100` everywhere.** The main loss and every depth loss use the same masking convention, so packed sequences with padding behave identically in both paths.
- **Empty-pairs case.** When `usable <= 0` for every depth (e.g. `seq_len <= 2` at depth 1), `mtp_pairs` is empty and the function returns `(main_loss, main_loss, zeros)`. The MTP slot is a zero tensor, not `main_loss` — a subtle contract that `tests/test_models.py` pins down (`test_compute_loss_no_mtp_pairs`).
- **Mean across depths, then weighted.** The `torch.stack(depth_losses).mean()` is the $\frac{1}{D_{\max}}\sum_d$ in the derivation; the single `self.mtp_weight` scales the whole mean, not each depth separately.

### 8.5 How the trainer consumes it

In `training/pretrain.py:Pretrainer.train_step`, the MTP branch is:

```python
if self.mtp_wrapper is not None:
    main_logits, mtp_pairs = self.model(tokens)
    total_loss, main_loss, mtp_loss = self.mtp_wrapper.compute_loss(main_logits, targets, mtp_pairs)
    # Defer host round-trip to the logger (avoids per-step GPU sync).
    _ce_loss_val = main_loss.detach()
    _mtp_loss_val = mtp_loss.detach() if mtp_pairs else None
    loss = total_loss / self.config.gradient_accumulation_steps
```

Only the **total** loss is divided by `gradient_accumulation_steps`; main and MTP components are not separately scaled. The NaN guard, gradient clipping, and optimizer step all operate on this single scaled total, so MTP parameters are clipped and updated exactly like trunk parameters. `main_loss` and `mtp_loss` are detached for logging only — the logger never forces a GPU→host sync inside the hot loop.

---

## Length-Alignment Algebra — Deriving `usable`

### 9.1 Why alignment is non-trivial

The MTP block at depth $D$ consumes *two* inputs of the same length — the hidden states and the conditioning embeddings — and produces logits that must be compared against targets of that same length. The three sequences are shifted relative to each other by construction:

| Tensor | What it contains | Positions used |
|---|---|---|
| `h_in` | trunk hidden states | $0 \dots \texttt{usable}-1$ |
| `emb_in` | embeddings of conditioning tokens | $D \dots S-2$ |
| `tgt` | prediction targets | $D+1 \dots S-1$ |

All three must have exactly `usable` elements, and the shifts must line up so that row $t$ of `h_in` + row $t$ of `emb_in` predicts row $t$ of `tgt`.

### 9.2 The derivation

Fix 0-indexed depth `d` (depth value $D = d+1$). The block at position $t$ predicts $x_{t+D+1} = x_{t+d+2}$, conditioned on $h_t$ and $e_{t+D} = e_{t+d+1}$.

**Targets.** The last valid target position is $S-1$ (the final token of the sequence), and the first is $d+2$ (position $t=0$ predicts $x_{d+2}$). So

$$
\texttt{tgt} = x_{d+2}\, \dots\, x_{S-1}, \qquad \text{length} = (S-1) - (d+2) + 1 = S - d - 2.
$$

**Conditioning embeddings.** Position $t$ conditions on $x_{t+d+1}$; for $t = 0 \dots \texttt{usable}-1$ that is $x_{d+1} \dots x_{d+1+\texttt{usable}-1}$. Since $d+1+\texttt{usable} = d+1 + S - d - 2 = S-1$, the slice is

$$
\texttt{emb\_in} = x_{d+1}\, \dots\, x_{S-2}, \qquad \text{length} = (S-2) - (d+1) + 1 = S - d - 2 = \texttt{usable}.
$$

**Hidden states.** The block needs one hidden per target, starting at $t=0$:

$$
\texttt{h\_in} = h_0\, \dots\, h_{\texttt{usable}-1}, \qquad \text{length} = \texttt{usable}.
$$

All three lengths agree, which is why the code can write the slices as

```python
usable = seq_len - d - 2
h_in  = prev_h[:, :usable]
emb   = embed(tokens[:, d + 1 : d + 1 + usable])
tgt   = tokens[:, d + 2 : d + 2 + usable]
```

**Sanity check at the last valid position.** Take $t = \texttt{usable}-1 = S-d-3$. Then $h_t$ is the last row of `h_in`, the conditioning token is $x_{t+d+1} = x_{S-2}$ (the last row of `emb_in`), and the predicted token is $x_{t+d+2} = x_{S-1}$ (the last row of `tgt`). Everything closes exactly at the sequence end.

**What is lost.** The first $d+1$ tokens ($x_0 \dots x_{d+1}$) never appear as targets of depth $d$ — they would need conditioning tokens before position 0. The last $d+1$ tokens ($x_{S-d-2} \dots x_{S-1}$) never appear as `h_in` — they have no token after them to predict. Depth $D$ therefore contributes $S-D-1$ supervised positions out of $S$.

### 9.3 Worked example, depth 1

`seq_len=6`, tokens `[a, b, c, d, e, f]`, `d=0`:

```
usable = 6 - 0 - 2 = 4
h_in  = hidden[:, 0:4]      positions 0,1,2,3
emb   = embed([b,c,d,e])    tokens at t+1
tgt   = [c,d,e,f]           tokens at t+2

h[0] + emb(b) → predict c
h[1] + emb(c) → predict d
h[2] + emb(d) → predict e
h[3] + emb(e) → predict f
```

### 9.4 Worked example, depth 2 (if enabled)

`seq_len=6`, `d=1`:

```
usable = 6 - 1 - 2 = 3
h_in  = hidden[:, 0:3]      positions 0,1,2   (depth-1 output hidden)
emb   = embed([c,d,e])      tokens at t+2
tgt   = [d,e,f]             tokens at t+3

h[0] + emb(c) → predict d
h[1] + emb(d) → predict e
h[2] + emb(e) → predict f
```

Note the chaining: `prev_h = hidden` after depth 1, so depth 2's `h_in` is the depth-1 *output* hidden at the same positions — the depth-1 output at position $t$ already encodes $x_{t+1}$ (it was conditioned on $e_{t+1}$), which is exactly the information depth 2 needs to predict $x_{t+3}$.

### 9.5 Edge cases

- **`usable <= 0`** ⟺ `seq_len <= d + 2`. For depth 1 that means sequences of length ≤ 2 produce no MTP pairs at all; `MultiTokenPrediction.forward` breaks out of the loop and `compute_loss` returns the zero-tensor MTP slot. `tests/test_models.py:TestMultiTokenPrediction.test_forward_short_sequence` guards this.
- **`tgt.numel() == 0`** is checked again inside `compute_loss` — defensive, since the forward already skips empty windows.
- The `prev_hidden.shape != target_emb.shape` check in `models/mtp.py:MTPModule.forward` is the runtime backstop: if the alignment algebra ever produced mismatched lengths, the error surfaces as a clear `ValueError` instead of a silent broadcast.

---

## Shared-Head Mechanics — `set_output_head`

### 10.1 Why share the head at all

The LM head is a `Linear(D, V)` with $768 \times 100\,018 \approx 76.8$M parameters — nearly a fifth of the whole model. A separate head per MTP depth would add ~76.8M parameters per depth and, worse, would let the auxiliary losses drift the head away from the main objective. DeepSeek-V3's design shares one head across the main model and every depth, so the MTP loss and the main loss *compete and cooperate* on the same projection.

### 10.2 The mechanism

`models/mtp.py:MTPModule.set_output_head` is a plain setter:

```python
def set_output_head(self, head: nn.Linear) -> None:
    self.output_head = head
```

`MultiTokenPrediction.__init__` wires every depth to the main model's head:

```python
shared_head = main_model.head
for mtp in self.mtp_modules:
    mtp.set_output_head(shared_head)
```

Because `nn.Module.__setattr__` registers `Module` values as submodules, `output_head` becomes part of `MTPModule`'s `state_dict` (keys `output_head.weight`) — but it is the *same tensor object* as `main_model.head.weight`, which under weight tying is itself `embed.weight` (`models/transformer.py:Transformer.__init__` sets `self.head.weight = self.embed.weight` when `weight_tying: true`).

### 10.3 Parameter accounting

`models/transformer.py:count_parameters` deduplicates by tensor id:

```python
for p in model.parameters():
    pid = id(p)
    if pid in seen:
        continue
    seen.add(pid)
    ...
```

So the shared head and shared embedding are counted **once** across the whole wrapper. The MTP wrapper's *added* parameters are exactly one `MTPBlock` + one RMSNorm per depth: 7,081,728 for depth 1, matching the measured 418,713,984 − 411,632,256 = 7,081,728 delta. The optimizer likewise sees each shared tensor once, so there is no double-update of the head.

### 10.4 Gradient coupling

Because the head is shared, the head's gradient is the sum of both paths:

$$
\nabla_{W_{\text{head}}} \mathcal{L}_{\text{total}} = \nabla_{W_{\text{head}}} \mathcal{L}_{\text{main}} + \lambda \sum_d \nabla_{W_{\text{head}}} \mathcal{L}_d .
$$

There is no gradient-conflict resolution — a plain sum, with $\lambda = 0.3$ chosen to keep the auxiliary contribution from overwhelming the main one. The same coupling flows into the trunk: every position $t$ receives gradient from the main loss (as predictor of $x_{t+1}$) *and* from the depth-1 loss (as predictor of $x_{t+2}$), which is the "denser training signal" of section [2](#motivation--denser-training-signal). `tests/test_models.py:TestMTPGradientFlow.test_mtp_loss_flows_into_main_model` verifies that `embed.weight` and an MLA attention weight receive gradients after a backward through the MTP loss.

### 10.5 Pitfalls

- **Forward before wiring.** `MTPModule.forward` raises `RuntimeError("MTPModule(depth=...): output_head not set.")` if `set_output_head` was never called. The training wrapper always wires it; standalone construction (as in `inference/generate.py`) does **not** — see the pitfall in section [12](#the-repos-decoder--code-walkthrough).
- **`strict=False` on load.** The checkpoint stores `output_head.weight` (it was wired during training), but a standalone `MTPModule` with `output_head=None` has no such key; `load_state_dict(..., strict=False)` silently ignores the mismatch. The shared head is restored through the main model's own `head.weight`/`embed.weight`, so nothing is lost — but the silent ignore is easy to misread as "the head loaded".
- **Don't create a second head.** The invariant is: never construct a separate `nn.Linear` for MTP. If you do, `count_parameters` will count it, the optimizer will train it, and the draft distribution will drift from the main model's — silently breaking the acceptance test's premise.

---

## Speculative Decoding Theory

### 11.1 60-second summary

Speculative decoding lets a cheap **draft** model propose tokens while the expensive **target** model verifies them in a single forward pass. When the draft is right, you emit several tokens per target forward; when it is wrong, you fall back to the target's own token. The MTP head trained in sections [8](#the-mtp-loss--derivation-and-semantics)–[10](#shared-head-mechanics--set_output_head) is exactly such a draft model — it predicts $x_{t+2}$ from the trunk hidden state, so at inference it can propose the token *after* the one the main model just committed.

### 11.2 Why it exists

Autoregressive decoding is sequential: one token per forward pass, and each forward is latency-bound (a length-1 decode step moves a tiny amount of data through a huge model). The wall-clock cost of generating $N$ tokens is $N$ sequential steps. Speculative decoding breaks the "one forward = one token" coupling: the draft proposes $K$ tokens cheaply, the target scores all $K+1$ candidates in one batched forward, and you emit the longest verified prefix. The target's distribution is what you sample from — the draft only decides *where* to look.

### 11.3 The exact acceptance scheme (Leviathan et al. 2023)

Let $q$ be the draft distribution and $p$ the target distribution over the vocabulary. The exact scheme is:

1. Sample $x \sim q$.
2. Accept $x$ with probability $\min\!\left(1,\, \frac{p(x)}{q(x)}\right)$.
3. If rejected, resample from the *corrected* distribution $p_{\text{res}}(x) \propto \max(0,\, p(x) - q(x))$ and emit that instead.

**Why this preserves the target distribution.** Consider the probability that the emitted token is $x$:

$$
P(\text{emit } x) = q(x) \cdot \min\!\left(1, \frac{p(x)}{q(x)}\right) + P(\text{reject}) \cdot p_{\text{res}}(x).
$$

The rejection probability is

$$
P(\text{reject}) = \sum_y q(y)\left(1 - \min\!\left(1, \frac{p(y)}{q(y)}\right)\right) = \sum_y \max(0,\, q(y) - p(y)) =: R.
$$

If $p(x) \le q(x)$: the first term is $q(x) \cdot \frac{p(x)}{q(x)} = p(x)$, and $p_{\text{res}}(x) = 0$ (since $p(x) - q(x) \le 0$), so $P(\text{emit } x) = p(x)$. If $p(x) > q(x)$: the first term is $q(x)$, and $p_{\text{res}}(x) = \frac{p(x) - q(x)}{R'}$ where $R' = \sum_z \max(0, p(z) - q(z))$. A short identity shows $R = R'$ (both equal the total mass where one distribution exceeds the other), so

$$
P(\text{emit } x) = q(x) + R \cdot \frac{p(x) - q(x)}{R} = p(x).
$$

Either way, the emitted token is distributed exactly as $p$. The scheme is **lossless**: the output distribution is indistinguishable from sampling the target model directly.

**Expected tokens per step.** With a 1-token draft, the expected number of emitted tokens per step is

$$
1 + \mathbb{E}[\text{accept}] = 1 + \sum_x q(x) \min\!\left(1, \frac{p(x)}{q(x)}\right) = 1 + \sum_x \min(p(x), q(x)) =: 1 + \alpha,
$$

where $\alpha \in [0, 1]$ is the *acceptance rate* — the total probability mass the two distributions agree on. $\alpha = 1$ iff $p = q$; $\alpha$ shrinks as the draft diverges from the target. For a $K$-token draft, assuming i.i.d. acceptance, the expected tokens per verification pass is

$$
\sum_{i=1}^{K+1} \alpha^{i-1} = \frac{1 - \alpha^{K+1}}{1 - \alpha},
$$

which is why deeper drafts (larger $K$) pay off when $\alpha$ is high: at $\alpha = 0.8$, $K=1$ gives 1.8 tokens/step, $K=4$ gives $\frac{1-0.8^5}{0.2} \approx 3.36$.

### 11.4 The repo's variant: greedy verification

`inference/speculative.py:SpeculativeDecoder.generate_step` implements a **deliberate simplification** of the exact scheme. The draft is not sampled — it is the greedy argmax of the draft distribution — and acceptance is a deterministic threshold test:

```python
# Acceptance compares raw (unscaled) probabilities of the draft token.
p_main_of_draft = main_probs[0, token_draft[0]].item()
p_draft_of_draft = draft_probs[0, token_draft[0]].item()
return token_main, token_draft, p_main_of_draft >= self.threshold * max(p_draft_of_draft, 1e-12)
```

Let $x^* = \arg\max_x q(x)$ be the greedy draft token. The rule accepts iff

$$
p(x^*) \ge \tau \cdot q(x^*), \qquad \tau = \texttt{acceptance\_threshold} = 0.8.
$$

**How this differs from the exact scheme — documented honestly:**

1. **No rejection resampling.** The exact scheme resamples from $p_{\text{res}} \propto \max(0, p - q)$ on rejection, which is what makes the output distribution *exactly* $p$. This repo emits only `token_main` on rejection and moves on. The output distribution is therefore **not** guaranteed to match the target model's — the rule is a throughput heuristic for greedy-ish decoding, not a distribution-preserving sampler.
2. **Deterministic, not probabilistic.** The exact scheme accepts with probability $\min(1, p/q)$; this repo accepts iff the ratio clears a fixed bar. A draft token with $p(x^*) = 0.9\,q(x^*)$ is accepted with probability 1 here, but with probability 0.9 in the exact scheme.
3. **Even at temperature 0, the output can deviate from pure greedy decoding.** `token_main` is the main model's argmax, but an accepted `token_draft` only needs $p(x^*) \ge 0.8\,q(x^*)$ — it can differ from $\arg\max p$. The emitted sequence is "greedy with draft-flavoured detours", not the main model's greedy continuation.
4. **The draft is always greedy.** The draft proposes its single most-likely continuation, and the verifier checks that one token. There is no sampling from $q$, so the expected-tokens-per-step formula becomes a Bernoulli: $1 + \mathbf{1}[p(x^*) \ge \tau q(x^*)]$.

**Why this is a reasonable choice for this repo.** The exact scheme requires matching draft and target distributions and a resampling step; the greedy verifier is a few lines, deterministic, and testable. It is the right call for a pedagogical codebase whose draft head is a single MTP block — but it must not be sold as lossless speculative sampling. Use it where exact distribution matching doesn't matter (chat, code completion); don't claim it as lossless.

### 11.5 Honest compute accounting

Each `generate_step` runs the main model **twice** — once on the last token (to produce `token_main`) and once on `token_main` itself (to produce the hidden state the MTP block conditions on) — plus one MTP-block pass. Let $\hat\alpha$ be the empirical acceptance rate. Then:

$$
\text{main-model forwards per emitted token} = \frac{2}{1 + \hat\alpha}.
$$

This is **≥ 1.0**, equal at $\hat\alpha = 1$ and strictly worse below that. Standard decode is exactly 1.0. So this implementation never reduces main-model forward count — the earlier claim of "~0.55 forward passes per token" in this doc's history was an arithmetic error (it divided 1 by 1.8, counting one forward per step when a step is two forwards; the correct figure at $\hat\alpha = 0.8$ is $2/1.8 \approx 1.11$).

The paper's speedup comes from a structural difference this repo does not implement: there, the draft hidden states are produced *inside* the single verification forward (the target scores all draft positions at once), and the MTP head is a ~1-block add-on rather than a full trunk re-run. Here, `forward_with_hidden` re-runs all 18 layers on `token_main` to obtain its hidden state — the same compute a standard decode step would spend anyway, but not amortized.

What the repo's loop *does* buy, at high acceptance: the number of sequential step boundaries halves (two tokens per `generate()` iteration instead of one), which removes one round of per-step fixed overhead (Python loop, sampling call, kernel launches) per accepted pair. Whether that wins wall-clock depends on how large that fixed overhead is relative to a decode forward — a crossover that must be measured per harness and per corpus. `.benchmarks/` is empty and no trained checkpoint exists, so **no measured speedup number is available**; treat any throughput claim as an estimate.

**Measured acceptance caveat.** "~0.8 acceptance → ~1.8× throughput" came from **smoke tests, not a trained model** — this repo has no completed checkpoint, and with random MTP weights the draft head is near-useless (acceptance ≈ 0). Expect meaningful 1.5–2× speedups only after the MTP head is actually trained; even then, acceptance is prompt-dependent and must be measured per corpus.

### 11.6 A worked acceptance example

Concrete numbers make the difference between the two schemes visible. Take a 3-token vocabulary $\{A, B, C\}$ with draft distribution $q = (0.7, 0.2, 0.1)$ and target distribution $p = (0.5, 0.4, 0.1)$.

**Exact scheme.** The greedy draft is $x^* = A$. Acceptance is probabilistic: accept $A$ with probability $\min(1, 0.5/0.7) \approx 0.714$; $B$ and $C$ would be accepted with probability 1 if sampled. The expected number of tokens per step is

$$
1 + \alpha = 1 + \sum_x \min(p(x), q(x)) = 1 + (0.5 + 0.2 + 0.1) = 1.8.
$$

On rejection (prob $1 - 0.8 = 0.2$), the resample comes from $p_{\text{res}} \propto \max(0, p - q) = (0, 0.2, 0)$, i.e. deterministically $B$ — which is exactly the mass the target distribution has that the draft lacks.

**Repo's rule** ($\tau = 0.8$). The check is $p(A) = 0.5 \ge 0.8 \cdot q(A) = 0.56$? No → reject. Expected tokens per step: 1.0. The draft over-proposes $A$ (0.7 vs 0.5), the verifier catches it, and the step falls back to the main token.

**Same draft, slightly different target.** If $p(A) = 0.6$ (so $p = (0.6, 0.3, 0.1)$): the exact scheme accepts $A$ with probability $0.6/0.7 \approx 0.857$, giving $1 + \alpha = 1 + (0.6 + 0.2 + 0.1) = 1.9$ tokens/step. The repo's rule accepts with probability 1 ($0.6 \ge 0.56$), giving 2.0 tokens/step — *more* than the exact scheme's expectation. This is the flip side of the deterministic rule: when the ratio clears the bar it accepts unconditionally, which is exactly why the emitted distribution can drift from $p$ (see [11.4](#114-the-repos-variant-greedy-verification)).

---

## The Repo's Decoder — Code Walkthrough

`inference/speculative.py:SpeculativeDecoder.generate_step` in full:

```python
@torch.inference_mode()
def generate_step(self, last_token: torch.Tensor, start_pos: int, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    main_logits = self.main_model(last_token, start_pos=start_pos, use_cache=True)
    main_probs = torch.softmax(main_logits[:, -1, :], dim=-1)
    # Sample the main token with temperature (argmax when temperature == 0);
    # the draft stays greedy (its most-likely continuation is what we verify).
    token_main = Transformer._sample(main_logits[:, -1, :], temperature, top_p=1.0, top_k=0).squeeze(0)
    t1_pos = start_pos + 1
    _, hidden = self.main_model.forward_with_hidden(token_main.unsqueeze(0), start_pos=t1_pos, use_cache=True)
    hidden_last = hidden[:, -1:, :]
    token_main_emb = self.main_model.embed(token_main.unsqueeze(-1))
    draft_logits, _ = self.mtp(hidden_last, token_main_emb)
    draft_probs = torch.softmax(draft_logits[:, -1, :], dim=-1)
    token_draft = draft_probs.argmax(dim=-1)
    # Acceptance compares raw (unscaled) probabilities of the draft token.
    p_main_of_draft = main_probs[0, token_draft[0]].item()
    p_draft_of_draft = draft_probs[0, token_draft[0]].item()
    return token_main, token_draft, p_main_of_draft >= self.threshold * max(p_draft_of_draft, 1e-12)
```

Step by step:

1. **Score the last token.** `main_model(last_token, start_pos, use_cache=True)` runs the trunk on the last emitted token and writes its KV entry (see [section 13](#cache-consistency-walkthrough)). `main_logits[:, -1, :]` is the distribution over the next token.
2. **Sample the main token.** `Transformer._sample(..., temperature, top_p=1.0, top_k=0)` — temperature scaling only (nucleus/top-k disabled); `temperature == 0` degenerates to argmax (`models/transformer.py:Transformer._sample`).
3. **Draft conditioning.** The MTP block predicts $x_{t+2}$ from $(h_{t+1}, e_{x_{t+1}})$ — i.e. from the hidden state of `token_main` and the embedding of `token_main` itself. `forward_with_hidden` supplies the pre-norm hidden; `self.main_model.embed` supplies the embedding (the same shared `nn.Embedding` as training).
4. **Greedy draft.** `token_draft = draft_probs.argmax(dim=-1)` — the draft's single most-likely continuation.
5. **Verify.** Accept iff the main model's probability of the draft token is at least `threshold ×` the draft's own probability of it. The `max(p_draft_of_draft, 1e-12)` floor prevents a spurious accept when the draft's argmax probability underflows to ~0.

The outer loop (`inference/speculative.py:SpeculativeDecoder.generate`):

```python
@torch.inference_mode()
def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 512, temperature: float = 1.0,
             eos_token_id: Optional[int] = None) -> torch.Tensor:
    output = input_ids.clone()
    n_generated = 0
    if hasattr(self.main_model, "reset_cache"):
        self.main_model.reset_cache()
    _ = self.main_model(output, start_pos=0, use_cache=True)
    while n_generated < max_new_tokens:
        start_pos = output.size(1) - 1
        last_token = output[:, -1:]
        token_main, token_draft, was_accepted = self.generate_step(last_token, start_pos=start_pos, temperature=temperature)
        output = torch.cat([output, token_main.unsqueeze(0)], dim=1)
        n_generated += 1
        if eos_token_id is not None and token_main.item() == eos_token_id:
            break
        if was_accepted and n_generated < max_new_tokens:
            output = torch.cat([output, token_draft.unsqueeze(0)], dim=1)
            n_generated += 1
            if eos_token_id is not None and token_draft.item() == eos_token_id:
                break
    return output
```

Invariants of the loop:

- **Cache reset before prefill.** `reset_cache()` (via `models/transformer.py:Transformer.reset_cache`) clears every layer's MLA cache, then the prompt is prefilled with one forward at `start_pos=0`. Forgetting this between conversations leaves stale entries from the previous prompt.
- **EOS is checked on both tokens.** The draft token is only appended if `n_generated < max_new_tokens` — the budget is enforced per token, not per step.
- **`token_main` is always emitted.** Even a rejected draft costs nothing in output quality: the fallback is exactly the main model's own token.

### 12.1 Training/inference alignment

The draft head's inference usage is the *same function* it was trained to compute — the only difference is where the conditioning token comes from:

| Training | Inference |
|---|---|
| Predict $x_{t+2}$ from $(h_t, e_{t+1})$ | Draft after main commits $t_1$ |
| CE loss on draft | Accept/reject vs main distribution |
| Shared head weights | Same `output_head` |
| Teacher forcing on $e_{t+1}$ (ground truth) | Greedy $e_{t_1}$ from main model's argmax |

This alignment is what makes MTP usable as a draft model without a separate training run: the head never sees a distribution shift at inference beyond the conditioning token being the model's own output instead of the ground truth. The better the main model's greedy choices match the training distribution, the better the draft's proposals — which is why acceptance rate correlates with MTP training quality and degrades under heavy sampling (see [section 14](#temperature-handling)).

**Pitfall — the interactive entry point does not wire the head.** `inference/generate.py:generate_interactive` constructs `MTPModule(model_cfg, depth=1)` and hands it to `SpeculativeDecoder` without calling `set_output_head`. As shipped, `python -m inference.generate --use_speculative` raises `RuntimeError("MTPModule(depth=1): output_head not set.")` on the first `generate_step`. The tests wire it explicitly (`mtp_module.set_output_head(model.head)` in `tests/test_inference.py`); the interactive path needs the same call after construction. [verified in source]

---

## Cache-Consistency Walkthrough

### 13.1 The question

`generate_step` runs the main model twice per step: once on `last_token` and once on `token_main` (via `forward_with_hidden`). Both calls write to the same MLA KV cache. Why is `token_main` written **twice** — and why is that safe?

### 13.2 Position-by-position trace

Let the prompt have $P$ tokens; after prefill the cache covers positions $0 \dots P-1$. Consider a step whose last emitted token sits at global position $s$ (`start_pos = s`).

1. **Forward #1 — `main_model(last_token, start_pos=s, use_cache=True)`.** The MLA layer writes the KV entry for position $s$ (`models/mla.py:MultiHeadLatentAttention.forward`):

   ```python
   if use_cache:
       self.kv_cache[:bsz, start_pos:end_pos] = kv_normed.detach()
       self.pe_cache[:bsz, start_pos:end_pos] = k_pe.detach()
   ```

   Cache now covers $0 \dots s$. `main_logits[:, -1]` scores the next token; `token_main` is sampled.

2. **Forward #2 — `forward_with_hidden(token_main, start_pos=s+1, use_cache=True)`.** This is the *draft* pass: the MTP block needs the trunk hidden state of `token_main` to predict the token after it. The forward writes the KV entry for position $s+1$. Cache now covers $0 \dots s+1$ — **one position ahead of the emitted output**, which still ends at $s$.

3. **Accepted.** Emit `token_main` and `token_draft`. Next step: `start_pos = s+2`, `last_token = token_draft`. Forward #1 writes position $s+2$. Every position is written exactly once; the cache is always a prefix of the emitted sequence.

4. **Rejected.** Emit only `token_main`. Next step: `start_pos = s+1`, `last_token = token_main`. Forward #1 processes `token_main` **again** at position $s+1$ — the same position forward #2 wrote in the previous step. The rewrite is **idempotent**: same token, same position, same prefix cache ($0 \dots s$ unchanged), deterministic forward → identical KV values. No corruption; one redundant forward pass.

### 13.3 Why the double write is structural

The draft head's contract is $x_{t+2} \leftarrow (h_{t+1}, e_{x_{t+1}})$: it needs the *hidden state* of the token it conditions on. The only way to obtain that hidden state is to run the trunk on `token_main` — that is forward #2, and its cache write is unavoidable. If the draft is then rejected, `token_main` is also the last emitted token, so the next step must run the trunk on it again as the query — forward #1 of the next iteration. Two writes, same position, same values. If the draft is accepted, `token_main` is written exactly once and never re-processed.

Note also that forward #2's logits are discarded (`_, hidden = ...`): the draft logits come from the MTP head, not the trunk. And the MTP block's own `nn.MultiheadAttention` attends only within its length-1 window (causal mask of size 1), so it needs no KV cache of its own — see [section 15](#kv-cache-sharing).

### 13.4 What could break consistency

- **Forgetting `reset_cache()`** between prompts: stale entries at positions beyond the new prompt's length get attended to, silently corrupting every subsequent token. `SpeculativeDecoder.generate` resets internally; `generate_step` alone does not.
- **Calling `generate_step` with a `start_pos` that does not match the cache's actual length**: the cache write is positional (`kv_cache[:, start_pos:end_pos]`), so a wrong `start_pos` writes into the wrong slot and the mask math (`models/transformer.py:Transformer._build_causal_mask`) will not catch it — the model will attend to garbage or future tokens.
- **Mutating the cache between the two forwards** (e.g. a second decoder sharing the same model): forward #2's KV values depend on the prefix, so any interleaved write breaks the idempotence argument.

---

## Temperature Handling

### 14.1 What the code does

`generate_step` takes a `temperature` argument and applies it in exactly one place: sampling `token_main` via `models/transformer.py:Transformer._sample`:

```python
if temperature == 0.0:
    return logits.argmax(dim=-1, keepdim=True)
logits = logits / temperature
...
next_token = torch.multinomial(probs, num_samples=1)
```

with `top_p=1.0, top_k=0` — so only temperature scaling is active; nucleus and top-k filtering are disabled in the speculative path.

### 14.2 Three deliberate asymmetries

1. **The draft is always greedy.** `token_draft = draft_probs.argmax(dim=-1)` regardless of temperature. This is intentional: the verifier checks the draft's *most-likely* continuation, and a sampled draft would make the acceptance test noisy. The comment in the source says exactly this: "the draft stays greedy (its most-likely continuation is what we verify)".
2. **Acceptance uses raw probabilities.** `main_probs` and `draft_probs` are softmaxes of the *unscaled* logits, computed before `_sample` applies temperature. So even at `temperature=0.7`, the threshold test compares the temperature-1 distributions. A temperature-aware verifier would compare the scaled distributions; this one does not.
3. **Temperature shapes the draft's input, not its output.** With `temperature > 0`, `token_main` may not be the argmax, and the hidden state fed to the MTP block is that of a *sampled* token. The draft then proposes a continuation of a token the main model chose probabilistically — which is fine for sampling-style generation, but it means acceptance rate and temperature interact: higher temperature pushes `token_main` away from the mode, and the draft (trained under teacher forcing on the true next token) is less likely to agree.

### 14.3 Pitfalls

- **`temperature=0` is the only setting where the emitted sequence is close to deterministic** — and even then, as section [11.4](#114-the-repos-variant-greedy-verification) notes, an accepted draft token can deviate from the main model's own argmax.
- **The `1e-12` floor** in `p_main_of_draft >= self.threshold * max(p_draft_of_draft, 1e-12)` guards the degenerate case where the draft's argmax probability underflows to zero (e.g. after aggressive temperature scaling of the *draft* logits — not done here, but easy to add later). Without the floor, `0 >= 0.8 * 0` would spuriously accept.
- **`Transformer.generate` (standard path) supports `top_p`/`top_k`; the speculative path does not.** If you need nucleus sampling with MTP, the acceptance math would need re-deriving — the threshold rule is calibrated for the raw distribution.

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

MTP weights are saved with `mtp.` prefix. In `training/pretrain.py:Pretrainer.save_checkpoint`:

```python
if self.mtp_wrapper is not None:
    mtp_mod = self.mtp_wrapper
    orig = getattr(mtp_mod, "_orig_mod", mtp_mod)
    mtp_state = {f"mtp.{k}": v for k, v in orig.state_dict().items() if k.startswith("mtp_modules.")}
    state.update(mtp_state)
```

The `mtp_modules.` filter deliberately excludes the shared `embed.weight` and `main_model.*` keys (restored via the base model's own state dict) — but it *includes* `mtp_modules.0.output_head.weight`, since the shared head is a registered submodule of each `MTPModule`. On load (`training/pretrain.py:Pretrainer` resume path and `inference/generate.py`):

```python
mtp_state = {k.removeprefix("mtp."): v for k, v in state.items() if k.startswith("mtp.")}
mtp_orig.load_state_dict(mtp_state, strict=False)
```

`strict=False` absorbs the `output_head.weight` key when the loading module has no head wired (see [section 10.5](#105-pitfalls)). The checkpoint metadata carries `has_mtp`, and the resume path only attempts the MTP restore when it is true. Optimizer state for MTP is **not** saved separately — MTP params are included in the main optimizer state dict via `MultiTokenPrediction.parameters()`.

---

## Comparison: MTP vs Standard Decode

| Aspect | Standard `generate()` | `SpeculativeDecoder` |
|---|---|---|
| Tokens per step | 1 | 1 or 2 |
| Main-model forwards per token | 1 | $2/(1+\hat\alpha) \ge 1$ (2 forwards per step; 1.8 tokens at $\hat\alpha=0.8$ → 1.11) |
| Requires MTP weights | no | yes |
| Sampling | temperature, top-k, top-p | temperature on main token; draft greedy; top-k/top-p disabled |
| Output distribution | exactly the sampled target distribution | greedy-verification heuristic; can deviate from target (see [11.4](#114-the-repos-variant-greedy-verification)) |
| KV cache | main MLA | main MLA only |
| Training overhead | none (if mtp_depth=0) | +30% loss weight, ~7.1M params, ~5% FLOPs (estimate) |
| Wall-clock | 1 step boundary per token | 1 step boundary per 1–2 tokens; win only if per-step overhead is significant (unmeasured) |

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

### Per-position gradient density

For a sequence of length $S$ at depth 1, position $t$ contributes to the total loss in two ways: as the predictor of $x_{t+1}$ (main loss) and, for $t \le S-3$, as the predictor of $x_{t+2}$ (MTP loss). The last two positions only receive the main-loss gradient. With `mtp_depth=1` the trunk receives ~2× the gradient signal per position on average; deeper stacks multiply this further. This is the precise sense in which MTP "densifies" training.

---

## Depth > 1 Extension Sketch

`MultiTokenPrediction` uses `ModuleList` of `MTPModule` length `mtp_depth`. Each depth adds another $(h, e_{t+d})$ fusion predicting $t+d+1$. This repo ships `mtp_depth=1` only; extending requires:

1. YAML `mtp_depth: K`
2. Loop depths in `forward` with shifting embed slices
3. Sum MTP losses with optional per-depth weights
4. Speculative decode chain for multi-token drafts

Untested beyond depth=1 in this codebase. The alignment algebra of [section 9](#length-alignment-algebra--deriving-usable) already generalizes: depth $d$ uses `usable = seq_len - d - 2` and chains `prev_h = hidden`.

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

---

## Appendix A — Worked Alignment Example

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

## Appendix B — Gradient Flow

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

**Q: Can I disable MTP?** Set `mtp_depth: 0` or `mtp_loss_weight: 0.0` in config. `Pretrainer` skips the wrapper (it only builds `MultiTokenPrediction` when both `mtp_depth > 0` and `config.mtp_weight > 0.0`).

**Q: Does speculative decoding work without trained MTP weights?** It runs but acceptance rate collapses — the draft head is random.

**Q: Why greedy only in speculative decode?** Rejection sampling with temperature requires matching draft and target distributions exactly. The current implementation uses argmax for simplicity.

**Q: Is the acceptance rule lossless?** No. True speculative decoding resamples from a corrected distribution on rejection so the output distribution exactly matches the target model's. The threshold rule here just truncates — when the draft over-proposes a token, the rule either accepts it (biasing the output toward draft-favoured tokens) or falls back to greedy. It is a throughput heuristic for greedy-ish decoding, not a distribution-preserving sampler.

**Q: Why is `token_main` run through the model twice?** Once to obtain its hidden state for the draft (unavoidable — the MTP head conditions on it), and again as the query of the next step when the draft is rejected. The second write is idempotent (same token, same position, same prefix), so the cache stays consistent.

**Q: Does temperature affect the draft?** No — the draft is always greedy. Temperature only shapes `token_main`, and the acceptance test always compares raw (temperature-1) probabilities.

**Q: What is the real speedup?** Unknown — no trained checkpoint exists and `.benchmarks/` is empty. The implementation does not reduce main-model forward count (2/(1+α̂) ≥ 1 per token); any wall-clock win depends on per-step overhead and must be measured.

---

## Appendix D — Glossary

| Term | Meaning |
|---|---|
| `mtp_depth` | Number of MTP heads (canonical: 1) |
| `mtp_loss_weight` | λ in `L = L_main + λ·L_mtp` (0.3) |
| `usable` | `seq_len - d - 2`, valid MTP window length for 0-indexed depth `d` |
| `mtp_pairs` | List of `(logits, targets)` per depth |
| `prev_h` | Pre-norm trunk hidden from `forward_with_hidden` |
| `acceptance_threshold` | Min probability ratio to accept draft (0.8) |
| `α` (alpha) | Acceptance rate `Σ_x min(p(x), q(x))` in the exact scheme |
| `forward_with_hidden` | Trunk forward returning `(logits, pre-norm hidden)` |
| `set_output_head` | Wires the shared LM head into an `MTPModule` |

---

## Load-Bearing Invariants

1. **Shared output head** — `mtp.set_output_head(main_model.head)`. Never create a separate LM head for MTP.
2. **Shared embedding** — `self.add_module("embed", main_model.embed)`.
3. **`forward_with_hidden` returns pre-norm `h`** — MTP applies its own `RMSNorm` before the shared head.
4. **`use_cache=False` during training** — MTP training path does not use MLA KV cache.
5. **MTP checkpoint prefix** — `mtp.` keys in safetensors; strip on load.
6. **Alignment** — `usable = seq_len - d - 2`; `h_in`, `emb_in`, `tgt` all have exactly `usable` elements.
7. **Cache idempotence** — the double write of `token_main` is safe because the forward is deterministic and the prefix is unchanged.

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

> **See also:** [[Docs/10_Inference_and_Serving|Inference]] and [[Docs/08_Training_Pipeline|Training]] for generation and training pipeline integration.

<!-- docs:verified 2026-08-04 · 59aeef3 -->
