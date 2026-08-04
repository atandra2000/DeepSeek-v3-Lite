# Mixture-of-Experts (MoE) — AuxLossFreeGate + DeepSeekMoE

> **Prerequisites:** [[Docs/02_Model_Architecture|Model Architecture]].

> **Read this if** you're debugging expert collapse, routing histograms, or aux-loss-free bias updates. **Skip if** you only need YAML knobs → [[Docs/08_Training_Pipeline|Training]].

**Depends on:** [[Docs/02_Model_Architecture|Model Architecture]] · **Read next:** [[Docs/05_Multi_Token_Prediction|MTP]], [[Docs/08_Training_Pipeline|Training]]

---

## Table of Contents

1. [Abstract](#abstract)
2. [Motivation — Capacity Without Dense Compute](#motivation-capacity-without-dense-compute)
3. [DeepSeek MoE Design](#deepseek-moe-design)
4. [Mathematical Formulation (DeepSeek-V3 Paper)](#mathematical-formulation-deepseek-v3-paper)
5. [Expert Capacity and Sparsity Math](#expert-capacity-and-sparsity-math)
6. [Auxiliary-Loss-Free Load Balancing](#auxiliary-loss-free-load-balancing)
7. [The Bias Controller as a Feedback System](#the-bias-controller-as-a-feedback-system)
8. [Why Not a Standard Auxiliary Loss?](#why-not-a-standard-auxiliary-loss)
9. [The SwiGLU Expert](#the-swiglu-expert)
10. [Dimension Breakdown](#dimension-breakdown)
11. [Implementation in This Repo](#implementation-in-this-repo)
    - [Class Structure](#class-structure)
    - [AuxLossFreeGate Forward Pass](#auxlossfreegate-forward-pass)
    - [DeepSeekMoE Forward Pass](#deepseekmoe-forward-pass)
    - [Sorted-Token Dispatch Layout](#sorted-token-dispatch-layout)
    - [Stacked Dispatch Path](#stacked-dispatch-path)
    - [Triton Grouped Dispatch Path](#triton-grouped-dispatch-path)
    - [Shared Expert Path](#shared-expert-path)
    - [Training Integration](#training-integration)
12. [Comparison: MoE vs Dense FFN](#comparison-moe-vs-dense-ffn)
13. [Performance Characteristics](#performance-characteristics)
14. [Appendix A — A worked numerical example](#appendix-a--a-worked-numerical-example)
15. [Appendix B — The Triton grouped-GEMM path](#appendix-b--the-triton-grouped-gemm-path)
16. [Appendix C — Gradient flow in MoE](#appendix-c--gradient-flow-in-moe)
17. [Appendix D — Load-balance metrics](#appendix-d--load-balance-metrics)
18. [Appendix E — Bias-update lifecycle (state diagram)](#appendix-e--bias-update-lifecycle-state-diagram)
19. [Appendix F — Numerical stability notes](#appendix-f--numerical-stability-notes)
20. [Appendix G — Glossary](#appendix-g--glossary)
21. [Appendix H — Frequently Asked Questions](#appendix-h--frequently-asked-questions)
22. [Check Your Understanding](#check-your-understanding)
23. [Load-Bearing Invariants (Do Not Break)](#load-bearing-invariants-do-not-break)
24. [Implementation Checklist](#implementation-checklist)
25. [References](#references)
26. [Sigmoid vs Softmax Rationale](#sigmoid-vs-softmax-rationale)
27. [Bias Update Mechanism — Deadband Rule](#bias-update-mechanism--deadband-rule)
28. [Original Scores for Weights](#original-scores-for-weights)
29. [MoE Comparison Table — DeepSeek vs GPT-OSS-Lite](#moe-comparison-table--deepseek-vs-gpt-oss-lite)

---

## Abstract

**Mixture-of-Experts (MoE)** replaces the dense feed-forward network (FFN) in selected transformer layers with a bank of sparsely activated expert networks. Each token is routed to a small subset of experts (top-*k*), so the model's **parameter capacity grows linearly** with the number of experts while **active compute grows only with *k***.

DeepSeek-V3 introduces two MoE innovations that this repo reproduces faithfully:

1. **Fine-grained expert decomposition** — many small experts (20 routed) with top-4 activation, giving a 20% routed sparsity ratio close to the full V3 design.
2. **Auxiliary-loss-free load balancing** — a per-expert **bias buffer** shifts top-*k* *selection* based on observed token counts, without injecting a load-balancing gradient into the task loss. Routing **weights** come from unbiased sigmoid scores.

The implementation lives in `models/moe.py`. An optional Triton fused kernel in `models/moe_triton.py` accelerates the routed-expert dispatch path on supported hardware.

---

## Motivation — Capacity Without Dense Compute

A standard transformer FFN (SwiGLU) at every layer costs:

```
FLOPs/token/layer ≈ 6 · d · d_ff     (three matmuls: W1, W3 input; W2 output)
Params/layer       ≈ 3 · d · d_ff
```

For DeepSeek-V3-Lite (canonical `pretrain_a100_422m` config — the "422m" name is the historical filename; the deduped count is 411.6M): `d = 768`, dense `d_ff = 1536`, MoE `d_ff = 384`.

| Layer type | Layers | Active FFN dim | Params per FFN | Active FLOPs factor |
|---|---|---|---|---|
| Dense SwiGLU | 2 | 1536 | 3.5M | 1.0× |
| MoE (top-4 of 20 + 1 shared) | 16 | 384 per expert | 18.6M stored | ~5× 384-dim SwiGLUs |

**The key trade-off:** MoE stores ~5× more FFN parameters per layer than a dense layer of equivalent width, but each token only *executes* 4 routed experts + 1 shared expert. You get **high capacity, moderate compute**.

### Why sparsity breaks without load balancing

If routing is purely data-driven with no balancing mechanism, a positive-feedback loop emerges:

```
Expert A gets slightly more tokens
    → Expert A's weights adapt faster
    → Gate routes more tokens to Expert A
    → Other experts starve
    → Model capacity collapses to a few experts
```

Formalize the loop. Let `c_e(t)` be the number of token slots routed to expert `e` at step `t`. Experts update with per-token gradients, so an expert's effective learning signal scales (roughly) with `c_e(t)`. A slightly higher `c_A` means expert A's output adapts faster to the training data, which (in the early-data regime where the gate learns to trust whatever expert produces the right answer) raises A's scores further — `E[c_A(t+1)] > E[c_A(t)]`. This is a linear instability: the load vector drifts toward a corner of the simplex, and the model silently loses most of its parameter capacity. Classic MoE papers solve this with an **auxiliary load-balancing loss** added to the training objective. DeepSeek-V3's insight: you can achieve the same equilibrium with a **bias buffer** updated out-of-band, keeping the task loss gradient clean.

---

## DeepSeek MoE Design

DeepSeek's MoE architecture (introduced in [DeepSeekMoE, 2024](https://arxiv.org/abs/2401.06066) and refined in V3) has three structural choices this repo mirrors:

### 1. Fine-grained experts

Instead of 8 large experts (Mixtral-style), DeepSeek uses **many small experts** (256 in V3-671B; 20 in this reproduction). Fine-grained decomposition improves expert specialisation: each expert can capture a narrower function. With top-4 of 20, the router has `C(20, 4) = 4,845` distinct expert *subsets* it can compose per token, versus `C(8, 2) = 28` for a Mixtral-style 8-expert/top-2 layout — the combinatorics of *combinations* is where the routing flexibility lives, and it grows much faster than the expert count.

### 2. Shared experts

One (or more) **shared experts** run on **every token**, bypassing the router entirely. Shared experts capture common patterns (syntax, high-frequency constructions) that every token needs, freeing routed experts to specialise.

This repo: **`n_shared_experts = 1`**, always active. (The README once mentioned 2; the canonical config and code use 1.)

### 3. Top-*k* routing with sigmoid gating

Each token selects `n_activated_experts = 4` of `n_routed_experts = 20` routed experts. The gate uses **sigmoid** scores (not softmax over all experts), so multiple experts can be independently "on" without competing for a fixed probability mass.

### Layer placement in this repo

```
Layer 0:  MLA + Dense SwiGLU  (inter_dim = 1536)
Layer 1:  MLA + Dense SwiGLU  (inter_dim = 1536)
Layers 2–17: MLA + DeepSeekMoE (16 MoE layers)
              ├─ 20 routed experts (top-4 per token)
              └─ 1 shared expert (always on)
```

Controlled by `n_dense_layers = 2` in `TransformerBlock` (`models/transformer.py:TransformerBlock.forward`): MoE replaces FFN when `layer_id >= n_dense_layers`.

---

## Mathematical Formulation (DeepSeek-V3 Paper)

Let `h_t ∈ ℝ^d` be the post-attention hidden state for token `t` (after RMSNorm). The MoE FFN computes:

### Gating

```
s_t     = σ(h_t W_g^T)                    (1)  scores ∈ (0,1)^N
s̃_t     = s_t + b                          (2)  biased scores (selection only)
ℐ_t     = TopK(s̃_t, k)                    (3)  expert indices, |ℐ_t| = k
w_{t,i} = s_{t,i} / Σ_{j∈ℐ_t} s_{t,j}     (4)  normalised weights (from raw s, NOT s̃)
```

Where:

| Symbol | Shape | Purpose |
|---|---|---|
| `W_g` | `N × d` | Gate projection (`nn.Parameter`) |
| `b` | `N` | Per-expert bias buffer (NOT a Parameter) |
| `σ` | — | Sigmoid activation |
| `N` | scalar | `n_routed_experts` (= 20) |
| `k` | scalar | `n_activated_experts` (= 4) |

**Critical split:** `b` affects *which* experts are selected (eq. 2–3) but *not* the routing weights used in the weighted sum (eq. 4). This decouples load balancing from the gradient path.

### Expert computation

Each expert `e` is a SwiGLU FFN:

```
f_e(h) = W_{2,e} · silu(W_{1,e} h) ⊙ W_{3,e} h        (5)
```

### Routed output

```
y_t^routed = Σ_{i ∈ ℐ_t} w_{t,i} · f_i(h_t)            (6)
```

### Shared output

```
y_t^shared = Σ_{s=1}^{N_s} f_s^{shared}(h_t)            (7)
```

With `N_s = n_shared_experts = 1` in this repo, this is a single SwiGLU.

### Final MoE output

```
y_t = y_t^routed + y_t^shared                           (8)
```

The residual connection (`x + ffn(ffn_norm(x))`) is handled by `TransformerBlock`, not inside `DeepSeekMoE`.

---

## Expert Capacity and Sparsity Math

Before the gate details, it is worth building the quantitative picture of *how sparse* the MoE really is, because every later number (load balance, Triton kernel sizing, the register cap) hangs off these counts.

### Routing slots

Each forward pass flattens `(B, S, D)` into `(T, D)` with `T = B × S` and produces, for every token, `k` expert assignments. The total number of **routing slots** per MoE layer per forward is therefore:

```
slots = T × k = B × S × k
```

Canonical micro-batch (`B=8, S=2048, k=4`):

```
slots = 8 × 2048 × 4 = 65,536
```

If every expert received exactly its fair share, each of the `N = 20` experts would see

```
μ = slots / N = 65,536 / 20 = 3,276.8 tokens
```

per layer per forward.

### Load imbalance factor and capacity factor

Define the **load imbalance factor** as the ratio of the busiest expert's count to the fair share:

```
ρ = max_e count_e / (slots / N)
```

`ρ` is exactly the *capacity factor* a GShard-style implementation would need to allocate so that the busiest expert fits in a single preallocated batch without dropping tokens. `ρ = 1` means perfectly balanced; `ρ = 1.25` (the classic GShard/Switch setting) tolerates up to 25% over the mean.

### How much imbalance is "just noise"?

Suppose the gate routed each of the 65,536 slots independently and uniformly across 20 experts (a null model with no balancing mechanism). Then the count vector is multinomial with `slots` trials and probability `1/N` per expert. The per-expert standard deviation is

```
σ = √(slots · (1/N) · (1 − 1/N)) = √(65,536 × 0.05 × 0.95) ≈ 55.8
```

The largest of 20 independent draws concentrates near

```
E[max_e count_e] ≈ μ + σ·√(2 ln N) ≈ 3,276.8 + 55.8 × 2.45 ≈ 3,413
```

so even a *perfectly* uniform router produces `ρ ≈ 3,413 / 3,276.8 ≈ 1.04` — about 4% slack — purely from sampling noise. Any capacity planning must budget at least that. (Derived under the i.i.d. uniform model; the real gate is not i.i.d., but the magnitude is right.)

### This repo drops no tokens

GShard and Switch allocate `capacity = ceil(capacity_factor × slots/N)` token slots per expert and **drop** (or relegate to a residual connection) any token that arrives when the expert's buffer is full. This repo does neither: every one of the 65,536 slots is dispatched, and each expert simply runs a GEMM of size `count_e × I`. Imbalance therefore costs **compute and wall-clock** (the hot expert's chunk is bigger, and the per-expert GEMMs are sized by the real counts), never correctness. The `counts` vector is the exact histogram, and `max_e count_e` is the true load each layer must absorb.

### Active parameters and FLOPs per token

Stored FFN capacity per MoE layer: 20 routed + 1 shared expert = 21 experts × `3·D·I` = 21 × 884,736 = 18,579,456, plus the gate `N × D` = 15,360 → **18,594,816 (~18.6M)**.

Active per token: 4 routed + 1 shared = 5 experts:

```
active FFN per token per MoE layer = 5 × 6·D·I FLOPs = 5 × 6 × 768 × 384 ≈ 8.85M FLOPs
active FFN params per token per MoE layer = 5 × 884,736 = 4,423,680
```

So per token, a MoE layer executes **5/21 ≈ 23.8%** of its stored FFN parameters. That ratio — not `k/N` — is the true "active fraction" of a MoE layer, because the shared expert is always on: `(k + N_s)/(N + N_s) = 5/21 ≈ 0.238`. The routed-only sparsity `k/N = 4/20 = 20%` is the more familiar headline number; both are worth keeping straight.

The whole-model active budget (measured by instantiating `Transformer` with the canonical config and counting the tensors each token touches):

```
embedding + tied head:  76,813,824   (100,018 × 768, counted once)
attention: 18 × 1,677,504 = 30,195,072
dense FFN: 2 × 3,538,944  =  7,077,888
MoE: 16 × (15,360 gate + 5 × 884,736 experts) = 71,024,640
                                total ≈ 185,111,424 ≈ 185.1M
```

This is the canonical **~185M active parameters per token** (411.6M stored), i.e. about 45% of stored parameters are exercised on every token. The rest of the 411.6M is routed-expert capacity that only comes alive for the tokens that select it.

### Sparsity takeaways

| Quantity | Value |
|---|---|
| Routed sparsity `k/N` | 20% |
| True active fraction `(k+N_s)/(N+N_s)` | 23.8% |
| Slots per forward (B=8, S=2048) | 65,536 |
| Fair-share count per expert | 3,276.8 |
| Noise floor `σ` (uniform null model) | ≈ 55.8 (ρ ≈ 1.04 worst-expert) |
| Active params per token | ~185.1M of 411.6M |

---

## Auxiliary-Loss-Free Load Balancing

DeepSeek-V3 §2.3.3 replaces the auxiliary loss with a **dynamic bias update** on the gate scores.

### The bias buffer

```python
self.register_buffer("bias", torch.zeros(n_routed_experts, dtype=torch.float32))
```

Properties:

| Property | Value | Why it matters |
|---|---|---|
| Registered as `buffer` | yes | Not in `optimizer.param_groups` — no SGD/AdamW updates |
| In `state_dict()` | yes | Survives checkpoint save/load (`test_bias_in_state_dict`) |
| In `parameters()` | no | `test_bias_not_in_parameters` enforces this |
| Dtype | FP32 | Stable accumulation across thousands of bias steps |
| Gradients | never | `@torch.no_grad()` on `update_bias` |

### Update rule

Every `bias_update_every` optimizer steps, `Pretrainer._update_moe_bias` calls `DeepSeekMoE.update_gate_bias`, which counts how many token slots were routed to each expert in the last forward and applies (`models/moe.py:AuxLossFreeGate.update_bias`):

```python
avg = counts.mean()
bias[counts > avg * (1 + upper)] -= speed    # over-loaded → demote
bias[counts < avg * (1 - lower)] += speed    # under-loaded → promote
```

Default thresholds: `upper = lower = 0.10`, `speed = 0.001`.

**Intuition:** An over-loaded expert's bias decreases, making it less likely to win top-*k* in the next window. An under-loaded expert's bias increases. Experts within the deadband (`avg × 0.9` to `avg × 1.1`) are untouched.

### Worked bias example

Suppose `N = 4` experts, `T × k = 20` total routing slots in the counting window:

```
counts = [10, 5, 5, 0]
avg    = 5.0
upper threshold = 5.5,  lower threshold = 4.5

Expert 0: 10 > 5.5  → bias -= speed  (over-loaded)
Expert 1: 5 in deadband → unchanged
Expert 2: 5 in deadband → unchanged
Expert 3: 0 < 4.5   → bias += speed  (under-loaded)
```

This is exactly what `TestDeepSeekMoEAdditional::test_update_bias_sign_rule` verifies.

### Why routing weights ignore the bias

```python
biased = scores + self.bias          # used ONLY for topk selection
indices = biased.topk(self.topk)[1]
weights = scores.gather(1, indices)  # raw sigmoid, no bias
```

If weights came from `biased`, the bias would enter the autograd graph through the weighted expert sum, and the optimizer would fight the out-of-band bias updates. By using raw `scores` for weights, the bias is a **pure inference-time routing knob** for load balancing.

---

## The Bias Controller as a Feedback System

The bias update is not an optimizer step and it is not a loss term — it is a **discrete-time feedback controller**, and thinking about it in control-theory terms explains every design choice (the deadband, the fixed step size, the update cadence, and the failure modes). This section reads `models/moe.py:AuxLossFreeGate.update_bias` and `models/moe.py:DeepSeekMoE.update_gate_bias` as a control loop.

### The closed loop

Map the pieces of the loop onto the standard control vocabulary:

| Control concept | This repo |
|---|---|
| Plant | The routing distribution: `P(e ∈ ℐ_t)` as a function of the bias |
| Control input `b` | The per-expert bias buffer, `(N,)` |
| Measured output | `counts_e` = `bincount(_last_indices)` from the last forward |
| Reference signal | Uniform load: every expert at `slots / N` |
| Actuator | Additive bias shifts the top-*k* *selection* boundary |
| Controller | The deadband + fixed-step update in `update_bias` |

Each optimizer step (when `bias_update_every == 1`), the loop runs once:

```
measure counts(c_k) ──► deadband comparator ──► b_{k+1} = b_k + u(c_k) ──► next forward selects with new b
```

where the control law is

```
u_e(c) = −speed · 1[ c_e > avg·(1+upper) ]  +  speed · 1[ c_e < avg·(1−lower) ]
```

### The deadband is a deadzone that suppresses limit cycles

The two thresholds implement a **deadzone** around the reference: inside `[avg·(1−lower), avg·(1+upper)]` the controller does nothing. Without the deadzone, pure sampling noise would flip the update sign every step, and the bias would random-walk — the controller would *inject* noise into routing instead of removing it.

Quantify with the canonical micro-batch. The null-model standard deviation is `σ ≈ 55.8` (Section 5), and the deadband half-width is `0.10 × 3,276.8 ≈ 327.7` tokens — about **5.9σ**. Under the null model the probability that one expert's count strays past the deadband purely by chance is

```
P(|c_e − μ| > 0.1μ) ≈ 2·(1 − Φ(5.87)) ≈ 4×10⁻⁹
```

per expert per step — essentially never. The controller therefore fires only on *systematic* imbalance (a real load drift), never on benign statistical fluctuation. This is the whole point of pairing a deadband with a small fixed step: the two together give a controller that is both noiseless at the equilibrium and slow-but-sure far from it.

### Integral-only control: no P term, no damping

The update accumulates a persistent correction (`b_{k+1} = b_k ± speed`), which makes it a pure **integral (I) controller** with a fixed step. There is no proportional term: the correction does not scale with the size of the imbalance. Why fixed step instead of proportional? A proportional controller needs the sensitivity `∂P(e ∈ top-k)/∂b_e` — the slope of the selection-probability curve at the current operating point — which is data-dependent, changes as the gate trains, and is not directly observable from counts alone. A fixed step is robust to all of that: it is the safest thing you can do without a plant model.

The consequences are the classic I-controller trade-off:

- **No overshoot.** The loop is monotone (see below), so the bias never oscillates past the equilibrium due to gain.
- **Slow convergence for large errors.** The bias moves `±speed` per step regardless of how far out of balance an expert is. The worst case is an expert whose score sits so far below the selection boundary that it needs a large bias offset `Δb*` to win top-*k*; the number of steps is `Δb* / speed`. Scores live in `(0, 1)` and the boundary gaps seen in practice are on the order of 0.01–0.1, so a cold expert needs on the order of 10–100 optimizer steps to be lifted into the rotation — fast compared to the training horizon, and harmless because the expert's weights are not decaying while it waits.
- **No damping term.** There is no velocity feedback, but none is needed: the plant (routing frequency) is monotone in the bias, so the loop cannot overshoot the way a spring-mass system does.

### Monotonicity gives stability

The key structural property of sigmoid + additive-bias + top-*k* routing is that raising `b_e` can only help expert `e`:

```
P(e ∈ ℐ_t) is non-decreasing in b_e
```

because increasing `b_e` raises `s̃_{t,e} = s_{t,e} + b_e`, and an element that rises in the sorted order of `s̃_t` can only move up into the top-*k* set, never out of it (with ties resolved in favor of the raised expert). The same argument shows the *expected* load `E[c_e | b]` is non-decreasing in `b_e`.

Now chain the loop: an over-loaded expert (count above the deadband) gets `b_e` decreased, which strictly decreases its expected load; an under-loaded expert gets the opposite. The update is therefore **sign-consistent with the error**: every move of the bias pushes the expert's expected load back toward the deadband. Because the expected load is monotone in the bias, there is a fixed point inside the deadband, and because the correction is bounded (`|Δb_e| ≤ speed` per step) the bias is bounded as well: a bias that grew without bound would push the expert's selection frequency to 1, its count above the deadband, and the update sign would flip. The loop cannot diverge; at worst it orbits the equilibrium with amplitude bounded by one step's effect.

The equilibrium is characterized exactly: at steady state,

```
counts_e ∈ [avg·(1 − lower), avg·(1 + upper)]   for all e
```

Since `Σ_e counts_e = slots` is conserved by routing, the only configuration in which *every* expert sits inside the deadband is one very close to uniform load — which is precisely the desired operating point.

### Why counts and not weights

The controlled variable is selection **frequency** (`counts`), not contribution **mass** (weighted sums). The bias exists to equalize *load* — how often each expert's weights are updated — because that is what prevents the positive-feedback collapse of Section 2. The routing *weights* encode the model's confidence in each expert for each token and are deliberately left untouched; the controller would corrupt them if it scaled them, since they are the task's own signal. This separation (frequency controlled, mass untouched) is the design's core.

### Window semantics: one forward, not an EMA

The counts come from `_last_indices` — the single most recent forward's routing (65,536 slots at B=8), stashed detached in `DeepSeekMoE.forward`. There is no exponential moving average and no multi-step accumulation window. `bias_update_every` (canonical: 1) controls how often that snapshot is *consumed*, and the consumption happens on optimizer steps — not micro-steps — so gradient-accumulation runs (4 micro-steps) update the bias once per optimizer step, on the routing of the final micro-step. The trade-off: a single micro-batch is a noisy sample of the true distribution, but the deadband (5.9σ) is sized precisely so that this noise does not reach the actuator; and reacting to the *latest* routing keeps the controller tracking the LR-driven drift in routing preferences instead of lagging a window behind it.

### Failure modes (what breaks the loop)

| Failure | Symptom | Mechanism |
|---|---|---|
| `update_gate_bias` never called | Slow collapse over thousands of steps | Open loop: the positive feedback of Section 2 runs unchecked |
| Bias made a `Parameter` | Bias drifts under optimizer noise | The optimizer's updates fight the controller's; `test_bias_not_in_parameters` exists to prevent this |
| `update_bias` called inside autograd | Bias receives gradients | In-place buffer writes under `requires_grad` corrupt the graph; `@torch.no_grad()` prevents it |
| `speed` too large | Dithering near the boundary | Each step overshoots the deadband, flipping the sign; routing quality degrades |
| `bias_update_every` large + small batch | Controller reacts late | The counts snapshot is old by the time it is consumed |
| Bias grows without bound (monitor) | All experts tied; balance metric at ceiling | The loop is open (one of the above); check the monitor alarms in Appendix D |

---

## Why Not a Standard Auxiliary Loss?

The classic Switch/GShard auxiliary loss (used in GPT-OSS-Lite and many other MoE models) adds:

```
L_aux = α · N · Σ_i f_i · P_i
```

where `f_i` is the fraction of tokens routed to expert `i` and `P_i` is the mean routing probability. This term pushes the gate toward uniform routing.

**Problems with auxiliary loss in DeepSeek-V3's design:**

1. **Gradient contamination** — `L_aux` backprops through the gate, competing with the language-modelling objective. The optimal α is task-dependent and often requires tuning.
2. **Sigmoid gates** — DeepSeek uses independent sigmoid scores, not a softmax simplex. The aux-loss derivation assumes softmax routing probabilities that sum to 1.
3. **Empirical regression** — Replacing bias updates with aux loss in this codebase **silently breaks MoE balance** (see `AGENTS.md` hard rule 3). The bias mechanism and sigmoid gate are co-designed.

This repo exposes `get_load_balance_loss()` for **logging only** — it computes the classic `(f · P) · N` surrogate from the last forward's routing cache. `balance_loss_alpha` is not wired into the training loss; the metric appears in W&B logs as `balance_loss`. Section 7 explains *why* the two approaches are not interchangeable: the aux loss optimizes an objective inside the gradient path, while the bias controller regulates a measured quantity outside it. The controller can also act on quantities a loss cannot — e.g. hard counts, which are non-differentiable — which is exactly why the aux-loss-free design can use the (non-differentiable) top-*k* selection statistics directly.

---

## The SwiGLU Expert

Both routed and shared experts share the same `Expert` class (`models/moe.py:Expert.forward`):

```python
class Expert(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

This is identical to `SwiGLUFFN` in `models/transformer.py` (`models/transformer.py:SwiGLUFFN.forward`) — the dense FFN used in layers 0–1. The only difference is width: dense layers use `inter_dim = 1536`, MoE experts use `moe_inter_dim = 384`.

### Parameter layout

| Weight | `nn.Linear` shape | Stored as |
|---|---|---|
| `w1` | `(I, D)` | gate projection |
| `w2` | `(D, I)` | down projection |
| `w3` | `(I, D)` | up projection |

Per expert: `3 · D · I` parameters.

For the canonical config (`D=768, I=384`): **884,736 params/expert**.

### Why SwiGLU for experts

SwiGLU (Shazeer, 2020) uses a gated linear unit with SiLU activation:

```
SwiGLU(x) = W₂ · (silu(W₁x) ⊙ W₃x)
```

Compared to vanilla GELU-FFN, SwiGLU adds one extra projection (`W₃`) but consistently improves quality per parameter at the same FLOP budget. DeepSeek-V3 uses SwiGLU everywhere (dense and MoE), and this repo follows suit.

---

## Dimension Breakdown

### DeepSeek-V3 (original, 671B)

| Parameter | Value | Description |
|---|---|---|
| `d_model` | 7,168 | Hidden dimension |
| `n_routed_experts` | 256 | Fine-grained routed experts |
| `n_activated_experts` | 8 | Top-*k* per token |
| `n_shared_experts` | 1 | Always-active shared expert |
| `moe_inter_dim` | 2,048 | Expert hidden width |
| MoE layers | 58 of 61 | First 3 layers dense |
| Routed sparsity | 8/256 = 3.1% | Active routed fraction |

### DeepSeek-V3-Lite (this repo, canonical config — 411.6M deduped)

| Parameter | Value | Description |
|---|---|---|
| `dim` | 768 | Hidden dimension |
| `n_routed_experts` | 20 | Routed experts per MoE layer |
| `n_activated_experts` | 4 | Top-*k* per token |
| `n_shared_experts` | 1 | Shared expert per MoE layer |
| `inter_dim` | 1536 | Dense FFN width (layers 0–1) |
| `moe_inter_dim` | 384 | MoE expert width |
| `n_dense_layers` | 2 | Layers 0–1 are dense |
| MoE layers | 16 of 18 | Layers 2–17 |
| Routed sparsity | 4/20 = 20% | Active routed fraction |

### Parameter accounting (per MoE layer, canonical config)

```
Gate:           N × D           = 20 × 768           =     15,360
Routed experts: N × 3 × D × I   = 20 × 3 × 768 × 384 = 17,694,720
Shared experts: N_s × 3 × D × I =  1 × 3 × 768 × 384 =    884,736
                                                          ──────────
Per MoE layer total:                                      18,594,816  (~18.6M)
× 16 MoE layers:                                          297,517,056 (~298M)
```

The gate bias (`N` floats) is negligible. MoE FFN parameters dominate the model budget — 297.5M of the 411.6M total is **~72%**.

### Active compute per token (MoE layer)

Each token executes:

- 4 routed SwiGLU experts at width 384
- 1 shared SwiGLU expert at width 384
- 1 gate matmul: `(D,) × (N, D)^T`

Effective FFN FLOPs ≈ `5 × 6 × D × I = 5 × 6 × 768 × 384 ≈ 8.8M` per MoE layer, versus `6 × 768 × 1536 ≈ 7.1M` for a dense layer. MoE is slightly more compute per token but provides ~5× the FFN parameter capacity.

---

## Implementation in This Repo

Source files:

| File | Role |
|---|---|
| `models/moe.py` | `AuxLossFreeGate`, `Expert`, `DeepSeekMoE` |
| `models/moe_triton.py` | Fused grouped-GEMM kernel + PyTorch reference |
| `models/_triton_dispatch.py` | `ENABLE_TRITON_KERNELS` env-var guard |
| `training/pretrain.py` | Bias update scheduling, balance metric logging |

### Class Structure

```
DeepSeekMoE
├── gate: AuxLossFreeGate
│   ├── weight: Parameter (N, D)
│   └── bias: Buffer (N,)  fp32
├── experts: ModuleList[Expert]  × N routed
├── shared_experts: ModuleList[Expert]  × N_s
├── moe_dispatch: str  ("stacked" | "triton_grouped")
├── _stacked_w{1,2,3}: Tensor (E, ...)  rebuilt each forward
├── _shared_w{1,2,3}: Tensor (E, ...)   rebuilt each forward
├── _last_weights: Tensor (T, k)  detached routing cache
└── _last_indices: Tensor (T, k)  detached routing cache
```

### AuxLossFreeGate Forward Pass

The gate is the smallest interesting piece of the whole MoE, so read it line by line (`models/moe.py:AuxLossFreeGate.forward`):

```python
def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    T = x.size(0)
    scores = F.linear(x, self.weight).sigmoid()
    biased = scores + self.bias.to(scores.dtype)
    indices = biased.topk(self.topk, dim=-1)[1]
    weights = scores.gather(1, indices)
    weights = (weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-10) * self.route_scale).to(x.dtype)
    return weights, indices
```

**Inputs:** `x` of shape `(T, D)` where `T = batch × seq` (the MoE flattens `(B, S, D)` before gating).

**Line 1 — `scores = F.linear(x, self.weight).sigmoid()`.** `weight` is a bare `nn.Parameter` of shape `(N, D)`, not an `nn.Linear`, so the code calls `F.linear(x, weight)` explicitly, which computes `x @ weightᵀ` → `(T, N)` logits, then applies the sigmoid elementwise. The result is `N` *independent* per-expert scores in `(0, 1)` — independence is the property that distinguishes this gate from softmax routing (see the final section).

Where do the logits live early in training? The gate weight is initialized `nn.init.normal_(self.weight, std=0.006)` (`models/moe.py:AuxLossFreeGate.__init__`), so a logit is a sum of `D = 768` i.i.d. terms of std 0.006 → logit std ≈ `0.006·√768 ≈ 0.166`. The sigmoid's linear region spans roughly `|z| ≲ 0.5`, so early scores cluster tightly around `σ(0) = 0.5` — nearly equal weights for every expert. Specialization (scores spreading toward 0 and 1) emerges only as the gate weight grows during training. This is also the answer to "why doesn't the sigmoid saturate at init": the init is deliberately small so the gate operates in the sigmoid's linear band.

**Line 2 — `biased = scores + self.bias.to(scores.dtype)`.** The bias buffer is stored FP32 (accumulation stability) and cast to the *scores* dtype before adding. Under autocast, `x` is BF16 and `scores` ends up BF16, so the cast keeps the add in a single dtype. The bias values themselves are tiny (`±0.001` per update, rarely exceeding ±0.1), so the cast loses nothing.

**Line 3 — `indices = biased.topk(self.topk, dim=-1)[1]`.** Selection only. `topk` returns `(values, indices)`; the values are discarded. Only the *order* of `biased` matters. Note what this means for gradients: `topk`'s indices are non-differentiable, and `biased`'s values are discarded here, so **`biased` is provably outside the autograd graph** — no path from `bias` to the loss exists, by construction.

**Line 4 — `weights = scores.gather(1, indices)`.** The raw sigmoid scores at the selected positions — *not* the biased scores. Shape `(T, k)`. This is the load-bearing decoupling: selection used `biased`, weights use `scores`.

**Line 5 — normalization.** `weights.sum(dim=-1, keepdim=True)` sums the `k` selected raw scores per token; `.clamp(min=1e-10)` guards the division; then `* self.route_scale` (canonical `1.0`) scales without re-normalizing. Why the clamp? Sigmoid outputs are strictly positive, but under BF16 a score can underflow to exactly `0.0`, and `k` zero-scores would give a divide-by-zero (NaN that then poisons the whole layer). The clamp is a two-instruction insurance policy against a whole class of NaN bugs.

**Line 6 — `.to(x.dtype)`.** Forces the returned weights to the activation dtype (BF16 under autocast), so the subsequent `out * weights` multiply does not silently promote to FP32.

**Outputs:**
- `weights`: `(T, k)` — normalised routing weights, sum to `route_scale` (default 1.0) per token
- `indices`: `(T, k)` — expert IDs in `[0, N)`

### DeepSeekMoE Forward Pass

High-level flow (`models/moe.py:DeepSeekMoE.forward`):

```
x (B, S, D)
  │
  ├─ flatten → flat (T, D)          T = B·S
  │
  ├─ gate(flat) → weights (T, k), indices (T, k)
  │     └─ stash detached copies in _last_weights/_last_indices
  │
  ├─ stack expert weights → _stacked_w{1,2,3}  (E, I, D) / (E, D, I)
  │
  ├─ routed forward (stacked or triton_grouped) → y_routed (T, D)
  │
  ├─ shared forward → y_shared (T, D)
  │
  ├─ y = y_routed + y_shared
  │
  └─ reshape → (B, S, D)
```

Two details are easy to miss:

1. **The gate output is stashed detached** (`self._last_weights = weights.detach()`) for the balance metric and the bias update, while the *live* `weights`/`indices` carry gradients into the dispatch. The Triton path's docstring says this explicitly: "`weights`/`indices` carry grad to the gate (unlike the detached `_last_*` snapshots used only for the balance metric / bias update)."
2. **Weights are re-stacked every forward** (`torch.stack([ex.w1.weight for ex in self.experts], dim=0)`). This is load-bearing: caching stacked weights across steps left stale copies after `optimizer.step()` (regression test: `test_stacked_weights_refresh_after_optimizer_step`). The stack is a view-copy of the parameters into one `(E, I, D)` tensor, moved to `flat.device`/`flat.dtype` — under autocast this is also what gets the BF16 copies into one contiguous buffer for the kernels.

### Sorted-Token Dispatch Layout

Both dispatch paths share the same **sort-by-expert** preprocessing. Given `indices (T, k)`:

```python
flat_idx   = indices.reshape(-1)           # (T·k,)  expert ID per slot
flat_w     = weights.reshape(-1)           # (T·k,)  weight per slot
token_id   = arange(T).repeat_interleave(k)  # which token each slot belongs to

order            = argsort(flat_idx)       # group slots by expert
sorted_token_ids = token_id[order]
sorted_weights   = flat_w[order]

expert_counts  = bincount(flat_idx, minlength=N)
expert_offsets = cumsum(counts)[:-1]       # start index per expert in sorted layout
```

**Why sort?** After sorting, all tokens assigned to expert `e` are contiguous in memory. The expert's SwiGLU runs as one batched GEMM over `cnt` tokens instead of `cnt` separate single-token calls.

Example with `T=3, k=2, N=4`:

```
Token 0 → experts [1, 3]   weights [0.6, 0.4]
Token 1 → experts [1, 2]   weights [0.5, 0.5]
Token 2 → experts [0, 1]   weights [0.7, 0.3]

flat_idx = [1, 3, 1, 2, 0, 1]
After argsort:
  expert 0: token 2 (w=0.7)
  expert 1: tokens 0, 1, 2 (w=0.6, 0.5, 0.3)
  expert 2: token 1 (w=0.5)
  expert 3: token 0 (w=0.4)
```

Read the layout data structures the way the kernels do:

- `expert_counts[e]` — number of slots assigned to expert `e` (the histogram of Section 5).
- `expert_offsets[e]` — the start index of expert `e`'s contiguous block in the sorted layout; `counts[e]` elements follow. Formed as `cat([0], cumsum(counts)[:-1])`, so `offsets[e] = Σ_{i<e} counts[i]`. (The stacked path uses the `(E,)` form; the PyTorch reference `grouped_moe_pytorch` uses an `(E+1,)` form with the end boundary included — same information, two conventions; do not confuse them when reading code.)
- `sorted_token_ids` — the token each slot belongs to, in expert order; this is what lets the results scatter back with `index_add_`.

A token appears in the sorted layout exactly `k` times (once per selected expert), so the scatter-back **must accumulate** (add), never assign: `y_routed[t]` is the sum of `k` expert outputs.

### Stacked Dispatch Path

Default: `moe_dispatch = "stacked"` (`models/moe.py:DeepSeekMoE._routed_forward_stacked`):

```python
y_routed = torch.zeros_like(flat)
for e in range(self.n_routed_experts):
    cnt = counts_cpu[e]
    if cnt == 0:
        continue
    start = offsets_cpu[e]
    end = start + cnt
    chunk_tokens = sorted_token_ids[start:end]
    chunk_weights = sorted_weights[start:end]
    expert_in = flat[chunk_tokens]
    gate = expert_in @ self._stacked_w1[e].t()
    up = expert_in @ self._stacked_w3[e].t()
    h = torch.nn.functional.silu(gate) * up
    out = h @ self._stacked_w2[e].t()
    y_routed = y_routed.index_add(0, chunk_tokens, out * chunk_weights.unsqueeze(-1))
return y_routed
```

Walk through the mechanics:

1. **The Python loop is over experts, not tokens.** Each iteration handles *all* of one expert's tokens as a single batched GEMM: `expert_in` is `(cnt, D)`, and the three matmuls produce `(cnt, I)`, `(cnt, I)`, `(cnt, D)`. With perfect balance, `cnt ≈ 3,277` — a perfectly sized batched GEMM. With collapse, one expert's chunk is huge and 19 loops are near-empty.
2. **The GEMMs are plain `@` with transposed stacked weights.** `self._stacked_w1[e]` is expert `e`'s `(I, D)` weight; `.t()` makes it `(D, I)` for `expert_in @ w1ᵀ`.
3. **`index_add(0, chunk_tokens, ...)` accumulates into `y_routed`.** This is the scatter-back that handles duplicate token ids (the `k` slots per token). `torch.index_add` is differentiable and deterministic in accumulation order — important because the *same* token's `k` contributions must be summed, and any expert's output is attributed back to its original row via `chunk_tokens`.
4. **The weight multiply is outside any custom op**: `out * chunk_weights.unsqueeze(-1)` is plain autograd (see Appendix B for why this placement matters).
5. **The loop reads counts/offsets from CPU** (`counts_cpu = expert_counts.tolist()`). With only `N = 20` entries this is a cheap host sync *per MoE layer* — 16 layers per forward, 4 micro-steps per optimizer step → ~64 tiny host-device round trips per step. Not the dominant cost (the GEMM launches are), but one of the reasons the fused path exists.

**Characteristics:**
- One Python `for e in range(E)` loop per MoE layer (E=20)
- 3 GEMMs per non-empty expert → up to 60 GEMM launches per layer
- 16 MoE layers → up to 960 expert GEMM launches per training step
- Always available (CPU, Mac, no Triton dependency)
- Correctness reference for the Triton path

**Weight stacking:** Expert `nn.Linear` weights are stacked into `_stacked_w{1,2,3}` tensors **every forward pass**. This is load-bearing: caching stacked weights across steps left stale copies after `optimizer.step()` (regression test: `test_stacked_weights_refresh_after_optimizer_step`).

### Triton Grouped Dispatch Path

Opt-in: `moe_dispatch = "triton_grouped"` + `ENABLE_TRITON_KERNELS=1`. The entry point is `models/moe_triton.py:triton_grouped_moe_dispatch`, invoked from `models/moe.py:DeepSeekMoE._routed_forward_triton`:

```python
x_sorted = flat[sorted_token_ids].contiguous()
y_sorted = triton_grouped_moe_dispatch(
    x_sorted=x_sorted,
    w1=self._stacked_w1,
    w2=self._stacked_w2,
    w3=self._stacked_w3,
    sorted_weights=sorted_weights_1d,
    expert_offsets=expert_offsets,
)
y_routed = torch.zeros_like(flat)
y_routed.index_add_(0, sorted_token_ids, y_sorted)
```

The Triton kernel fuses the per-expert SwiGLU loop into one launch over the sorted layout. See [Appendix B](#appendix-b--the-triton-grouped-gemm-path).

#### The register cap and why the canonical config cannot use this kernel

The kernel sizes its tiles as `BLOCK_I = next_pow2(I)`, `BLOCK_D = next_pow2(D)` with `BLOCK_T = 32`, and `models/moe_triton.py:_check_dim_limits` hard-fails when either exceeds 256:

```
ValueError: triton_grouped_moe_dispatch: BLOCK_I=ceil_pow2(I) and
BLOCK_D=ceil_pow2(D) must each be ≤ 256. Got I=..., D=...
```

The canonical config has `I = 384` → `BLOCK_I = 512` and `D = 768` → `BLOCK_D = 1024`, both over the cap. `triton_grouped_moe_dispatch` therefore raises `ValueError`, which `DeepSeekMoE.forward` catches and converts into a one-time fallback to `stacked`:

```python
except (ImportError, ValueError) as exc:
    if not getattr(self, "_triton_fallback_warned", False):
        print(f"[moe] triton_grouped unavailable ({type(exc).__name__}: {exc}); "
              f"falling back to 'stacked' for this model.")
        self._triton_fallback_warned = True
    y_routed = self._routed_forward_stacked(flat, indices, weights)
```

There are *two* independent gates before any kernel runs, and they must not be confused:

1. **Construction-time force-back** — `models/_triton_dispatch.py:enforce_triton_env_var` rewrites `moe_dispatch: "triton_grouped"` → `"stacked"` (with a single warning) unless `ENABLE_TRITON_KERNELS=1`, at `Transformer.__init__` and `Pretrainer.__init__`. This catches the "I asked for triton but forgot the env var" misconfiguration.
2. **Runtime fallback** — `_check_dim_limits`'s `ValueError` (or `ImportError` if triton is missing) inside `_routed_forward_triton`, caught by `DeepSeekMoE.forward`, one warning per model instance, silent afterwards.

**Why a 256 cap? Register pressure.** A CUDA thread has 255 registers, and Triton must fit every live tile of the kernel in them (anything beyond spills to local memory, which destroys throughput). With `num_warps=4` → 128 threads per program, a single FP32 accumulator of shape `(BLOCK_T, BLOCK_I) = (32, 256)` holds `8,192` values — 64 registers per thread. The forward kernel needs *two* such accumulators (`gate_acc`, `up_acc`) → 128 registers per thread before any operand tile is even loaded; the `w1` tile alone `(256, 256)` BF16 is another 128 registers per thread. The kernel at the 256/256 cap is already at the edge of the register file and relies on Triton's spilling; doubling `BLOCK_I` to 512 would put the weight tile alone over the entire 255-register budget. Hence the design decision: hard-reject rather than ship a kernel that spills catastrophically. (The arithmetic here is a [derived] estimate of what the compiler must allocate; the `ValueError` itself is the measured, enforced behavior.)

**Honest framing of the kernel's current shape (a design constraint, not an implementation accident):**

| Constraint | Consequence |
|---|---|
| `BLOCK_I = next_pow2(I)` is materialized whole in the forward and dx kernels | The forward and dx kernels **cannot tile I**: `i_idx = tl.arange(0, BLOCK_I)` is a full tile and the only tiling loop is over `D`. Supporting `I = 384` would require restructuring the kernels to loop over `i_start`, not just raising the cap. |
| The backward-dw kernel *does* loop over both `i_start` and `d_start`, but its `dh = tl.dot(dy_tile, w2_tile)` is computed per `(i_start, d_start)` tile **without accumulating across `d_start`** | At `D ≤ 256` there is exactly one `D`-block, so `dh` is correct today — but the accumulation is single-`D`-block by construction. If `D`-tiling were ever enabled (e.g. by raising the cap), `dh` would silently reflect only the last `D`-block. This is a **latent correctness bug** that is currently unreachable, precisely because `_check_dim_limits` enforces the single-block regime. Any future cap relaxation must fix `dh` first. |
| `BLOCK_T = 32` token-blocking is the only dimension that tiles `T` | Fine for large `cnt` per expert; small experts run masked, wasting lanes — acceptable since counts cluster near 3,277 at balance. |

The bottom line, stated plainly: **the fused kernel is valid only at smoke-config dimensions** (`I, D ≤ 256`), and the ≥1.5× MoE A100 speedup target is structurally blocked at canonical dims (I=384, D=768) until the kernel gains I-tiling and a corrected multi-D-block `dh`. The fallback is not a workaround — it is the designed behavior of the current kernel boundary, and `stacked` remains the correctness reference. See [[Docs/12_Triton_Kernels|Triton Kernels]] for the kernel design and the roadmap.

**Training-only:** The Triton MoE kernel is not validated for inference decode (sparse per-step routing shapes differ from training's batched layout). Use `stacked` for inference.

### Shared Expert Path

The shared expert runs on every token and needs no sorting (`models/moe.py:DeepSeekMoE._shared_forward`):

```python
E = self.n_shared_experts
gate = torch.bmm(flat.unsqueeze(0).expand(E, -1, -1), self._shared_w1.transpose(-1, -2))
up = torch.bmm(flat.unsqueeze(0).expand(E, -1, -1), self._shared_w3.transpose(-1, -2))
h = torch.nn.functional.silu(gate) * up
out = torch.bmm(h, self._shared_w2.transpose(-1, -2))
return out.sum(dim=0)   # (T, D)
```

Deconstruct:

1. **`flat.unsqueeze(0).expand(E, -1, -1)`** turns `(T, D)` into a virtual `(E, T, D)` batch via stride-0 expansion — no memory is copied. For `E = 1` this is exactly one matmul with a batch dimension of 1, which cuBLAS executes as a plain GEMM.
2. **`self._shared_w1.transpose(-1, -2)`** — the stacked shared weight `(E, I, D)` becomes `(E, D, I)` so `bmm` computes `(E, T, D) @ (E, D, I) → (E, T, I)`. Same for `w3`; `w2` runs `(E, T, I) @ (E, I, D) → (E, T, D)`.
3. **`out.sum(dim=0)`** merges the `E` shared experts — with `N_s = 1` it is a no-op squeeze, and with `n_shared_experts = 0` the method returns `torch.zeros_like(flat)` early (supported but not recommended, per FAQ Q4).
4. **Re-stack every forward**, exactly like the routed path — the same staleness bug class applies to `_shared_w*`.

The shared path is a small fraction of MoE compute (~1 shared vs 4 routed per token, i.e. ~1/5 of the active FFN work) but it is the *most frequently updated* FFN in the model — every token backprops through it every layer (see Appendix C).

### Training Integration

In `training/pretrain.py`, two hooks touch the MoE. The balance metric is computed inside the autocast region of `Pretrainer.train_step` (`training/pretrain.py:Pretrainer._moe_balance_metric`):

```python
def _moe_balance_metric(self) -> torch.Tensor:
    """Return the on-device balance loss tensor — .item() is deferred to the logger path (avoid per-step sync)."""
    losses = [moe.get_load_balance_loss() for moe in self.raw_model.moe_layers()]
    if not losses:
        return torch.tensor(0.0, device=self.device)
    return torch.stack(losses).sum()
```

`Transformer.moe_layers` (`models/transformer.py:Transformer.moe_layers`) yields exactly the 16 `DeepSeekMoE` instances:

```python
def moe_layers(self):
    for layer in self.layers:
        if isinstance(layer.ffn, DeepSeekMoE):
            yield layer.ffn
```

The bias update is scheduled on optimizer steps in `training/pretrain.py:Pretrainer.train_step`, after `optimizer.step()` / `scheduler.step()` / `zero_grad()`:

```python
self._opt_steps += 1
if self._opt_steps % self.config.bias_update_every == 0:
    self._update_moe_bias()
```

which fans out through `training/pretrain.py:Pretrainer._update_moe_bias` → `DeepSeekMoE.update_gate_bias` → `AuxLossFreeGate.update_bias` (Section 7). Note the ordering: the counts consumed are from the forward that just completed, and the updated bias applies to the *next* forward — a one-step-lookbehind feedback loop, consistent with Section 7's model.

Config keys (`configs/pretrain_a100_422m.yaml`):

```yaml
bias_update_speed:  0.001
bias_update_every:  1        # update bias every optimizer step
moe_dispatch:       "stacked" # canonical; triton opt-in
```

The bias update uses routing counts from the **last forward** (`_last_indices`), not an accumulated window. With `bias_update_every = 1`, each optimizer step's routing distribution directly drives the next step's bias.

---

## Comparison: MoE vs Dense FFN

| Aspect | Dense SwiGLU (layers 0–1) | DeepSeekMoE (layers 2–17) |
|---|---|---|
| Width | `inter_dim = 1536` | `moe_inter_dim = 384` per expert |
| Params/layer | 3.5M | 18.6M |
| Active experts/token | 1 (always) | 4 routed + 1 shared |
| Routing | none | sigmoid gate + top-4 |
| Load balancing | N/A | aux-loss-free bias |
| Dispatch | single matmul stack | sorted-token + per-expert GEMM |
| Parameter efficiency | baseline | ~5× capacity per layer |
| Compute/token | 7.1M FLOPs | ~8.8M FLOPs |

---

## Performance Characteristics

### Bottleneck analysis (canonical config, B=8, S=2048, 16 MoE layers)

Estimated step-time share (profiling estimates — not yet measured on GPU):

| Component | Step-time share | Notes |
|---|---|---|
| MLA attention (SDPA) | ~35–40% | FA2-accelerated |
| MoE dispatch loop | ~25–30% | 20 experts × 3 GEMMs × 16 layers |
| Dense FFN (2 layers) | ~5% | Small fraction |
| Embedding + LM head | ~10% | Vocab 100K |
| MTP auxiliary | ~5% | depth=1 |
| Other (norm, gate, etc.) | ~10% | |

The MoE per-expert Python loop is the **second-largest hotspot** after attention. The Triton grouped kernel targets ≥1.5× speedup on the MoE path (see `AGENTS.md` §Hard rules). Two structural costs make the Python loop slow on GPU, beyond the raw GEMM count: (a) up to 60 small-GEMM launches per layer where each launch pays fixed driver overhead (typically microseconds, which matters when the GEMM itself runs in tens of microseconds at these widths), and (b) the per-layer host-device sync from `counts_cpu`/`offsets_cpu = .tolist()`, which serializes the loop with the GPU. The fused kernel removes both: one launch per layer, everything on device. (Both costs are [INFERENCE]-grade estimates of *why* the loop is slow; the 25–30% share itself is unmeasured until a GPU run exists.)

### Memory

Per MoE layer (BF16 weights):

```
Routed: 20 × 3 × 768 × 384 × 2 bytes ≈ 35.4 MB
Shared:  1 × 3 × 768 × 384 × 2 bytes ≈  1.8 MB
Gate:    20 × 768 × 2 bytes             ≈  0.03 MB
× 16 layers ≈ 595 MB MoE FFN weights
```

Activations scale with `T × k × I` for the intermediate SwiGLU state inside each expert chunk. Gradient checkpointing (`grad_checkpoint: true`) recomputes these during backward, trading compute for memory.

---

## Appendix A — A worked numerical example

**Config:** `D=4, I=2, N=3, k=2, N_s=1`. One token, hidden `h = [1, 0, 1, 0]`.

### Gate

```
W_g (3×4):                bias: [0, 0.5, -0.5]
  [0.1, 0.2, 0.1, 0.2]     Expert 0: unbiased
  [0.3, 0.1, 0.3, 0.1]     Expert 1: +0.5 (slightly favoured)
  [0.1, 0.3, 0.1, 0.3]     Expert 2: -0.5 (slightly penalised)

scores = σ(h @ W_g^T) = σ([0.2, 0.5, 0.4]) ≈ [0.55, 0.62, 0.60]
biased = [0.55, 1.12, 0.10]
top-2 indices = [1, 2]  (biased scores 1.12 > 0.10; expert 0 misses despite score 0.55)
weights = scores[[1,2]] / sum = [0.62, 0.60] / 1.22 ≈ [0.508, 0.492]
```

Notice expert 0 had the *second-highest raw score* (0.55) but lost top-2 because bias shifted selection. Its weight is irrelevant because it wasn't selected.

### Expert output (simplified)

Suppose `f_1(h) = [0.5, 0.5, 0.5, 0.5]` and `f_2(h) = [1.0, 0.0, 1.0, 0.0]`:

```
y_routed = 0.508 · f_1 + 0.492 · f_2
         ≈ [0.754, 0.254, 0.754, 0.254]

y_shared = f_s(h) = [0.2, 0.2, 0.2, 0.2]  (always on)

y = y_routed + y_shared ≈ [0.954, 0.454, 0.954, 0.454]
```

---

## Appendix B — The Triton grouped-GEMM path

**File:** `models/moe_triton.py`
**Replaces:** the `for e in range(E)` loop in `_routed_forward_stacked`.

### Design

One Triton kernel launch per MoE layer. Grid: `(E, ceil(T / BLOCK_T))` — one program per (expert, token-block) pair.

**Forward (per program):**

1. Load `expert_offsets[e]` → `[start, end)` token range in sorted layout
2. Tile over input dim `D` in `BLOCK_D` chunks, accumulate `gate` and `up` in FP32
3. `h = silu(gate) * up`
4. Down-project: `out = h @ w2^T`
5. Store unweighted `out` to `y_sorted`

The gate-weight multiply happens **outside** the autograd Function:

```python
out = _TritonGroupedMoeFunction.apply(x_sorted, w1, w2, w3, expert_offsets)
return out * sorted_weights.unsqueeze(-1)
```

This ensures gradients flow to the gate weights through the routing-weight multiply.

### Why the weight multiply lives outside the autograd Function

This placement is a deliberate gradient-routing decision, and it is worth deriving exactly what happens in both designs. The Function's `backward` returns one gradient per *input* it was given:

```python
return (dx, dw1.to(w1.dtype), dw2.to(w2.dtype), dw3.to(w3.dtype), None)
```

`x_sorted, w1, w2, w3` get gradients; `expert_offsets` gets `None` (it is integer data, non-differentiable). `sorted_weights` is **not** an input at all. If the multiply were inside the Function, the backward would need to produce a sixth gradient for `sorted_weights`, thread it through the kernel or an extra autograd node, and save it for backward — more code and more saved state for zero benefit. Instead, the wrapper returns `out * sorted_weights.unsqueeze(-1)`, an ordinary autograd multiply, so the chain

```
y_routed[t] = Σ_j w_{t,j} · f_j(h_t)          (eq. 6)
```

is differentiated by PyTorch's own `MulBackward`/`index_add` machinery:

```
dL/dw_{t,j} = ⟨ f_j(h_t), dL/dy_routed[t] ⟩    (inner product over D)
```

and from there the gradient flows `w → normalization → gathered scores → σ(x W_gᵀ) → W_g` entirely through native ops (`gather`, `div`, `sigmoid`, `F.linear`) in `AuxLossFreeGate.forward`. The kernel never has to know routing weights exist: its backward receives `dy` *already scaled* by the weight (autograd applies the chain rule for the outside multiply), which is exactly why the dx kernel's comment says "dy arrives already scaled by the gate weight (autograd handles the outside multiply), so no gw load here."

The same reasoning applies to the stacked path, where `out * chunk_weights.unsqueeze(-1)` and the `index_add` scatter are also native autograd ops. Both dispatch paths therefore give the gate an identical gradient path — one of the invariants verified by `torch.autograd.gradcheck` on float32 tiny configs.

### Backward (FA2 re-compute pattern)

Forward saves: `x_sorted, w1, w2, w3, expert_offsets` (minimal).

Backward re-computes `gate`, `up`, `h` on the fly:

- `_grouped_moe_bwd_dx_kernel` — gradient w.r.t. `x_sorted`
- `_grouped_moe_bwd_dw_kernel` — gradient w.r.t. `w1, w2, w3` (one program per expert, no atomics)

Weight gradients accumulate in FP32, cast back to BF16 before return.

### PyTorch reference

`grouped_moe_pytorch()` (`models/moe_triton.py:grouped_moe_pytorch`) in the same file implements identical arithmetic for CPU tests. `tests/test_moe_triton.py` asserts Triton ≈ reference within `atol=1e-2` for BF16.

### Env-var guard

```python
# models/_triton_dispatch.py
if os.environ.get("ENABLE_TRITON_KERNELS", "0") != "1":
  moe_dispatch "triton_grouped" → force-backed to "stacked"
```

Called at `Transformer.__init__` and `Pretrainer.__init__` via `models/_triton_dispatch.py:enforce_triton_env_var`. One warning at construction, not per layer.

---

## Appendix C — Gradient flow in MoE

### Gate path

```
loss → y_routed → Σ w_i · f_i(h) → w_i = scores.gather(indices) / sum(scores)
                                              ↑
                                         scores = σ(h @ W_g^T)
                                              ↑
                                         W_g receives gradients
```

The bias `b` is **not** in this graph. It is updated only via `update_bias()` in Python.

Derive the two gradient legs explicitly. The routed contribution is

```
y_routed = Σ_{t} Σ_{j∈ℐ_t} w_{t,j} f_j(h_t)
```

**Leg 1 — through the expert weights.** For a fixed selected pair `(t, j)`:

```
∂L/∂W_{1,j}  ←  ∂L/∂y_routed[t] · w_{t,j} · ∂f_j/∂W_{1,j}      (ditto W2, W3)
```

Only the 4 routed experts per token receive gradients; the other 16 are idle for that token (no compute, no grad). Because `w_{t,j}` multiplies the expert's output *before* the chain reaches `f_j`'s weights, the routing weight acts as a per-token scaling of the expert's gradient — an expert demoted by a small weight learns less from that token, which is exactly the intended semantics of weighted routing.

**Leg 2 — through the gate.** The same products give the gradient for the routing weights:

```
∂L/∂w_{t,j} = ⟨ f_j(h_t), ∂L/∂y_routed[t] ⟩
```

then the normalization chain inside `AuxLossFreeGate.forward`:

```
w_{t,j} = s_{t,j} / Σ_{i∈ℐ_t} s_{t,i}
∂L/∂s_{t,j} = (1/Σs) · [∂L/∂w_{t,j}] − s_{t,j}·(Σ_i ∂L/∂w_{t,i}) / (Σs)²
```

and finally `∂L/∂W_g = xᵀ · (∂L/∂scores ⊙ σ′(x W_gᵀ))` via the sigmoid's derivative `σ′(z) = σ(z)(1−σ(z))`. Every step is native autograd in both dispatch paths (Appendix B), so the stacked and Triton paths produce identical gate gradients — verified by `torch.autograd.gradcheck` on float32 tiny configs in `test_moe_triton.py`.

**Where the bias sits:** `bias` enters only through `biased`, whose values are discarded by `topk` (only the index order is used, and index selection is non-differentiable). There is no path from `bias` to the loss. This is a *structural* property of the code, not a convention.

### Expert path

```
loss → y_routed → f_i(h) → w1, w2, w3 of expert i
```

Only the 4 routed experts per token receive gradients. The other 16 experts are idle for that token (no compute, no grad).

### Shared expert path

```
loss → y_shared → f_s(h) → shared w1, w2, w3
```

Every token backprops through the shared expert — it trains ~T/k times more frequently than any single routed expert. Concretely, over one forward the shared expert receives `T = 16,384` token-gradients while a routed expert receives its `count_e ≈ 3,277` — the shared expert's weights move on every token and converge to the "common-pattern" solution fastest; the routed experts specialize on the residual. This is the structural reason shared experts exist (Section 3).

### Stacked vs Triton gradient equivalence

Both paths must produce identical gradients (verified by `torch.autograd.gradcheck` on float32 tiny configs in `test_moe_triton.py`). The Triton path's re-compute backward must match the PyTorch autograd graph of the stacked loop.

---

## Appendix D — Load-balance metrics

### `get_load_balance_loss()` — logging surrogate

Full source (`models/moe.py:DeepSeekMoE.get_load_balance_loss`):

```python
def get_load_balance_loss(self) -> torch.Tensor:
    if self._last_weights is None or self._last_indices is None:
        return torch.tensor(0.0, device=self.gate.weight.device)
    weights = self._last_weights
    indices = self._last_indices
    T = weights.size(0)
    counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).float()
    f = counts / counts.sum().clamp(min=1e-10)
    one_hot = F.one_hot(indices.flatten(), num_classes=self.n_routed_experts).float()
    P = (one_hot * weights.flatten().unsqueeze(-1)).view(T, -1, self.n_routed_experts).sum(dim=1).mean(dim=0)
    return (f * P).sum() * self.n_routed_experts
```

**Derivation.** Let `slots = T·k` be the total routing slots, `1_e(·)` the indicator of expert `e`, and `w_{t,j}` the (normalized) weight of token `t`'s `j`-th selected expert.

1. **Load fraction:** `f_e = count_e / slots = (1/slots) Σ_{t,j} 1_e(ℐ_{t,j})` — the empirical selection frequency, computed by `bincount` + normalize. Note the `.clamp(min=1e-10)` on the sum: it is a divide-by-zero guard for the degenerate empty-forward case.

2. **Mean routing probability:** the paper's `P_e` is the mean probability the gate assigns to expert `e` per token. With hard top-*k* selection, the natural proxy is the mean *weight* mass on `e`:

```
P_e = (1/T) Σ_t  Σ_{j: ℐ_{t,j}=e} w_{t,j}
```

The implementation computes this with a one-hot trick: `one_hot` is `(slots, N)` with a 1 in the selected column of each slot; multiplying by `weights.flatten().unsqueeze(-1)` leaves the slot's weight in exactly that column; reshaping to `(T, k, N)` and summing over `k` gives per-token per-expert weight mass; `.mean(dim=0)` averages over tokens → `P`. Since the weights are normalized (`Σ_j w_{t,j} = route_scale = 1` per token), `Σ_e P_e = 1` exactly.

3. **The surrogate:** `L_bal = N · Σ_e f_e · P_e = N·⟨f, P⟩` — the Switch Transformer auxiliary-loss formula, computed from the **detached** routing cache.

**Interpretation.** With both `f` and `P` normalized distributions:

- **Perfectly balanced, decorrelated routing:** `f_e = P_e = 1/N` → `⟨f,P⟩ = 1/N` → `L_bal = 1.0`. (This is the "healthy" value; drift above it is drift toward imbalance.)
- **Collapse (all tokens to expert 0, weight 1):** `f_0 = P_0 = 1` → `⟨f,P⟩ = 1` → `L_bal = N = 20`. This is the collapse alarm ceiling.
- **General range:** Cauchy–Schwarz gives `⟨f,P⟩ ≤ ‖f‖·‖P‖ ≤ 1`, so `L_bal ≤ N` always. The lower bound is *not* 1: if `f` and `P` anti-correlate (e.g. `f = (0.5, 0.5, 0)`, `P = (0, 0.5, 0.5)`), `⟨f,P⟩ = 0.25` and `L_bal = 0.75 < 1`. In this repo `f` and `P` come from the *same* routing decisions, so they move together and the practically relevant range is `[1, N]`; but treat the metric as a **relative drift monitor**, not an absolute score — which is all it is used for, since it never enters the loss.

**Why detached caches:** `DeepSeekMoE.forward` stashes `_last_weights = weights.detach()` and `_last_indices = indices.detach()`. If the metric consumed live tensors *and* were added to the loss, it would inject exactly the aux-loss gradient Section 8 argues against; the detach makes it structurally impossible for `get_load_balance_loss` to backprop — it is observability only. (Before the first forward the caches are `None` and the method returns `0.0` rather than crashing, which keeps the training-loop `torch.stack` over 16 layers safe at step 0.)

**Interpretation during training:**

| Metric | Healthy range | Alarm signal |
|---|---|---|
| `balance_loss` | 1.0–2.0 (slow drift) | Sustained > 5.0 → routing collapse |
| Per-expert counts (manual) | Within ±20% of mean | Any expert at 0 for many steps |
| `gate.bias` magnitude | Slowly growing, bounded | Runaway bias → all experts tied |

---

## Appendix E — Bias-update lifecycle (state diagram)

```
┌─────────────────────┐
│  Model.__init__     │
│  bias = zeros(N)    │
└──────────┬──────────┘
           │ forward pass
           ▼
┌──────────────────────────────────────────────┐
│  gate(flat) → weights, indices             │
│  _last_weights = weights.detach()            │
│  _last_indices = indices.detach()            │
│  ... expert dispatch ...                     │
└──────────┬───────────────────────────────────┘
           │ optimizer.step() (every N micro-steps)
           ▼
┌──────────────────────────────────────────────┐
│  if opt_steps % bias_update_every == 0:      │
│    counts = bincount(_last_indices)          │
│    gate.update_bias(counts, speed)  @no_grad │
└──────────┬───────────────────────────────────┘
           │ next forward
           ▼
┌──────────────────────────────────────────────┐
│  biased = scores + bias  (new bias applied)  │
│  different top-k selection                   │
└──────────────────────────────────────────────┘

Checkpoint save/load:
  bias in state_dict (buffer) → survives resume
  _last_* caches are NOT saved (re-populated on first forward)
```

**Common pitfalls:**

- Making `bias` a `Parameter` → optimizer overwrites load-balancing signal
- Calling `update_bias` inside the autograd graph → bias gets unintended gradients
- Using `biased` scores for `weights` → aux-loss contamination through the task loss
- Forgetting to call `update_bias` → slow expert collapse over thousands of steps
- `bias_update_every` too high with small batch → noisy counts, oscillating bias

---

## Appendix F — Numerical stability notes

| Concern | Mitigation in this repo |
|---|---|
| Sigmoid saturation | Gate init `std=0.006` keeps scores near 0.5 early in training |
| Weight normalisation divide-by-zero | `.clamp(min=1e-10)` on `weights.sum()` |
| Bias dtype mismatch | `bias.to(scores.dtype)` at forward; stored as FP32 |
| Bias update device mismatch | `counts` kept on bias device before boolean indexing (`models/moe.py:DeepSeekMoE.update_gate_bias`) |
| BF16 GEMM accumulation | PyTorch bmm accumulates FP32 on Ampere+; Triton uses explicit FP32 accumulators |
| Stale stacked weights | Re-stack every forward (not cached across steps) |
| Triton BF16 dot dtype | `h` cast to input dtype before `w2` dot in kernel |
| Metric f-normalization divide-by-zero | `.clamp(min=1e-10)` on `counts.sum()` in `get_load_balance_loss` |

Two Triton-side numerics worth spelling out:

1. **FP32 accumulators with BF16 operands.** Both `gate_acc` and `up_acc` are `tl.zeros((BLOCK_T, BLOCK_I), dtype=tl.float32)` and every `tl.dot` accumulates into them with `acc=`. The per-expert `gate` and `up` projections contract over `D = 768` in chunks of `BLOCK_D`; FP32 accumulation keeps the sum of 768 BF16 products from rounding into BF16 at every step.
2. **The `h` cast before the second GEMM.** `h = tl.sigmoid(gate_acc) * gate_acc * up_acc` is FP32, but the down-projection dot requires matching operand dtypes, so the kernel casts `h` back to the input dtype first (`h_typed = h.to(x_ptr.dtype.element_ty)`). The cast loses the FP32 precision of the SwiGLU intermediate — a deliberate trade of accuracy for a valid BF16·BF16 tensor-core dot (and the `atol=1e-2` reference test is what keeps this honest).

### `route_scale`

Default `1.0`. Multiplying normalised weights by `route_scale` scales the routed expert contribution without re-normalising. Values ≠ 1.0 change the relative magnitude of routed vs shared expert output (shared is not scaled). Not used in the canonical config.

---

## Appendix G — Glossary

| Symbol / Name | Meaning |
|---|---|
| `N` / `n_routed_experts` | Number of routed experts (20 in canonical config) |
| `k` / `n_activated_experts` | Top-*k* experts per token (4) |
| `N_s` / `n_shared_experts` | Shared experts, always active (1) |
| `D` / `dim` | Model hidden dimension (768) |
| `I` / `moe_inter_dim` | Expert intermediate width (384) |
| `W_g` / `gate.weight` | Gate projection `(N, D)` |
| `b` / `gate.bias` | Per-expert bias buffer `(N,)`, FP32 |
| `s_t` | Sigmoid gate scores (routing weights source) |
| `s̃_t` | Biased scores (top-*k* selection only) |
| `ℐ_t` | Set of selected expert indices for token `t` |
| `w_{t,i}` | Normalised routing weight for expert `i` |
| `f_e` | SwiGLU expert function |
| `T` | Total tokens in a forward pass (`B × S`) |
| `slots` | Routing slots per layer per forward (`T × k`) |
| `ρ` | Load imbalance / capacity factor (`max_e count_e / (slots/N)`) |
| `flat_idx` | Flattened expert assignments `(T·k,)` |
| `sorted_token_ids` | Token indices reordered by expert |
| `expert_offsets` | Cumulative boundaries in sorted layout |
| `moe_dispatch` | `"stacked"` (default) or `"triton_grouped"` |
| `bias_update_every` | Optimizer steps between bias updates |
| `bias_update_speed` | Step size for bias increment/decrement |
| `balance_loss` | Logged load-balance surrogate (not in training loss) |
| `route_scale` | Scalar multiplier on normalised routing weights |

---

## Appendix H — Frequently Asked Questions

### Q1. Why sigmoid instead of softmax for gating?

Softmax forces experts to compete: increasing one expert's probability decreases all others. Sigmoid treats each expert independently — a token can strongly activate multiple experts without suppressing the rest. This pairs naturally with top-*k* selection (pick the *k* highest, not sample from a distribution).

### Q2. Can I replace bias updates with an auxiliary loss?

**No.** This is a load-bearing invariant. The bias mechanism, sigmoid gate, and raw-score weighting are co-designed. Swapping in a standard aux loss silently breaks load balance in this codebase. If you need aux-loss routing (e.g., for comparison with GPT-OSS-Lite), fork the gate class — do not modify `AuxLossFreeGate` in place.

### Q3. Why is the Triton kernel disabled for the canonical config?

The kernel's `BLOCK_I` and `BLOCK_D` are `next_pow2(I)` and `next_pow2(D)`, capped at 256 for register pressure. The canonical config has `I=384, D=768`, both exceeding the cap. `triton_grouped` raises `ValueError` (`models/moe_triton.py:_check_dim_limits`) and falls back to `stacked`. Extending the kernel to larger dims is discussed in [[Docs/12_Triton_Kernels|Triton Kernels]]; the specific blockers (no I-tiling in fwd/dx, single-D-block `dh` in dw) are in Section 11.6.

### Q4. Does the shared expert hurt specialisation?

Empirically, no — it handles "universal" FFN patterns (common subword mappings, high-frequency syntax) so routed experts can specialise on rarer functions. The shared expert trains on every token, so it converges faster. Removing it (`n_shared_experts=0`) is supported in code but not recommended.

### Q5. How often should `bias_update_every` be set?

The canonical config uses `1` (every optimizer step). Larger values smooth the bias signal but react slower to routing drift. The default dataclass value is `10`; ensure your YAML overrides it. With gradient accumulation, bias updates happen on optimizer steps, not micro-steps.

### Q6. Why re-stack expert weights every forward?

A regression was found where cached `_stacked_w*` tensors pointed to pre-optimizer-step weight values, freezing experts at their init values. Re-stacking costs a small `torch.stack` per layer but guarantees correctness. The Triton path uses the same freshly stacked tensors.

### Q7. Is group-limited routing (DeepSeek-V3's expert groups) implemented?

Not in this repo. The full V3 model routes first to expert *groups*, then within groups. The canonical config uses a single group (`n_expert_groups=1` equivalent). Adding group routing would require extending `AuxLossFreeGate.forward`.

### Q8. How do I monitor expert utilisation during training?

```python
for i, moe in enumerate(model.moe_layers()):
    counts = torch.bincount(moe._last_indices.flatten(), minlength=moe.n_routed_experts)
    print(f"Layer {i+2}: {counts.tolist()}")
```

For production, log `balance_loss` from the training loop (already wired in `Pretrainer`).

---

## Check Your Understanding

**Q1. The gate computes `biased = scores + bias` for selection but `weights = scores.gather(indices)` for weighting. Why the asymmetry?**

*A1.* The bias is a load-balancing *actuator*: it must be able to change *which* experts win top-*k* (that is how it redistributes load). But it must not change *how much* each selected expert contributes, and it must not enter the autograd graph. Using biased values as weights would (a) scale expert outputs by a control signal the optimizer does not see, and (b) give the bias an implicit gradient path through the task loss. Using raw scores for weights keeps the bias provably outside the graph (its values are discarded by `topk`) while fully in charge of selection.

**Q2. The Triton kernel returns the *unweighted* expert output and the wrapper multiplies by `sorted_weights` outside the autograd Function. Why not inside?**

*A2.* The Function's `backward` returns gradients only for its inputs (`x_sorted, w1, w2, w3, expert_offsets`). `sorted_weights` is not an input, so an in-Function multiply would force the backward to manufacture a sixth gradient and thread it back through the kernel. Placing the multiply outside lets native autograd (`MulBackward`, `gather`, `index_add`) carry `dL/dw` to the gate: `dL/dw_{t,j} = ⟨f_j(h_t), dL/dy_routed[t]⟩`. The kernel's backward then receives `dy` already scaled and needs no knowledge of routing weights at all.

**Q3. All tokens route to expert 0 with weight 1.0 in every layer. What is `balance_loss` per layer?**

*A3.* `f_0 = 1`, `P_0 = 1`, everything else 0 → `⟨f,P⟩ = 1` → `L_bal = N·1 = 20`. That is the collapse ceiling; sustained values above ~5 are the alarm trigger in Appendix D.

**Q4. Can the canonical config (I=384, D=768) run the grouped Triton MoE kernel today? Why or why not?**

*A4.* No. `_check_dim_limits` rejects any `I` or `D` over 256 because `BLOCK_I = next_pow2(384) = 512` and `BLOCK_D = next_pow2(768) = 1024` would blow the register budget (the forward kernel already needs ~128 registers per thread for its two FP32 accumulators at the cap). `triton_grouped_moe_dispatch` raises `ValueError`, and `DeepSeekMoE.forward` falls back to `stacked` with a one-time warning. Extending the kernel needs I-tiling in the fwd/dx kernels and a multi-D-block `dh` accumulation fix in the dw kernel first — both currently absent by design, which is why the cap is enforced rather than raised.

---

## Load-Bearing Invariants (Do Not Break)

1. **Bias is a `buffer`, not a `Parameter`.** No autograd, but `state_dict()` persists it. Tests: `test_bias_not_in_parameters`, `test_bias_in_state_dict`.

2. **Bias updates are `@torch.no_grad()` and out-of-band.** Run by `Pretrainer._update_moe_bias` every `bias_update_every` optimizer steps. Never inside the autograd graph. The Triton kernel only fuses the routed-expert forward — bias updates stay in Python.

3. **Routing weights come from raw `sigmoid` scores, not `biased`.** The bias shifts top-*k* *selection*; weights are `scores.gather(indices)`. No bias gradient contaminates the task loss.

4. **Stacked weights are re-built every forward.** Do not cache `_stacked_w*` across optimizer steps.

5. **`moe_dispatch="triton_grouped"` requires `ENABLE_TRITON_KERNELS=1`.** Without it, `enforce_triton_env_var` force-backs to `"stacked"` at construction with a single warning. No silent per-layer fallback during training.

6. **Triton runtime fallback is explicit.** If Triton is missing or dims exceed 256, `_routed_forward_triton` raises and `forward()` catches it, falling back to `stacked` with a one-time warning per model instance. This is a construction-time misconfiguration signal, not a silent training-path switch.

7. **Default dispatch is `"stacked"` everywhere** — config, env-var guard, CI tests. CPU/Mac runs need no Triton dependency.

8. **Do not disable the NaN guard** to mask MoE numerical issues. If a kernel produces NaNs, fix the kernel before re-enabling.

---

## Implementation Checklist

When modifying MoE code, verify:

- [ ] `pytest tests/test_models.py -k "MoE or AuxLossFree or Expert"` passes
- [ ] `pytest tests/test_moe_triton.py` passes (CPU reference path always; GPU Triton if available)
- [ ] `pytest tests/test_force_back.py` passes (env-var guard)
- [ ] `test_bias_not_in_parameters` — bias not in `parameters()`
- [ ] `test_bias_in_state_dict` — bias survives checkpoint roundtrip
- [ ] `test_update_bias_sign_rule` — over/under/deadband behaviour
- [ ] `test_stacked_weights_refresh_after_optimizer_step` — no stale weight copies
- [ ] `test_moe_bias_update_during_training` — integration with `Pretrainer`
- [ ] Triton path (if touched): `atol=1e-2` vs `grouped_moe_pytorch` reference
- [ ] Triton path (if touched): `torch.autograd.gradcheck` on float32 tiny config
- [ ] No new magic numbers without named config keys

---

## References

- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — §2.3 MoE architecture, §2.3.3 auxiliary-loss-free balancing
- [DeepSeekMoE: Towards Ultimate Expert Specialization](https://arxiv.org/abs/2401.06066) — fine-grained expert decomposition, shared experts
- [Switch Transformers](https://arxiv.org/abs/2101.03961) — original auxiliary load-balancing loss formulation
- [GShard](https://arxiv.org/abs/2006.16668) — top-2 routing, capacity factors
- [Outrageously Large Neural Networks (Sparsely-Gated MoE)](https://arxiv.org/abs/1701.06538) — foundational MoE work
- `docs/12_Triton_Kernels.md` — Triton grouped-GEMM design and benchmarks
- `models/moe.py` — authoritative implementation
- `models/moe_triton.py` — Triton kernel + PyTorch reference

---

### Sigmoid vs Softmax Rationale

DeepSeek uses **sigmoid** gating instead of softmax for three reasons:

1. **Independent activation, not competition** — Softmax forces experts to compete for probability mass (zero-sum). Sigmoid treats each expert independently — a token can strongly activate multiple experts without suppressing others.
2. **Bias update meaningfulness** — Increasing `bias[e]` directly increases expert $e$'s chance of being selected, without decreasing other experts' chances. With softmax, boosting one expert necessarily suppresses all others.
3. **Natural top-k pairing** — The top-k mechanism picks the highest-scored experts; sigmoid probabilities shift ranking without changing the fundamental independent scores. No redistribution of probability mass occurs.

Standard MoE (Switch Transformer) uses softmax because the auxiliary loss derivation assumes probabilities summing to 1. DeepSeek's aux-loss-free design breaks this assumption — sigmoid scores are independent $(0,1)$ values, not a simplex distribution.

---

### Bias Update Mechanism — Deadband Rule

The bias update implements a **deadband controller** — a classic control-systems pattern where intervention only fires when the signal exceeds a tolerance band around the target:

```python
@torch.no_grad()
def update_bias(self, counts, speed=0.001):
    counts = counts.float()
    avg = counts.mean()
    self.bias[counts > avg * (1.0 + self.bias_upper)] -= speed  # over-utilized: demote
    self.bias[counts < avg * (1.0 - self.bias_lower)] += speed  # under-utilized: promote
```

| Situation | Expert receives | Bias change | Effect |
|---|---|---|---|
| Over-utilized | > 110% of average | `bias[e] -= 0.001` | Less likely to be selected next time |
| Under-utilized | < 90% of average | `bias[e] += 0.001` | More likely to be selected next time |
| Balanced (deadband) | 90–110% of average | No change | No intervention |

**Why it's aux-loss-free:**

1. **No gradient on bias** — The bias is a buffer, not a parameter. It doesn't receive gradients. The task loss is pure.
2. **No aux loss term** — There's no $\mathcal{L}_{\text{aux}}$ in the loss function. The total loss is just the task loss (+ optional MTP loss).
3. **Out-of-band updates** — The bias is updated periodically (every `bias_update_every` steps) based on observed token counts. This is a **control system**, not an optimization objective.

**Integration with training loop:**

```python
# In Pretrainer.train_step (every optimizer step):
if self._opt_steps % self.config.bias_update_every == 0:
    self._update_moe_bias()  # update bias for all MoE layers

def _update_moe_bias(self):
    for moe in self.raw_model.moe_layers():
        moe.update_gate_bias(speed=self.config.bias_update_speed)
```

With `bias_update_every=1` (canonical), the bias is updated after every optimizer step, keeping it responsive to changing routing patterns.

---

### Original Scores for Weights

The routing **decision** uses `biased` scores (bias affects which experts are selected). But the routing **weights** use the original `scores` (without bias):

```python
biased = scores + self.bias          # used ONLY for topk selection
indices = biased.topk(self.topk)[1]
weights = scores.gather(1, indices)  # raw sigmoid, no bias
```

**Why this matters:** If weights came from `biased`, the bias would also scale the expert's output magnitude — which is not desired. The bias should affect *selection* only, not *contribution magnitude*. Using raw `scores` for weights ensures the bias is a pure routing knob that doesn't enter the autograd graph through the weighted expert sum. If it did, the optimizer would fight the out-of-band bias updates.

---

### MoE Comparison Table — DeepSeek vs GPT-OSS-Lite

| Property | DeepSeek-V3-Lite | GPT-OSS-Lite |
|---|---|---|
| Routed experts | 20 | 8 |
| Active experts | 4 (top-4) | 2 (top-2) |
| Shared experts | 1 | 1 |
| Gate activation | Sigmoid | Softmax |
| Load balancing | Aux-loss-free bias | Standard aux loss ($\alpha = 0.01$) |
| Bias/aux gradient | None (buffer) | Gradient through aux loss |
| Expert `inter_dim` | 384 | 1536 |
| Dispatch | Stacked bmm (same) | Stacked `F.linear` (same) |

Both use stacked dispatch and SwiGLU experts, but the routing philosophy is completely different. DeepSeek uses finer-grained experts (20 vs 8) with sigmoid routing and no aux loss. GPT-OSS uses coarser experts with softmax routing and standard aux loss. The aux-loss-free approach keeps the task gradient pure — the optimizer never receives a load-balancing signal, only task performance feedback.

> **See also:** [[Docs/02_Model_Architecture|Model Architecture]] and [[Docs/13_Portfolio_Comparison|Portfolio Comparison]] for overall MoE topology and cross-project comparisons.

<!-- docs:verified 2026-08-04 · 59aeef3 -->
