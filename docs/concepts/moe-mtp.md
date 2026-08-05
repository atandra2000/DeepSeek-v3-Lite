# DeepSeek-v3-Lite — DeepSeekMoE & MTP

> **Prerequisites:** [Foundations & Architecture](../concepts/foundations.md).

> **Read this if** you're debugging expert collapse, routing histograms, or aux-loss-free bias updates. **Skip if** you only need YAML knobs → [Training](../training.md).

**Depends on:** [Foundations & Architecture](../concepts/foundations.md) · **Read next:** [Training](../training.md), [Parallelism](../concepts/parallelism.md)

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

```
+-----------------------------------------------------------------------------------+
|                              AuxLossFreeGate Feedback Loop                        |
|                                                                                   |
|   Token Input x (B,S,D) ----> [Sigmoid Gate: x @ W_g^T] ---> Raw Scores (B,S,E)    |
|                                       |                             |             |
|                                       | + Bias                      | Selection   |
|                                       v                             v Weights     |
|                                [Top-k Selection] ----------> [Expert Dispatch]    |
|                                       |                             |             |
|                                       v Routing Indices             v Output      |
|                              [Token Bincount c_e]            y_routed (B,S,D)     |
|                                       |                                           |
|                                       v                                           |
|                              [Deadband Comparator]                                |
|                      c_e > avg*(1+upper) => b_e -= speed                          |
|                      c_e < avg*(1-lower) => b_e += speed                          |
|                                       |                                           |
|                                       +----> Updated Bias Buffer b (FP32)         |
+-----------------------------------------------------------------------------------+
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

The bottom line, stated plainly: **the fused kernel is valid only at smoke-config dimensions** (`I, D ≤ 256`), and the ≥1.5× MoE A100 speedup target is structurally blocked at canonical dims (I=384, D=768) until the kernel gains I-tiling and a corrected multi-D-block `dh`. The fallback is not a workaround — it is the designed behavior of the current kernel boundary, and `stacked` remains the correctness reference. See [Triton Kernels](../concepts/kernels-and-ops.md) for the kernel design and the roadmap.

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
weights = scores1,2 / sum = [0.62, 0.60] / 1.22 ≈ [0.508, 0.492]
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

**File:** `models/moe_triton.py` **Replaces:** the `for e in range(E)` loop in `_routed_forward_stacked`.

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

The kernel's `BLOCK_I` and `BLOCK_D` are `next_pow2(I)` and `next_pow2(D)`, capped at 256 for register pressure. The canonical config has `I=384, D=768`, both exceeding the cap. `triton_grouped` raises `ValueError` (`models/moe_triton.py:_check_dim_limits`) and falls back to `stacked`. Extending the kernel to larger dims is discussed in [Triton Kernels](../concepts/kernels-and-ops.md); the specific blockers (no I-tiling in fwd/dx, single-D-block `dh` in dw) are in Section 11.6.

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


- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — §2.3 MoE architecture, §2.3.3 auxiliary-loss-free balancing
- [DeepSeekMoE: Towards Ultimate Expert Specialization](https://arxiv.org/abs/2401.06066) — fine-grained expert decomposition, shared experts
- [Switch Transformers](https://arxiv.org/abs/2101.03961) — original auxiliary load-balancing loss formulation
- [GShard](https://arxiv.org/abs/2006.16668) — top-2 routing, capacity factors
- [Outrageously Large Neural Networks (Sparsely-Gated MoE)](https://arxiv.org/abs/1701.06538) — foundational MoE work
- [Kernels & Ops](../concepts/kernels-and-ops.md) — Triton grouped-GEMM design and benchmarks
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

> **See also:** [Model Architecture](../concepts/foundations.md) and [Portfolio Comparison](../concepts/foundations.md) for overall MoE topology and cross-project comparisons.

---

## Multi-Token Prediction (MTP) + Speculative Decoding

> **Prerequisites:** [Model Architecture](../concepts/foundations.md), [MLA](../concepts/attention-and-precision.md).

> **Read this if** you're working on MTP loss alignment or speculative decoding. **Skip if** you only need the standard train loop → [Training](../training.md).

**Depends on:** [Model Architecture](../concepts/foundations.md), [MLA](../concepts/attention-and-precision.md) · **Read next:** [Inference](../inference.md)

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


```
=== Multi-Token Prediction (MTP) Depth-1 Data Flow & Token Alignment ===

Trunk Tokens:  [x_0] --------> [x_1] --------> [x_2] --------> [x_3]
                 |               |               |               |
Main Trunk:    [Block 0..17] -> [Block 0..17] -> [Block 0..17] -> [Block 0..17]
                 |               |               |               |
Trunk Hidden:   h_0             h_1             h_2             h_3
                 |               |               |               |
Main Head:      p(x_1|x_0)      p(x_2|x_0..1)   p(x_3|x_0..2)   p(x_4|x_0..3)
Target:          x_1             x_2             x_3             x_4
                 |               |               |
               +---------------+---------------+
               | (Concatenate / Linear Projection)
               v
MTP Block:     [MTP Layer: MHA + Norm + FFN] (Fused h_t & Embed(x_{t+1}))
               |
MTP Logits:    p(x_2|x_0..1)   p(x_3|x_0..2)   p(x_4|x_0..3)
Target:          x_2             x_3             x_4
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


- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — MTP module design
- [Speculative Decoding (Leviathan et al.)](https://arxiv.org/abs/2211.17192)
- `models/mtp.py`, `inference/speculative.py` — authoritative implementation

## References

- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — §2.3 MoE architecture, §2.3.3 auxiliary-loss-free balancing
- [DeepSeekMoE: Towards Ultimate Expert Specialization](https://arxiv.org/abs/2401.06066) — fine-grained expert decomposition, shared experts
- [Switch Transformers](https://arxiv.org/abs/2101.03961) — original auxiliary load-balancing loss formulation
- [GShard](https://arxiv.org/abs/2006.16668) — top-2 routing, capacity factors
- [Outrageously Large Neural Networks (Sparsely-Gated MoE)](https://arxiv.org/abs/1701.06538) — foundational MoE work
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — MTP module design
- [Speculative Decoding (Leviathan et al.)](https://arxiv.org/abs/2211.17192)
- [Training](../training.md) — how the trainer consumes MTP loss and MoE bias updates
- [Inference](../inference.md) — speculative decoding at serving time
- [Kernels & Ops](../concepts/kernels-and-ops.md) — Triton grouped-GEMM path
