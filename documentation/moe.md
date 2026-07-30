# Mixture-of-Experts (MoE) — AuxLossFreeGate + DeepSeekMoE

## A Comprehensive Technical Reference

> **Prerequisites:** [foundations.md](foundations.md) §8, [transformer.md](transformer.md).

> **Covers**: DeepSeek-V2/V3 MoE design, auxiliary-loss-free load balancing (§2.3.3), fine-grained expert routing, shared experts, sorted-token dispatch, and the implementation in this repo (`models/moe.py`, `models/moe_triton.py`).

---

## Table of Contents

1. [Abstract](#abstract)
2. [Motivation — Capacity Without Dense Compute](#motivation-capacity-without-dense-compute)
3. [DeepSeek MoE Design](#deepseek-moe-design)
4. [Mathematical Formulation (DeepSeek-V3 Paper)](#mathematical-formulation-deepseek-v3-paper)
5. [Auxiliary-Loss-Free Load Balancing](#auxiliary-loss-free-load-balancing)
6. [Why Not a Standard Auxiliary Loss?](#why-not-a-standard-auxiliary-loss)
7. [The SwiGLU Expert](#the-swiglu-expert)
8. [Dimension Breakdown](#dimension-breakdown)
9. [Implementation in This Repo](#implementation-in-this-repo)
   - [Class Structure](#class-structure)
   - [AuxLossFreeGate Forward Pass](#auxlossfreegate-forward-pass)
   - [DeepSeekMoE Forward Pass](#deepseekmoe-forward-pass)
   - [Sorted-Token Dispatch Layout](#sorted-token-dispatch-layout)
   - [Stacked Dispatch Path](#stacked-dispatch-path)
   - [Triton Grouped Dispatch Path](#triton-grouped-dispatch-path)
   - [Shared Expert Path](#shared-expert-path)
   - [Training Integration](#training-integration)
10. [Comparison: MoE vs Dense FFN](#comparison-moe-vs-dense-ffn)
11. [Performance Characteristics](#performance-characteristics)
12. [Appendix A — A worked numerical example](#appendix-a--a-worked-numerical-example)
13. [Appendix B — The Triton grouped-GEMM path](#appendix-b--the-triton-grouped-gemm-path)
14. [Appendix C — Gradient flow in MoE](#appendix-c--gradient-flow-in-moe)
15. [Appendix D — Load-balance metrics](#appendix-d--load-balance-metrics)
16. [Appendix E — Bias-update lifecycle (state diagram)](#appendix-e--bias-update-lifecycle-state-diagram)
17. [Appendix F — Numerical stability notes](#appendix-f--numerical-stability-notes)
18. [Appendix G — Glossary](#appendix-g--glossary)
19. [Appendix H — Frequently Asked Questions](#appendix-h--frequently-asked-questions)
20. [Load-Bearing Invariants (Do Not Break)](#load-bearing-invariants-do-not-break)
21. [Implementation Checklist](#implementation-checklist)
22. [References](#references)

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

For DeepSeek-V3-Lite (422M config): `d = 768`, dense `d_ff = 1536`, MoE `d_ff = 384`.

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

Classic MoE papers solve this with an **auxiliary load-balancing loss** added to the training objective. DeepSeek-V3's insight: you can achieve the same equilibrium with a **bias buffer** updated out-of-band, keeping the task loss gradient clean.

---

## DeepSeek MoE Design

DeepSeek's MoE architecture (introduced in [DeepSeekMoE, 2024](https://arxiv.org/abs/2401.06066) and refined in V3) has three structural choices this repo mirrors:

### 1. Fine-grained experts

Instead of 8 large experts (Mixtral-style), DeepSeek uses **many small experts** (256 in V3-671B; 20 in this 422M reproduction). Fine-grained decomposition improves expert specialisation: each expert can capture a narrower function.

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

Controlled by `n_dense_layers = 2` in `TransformerBlock`: MoE replaces FFN when `layer_id >= n_dense_layers`.

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

Every `bias_update_every` optimizer steps, `Pretrainer._update_moe_bias` calls `DeepSeekMoE.update_gate_bias`, which counts how many token slots were routed to each expert in the last forward and applies:

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

This repo exposes `get_load_balance_loss()` for **logging only** — it computes the classic `(f · P) · N` surrogate from the last forward's routing cache. `balance_loss_alpha` is not wired into the training loss; the metric appears in W&B logs as `balance_loss`.

---

## The SwiGLU Expert

Both routed and shared experts share the same `Expert` class:

```python
class Expert(nn.Module):
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

This is identical to `SwiGLUFFN` in `models/transformer.py` — the dense FFN used in layers 0–1. The only difference is width: dense layers use `inter_dim = 1536`, MoE experts use `moe_inter_dim = 384`.

### Parameter layout

| Weight | `nn.Linear` shape | Stored as |
|---|---|---|
| `w1` | `(I, D)` | gate projection |
| `w2` | `(D, I)` | down projection |
| `w3` | `(I, D)` | up projection |

Per expert: `3 · D · I` parameters.

For the 422M config (`D=768, I=384`): **884,736 params/expert**.

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

### DeepSeek-V3-Lite (this repo, 422M)

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

### Parameter accounting (per MoE layer, 422M)

```
Gate:           N × D           = 20 × 768           =     15,360
Routed experts: N × 3 × D × I   = 20 × 3 × 768 × 384 = 17,694,720
Shared experts: N_s × 3 × D × I =  1 × 3 × 768 × 384 =    884,736
                                                          ──────────
Per MoE layer total:                                      18,594,816  (~18.6M)
× 16 MoE layers:                                          297,517,056 (~298M)
```

The gate bias (`N` floats) is negligible. MoE FFN parameters dominate the 422M budget (~70% of total params).

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

```python
def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
    scores = F.linear(x, self.weight).sigmoid()       # (T, N)
    biased = scores + self.bias.to(scores.dtype)     # (T, N)
    indices = biased.topk(self.topk, dim=-1)[1]      # (T, k)
    weights = scores.gather(1, indices)              # (T, k) — raw scores
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-10)
    weights = (weights * self.route_scale).to(x.dtype)
    return weights, indices
```

**Inputs:** `x` of shape `(T, D)` where `T = batch × seq` (the MoE flattens `(B, S, D)` before gating).

**Outputs:**
- `weights`: `(T, k)` — normalised routing weights, sum to `route_scale` (default 1.0) per token
- `indices`: `(T, k)` — expert IDs in `[0, N)`

**Init:** `W_g ~ N(0, 0.006²)`, `bias = 0`.

### DeepSeekMoE Forward Pass

High-level flow:

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

### Stacked Dispatch Path

Default: `moe_dispatch = "stacked"`.

```python
for e in range(E):
    cnt = counts[e]
    if cnt == 0: continue
    chunk_tokens = sorted_token_ids[start:end]
    expert_in = flat[chunk_tokens]
    gate = expert_in @ _stacked_w1[e].t()
    up   = expert_in @ _stacked_w3[e].t()
    h    = silu(gate) * up
    out  = h @ _stacked_w2[e].t()
    y_routed.index_add_(0, chunk_tokens, out * weights.unsqueeze(-1))
```

**Characteristics:**
- One Python `for e in range(E)` loop per MoE layer (E=20)
- 3 GEMMs per non-empty expert → up to 60 GEMM launches per layer
- 16 MoE layers → up to 960 expert GEMM launches per training step
- Always available (CPU, Mac, no Triton dependency)
- Correctness reference for the Triton path

**Weight stacking:** Expert `nn.Linear` weights are stacked into `_stacked_w{1,2,3}` tensors **every forward pass**. This is load-bearing: caching stacked weights across steps left stale copies after `optimizer.step()` (regression test: `test_stacked_weights_refresh_after_optimizer_step`).

### Triton Grouped Dispatch Path

Opt-in: `moe_dispatch = "triton_grouped"` + `ENABLE_TRITON_KERNELS=1`.

```python
x_sorted = flat[sorted_token_ids].contiguous()
y_sorted = triton_grouped_moe_dispatch(x_sorted, w1, w2, w3, sorted_weights, expert_offsets)
y_routed.index_add_(0, sorted_token_ids, y_sorted)
```

The Triton kernel fuses the per-expert SwiGLU loop into one launch over the sorted layout. See [Appendix B](#appendix-b--the-triton-grouped-gemm-path).

**Hard limits (current kernel):**

| Constraint | Limit | 422M config | Result |
|---|---|---|---|
| `moe_inter_dim` (I) | ≤ 256 | 384 | **exceeds** |
| `dim` (D) | ≤ 256 | 768 | **exceeds** |

The canonical 422M config **cannot** use the Triton MoE kernel today. `triton_grouped` raises `ValueError` and auto-falls back to `stacked` with a one-time warning. The kernel is validated on smoke configs (`dim ≤ 64`, `moe_inter_dim ≤ 64`) and is the target of ongoing work (see `documentation/triton_kernels.md` §3).

**Training-only:** The Triton MoE kernel is not validated for inference decode (sparse per-step routing shapes differ from training's batched layout). Use `stacked` for inference.

### Shared Expert Path

```python
gate = bmm(flat.expand(E_s, -1, -1), _shared_w1.transpose(-1, -2))
up   = bmm(flat.expand(E_s, -1, -1), _shared_w3.transpose(-1, -2))
h    = silu(gate) * up
out  = bmm(h, _shared_w2.transpose(-1, -2))
return out.sum(dim=0)   # (T, D)
```

With `N_s = 1`, this is one SwiGLU applied to all `T` tokens. No routing, no sorting. The shared path is already batched and is a small fraction of total MoE compute (~1 shared vs 4 routed per token).

### Training Integration

In `training/pretrain.py`:

```python
# Every train_step (logging only — NOT added to loss):
balance_loss = sum(moe.get_load_balance_loss() for moe in model.moe_layers())

# Every bias_update_every optimizer steps (default: 1):
if opt_steps % bias_update_every == 0:
    for moe in model.moe_layers():
        moe.update_gate_bias(speed=bias_update_speed)
```

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

### Bottleneck analysis (422M, B=8, S=2048, 16 MoE layers)

From `documentation/triton_kernels.md` profiling estimates:

| Component | Step-time share | Notes |
|---|---|---|
| MLA attention (SDPA) | ~35–40% | FA2-accelerated |
| MoE dispatch loop | ~25–30% | 20 experts × 3 GEMMs × 16 layers |
| Dense FFN (2 layers) | ~5% | Small fraction |
| Embedding + LM head | ~10% | Vocab 100K |
| MTP auxiliary | ~5% | depth=1 |
| Other (norm, gate, etc.) | ~10% | |

The MoE per-expert Python loop is the **second-largest hotspot** after attention. The Triton grouped kernel targets ≥1.5× speedup on the MoE path (see `AGENTS.md` §Hard rules).

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

### Backward (FA2 re-compute pattern)

Forward saves: `x_sorted, w1, w2, w3, expert_offsets` (minimal).

Backward re-computes `gate`, `up`, `h` on the fly:

- `_grouped_moe_bwd_dx_kernel` — gradient w.r.t. `x_sorted`
- `_grouped_moe_bwd_dw_kernel` — gradient w.r.t. `w1, w2, w3` (one program per expert, no atomics)

Weight gradients accumulate in FP32, cast back to BF16 before return.

### PyTorch reference

`grouped_moe_pytorch()` in the same file implements identical arithmetic for CPU tests. `tests/test_moe_triton.py` asserts Triton ≈ reference within `atol=1e-2` for BF16.

### Env-var guard

```python
# models/_triton_dispatch.py
if os.environ.get("ENABLE_TRITON_KERNELS", "0") != "1":
  moe_dispatch "triton_grouped" → force-backed to "stacked"
```

Called at `Transformer.__init__` and `Pretrainer.__init__`. One warning at construction, not per layer.

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

### Expert path

```
loss → y_routed → f_i(h) → w1, w2, w3 of expert i
```

Only the 4 routed experts per token receive gradients. The other 16 experts are idle for that token (no compute, no grad).

### Shared expert path

```
loss → y_shared → f_s(h) → shared w1, w2, w3
```

Every token backprops through the shared expert — it trains ~T/k times more frequently than any single routed expert.

### Stacked vs Triton gradient equivalence

Both paths must produce identical gradients (verified by `torch.autograd.gradcheck` on float32 tiny configs in `test_moe_triton.py`). The Triton path's re-compute backward must match the PyTorch autograd graph of the stacked loop.

---

## Appendix D — Load-balance metrics

### `get_load_balance_loss()` — logging surrogate

```python
counts = bincount(indices.flatten(), minlength=N).float()
f = counts / counts.sum()                    # fraction of slots per expert
P = mean over tokens of (one_hot(indices) * weights).sum(dim=1)  # mean prob
return (f * P).sum() * N
```

This is the Switch Transformer auxiliary loss formula, computed from the **detached** routing cache. It is logged as `balance_loss` in training metrics but **never added to the training loss**.

**Interpretation:**
- Perfect balance → `f_i ≈ 1/N` for all `i`, and `P_i ≈ 1/N` → loss ≈ `N × (1/N)² × N = 1.0`
- Collapsed routing (all tokens to expert 0) → loss → `N`
- Lower is more balanced

### What to watch during training

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
| Bias update device mismatch | `counts` kept on bias device before boolean indexing |
| BF16 GEMM accumulation | PyTorch bmm accumulates FP32 on Ampere+; Triton uses explicit FP32 accumulators |
| Stale stacked weights | Re-stack every forward (not cached across steps) |
| Triton BF16 dot dtype | `h` cast to input dtype before `w2` dot in kernel |

### `route_scale`

Default `1.0`. Multiplying normalised weights by `route_scale` scales the routed expert contribution without re-normalising. Values ≠ 1.0 change the relative magnitude of routed vs shared expert output (shared is not scaled). Not used in the canonical 422M config.

---

## Appendix G — Glossary

| Symbol / Name | Meaning |
|---|---|
| `N` / `n_routed_experts` | Number of routed experts (20 in 422M) |
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

### Q3. Why is the Triton kernel disabled for the 422M config?

The kernel's `BLOCK_I` and `BLOCK_D` are `next_pow2(I)` and `next_pow2(D)`, capped at 256 for register pressure. The 422M config has `I=384, D=768`, both exceeding the cap. `triton_grouped` raises `ValueError` and falls back to `stacked`. Extending the kernel to larger dims is tracked in `documentation/triton_kernels.md`.

### Q4. Does the shared expert hurt specialisation?

Empirically, no — it handles "universal" FFN patterns (common subword mappings, high-frequency syntax) so routed experts can specialise on rarer functions. The shared expert trains on every token, so it converges faster. Removing it (`n_shared_experts=0`) is supported in code but not recommended.

### Q5. How often should `bias_update_every` be set?

The canonical config uses `1` (every optimizer step). Larger values smooth the bias signal but react slower to routing drift. The default dataclass value is `10`; ensure your YAML overrides it. With gradient accumulation, bias updates happen on optimizer steps, not micro-steps.

### Q6. Why re-stack expert weights every forward?

A regression was found where cached `_stacked_w*` tensors pointed to pre-optimizer-step weight values, freezing experts at their init values. Re-stacking costs a small `torch.stack` per layer but guarantees correctness. The Triton path uses the same freshly stacked tensors.

### Q7. Is group-limited routing (DeepSeek-V3's expert groups) implemented?

Not in this repo. The full V3 model routes first to expert *groups*, then within groups. The 422M config uses a single group (`n_expert_groups=1` equivalent). Adding group routing would require extending `AuxLossFreeGate.forward`.

### Q8. How do I monitor expert utilisation during training?

```python
for i, moe in enumerate(model.moe_layers()):
    counts = torch.bincount(moe._last_indices.flatten(), minlength=moe.n_routed_experts)
    print(f"Layer {i+2}: {counts.tolist()}")
```

For production, log `balance_loss` from the training loop (already wired in `Pretrainer`).

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
- `documentation/triton_kernels.md` §3 — Triton grouped-GEMM design and benchmarks
- `models/moe.py` — authoritative implementation
- `models/moe_triton.py` — Triton kernel + PyTorch reference

<!-- docs:verified 2026-07-31 · 88cb863 -->
