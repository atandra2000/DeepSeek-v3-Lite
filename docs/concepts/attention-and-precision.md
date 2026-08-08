# DeepSeek-v3-Lite — MLA & Mixed Precision

> **Canonical MLA reference for this repo.** Theory (DeepSeek-V2/V3), the absorption trick, decoupled RoPE, and the implementation in `models/mla.py` / `models/mla_triton.py`. If prose and code disagree, `models/mla.py` wins.

> **Read this if** you're debugging attention, KV-cache layout, or the absorption trick. **Skip if** you only need YAML knobs → [Training](../training.md).

**Depends on:** [Foundations & Architecture](../concepts/foundations.md) · **Read next:** [MoE & MTP](../concepts/moe-mtp.md)

## A Comprehensive Technical Reference

> **Covers**: DeepSeek-V2/V3 original formulation, the absorption trick, decoupled RoPE, and the implementation in this repo (`models/mla.py`).

---

## Table of Contents

1. [Abstract](#abstract)
2. [Motivation — The KV Cache Problem](#motivation--the-kv-cache-problem)
3. [Core Innovation — Low-Rank KV Compression](#core-innovation--low-rank-kv-compression)
4. [Mathematical Formulation (DeepSeek-V3 Paper)](#mathematical-formulation-deepseek-v3-paper)
5. [The Absorption Trick](#the-absorption-trick)
6. [Decoupled RoPE](#decoupled-rope)
7. [Query-Side Compression](#query-side-compression)
8. [Dimension Breakdown](#dimension-breakdown)
9. [Implementation in This Repo](#implementation-in-this-repo)
   - [Class Structure](#class-structure)
   - [Forward Pass: SDPA Path](#forward-pass-sdpa-path)
   - [Forward Pass: Manual Path (True Absorption)](#forward-pass-manual-path-true-absorption)
   - [KV Cache Management](#kv-cache-management)
   - [RoPE Helpers](#rope-helpers)
10. [Comparison: MLA vs MHA vs GQA vs MQA](#comparison-mla-vs-mha-vs-gqa-vs-mqa)
11. [Performance Characteristics](#performance-characteristics)
12. [V3-Lite Architecture Deep Dives](#v3-lite-architecture-deep-dives)
13. [Appendix A — FlashAttention in 60 seconds](#appendix-a--flashattention-in-60-seconds)
14. [Appendix B — A worked numerical example](#appendix-b--a-worked-numerical-example)
15. [Appendix C — The Triton path](#appendix-c--the-triton-path)
16. [Appendix D — Training vs. inference pathway](#appendix-d--training-vs-inference-pathway)
17. [Appendix E — Why `wkv_b` has shape `(H, d_nope + d_v, R)`](#appendix-e--why-wkv_b-has-shape-h-d_nope--d_v-r)
18. [Appendix F — KV-cache lifecycle (state diagram)](#appendix-f--kv-cache-lifecycle-state-diagram)
19. [Appendix G — Gradient flow in MLA](#appendix-g--gradient-flow-in-mla)
20. [Appendix H — Numerical stability notes](#appendix-h--numerical-stability-notes)
21. [Appendix I — Glossary](#appendix-i--glossary)
22. [Appendix J — Frequently Asked Questions](#appendix-j--frequently-asked-questions)
23. [Check Your Understanding](#check-your-understanding)
24. [Implementation Checklist](#implementation-checklist)
25. [References](#references)


> **Prerequisites:** Read [foundations](../concepts/foundations.md) §5 (attention), §6 (RoPE), and §11 (KV caching) before this chapter. This document assumes you understand standard multi-head attention and why KV-cache memory dominates inference.

---

## Abstract

**Multi-Head Latent Attention (MLA)** is the attention mechanism introduced in DeepSeek-V2 (May 2024) and refined in DeepSeek-V3 (Dec 2024). Its central innovation is **low-rank joint compression of keys and values**: instead of caching full per-head K/V tensors during autoregressive generation (the standard "KV cache"), MLA compresses them into a small latent vector and reconstructs them on the fly. This yields a **~7× reduction in KV-cache memory at the 411.6M scale** (up to ~30× at DeepSeek-V3's 671B scale) at no quality loss, making long-context inference dramatically more memory-efficient.

MLA achieves this through four interlocking mechanisms:

1. **Low-rank KV down-projection** — a learned matrix compresses the hidden state into a compact latent.
2. **The absorption trick** — the key/value up-projection matrices are algebraically absorbed into the query and output projections, so full K/V are never materialised during inference.
3. **Decoupled RoPE** — a separate, small positional embedding path (single shared K head) preserves RoPE compatibility without breaking the absorption algebra.
4. **Optional query compression** — a parallel low-rank path reduces activation memory during training.

---

## Motivation — The KV Cache Problem

During autoregressive decoding, every new token must attend to all preceding tokens. To avoid recomputing keys and values for every past position at each step, transformers store them in a **Key-Value (KV) cache**. The memory cost of this cache scales linearly with sequence length and quadratically with the number of attention heads.

**KV cache size per sequence** (standard MHA, FP16):

```
Bytes = 2 × L × n_layers × n_heads × d_head × 2 (FP16)
```

For a 70B-class model at 128K context length:

| Variant | KV cache per token per layer | Per sequence (128K, 61 layers) |
|---|---|---|
| MHA | 32,768 floats | 256 GB |
| GQA (8 groups) | 2,048 floats | 16 GB |
| MQA | 256 floats | 2 GB |
| **MLA** | **576 floats** | **2.18 GB** |

This isn't just about capacity — it's about **memory bandwidth**. During each decode step, the entire KV cache must be read from HBM into on-chip SRAM to compute attention scores. For MHA at 128K context, this means reading 256 GB per decode step. MLA reduces this to 2.18 GB — a **~120× reduction** in bandwidth demand.

**The real bottleneck in LLM inference is not compute — it's memory bandwidth.** MLA directly attacks this bottleneck.

### Historical placement

| Year | Model | Attention mechanism | KV per token per layer (typical) |
|---|---|---|---|
| 2017 | Transformer | MHA | $2 \times n_h \times d_h$ |
| 2019 | GShard / Switch | MHA + MoE | same |
| 2023 | GQA (LLaMA-2) | Grouped-query | $2 \times n_g \times d_h$ |
| 2024 | DeepSeek-V2/V3 | **MLA** | $R + d_{\text{rope}}$ (latent + rope slice) |

MLA is not merely "another GQA variant" — it changes **what is stored** in the cache (a latent vector) and **when** full keys/values are materialised (absorbed into query projection at inference).


---

## Core Innovation — Low-Rank KV Compression

Standard Multi-Head Attention (MHA) computes keys and values for token `t` as:

```
k_t = h_t W^K      (n_heads × d_head)
v_t = h_t W^V      (n_heads × d_head)
```

MLA inserts a **down-projection** that compresses the hidden state into a low-dimensional latent **before** the per-head key/value projections:

```
c_t^KV = h_t W^{DKV}    c_t^KV ∈ ℝ^{d_c}
```

where `d_c` (the KV compression dimension) is much smaller than `n_heads × d_head` (e.g., 512 vs 4096 for DeepSeek-V3). **Only this latent is cached.**

At attention time, the latent is up-projected back to the full head dimension:

```
k_t^C = c_t^KV W^{UK}     (n_heads × d_h)
v_t^C = c_t^KV W^{UV}     (n_heads × d_h)
```

Critically, `W^{DKV}` is **shared across all heads** — a single compression matrix, one latent per token. This is what gives MLA its dramatic cache savings.

---

## Mathematical Formulation (DeepSeek-V3 Paper)

The DeepSeek-V3 technical report (arXiv:2412.19437) specifies MLA as follows.

Let `d` be the model dimension, `n_h` the number of attention heads, `d_h` the per-head dimension, and `h_t ∈ ℝ^d` the input to the attention layer at position `t`.

### KV Compression

```
c_t^KV = W^{DKV} h_t                                          (1)
k_t^C  = [k_{t,1}^C; k_{t,2}^C; ...; k_{t,n_h}^C] = W^{UK} c_t^KV   (2)
k_t^R  = RoPE(W^{KR} h_t)                                     (3)
k_{t,i} = [k_{t,i}^C; k_t^R]                                   (4)
v_t^C  = [v_{t,1}^C; v_{t,2}^C; ...; v_{t,n_h}^C] = W^{UV} c_t^KV   (5)
```

Where:

| Symbol | Shape | Purpose |
|---|---|---|
| `W^{DKV}` | `d_c × d` | KV down-projection (compression) |
| `W^{UK}` | `n_h · d_h × d_c` | Key up-projection (recovery) |
| `W^{UV}` | `n_h · d_h × d_c` | Value up-projection (recovery) |
| `W^{KR}` | `d_h^R × d` | Decoupled RoPE key projection |
| `c_t^KV` | `d_c` | **Cached latent** |
| `k_t^R` | `d_h^R` | **Cached RoPE key** |

**Only `c_t^KV` and `k_t^R` are cached** — the blue-boxed quantities in the paper's notation. Everything else is reconstructed on the fly.

### Query Compression (Optional)

```
c_t^Q = W^{DQ} h_t                                            (6)
q_t^C = [q_{t,1}^C; ...; q_{t,n_h}^C] = W^{UQ} c_t^Q        (7)
q_t^R = RoPE(W^{QR} c_t^Q)                                    (8)
q_{t,i} = [q_{t,i}^C; q_t^R]                                  (9)
```

### Attention Output

The final attention output for head `i` at position `t`:

```
o_{t,i} = ∑_j softmax_j( q_{t,i}^T k_{j,i} / √(d_h) ) v_{j,i}
u_t = W^O [o_{t,1}; o_{t,2}; ...; o_{t,n_h}]
```

---

## The Absorption Trick

The absorption trick is what makes MLA efficient at inference time. If you compute attention scores naively by reconstructing full K/V at every step, you lose the cache benefit. The trick: **fold the up-projections into the query and output projections.**

### Score computation with absorption

Standard score computation (what you'd write naively):

```
score = (c_q^Q W^{UQ})^T (c_k^{KV} W^{UK})
```

But matrix multiplication is associative — re-parenthesise:

```
score = c_q^{Q^T} (W^{UQ} W^{UK^T}) c_k^{KV}
         _____/
       precompute once
```

The product `W^{UQ} W^{UK^T}` is a **constant matrix** that can be computed once at model load time. At inference, you only multiply latent against latent — the 128-dimensional inner products never appear.

### Tensor shapes — why the savings are real

Let `R = kv_lora_rank` (192 in this repo) and `H = n_heads` (12), and let `d_nope = qk_nope_head_dim` (48). For one token, one query head, one key position:

| Quantity                    | Shape              | Reads/writes |
|-----------------------------|--------------------|--------------|
| Latent `c^KV`               | `R`                | 192 floats   |
| Full K (`c^KV · W^{UK}`)    | `H · d_nope = 576` | 576 floats   |

So the **per-token, per-step memory traffic** for the content path drops from 576 floats to 192 floats — exactly a `d_nope : R = 48 : 192` ratio, which is `1/3`. The full picture (including the RoPE key of 24 floats) gives `576 → 216`, a **~2.67× drop** at this config; at DeepSeek-V3 scale it is `d_nope × H + d_rope × H → d_c + d_rope = 16384+8192 → 576`, the headline **~30× number** in §2.

### Value side absorption

Similarly, the value up-projection `W^{UV}` is absorbed into the output projection `W^O`. The post-attention weighted sum of latents can be directly up-projected to the model dimension:

```
output = (attn_weights × C^{KV}) W^{UV} W^O
```

The product `W^{UV} W^O` is precomputed, so the value expansion never materialises.

### Why this matters

| Step | Without absorption | With absorption |
|---|---|---|
| Score computation | Expand latent to full K (n_h × d_h), then score | Latent-to-latent inner product (d_c) |
| Value aggregation | Expand latent to full V (n_h × d_h), attend, project back | Attend directly in latent space, project once |
| Memory reads per token | n_h × (d_h_k + d_h_v) ≈ 16K floats | d_c + d_h^R ≈ 576 floats |

**The absorption trick transforms MLA from a computationally expensive curiosity into the most memory-efficient attention variant available.**

```
=== Materialized Attention Path (SDPA / Standard Training) ===
Hidden State x ---> c_KV (B, S, R=192) ---> [Up-Project W_UK/W_UV] ---> Full K, V (B, H, S, d_head)
                                                                                   |
Queries q_nope ---> [Dot Product in Full d_head Space] <---------------------------+

=== Absorbed Attention Path (Manual / Inference Decode) ===
Queries q_nope ---> [Fold W_UK: q_nope * W_UK] ---> Absorbed Query q~ (B, H, S_q, R=192)
                                                                    |
Cached Latent c_KV (B, S_kv, R=192) -------------> [Dot Product in Latent R-Space] ---> Score (B, H, S_q, S_kv)
```

#### Matrix Shape Transformations Across Execution Phases

| Execution Phase | Operand | SDPA Path (`"sdpa"`) | Manual Path (`"manual"`) | Triton Kernel Path (`"triton"`) |
|---|---|---|---|---|
| **Prefill** ($S_q > 1$) | Queries | `(B, H, S_q, d_nope+d_rope)` | `q_nope` `(H, B·S_q, d_nope)` | `q_nope` `(B, S_q, H, d_nope)` |
| | Keys | `(B, H, S_kv, d_nope+d_rope)` | `c_KV` `(B, S_kv, R)` | `c_KV` `(B, S_kv, R)` (HBM) |
| | Transformed Query | N/A | $\tilde{q} = q \cdot W_{UK}$ `(H, B·S_q, R)` | Materialized in registers |
| | Scores | `(B, H, S_q, S_kv)` | $\tilde{Q} C_{KV}^\top$ `(B, H, S_q, S_kv)` | Fused online softmax |
| **Decode** ($S_q = 1$) | Latent Cache Read | `c_KV` `(B, 1, R)` | `c_KV` `(B, 1, R)` | `c_KV` `(B, 1, R)` |
| | Up-projection | Full $K, V$ computed | Absorbed into $\tilde{q}$ once | Materialized in registers |


### The full two-direction derivation (with the repo's tensors)

The forward pass of `models/mla.py:MultiHeadLatentAttention.forward` contains the absorption algebra as executable tensors. This subsection derives the trick in **both directions**: from the materialised form to the absorbed form (Direction 1, what the manual path computes), and from the absorbed form back to the materialised form (Direction 2, what the SDPA path computes). Both directions anchor to the same two lines of code:

```python
q_nope_proj_h = torch.bmm(q_nope_h, wkv_b_k)                     # (H, B·S_q, R) — absorbed query
scores_content = self._per_batch_bmm(q_nope_proj, ctx_kv)        # (B, H, S_q, S_k)
```

where `_per_batch_bmm` (`models/mla.py:MultiHeadLatentAttention._per_batch_bmm`) is the one-liner that turns "batched query" and "broadcast key" into scores:

```python
return torch.matmul(q.transpose(1, 2), k.unsqueeze(1).transpose(2, 3))
```

**Notation.** Fix one head $h$. Let $c_t \in \mathbb{R}^{R}$ be the cached latent of token $t$ (a row vector), and let $U_h := \texttt{wkv\_b\_k}[h] \in \mathbb{R}^{d_{\text{nope}} \times R}$ be the per-head key up-projection **as stored** (note: `wkv_b` is an `nn.Linear(R → H·(d_nope + d_v))`, so its weight view `(H, d_nope + d_v, R)` already has the contraction dim last). The materialised content key is

$$k^{C}_{t,h} = c_t\, U_h^{\top} \in \mathbb{R}^{d_{\text{nope}}}.$$

For a content query $q^{C}_{s,h} \in \mathbb{R}^{d_{\text{nope}}}$ (row vector), the score is the dot product

$$\text{score}_{s,t,h} \;=\; \big\langle q^{C}_{s,h},\, k^{C}_{t,h}\big\rangle \;=\; q^{C}_{s,h}\, U_h\, c_t^{\top}.$$

**Direction 1 — fold the up-projection into the query.** Re-parenthesise:

$$\text{score}_{s,t,h} \;=\; \underbrace{\big(q^{C}_{s,h}\, U_h\big)}_{\tilde{q}_{s,h} \,\in\, \mathbb{R}^{R}} \cdot\, c_t^{\top} \;=\; \big\langle \tilde{q}_{s,h},\, c_t \big\rangle.$$

The map $q \mapsto q U_h$ is a constant linear transformation — it does not depend on the key index $t$ — so it can be applied **once per query**, and every score becomes an inner product in the shared latent space $\mathbb{R}^{R}$. The code executes this literally: `q_nope_h` stacks the $(B \cdot S_q) \times d_{\text{nope}}$ content queries of head $h$ as rows, `torch.bmm(q_nope_h, wkv_b_k)` contracts $d_{\text{nope}}$ and emits the $\tilde{q}$ rows of shape $(H, B \cdot S_q, R)$, and `_per_batch_bmm(q_nope_proj, ctx_kv)` computes $\tilde{Q}_{b,h} C_b^{\top}$ for every $(b, h)$ in one batched matmul: $(B, H, S_q, R) \cdot (B, H, R, S_k) \to (B, H, S_q, S_k)$. **The full key is never formed** — the inner product runs over $R = 192$ dimensions rather than $d_{\text{nope}} = 48$.

**Direction 2 — unfold back to the materialised form.** Start from the absorbed computation and substitute $\tilde{q}$ back out:

$$\big\langle \tilde{q}_{s,h},\, c_t \big\rangle \;=\; \big(q^{C}_{s,h}\, U_h\big)\, c_t^{\top} \;=\; q^{C}_{s,h}\, \big(U_h\, c_t^{\top}\big) \;=\; q^{C}_{s,h}\, \big(k^{C}_{t,h}\big)^{\top} \;=\; \big\langle q^{C}_{s,h},\, k^{C}_{t,h} \big\rangle.$$

The chain is pure associativity of matrix multiplication. This is the second direction of the trick: **given the same weights and the same latents, the absorbed computation is exactly the materialised computation** — no approximation, no learned compromise, no information lost. The SDPA path of `forward` is Direction 2 executed on the repo's tensors: it materialises

```python
KV_nope_h = torch.bmm(ctx_kv_bmm, wkv_b_kv.transpose(-1, -2))   # K_nope and V together
K_nope_h, V_h = KV_nope_h.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
```

and feeds `Q_full`, `K_full`, `V` to `F.scaled_dot_product_attention`. Both directions coexist inside the same method, which is why the parity tests (`tests/test_mla_triton.py`) can assert they agree.

**Why both directions matter.** Direction 1 is the *inference* story: because the score needs only the $R$-vector $\tilde{q}$ and the cached $R$-vector $c_t$, the per-head up-projection never has to run over the cache — this is where the ~7× cache-size reduction turns into a ~7× reduction in per-step HBM reads. Direction 2 is the *training* story: with no cache to save bandwidth on, you instead want gradients with respect to $U_h$ itself, and materialising $K$ routes them through a plain `bmm` backward into `wkv_b.weight` while letting the fused FlashAttention-2 kernel own the score/softmax/output work. Both directions are the same algebra; the implementation choice is about which operand is cheaper to move.

**The value side, derived.** The same associativity applies to outputs. With $W_h := \texttt{wkv\_b\_v}[h] \in \mathbb{R}^{d_v \times R}$ and attention weights $a_{s,t,h}$:

$$\text{out}_{s,h} \;=\; \sum_t a_{s,t,h}\, \big(c_t\, W_h^{\top}\big) \;=\; \underbrace{\Big(\sum_t a_{s,t,h}\, c_t\Big)}_{o^{\text{latent}}_{s,h} \,\in\, \mathbb{R}^{R}} W_h^{\top}.$$

The weighted sum can be accumulated **in latent space**, producing one $R$-vector per query per head, and the value up-projection applied once at the end. The manual path does exactly this:

```python
out_latent[b] = torch.bmm(attn[b], ctx_kv[b].unsqueeze(0).expand(h, -1, -1))   # Σ_t a·c_t in R
out_v = torch.bmm(out_h, wkv_b_v.transpose(-1, -2))                           # apply W_h — once
```

One more re-parenthesisation folds $W_h$ into the output projection $W^{O}_{h} \in \mathbb{R}^{d_v \times d_{\text{model}}}$:

$$\text{out}_{s,h}\, W^{O}_{h} \;=\; o^{\text{latent}}_{s,h}\, \big(W_h^{\top} W^{O}_{h}\big),$$

so the per-layer product $W_h^{\top} W^{O}_{h} \in \mathbb{R}^{R \times d_{\text{model}}}$ is a constant matrix computable once per layer — this is the paper's "value absorption". `[INFERENCE]` This repo keeps `wkv_b_v` and `wo` as two separate matmuls instead of fusing them into a precomputed product: `wo` is a first-class `nn.Linear` parameter (it must stay a parameter for the optimiser and is shared by all three paths), and the two-step version costs one extra kernel launch but no extra FLOPs. The algebra permits the fusion; the code chooses the simpler parameterisation.

**Exactness caveat (floating point).** In exact arithmetic Directions 1 and 2 agree; in floating point they do not. Direction 1 contracts over $R = 192$ dims after forming $\tilde{q}$; Direction 2 contracts over $d_{\text{nope}} = 48$ dims after forming $k^{C}$ — different accumulation orders, different roundings. Parity tests tolerate this with `atol=1e-3` in FP32; in BF16 the two paths can differ at the 1e-2 level on long contexts. Neither is "more correct" — they are two roundings of the same exact value.

**FLOP accounting.** Per query–key pair, the absorbed score contracts $R = 192$ dims; the materialised score contracts $d_{\text{nope}} = 48$. That $R/d_{\text{nope}} = 4\times$ ratio is the classic "MLA trades FLOPs for memory bandwidth" line. The asymmetry flips on the value side: Direction 2 must materialise $V$ from *every* cached latent every step ($R \cdot d_v$ MACs per key), while Direction 1 accumulates the weighted sum in latent space and pays the $R \cdot d_v$ up-projection only once per query. At decode ($S_q = 1$) the manual path is therefore the FLOP-cheaper of the two — the reason production still prefers SDPA/Triton is kernel efficiency (no per-batch Python loop, fused online softmax), not FLOPs.

---

## Decoupled RoPE

### A 90-second RoPE primer (read this first if RoPE is fuzzy)

Rotary Position Embeddings encode position by rotating each query/key vector in 2D subspaces. For a vector `x ∈ ℝ^{2d}`, RoPE pairs coordinates `(x_{2i}, x_{2i+1})` and rotates the `i`-th pair by angle `θ_i · pos`, where the frequency schedule is:

```
θ_i = base^{-2i / d},   base = 10000 (the canonical rope_theta)
```

Concretely, for one pair at position `p`:

```
[x_{2i}]    [cos(θ_i p)  -sin(θ_i p)] [x_{2i}]
[x_{2i+1}] → [sin(θ_i p)   cos(θ_i p)] [x_{2i+1}]
```

This is a 2×2 rotation per pair, hence the half-dim usage: `qk_rope_head_dim` real dims = `qk_rope_head_dim / 2` independent rotations. The code implements this with complex numbers (`view_as_complex`, `torch.polar(ones, freqs)`) for a clean Hadamard-style multiplication.

**The crucial property** that makes RoPE useful for attention:

```
⟨ R(θ, pos_q) q,  R(θ, pos_k) k ⟩ = g(q, k, pos_q - pos_k)
```

i.e., the inner product of two RoPE-rotated vectors depends only on the *relative* position. This is what gives transformers length-generalisation and clean position-equivariance. **But it also makes RoPE non-linear in position**, which is the heart of the absorption problem below.

### The problem

RoPE is not a linear operation — the rotation depends on the token position. If you try to absorb `W^{UK}` into `W^{UQ}` as described above, the RoPE rotation ends up **between** the two matrices, breaking the associative reordering:

```
score = c_q^Q W^{UQ^T} R(θ, Δ) W^{UK} c_k^{KV}
```

`R(θ, Δ)` is position-dependent, so `W^{UQ^T} R(θ, Δ) W^{UK}` cannot be precomputed. (For every relative position `Δ`, you'd need a different `R`, so the savings vanish.)

### The solution: decoupled RoPE

DeepSeek's fix splits the head dimension into two parts:

- **Content part** (`qk_nope_head_dim`): carries semantic content, NO RoPE. Goes through the latent compression as described. Supports absorption.
- **Position part** (`qk_rope_head_dim`): carries positional information, uses RoPE. **Not compressed** — operates as Multi-Query Attention (single shared K head across all Q heads).

This means:

```
k_t = [k_t^C ;  k_t^R]        (concat of content + positional keys)
q_t = [q_t^C ;  q_t^R]        (concat of content + positional queries)

score = q_t^C k_s^C^T + q_t^R k_s^R^T
        _____________/   ___________/
       content (absorbed)   position (MQA)
```

The content score uses the absorption trick (linear, precomputable). The position score uses standard RoPE but with a **single shared key head** — no different from MQA's positional cost.

### Cache impact

The decoupled RoPE key `k_t^R` (typically 64 dims for DeepSeek-V3, 16 dims in this repo) is the **second** cached quantity:

```
Cached per token per layer:  c_t^KV (d_c)  +  k_t^R (d_h^R)
                           =  192          +  24         =  216  (this repo, 422M config)
```

---

## Query-Side Compression

DeepSeek-V3 also compresses the **query** using a parallel low-rank path. This doesn't affect the cache size (queries aren't cached) but reduces **activation memory** during training.

When `q_lora_rank > 0`:

```
c_t^Q = h_t W^{DQ}           c_t^Q ∈ ℝ^{d'_c}
q_t   = c_t^Q W^{UQ}         q_t   ∈ ℝ^{n_h × qk_head_dim}
```

This is a bottleneck: the hidden state is first compressed to `d'_c` (e.g., 1,536 for DeepSeek-V3), then expanded to the full head dimension.

In this repo's 422M config, `q_lora_rank = 0` (no query compression) as a simplification at the smaller scale.

---

## Dimension Breakdown

### DeepSeek-V3 (original, 671B total)

| Parameter | Value | Description |
|---|---|---|
| `d_model` | 7,168 | Hidden dimension |
| `n_heads` | 128 | Number of attention heads |
| `d_head` | 128 | Per-head dimension |
| `qk_nope_head_dim` | 128 | Content key/query dimension per head |
| `qk_rope_head_dim` | 64 | Positional key/query dimension per head |
| `qk_head_dim` | 192 | Total QK head dim (128 + 64) |
| `v_head_dim` | 128 | Value head dimension |
| `kv_lora_rank` | 512 | KV compression latent dimension |
| `q_lora_rank` | 1,536 | Query compression latent dimension |
| **KV cache per token** | **576** | 512 (latent) + 64 (RoPE key) |

### DeepSeek-V3-Lite (this repo, 422M)

| Parameter | Value | Description |
|---|---|---|
| `dim` | 768 | Hidden dimension |
| `n_heads` | 12 | Number of attention heads |
| `qk_nope_head_dim` | 48 | Content key/query dimension per head |
| `qk_rope_head_dim` | 24 | Positional key/query dimension per head |
| `qk_head_dim` | 72 | Total QK head dim (48 + 24) |
| `v_head_dim` | 64 | Value head dimension |
| `kv_lora_rank` | 192 | KV compression latent dimension |
| `q_lora_rank` | 0 | No query compression (simplified) |
| `n_layers` | 18 | 2 dense + 16 MoE |
| `n_dense_layers` | 2 | First 2 layers use dense SwiGLU |
| `max_seq_len` | 2048 | Training sequence length |
| **KV cache per token** | **216** | 192 (latent) + 24 (RoPE key) |

The 422M config is a 2.7×-scale-up of the 1650 smoke-test config (`dim=64, n_heads=4, kv_lora_rank=16, qk_rope_head_dim=16`). Compared to DeepSeek-V3's 671B it scales kv_lora_rank down by 2.7× and qk_rope_head_dim down by 2.7×, but preserves the compression ratio: MLA caches 216 floats per token vs MHA's 1536 (K+V at `2 × 12 × 64`), a **~7.1× KV-cache reduction** at the 422M scale.

> **Which reduction factor?** The ratio depends on the MHA baseline.
> - **~7.1×** — per-token floats at 422M: MHA caches K+V at `d_head=64` (`2·H·d_head = 1536`) vs MLA `216`. This is the canonical figure used throughout these docs.
> - **~8×** — same but MHA at `d_head=72` (`2·12·72 = 1728`).
> - **~30×** — DeepSeek-V3's 671B scale (`d_nope·H + d_rope·H = 16384 + 8192 → d_c + d_rope = 576`).
> - "~46×" / "~120×" figures in older notes mixed bytes, batch size, or 70B-class long-context numbers into the comparison and are **wrong** at the 422M scale.
> Bytes-per-float and batch size scale both sides equally, so they never change the ratio.

---

## Implementation in This Repo

The MLA implementation lives in `models/mla.py`. Here's a walkthrough of every major component.

### Class Structure

```python
class MultiHeadLatentAttention(nn.Module):
```

**Key class attributes (422M config in parentheses):**

| Attribute | Source | Description |
|---|---|---|
| `dim` | `config["dim"]` | Model hidden size (768) |
| `n_heads` | `config["n_heads"]` | Total attention heads (12) |
| `kv_lora_rank` | `config["kv_lora_rank"]` | KV compression dim (192) |
| `qk_nope_head_dim` | `config["qk_nope_head_dim"]` | Content-only QK dim per head (48) |
| `qk_rope_head_dim` | `config["qk_rope_head_dim"]` | Positional QK dim per head (24) |
| `qk_head_dim` | computed | `qk_nope_head_dim + qk_rope_head_dim` (72) |
| `v_head_dim` | `config["v_head_dim"]` | Value head dimension (64) |
| `max_seq_len` | `config["max_seq_len"]` | Maximum sequence length (2048) |

### Learned Projections

```
                              KV Compression Path
┌────────────────────────────────────────────────────────────────────────┐
│  x (bsz, seqlen, 768)                                                  │
│    │                                                                   │
│    ▼                                                                   │
│  wkv_a: Linear(768 → 192 + 24)  ←── joint projection                  │
│    │                                                                   │
│    ├── kv_latent (192): stored in cache                                │
│    ├── k_pe_raw (24): RoPE'd and stored in pe_cache                    │
│    │                                                                   │
│    ▼                                                                   │
│  kv_norm: RMSNorm(192)  ←── normalise latent before cache             │
│                                                                        │
│  wkv_b.weight reshaped → (n_heads=12, qk_nope+v_head=48+64, 192)       │
│    ├── wkv_b_k[:48]  : key up-projection (12 heads × 48 → from 192)   │
│    └── wkv_b_v[48:]  : value up-projection (12 heads × 64 → from 192) │
└────────────────────────────────────────────────────────────────────────┘

                              Query Path (no compression; q_lora_rank=0)
┌────────────────────────────────────────────────────────────────────────┐
│  x (bsz, seqlen, 768)                                                  │
│    │                                                                   │
│    ▼                                                                   │
│  wq: Linear(768 → 12 × 72 = 864)  ←── no compression                  │
│    │                                                                   │
│    ▼                                                                   │
│  reshape → (bsz, seqlen, 12, 72)                                       │
│    │                                                                   │
│    ├── q_nope (48): content, no RoPE                                   │
│    └── q_pe (24): RoPE'd for positional scoring                        │
└────────────────────────────────────────────────────────────────────────┘

                              Output Projection
┌────────────────────────────────────────────────────────────────────────┐
│  wo: Linear(12 × 64 = 768 → 768)  ←── projects attended values back    │
└────────────────────────────────────────────────────────────────────────┘
```

### Forward Pass: SDPA Path

The `attn_impl == "sdpa"` path is the default and uses FlashAttention-2 via PyTorch's `F.scaled_dot_product_attention`. This is the path used for both training and inference.

**Step-by-step:**

1. **Query projection** (lines 114-120):
   ```python
   q = self.wq(x)                              # (bsz, seqlen, n_heads * qk_head_dim)
   q = q.view(bsz, seqlen, n_heads, qk_head_dim)
   q_nope, q_pe = q.split([48, 24], dim=-1)   # split content vs position
   q_pe = self._apply_rope(q_pe, start_pos, seqlen)
   ```

2. **KV compression** (lines 122-126):
   ```python
   kv_a = self.wkv_a(x)                        # joint projection
   kv_latent, k_pe_raw = kv_a.split([192, 24], dim=-1)
   kv_normed = self.kv_norm(kv_latent)         # normalise latent
   k_pe = self._apply_rope(k_pe_raw.unsqueeze(2), ...).squeeze(2)
   ```

3. **Cache write/read** (lines 127-134):
   ```python
   if use_cache:
       self.kv_cache[:bsz, start_pos:end_pos] = kv_normed.detach()
       self.pe_cache[:bsz, start_pos:end_pos] = k_pe.detach()
       ctx_kv = self.kv_cache[:bsz, :end_pos]     # full context latents
       ctx_pe = self.pe_cache[:bsz, :end_pos]      # full context rope keys
   else:
       ctx_kv = kv_normed                          # current sequence only
       ctx_pe = k_pe
   ```

4. **Split wkv_b weights** (lines 136-138):
   ```python
   wkv_b_full = self.wkv_b.weight.view(n_heads, 48+64, 192)
   wkv_b_k = wkv_b_full[:, :48]     # (h, 48, 192) — key up-projection
   wkv_b_v = wkv_b_full[:, 48:]     # (h, 64, 192) — value up-projection
   ```

5. **Materialise K_nope and V from latents (the *unabsorbed* SDPA path):**

   This is where the SDPA path differs from the manual path. Instead of computing scores in latent space, it **materialises full K_nope and V by multiplying the latent with wkv_b weights**, then runs a single `scaled_dot_product_attention` call:

   ```python
   # Fused: one bmm over ctx_kv produces K_nope and V together.
   wkv_b_kv = torch.cat([wkv_b_k, wkv_b_v], dim=1)       # (H, d_nope+d_v, R)
   KV_nope_h = torch.bmm(ctx_kv_bmm, wkv_b_kv.transpose(-1, -2))   # (H, B·S_k, d_nope+d_v)
   K_nope_h, V_h = KV_nope_h.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
   ```

   The cat is intentional: one bmm that produces both K and V in one launch. After materialisation, K is `(B, H, S_k, d_nope)` and V is `(B, H, S_k, d_v)`.

> > **Note — important trade-off.** This materialises K and V for every > step. The K tensor is `(B, H, S_k, 48)` floats — at `B=4, H=12, > S_k=2048` that's `4·12·2048·48 ≈ 4.7M` floats ≈ 9.4 MB in BF16 per > layer. For 18 layers that's ~170 MB — substantial but not blocking. > On GPU with FlashAttention-2 the *attention matrix* `QK^T` is never > materialised (that's the SDPA win), so the SDPA path trades a small > K materialisation for the elimination of the full `(S_q × S_k)` > attention matrix in HBM. This is not the "true" absorption trick — > that's in the manual path. The SDPA path is *fast, but cache-blind*; > the manual path is *cache-aware, but slow*. The Triton path (§C) is > the best of both.

6. **Concatenate RoPE keys** (lines 170-173):
   ```python
   Q_full = torch.cat([Q_nope, Q_rope], dim=-1)    # (bsz, h, seqlen_q, 72)
   K_full = torch.cat([K_nope, K_rope], dim=-1)    # (bsz, h, seqlen_k, 72)
   ```

   Here K_rope is the **shared** RoPE key `ctx_pe` expanded to all heads:
   ```python
   K_rope = ctx_pe.unsqueeze(1).expand(-1, h, -1, -1)
   ```

7. **FlashAttention call** (lines 174-175):
   ```python
   attn = F.scaled_dot_product_attention(
       Q_full, K_full, V,
       attn_mask=attn_mask,
       scale=self.softmax_scale,
   )
   ```

8. **Output** (line 176):
   ```python
   return self.wo(attn.transpose(1, 2).contiguous().flatten(2))
   ```

### Forward Pass: Manual Path (True Absorption)

The `attn_impl == "manual"` path implements the **true absorption trick**. It keeps everything in latent space and only recovers V at the very end.

**Step-by-step (the `attn_impl == "manual"` branch of `forward`):**

1. **Project queries into latent space (the absorption step):**

   ```python
   q_nope_h = q_nope.permute(2, 0, 1, 3).reshape(h, bsz * seqlen_q, d)  # (H, B·S_q, d_nope)
   q_nope_proj_h = torch.bmm(q_nope_h, wkv_b_k)                          # (H, B·S_q, R)
   q_nope_proj = q_nope_proj_h.reshape(h, bsz, seqlen_q, R).permute(1, 2, 0, 3).contiguous()
   ```

   `q_nope` is `(B, S_q, H, d_nope) = (B, S_q, H, 48)`. Multiplying by `wkv_b_k` (shape `(H, d_nope, R) = (H, 48, 192)`) per head projects each query from per-head content space (48 dims) into shared latent space (192 dims). **The result `q_nope_proj` is `q · W^{UK}`, the pre-absorbed query** — i.e., if we ever materialised K we could skip the up-projection entirely. After this step, `q_nope_proj` is exactly the LHS of `score = q_nope_proj · c_kv`.

2. **Content scores in latent space (true absorption — no full K):**

   ```python
   scores_content = self._per_batch_bmm(q_nope_proj, ctx_kv)  # (B, H, S_q, S_k)
   ```

   `_per_batch_bmm` is implemented as `q.transpose(1,2) @ k.unsqueeze(1).transpose(2,3)`, producing `(B, H, S_q, S_k)`. **Notice: the inner product is over `R=192` dims, not `d_nope=48`.** This is 4× more FLOPs per inner product than standard attention (192/48 = 4). It's the cost we pay for never materialising K — the up-projection is fused into the matmul.

3. **Position scores via MQA (shared RoPE key):**

   ```python
   scores_rope = self._per_batch_bmm(q_pe, ctx_pe)            # (B, H, S_q, S_k)
   ```

   `q_pe` is `(B, S_q, H, d_rope=24)`. `ctx_pe` is `(B, S_k, d_rope=24)` with no head dim — single shared RoPE key. The unsqueeze in `_per_batch_bmm` broadcasts this shared key against all `H` heads. This is exactly MQA's positional cost (one K head for many Q heads).

4. **Combine, scale, mask, softmax:**

   ```python
   scores = (scores_content + scores_rope) * self.softmax_scale
   if mask is not None:
       scores = scores + mask.expand(bsz, h, seqlen_q, -1)
   attn = scores.softmax(dim=-1, dtype=torch.float32).to(x.dtype)
   ```

   The `softmax(dtype=torch.float32)` is **a numerical-stability choice**: softmax involves `exp()`, which overflows easily in BF16 for scores `> ~10`. Up-casting to FP32 for the softmax and casting back is standard practice.

5. **Weighted sum in latent space:**

   ```python
   out_latent = torch.empty(bsz, h, seqlen_q, R, dtype=x.dtype, device=x.device)
   for b in range(bsz):                                       # ← Python loop!
       out_latent[b] = torch.bmm(attn[b], ctx_kv[b].unsqueeze(0).expand(h, -1, -1))
   ```

   Attends over `ctx_kv` in latent space, producing per-head latent representations. **This Python `for b in range(bsz)` loop is the reason the manual path is slow on GPU** — each iteration launches a small CUDA kernel with overhead. PyTorch's SDPA path avoids this by materialising K (step 5 in §9.2) and then calling a single fused `scaled_dot_product_attention` kernel that does softmax+matmul as one FlashAttention-style operation.

6. **Recover V via `wkv_b_v` (one bmm per head, no Python loop):**

   ```python
   out_h = out_latent.permute(1, 0, 2, 3).reshape(h, bsz*seqlen_q, R)   # (H, B·S_q, R)
   out_v = torch.bmm(out_h, wkv_b_v.transpose(-1, -2))                  # (H, B·S_q, d_v)
   ```

   Now — and only now — we materialise values in `(B, S_q, H, d_v=64)` space.

7. **Output projection:**

   ```python
   return self.wo(out.flatten(2))   # out: (B, S_q, H*d_v) → (B, S_q, dim)
   ```

This path is slower than SDPA for typical GPU hardware due to the per-batch bmm loops, but it demonstrates the true absorption mechanism and serves as the reference implementation that the Triton path emulates in a fused kernel.

**Why two paths exist at all.** The SDPA path is what we run in production — fast, hardware-fused, no materialisation of the attention matrix. The manual path exists for two reasons:

1. **Correctness reference.** It's the literal implementation of the
   absorption algebra. If the SDPA path disagrees with the manual path in unit tests, something is wrong.
2. **CPU / portability.** On CPU/Mac or under Triton-kernel-kill,
   the manual path is the fallback that keeps the test suite green without requiring a fused kernel.

In real production on A100/H100, neither runs alone — the Triton path (§C) fuses both and matches the manual path's algebra while approaching SDPA's speed.

### KV Cache Management

The KV cache stores two things per layer:

```python
self.kv_cache  # (batch, max_seq_len, kv_lora_rank=192) — compressed latents
self.pe_cache  # (batch, max_seq_len, qk_rope_head_dim=24) — RoPE keys
```

**Allocation** (`_ensure_cache`):
- Lazily allocated on first forward call
- Grown in doubling steps (min 16) to amortise reallocation
- Handles device/dtype changes

**Write** (`forward` lines 128-129):
```python
self.kv_cache[:bsz, start_pos:end_pos] = kv_normed.detach()
self.pe_cache[:bsz, start_pos:end_pos] = k_pe.detach()
```

**Why `.detach()` is non-negotiable.** PyTorch's autograd builds a DAG where every tensor remembers the operations that produced it. If we stored `kv_normed` (not its detached copy) into the cache, the cache would carry *edges* back to the forward pass that produced it. On the next forward, the cache's slice re-enters autograd as input — and the DAG gains a back-edge from step `t+1` to step `t`. Three bad consequences:

1. **Memory leak across forwards.** The DAG from step `t` is pinned
   alive by step `t+1`'s reference. After 1000 decode steps you'd be holding 1000× the per-step graph.
2. **Gradient backflow.** Gradients would flow *through the cache* into
   a previous forward's parameters, breaking the assumption that each step's loss only depends on the current step.
3. **Triton / CUDA-graph incompatibility.** Cached tensors with live
   autograd state can't be safely treated as plain storage.

`.detach()` returns a view with `requires_grad=False` and no grad-fn — a plain `Tensor` that behaves like a constant buffer. The math result is identical (same storage); the autograd metadata is severed.

**Read** (lines 130-131):
```python
ctx_kv = self.kv_cache[:bsz, :end_pos]   # full prefix up to current pos
ctx_pe = self.pe_cache[:bsz, :end_pos]
```

**Reset** (`reset_cache`):
```python
def reset_cache(self):
    self.kv_cache = None
    self.pe_cache = None
    self._cache_batch = 0
```

**Arbitrary-offset writes (how "prefix caching" actually works).** There is no `prefill_cache` method in this repo — the mechanism for writing pre-computed latents at an arbitrary offset is the forward pass itself. Calling `forward(x, start_pos=k, use_cache=True)` writes exactly the slice `kv_cache[:bsz, k:k+seqlen]` and then reads `ctx_kv = kv_cache[:bsz, :k+seqlen]`. A shared prompt prefix is therefore handled by processing the prefix once with `use_cache=True`, then continuing with `start_pos = len(prefix)`; no separate API exists or is needed. (Earlier drafts of this chapter referenced a `prefill_cache` helper — that symbol does not exist in `models/mla.py`; the slice-write contract above is the ground truth.)

#### The cache lifecycle as a state machine

The KV cache has exactly two states and a small set of transitions, all driven by `models/mla.py:MultiHeadLatentAttention._ensure_cache`, `models/mla.py:MultiHeadLatentAttention.reset_cache`, and the first lines of `MultiHeadLatentAttention.forward`:

| # | State / transition | Trigger | What happens |
|---|---|---|---|
| 1 | **INIT** | `__init__` | `kv_cache = None`, `pe_cache = None`, `_cache_batch = 0` |
| 2 | INIT → **READY** | first `forward(use_cache=True)` | `_ensure_cache` allocates `(B, max_seq_len, R)` and `(B, max_seq_len, d_rope)` zeros |
| 3 | READY → READY (reuse) | `forward(use_cache=True)` with `bsz ≤ _cache_batch`, same device & dtype | no-op; cache is reused in place |
| 4 | READY → READY (grow) | `bsz > _cache_batch`, or device/dtype changed | **reallocate** at `new_bsz = max(bsz, 2·_cache_batch, 16)` — old contents are discarded |
| 5 | READY → **ERROR** | `start_pos + seqlen > max_seq_len` | `RuntimeError("end_pos exceeds max_seq_len")` — the sequence axis is *never* grown |
| 6 | READY → INIT | `reset_cache()` | back to state 1 (also via `Transformer.reset_cache`, which loops over layers) |
| 7 | (any) | `forward(use_cache=False)` | cache never touched; `ctx_kv`/`ctx_pe` are the current step's latents |

The growth policy in state 4 deserves attention:

```python
new_bsz = max(bsz, self._cache_batch * 2, 16)
```

Three terms, three reasons: `bsz` covers the immediate request, `* 2` doubles so that a serving loop whose batch creeps upward reallocates amortised (`O(log n)` reallocations instead of `O(n)`), and `16` is the minimum allocation so tiny smoke tests don't thrash. The reallocation conditions are checked in `_ensure_cache` (`need_alloc`), and the check is a pure guard: when nothing changed, the function returns immediately.

**Pitfall — growing the batch axis drops the contents.** Transitions 3 and 4 are not equivalent to "resizing". `_ensure_cache` *allocates a fresh tensor* when it decides to grow; it never copies the old rows. In practice this means you must not grow the batch mid-request: if the first forward of a request used `bsz=4` and a later forward of the same logical sequence uses `bsz=8`, the fresh zero tensor erases the prefix latents. The contract is: **pick the batch size on the first `use_cache=True` call and keep it fixed until `reset_cache()`**. The doubling policy only pays off when growth happens *between* requests.

**Pitfall — the sequence axis is a hard bound, not a growable dim.** The cache is allocated `(B, max_seq_len, R)`. Exceeding it raises (state 5) rather than reallocating, because `max_seq_len` is also the RoPE table bound (`_extend_rope` clamps `grow_to` to `max_seq_len`) and the training context length. Decoding past `max_seq_len` is a hard error by design — see also `models/mla.py:MultiHeadLatentAttention._extend_rope`.

**Pitfall — `reset_cache()` between requests.** `Transformer.generate` calls `self.reset_cache()` before generating (see `models/transformer.py:Transformer.generate`), but a serving loop that calls `forward` directly must do it by hand: without a reset, the next request reads the previous request's latents from positions `[0, end_pos)` — silent cross-request leakage.

**Memory cost of the cache (derived arithmetic, not measured).** Per layer the cache holds `B · max_seq_len · (R + d_rope)` floats. At the canonical config (`B=8`, `max_seq_len=2048`, `R=192`, `d_rope=24`, 18 layers) that is `8 · 2048 · 216 · 18 ≈ 63.7M` floats ≈ **127 MB in BF16** across the whole model. The MHA baseline at the same shapes (`2·H·d_v = 1536` floats/token) would be `8 · 2048 · 1536 · 18 ≈ 453M` floats ≈ **906 MB** — the same ~7.1× ratio as the per-token comparison in the Dimension Breakdown section. (These are derived estimates; no GPU run has executed in this repo.)

### RoPE Helpers

**`_extend_rope(seq_len, device)`** (`models/mla.py:MultiHeadLatentAttention._extend_rope`):
- Lazily grows the precomputed RoPE frequency table up to `max_seq_len`
- Grows by 2x to amortise during autoregressive generation
- Supports YaRN scaling via `rope_factor`

The whole RoPE implementation of the layer is these two methods. First the frequency table:

```python
def _extend_rope(self, seq_len: int, device: torch.device) -> None:
    if seq_len <= self._rope_seq_len:
        return
    dim = self.qk_rope_head_dim
    inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    grow_to = max(seq_len, self._rope_seq_len * 2, 64)
    grow_to = min(grow_to, self.max_seq_len)
    t = torch.arange(grow_to, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    self.freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    self._rope_seq_len = grow_to
```

Four details are load-bearing:

1. **The frequency schedule.** `torch.arange(0, dim, 2)` produces
   `dim/2 = 12` exponents, and `inv_freq = 1.0 / theta ** (2i/dim)` is the standard RoPE schedule `theta ** (-2i/dim)` with `theta = rope_theta = 10000`. The frequencies are naturally bounded in `[1/10000, 1]` (Appendix H).
2. **`torch.polar` builds the rotation.** `freqs` is a `(grow_to, 12)`
   matrix of *angles* `θ_i · p`; `torch.polar(ones, freqs)` converts each angle into the complex number `cos(θ_i p) + i·sin(θ_i p)` in one call. The result is a `(grow_to, 12)` `complex64` table.
3. **Doubling growth with a hard cap.** `grow_to = max(seq_len,
   _rope_seq_len * 2, 64)` amortises the table build during decode (where `end_pos` creeps up by 1 per step), while `min(grow_to, max_seq_len)` keeps the table bounded by the training context. The early return `if seq_len <= self._rope_seq_len` makes steady-state decode calls free. The forward calls this with `end_pos = start_pos + seqlen`, so the table always covers the slice `[start_pos, start_pos + seqlen)` that `_apply_rope` will index.
4. **No YaRN in the table itself.** `rope_factor` is *not* applied here —
   the table is always built at the base `rope_theta`. YaRN enters only through the softmax scale (`mscale`, see the YaRN section below), which is this repo's deliberate simplification: the rotation frequencies are standard, and the attention temperature compensates.

**`_apply_rope(x, start_pos, seqlen)`** (`models/mla.py:MultiHeadLatentAttention._apply_rope`):
- Reshapes `x` from `(..., qk_rope_head_dim)` real → `(..., qk_rope_head_dim//2)` complex
- Multiplies by the matching slice of `freqs_cis`
- Reshapes back to real

```python
def _apply_rope(self, x: torch.Tensor, start_pos: int, seqlen: int) -> torch.Tensor:
    dtype = x.dtype
    x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = self.freqs_cis[start_pos: start_pos + seqlen].view(1, seqlen, 1, -1)
    return torch.view_as_real(x_c * freqs).flatten(-2).to(dtype)
```

The mechanics: `reshape(..., -1, 2)` pairs consecutive coordinates `(x_{2i}, x_{2i+1})` — the RoPE pairing convention — and `view_as_complex` turns each pair into one complex number; multiplying by `freqs` rotates each pair by `θ_i · pos`; `view_as_real` + `flatten(-2)` restores the real layout. The key line is the broadcast:

```python
freqs = self.freqs_cis[start_pos: start_pos + seqlen].view(1, seqlen, 1, -1)
```

That view shape `(1, S, 1, d/2)` is what allows the same `freqs_cis` table to broadcast against Q shaped `(B, S, H, d/2)` (or in the KV-side path, `(B, S, 1, d/2)` because we add a head dim of 1 — the shared-key trick from §6). Two further notes:

- **FP32 upcast.** The `.float()` before the complex multiply means RoPE
  rotates in FP32 even when activations are BF16, then casts back. This is a small but deliberate numerical choice: complex multiplication chains the two real products and a sum, and doing that in BF16 would accumulate rotation error across positions.
- **Absolute-position indexing.** Both `q_pe` and `k_pe` are rotated by
  their *absolute* positions (`start_pos` offsets the table slice). The relative-position property of the scores — `⟨R(q, p_q), R(k, p_k)⟩` depends only on `p_q − p_k` — is what makes this safe; §6 derives it.

### Forward pass, end to end: tensor shapes

For reference, here is the complete shape flow of the SDPA path at the canonical config (`B=1`, `S=4`, `dim=768`, `H=12`, `d_nope=48`, `d_rope=24`, `R=192`, `d_v=64`, cache-free prefill). Every line maps to a statement in `MultiHeadLatentAttention.forward`:

| Step | Code | Tensor shape |
|---|---|---|
| 1. Query projection | `q = self.wq(x)` | `(1, 4, 864)` = `(B, S, H·72)` |
| 2. Head view | `q.view(bsz, seqlen, n_heads, qk_head_dim)` | `(1, 4, 12, 72)` |
| 3. Content/position split | `q.split([48, 24], dim=-1)` | `(1, 4, 12, 48)` + `(1, 4, 12, 24)` |
| 4. RoPE on queries | `q_pe = self._apply_rope(q_pe, start_pos, seqlen)` | `(1, 4, 12, 24)` |
| 5. Joint KV projection | `kv_a = self.wkv_a(x)` | `(1, 4, 216)` = `(B, S, R + d_rope)` |
| 6. Latent/rope-key split | `kv_a.split([192, 24], dim=-1)` | `(1, 4, 192)` + `(1, 4, 24)` |
| 7. Latent norm | `kv_normed = self.kv_norm(kv_latent)` | `(1, 4, 192)` |
| 8. RoPE on keys | `k_pe = self._apply_rope(k_pe_raw.unsqueeze(2), ...).squeeze(2)` | `(1, 4, 24)` |
| 9. Cache write | `kv_cache[:1, 0:4] = kv_normed.detach()` | slice of `(1, 2048, 192)` |
| 10. Absorb queries | `q_nope_h = q_nope.permute(2,0,1,3).reshape(h, bsz·seqlen_q, d)` | `(12, 4, 48)` |
| 11. Absorbed bmm | `q_nope_proj_h = torch.bmm(q_nope_h, wkv_b_k)` | `(12, 4, 192)` |
| 12. Back to batch layout | `q_nope_proj` (permute + contiguous) | `(1, 4, 12, 192)` |
| 13. Materialise K+V | `KV_nope_h = torch.bmm(ctx_kv_bmm, wkv_b_kv.transpose(-1,-2))` | `(12, 4, 112)` |
| 14. K/V split + layout | `K_nope`, `V` | `(1, 12, 4, 48)` + `(1, 12, 4, 64)` |
| 15. Full Q/K concat | `torch.cat([Q_nope, Q_rope], -1)` / `cat([K_nope, K_rope], -1)` | `(1, 12, 4, 72)` each |
| 16. Fused attention | `F.scaled_dot_product_attention(Q_full, K_full, V, ...)` | `(1, 12, 4, 64)` |
| 17. Output projection | `self.wo(attn.transpose(1,2).contiguous().flatten(2))` | `(1, 4, 768)` |

Steps 10–12 are the Direction-1 absorption algebra of §5 (computed unconditionally, consumed only by the manual path); steps 13–16 are Direction 2. The manual path diverges at step 13: instead of materialising K+V it computes `scores_content = self._per_batch_bmm(q_nope_proj, ctx_kv)` → `(1, 12, 4, 4)` directly from the latent, keeping the step-12 `(B, S_q, H, R)` tensors as the only per-query representation.

### KV Cache Management

For extended context lengths, the softmax scale is adjusted. The exact code (`models/mla.py:MultiHeadLatentAttention.__init__`) is:

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

> **Code-vs-paper note.** DeepSeek-V3's paper formula is `0.1·ln(rope_factor) + 1.0`; this repo's code now matches it (`mscale = 0.1·mscale_raw·log(rope_factor) + 1.0`). At the canonical `rope_factor = 1.0` the path is dormant (`mscale = mscale_raw = 1.0`), so it never fires in the 422M config.

**Semantics — two independent gates.** Note that the two `if`s test *different* conditions, and that matters:

| Condition | Effect |
|---|---|
| `rope_factor > 1.0` | makes `mscale` deviate from `mscale_raw` (YaRN temperature `t = 0.1·ln(f) + 1.0`, scaled by the config's `mscale`) |
| `max_seq_len > 4096` | makes `softmax_scale` deviate from the vanilla `qk_head_dim ** -0.5` by a factor of `mscale ** 2` |

The first gate alone (e.g. `rope_factor = 4` with `max_seq_len = 2048`) computes an `mscale` that is **never used** — the softmax scale stays vanilla because the second gate is closed. Conversely, `max_seq_len > 4096` with `rope_factor = 1.0` gives `mscale = 1.0`, so `mscale ** 2 = 1` and the scale is again vanilla. The YaRN path only *does* something when both gates are open.

**Why `mscale²`?** When RoPE frequencies are stretched by `rope_factor`, the argument `θ_i · pos` of every rotation grows, the rotated vectors decorrelate faster with distance, and the raw score distribution flattens (its scale changes relative to the training-time distribution). YaRN's fix is an attention "temperature": `t = 0.1·ln(s) + 1.0` for extension ratio `s`, applied so that the effective softmax temperature becomes `sqrt(d)·t`. The DeepSeek convention — as adopted here — enters the scale as `mscale²` multiplied onto the vanilla `1/sqrt(d)`; `mscale_raw` lets a caller tune the temperature by hand on top of the log-based default. `[INFERENCE]` The exact rationale for the square (rather than a single power) is the DeepSeek implementation convention; what is verifiable in this repo is that the formula is applied verbatim above and that the path is dormant at the canonical config.

**The scale is shared by all three paths.** `self.softmax_scale` is consumed identically by every attention implementation: the SDPA path passes it as `scale=self.softmax_scale` to `F.scaled_dot_product_attention`, the manual path multiplies `scores = (scores_content + scores_rope) * self.softmax_scale`, and `_forward_triton` forwards it as the `softmax_scale` scalar argument of `triton_mla_attention`. A change to `mscale` therefore propagates to every path with no per-path special-casing.

This prevents attention score underflow when the model is used beyond its original max sequence length.

### Decode vs Prefill: two attention regimes

MLA is called in two structurally different regimes, and the causal-masking contract differs between them. Understanding the contract is the key to the chunked-prefill fix described below.

**Decode (autoregressive, one token at a time).** `seqlen = 1`. The transformer (`models/transformer.py:Transformer.forward`) skips mask construction entirely (`if seqlen > 1: ... else: mask = None`), because a single query at global position `start_pos` can only ever see keys in `[0, end_pos)` — every cached key is at or before its own position, so `q >= k` holds for all pairs and the causal mask is the identity. The SDPA path runs with `attn_mask=None`; the Triton path runs with `is_causal=False` and no masking. Nothing to do, by construction.

**Prefill (a whole chunk at once).** `seqlen > 1`. Queries at positions `[start_pos, start_pos + seqlen)` must be masked against keys at positions `[0, end_pos)` — the causal condition is `q_global >= k_global`. Two cases:

- *Training / cache-free prefill* (`use_cache=False`): the KV context is the
  chunk itself, `start_pos = 0`, so the mask is the familiar lower-triangular `(1, 1, S, S)` additive mask.
- *Chunked prefill with cache* (`use_cache=True`): the KV context is
  `ctx_kv = kv_cache[:bsz, :end_pos]` — the *whole* prefix written so far, which is longer than the current chunk whenever `start_pos > 0`.

**The mask spans the whole KV context.** `Transformer._build_causal_mask` (`models/transformer.py:Transformer._build_causal_mask`) is the single source of truth:

```python
q = torch.arange(seqlen, device=device)[:, None] + start_pos   # GLOBAL query positions
k = torch.arange(kv_len, device=device)[None, :]               # GLOBAL key positions
mask = torch.where(q >= k, torch.zeros((), device=device),
                   torch.full((), float("-inf"), device=device))
```

and the call site chooses `kv_len = end_pos if use_cache else seqlen`. Two consequences. First, the mask is causal by **global** position: `q` is offset by `start_pos`, `k` runs to `kv_len = end_pos`, so a query at global position `g` is masked against keys at positions `> g` — including keys inside its own chunk. Second, the mask's `S_kv` axis is exactly the number of rows of `ctx_kv`, which is the shape contract the SDPA path relies on (`attn_mask = mask.expand(bsz, h, seqlen_q, -1)` must broadcast against `(B, H, S_q, S_kv)`).

**The Triton path receives the same contract as `q_start`.** The fused kernel builds its own causal mask in SRAM, so `_forward_triton` (`models/mla.py:MultiHeadLatentAttention._forward_triton`) translates the Python-level contract into the kernel's terms:

```python
is_causal = mask is not None
q_start = start_pos if (is_causal and use_cache) else 0
out = triton_mla_attention(..., is_causal=is_causal, q_start=q_start)
```

Inside `models/mla_triton.py:triton_mla_attention`, `q_start` offsets the in-kernel query row indices before the causal comparison

```python
causal_mask = (s_q_off[:, None] + q_start >= k_off[None, :])
```

so `s_q_off` (local to the query block) plus `q_start` reproduces the same global-position test as `_build_causal_mask`. The pure-PyTorch reference used by the CPU tests applies the identical rule (`models/mla_triton.py:mla_attention_reference`): `q_idx = arange(S_q) + q_start; k_idx = arange(S_kv); mask = q_idx >= k_idx`. `q_start` is passed through the autograd `Function` unchanged, so forward and backward see the same mask.

**Why `q_start` exists — the chunked-prefill correctness fix.** Before the 2026-08-04 fix, the Triton path masked queries as if they were local positions (`s_q_off` with no offset). For a cached mid-sequence prefill — say chunk `[1024, 1536)` with `S_q = 512` — the kernel would allow query 1024 to attend keys 1024–1535 inside its own chunk, **leaking future tokens** into the attention scores. The SDPA path had the dual failure: its mask was built over the chunk length only, so the `(B, H, S_q, S_chunk)` mask did not broadcast against the `(B, H, S_q, S_kv = end_pos)` tensors and the call crashed. Both failure modes are now closed by the same invariant, enforced in two places: the transformer's mask is `(1, 1, S_q, kv_len)` causal-by-global-position with `kv_len = end_pos` when caching (so SDPA is shape-correct), and the kernel offsets queries by `q_start` (so Triton is value-correct). `models/mla.py` passes `start_pos` straight through, and `Transformer._build_causal_mask` caches masks by `(seqlen, kv_len, start_pos, device)` so the common decode pattern (`start_pos` advancing by 1 each step) does not rebuild the mask every step.

**Regime summary:**

| Regime | `seqlen` | `mask` (Python) | Triton `is_causal` / `q_start` | What guards causality |
|---|---|---|---|---|
| Decode | 1 | `None` | `False` / 0 | nothing needed (single query ≤ all keys) |
| Prefill, cache-free | `S` | `(1,1,S,S)` lower-tri | `True` / 0 | mask / kernel both global with `start_pos=0` |
| Chunked prefill, cached | `S` | `(1,1,S,end_pos)` global | `True` / `start_pos` | mask + `q_start` agree by global position |
| Direct MLA call, no mask | any | `None` | `False` / 0 | reference falls back to triangular when `S_q == S_kv` |

The last row is the standalone-call convention of `mla_attention_reference`: when no mask is supplied but query and key lengths match, it applies a plain upper-triangular mask, so unit tests calling the kernel directly get causal behaviour by default.

---

## Comparison: MLA vs MHA vs GQA vs MQA

| Property | MHA | GQA (8 groups) | MQA | **MLA** |
|---|---|---|---|---|
| KV heads | `n_heads` | `n_groups` | 1 | 1 latent + 1 RoPE key |
| KV cache per token | `2 × n_h × d_h` | `2 × g × d_h` | `2 × d_h` | `d_c + d_h^R` |
| Cache ratio (vs MHA) | 1× | `1/n_groups` × | `1/n_h` × | **~0.02×** |
| Quality vs MHA | baseline | slight drop | measurable drop | **matches MHA** |
| Compute at decode | lowest | low | low | **~4× MHA** |
| Memory bandwidth | highest | medium | low | **lowest** |
| RoPE compatibility | native | native | native | requires decoupled |
| Training compatibility | native | native | native | native |

**Key insight**: MLA trades FLOPs for memory bandwidth. The attention computation is ~4× more expensive than MHA, but memory reads are ~30× cheaper. Since decode is overwhelmingly memory-bandwidth-bound at long contexts, MLA wins on throughput.

### Ablation results (from DeepSeek-V2 paper)

| Variant | PPL | KV cache |
|---|---|---|
| MHA baseline | 100% | 32,768 floats |
| GQA (8 groups) | +0.5 PPL | 2,048 |
| MQA | +1.5 PPL | 256 |
| **MLA** | **≤0.0 PPL** | **576** |

MLA matches MHA perplexity while GQA and MQA incur measurable degradation.

---

## Performance Characteristics

### When MLA wins

| Workload | Benefit |
|---|---|
| Long-context serving (32K–128K) | KV cache no longer dominates HBM |
| High-batch decode | More sequences fit per GPU; throughput up 3–5× |
| Edge / single-GPU, long context | Becomes feasible at small batch |
| MoE serving (DeepSeek-V3) | Frees HBM for expert weights |

### When MLA is not optimal

| Workload | Reason |
|---|---|
| Short-context (≤2K) | KV cache is small anyway; MLA's extra projections add latency |
| Pure compute-bound scenarios | The 4× extra compute hurts without bandwidth relief |
| Greenfield non-attention architectures | Linear attention or SSMs compress further |
| Existing pretrained models | Cannot retrofit MLA without retraining from scratch |

### Hardware implications

- **Without FlashAttention**: The materialised K/V in the SDPA path creates large intermediate tensors — the batch × heads × seqlen × d_head attention matrix must be written to HBM and read back
- **With FlashAttention-2/3**: The fused kernel never materialises the full attention matrix, making the SDPA path highly efficient
- **CUDA Graph compatibility**: The dynamic cache allocation complicates static CUDA graphs; prefill shared prompt prefixes once via `forward(..., start_pos=k, use_cache=True)` and replay only the decode steps to amortise this
- **`torch.compile`**: Supported out of the box; the critical paths (bmm, split, concat, SDPA) are inductor-friendly

---

## V3-Lite Architecture Deep Dives

The following sections provide concrete numerical analysis and step-by-step derivations specific to the DeepSeek-V3-Lite (422m) configuration. They complement the conceptual sections above with worked calculations.

### KV Cache Problem Analysis

During autoregressive decoding, every new token must attend to all preceding tokens. To avoid recomputing keys and values for every past position at each step, transformers store them in a **KV cache**.

**Standard MHA KV cache per token per layer:**

$$
\text{KV cache per token per layer} = 2 \times n_{\text{heads}} \times d_{\text{head}}
$$

For V3-Lite with 12 heads × 64 head_dim:

$$
2 \times 12 \times 64 = 1536 \text{ floats per token per layer}
$$

At 18 layers and 128K context:

$$
18 \times 131072 \times 1536 \times 2 \text{ bytes (BF16)} = 7.3 \text{ GB}
$$

**The real bottleneck is memory bandwidth.** During each decode step, the entire KV cache must be read from HBM into on-chip SRAM to compute attention scores. At 128K context, this means reading 7.3 GB per decode step — and this read dominates decode latency. MLA directly attacks this by reducing what's cached from 1,536 to 216 floats per token.

### Absorption Trick Derivation

The absorption trick is what makes MLA efficient at inference. Without it, reconstructing full K and V at every decode step would negate the cache savings.

**The key insight: matrix associativity.** The attention score between query $q$ and key $k$ is:

$$
\text{score} = q^T k = q^T (W^{UK} c^{KV})
$$

Since matrix multiplication is associative, we can re-parenthesize:

$$
\text{score} = (q^T W^{UK}) c^{KV} = q'^T c^{KV}
$$

where $q' = W^{UK\,T} q$ is the query projected into latent space.

**Score computation — naive vs absorbed:**

Naive (without absorption):
```
score = (c_q · W^{UQ})^T · (c_k^{KV} · W^{UK})
         ↑ expand Q         ↑ expand K from latent
```

With absorption:
```
score = c_q^T · (W^{UQ} · W^{UK^T}) · c_k^{KV}
         \_________________________/
             precompute once → W_absorbed
```

At inference: (1) precompute $W_{\text{absorbed}} = W^{UQ} \cdot W^{UK\,T}$ once at model load; (2) project the query $q' = W_{\text{absorbed}} \cdot c_q$; (3) compute score as $\text{score} = q'^T \cdot c_k^{KV}$ — a dot product in 192-dim latent space. **The full K (12 × 48 = 576 floats) is never materialized.**

**Value-side absorption** works identically — $W^{UV}$ is absorbed into the output projection $W^O$:

```
output = attn_weights · c^{KV} · (W^{UV} · W^O)
                          \________________/
                          precompute once → W_absorbed_v
```

**Impact table:**

| Step | Without absorption | With absorption |
|---|---|---|
| Score computation | Expand latent → full K (576 floats), then score | Latent-to-latent inner product (192 floats) |
| Value aggregation | Expand latent → full V (768 floats), attend, project back | Attend directly in latent space, project once |
| Memory reads per token | 1,536 floats (K + V) | 216 floats (latent + RoPE key) |

The absorption trick transforms MLA from a computationally expensive curiosity into the **most memory-efficient attention variant available**.

### Decoupled RoPE Rationale

RoPE encodes position by rotating query/key vectors in 2D subspaces. The crucial property: the inner product of two RoPE-rotated vectors depends only on the *relative* position. But this is also the problem — RoPE is **not a linear operation**, and it breaks the absorption algebra.

**Why RoPE breaks absorption:**

$$
\text{score} = q^T R(\theta, \Delta_{\text{pos}}) k
$$

If we try to absorb $W^{UK}$ into the query when RoPE is present:

$$
\text{score} = q^T W^{UK\,T} R(\theta, \Delta_{\text{pos}}) W^{UK} c^{KV}
$$

The rotation $R(\theta, \Delta_{\text{pos}})$ sits **between** $W^{UK\,T}$ and $W^{UK}$. Since $R$ depends on position, the product $W^{UK\,T} R W^{UK}$ **cannot be precomputed** — it changes for every query-key pair.

**The split-head solution:**

| Part | Dimension | RoPE? | Compressed? | Role |
|---|---|---|---|---|
| Content (`qk_nope_head_dim`) | 48 | No | Yes (via latent) | Semantic information |
| Position (`qk_rope_head_dim`) | 24 | Yes | No (separate path) | Positional information |

Total query/key head dim = 48 + 24 = 72. The content part goes through latent compression + absorption (no RoPE, so absorption works). The position part uses standard RoPE but operates as Multi-Query Attention (a single shared key head).

**The split score:**

$$
\text{score} = \underbrace{q_{\text{content}}^T \cdot k_{\text{content}}}_{\text{content (absorbed)}} + \underbrace{q_{\text{rope}}^T \cdot k_{\text{rope}}}_{\text{position (MQA)}}
$$

The two scores are simply added. This preserves RoPE compatibility without breaking the absorption algebra.

**Cache impact:** The decoupled RoPE key is shared across all heads (MQA-style), so only 24 extra floats per token are cached — not $12 \times 24 = 288$. The total cache is $192 + 24 = 216$ floats per token.

### MLA vs MHA vs GQA Comparison (V3-Lite Numbers)

| Property | MHA | GQA (4 groups) | MQA | **MLA** |
|---|---|---|---|---|
| KV cache per token | 1,536 | 512 | 128 | **216** |
| Cache ratio (vs MHA) | 1× | 3.3× reduction | 12× reduction | **7.1× reduction** |
| Quality vs MHA | baseline | slight drop | measurable drop | **matches MHA** |
| Compute at decode | lowest | low | low | ~4× MHA |
| Memory bandwidth | highest | medium | low | **lowest** |
| RoPE compatibility | native | native | native | requires decoupled |

MLA is the only mechanism that matches MHA quality while reducing cache. GQA and MQA both degrade quality — the more aggressive the compression, the worse the degradation. MLA's low-rank compression preserves full expressiveness because the up-projection $W^{UK}$ is learned, not a fixed sharing pattern.

### Dimension Summary (V3-Lite Complete Reference)

| Symbol | Value | Description |
|---|---|---|
| `dim` | 768 | Model hidden size |
| `n_heads` | 12 | Number of attention heads |
| `qk_nope_head_dim` | 48 | Content key/query dim per head (no RoPE) |
| `qk_rope_head_dim` | 24 | Positional key/query dim per head (with RoPE) |
| `qk_head_dim` | 72 | Total QK head dim (48 + 24) |
| `v_head_dim` | 64 | Value head dim |
| `kv_lora_rank` | 192 | KV compression latent dim |
| `q_lora_rank` | 0 | Query compression (disabled at this scale) |
| **KV cache per token** | **216** | 192 (latent) + 24 (RoPE key) |
| **MHA equivalent** | 1,536 | 2 × 12 × 64 (K + V) |
| **Reduction** | **7.1×** | vs MHA |

> **See also:** [Model Architecture](../concepts/foundations.md) for overall model topology and system design.

---

## Appendix A — FlashAttention in 60 seconds

This doc repeatedly references "FlashAttention-2 / SDPA fused kernel." If that phrase is opaque, read this.

**Standard attention** materialises a full `(S_q × S_k)` attention matrix in HBM:

```
scores = (Q @ K^T) * scale          # (B, H, S_q, S_k) — written to HBM
weights = softmax(scores)            # another HBM roundtrip
out = weights @ V                    # yet another
```

For `S_q = S_k = 8192` and `H = 12`, that matrix is `12 × 8192 × 8192` floats = ~770 MB **per layer, per forward**. The full attention matrix is never read back as a tensor — it's just an intermediate.

**FlashAttention** tiles the computation along the `S_k` axis. For each query block `(B_q rows)` and key block `(B_k rows)`:

1. Load Q block into SRAM.
2. For each K block: load K block, compute partial scores in SRAM,
   update running `(m_i, ℓ_i, o_i)` for online softmax (FlashAttention's trick for stable softmax across tiles).
3. Final `o_i` is the attention output for that query block.

The `(S_q × S_k)` matrix is never written to HBM — only the output `(S_q × d_v)` block is. **Memory traffic drops from O(S²) to O(S).**

`torch.nn.functional.scaled_dot_product_attention` (a.k.a. SDPA) is PyTorch's wrapper; on CUDA it picks FlashAttention-2 if available, falls back to a memory-efficient or math kernel otherwise. On Mac it picks the Metal-equivalent kernel.

**Why this matters for MLA.** The manual path *does* materialise the attention matrix (`attn = scores.softmax(...)`). For long contexts this is the killer. The SDPA path *avoids* that materialisation by calling `F.scaled_dot_product_attention` with materialised K and V — still the expensive K materialisation, but no attention matrix in HBM. The Triton path (§C) goes one step further: it never materialises K either, fusing the latent→K up-projection into the FlashAttention tile.

---

## Appendix B — A worked numerical example

Let's trace a single MLA forward pass on a deliberately tiny model: `dim=8, n_heads=2, qk_nope_head_dim=2, qk_rope_head_dim=2, v_head_dim=2, kv_lora_rank=3, max_seq_len=4, batch=1, seqlen=3`.

### Inputs

```
x = [[[1, 2, 3, 4, 5, 6, 7, 8],        # token 0
      [8, 7, 6, 5, 4, 3, 2, 1],        # token 1
      [1, 1, 1, 1, 1, 1, 1, 1]]]      # token 2
# shape: (1, 3, 8)
```

### Step 1 — Query projection

```
wq.weight shape: (2 * (2+2), 8) = (8, 8)
q = x @ wq.weight.T   # shape (1, 3, 8) — 2 heads × 4 dim each
q reshape to (1, 3, 2, 4)
q_nope = q[..., :2]   # (1, 3, 2, 2)  — content part
q_pe   = q[..., 2:]   # (1, 3, 2, 2)  — RoPE part
```

### Step 2 — KV compression

```
wkv_a.weight shape: (3 + 2, 8) = (5, 8)   # joint compression + RoPE key proj
kv_a = x @ wkv_a.weight.T   # shape (1, 3, 5)
kv_latent = kv_a[..., :3]   # (1, 3, 3) — latent, will be cached
k_pe_raw  = kv_a[..., 3:]   # (1, 3, 2) — RoPE key, will be cached

kv_normed = rmsnorm(kv_latent)        # (1, 3, 3), normalise per row
k_pe      = apply_rope(k_pe_raw)      # (1, 3, 2), rotated by position
```

**Cache write** (assuming `use_cache=True, start_pos=0`):

```
self.kv_cache[0, 0:3] = kv_normed.detach()   # (3, 3)  =  9 floats
self.pe_cache[0, 0:3] = k_pe.detach()         # (3, 2)  =  6 floats
```

Per token per layer: 3 + 2 = **5 floats** (the latent + RoPE key). Compare to MHA at this scale: 2 heads × 4 head_dim = 8 floats per token for K, plus another 8 for V = **16 floats**. So even at this tiny scale MLA caches 5/16 ≈ **31% of MHA**, a 3.2× reduction.

### Step 3 — Materialise K_nope, V (SDPA path)

```
wkv_b.weight shape: (n_heads * (qk_nope+v), kv_lora) = (2*4, 3) = (8, 3)
wkv_b reshape → (n_heads=2, qk_nope+v=4, kv_lora=3) = (2, 4, 3)
wkv_b_k = wkv_b[:, :2]   # (2, 2, 3)  — per-head key up-projection
wkv_b_v = wkv_b[:, 2:]   # (2, 2, 3)  — per-head value up-projection

For each cached latent (1, 3) — apply per-head up-projection:
   K_nope[h, t] = cached_latent[t] @ wkv_b_k[h].T    # (2,)
   V    [h, t] = cached_latent[t] @ wkv_b_v[h].T    # (2,)
```

So `K_nope.shape = (2, 3, 2)`, `V.shape = (2, 3, 2)`. After batch and head dim promotion for SDPA: `(B=1, H=2, S_k=3, d=2)` each.

### Step 4 — Material scores and softmax

```
Q_nope = q_nope.transpose(1, 2)             # (1, 2, 3, 2)
Q_rope = apply_rope(q_pe).transpose(1, 2)   # (1, 2, 3, 2)
K_rope = ctx_pe.unsqueeze(1).expand(...)    # (1, 2, 3, 2) — broadcast from (1, 3, 2)

Q_full = concat([Q_nope, Q_rope], dim=-1)   # (1, 2, 3, 4)
K_full = concat([K_nope, K_rope], dim=-1)   # (1, 2, 3, 4)

softmax_scale = qk_head_dim ** -0.5 = 4 ** -0.5 = 0.5

attn = sdpa(Q_full, K_full, V, scale=0.5)   # (1, 2, 3, 2)
```

For the manual path, the equivalent is `q_nope_proj = q_nope @ wkv_b_k` (per head) which puts the dot product in `R=3`-dim space instead of `d_nope=2`. The two paths agree mathematically (one is just a re-parenthesisation), but the manual path does ~50% more inner-product FLOPs (3 vs 2 dims).

### Step 5 — Output

```
attn reshaped → (1, 3, 4)          # concat H, d_v
output = attn @ wo.weight.T        # wo: (8, 4) → (1, 3, 8)
```

This `(1, 3, 8)` is the MLA output for the layer. Residue added, onto the next layer, the next block, etc.

**Take-away:** every operation has clear shapes, the cache stores a 5-float summary per token, and the materialisation step is the only place where MHA-style sizes appear — and that's the cost we're paying to make the SDPA kernel happy.

---

## Appendix C — The Triton path

`models/mla_triton.py` ships a fused kernel that **never materialises K at all** — it computes the up-projection `c^KV · W^{UK}` inside the attention tile, online. This is the production path on A100/H100 when `attn_impl="triton"` and `ENABLE_TRITON_KERNELS=1`.

### Why a custom kernel

The SDPA path materialises K to call `F.scaled_dot_product_attention`, which is good for compute (FA-2 tiling) but wastes memory bandwidth (K materialisation is `(B, H, S, d_nope)` per step). The manual path stays in latent space but uses a Python `for b in range(bsz)` loop, which is bad for GPU launch overhead. **The Triton path merges both strengths**:

- Up-projection happens *inside* the FlashAttention tile (per K-block,
  in SRAM), so K is never written to HBM.
- No Python loop — the kernel is one launch covering all batches and
  heads.

### Kernel contract

Inputs (per the wrapper in `models/mla.py:MultiHeadLatentAttention._forward_triton`):

| Arg        | Shape                          | Description                       |
|------------|--------------------------------|-----------------------------------|
| `q_nope`   | `(B, H, S_q, d_nope)`          | Q content, post-RoPE-free         |
| `q_pe`     | `(B, H, S_q, d_rope)`          | Q position, **already RoPE-rotated** |
| `ctx_kv`   | `(B, S_kv, R)`                 | Cached KV latent                  |
| `ctx_pe`   | `(B, S_kv, d_rope)`            | Cached RoPE key                   |
| `wkv_b_k`  | `(H, d_nope, R)`               | Per-head K up-projection          |
| `wkv_b_v`  | `(H, d_v, R)`                  | Per-head V up-projection          |
| `softmax_scale` | scalar                     | `1/sqrt(d)` (or `* mscale²`)      |
| `is_causal`| bool                     | True iff a mask was passed (`mask is not None`); enables in-kernel causal masking with `q_start` |

Output: `(B, S_q, H, d_v)` — the per-head attended values.

### Algorithm (per tile)

For each query block of size `BLOCK_T`:

```
Load Q_nope tile (BLOCK_T, d_nope), Q_pe tile (BLOCK_T, d_rope) into SRAM.
For each K/V block of size BLOCK_I along S_kv:
    Load ctx_kv tile (BLOCK_I, R) into SRAM.
    # Up-project tile-local — never write K to HBM
    K_nope_tile[h] = ctx_kv_tile @ wkv_b_k[h].T    # (BLOCK_I, d_nope), in SRAM
    V_tile    [h] = ctx_kv_tile @ wkv_b_v[h].T    # (BLOCK_I, d_v),    in SRAM
    K_full = concat([K_nope_tile, ctx_pe_tile], dim=-1)   # RoPE key from cache
    scores = (Q_nope · K_nope^T  +  Q_pe · ctx_pe^T) * scale   # in SRAM
    Update online-softmax state (m, ℓ, o) for this tile (FA-2 trick).
After all K/V tiles: write o / ℓ → final attention output.
```

This is essentially the FlashAttention-2 forward kernel with two augmentations:

1. **Online K materialisation:** the first matmul `ctx_kv · wkv_b_k[h].T`
   happens per tile, in SRAM.
2. **Two Q vectors per query block** (`q_nope`, `q_pe`) summed into a
   single score via two separate dot products — one against the tile-materialised K_nope, one against the cached RoPE key.

### Backward

Like FlashAttention-2, the backward pass **recomputes** the tile-local K_nope from the cached latent (which is held in HBM, gradient-free). This trades HBM-bandwidth-for-recompute: instead of stashing the materialised K in HBM for the backward, we recompute it. Net effect: lower peak memory, slightly more compute. The autograd `Function` exposes `forward` (saves `(q_nope, q_pe, ctx_kv, ctx_pe, wkv_b_k, wkv_b_v)`) and `backward` (re-materialises K_nope and V on the fly).

### Triton-only constraints

The kernel is gated by two env-level decisions:

- `HAS_TRITON` (set at import time, never raises) — controls whether
  the kernel module even loads.
- `ENABLE_TRITON_KERNELS` env-var — must be `1` to opt in. The default
  (`0`) keeps raw-PyTorch paths so that CPU/Mac test suites stay green.
- `attn_impl="triton"` in the config — required to dispatch into the
  fused kernel.

When all three are set, you get the full path. If `triton` import fails or the kernel reports an unrecoverable `ValueError`, the wrapper in `models/mla.py` (inside `MultiHeadLatentAttention.forward`) prints a one-time warning and falls back to SDPA for that model (`self.attn_impl = "sdpa"` after the first failure). The fallback is **silent per forward** after the first warning — so production logs won't spam.

---

## Appendix D — Training vs. inference pathway

| Aspect                | Training                                | Inference (autoregressive decode) |
|-----------------------|-----------------------------------------|------------------------------------|
| `use_cache`           | `False` (in the trainer)                | `True` (one token per step)        |
| `start_pos`           | 0 (full sequence in one call)           | `t` (cumulative position)          |
| Cache writes          | none (latents live in this call only)   | writes `t:t+1` slice each step     |
| Gradient flow         | through every up-projection             | not needed (no autograd)           |
| `attn_impl` choice    | `sdpa` (manual path is correctness ref) | `triton` for long contexts, else `sdpa` |
| `mask` shape          | `(1, 1, S_q, S_kv)` additive, causal by global position (`kv_len = seqlen` when cache-free) | `None` for single-token decode; `(1, 1, S_q, end_pos)` global mask for chunked prefill |
| Memory savings        | none on KV (no caching), but `qk_nope`  | full MLA benefit (cache)           |
|                      | up-projection is gradient-cheap in BF16 |                                    |

**Why the SDPA path dominates training even though it's "cache-blind":** training doesn't use a KV cache (full sequence is re-processed every step with proper gradients), so materialising K for one forward is cheap relative to all the matmuls upstream and downstream. The savings of the manual path — never materialising K — only pay off if you can reuse the latent across forwards, which only happens with a cache.

**Why `use_cache=False` matters in training.** When the trainer calls `forward(x)`, it sets `use_cache=False`. The code then takes the `else` branch of the cache block (where `ctx_kv = kv_normed`) and never touches `self.kv_cache` — `ctx_kv` and `ctx_pe` are simply the *current step's* latents. This guarantees:

- `requires_grad=True` flows through `kv_normed` and `k_pe` into
  `wkv_a`, `kv_norm`, and the RoPE helpers.
- No stale autograd graph pinned in the cache (which would also be
  a leak during the multi-step backward).
- `kv_cache` and `pe_cache` stay `None` (memory-zero) for that
  training step.

**Why the manual path is the unit-test reference.** Tests can't easily distinguish "SDPA math is wrong" from "kernel is fast but slightly imprecise" if they only test SDPA. The manual path is the literal absorption algebra, so a `test_mla_parity.py` style test compares both and forces agreement within `atol=1e-3` in FP32.

---

## Appendix E — Why `wkv_b` has shape `(H, d_nope + d_v, R)`

This is a subtle design choice. Naively, K and V have separate up-projection matrices:

```
W^{UK} : (H · d_nope, R)
W^{UV} : (H · d_v,   R)
```

This repo combines them into a single `wkv_b.weight` of shape `(H · (d_nope + d_v), R)` and *views* it as `(H, d_nope + d_v, R)` to split per head. Why?

1. **Single nn.Linear registration.** PyTorch's `nn.Linear` stores
   weights as a 2D matrix `(out, in)`. Two separate `nn.Linear`s would create two parameters, two optimiser entries, and two separate kernels for forward/backward. One fused `Linear` is faster.

2. **Fused K/V materialisation.** The SDPA path can compute
   `K_nope, V = split(ctx_kv @ wkv_b_kv.T)` in a single bmm. The Triton path exploits the same layout (per-tile up-projection with both K and V slices).

3. **No quality difference.** Concatenating two unrelated projection
   matrices along the output dim is algebraically identical to applying them separately. There's no "cross-talk" because the K and V columns are sliced at the boundary, not mixed.

The trade-off: the K and V up-projections share a *weight initialisation distribution* but learn independently. This is fine — they're independent parameters anyway.

---

## Appendix F — KV-cache lifecycle (state diagram)

```
┌──────────────────┐
│  Layer.__init__  │
│  cache=None,     │
│  _cache_batch=0  │
└────────┬─────────┘
         │ first forward(use_cache=True)
         ▼
┌──────────────────────────────────────────────┐
│  _ensure_cache(bsz, device, dtype)           │
│   - alloc kv_cache (B, max_seq_len, R)       │
│   - alloc pe_cache (B, max_seq_len, d_rope)  │
│   - B = max(bsz, prev*2, 16)                 │
└────────┬─────────────────────────────────────┘
         │ every forward with use_cache=True
         ▼
┌──────────────────────────────────────────────┐
│  self.kv_cache[:bsz, start_pos:end_pos] =    │
│       kv_normed.detach()                     │
│  self.pe_cache[:bsz, start_pos:end_pos] =    │
│       k_pe.detach()                          │
│                                              │
│  ctx_kv = self.kv_cache[:bsz, :end_pos]      │
│  ctx_pe = self.pe_cache[:bsz, :end_pos]      │
└────────┬─────────────────────────────────────┘
         │ end of generation (or new request)
         ▼
┌──────────────────┐
│  reset_cache()   │ ──► cache=None, _cache_batch=0
└──────────────────┘

   ┌───────────────────────────────────────────┐
   │ forward(x, start_pos=k, use_cache=True)   │  arbitrary-offset write
   │   → kv_cache[:bsz, k:k+seqlen] = ...      │  (= "prefix caching" here;
   └───────────────────────────────────────────┘   no prefill_cache API exists)

   Reallocation note: if a later forward arrives with bsz > _cache_batch,
   _ensure_cache allocates a FRESH (max(bsz, 2·_cache_batch, 16), …) tensor —
   the old rows are dropped, so the batch size must be fixed per request.
   Exceeding max_seq_len along the sequence axis is a RuntimeError, not a grow.
```

**Common pitfalls:**

- Forgetting `reset_cache()` between requests in a serving loop → next
  request reads garbage from the previous request's positions.
- Forgetting `use_cache=False` during training → cache fills with
  detached latents that have no grad and no replay, silently making the model "single-step" only.
- Growing the batch mid-request → `_ensure_cache` reallocates and the
  prefix latents vanish (fresh zeros), silently corrupting the request.
- Allocating cache with `max_seq_len=2048` and then trying to
  decode past 2048 → `RuntimeError: end_pos exceeds max_seq_len`.
- Multi-GPU: cache lives on each device, no cross-process sharing.
  For tensor-parallel, each rank holds its own cache shard (the caching is naturally partitioned along the head dim).

---

## Appendix G — Gradient flow in MLA

The gradient story follows the same three-way split as the forward paths, and it is anchored in one method: everything below is the autograd graph produced by `models/mla.py:MultiHeadLatentAttention.forward` under different `use_cache` / `attn_impl` combinations. Two structural facts about `forward` shape every mode:

1. **The absorption projection is computed unconditionally.** The lines
   `q_nope_h → q_nope_proj_h = torch.bmm(q_nope_h, wkv_b_k) → q_nope_proj` run *before* the `attn_impl` dispatch, for every path. In the SDPA and Triton paths the result is never consumed by the output, so autograd simply never reaches it: the loss has no path back through `q_nope_proj`, and backward prunes it as a dead branch (no gradient is produced for `wkv_b_k` through this route, and the bmm costs FLOPs but nothing else). Only the manual path consumes it.
2. **The cache is the autograd boundary.** `kv_normed.detach()` and
   `k_pe.detach()` sever the graph before the slice-write. Gradients flow into `kv_normed` (and from there to `wkv_a` / `kv_norm`) only through *this step's* latents, never through the cache. The `.detach()` is what makes "the loss of step t+1 depends only on step t+1's parameters" true.

### Mode 1 — Training, SDPA path (`use_cache=False, attn_impl="sdpa"`)

```
loss
 ↑ dL/d_output
wo            ← dL/dwo from loss.backward() (standard Linear backward)
 ↑
attn          ← SDPA backward (FA-2 recompute if applicable)
 ↑   ↑   ↑
Q_full, K_full, V
 ↑   ↑   ↑
wo   cat[K_nope, K_rope]  ← cat backward splits into Q_nope/Q_rope/K_nope/K_rope
wkv_b    ← dL/dwkv_b from the fused KV bmm (KV_nope_h = bmm(ctx_kv_bmm, wkv_b_kvᵀ))
wkv_a    ← dL/dwkv_a from kv_a = wkv_a(x)
kv_norm  ← dL/dkv_norm from kv_normed = kv_norm(kv_latent)
```

All parameters receive gradients; the autograd graph is one connected component from `loss` to `wkv_a.weight`. `wkv_b.weight` receives its gradient through the single fused bmm `torch.bmm(ctx_kv_bmm, wkv_b_kv.transpose(-1, -2))`, where `wkv_b_kv = torch.cat([wkv_b_k, wkv_b_v], dim=1)` is a cat of two views of the same parameter — the cat backward routes `dL/dKV_nope_h` back into the correct 48/64-column slices automatically. The cache is `None` (or untouched), so no back-edge through it. Note also that `use_cache=False` means `ctx_kv = kv_normed` (the live tensor, not a cache slice), which is what lets gradients reach `kv_norm` and `wkv_a` in the first place.

### Mode 2 — Training, manual path

Same overall graph, but:

- `q_nope_proj = torch.bmm(q_nope_h, wkv_b_k)` *is* on the loss path here
  (via `scores_content`), so `wkv_b.weight` receives **two** gradient contributions in manual mode: one through `wkv_b_k` (from `q_nope_proj_h = q_nope_h @ wkv_b_k`) and one through `wkv_b_v` (from `out_v = out_h @ wkv_b_v.transpose(-1, -2)`). They accumulate into the same parameter's `.grad`; the slices are disjoint (columns `[:48]` vs `[48:]`), so there is no double-counting of any single entry.
- The Python `for b in range(bsz)` loop in `out_latent` accumulation
  produces a sequence of bmm backwards, each with its own kernel launch — slow but correct.
- The value-side chain is `out_latent → bmm(attn[b], ctx_kv[b]) → attn →
  softmax → scores_content + scores_rope`, so gradients to `q_nope` arrive through both the content bmm (in `R`-space, via `q_nope_proj`) and the RoPE bmm (in `d_rope`-space, via `q_pe`).

### Mode 3 — Inference (`use_cache=True`)

No autograd: generation runs under `@torch.inference_mode()` (see `models/transformer.py:Transformer.generate` and `inference/generate.py:generate_interactive`), which is stricter than `no_grad` — it also disables the version counters autograd uses to detect in-place modification. Gradients are zero for every parameter, including `wkv_a` and `wkv_b`. The cache `detach()` calls are belt-and-suspenders: even if a user calls `forward` without an inference/no-grad context, the cache won't accumulate graph edges across steps (see "Why `.detach()` is non-negotiable" in §9).

### Mode 4 — Training, Triton path (`attn_impl="triton"`)

The fused kernel is wrapped in a `torch.autograd.Function` (`models/mla_triton.py:triton_mla_attention`), and its backward is a **correctness-first v1 stub**: it re-runs the pure-PyTorch reference (`mla_attention_reference`) and lets autograd differentiate it:

```python
out_ref = mla_attention_reference(q_nope=q_nope, q_pe=q_pe, ctx_kv=ctx_kv, ...)
grads = torch.autograd.grad(out_ref, [q_nope, q_pe, ctx_kv, ctx_pe, wkv_b_k, wkv_b_v], grad_outputs=dout)
```

Because the reference reproduces the SDPA algebra exactly, the gradients match the SDPA path — this is asserted by `tests/test_mla_triton.py` — at the cost of forfeiting the fused-kernel speedup in backward. The forward is fused and fast; the backward is reference and slow; correctness is guaranteed in both. (The docstring in the kernel file marks the fused recompute-backward as the obvious next optimisation.)

**Why this matters for `torch.compile`.** The SDPA path's gradient graph is essentially one big matmul stack with one `cat` and one SDPA node — inductor-friendly. The manual path's `for b in range(bsz)` loop **breaks inductor's tracing**, because Python control flow inside a compiled region forces graph breaks. In practice, the manual path is used only as a correctness reference and CPU fallback; production training uses SDPA (or Triton on A100/H100).

---

## Appendix H — Numerical stability notes

| Concern                       | Mitigation in this repo                                  |
|-------------------------------|----------------------------------------------------------|
| Softmax overflow in BF16      | `softmax(dtype=torch.float32)` then cast back            |
| `kv_latent` scale drift       | `RMSNorm(kv_lora_rank)` between compression and cache    |
| RoPE frequency extreme values | `inv_freq = base^{-2i/d}` naturally bounded in [1/base, 1] |
| Long-context attention underflow | `softmax_scale *= mscale^2` when `max_seq_len > 4096`  |
| BMM accumulation in BF16      | PyTorch's bmm accumulates in FP32 by default on Ampere+  |
| `torch.compile` numerics      | `torch.float32` softmax-dtype guard prevents re-compile drift |

### `mscale` math

```
mscale = mscale_raw                                  (rope_factor <= 1.0)
mscale = 0.1 · mscale_raw · log(rope_factor) + 1.0   (rope_factor > 1.0 — matches the paper)
```

If `rope_factor = 1.0` (the canonical config), `mscale = mscale_raw = 1.0` and `softmax_scale = 1/sqrt(d)` as usual. For `rope_factor > 1.0`, the code now includes the paper's `+1.0` (this was corrected in the 2026-08-04 session; earlier versions of this chapter described a formula without it). With the default `mscale_raw = 1.0`, `mscale > 1` for any `rope_factor > 1` (e.g. `mscale ≈ 1.39` at `rope_factor = 50`), so `mscale² > 1` and — only when `max_seq_len > 4096` — `softmax_scale` *increases*: the YaRN temperature is a widening correction that compensates for the flattened score distribution of stretched RoPE frequencies. The path is **dormant and unverified** at the 422M config and should be exercised before extending context.

**At this repo's 422M config, `rope_factor = 1.0`**, so the YaRN path is dormant. The mscale machinery exists for future long-context fine-tuning and serves as the on-ramp if someone wants to extend the model to 8K or 16K context.

### Dtype policy

- **Linear weights:** BF16 (matches input).
- **Activations:** BF16 throughout, with explicit `.float()` upcasts
  before `softmax` and `.to(x.dtype)` downcasts after.
- **KV cache:** same dtype as input (BF16 in production). Storing FP32
  cache would double memory cost and force a dtype-aware SDPA call.
- **Gradients:** FP32 master weights in AdamW (outside MLA's scope).

---

## Appendix I — Glossary

| Symbol            | Meaning                                                    |
|-------------------|------------------------------------------------------------|
| `c_t^KV`          | Compressed KV latent for token t (cached)                  |
| `k_t^R`           | Decoupled RoPE key for token t (cached)                    |
| `k_t^C`           | Reconstructed content key (NOT cached — derived)           |
| `v_t^C`           | Reconstructed value (NOT cached — derived)                 |
| `q_t^C`           | Query content part                                         |
| `q_t^R`           | Query position part (RoPE-rotated)                         |
| `W^{DKV}`         | KV down-projection (compression matrix)                    |
| `W^{UK}`          | Key up-projection (per-head content recovery)              |
| `W^{UV}`          | Value up-projection (per-head value recovery)              |
| `W^{KR}`          | Decoupled RoPE key projection (shared K head)              |
| `W^{DQ}`, `W^{UQ}`| Query-side LoRA-style compression                          |
| `R` / `kv_lora_rank` | Latent dimension (192 in 422M, 512 in V3-671B)           |
| `d_nope`          | Per-head content dimension (48 in 422M, 128 in V3)         |
| `d_rope`          | Per-head RoPE dimension (24 in 422M, 64 in V3)             |
| `qk_head_dim`     | `d_nope + d_rope` per head (72 in 422M, 192 in V3)         |
| `v_head_dim`      | Per-head value dim (64 in 422M, 128 in V3)                 |
| `H` / `n_heads`   | Number of attention heads (12 in 422M, 128 in V3)          |
| `mscale`          | YaRN attention-temperature multiplier                      |
| `rope_factor`     | Position-frequency scale; 1.0 = standard, >1 = YaRN       |
| `use_cache`       | Whether to read/write the KV cache during this forward     |
| `start_pos`       | Position of the first token in the current call            |
| `end_pos`         | `start_pos + seqlen` — end of the slice being written      |
| `ctx_kv`          | Slice of the KV cache covering positions `[0, end_pos)`    |
| `ctx_pe`          | Slice of the RoPE cache covering positions `[0, end_pos)`  |
| `wkv_b`           | Combined `(W^{UK}, W^{UV})` parameter                       |
| `q_lora_rank`     | Query-side compression dim (0 = no compression)            |

---

## Appendix J — Frequently Asked Questions

### Q1. Why doesn't MLA also compress Q by default?

It can — `q_lora_rank > 0` enables the Q compression path (`wq_a → q_norm → wq_b`, see `MultiHeadLatentAttention.__init__` in `models/mla.py`). At DeepSeek-V3 scale, `q_lora_rank = 1536`. The 422M config uses `q_lora_rank = 0` because the smaller model has too little capacity for an extra bottleneck to pay off. This is a deliberate simplification, not a deficiency.

### Q2. Can I retrofit MHA → MLA on a pretrained model?

No. The latent dimension, the up-projection matrices, the decoupled RoPE path, and the `wkv_b` split are all learned during pretraining. You can *initialise* an MLA layer to mimic MHA (`R = n_heads × d_head`, `W^{UK} = identity`, `W^{UV} = identity`-like), but the resulting quality is far below what MHA achieves with the same param count. The architecture choice is binding at pretraining time.

### Q3. What's the actual compute overhead of MLA vs MHA?

Per token per layer, per query-key inner product:

- MHA: `d_h` (e.g., 64) FLOPs.
- MLA manual path: `R` (e.g., 192) FLOPs → ~3× more.
- MLA SDPA path: `d_nope` (e.g., 48) FLOPs (same as MHA per inner
  product) + the cost of materialising K first.

So MLA's manual path is ~3-4× more compute than MHA at the inner product, but its KV bandwidth is `~30×` less at long context. Net: MLA wins at long context (memory-bound), loses at short context (compute-bound and KV-cache small).

### Q4. Why is the RoPE key shared (single head) and not per-head?

Because of the absorption algebra. If each head had its own RoPE key, the absorption trick would break — you'd need a different `W^{UQ} R(θ, Δ) W^{UK}` per head, and the precomputation savings disappear. Sharing the RoPE key across all heads is the *only* way to keep the up-projection matrix constant in `Δ`. This is the "decoupled" in decoupled RoPE.

### Q5. What happens if `rope_factor = 1.0`?

YaRN is dormant. `mscale = 1.0`, `softmax_scale = 1/sqrt(qk_head_dim)`, no attention-temperature adjustment. This is the canonical training configuration in this repo.

### Q6. Why does `wo` (output projection) take input of shape `(B, S_q, H * d_v)`?

Because after attention, the per-head value outputs are concatenated along the head dimension. PyTorch's `nn.Linear` operates on the last dim, so the output projection sees a single tensor of shape `(B, S_q, H * d_v)` and projects to `(B, S_q, dim)`. There's no "per-head output projection" — heads are mixed at this step.

### Q7. What if `max_seq_len` is exceeded mid-decode?

`_extend_rope` returns silently if `seq_len ≤ self._rope_seq_len`. The `forward` method has an explicit `RuntimeError` for `end_pos > self.max_seq_len`. The KV cache allocation is `max_seq_len`-bounded, so exceeding it is a hard error, not a silent overflow.

### Q8. Why is `kv_norm` an `RMSNorm` rather than `LayerNorm`?

RMSNorm is parameter-free (no bias, no learned scale) — cheaper and empirically equivalent for this use case. Both apply a normalisation before the cached latent is stored, keeping its scale consistent across tokens.

### Q9. Can the manual path run on the same batch size as SDPA?

Yes — the only constraint is GPU memory for the latent-space attention matrix `(B, H, S_q, S_k)`. For 422M at `B=4, H=12, S=2048`, that's `4·12·2048·2048 ≈ 200M` floats ≈ 400 MB in FP32 — substantial but fits on a 24 GB GPU. Smaller-batch use is fine on CPU.

### Q10. Does MLA support sliding-window or sparse attention?

Not natively. The MLA layer is dense: every query attends to every key (within causal mask). Sparse patterns require either custom Triton kernels or post-hoc sparsification, neither of which is in this repo.

---

## Check Your Understanding

### Q1. The two directions of the absorption trick

In exact arithmetic, why do the manual path and the SDPA path of `MultiHeadLatentAttention.forward` produce identical scores? And where — if anywhere — do they disagree in floating point?

**Answer.** The identity is pure associativity of matrix multiplication: $q U_h c_t^\top = (q U_h) c_t^\top = q (U_h c_t^\top)$. Direction 1 (manual) forms $\tilde q = q U_h$ once per query and contracts $R=192$ dims per score; Direction 2 (SDPA) forms $k^C = c_t U_h^\top$ once per key and contracts $d_{\text{nope}}=48$ dims. In exact arithmetic the values are identical; in floating point the two accumulation orders round differently, which is why the parity tests use `atol=1e-3` in FP32 and why BF16 can drift to 1e-2 on long contexts. Neither path is "more correct".

### Q2. The cache and a growing batch

You are serving a request with `bsz=4`. Mid-request, a caller invokes the same layer with `bsz=8` before `reset_cache()`. What happens to the prefix latents, and what is the correct usage contract?

**Answer.** `_ensure_cache` decides `need_alloc` because `bsz > _cache_batch` and allocates a *fresh* `(max(8, 2·4, 16), max_seq_len, R)` zero tensor — it never copies the old rows. The prefix latents are silently gone, and the next attention reads zeros for every earlier position. The contract is: choose the batch on the first `use_cache=True` call and keep it fixed until `reset_cache()`. The doubling policy only pays off for growth *between* requests.

### Q3. Chunked prefill and causality

A cached prefill processes chunk `[1024, 1536)` (`seqlen = 512`, `start_pos = 1024`, `use_cache = True`). What mask does the SDPA path see, what does the Triton kernel see, and how do both stop query 1024 from attending key 1100?

**Answer.** The transformer builds `mask = _build_causal_mask(512, 1536, 1024)` — a `(1, 1, 512, 1536)` additive mask comparing *global* positions (`q = arange(512) + 1024`, `k = arange(1536)`), so the SDPA path sees `attn_mask` with `-inf` exactly where `q_global < k_global`, including `(1024, 1100)`. The Triton path does not receive the mask; it receives `is_causal=True` and `q_start=1024` (set by `_forward_triton`), and the kernel tests `s_q_off + q_start >= k_off`, i.e. the same global comparison. Both mechanisms agree because both use absolute positions; the query at global 1024 is masked against every key at global position > 1024.

### Q4. The dead absorption bmm

In the SDPA path, `q_nope_proj_h = torch.bmm(q_nope_h, wkv_b_k)` executes on every forward, yet training produces no gradient for `wkv_b_k` through it. Why?

**Answer.** `q_nope_proj` is only consumed by the manual path. In the SDPA path the loss has no path back to it (the scores come from `F.scaled_dot_product_attention` on the materialised K/V), so autograd's backward pass prunes the branch as dead: the bmm costs FLOPs but contributes nothing to any `.grad`. `wkv_b.weight` receives its SDPA-path gradient through the fused materialisation bmm (`KV_nope_h = torch.bmm(ctx_kv_bmm, wkv_b_kv.transpose(-1, -2))`) instead.

---

## Implementation Checklist

To verify a correct MLA implementation, check these invariants:

1. **Cache size**: `kv_lora_rank + qk_rope_head_dim` floats per token per layer. Never `n_heads * v_head_dim`.
2. **Content-position split**: `qk_nope_head_dim + qk_rope_head_dim == qk_head_dim`. No overlap, no gap.
3. **Shared RoPE key**: `k_pe` is produced once per layer (not per head) and expanded to all query heads.
4. **Cache detach**: Latents written to cache always carry `.detach()` to prevent cross-forward autograd leaks.
5. **Weight absorption in inference**: The SDPA path should be functionally equivalent to the manual path at inference (test by comparing outputs).
6. **Gradient flows through cache**: During training (`use_cache=False`), gradients flow through the latent and up-projection paths correctly without caching.

---


1. **DeepSeek-V2** (May 2024) — *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*.
   [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)
   — Original introduction of MLA.

2. **DeepSeek-V3** (Dec 2024) — *DeepSeek-V3 Technical Report*.
   [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
   — Refined MLA with auxiliary-loss-free MoE and MTP.

3. **Chris McCormick** (Apr 2025) — *The Inner Workings of Multihead Latent Attention (MLA)*.
   [mccormickml.com](http://mccormickml.com/2025/04/26/inner-workings-of-mla/)
   — Excellent deep-dive on the algebra and interpretability of MLA.

4. **tutorialQ / Mahi Mullapudi** (Apr 2026) — *Multi-Head Latent Attention (MLA) — KV-Cache Compression*.
   [tutorialq.com](https://tutorialq.com/ai/dl-foundations/multi-head-latent-attention)
   — Practical overview with code sketch and cache-size calculator.

5. **Hardware-Centric Analysis of DeepSeek's MLA** (2025) — *Hardware-Centric Analysis of DeepSeek's Multi-Head Latent Attention*.
   [arXiv:2506.02523](https://arxiv.org/abs/2506.02523)
   — Detailed analysis of MLA computation orders and hardware efficiency.

6. **PyTorch torchtitan** — Reference implementation in PyTorch's distributed training framework.
   [github.com/pytorch/torchtitan](https://github.com/pytorch/torchtitan)

7. **DeepSeek-V3-Lite** — This repo. 422M faithful reimplementation.
   [github.com/atandra2000/DeepSeek-V3-Lite](https://github.com/atandra2000/DeepSeek-V3-Lite)

---

## 06 — FP8 Mixed Precision Training & Quantization

> **Canonical** for the DeepSeek-V3 FP8 scheme: E4M3 format for both forward and backward GEMMs, tile-wise and block-wise fine-grained scaling, and the high-precision accumulation that makes 8-bit training stable. Educational textbook chapter — from-scratch, with the math and the intuition.

> How DeepSeek-V3 cuts matmul memory bandwidth and Tensor Core throughput in half using 8-bit floats, and why a single global scale factor fails. **Status in this repo:** FP8 is **paper-spec only — not implemented.** This reproduction trains in BF16 throughout (see [08 Training Pipeline](../training.md)). This chapter exists so the portfolio documents the technique that makes full-scale DeepSeek-V3 feasible; at 411.6M params on a single A100, BF16 is already compute-efficient and FP8's complexity is not justified.

**Depends on:** [02 Model Architecture](../concepts/foundations.md), [03 Multi Head Latent Attention](../concepts/attention-and-precision.md) · **Read next:** [07 DualPipe Parallelism](../concepts/parallelism.md)

---

## 0. Status in this repo

| Aspect | DeepSeek-V3 (paper) | DeepSeek-v3-Lite (this repo) |
|---|---|---|
| Forward GEMM dtype | FP8 E4M3 | BF16 |
| Backward GEMM dtype | FP8 E4M3 (uniform; no E5M2) | BF16 |
| Weight scaling | block-wise `128×128` | n/a (BF16) |
| Activation scaling | tile-wise `1×128` | n/a (BF16) |
| Master weights / optim | FP32 | FP32 (AdamW master + m + v) |
| Reason to skip | — | 411.6M params fits BF16 on one A100; FP8's stability engineering is disproportionate at this scale |

The rest of this chapter explains the V3 scheme so the technique is documented; it is not a description of running code. `[INFERENCE]` marks derived figures not measured here.

---

## 1. Why 8-bit floats — the bandwidth wall

A matmul `Y = X W` is memory-bound for anything but the largest GEMMs: the time is dominated by **loading** `X` and `W` from HBM, not by the multiply-adds. Halving the storage dtype halves the bytes moved:

| dtype | bytes/element | Relative bandwidth |
|---|---|---|
| FP32 | 4 | 1.0× |
| BF16 / FP16 | 2 | 2.0× |
| **FP8 (E4M3 / E5M2)** | **1** | **4.0×** |

Modern Tensor Cores (Hopper `fp8` tensor cores) also issue twice the FLOPs/s in FP8 vs BF16. So FP8 is a ~2× throughput and ~2× memory win simultaneously — the single largest engineering lever in DeepSeek-V3's training cost reduction.

The catch: 8 bits give you either **range** (E5M2: ±57,344) or **precision** (E4M3: ±448 with 3 mantissa bits), not both. LLM activations and gradients have outlier channels that a single 8-bit scale cannot represent without either saturating the outliers or quantizing the inliers to zero. That tension is the entire problem FP8 training solves.

### 1.1 The quantization error, precisely

Let $x$ be a real value and $S$ the scale. The quantization maps

$$x \mapsto \hat{x} = \text{clip}\!\left(\text{round}\!\left(\frac{x}{S}\right), -M, M\right) \cdot S$$

where $M$ is the format's maximum integer magnitude ($448$ for E4M3, $57344$ for E5M2). Two error regimes:

- **Rounding error** (inliers): $\hat{x} = x + \epsilon$ with $|\epsilon| \le S/2$. As a **relative** error, $\epsilon/x \le S/(2|x|)$. For a value of magnitude $|x|$, the representable resolution is $S$; relative error grows as values shrink toward zero.
- **Saturation error** (outliers): any $|x| > MS$ clips to $\pm MS$, a relative error of $|x|/(MS) - 1$ that can be arbitrarily large.

The scale is therefore a trade: raise $S$ to cover outliers (and lose inlier precision), or lower $S$ for inlier fidelity (and clip more values). For a fixed budget of $b$ mantissa bits, the number of distinct representable values in $[-MS, MS]$ is fixed — so the *relative* precision of typical values is what fine-grained scaling protects.

---

## 2. The two FP8 formats

```
E4M3  (DeepSeek-V3 forward AND backward)       Range: [-448, 448]
┌──────┬─────────────┬──────────────┐
│ sign │ exponent 4b │ mantissa 3b  │   More mantissa → finer steps,
└──────┴─────────────┴──────────────┘   small range. Good for values near 1.

E5M2  (common alternative, NOT used in V3)     Range: [-57344, 57344]
┌──────┬─────────────┬──────────────┐
│ sign │ exponent 5b │ mantissa 2b  │   More exponent → huge range,
└──────┴─────────────┴──────────────┘   coarse steps.
```

| Property | BF16 | FP8 E4M3 | FP8 E5M2 |
|---|---|---|---|
| Total bits | 16 | 8 | 8 |
| Exponent | 8 | 4 | 5 |
| Mantissa | 7 | 3 | 2 |
| Dynamic range | wide | narrow (±448) | wide (±57,344) |
| Mantissa precision | medium | high (3 bits) | low (2 bits) |
| Used for | master weights, RMSNorm | **all V3 GEMMs** | (not used by V3) |

`★ Insight ─────────────────────────────────────` Naive intuition says gradients are heavy-tailed and need E5M2's range. DeepSeek-V3's empirical answer is the opposite: **E4M3 everywhere**, with fine-grained per-block scaling doing the range work that the exponent field would otherwise do. The official DeepGEMM repository states this explicitly: "We don't support e5m2 because it isn't used in DeepSeek V3/R1." Because scales are per-tile/per-block (not per-tensor), each local block can pick an $S$ that fits *its* values, so a global dynamic range is unnecessary — the 3 mantissa bits of E4M3 (twice the resolution of E5M2) are worth more than a wide exponent range that scaling already provides. `─────────────────────────────────────────────────`

### 2.1 The quantize/de-quantize primitive

Every FP8 GEMM is really a dequant→matmul→quant dance with the scale folded in:

$$X_{\text{FP8}} = \text{clip}\!\left(\text{round}\!\left(\frac{X}{S}\right), -448, 448\right), \qquad \hat{X} = X_{\text{FP8}} \times S$$

The scale `S` is the whole game. The question FP8 training answers is: **what is `S`, and can one `S` work for a whole matrix?** (Spoiler: no.)

### 2.2 E4M3 bit patterns, worked

E4M3: 1 sign + 4 exponent + 3 mantissa bits, bias 7 (like BF16's bias but fewer exponent bits). The value is

$$(-1)^s \times 2^{e-7} \times (1.m_2)$$

with exponent `e` in `[1, 14]` (subnormals `e=0` have no implicit leading 1 and represent `2^{-7} \times 0.m_2`). Worked encodings:

| Value | sign | exponent (4b) | mantissa (3b) | notes |
|---|---|---|---|---|
| 1.0 | 0 | 0111 (7) | 000 | $2^{0} \times 1.0$ |
| 1.25 | 0 | 0111 | 010 | $2^{0} \times 1.25$ |
| 0.5 | 0 | 0110 (6) | 000 | $2^{-1}$ |
| 448 | 0 | 1111 (15) | 111 | max normal: $2^{8} \times 1.875$ |
| 0.0078125 | 0 | 0000 | 001 | smallest subnormal: $2^{-7} \times 0.125$ |

Two consequences the scaling math leans on:

- **Granularity is exponential, not uniform.** Between 1.0 and 2.0 the step is $2^{-3} = 0.125$; between 64 and 128 it is 8.0. Relative precision is constant (~6% per step at 3 mantissa bits: $2^{-3}$), which is why *relative* error (not absolute) is the right lens — and why a scale that shifts values into the $[1, 2)$ decade gets the best resolution per bit.
- **448 vs 57344 is a 128× range difference.** E5M2 trades 1 mantissa bit for 1 exponent bit: same constant relative precision class ($2^{-2} = 25\%$ steps) but a range wide enough that a per-tensor max rarely saturates. The V3 choice of E4M3 says: with fine-grained scales, saturation is controllable; precision is not recoverable.

---

## 3. Why a single global scale fails — the outlier problem

Naive per-tensor quantization picks one `S = max(|X|) / 448` for the entire matrix. Consider an activation matrix where 99% of channels have magnitude ~1 and 1% are outliers at ~100:

- `S = 100/448 ≈ 0.223`. The 99% inliers map to `round(1/0.223) = 4` → dequant `4 × 0.223 = 0.89`. **You've lost ~11% of every normal value to rounding**, and with only 3 mantissa bits the inlier resolution is now ~0.223 — coarser than the signal.
- The outlier (100) maps to `448` (clipped) — fine.

The inliers, which carry the actual signal, get quantized to noise. This is the **outlier channel** problem documented across every LLM quantization paper: a tiny fraction of channels forces a scale that destroys everyone else.

### 3.1 Why gradients are the worst case

During backprop, gradient magnitudes across a layer's channels are famously **heavy-tailed**: a log-normal-like distribution where the top few percent of channels carry most of the mass. A per-tensor max over such a distribution is dominated by the tail, so the scale is almost always "wrong" for the body of the distribution. This is precisely the failure that per-tensor FP8 training hit in early experiments, and why the field converged on per-tile/per-block scaling: the tail must be *localized*, not global.

---

## 4. Fine-grained scaling — DeepSeek-V3's answer

Instead of one scale per tensor, DeepSeek-V3 uses **many small scales**, each covering a block small enough that outliers are local and don't poison distant inliers.

### 4.1 Tile-wise activation scaling (`1 × 128`)

Activations are quantized per **128-token tile** along the channel dimension:

```
Activation X  (T tokens × C channels)
┌──────────────────────────────────────────┐
│  tile 0 : scale S_0  (1 × 128 tokens)     │  ← own FP32 scale
│  tile 1 : scale S_1  (1 × 128 tokens)     │  ← own FP32 scale
│  ...                                      │
└──────────────────────────────────────────┘
```

Each 128-token strip gets its own `S_i = max(|X_i|) / 448`, so an outlier in token row 500 only inflates the scale for its own 128-row tile; rows 0–127 keep a tight scale and full precision.

### 4.2 Block-wise weight scaling (`128 × 128`)

Weight matrices are partitioned into `128 × 128` sub-blocks, each with an independent FP32 scale:

```
Weight W  (Out_Channels × In_Channels)
┌──────────────┬──────────────┐
│ S_(0,0) 128² │ S_(0,1) 128² │   ← each 128×128 block has its own FP32 scale
├──────────────┼──────────────┤
│ S_(1,0) 128² │ S_(1,1) 128² │
└──────────────┴──────────────┘
```

`★ Insight ─────────────────────────────────────` Why `128`? It is the **Tensor Core tile size**. Hopper FP8 Tensor Cores operate on `16×128` (activation) × `128×128` (weight) tiles natively. Choosing the scale block to match the hardware tile means the scale is applied *inside* the Tensor Core accumulator, with zero overhead — the dequant scale multiply is folded into the dot product. A scale block that didn't align with the tile boundary would force extra rescales between cores. The "fine-grained" granularity is not arbitrary; it is the largest block that is (a) small enough to localize outliers and (b) exactly the Tensor Core's native shape. `─────────────────────────────────────────────────`

### 4.4 Why the two shapes are asymmetric (1×128 vs 128×128)

Activations are tiled `1 × 128` (128 *tokens*), weights `128 × 128`. The asymmetry is structural, not an oversight:

- **Activations vary per token row.** Row $t$'s values are a function of the *input*, so outliers are a property of individual tokens. Tiling by 128 tokens isolates a bad token's outliers to its own strip — and because a batch's activations change every step, the scale must be recomputed per step anyway, so the granularity cost (128 scales per matrix) is paid once and amortized over the GEMM.
- **Weights vary per output/input channel pair.** A weight matrix is *static* between optimizer steps, so its scales are computed once and reused (see §6). Grouping 128 output × 128 input channels into one scale exploits channel locality: outlier *rows* of the weight matrix (a common LLM phenomenon) are confined to their own 128-row band.

The contraction dimension (the shared $k$ in $Y = XW$) is covered by *both* scales: each output element carries $S^X_{i, k/128} \cdot S^W_{k/128, j/128}$, so an outlier in a single activation tile and an outlier in a single weight block never jointly poison a wide region of $Y$.

### 4.3 The per-element scaling flow for one GEMM

$$Y_{ij} = \sum_k X_{ik}\, W_{kj} \;\;\longrightarrow\;\; Y_{ij} = \sum_k \underbrace{(X^{\text{FP8}}_{ik} \cdot S^X_{i,k/128})}_{\text{dequant }X} \;\underbrace{(W^{\text{FP8}}_{kj} \cdot S^W_{k/128,\,j/128})}_{\text{dequant }W}$$

Each product term carries two scales (one from X's tile, one from W's block); the Tensor Core accumulates the FP8 products in **FP32**, then the surrounding code applies the combined `S^X \cdot S^W` rescale. The scales themselves are FP32, so the dynamic range is preserved; only the bulk storage and FLOPs are 8-bit.

---

## 5. High-precision accumulation — the stability floor

FP8 is only the *compute* dtype; the parts that must stay accurate stay wider:

- **Tensor Core accumulation** happens in **FP32** internal registers, then down-scales to BF16/FP8 for the output. The accumulation never loses precision to FP8 — only the storage and the multiply do.
- **Master weights** are kept in **FP32** (or BF16) in the optimizer; the FP8 weight is a *cast* used for the GEMM, not the source of truth.
- **AdamW state** (moments `m, v`) is **FP32**. The update `θ ← θ − lr · m / (√v + ε)` is done in FP32, then the result is re-cast to FP8 for the next forward.

This is the same pattern as BF16 training (FP32 master + BF16 compute), just one notch lower in the compute dtype. The principle is invariant: **the dtype ladder puts precision where gradients accumulate and savings where bytes move.**

---

## 6. The online scale amortization trick

Computing `S = max(|X|)/448` per tile every step is itself a reduction over the whole activation — cheap, but not free. DeepSeek-V3 amortizes it:

- **Activation scales** are computed every step (activations change each step).
- **Weight scales** are computed **once** and reused for many steps — weights move slowly under AdamW, so the scale drifts negligibly. The scale is recomputed periodically (e.g. every N steps) rather than every step.

This is a second-order optimization that matters at scale; it is part of why FP8's gains aren't eaten by the scale-computation overhead.

---

## 7. Why this repo skips FP8 — a deliberate scope call

Implementing FP8 training correctly is a large engineering surface:

1. A custom FP8 GEMM path (Triton or CUTLASS) with tile/block scale plumbing.
2. Per-tile/per-block scale buffers carried alongside every weight and activation.
3. E4M3 casting at the forward/backward boundary with the scale flow of §4.3.
4. An autocast context that swaps dtypes per GEMM while keeping master weights FP32.
5. Stability instrumentation: overflow/underflow counters, scale-histogram logging, fallback to BF16 on divergence.

For a 411.6M-param model on a single A100, BF16 already saturates the compute efficiently and the memory fits with room to spare — the ~2× FP8 win buys throughput that isn't the bottleneck here. The bottleneck at this scale is data pipeline and launch overhead, which FP8 does not address. So FP8 is **documented but deliberately not built**; it is the natural next lever if the model were scaled up to where BF16 memory or throughput became the constraint.

### 7.1 Why quantized training error is a *propagation* problem, not a per-layer one

A transformer is a residual stream: every layer adds its output to the running hidden state $h_{t+1} = h_t + f(h_t)$. Quantization noise at layer $\ell$ therefore does not stay at layer $\ell$ — it is carried into every subsequent layer's input:

$$\hat{h}_L = h_0 + \sum_{\ell=1}^{L} \left( f_\ell(\hat{h}_{\ell-1}) + \epsilon_\ell \right), \qquad \text{noise } \sum_\ell \epsilon_\ell \text{ accumulates along the stream}$$

with $\epsilon_\ell$ the per-layer quantization error (activation + weight). Two consequences:

- **Depth multiplies the budget.** With 18 layers in this repo (671B-scale V3 has hundreds), a per-layer relative error budget of $\delta$ becomes an end-of-stream error that can grow as $\sqrt{L}\,\delta$ (random signs) up to $L\delta$ (aligned). The fine-grained scales of §4 exist precisely to keep $\delta$ small enough that $L\delta$ stays under the noise floor the loss can tolerate.
- **The head and the embedding are the sensitive ends.** The embedding maps tokens into the stream; the LM head reads the final hidden state and produces logits that the softmax turns into probabilities. Errors at the embedding are multiplied by all layers; errors at the head directly perturb the training signal. This is why FP8 systems keep the embedding/head in higher precision even when the bulk GEMMs are 8-bit — and why this repo's BF16 autocast lets RMSNorm, embeddings and the CE loss run FP32 (see [08 Training Pipeline](../training.md)).

The practical takeaway: **FP8 is not "make everything 8-bit"** — it is a budgeted precision allocation where the per-layer error allowance is dictated by residual-stream depth, and only the bandwidth-dominated GEMMs get the 8-bit treatment.

---

## 8. FP8 vs what this repo actually does — the same ladder, one rung up

The repo's BF16 training (`training/pretrain.py:Pretrainer._amp_context`) already embodies the two structural ideas FP8 pushes further:

1. **Compute dtype ≠ master dtype.** BF16 autocast runs the GEMMs in BF16 while AdamW keeps FP32 master weights, moments, and the update math (`training/pretrain.py:Pretrainer.__init__` builds the FP32 optimizer state). FP8 would only change *which* dtype the compute GEMMs use.
2. **Accumulators stay FP32.** Every Triton kernel in this repo accumulates in `tl.float32` (see [12 Triton Kernels](../concepts/kernels-and-ops.md)); PyTorch BF16 matmuls accumulate in FP32 internally. FP8 Tensor Cores do exactly the same — 8-bit multiply, FP32 accumulate.

The gap between this repo and V3 is therefore not architectural principle but *granularity of the quantize step*: BF16 needs no scale at all (its 8-bit exponent gives ±3.4e38 range natively), so the entire scale-buffer machinery of §4 vanishes. BF16 is "FP8 with the scale folded into every element" — the range is paid for in 8 extra bits per element, which at 411.6M params is a rounding error in the memory budget (0.82 GB weights vs ~12 GB optimizer state; see [11 Operations and Testing](../concepts/kernels-and-ops.md)).

`★ Insight ─────────────────────────────────────` The decision rule in this repo: adopt FP8 when **bytes moved** (not FLOPs) is the binding constraint. At 8 × 2048 × 18 layers on one A100, activations and weights are small enough that BF16's extra bytes cost far less than FP8's engineering + stability risk. The moment the model outgrows a single GPU (or the batch doubles), the trade flips. `─────────────────────────────────────────────────`

---

## 9. Worked example — an FP8 quantization error budget

Setup: a 4096-wide activation row, 99% of entries ~N(0, 1) (inliers), 1% outliers at magnitude 100. Compare per-tensor vs per-128-tile scaling for E4M3.

**Per-tensor:** `S = 100/448 ≈ 0.223`. Inlier values ~1 map to `round(4.48) = 4` → dequant `0.893`. Relative error ~11%; resolution (step size) 0.223.

**Per-tile (outlier isolated):** the tile with outliers uses `S = 0.223` as above; the 31 outlier-free tiles use `S = max(|N(0,1)|)/448 ≈ 3.5/448 ≈ 0.0078`. Inliers map to `round(1/0.0078) = 128` → step size 0.0078, relative error ~0.4% — a **28× improvement** in inlier fidelity, at the cost of 32 FP32 scale values instead of 1.

**Accumulation:** with a 4096-length K-dimension and FP32 accumulation, the dot-product error is dominated by the *input* quantization (relative ~0.4–11% per element), not by the accumulation — FP32 accumulators add ~2⁻²⁴ relative error per add, far below the 8-bit input noise. This is why the "accumulator must be FP32" rule matters but is not the hard part; the hard part is always the scale.

---

## 10. Check your understanding

1. **Q:** Why does DeepSeek-V3 use E4M3 for backward GEMMs instead of the wider-range E5M2? **A:** Because per-block fine-grained scaling already localizes the dynamic range; E4M3's extra mantissa bit (3 vs 2) halves the inlier rounding error. DeepSeek's own DeepGEMM repo confirms E5M2 is unused in V3/R1.
2. **Q:** Why is per-tensor scaling insufficient for LLM activations? **A:** Outlier channels dominate `max(|X|)`, inflating S and destroying inlier precision (worked example §9: ~11% vs ~0.4% relative error).
3. **Q:** What stays FP32 in an FP8 training system? **A:** Tensor Core accumulators, scale factors, master weights, and AdamW state (m, v) — the same set this repo keeps FP32 in BF16 training.
4. **Q:** What would this repo need to adopt FP8? **A:** A custom FP8 GEMM path with tile/block scale plumbing (§7) — and a measured memory/throughput constraint that BF16 actually hits first.
5. **Q:** Why does residual depth matter for quantization budgets? **A:** Per-layer noise accumulates along the residual stream (~$\sqrt{L}$ to $L$ growth); deeper models need tighter per-layer error, which is why fine-grained scales and FP32 embeddings/heads are non-negotiable parts of an FP8 recipe (§7.1).

### 10.1 A note on FP8 inference (W8A8) vs FP8 training

This chapter is about **training**. FP8 *inference* (weight-only W8A8 quantization) is a different, much simpler problem: weights are quantized once, offline, with calibration; activations are quantized at serving time with running-statistic scales. Training is harder because activations change every step and the *backward* pass re-introduces heavy-tailed gradients — which is why the scale machinery here is per-step and per-block, and why this repo's only 8-bit-adjacent work is the FP32-accumulator discipline in its Triton kernels (see [12 Triton Kernels](../concepts/kernels-and-ops.md)). Do not conflate the two: W8A8 inference would be a plausible *future* optimization for this repo's serving path; FP8 training remains out of scope at 411.6M.

## References

- `models/mla.py`, `models/mla_triton.py` — MLA implementation + Triton kernel
- [Triton Kernels](../concepts/kernels-and-ops.md) — fused kernel design
