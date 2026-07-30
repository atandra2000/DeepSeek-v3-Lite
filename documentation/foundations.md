# Foundations — Building Blocks of DeepSeek-V3-Lite

> **Purpose:** A self-contained textbook chapter covering every prerequisite concept you need before reading the component-specific docs (MLA, MoE, MTP, training, inference). Read this first if you are learning the project from scratch.

> **Read this if** transformer basics, RoPE, or Chinchilla are unfamiliar. **Skip if** you're ready to run smoke tests → [getting_started.md](getting_started.md).

---

## Table of Contents

1. [What Is a Language Model?](#what-is-a-language-model)
2. [The Transformer Architecture](#the-transformer-architecture)
3. [Pre-Norm Residual Blocks](#pre-norm-residual-blocks)
4. [RMSNorm — Root Mean Square Layer Normalization](#rmsnorm--root-mean-square-layer-normalization)
5. [Multi-Head Attention — From Dot Product to Causal Masking](#multi-head-attention--from-dot-product-to-causal-masking)
6. [Rotary Position Embeddings (RoPE)](#rotary-position-embeddings-rope)
7. [The Feed-Forward Network — SwiGLU](#the-feed-forward-network--swiglu)
8. [Mixture of Experts (MoE) — Intuition](#mixture-of-experts-moe--intuition)
9. [Causal Language Modeling Loss](#causal-language-modeling-loss)
10. [Weight Tying](#weight-tying)
11. [KV Caching for Autoregressive Inference](#kv-caching-for-autoregressive-inference)
12. [Scaling Laws and the Chinchilla Optimum](#scaling-laws-and-the-chinchilla-optimum)
13. [Maximal Update Parametrization (μP)](#maximal-update-parametrization-μp)
14. [Mixed Precision — BF16 Training](#mixed-precision--bf16-training)
15. [Gradient Checkpointing](#gradient-checkpointing)
16. [AdamW and Learning Rate Schedules](#adamw-and-learning-rate-schedules)
17. [How the Pieces Map to This Repo](#how-the-pieces-map-to-this-repo)
18. [Worked Example — One Forward Pass at 422M Scale](#worked-example--one-forward-pass-at-422m-scale)
22. [Byte-Pair Encoding (BPE) and Tokenization](#byte-pair-encoding-bpe-and-tokenization)
23. [Perplexity](#perplexity)
24. [Softmax Numerical Stability](#softmax-numerical-stability)
25. [FlashAttention and SDPA](#flashattention-and-sdpa--memory-efficient-attention)
26. [FLOP Accounting](#flop-accounting)
27. [Extended μP Theory](#extended-μp-theory-optional-deep-dive)
28. [Load-Bearing Invariants](#load-bearing-invariants)
29. [FAQ](#faq)
30. [References](#references)

---

## What Is a Language Model?

A **language model** assigns a probability distribution over the next token given all preceding tokens. Given a sequence of token IDs $x_1, x_2, \ldots, x_T$ drawn from a vocabulary $\mathcal{V}$ of size $V$, a causal language model defines:

$$
P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, \ldots, x_{t-1})
$$

**Intuition:** Reading left to right, the model predicts each word from context — exactly how humans complete sentences. Training teaches this by showing billions of real text sequences and penalizing wrong predictions.

**In this repo:** Tokens are integers in $[0, V-1]$ with $V = 100018$ (DeepSeek-Coder-V2-Lite tokenizer). The model outputs logits $\ell_t \in \mathbb{R}^V$ at each position; softmax converts them to probabilities.

### Tokens and embeddings

Raw text is split into **tokens** — subword units from a fixed vocabulary. Each token ID $x_t$ maps to a dense vector via an embedding matrix $E \in \mathbb{R}^{V \times d}$:

$$
\mathbf{h}_t^{(0)} = E[x_t] \in \mathbb{R}^d
$$

At the canonical 422M config: $V = 100018$, $d = 768$. The embedding table alone holds $100018 \times 768 \approx 77$M parameters — the second-largest component after MoE FFN weights.

**Design note:** We use a code-oriented tokenizer (`deepseek-ai/deepseek-coder-v2-lite`) because this project targets understanding DeepSeek-V3 architecture, not building a general chatbot. Code tokenization is efficient for Python/technical text in the training mixture.

---

## The Transformer Architecture

The **Transformer** (Vaswani et al., 2017, arXiv:1706.03762) replaced recurrence with **self-attention**: every token can directly attend to every other token in one parallel pass. DeepSeek-V3-Lite implements a **decoder-only** (GPT-style) variant:

```
Token IDs  →  Embedding  →  [Block × L]  →  RMSNorm  →  LM Head  →  Logits
```

Each **block** contains:
1. **Attention** — mix information across positions (with causal masking)
2. **FFN** — per-position nonlinear transformation (dense SwiGLU or sparse MoE)

**In this repo:** $L = 18$ layers. Layers 0–1 use dense SwiGLU; layers 2–17 use DeepSeekMoE. Every layer uses MLA (not standard MHA). See [architecture.md](architecture.md) for the full stack diagram.

### Why decoder-only?

Encoder-decoder models (original Transformer, T5) separate "read input" from "write output." Decoder-only models unify both: the same stack processes prompt and generation. This simplifies training (one objective: next-token prediction on all positions) and inference (one KV cache, one forward path).

**Trade-off:** Decoder-only models cannot do bidirectional encoding natively. For code completion and generation — this project's focus — that is acceptable.

---

## Pre-Norm Residual Blocks

Each sub-layer uses the **pre-norm** pattern (Xiong et al., 2020):

$$
\mathbf{x}' = \mathbf{x} + \mathrm{Sublayer}(\mathrm{RMSNorm}(\mathbf{x}))
$$

**Contrast with post-norm** (original Transformer): $\mathbf{x}' = \mathrm{RMSNorm}(\mathbf{x} + \mathrm{Sublayer}(\mathbf{x}))$.

**Why pre-norm?**
- Gradients flow more directly through the residual "highway" $\mathbf{x} \to \mathbf{x}'$.
- Training is stable at depth without careful learning-rate tuning.
- Modern LLMs (GPT-3, LLaMA, DeepSeek) universally use pre-norm.

**In this repo** (`models/transformer.py:TransformerBlock.forward`):

```python
x = x + self.attn(self.attn_norm(x), start_pos, mask, use_cache)
x = x + self.ffn(self.ffn_norm(x))
```

The residual stream $\mathbf{x}$ maintains shape $(B, S, d)$ throughout all 18 layers.

---

## RMSNorm — Root Mean Square Layer Normalization

**Layer normalization** stabilizes activations by normalizing across the feature dimension. **RMSNorm** (Zhang & Sennrich, 2019) drops the mean-centering step:

$$
\mathrm{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \odot \boldsymbol{\gamma}
$$

where $\boldsymbol{\gamma} \in \mathbb{R}^d$ is a learned scale (no bias), $\epsilon = 10^{-6}$.

**Derivation intuition:** Standard LayerNorm subtracts the mean $\mu = \frac{1}{d}\sum x_i$ before scaling. RMSNorm argues that re-centering is unnecessary for transformers — only the **scale** of activations matters for stable gradients. Removing mean subtraction saves compute and slightly improves throughput.

**Comparison:**

| | LayerNorm | RMSNorm |
|---|---|---|
| Mean subtraction | Yes | No |
| Learned scale $\gamma$ | Yes | Yes |
| Learned bias $\beta$ | Yes | No |
| Params per norm | $2d$ | $d$ |

At $d = 768$ and 37 norm layers (2 per block × 18 + final), RMSNorm saves $37 \times 768 \approx 28$K parameters — negligible, but the speed win is real.

**In this repo:** `nn.RMSNorm(dim, eps=1e-6)` in `TransformerBlock`, MLA, MTP blocks, and the final `self.norm` before the LM head.

---

## Multi-Head Attention — From Dot Product to Causal Masking

### Scaled dot-product attention

Given queries $\mathbf{Q}$, keys $\mathbf{K}$, values $\mathbf{V} \in \mathbb{R}^{S \times d}$:

$$
\mathrm{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \mathrm{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}} + \mathbf{M}\right)\mathbf{V}
$$

- $\mathbf{Q}\mathbf{K}^\top$ computes pairwise similarity between every query and key position.
- $\sqrt{d}$ prevents dot products from growing large (which would push softmax into saturation).
- $\mathbf{M}$ is a causal mask: $M_{ij} = -\infty$ if $j > i$, else $0$.

**Intuition:** Position $i$ asks "which previous positions are relevant to me?" Softmax turns similarities into weights summing to 1; the weighted sum of values is the output.

### Multi-head attention (MHA)

Split $d$ into $H$ heads of dimension $d_h = d / H$. Each head has independent projections:

$$
\mathrm{head}_h = \mathrm{Attention}(\mathbf{Q}_h, \mathbf{K}_h, \mathbf{V}_h), \quad \mathbf{O} = \mathbf{W}_O [\mathrm{head}_1; \ldots; \mathrm{head}_H]
$$

**Why multiple heads?** Different heads can specialize: syntax vs. semantics vs. long-range dependencies. Empirically, $H = 12$ at $d = 768$ ($d_h = 64$) is a sweet spot for small models.

**In this repo:** Standard MHA appears only in MTP blocks (`models/mtp.py:MTPBlock`). The main trunk uses **MLA** — a compressed variant covered in [MLA.md](MLA.md). The causal mask logic is identical.

### Causal masking during training

For sequence length $S$, the mask is lower-triangular:

```
     pos:  0    1    2    3
  0        0   -inf -inf -inf
  1        0    0   -inf -inf
  2        0    0    0   -inf
  3        0    0    0    0
```

(`0` = attend, `-inf` = block)

**In this repo:** `Transformer._build_causal_mask` caches an additive mask of shape $(1, 1, S, S)$. MLA applies it via `F.scaled_dot_product_attention` (SDPA) or manually.

---

## Rotary Position Embeddings (RoPE)

**Problem:** Attention is permutation-invariant without position information. We must inject **where** each token sits in the sequence.

**RoPE** (Su et al., 2021, arXiv:2104.09870) encodes position by rotating query and key vectors in 2D subspaces:

For dimension pair $(2k, 2k+1)$ at position $m$:

$$
\begin{pmatrix} q_{2k} \\ q_{2k+1} \end{pmatrix} \mapsto
\begin{pmatrix} \cos m\theta_k & -\sin m\theta_k \\ \sin m\theta_k & \cos m\theta_k \end{pmatrix}
\begin{pmatrix} q_{2k} \\ q_{2k+1} \end{pmatrix}
$$

where $\theta_k = \text{base}^{-2k/d_{\text{rope}}}$ and `base` = `rope_theta` (10,000 in our config).

**Key property:** $\langle \mathrm{RoPE}(\mathbf{q}, m), \mathrm{RoPE}(\mathbf{k}, n) \rangle$ depends only on relative position $m - n$. This is exactly what attention needs.

**Intuition:** Instead of adding a position vector (original Transformer) or learning position embeddings (GPT-2), RoPE **twists** vectors so their dot product encodes relative distance. Longer contexts require either larger `rope_theta` or **YaRN** extrapolation (`rope_factor` > 1.0 in config; dormant at 1.0 here).

**In this repo:** MLA splits each head's Q/K into **nope** (non-positional, 48 dims) and **rope** (positional, 24 dims) components. Only the rope part gets RoPE. See [MLA.md](MLA.md) §Decoupled RoPE.

---

## The Feed-Forward Network — SwiGLU

Each transformer block has a position-wise FFN. **SwiGLU** (Shazeer, 2020) replaces the standard two-layer MLP:

$$
\mathrm{SwiGLU}(\mathbf{x}) = W_2\bigl(\mathrm{SiLU}(W_1 \mathbf{x}) \odot W_3 \mathbf{x}\bigr)
$$

where $\mathrm{SiLU}(z) = z \cdot \sigma(z)$ (aka swish), and $\odot$ is element-wise multiplication.

**Why three matrices?** The gating path $W_3 \mathbf{x}$ acts as a learned switch controlling which features $W_1$ extracts. Empirically, SwiGLU outperforms ReLU/GELU FFNs at equal parameter count.

**Parameter count** for one SwiGLU layer:

$$
\underbrace{d \cdot I}_{W_1} + \underbrace{I \cdot d}_{W_2} + \underbrace{d \cdot I}_{W_3} = 3dI
$$

At dense layers: $d = 768$, $I = 1536$ → $3 \times 768 \times 1536 \approx 3.5$M params per FFN.

**In this repo:** `SwiGLUFFN` in `models/transformer.py`. MoE experts use the same SwiGLU structure with smaller $I = 384$ (`moe_inter_dim`). See [moe.md](moe.md).

---

## Mixture of Experts (MoE) — Intuition

A **Mixture of Experts** replaces one large FFN with $N$ smaller **expert** FFNs plus a **router** that selects top-$k$ experts per token.

$$
\mathrm{MoE}(\mathbf{x}) = \sum_{i \in \mathrm{TopK}(\mathbf{x})} g_i(\mathbf{x}) \cdot \mathrm{Expert}_i(\mathbf{x}) + \mathrm{SharedExpert}(\mathbf{x})
$$

**Intuition:** Not every token needs the same computation. Factual tokens might route to a "knowledge" expert; syntactic tokens to a "grammar" expert. Sparsity means we store $N$ experts but only **execute** $k$ per token — gaining capacity without proportional compute.

**The load-balancing problem:** Without intervention, routers collapse — sending 99% of tokens to one expert. Solutions:
1. **Auxiliary load-balancing loss** (Switch Transformer) — adds a gradient term penalizing imbalance.
2. **Aux-loss-free bias updates** (DeepSeek-V3) — adjust router biases out-of-band. **This repo uses #2.** See [moe.md](moe.md).

**At 422M scale:** 16 MoE layers × 20 routed experts + 1 shared, top-4 routing. ~70% of all parameters live in expert weights.

---

## Causal Language Modeling Loss

Training minimizes **cross-entropy** between predicted and true next tokens:

$$
\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})
$$

In practice, for logits $\ell_t \in \mathbb{R}^V$ and target $y_t$:

$$
\mathcal{L} = \frac{1}{T}\sum_{t=1}^{T} \mathrm{CE}(\ell_t, y_t) = \frac{1}{T}\sum_{t=1}^{T} \left[-\log \frac{e^{\ell_{t,y_t}}}{\sum_j e^{\ell_{t,j}}}\right]
$$

**MTP extension:** This repo adds auxiliary losses for predicting $t+2, t+3, \ldots$ with weight $\lambda = 0.3$. See [mtp.md](mtp.md).

**Label smoothing:** Not used here. The model is trained with hard targets (one-hot).

---

## Weight Tying

**Weight tying** shares the embedding matrix $E$ with the output projection (LM head):

$$
\ell_t = E^\top \cdot \mathrm{RMSNorm}(\mathbf{h}_t)
$$

**Why?** The embedding maps token → vector; the LM head maps vector → token logits. These are dual operations. Tying:
- Halves the largest matrix's parameter count ($77$M saved at 422M scale).
- Acts as regularization (Sutskever et al., 2014).

**In this repo:** `Transformer` with `weight_tying=True` sets `self.head.weight = self.embed.weight`. `count_parameters` deduplicates by tensor ID so tied weights count once.

---

## KV Caching for Autoregressive Inference

During training, the full sequence is processed in parallel — no cache needed.

During inference, generating token $t$ requires attending to tokens $1, \ldots, t-1$. Recomputing their K/V every step is $O(T^2)$ per generated token. **KV caching** stores past K/V tensors:

```
Step 1 (prefill):  process prompt [t₁, t₂, ..., tₙ]  → cache K/V for positions 0..n-1
Step 2 (decode):   process [tₙ₊₁] only               → read cache[0..n-1], write cache[n]
Step 3 (decode):   process [tₙ₊₂] only               → read cache[0..n], write cache[n+1]
```

**Memory:** Per layer, cache size grows as $O(S \cdot d_{\text{kv}})$. MLA compresses $d_{\text{kv}}$ dramatically. See [MLA.md](MLA.md) and [inference.md](inference.md).

**Critical invariant:** `use_cache=False` during training. Enabling cache in training silently breaks gradient flow through past positions.

---

## Scaling Laws and the Chinchilla Optimum

**Kaplan scaling laws** (2020) suggested training larger models on fewer tokens. **Chinchilla** (Hoffmann et al., 2022, arXiv:2203.15556) revised this:

> For a fixed compute budget, model size and training tokens should scale **equally**.

**Chinchilla-optimal token count:**

$$
N_{\text{tokens}} \approx 20 \times N_{\text{params}}
$$

For 422M parameters: $20 \times 4.22 \times 10^8 \approx 8.4 \times 10^9$ tokens.

**In this repo:**
- Effective batch: $8 \times 4 \times 2048 = 65536$ tokens per optimizer step.
- 512,000 optimizer steps → many passes over the 8.4B-token corpus (multi-epoch).
- The data pipeline targets exactly this token budget. See [data_pipeline.md](data_pipeline.md).

### Model FLOPs Utilisation (MFU)

$$
\mathrm{MFU} = \frac{\text{actual FLOPs/sec}}{\text{peak hardware FLOPs/sec}}
$$

Target: 35–40% on A100 80GB. MFU < 20% usually means memory-bound (activations, MoE dispatch) or unfused kernels. See [scripts.md](scripts.md) §step_time_a100.

---

## Maximal Update Parametrization (μP)

When scaling model width, naively keeping the same learning rate causes instability or under-training. **μP** (Yang et al., 2021; Tensor Programs) prescribes:

1. Init scales with width.
2. Learning rate scales as $1/\text{width}$ for hidden weights.
3. **LR transfer:** A hyperparameter tuned on a small model transfers to a large model if both use μP.

**This repo's pragmatic shortcut** (`training/pretrain.py:Pretrainer.__init__`):

$$
\eta_{\text{new}} = \eta_{\text{ref}} \sqrt{\frac{N_{\text{ref}}}{N_{\text{actual}}}}
$$

With $\eta_{\text{ref}} = 6 \times 10^{-4}$ at $N_{\text{ref}} = 757226496$ (DeepSeek-V3 reference). For 422M params:

$$
\eta_{\text{new}} = 6 \times 10^{-4} \times \sqrt{757226496 / 422000000} \approx 8.07 \times 10^{-4}
$$

**Important:** μP scaling runs **after** MTP wrapping, because MTP heads add ~3M params that affect the count. See [training.md](training.md).

---

## Mixed Precision — BF16 Training

**BF16** (bfloat16) uses 8 exponent bits (same as FP32) and 7 mantissa bits. Compared to FP16:

| Format | Exponent | Mantissa | Range | Precision |
|---|---|---|---|---|
| FP32 | 8 | 23 | Large | High |
| FP16 | 5 | 10 | Small | Medium |
| BF16 | 8 | 7 | Large | Lower |

**Why BF16 for training?** No loss scaling needed (unlike FP16). Gradients and activations rarely underflow. A100 Tensor Cores natively support BF16 matmul.

**In this repo:** `autocast("cuda", dtype=torch.bfloat16)` wraps forward; AdamW maintains **FP32 master weights** internally (fused optimizer). No `GradScaler`.

---

## Gradient Checkpointing

**Problem:** Backpropagation stores activations for every layer. For 18 layers × batch 8 × seq 2048 × dim 768, activation memory dominates VRAM.

**Solution:** Don't store all activations. During backward, **recompute** them from checkpoints:

```
Forward:  compute layers 1..18, save only layers {6, 12, 18} as checkpoints
Backward: recompute layers 7..12 from checkpoint at 6, then continue
```

**Trade-off:** ~33% extra compute for ~3× activation memory savings.

**In this repo:** `Transformer(use_checkpoint=True)` wraps each `TransformerBlock.forward` in `torch.utils.checkpoint.checkpoint`. Enabled in 422M config, disabled in 1650 2M smoke config.

---

## AdamW and Learning Rate Schedules

**AdamW** (Loshchilov & Hutter, 2019) decouples weight decay from the gradient update:

$$
\theta_{t+1} = \theta_t - \eta \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t\right)
$$

**In this repo:**
- $\beta_1 = 0.9$, $\beta_2 = 0.95$ (higher $\beta_2$ than default 0.999 — common in LLM training for smoother second moments).
- Weight decay $\lambda = 0.1$ on **dim ≥ 2** params only (matrices). Biases and norm scales get $\lambda = 0$.
- Global grad clip at norm 1.0 before optimizer step.

**LR schedule** (`make_warmup_cosine_lambda`):

1. **Warmup** (steps 0–2000): linear ramp $0 \to \eta$.
2. **Cosine decay** (steps 2000–512000): $\eta_t = \eta_{\min} + (\eta - \eta_{\min}) \cdot \frac{1}{2}(1 + \cos(\pi \cdot \text{progress}))$.
3. **Floor:** $\eta_{\min} = 0.05 \times \eta$.

---

## How the Pieces Map to This Repo

| Concept | Where in code | Deep-dive doc |
|---|---|---|
| Embedding + blocks + head | `models/transformer.py` | [transformer.md](transformer.md) |
| MLA attention | `models/mla.py` | [MLA.md](MLA.md) |
| MoE routing + experts | `models/moe.py` | [moe.md](moe.md) |
| MTP auxiliary heads | `models/mtp.py` | [mtp.md](mtp.md) |
| Training loop | `training/pretrain.py` | [training.md](training.md) |
| Generation + speculative | `inference/` | [inference.md](inference.md) |
| Data pipeline | `data/prepare_data.py` | [data_pipeline.md](data_pipeline.md) |
| Config YAML | `configs/*.yaml` | [configs.md](configs.md) |
| Triton kernels | `models/*_triton.py` | [triton_kernels.md](triton_kernels.md) |

---

## Worked Example — One Forward Pass at 422M Scale

**Input:** batch $B = 2$, sequence $S = 4$, token IDs:

```
[[101, 2345, 678, 9012],
 [55,  1234, 5678, 90]]
```

**Step 1 — Embedding:** $(2, 4) \to (2, 4, 768)$

**Step 2 — Layer 0 (dense):**
- RMSNorm → MLA(12 heads, R=192) → residual
- RMSNorm → SwiGLU(768 → 1536 → 768) → residual

**Step 3 — Layer 1 (dense):** same structure.

**Step 4 — Layers 2–17 (MoE):**
- RMSNorm → MLA → residual
- RMSNorm → DeepSeekMoE: route to top-4 of 20 experts + 1 shared → residual

**Step 5 — Final RMSNorm → LM head:** $(2, 4, 768) \to (2, 4, 100018)$

**Parameter touch count:** MoE layers execute 5 of 21 experts per token (4 routed + 1 shared), so FLOPs are ~5/21 of dense-equivalent, but all 21 expert weights remain in memory.

**MTP path (training only):** `forward_with_hidden` returns hidden states; MTPModule takes $(h_t, e_{t+1})$ and predicts token $t+2$.

---

## Byte-Pair Encoding (BPE) and Tokenization

### Motivation

Neural networks operate on **discrete symbols**. Raw UTF-8 bytes are too fine-grained (256 symbols, long sequences). Whole words are too coarse (unbounded vocabulary). **Subword tokenization** splits the compromise: frequent words stay whole; rare words decompose into pieces.

### BPE algorithm (intuition)

1. Start with byte- or character-level vocabulary.
2. Count all adjacent symbol pairs in the training corpus.
3. Merge the most frequent pair into a new symbol.
4. Repeat until vocabulary reaches target size $V$.

### DeepSeek-Coder-V2-Lite tokenizer (this repo)

| Property | Value | Implication |
|---|---|---|
| `vocab_size` | 100,018 | Embedding rows must match exactly |
| `eos_token_id` | 100,017 | Appended at document boundaries |
| `byte_fallback` | enabled | Any byte 0-255 is representable |

**Footgun:** Using `vocab_size=100000` breaks `nn.Embedding` — always verify `len(tokenizer) == 100018`.

### Tokens per parameter

Chinchilla-optimal ratio: $N_{\text{tokens}} / N_{\text{params}} \approx 20$. At 422M params and 8.4B tokens: ratio $\approx 19.9$.

---

## Perplexity — Measuring Model Quality

$$
\mathrm{PPL} = \exp(\mathcal{L}) = \exp\left(-\frac{1}{T}\sum_{t=1}^{T} \log P(x_t \mid x_{<t})\right)
$$

**Intuition:** PPL is the effective branching factor at each step. PPL = 1 is perfect; PPL = $V$ is uniform random.

| Loss (nats) | PPL | Interpretation |
|---|---|---|
| 3.0 | 20.1 | Mid-training 422M |
| 2.5 | 12.2 | Strong mixed corpus |
| 2.0 | 7.4 | Very good small LM |

**In this repo:** `TrainingLogger` prints `ppl = exp(avg_loss)` (`utils/logging.py`).

---

## Softmax Numerical Stability

Naive softmax overflows for large logits. The log-sum-exp trick:

$$
\log \sum_j e^{z_j} = m + \log \sum_j e^{z_j - m}, \quad m = \max_j z_j
$$

`F.cross_entropy` fuses log-softmax + NLL — avoids materialising $(B, S, V)$ probability tensors when $V = 100018$.

---

## FlashAttention and SDPA

Standard attention materialises $\mathbf{A} \in \mathbb{R}^{S \times S}$ per head. At $S=2048$, $H=12$, BF16: $\approx 100$ MB/layer for $\mathbf{A}$ alone.

**FlashAttention** (Dao et al., 2022, arXiv:2205.14135) tiles in SRAM: $O(S^2)$ FLOPs, $O(S)$ memory.

This repo uses `attn_impl: sdpa` → `F.scaled_dot_product_attention`. MLA SDPA vs manual absorption paths are tested in `test_sdpa_and_manual_agree`.

---

## Information-Theoretic View

$$
\mathcal{L}_{\text{CE}} \approx H + D_{\mathrm{KL}}(P_{\text{data}} \| P_\theta)
$$

Minimising CE minimises KL divergence — the model learns a distribution close to the data.

---

## Weight Initialization

```python
nn.init.normal_(self.embed.weight, std=0.006)
```

Small $\sigma$ prevents logit saturation at depth $L=18$. MoE gates use the same std (`models/moe.py`).

---

## FLOP Accounting — One Training Step

$$
\text{FLOPs/token} \approx 6 N_{\text{non-embed}}
$$

| Quantity | 422M value |
|---|---|
| $N_{\text{non-embed}}$ | ~345M |
| FLOPs/token | ~2.1G |
| Tokens/optim step | 65,536 ($8 \times 4 \times 2048$) |
| FLOPs/step | $\approx 1.4 \times 10^{14}$ |
| A100 BF16 peak | 312 TFLOPS |
| Target @ 35% MFU | ~1.3 s/step |

Validated empirically by `scripts/step_time_a100.py`.

---


## Backpropagation Sketch — Attention Block

For one head, omitting batch/head indices. Let $\mathbf{A} = \mathrm{softmax}(\mathbf{Q}\mathbf{K}^	op / \sqrt{d})$, output $\mathbf{O} = \mathbf{A}\mathbf{V}$.

**Forward:** $\mathbf{Q},\mathbf{K},\mathbf{V}$ are linear projections of normalized input.

**Backward intuition:**
- Gradient w.r.t. $\mathbf{V}$: $\mathbf{A}^	op \frac{\partial \mathcal{L}}{\partial \mathbf{O}}$
- Gradient w.r.t. $\mathbf{A}$: $\frac{\partial \mathcal{L}}{\partial \mathbf{O}} \mathbf{V}^	op$
- Softmax backward couples all positions in the row — why attention is $O(S^2)$ in **both** forward and backward

**MLA twist:** $\mathbf{K},\mathbf{V}$ are low-rank functions of compressed cache — backward flows through LoRA adapters $W_{kv}$, absorption matrices, and RoPE rotation. See [MLA.md](MLA.md) §Gradient Flow.

---

## Softmax Numerical Stability

Naive: $p_i = e^{z_i} / \sum_j e^{z_j}$ overflows when $z_i$ is large.

**Stable form:** subtract row max $m = \max_j z_j$:

$$
p_i = \frac{e^{z_i - m}}{\sum_j e^{z_j - m}}
$$

PyTorch `F.cross_entropy` fuses log-softmax + NLL for the same reason — never implement CE as `log(softmax)` manually in training code.

---

## Matrix Calculus Primer (Used in This Repo)

| Operation | Forward | Gradient w.r.t. input |
|---|---|---|
| $\mathbf{y} = W\mathbf{x}$ | matmul | $W^	op \frac{\partial \mathcal{L}}{\partial \mathbf{y}}$ |
| $\mathbf{y} = \mathbf{x} \odot \mathbf{z}$ (SwiGLU gate) | elementwise | $\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \odot \mathbf{z}$ |
| RMSNorm | scale by RMS | see Zhang & Sennrich 2019 |
| RoPE | 2D rotations per pair | orthogonal — norm-preserving |

---

## Initialization — Why $\mathcal{N}(0, 0.006^2)$ Embeddings

`Transformer.__init__` uses `std=0.006` for `nn.Embedding`. At vocab 100k, expected embedding norm per row stays $O(1)$ — prevents initial logits from saturating softmax.

Linear layers use default PyTorch Kaiming/Uniform unless overridden in MLA/MoE modules (see component docs).

---

## Information-Theoretic View of LM Training

Minimising CE is equivalent to minimising KL divergence from the empirical token distribution to the model:

$$
\mathcal{L} \approx D_{\mathrm{KL}}(\hat{p}_{\text{data}} \,\|\, p_\theta) + H(\hat{p}_{\text{data}})
$$

The entropy of natural language is unobservable directly — but **compression** intuition holds: lower CE ⇒ better compressibility of the corpus under the model.

---

## Extended Worked Example — Perplexity from Loss

If average CE at step 50,000 is $\mathcal{L} = 3.20$ nats/token:

$$
\mathrm{PPL} = e^{3.20} \approx 24.5
$$

Interpretation: on average the model assigns probability mass equivalent to choosing uniformly among ~24.5 tokens. Random baseline over 100k vocab: $\approx 100000$ — good models sit far below that within the first epoch.

---

## Study Guide — Chapter Dependencies

```
§1 LM objective ──► §9 Loss / CE
§5 Attention ─────► MLA.md (low-rank KV)
§6 RoPE ──────────► MLA.md (decoupled PE)
§7 SwiGLU ────────► transformer.md (dense FFN)
§8 MoE intuition ─► moe.md (aux-loss-free)
§12 Chinchilla ───► training.md (token budget)
§13 μP ───────────► training.md (LR scaling)
```

---

## Practice Problems

1. **RoPE period:** With `rope_theta=10000`, `qk_rope_head_dim=24`, compute $	heta_0$ for the first frequency pair.
2. **SwiGLU params:** Derive $3 d I$ parameter count for dense FFN with dim $d$, intermediate $I$.
3. **MoE FLOPs:** Compare one dense layer ($I=1536$) vs one MoE layer (top-4 of 20, $I_{moe}=384$, +1 shared).
4. **μP scaling:** If params double, by what factor should LR change under square-root scaling?
5. **KV bytes:** Compute MLA cache bytes/token/layer vs full MHA with $d_h=64$, 12 heads.



## Extended μP Theory

Full μP (Yang et al., 2021) requires width-dependent init and per-tensor LR. This repo uses pragmatic transfer:

$$
\eta_{\text{new}} = \eta_{\text{ref}} \sqrt{\frac{N_{\text{ref}}}{N_{\text{actual}}}}
$$

Applied in `Pretrainer.__init__` **after** MTP wrap. Locked by `test_mup_lr_scaling`.

---

## Load-Bearing Invariants

| Invariant | Why it matters |
|---|---|
| Causal mask on all attention paths | Prevents future-token leakage |
| `use_cache=False` in training | Broken gradients if cache enabled |
| μP LR after MTP param count | Wrong LR if counted before wrap |
| Weight tying dedup in param count | Double-counting inflates μP denominator |
| BF16 forward + FP32 optimizer state | Standard stable mixed-precision recipe |
| MoE bias is a buffer, not Parameter | Aux-loss-free routing must not get weight decay |

---

## FAQ

**Q: Do I need to read this entire chapter before MLA.md?**
A: Skim §5 (attention), §6 (RoPE), and §11 (KV cache). Return here when other sections reference unfamiliar concepts.

**Q: Why not use HuggingFace Transformers?**
A: This project is an educational from-scratch implementation. Every line must be inspectable. HF hides MLA absorption, aux-loss-free routing, and MTP alignment behind abstractions.

**Q: What's the minimum math background?**
A: Linear algebra (matrix multiply, softmax), basic calculus (chain rule for backprop intuition), and probability (cross-entropy). No measure theory required.

**Q: How does this relate to the full DeepSeek-V3 (671B)?**
A: Same architectural family (MLA + aux-loss-free MoE + MTP), scaled down to 422M for single-GPU training. The math is identical; only dimensions and expert counts differ.

---

## References

| Topic | Citation |
|---|---|
| Transformer | Vaswani et al., 2017 — arXiv:1706.03762 |
| RoPE | Su et al., 2021 — arXiv:2104.09870 |
| RMSNorm | Zhang & Sennrich, 2019 — arXiv:1910.07467 |
| SwiGLU | Shazeer, 2020 — arXiv:2002.05202 |
| Chinchilla | Hoffmann et al., 2022 — arXiv:2203.15556 |
| μP | Yang et al., 2021 — arXiv:2203.03466 |
| AdamW | Loshchilov & Hutter, 2019 — ICLR |
| DeepSeek-V2 (MLA) | Liu et al., 2024 — arXiv:2405.04434 |
| DeepSeek-V3 | DeepSeek-AI, 2024 — arXiv:2412.19437 |
| MoE survey | Fedus et al., 2022 — JMLR |

**Source files:** `models/transformer.py`, `training/pretrain.py`, `configs/pretrain_a100_422m.yaml`

## Causal Masking — Why Upper Triangular?

Attention with mask $M_{ij} = 0$ if $j \le i$, else $-\infty$:

$$
\text{softmax}\left(\frac{QK^\top}{\sqrt{d}} + M\right)
$$

Position $i$ cannot attend to future positions $j > i$. **Training** uses full $(S, S)$ mask. **Inference** with cache uses $S_{\text{query}}=1$ — mask unnecessary.

---

## Expert Routing Intuition (MoE Preview)

Given token embedding $\mathbf{x}$:

1. Router scores: $\mathbf{s} = \sigma(W_g \mathbf{x})$ (sigmoid gate)
2. Select top-$k$ experts by $s_i + b_i$ (learnable bias buffer $b$)
3. Run $k$ expert FFNs + shared expert; combine weighted outputs

**Aux-loss-free:** No $\mathcal{L}_{\text{aux}}$ in gradient; bias updated out-of-band. See [moe.md](moe.md).

---

## Practice Problems (with hints)

1. **RoPE period:** With $\theta_k = 10000^{-2k/d_{\text{rope}}}$, which dimensions rotate fastest? *Hint: smallest $k$.*

2. **Chinchilla tokens:** How many tokens for a 1B param model? *Answer: ~20B.*

3. **μP scaling:** If params quadruple, how does sqrt-scaling change LR? *Answer: halved.*

4. **KV bytes:** Compute MLA cache per layer per token at $R=192$, $d_{\text{rope}}=24$, BF16. *Answer: 432 bytes.*

5. **MoE FLOPs:** 20 experts, top-4 + 1 shared — what fraction of expert weights executed per token? *Answer: 5/21 stored, 5 executed.*

---

## Glossary (Extended)

| Term | Definition |
|---|---|
| Teacher forcing | Training uses ground-truth previous tokens as input |
| Autoregressive | Each generated token becomes input to next step |
| Absorption trick | Fold KV projection into Q to avoid materialising full K,V |
| Chinchilla-optimal | ~20 tokens per parameter for compute-efficient training |
| MFU | Achieved FLOPs / peak hardware FLOPs |
| Speculative decode | Draft model proposes tokens; main model verifies |

<!-- docs:verified 2026-07-31 · 5a880d2 -->
