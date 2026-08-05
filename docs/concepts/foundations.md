# DeepSeek-v3-Lite — Foundations & Architecture

> **Canonical** for the from-scratch primitives every other chapter assumes: causal LM objective, pre-norm residuals, attention, RoPE, KV caching, Chinchilla scaling, μP, mixed precision, and the numerical tools you need to read the rest of the docs. Educational textbook chapter.

> The conceptual base layer. DeepSeek lineage (V1 → V2 → V3) plus the mathematical primitives that every later chapter builds on. Read this before [Attention & Precision](../concepts/attention-and-precision.md), [MoE & MTP](../concepts/moe-mtp.md), or [Training](../training.md) — each of those cites sections here by number.

**Depends on:** [Getting Started](../guides/getting-started.md) · **Read next:** [Attention & Precision](../concepts/attention-and-precision.md), [MoE & MTP](../concepts/moe-mtp.md)

---

## Table of Contents

1. [The DeepSeek evolutionary lineage](#1-the-deepseek-evolutionary-lineage)
2. [Causal language modeling & cross-entropy loss](#2-causal-language-modeling--cross-entropy-loss)
3. [Pre-norm residual blocks & RMSNorm](#3-pre-norm-residual-blocks--rmsnorm)
4. [SwiGLU feed-forward network](#4-swiglu-feed-forward-network)
5. [Multi-head attention — the standard baseline](#5-multi-head-attention--the-standard-baseline)
6. [Rotary position embeddings (RoPE)](#6-rotary-position-embeddings-rope)
7. [Attention scaling & causal masking](#7-attention-scaling--causal-masking)
8. [The residual stream & information flow](#8-the-residual-stream--information-flow)
9. [The training loss — CE plus the MTP auxiliary](#9-the-training-loss--ce-plus-the-mtp-auxiliary)
10. [Optimization — AdamW, warmup-cosine, gradient clipping](#10-optimization--adamw-warmup-cosine-gradient-clipping)
11. [KV caching — the inference bottleneck](#11-kv-caching--the-inference-bottleneck)
12. [The Chinchilla token budget & compute accounting](#12-the-chinchilla-token-budget--compute-accounting)
13. [μP — maximal-update parameterization](#13-μp--maximal-update-parameterization)
14. [Mixed precision — BF16 training](#14-mixed-precision--bf16-training)
15. [Gradient checkpointing](#15-gradient-checkpointing)
16. [Weight tying](#16-weight-tying)
17. [Tokenization — BPE and the DeepSeek tokenizer](#17-tokenization--bpe-and-the-deepseek-tokenizer)
18. [The matrix calculus you need](#18-the-matrix-calculus-you-need)
19. [Weight initialization](#19-weight-initialization)
20. [Worked example — one forward pass at 411.6M scale](#20-worked-example--one-forward-pass-at-4116m-scale)
21. [The loss landscape & training dynamics](#21-the-loss-landscape--training-dynamics)
22. [Practice problems (with answers)](#22-practice-problems-with-answers)
23. [Core notation & glossary](#23-core-notation--glossary)
24. [Load-bearing invariants](#24-load-bearing-invariants)
25. [Check your understanding](#25-check-your-understanding)
26. [References](#26-references)

---

## 1. The DeepSeek evolutionary lineage

```
DeepSeek-V1 (2024)
   │  • Standard Dense Transformer + basic MoE (Top-2 routing over coarse experts)
   │  • Problem: high KV-cache memory footprint; conventional aux-loss degrades main loss
   ▼
DeepSeek-V2 (2024)
   │  • Multi-Head Latent Attention (MLA) — low-rank KV compression (paper d_c = 512)
   │  • DeepSeekMoE — fine-grained experts (paper: 64 experts, 6 active) + 2 shared experts
   │  • Result: up to 93.3% KV-cache memory reduction; higher expert specialization
   ▼
DeepSeek-V3 (2024)
      • Auxiliary-loss-free load balancing via dynamic expert bias adjustment
      • Multi-Token Prediction (MTP) for dense training signal + speculative decoding
      • FP8 mixed precision (E4M3/E5M2) with block-wise scaling      [paper-spec in this repo]
      • DualPipe bidirectional pipeline parallelism                  [paper-spec in this repo]
```

> [!NOTE] **Lineage numbers are the paper's, not this repo's.** The V2/V3 rows above describe the *published* DeepSeek models. This reproduction (DeepSeek-**v3-Lite**) is a ~412 M-param single-GPU model: `d_c = 192`, 20 routed experts (4 active) + 1 shared, BF16. The canonical config (`configs/pretrain_a100_422m.yaml` — "422m" is the *filename*, a historical nominal) instantiates **411,632,256** deduped parameters (411.6 M) without MTP and **418,713,984** (418.7 M) with the MTP module. Don't conflate the paper's 14 B / 64-expert numbers with this repo's 411.6 M / 20-expert numbers — they are different scales of the same architecture.

The lineage matters because each V3 component is the *answer to a specific failure*. Read the chapter through that lens:

| V3 component | Failure it answers | Chapter |
|---|---|---|
| **MLA** (§3 of [03 Multi Head Latent Attention](../concepts/attention-and-precision.md)) | KV-cache memory dominates long-context inference | §5, §6, §11 here |
| **DeepSeekMoE, aux-loss-free** | Dense FFN cost scales with capacity; aux-loss gradient contaminates the task loss | §4, §12 |
| **MTP** | One next-token target per position under-supervises the trunk | §9 |
| **μP LR transfer** | LR tuned at one width fails at another | §13 |

### 1.1 How to read this chapter

Each of the next twenty sections is a *from-scratch* derivation of one primitive, in the order a transformer needs them: first the objective (§2), then the building blocks of the trunk (§3–§8), then training-time machinery (§9–§15), then the vocabulary machinery (§16–§17), then the numerical tools (§18–§19), then two worked passes (§20–§21) and the reference tables (§22–§26). Every code claim is anchored to a real symbol in `models/` or `training/` — you can open the file and find the exact line by searching for the symbol name. Where a number is an estimate rather than a measurement, it is labelled as such; no GPU training run has been executed in this repo, so every memory and latency figure is a budget estimate `[INFERENCE]` derived from the config, not a measurement.

---

## 2. Causal language modeling & cross-entropy loss

### 2.1 The autoregressive decomposition

A language model assigns probability to a token sequence $\mathbf{x} = (x_1, \ldots, x_T)$ by factorizing the joint into conditionals (autoregressive decomposition):

$$p_\theta(\mathbf{x}) = \prod_{t=1}^{T} p_\theta(x_t \mid x_{<t})$$

This is just the chain rule of probability, applied left-to-right. It converts "model the whole sequence" into "model each next token given the past" — which is why the architecture is causal and why a single training objective on all positions doubles as a definition of the model. No independence assumption is made; the conditionals share parameters through the trunk.

### 2.2 From logits to probabilities

The model does not output probabilities directly; it outputs **logits** $z \in \mathbb{R}^{V}$ (unnormalized scores), one per vocabulary item. The **softmax** turns them into a valid distribution:

$$p_i = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}$$

Three properties to internalize:

1. **Softmax is invariant to adding a constant to all logits** — $p$ is unchanged if $z \to z + c$. Only *differences* of logits matter. This is why a "temperature" $\tau$ is applied as $z_i / \tau$: scaling logits changes their spread, sharpening ($\tau < 1$) or flattening ($\tau > 1$) the distribution. The repo implements temperature (plus top-k/top-p) in `models/transformer.py:Transformer._sample`, used by `models/transformer.py:Transformer.generate` at inference.
2. **Softmax is a smooth argmax.** It returns near-one-hot distributions when logits are widely separated and near-uniform when they are close.
3. **The gradient of the cross-entropy through softmax is beautifully simple** (§2.4) — this is why classification heads use logits + softmax rather than direct probability regression.

### 2.3 The objective

We train by maximizing the likelihood of the data, equivalently **minimizing the cross-entropy** of the predicted next-token distribution against the observed token:

$$\mathcal{L}_{\text{CE}}(\theta) = -\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})$$

`★ Insight ─────────────────────────────────────` Cross-entropy is *negative log-likelihood* — it is the number of bits (in nats) the model "needs" to encode the true token under its distribution. A perfect model assigns probability 1 to the right token (loss 0); a uniform-over-vocab model assigns $1/V$ (loss $\log V$). The gradient $\nabla_\theta \mathcal{L}_{\text{CE}}$ flows only through the position being predicted, which is why the causal mask (§7) is essential: without it, position $t$ could "see" $x_t$ and the loss would be trivially zero. `─────────────────────────────────────────────────`

The connection to information theory: cross-entropy decomposes as

$$\text{CE}(p_{\text{data}}, p_\theta) = H(p_{\text{data}}) + D_{\mathrm{KL}}(p_{\text{data}} \,\|\, p_\theta)$$

where $H$ is the (fixed) entropy of the data distribution and $D_{\mathrm{KL}}$ is the Kullback–Leibler divergence the model is actually minimizing. The entropy term is a constant the model can't affect, so minimizing CE is *exactly* minimizing the KL divergence between the data and the model — i.e., learning the data distribution. This "compression" framing is why lower CE ⇒ the model compresses the corpus better.

### 2.4 The softmax-minus-target gradient

Write $p = \text{softmax}(z)$ and let the loss at one position be $\mathcal{L} = -\log p_y$ (the true token is $y$). Then:

$$\frac{\partial \mathcal{L}}{\partial z_i} = p_i - \mathbf{1}[i = y]$$

**"Softmax output minus one-hot target."** This single identity is why logits-softmax-CE is the workhorse: the gradient is a *difference of distributions*, it costs nothing extra to compute (the softmax probabilities are already in hand), and it is bounded in $[-1, 1]$ per element — numerically benign compared to, say, the squared-error gradient through a sigmoid, which saturates.

The derivation is two lines and worth doing once. Write $\mathcal{L} = -z_y + \log \sum_j e^{z_j}$. Then $\partial \mathcal{L}/\partial z_i = -\mathbf{1}[i=y] + e^{z_i}/\sum_j e^{z_j} = p_i - \mathbf{1}[i=y]$. Note also that $\sum_i \partial \mathcal{L}/\partial z_i = 1 - 1 = 0$: the gradient is always **zero-mean**, which reflects the shift-invariance of softmax (§2.2) — the model can only learn *differences* between logits, never their absolute level. If you ever see a logit-gradient that does not sum to zero, there is a bug.

### 2.5 Numerical stability

Naively computing $e^{z_i}$ overflows when $z_i$ is large (e.g. $e^{100}$ exceeds float range). The stable form subtracts the row max $m = \max_j z_j$:

$$p_i = \frac{e^{z_i - m}}{\sum_j e^{z_j - m}}$$

Since softmax is shift-invariant (§2.2), this is mathematically identical. PyTorch's `F.cross_entropy` fuses log-softmax + NLL for exactly this reason — never implement CE as `log(softmax(...))` in training code; it both wastes memory and risks overflow.

### 2.6 Perplexity

**Perplexity** is the standard scalar report of LM quality:

$$\mathrm{PPL} = \exp(\mathcal{L}_{\text{CE}})$$

Intuitively it is the **effective branching factor**: how many tokens the model is, on average, choosing uniformly among at each step.

| Loss (nats) | PPL | Interpretation |
|---|---|---|
| $\log V \approx 11.5$ | $\approx 100000$ | Uniform over the 100K vocab (random init) |
| 4.0 | 54.6 | Learned subword structure |
| 2.8 | 16.4 | Mid-training ~412M |
| 2.0 | 7.4 | Very good small LM |

In code, `TrainingLogger` prints `ppl = exp(avg_loss)` (`utils/logging.py:TrainingLogger.log`). A PPL equal to $V$ means the model has learned nothing; every factor of $e$ reduction in loss halves the branching factor. Because nats and bits are related by a constant ($1$ nat $= 1/\ln 2 \approx 1.44$ bits), a loss of 2.8 nats is about 4.0 bits per token — both appear in the literature; this repo logs nats.

### 2.7 In this repo

`targets` is `tokens` shifted by one position — predict $x_{t+1}$ from $x_{\le t}$. The loss is one line on the flattened logits (`training/pretrain.py`):

```python
main_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
```

`ignore_index=-100` is the standard mask sentinel: padded positions (if any) are labelled $-100$ and contribute no gradient. Label smoothing is **not** used here — the model trains on hard one-hot targets.

### 2.8 Teacher forcing

Notice what the objective does *not* do: at every training step the model predicts $x_{t+1}$ from the **ground-truth** prefix $x_{\le t}$, never from its own previously sampled tokens. This is **teacher forcing**: the "teacher" (the true text) is fed in, and the model is only ever asked to score the correct continuation one position ahead.

Why is this the right thing for pretraining? The alternative — feeding the model its own samples ("free-running") — has two fatal problems:

1. **The sampled token is discrete.** A sampled token $x_{t+1}$ has no gradient w.r.t. the model parameters (an `argmax` or `multinomial` is not differentiable), so a free-running model cannot be trained by backprop through its own outputs without additional machinery (e.g., REINFORCE-style estimators, which are high-variance).
2. **Errors compound.** If the model feeds its own mistakes forward, the training distribution drifts further from the true text the deeper the rollout goes, and the learning signal becomes noise.

Teacher forcing sidesteps both: the forward pass is a pure differentiable function of the input tokens, and every position is supervised against the true next token. The price is **exposure bias** — at inference the model free-runs on its own samples (see `models/transformer.py:Transformer.generate`), so the input distribution shifts from ground truth to model output. For causal LMs this is largely absorbed by the fact that each step is a *local* correction (one token at a time, with the KV cache), and it is the standard training regime for every model in this repo and in the DeepSeek lineage.

### 2.9 The loss is a mean over positions and batches

The flattened `F.cross_entropy` call reduces over *all* positions in the batch: with micro-batch 8 and sequence 2048, that is 16,384 positions per micro-step, and the returned scalar is the mean per-position loss. Two consequences:

- **The gradient is an average too.** `∂L/∂θ` is the mean over 16,384 positions of the per-position gradient. This keeps the gradient scale independent of sequence length and batch size — if the loss summed instead of averaged, doubling the sequence would double every gradient.
- **`ignore_index` renormalizes.** PyTorch's `cross_entropy` with `ignore_index=-100` excludes ignored positions from *both* the numerator and the denominator, so the mean is over valid positions only. In packed pretraining (see [09 Data Pipeline](../concepts/data-pipeline.md)) padding is rare, so `-100` is mostly a safety sentinel — but the semantics matter the day you mix padded and packed batches.

The repo divides by the gradient-accumulation factor before backprop (`training/pretrain.py:Pretrainer.train_step`), so four micro-steps of 16,384 positions each accumulate into one optimizer step whose gradient is the mean over 65,536 positions:

```python
loss = main_loss / self.config.gradient_accumulation_steps
```

### 2.10 What "good" looks like at this scale

The uniform baseline is $\log V = \log 100018 \approx 11.51$ nats. Anything below ~6 nats means the model has learned token statistics; below ~4 nats it has learned subword and syntactic structure; below ~3 nats (PPL < 20) it is a functioning small LM. There is no GPU run in this repo yet, so the "mid-training 2.8" row in the §2.6 table is a planning target, not a measurement `[INFERENCE]`. The practical way to watch the loss in this repo is `utils/logging.py:TrainingLogger.log`, which prints `loss`, `ppl`, `lr`, and tokens/sec every `log_interval` steps.

---

## 3. Pre-norm residual blocks & RMSNorm

### 3.1 The residual block

Every transformer layer is a **residual block** with a pre-norm:

$$h \leftarrow h + f(\text{RMSNorm}(h))$$

where $f$ is attention or the FFN. The residual connection is not a convenience — it is what makes deep training possible. Without it, the gradient at layer $\ell$ would be the product of many layer-wise Jacobians; with it, the Jacobian of the block is $I + J_f$ (identity plus a correction), so gradients flow through the identity term essentially unattenuated. This is the "residual highway" that lets gradients reach the embedding at depth 18+.

### 3.2 Pre-norm vs post-norm

- **Pre-norm:** the normalization is applied to the *input* of the sublayer, so the residual stream $h$ is never normalized away. Post-norm (original Transformer) normalizes the output of each sublayer, which re-scales the signal before it re-enters the residual path; this is empirically unstable past ~12 layers and requires careful init/LR.
- **Consequence:** with pre-norm, the *final* residual is unnormalized, so a final `RMSNorm` must be applied before the LM head (`self.head(self.norm(h))` in `models/transformer.py:Transformer.forward`). The repository does exactly this.

### 3.3 LayerNorm vs RMSNorm

**LayerNorm** (Ba, Kiros & Hinton, 2016) normalizes with both a learned mean-shift and scale:

$$\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta, \qquad \mu = \tfrac{1}{d}\sum_i x_i,\; \sigma^2 = \tfrac{1}{d}\sum_i (x_i - \mu)^2$$

**RMSNorm** (Zhang & Sennrich, 2019) keeps only the root-mean-square scaling, dropping the mean subtraction *and* the bias:

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\tfrac{1}{d}\sum_i x_i^2 + \epsilon}} \odot \gamma$$

| | LayerNorm | RMSNorm |
|---|---|---|
| Mean subtraction | Yes | No |
| Learned scale $\gamma$ | Yes | Yes |
| Learned bias $\beta$ | Yes | No |
| Params per norm | $2d$ | $d$ |
| Scale-invariant ($x \to c x$) | Yes | Yes (up to sign) |
| Translation-invariant ($x \to x + c$) | Yes | **No** |

The key property is **scale-invariance**: $\text{RMSNorm}(c x) = \operatorname{sgn}(c)\,\text{RMSNorm}(x)$, so the norm resets the magnitude of the stream regardless of how large it has grown — this is what stabilizes the residual stream. LayerNorm additionally subtracts the mean; RMSNorm argues this re-centering is unnecessary for transformers, where activations sit near zero anyway, and that removing it cuts a reduction + broadcast per norm (a real speed win at hundreds of norm layers) with marginally better gradient flow. Every DeepSeek (and LLaMA/Mamba) block uses `nn.RMSNorm(dim, eps=1e-6)`.

**Why `eps=1e-6`?** The epsilon is a numerical floor against dividing by a near-zero RMS. This repo (and DeepSeek) use $10^{-6}$, tighter than the $10^{-5}$ many LLaMA implementations use — allowing the norm to be "more aggressive" when the RMS is small, at the cost of requiring initialization that keeps activations away from zero (§19).

### 3.4 The canonical block

The repo's `TransformerBlock` (`models/transformer.py`) is the canonical form:

```python
def forward(self, x: torch.Tensor, start_pos: int = 0, mask: Optional[torch.Tensor] = None, use_cache: bool = True) -> torch.Tensor:
    x = x + self.attn(self.attn_norm(x), start_pos, mask, use_cache)
    x = x + self.ffn(self.ffn_norm(x))
    return x
```

Two residual additions, two pre-norms — attention then FFN. The residual stream `x` carries information; the sublayers *add corrections* to it.

### 3.5 RMSNorm, derived from scratch

**What normalization must do.** Deep networks suffer from "internal covariate shift"-adjacent problems: the scale of activations is not controlled by anything, so a slightly-too-large weight at layer 5 can compound into saturated or vanishing signals by layer 18. Normalization forces each layer to see an input with a *fixed, known scale* — the layer then only has to learn the shape, not the magnitude.

**Why drop the mean.** The mean subtraction in LayerNorm exists to re-center activations. In a transformer with pre-norm residuals, the input to every norm is a sum of corrections written into the residual stream; with symmetric initialization these sums hover near zero mean already, and any constant offset is a *bias-like* signal the model can learn to use or ignore. RMSNorm's bet is that the mean carries no information worth a dedicated per-layer estimate — the win is one fewer reduction and one fewer broadcast per norm, plus one fewer set of parameters ($d$ instead of $2d$).

**The forward pass.** Define the mean square $m(x) = \tfrac{1}{d}\sum_i x_i^2$ and the RMS $r(x) = \sqrt{m(x) + \epsilon}$. Then

$$y_i = \frac{x_i}{r(x)} \qquad\text{(then } \tilde{y}_i = \gamma_i y_i\text{)}$$

The output has RMS $\approx 1$ (up to $\epsilon$): $\frac{1}{d}\sum_i y_i^2 = \frac{m(x)}{m(x)+\epsilon} \to 1$. The learned gain $\gamma \in \mathbb{R}^d$ (initialized to ones by `nn.RMSNorm`) then rescales each channel to whatever variance the following layer wants.

**Scale-invariance, proved.** For $c > 0$: $m(cx) = c^2 m(x)$, so $r(cx) = c\, r(x)$, and $y(cx)_i = cx_i/(c\,r(x)) = y(x)_i$. For $c < 0$ the sign flips: $y(cx) = \operatorname{sgn}(c)\,y(x)$. Two consequences: (1) the norm output is insensitive to the absolute scale of the stream — the stream can grow or shrink across depth without the sublayers noticing; (2) the *norm's input scale* is not a learnable signal, which is precisely why the residual stream needs no per-layer rescaling.

**The gradient.** Let $g = \partial \mathcal{L}/\partial y$ be the upstream gradient. Differentiate $y_i = x_i / r$:

$$\frac{\partial y_i}{\partial x_j} = \frac{\delta_{ij}}{r} - \frac{x_i}{r^2}\frac{\partial r}{\partial x_j}, \qquad \frac{\partial r}{\partial x_j} = \frac{x_j}{d\, r}$$

Chaining with $g$ and pulling the sum through:

$$\frac{\partial \mathcal{L}}{\partial x_j} = \frac{1}{r}\left( g_j - x_j \cdot \frac{\langle g, x\rangle}{d\, r^2} \right), \qquad \langle g, x\rangle = \sum_i g_i x_i$$

Read this as: **"upstream gradient minus its projection onto the input, divided by the RMS."** The first term is the direct path; the second subtracts the component of $g$ parallel to $x$ — the norm "explains away" any gradient that would simply change the overall magnitude, because magnitude changes are invisible to the next layer (scale-invariance). With the learned gain, $g$ is first multiplied by $\gamma$; PyTorch's `nn.RMSNorm` computes exactly $y = (x/\sqrt{\text{mean}(x^2)+\epsilon}) \cdot \gamma$ and leaves the backward to autograd. The gradient is $O(1)$ in the input magnitude (the $1/r$ factor), which is what makes the norm numerically benign compared to saturating nonlinearities.

### 3.6 Code walkthrough — where the norms live

Every normalization in this repo is `nn.RMSNorm(dim, eps=1e-6)`. At the canonical config the instances are:

| Location | Instance | Count |
|---|---|---|
| `models/transformer.py:TransformerBlock.__init__` | `attn_norm`, `ffn_norm` — one pair per layer | $18 \times 2 = 36$ |
| `models/transformer.py:Transformer.__init__` | `self.norm` — final norm before the LM head | 1 |
| `models/mla.py:MultiHeadLatentAttention.__init__` | `kv_norm` — normalizes the KV latent | 18 |
| `models/mtp.py:MTPBlock.__init__` + `models/mtp.py:MTPModule.__init__` | `norm_h`, `norm_e`, `norm_attn`, `norm_ffn`, `norm` | 5 |

That is 37 norms in the trunk alone (the "37+" figure) and 60 total once MLA and MTP are counted. The trunk construction is visible in `models/transformer.py:TransformerBlock.__init__`:

```python
self.attn_norm = nn.RMSNorm(self.dim, eps=1e-6)
self.attn = MultiHeadLatentAttention(config, layer_id)
self.ffn_norm = nn.RMSNorm(self.dim, eps=1e-6)
self.ffn = SwiGLUFFN(self.dim, config["inter_dim"]) if layer_id < self.n_dense_layers else DeepSeekMoE(config)
```

MLA adds two more normalization sites — `q_norm` (only when `q_lora_rank > 0`, so *absent* at the canonical config where `q_lora_rank: 0`) and `kv_norm` (always present, normalizing the compressed latent before it is written to cache or expanded):

```python
self.kv_norm = nn.RMSNorm(self.kv_lora_rank, eps=1e-6)
```

The `kv_norm` placement is deliberate and load-bearing: the cache stores the *normalized* latent, so a change to the norm would invalidate cached values — see §11.5 and [03 Multi Head Latent Attention](../concepts/attention-and-precision.md).

### 3.7 RMSNorm pitfalls

- **eps too large flattens small activations.** With `eps=1e-6` the norm faithfully rescales even RMS-0.01 activations; with `1e-5` (a common LLaMA value) a 0.01-RMS signal would be pulled toward $1/\sqrt{1+100\epsilon}$-ish distortion. Consistency across the 60 norms matters: mixing eps values changes the effective scale of the residual stream per layer.
- **RMSNorm is not translation-invariant.** An input $x + c\mathbf{1}$ changes the output (unlike LayerNorm). This is why the embedding is initialized with a tiny std (0.006, §19): a large DC offset at the stream's birth would be amplified by the first norm.
- **Under autocast, norms stay in FP32.** PyTorch's `autocast` casts matmul-heavy ops to BF16 but keeps normalization (and softmax) in FP32 — the RMS reduction is a sum of $d$ squares, which is exactly the kind of reduction that loses precision in BF16. Don't "optimize" by forcing the norm into BF16.
- **The final norm is the head's only scale control.** With weight tying (§16), the LM head *is* the embedding matrix; the final `RMSNorm` in `models/transformer.py:Transformer.forward` (`self.head(self.norm(h))`) is what keeps logits from exploding as the stream grows across 18 layers.

---

## 4. SwiGLU feed-forward network

### 4.1 The vanilla FFN and its problem

The classic transformer FFN is two matrices with a pointwise nonlinearity in between: $\text{FFN}(x) = W_2\, \phi(W_1 x)$ with $\phi = \text{ReLU}$ or $\text{GELU}$. This is a per-position, channel-mixing computation: it transforms each token independently (unlike attention, which mixes across positions). Its capacity is limited by how expressive a single fixed nonlinearity can be.

### 4.2 Gating: the SwiGLU idea

**SwiGLU** (Shazeer, 2020) replaces the single path with a **gated** path — two parallel projections, one acting as a value, the other as a smooth gate:

$$\text{SwiGLU}(x) = W_2\big(\text{silu}(W_1 x) \odot (W_3 x)\big)$$

Three weight matrices: $W_1, W_3$ project up to `inter_dim`, the SiLU gate selects, $W_2$ projects back down. The elementwise product $\text{silu}(W_1 x) \odot (W_3 x)$ is the gating — $W_3 x$ provides a "value" and $\text{silu}(W_1 x)$ a smooth "gate" that lets the model pass or suppress each hidden dimension.

**SiLU (a.k.a. Swish)** is $\sigma(x) \cdot x$ — the sigmoid times the input:

- **Smooth and monotonic**, unlike ReLU.
- **Bounded below, unbounded above** — the negative branch saturates to 0 but never exactly 0 (no dead neurons).
- Its gradient is $\sigma(x) + x\,\sigma(x)(1 - \sigma(x))$, which is **non-zero almost everywhere** — every channel learns throughout training, whereas ReLU can pin channels to zero gradient forever.

SwiGLU beats the vanilla two-matrix FFN (ReLU/GELU) at equal parameter count because the gate adds a *multiplicative* nonlinearity: the effective function is a *product* of two learnable transformations, which is strictly more expressive than a single nonlinearity applied to one transformation. Empirically (T5/PaLM-era ablations) SwiGLU wins at fixed FLOPs and fixed parameter counts alike.

### 4.3 Parameter and FLOP accounting

$$\text{Params} = 3\,d\,I, \qquad \text{FLOPs/token} \approx 6\,d\,I$$

where $I$ is the intermediate width. The three matrices are $W_1: d \to I$, $W_3: d \to I$, $W_2: I \to d$; the "6" is the three matmuls, each with a multiply and an add per element (forward; backward is another $2\times$ on top, see §12.3).

At the canonical 411.6M scale, dense layers use $d=768$, $I=1536$ → **3.5M params / ~7.1 MFLOPs per token per layer**. Each MoE expert is a smaller SwiGLU with $I=384$ (see [04 DeepSeekMoE](../concepts/moe-mtp.md)).

### 4.4 SwiGLU in code, annotated

The repo's dense FFN is exactly the formula above (`models/transformer.py:SwiGLUFFN.forward`):

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

Read the line right-to-left: `w1(x)` is the *gate* path (silu-activated), `w3(x)` is the *value* path, the `*` is the elementwise gating product, and `w2` is the down-projection. Every MoE expert is the same three-matrix structure with a smaller intermediate width (`models/moe.py:Expert`), so everything in this section applies to experts too — the only difference is which tokens flow through which expert (see [04 DeepSeekMoE](../concepts/moe-mtp.md)).

**The gated backward.** Because the forward is a product $h = \text{silu}(a) \odot b$ with $a = W_1 x$, $b = W_3 x$, the backward splits cleanly by the product rule:

$$\frac{\partial \mathcal{L}}{\partial b} = \frac{\partial \mathcal{L}}{\partial h} \odot \text{silu}(a), \qquad \frac{\partial \mathcal{L}}{\partial a} = \frac{\partial \mathcal{L}}{\partial h} \odot b \odot \text{silu}'(a)$$

Each channel receives the *other* channel's forward value as a coefficient. This is where the gate's multiplicative structure shows up in gradients: the gate path is modulated by the value path and vice versa, so the two projections co-adapt. The same identity is what the fused Triton grouped-MoE kernel has to reproduce in its `dx`/`dw` passes (see [12 Triton Kernels](../concepts/kernels-and-ops.md)).

**Pitfall — no biases anywhere.** Every FFN and expert `Linear` in this repo is `bias=False`. Biases are omitted in the DeepSeek lineage because the gating product + norms make them redundant, and dropping them removes a class of update-scale problems. If you add a bias "to be safe," you are introducing a deviation from every documented parameter count in [02 Model Architecture](../concepts/foundations.md).

---

## 5. Multi-head attention — the standard baseline

Standard Multi-Head Attention (MHA) is what MLA compresses. Recap it here so the compression in [03 Multi Head Latent Attention](../concepts/attention-and-precision.md) has a reference point.

### 5.1 Projections and scaled dot-product attention

For each head $h$, project queries/keys/values and compute scaled dot-product attention:

$$Q_h = x W_h^Q, \quad K_h = x W_h^K, \quad V_h = x W_h^V \in \mathbb{R}^{T \times d_h}$$

$$\text{Attn}_h = \text{softmax}\!\left(\frac{Q_h K_h^\top}{\sqrt{d_h}}\right) V_h$$

Think of attention as a **soft dictionary lookup**: the queries "ask", the keys are the "index" of every stored entry, and the values are the "content" retrieved. The softmax turns the similarity scores $\langle q, k\rangle$ into retrieval weights; the output is a weighted average of the values. Attention is a *data-dependent* weighted average — the weights depend on the input, which is what lets tokens route information to the tokens that need it.

### 5.2 Why scale by $\sqrt{d_h}$

The dot product $\langle q, k\rangle$ of two vectors whose coordinates are roughly $\mathcal{N}(0, 1)$ has mean 0 and variance $d_h$ (sum of $d_h$ independent products). So scores grow with dimension, and unscaled scores push softmax into a near-one-hot regime where gradients vanish (§7.1). Dividing by $\sqrt{d_h}$ holds the variance at 1. This is the same argument as §7.

### 5.3 Multi-head: parallel subspaces

Split $d$ into $H$ heads of dimension $d_h = d/H$. Each head has independent projections:

$$\text{head}_h = \text{Attention}(\mathbf{Q}_h, \mathbf{K}_h, \mathbf{V}_h), \quad \mathbf{O} = \mathbf{W}_O [\text{head}_1; \ldots; \text{head}_H]$$

**Why multiple heads?** A single attention pattern is one learned similarity function. Different heads learn *different* similarity functions in parallel — syntax, coreference, long-range dependencies — then $W_O$ mixes them. Intuitively, heads are an ensemble of retrieval mechanisms sharing the same tokens. At the canonical scale, $H=12$, $d_h = 64$ (for values) — a sweet spot for small models.

**Note for later:** standard MHA appears in this repo only inside the MTP blocks (`models/mtp.py:MTPBlock`). The main trunk uses **MLA**, a compressed variant ([03 Multi Head Latent Attention](../concepts/attention-and-precision.md)).

### 5.4 Complexity

- **Time:** $O(T^2 d)$ — the $QK^\top$ product is $O(T^2 d_h)$ per head.
- **Memory (naive):** the score matrix $QK^\top$ is $O(T^2)$ per head — this is what FlashAttention/SDPA eliminate (§7.3).
- **KV cache (inference):** $O(T \cdot L \cdot H \cdot d_h)$ — every head stores full $K_h$ and $V_h$ for every past token (§11). This is the memory MLA attacks.

### 5.5 Attention from first principles

Before the matrix machinery, build attention as a *mechanism*. Suppose token $i$ needs information that lives at other positions. Three questions, in order:

1. **What is token $i$ looking for?** A query vector $q_i \in \mathbb{R}^{d_h}$.
2. **What does each candidate position $j$ offer?** Two vectors: a key $k_j$ that answers "how relevant am I to this query" and a value $v_j$ that is the actual content to be retrieved.
3. **How do we combine them?** Score each candidate by similarity $s_{ij} = q_i \cdot k_j$, convert scores to non-negative weights summing to 1 via softmax, and output the weighted average $\sum_j w_{ij} v_j$.

A concrete two-token example makes the "soft retrieval" concrete. Let $d_h = 2$:

$$q_0 = (1, 0),\quad k_0 = (1, 0),\quad k_1 = (0, 1),\quad v_0 = (2, 0),\quad v_1 = (0, 2)$$

Token 0's scores are $s_{00} = 1$, $s_{01} = 0$; $\text{softmax}(1, 0) = (0.731, 0.269)$; the output is $0.731 \cdot (2,0) + 0.269 \cdot (0,2) = (1.462, 0.538)$. Token 0 mostly retrieved its own value but pulled a quarter of token 1's. If token 1's key had pointed *at* token 0 (say $k_1 = (1, 0)$), the retrieval weight would have shifted — that data-dependence is the whole point: **the weights are a function of the inputs, not a fixed pattern.**

The batched form is what the code computes: stack queries and keys into matrices and write the attention as

$$\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}\right) V$$

One pass of $QK^\top$ computes all $T^2$ pair scores; one multiplication by $V$ produces all outputs. This is the operation `F.scaled_dot_product_attention` implements under the hood (with the $O(T^2)$ score matrix kept in SRAM — §7.3).

### 5.6 Why multiple heads — the representation argument

One head computes one quadratic form $\langle q_i, k_j\rangle = x_i^\top W_Q^\top W_K x_j$ — a *single* learned similarity function over pairs of tokens. A single function cannot simultaneously track "is $j$ the verb that $i$ modifies," "is $j$ the antecedent of the pronoun at $i$," and "is $j$ a nearby token in the same sentence" — those are different notions of similarity. Heads are how the model gets several similarity functions in parallel: head 1 may specialize in local syntactic relations, head 2 in long-range coreference, and so on, each with its own $W_Q^h, W_K^h, W_V^h$. The output projection $W_O$ then mixes the per-head retrievals into the token's final representation.

The parameter cost is *not* multiplicative: each head uses $3 d_h d$ parameters and the total is $3 d^2$ regardless of how you split it (since $\sum_h d_h = d$). What changes with $H$ is the *number of independently learned similarity functions* and the per-head dimension $d_h$ (which sets the expressiveness of each retrieval and the scale of the scores, §5.2). At $H = 12$, $d_h = 64$, the model has 12 retrieval mechanisms per layer.

### 5.7 From MHA to MLA — why compress

At inference, MHA must remember, for every past token and every layer, all per-head keys and values: $H \cdot (d_k + d_v)$ floats per token per layer. At the canonical config that is $12 \times (72 + 64) = 1632$ floats per token per layer — 3264 bytes in BF16. The full table is in §11.3; the takeaway is that the KV cache is the dominant memory cost of long-context decoding.

**MLA's bet:** the per-head keys and values are *redundant*. They are produced by an up-projection from a shared, lower-dimensional representation, so instead of caching the expanded K and V, cache the small latent $c$ (192 dims) and a tiny position-only key (24 dims) — 216 floats per token per layer, a ~7.6× reduction — and re-expand per-head K, V on the fly at attention time via the shared matrix `wkv_b`. The expansion happens *inside* `models/mla.py:MultiHeadLatentAttention.forward`, which materializes `K_nope`, `K_rope`, and `V` from the cached latent with two batched matmuls before calling SDPA. This is the compression that [03 Multi Head Latent Attention](../concepts/attention-and-precision.md) derives in full; §11 here quantifies why it is worth it.

---

## 6. Rotary position embeddings (RoPE)

### 6.1 The problem

Attention is permutation-invariant: $\text{softmax}(QK^\top)V$ doesn't know the *order* of the tokens unless positions are injected. Position information must be added to queries and keys.

### 6.2 The rotation

RoPE encodes **relative position** by rotating query/key pairs in 2-D subspaces. For dimension pair $(d_{2i}, d_{2i+1})$ at position $t$, apply a rotation by angle $t \cdot \theta_i$:

$$\begin{pmatrix} q_{2i}' \\ q_{2i+1}' \end{pmatrix} = \begin{pmatrix} \cos(t\theta_i) & -\sin(t\theta_i) \\ \sin(t\theta_i) & \cos(t\theta_i) \end{pmatrix} \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}, \qquad \theta_i = \theta^{-2i/d}$$

Because the rotation is block-diagonal over $d/2$ pairs, the whole transformation is a **rotation matrix** $R(t\theta)$ of the $d$-dim space. The frequency schedule $\theta_i = \text{base}^{-2i/d}$ is *geometric*: dimension 0 rotates fastest (period $2\pi$), and the highest dimension rotates slowest (period $2\pi \cdot \text{base}^{(d-2)/d}$, which for $\text{base}=10^4$, $d=24$ is enormous). Slow rotations preserve long-range structure; fast rotations encode fine local offsets — the frequency schedule is what makes RoPE multi-scale.

### 6.3 Why it encodes *relative* position — the key lemma

Rotations compose: rotating by angle $m\theta$ then *undoing* $n\theta$ gives a rotation by $(m-n)\theta$. In matrix form, $R(m\theta)^\top R(n\theta) = R((n-m)\theta)$. Then:

$$\langle \text{RoPE}(q, m), \text{RoPE}(k, n) \rangle = q^\top R(m\theta)^\top R(n\theta)\, k = q^\top R((n-m)\theta)\, k$$

The dot product of two rotated vectors depends only on the **relative** position $n - m$ — absolute positions cancel. This is the entire magic: attention scores become translation-equivariant, and the model can generalize to positions it never trained on (length generalization follows from the same property).

`★ Insight ─────────────────────────────────────` RoPE is a *multiplicative* position encoding, injected into Q and K (not V). The rotation is applied *before* the dot product, so relative position falls out of $QK^\top$ for free. This is why DeepSeek can do **decoupled RoPE** in MLA: a small dedicated RoPE subspace (`d_R = 24` dims) is rotated separately from the content subspace, so the position information — which cannot be absorbed into the latent — stays outside the compressed cache. See [03 Multi Head Latent Attention](../concepts/attention-and-precision.md) §Decoupled RoPE. `─────────────────────────────────────────────────`

### 6.4 Implementation via complex numbers

A rotation in 2D is a multiplication by a unit complex number: $(x, y) \mapsto (x + iy) \cdot e^{i\theta}$. So RoPE can be applied to the whole vector at once by viewing it as a complex tensor and multiplying by a precomputed table of phasors:

```
freqs_cis[t, i] = cos(t·θ_i) + i·sin(t·θ_i)     # shape (max_seq_len, d_rope/2), complex64
x_c = view_as_complex(x.reshape(..., d/2, 2))
out = view_as_real(x_c * freqs_cis[...]).flatten(-2)
```

This is exactly what `models/mla.py:MultiHeadLatentAttention._apply_rope` does: one complex multiply instead of $d/2$ 2×2 matrix multiplies. The `freqs_cis` table is computed once (and grown lazily by `_extend_rope`) and reused across forwards — cheaper than reconstructing cos/sin per step.

### 6.5 Beyond training length — interpolation, extrapolation, YaRN

- **Extrapolation** (using a model at contexts longer than trained) degrades because the model has never seen large absolute positions — the fast-rotating dimensions alias and the attention-density statistics change.
- **YaRN** (Peng et al., 2023) *stretches* the frequency schedule by a factor $s$ ($\theta_i / s$) — **interpolation** rather than extrapolation — and corrects the softmax temperature with a **mscale** factor to compensate for the changed attention-score statistics.

In this repo: `rope_factor` is the YaRN stretch factor, **off at training** (`rope_factor = 1.0`). The base is `rope_theta = 10000`. The mscale machinery exists for future 8K/16K fine-tuning but is dormant at the canonical 2048 context; the exact formulas are in `models/mla.py:MultiHeadLatentAttention.__init__` (§6.8).

### 6.6 Deriving the rotation schedule

**Why block-diagonal rotations?** A rotation of the full $d$-dim vector would mix every coordinate, destroying the per-coordinate semantics the projections learned. Instead RoPE rotates *pairs* of coordinates $(x_{2i}, x_{2i+1})$ by independent angles $t\theta_i$ — a block-diagonal rotation matrix with $d/2$ blocks:

$$R(t\theta) = \begin{pmatrix} R(t\theta_0) & & \\ & \ddots & \\ & & R(t\theta_{d/2-1}) \end{pmatrix}, \qquad R(\phi) = \begin{pmatrix} \cos\phi & -\sin\phi \\ \sin\phi & \cos\phi \end{pmatrix}$$

A rotation is an **orthogonal** transformation ($R^\top R = I$), so RoPE preserves norms: $\|\text{RoPE}(x)\| = \|x\|$. This matters twice: attention scores are dot products (norm-preserving rotations leave their *distribution* alone), and the backward pass through a rotation is numerically well-behaved ($R^\top$ is the inverse).

**Why the geometric schedule?** With a single frequency $\theta$, all pairs would rotate at the same speed: position $t$ and $t + 2\pi/\theta$ would produce identical embeddings — aliasing. The geometric schedule $\theta_i = \text{base}^{-2i/d}$ makes the pairs rotate at *different* rates, so the trajectory of a token's position embedding is a helix-like curve that doesn't repeat until enormous offsets. For $d=24$, $\text{base}=10^4$:

$$\theta_i = 10000^{-2i/24} = 10^{-i/3} \quad\Rightarrow\quad \theta_0 = 1,\; \theta_1 \approx 0.464,\; \theta_3 = 0.1,\; \theta_6 = 0.01,\; \theta_{11} \approx 2.15\times 10^{-4}$$

The period of pair $i$ is $2\pi/\theta_i$: pair 0 completes a full rotation every $2\pi \approx 6.3$ positions (fine-grained local offsets), while pair 11 needs $2\pi \cdot 4642 \approx 29\,160$ positions (essentially a global, position-invariant bias over the 2048-token window). The model can use fast pairs for exact offsets and slow pairs for rough distance bands — a multi-scale position code.

### 6.7 RoPE in code — the complex-multiply walkthrough

The table is built lazily and grown geometrically (`models/mla.py:MultiHeadLatentAttention._extend_rope`):

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

Line by line:

- `inv_freq` is $1/\theta_i = \text{rope\_theta}^{i/(d/2)}$ computed over the $d/2 = 12$ even indices — exactly the $\theta_i = \text{base}^{-2i/d}$ schedule of §6.2.
- `torch.outer(t, inv_freq)` builds the angle matrix $\Phi_{t,i} = t\,\theta_i$ for all positions $0 \le t < \text{grow\_to}$.
- `torch.polar(1, Φ)` turns angles into unit complex numbers $e^{i\Phi}$ — the phasor table `freqs_cis`, dtype `complex64`.
- Growth is geometric (`_rope_seq_len * 2`), capped at `max_seq_len`; the cap is what makes an over-long sequence fail loudly in `forward` rather than silently building an out-of-bounds table.

The application (`models/mla.py:MultiHeadLatentAttention._apply_rope`):

```python
def _apply_rope(self, x: torch.Tensor, start_pos: int, seqlen: int) -> torch.Tensor:
    dtype = x.dtype
    x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = self.freqs_cis[start_pos: start_pos + seqlen].view(1, seqlen, 1, -1)
    return torch.view_as_real(x_c * freqs).flatten(-2).to(dtype)
```

Line by line:

- `x.float()` upcasts to FP32 (complex multiply in BF16 would lose the mantissa); the `reshape(..., -1, 2)` groups coordinates into pairs; `view_as_complex` reinterprets each pair $(x_{2i}, x_{2i+1})$ as the complex number $x_{2i} + i\,x_{2i+1}$.
- `freqs_cis[start_pos : start_pos + seqlen]` picks the rows for **absolute** positions `start_pos … start_pos+seqlen-1` — this is how a cached mid-sequence prefill gets globally-correct positions (the local offset $i$ inside the chunk maps to global position $\text{start\_pos} + i$).
- `x_c * freqs` multiplies each complex coordinate by $e^{i t\theta_i} = \cos + i\sin$: by Euler's formula this is exactly the 2×2 rotation of §6.2 applied to the pair. `view_as_real` recovers the two rotated coordinates; `flatten(-2)` reassembles the $d$-dim vector; `.to(dtype)` casts back to the input dtype (BF16 under autocast).

The multiplication is a *rotation in the complex plane*: $z \cdot e^{i\phi}$ preserves $|z|$, mirroring the orthogonality of the matrix form. The whole operation is one fused complex multiply per position instead of $d/2$ matrix-vector products.

### 6.8 Frequencies in this repo — and the dormant YaRN path

At the canonical config the RoPE subspace is $d_R = 24$ (`qk_rope_head_dim: 24`) with `rope_theta: 10000`, `rope_factor: 1.0`, `mscale: 1.0`, `max_seq_len: 2048`. The softmax scale is set in `models/mla.py:MultiHeadLatentAttention.__init__`:

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

Two branches to internalize:

- **Canonical (`max_seq_len ≤ 4096`):** `softmax_scale = 72^{-1/2}` — the standard $\sqrt{d_k}$ scaling of §7.1, with $d_k = qk\_head\_dim = 48 + 24 = 72$ (content + rope). YaRN is fully dormant: `rope_factor = 1.0` so `mscale = 1.0` and the rope table is built at un-stretched frequencies.
- **Long-context path (`max_seq_len > 4096`, future 8K/16K runs):** the frequencies are stretched by the YaRN factor inside `_extend_rope` (via `rope_factor`), and the softmax scale is corrected by `mscale²` to compensate for the changed score statistics under interpolation.

**Pitfall — `d_R` is not a free knob.** The rope head dim appears in three coupled places: the `qk_head_dim` used for `softmax_scale`, the split `q_nope/q_pe` in `forward`, and the size of `pe_cache`. Changing `qk_rope_head_dim` without touching all three silently corrupts either the scores or the cache layout.

---

## 7. Attention scaling & causal masking

### 7.1 Why the $\frac{1}{\sqrt{d_h}}$ scale

For two independent vectors $q, k \in \mathbb{R}^{d_h}$ with coordinates $\sim \mathcal{N}(0,1)$, the dot product is a sum of $d_h$ independent terms, so:

$$\mathbb{E}[\langle q, k\rangle] = 0, \qquad \text{Var}[\langle q, k\rangle] = d_h$$

Unscaled scores grow with dimension: for $d_h = 64$, typical score magnitude is $\sqrt{64} = 8$. Feeding scores with variance $d_h$ into softmax pushes them into the saturation region where one entry dominates and the gradient (a difference of near-0/1 probabilities, §2.4) vanishes. Dividing by $\sqrt{d_h}$ holds score variance at 1 — the softmax operates in its sensitive, high-gradient regime.

The scale shows up explicitly in MLA as `softmax_scale` (§6.8), passed to `F.scaled_dot_product_attention` in `models/mla.py:MultiHeadLatentAttention.forward` — the repo never hardcodes the $\sqrt{d}$ inside the attention call; it is a computed attribute of the config.

### 7.2 The causal mask

The **causal mask** prevents position $t$ from attending to positions $> t$: an additive $-\infty$ above the diagonal of the $T \times T$ score matrix.

```
     pos:  0    1    2    3
  0        0   -inf -inf -inf
  1        0    0   -inf -inf
  2        0    0    0   -inf
  3        0    0    0    0
```

**Why additive $-\infty$, not a multiplicative 0/1?** Softmax exponentiates the scores, so masking must zero out the *exponentiated* weights. An additive $-\infty$ becomes $e^{-\infty} = 0$ in the numerator — exactly "do not attend". A multiplicative 0/1 mask on raw scores would instead leave $e^{0} = 1$ entries, which is wrong. This is why additive $-\infty$ masks are the universal convention.

The repo builds it once and caches it keyed by `(seqlen, kv_len, start_pos, device)` (`models/transformer.py:Transformer._build_causal_mask`). The mask is causal by global position: with a KV cache spanning the past, a cached mid-sequence prefill still cannot attend its own future. At inference decode, only the *new* token's queries attend to all cached keys, so the mask collapses to "attend to everything up to me" — **no masking needed during decode**, only during prefill/training when query length equals key length. The `seqlen == 1` fast path skips the mask entirely.

### 7.3 FlashAttention and SDPA

Naive attention materialises the score matrix $\mathbf{A} \in \mathbb{R}^{S \times S}$ per head in HBM. At $S=2048$, $H=12$, BF16: that's $\approx 100$ MB/layer just for the intermediate scores — and it's written and re-read three times (scores → softmax → output).

**FlashAttention** (Dao et al., 2022) tiles the computation along the key axis and keeps the running $(m, \ell, o)$ online-softmax accumulators *in SRAM*:

1. Load a query block into SRAM.
2. For each key block: load it, compute partial scores in SRAM, update the running max $m$, normalizer $\ell$, and output accumulator $o$ (online softmax — rescaling as the running max grows).
3. Only the output block is written back.

The $O(S^2)$ score matrix **never touches HBM**: memory traffic drops from $O(S^2)$ to $O(S)$ while FLOPs stay $O(S^2)$. This is why `F.scaled_dot_product_attention` (PyTorch's wrapper — FlashAttention-2 on CUDA, memory-efficient or math kernels elsewhere) is the default attention backend in this repo (`attn_impl: "sdpa"`), and why the custom MLA Triton kernel (§docs/12) is FA2-style.

For long context (`max_seq_len > 4096`), MLA applies a **YaRN mscale** correction to the softmax scale to counteract the density loss of stretched RoPE frequencies (`models/mla.py:MultiHeadLatentAttention.__init__`). At the canonical 2048 context this path is inactive.

### 7.4 The mask in code — three geometries, one function

The mask construction (`models/transformer.py:Transformer._build_causal_mask`) is worth reading in full because it encodes the whole KV-cache geometry in one comparison:

```python
def _build_causal_mask(self, seqlen: int, kv_len: int, start_pos: int, device: torch.device) -> torch.Tensor:
    """Additive causal mask (1,1,S_q,S_kv), causal by global position.
    ...
    """
    key = (seqlen, kv_len, start_pos, device)
    if self._mask_cache is None or key != self._mask_key:
        q = torch.arange(seqlen, device=device)[:, None] + start_pos
        k = torch.arange(kv_len, device=device)[None, :]
        mask = torch.where(q >= k, torch.zeros((), device=device), torch.full((), float("-inf"), device=device))
        self._mask_cache = mask.unsqueeze(0).unsqueeze(0)
        self._mask_key = key
    return self._mask_cache
```

- **Query rows are global positions:** `q = arange(seqlen) + start_pos` — a query at local offset $i$ occupies global position $\text{start\_pos} + i$.
- **Key columns are cached positions:** `k = arange(kv_len)` runs $0 \ldots \text{kv\_len}-1$, which is exactly the set of global positions in the cache (positions $0 \ldots \text{end\_pos}-1$, since `kv_len = end_pos` when caching).
- **The rule:** query $p$ may attend to key $j$ iff $p \ge j$ — additive `0` (allowed) or `-inf` (blocked), shaped `(1, 1, S_q, S_kv)` and broadcast to `(bsz, H, S_q, S_kv)` by `mask.expand(...)` inside the MLA forward.
- **The cache:** identical `(seqlen, kv_len, start_pos, device)` tuples reuse one tensor — during training the same geometry repeats every step, so the mask is built once per unique shape instead of once per step.

The three geometries it covers:

| Scenario | `seqlen` | `start_pos` | `kv_len` | Mask |
|---|---|---|---|---|
| Training / prefill | 2048 | 0 | 2048 | standard lower-triangular |
| Cached mid-sequence prefill | > 1 | > 0 | `end_pos` | causal by **global** position |
| Decode | 1 | $t$ | — | `None` (attend to all cached keys) |

The mid-sequence row is the subtle one: without the `+ start_pos` offset, a chunk being prefilled into the cache could attend to its own future (leakage) or be blocked from the past (broken context). The `seqlen == 1` decode path skips the mask entirely because a single query may attend to every cached key by construction.

---

## 8. The residual stream & information flow

Think of the transformer as a **residual stream** $h$ that each layer *reads from and writes to*. Embeddings write the token into the stream; each block adds an attention correction (mix information across positions) and an FFN correction (transform per-position); the final norm + LM head read the stream out as logits.

This framing is more than a metaphor — it has two operational consequences used later:

1. **The stream is the shared working memory.** Attention is the only component that moves information *between positions*; the FFN is the only component that mixes *within a position* at scale. Every token's "understanding" of the sequence is the superposition of corrections written into its stream slot by both.
2. **Sub-modules read, not overwrite.** Because each block *adds* its correction, the stream's content at depth $\ell$ is the sum of all earlier contributions. This is why ablating or compressing an intermediate component (MLA's KV, MTP's hidden) is safe — you are trimming a correction, not destroying the signal.

This framing matters for two later chapters:
- **MLA** compresses the part of the stream that attention *reads* (keys/values) — the *residual* stream itself is untouched.
- **MTP** feeds the *pre-norm* trunk hidden $h$ (not the logits) to the MTP block, which has its own norm — so MTP predicts from the model's internal representation, not the output. See [05 Multi Token Prediction](../concepts/moe-mtp.md).

`models/transformer.py:Transformer.forward_with_hidden` returns both `(logits, h)` precisely so MTP can use $h$.

### 8.3 Why gradients survive depth — the residual identity

The depth-robustness argument from §3.1, made precise. Stacking $L$ residual blocks gives a composed map $h_L = h_0 + \sum_\ell f_\ell(h_{\ell-1})$, and by the chain rule the gradient of the loss at the input is

$$\frac{\partial \mathcal{L}}{\partial h_0} = \frac{\partial \mathcal{L}}{\partial h_L} \prod_{\ell=1}^{L} \left( I + \frac{\partial f_\ell}{\partial h_{\ell-1}} \right)$$

If each block were a bare function $f_\ell$ (no skip), the gradient would be the product of $L$ Jacobians — exponentially vanishing or exploding with depth. With the identity term, expanding the product gives $I + \sum_\ell J_\ell + \sum_{\ell<\ell'} J_\ell J_{\ell'} + \cdots$: the **identity path carries the gradient through all $L$ layers unattenuated**, and the Jacobian terms are corrections on top. Pre-norm strengthens this: because each $f_\ell$ sees a normalized input, its Jacobian is bounded (the norm keeps the input in a fixed-scale region where the layer's Lipschitz constant is controlled), so the corrections stay small and the identity term dominates.

The whole loop is `models/transformer.py:Transformer._run_layers`, and the bookends are visible in `models/transformer.py:Transformer.forward`:

```python
h = self.embed(tokens)
...
h = self._run_layers(h, start_pos, mask, use_cache)
return self.head(self.norm(h))
```

Embedding writes into the stream; the final norm + head reads it out. Everything in between is correction.

---

## 9. The training loss — CE plus the MTP auxiliary

The full training objective combines the main next-token loss with the MTP auxiliary losses (see [05 Multi Token Prediction](../concepts/moe-mtp.md)):

$$\mathcal{L}(\theta) = \mathcal{L}_{\text{CE}} + \lambda \sum_{d=1}^{D} \mathcal{L}_{\text{MTP}}^{(d)}$$

with depth $D = 1$ and weight $\lambda = 0.3$ at the canonical config. The MTP loss is a cross-entropy on the depth-$d$ prediction target; the total is averaged across depths.

**Why it helps.** The main head supervises each position with one target ($x_{t+1}$). The MTP head additionally supervises the *same trunk* with $x_{t+2}$ from $(h_t, e_{t+1})$. So:

1. **Denser gradient per token** — every token contributes to both a $+1$ and a $+2$ prediction, so the trunk receives two supervision signals per position instead of one, without needing more data.
2. **Shared head, shared trunk** — MTP shares the embedding and the output head with the main model, so the auxiliary loss's gradients flow through the *same* `head.weight` and *same* trunk, improving the main head too (see [05 Multi Token Prediction](../concepts/moe-mtp.md) §Gradient flow).
3. **A free draft head** — the trained MTP module becomes a speculative-decoding draft model at inference, with no separate training run.

**Why $\lambda = 0.3$?** Large enough to shape the trunk representation, small enough that the primary CE objective dominates. DeepSeek-V3 uses $\lambda \in [0.1, 0.3]$.

### 9.1 The MTP loss in code

`models/mtp.py:MultiTokenPrediction.compute_loss` returns `(total_loss, main_loss, mtp_loss)`; only `total_loss` is backpropped, the components are detached for logging (avoiding per-step GPU syncs):

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

Note the `usable = seq_len - d - 2` length alignment: at depth $d$ the block predicts token $t + d + 1$ from hidden $h_t$ and embedding $e_{t+1}$, so the last $d+1$ positions have no target and are sliced off inside `models/mtp.py:MultiTokenPrediction.forward`. The depth-0 MTP loss therefore covers $S-2$ positions, one fewer than the main loss — a deliberate, load-bearing slice that [05 Multi Token Prediction](../concepts/moe-mtp.md) derives in full.

In `training/pretrain.py:Pretrainer.train_step`, the MTP branch runs the wrapped model and divides the total by the accumulation factor exactly like the main loss:

```python
main_logits, mtp_pairs = self.model(tokens)
total_loss, main_loss, mtp_loss = self.mtp_wrapper.compute_loss(main_logits, targets, mtp_pairs)
...
loss = total_loss / self.config.gradient_accumulation_steps
```

---

## 10. Optimization — AdamW, warmup-cosine, gradient clipping

### 10.1 Adam — the adaptive per-parameter step

Adam (Kingma & Ba, 2015) maintains per-parameter running estimates of the gradient's first moment ($m$, the mean) and second moment ($v$, the mean square):

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

Because both start at zero, early estimates are biased toward 0. The **bias correction** divides by the sum of the exponential weights:

$$\hat{m}_t = \frac{m_t}{1-\beta_1^{\,t}}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^{\,t}}$$

Then the update is a normalized step:

$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Each parameter moves by $\pm \eta$ in direction $m_t$, **rescaled by the inverse RMS of its own recent gradients**. This is the crucial property: Adam gives each parameter a step of roughly constant magnitude regardless of its gradient scale — it is *scale-free per parameter*, which is what makes it robust to the wildly differing gradient scales across an LLM (embedding rows vs norm gains vs expert weights).

### 10.2 AdamW — decoupled weight decay

**Weight decay** regularizes by shrinking weights toward zero. In L2 regularization, $\lambda\theta$ is added *into* the gradient and thus passes through the moment estimates; in **AdamW** (Loshchilov & Hutter, 2019) it is decoupled and applied directly to the weights:

$$\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda\, \theta_t \right)$$

This matters because Adam's per-parameter rescaling *fights* L2: a weight with large moments gets a large L2 gradient but a small normalized update, so L2's effect is distorted by the adaptive step. Decoupling restores weight decay as a clean, predictable force. **This repo:** $\beta_1=0.9$, $\beta_2=0.95$, weight decay $\lambda = 0.1$ on **2-D params only** (`p.dim() >= 2`).

**Why exclude 1-D params from decay?** Norm gains ($\gamma$), biases, and embeddings are low-dimensional; decaying them destabilizes training or double-penalizes the embedding. This is the standard LLM recipe (GPT-3, LLaMA). Note the MoE gate *bias* is a buffer, not a parameter, so it never sees weight decay at all (see [04 DeepSeekMoE](../concepts/moe-mtp.md)).

**Why $\beta_2 = 0.95$ and not the default $0.999$?** The second moment's effective window is $1/(1-\beta_2)$ steps: 20 for 0.95 vs 1000 for 0.999. LLM gradients' scale shifts rapidly during training (warmup, schedule, expert routing), so a very long window makes $v_t$ stale and the normalized step sluggish. A shorter window adapts faster.

**Fused AdamW:** the repo uses `fused=True` when CUDA is available — one fused kernel for the whole update with FP32 master weights held internally.

### 10.3 Warmup-cosine LR schedule

$$\text{phase 1 (warmup):}\;\; \eta_t = \eta \cdot \frac{t}{t_{\text{warm}}}, \qquad \text{phase 2 (cosine):}\;\; \eta_t = \eta_{\min} + (\eta - \eta_{\min})\cdot \tfrac{1}{2}\left(1 + \cos\!\left(\pi \cdot \frac{t - t_{\text{warm}}}{T - t_{\text{warm}}}\right)\right)$$

with $\eta_{\min} = 0.05\,\eta$, `warmup_steps = 2000`, `total_steps = 512000` (`training/pretrain.py:make_warmup_cosine_lambda`).

- **Why warmup?** Adam's bias correction is imperfect in the first few steps and the adaptive step can be huge when $v_t$ is tiny (early in training, before gradients accumulate statistics). Warmup lets the moments stabilize and avoids early divergence — the LR ramp is a *guard rail*, not a performance tweak.
- **Why cosine?** Cosine decay spends meaningful time at intermediate LR values before a long, gentle tail — empirically better than linear or step decay for LLM pretraining. The **floor at 5%** (not zero) keeps the model learning through the tail; a zero floor wastes the final steps.

### 10.4 Gradient clipping

Global-norm clipping: if $\|g\|_2 > c$, rescale the whole gradient $g \leftarrow g \cdot \frac{c}{\|g\|_2}$. With `grad_clip = 1.0`, the update norm never exceeds 1.0 — a safety rail against rare huge batches (or MoE routing spikes) that would otherwise blow up the step. Clipping preserves the *direction* of the gradient, only capping its magnitude.

### 10.5 The precision ladder

The optimizer's FP32 master copy of weights is the source of truth; forward/backward compute in BF16 (see §14). The ladder "FP32 master → BF16 compute → (paper) FP8 compute" puts precision where gradients accumulate and savings where bytes move.

### 10.6 The optimizer in code

The decay/no-decay split and the optimizer construction happen in `training/pretrain.py:Pretrainer.__init__`:

```python
decay_params = [p for p in all_params if p.dim() >= 2]
no_decay_params = [p for p in all_params if p.dim() < 2]
self.optimizer = AdamW([
    {"params": decay_params, "weight_decay": config.weight_decay},
    {"params": no_decay_params, "weight_decay": 0.0},
], lr=config.lr, betas=(config.beta1, config.beta2), fused=torch.cuda.is_available())
```

Two details worth noting. First, `all_params` is *deduplicated by tensor id* before the split — this is what makes weight tying safe: `head.weight` and `embed.weight` are the same tensor, and without the dedup the tied weight would be registered twice in the optimizer and updated twice per step (see §16.2). Second, `fused=torch.cuda.is_available()` switches to the fused CUDA kernel on GPU; on a CPU laptop (where this repo's test suite runs) it silently falls back to the eager path — the 199-test suite (189 pass + 10 GPU-gated skips) exercises the eager path.

### 10.7 Gradient clipping in code

Clipping happens once per optimizer step, after the accumulated backward and before `optimizer.step()` (`training/pretrain.py:Pretrainer.train_step`):

```python
loss.backward()
if is_opt_step:
    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
    self.optimizer.step()
    self.scheduler.step()
    self.optimizer.zero_grad(set_to_none=True)
```

`nn.utils.clip_grad_norm_` computes the **global** L2 norm over all parameters of the model and rescales if it exceeds `max_grad_norm = 1.0`. Because it is applied to `self.model.parameters()` (the MTP wrapper when MTP is on), the clip covers the MTP heads and the main trunk together — one gradient, one norm, one rescale. Note the order: clip, step, scheduler step, then zero. Also note the bias-update timing in the same block: `_update_moe_bias()` runs every `bias_update_every` optimizer steps (see [04 DeepSeekMoE](../concepts/moe-mtp.md)).

---

## 11. KV caching — the inference bottleneck

### 11.1 The problem: $O(T^2)$ recomputation

During autoregressive generation, each new token attends to *all* past tokens. Recomputing all past keys/values every step is $O(T^2)$ total (each of $T$ generated tokens attends to up to $T$ past keys). The fix: **cache** $K$ and $V$ for past tokens and only compute the new token's contributions — amortized $O(T)$ per sequence.

### 11.2 The lifecycle

```
prefill: forward(prompt, start_pos=0, use_cache=True)
         → cache written for positions [0, prompt_len)
decode:  forward([new_token], start_pos=t, use_cache=True)
         → reads cache[0:t], writes cache[t:t+1], outputs 1 token
reset:   reset_cache()  → cache=None  (required between independent generations)
```

The cache grows one slot per generated token and is capped by `max_seq_len` (exceeding it is a hard error, not silent truncation — `models/mla.py:MultiHeadLatentAttention.forward` raises `RuntimeError`). The `start_pos` argument is the cache write offset; an off-by-one here is the classic cause of "gibberish after the first token".

### 11.3 The cost

The cache grows with context. For MHA it is $O(T \cdot L \cdot H \cdot d_h)$ (§5). At $B=8$, $T = 2048$, $L = 18$, $H = 12$, $d_h = 72$ that is ~1 GB just for keys and values (K+V = $2 \cdot 12 \cdot 72 = 1728$ floats/token/layer) — comparable to the model weights (~0.84 GB). This is why every efficient LLM family addresses the cache: GQA shares heads, SWA windows it, SSM eliminates it, and **MLA compresses it** into a low-rank latent (see [03 Multi Head Latent Attention](../concepts/attention-and-precision.md) §KV cache, and [13 Portfolio Comparison](../concepts/foundations.md) §2). The repo's MLA cache stores only `kv_cache (B, T, 192)` + `pe_cache (B, T, 24)` per layer — ~0.13 GB at the same config, a ~8× reduction. All of these numbers are budget estimates from the config, not measurements `[INFERENCE]`.

### 11.4 Memory bandwidth, not just capacity

The deeper reason the cache matters is **bandwidth**: every decode step must read the entire cache from HBM into SRAM to compute attention scores. At 128K context, that is a multi-GB read *per generated token*. MLA's reduction of *what is cached* therefore reduces the per-step memory traffic — the actual wall-clock bottleneck of long-context decoding. This is why MLA is a throughput win, not just a memory win.

### 11.5 Cache mechanics in code

The cache buffer is allocated lazily and grown by doubling (`models/mla.py:MultiHeadLatentAttention._ensure_cache`):

```python
def _ensure_cache(self, bsz: int, device: torch.device, dtype: torch.dtype) -> None:
    need_alloc = self.kv_cache is None or bsz > self._cache_batch or self.kv_cache.device != device or self.kv_cache.dtype != dtype
    if not need_alloc:
        return
    new_bsz = max(bsz, self._cache_batch * 2, 16)
    self.kv_cache = torch.zeros(new_bsz, self.max_seq_len, self.kv_lora_rank, device=device, dtype=dtype)
    self.pe_cache = torch.zeros(new_bsz, self.max_seq_len, self.qk_rope_head_dim, device=device, dtype=dtype)
    self._cache_batch = new_bsz
```

Three properties are load-bearing:

1. **Zero-initialized, never garbage.** The buffer is `torch.zeros`, so reading a never-written slot is a silent zero, not a crash — which is also why `start_pos` bookkeeping must be exact (a stale zero slot attended to as if it were a real key produces garbage logits, not an error).
2. **Batch size doubles.** `new_bsz = max(bsz, 2 * _cache_batch, 16)` — the allocation policy grows the batch dimension geometrically, so a batch-16 run costs one reallocation at most.
3. **Device and dtype are part of the key.** If a subsequent forward moves to another device or dtype, the cache is reallocated rather than silently misread.

Writes are detached snapshots of the *normalized* latent and the *rotated* rope key (`models/mla.py:MultiHeadLatentAttention.forward`):

```python
if use_cache:
    self.kv_cache[:bsz, start_pos:end_pos] = kv_normed.detach()
    self.pe_cache[:bsz, start_pos:end_pos] = k_pe.detach()
    ctx_kv = self.kv_cache[:bsz, :end_pos]
    ctx_pe = self.pe_cache[:bsz, :end_pos]
```

The `.detach()` is essential: during training `use_cache=False`, so this branch is inference-only — but even so, cache writes must never carry autograd graph references, or a decode-step forward would start building a graph through every past token. `models/mla.py:MultiHeadLatentAttention.reset_cache` (and `models/transformer.py:Transformer.reset_cache`, which fans out to every layer) drops the buffers entirely, so the next forward reallocates fresh — this is what `generate` calls before each generation (see `models/transformer.py:Transformer.generate`).

---

## 12. The Chinchilla token budget & compute accounting

### 12.1 The scaling law

**Chinchilla** (Hoffmann et al., 2022) fits the empirical loss as a sum of three power laws — one in parameters $N$, one in data $D$, plus a floor:

$$L(N, D) = a\, N^{-\alpha} + b\, D^{-\beta} + c$$

Minimizing this subject to a *fixed compute budget* $C \approx 6 N D$ (roughly, FLOPs ≈ 6 × params × tokens) yields the compute-optimal frontier: **$N$ and $D$ should grow at equal rates**. The convenient rule of thumb falls out:

$$N_{\text{tokens}} \approx 20 \cdot N_{\text{params}}$$

### 12.2 What it means for this repo

Train a 411.6 M model on ~8.4 B tokens — not the "train forever" heuristic that wastes compute on too-small models (undertraining leaves the model data-hungry) and not too many epochs on too few tokens (overtraining burns FLOPs a larger model would use better).

The canonical config targets **8.4 B tokens over 512,000 micro-steps** (micro-batch 8 × sequence 2048 = 16,384 tokens per micro-step, gradient accumulation 4 → effective batch 32 → 65,536 tokens per optimizer step). The shared universal pipeline in `LLM/shared_data/` produces exactly this corpus, sharded as `uint32` memmap (see [09 Data Pipeline](../concepts/data-pipeline.md)).

**Step accounting (derived from `training/pretrain.py:Pretrainer.train`).** The training loop's `global_step` counter increments once per micro-batch, so `total_steps: 512000` bounds **micro-steps**: 512,000 × 16,384 = 8.39 B token exposures — one pass over the ~8.4 B Chinchilla-optimal corpus — and 512,000 / 4 = **128,000 optimizer steps**. Two consequences worth knowing:

1. **One epoch, not four.** The 8.4 B-token budget and the 512 K-micro-step budget are the *same* number; the corpus is sized so the run is exactly one pass. (A naive "512,000 steps × 65,536 tokens" product conflates micro-step count with optimizer-step token count and arrives at a spurious 4 epochs.)
2. **The cosine schedule's denominator is the micro-step budget.** `make_warmup_cosine_lambda(..., total_steps=512000)` advances once per *optimizer* step, so after 128,000 scheduler steps the schedule has traversed only ~25% of its cosine arc — the LR decays from peak toward ~0.86 × peak and never reaches the 5% floor in a full run `[INFERENCE from config+code; flagged for [08 Training Pipeline](../training.md)]`. This is harmless for a first run (a gentle LR is conservative) but should be reconciled before treating the schedule as fully exercised.

### 12.3 FLOP accounting and MFU

**The "6N" rule:** a transformer forward+backward costs $\approx 6 \cdot N_{\text{non-embedding}}$ FLOPs per token (2 per parameter for the forward matmuls, 4 for the backward). So the training budget is:

$$C \approx 6 \cdot N_{\text{non-embed}} \cdot D_{\text{tokens}}$$

At the canonical 411.6 M model ($N_{\text{non-embed}} = 411\,632\,256 - 76\,813\,824 \approx 334.8$ M; the embedding is memory-bound, not FLOP-bound):

| Quantity | Value |
|---|---|
| FLOPs/token | $\approx 2.0$ GFLOPs |
| Tokens/opt step | 65,536 |
| FLOPs/opt step | $\approx 1.3 \times 10^{14}$ |
| A100 BF16 peak | 312 TFLOPS |

**Model FLOPs Utilisation (MFU)** is the fraction of peak hardware throughput actually achieved:

$$\mathrm{MFU} = \frac{\text{achieved FLOPs/sec}}{\text{peak BF16 FLOPs/sec}}$$

At 35–40% MFU (the planning target, measured by `scripts/step_time_a100.py`), each optimizer step takes ~1.1–1.2 s, so the 128,000-step run is **~40 wall-clock hours** `[estimate — no GPU run exists]`. The config header quotes a 13–15 h target; that figure does not reproduce from the FLOP budget at 35–40% MFU (it would require ≈1.1× A100 peak), so treat every wall-clock number here as a budget estimate pending a real run. MFU < 25% usually means memory-bound (MoE dispatch, activation recomputation from grad checkpointing) or a batch too small to saturate the SMs.

### 12.4 Deriving the 20× rule

Where does "20 tokens per parameter" come from? Minimize the empirical loss under a compute budget. The fitted scaling law is $L(N, D) = a N^{-\alpha} + b D^{-\beta} + c$ with $\alpha \approx 0.34$, $\beta \approx 0.28$ (Chinchilla's fits; the constants vary by corpus). Holding $C = 6 N D$ fixed, use a Lagrange multiplier:

$$\mathcal{L}(N, D, \lambda) = a N^{-\alpha} + b D^{-\beta} + c + \lambda\,(N D - C/6)$$

Setting the $N$ and $D$ derivatives to zero and eliminating $\lambda$:

$$\frac{\partial}{\partial N}: -\alpha a N^{-\alpha-1} + \lambda D = 0, \qquad \frac{\partial}{\partial D}: -\beta b D^{-\beta-1} + \lambda N = 0$$

$$\frac{\alpha a\, N^{-\alpha}}{\beta b\, D^{-\beta}} = 1 \quad\Rightarrow\quad \beta \ln D - \alpha \ln N = \ln\!\frac{\beta b}{\alpha a}$$

The right-hand side is a constant, so along the optimal frontier $\ln D$ and $\ln N$ are *linearly related*: doubling the model must be accompanied by a fixed multiplicative increase in tokens — the "grow them at equal rates" result. The specific constant ($D \approx 20 N$) comes from plugging the fitted $\alpha, \beta, a, b$ into that equation. Two practical readings:

- **Halving loss requires 4–10× more compute** — the exponents $\alpha, \beta < 1$ mean returns diminish hard; this is why "train a bigger model longer" is rarely the efficient move for a fixed budget.
- **Small models are data-hungry.** 411.6 M × 20 = 8.23 B tokens is a lot of text for a model that fits in ~1 GB — but the alternative (training it on 2 B tokens) leaves the power-law $b D^{-\beta}$ term dominating the loss.

---

## 13. μP — maximal-update parameterization

### 13.1 The transfer problem

When you scale model width, naively keeping the same learning rate causes instability or under-training. The best LR depends on width, so you must re-tune it per size. **μP** (Yang et al., 2021) prescribes width-dependent *initialization* and *per-tensor LR* such that the update magnitude per parameter stays width-invariant — making a LR tuned on a small reference model valid on a large target.

### 13.2 Why the naive LR breaks

At initialization, a hidden preactivation $y = W x$ with $x \in \mathbb{R}^{n}$ has magnitude $\sqrt{n}$ if $W$'s entries are $\mathcal{O}(1)$ (the sum of $n$ terms). To keep activations width-independent, hidden weights are typically scaled as $\mathcal{O}(1/\sqrt{n})$ — then $y$ is $\mathcal{O}(1)$ but each *update* $\Delta W \sim \eta \cdot \nabla L$ perturbs $y$ by $\sqrt{n} \cdot \Delta W \sim \sqrt{n} \eta$, which *grows with width*. μP compensates: hidden-weight LRs scale as $1/n$, embedding/output LRs as $1$ (or as their own width rules), so the *function-level* update stays $\mathcal{O}(1)$ at every width. The result is **hyperparameter transfer**: tune on a cheap small model, deploy on the large one.

### 13.3 This repo's pragmatic shortcut

The repo does not implement full μP (per-tensor LR, width-dependent init everywhere). It uses the count-based **transfer rule** (`training/pretrain.py`):

$$\text{lr}_{\text{target}} = \text{lr}_{\text{ref}} \cdot \left(\frac{N_{\text{ref}}}{N_{\text{target}}}\right)^{1/2}$$

with reference $\text{lr}_{\text{ref}} = 6.0\text{e-4}$ at $N_{\text{ref}} = 757\,226\,496$ params. The canonical config sets `mup_lr: true`, so the configured `lr: 8.0e-4` is overridden by the μP-scaled value ($\approx 8.07\text{e-4}$ with MTP, $8.14\text{e-4}$ for the 411.6M base).

`★ Insight ─────────────────────────────────────` The intuition: wider models have more parameters contributing to each update; if every param updates at the same LR, the aggregate update to the *function* grows with width and destabilizes training. The $\sqrt{N}$ scaling keeps the effective update magnitude stable as width grows — it is the continuous analog of averaging more independent estimates. This is why μP lets you tune hyperparameters on a cheap small model and lift them to the expensive large one, instead of retuning on the target. See [08 Training Pipeline](../training.md) §μP Learning Rate Scaling. `─────────────────────────────────────────────────`

**Caveat:** the count $N$ is measured *after* the MTP wrap, so the MTP head's ~7M params inflate the denominator slightly. This is documented and locked by `test_mup_lr_scaling`.

### 13.4 The real numbers in code

The scaling is applied once at construction (`training/pretrain.py:Pretrainer.__init__`):

```python
if config.mup_lr:
    new_lr = config.mup_lr_reference * (config.mup_lr_reference_params / total) ** 0.5
    self._log(f"µP LR scaling: {config.lr:.2e} → {new_lr:.2e} (ref {config.mup_lr_reference:.2e} @ {config.mup_lr_reference_params:,} params)")
    config.lr = new_lr
```

where `total` is the deduplicated parameter count of the *training model* — `count_parameters` on the raw transformer for the base case (411,632,256), or on the MTP wrapper when `mtp_depth > 0` (418,713,984). Working the numbers:

$$\text{base:}\quad 6.0\text{e-4} \times \sqrt{\frac{757\,226\,496}{411\,632\,256}} = 6.0\text{e-4} \times 1.3563 \approx 8.14\text{e-4}$$

$$\text{with MTP:}\quad 6.0\text{e-4} \times \sqrt{\frac{757\,226\,496}{418\,713\,984}} = 6.0\text{e-4} \times 1.3448 \approx 8.07\text{e-4}$$

The 0.07e-4 gap between the two is the MTP head's 7,081,728 params (~7.1 M) inflating the denominator — small but real, which is why the order of operations matters: `total` must be the *final wrapped* count, and `count_parameters` must deduplicate the tied embedding (else the denominator double-counts 76.8 M and the LR comes out wrong). Both are enforced by `test_mup_lr_scaling`. Note that because `new_lr` overwrites `config.lr` in place, the *logged* and *scheduled* LR is always the μP value; the `8.0e-4` in the YAML is only the pre-scaling nominal.

---

## 14. Mixed precision — BF16 training

### 14.1 Why not full FP32

A 411.6M model in FP32 needs ~1.6 GB just for weights; AdamW's FP32 state (master weights + first + second moments, 12 bytes per parameter) adds ~4.9 GB. In BF16 the compute weights halve to ~0.8 GB, and A100 Tensor Cores run BF16 matmuls at 2× the FP32 rate. Mixed precision gets most of the speed and memory of FP16 with none of the numerical fragility. (All memory figures here are budget estimates from the parameter count, not measurements `[INFERENCE]`.)

### 14.2 BF16 vs FP16

| Format | Exponent | Mantissa | Range | Precision |
|---|---|---|---|---|
| FP32 | 8 | 23 | $\pm 3.4 \times 10^{38}$ | High |
| FP16 | 5 | 10 | $\pm 65504$ | Medium |
| BF16 | 8 | 7 | $\pm 3.4 \times 10^{38}$ | Lower |

BF16 keeps FP32's **8 exponent bits** — the same *range* as FP32 — and sacrifices only mantissa precision. FP16 has more mantissa but a tiny range, which is why FP16 training needs loss scaling (a global multiplier to keep small gradients representable). BF16's wide range means **no loss scaling is needed**: gradients and activations rarely under/overflow, and the FP32 master weights (held in AdamW state) absorb the mantissa loss. This is why the repo uses `autocast("cuda", dtype=torch.bfloat16)` and *no* `GradScaler`.

**Where precision is preserved:** the FP32 master copy of weights lives in AdamW's state; the BF16 weights used for compute are a cast. Softmax is up-cast to FP32 explicitly (`softmax(dtype=torch.float32)`) because `exp()` overflows easily in BF16. Tensor-core matmul accumulation happens in FP32 internally. This is the same principle FP8 would extend one notch lower (see [06 FP8 Mixed Precision](../concepts/attention-and-precision.md)).

### 14.3 TF32 matmuls

On Ampere+, `torch.backends.cuda.matmul.allow_tf32 = True` (set in `Pretrainer.__init__` on CUDA) lets FP32 matmuls run on Tensor Cores at TF32 precision (~10 mantissa bits) — roughly 8× the FP32 matmul throughput with negligible accuracy change at BF16 training scale. This, plus `torch.compile(mode="max-autotune")`, is the hardware-utilization stack.

### 14.4 FP32 accumulation — where the precision actually lives

The claim "BF16 forward" deserves precision: BF16 applies to *inputs and outputs* of matmuls; the *accumulation* inside a matmul happens in FP32. That distinction is the entire numerics story of mixed-precision training. Consider a dot product of length $n$ in BF16. Each product $a_i b_i$ rounds to ~8 mantissa bits, and if the running sum were kept in BF16, each addition would round again — a per-step relative error of $2^{-8}$, accumulating to $O(\sqrt{n}\, 2^{-8})$ over $n$ terms. Keeping the accumulator in FP32 ($2^{-24}$ per rounding) drops the accumulation error to $O(\sqrt{n}\, 2^{-24})$ — below the input quantization error, which is why it is "free" precision. PyTorch's `torch.bmm`/`torch.matmul`/`nn.Linear` on CUDA Tensor Cores accumulate in FP32 by construction, and the same is true inside the custom Triton kernels ([12 Triton Kernels](../concepts/kernels-and-ops.md)).

The other two places precision is deliberately preserved:

1. **Softmax runs in FP32.** The manual attention path in `models/mla.py:MultiHeadLatentAttention.forward` computes `scores.softmax(dim=-1, dtype=torch.float32)` — the exponentiation and normalization happen in FP32, then the result is cast back to the input dtype. A BF16 softmax over 2048 keys would overflow (`e^{80}` already exceeds BF16 range) and would also lose the small probability mass that matters for gradient flow.
2. **Normalization runs in FP32.** Under `autocast`, PyTorch keeps normalization-style ops (RMSNorm/LayerNorm) in FP32 even when the surrounding matmuls are BF16 — the RMS reduction is a sum of $d$ squares, the canonical precision-sensitive reduction.

This is also why there is **no GradScaler** anywhere in this repo: scaling exists to keep *small FP16 values representable*; BF16's exponent range already covers them, and the FP32 accumulator covers the sums. A `GradScaler` with BF16 would be dead weight at best and a silent LR change at worst.

### 14.5 Autocast in code

The dtype is fixed once (`self.amp_dtype = torch.bfloat16`) and every training forward runs inside an autocast context (`training/pretrain.py:Pretrainer._amp_context`):

```python
def _amp_context(self):
    return autocast(self.device.type, dtype=self.amp_dtype)
```

`autocast` is *policy-based*: it casts only the ops on its "lower-precision" list (matmuls, convolutions, embeddings' matmul-like paths) to BF16 and leaves the rest (norms, softmax, losses) in FP32 — which is exactly the division of labor §14.4 argues for. The forward/backward in `training/pretrain.py:Pretrainer.train_step` wraps the whole compute:

```python
with self._amp_context():
    if self.mtp_wrapper is not None:
        main_logits, mtp_pairs = self.model(tokens)
        total_loss, main_loss, mtp_loss = self.mtp_wrapper.compute_loss(main_logits, targets, mtp_pairs)
        ...
        loss = total_loss / self.config.gradient_accumulation_steps
    else:
        logits = self.model(tokens, start_pos=0, use_cache=False)
        main_loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        ...
        loss = main_loss / self.config.gradient_accumulation_steps
```

Two details in this snippet that are easy to miss:

- **`use_cache=False` in the training path** — the training forward never touches the KV cache; the cache is an inference-only structure (§11), and a cache write would break gradient flow through past positions.
- **`ignore_index=-100` in both losses** — the MTP path forwards the same sentinel through `compute_loss` (§9.1), so padded positions are excluded from the main and auxiliary losses identically.

The gradient side of the ladder: `loss.backward()` runs under the same autocast context (the context is still active), so gradients are computed against BF16 activations but accumulated into FP32 gradient buffers; `clip_grad_norm_` and AdamW then operate in FP32. The FP32 master copy in AdamW's state is what actually moves the weights — the BF16 cast happens fresh each forward.

### 14.6 The dtype boundary, op by op

"BF16 training" is shorthand; the precise picture under `autocast` is a *per-op* policy. For the ops this repo uses:

| Op | dtype under autocast | Why |
|---|---|---|
| `nn.Linear`, `torch.bmm`, `torch.matmul` | BF16 inputs, **FP32 accumulation** | Tensor-core path; accumulation error $O(\sqrt{n}\,2^{-24}) \ll$ input quantization |
| `nn.Embedding` gather | BF16 (cast of the FP32 master) | The weight is cast to BF16 for the gather |
| `nn.RMSNorm` | FP32 | Sum-of-squares reduction is precision-sensitive |
| `softmax` (incl. attention) | FP32 | `exp()` overflows BF16; small probability mass matters for gradients |
| `F.cross_entropy` | FP32 | Loss must be exact; it is a scalar anyway |
| `F.silu` (pointwise) | BF16 (input dtype) | No accumulation; quantization is benign |

The pattern: **any op that sums or exponentiates runs in FP32; any op that is a pointwise or tensor-core matmul runs in BF16.** This division is what makes the "no loss scaling" claim (§14.4) hold: the numerically fragile parts never see BF16. If you ever add a custom op to the training path, apply the same test — if it sums more than a handful of BF16 values or exponentiates, it belongs in FP32.

---

## 15. Gradient checkpointing

### 15.1 The problem

Backprop stores every layer's activations to compute gradients. For 18 layers × batch 8 × seq 2048 × dim 768, activation memory dominates VRAM — on the order of 10 GB with the PaLM 24× factor. Without checkpointing, the 411.6M config would not fit comfortably alongside optimizer state on the A100.

### 15.2 The mechanism

**Gradient checkpointing** does not store every activation — it stores only a checkpoint (the block input) at each layer and **recomputes the forward** during backward:

```
Forward:  compute layers 1..18; save only each layer's INPUT as a checkpoint
Backward: to get layer ℓ's activations, re-run layers' forwards from the nearest checkpoint
```

In this repo every layer is checkpointed (`Transformer(use_checkpoint=True)`), and the per-layer wrap lives in `models/transformer.py:Transformer._run_layers`:

```python
def _run_layers(self, h: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor], use_cache: bool) -> torch.Tensor:
    for layer in self.layers:
        if self.use_checkpoint and self.training:
            h = torch.utils.checkpoint.checkpoint(
                layer, h, start_pos, mask, use_cache, use_reentrant=False,
            )
        else:
            h = layer(h, start_pos, mask, use_cache)
    return h
```

Note the `self.training` guard: the same `_run_layers` serves inference (`generate`), where checkpointing would be pure overhead. The `use_reentrant=False` flag selects the modern, non-reentrant checkpoint API — required for `torch.compile` compatibility and the default in PyTorch 2.x.

### 15.3 The trade-off

$$3\times \text{activation memory savings} \iff \approx 33\%\ \text{extra backward FLOPs}$$

The recomputation is a *forward* pass — roughly half the cost of the forward+backward pair — so total FLOPs rise ~33% while peak memory drops ~3×. For a memory-bound single-GPU run this is the right trade: the bottleneck is VRAM, not FLOPs. `use_reentrant=False` is the PyTorch 2.x default and keeps `torch.compile` compatibility.

---

## 16. Weight tying

### 16.1 The dual operation

The embedding maps token → vector ($E \in \mathbb{R}^{V \times d}$); the LM head maps vector → logits. These are **dual operations** — the transpose of each other's job. **Weight tying** (Inan et al., 2016; Press & Wolf, 2017) makes the output projection use the embedding matrix:

$$\ell_t = E^\top \cdot \text{RMSNorm}(h_t)$$

### 16.2 Why it works and what it saves

- **Saves ~77M params** at the canonical scale: without tying, embedding + head = $2 \times 100018 \times 768 = 153.6$M; with tying, $100018 \times 768 = 76.8$M counted once. That is ~18% of the 411.6M model.
- **Acts as a regularizer** — the same vectors must serve as both input features and output logits, coupling the two roles.
- In code: `models/transformer.py:Transformer.__init__` sets `self.head.weight = self.embed.weight` (same storage), `models/transformer.py:count_parameters` deduplicates by tensor `id`, and `CheckpointManager` dedups the shared tensor on save (safetensors rejects duplicate storage).

The dedup is visible in `models/transformer.py:count_parameters`:

```python
def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """(total, trainable) — deduplicated by tensor id (shared weights counted once)."""
    seen = set()
    total = 0
    trainable = 0
    for p in model.parameters():
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return total, trainable
```

Because `head.weight` *is* `embed.weight`, iterating `model.parameters()` yields the same tensor object twice; `id(p)` dedup makes `total` come out to 411,632,256 instead of 488,446,080 — the latter being what a naive sum would report. The same dedup is used in the optimizer construction (§10.6), the per-component breakdown in `Pretrainer.__init__`, and checkpoint saving (the duplicate `head.weight` entry is dropped from the state dict).

**Load-bearing:** removing tying (`weight_tying: false`) breaks generation quality and changes the parameter count the μP denominator uses — both are documented invariants (see [11 Operations and Testing](../concepts/kernels-and-ops.md)).

### 16.3 Embeddings in detail

The embedding is a plain `nn.Embedding(vocab_size, dim)` — 100,018 rows × 768 columns — initialized with the small-std scheme of §19 (`nn.init.normal_(self.embed.weight, std=0.006)`). Three operational facts:

1. **Row count is load-bearing.** `vocab_size: 100018` must equal `len(tokenizer)` exactly; the tokenizer's `byte_fallback` guarantees any byte sequence is tokenizable, so *every* row is reachable in principle (§17). A config with 100,000 or 102,400 rows silently misaligns or crashes the embedding.
2. **The dtype cast happens at the boundary.** `models/transformer.py:Transformer.forward` casts non-Long inputs (`tokens.to(torch.long)`) because `nn.Embedding` requires Long indices — the dataset stores `uint32` tokens (4 bytes vs 8) for memory efficiency, and the cast happens once per forward.
3. **Tying makes the embedding the head.** With `self.head.weight = self.embed.weight`, the *same* rows serve as input features (rows of $E$, gathered by token id) and output logits (columns of $E^\top$). The gradient for a token row accumulates from both roles: the embedding gather (for tokens seen as input) and the head matmul (for tokens seen as targets). This is the "dual operation" of §16.1 made concrete — and why the embedding is the single largest parameter tensor (76.8 M of the 411.6 M) whose gradient statistics AdamW must handle (see [08 Training Pipeline](../training.md) for the per-component breakdown log).

---

## 17. Tokenization — BPE and the DeepSeek tokenizer

### 17.1 The problem

Neural networks operate on **discrete symbols**. Raw UTF-8 bytes are too fine-grained (256 symbols → very long sequences). Whole words are too coarse (unbounded vocabulary, no way to handle unseen words). **Subword tokenization** splits the compromise: frequent words stay whole; rare words decompose into pieces.

### 17.2 Byte-Pair Encoding

BPE (Sennrich et al., 2016) is an iterative merge algorithm:

1. Start with a byte- or character-level vocabulary.
2. Count all adjacent symbol pairs in the training corpus.
3. Merge the most frequent pair into a new symbol.
4. Repeat until the vocabulary reaches the target size $V$.

The result is a **tokenizer with a fixed vocabulary** where the merge operations encode the corpus's most frequent subword patterns. The **byte_fallback** extension (used by the DeepSeek tokenizer) adds: any byte 0–255 not covered by a merge is itself a valid token — so *any* input can be tokenized, including arbitrary or non-UTF-8 bytes. This is the unusual property that makes the embedding's row count load-bearing.

### 17.3 The DeepSeek tokenizer (this repo)

| Property | Value | Implication |
|---|---|---|
| Name | `deepseek-ai/deepseek-coder-v2-lite` | Public HF tokenizer |
| `vocab_size` | **100,018** | Embedding rows must match exactly |
| `eos_token_id` | 100,017 | Appended at document boundaries |
| `pad_token_id` | 100,016 | Padding (rare in packed pretrain) |
| `byte_fallback` | yes | Tokens 0–255 may be raw bytes |

**The footgun:** `vocab_size` is not a free knob. Using 100,000 or 102,400 silently misaligns or crashes the embedding (`nn.Embedding(vocab_size, dim)`). Always verify `len(tokenizer) == 100018` and that the model config matches (see [09 Data Pipeline](../concepts/data-pipeline.md) §Tokenizer Deep Dive).

### 17.4 The tokenizer in the data pipeline

The pipeline configuration (`data/data_config.yaml`) fixes the three special ids the whole stack agrees on:

```yaml
vocab_size:    100018
eos_token_id:  100017
pad_token_id:  100016
add_eos:       true
```

These are not arbitrary: `eos_token_id = vocab_size - 1` and `pad_token_id = vocab_size - 2` are the *last two rows* of the embedding table, and `add_eos: true` means every document gets the EOS token appended when it is packed into the token corpus (`data/prepare_data.py` routes the same constants into the shared pipeline). The EOS token therefore does double duty: it marks document boundaries in the packed stream (so the model learns "a document ends here") and it is the natural stopping condition for generation (`eos_token_id` in `models/transformer.py:Transformer.generate`).

Because documents are **packed** end-to-end (EOS-separated, no padding), a training batch is a continuous token window: there is no padding structure, which is why `ignore_index=-100` (§2.7) is a rare-path safety net rather than a daily feature. The byte_fallback guarantee closes the loop on tokenization: *every* byte sequence maps to valid token ids, so the uint32 shards can never contain an id that would index past the 100,018 rows — the embedding and the tokenizer are mutually consistent by construction.

---

## 18. The matrix calculus you need

You only need three rules to read every gradient argument in this repo. Let $y = Wx$ be a matmul and $\frac{\partial \mathcal{L}}{\partial y}$ the upstream gradient.

| Operation | Forward | Gradient w.r.t. input | Gradient w.r.t. weight |
|---|---|---|---|
| Matmul $y = Wx$ | $W x$ | $W^\top \frac{\partial \mathcal{L}}{\partial y}$ | $\frac{\partial \mathcal{L}}{\partial y} x^\top$ |
| Elementwise $y = x \odot z$ | $x \odot z$ | $\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \odot z$ | — |
| Linear with bias $y = Wx + b$ | $W x + b$ | $W^\top \frac{\partial \mathcal{L}}{\partial y}$ | $b$: sum of the gradient over the batch |
| RMSNorm | scale by RMS | (chain rule through the RMS denominator) | via $\gamma$ |
| Softmax $p = \text{softmax}(z)$ | $p_i = e^{z_i}/\sum_j e^{z_j}$ | $\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)$ | — |

Two things to notice:

1. **Matmul gradients are matmuls.** `∂L/∂W = (∂L/∂y)·xᵀ` is why backward costs the same FLOPs as forward — and why fused kernels ("GEMM backward") reuse the same tiling machinery.
2. **The SwiGLU gate is a product** (§4). The backward of `h = silu(a) ⊙ b` splits into `∂L/∂b = ∂L/∂h ⊙ silu(a)` and `∂L/∂a = ∂L/∂h ⊙ b ⊙ silu'(a)` — two channels, each getting the *other* channel's forward value as a coefficient. This "gating backward" is exactly where the gate's multiplicative structure shows up in code.

The added softmax row completes the picture: it is the Jacobian behind §2.4's "softmax minus one-hot" — chaining $(\partial \mathcal{L}/\partial p) \cdot (\partial p/\partial z)$ with $\mathcal{L} = -\log p_y$ collapses to $p - \text{onehot}(y)$.

Orthogonality matters for RoPE: a rotation is an orthogonal matrix ($R^\top R = I$), so RoPE is **norm-preserving** and its backward is numerically well-behaved.

---

## 19. Weight initialization

### 19.1 Why init matters at depth

At 18 layers, any systematic activation growth compounds. If each layer scales activations by a factor slightly above 1, the stream saturates by layer 18; slightly below 1, it vanishes. Initialization must keep per-layer scale $\approx 1$ so the depth product stays bounded.

### 19.2 This repo's choices

```python
nn.init.normal_(self.embed.weight, std=0.006)          # models/transformer.py:Transformer.__init__
nn.init.normal_(self.gate.weight, std=0.006)           # models/moe.py:AuxLossFreeGate.__init__
# all other Linear layers: PyTorch default init
```

- **Embedding and gate: `std = 0.006`**, far smaller than the GPT-2/LLaMA convention of `0.02`. With vocab 100K and dim 768, the embedding has 76.8M params; a small std keeps initial embeddings close together (similar tokens start similar) and, critically, keeps the *initial logits* from saturating softmax — a large embedding norm at init would push logits to $\pm$large values and the loss into the flat tail of CE (§2.4).
- **MoE gate `std = 0.006`** keeps the sigmoid scores near 0.5 at init (`sigmoid(0) = 0.5`), avoiding saturated routing early in training (see [04 DeepSeekMoE](../concepts/moe-mtp.md) §Numerical stability).
- **Other Linears use PyTorch defaults** (Kaiming/Uniform per weight shape). At this scale with RMSNorm resetting the stream's magnitude each layer (§3.3), the default init is sufficient; only the embedding and gate need the small-std treatment.

### 19.3 Why 0.006 — the arithmetic

The choice is forced by the *coupling* of the tied embedding: the same matrix is both input features and output logits (§16.3), so one std controls both the stream's entry scale and the initial softmax temperature.

**Logit scale at init.** After 18 layers of RMSNorm, the final hidden $h$ has RMS $\approx 1$, so $\|h\|_2 \approx \sqrt{768} \approx 27.7$. The logit for token $v$ is $\ell_v = \langle h, e_v\rangle$ where the embedding row $e_v \sim \mathcal{N}(0, \sigma^2 I)$. For fixed $h$:

$$\mathrm{Var}(\ell_v) = \sigma^2 \|h\|_2^2 \approx \sigma^2 \cdot 768$$

- With $\sigma = 0.006$: $\mathrm{std}(\ell) \approx 0.006 \cdot 27.7 \approx 0.166$ — logits are small, the softmax is diffuse, and the CE gradient $p - \text{onehot}$ is far from its $\pm 1$ bounds. The initial loss sits near $\log V \approx 11.5$ and every position provides a healthy gradient.
- With the LLaMA-style $\sigma = 0.02$: $\mathrm{std}(\ell) \approx 0.55$ — logits of $\pm 2$–3, the softmax starts concentrating, and a few positions begin saturating. Not fatal at depth 1, but it compounds: the embedding is also the *input*, so large rows inject a large stream norm at position 0 that the first norms must absorb.

**The gate.** The routed gate computes $\text{sigmoid}(\langle w_g, x\rangle)$ with $w_g \sim \mathcal{N}(0, 0.006^2 I)$ and normalized $x$: the pre-activation has std $\approx 0.006 \cdot 27.7 \approx 0.166$, so scores cluster around 0.5 — every expert starts with roughly uniform routing probability, and the bias-update mechanism ([04 DeepSeekMoE](../concepts/moe-mtp.md)) takes over from there. A larger std would start the model with near-deterministic routing and a load imbalance the bias updates would have to dig out of.

**What is *not* controlled:** the dense/MoE weight matrices use PyTorch's default Kaiming-uniform per shape. The reason this is safe is exactly §3.5: every sublayer input is RMSNorm'd to unit RMS, so the input scale is fixed regardless of how the previous layer was initialized. Init and normalization are a matched pair — change one and the other's guarantees shift.

---

## 20. Worked example — one forward pass at 411.6M scale

**Input:** batch $B = 2$, sequence $S = 4$, token IDs:

```
[[101, 2345, 678, 9012],
 [55,  1234, 5678, 90]]
```

| Step | Operation | Shape |
|---|---|---|
| Embedding | `embed(tokens)` | $(2, 4) \to (2, 4, 768)$ |
| Layer 0 (dense) | `RMSNorm → MLA → + ; RMSNorm → SwiGLU(768→1536→768) → +` | $(2, 4, 768)$ |
| Layer 1 (dense) | same structure | $(2, 4, 768)$ |
| Layers 2–17 (MoE) | `RMSNorm → MLA → + ; RMSNorm → DeepSeekMoE → +` (top-4 of 20 + 1 shared) | $(2, 4, 768)$ |
| Final norm + head | `RMSNorm → Linear(768 → 100018)` | $(2, 4, 100018)$ |

The MLA internals at one layer, with the canonical dims: `wq` projects $(2,4,768) \to (2,4,864)$ (12 heads × 72), split into `q_nope` (48) and `q_pe` (24); the rope half is rotated (§6.7) while the content half is absorbed into the latent space; `wkv_a` produces the 192-dim latent + 24-dim rope key; `kv_norm` normalizes the latent; and the attention output — 12 heads × 64 value dims — is projected back to 768 by `wo`. The shapes all derive from `models/mla.py:MultiHeadLatentAttention.forward`; [02 Model Architecture](../concepts/foundations.md) walks every tensor shape in the full 18-layer stack.

**Parameter touch count:** MoE layers store 21 experts each but execute 5 per token (4 routed + 1 shared) — so the *active* parameter count per token is ~185M of the ~411.6M deduped total (see [02 Model Architecture](../concepts/foundations.md) §Active vs Total Parameters).

**MTP path (training only):** `forward_with_hidden` returns the pre-norm hidden; `MTPModule` takes $(h_t, e_{t+1})$ and predicts token $t+2$ (see §9, [05 Multi Token Prediction](../concepts/moe-mtp.md)). With `mtp_depth: 1`, the wrapped model carries 418,713,984 total params — the +7.1M MTP delta.

---

## 21. The loss landscape & training dynamics

### 21.1 The geometry of the objective

The per-position loss $\mathcal{L}(z) = -z_y + \log\sum_j e^{z_j}$ is **convex in the logits**: its Hessian is $\nabla^2\mathcal{L} = \operatorname{diag}(p) - pp^\top$, the covariance matrix of a categorical distribution, which is positive semi-definite. So in logit space the objective is a bowl with a unique minimum at the one-hot vector $z_y \to \infty$, $z_{j\ne y} \to -\infty$. This convexity is why gradient descent on the *parameters* works at all: at every position, the gradient $p - \text{onehot}$ (§2.4) points in a direction that strictly decreases the local loss, and the model's job is to make the parameter-to-logit map line those directions up.

Two structural features of the landscape matter for training:

1. **Flat directions are built in.** Softmax is shift-invariant, so adding a constant to all logits of a position leaves the loss unchanged — the gradient is orthogonal to the all-ones direction (it sums to zero, §2.4). RMSNorm adds a second family: scaling an entire stream slot is undone by the next norm, so the loss is (near-)invariant along those directions too. Nothing in the loss breaks these equivalences, which is why init (§19) and weight decay (§10.2) exist: they are the optimizer's tie-breakers among equally good solutions.
2. **The gradient is bounded.** $\|p - \text{onehot}\|_2 \le \sqrt{2}$ at every position, and it *shrinks* as $p$ approaches the target: learning slows exactly as the model becomes confident — a "saturation plateau" that is inherent to CE, not a bug. This is why a loss that stalls at, say, 3.0 while perplexity keeps dropping (§2.6) is normal: the residual gradient is concentrated in the long tail of wrong predictions.

### 21.2 The three phases of a pretraining run

A healthy run has a recognizable shape, visible directly in the `loss`/`ppl`/`lr` columns of `utils/logging.py:TrainingLogger.log`:

- **Phase 1 — token statistics (warmup, ~first 2,000 steps).** Loss collapses from $\log V \approx 11.5$ toward the 5–6 range within the first few thousand steps: the model learns unigram and short-range statistics almost immediately. The LR ramp (§10.3) exists precisely to keep this fast descent from overshooting while Adam's moments are still cold.
- **Phase 2 — structure (the long middle).** Loss moves 4 → 3 as the model acquires subword, syntactic, and local semantic structure; MoE routing stabilizes as the bias updates (§10.7) balance expert loads. This is where the run spends most of its 128,000 optimizer steps, and where the per-step loss becomes noisy — gradient noise dominates over the mean descent.
- **Phase 3 — the power-law tail.** Loss improvements slow to a crawl (the $b D^{-\beta}$ term of §12.4); gains come from rare patterns and long-range dependencies. This is the regime where the LR cosine decay matters most: small LR + long tail = fine-grained fitting without divergence.

The phases are not phases of the *schedule* (warmup/cosine); they are phases of the *data-fitting process*, and they are why people read loss curves rather than final loss alone. A run whose Phase 1 is too fast (loss explodes mid-warmup) points at LR/init; a run whose Phase 2 plateaus early points at capacity or data.

### 21.3 Loss spikes and the NaN guard

Rare events — a pathological batch, a transient expert-load concentration, an unlucky large gradient — produce **loss spikes**: one step's loss jumps 2–3 nats before resuming its descent. The gradient clip (§10.4) absorbs the usual cases. The catastrophic case is NaN/Inf: once a NaN enters the loss, every subsequent gradient is corrupted and the model never recovers on its own. The repo runs a two-stage state machine (`training/pretrain.py:Pretrainer.train_step` and `Pretrainer.train`):

```python
if self.config.nan_guard and (torch.isnan(loss).any().item() or torch.isinf(loss).any().item()):
    self._log(f"[nan-guard] NaN/Inf at micro_step={micro_step}, opt_steps={self._opt_steps}. Skipping backward.")
    self.optimizer.zero_grad(set_to_none=True)
    return None
```

- **Stage 1 (per micro-step):** the loss is checked *before* `backward()`; on NaN/Inf the step is skipped, gradients are zeroed, and `train_step` returns `None` — the accumulation for that micro-step is simply lost.
- **Stage 2 (per streak):** the training loop counts consecutive skipped steps (`nan_guard_streak`); at `nan_guard_max_consecutive = 5` it restores the latest checkpoint (rolling back the corrupted weights) and continues. If there is no checkpoint, the run aborts with a clear `RuntimeError` rather than silently training on garbage.

This is the repo's answer to the landscape's worst feature: it treats NaN as a *recoverable* event (rare, transient) and only escalates to rollback when the corruption persists. For the full state machine, see [08 Training Pipeline](../training.md) §NaN Guard.

### 21.4 What the numbers mean in this repo

The logged loss components map directly onto the objectives of §9: `loss` (main CE), `mtp_loss` (auxiliary), and `balance_loss` (the MoE load-balance metric from `models/moe.py:DeepSeekMoE.get_load_balance_loss`), all detached from the graph and rounded once per log interval to avoid per-step GPU syncs. Reading them together tells you *where* a run is struggling:

| Symptom | Likely cause |
|---|---|
| `loss` high, `mtp_loss` fine | Main-head capacity / LR too low |
| `loss` fine, `mtp_loss` high | MTP weight too aggressive or trunk hidden under-specified |
| `balance_loss` creeping up | Expert load imbalance; bias updates too slow (`bias_update_speed`) |
| `loss` → NaN, no rollback | Uncheckpointed corruption or an unguarded op outside autocast |

There is no GPU run in this repo yet, so "mid-training 2.8 nats" remains a planning target, not an observation `[INFERENCE]` — the landscape story above is the *shape* to expect, derived from the objective and the config, not measured.

### 21.5 Gradient noise, batch size, and the effective batch

The loss curve is a noisy curve, and the noise is not a nuisance — it is information about the landscape. The gradient computed from a batch of $B$ sequences is a Monte Carlo estimate of the true gradient: writing $\sigma$ for the per-token gradient noise, the estimator's standard error is $\sigma/\sqrt{B_{\text{eff}}}$, where $B_{\text{eff}}$ is the number of *independent* tokens. The canonical config's effective batch — 32 sequences × 2048 tokens = 65,536 tokens per optimizer step — is a deliberate middle ground:

- **Too small** (e.g., micro-batch 8, no accumulation): the gradient estimate is dominated by noise; the loss curve jitters; AdamW's moments are chasing noise, and the effective step direction wanders.
- **Too large**: the optimizer steps on a nearly exact gradient, which sounds good but wastes compute — the descent direction changes slowly, and the schedule has fewer steps over which to anneal the LR.

Gradient accumulation (§2.9) is the repo's tool for tuning $B_{\text{eff}}$ without touching memory: 4 micro-steps of batch 8 accumulate into one optimizer step with the *mean* gradient, so the 65,536-token effective batch costs the memory of a batch-8 forward. This is also why `loss.backward()` sees `loss / gradient_accumulation_steps` — the division keeps the accumulated gradient a mean rather than a sum, so the clip norm (§10.7) and AdamW's moments see a batch-size-independent scale.

There is a second, subtler noise source in a MoE model: **routing noise**. Each token's expert assignment is a function of the current gate weights, so the gradient the FFN experts see is a *routed* gradient — tokens arrive at experts in clumps, and the per-expert gradient is noisier than the per-token average would suggest. The bias-update mechanism (§10.7, [04 DeepSeekMoE](../concepts/moe-mtp.md)) exists partly to keep the routing distribution (and hence the per-expert gradient noise) stable over time.

---

## 22. Practice problems (with answers)

1. **RoPE periods.** With `rope_theta = 10000` and `qk_rope_head_dim = 24`, the frequencies are $\theta_i = 10000^{-2i/24}$. Which dimension rotates fastest? *Answer: $i=0$ — $\theta_0 = 1$, period $2\pi$. The slowest is $i=11$, $\theta_{11} = 10000^{-22/24} \approx 2.15\times10^{-4}$, period $2\pi \cdot 10000^{22/24} \approx 2\pi \cdot 4642 \approx 29\,160$ positions.*

2. **SwiGLU params.** Derive the parameter count of a dense FFN with dim $d$ and intermediate $I$. *Answer: $3dI$ ($W_1, W_3: d \to I$; $W_2: I \to d$). At $d=768, I=1536$: $\approx 3.5$M.*

3. **MoE FLOPs.** Compare one dense layer ($I=1536$) vs one MoE layer (top-4 of 20 routed + 1 shared, $I_{moe}=384$). *Answer: dense $\approx 6 \cdot 768 \cdot 1536 \approx 7.1$ MFLOPs/token; MoE $\approx 5 \cdot 6 \cdot 768 \cdot 384 \approx 8.8$ MFLOPs/token. MoE is slightly more compute but stores ~5× the FFN capacity.*

4. **μP scaling.** If the parameter count quadruples, by what factor does the μP LR change? *Answer: $\sqrt{N_{ref}/N_{target}} = \sqrt{1/4} = 1/2$ — the LR halves.*

5. **KV bytes.** Compute MLA cache bytes/token/layer vs full MHA at $R=192$, $d_{\text{rope}}=24$, $d_{\text{head}}=64$, $H=12$, BF16. *Answer: MLA = $(192+24) \times 2 = 432$ bytes/token/layer; MHA (K+V) = $2 \times 12 \times 64 \times 2 = 3072$ bytes; ratio $\approx 7.1\times$.*

6. **Chinchilla tokens.** How many tokens for a 1B-param model? *Answer: $\approx 20$B.*

7. **Softmax gradient.** If the model assigns $p = [0.7, 0.2, 0.1]$ and the true token is class 0, what is $\partial \mathcal{L}/\partial z$? *Answer: $p - \text{onehot} = [0.7-1, 0.2, 0.1] = [-0.3, 0.2, 0.1]$.* Notice it sums to zero — softmax gradients are zero-mean.

8. **Activation memory.** Why does gradient checkpointing roughly triple memory savings while adding ~33% FLOPs? *Answer: it avoids storing all $O(L \cdot S \cdot d)$ activations, keeping only $O(L)$ checkpoints; backward re-runs the forward, adding one forward's FLOPs to the backward's ~2 forward-equivalents.*

9. **Non-embedding FLOPs.** What is $N_{\text{non-embed}}$ for the canonical config? *Answer: $411\,632\,256 - 76\,813\,824 = 334\,818\,432 \approx 334.8$M — the embedding (76.8M, tied with the head) is memory-bound, not FLOP-bound.*

10. **Optimizer steps.** How many optimizer steps does a full run take? *Answer: `total_steps: 512000` bounds micro-steps; with `gradient_accumulation_steps: 4` that is 128,000 optimizer steps (512,000 × 16,384 = 8.39B tokens ≈ one pass over the ~8.4B corpus).*

---

## 23. Core notation & glossary

| Symbol | Meaning (canonical value) |
|---|---|
| $V$ | Vocabulary size (100,018) |
| $d$, `dim` | Model hidden dimension (768) |
| $L$, `n_layers` | Layer count (18) |
| $H$, `n_heads` | Attention heads (12) |
| $d_h$ | Per-head dimension (value 64) |
| $T$, $S$ | Sequence length (2,048) |
| $B$ | Batch size (micro-batch 8) |
| $I$ / `inter_dim` | Dense FFN width (1,536) |
| `moe_inter_dim` | Expert FFN width (384) |
| $d_c$ / `kv_lora_rank` | MLA KV latent dim (192) |
| $d_R$ / `qk_rope_head_dim` | Decoupled RoPE dim (24) |
| $x_{<t}$ | Tokens strictly before $t$ |
| $h_t$ | Residual-stream hidden at position $t$ |
| $\eta$ | Learning rate |
| $\lambda$ | Weight decay (0.1) or MTP weight (0.3), from context |
| `rope_theta` | RoPE base frequency (10,000) |
| MFU | Achieved FLOPs/s ÷ peak FLOPs/s |
| PPL | $\exp(\mathcal{L}_{\text{CE}})$ — effective branching factor |

---

## 24. Load-bearing invariants

| Invariant | Why it matters | Enforced by |
|---|---|---|
| Causal mask on all attention paths | Future-token leakage makes the loss trivial | `models/transformer.py:Transformer._build_causal_mask`, SDPA `attn_mask` |
| `use_cache=False` in training | A cache write breaks gradient flow through past positions | `training/pretrain.py:Pretrainer.train_step`, `forward_with_hidden` default |
| RMSNorm with `eps=1e-6` everywhere | Consistency of scale across the 60 norms | `models/*.py` |
| $\mu$P LR after MTP param count | Wrong count ⇒ wrong LR | `test_mup_lr_scaling` |
| Weight-tying dedup in param count | Double-counting inflates the μP denominator | `models/transformer.py:count_parameters` |
| BF16 forward + FP32 optimizer master | Stable low-precision training, no GradScaler | `autocast`, AdamW `fused` |
| MoE gate bias is a buffer, not a Parameter | Aux-loss-free routing must not get weight decay or autograd | `test_bias_not_in_parameters` |
| `vocab_size == len(tokenizer) == 100018` | Embedding row count is load-bearing (byte_fallback) | CI import check, data validation |
| `max_seq_len` caps the KV cache | Exceeding it is a hard error, not silent truncation | `models/mla.py:MultiHeadLatentAttention.forward` `RuntimeError` |

---

## 25. Check your understanding

**Q1.** Derive the RMSNorm gradient. For $y = x / r$ with $r = \sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}$ and upstream gradient $g = \partial \mathcal{L}/\partial y$, show that $\partial \mathcal{L}/\partial x_j = \frac{1}{r}\left(g_j - x_j \frac{\langle g, x\rangle}{d\, r^2}\right)$, and explain the two terms.

*Answer (short version):* differentiate $y_i = x_i/r$: $\partial y_i/\partial x_j = \delta_{ij}/r - x_i x_j/(d\,r^3)$; chain with $g$ and sum over $i$. The first term is the direct path scaled by $1/r$; the second subtracts the component of $g$ parallel to $x$ — the norm explains away magnitude-changing gradients because magnitude is invisible to the next layer (scale-invariance, §3.5).

**Q2.** Show that RoPE makes attention scores depend only on relative position.

*Answer (short version):* $\text{RoPE}(x, t) = R(t\theta)\,x$ with block-diagonal rotation $R$; rotations compose as $R(m\theta)^\top R(n\theta) = R((n-m)\theta)$. Hence $\langle \text{RoPE}(q,m), \text{RoPE}(k,n)\rangle = q^\top R((n-m)\theta)\,k$ — only $n-m$ appears (§6.3). In the repo, this falls out of the complex multiply in `models/mla.py:MultiHeadLatentAttention._apply_rope`, where the table row index is the absolute position.

**Q3.** Why does BF16 training in this repo need **no** loss scaling while FP16 training does?

*Answer (short version):* loss scaling exists to keep small FP16 values representable — FP16 has 5 exponent bits (range ±65504, underflow below ~6e-5). BF16 keeps FP32's 8 exponent bits (same ±3.4e38 range), so gradients and activations stay representable without a multiplier; the mantissa loss is absorbed by FP32 accumulation in matmuls and the FP32 master weights in AdamW (§14.2, §14.4).

**Q4.** Is the canonical config Chinchilla-consistent, and how many optimizer steps does a full run take?

*Answer (short version):* yes — 20 × 411.6M ≈ 8.2B ≈ the ~8.4B-token corpus (§12.4). At 65,536 tokens per optimizer step (batch 32 × 2048), one pass is ≈ 125,000–128,000 optimizer steps = 512,000 micro-steps at accumulation 4, matching `total_steps: 512000` (§12.2).

---

## 26. References

| Topic | Citation |
|---|---|
| Transformer | Vaswani et al., 2017 — arXiv:1706.03762 |
| RoPE | Su et al., 2021 — arXiv:2104.09870 |
| RMSNorm | Zhang & Sennrich, 2019 — arXiv:1910.07467 |
| LayerNorm | Ba, Kiros & Hinton, 2016 — arXiv:1607.06450 |
| SwiGLU | Shazeer, 2020 — arXiv:2002.05202 |
| FlashAttention | Dao et al., 2022 — arXiv:2205.14135 |
| BPE | Sennrich et al., 2016 — arXiv:1508.07909 |
| Adam | Kingma & Ba, 2015 — arXiv:1412.6980 |
| AdamW | Loshchilov & Hutter, 2019 — ICLR |
| Weight tying | Press & Wolf, 2017 — arXiv:1608.05859 |
| Chinchilla | Hoffmann et al., 2022 — arXiv:2203.15556 |
| μP / Tensor Programs | Yang et al., 2021 — arXiv:2203.03466 |
| YaRN | Peng et al., 2023 — arXiv:2309.00071 |
| DeepSeek-V2 (MLA) | Liu et al., 2024 — arXiv:2405.04434 |
| DeepSeek-V3 | DeepSeek-AI, 2024 — arXiv:2412.19437 |
| MoE survey | Fedus et al., 2022 — JMLR |

**Source files:** `models/transformer.py`, `models/mla.py`, `models/moe.py`, `models/mtp.py`, `training/pretrain.py`, `configs/pretrain_a100_422m.yaml`

---

## Cross-cutting lessons

1. **Every choice trades framework magic for readable math.** Pre-norm over post-norm, RMSNorm over LayerNorm, SwiGLU over ReLU-FFN, RoPE over learned position embeddings — each is the simpler, more inspectable, empirically-better option. DeepSeek's lineage is a sequence of "replace the magic with math that works better."
2. **The KV cache is the inference bottleneck.** §11 is the reason MLA exists; it is the connective tissue between attention (§5) and the architecture chapter.
3. **Scaling laws constrain everything.** Chinchilla (§12) sets the data budget; μP (§13) sets the LR. They are why a 411.6 M single-GPU run is a *deliberate* choice, not a limitation.
4. **Precision is a ladder, not a switch.** FP32 master → BF16 compute (§14) → FP8 (paper-spec; see [MLA & Mixed Precision](../concepts/attention-and-precision.md)): precision where gradients accumulate, savings where bytes move. Every component in this repo sits on the same principle.

---

## Architecture Overview

> **Purpose:** A single map of how every component in DeepSeek-V3-Lite fits together. Read this after [foundations](../concepts/foundations.md) and before diving into component-specific docs.

> **Read this if** you need the full system map. **Skip if** you only need MLA → [MLA](../concepts/attention-and-precision.md).

**Depends on:** (see Part I above) · **Read next:** [MLA](../concepts/attention-and-precision.md), [MoE & MTP](../concepts/moe-mtp.md)

**60-second summary:** DeepSeek-V3-Lite is an 18-layer, decoder-only, causal language model built from three architectural pillars — MLA attention (a low-rank-compressed attention that also replaces the KV cache), an aux-loss-free MoE FFN (20 routed + 1 shared expert per layer, top-4 active), and a depth-1 MTP head that turns the trunk into a speculative decoder. The whole model is 411.6M deduplicated parameters (~185M active per token) and is trained single-GPU on 8.4B Chinchilla-optimal tokens. Every number below was verified by instantiating the canonical config on CPU and walking `count_parameters` over it.

---

## Prerequisites

You should understand (or have skimmed):
- Causal language modeling objective — [foundations](../concepts/foundations.md) §2
- Pre-norm residual blocks — [foundations](../concepts/foundations.md) §3
- Chinchilla token budget — [foundations](../concepts/foundations.md) §12

---

## Table of Contents

Part I — Architecture Overview
1. [Design Goals](#design-goals)
2. [System Diagram](#system-diagram)
3. [Layer Topology](#layer-topology)
4. [Data Flow — Training](#data-flow--training)
5. [Data Flow — Inference](#data-flow--inference)
6. [Parameter Budget](#parameter-budget)
   - 6.1 [Component breakdown (canonical)](#61-component-breakdown-canonical)
   - 6.2 [Active vs total parameters](#62-active-vs-total-parameters)
   - 6.3 [Derivation sketch](#63-derivation-sketch)
7. [Memory Budget](#memory-budget)
   - 7.1 [Component estimate](#71-component-estimate)
   - 7.2 [Why the estimate is a lower bound](#72-why-the-estimate-is-a-lower-bound)
   - 7.3 [End-to-end forward memory timeline](#73-end-to-end-forward-memory-timeline)
8. [File Map](#file-map)
9. [Config → Code Routing](#config--code-routing)
10. [Module Dependency Graph](#module-dependency-graph)
11. [Scaling Knobs](#scaling-knobs)
12. [Request Lifecycles](#request-lifecycles)
13. [Load-Bearing Invariants](#load-bearing-invariants)
14. [Further Reading](#further-reading)

Part II — Transformer: Top-Level Wiring
15. [Abstract](#abstract)
16. [Pre-Norm Residual Architecture](#pre-norm-residual-architecture)
17. [Layer Topology (411.6M)](#layer-topology-4116m)
18. [FLOP Accounting per Layer Type](#flop-accounting-per-layer-type)
19. [SwiGLUFFN](#swigluffn)
20. [TransformerBlock](#transformerblock)
21. [Transformer Class](#transformer-class)
22. [Forward Contracts](#forward-contracts)
23. [Causal Mask Cache](#causal-mask-cache)
24. [Weight Tying](#weight-tying)
25. [Generation API — `generate()` and `_sample()` Walkthrough](#generation-api--generate-and-_sample-walkthrough)
26. [Gradient Checkpointing](#gradient-checkpointing)
27. [Config Shape](#config-shape)
28. [Tensor-Shape Walkthrough per Module](#tensor-shape-walkthrough-per-module)
29. [Comparison with GPT-2 / LLaMA](#comparison-with-gpt-2--llama)
30. [`enforce_triton_env_var` Integration](#enforce_triton_env_var-integration)
31. [Extension Points](#extension-points)
32. [Appendix A — Tensor shape trace](#appendix-a--tensor-shape-trace)
33. [Appendix B — FAQ](#appendix-b--faq)
34. [Appendix C — Glossary](#appendix-c--glossary)
35. [Load-Bearing Invariants (Part II)](#load-bearing-invariants-part-ii)
36. [References](#references)

Part III — Design Rationale and Transformer Block Details
37. [Chinchilla Scaling — Step-by-Step Derivation](#chinchilla-scaling--step-by-step-derivation)
38. [Memory Breakdown — Why 411.6M Fits on One A100](#memory-breakdown--why-4116m-fits-on-one-a100)
39. [Why MLA over GQA — Trade-off Analysis](#why-mla-over-gqa--trade-off-analysis)
40. [Why Aux-Loss-Free MoE](#why-aux-loss-free-moe)
41. [Why Top-4 of 20 Experts](#why-top-4-of-20-experts)
42. [Active vs Total Parameters — Deep Derivation](#active-vs-total-parameters--deep-derivation)
43. [Why 2 Dense Layers](#why-2-dense-layers)
44. [Why seq_len=2048](#why-seq_len2048)
45. [What This Project Deliberately Excludes](#what-this-project-deliberately-excludes)
46. [Transformer Block Details](#transformer-block-details)
   - 46.1 [Pre-Norm Rationale](#pre-norm-rationale)
   - 46.2 [RMSNorm Math and eps=1e-6](#rmsnorm-math-and-eps1e-6)
   - 46.3 [Embedding init std=0.006](#embedding-init-std0006)
   - 46.4 [Weight Tying — Savings Math](#weight-tying--savings-math)
   - 46.5 [Causal Mask Caching](#causal-mask-caching)
   - 46.6 [Gradient Checkpointing — Every Layer, use_reentrant=False](#gradient-checkpointing--every-layer-use_reentrantfalse)
   - 46.7 [Forward Contracts (detail)](#forward-contracts-detail)
47. [Check Your Understanding](#check-your-understanding)

---

## Design Goals

1. **Faithful V3 reproduction** — MLA, aux-loss-free MoE, MTP, SwiGLU, RMSNorm, RoPE. No architectural shortcuts.
2. **Raw PyTorch** — no HuggingFace Trainer, no Lightning. Every line is inspectable.
3. **Single-GPU training** — 1× A100 80GB, Chinchilla-optimal 8.4B tokens.
4. **CPU-testable** — all correctness tests run without CUDA or Triton.
5. **Optional Triton** — fused kernels for MoE dispatch and MLA attention (opt-in).

### Design philosophy — inspectability over convenience

Every architectural choice trades **framework magic** for **readable math**:

| Choice | Alternative rejected | Why this repo chose inspectability |
|---|---|---|
| Raw PyTorch modules | HuggingFace `PreTrainedModel` | MLA absorption and MoE bias updates are visible |
| Single-GPU loop | FSDP / DeepSpeed | One file (`training/pretrain.py`) explains the full train path |
| Explicit `uint32` shards | WebDataset streaming | On-disk format matches `PretrainDataset` line-by-line |
| Opt-in Triton | Always-on fused kernels | CPU tests compare against PyTorch reference |

**Pedagogical implication:** When you see a 15-line function instead of a one-liner, it is usually guarding an invariant documented in [testing](../concepts/kernels-and-ops.md).

---

## System Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PATH                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  data/pretrain_chinchilla/                                              │
│    shard_*.bin (uint32 tokens)                                          │
│         │                                                               │
│         ▼                                                               │
│  PretrainDataset ──► DataLoader ──► Pretrainer.train_step()             │
│                                         │                               │
│                    ┌────────────────────┼────────────────────┐          │
│                    ▼                    ▼                    ▼          │
│            MultiTokenPrediction    AdamW (FP32 master)   MoE bias      │
│                    │                    │               update         │
│                    ▼                    ▼                    │          │
│              Transformer ◄── torch.compile ── LR scheduler ─┘          │
│                    │                                                    │
│         ┌──────────┼──────────┐                                        │
│         ▼          ▼          ▼                                        │
│    Embedding   18× Block   LM Head (tied)                              │
│                    │                                                    │
│              ┌─────┴─────┐                                             │
│              ▼           ▼                                             │
│            MLA         FFN                                             │
│         (all layers)  dense/MoE                                        │
│                                                                         │
│  CheckpointManager ──► model_step_N.safetensors                         │
│                     ──► optim_step_N.pt                                 │
│                     ──► meta_step_N.json                                │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                        INFERENCE PATH                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Prompt tokens ──► Transformer.generate()  OR  SpeculativeDecoder       │
│                         │                           │                   │
│                    KV cache (MLA)              MTP draft head         │
│                         │                           │                   │
│                         ▼                           ▼                   │
│                    next token(s) ◄──── accept/reject ────┘              │
│                         │                                               │
│                         ▼                                               │
│                   detokenize (deepseek-coder-v2-lite)                   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Training path — tensor lifecycle

Understanding **where dtypes and shapes change** prevents subtle bugs:

| Stage | Tensor | dtype | Shape |
|---|---|---|---|
| Disk shard | raw tokens | uint32 | $(N_{\text{tokens}},)$ |
| `PretrainDataset` | chunk | uint32 → int64 at train boundary | $(S+1,)$ |
| `train_step` input | tokens, targets | int64 | $(B, S)$ |
| After embed | hidden | bf16 | $(B, S, 768)$ |
| MLA output | hidden | bf16 | $(B, S, 768)$ |
| MoE output | hidden | bf16 | $(B, S, 768)$ |
| Logits | float | bf16 | $(B, S, 100018)$ |
| Loss | scalar | fp32 | — |

**Gradient path:** `loss.backward()` flows through MTP heads (if enabled) into trunk, through MoE experts (only top-4 + shared receive gradients per token), through MLA, into embedding. MoE **gate bias** is updated out-of-band — not via autograd.

**Weight path:** uint32 shard → CPU DataLoader → `.to(cuda)` → int64 cast in `train_step` → BF16 activations; AdamW keeps FP32 master weights.

---

## Layer Topology

411.6M canonical config (`configs/pretrain_a100_422m.yaml` — the filename says "422m", the actual deduplicated count is 411,632,256):

```
Token IDs (B, S)     vocab = 100,018
       │
       ▼
  nn.Embedding (768) ─────────────────────────────┐ weight tying
       │                                          │
       ▼                                          │
  ┌─ Layer 0 ─────────────────────────────┐      │
  │  RMSNorm → MLA (12 heads, R=192)      │      │
  │  RMSNorm → SwiGLU (inter_dim=1536)    │      │  DENSE
  └───────────────────────────────────────┘      │
  ┌─ Layer 1 ─────────────────────────────┐      │
  │  RMSNorm → MLA                        │      │  DENSE
  │  RMSNorm → SwiGLU (inter_dim=1536)    │      │
  └───────────────────────────────────────┘      │
  ┌─ Layers 2–17 (×16) ───────────────────┐      │
  │  RMSNorm → MLA                        │      │  MoE
  │  RMSNorm → DeepSeekMoE                │      │
  │    ├─ AuxLossFreeGate (20 experts)    │      │
  │    ├─ 4 routed SwiGLU (I=384)         │      │
  │    └─ 1 shared SwiGLU (I=384)         │      │
  └───────────────────────────────────────┘      │
       │                                          │
       ▼                                          │
  RMSNorm → Linear(768 → 100018) ◄───────────────┘
       │
       ▼
  Logits (B, S, 100018)

  Parallel path (training only):
  MTPModule(depth=1) takes (hidden, embed(t+1)) → predicts token t+2
```

### Why two dense layers before MoE?

DeepSeek-V3 places **dense** SwiGLU FFN in the first layers and MoE in deeper layers. Rationale (design, not enforced in code):

1. **Early layers** learn local syntax and token co-occurrence — dense FFN is simpler and well-conditioned.
2. **Deep layers** benefit from **specialised experts** (code vs math vs natural language patterns in the mixture).
3. **Routing stability** — routing on unprocessed early representations is noisier.

At 411.6M scale: `n_dense_layers=2` of 18 total. The 1650 smoke config preserves the same pattern at 2 of 4 layers.

### The canonical dimensions, tabulated

Every architectural number below comes from `model:` in `configs/pretrain_a100_422m.yaml` and lands in exactly one module:

| Key | Value | Consumed by | What it sets |
|---|---|---|---|
| `vocab_size` | 100,018 | `Transformer.embed`, `Transformer.head` | token table width; logits width |
| `dim` | 768 | every module | hidden width of the residual stream |
| `n_layers` | 18 | `Transformer.layers` | stack depth (2 dense + 16 MoE) |
| `n_heads` | 12 | `MultiHeadLatentAttention` | query heads; MLA head count |
| `n_dense_layers` | 2 | `TransformerBlock.__init__` | layers 0–1 dense, 2–17 MoE |
| `n_routed_experts` | 20 | `AuxLossFreeGate`, `DeepSeekMoE` | routed experts per MoE layer |
| `n_shared_experts` | 1 | `DeepSeekMoE` | always-active experts per MoE layer |
| `n_activated_experts` | 4 | `AuxLossFreeGate` | top-k routing depth |
| `inter_dim` | 1536 | `SwiGLUFFN` | dense FFN hidden width |
| `moe_inter_dim` | 384 | `Expert` | per-expert hidden width |
| `kv_lora_rank` | 192 | `MultiHeadLatentAttention` | shared KV latent width $R$ |
| `q_lora_rank` | 0 | `MultiHeadLatentAttention` | 0 → `wq` direct, no `wq_a`/`q_norm` |
| `qk_nope_head_dim` | 48 | `MultiHeadLatentAttention` | non-RoPE part of each query/key head |
| `qk_rope_head_dim` | 24 | `MultiHeadLatentAttention` | RoPE part of each query/key head |
| `v_head_dim` | 64 | `MultiHeadLatentAttention` | value head width |
| `max_seq_len` | 2048 | `Transformer`, MLA cache | context cap; cache bound |
| `rope_theta` | 10000 | `_extend_rope` | RoPE base frequency |
| `mtp_depth` / `mtp_loss_weight` | 1 / 0.3 | `MultiTokenPrediction` | side-branch depth and loss share |

Two invariants to note: `qk_nope_head_dim + qk_rope_head_dim = 72 = n_heads`-compatible query width, and `kv_lora_rank + qk_rope_head_dim = 216` — the exact number of floats cached per token per layer (the "216" of the MLA cache contract).

---

## Data Flow — Training

```
1. Dataset yields (tokens, targets)     shape: (B, S) each, uint32 on disk
2. Cast to int64 at train_step boundary
3. MultiTokenPrediction.forward():
     a. main_model.forward_with_hidden() → main_logits (B,S,V), hidden (B,S,D)
     b. For each MTP depth d:
        - h_in  = hidden[:, :usable]
        - emb   = embed(tokens[:, d+1 : d+1+usable])
        - tgt   = tokens[:, d+2 : d+2+usable]
        - mtp_logits = MTPModule(h_in, emb)
4. compute_loss():
     main_loss = CE(main_logits, targets)
     mtp_loss  = mean_depth(CE(mtp_logits, tgt))
     total     = main_loss + 0.3 * mtp_loss
5. loss / grad_accum → backward → clip → optimizer.step
6. Every bias_update_every steps: update MoE gate biases
7. Every save_interval steps: atomic checkpoint
```

### Training loss decomposition

When MTP is enabled (`mtp_depth=1`, `mtp_loss_weight=0.3`):

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}^{(0)} + 0.3 \cdot \mathcal{L}_{\text{CE}}^{(1)}
$$

where $\mathcal{L}_{\text{CE}}^{(0)}$ is next-token loss on the main head and $\mathcal{L}_{\text{CE}}^{(1)}$ predicts $t+2$ from $(h_t, e_{t+1})$.

**Alignment detail:** For sequence length $S$, MTP usable positions = $S - 2$ (need $t$, $t+1$, $t+2$). See `models/mtp.py:MultiTokenPrediction.forward`.

Key: `use_cache=False` during training. KV cache is inference-only.

---

## Data Flow — Inference

### Standard decode

```
1. reset_cache()
2. Prefill: forward(prompt, start_pos=0, use_cache=True)  → logits for all prompt positions
3. Sample token t₁ from last logits
4. Loop: forward([tᵢ], start_pos=prompt_len+i-1, use_cache=True) → logits → sample tᵢ₊₁
```

### Speculative decode

```
1. reset_cache()
2. Prefill prompt
3. Each step:
   a. Main model decodes t₁ (greedy)
   b. MTP drafts t₂ from (hidden, embed(t₁))
   c. Accept t₂ if p_main(t₂) ≥ threshold × p_draft(t₂)
   d. Emit 1 or 2 tokens
```

### Speculative decode — probability contract

`SpeculativeDecoder.generate_step` (`inference/speculative.py:SpeculativeDecoder.generate_step`) implements a simplified accept rule:

$$
\text{accept}(t_2) \iff p_{\text{main}}(t_2) \geq \tau \cdot p_{\text{draft}}(t_2)
$$

with $\tau = 0.8$ default. This is **not** the full optimal speculative sampling algorithm (Leviathan et al.) but a pedagogical approximation that preserves correctness (reject → fall back to one token).

See [inference](../inference.md) and [mtp](../concepts/moe-mtp.md).

---

## Parameter Budget

> **60-second summary.** All three parameter-budget tables that used to live in this document have been consolidated into this one canonical table. The numbers were verified by instantiating `models/transformer.py:Transformer` from the canonical config on CPU and summing `models/transformer.py:count_parameters` per component; every row below is exact, not rounded-down prose. The per-module API surface and a second view of the same numbers live in [R2 transformer api](../references/R2_transformer_api.md).

| Component | Exact params | ~M | % of total |
|---|---|---|---|
| MoE routed experts (16 layers × 20 SwiGLU, $I=384$) | 283,115,520 | 283.1M | 68.8% |
| Embedding + LM head (tied — counted once) | 76,813,824 | 76.8M | 18.7% |
| MLA attention (18 layers) | 30,195,072 | 30.2M | 7.3% |
| MoE shared experts (16 layers × 1 SwiGLU, $I=384$) | 14,155,776 | 14.2M | 3.4% |
| Dense SwiGLU (2 layers, $I=1536$) | 7,077,888 | 7.1M | 1.7% |
| MoE gate weights (16 × 768 × 20) | 245,760 | 0.25M | 0.1% |
| RMSNorm (36 block norms + final norm) | 28,416 | 0.03M | <0.1% |
| **Total — base model (deduplicated)** | **411,632,256** | **411.6M** | 100% |
| MTP heads (depth 1) | +7,081,728 | +7.1M | — |
| **Total — with MTP** | **418,713,984** | **418.7M** | — |

Three things to internalise from this table:

1. **The MoE experts are 72% of the model** (283.1M routed + 14.2M shared = 297.3M of 411.6M). This is the entire point of sparse FFNs: you pay storage for 21 experts per layer, but only 5 execute per token.
2. **The embedding is the second-largest line item** (76.8M) *because* the vocabulary is huge (100,018) and weight tying means it is counted once. If tying were off, the head would add another 76.8M and the total would read ~488M.
3. **The "422M" name is a filename, not a measurement.** The config file is `pretrain_a100_422m.yaml` for historical reasons; the actual deduplicated count is 411,632,256 (411.6M), 418,713,984 (418.7M) with MTP. Any doc or log that says "422M parameters" is quoting the nominal, not the model.

The same component breakdown is logged at training init when `training.log_per_component_params: true` by `training/pretrain.py:Pretrainer._log_per_component_params` — the canonical table above is exactly what that logger prints, with one caveat: the logger's `rmsnorm` bucket misses the final `norm.weight` (name length quirk, see Pitfalls below), so its `other` bucket shows 768 params.

**Pitfall (logger quirk):** the repo's component classifier tests `name.endswith(".norm.weight")`, and `"norm.weight"` (11 chars) cannot end with `".norm.weight"` (12 chars). The final `Transformer.norm` therefore lands in the `other` bucket, reported as 768 params (0.00%). Cosmetic only — the total is unaffected.

**How to reproduce the table yourself** (CPU-safe, ~5 s):

```python
import yaml
from models.transformer import Transformer, count_parameters

cfg = yaml.safe_load(open("configs/pretrain_a100_422m.yaml"))
total, trainable = count_parameters(Transformer(cfg["model"]))
# (411632256, 411632256)
```

The per-component rows come from the same loop the trainer logs: walk `named_parameters()`, skip any `id(p)` already seen (that is what deduplicates the tied head), and bucket by name substrings (`embed` / `.attn.` / `.experts.` / `shared_experts` / `.ffn.w` / `.gate.` / norms). `training/pretrain.py:Pretrainer._log_per_component_params` is the canonical implementation; the table above is its output for the canonical config.

### 6.1 Component breakdown (canonical)

The table above is the single source of truth for this document. Older drafts of this chapter carried three further budget tables with drifting numbers (MLA "35M" / "45M" / "52M" / "58M" in various drafts; MTP "3M" vs "7M"; routed experts "210M" vs "298M"). All were replaced by the verified table. If you see a component figure in this repo's docs that disagrees with the table, the table wins — it was produced by running the code.

### 6.2 Active vs total parameters

| Component | Active params per token |
|---|---|
| Dense SwiGLU (2 layers) | 7,077,888 |
| MoE: 16 layers × (4 routed + 1 shared) experts | 70,778,880 |
| Non-FFN (embed + MLA + norms + gates + tied head) | 107,283,072 |
| **Active per token** | **185,139,840 (~185M, 45.0%)** |
| Inactive (unselected routed experts) | 226,492,416 (~226M, 55.0%) |
| **Total (deduplicated, weight-tied)** | **411,632,256** |

**Why ~185M of 411.6M?** Per token, each MoE layer executes exactly 5 experts (4 routed + 1 shared) out of the 21 it stores. The 16 unselected routed experts per layer — $16 \times 16 \times 884\,736 = 226\,492\,416$ parameters — sit in VRAM but receive neither activations nor gradients. Everything else (embedding, MLA, norms, gates, dense FFN, and the tied head) is active for every token. See [§42](#active-vs-total-parameters--deep-derivation) for the full decomposition.

### 6.3 Derivation sketch

**MoE expert params** (one SwiGLU expert, `models/moe.py:Expert`): $3 \times d \times I_{\text{moe}} = 3 \times 768 \times 384 = 884\,736$.

- Routed: $16 \times 20 \times 884\,736 = 283\,115\,520$
- Shared: $16 \times 1 \times 884\,736 = 14\,155\,776$

**Embedding + tied head:** $100\,018 \times 768 = 76\,813\,824$ (counted once because `head.weight = embed.weight` shares storage — see `models/transformer.py:Transformer.__init__`).

**MLA per layer** (`models/mla.py:MultiHeadLatentAttention.__init__`, with `q_lora_rank=0` so `wq_a`/`q_norm` do not exist):

| Sub-module | Weight shape | Params |
|---|---|---|
| `wq` | (864, 768) | 663,552 |
| `wkv_a` | (216, 768) | 165,888 |
| `kv_norm` | (192,) | 192 |
| `wkv_b` | (1344, 192) | 258,048 |
| `wo` | (768, 768) | 589,824 |
| **Total per layer** | — | **1,677,504** |

Eighteen layers: $18 \times 1\,677\,504 = 30\,195\,072$.

**Dense SwiGLU (2 layers):** $2 \times 3 \times 768 \times 1536 = 7\,077\,888$.

**Gates:** $16 \times 768 \times 20 = 245\,760$. **Norms:** $18 \times 2 \times 768 + 768 = 28\,416$.

Sum: $76\,813\,824 + 283\,115\,520 + 14\,155\,776 + 245\,760 + 7\,077\,888 + 30\,195\,072 + 28\,416 = 411\,632\,256$. MTP adds $7\,081\,728 \rightarrow 418\,713\,984$.

---

## Memory Budget

> **60-second summary.** Training on one A100 80GB (`B=8, S=2048, grad_checkpoint=True`) is estimated at ~30.5 GB peak — comfortable headroom on 80 GB. The estimate comes from `utils/memory.py:estimate_model_memory_gb`, a closed-form sum of params + optimizer state + activations + static overhead. **No GPU training run has ever been executed** — every figure in this section is a budget estimate from arithmetic, not a measurement (`.benchmarks/` is empty; see [testing](../concepts/kernels-and-ops.md) for the formula provenance).

### 7.1 Component estimate

| Component | ~GB |
|---|---|
| BF16 params | 0.8 |
| AdamW FP32 state | 5.0 |
| Activations (×24 factor, grad-checkpoint on) | 10.9 |
| CUDA overhead | 13.7 |
| **Total estimate** | **~30.5** |

Estimated peak with 15–20% headroom: **~30–36 GB**. No GPU run has been executed yet — these are budget figures, not measurements.

The estimator (`utils/memory.py`) sums:

$$
M_{\text{total}} = M_{\text{params}} + M_{\text{optim}} + M_{\text{act}} + M_{\text{overhead}}
$$

- $M_{\text{params}} = 2N$ bytes (BF16): $411\,632\,256 \times 2 \approx 0.82$ GB.
- $M_{\text{optim}} = 12N$ bytes (FP32 master + first moment + second moment, 4 bytes each): $411\,632\,256 \times 12 \approx 4.9$ GB (5.0 with MTP).
- $M_{\text{act}} = 24 \times B \cdot S \cdot D \cdot L \times 2$ bytes with grad checkpointing (PaLM Appendix A factor; 36× without): $24 \times 8 \times 2048 \times 768 \times 18 \times 2 = 10\,871\,635\,968 \approx 10.9$ GB.
- $M_{\text{overhead}}$: 2 GB on CPU, `min(13.7, 0.17 × total)` GB on CUDA (`STATIC_PYTORCH_OVERHEAD_GB = 13.7` in `utils/memory.py`).

Note that the **KV cache does not appear in the training estimate**: `use_cache=False` during training means `models/mla.py:MultiHeadLatentAttention._ensure_cache` is never called, so no cache tensors are allocated. The cache only exists at inference: 18 layers × $B \cdot S \cdot (192 + 24) \times 2$ bytes = 0.13 GB at $B{=}8, S{=}2048$.

### 7.2 Why the estimate is a lower bound

Analytical estimate (`utils/memory.py`) sums:

$$
M_{\text{total}} = M_{\text{params}} + M_{\text{optim}} + M_{\text{act}} + M_{\text{overhead}}
$$

**Sources of underestimate:**
1. `torch.compile` workspace (not modelled)
2. CUDA caching allocator retaining freed blocks
3. MoE dispatch temporaries (stacked path copies expert weights)
4. Gradient buckets during backward
5. The logits tensor and cross-entropy intermediates — see the timeline below

Rule of thumb: add **15–20% headroom** above estimate before launching multi-hour runs.

### 7.3 End-to-end forward memory timeline

Walking one training micro-step (`B=8, S=2048`, BF16, SDPA attention, grad checkpointing) through `models/transformer.py:Transformer.forward` and `training/pretrain.py:Pretrainer.train_step`, in allocation order. All numbers are derived from the config and the tensor shapes — again, estimates, not measurements.

| # | Stage | What allocates | Size | Notes |
|---|---|---|---|---|
| 0 | Model construction | BF16 weights | 0.82 GB | `Transformer.__init__` builds in fp32 (~1.6 GB transient) then `train_step` moves to BF16 |
| 1 | Optimizer init | FP32 master + m1 + m2 | 4.9 GB | AdamW state, 12 bytes/param |
| 2 | Dataloader | tokens/targets int64 | ~0.3 MB | $(B,S{+}1)$ int64 — negligible |
| 3 | Embed | hidden $h$ | 25.2 MB | $(8, 2048, 768)$ BF16 |
| 4 | Per-layer forward | RMSNorm outputs, MLA q/kv/attn, FFN intermediates | tens of MB/layer, freed as layers complete | MLA activations are small ($q{:} (8,2048,12,72)$ = 28 MB); dense gate/up $(8,2048,1536)$ = 50 MB each |
| 4a | MoE layers | **Stacked expert copies** | 0.62 GB total, held | `_routed_forward_stacked` re-stacks 21 experts × 3 weights per layer (37.2 MB/layer) into `_stacked_w{1,2,3}`; references persist until the next forward |
| 5 | LM head | **Logits** | **3.3 GB** | $(8, 2048, 100018)$ BF16 — the single largest tensor in the step |
| 6 | Loss | CE softmax buffer | ~3.3 GB | `F.cross_entropy` over 1.6M rows × 100K classes; another buffer of logits scale |
| 7 | Backward | Gradients + recomputed activations | grads 0.82 GB; activations peak inside the 24× envelope (10.9 GB) | With grad checkpointing, each layer's forward is re-run during backward, so peak backward activation ≈ one layer's activations, not 18 layers' |
| 8 | Optimizer step | in-place on FP32 masters | — | Gradients freed after `zero_grad` + step |
| 9 | Every 4000 steps | Checkpoint write | disk: 0.82 + 4.9 GB | `utils/checkpoint.py:CheckpointManager` atomic save |

**What this timeline teaches:**

- **The logits are the memory bottleneck, not the trunk.** At $B{=}8, S{=}2048$ the logits + CE pair is ~6.6 GB of transient traffic through the caching allocator. This is the direct consequence of a 100K vocabulary on a small hidden size — see the FLOP section for the compute twin of this effect.
- **The 24× activation factor already covers the spiky stages.** The PaLM-formula 10.9 GB envelope is meant to absorb forward activations plus the logits/CE transients; the timeline shows why the estimator can still under-report (transients can spike above the envelope briefly) — hence the 15–20% headroom rule.
- **MoE layers hold a quiet 0.62 GB of stacked copies.** Re-stacking every forward (a deliberate staleness fix, see `models/moe.py:DeepSeekMoE.forward`) means each MoE layer keeps its 37 MB of stacked BF16 weights alive between forwards.
- **Inference is a different regime:** no optimizer state (drop 4.9 GB), no gradient activations (drop ~10.9 GB), add KV cache (0.13 GB). An A100-sized inference load is dominated by weights (0.82 GB) and the logits transient for large batches.

### The backward pass (same micro-step)

Backward runs the graph in reverse, layer 18 → 0, and looks like this in memory:

| # | Stage | What allocates | Size | Notes |
|---|---|---|---|---|
| 10 | Loss grad | `d_logits` | 3.3 GB | gradient of the same shape as logits, freed as the head backward consumes it |
| 11 | Head backward | grad for `norm(h)` and `head.weight` | 25 MB + 0.15 GB | head.weight grad is written (it feeds the tied embedding grad later) |
| 12 | Checkpointed recompute | per-layer forward re-run, layer by layer | ≈ one layer's activations at a time | `torch.utils.checkpoint` discards each layer's graph after its backward (functional, `use_reentrant=False`) |
| 13 | Per-layer grads | expert `w1/w2/w3` grads, gate grad, `wq/wkv/wo` grads | 0.82 GB total once all layers done | MoE expert grads exist **only** for experts that fired this step (scatter-add backward) |
| 14 | Optimizer step | in-place FP32 updates; grads zeroed | — | master + m1 + m2 live in 4.9 GB the whole time |

Two subtleties worth knowing:

- **Gradients are sparse in the MoE sense, dense in storage.** Each token contributed to only 5 experts, but the per-expert grads are full matrices — the *sparsity* saves compute in the backward GEMMs, not memory for the grad tensors.
- **The recompute window is what makes grad checkpointing win.** Without it, stages 4's activations for all 18 layers would have to survive until stage 12; with it, only the region input (25 MB) survives and each layer re-runs inside backward. Peak backward activation is therefore ≈ one layer, not 18.

### Inference-mode timeline (for contrast)

`Transformer.generate` (B=1, S=2048 context):

| Stage | Size | Notes |
|---|---|---|
| Weights (BF16) | 0.82 GB | loaded from checkpoint |
| KV cache (18 layers × (192+24)/token) | 15.9 MB | `_ensure_cache` allocates on first cached forward; `max(bsz, _cache_batch*2, 16)` growth policy |
| Prefill activations | tens of MB | one pass, layers release as they complete |
| Prefill logits | 0.41 GB | $(1, 2048, 100018)$ BF16 — still the biggest transient at B=1 |
| Decode step | <1 MB/step | single-token forward; logits $(1,1,100018)$ |

Decode memory is dominated by weights + cache — which is exactly why MLA's 216-float cache per token per layer matters (vs 1536 for MHA at these dims).

---

## File Map

### Directory layout

```
models/
  transformer.py       # Transformer, TransformerBlock, SwiGLUFFN
  mla.py               # MultiHeadLatentAttention
  mla_triton.py        # Optional fused MLA kernel
  moe.py               # AuxLossFreeGate, DeepSeekMoE
  moe_triton.py        # Optional fused MoE kernel
  mtp.py               # MultiTokenPrediction
  _triton_dispatch.py  # ENABLE_TRITON_KERNELS guard

training/
  pretrain.py          # Pretrainer, PretrainDataset, TrainingConfig

inference/
  generate.py          # Interactive CLI
  speculative.py       # SpeculativeDecoder

utils/
  checkpoint.py        # CheckpointManager
  memory.py            # VRAM estimation
  logging.py           # TrainingLogger + WandB

data/
  prepare_data.py      # Shim to universal pipeline (LLM/shared_data)

configs/
  pretrain_a100_422m.yaml   # Canonical recipe
  pretrain_1650_2m.yaml     # Tiny smoke config

scripts/
  launch_a100.sh, microbench_a100.py, step_time_a100.py, smoke_forward.py, ...

tests/
  test_models.py, test_training.py, test_inference.py, test_moe_triton.py, ...
```

### Quick lookup

| Question | File |
|---|---|
| Where is LR schedule? | `training/pretrain.py` (`make_warmup_cosine_lambda`) |
| Where is MoE routing? | `models/moe.py` (`AuxLossFreeGate`) |
| Where is MLA absorption? | `models/mla.py` (SDPA path) |
| Where is speculative accept? | `inference/speculative.py` (`generate_step`) |
| Where is config parsed? | `training/pretrain.py` (`main`) |
| Where are test fixtures? | `tests/conftest.py` |

### Key files by size

| Path | LOC | Responsibility |
|---|---|---|
| `models/moe_triton.py` | 536 | Fused grouped-GEMM MoE kernel |
| `training/pretrain.py` | 445 | Train loop |
| `models/mla_triton.py` | 339 | Fused MLA attention kernel |
| `models/mla.py` | 230 | MLA attention + cache |
| `models/moe.py` | 207 | Aux-loss-free MoE |
| `models/transformer.py` | 196 | Stack + generate |
| `models/mtp.py` | 117 | MTP heads |
| `utils/checkpoint.py` | 110 | Atomic I/O |
| `inference/generate.py` | 125 | Interactive CLI |
| `inference/speculative.py` | 60 | Speculative decode |

---

## Config → Code Routing

| YAML key | Read by | Effect |
|---|---|---|
| `model.n_dense_layers` | `TransformerBlock` | dense vs MoE FFN |
| `model.attn_impl` | `MultiHeadLatentAttention` | sdpa / manual / triton |
| `model.moe_dispatch` | `DeepSeekMoE` | stacked / triton_grouped |
| `model.mtp_depth` | `Pretrainer`, `MultiTokenPrediction` | enables MTP wrapper |
| `training.mup_lr` | `Pretrainer.__init__` | μP LR scaling |
| `training.bias_update_every` | `train_step` | MoE bias cadence |
| `training.nan_guard` | `train` loop | checkpoint rollback |

Full reference: [configs](../training.md) and [R1 config schema](../references/R1_config_schema.md).

---

## Module Dependency Graph

**Call graph (runtime):**

```
main() / generate CLI
  └─ Pretrainer.train()              training/pretrain.py
       ├─ PretrainDataset            mmap uint32 shards
       ├─ MultiTokenPrediction       models/mtp.py (if mtp_depth>0)
       │    └─ Transformer.forward_with_hidden
       ├─ CheckpointManager          utils/checkpoint.py
       └─ TrainingLogger             utils/logging.py

Transformer.forward
  └─ TransformerBlock × L
       ├─ MultiHeadLatentAttention   models/mla.py [→ mla_triton.py opt]
       └─ SwiGLUFFN | DeepSeekMoE    models/moe.py [→ moe_triton.py opt]
```

**Import graph (build time):**

```
configs/*.yaml
    ├── models/transformer.py ──┬── models/mla.py
    │                           ├── models/moe.py ── models/moe_triton.py (opt)
    │                           ├── models/mtp.py
    │                           └── models/_triton_dispatch.py
    ├── training/pretrain.py ─── utils/{checkpoint,memory,logging}.py
    ├── data/prepare_data.py ── LLM/shared_data/ (external)
    └── inference/generate.py ── inference/speculative.py

inference/generate.py
  ├── models/transformer.py
  ├── models/mtp.py
  └── inference/speculative.py
```

**Acyclic rule:** `models/` never imports `training/` or `inference/`. Utilities are leaf nodes.

---

## Scaling Knobs

### Config presets

| Stage | Config | Purpose |
|---|---|---|
| Smoke | `pretrain_1650_2m.yaml` | Structural correctness, 4GB GPU |
| Research | `pretrain_a100_422m.yaml` | Chinchilla-optimal single-GPU run |
| Production | not in repo | Multi-GPU, larger width — out of scope |

The **same code paths** run at all scales; only YAML dimensions and batch settings change.

### Experiment goals

| Goal | Knobs | Doc |
|---|---|---|
| Fit smaller GPU | ↓ `micro_batch_size`, ↓ `max_seq_len`, enable `grad_checkpoint` | [configs](../training.md) |
| Faster iteration | `pretrain_1650_2m.yaml`, `build_small_pretrain_data.py` | [scripts](../concepts/kernels-and-ops.md) |
| Higher MFU | `ENABLE_TRITON_KERNELS=1`, `moe_dispatch: triton_grouped` | [triton kernels](../concepts/kernels-and-ops.md) |
| Longer context | ↑ `max_seq_len`, consider YaRN `rope_factor` | [MLA](../concepts/attention-and-precision.md) |

### Width / depth trade-offs (411.6M → larger)

| Knob | Effect | Risk |
|---|---|---|
| `dim`, `n_layers` | Linear param growth | VRAM, μP LR |
| `n_routed_experts` | Capacity without dense FLOPs | Routing instability |
| `n_activated_experts` | Compute per token | Expert collapse |
| `max_seq_len` | Quadratic attention memory | OOM |
| `micro_batch_size` | Throughput vs memory | MFU |

Full DeepSeek-V3 (671B) uses the same **family** of components at different scales — read component docs for which knobs are architectural vs scale-specific.

---

## Request Lifecycles

### Training job

```
launch_a100.sh
  → python -m training.pretrain --config configs/pretrain_a100_422m.yaml
    → Pretrainer reads YAML
    → PretrainDataset mmap shards
    → train loop: forward/backward/optim
    → CheckpointManager writes triplets every save_every
```

### Inference session

```
python -m inference.generate --config ... --checkpoint ...
  → Transformer + optional MTPModule load
  → REPL: chat template → tokenize → generate / speculative
  → KV cache lives until process exit or reset_cache()
```

---

## Load-Bearing Invariants

| Invariant | Where enforced | Test (if any) |
|---|---|---|
| `use_cache=False` during training | `train_step`, `forward_with_hidden` default | `test_training.py` |
| MoE bias buffer not Parameter | `models/moe.py:AuxLossFreeGate` | `test_bias_not_in_parameters` |
| Triton gated by env var | `models/_triton_dispatch.py:enforce_triton_env_var` | `test_force_back.py` |
| Weight tying dedup on save | `training/pretrain.py:Pretrainer.save_checkpoint` | checkpoint roundtrip tests |
| μP LR after MTP wrap | `training/pretrain.py:Pretrainer.__init__` | `test_mup_lr_scaling` |

Full test mapping: [testing](../concepts/kernels-and-ops.md).

---

## Further Reading

- Attention deep-dive: [MLA](../concepts/attention-and-precision.md)
- MoE deep-dive: [moe](../concepts/moe-mtp.md)
- Training loop: [training](../training.md)
- Learning path: [getting started](../guides/getting-started.md)



---

## Transformer — Top-Level Wiring

## A Comprehensive Technical Reference

> **Prerequisites:** [foundations](../concepts/foundations.md), [MLA](../concepts/attention-and-precision.md), [moe](../concepts/moe-mtp.md).

> **Covers**: `Transformer`, `TransformerBlock`, `SwiGLUFFN` in `models/transformer.py` — how the full DeepSeek-V3 stack is assembled and executed.

> **Read this if** you need the end-to-end forward/generate wiring. **Read first** if new to the repo: [architecture](../concepts/foundations.md) → [MLA](../concepts/attention-and-precision.md) / [moe](../concepts/moe-mtp.md).

---

## Abstract

The `Transformer` class is the root `nn.Module` for DeepSeek-V3-Lite. It stacks 18 `TransformerBlock` layers (2 dense SwiGLU + 16 MoE), each consisting of pre-norm MLA attention and pre-norm FFN. It exposes three critical interfaces: `forward` (training/inference logits), `forward_with_hidden` (MTP training), and `generate` (autoregressive decode with KV cache). This part walks the wiring top to bottom: construction, the residual architecture, every forward contract, the mask cache, weight tying, generation, checkpointing, and — new in this edition — a per-module tensor-shape walkthrough ([§28](#tensor-shape-walkthrough-per-module)) and a full FLOP accounting ([§18](#flop-accounting-per-layer-type)). The per-symbol API reference lives in [R2 transformer api](../references/R2_transformer_api.md).

---

## Line-by-Line Construction Walkthrough

### `Transformer.__init__` (`models/transformer.py:Transformer.__init__`)

```python
model_cfg = config.get("model", config)
enforce_triton_env_var(model_cfg, print)  # AGENTS rule #7
```

**Step 1 — Config unwrap:** Tests pass flat dicts; YAML passes nested `{"model": {...}}`. Single unwrap point prevents double-nesting bugs.

**Step 2 — Triton guard:** If `attn_impl=triton` or `moe_dispatch=triton_grouped` without `ENABLE_TRITON_KERNELS=1`, force-back to `sdpa`/`stacked` with one warning. The `print` argument routes the warning to stdout (this construction path is outside the pretrain entry point, which uses its own logger).

**Step 3 — Embedding:** `nn.Embedding(vocab_size, dim)` with `N(0, 0.006²)` init.

**Step 4 — Layers:** `ModuleList([TransformerBlock(i, model_cfg) for i in range(n_layers)])`. Layer index determines dense vs MoE.

**Step 5 — Head + tying:** `Linear(dim, vocab_size, bias=False)`. If `weight_tying`: `head.weight = embed.weight` (same storage).

### `TransformerBlock.forward` (`models/transformer.py:TransformerBlock.forward`)

```python
def forward(self, x: torch.Tensor, start_pos: int = 0, mask: Optional[torch.Tensor] = None, use_cache: bool = True) -> torch.Tensor:
    x = x + self.attn(self.attn_norm(x), start_pos, mask, use_cache)
    x = x + self.ffn(self.ffn_norm(x))
    return x
```

**Residual highway:** $\mathbf{x}$ shape $(B, S, d)$ unchanged across both sub-layers. Gradients flow through `+` directly to earlier layers (pre-norm benefit).

**MoE note:** `DeepSeekMoE.forward(x)` flattens to $(B \cdot S, d)$ internally for routing — position-agnostic expert assignment. See [§28](#tensor-shape-walkthrough-per-module) for the full trace.

### `count_parameters` — deduplication logic (`models/transformer.py:count_parameters`)

```python
seen = set()
for p in model.parameters():
    if id(p) not in seen:
        seen.add(id(p))
        total += p.numel()
```

Weight tying means `embed.weight` and `head.weight` share `id` — counted once. Critical for μP LR denominator.

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

### Why pre-norm preserves gradients (one line of calculus)

Write a block as $x_{\ell} = x_{\ell-1} + f_{\ell}(\text{RMSNorm}(x_{\ell-1}))$. The Jacobian of the residual path is

$$
\frac{\partial x_{\ell}}{\partial x_{\ell-1}} = I + J_{\ell}, \qquad J_{\ell} = \frac{\partial f_{\ell}(\text{RMSNorm}(x_{\ell-1}))}{\partial x_{\ell-1}}
$$

so by the chain rule the gradient through the full stack expands into products of $(I + J)$ factors. The identity term $I$ survives every multiplication: there is always a "straight-through" gradient path from the loss to the embeddings that never passes through a norm's inverse-scale factor. Post-norm blocks ($x_\ell = \text{Norm}(x_{\ell-1} + f_\ell(x_{\ell-1}))$) lack this clean identity — the norm's derivative multiplies the residual sum and can shrink the path at depth. That is the formal reason 18-layer stacks with pre-norm train without normalisation tricks beyond RMSNorm itself.

---

## Layer Topology (411.6M)

| Layer ID | FFN type | FFN width | Attention |
|---|---|---|---|
| 0, 1 | `SwiGLUFFN` | `inter_dim=1536` | MLA |
| 2–17 | `DeepSeekMoE` | `moe_inter_dim=384` × 21 experts | MLA |

Selection logic in `TransformerBlock.__init__` (`models/transformer.py:TransformerBlock.__init__`):

```python
self.ffn = SwiGLUFFN(self.dim, config["inter_dim"]) if layer_id < self.n_dense_layers else DeepSeekMoE(config)
```

With `n_dense_layers=2`, layers 0–1 are dense, 2–17 are MoE.

---

## FLOP Accounting per Layer Type

> **60-second summary.** A matmul of shape $(M, K) \times (K, N)$ costs $2 M N K$ FLOPs. On that basis, one dense FFN layer costs 7.1 MFLOPs/token, one MoE layer costs 8.8 MFLOPs/token (5 experts executed), and the LM head — often forgotten — costs 153.6 MFLOPs/token, making it the largest single line item in the model. At $S{=}2048$ the whole forward is ~0.49 GFLOPs/token. All figures below are **derived** from config dimensions and the exact matrix shapes in the code — none are measured (no GPU run has executed).

### The primitive

A linear layer $y = x W^\top$ with $x \in \mathbb{R}^{M \times K}$, $W \in \mathbb{R}^{N \times K}$ costs

$$
\text{FLOPs} = 2 M N K.
$$

Per token ($M = 1$), every FLOP formula below drops the $M$ factor.

### FFN layers

**Dense SwiGLU** (`models/transformer.py:SwiGLUFFN.forward`): three matmuls — gate, up, down:

$$
\text{FLOPs}_{\text{dense}} = 6 \cdot d \cdot I = 6 \times 768 \times 1536 = 7\,077\,888 \approx 7.1 \text{ MFLOP/token}
$$

**MoE layer** (`models/moe.py:DeepSeekMoE.forward`): the gate is negligible ($2 \cdot d \cdot E = 30\,720$); the executed experts dominate. Per token, 4 routed + 1 shared expert each run the same three matmuls at $I_{\text{moe}} = 384$:

$$
\text{FLOPs}_{\text{MoE}} = 5 \times 6 \cdot d \cdot I_{\text{moe}} = 5 \times 6 \times 768 \times 384 = 8\,847\,360 \approx 8.8 \text{ MFLOP/token}
$$

The striking comparison: **one MoE layer executes only 25% more FLOPs than one dense layer (8.8 vs 7.1 MFLOP/token) while storing 21× the experts.** Sparse FFNs trade memory for capacity at nearly constant compute — that is the whole point of the design.

### MLA attention (`models/mla.py:MultiHeadLatentAttention.forward`)

MLA splits into a sequence-independent projection budget and an $S$-dependent attention core.

**Projections (per token, independent of $S$):**

| Matmul | Shape | FLOPs |
|---|---|---|
| `wq`: $x \to q$ | $d \to H \cdot d_{qk} = 768 \to 864$ | $2 \cdot 768 \cdot 864 = 1\,327\,104$ |
| `wkv_a`: $x \to (kv\text{-}latent, k_{pe})$ | $d \to R + d_{rope} = 768 \to 216$ | $2 \cdot 768 \cdot 216 = 331\,776$ |
| KV materialize (SDPA path): $\text{ctx}\_{kv} \to (K_{nope}, V)$ | $R \to H \cdot (d_{nope} + d_v) = 192 \to 1344$ | $2 \cdot 192 \cdot 1344 = 516\,096$ |
| `wo`: attention out $\to x$ | $H \cdot d_v \to d = 768 \to 768$ | $2 \cdot 768 \cdot 768 = 1\,179\,648$ |
| **Sub-total** | | **3,354,624 ≈ 3.35 MFLOP** |

(The `q_nope` absorption bmm $2 \cdot H \cdot d_{nope} \cdot R = 221\,184$ appears only in the manual path; the SDPA path materialises $K_{nope}$ instead. Details in [MLA](../concepts/attention-and-precision.md).)

**Attention core (scales with $S$):**

- SDPA path: scores $2 \cdot H \cdot S \cdot d_{qk}$ + $\text{attn} \cdot V$: $2 \cdot H \cdot S \cdot d_v$:

$$
2 H S (d_{qk} + d_v) = 2 \times 12 \times S \times (72 + 64) \approx 6.68 \text{ MFLOP/token at } S = 2048
$$

- Manual path (the "manual" `attn_impl`): scores live in the $R{=}192$ latent space instead of $d_{qk}{=}72$, plus an attention-over-latents step:

$$
2 H S (R + d_{rope}) + 2 H S R + 2 H R d_v \approx 20.3 \text{ MFLOP/token at } S = 2048
$$

The manual path is ~3× costlier in the attention core — the price of never materialising $K_{nope}$. The SDPA path materialises it (extra projection FLOPs) but scores in the cheaper 72-dim space. This is the FLOP twin of the memory-bandwidth trade-off analysed in [§39](#why-mla-over-gqa--trade-off-analysis).

**MLA total: ≈ 3.35 + 6.68 ≈ 10.0 MFLOP/token** at $S{=}2048$ (SDPA path). The attention core is linear in $S$ per token, so per-sequence attention FLOPs grow $\mathcal{O}(S^2)$.

### The LM head — the largest line item

`self.head(self.norm(h))` with $V = 100\,018$:

$$
\text{FLOPs}_{\text{head}} = 2 \cdot d \cdot V = 2 \times 768 \times 100\,018 = 153\,627\,648 \approx 153.6 \text{ MFLOP/token}
$$

At this scale the untied-vocabulary projection costs more than all 18 attention layers put together ($18 \times 10.0 = 180.7$ MFLOP/token). This is the "logits bottleneck" of small models with big vocabularies — the reason `mtp` reuses the same head instead of growing a new one per depth.

### Whole-model forward

| Line item | MFLOP/token (S=2048) |
|---|---|
| 18 × MLA | 180.7 |
| 2 × dense SwiGLU | 14.2 |
| 16 × MoE (executed) | 141.6 |
| 16 × MoE gate | 0.5 |
| LM head | 153.6 |
| **Total forward** | **≈ 490 (0.49 GFLOP/token)** |

**Decode regime:** with $S_q = 1$ the attention core collapses to $2 H (d_{qk} + d_v) = 3\,264$ FLOPs/token; decode cost is the projection budget (~3.35 MFLOP/token) and — dominantly — **memory bandwidth** reading $216$ cache floats per token per layer. Decode is bandwidth-bound, not FLOP-bound; see [inference](../inference.md).

**Training budget:** $0.49 \text{ GFLOP/token} \times 8.4\text{B tokens} \approx 4.1$ EFLOPs forward; with backward ≈ 2× forward plus grad-checkpoint recompute of the forward, ~12 EFLOPs total. On an A100 (312 TFLOP/s BF16 dense) that is ~39,700 s at 100% MFU, ~27–32 h at 35–40% MFU. The config's stated target is 13–15 h at 35–40% MFU, which implies ~75% MFU — **optimistic**. Both figures are estimates; nothing has been measured yet. [G4 benchmarking](../guides/G4_benchmarking.md) covers what a real measurement must capture (activation recompute, allocator churn, the logits transient).

### Sequence-level view and the three regimes

Per-sequence forward at $S{=}2048$: $2048 \times 490.6\text{ MFLOP} \approx 1.0$ TFLOPs; 8.4B tokens is $4.1 \times 10^6$ sequences, and $4.1 \times 10^6 \times 1.0$ TFLOPs $= 4.1$ EFLOPs — consistent with the per-token budget above.

| Regime | $S_q$ | Attention core per token | Total per token | What dominates |
|---|---|---|---|---|
| Training / prefill ($S{=}2048$) | 2048 | 6.68 MFLOP | ~490 MFLOP | LM head (153.6M) + 18× attention (180.7M) |
| Prefill with cache ($S{=}2048$) | 2048 | 6.68 MFLOP | ~490 MFLOP | identical to training minus backward |
| Decode step ($S_q{=}1$) | 1 | 3.3 KFLOP | ~3.4 MFLOP | projections; attention core vanishes |

The table makes the two regimes legible: **prefill is compute-bound and quadratic in $S$; decode is memory-bound and linear in $S$** (each step reads the whole cache). At $S{=}2048$ the training attention core ($2 H S (d_{qk}+d_v)$) is only 1.4% of the per-token FLOPs — the logits bottleneck, not attention, sets this model's compute profile. That is a scale-dependent statement: at $S{=}4096$+ and larger $d$ the attention terms grow quadratically and eventually dominate.

**MFU arithmetic** (so the number is checkable): $12\text{ EFLOP} / (0.40 \times 312\text{ TFLOP/s}) = 96\,200\text{ s} \approx 27\text{ h}$. Getting to 13–15 h would require 0.75–0.85 MFU, which is above what dense BF16 transformers typically sustain on one A100 — MoE dispatch overhead and the logits GEMM (thin, memory-bound) push the other way. [G4 benchmarking](../guides/G4_benchmarking.md) explains how to measure rather than guess.

---

## SwiGLUFFN

Dense FFN used in layers 0–1 and as the building block inside each MoE expert:

```
SwiGLU(x) = W₂ · silu(W₁x) ⊙ W₃x
```

```python
class SwiGLUFFN(nn.Module):
    """SwiGLU FFN: W2(silu(W1(x)) * W3(x))."""
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

(Structure sketch: `w1: Linear(dim → inter_dim)`, `w2: Linear(inter_dim → dim)`, `w3: Linear(dim → inter_dim)`.)

**Why the gate/up split?** The gate branch $\text{silu}(W_1 x)$ is a smooth, self-gating activation in $(-\infty, \infty)$; the up branch $W_3 x$ supplies the magnitude. Multiplying them gives the model an on/off switch per hidden unit without a hard nonlinearity — the property that makes SwiGLU train more stably than GELU MLPs at equal width. The shared input $x$ means the two branches are computed with one read of the activations but two matmuls — hence the $6 d I$ FLOP count rather than $4 d I + 2 I d$ (same thing, but it explains why SwiGLU is ~1.5× the FLOPs of a 2-matrix MLP of the same width).

**Parameter count (dense layer):** `3 × dim × inter_dim = 3 × 768 × 1536 ≈ 3.5M`.

**FLOPs per token:** `≈ 6 × dim × inter_dim` (three matmuls, gate+up share input).

**Tensor-shape walkthrough** (`models/transformer.py:SwiGLUFFN.forward`, dense config, $B{=}2, S{=}128$):

| Step | Op | Shape |
|---|---|---|
| input | `x` from `ffn_norm` | $(2, 128, 768)$ |
| gate | `self.w1(x)` | $(2, 128, 1536)$ |
| up | `self.w3(x)` | $(2, 128, 1536)$ |
| gate-act | `F.silu(gate) * up` | $(2, 128, 1536)$ |
| down | `self.w2(...)` | $(2, 128, 768)$ |
| output | residual add in `TransformerBlock` | $(2, 128, 768)$ |

The intermediate width is $4\times$ the hidden width — the dense layer is where a (comparatively) large chunk of compute happens for only 1.7% of the parameters.

---

### The residual stream — central design object

Anthropic's "residual stream" framing applies directly: each layer reads from and writes to the same tensor $\mathbf{x} \in \mathbb{R}^{B \times S \times d}$. Attention and FFN are **updates** to this stream, not replacements.

$$
\mathbf{x}_{\ell+1} = \mathbf{x}_\ell + \mathrm{Attn}_\ell(\mathrm{RMSNorm}(\mathbf{x}_\ell)) + \mathrm{FFN}_\ell(\mathrm{RMSNorm}(\mathbf{x}_\ell'))
$$

At 411.6M: $d=768$, $L=18$. The stream width never changes — only the information content grows deeper in the stack.

---

## TransformerBlock

```python
class TransformerBlock(nn.Module):
    attn_norm: RMSNorm(dim)
    attn: MultiHeadLatentAttention(config, layer_id)
    ffn_norm: RMSNorm(dim)
    ffn: SwiGLUFFN | DeepSeekMoE
```

Forward (`models/transformer.py:TransformerBlock.forward`):

```python
def forward(self, x: torch.Tensor, start_pos: int = 0, mask: Optional[torch.Tensor] = None, use_cache: bool = True) -> torch.Tensor:
    x = x + self.attn(self.attn_norm(x), start_pos, mask, use_cache)
    x = x + self.ffn(self.ffn_norm(x))
    return x
```

**Note:** MoE `DeepSeekMoE.forward` only takes `x` — no `start_pos` or cache. Routing is position-agnostic (token-level, not sequence-level).

**Tensor-shape walkthrough** ($B{=}2, S{=}128$, `models/transformer.py:TransformerBlock.forward`):

| Step | Op | Shape |
|---|---|---|
| input | `x` (trunk hidden) | $(2, 128, 768)$ |
| attn branch | `attn_norm(x)` → `attn(..., start_pos, mask, use_cache)` | $(2, 128, 768)$ |
| residual add | `x = x + attn_out` | $(2, 128, 768)$ |
| ffn branch | `ffn_norm(x)` → `ffn(...)` | $(2, 128, 768)$ |
| residual add | `x = x + ffn_out` | $(2, 128, 768)$ |

**What this block does NOT pass on:** `start_pos`/`mask`/`use_cache` are consumed by `attn` only. The FFN sees pure hidden states, so a cached decode step still runs the full FFN on the single token. Parameters per block: 1,677,504 (MLA) + 1,536 (norms) + FFN (3,538,944 dense | 18,579,456 MoE: 21 experts × 884,736 + gate 15,360).

---

## Transformer Class

### Construction (`models/transformer.py:Transformer.__init__`)

```python
Transformer(config, use_checkpoint=False)
```

1. Unwrap nested config: `model_cfg = config.get("model", config)`
2. `enforce_triton_env_var(model_cfg, print)` — force-back Triton keys if env var unset
3. `nn.Embedding(vocab_size, dim)` — init `N(0, 0.006²)`
4. `ModuleList` of 18 `TransformerBlock`s
5. Final `RMSNorm(dim)`
6. `Linear(dim, vocab_size)` — LM head
7. Optional weight tying: `head.weight = embed.weight`

### `moe_layers()` (`models/transformer.py:Transformer.moe_layers`)

Generator yielding `DeepSeekMoE` instances from MoE layers. Used by `Pretrainer` for bias updates and balance metrics.

### `reset_cache()` (`models/transformer.py:Transformer.reset_cache`)

Calls `attn.reset_cache()` on every MLA layer. **Required** before each new generation session.

---

## Forward Contracts

### `forward(tokens, start_pos=0, use_cache=True) → (B, S, V)` (`models/transformer.py:Transformer.forward`)

Primary interface. Returns vocabulary logits.

- Casts non-Long tokens to `int64` at boundary (uint32 shards).
- Builds causal mask when `seqlen > 1`; `None` for single-token decode.
- Runs all layers via `_run_layers`.
- Returns `head(norm(h))`.

**Training:** `use_cache=False` (set in `train_step`). No KV cache writes.

**Inference prefill:** `use_cache=True`, `start_pos=0`, full prompt length.

**Inference decode:** `use_cache=True`, `start_pos=prompt_len+step`, `seqlen=1`.

### `forward_with_hidden(tokens, start_pos=0, use_cache=False) → (logits, h)` (`models/transformer.py:Transformer.forward_with_hidden`)

Returns `(head(norm(h)), h)` where `h` is the **pre-norm** trunk hidden.

- Used by `MultiTokenPrediction` to feed MTP heads.
- `use_cache=False` during training.
- `use_cache=True` in `SpeculativeDecoder.generate_step` (cache grows).

### Shape reference (411.6M, B=2, S=128)

| Tensor | Shape |
|---|---|
| `tokens` | (2, 128) |
| `h` (embedding) | (2, 128, 768) |
| per-layer `x` | (2, 128, 768) |
| `logits` | (2, 128, 100018) |

### `forward_with_hidden` — MTP contract

```python
h = embed(tokens)
h = _run_layers(h, start_pos, mask, use_cache=False)
return head(norm(h)), h   # logits use normed h; MTP consumes raw h
```

MTP blocks apply **their own** `RMSNorm` before fusion — the returned `h` is the pre-final-norm trunk state, matching DeepSeek-V3's "main hidden before head norm" convention in this repo.

### The call-site contract at a glance

Every entry point that reaches `forward` picks a `(use_cache, start_pos, mask)` triple; the correctness of the whole repo rides on these three:

| Call site | `use_cache` | `start_pos` | `mask` |
|---|---|---|---|
| `Pretrainer.train_step` (via `MultiTokenPrediction`) | False | 0 | full causal ($S_q = S_{kv} = S$) |
| `generate` prefill | True | 0 | full causal ($S_q = S_{kv} = S$) |
| `generate` decode step | True | `prompt_len + step` | `None` ($S_q = 1$) |
| `SpeculativeDecoder.generate_step` | True | current cache length | `None` ($S_q = 1$) |

The pattern: **training never touches the cache, inference always does, and a multi-token cached forward must pass both `start_pos` and `kv_len=end_pos` so the mask is global.** The one place this can go wrong is a hand-rolled cached prefill that passes `start_pos=0` with a warm cache — `_build_causal_mask` would then build a local mask and the chunk could attend its own future (the bug class fixed in the 2026-08-04 session).

---

## Causal Mask Cache

### `_build_causal_mask(seqlen, kv_len, start_pos, device)` (`models/transformer.py:Transformer._build_causal_mask`)

Verbatim from source (the additive mask is `(1,1,S_q,S_kv)`, causal by **global** position):

```python
key = (seqlen, kv_len, start_pos, device)
if self._mask_cache is None or key != self._mask_key:
    q = torch.arange(seqlen, device=device)[:, None] + start_pos
    k = torch.arange(kv_len, device=device)[None, :]
    mask = torch.where(q >= k, torch.zeros((), device=device), torch.full((), float("-inf"), device=device))
    self._mask_cache = mask.unsqueeze(0).unsqueeze(0)
    self._mask_key = key
return self._mask_cache
```

Cached in `_mask_cache` keyed by `(seqlen, kv_len, start_pos, device)`. `kv_len` is `end_pos` when a KV cache spans the past, `seqlen` otherwise, so a cached mid-sequence prefill cannot attend its own future.

**Skipped when `seqlen == 1`:** During autoregressive decode, each step passes a single token. A causal mask is unnecessary (one query, many keys from cache). Skipping saves allocation and SDPA overhead.

### When is the mask used?

| Call context | seqlen | kv_len | mask |
|---|---|---|---|
| Training / prefill | 2048 | 2048 (start_pos=0) | full causal |
| Cached mid-seq prefill | chunk | end_pos | causal vs the whole context |
| Decode | 1 | — | `None` (no self-attention mask needed) |

Changing `max_seq_len` in config without clearing `_mask_cache` is safe — cache miss triggers rebuild when any key part differs.

**Why the mask is by global position:** with a cache, query positions are `arange(seqlen) + start_pos`; keys are `arange(kv_len)`. A chunk prefilled at `start_pos = 128` must not attend keys at positions ≥ its own global position — a *local* causal mask (`q >= k` on the chunk index) would wrongly allow it. This is the contract fixed in the 2026-08-04 session: chunked prefill previously crashed (SDPA) or leaked future tokens (Triton) under a local mask.

### Mask algebra, concretely

The mask is the predicate "query may attend key", broadcast to 4 dims:

$$
M[q_{\text{idx}}, k_{\text{idx}}] = \begin{cases} 0 & \text{if } (\text{start\_pos} + q_{\text{idx}}) \ge k_{\text{idx}} \\ -\infty & \text{otherwise} \end{cases}
$$

**Training ($S{=}4$, no cache):** $q, k \in \{0,1,2,3\}$, allowed iff $q \ge k$ — the familiar lower triangle:

$$
M = \begin{pmatrix} 0 & -\infty & -\infty & -\infty \\ 0 & 0 & -\infty & -\infty \\ 0 & 0 & 0 & -\infty \\ 0 & 0 & 0 & 0 \end{pmatrix}
$$

**Cached chunk prefill at $\text{start\_pos}=128$, chunk $S_c{=}4$, `kv_len=end_pos=132`:** query rows are global 128–131, keys 0–131. Row $q{=}129$ is allowed keys $k \le 129$ (130 allowed cells, 2 masked). The same chunk under a *local* mask (rows/cols 0–3) would block keys 128–129 — past context the chunk is entitled to — and would allow key 131+ — its own future. Hence `kv_len` and `start_pos` must travel together through `forward`:

```python
kv_len = end_pos if use_cache else seqlen
mask = self._build_causal_mask(seqlen, kv_len, start_pos if use_cache else 0, tokens.device)
```

**Decode ($S_q{=}1$):** one query at global position $g$; every key is $< g$ by construction (the cache only ever contains the past). The mask would be all-zeros — pointless to build. `forward` short-circuits to `mask=None` when `seqlen == 1`.

**Why the cache is not masked at all:** `ctx_kv = self.kv_cache[:bsz, :end_pos]` (`models/mla.py:MultiHeadLatentAttention.forward`) already slices to the *written* prefix; uninitialised tail slots are structurally unreachable. The mask only ever guards query↔key pairs that both exist.

**Cache-bound guard:** `MultiHeadLatentAttention.forward` raises `RuntimeError(f"Layer {layer_idx}: end_pos ... exceeds max_seq_len ...")` if `start_pos + seqlen > max_seq_len` — the last line of defence before a cache write would go out of bounds. `generate` avoids it by breaking when `output.size(1) >= self.max_seq_len`.

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

## Generation API — `generate()` and `_sample()` Walkthrough

> **60-second summary.** `generate` is the inference entry point: it prefills the prompt through the whole stack with a causal mask, then loops one token at a time, each step sampling from the last row of logits and appending to the cache. It is the only production path that sets `use_cache=True` on every layer. This section walks the actual source line by line.

### `generate(input_ids, max_new_tokens=512, temperature=1.0, top_p=0.9, top_k=0, eos_token_id=None)` (`models/transformer.py:Transformer.generate`)

Verbatim from source:

```python
@torch.inference_mode()
def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 512, temperature: float = 1.0,
             top_p: float = 0.9, top_k: int = 0, eos_token_id: Optional[int] = None) -> torch.Tensor:
    """Autoregressive generation with KV-cache, top-p and top-k sampling."""
    if temperature < 0.0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")
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

### Step-by-step

1. **Validation:** negative temperature raises `ValueError` before any state is touched.
2. **Mode save/restore:** `was_training` records the calling mode; the model is forced to `eval()` and restored afterwards. `generate` is safe to call from inside a training context.
3. **`reset_cache()`** (`models/transformer.py:Transformer.reset_cache`) — mandatory. Without it, a second `generate()` appends to stale cache tensors and logits reference wrong positions.
4. **Prefill:** `self.forward(output, start_pos=0, use_cache=True)` runs the *entire prompt* in one pass with the full causal mask; every layer's MLA writes its `kv_cache`/`pe_cache` slots `[0:prompt_len)`. Logits for all prompt positions are produced; only the last row is kept (`prefill_logits[:, -1, :]`).
5. **Decode loop invariant:** at iteration `step`, `next_logits` holds the logits of the most recent token, and the KV cache spans positions `[0, prompt_len + step)`. Each iteration:
   - samples `next_token` via `_sample`; `torch.cat` appends it to the output (this is the only place output grows);
   - updates the `finished` mask row-wise on EOS; breaks only when **all** rows finished (`finished.all()`);
   - breaks at the hard context limit `max_seq_len`;
   - runs the one-token forward at `start_pos = prompt_len + step` — note the cache write lands exactly at `end_pos = prompt_len + step + 1`.
6. **Return:** the full `input_ids + generated` tensor; the caller detokenises and strips the prompt.

### `_sample(logits, temperature, top_p, top_k)` (`models/transformer.py:Transformer._sample`)

Verbatim from source:

```python
if temperature == 0.0:
    return logits.argmax(dim=-1, keepdim=True)
logits = logits / temperature
if top_k > 0:
    kth_vals = logits.topk(min(top_k, logits.size(-1)), dim=-1)[0][:, -1:]
    logits = logits.masked_fill(logits < kth_vals, float("-inf"))
probs = torch.softmax(logits, dim=-1)
if top_p < 1.0:
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cumulative = sorted_probs.cumsum(dim=-1)
    remove = (cumulative - sorted_probs) > top_p
    sorted_probs = sorted_probs.masked_fill(remove, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)
    next_token = sorted_idx.gather(-1, torch.multinomial(sorted_probs, num_samples=1))
else:
    next_token = torch.multinomial(probs, num_samples=1)
return next_token
```

| Setting | Behaviour |
|---|---|
| `temperature=0` | Greedy argmax (short-circuits before any scaling) |
| `temperature>0` | Scale logits, then sample |
| `top_k>0` | Mask logits below k-th largest |
| `top_p<1` | Nucleus sampling on sorted probabilities, renormalised |

**Sampling theory** (temperature, top-k, top-p semantics and why the ordering matters) is derived in [inference](../inference.md).

### Batch generation caveat

All rows in batch `B>1` share the same loop length; EOS only stops a row logically — the implementation checks `finished.all()` but still runs decode steps until all rows finish or `max_new_tokens` is exhausted.

### Cache-position table for a prompt of length 4

| Step | `start_pos` | Token fed | Cache write slot | Output length |
|---|---|---|---|---|
| prefill | 0 | `[t₀ t₁ t₂ t₃]` | 0..3 | 4 |
| 0 | 4 | `t₄` | 4 | 5 |
| 1 | 5 | `t₅` | 5 | 6 |
| … | … | … | … | … |

### A worked decode, end to end

Prompt `[3, 7, 9, 2]` (length 4), greedy (`temperature=0`), `max_new_tokens=3`, no EOS:

1. **Prefill** — `forward([3,7,9,2], start_pos=0, use_cache=True)`:
   - causal mask $(1,1,4,4)$; every MLA layer writes `kv_cache[0:4]` and `pe_cache[0:4]`.
   - `next_logits = prefill_logits[:, -1, :]` → the logits row for position 3 (predicting $t_4$).
2. **Step 0** — `_sample` (greedy) → `t₄ = 5`. Output `[3,7,9,2,5]`.
   - `forward([5], start_pos=4, use_cache=True)`: mask `None`; RoPE reads `freqs_cis[4:5]`; each layer writes cache slot 4; logits at position 4 predict $t_5$.
3. **Step 1** — `t₅ = 11`; `forward([11], start_pos=5, ...)`; output `[3,7,9,2,5,11]`.
4. **Step 2** — `t₆ = 42`; output `[3,7,9,2,5,11,42]`; loop ends after `max_new_tokens`.

Final cache state: `kv_cache[0:7]` filled, `output = [3,7,9,2,5,11,42]`. The one thing to notice: **the cache offset and the RoPE offset are the same integer** — `start_pos` drives both the cache write (`kv_cache[:bsz, start_pos:end_pos]`) and the frequency slice (`freqs_cis[start_pos:start_pos+seqlen]` in `models/mla.py:MultiHeadLatentAttention._apply_rope`). If they ever diverge, positions silently shift.

### Pitfalls

- **Negative temperature raises** — the `temperature < 0.0` check fires before any cache mutation, so a bad call cannot corrupt state.
- **`temperature=0` ignores `top_k`/`top_p`** — the greedy short-circuit returns before scaling. Setting `temperature=0, top_p=0.9` is greedy, not nucleus.
- **The `max_seq_len` guard is checked *after* the append** — `output.size(1) >= self.max_seq_len` breaks the loop, but the token that exceeded the limit is already in `output`. Callers that need a hard context cap should trim.
- **EOS is all-batch, not per-row** — with `B>1`, decoding stops only when *every* row has hit EOS (or the token budget). Long tails pay full price.
- **Cache lifetime** — `generate` calls `reset_cache()` itself, but `forward(..., use_cache=True)` outside `generate` (e.g. `SpeculativeDecoder.generate_step`) does not; call `reset_cache()` between requests or stale positions will be attended.
- **BF16 logits** — `_sample` divides by temperature in the logits' dtype; for very small temperatures the bf16 range can clip. The softmax itself is computed in fp32 internally by PyTorch.

---

## Gradient Checkpointing

### `_run_layers` (`models/transformer.py:Transformer._run_layers`)

Verbatim from source:

```python
def _run_layers(self, h, start_pos, mask, use_cache):
    for layer in self.layers:
        if self.use_checkpoint and self.training:
            h = torch.utils.checkpoint.checkpoint(
                layer, h, start_pos, mask, use_cache, use_reentrant=False,
            )
        else:
            h = layer(h, start_pos, mask, use_cache)
    return h
```

```python
Transformer(config, use_checkpoint=True)
```

**Design choices:**

- **All 18 layers are checkpointed** (not every Nth like GPT-OSS-Lite's every-3rd). DeepSeek-V3 checkpoints every layer for maximum memory savings.
- **`use_reentrant=False`:** PyTorch 2.x recommended default, better `torch.compile` compatibility.
- **Only during training:** the `self.training` check disables checkpointing at inference (no backward pass needed).
- **The whole block is the checkpointed unit:** `layer, h, start_pos, mask, use_cache` are the function + inputs, so MLA cache writes inside a checkpointed forward are recomputed during backward — fine, because training never writes the cache (`use_cache=False`).

**Memory impact (derived, $B{=}8, S{=}2048$):** without checkpointing, every layer holds its output `h` (25.2 MB each, 453 MB over 18 layers) plus its intermediates until backward; with checkpointing, only the region input `h` (25.2 MB) is retained and each layer's forward is recomputed inside the backward pass. Trade-off: roughly 3× less activation memory for ~33% more total FLOPs (backward grows from ~2× to ~3× the forward). The 411.6M config sets `grad_checkpoint: true`; the 1650 smoke config sets `false` (the model fits without it).

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

`Pretrainer` passes the full YAML dict; `Transformer` unwraps internally. MLA/MoE receive the unwrapped flat dict via `TransformerBlock`. The canonical config is documented key-by-key in [R1 config schema](../references/R1_config_schema.md) and `training/pretrain.py:TrainingConfig`.

---

## Tensor-Shape Walkthrough per Module

> **60-second summary.** Three modules are worth a full shape trace because each reshapes in a non-obvious way: MLA (split/recombine around the latent), DeepSeekMoE (flatten → route → sort → gather back), and the gate (sigmoid + bias before top-k). Trace them once and every shape error in the repo becomes legible.

### MultiHeadLatentAttention (`models/mla.py:MultiHeadLatentAttention.forward`)

Config: $d{=}768$, $H{=}12$, $R{=}192$, $d_{nope}{=}48$, $d_{rope}{=}24$, $d_v{=}64$, $d_{qk}{=}72$, `q_lora_rank=0`. Training mode ($B{=}2, S{=}128$, `use_cache=False`), SDPA path.

| # | Step | Op | Shape |
|---|---|---|---|
| 1 | input | `x` from `attn_norm` | $(2, 128, 768)$ |
| 2 | query | `self.wq(x)` ($q_{lora}=0$, no `wq_a`/`q_norm`) | $(2, 128, 864)$ |
| 3 | view | `q.view(bsz, seqlen, H, d_qk)` | $(2, 128, 12, 72)$ |
| 4 | split | `q.split([48, 24], dim=-1)` → `q_nope`, `q_pe` | $(2, 128, 12, 48)$ · $(2, 128, 12, 24)$ |
| 5 | rope | `_apply_rope(q_pe, start_pos, seqlen)` | $(2, 128, 12, 24)$ |
| 6 | KV down | `self.wkv_a(x)` | $(2, 128, 216)$ |
| 7 | split | `kv_a.split([192, 24], dim=-1)` → `kv_latent`, `k_pe_raw` | $(2, 128, 192)$ · $(2, 128, 24)$ |
| 8 | norm | `kv_norm(kv_latent)` | $(2, 128, 192)$ |
| 9 | rope | `_apply_rope(k_pe_raw.unsqueeze(2), ...).squeeze(2)` → `k_pe` | $(2, 128, 24)$ |
| 10 | no cache | `ctx_kv = kv_normed`, `ctx_pe = k_pe` | $(2, 128, 192)$ · $(2, 128, 24)$ |
| 11 | KV up | `wkv_b.weight` viewed as $(H, d_{nope}{+}d_v, R)$ → split into `wkv_b_k` $(12,48,192)$, `wkv_b_v` $(12,64,192)$ | — |
| 12 | materialise | `torch.bmm(ctx_kv_bmm, wkv_b_kvᵀ)` → split `K_nope`, `V` | $(2, 12, 128, 48)$ · $(2, 12, 128, 64)$ |
| 13 | scores | SDPA over `cat([Q_nope, Q_rope])`, `cat([K_nope, K_rope])` with `scale=softmax_scale` | $(2, 12, 128, 128)$ (implicit) |
| 14 | output | `attn ⊙ V` → `transpose` → `flatten(2)` → `wo` | $(2, 128, 768)$ |

Cache variant (`use_cache=True`, decode): steps 10 becomes writes `self.kv_cache[:bsz, start_pos:end_pos] = kv_normed.detach()` and `self.pe_cache[:bsz, start_pos:end_pos] = k_pe.detach()` — cache tensors are $(B, S_{max}, 192)$ and $(B, S_{max}, 24)$ per layer (see `models/mla.py:MultiHeadLatentAttention._ensure_cache`); `ctx_kv`/`ctx_pe` become the cache slices `[:bsz, :end_pos]`.

The **manual path** (non-SDPA `attn_impl`) replaces steps 12–13 with the absorption trick: `q_nope` is projected through `wkv_b_k` first (`models/mla.py:MultiHeadLatentAttention._per_batch_bmm`), scores are computed directly against the latent `ctx_kv` in $R{=}192$ space, and the attention output is up-projected through `wkv_b_v` after the softmax. Full algebra in [MLA](../concepts/attention-and-precision.md).

### DeepSeekMoE (`models/moe.py:DeepSeekMoE.forward`)

Training mode ($B{=}2, S{=}128 \rightarrow T{=}256$ tokens, $E{=}20$, $k{=}4$, 1 shared, $I{=}384$):

| # | Step | Op | Shape |
|---|---|---|---|
| 1 | input | `x` from `ffn_norm` | $(2, 128, 768)$ |
| 2 | flatten | `x.view(-1, dim)` | $(256, 768)$ |
| 3 | gate scores | `F.linear(x, gate.weight).sigmoid()` | $(256, 20)$ |
| 4 | bias | `scores + gate.bias` (a **buffer**, not a Parameter) | $(256, 20)$ |
| 5 | top-k | `biased.topk(4, dim=-1)` → `indices`, re-gathered `weights` | $(256, 4)$ · $(256, 4)$ |
| 6 | normalise | `weights / sum(weights) * route_scale` | $(256, 4)$ |
| 7 | snapshot | `_last_weights`, `_last_indices` (detached, for bias update) | $(256, 4)$ each |
| 8 | stack | 21 experts × 3 weights → `_stacked_w{1,2,3}` | $(21, 384, 768)$ · $(21, 768, 384)$ · $(21, 384, 768)$ |
| 9 | sort | argsort expert ids; `sorted_token_ids` (T·k), `sorted_weights`, per-expert `expert_offsets` | $(1024,)$ vectors |
| 10 | per-expert GEMM | for each routed expert with `cnt>0`: gate/up/down on `chunk_tokens` | $(c_e, 384) \rightarrow (c_e, 768)$ |
| 11 | scatter-add | `y_routed.index_add(0, chunk_tokens, out * weights)` | $(256, 768)$ |
| 12 | shared | `_shared_forward(flat)` — batched bmm over 1 shared expert | $(256, 768)$ |
| 13 | combine | `y = y_routed + shared`, `y.view(shape)` | $(2, 128, 768)$ |

Two invariants worth memorising from this trace:

- **Routing happens on the flattened token axis.** Expert assignment is per-token, never per-sequence — the batch dimension is gone by step 2 and restored only at step 13. Position information does not influence routing.
- **The gate's `bias` buffer never receives gradients.** `update_gate_bias` (`models/moe.py:DeepSeekMoE.update_gate_bias`) moves counts into `AuxLossFreeGate.update_bias` (`models/moe.py:AuxLossFreeGate.update_bias`), which bumps the buffer up/down by `bias_update_speed` under `torch.no_grad()`. Load balancing is a control loop, not an optimiser term. The Triton grouped path (`models/moe_triton.py`) keeps the same sort/scatter contract; see [triton kernels](../concepts/kernels-and-ops.md).

### MTPModule — the shape trace that bites (`models/mtp.py:MTPModule.forward`)

MTP is *not* another trunk layer; it is a depth-1 side branch over length-aligned slices. Training, $B{=}2$, $S{=}128$, `mtp_depth=1`:

| # | Step | Op | Shape |
|---|---|---|---|
| 1 | trunk | `main_model.forward_with_hidden(tokens)` | `main_logits` $(2,128,V)$; `prev_h` $(2,128,768)$ |
| 2 | slice | `usable = seq_len - d - 2 = 125`; `h_in = prev_h[:, :125]` | $(2,125,768)$ |
| 3 | embed | `emb_in = embed(tokens[:, 1:126])` | $(2,125,768)$ |
| 4 | target | `tgt = tokens[:, 2:127]` (label for $t{+}2$) | $(2,125)$ |
| 5 | fusion | `proj(cat([norm_h(h_in), norm_e(emb_in)], -1))` | $(2,125,1536) \rightarrow (2,125,768)$ |
| 6 | attn | `nn.MultiheadAttention` (plain MHA, not MLA!) with its own causal mask | $(2,125,768)$ |
| 7 | FFN | SwiGLU, `inter_dim=1536` | $(2,125,768)$ |
| 8 | head | `norm(h) → output_head(h_norm)` (shared with trunk head) | $(2,125,V)$ |

The alignment algebra (`usable = S − d − 2`, inputs at offsets $d{+}1$, targets at $d{+}2$) is the length contract between the trunk and the side branch — full derivation in [MTP](../concepts/moe-mtp.md). Note step 6 is regular multi-head attention, not MLA: the MTP block is small (1.7% of params) and its cache-free training path does not need MLA's bandwidth trick.

### Shape-error gallery (every one of these has happened)

| Symptom | Root cause | Fix |
|---|---|---|
| `RuntimeError: end_pos 2050 exceeds max_seq_len 2048` | `start_pos + seqlen` beyond the cache bound in `models/mla.py:MultiHeadLatentAttention.forward` | cap decode at `max_seq_len` (or raise the config) |
| SDPA shape mismatch at `mask.expand(bsz, h, seqlen_q, -1)` | hand-built mask with wrong dims instead of `_build_causal_mask`'s $(1,1,S_q,S_kv)$ | always go through `models/transformer.py:Transformer._build_causal_mask` |
| MoE output looks right for the first tokens, wrong later | cache of `_stacked_w*` surviving an `optimizer.step()` — experts frozen at init | the re-stack-every-forward in `DeepSeekMoE.forward` is deliberate; never "optimise" it into a cached stack |
| `bias` not moving despite skewed routing | `update_gate_bias` returns early when `_last_indices is None` (forward never ran) | call a forward before updating |
| `q.view(bsz, seqlen, H, d_qk)` mismatch | changed `qk_nope_head_dim` without changing `qk_rope_head_dim` — the sum must equal the `wq` output width | keep $d_{nope} + d_{rope} = d_{qk}$ |
| KV cache silently reallocated mid-run | dtype/device change (fp32 → bf16 move) triggers `_ensure_cache` reallocation | construct the model on the final device/dtype before caching |
| MTP `Shape mismatch: (2,125,768) vs (2,124,768)` | hand-sliced `h_in`/`emb_in` with different lengths | use `MultiTokenPrediction.forward`'s slicing, never DIY offsets |

---

## Comparison with GPT-2 / LLaMA

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

Called at `Transformer.__init__` (`models/_triton_dispatch.py:enforce_triton_env_var`) and mirrored in `Pretrainer`. Ensures default-config runs never silently enable Triton (AGENTS.md hard rule). Tests in `test_force_back.py` lock this behaviour.

---

## Extension Points

### Safe vs Unsafe

| Change | Safe? | Notes |
|---|---|---|
| `n_dense_layers` | Yes | Test with `small_cfg` first |
| `weight_tying: false` | Yes | Doubles embedding+head params; update μP count |
| `n_layers` | Yes | Watch VRAM linear in depth |
| Remove `enforce_triton_env_var` | **No** | Violates AGENTS hard rule |
| Post-norm residuals | **No** | Untested; breaks training stability |
| `use_cache=True` in training | **No** | Breaks gradients / leaks stale state |

### Where to change things

| Goal | Where to change |
|---|---|
| Add layer type | `TransformerBlock.__init__` branch on `layer_id` |
| New position encoding | `models/mla.py` RoPE section |
| Batched speculative decode | `inference/speculative.py` (not implemented) |
| Multi-GPU | New `training/pretrain_distributed.py` (out of scope) |
| Different vocab | YAML `vocab_size` + re-tokenize data |

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

---

## Appendix A — Tensor shape trace

Single forward, `B=1, S=4`, 411.6M:

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

**Q: Can I make all layers MoE?** Set `n_dense_layers=0`. Untested at 411.6M scale but architecturally valid.

**Q: Does `generate` support batch size > 1?** Yes, but all sequences in the batch share the same generation length (no per-sequence early stopping except EOS).

**Q: Why do all `Linear` layers have `bias=False`?** The model is deliberately bias-free, matching DeepSeek-V3: RMSNorm absorbs the per-feature offset, and every projection in `mla.py`, `moe.py`, and `transformer.py` drops the bias. The only bias-like object is the MoE gate's `bias` buffer — which is a load-balancing control input, not a learned weight, and lives outside the parameter count (see [§6](#parameter-budget)).

**Q: What happens if I set `rope_factor > 1`?** RoPE frequencies are recomputed for a longer context (`_extend_rope`), and `mscale` kicks in: `mscale = 0.1 * mscale_raw * log(rope_factor) + 1.0`; when `max_seq_len > 4096` the softmax scale becomes `(qk_head_dim ** -0.5) * (mscale ** 2)`. This is the YaRN contract — see [MLA](../concepts/attention-and-precision.md).

**Q: When does the `manual` attention path run?** Only when `attn_impl: "manual"` in the config. It is the pedagogical absorption-trick path (scores in the 192-dim latent space); the default `sdpa` path materialises `K_nope` and scores in 72 dims. Both are exercised by tests; the Triton path is a third option gated by `ENABLE_TRITON_KERNELS=1`.

**Q: What exactly does `count_parameters` count when MTP is on?** The MTP wrapper (`MultiTokenPrediction`) shares the trunk's head and embedding, so its *added* parameters are only the per-depth `MTPModule` blocks (~7.1M for depth 1). `count_parameters` over the wrapped model deduplicates across the shared tensors the same way — that is how the 418.7M figure stays honest.

**Q: Why does `_log_per_component_params` show an `other` bucket?** The 768-parameter final `norm.weight` (see the logger-quirk pitfall in [§6](#parameter-budget)). The total is unaffected.

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
| `kv_lora_rank` (R) | Width of the shared KV latent (192) |
| `d_qk`, `d_nope`, `d_rope`, `d_v` | 72 / 48 / 24 / 64 head dims |

---

## Load-Bearing Invariants (Part II)

1. **Weight tying** — `head.weight = embed.weight`. Removing breaks generation.
2. **Mask skip at seqlen=1** — required for efficient decode.
3. **`use_cache=False` in training** — prevents detached cache pollution.
4. **`reset_cache()` before generate** — prevents cross-request cache bleed.
5. **Config unwrap** — always `config.get("model", config)` at Transformer boundary.
6. **`enforce_triton_env_var`** — called at construction, not per-forward.

---


- `models/transformer.py` — authoritative source


---

## Design Rationale and Transformer Block Details

> **Design Rationale & Architecture Details.** This section consolidates unique architectural design choices and transformer block topology details. Cross-portfolio comparison lives in [Portfolio Comparison](../concepts/foundations.md).

---

## Design Rationale

> Why does DeepSeek-V3-Lite exist? Why MLA over GQA? Why aux-loss-free MoE? Why dense + MoE topology? This covers the reasoning behind every major design decision.

---

### Chinchilla Scaling — Step-by-Step Derivation

The Chinchilla paper establishes ~20 tokens per parameter as optimal compute efficiency. For 411.6M deduplicated parameters:

$$
T_{\text{optimal}} = 20 \times 411.6\text{M} \approx 8.2\text{B tokens}
$$

The config rounds this to **8.4B tokens** (the figure quoted everywhere in this repo). At batch=8, grad_accum=4, seq=2048:

$$
\text{tokens/step} = 8 \times 4 \times 2048 = 65536
$$

$$
\text{total\_steps} = \frac{8.4\text{B}}{65536} \approx 128000 \text{ optimizer steps}
$$

With grad_accum=4, this is 512,000 micro-steps. At 35–40% MFU on A100, the config targets **~30–45 hours** — an estimate; see [§18](#flop-accounting-per-layer-type) for the FLOP-derived cross-check.

---

### Memory Breakdown — Why 411.6M Fits on One A100

The constraint is single A100 80GB training. The 411.6M model fits comfortably (all figures estimates — no GPU run has executed):

| Component | Memory |
|---|---|
| Parameters (BF16) | ~0.82 GB |
| Optimizer states (FP32 AdamW) | ~4.9 GB |
| Activations (with grad-ckpt, PaLM 24×) | ~10.9 GB |
| KV cache (MLA: 192+24 per token, inference only) | ~0.13 GB |
| Overhead (env-dependent) | 2 – 13.7 GB |
| **Total estimate** | **~18 – 30.5 GB** |

Enormous headroom on 80GB. The 411.6M size was chosen for Chinchilla optimality at 8.4B tokens, not memory constraints. The estimator formula lives in `utils/memory.py:estimate_model_memory_gb`; the training-mode breakdown is §7 of Part I.

---

### Why MLA over GQA — Trade-off Analysis

**Why not GQA?** GQA (Grouped-Query Attention) shares KV heads across query heads, giving a constant-factor KV reduction (2× with 8 groups, 4× with 4 groups). But:

- **Fixed compression ratio:** GQA's reduction is limited by the group ratio. You can't compress beyond `n_kv_heads=1` (MQA).
- **Quality degradation:** MQA (1 KV head) shows measurable perplexity degradation. GQA with 4–8 groups shows slight degradation.
- **No absorption trick:** GQA still materializes full K/V at every decode step. The memory bandwidth savings are limited to the cache size, not the decode computation.

**Why MLA?** MLA achieves **~7× KV-cache reduction** with **no quality loss** (matches MHA perplexity). The key innovations:

1. **Low-rank compression:** Instead of caching `n_heads × d_head` floats per token for K *and* V, MLA caches `kv_lora_rank + qk_rope_head_dim = 192 + 24 = 216` floats — a ~7× reduction vs MHA's `2 × 12 × 64 = 1536` floats (K and V at `d_head=64`).
2. **Absorption trick:** At inference, the K up-projection is algebraically absorbed into the Q projection, so full K is never materialized. The attention score is computed directly in the 192-dim latent space.
3. **Decoupled RoPE:** A separate 24-dim RoPE path preserves position encoding without breaking the absorption algebra.

**The trade-off:** MLA trades **FLOPs for memory bandwidth**:

- Attention computation is more expensive than MHA if scored in the latent space (manual path: scores live in $R{=}192$ vs MHA's $d_{head}{=}64$ — ~3× per score; see [§18](#flop-accounting-per-layer-type)). The SDPA path instead materialises $K_{nope}$ and scores in the 72-dim space, spending the extra FLOPs on projections rather than scores.
- Memory reads are ~7× cheaper (reading 216 floats vs 1536 per token for K+V at `d_head=64`).

Since decode is overwhelmingly **memory-bandwidth-bound** at long context, MLA wins on throughput. This is a fundamentally different trade-off from GQA (which reduces both FLOPs and bandwidth proportionally).

**The absorption trick in one line:** the attention score needs $q^\top k$; with $k = W_{b,k} \cdot \text{kv\_latent}$ (the latent up-projected), the score factorises:

$$
q^\top k = q^\top (W_{b,k}\, c) = (W_{b,k}^\top q)^\top c = \tilde{q}^\top c
$$

so instead of materialising $k$ for every key (a $(S \times d_{nope})$ tensor per layer per step) the model pre-computes $\tilde q = W_{b,k}^\top q$ per query — $H \cdot d_{nope} \times R$ FLOPs once, not per key — and scores queries directly against the 192-dim latent $c$. That is exactly the `q_nope_proj = torch.bmm(q_nope_h, wkv_b_k)` step in `models/mla.py:MultiHeadLatentAttention.forward` (manual path). The SDPA path makes the mirror-image choice: materialise $K_{nope}$ once per forward and let FlashAttention-style kernels score in 72 dims. Both are correct; they sit at different points on the FLOPs↔bandwidth curve ([§18](#flop-accounting-per-layer-type) quantifies the 3× attention-core gap). The full two-direction derivation is [§4 of the MLA chapter](../concepts/attention-and-precision.md).

---

### Why Aux-Loss-Free MoE

**The standard approach and its problem:** Standard MoE (Switch Transformer, GShard) adds an auxiliary load-balancing loss to the training objective:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \alpha \cdot \mathcal{L}_{\text{aux}}
$$

The aux loss pushes the router toward uniform expert utilization. But it **contaminates the task gradient** — the gradient from the aux loss affects the router weights, which affects routing decisions, which affects the task loss. This coupling can degrade task performance.

**DeepSeek's solution:** Instead of an aux loss, DeepSeek-V3 uses a **per-expert bias** on the gate logits:

$$
\text{biased\_score}_e = \text{sigmoid}(\text{logit}_e) + \text{bias}_e
$$

The bias is **not a learned parameter** — it's a buffer updated out-of-band:

- If expert $e$ receives more than its fair share of tokens: `bias[e] -= speed` (make it less attractive).
- If expert $e$ receives less than its fair share: `bias[e] += speed` (make it more attractive).

This achieves load balancing **without any gradient on the bias** and **without any aux loss term**. The task gradient is pure — it only optimizes task performance, not load balancing. Implemented in `models/moe.py:AuxLossFreeGate.forward` (routing) and `models/moe.py:AuxLossFreeGate.update_bias` (the control loop); the `bias` buffer is asserted not-a-Parameter by `test_bias_not_in_parameters`.

**Reading the control loop as a controller** (`models/moe.py:AuxLossFreeGate.update_bias`): after each forward, `DeepSeekMoE.update_gate_bias` counts how many tokens each expert received (`torch.bincount` over `_last_indices`), and the gate moves the bias against the deviation:

$$
b_e \leftarrow b_e - \eta_b \cdot \mathbf{1}\left[c_e > \bar{c}(1 + \alpha)\right] + \eta_b \cdot \mathbf{1}\left[c_e < \bar{c}(1 - \alpha)\right], \qquad \bar{c} = \frac{1}{E}\sum_e c_e
$$

with $\eta_b = 0.001$ (`bias_update_speed`) and deadband $\alpha = 0.10$ (`bias_upper_threshold`/`bias_lower_threshold`). Three design properties fall out:

1. **Proportional control with a deadband** — experts near the mean (within ±10%) are untouched, so the loop does not oscillate on sampling noise.
2. **The bias acts on selection, the weights act on values** — `biased = scores + bias` chooses the top-4; the softmax-normalised `weights` (from *unbiased* sigmoid scores) set the mixture proportions. Bias therefore cannot distort the actual expert outputs, only which experts get chosen.
3. **Gradient purity** — because `bias` is a buffer updated under `torch.no_grad()`, the task loss has no path through the balancing term; there is no auxiliary-loss gradient contamination. The `route_scale` factor (default 1.0) is the only other knob on the routed output magnitude.

The config updates bias **every step** (`bias_update_every: 1`) — the fastest safe cadence, since each step's counts are a fresh sample. Slowing it (`bias_update_every: k`) is the right lever if the router starts chattering between two experts.

---

### Why Top-4 of 20 Experts

- **20 experts** (finer granularity than GPT-OSS's 8) gives more specialization at the same parameter budget.
- **Top-4** (vs GPT-OSS's top-2) gives more expert diversity per token, improving quality.
- **1 shared expert** (always active) provides a baseline transformation regardless of routing.
- **moe_inter_dim=384** (vs dense inter_dim=1536): each expert is smaller, but 4+1 are active, giving `5 × 3 × 768 × 384 = 4.4M` active params per MoE layer vs `3 × 768 × 1536 = 3.5M` per dense layer — comparable compute, more total capacity (the FLOP comparison in [§18](#flop-accounting-per-layer-type) shows 8.8 vs 7.1 MFLOP/token).

**Capacity vs compute, in numbers:** each MoE layer *stores* $21 \times 884\,736 = 18.6$M params — 5.2× the dense layer's 3.5M — but *executes* only 4.4M of them. The ratio $\text{capacity}/\text{compute} = 21/5 \approx 4.2$ is the layer's sparsity leverage: a 16-MoE-layer stack presents the optimizer with 297M expert weights while costing, per token, the FLOPs of a ~72M dense stack. The catch is the tail: with top-4 of 20, the *least-used* expert still needs enough tokens to train (the bias loop keeps utilisation within the ±10% deadband, but pathological tokens can still collapse a specialist), which is why the gate's `get_load_balance_loss` metric (`models/moe.py:DeepSeekMoE.get_load_balance_loss`) is logged every step even though no gradient flows from it.

---

### Active vs Total Parameters — Deep Derivation

Per token, only the selected experts execute. Of the 411.6M deduped params, **185.1M are active** (44.98%):

| Component | Active params |
|---|---|
| Dense SwiGLU (2 layers) | 7,077,888 |
| MoE: 16 layers × (4 routed + 1 shared) experts | 70,778,880 |
| Non-FFN (embed + MLA + norms + gates + tied head) | 107,283,072 |
| **Active per token** | **185,139,840 (~185M)** |
| Inactive (unselected routed experts) | 226,492,416 (~226M) |
| **Total (deduped, weight-tied)** | **411,632,256** |

Derivation, one expert = `3·dim·moe_inter_dim = 3·768·384 = 884,736` params:

- Active MoE = `16 × 5 × 884,736 = 70,778,880`
- Inactive routed = `16 × 16 × 884,736 = 226,492,416`
- Non-FFN = `411,632,256 − 16·21·884,736 − 2·(3·768·1536) = 107,283,072` (equivalently: embed 76.8M + MLA 30.2M + norms 0.03M + gates 0.25M)
- Check: `7,077,888 + 70,778,880 + 107,283,072 = 185,139,840` and `185,139,840 + 226,492,416 = 411,632,256`.

**Interpretation:** ~55% of the weights are "stored but silent" on any given token — this is the sparsity bargain. Storage is paid for all 21 experts; FLOPs and gradient traffic are paid only for the 5 that fire. See the canonical table in [§6](#parameter-budget).

---

### Why 2 Dense Layers

The first 2 layers are **dense SwiGLU FFNs** (no MoE routing). The rationale:

1. **Low-level feature extraction:** Early layers learn general features (tokenization, syntax) that all tokens need. These don't benefit from expert specialization.
2. **Routing stability:** The router needs meaningful hidden states to make good routing decisions. If the first layer is MoE, the router sees raw embeddings, which are less informative.
3. **DeepSeek-V3 design:** The original DeepSeek-V3 also starts with dense layers before switching to MoE.

```python
# In TransformerBlock.__init__:
self.ffn = SwiGLUFFN(self.dim, config["inter_dim"]) if layer_id < self.n_dense_layers else DeepSeekMoE(config)
```

With `n_dense_layers=2`:
- **Layers 0–1:** Dense SwiGLU FFN (inter_dim=1536)
- **Layers 2–17:** MoE FFN (20 routed top-4 + 1 shared, moe_inter_dim=384)

---

### Why seq_len=2048

DeepSeek-V3-Lite uses `max_seq_len=2048`, shorter than GPT-OSS-Lite's 4096:

1. **More optimizer steps per token budget:** At seq=2048, each step processes `8 × 4 × 2048 = 65,536` tokens. At seq=4096, each step processes 131,072 tokens. With 8.4B total tokens, seq=2048 gives 128K optimizer steps vs 64K at seq=4096. More steps = more gradient updates = potentially better convergence.
2. **MLA doesn't need long context for the compression benefit:** MLA's cache savings apply at any context length. The YaRN scaling (rope_factor=1.0 at training, can be increased for inference) handles extrapolation.
3. **DeepSeek-V3 uses a similar strategy:** The original model trains at 4K but extrapolates with YaRN at inference.

---

### What This Project Deliberately Excludes

| Excluded | Reason |
|---|---|
| GQA | MLA is the attention mechanism (GQA would be a different project) |
| Standard aux loss | Aux-loss-free bias is the DeepSeek approach (standard aux is GPT-OSS's approach) |
| Sliding-window attention | GPT-OSS territory |
| Attention sinks | GPT-OSS territory |
| SSM / Mamba | Mamba-3-Lite territory |
| Compressed queries (q_lora_rank=0) | Simplified for 411.6M scale; DeepSeek-V3 full uses q_lora_rank=1536 |
| FP8 training | Paper-spec only; BF16 fits on one A100 — see [MLA & Mixed Precision](attention-and-precision.md) |

---

## Transformer Block Details

> Dense vs MoE layers, RMSNorm, weight tying, parallel embedding, causal mask caching, and gradient checkpointing.

---

### Pre-Norm Rationale

DeepSeek-V3 uses **pre-normalization** (RMSNorm before each sub-layer):

```
x = x + Attention(RMSNorm(x))
x = x + FFN(RMSNorm(x))
```

**Why pre-norm (not post-norm)?**

1. Better gradient flow — the residual path is unnormalized, gradients flow directly through the residual "highway."
2. Training stability — the norm "resets" the scale before each sublayer, preventing activation magnitude blowup at depth 18+.
3. A final `RMSNorm` is needed before the output head (because pre-norm leaves the final residual unnormalized).

Two separate RMSNorm instances per block: `attn_norm` for attention and `ffn_norm` for the FFN. They don't share weights because the attention and FFN sublayers have different scale characteristics.

---

### RMSNorm Math and eps=1e-6

DeepSeek-V3-Lite uses PyTorch's built-in `nn.RMSNorm`:

$$
\text{RMSNorm}(x) = \gamma \cdot \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}}
$$

- No mean subtraction (unlike LayerNorm) — simpler, faster.
- No bias parameter — only scale $\gamma$.
- The internal computation uses FP32 for the reduction (handled by PyTorch's `nn.RMSNorm`).
- DeepSeek uses `eps=1e-6` (tighter than the typical `1e-5`). This allows the norm to be more aggressive when the RMS is very small, but requires careful initialization to avoid division by near-zero values.

---

### Embedding init std=0.006

This repo uses `nn.Embedding(vocab_size, dim)` directly — no custom embedding class — initialised with `nn.init.normal_(self.embed.weight, std=0.006)` in `models/transformer.py:Transformer.__init__`. (The full DeepSeek-V3 uses a "ParallelEmbedding" that handles tensor-parallel splitting across GPUs; this single-GPU implementation has no such class.)

**Why std=0.006?** Much smaller than the typical `0.02` used in GPT-2/LLaMA. With vocab=100,018 and dim=768, the embedding has 76.8M params. Small init keeps the initial embeddings close together (similar tokens start similar); the model learns to spread them apart during training.

---

### Weight Tying — Savings Math

Without tying: `2 × 100,018 × 768 = 153.6M` (embedding + head). With tying: `100,018 × 768 = 76.8M` (single shared tensor). **Savings: 76.8M** — 18.7% of the 411.6M total.

```python
self.head = nn.Linear(dim, vocab_size, bias=False)  # 768 → 100,018
if self.weight_tying:
    self.head.weight = self.embed.weight  # share the same tensor!
```

Deduplication in parameter counting uses `id(p)` (Python object identity) to detect tied parameters. Without this, the count would be ~488M instead of 411.6M (double-counting the 76.8M tied weights).

---

### Causal Mask Caching

See Part II [§23](#causal-mask-cache) for the verbatim source. The full contract, restated:

- **Training / prefill (seqlen > 1, no cache):** `kv_len = seqlen`, `start_pos = 0` — a standard local causal mask, cached and reused.
- **Cached mid-sequence prefill (seqlen > 1, cache):** `kv_len = end_pos`, `start_pos > 0` — causal by global position, so a chunk cannot attend its own future.
- **Decode (seqlen == 1):** No mask needed — a single new token attends to all cached tokens, which are all in the past (causal by construction).
- **Different seqlen/kv_len/start_pos/device:** The mask is rebuilt when any key part changes.

---

### Gradient Checkpointing — Every Layer, use_reentrant=False

See Part II [§26](#gradient-checkpointing) for the verbatim `_run_layers` (`models/transformer.py:Transformer._run_layers`). Design choices, restated:

- **All 18 layers are checkpointed**, not every Nth — maximum memory savings at the cost of recompute.
- **`use_reentrant=False`:** PyTorch 2.x recommended default, `torch.compile`-compatible.
- **Only during training:** the `self.training` check disables checkpointing at inference.
- **Memory impact (derived):** without checkpointing, each layer retains its 25.2 MB output plus intermediates across the backward (≈453 MB just for the `h` tensors at $B{=}8, S{=}2048$); with checkpointing only the region input (25.2 MB) is retained and the forward is recomputed inside the backward, trading ~33% more total FLOPs for roughly 3× less activation memory.

---

### Forward Contracts (detail)

**`Transformer.forward`** — Primary interface, returns vocabulary logits (`models/transformer.py:Transformer.forward`):

```python
def forward(self, tokens, start_pos=0, use_cache=True):
    """(bsz, seqlen) → (bsz, seqlen, vocab_size)"""
    # `nn.Embedding` requires Long indices. Accept uint32 (common from mmap'd
    # token shards) by casting at the boundary.
    if tokens.dtype != torch.long:
        tokens = tokens.to(torch.long)
    bsz, seqlen = tokens.shape
    h = self.embed(tokens)
    end_pos = start_pos + seqlen
    if seqlen > 1:
        kv_len = end_pos if use_cache else seqlen
        mask = self._build_causal_mask(seqlen, kv_len, start_pos if use_cache else 0, tokens.device)
    else:
        mask = None
    h = self._run_layers(h, start_pos, mask, use_cache)
    return self.head(self.norm(h))
```

Note the dtype cast at the boundary (uint32 shards), and the `kv_len = end_pos if use_cache else seqlen` / `start_pos if use_cache else 0` pair — without the `use_cache` guards, a cached mid-sequence prefill would build a wrong (self-attending) mask.

**`Transformer.forward_with_hidden`** — MTP-compatible forward (`models/transformer.py:Transformer.forward_with_hidden`):

```python
def forward_with_hidden(self, tokens, start_pos=0, use_cache=False):
    """Returns (logits, h). h is the pre-norm trunk hidden (V3 feeds this
    to MTP blocks, which apply their own norms); logits use the normed h."""
    # `nn.Embedding` requires Long indices. Accept uint32 (common from mmap'd
    # token shards) by casting at the boundary.
    if tokens.dtype != torch.long:
        tokens = tokens.to(torch.long)
        tokens = tokens.to(torch.long)
    bsz, seqlen = tokens.shape
    h = self.embed(tokens)
    end_pos = start_pos + seqlen
    if seqlen > 1:
        kv_len = end_pos if use_cache else seqlen
        mask = self._build_causal_mask(seqlen, kv_len, start_pos if use_cache else 0, tokens.device)
    else:
        mask = None
    h = self._run_layers(h, start_pos, mask, use_cache)
    return self.head(self.norm(h)), h
```

Returns the **pre-final-norm** trunk hidden — `MTPBlock` applies its own `RMSNorm` before fusion (see [MTP](../concepts/moe-mtp.md)).

**`MultiHeadLatentAttention.forward`** — Takes and returns hidden states (`models/mla.py:MultiHeadLatentAttention.forward`):

```python
def forward(self, x: torch.Tensor, start_pos: int = 0, mask: Optional[torch.Tensor] = None, use_cache: bool = True) -> torch.Tensor:
    """(bsz, seqlen, dim) → (bsz, seqlen, dim)"""
    # ... query/KV/attention computation ...
    return self.wo(attn_output)
```

MLA takes hidden states and returns hidden states — same shape, same contract as standard attention. The full shape trace is [§28](#tensor-shape-walkthrough-per-module).

---

## Check Your Understanding

**Q1: Why is the parameter count 411.6M, not 422M, and what does "deduplicated" mean?**

The config file is *named* `pretrain_a100_422m.yaml` for historical reasons, but instantiating it and summing `count_parameters` gives 411,632,256. "Deduplicated" refers to weight tying: `head.weight` and `embed.weight` are the same storage, so `count_parameters` counts the shared tensor once via `id(p)`. Counting naively (embedding and head separately) would give ~488M. With MTP the total is 418,713,984.

**Q2: Why are only ~185M of the 411.6M parameters active per token, and where do the inactive ones live?**

Per token each MoE layer executes 5 of its 21 experts (4 routed + 1 shared). The 16 unselected routed experts per layer — 16 layers × 16 experts × 884,736 params = 226.5M — receive no activations and no gradients. Everything else (embedding, MLA, norms, gates, dense FFN, tied head: 107.3M) plus the active experts (70.8M) sums to 185.1M. The inactive weights still occupy VRAM and are still updated on *other* tokens.

**Q3: Which single operation costs the most FLOPs per token, and why does that matter for training time?**

The LM head: 2·768·100,018 ≈ 153.6 MFLOP/token — more than all 18 MLA layers combined (≈180.7 MFLOP/token). It is the "logits bottleneck" of a 100K vocabulary on a small hidden size. Combined with ~490 MFLOP/token total forward, a pure-FLOP training estimate at 40% MFU lands near ~27–32 h for 8.4B tokens — comfortably above the config's 13–15 h target, which is itself an unverified estimate.

**Q4: During decode, why is `mask=None` safe, and what role does `start_pos` play instead?**

With `seqlen == 1`, there is exactly one query and it is at the newest global position; every cached key is strictly in its past, so no mask is needed (a causal mask would be a no-op anyway). `start_pos` carries the position information: it offsets RoPE (`_apply_rope` reads `freqs_cis[start_pos:start_pos+seqlen]`) and selects the cache write slot (`kv_cache[:bsz, start_pos:end_pos] = ...`). It also feeds `_build_causal_mask` when `seqlen > 1` in a cached prefill, where the mask must be causal by *global* position so a chunk cannot attend its own future.

**Q5: Where is the biggest transient allocation in a training step, and how does grad checkpointing change the memory profile?**

The logits tensor, $(8, 2048, 100018)$ BF16 ≈ 3.3 GB, plus a similar-scale cross-entropy buffer — together ~6.6 GB of transient traffic. Gradient checkpointing does not shrink that; it shrinks per-layer activations by retaining only the region input and recomputing each layer's forward inside the backward (≈3× less activation memory, ~33% more total FLOPs). The estimator's 24× PaLM activation factor is meant to envelope these spikes — which is why the docs still recommend 15–20% headroom on top of the ~30.5 GB sum.

---

## Portfolio Architecture Comparison

> **Purpose:** Compare DeepSeek-V3-Lite's architectural choices with its sibling projects in the CoreProjects portfolio: GPT-OSS-Lite, LLaMA-3-Lite, and Mamba-3-Lite. Each project is mechanistically distinct and addresses core LLM design challenges differently.

> **60-second summary:** DeepSeek-V3-Lite is the portfolio's answer to the two structural costs of modern LLMs — the KV cache and the dense FFN. MLA compresses the per-token cache from 1,536 floats (MHA) to 216 — a 7.1× cut with no quality loss — and the aux-loss-free MoE stores 411.6M parameters while activating only ~185M per token. This chapter compares those choices against the siblings: GPT-OSS-Lite (GQA + sliding-window alternation, classic aux-loss MoE), LLaMA-3-Lite (pure dense, GQA), and Mamba-3-Lite (no attention at all — a constant-state SSM).

---

## 1. Architecture Comparison Matrix

| Property | DeepSeek-V3-Lite | GPT-OSS-Lite | LLaMA-3-Lite | Mamba-3-Lite |
|---|---|---|---|---|
| **Attention** | MLA (latent KV) | GQA + SWA/full alt | GQA | — (SSM) |
| **KV compression** | Low-rank latent (192+24) | Sliding window (128) | None (standard GQA) | Constant-state SSM |
| **KV cache reduction** | 7.1× vs MHA | 2.0× via SWA/full alt | 2× via GQA groups | O(1) state |
| **Long-context** | YaRN (decode only) | YaRN 128K (train+decode) | θ=500K (train@2K) | Constant-state |
| **MoE** | ✅ 20 routed top-4 + 1 shared | ✅ 8 routed top-2 + 1 shared | ❌ | ❌ |
| **Load balancing** | Aux-loss-free bias | Standard aux loss (α=0.01) | N/A | N/A |
| **Attention sinks** | ❌ | ✅ Per-head learned | ❌ | N/A |
| **MTP** | ✅ Depth=1 + speculative | ❌ | ❌ | ❌ |
| **Position encoding** | Decoupled RoPE (24-dim) | YaRN-scaled RoPE (72-dim) | Standard RoPE (θ=500K) | — (implicit in SSM) |
| **Normalization** | RMSNorm (eps=1e-6) | RMSNorm (eps=1e-5) | RMSNorm | RMSNorm |
| **Weight tying** | ✅ | ✅ | ✅ | ✅ |
| **Tokenizer** | deepseek-coder-v2-lite (100,018) | LLaMA-3 (128,000) | LLaMA-3 (128,000) | — |
| **Total params** | ~411.6M (418.7M w/ MTP) | ~502M | — | — |
| **Training context** | 2,048 | 4,096 | 2,048 | — |
| **Eval context** | YaRN-scaled | 131,072 | — | Constant-state |

> **Param-count note:** "422M" survives only as the config filename (`pretrain_a100_422m.yaml`); the verified deduplicated counts are 411.6M base and 418.7M with the MTP head (MTP adds ~7.1M). ~185M parameters are active per token. See [Model Architecture](../concepts/foundations.md) §6 for the full budget.

---

## 2. Attention Mechanism Comparison

### 2.1 MLA vs GQA vs SWA-Alternation vs SSM

| Mechanism | How it reduces KV | Quality impact | Unique to |
|---|---|---|---|
| **MLA** | Low-rank compression → 192-dim latent | No loss (matches MHA) | DeepSeek |
| **GQA** | Share KV heads → fewer heads | Slight loss | LLaMA-3, GPT-OSS |
| **SWA/full alt** | Windowed layers cache only 128 tokens | Good (global layers retain context) | GPT-OSS |
| **SSM** | No KV cache at all — constant-size state | Different mechanism entirely | Mamba |

### 2.2 The Key Distinction

- **DeepSeek-V3-Lite (MLA):** Compresses K/V **into a latent** via learned projections. The compression is lossless (up-projection recovers full K/V). The absorption trick eliminates the up-projection at inference.
- **GPT-OSS-Lite (SWA/full):** Compresses the cache by **reducing what's stored** — windowed layers only keep the last 128 tokens. The compression is lossy (distant context is forgotten in windowed layers) but compensated by global layers.
- **LLaMA-3-Lite (GQA):** Compresses by **sharing K/V across heads**. Simple but limited — can't compress beyond 1 KV head (MQA).
- **Mamba-3-Lite (SSM):** No attention at all — uses a state-space model with constant-size state. No KV cache needed.

### 2.3 The KV-Cache Axis — MLA vs GQA vs MHA vs Linear Attention

Every attention variant is a different answer to one question: **what does the model keep around for past tokens?** That answer determines both how much memory a long conversation costs and how many bytes each decode step must read from HBM. The table below lays the four families side by side on this single axis, using the canonical 411.6M config's dimensions (derivation in [MLA](../concepts/attention-and-precision.md) §10):

| Property | MHA | GQA (4 groups) | MQA | **MLA (this repo)** | Linear attention / SSM |
|---|---|---|---|---|---|
| Cached per token per layer | 1,536 floats ($2 \cdot H \cdot d_v$) | 512 ($2 \cdot 4 \cdot 64$) | 128 ($2 \cdot 64$) | **216** ($R=192 + d_{\text{rope}}=24$) | constant state, not a cache |
| Growth with context | $O(S)$ | $O(S)$ | $O(S)$ | $O(S)$ | $O(1)$ |
| Reduction vs MHA | 1× | ~3× | 12× | **7.1×** | unbounded (state independent of $S$) |
| Quality vs MHA | baseline | slight drop | measurable drop | **matches MHA** | different mechanism — not directly comparable |
| Decode compute | lowest | low | low | ~4× MHA | low (recurrent scan) |
| Decode memory bandwidth | highest | medium | low | **lowest** | **lowest** (fixed-size state read) |

> **Footnote on ratios:** the per-token float counts are exact for this config. The GQA ratio is 3.0× on the $d_v=64$ baseline used here; [MLA](../concepts/attention-and-precision.md) §10 rounds it to 3.3× on a $d_{qk}=72$ baseline. The MLA 7.1× figure is the canonical one used throughout these docs.

**What the axis teaches.** LLaMA-3-Lite implements the GQA column (2× group ratio, §1) — a fixed sharing pattern that cannot compress below one KV head. GPT-OSS-Lite combines GQA with sliding-window alternation, cutting the *number of positions cached* (windowed layers keep 128) rather than the bytes per position. Mamba-3-Lite sits at the linear-attention end: a fixed-size recurrent state ($N=64$ complex64 per layer) replaces the cache entirely, so memory is $O(1)$ in context — hence no RoPE and no extrapolation story (§4). MLA is the only entry that compresses *into a latent*: 216 floats as a learned low-rank code, not a subsample of the K/V heads — which is why quality holds where GQA/MQA give ground.

**The counter-intuitive bit: cache size and decode speed are different axes.** A smaller cache only speeds up decode if each step also *reads* less. The absorption trick (`models/mla.py:MultiHeadLatentAttention.forward`) folds the key/value up-projections into the query and output projections, so scores are computed directly in latent space — each step reads the 216 cached floats, never the 1,536 they expand to. The trade hidden in the table: MLA's decode *compute* is higher than GQA's (~4× MHA — the absorbed query runs over $R=192$ and every score is an $R$-dimensional inner product). It wins anyway because decode is bandwidth-bound, not FLOP-bound ([Model Architecture](../concepts/foundations.md) §39). Absorption is inference-only: during training the same method materialises full K/V so gradients reach the up-projection weights.

**Pitfalls when comparing attention families:**

- **"7.1× smaller cache" ≠ "7.1× faster decode"** — the bandwidth win needs absorption; without it you still read the expanded K/V.
- **Reduction ratios are baseline-dependent** — always quote the MHA baseline (head count × head dim) alongside the ratio.
- **GQA cannot compress below one KV head** — MQA is exactly that limit and it measurably degrades quality.
- **"No KV cache" (SSM) is a mechanism change, not a compression** — Mamba's quality sits on a different axis and is not directly comparable to attention's.

---

## 3. MoE Comparison

| Property | DeepSeek-V3-Lite | GPT-OSS-Lite |
|---|---|---|
| Routed experts | 20 | 8 |
| Active experts | 4 (top-4) | 2 (top-2) |
| Shared experts | 1 | 1 |
| Gate activation | Sigmoid | Softmax |
| Load balancing | Aux-loss-free bias | Standard aux loss (α=0.01) |
| Expert inter_dim | 384 | 1536 |
| Dispatch | Stacked bmm | Stacked F.linear |

### Key Philosophical Difference

**DeepSeek:** Finer-grained experts (20) with sigmoid routing and **no aux loss**. The bias update is a control system — it adjusts the gate logits out-of-band, without contaminating the task gradient.

**GPT-OSS:** Coarser experts (8) with softmax routing and **standard aux loss**. The aux loss is a regularizer — it adds a gradient signal that pushes toward uniform utilization.

Both are valid approaches. The aux-loss-free method is more elegant (pure task gradient) but requires careful bias update tuning. The standard aux loss is simpler and more widely validated but introduces gradient coupling.

### 3.1 MoE vs Dense — the Sparsity Trade

The comparison above pits the two MoE projects against each other, but the sharper portfolio question is **why MoE at all** — LLaMA-3-Lite spends a comparable budget on a fully dense stack and never routes a single token. The trade has one big benefit and three real costs:

| | Dense (LLaMA-3-Lite; DeepSeek layers 0–1) | MoE (DeepSeek layers 2–17; GPT-OSS-Lite) |
|---|---|---|
| Stored FFN params | 100% of the stack | 100% (more, per layer) |
| Active per token | 100% | DeepSeek: 5/21 ≈ 23.8% of a MoE layer's FFN; GPT-OSS: (2+1)/9 ≈ 33% of its routed stack |
| Whole-model active | all params | DeepSeek: ~185.1M of 411.6M (≈45%) |
| Compute per token | proportional to stored params | proportional to *k* routed + shared experts |
| Failure mode | none | expert collapse without a balancing mechanism |

**The benefit.** Parameter capacity decouples from FLOPs: DeepSeek-V3-Lite stores 411.6M deduplicated params but only ~185.1M are exercised per token (measured by instantiating the canonical config — [Model Architecture](../concepts/foundations.md) §42), and each of the 16 MoE layers executes only 4 routed + 1 shared of its 21 experts. LLaMA-3-Lite's dense stack activates everything on every token — simple and predictable, but capacity is priced in FLOPs. The 411.6M/185.1M split is a 2.2× capacity multiplier over active compute.

**The costs.**

1. **Every expert lives in memory.** VRAM, checkpoint size, and load time track *stored* params, not active ones — sparsity saves compute, never memory.
2. **Sparsity without balance collapses.** If one expert wins slightly, it trains faster and wins more — a positive feedback loop that silently burns capacity ([MoE](../concepts/moe-mtp.md) §2). DeepSeek-V3-Lite's answer is the out-of-band bias controller (`models/moe.py:AuxLossFreeGate.forward`); GPT-OSS-Lite's is the auxiliary loss.
3. **Dispatch overhead and imbalance.** Tokens are sorted into per-expert batches, and the busiest expert's chunk sizes the wall-clock; this repo drops no tokens, so imbalance costs time, never correctness ([MoE](../concepts/moe-mtp.md) §5).

At 411.6M the trade resolves cleanly: the model is trained on 8.4B Chinchilla-optimal tokens — under-trained relative to its capacity — so it benefits more from stored capacity than from raw FLOPs, the classic argument for sparsity at small scale. Dense-only LLaMA-3-Lite is the control arm of the portfolio: same budget class, no routing, no balancing risk.

---

## 4. Long-Context Strategy Comparison

| Project | Strategy | Training context | Eval context | How it works |
|---|---|---|---|---|
| DeepSeek-V3-Lite | YaRN (decode only) | 2,048 | YaRN-scaled | Train without YaRN, apply at decode time |
| GPT-OSS-Lite | YaRN (train+decode) | 4,096 | 131,072 | Train with YaRN active, extrapolate 32× |
| LLaMA-3-Lite | θ=500K | 2,048 | — | Large RoPE base, moderate extrapolation |
| Mamba-3-Lite | Constant-state | — | — | SSM doesn't need extrapolation |

### DeepSeek vs GPT-OSS on YaRN

- **DeepSeek:** `rope_factor=1.0` at training (no YaRN). At inference, increase `rope_factor` to scale RoPE frequencies for longer context. This is a **decode-time patch** — the model wasn't trained for long context.
- **GPT-OSS:** `yarn_scale_factor=32` at training. The model learns the YaRN frequency ramp during training. This is **true length extrapolation** — the model genuinely generalizes to 32× its training context.

The trade-off: DeepSeek's approach is simpler (no YaRN at training) but less reliable at extrapolation. GPT-OSS's approach is more complex (YaRN must be configured correctly) but produces genuine extrapolation capability.

---

## 5. Unique Innovations Per Project

| Project | Unique innovations (not in siblings) |
|---|---|
| **DeepSeek-V3-Lite** | MLA (low-rank KV compression + absorption), aux-loss-free MoE bias, MTP + speculative decoding, μP LR scaling, dense+MoE topology |
| **GPT-OSS-Lite** | Sliding-window/full alternation, per-head learned attention sinks, YaRN at training time, pruned RoPE on global layers |
| **LLaMA-3-Lite** | 78% memory stack optimization, chunked cross-entropy, async prefetch, GQA with θ=500K |
| **Mamba-3-Lite** | Complex-valued SSD (N=64 complex64), MIMO head mixing, zero causal conv, A100-optimized chunkwise |

---

## 6. What Each Project Teaches

### DeepSeek-V3-Lite
- **MLA** is the most memory-efficient attention variant. It teaches low-rank compression, the absorption trick (matrix algebra for eliminating intermediate computations), and decoupled RoPE (splitting position from content to preserve absorption).
- **Aux-loss-free MoE** teaches an alternative to gradient-based load balancing — using control theory (bias updates) instead of optimization objectives.
- **MTP** teaches multi-token prediction as both a training regularizer and an inference accelerator (speculative decoding).

### GPT-OSS-Lite
- **Sliding-window alternation** teaches how to trade context range for cache efficiency while maintaining global information flow.
- **Attention sinks** teach the softmax normalization problem and how learned per-head biases solve it.
- **YaRN at training** teaches true length extrapolation (vs decode-time patching).

### LLaMA-3-Lite
- **GQA** teaches the simplest effective KV reduction.
- **Memory optimization** teaches the 78% peak memory reduction stack (a production engineering masterclass).

### Mamba-3-Lite
- **SSD (State-Space Duality)** teaches an alternative to attention entirely — a mathematically grounded approach to sequence modeling with constant memory.

---

## 7. Cross-Project Lessons

1. **KV cache is the dominant inference bottleneck.** Every project addresses it differently — MLA compresses it, SWA reduces what's cached, GQA shares it, SSM eliminates it.
2. **Load balancing in MoE is a design choice, not a settled question.** DeepSeek uses bias updates (control theory), GPT-OSS uses aux loss (optimization). Both work; they have different trade-offs.
3. **Position encoding for long context is still an open problem.** YaRN (GPT-OSS), decode-time YaRN (DeepSeek), large θ (LLaMA-3), and implicit positioning (Mamba) are all valid approaches with different extrapolation properties.
4. **Weight tying is universal.** All four projects use it. The savings (76–98M params) are significant relative to model size.
5. **RMSNorm has won.** All four projects use RMSNorm, not LayerNorm. The simpler computation and slightly better gradient flow have made it the standard.

---

## 8. Why These Design Choices at 411.6M

Read together, DeepSeek-V3-Lite's decisions form one coherent answer to a single constraint: **train a faithful DeepSeek-V3 on one A100 80GB at Chinchilla-optimal scale without dropping any paper mechanism.** Every choice either shrinks bytes moved, keeps FLOPs within single-GPU reach, or buys a capability for free.

1. **MLA instead of GQA — because decode is bandwidth-bound even at 411.6M.** The 7.1× cache cut with absorption (`models/mla.py:MultiHeadLatentAttention.forward`) is the largest per-token memory win available, at zero quality cost — GQA's fixed head-sharing caps at one KV head and degrades before reaching it ([Model Architecture](../concepts/foundations.md) §39). At 2,048 training context the absolute cache is small; the choice is about the long-context decode regime, where `models/transformer.py:Transformer.generate` reads the latent cache one token at a time ([Inference](../inference.md)).

2. **MoE instead of dense — because capacity, not FLOPs, is the scarce resource at 8.4B tokens.** Storing 411.6M and activating ~185.1M (≈45%) is a 2.2× capacity multiplier over active compute (§3.1), with no quality-reducing trickery in the attention stack. The aux-loss-free bias controller keeps the task gradient pure — a deliberate contrast with GPT-OSS-Lite's aux loss.

3. **Fine-grained experts, top-4 of 20.** With $C(20, 4) = 4\,845$ routable subsets per token versus GPT-OSS-Lite's $C(8, 2) = 28$, the router composes far more specialised behaviour from the same capacity class ([MoE](../concepts/moe-mtp.md) §3).

4. **Two dense layers up front.** Routing on raw early representations is noisy — the first layers handle universal token-level patterns densely, and only layers 2–17 route ([Model Architecture](../concepts/foundations.md) §43).

5. **MTP — a regularizer that pays for itself at inference.** Training adds a depth-1 head predicting token $t+2$ (`models/mtp.py:MultiTokenPrediction.forward`, loss weight 0.3) for ~7.1M extra params (≈1.7% of 418.7M). At decode, the same head drafts tokens that `inference/speculative.py:SpeculativeDecoder.generate_step` verifies against the trunk — the training overhead converts into decode speedup ([MTP](../concepts/moe-mtp.md)).

6. **μP LR — one hyperparameter that transfers across scales.** $6.0e-4 \times \sqrt{757226496 / N}$ gives 8.14e-4 (base) and 8.07e-4 (with MTP), so the training recipe survives model-size changes without retuning ([Training](../training.md)).

**What the portfolio adds up to.** Each sibling attacks a different bottleneck: GPT-OSS-Lite attacks context (train-time YaRN to 128K), LLaMA-3-Lite attacks memory engineering (78% peak-memory reduction), Mamba-3-Lite attacks the quadratic core itself (constant-state SSD). DeepSeek-V3-Lite attacks the two structural costs — the KV cache and the dense FFN — while keeping exact-quality attention and a paper-faithful training story. The four are not four implementations of one idea; they are four points in the design space, and T-chapters 02–05 and 08 are the deep dives behind the columns of §1.

---

## 9. Check Your Understanding

**Q1. Why does MLA match MHA quality while GQA and MQA degrade?**

A1. GQA/MQA reduce the cache by *sharing or dropping* key/value heads — a fixed, information-losing subsample. MLA's latent is produced by a learned compression matrix, and the per-head K/V are recovered by a learned up-projection; the absorption algebra ([MLA](../concepts/attention-and-precision.md) §5) shows the recovered keys/values are exactly what materialised MHA would compute. The compression is learned, not structural.

**Q2. A model advertises "3× KV-cache reduction". What must you ask before believing it?**

A2. Against which baseline? The same mechanism is 3.0× against a $d_v=64$ MHA baseline and 3.3× against a $d_{qk}=72$ one (§2.3). Then ask what decode *reads* per step, not just what it stores — absorption is what turns MLA's 216 stored floats into a bandwidth win.

**Q3. DeepSeek-V3-Lite stores 411.6M params. Why is it wrong to call it "a 411.6M-FLOPs-per-token model"?**

A3. Because only ~185.1M (≈45%) are active per token — the rest is routed-expert capacity that wakes only for tokens that select it. FLOPs track active params (top-4 + shared), while memory and checkpoints track stored params (§3.1).

---

## Summary

DeepSeek-V3-Lite is the **attention compression** and **MoE balancing** project in the portfolio:
1. [MLA](../concepts/attention-and-precision.md) — the only low-rank KV compression with the absorption trick.
2. [MoE](../concepts/moe-mtp.md) — the only control-theory-based load balancing.
3. [MTP](../concepts/moe-mtp.md) — the only multi-token prediction with speculative decoding.
4. [Training](../training.md) — the only principled μP LR scaling across model sizes.

Combined with GPT-OSS-Lite (sinks + SWA + YaRN), LLaMA-3-Lite (GQA + memory optimization), and Mamba-3-Lite (SSM), the portfolio covers the full spectrum of modern LLM architecture innovations — each project mechanistically distinct, each teaching a different fundamental approach.

## References

- [R2 — Transformer API](../references/R2_transformer_api.md) — per-symbol API reference
- `models/transformer.py`, `models/mla.py`, `models/moe.py`, `models/mtp.py`, `training/pretrain.py` — authoritative sources
- [MoE & MTP](../concepts/moe-mtp.md) — component deep-dives
