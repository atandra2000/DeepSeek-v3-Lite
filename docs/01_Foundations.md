# 01 — Architectural Foundations

> **Canonical** for the from-scratch primitives every other chapter assumes: causal LM objective, pre-norm residuals, attention, RoPE, KV caching, Chinchilla scaling, μP. Educational textbook chapter.

> The conceptual base layer. DeepSeek lineage (V1 → V2 → V3) plus the mathematical primitives that every later chapter builds on. Read this before [[Docs/02_Model_Architecture]], [[Docs/03_Multi_Head_Latent_Attention]], or [[Docs/08_Training_Pipeline]] — each of those cites sections here by number.

**Depends on:** [[Docs/00_Getting_Started]] · **Read next:** [[Docs/02_Model_Architecture]], [[Docs/03_Multi_Head_Latent_Attention]]

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

> [!NOTE] **Lineage numbers are the paper's, not this repo's.** The V2/V3 rows above describe the *published* DeepSeek models. This reproduction (DeepSeek-**v3-Lite**) is a 422 M-param single-GPU model: `d_c = 192`, 20 routed experts (4 active) + 1 shared, BF16. See [[Reference]] for the canonical spec. Don't conflate the paper's 14 B / 64-expert numbers with this repo's 422 M / 20-expert numbers — they are different scales of the same architecture.

---

## 2. Causal language modeling & cross-entropy loss

A language model assigns probability to a token sequence $\mathbf{x} = (x_1, \ldots, x_T)$ by factorizing the joint into conditionals (autoregressive decomposition):

$$p_\theta(\mathbf{x}) = \prod_{t=1}^{T} p_\theta(x_t \mid x_{<t})$$

The model outputs a distribution over the vocabulary at each position. We train by maximizing the likelihood of the data, equivalently **minimizing the cross-entropy** of the predicted next-token distribution against the observed token:

$$\mathcal{L}_{\text{CE}}(\theta) = -\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})$$

`★ Insight ─────────────────────────────────────`
Cross-entropy is *negative log-likelihood* — it is the number of bits (in nats) the model "needs" to encode the true token under its distribution. A perfect model assigns probability 1 to the right token (loss 0); a uniform-over-vocab model assigns $1/V$ (loss $\log V$). The gradient $\nabla_\theta \mathcal{L}_{\text{CE}}$ flows only through the position being predicted, which is why the causal mask (§7) is essential: without it, position $t$ could "see" $x_t$ and the loss would be trivially zero.
`─────────────────────────────────────────────────`

In code (`training/pretrain.py`), this is one line on the flattened logits:
```python
main_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
```
`targets` is `tokens` shifted by one position — predict $x_{t+1}$ from $x_{\le t}$.

---

## 3. Pre-norm residual blocks & RMSNorm

Every transformer layer is a **residual block** with a pre-norm:

$$h \leftarrow h + f(\text{RMSNorm}(h))$$

where $f$ is attention or the FFN. Two properties matter:

1. **Pre-norm (not post-norm):** the normalization is applied to the *input* of the sublayer, so the residual stream $h$ is never normalized away. This makes deep models trainable — gradients flow through the un-normalized residual path. Post-norm (original Transformer) normalizes the output and is unstable past ~12 layers.
2. **RMSNorm (not LayerNorm):** RMSNorm drops the mean-subtraction of LayerNorm, keeping only the root-mean-square scaling:

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\tfrac{1}{d}\sum_i x_i^2 + \epsilon}} \cdot \gamma$$

LayerNorm computes $\frac{x - \mu}{\sigma}$; RMSNorm computes $\frac{x}{\text{RMS}(x)}$. The mean shift is removed because for activations centred near zero it contributes little, and removing it cuts a reduction + broadcast per norm — cheaper, marginally better gradient flow. Every DeepSeek (and LLaMA/Mamba) block uses `nn.RMSNorm(dim, eps=1e-6)`.

The repo's `TransformerBlock` (`models/transformer.py`) is the canonical form:
```python
def forward(self, x, start_pos=0, mask=None, use_cache=True):
    x = x + self.attn(self.attn_norm(x), start_pos, mask, use_cache)
    x = x + self.ffn(self.ffn_norm(x))
    return x
```
Two residual additions, two pre-norms — attention then FFN. The residual stream `x` carries information; the sublayers *add corrections* to it.

---

## 4. SwiGLU feed-forward network

The FFN is a **SwiGLU** — a gated linear unit with a Swish (SiLU) gate:

$$\text{SwiGLU}(x) = W_2\big(\text{silu}(W_1 x) \odot (W_3 x)\big)$$

Three weight matrices: $W_1, W_3$ project up to `inter_dim`, the SiLU gate selects, $W_2$ projects back down. The elementwise product $\text{silu}(W_1 x) \odot (W_3 x)$ is the gating — $W_3 x$ provides a "value" and $\text{silu}(W_1 x)$ a smooth "gate" that lets the model pass or suppress each hidden dimension. SwiGLU beats the vanilla two-matrix FFN (ReLU/GELU) at equal parameter count because the gate adds a multiplicative nonlinearity.

```python
class SwiGLUFFN(nn.Module):
    def forward(self, x): return self.w2(F.silu(self.w1(x)) * self.w3(x))
```
Dense layers (0–1 in the 422 M config) use `SwiGLUFFN`; MoE layers (2–17) replace it with `DeepSeekMoE`, where each *expert* is itself a SwiGLU (see [[Docs/04_DeepSeekMoE]]).

---

## 5. Multi-head attention — the standard baseline

Standard Multi-Head Attention (MHA) is what MLA compresses. Recap it here so the compression in [[Docs/03_Multi_Head_Latent_Attention]] has a reference point.

For each head $h$, project queries/keys/values and compute scaled dot-product attention:

$$Q_h = x W_h^Q, \quad K_h = x W_h^K, \quad V_h = x W_h^V \in \mathbb{R}^{T \times d_h}$$

$$\text{Attn}_h = \text{softmax}\!\left(\frac{Q_h K_h^\top}{\sqrt{d_h}}\right) V_h$$

Heads run in parallel and are concatenated, then projected by $W^O$. The cost: at inference, every head stores **full** $K_h$ and $V_h$ for every past token. With $H$ heads and sequence length $T$:

$$\text{KV cache (MHA)} = O(T \cdot L \cdot H \cdot d_h)$$

This is the memory MLA attacks — at long context, this cache dominates GPU memory (see §11).

---

## 6. Rotary position embeddings (RoPE)

RoPE encodes **relative position** by rotating query/key pairs in 2-D subspaces. For dimension pair $(d_{2i}, d_{2i+1})$ at position $t$, apply a rotation by angle $t \cdot \theta_i$:

$$\begin{pmatrix} q_{2i}' \\ q_{2i+1}' \end{pmatrix} = \begin{pmatrix} \cos(t\theta_i) & -\sin(t\theta_i) \\ \sin(t\theta_i) & \cos(t\theta_i) \end{pmatrix} \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}, \qquad \theta_i = \theta^{-2i/d}$$

The frequency $\theta_i$ decreases with dimension index, so high dims rotate slowly (capture long-range relations) and low dims rotate fast (capture local relations). The magic: because $Q$ and $K$ are rotated by the *same* angle schedule, the dot product $Q \cdot K$ depends only on the **relative** position $t - s$, not the absolute positions — attention scores are translation-equivariant.

`★ Insight ─────────────────────────────────────`
RoPE is a *multiplicative* position encoding, injected into Q and K (not V). The rotation is applied *before* the dot product, so relative position falls out of $QK^\top$ for free. This is why DeepSeek can do **decoupled RoPE** in MLA: a small dedicated RoPE subspace (`d_R = 24` dims) is rotated separately from the content subspace, so the position information — which cannot be absorbed into the latent — stays outside the compressed cache. See [[Docs/03_Multi_Head_Latent_Attention]] §Decoupled RoPE.
`─────────────────────────────────────────────────`

The repo builds the rotation table as complex exponentials and applies them via complex multiplication (`models/mla.py:_apply_rope`), with the frequency base `rope_theta = 10000` and a `rope_factor` knob for decode-time YaRN scaling (off in training, `rope_factor = 1.0`).

---

## 7. Attention scaling & causal masking

The softmax denominator needs the $\frac{1}{\sqrt{d_h}}$ scale to keep dot products in a sensible range: random unit vectors of dim $d_h$ have dot product $\sim \mathcal{N}(0, d_h)$, so unscaled scores grow with dimension and saturate softmax to one-hot. Dividing by $\sqrt{d_h}$ holds variance at 1.

The **causal mask** prevents position $t$ from attending to positions $> t$: an additive $-\infty$ above the diagonal of the $T \times T$ score matrix. The repo builds it once and caches it by sequence length (`Transformer._build_causal_mask`). At inference with a KV cache, only the *new* token's queries attend to all cached keys, so the mask collapses to "attend to everything up to me" — no masking needed during decode, only during prefill/training when query length equals key length.

For long context (`max_seq_len > 4096`), MLA applies a **YaRN mscale** correction to the softmax scale to counteract the density loss of stretched RoPE frequencies (`mla.py` line 50–53). At the canonical 2048 context this path is inactive.

---

## 8. The residual stream & information flow

Think of the transformer as a **residual stream** $h$ that each layer *reads from and writes to*. Embeddings write the token into the stream; each block adds an attention correction (mix information across positions) and an FFN correction (transform per-position); the final norm + LM head read the stream out as logits.

This framing matters for two later chapters:
- **MLA** compresses the part of the stream that attention *reads* (keys/values) — the *residual* stream itself is untouched.
- **MTP** feeds the *pre-norm* trunk hidden $h$ (not the logits) to the MTP block, which has its own norm — so MTP predicts from the model's internal representation, not the output. See [[Docs/05_Multi_Token_Prediction]].

`Transformer.forward_with_hidden` returns both `(logits, h)` precisely so MTP can use $h$.

---

## 9. The training loss — CE plus the MTP auxiliary

The full training objective combines the main next-token loss with the MTP auxiliary losses (see [[Docs/05_Multi_Token_Prediction]]):

$$\mathcal{L}(\theta) = \mathcal{L}_{\text{CE}} + \lambda \sum_{d=1}^{D} \mathcal{L}_{\text{MTP}}^{(d)}$$

with depth $D = 1$ and weight $\lambda = 0.3$ at the 422 M config. The MTP loss is a cross-entropy on the depth-$d$ prediction target; the total is averaged across depths. The key property: the MTP term **densifies the training signal** — every token contributes to both the main prediction and a +1-ahead prediction, so the gradient per token is richer without needing more data.

In `training/pretrain.py`, `MultiTokenPrediction.compute_loss` returns `(total_loss, main_loss, mtp_loss)`; only `total_loss` is backpropped, the components are detached for logging (avoiding per-step GPU syncs).

---

## 10. Optimization — AdamW, warmup-cosine, gradient clipping

- **AdamW** with β = (0.9, 0.95), weight decay 0.1 (applied to 2-D params only — norms/embeddings/biases excluded). The "fused" CUDA implementation is used when available.
- **Warmup-cosine schedule** (`make_warmup_cosine_lambda`): linear warmup over 2000 steps (avoid early instability), then cosine decay to 5% of peak LR over the remaining steps. The cosine floor (not zero) keeps learning in the tail.
- **Gradient clipping** at norm 1.0 (`clip_grad_norm_`) — a safety rail against rare large batches that would otherwise spike the update.

The FP32 master copy of weights lives in AdamW's state, so BF16 compute + FP32 master is the precision ladder (the same pattern FP8 would extend one notch lower, see [[Docs/06_FP8_Mixed_Precision]]).

---

## 11. KV caching — the inference bottleneck

During autoregressive generation, each new token attends to *all* past tokens. Recomputing all past keys/values every step is $O(T^2)$ total. The fix: **cache** $K$ and $V$ for past tokens and only compute the new token's contributions — $O(T)$ amortized.

The cost is **memory**: the cache grows with context. For MHA it is $O(T \cdot L \cdot H \cdot d_h)$ (§5). At $T = 2048$, $L = 18$, $H = 12$, $d_h = 72$ that is ~6 GB just for keys/values — comparable to the model weights. This is why every efficient LLM family addresses the cache: GQA shares heads, SWA windows it, SSM eliminates it, and **MLA compresses it** into a low-rank latent (see [[Docs/03_Multi_Head_Latent_Attention]] §KV cache, and [[Docs/13_Portfolio_Comparison]] §2). The repo's MLA cache stores only `kv_cache (B, T, 192)` + `pe_cache (B, T, 24)` per layer — ~0.13 GB at the same config, a ~46× reduction.

---

## 12. The Chinchilla token budget

**Chinchilla scaling** (Hoffmann et al. 2022): for compute-optimal training, the number of training tokens should scale ~20× the parameter count. Train a 422 M model on ~8.4 B tokens — not the "train forever" heuristic that wastes compute on too-small models.

$$N_{\text{tokens}} \approx 20 \cdot N_{\text{params}}$$

The 422 M config targets **8.4 B tokens over 512 000 steps** (micro-batch 8 × grad-accum 4 = effective batch 32, sequence 2048 → ~8.4 B tokens). The shared universal pipeline in `LLM/shared_data/` produces exactly this corpus, sharded as `uint32` memmap (see [[Docs/09_Data_Pipeline]]). Undertraining (fewer tokens) leaves the model data-hungry; overtraining wastes FLOPs that a larger model would use better.

---

## 13. μP — maximal-update parameterization

**μP** (Yang et al. 2022) makes the optimal learning rate **transfer** across model widths. In standard training, the best LR depends on width, so you must tune it per size. μP parameterizes every weight's LR scale so that the *update magnitude per parameter* stays width-invariant, making a LR tuned on a small reference model valid on a large target.

The repo's scaling law (`training/pretrain.py`):
$$\text{lr}_{\text{target}} = \text{lr}_{\text{ref}} \cdot \left(\frac{N_{\text{ref}}}{N_{\text{target}}}\right)^{1/2}$$
with reference $\text{lr}_{\text{ref}} = 6.0\text{e-4}$ at $N_{\text{ref}} = 757\,226\,496$ params. The 422 M config sets `mup_lr: true`, so the configured `lr: 8.0e-4` is overridden by the μP-scaled value.

`★ Insight ─────────────────────────────────────`
The intuition: wider models have more parameters contributing to each update; if every param updates at the same LR, the aggregate update to the *function* grows with width and destabilizes training. The $\sqrt{N}$ scaling keeps the effective update magnitude stable as width grows — it is the continuous analog of averaging more independent estimates. This is why μP lets you tune hyperparameters on a cheap small model and lift them to the expensive large one, instead of retuning on the target. See [[Docs/08_Training_Pipeline]] §μP Learning Rate Scaling.
`─────────────────────────────────────────────────`

---

## Cross-cutting lessons

1. **Every choice trades framework magic for readable math.** Pre-norm over post-norm, RMSNorm over LayerNorm, SwiGLU over ReLU-FFN, RoPE over learned position embeddings — each is the simpler, more inspectable, empirically-better option. DeepSeek's lineage is a sequence of "replace the magic with math that works better."
2. **The KV cache is the inference bottleneck.** §11 is the reason MLA exists; it is the connective tissue between attention (§5) and the architecture chapter.
3. **Scaling laws constrain everything.** Chinchilla (§12) sets the data budget; μP (§13) sets the LR. They are why a 422 M single-GPU run is a *deliberate* choice, not a limitation.

> **Next:** [[Docs/02_Model_Architecture]] — full 18-layer topology, tensor shapes, and the parameter budget that realizes these primitives.