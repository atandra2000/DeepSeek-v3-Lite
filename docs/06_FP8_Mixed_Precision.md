# 06 — FP8 Mixed Precision Training & Quantization

> **Canonical** for the DeepSeek-V3 FP8 scheme: E4M3 format for both forward and
> backward GEMMs, tile-wise and block-wise fine-grained scaling, and the
> high-precision accumulation that makes 8-bit training stable. Educational
> textbook chapter — from-scratch, with the math and the intuition.

> How DeepSeek-V3 cuts matmul memory bandwidth and Tensor Core throughput in
> half using 8-bit floats, and why a single global scale factor fails.
> **Status in this repo:** FP8 is **paper-spec only — not implemented.** This
> reproduction trains in BF16 throughout (see [[Docs/08_Training_Pipeline]]).
> This chapter exists so the portfolio documents the technique that makes
> full-scale DeepSeek-V3 feasible; at 411.6M params on a single A100, BF16 is
> already compute-efficient and FP8's complexity is not justified.

**Depends on:** [[Docs/02_Model_Architecture]], [[Docs/03_Multi_Head_Latent_Attention]] · **Read next:** [[Docs/07_DualPipe_Parallelism]]

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

`★ Insight ─────────────────────────────────────`
Naive intuition says gradients are heavy-tailed and need E5M2's range. DeepSeek-V3's empirical answer is the opposite: **E4M3 everywhere**, with fine-grained per-block scaling doing the range work that the exponent field would otherwise do. The official DeepGEMM repository states this explicitly: "We don't support e5m2 because it isn't used in DeepSeek V3/R1." Because scales are per-tile/per-block (not per-tensor), each local block can pick an $S$ that fits *its* values, so a global dynamic range is unnecessary — the 3 mantissa bits of E4M3 (twice the resolution of E5M2) are worth more than a wide exponent range that scaling already provides.
`─────────────────────────────────────────────────`

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

`★ Insight ─────────────────────────────────────`
Why `128`? It is the **Tensor Core tile size**. Hopper FP8 Tensor Cores operate on `16×128` (activation) × `128×128` (weight) tiles natively. Choosing the scale block to match the hardware tile means the scale is applied *inside* the Tensor Core accumulator, with zero overhead — the dequant scale multiply is folded into the dot product. A scale block that didn't align with the tile boundary would force extra rescales between cores. The "fine-grained" granularity is not arbitrary; it is the largest block that is (a) small enough to localize outliers and (b) exactly the Tensor Core's native shape.
`─────────────────────────────────────────────────`

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
- **The head and the embedding are the sensitive ends.** The embedding maps tokens into the stream; the LM head reads the final hidden state and produces logits that the softmax turns into probabilities. Errors at the embedding are multiplied by all layers; errors at the head directly perturb the training signal. This is why FP8 systems keep the embedding/head in higher precision even when the bulk GEMMs are 8-bit — and why this repo's BF16 autocast lets RMSNorm, embeddings and the CE loss run FP32 (see [[Docs/08_Training_Pipeline]]).

The practical takeaway: **FP8 is not "make everything 8-bit"** — it is a budgeted precision allocation where the per-layer error allowance is dictated by residual-stream depth, and only the bandwidth-dominated GEMMs get the 8-bit treatment.

---

## 8. FP8 vs what this repo actually does — the same ladder, one rung up

The repo's BF16 training (`training/pretrain.py:Pretrainer._amp_context`) already embodies the two structural ideas FP8 pushes further:

1. **Compute dtype ≠ master dtype.** BF16 autocast runs the GEMMs in BF16 while AdamW keeps FP32 master weights, moments, and the update math (`training/pretrain.py:Pretrainer.__init__` builds the FP32 optimizer state). FP8 would only change *which* dtype the compute GEMMs use.
2. **Accumulators stay FP32.** Every Triton kernel in this repo accumulates in `tl.float32` (see [[Docs/12_Triton_Kernels]]); PyTorch BF16 matmuls accumulate in FP32 internally. FP8 Tensor Cores do exactly the same — 8-bit multiply, FP32 accumulate.

The gap between this repo and V3 is therefore not architectural principle but *granularity of the quantize step*: BF16 needs no scale at all (its 8-bit exponent gives ±3.4e38 range natively), so the entire scale-buffer machinery of §4 vanishes. BF16 is "FP8 with the scale folded into every element" — the range is paid for in 8 extra bits per element, which at 411.6M params is a rounding error in the memory budget (0.82 GB weights vs ~12 GB optimizer state; see [[Docs/11_Operations_and_Testing]]).

`★ Insight ─────────────────────────────────────`
The decision rule in this repo: adopt FP8 when **bytes moved** (not FLOPs) is the binding constraint. At 8 × 2048 × 18 layers on one A100, activations and weights are small enough that BF16's extra bytes cost far less than FP8's engineering + stability risk. The moment the model outgrows a single GPU (or the batch doubles), the trade flips.
`─────────────────────────────────────────────────`

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

This chapter is about **training**. FP8 *inference* (weight-only W8A8 quantization) is a different, much simpler problem: weights are quantized once, offline, with calibration; activations are quantized at serving time with running-statistic scales. Training is harder because activations change every step and the *backward* pass re-introduces heavy-tailed gradients — which is why the scale machinery here is per-step and per-block, and why this repo's only 8-bit-adjacent work is the FP32-accumulator discipline in its Triton kernels (see [[Docs/12_Triton_Kernels]]). Do not conflate the two: W8A8 inference would be a plausible *future* optimization for this repo's serving path; FP8 training remains out of scope at 411.6M.

> **Next:** [[Docs/07_DualPipe_Parallelism]] — bidirectional pipeline parallelism, the other DeepSeek-V3 paper technique not implemented in this single-GPU repo.


<!-- docs:verified 2026-08-04 · 59aeef3 -->
