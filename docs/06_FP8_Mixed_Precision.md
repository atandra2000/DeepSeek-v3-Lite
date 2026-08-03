# 06 — FP8 Mixed Precision Training & Quantization

> **Canonical** for the DeepSeek-V3 FP8 scheme: E4M3/E5M3 formats, tile-wise and block-wise fine-grained scaling, and the high-precision accumulation that makes 8-bit training stable. Educational textbook chapter — from-scratch, with the math and the intuition.

> How DeepSeek-V3 cuts matmul memory bandwidth and Tensor Core throughput in half using 8-bit floats, and why a single global scale factor fails. **Status in this repo:** FP8 is **paper-spec only — not implemented.** This reproduction trains in BF16 throughout (see [[Docs/08_Training_Pipeline]]). This chapter exists so the portfolio documents the technique that makes full-scale DeepSeek-V3 feasible; at 422 M params on a single A100, BF16 is already compute-efficient and FP8's complexity is not justified.

**Depends on:** [[Docs/02_Model_Architecture]], [[Docs/03_Multi_Head_Latent_Attention]] · **Read next:** [[Docs/07_DualPipe_Parallelism]]

---

## 0. Status in this repo

| Aspect | DeepSeek-V3 (paper) | DeepSeek-v3-Lite (this repo) |
|---|---|---|
| Forward GEMM dtype | FP8 E4M3 | BF16 |
| Backward GEMM dtype | FP8 E5M2 | BF16 |
| Weight scaling | block-wise `128×128` | n/a (BF16) |
| Activation scaling | tile-wise `1×128` | n/a (BF16) |
| Master weights / optim | FP32 | FP32 (AdamW master + m + v) |
| Reason to skip | — | 422 M params fits BF16 on one A100; FP8's stability engineering is disproportionate at this scale |

The rest of this chapter explains the V3 scheme so the technique is documented; it is not a description of running code.

---

## 1. Why 8-bit floats — the bandwidth wall

A matmul `Y = X W` is memory-bound for anything but the largest GEMMs: the time is dominated by **loading** `X` and `W` from HBM, not by the multiply-adds. Halving the storage dtype halves the bytes moved:

| dtype | bytes/element | Relative bandwidth |
|---|---|---|
| FP32 | 4 | 1.0× |
| BF16 / FP16 | 2 | 2.0× |
| **FP8 (E4M3 / E5M2)** | **1** | **4.0×** |

Modern Tensor Cores (Hopper `fp8` tensor cores) also issue twice the FLOPs/s in FP8 vs BF16. So FP8 is a ~2× throughput and ~2× memory win simultaneously — the single largest engineering lever in DeepSeek-V3's training cost reduction.

The catch: 8 bits give you either **range** (E5M2: ±57 344) or **precision** (E4M3: ±448 with 3 mantissa bits), not both. LLM activations and gradients have outlier channels that a single 8-bit scale cannot represent without either saturating the outliers or quantizing the inliers to zero. That tension is the entire problem FP8 training solves.

---

## 2. The two FP8 formats

```
E4M3  (forward — weights & activations)          Range: [-448, 448]
┌──────┬─────────────┬──────────────┐
│ sign │ exponent 4b │ mantissa 3b  │   More mantissa → finer steps,
└──────┴─────────────┴──────────────┘   small range. Good for values near 1 (weights/activations).

E5M2  (backward — gradients)                    Range: [-57344, 57344]
┌──────┬─────────────┬──────────────┐
│ sign │ exponent 5b │ mantissa 2b  │   More exponent → huge range,
└──────┴─────────────┴──────────────┘   coarse steps. Good for gradients that span orders of magnitude.
```

| Property | BF16 | FP8 E4M3 | FP8 E5M2 |
|---|---|---|---|
| Total bits | 16 | 8 | 8 |
| Exponent | 8 | 4 | 5 |
| Mantissa | 7 | 3 | 2 |
| Dynamic range | wide | narrow (±448) | wide (±57 344) |
| Mantissa precision | medium | high (3 bits) | low (2 bits) |
| Used for | master weights, RMSNorm | forward `W, X` GEMMs | backward `∇Y` GEMMs |

`★ Insight ─────────────────────────────────────`
The split is **asymmetric on purpose**. Forward activations/weights are well-behaved, bounded distributions centred near 1 — they benefit from *precision* (E4M3's 3 mantissa bits) and don't need range. Backward *gradients* are notoriously heavy-tailed — a few channels carry orders-of-magnitude larger gradients than the rest — so they need *range* (E5M2's 5 exponent bits) and tolerate coarse steps. Using one format for both would waste either precision or range; using two formats matches each phase's distribution.
`─────────────────────────────────────────────────`

### 2.1 The quantize/de-quantize primitive

Every FP8 GEMM is really a dequant→matmul→quant dance with the scale folded in:

$$X_{\text{FP8}} = \text{clip}\!\left(\text{round}\!\left(\frac{X}{S}\right), -448, 448\right), \qquad \hat{X} = X_{\text{FP8}} \times S$$

The scale `S` is the whole game. The question FP8 training answers is: **what is `S`, and can one `S` work for a whole matrix?** (Spoiler: no.)

---

## 3. Why a single global scale fails — the outlier problem

Naive per-tensor quantization picks one `S = max(|X|) / 448` for the entire matrix. Consider an activation matrix where 99% of channels have magnitude ~1 and 1% are outliers at ~100:

- `S = 100/448 ≈ 0.223`. The 99% inliers map to `round(1/0.223) = 4` → dequant `4 × 0.223 = 0.89`. **You've lost ~11% of every normal value to rounding**, and with only 3 mantissa bits the inlier resolution is now ~0.223 — coarser than the signal.
- The outlier (100) maps to `448` (clipped) — fine.

The inliers, which carry the actual signal, get quantized to noise. This is the **outlier channel** problem documented across every LLM quantization paper: a tiny fraction of channels forces a scale that destroys everyone else.

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
3. E4M3↔E5M2 casting at the forward/backward boundary.
4. An autocast context that swaps dtypes per GEMM while keeping master weights FP32.
5. Stability instrumentation: overflow/underflow counters, scale-histogram logging, fallback to BF16 on divergence.

For a 422 M-param model on a single A100, BF16 already saturates the compute efficiently and the memory fits with room to spare — the ~2× FP8 win buys throughput that isn't the bottleneck here. The bottleneck at this scale is data pipeline and launch overhead, which FP8 does not address. So FP8 is **documented but deliberately not built**; it is the natural next lever if the model were scaled up to where BF16 memory or throughput became the constraint.

> **Next:** [[Docs/07_DualPipe_Parallelism]] — bidirectional pipeline parallelism, the other DeepSeek-V3 paper technique not implemented in this single-GPU repo.