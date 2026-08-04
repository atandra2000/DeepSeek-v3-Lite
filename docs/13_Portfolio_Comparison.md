# Portfolio Architecture Comparison

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

> **Param-count note:** "422M" survives only as the config filename (`pretrain_a100_422m.yaml`); the verified deduplicated counts are 411.6M base and 418.7M with the MTP head (MTP adds ~7.1M). ~185M parameters are active per token. See [[Docs/02_Model_Architecture|Model Architecture]] §6 for the full budget.

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

Every attention variant is a different answer to one question: **what does the model keep around for past tokens?** That answer determines both how much memory a long conversation costs and how many bytes each decode step must read from HBM. The table below lays the four families side by side on this single axis, using the canonical 411.6M config's dimensions (derivation in [[Docs/03_Multi_Head_Latent_Attention|MLA]] §10):

| Property | MHA | GQA (4 groups) | MQA | **MLA (this repo)** | Linear attention / SSM |
|---|---|---|---|---|---|
| Cached per token per layer | 1,536 floats ($2 \cdot H \cdot d_v$) | 512 ($2 \cdot 4 \cdot 64$) | 128 ($2 \cdot 64$) | **216** ($R=192 + d_{\text{rope}}=24$) | constant state, not a cache |
| Growth with context | $O(S)$ | $O(S)$ | $O(S)$ | $O(S)$ | $O(1)$ |
| Reduction vs MHA | 1× | ~3× | 12× | **7.1×** | unbounded (state independent of $S$) |
| Quality vs MHA | baseline | slight drop | measurable drop | **matches MHA** | different mechanism — not directly comparable |
| Decode compute | lowest | low | low | ~4× MHA | low (recurrent scan) |
| Decode memory bandwidth | highest | medium | low | **lowest** | **lowest** (fixed-size state read) |

> **Footnote on ratios:** the per-token float counts are exact for this config. The GQA ratio is 3.0× on the $d_v=64$ baseline used here; [[Docs/03_Multi_Head_Latent_Attention|MLA]] §10 rounds it to 3.3× on a $d_{qk}=72$ baseline. The MLA 7.1× figure is the canonical one used throughout these docs.

**What the axis teaches.** LLaMA-3-Lite implements the GQA column (2× group ratio, §1) — a fixed sharing pattern that cannot compress below one KV head. GPT-OSS-Lite combines GQA with sliding-window alternation, cutting the *number of positions cached* (windowed layers keep 128) rather than the bytes per position. Mamba-3-Lite sits at the linear-attention end: a fixed-size recurrent state ($N=64$ complex64 per layer) replaces the cache entirely, so memory is $O(1)$ in context — hence no RoPE and no extrapolation story (§4). MLA is the only entry that compresses *into a latent*: 216 floats as a learned low-rank code, not a subsample of the K/V heads — which is why quality holds where GQA/MQA give ground.

**The counter-intuitive bit: cache size and decode speed are different axes.** A smaller cache only speeds up decode if each step also *reads* less. The absorption trick (`models/mla.py:MultiHeadLatentAttention.forward`) folds the key/value up-projections into the query and output projections, so scores are computed directly in latent space — each step reads the 216 cached floats, never the 1,536 they expand to. The trade hidden in the table: MLA's decode *compute* is higher than GQA's (~4× MHA — the absorbed query runs over $R=192$ and every score is an $R$-dimensional inner product). It wins anyway because decode is bandwidth-bound, not FLOP-bound ([[Docs/02_Model_Architecture|Model Architecture]] §39). Absorption is inference-only: during training the same method materialises full K/V so gradients reach the up-projection weights.

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

**The benefit.** Parameter capacity decouples from FLOPs: DeepSeek-V3-Lite stores 411.6M deduplicated params but only ~185.1M are exercised per token (measured by instantiating the canonical config — [[Docs/02_Model_Architecture|Model Architecture]] §42), and each of the 16 MoE layers executes only 4 routed + 1 shared of its 21 experts. LLaMA-3-Lite's dense stack activates everything on every token — simple and predictable, but capacity is priced in FLOPs. The 411.6M/185.1M split is a 2.2× capacity multiplier over active compute.

**The costs.**

1. **Every expert lives in memory.** VRAM, checkpoint size, and load time track *stored* params, not active ones — sparsity saves compute, never memory.
2. **Sparsity without balance collapses.** If one expert wins slightly, it trains faster and wins more — a positive feedback loop that silently burns capacity ([[Docs/04_DeepSeekMoE|MoE]] §2). DeepSeek-V3-Lite's answer is the out-of-band bias controller (`models/moe.py:AuxLossFreeGate.forward`); GPT-OSS-Lite's is the auxiliary loss.
3. **Dispatch overhead and imbalance.** Tokens are sorted into per-expert batches, and the busiest expert's chunk sizes the wall-clock; this repo drops no tokens, so imbalance costs time, never correctness ([[Docs/04_DeepSeekMoE|MoE]] §5).

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

1. **MLA instead of GQA — because decode is bandwidth-bound even at 411.6M.** The 7.1× cache cut with absorption (`models/mla.py:MultiHeadLatentAttention.forward`) is the largest per-token memory win available, at zero quality cost — GQA's fixed head-sharing caps at one KV head and degrades before reaching it ([[Docs/02_Model_Architecture|Model Architecture]] §39). At 2,048 training context the absolute cache is small; the choice is about the long-context decode regime, where `models/transformer.py:Transformer.generate` reads the latent cache one token at a time ([[Docs/10_Inference_and_Serving|Inference]]).

2. **MoE instead of dense — because capacity, not FLOPs, is the scarce resource at 8.4B tokens.** Storing 411.6M and activating ~185.1M (≈45%) is a 2.2× capacity multiplier over active compute (§3.1), with no quality-reducing trickery in the attention stack. The aux-loss-free bias controller keeps the task gradient pure — a deliberate contrast with GPT-OSS-Lite's aux loss.

3. **Fine-grained experts, top-4 of 20.** With $C(20, 4) = 4\,845$ routable subsets per token versus GPT-OSS-Lite's $C(8, 2) = 28$, the router composes far more specialised behaviour from the same capacity class ([[Docs/04_DeepSeekMoE|MoE]] §3).

4. **Two dense layers up front.** Routing on raw early representations is noisy — the first layers handle universal token-level patterns densely, and only layers 2–17 route ([[Docs/02_Model_Architecture|Model Architecture]] §43).

5. **MTP — a regularizer that pays for itself at inference.** Training adds a depth-1 head predicting token $t+2$ (`models/mtp.py:MultiTokenPrediction.forward`, loss weight 0.3) for ~7.1M extra params (≈1.7% of 418.7M). At decode, the same head drafts tokens that `inference/speculative.py:SpeculativeDecoder.generate_step` verifies against the trunk — the training overhead converts into decode speedup ([[Docs/05_Multi_Token_Prediction|MTP]]).

6. **μP LR — one hyperparameter that transfers across scales.** $6.0e-4 \times \sqrt{757226496 / N}$ gives 8.14e-4 (base) and 8.07e-4 (with MTP), so the training recipe survives model-size changes without retuning ([[Docs/08_Training_Pipeline|Training]]).

**What the portfolio adds up to.** Each sibling attacks a different bottleneck: GPT-OSS-Lite attacks context (train-time YaRN to 128K), LLaMA-3-Lite attacks memory engineering (78% peak-memory reduction), Mamba-3-Lite attacks the quadratic core itself (constant-state SSD). DeepSeek-V3-Lite attacks the two structural costs — the KV cache and the dense FFN — while keeping exact-quality attention and a paper-faithful training story. The four are not four implementations of one idea; they are four points in the design space, and T-chapters 02–05 and 08 are the deep dives behind the columns of §1.

---

## 9. Check Your Understanding

**Q1. Why does MLA match MHA quality while GQA and MQA degrade?**

A1. GQA/MQA reduce the cache by *sharing or dropping* key/value heads — a fixed, information-losing subsample. MLA's latent is produced by a learned compression matrix, and the per-head K/V are recovered by a learned up-projection; the absorption algebra ([[Docs/03_Multi_Head_Latent_Attention|MLA]] §5) shows the recovered keys/values are exactly what materialised MHA would compute. The compression is learned, not structural.

**Q2. A model advertises "3× KV-cache reduction". What must you ask before believing it?**

A2. Against which baseline? The same mechanism is 3.0× against a $d_v=64$ MHA baseline and 3.3× against a $d_{qk}=72$ one (§2.3). Then ask what decode *reads* per step, not just what it stores — absorption is what turns MLA's 216 stored floats into a bandwidth win.

**Q3. DeepSeek-V3-Lite stores 411.6M params. Why is it wrong to call it "a 411.6M-FLOPs-per-token model"?**

A3. Because only ~185.1M (≈45%) are active per token — the rest is routed-expert capacity that wakes only for tokens that select it. FLOPs track active params (top-4 + shared), while memory and checkpoints track stored params (§3.1).

---

## Summary

DeepSeek-V3-Lite is the **attention compression** and **MoE balancing** project in the portfolio:
1. [[Docs/03_Multi_Head_Latent_Attention|MLA]] — the only low-rank KV compression with the absorption trick.
2. [[Docs/04_DeepSeekMoE|MoE]] — the only control-theory-based load balancing.
3. [[Docs/05_Multi_Token_Prediction|MTP]] — the only multi-token prediction with speculative decoding.
4. [[Docs/08_Training_Pipeline|Training]] — the only principled μP LR scaling across model sizes.

Combined with GPT-OSS-Lite (sinks + SWA + YaRN), LLaMA-3-Lite (GQA + memory optimization), and Mamba-3-Lite (SSM), the portfolio covers the full spectrum of modern LLM architecture innovations — each project mechanistically distinct, each teaching a different fundamental approach.

<!-- docs:verified 2026-08-04 · 59aeef3 -->
