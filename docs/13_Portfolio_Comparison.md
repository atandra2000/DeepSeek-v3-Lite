# Portfolio Architecture Comparison

> **Purpose:** Compare DeepSeek-V3-Lite's architectural choices with its sibling projects in the CoreProjects portfolio: GPT-OSS-Lite, LLaMA-3-Lite, and Mamba-3-Lite. Each project is mechanistically distinct and addresses core LLM design challenges differently.

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
| **Total params** | ~422M | ~502M | — | — |
| **Training context** | 2,048 | 4,096 | 2,048 | — |
| **Eval context** | YaRN-scaled | 131,072 | — | Constant-state |

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

## Summary

DeepSeek-V3-Lite is the **attention compression** and **MoE balancing** project in the portfolio:
1. [[Docs/03_Multi_Head_Latent_Attention|MLA]] — the only low-rank KV compression with the absorption trick.
2. [[Docs/04_DeepSeekMoE|MoE]] — the only control-theory-based load balancing.
3. [[Docs/05_Multi_Token_Prediction|MTP]] — the only multi-token prediction with speculative decoding.
4. [[Docs/08_Training_Pipeline|Training]] — the only principled μP LR scaling across model sizes.

Combined with GPT-OSS-Lite (sinks + SWA + YaRN), LLaMA-3-Lite (GQA + memory optimization), and Mamba-3-Lite (SSM), the portfolio covers the full spectrum of modern LLM architecture innovations — each project mechanistically distinct, each teaching a different fundamental approach.
