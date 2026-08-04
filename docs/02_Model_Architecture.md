# Architecture Overview

> **Purpose:** A single map of how every component in DeepSeek-V3-Lite fits together. Read this after [[Docs/01_Foundations|foundations]] and before diving into component-specific docs.

> **Read this if** you need the full system map. **Skip if** you only need MLA → [[Docs/03_Multi_Head_Latent_Attention|MLA]].

**Depends on:** (none — start here) · **Read next:** [[Docs/03_Multi_Head_Latent_Attention|MLA]], [[Docs/04_DeepSeekMoE|MoE]]

**60-second summary:** DeepSeek-V3-Lite is an 18-layer, decoder-only, causal language model built from three architectural pillars — MLA attention (a low-rank-compressed attention that also replaces the KV cache), an aux-loss-free MoE FFN (20 routed + 1 shared expert per layer, top-4 active), and a depth-1 MTP head that turns the trunk into a speculative decoder. The whole model is 411.6M deduplicated parameters (~185M active per token) and is trained single-GPU on 8.4B Chinchilla-optimal tokens. Every number below was verified by instantiating the canonical config on CPU and walking `count_parameters` over it.

---

## Prerequisites

You should understand (or have skimmed):
- Causal language modeling objective — [[Docs/01_Foundations|foundations]] §2
- Pre-norm residual blocks — [[Docs/01_Foundations|foundations]] §3
- Chinchilla token budget — [[Docs/01_Foundations|foundations]] §12

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

**Pedagogical implication:** When you see a 15-line function instead of a one-liner, it is usually guarding an invariant documented in [[Docs/11_Operations_and_Testing|testing]].

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

See [[Docs/10_Inference_and_Serving|inference]] and [[Docs/05_Multi_Token_Prediction|mtp]].

---

## Parameter Budget

> **60-second summary.** All three parameter-budget tables that used to live in this document have been consolidated into this one canonical table. The numbers were verified by instantiating `models/transformer.py:Transformer` from the canonical config on CPU and summing `models/transformer.py:count_parameters` per component; every row below is exact, not rounded-down prose. The per-module API surface and a second view of the same numbers live in [[reference/R2_transformer_api]].

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

> **60-second summary.** Training on one A100 80GB (`B=8, S=2048, grad_checkpoint=True`) is estimated at ~30.5 GB peak — comfortable headroom on 80 GB. The estimate comes from `utils/memory.py:estimate_model_memory_gb`, a closed-form sum of params + optimizer state + activations + static overhead. **No GPU training run has ever been executed** — every figure in this section is a budget estimate from arithmetic, not a measurement (`.benchmarks/` is empty; see [[Docs/11_Operations_and_Testing|testing]] for the formula provenance).

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

Full reference: [[Docs/08_Training_Pipeline|configs]] and [[reference/R1_config_schema]].

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
| Fit smaller GPU | ↓ `micro_batch_size`, ↓ `max_seq_len`, enable `grad_checkpoint` | [[Docs/08_Training_Pipeline|configs]] |
| Faster iteration | `pretrain_1650_2m.yaml`, `build_small_pretrain_data.py` | [[Docs/11_Operations_and_Testing|scripts]] |
| Higher MFU | `ENABLE_TRITON_KERNELS=1`, `moe_dispatch: triton_grouped` | [[Docs/12_Triton_Kernels|triton kernels]] |
| Longer context | ↑ `max_seq_len`, consider YaRN `rope_factor` | [[Docs/03_Multi_Head_Latent_Attention|MLA]] |

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

Full test mapping: [[Docs/11_Operations_and_Testing|testing]].

---

## Further Reading

- Attention deep-dive: [[Docs/03_Multi_Head_Latent_Attention|MLA]]
- MoE deep-dive: [[Docs/04_DeepSeekMoE|moe]]
- Training loop: [[Docs/08_Training_Pipeline|training]]
- Learning path: [[Docs/00_Getting_Started|getting started]]

<!-- docs:verified 2026-08-01 · e8553c4 -->


---

# Transformer — Top-Level Wiring

## A Comprehensive Technical Reference

> **Prerequisites:** [[Docs/01_Foundations|foundations]], [[Docs/03_Multi_Head_Latent_Attention|MLA]], [[Docs/04_DeepSeekMoE|moe]].

> **Covers**: `Transformer`, `TransformerBlock`, `SwiGLUFFN` in `models/transformer.py` — how the full DeepSeek-V3 stack is assembled and executed.

> **Read this if** you need the end-to-end forward/generate wiring. **Read first** if new to the repo: [[Docs/02_Model_Architecture|architecture]] → [[Docs/03_Multi_Head_Latent_Attention|MLA]] / [[Docs/04_DeepSeekMoE|moe]].

---

## Abstract

The `Transformer` class is the root `nn.Module` for DeepSeek-V3-Lite. It stacks 18 `TransformerBlock` layers (2 dense SwiGLU + 16 MoE), each consisting of pre-norm MLA attention and pre-norm FFN. It exposes three critical interfaces: `forward` (training/inference logits), `forward_with_hidden` (MTP training), and `generate` (autoregressive decode with KV cache). This part walks the wiring top to bottom: construction, the residual architecture, every forward contract, the mask cache, weight tying, generation, checkpointing, and — new in this edition — a per-module tensor-shape walkthrough ([§28](#tensor-shape-walkthrough-per-module)) and a full FLOP accounting ([§18](#flop-accounting-per-layer-type)). The per-symbol API reference lives in [[reference/R2_transformer_api]].

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

(The `q_nope` absorption bmm $2 \cdot H \cdot d_{nope} \cdot R = 221\,184$ appears only in the manual path; the SDPA path materialises $K_{nope}$ instead. Details in [[Docs/03_Multi_Head_Latent_Attention|MLA]].)

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

**Decode regime:** with $S_q = 1$ the attention core collapses to $2 H (d_{qk} + d_v) = 3\,264$ FLOPs/token; decode cost is the projection budget (~3.35 MFLOP/token) and — dominantly — **memory bandwidth** reading $216$ cache floats per token per layer. Decode is bandwidth-bound, not FLOP-bound; see [[Docs/10_Inference_and_Serving|inference]].

**Training budget:** $0.49 \text{ GFLOP/token} \times 8.4\text{B tokens} \approx 4.1$ EFLOPs forward; with backward ≈ 2× forward plus grad-checkpoint recompute of the forward, ~12 EFLOPs total. On an A100 (312 TFLOP/s BF16 dense) that is ~39,700 s at 100% MFU, ~27–32 h at 35–40% MFU. The config's stated target is 13–15 h at 35–40% MFU, which implies ~75% MFU — **optimistic**. Both figures are estimates; nothing has been measured yet. [[guides/G4_benchmarking]] covers what a real measurement must capture (activation recompute, allocator churn, the logits transient).

### Sequence-level view and the three regimes

Per-sequence forward at $S{=}2048$: $2048 \times 490.6\text{ MFLOP} \approx 1.0$ TFLOPs; 8.4B tokens is $4.1 \times 10^6$ sequences, and $4.1 \times 10^6 \times 1.0$ TFLOPs $= 4.1$ EFLOPs — consistent with the per-token budget above.

| Regime | $S_q$ | Attention core per token | Total per token | What dominates |
|---|---|---|---|---|
| Training / prefill ($S{=}2048$) | 2048 | 6.68 MFLOP | ~490 MFLOP | LM head (153.6M) + 18× attention (180.7M) |
| Prefill with cache ($S{=}2048$) | 2048 | 6.68 MFLOP | ~490 MFLOP | identical to training minus backward |
| Decode step ($S_q{=}1$) | 1 | 3.3 KFLOP | ~3.4 MFLOP | projections; attention core vanishes |

The table makes the two regimes legible: **prefill is compute-bound and quadratic in $S$; decode is memory-bound and linear in $S$** (each step reads the whole cache). At $S{=}2048$ the training attention core ($2 H S (d_{qk}+d_v)$) is only 1.4% of the per-token FLOPs — the logits bottleneck, not attention, sets this model's compute profile. That is a scale-dependent statement: at $S{=}4096$+ and larger $d$ the attention terms grow quadratically and eventually dominate.

**MFU arithmetic** (so the number is checkable): $12\text{ EFLOP} / (0.40 \times 312\text{ TFLOP/s}) = 96\,200\text{ s} \approx 27\text{ h}$. Getting to 13–15 h would require 0.75–0.85 MFU, which is above what dense BF16 transformers typically sustain on one A100 — MoE dispatch overhead and the logits GEMM (thin, memory-bound) push the other way. [[guides/G4_benchmarking]] explains how to measure rather than guess.

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

**Sampling theory** (temperature, top-k, top-p semantics and why the ordering matters) is derived in [[Docs/10_Inference_and_Serving|inference]].

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

`Pretrainer` passes the full YAML dict; `Transformer` unwraps internally. MLA/MoE receive the unwrapped flat dict via `TransformerBlock`. The canonical config is documented key-by-key in [[reference/R1_config_schema]] and `training/pretrain.py:TrainingConfig`.

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

The **manual path** (non-SDPA `attn_impl`) replaces steps 12–13 with the absorption trick: `q_nope` is projected through `wkv_b_k` first (`models/mla.py:MultiHeadLatentAttention._per_batch_bmm`), scores are computed directly against the latent `ctx_kv` in $R{=}192$ space, and the attention output is up-projected through `wkv_b_v` after the softmax. Full algebra in [[Docs/03_Multi_Head_Latent_Attention|MLA]].

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
- **The gate's `bias` buffer never receives gradients.** `update_gate_bias` (`models/moe.py:DeepSeekMoE.update_gate_bias`) moves counts into `AuxLossFreeGate.update_bias` (`models/moe.py:AuxLossFreeGate.update_bias`), which bumps the buffer up/down by `bias_update_speed` under `torch.no_grad()`. Load balancing is a control loop, not an optimiser term. The Triton grouped path (`models/moe_triton.py`) keeps the same sort/scatter contract; see [[Docs/12_Triton_Kernels|triton kernels]].

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

The alignment algebra (`usable = S − d − 2`, inputs at offsets $d{+}1$, targets at $d{+}2$) is the length contract between the trunk and the side branch — full derivation in [[Docs/05_Multi_Token_Prediction|MTP]]. Note step 6 is regular multi-head attention, not MLA: the MTP block is small (1.7% of params) and its cache-free training path does not need MLA's bandwidth trick.

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

**Q: What happens if I set `rope_factor > 1`?** RoPE frequencies are recomputed for a longer context (`_extend_rope`), and `mscale` kicks in: `mscale = 0.1 * mscale_raw * log(rope_factor) + 1.0`; when `max_seq_len > 4096` the softmax scale becomes `(qk_head_dim ** -0.5) * (mscale ** 2)`. This is the YaRN contract — see [[Docs/03_Multi_Head_Latent_Attention|MLA]].

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

## References

- [[Docs/02_Model_Architecture|transformer]] → [[Docs/03_Multi_Head_Latent_Attention|MLA]], [[Docs/04_DeepSeekMoE|moe]]
- `models/transformer.py` — authoritative source
- [[Docs/02_Model_Architecture|architecture]] — system-level overview
- [[reference/R2_transformer_api]] — per-symbol API reference

<!-- docs:verified 2026-08-01 · e8553c4 -->

---

# Design Rationale and Transformer Block Details

> **Design Rationale & Architecture Details.** This section consolidates unique architectural design choices and transformer block topology details. Cross-portfolio comparison lives in [[Docs/13_Portfolio_Comparison|Portfolio Comparison]].

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

so instead of materialising $k$ for every key (a $(S \times d_{nope})$ tensor per layer per step) the model pre-computes $\tilde q = W_{b,k}^\top q$ per query — $H \cdot d_{nope} \times R$ FLOPs once, not per key — and scores queries directly against the 192-dim latent $c$. That is exactly the `q_nope_proj = torch.bmm(q_nope_h, wkv_b_k)` step in `models/mla.py:MultiHeadLatentAttention.forward` (manual path). The SDPA path makes the mirror-image choice: materialise $K_{nope}$ once per forward and let FlashAttention-style kernels score in 72 dims. Both are correct; they sit at different points on the FLOPs↔bandwidth curve ([§18](#flop-accounting-per-layer-type) quantifies the 3× attention-core gap). The full two-direction derivation is [[Docs/03_Multi_Head_Latent_Attention|§4 of the MLA chapter]].

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
| FP8 training | Paper-spec only; BF16 fits on one A100 — see [FP8](06_FP8_Mixed_Precision.md) |

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

Without tying: `2 × 100,018 × 768 = 153.6M` (embedding + head).
With tying: `100,018 × 768 = 76.8M` (single shared tensor).
**Savings: 76.8M** — 18.7% of the 411.6M total.

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

Returns the **pre-final-norm** trunk hidden — `MTPBlock` applies its own `RMSNorm` before fusion (see [[Docs/05_Multi_Token_Prediction|MTP]]).

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

<!-- docs:verified 2026-08-04 · 59aeef3 -->
