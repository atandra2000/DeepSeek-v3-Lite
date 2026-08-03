# Architecture Overview

> **Purpose:** A single map of how every component in DeepSeek-V3-Lite fits together. Read this after [[Docs/01_Foundations|foundations]] and before diving into component-specific docs.

> **Read this if** you need the full system map. **Skip if** you only need MLA → [[Docs/03_Multi_Head_Latent_Attention|MLA]].

**Depends on:** (none — start here) · **Read next:** [[Docs/03_Multi_Head_Latent_Attention|MLA]], [[Docs/04_DeepSeekMoE|MoE]]

---

## Prerequisites

You should understand (or have skimmed):
- Causal language modeling objective — [[Docs/01_Foundations|foundations]] §1
- Pre-norm residual blocks — [[Docs/01_Foundations|foundations]] §3
- Chinchilla token budget — [[Docs/01_Foundations|foundations]] §12

---

## Table of Contents

1. [Design Goals](#design-goals)
2. [System Diagram](#system-diagram)
3. [Layer Topology](#layer-topology)
4. [Data Flow — Training](#data-flow--training)
5. [Data Flow — Inference](#data-flow--inference)
6. [Parameter Budget](#parameter-budget)
7. [Memory Budget](#memory-budget)
8. [File Map](#file-map)
9. [Config → Code Routing](#config--code-routing)
10. [Module Dependency Graph](#module-dependency-graph)
11. [Scaling Knobs](#scaling-knobs)
12. [Request Lifecycles](#request-lifecycles)
13. [Load-Bearing Invariants](#load-bearing-invariants)
14. [Further Reading](#further-reading)

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

422M canonical config (`configs/pretrain_a100_422m.yaml`):

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

At 422M scale: `n_dense_layers=2` of 18 total. The 1650 smoke config preserves the same pattern at 2 of 4 layers.

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

`SpeculativeDecoder.generate_step` (`inference/speculative.py`) implements a simplified accept rule:

$$
\text{accept}(t_2) \iff p_{\text{main}}(t_2) \geq \tau \cdot p_{\text{draft}}(t_2)
$$

with $\tau = 0.8$ default. This is **not** the full optimal speculative sampling algorithm (Leviathan et al.) but a pedagogical approximation that preserves correctness (reject → fall back to one token).

See [[Docs/10_Inference_and_Serving|inference]] and [[Docs/05_Multi_Token_Prediction|mtp]].

---

## Parameter Budget

Approximate breakdown for 422M (with weight tying):

| Component | Params | Share |
|---|---|---|
| MoE FFN (16 layers × ~18.6M) | ~298M | ~70% |
| Embedding + LM head (tied) | ~77M | ~18% |
| MLA attention (18 layers) | ~35M | ~8% |
| Dense SwiGLU (2 layers) | ~7M | ~2% |
| MTP heads | ~3M | ~1% |
| RMSNorm + gates | ~1M | <1% |

MoE dominates because each of 16 layers stores 21 SwiGLU experts (20 routed + 1 shared), but only 5 execute per token.

### Derivation sketch

**MoE expert params** (one SwiGLU expert): $3 \times d \times I = 3 \times 768 \times 384 = 884736$.

Per MoE layer: $(20 + 1) \times 884736 \approx 18.6$M. Sixteen MoE layers: $\approx 298$M.

**Embedding + tied head:** $100018 \times 768 \approx 77$M (counted once).

**MLA per layer (approx.):** $4dR + 2d(q_{\mathrm{k,nope}} + q_{\mathrm{k,rope}})H + \ldots \approx 2$M × 18 layers.

Full component breakdown logged at train init when `log_per_component_params: true`.

---

## Memory Budget

Training on A100 80GB (`B=8, S=2048, grad_checkpoint=True`):

| Component | ~GB |
|---|---|
| BF16 params | 0.8 |
| AdamW FP32 state | 5.1 |
| Activations (×24 factor) | 10.9 |
| CUDA overhead | 13.6 |
| **Total estimate** | **~30.5** |

Measured peak: ~35 GB (working set spikes during compile and backward recompute).

See [[Docs/11_Operations_and_Testing|utils]] for formulas.

### Why measured > estimated

Analytical estimate (`utils/memory.py`) sums:

$$
M_{\text{total}} = M_{\text{params}} + M_{\text{optim}} + M_{\text{act}} + M_{\text{overhead}}
$$

**Sources of underestimate:**
1. `torch.compile` workspace (not modelled)
2. CUDA caching allocator retaining freed blocks
3. MoE dispatch temporaries (stacked path copies expert weights)
4. Gradient buckets during backward

Rule of thumb: add **15–20% headroom** above estimate before launching multi-hour runs.

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
| `training/pretrain.py` | 441 | Train loop |
| `models/mla_triton.py` | 323 | Fused MLA attention kernel |
| `models/mla.py` | 225 | MLA attention + cache |
| `models/moe.py` | 207 | Aux-loss-free MoE |
| `models/transformer.py` | 175 | Stack + generate |
| `models/mtp.py` | 117 | MTP heads |
| `utils/checkpoint.py` | 109 | Atomic I/O |
| `inference/generate.py` | 125 | Interactive CLI |
| `inference/speculative.py` | 56 | Speculative decode |

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

Full reference: [[Docs/08_Training_Pipeline|configs]].

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

### Width / depth trade-offs (422M → larger)

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
| Triton gated by env var | `models/_triton_dispatch.py` | `test_force_back.py` |
| Weight tying dedup on save | `Pretrainer.save_checkpoint` | checkpoint roundtrip tests |
| μP LR after MTP wrap | `Pretrainer.__init__` | `test_mup_lr_scaling` |

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

## Table of Contents

1. [Abstract](#abstract)
2. [Pre-Norm Residual Architecture](#pre-norm-residual-architecture)
3. [Layer Topology (422M)](#layer-topology-422m)
4. [SwiGLUFFN](#swiglu-ffn)
5. [TransformerBlock](#transformerblock)
6. [Transformer Class](#transformer-class)
7. [Forward Contracts](#forward-contracts)
8. [Causal Mask Cache](#causal-mask-cache)
9. [Weight Tying](#weight-tying)
10. [Generation API](#generation-api)
11. [Gradient Checkpointing](#gradient-checkpointing)
12. [Config Shape](#config-shape)
13. [Appendix A — Tensor shape trace](#appendix-a--tensor-shape-trace)
14. [Appendix B — FAQ](#appendix-b--faq)
15. [Appendix C — Glossary](#appendix-c--glossary)
16. [Load-Bearing Invariants](#load-bearing-invariants)
17. [References](#references)

---

## Abstract

The `Transformer` class is the root `nn.Module` for DeepSeek-V3-Lite. It stacks 18 `TransformerBlock` layers (2 dense SwiGLU + 16 MoE), each consisting of pre-norm MLA attention and pre-norm FFN. It exposes three critical interfaces: `forward` (training/inference logits), `forward_with_hidden` (MTP training), and `generate` (autoregressive decode with KV cache).



---

## Line-by-Line Construction Walkthrough

### `Transformer.__init__` (`models/transformer.py`)

```python
model_cfg = config.get("model", config)
enforce_triton_env_var(model_cfg, print)  # AGENTS rule #7
```

**Step 1 — Config unwrap:** Tests pass flat dicts; YAML passes nested `{"model": {...}}`. Single unwrap point prevents double-nesting bugs.

**Step 2 — Triton guard:** If `attn_impl=triton` or `moe_dispatch=triton_grouped` without `ENABLE_TRITON_KERNELS=1`, force-back to `sdpa`/`stacked` with one warning.

**Step 3 — Embedding:** `nn.Embedding(vocab_size, dim)` with `N(0, 0.006²)` init.

**Step 4 — Layers:** `ModuleList([TransformerBlock(i, model_cfg) for i in range(n_layers)])`. Layer index determines dense vs MoE.

**Step 5 — Head + tying:** `Linear(dim, vocab_size, bias=False)`. If `weight_tying`: `head.weight = embed.weight` (same storage).

### `TransformerBlock.forward`

```python
x = x + self.attn(self.attn_norm(x), start_pos, mask, use_cache)
x = x + self.ffn(self.ffn_norm(x))
```

**Residual highway:** $\mathbf{x}$ shape $(B, S, d)$ unchanged across both sub-layers. Gradients flow through `+` directly to earlier layers (pre-norm benefit).

**MoE note:** `DeepSeekMoE.forward(x)` flattens to $(B \cdot S, d)$ internally for routing — position-agnostic expert assignment.

### `count_parameters` — deduplication logic

```python
seen = set()
for p in model.parameters():
    if id(p) not in seen:
        seen.add(id(p))
        total += p.numel()
```

Weight tying means `embed.weight` and `head.weight` share `id` — counted once. Critical for μP LR denominator.

---

## FLOP Breakdown Per Layer (422M)

| Layer type | Attention FLOPs/token | FFN FLOPs/token |
|---|---|---|
| Dense (0-1) | $\approx 4 S d^2$ (MLA compressed) | $6 d \cdot I = 6 \times 768 \times 1536$ |
| MoE (2-17) | same | $6 d \cdot I_{\text{moe}} \times k = 6 \times 768 \times 384 \times 4$ routed + shared |

MoE **executes** 5 experts but **stores** 21 — memory bound, not compute bound at inference.

---

## Comparison with GPT-2 / LLaMA Block

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

Called at `Transformer.__init__` and mirrored in `Pretrainer`. Ensures default-config runs never silently enable Triton (AGENTS.md hard rule). Tests in `test_force_back.py` lock this behaviour.


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

---

## Layer Topology (422M)

| Layer ID | FFN type | FFN width | Attention |
|---|---|---|---|
| 0, 1 | `SwiGLUFFN` | `inter_dim=1536` | MLA |
| 2–17 | `DeepSeekMoE` | `moe_inter_dim=384` × 21 experts | MLA |

Selection logic in `TransformerBlock.__init__`:

```python
self.ffn = SwiGLUFFN(dim, inter_dim) if layer_id < n_dense_layers else DeepSeekMoE(config)
```

With `n_dense_layers=2`, layers 0–1 are dense, 2–17 are MoE.

---

## SwiGLUFFN

Dense FFN used in layers 0–1 and as the building block inside each MoE expert:

```
SwiGLU(x) = W₂ · silu(W₁x) ⊙ W₃x
```

```python
class SwiGLUFFN(nn.Module):
    w1: Linear(dim → inter_dim)
    w2: Linear(inter_dim → dim)
    w3: Linear(dim → inter_dim)
```

**Parameter count (dense layer):** `3 × dim × inter_dim = 3 × 768 × 1536 ≈ 3.5M`.

**FLOPs per token:** `≈ 6 × dim × inter_dim` (three matmuls, gate+up share input).

---

### The residual stream — central design object

Anthropic's "residual stream" framing applies directly: each layer reads from and writes to the same tensor $\mathbf{x} \in \mathbb{R}^{B \times S \times d}$. Attention and FFN are **updates** to this stream, not replacements.

$$
\mathbf{x}_{\ell+1} = \mathbf{x}_\ell + \mathrm{Attn}_\ell(\mathrm{RMSNorm}(\mathbf{x}_\ell)) + \mathrm{FFN}_\ell(\mathrm{RMSNorm}(\mathbf{x}_\ell'))
$$

At 422M: $d=768$, $L=18$. The stream width never changes — only the information content grows deeper in the stack.

---

## TransformerBlock

```python
class TransformerBlock(nn.Module):
    attn_norm: RMSNorm(dim)
    attn: MultiHeadLatentAttention(config, layer_id)
    ffn_norm: RMSNorm(dim)
    ffn: SwiGLUFFN | DeepSeekMoE
```

Forward:

```python
def forward(x, start_pos, mask, use_cache):
    x = x + self.attn(self.attn_norm(x), start_pos, mask, use_cache)
    x = x + self.ffn(self.ffn_norm(x))
    return x
```

**Note:** MoE `DeepSeekMoE.forward` only takes `x` — no `start_pos` or cache. Routing is position-agnostic (token-level, not sequence-level).

---

## Transformer Class

### Construction

```python
Transformer(config, use_checkpoint=False)
```

1. Unwrap nested config: `model_cfg = config.get("model", config)`
2. `enforce_triton_env_var(model_cfg)` — force-back Triton keys if env var unset
3. `nn.Embedding(vocab_size, dim)` — init `N(0, 0.006²)`
4. `ModuleList` of 18 `TransformerBlock`s
5. Final `RMSNorm(dim)`
6. `Linear(dim, vocab_size)` — LM head
7. Optional weight tying: `head.weight = embed.weight`

### `moe_layers()`

Generator yielding `DeepSeekMoE` instances from MoE layers. Used by `Pretrainer` for bias updates and balance metrics.

### `reset_cache()`

Calls `attn.reset_cache()` on every MLA layer. **Required** before each new generation session.

---

## Forward Contracts

### `forward(tokens, start_pos=0, use_cache=True) → (B, S, V)`

Primary interface. Returns vocabulary logits.

- Casts non-Long tokens to `int64` at boundary (uint32 shards).
- Builds causal mask when `seqlen > 1`; `None` for single-token decode.
- Runs all layers via `_run_layers`.
- Returns `head(norm(h))`.

**Training:** `use_cache=False` (set in `train_step`). No KV cache writes.

**Inference prefill:** `use_cache=True`, `start_pos=0`, full prompt length.

**Inference decode:** `use_cache=True`, `start_pos=prompt_len+step`, `seqlen=1`.

### `forward_with_hidden(tokens, start_pos=0, use_cache=False) → (logits, h)`

Returns `(head(norm(h)), h)` where `h` is the **pre-norm** trunk hidden.

- Used by `MultiTokenPrediction` to feed MTP heads.
- `use_cache=False` during training.
- `use_cache=True` in `SpeculativeDecoder.generate_step` (cache grows).

### Shape reference (422M, B=2, S=128)

| Tensor | Shape |
|---|---|
| `tokens` | (2, 128) |
| `h` (embedding) | (2, 128, 768) |
| per-layer `x` | (2, 128, 768) |
| `logits` | (2, 128, 100018) |

---

## Causal Mask Cache

```python
def _build_causal_mask(seqlen, device):
    mask = triu(-inf, diagonal=1)    # (S, S)
    return mask.unsqueeze(0).unsqueeze(0)   # (1, 1, S, S)
```

Cached in `_mask_cache` keyed by `(seqlen, device)`.

**Skipped when `seqlen == 1`:** During autoregressive decode, each step passes a single token. A causal mask is unnecessary (one query, many keys from cache). Skipping saves allocation and SDPA overhead.

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

## Generation API

```python
@torch.inference_mode()
def generate(input_ids, max_new_tokens=512, temperature=1.0,
             top_p=0.9, top_k=0, eos_token_id=None) -> Tensor
```

### Flow

1. Save/restore `self.training` mode
2. `reset_cache()`
3. `eval()`
4. **Prefill:** `forward(prompt, start_pos=0, use_cache=True)` → logits for all prompt positions
5. **Decode loop:** sample from last logits, append token, `forward(single_token, start_pos=prompt_len+step)`
6. Stop on EOS or `max_seq_len`

### `_sample(logits, temperature, top_p, top_k)`

| Setting | Behaviour |
|---|---|
| `temperature=0` | Greedy argmax |
| `temperature>0` | Scale logits, then sample |
| `top_k>0` | Mask logits below k-th largest |
| `top_p<1` | Nucleus sampling on sorted probabilities |

---

## Gradient Checkpointing

```python
Transformer(config, use_checkpoint=True)
```

When `use_checkpoint and self.training`:

```python
h = torch.utils.checkpoint.checkpoint(
    layer, h, start_pos, mask, use_cache, use_reentrant=False
)
```

**Trade-off:** ~3× less activation memory, ~33% more backward FLOPs (recompute forward during backward).

422M config: `grad_checkpoint: true`. 1650 smoke config: `false` (model fits without it).

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

`Pretrainer` passes the full YAML dict; `Transformer` unwraps internally. MLA/MoE receive the unwrapped flat dict via `TransformerBlock`.

---


## `generate()` and `_sample()` — Full Walkthrough

See `models/transformer.py:110-158`. Generation is the only production code path that sets `use_cache=True` on every layer.

**Training forward:** `mask` built when `seqlen>1`; writes no cache when `use_cache=False`.
**Prefill:** `seqlen=prompt_len`, full causal mask, fills MLA `kv_cache`/`pe_cache` slots `[0:prompt_len)`.
**Decode:** `seqlen=1`, `mask=None` (fast path), writes one slot at `start_pos`.

### Why `reset_cache()` is mandatory

Without reset, a second `generate()` call appends to stale cache tensors — logits reference wrong positions. `generate()` always calls `reset_cache()` at entry.

### Batch generation caveat

All rows in batch `B>1` share the same loop length; EOS only stops a row logically — implementation checks `finished.all()` but still runs decode steps until all finish or `max_new_tokens` exhausted.

---

## Parameter Budget by Component (422M)

Approximate counts from `Pretrainer._log_per_component_params` categories:

| Component | ~Params | % of total |
|---|---|---|
| MoE routed experts | 298M | 70% |
| MLA attention | 52M | 12% |
| Embeddings + tied head | 77M | 18% (counted once) |
| Dense SwiGLU (layers 0–1) | 7M | 2% |
| MoE shared experts | 14M | 3% |
| MTP (depth=1) | 3M | <1% |
| RMSNorm + gates | <1M | <1% |

**μP note:** `count_parameters` deduplicates tied embed/head — do not sum embedding and lm_head rows separately.

---

## `forward_with_hidden` — MTP Contract

```python
h = embed(tokens)
h = _run_layers(h, start_pos, mask, use_cache=False)
return head(norm(h)), h   # logits use normed h; MTP consumes raw h
```

MTP blocks apply **their own** `RMSNorm` before fusion — the returned `h` is the pre-final-norm trunk state, matching DeepSeek-V3's "main hidden before head norm" convention in this repo.

---

## Causal Mask Cache

`_build_causal_mask(seqlen, device)` caches `(1,1,S,S)` upper-triangular `-inf` masks keyed by `(seqlen, device)`.

| Call context | seqlen | mask |
|---|---|---|
| Training | 2048 | full causal |
| Prefill | prompt_len | full causal |
| Decode | 1 | `None` (no self-attention mask needed) |

Changing `max_seq_len` in config without clearing `_mask_cache` is safe — cache miss triggers rebuild when `seqlen` differs.

---

## Extension Points (Safe vs Unsafe)

| Change | Safe? | Notes |
|---|---|---|
| `n_dense_layers` | Yes | Test with `small_cfg` first |
| `weight_tying: false` | Yes | Doubles embedding+head params; update μP count |
| `n_layers` | Yes | Watch VRAM linear in depth |
| Remove `enforce_triton_env_var` | **No** | Violates AGENTS hard rule |
| Post-norm residuals | **No** | Untested; breaks training stability |
| `use_cache=True` in training | **No** | Breaks gradients / leaks stale state |

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



## Appendix A — Tensor shape trace

Single forward, `B=1, S=4`, 422M:

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

**Q: Can I make all layers MoE?** Set `n_dense_layers=0`. Untested at 422M scale but architecturally valid.

**Q: Does `generate` support batch size > 1?** Yes, but all sequences in the batch share the same generation length (no per-sequence early stopping except EOS).

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

---

## Load-Bearing Invariants

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

## `generate()` — Full Source Walkthrough

`models/transformer.py:generate` (inference-only):

```python
@torch.inference_mode()
def generate(self, input_ids, max_new_tokens=512, temperature=1.0,
             top_p=0.9, top_k=0, eos_token_id=None):
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

**Batch semantics:** All sequences in batch share `max_new_tokens`; per-sequence early stop only when **all** rows hit EOS (`finished.all()`).

---

## Parameter Budget — 422M Breakdown

From `Pretrainer._log_per_component_params` (typical run):

| Component | ~Params | ~% |
|---|---|---|
| MoE routed experts | ~210M | 50% |
| Embedding (tied head) | ~77M | 18% |
| MLA attention | ~45M | 11% |
| MoE shared experts | ~25M | 6% |
| Dense SwiGLU (layers 0-1) | ~7M | 2% |
| MTP modules | ~3M | 1% |
| Gates + norms | remainder | |

MoE **stores** 21 experts per layer but **executes** 5 per token.

---

## `_build_causal_mask` Cache

```python
def _build_causal_mask(self, seqlen, device):
    if cache miss or device change:
        mask = triu(-inf, diagonal=1)  # (S, S)
        cache (1, 1, S, S)
    return cache
```

**Invalidation:** New `seqlen` or device change rebuilds. Decode with `seqlen=1` skips mask entirely (`mask=None`).

---

## Extension Points

| Goal | Where to change |
|---|---|
| Add layer type | `TransformerBlock.__init__` branch on `layer_id` |
| New position encoding | `models/mla.py` RoPE section |
| Batched speculative decode | `inference/speculative.py` (not implemented) |
| Multi-GPU | New `training/pretrain_distributed.py` (out of scope) |
| Different vocab | YAML `vocab_size` + re-tokenize data |

<!-- docs:verified 2026-08-01 · e8553c4 -->


---

# Foundations — Building Blocks of DeepSeek-V3-Lite

> **Purpose:** A self-contained textbook chapter covering every prerequisite concept you need before reading the component-specific docs (MLA, MoE, MTP, training, inference). Read this first if you are learning the project from scratch.

> **Read this if** transformer basics, RoPE, or Chinchilla are unfamiliar. **Skip if** you're ready to run smoke tests → [[Docs/00_Getting_Started|getting started]].

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

**In this repo:** $L = 18$ layers. Layers 0–1 use dense SwiGLU; layers 2–17 use DeepSeekMoE. Every layer uses MLA (not standard MHA). See [[Docs/02_Model_Architecture|architecture]] for the full stack diagram.

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

**In this repo:** Standard MHA appears only in MTP blocks (`models/mtp.py:MTPBlock`). The main trunk uses **MLA** — a compressed variant covered in [[Docs/03_Multi_Head_Latent_Attention|MLA]]. The causal mask logic is identical.

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

**In this repo:** MLA splits each head's Q/K into **nope** (non-positional, 48 dims) and **rope** (positional, 24 dims) components. Only the rope part gets RoPE. See [[Docs/03_Multi_Head_Latent_Attention|MLA]] §Decoupled RoPE.

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

**In this repo:** `SwiGLUFFN` in `models/transformer.py`. MoE experts use the same SwiGLU structure with smaller $I = 384$ (`moe_inter_dim`). See [[Docs/04_DeepSeekMoE|moe]].

---

## Mixture of Experts (MoE) — Intuition

A **Mixture of Experts** replaces one large FFN with $N$ smaller **expert** FFNs plus a **router** that selects top-$k$ experts per token.

$$
\mathrm{MoE}(\mathbf{x}) = \sum_{i \in \mathrm{TopK}(\mathbf{x})} g_i(\mathbf{x}) \cdot \mathrm{Expert}_i(\mathbf{x}) + \mathrm{SharedExpert}(\mathbf{x})
$$

**Intuition:** Not every token needs the same computation. Factual tokens might route to a "knowledge" expert; syntactic tokens to a "grammar" expert. Sparsity means we store $N$ experts but only **execute** $k$ per token — gaining capacity without proportional compute.

**The load-balancing problem:** Without intervention, routers collapse — sending 99% of tokens to one expert. Solutions:
1. **Auxiliary load-balancing loss** (Switch Transformer) — adds a gradient term penalizing imbalance.
2. **Aux-loss-free bias updates** (DeepSeek-V3) — adjust router biases out-of-band. **This repo uses #2.** See [[Docs/04_DeepSeekMoE|moe]].

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

**MTP extension:** This repo adds auxiliary losses for predicting $t+2, t+3, \ldots$ with weight $\lambda = 0.3$. See [[Docs/05_Multi_Token_Prediction|mtp]].

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

**Memory:** Per layer, cache size grows as $O(S \cdot d_{\text{kv}})$. MLA compresses $d_{\text{kv}}$ dramatically. See [[Docs/03_Multi_Head_Latent_Attention|MLA]] and [[Docs/10_Inference_and_Serving|inference]].

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
- The data pipeline targets exactly this token budget. See [[Docs/09_Data_Pipeline|data pipeline]].

### Model FLOPs Utilisation (MFU)

$$
\mathrm{MFU} = \frac{\text{actual FLOPs/sec}}{\text{peak hardware FLOPs/sec}}
$$

Target: 35–40% on A100 80GB. MFU < 20% usually means memory-bound (activations, MoE dispatch) or unfused kernels. See [[Docs/11_Operations_and_Testing|scripts]] §step_time_a100.

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

**Important:** μP scaling runs **after** MTP wrapping, because MTP heads add ~3M params that affect the count. See [[Docs/08_Training_Pipeline|training]].

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
| Embedding + blocks + head | `models/transformer.py` | [[Docs/02_Model_Architecture|transformer]] |
| MLA attention | `models/mla.py` | [[Docs/03_Multi_Head_Latent_Attention|MLA]] |
| MoE routing + experts | `models/moe.py` | [[Docs/04_DeepSeekMoE|moe]] |
| MTP auxiliary heads | `models/mtp.py` | [[Docs/05_Multi_Token_Prediction|mtp]] |
| Training loop | `training/pretrain.py` | [[Docs/08_Training_Pipeline|training]] |
| Generation + speculative | `inference/` | [[Docs/10_Inference_and_Serving|inference]] |
| Data pipeline | `data/prepare_data.py` | [[Docs/09_Data_Pipeline|data pipeline]] |
| Config YAML | `configs/*.yaml` | [[Docs/08_Training_Pipeline|configs]] |
| Triton kernels | `models/*_triton.py` | [[Docs/12_Triton_Kernels|triton kernels]] |

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

**MLA twist:** $\mathbf{K},\mathbf{V}$ are low-rank functions of compressed cache — backward flows through LoRA adapters $W_{kv}$, absorption matrices, and RoPE rotation. See [[Docs/03_Multi_Head_Latent_Attention|MLA]] §Gradient Flow.

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

## Documentation Chapter Dependencies

```
§1 LM objective ──► [[Docs/08_Training_Pipeline|Training]] §9 Loss / CE
§5 Attention ─────► [[Docs/03_Multi_Head_Latent_Attention|MLA]] (low-rank KV)
§6 RoPE ──────────► [[Docs/03_Multi_Head_Latent_Attention|MLA]] (decoupled PE)
§7 SwiGLU ────────► [[Docs/02_Model_Architecture|Model Architecture]] (dense FFN)
§8 MoE intuition ─► [[Docs/04_DeepSeekMoE|MoE]] (aux-loss-free)
§12 Chinchilla ───► [[Docs/08_Training_Pipeline|Training]] (token budget)
§13 μP ───────────► [[Docs/08_Training_Pipeline|Training]] (LR scaling)
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

**Q: Do I need to read this entire chapter before [[Docs/03_Multi_Head_Latent_Attention|MLA]]?**
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

**Aux-loss-free:** No $\mathcal{L}_{\text{aux}}$ in gradient; bias updated out-of-band. See [[Docs/04_DeepSeekMoE|moe]].

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

<!-- docs:verified 2026-08-01 · e8553c4 -->


---

# Getting Started — From Zero to a Running DeepSeek-V3-Lite

> **Purpose:** Onboard a strong ML student who has never seen this repo. You will learn *what* the project is, *why* each design choice exists, and *how* to verify your understanding with smoke tests — before diving into component chapters.

---

## Table of Contents

1. [What Problem Does This Project Solve?](#what-problem-does-this-project-solve)
2. [The DeepSeek-V3 Architecture in One Page](#the-deepseek-v3-architecture-in-one-page)
3. [Canonical Numbers — 422M Config](#canonical-numbers--422m-config)
4. [Mental Model — Three Execution Modes](#mental-model--three-execution-modes)
5. [Recommended Reading Order](#recommended-reading-order)
6. [Environment Setup](#environment-setup)
7. [Quick Smoke Test (CPU)](#quick-smoke-test-cpu)
8. [Quick GPU Smoke Test](#quick-gpu-smoke-test)
9. [Full Training Run (A100)](#full-training-run-a100)
10. [Interactive Inference](#interactive-inference)
11. [How to Read the Codebase](#how-to-read-the-codebase)
12. [Common Pitfalls — Theory and Fixes](#common-pitfalls--theory-and-fixes)
13. [Learning Exercises](#learning-exercises)
14. [FAQ](#faq)
15. [References](#references)

---

## What Problem Does This Project Solve?

Modern frontier LLMs (GPT-4, Claude, DeepSeek-V3) are **black boxes** — you can call an API, but you cannot inspect how Multi-Head Latent Attention compresses KV cache, how aux-loss-free MoE routing avoids expert collapse, or how Multi-Token Prediction enables speculative decoding.

**DeepSeek-V3-Lite** is a **pedagogical reproduction**: the full DeepSeek-V3 architecture at ~422M parameters, implemented in raw PyTorch with no HuggingFace Trainer abstraction. Every line is meant to be read.

### What you will learn

| Topic | Why it matters | Doc |
|---|---|---|
| **MLA** | 10× KV-cache compression via low-rank latent + absorption | [[Docs/03_Multi_Head_Latent_Attention|MLA]] |
| **MoE** | Sparse FFN capacity without proportional compute | [[Docs/04_DeepSeekMoE|moe]] |
| **MTP** | Denser training signal + inference draft head | [[Docs/05_Multi_Token_Prediction|mtp]] |
| **μP + Chinchilla** | Stable hyperparameter transfer across scales | [[Docs/08_Training_Pipeline|training]] |
| **Triton kernels** | Optional fused paths for production throughput | [[Docs/12_Triton_Kernels|triton kernels]] |

### What this project is NOT

- Not a production chatbot (422M is tiny by industry standards)
- Not a distributed training framework (single GPU only)
- Not a drop-in replacement for `transformers.AutoModel`

It **is** a complete, trainable, testable implementation you can run on one A100 in ~15 hours.

---

## The DeepSeek-V3 Architecture in One Page

```
                    ┌─────────────────────────────────────┐
  Token IDs ───────►│  Embedding (V=100018, d=768)        │
                    └──────────────┬──────────────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         │  Layer 0–1 (DENSE)      │                         │
         │  RMSNorm → MLA → +      │  SwiGLU FFN (I=1536)    │
         └─────────────────────────┼─────────────────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         │  Layer 2–17 (MoE)       │                         │
         │  RMSNorm → MLA → +      │  DeepSeekMoE            │
         │                         │   top-4 of 20 experts   │
         │                         │   + 1 shared expert     │
         └─────────────────────────┼─────────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │  RMSNorm → LM Head (weight-tied)      │
                    └──────────────┬──────────────────────┘
                                   │
                              Logits (B,S,V)

  Training-only parallel path:
  MTPModule(hidden_t, embed_{t+1}) → predicts token_{t+2}
```

**Key insight:** Attention (MLA) and FFN (MoE) are **independent innovations** glued by the standard pre-norm residual stack. Understanding each in isolation, then reading [[Docs/02_Model_Architecture|architecture]], gives you the full picture.

**Prerequisites:** If RMSNorm, RoPE, or SwiGLU are unfamiliar, read [[Docs/01_Foundations|foundations]] first.

---

## Canonical Numbers — 422M Config

These numbers appear everywhere. Memorise the orders of magnitude; exact values live in `configs/pretrain_a100_422m.yaml`.

| Quantity | Value | Formula / note |
|---|---|---|
| Parameters | ~422M | MoE dominates (~70%) |
| Training tokens | 8.4B | Chinchilla: $20 \times N_{\text{params}}$ |
| Layers | 18 | 2 dense + 16 MoE |
| Hidden dim $d$ | 768 | |
| Heads $H$ | 12 | MLA per head: 48 nope + 24 rope QK, 64 V |
| KV latent rank $R$ | 192 | MLA compression |
| MoE experts | 20 routed + 1 shared | Top-4 activated |
| Expert width | 384 | `moe_inter_dim` |
| Vocab $V$ | 100,018 | DeepSeek tokenizer |
| Seq length $S$ | 2,048 | |
| Micro-batch | 8 | |
| Grad accum | 4 | Effective batch = 32 seqs |
| Tokens / optim step | 65,536 | $32 \times 2048$ |
| Optim steps | 512,000 | |
| LR (μP-scaled) | ~8.07e-4 | From ref 6e-4 @ 757M params |
| Wall time (A100) | 13–15 h | Target MFU 35–40% |
| Training VRAM | ~35 GB | grad_checkpoint + compile |

---

## Mental Model — Three Execution Modes

The same codebase serves three purposes. Confusing them causes the most common bugs.

### Mode 1 — CPU correctness (development)

```bash
python -m pytest tests/ -q
```

- No CUDA, no Triton, no data download required
- Tests use `small_cfg` (dim=64, vocab=1024)
- **Purpose:** Prove math invariants (MLA absorption, MoE bias, MTP alignment)

### Mode 2 — GPU smoke (validation)

```bash
python scripts/smoke_forward.py
python scripts/microbench_a100.py
python scripts/step_time_a100.py
```

- Builds full 422M model on GPU
- One forward/backward, VRAM measurement, throughput
- **Purpose:** Confirm hardware headroom before a 15-hour run

### Mode 3 — Production training

```bash
python3 data/prepare_data.py --stage pretrain   # once
bash scripts/launch_a100.sh
```

- 8.4B-token corpus, 512K steps, checkpointing, WandB
- **Purpose:** Train a real checkpoint

| Mode | `use_cache` | `torch.compile` | Data |
|---|---|---|---|
| CPU tests | False | No | Random / tiny fixture |
| GPU smoke | False | Optional | Random |
| Training | **False** | Yes | 8.4B shards |
| Inference | **True** | No | Prompt tokens |

---

## Recommended Reading Order

### Track A — Theory first (recommended for learning)

| Step | Doc | Time | You will understand |
|---|---|---|---|
| 0 | [[Docs/01_Foundations|foundations]] | 2–3 h | Transformer basics, RoPE, Chinchilla, μP |
| 1 | [[Docs/02_Model_Architecture|architecture]] | 1 h | How components connect |
| 2 | [[Docs/03_Multi_Head_Latent_Attention|MLA]] | 3–4 h | KV compression + absorption trick |
| 3 | [[Docs/04_DeepSeekMoE|moe]] | 2–3 h | Aux-loss-free routing |
| 4 | [[Docs/05_Multi_Token_Prediction|mtp]] | 1–2 h | Multi-token prediction |
| 5 | [[Docs/02_Model_Architecture|transformer]] | 1 h | Wiring + generation |
| 6 | [[Docs/08_Training_Pipeline|training]] | 2 h | Full train loop |
| 7 | [[Docs/10_Inference_and_Serving|inference]] | 1 h | KV cache + speculative decode |

### Track B — Code first (recommended for debugging)

| Step | Start here | Question answered |
|---|---|---|
| 1 | `models/transformer.py` | What is the forward path? |
| 2 | `models/mla.py` | How does attention work? |
| 3 | `models/moe.py` | How does routing work? |
| 4 | `training/pretrain.py` | How is loss computed? |
| 5 | `tests/test_models.py` | What invariants must hold? |

### Track C — Operations only

| Doc | When |
|---|---|
| [[Docs/08_Training_Pipeline|configs]] | Tuning YAML |
| [[Docs/11_Operations_and_Testing|scripts]] | Launch / benchmark |
| [[Docs/09_Data_Pipeline|data pipeline]] | Preparing tokens |
| [[Docs/11_Operations_and_Testing|utils]] | Checkpoints / VRAM |
| [[Docs/11_Operations_and_Testing|testing]] | Writing tests |

---

## Environment Setup

### Dependencies

```bash
cd LLM/DeepSeek-v3-Lite
pip install torch safetensors pyyaml tqdm pytest
# Optional: transformers (inference CLI), wandb (logging), triton (GPU kernels)
```

### Hardware expectations

| Hardware | What works |
|---|---|
| Mac / Linux CPU | All `pytest` tests |
| GTX 1650 4GB | `configs/pretrain_1650_2m.yaml` smoke train |
| A100 80GB | Full 422M Chinchilla run |

### Environment variables (common)

```bash
export TOKENIZERS_PARALLELISM=false   # avoid fork warnings in DataLoader
export ENABLE_TRITON_KERNELS=1      # only when benchmarking Triton paths
export WANDB_PROJECT=deepseek-v3-lite-a100
```

---

## Quick Smoke Test (CPU)

```bash
python -m pytest tests/ -q --ignore=tests/test_moe_triton.py::TestMoeTritonKernelGPU
```

**Expected:** All tests pass in ~1–3 minutes on a modern laptop.

**What this proves:**
- MLA SDPA and manual paths agree (absorption correctness)
- MoE gate bias is a buffer, not a Parameter
- MTP shared head and alignment logic
- Checkpoint atomic save/load with MTP prefix
- Triton env-var guard forces PyTorch fallback

**If tests fail:** Read the failing test name in [[Docs/11_Operations_and_Testing|testing]] — each test documents an invariant.

---

## Quick GPU Smoke Test

```bash
python scripts/smoke_forward.py          # builds 422M, one forward
python scripts/microbench_a100.py        # VRAM estimate vs measured
python scripts/step_time_a100.py         # ms/step, MFU
```

### Interpreting microbench output

```
estimated peak   ≈ 30.5 GB   # from utils/memory.py formulas
measured peak    ≈ 33-35 GB # torch.cuda.max_memory_allocated
```

**Theory:** The gap is CUDA allocator overhead + `torch.compile` workspace spikes. If measured > 72 GB on an 80GB card, reduce `micro_batch_size` or `max_seq_len` before launching training.

### Interpreting step_time output

```
median ms/step ≈ 270-320 ms
MFU ≈ 35-40%
```

**MFU** (Model FLOPs Utilisation) = fraction of peak Tensor Core throughput used. Below 25% usually means memory-bound (MoE dispatch, activation recomputation).

---

## Full Training Run (A100)

### Step 1 — Prepare data (~8.4B tokens)

```bash
python3 data/prepare_data.py --stage pretrain
```

Output: `data/pretrain_chinchilla/shard_*.bin` (uint32 tokens). See [[Docs/09_Data_Pipeline|data pipeline]].

### Step 2 — Verify GPU headroom

```bash
python scripts/microbench_a100.py
```

### Step 3 — Launch

```bash
bash scripts/launch_a100.sh
# or manually:
python -u training/pretrain.py --config configs/pretrain_a100_422m.yaml
```

### Step 4 — Monitor

```bash
tail -f checkpoints/pretrain_a100/train.log
# step=  4000 | loss=2.85 | ppl=17.3 | lr=7.9e-04 | tps=128000
```

### Resume from checkpoint

```bash
python training/pretrain.py --config configs/pretrain_a100_422m.yaml --resume 4000
```

---

## Interactive Inference

```bash
python -m inference.generate \
  --config configs/pretrain_a100_422m.yaml \
  --checkpoint checkpoints/pretrain_a100 \
  --use_speculative
```

**Theory:** Standard decode generates one token per main-model forward. Speculative decode uses the MTP head to **draft** a second token, verified by the main model — potentially 2 tokens per forward. See [[Docs/10_Inference_and_Serving|inference]] and [[Docs/05_Multi_Token_Prediction|mtp]].

---

## How to Read the Codebase

### File priority map

```
models/transformer.py   ← start here (wiring)
models/mla.py           ← hardest, most important
models/moe.py           ← routing invariants
models/mtp.py           ← training + speculative bridge
training/pretrain.py    ← train loop
tests/test_models.py    ← executable specification
```

### Config → code routing

YAML keys are consumed at different depths. `Pretrainer` stores the **entire** YAML in `TrainingConfig.model_config`; sub-modules call `config.get("model", config)`:

| YAML key | Consumer | Effect |
|---|---|---|
| `model.n_dense_layers` | `TransformerBlock` | Layers 0–1 dense, 2–17 MoE |
| `model.attn_impl` | `MultiHeadLatentAttention` | sdpa / manual / triton |
| `model.mtp_depth` | `Pretrainer` | Wrap with `MultiTokenPrediction` |
| `training.mup_lr` | `Pretrainer.__init__` | LR scaling |
| `training.nan_guard` | `train()` loop | Rollback on NaN streak |

Full reference: [[Docs/08_Training_Pipeline|configs]].

---

## Common Pitfalls — Theory and Fixes

| Mistake | Symptom | Root cause | Fix |
|---|---|---|---|
| `vocab_size` ≠ 100018 | Shape error at embed | Embedding rows must match tokenizer | Use DeepSeek tokenizer everywhere |
| `use_cache=True` in training | Silent wrong loss / NaN | Cache breaks full-sequence backward | `use_cache=False` in `train_step` |
| Forget `reset_cache()` | Garbage generation | Stale KV from prior prompt | Call before each `generate()` |
| MoE aux loss instead of bias | Expert collapse | Gradient fights task loss | Read [[Docs/04_DeepSeekMoE|moe]] §Why Not Aux Loss |
| Triton without env var | No speedup, warning once | `enforce_triton_env_var` guard | `ENABLE_TRITON_KERNELS=1` |
| μP before MTP wrap | Wrong LR | Param count excludes MTP | Scaling after wrap in `Pretrainer` |
| Load ckpt with `strict=True` | Missing `head.weight` | Weight tying drops duplicate key | `strict=False` (default) |

---

## Learning Exercises

1. **Parameter audit:** Run `Pretrainer` init with `log_per_component_params: true`; verify MoE > 65% of params.
2. **MLA trace:** Set `attn_impl: manual` vs `sdpa` in a tiny config; confirm identical outputs (`test_sdpa_and_manual_agree`).
3. **MoE routing histogram:** Log `gate.get_expert_counts()` over 1000 steps; confirm no expert gets 0% load.
4. **MTP alignment:** Manually verify `usable = S - depth - 1` for depth=1, S=2048.
5. **Chinchilla check:** Compute $512000 \times 65536$ total token exposures vs 8.4B unique tokens — how many epochs?
6. **Speculative accept rate:** Log `was_accepted` in `SpeculativeDecoder.generate_step`; relate to threshold 0.8.

---

## FAQ

**Q: I only have a Mac. Can I learn this project?**
A: Yes. All correctness tests run on CPU. You cannot train 422M locally, but you can read every line and run `small_cfg` forwards.

**Q: How does this compare to nanoGPT / minGPT?**
A: Those teach classic GPT. This teaches **modern** DeepSeek-V3 mechanisms (MLA, aux-loss-free MoE, MTP) that nanoGPT omits.

**Q: Should I read the paper or the docs first?**
A: Read [[Docs/01_Foundations|foundations]] + [[Docs/02_Model_Architecture|architecture]], then the DeepSeek-V3 paper (arXiv:2412.19437) with this repo open. The docs explain *this implementation*; the paper explains *the full 671B system*.

**Q: Where do I ask questions in code?**

| Question | Start here |
|---|---|
| "How does attention work?" | `models/mla.py` + [[Docs/03_Multi_Head_Latent_Attention|MLA]] |
| "How does routing work?" | `models/moe.py` + [[Docs/04_DeepSeekMoE|moe]] |
| "How is loss computed?" | `models/mtp.py:compute_loss` + [[Docs/08_Training_Pipeline|training]] |
| "How do checkpoints work?" | `utils/checkpoint.py` + [[Docs/11_Operations_and_Testing|utils]] |
| "What do tests verify?" | [[Docs/11_Operations_and_Testing|testing]] |

---

## References

- [[Docs/01_Foundations|foundations]] — prerequisite theory
- [[Docs/02_Model_Architecture|architecture]] — system map
- [[Docs/README|README]] — documentation index
- DeepSeek-V3 paper — arXiv:2412.19437
- `configs/pretrain_a100_422m.yaml` — canonical recipe

## Day-1 Checklist

| Step | Command | Success criterion |
|---|---|---|
| 1 | `pip install torch safetensors pyyaml tqdm pytest` | No import errors |
| 2 | `python -m pytest tests/ -q` | All CPU tests pass |
| 3 | `python scripts/smoke_forward.py` (GPU) | `(B,S,V)` logits shape |
| 4 | `python scripts/microbench_a100.py` (GPU) | measured < 72 GB |
| 5 | Read [[Docs/01_Foundations|foundations]] §1–6 | Can explain CE loss + RoPE |

---

## Loss Curve — What to Expect

422M Chinchilla training on mixed corpus (rough guide, not guarantees):

| Step | Loss (nats) | PPL | Phase |
|---|---|---|---|
| 0–2k | 10–11 | ~50k | Warmup, random init |
| 10k | 4–5 | ~150 | Learning subword structure |
| 100k | 2.8–3.2 | ~16–25 | Mid-training |
| 400k+ | 2.3–2.7 | ~10–15 | Late cosine tail |

**Red flags:** loss flat after 50k steps (data path wrong), sudden spike to NaN (LR / shard corruption), PPL stuck > 100 at 200k (vocab mismatch).

---

## Checkpoint Inspection

```python
from safetensors.torch import load_file
state = load_file("checkpoints/pretrain_a100/model_step_4000.safetensors")
print(len(state), "keys")
print([k for k in state if k.startswith("mtp.")][:5])
```

Expect ~hundreds of keys; `head.weight` may be absent when `weight_tying: true`.

---

## Extending the Project — Safe Order

1. **Change depth/width** — edit YAML, run `pytest`, then `microbench`
2. **Add data source** — edit `mixture.yaml`, re-run `prepare_data.py`
3. **New attention path** — add test in `test_models.py` first, then implement
4. **Triton kernel** — set `ENABLE_TRITON_KERNELS=1`, compare against stacked in `test_moe_triton.py`

Never skip step 1 (tests) when touching `mla.py`, `moe.py`, or `mtp.py`.

<!-- docs:verified 2026-08-01 · e8553c4 -->


---

# Design Rationale, Transformer Block Details, and Portfolio Comparison

> **Design Rationale & Architecture Details.** This section consolidates unique architectural design choices, transformer block topology details, and portfolio comparisons.

---

## Design Rationale

> Why does DeepSeek-V3-Lite exist? Why MLA over GQA? Why aux-loss-free MoE? Why dense + MoE topology? This covers the reasoning behind every major design decision.

---

### Chinchilla Scaling — Step-by-Step Derivation

The Chinchilla paper establishes ~20 tokens per parameter as optimal compute efficiency. For 422M parameters:

$$
T_{\text{optimal}} = 20 \times 422\text{M} \approx 8.4\text{B tokens}
$$

At batch=8, grad_accum=4, seq=2048:

$$
\text{tokens/step} = 8 \times 4 \times 2048 = 65536
$$

$$
\text{total\_steps} = \frac{8.4\text{B}}{65536} \approx 128000 \text{ optimizer steps}
$$

With grad_accum=4, this is 512,000 micro-steps. At 35–40% MFU on A100, this takes ~13–15 hours.

---

### Memory Breakdown — Why 422M Fits on One A100

The constraint is single A100 80GB training. The 422M model fits comfortably:

| Component | Memory |
|---|---|
| Parameters (BF16) | ~0.84 GB |
| Optimizer states (FP32 AdamW) | ~5.1 GB |
| Activations (with grad-ckpt) | ~2.0 GB |
| KV cache (MLA: 192+24 per token) | ~0.3 GB |
| Overhead | ~2.0 GB |
| **Total** | **~10.2 GB** |

Enormous headroom on 80GB. The 422M size was chosen for Chinchilla optimality at 8.4B tokens, not memory constraints.

---

### Parameter Breakdown by Component

| Component | Params | % of total |
|---|---|---|
| Embedding (100,018 × 768) | 76.8M | 18.2% |
| MLA (18 layers) | ~58M | 13.7% |
| Dense SwiGLU (2 layers) | ~7.1M | 1.7% |
| MoE routed experts (16 × 20 × 3 × 768 × 384) | ~283M | 67.1% |
| MoE shared experts (16 × 1 × 3 × 768 × 384) | ~14.1M | 3.3% |
| MoE gate (16 × 768 × 20) | ~0.25M | 0.06% |
| RMSNorm (all layers + final) | ~0.07M | 0.02% |
| Output head (tied with embedding) | 0 (deduplicated) | 0% |

The MoE experts dominate at 67%+ of total parameters, but only 4 of 20 are active per token (20% sparsity), so the active compute is much smaller.

---

### Why MLA over GQA — Trade-off Analysis

**Why not GQA?** GQA (Grouped-Query Attention) shares KV heads across query heads, giving a constant-factor KV reduction (2× with 8 groups, 4× with 4 groups). But:

- **Fixed compression ratio:** GQA's reduction is limited by the group ratio. You can't compress beyond `n_kv_heads=1` (MQA).
- **Quality degradation:** MQA (1 KV head) shows measurable perplexity degradation. GQA with 4–8 groups shows slight degradation.
- **No absorption trick:** GQA still materializes full K/V at every decode step. The memory bandwidth savings are limited to the cache size, not the decode computation.

**Why MLA?** MLA achieves **~5× KV-cache reduction** with **no quality loss** (matches MHA perplexity). The key innovations:

1. **Low-rank compression:** Instead of caching `n_heads × d_head` floats per token, MLA caches `kv_lora_rank + qk_rope_head_dim = 192 + 24 = 216` floats — a 5× reduction vs MHA's `12 × 64 = 768` floats (for content + value).
2. **Absorption trick:** At inference, the K up-projection is algebraically absorbed into the Q projection, so full K is never materialized. The attention score is computed directly in the 192-dim latent space.
3. **Decoupled RoPE:** A separate 24-dim RoPE path preserves position encoding without breaking the absorption algebra.

**The trade-off:** MLA trades **FLOPs for memory bandwidth**:

- Attention computation is ~4× more expensive than MHA (latent dot products are in a higher-dim space).
- But memory reads are ~5× cheaper (reading 216 floats vs 768 per token).

Since decode is overwhelmingly **memory-bandwidth-bound** at long context, MLA wins on throughput. This is a fundamentally different trade-off from GQA (which reduces both FLOPs and bandwidth proportionally).

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

This achieves load balancing **without any gradient on the bias** and **without any aux loss term**. The task gradient is pure — it only optimizes task performance, not load balancing.

---

### Why Top-4 of 20 Experts

- **20 experts** (finer granularity than GPT-OSS's 8) gives more specialization at the same parameter budget.
- **Top-4** (vs GPT-OSS's top-2) gives more expert diversity per token, improving quality.
- **1 shared expert** (always active) provides a baseline transformation regardless of routing.
- **moe_inter_dim=384** (vs dense inter_dim=1536): each expert is smaller, but 4+1 are active, giving `5 × 3 × 768 × 384 = 4.4M` active params per MoE layer vs `3 × 768 × 1536 = 3.5M` per dense layer — comparable compute, more total capacity.

### Active vs Total Parameters

| | Per layer | 18 layers |
|---|---|---|
| Dense layers (2) | 3.5M | 7.1M |
| MoE active (4 routed + 1 shared) | 4.4M + gate | ~71M |
| Non-FFN (MLA + norm + embed) | — | ~135M |
| **Total active** | | **~213M** |
| **Total params** | | **~422M** |
| **Sparsity** | | **~50%** |

---

### Why 2 Dense Layers

The first 2 layers are **dense SwiGLU FFNs** (no MoE routing). The rationale:

1. **Low-level feature extraction:** Early layers learn general features (tokenization, syntax) that all tokens need. These don't benefit from expert specialization.
2. **Routing stability:** The router needs meaningful hidden states to make good routing decisions. If the first layer is MoE, the router sees raw embeddings, which are less informative.
3. **DeepSeek-V3 design:** The original DeepSeek-V3 also starts with dense layers before switching to MoE.

```python
# In TransformerBlock.__init__:
self.ffn = SwiGLUFFN(dim, inter_dim) if layer_id < n_dense_layers else DeepSeekMoE(config)
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
| Compressed queries (q_lora_rank=0) | Simplified for 422M scale; DeepSeek-V3 full uses q_lora_rank=1536 |

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

### ParallelEmbedding Init std=0.006

```python
class ParallelEmbedding(nn.Module):
    def __init__(self, vocab_size, dim):
        self.weight = nn.Parameter(torch.empty(vocab_size, dim))
        nn.init.normal_(self.weight, std=0.006)
    
    def forward(self, x):
        return F.embedding(x, self.weight)
```

The "Parallel" name refers to the full DeepSeek-V3's tensor-parallel splitting across GPUs. In this single-GPU implementation, it's a regular embedding.

**Why std=0.006?** Much smaller than the typical `0.02` used in GPT-2/LLaMA. With vocab=100,018 and dim=768, the embedding has 76.8M params. Small init ensures the initial embeddings are close together (similar tokens start similar). The model learns to spread them apart during training.

---

### Weight Tying — Savings Math

Without tying: `2 × 100,018 × 768 = 153.6M` (embedding + head).
With tying: `100,018 × 768 = 76.8M` (single shared tensor).
**Savings: 76.8M** — 18% of the 422M total.

```python
self.head = nn.Linear(dim, vocab_size, bias=False)  # 768 → 100,018
if self.weight_tying:
    self.head.weight = self.embed.weight  # share the same tensor!
```

Deduplication in parameter counting uses `id(p)` (Python object identity) to detect tied parameters. Without this, the count would be ~499M instead of 422M (double-counting the 76.8M tied weights).

---

### Causal Mask Caching

```python
def _build_causal_mask(self, seqlen, device):
    if self._mask_cache is None or seqlen != self._mask_seqlen or self._mask_cache.device != device:
        mask = torch.triu(torch.full((seqlen, seqlen), float("-inf"), device=device), diagonal=1)
        self._mask_cache = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S)
        self._mask_seqlen = seqlen
    return self._mask_cache
```

When is the mask used?

- **Training (seqlen > 1):** The mask is built and cached. On subsequent forward passes with the same seqlen, the cached mask is reused.
- **Decode (seqlen == 1):** No mask needed — a single new token attends to all cached tokens, which are all in the past (causal by construction).
- **Different seqlen:** The mask is rebuilt when seqlen changes (e.g., from training to inference).

---

### Gradient Checkpointing — Every Layer, use_reentrant=False

```python
def _run_layers(self, h, start_pos, mask, use_cache):
    for layer in self.layers:
        if self.use_checkpoint and self.training:
            def _block(h, layer=layer, sp=start_pos, m=mask, uc=use_cache):
                return layer(h, sp, m, uc)
            h = torch.utils.checkpoint.checkpoint(_block, h, use_reentrant=False)
        else:
            h = layer(h, start_pos, mask, use_cache)
    return h
```

**Design choices:**

- **All 18 layers are checkpointed** (not every Nth like GPT-OSS-Lite's every-3rd). DeepSeek-V3 checkpoints every layer for maximum memory savings.
- **`use_reentrant=False`:** PyTorch 2.x recommended default, better `torch.compile` compatibility.
- **Only during training:** The `self.training` check ensures checkpointing is disabled during inference (no backward pass needed).

**Memory impact:**

Without checkpointing: all 18 layers store activations → `18 × 2048 × 8 × 768 × 2 = 565 MB`.
With checkpointing: only the input to each layer is stored → `18 × 768 × 2 = 27 KB` (negligible).
Trade-off: ~33% more compute (recomputation during backward).

---

### Forward Contracts

**`Transformer.forward`** — Primary interface, returns vocabulary logits:

```python
def forward(self, tokens, start_pos=0, use_cache=True):
    """(bsz, seqlen) → (bsz, seqlen, vocab_size)"""
    h = self.embed(tokens)
    mask = self._build_causal_mask(seqlen, device) if seqlen > 1 else None
    h = self._run_layers(h, start_pos, mask, use_cache)
    return self.head(self.norm(h))
```

**`Transformer.forward_with_hidden`** — MTP-compatible forward:

```python
def forward_with_hidden(self, tokens, start_pos=0, use_cache=False):
    """Returns (logits, h_norm) — used by MTP."""
    h = self.embed(tokens)
    mask = self._build_causal_mask(seqlen, device) if seqlen > 1 else None
    h = self._run_layers(h, start_pos, mask, use_cache)
    h_norm = self.norm(h)
    return self.head(h_norm), h_norm  # both logits AND hidden states
```

Returns the normalized hidden states so the MTP module can use them as input.

**`MultiHeadLatentAttention.forward`** — Takes and returns hidden states:

```python
def forward(self, x, start_pos=0, mask=None, use_cache=True):
    """(bsz, seqlen, dim) → (bsz, seqlen, dim)"""
    # ... query/KV/attention computation ...
    return self.wo(attn_output)
```

MLA takes hidden states and returns hidden states — same shape, same contract as standard attention.

---

## Portfolio Comparison

> Comparing DeepSeek-V3-Lite with sibling projects: GPT-OSS-Lite, LLaMA-3-Lite, and Mamba-3-Lite. Each project is mechanistically distinct.

---

### Architecture Comparison Matrix

| Property | DeepSeek-V3-Lite | GPT-OSS-Lite | LLaMA-3-Lite | Mamba-3-Lite |
|---|---|---|---|---|
| **Attention** | MLA (latent KV) | GQA + SWA/full alt | GQA | — (SSM) |
| **KV compression** | Low-rank latent (192+24) | Sliding window (128) | None (standard GQA) | Constant-state SSM |
| **KV cache reduction** | 7.1× vs MHA | 2.0× via SWA/full alt | 2× via GQA groups | O(1) state |
| **Long-context** | YaRN (decode only) | YaRN 128K (train+decode) | θ=500K (train@2K) | Constant-state |
| **MoE** | 20 routed top-4 + 1 shared | 8 routed top-2 + 1 shared | No | No |
| **Load balancing** | Aux-loss-free bias | Standard aux loss (α=0.01) | N/A | N/A |
| **Attention sinks** | No | Per-head learned | No | N/A |
| **MTP** | Depth=1 + speculative | No | No | No |
| **Position encoding** | Decoupled RoPE (24-dim) | YaRN-scaled RoPE (72-dim) | Standard RoPE (θ=500K) | — (implicit in SSM) |
| **Normalization** | RMSNorm (eps=1e-6) | RMSNorm (eps=1e-5) | RMSNorm | RMSNorm |
| **Weight tying** | Yes | Yes | Yes | Yes |
| **Tokenizer** | deepseek-coder-v2-lite (100,018) | LLaMA-3 (128,000) | LLaMA-3 (128,000) | — |
| **Total params** | ~422M | ~502M | — | — |
| **Training context** | 2,048 | 4,096 | 2,048 | — |
| **Eval context** | YaRN-scaled | 131,072 | — | Constant-state |

---

### Attention Mechanism Comparison

| Mechanism | How it reduces KV | Quality impact | Unique to |
|---|---|---|---|
| **MLA** | Low-rank compression → 192-dim latent | No loss (matches MHA) | DeepSeek |
| **GQA** | Share KV heads → fewer heads | Slight loss | LLaMA-3, GPT-OSS |
| **SWA/full alt** | Windowed layers cache only 128 tokens | Good (global layers retain context) | GPT-OSS |
| **SSM** | No KV cache at all — constant-size state | Different mechanism entirely | Mamba |

**The key distinction:**

- **DeepSeek-V3-Lite (MLA):** Compresses K/V **into a latent** via learned projections. The compression is lossless (up-projection recovers full K/V). The absorption trick eliminates the up-projection at inference.
- **GPT-OSS-Lite (SWA/full):** Compresses the cache by **reducing what's stored** — windowed layers only keep the last 128 tokens. The compression is lossy (distant context is forgotten in windowed layers) but compensated by global layers.
- **LLaMA-3-Lite (GQA):** Compresses by **sharing K/V across heads**. Simple but limited — can't compress beyond 1 KV head (MQA).
- **Mamba-3-Lite (SSM):** No attention at all — uses a state-space model with constant-size state. No KV cache needed.

---

### MoE Comparison — DeepSeek vs GPT-OSS

| Property | DeepSeek-V3-Lite | GPT-OSS-Lite |
|---|---|---|
| Routed experts | 20 | 8 |
| Active experts | 4 (top-4) | 2 (top-2) |
| Shared experts | 1 | 1 |
| Gate activation | Sigmoid | Softmax |
| Load balancing | Aux-loss-free bias | Standard aux loss (α=0.01) |
| Expert inter_dim | 384 | 1536 |
| Dispatch | Stacked bmm | Stacked F.linear |

**Key philosophical difference:**

**DeepSeek:** Finer-grained experts (20) with sigmoid routing and **no aux loss**. The bias update is a control system — it adjusts the gate logits out-of-band, without contaminating the task gradient.

**GPT-OSS:** Coarser experts (8) with softmax routing and **standard aux loss**. The aux loss is a regularizer — it adds a gradient signal that pushes toward uniform utilization.

Both are valid approaches. The aux-loss-free method is more elegant (pure task gradient) but requires careful bias update tuning. The standard aux loss is simpler and more widely validated but introduces gradient coupling.

---

### Long-Context Strategy Comparison

| Project | Strategy | Training context | Eval context | How it works |
|---|---|---|---|---|
| DeepSeek-V3-Lite | YaRN (decode only) | 2,048 | YaRN-scaled | Train without YaRN, apply at decode time |
| GPT-OSS-Lite | YaRN (train+decode) | 4,096 | 131,072 | Train with YaRN active, extrapolate 32× |
| LLaMA-3-Lite | θ=500K | 2,048 | — | Large RoPE base, moderate extrapolation |
| Mamba-3-Lite | Constant-state | — | — | SSM doesn't need extrapolation |

**DeepSeek vs GPT-OSS on YaRN:**

- **DeepSeek:** `rope_factor=1.0` at training (no YaRN). At inference, increase `rope_factor` to scale RoPE frequencies for longer context. This is a **decode-time patch** — the model wasn't trained for long context.
- **GPT-OSS:** `yarn_scale_factor=32` at training. The model learns the YaRN frequency ramp during training. This is **true length extrapolation** — the model genuinely generalizes to 32× its training context.

The trade-off: DeepSeek's approach is simpler (no YaRN at training) but less reliable at extrapolation. GPT-OSS's approach is more complex (YaRN must be configured correctly) but produces genuine extrapolation capability.

---

### Unique Innovations Per Project

| Project | Unique innovations (not in siblings) |
|---|---|
| **DeepSeek-V3-Lite** | MLA (low-rank KV compression + absorption), aux-loss-free MoE bias, MTP + speculative decoding, μP LR scaling, dense+MoE topology |
| **GPT-OSS-Lite** | Sliding-window/full alternation, per-head learned attention sinks, YaRN at training time, pruned RoPE on global layers |
| **LLaMA-3-Lite** | 78% memory stack optimization, chunked cross-entropy, async prefetch, GQA with θ=500K |
| **Mamba-3-Lite** | Complex-valued SSD (N=64 complex64), MIMO head mixing, zero causal conv, A100-optimized chunkwise |

---

### Cross-Project Lessons

1. **KV cache is the dominant inference bottleneck.** Every project addresses it differently — MLA compresses it, SWA reduces what's cached, GQA shares it, SSM eliminates it.

2. **Load balancing in MoE is a design choice, not a settled question.** DeepSeek uses bias updates (control theory), GPT-OSS uses aux loss (optimization). Both work; they have different trade-offs.

3. **Position encoding for long context is still an open problem.** YaRN (GPT-OSS), decode-time YaRN (DeepSeek), large θ (LLaMA-3), and implicit positioning (Mamba) are all valid approaches with different extrapolation properties.

4. **Weight tying is universal.** All four projects use it. The savings (76–98M params) are significant relative to model size.

5. **RMSNorm has won.** All four projects use RMSNorm, not LayerNorm. The simpler computation and slightly better gradient flow have made it the standard.

---

> **See also:** [[Docs/13_Portfolio_Comparison|Portfolio Comparison]] for full cross-portfolio comparative analysis across DeepSeek-V3-Lite, GPT-OSS-Lite, LLaMA-3-Lite, and Mamba-3-Lite.
