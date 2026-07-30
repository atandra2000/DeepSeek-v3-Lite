# Architecture Overview

> **Purpose:** A single map of how every component in DeepSeek-V3-Lite fits together. Read this after [foundations.md](foundations.md) and before diving into component-specific docs.

---

## Prerequisites

You should understand (or have skimmed):
- Causal language modeling objective — [foundations.md](foundations.md) §1
- Pre-norm residual blocks — [foundations.md](foundations.md) §3
- Chinchilla token budget — [foundations.md](foundations.md) §12

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

**Pedagogical implication:** When you see a 15-line function instead of a one-liner, it is usually guarding an invariant documented in [testing.md](testing.md).

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

See [inference.md](inference.md) and [mtp.md](mtp.md).

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

See [utils.md](utils.md) for formulas.

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

Full reference: [configs.md](configs.md).

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
| Fit smaller GPU | ↓ `micro_batch_size`, ↓ `max_seq_len`, enable `grad_checkpoint` | [configs.md](configs.md) |
| Faster iteration | `pretrain_1650_2m.yaml`, `build_small_pretrain_data.py` | [scripts.md](scripts.md) |
| Higher MFU | `ENABLE_TRITON_KERNELS=1`, `moe_dispatch: triton_grouped` | [triton_kernels.md](triton_kernels.md) |
| Longer context | ↑ `max_seq_len`, consider YaRN `rope_factor` | [MLA.md](MLA.md) |

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

Full test mapping: [testing.md](testing.md).

---

## Further Reading

- Attention deep-dive: [MLA.md](MLA.md)
- MoE deep-dive: [moe.md](moe.md)
- Training loop: [training.md](training.md)
- Learning path: [getting_started.md](getting_started.md)

<!-- docs:verified 2026-07-31 · 88cb863 -->
