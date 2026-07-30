# Configuration Reference

> **Purpose:** Textbook-style reference for every YAML key in `configs/`, with theory for *why* each hyperparameter exists and *where* code consumes it.

> **Read this if** you're tuning hyperparameters or adding config keys. **Skip if** you're learning architecture theory → [architecture.md](architecture.md).
---

## Table of Contents

1. [Overview](#overview)
2. [pretrain_a100_422m.yaml — Canonical Recipe](#pretrain_a100_422myaml--canonical-recipe)
3. [pretrain_1650_2m.yaml — Smoke Test](#pretrain_1650_2myaml--smoke-test)
4. [Model Keys](#model-keys)
5. [Training Keys](#training-keys)
6. [Data Keys](#data-keys)
7. [Triton Dispatch Keys](#triton-dispatch-keys)
8. [Config Nesting](#config-nesting)
9. [Appendix — Quick comparison table](#appendix--quick-comparison-table)

---

## Overview

Configs are YAML files with three top-level sections:

```yaml
model:      # Architecture — consumed by Transformer, MLA, MoE, MTP
training:   # Loop hyperparameters — consumed by Pretrainer
data:       # Paths — consumed by Pretrainer and inference/generate.py
```

`Pretrainer` reads the full YAML into `TrainingConfig.model_config` (the entire dict). `Transformer` unwraps `config.get("model", config)`.

### Why YAML instead of Python dataclasses?

Hyperparameters are **experiment surface area** — you will sweep LR, batch size, and MoE width without touching code. YAML provides:

1. **Git-diffable recipes** — `pretrain_a100_422m.yaml` is a reproducible artifact
2. **No import side effects** — changing config does not re-run module-level code
3. **Nested sections** — `model` / `training` / `data` mirror mental model

`Pretrainer.main()` maps YAML → `TrainingConfig` dataclass at runtime (`training/pretrain.py:main`). Tests may use flat dicts via `conftest.py` fixtures — both shapes work via `config.get("model", config)` unwrapping.


---

## pretrain_a100_422m.yaml — Canonical Recipe

**Target:** 1× A100 80GB, ~422M params, 8.4B Chinchilla tokens, 13–15 h wall.

| Section | Highlights |
|---|---|
| model | 18 layers, vocab 100018, dim 768, 16 MoE layers |
| training | 512K steps, μP LR, nan_guard, compile, grad_checkpoint |
| data | `data/pretrain_chinchilla`, deepseek tokenizer |

---

## pretrain_1650_2m.yaml — Smoke Test

**Target:** GTX 1650 4GB end-to-end validation.

| Difference from 422M | Value |
|---|---|
| vocab | 50257 (GPT-2) |
| dim | 64 |
| n_layers | 4 (2 dense + 2 MoE) |
| n_routed_experts | 4 |
| n_activated_experts | 1 |
| max_seq_len | 128 |
| compile | false |
| grad_checkpoint | false |
| mup_lr | false |

Preserves all architectural invariants (MLA, aux-loss-free MoE, MTP) at tiny scale.

---

## Model Keys

| Key | 422M | Read by | Description |
|---|---|---|---|
| `vocab_size` | 100018 | Embedding, LM head | Must match tokenizer |
| `dim` | 768 | All layers | Hidden dimension |
| `n_layers` | 18 | Transformer | Total transformer blocks |
| `n_heads` | 12 | MLA, MTP MHA | Attention heads |
| `n_dense_layers` | 2 | TransformerBlock | Dense before MoE |
| `n_routed_experts` | 20 | DeepSeekMoE | Routed expert count |
| `n_shared_experts` | 1 | DeepSeekMoE | Always-on experts |
| `n_activated_experts` | 4 | AuxLossFreeGate | Top-k routing |
| `inter_dim` | 1536 | SwiGLUFFN | Dense FFN width |
| `moe_inter_dim` | 384 | Expert | MoE expert width |
| `kv_lora_rank` | 192 | MLA | KV compression dim |
| `q_lora_rank` | 0 | MLA | Query compression (off) |
| `qk_nope_head_dim` | 48 | MLA | Content QK dim/head |
| `qk_rope_head_dim` | 24 | MLA | RoPE QK dim/head |
| `v_head_dim` | 64 | MLA | Value dim/head |
| `max_seq_len` | 2048 | MLA cache, dataset | Training sequence length |
| `rope_theta` | 10000 | MLA | RoPE base frequency |
| `rope_factor` | 1.0 | MLA | YaRN scale (1.0=off) |
| `mscale` | 1.0 | MLA | Attention temp scale |
| `mtp_depth` | 1 | MultiTokenPrediction | MTP heads |
| `mtp_loss_weight` | 0.3 | Pretrainer, compute_loss | λ for MTP loss |
| `dtype` | bf16 | Documentation | Training precision |
| `attn_impl` | sdpa | MLA | sdpa / manual / triton |
| `moe_dispatch` | stacked | DeepSeekMoE | stacked / triton_grouped |
| `weight_tying` | true | Transformer | Share embed/head |


### Dimensional constraints (422M)

These keys are not independent — violating a constraint crashes at first forward or silently mis-trains:

| Constraint | Rule | Example violation |
|---|---|---|
| Head divisibility | $d \bmod H = 0$ | dim=770, n_heads=12 |
| MLA head dims | $q_{\mathrm{k,nope}} + q_{\mathrm{k,rope}} = d_h$ | 48+24=72, 768/12=64 — **must match** |
| Vocab = embed rows | `vocab_size` = tokenizer len | 100000 vs 100018 |
| MoE width | experts use `moe_inter_dim` | dense layers use `inter_dim` |
| Triton MoE limit | `moe_inter_dim ≤ 256` for triton_grouped | 384 → auto-fallback stacked |

---

## Training Keys

| Key | 422M | Description |
|---|---|---|
| `micro_batch_size` | 8 | Per-GPU batch |
| `gradient_accumulation_steps` | 4 | Micro-steps per optim step |
| `total_steps` | 512000 | Optimiser steps |
| `warmup_steps` | 2000 | LR warmup |
| `lr` | 8.0e-4 | Base LR (μP-adjusted) |
| `min_lr_ratio` | 0.05 | Cosine floor |
| `weight_decay` | 0.1 | AdamW WD |
| `beta1`, `beta2` | 0.9, 0.95 | AdamW betas |
| `grad_clip` | 1.0 | Global norm clip |
| `grad_checkpoint` | true | Activation checkpointing |
| `compile` | true | torch.compile |
| `save_interval` | 4000 | Steps between checkpoints |
| `log_interval` | 50 | Steps between log lines |
| `mup_lr` | true | Enable μP scaling |
| `mup_lr_reference` | 6.0e-4 | Reference LR |
| `mup_lr_reference_params` | 757226496 | Reference param count |
| `nan_guard` | true | NaN rollback |
| `nan_guard_max_consecutive` | 5 | Streak before rollback |
| `bias_update_speed` | 0.001 | MoE bias step size |
| `bias_update_every` | 1 | Optim steps between bias updates |
| `save_dir` | checkpoints/pretrain_a100 | Checkpoint directory |

### Training budget arithmetic

Derive total token exposures from YAML:

$$
N_{\text{tokens/step}} = B_{\text{micro}} \times G_{\text{accum}} \times S = 8 \times 4 \times 2048 = 65536
$$

$$
N_{\text{total exposures}} = N_{\text{tokens/step}} \times T_{\text{steps}} = 65536 \times 512000 \approx 33.5 \times 10^9
$$

Unique corpus size ≈ 8.4B → **~4 epochs** over the mixture (with shuffle each epoch). Chinchilla-optimal **unique** tokens ≈ $20 \times 422\text{M} \approx 8.4$B — the corpus size matches; multi-epoch training is intentional for a single-GPU budget.


---

## Data Keys

| Key | 422M | Description |
|---|---|---|
| `train_data_path` | data/pretrain_chinchilla | Shard directory or .bin file |
| `tokenizer_path` | deepseek-ai/deepseek-coder-v2-lite | HF tokenizer for inference |

---

## Triton Dispatch Keys

| Key | Triton value | PyTorch default | Env guard |
|---|---|---|---|
| `attn_impl` | triton | sdpa | `ENABLE_TRITON_KERNELS=1` |
| `moe_dispatch` | triton_grouped | stacked | `ENABLE_TRITON_KERNELS=1` |

Without env var, `enforce_triton_env_var` force-backs at `Transformer.__init__` with a single warning.

### μP keys — full derivation

When `mup_lr: true`:

```python
new_lr = mup_lr_reference * (mup_lr_reference_params / total_params) ** 0.5
```

| Key | 422M value | Meaning |
|---|---|---|
| `mup_lr_reference` | 6.0e-4 | LR known-good on reference model |
| `mup_lr_reference_params` | 757,226,496 | Param count of reference (~DeepSeek-V3 class) |
| `lr` (YAML) | 8.0e-4 | Overwritten at init if μP enabled |

**Intuition:** Wider models have more parameters contributing to each update; sqrt scaling keeps effective update magnitude stable. See [foundations.md](foundations.md) §13.


**422M note:** `moe_inter_dim=384` and `dim=768` exceed Triton MoE kernel limit (256). Falls back to stacked at runtime.

---

## Config Nesting

Both forms work:

```yaml
# Nested (canonical YAML)
model:
  dim: 768
training:
  lr: 8e-4

# Flat (tests)
dim: 768
lr: 8e-4
```

`Transformer`: `config.get("model", config)`
`Pretrainer`: passes full YAML as `model_config`; sub-components unwrap as needed.

---

## Appendix — Quick comparison table

| | 422M A100 | 1650 2M |
|---|---|---|
| Params | ~422M | ~2M |
| VRAM | ~35 GB | <4 GB |
| Vocab | 100018 | 50257 |
| MoE layers | 16 | 2 |
| Experts | 20 routed, top-4 | 4 routed, top-1 |
| Seq len | 2048 | 128 |
| Compile | yes | no |
| Purpose | Production training | Smoke / CI |

## Worked example — tracing a config key

**Question:** What happens when you set `model.mtp_depth: 1` and `model.mtp_loss_weight: 0.3`?

1. `Pretrainer.__init__` reads `mtp_depth` from nested model config
2. If `mtp_depth > 0` and `mtp_weight > 0`, wraps `Transformer` in `MultiTokenPrediction`
3. `train_step` calls `mtp_wrapper.forward` → `compute_loss` with λ=0.3
4. μP scaling uses **post-wrap** param count
5. Checkpoint saves `mtp.mtp_modules.*` keys with `has_mtp: true` in meta

Trace in: `training/pretrain.py:155-163`, `models/mtp.py:67-117`.


---



---

## Hyperparameter Rationale (422M)

### Why `dim=768`, `n_layers=18`?

Chinchilla-optimal sizing: for ~422M params, depth and width balance FLOPs and memory. 18 layers fits in A100 VRAM with MoE at batch 8 × seq 2048.

### Why `n_dense_layers=2`?

DeepSeek-V3 uses early dense layers for stable low-level features before sparse routing. Matches paper schedule.

### Why `n_routed_experts=20`, `n_activated_experts=4`?

20 experts × 384 dim stores capacity; top-4 activates ~20% of routed compute. +1 shared expert always runs (5 FFN evaluations per MoE token).

### Why `lr=8e-4` (μP-adjusted)?

Base YAML value; overwritten at init to ~8.07e-4 when `mup_lr: true`. Tuned for stable BF16 training with grad clip 1.0.

### Why `warmup_steps=2000`?

~0.4% of total micro-steps. Sufficient for Adam moment estimates to stabilise without wasting compute.

### Why `min_lr_ratio=0.05`?

Floor LR = 5% of peak during cosine tail — prevents LR hitting zero while still allowing late-stage fine adjustment.

### Why `grad_clip=1.0`?

Standard for LLM pretraining. MoE routing spikes can produce large gradients; clip prevents single-step destabilisation.

### Why `mtp_loss_weight=0.3`?

DeepSeek-V3 uses λ ∈ [0.1, 0.3]. 0.3 gives meaningful auxiliary signal without overpowering main CE loss.

### Why `bias_update_speed=0.001`?

Small steps on router bias buffer — load balancing is slow control loop, not gradient descent.

---

## Sensitivity Guide

| If you change... | Watch for... |
|---|---|
| `micro_batch_size` ↑ | OOM — run microbench first |
| `max_seq_len` ↑ | Quadratic attention memory in training |
| `n_activated_experts` ↑ | More FFN compute, routing diversity |
| `lr` ↑ | NaN guard triggers — watch loss |
| `mup_lr: false` | May need manual LR retuning |
| `compile: false` | ~15-25% slower steps |




## CLI Overrides vs YAML

`training/pretrain.py` CLI flags override YAML at runtime:

| CLI flag | Overrides |
|---|---|
| `--data-path` | `data.train_data_path` |
| `--checkpoint-dir` | `training.save_dir` |
| `--no-compile` | `training.compile: false` |
| `--no-checkpoint` | `training.grad_checkpoint: false` |
| `--resume N` | Load step N before training |

Model architecture keys have **no CLI override** — edit YAML or use test fixtures.

---

## Environment + Config Interaction

| Env var | Config key | Interaction |
|---|---|---|
| `ENABLE_TRITON_KERNELS` | `attn_impl`, `moe_dispatch` | Env required for triton values |
| `TORCH_COMPILE_MODE` | `training.compile` | Only when compile true |
| `WANDB_PROJECT` | — | Logging only, not in YAML |

---

## FAQ

**Q: Can I train 422M config on 24GB GPU?**
A: No without major changes — halve batch and seq len, possibly disable MTP/compile.

**Q: Why two configs?**
A: 422M = production research. 1650 2M = structural smoke test on 4GB.

**Q: Where is `seed` in YAML?**
A: Not in 422M YAML — defaults to 42 in `TrainingConfig`. Add `seed: N` under `training:` to override.


## References

- `configs/pretrain_a100_422m.yaml`
- `configs/pretrain_1650_2m.yaml`
- `tests/conftest.py` — `cfg`, `small_cfg` fixtures
- [training.md](training.md) — how YAML maps to `TrainingConfig`

## Full YAML Walkthrough — 422M Config

Annotated excerpt from `configs/pretrain_a100_422m.yaml`:

```yaml
model:
  vocab_size: 100018      # MUST match DeepSeek tokenizer len()
  dim: 768                # Hidden width d; divisibility: d % n_heads == 0
  n_layers: 18            # 2 dense + 16 MoE
  n_heads: 12             # MLA heads; d_h = 64 = 48 nope + 16 rope (see MLA.md)
  n_dense_layers: 2       # Layers 0-1 use SwiGLU; 2-17 use MoE
  n_routed_experts: 20    # Router selects top-k from these
  n_shared_experts: 1     # Always executed (not routed)
  n_activated_experts: 4  # Top-k per token (routed only)
  inter_dim: 1536         # Dense FFN width (layers 0-1)
  moe_inter_dim: 384      # Per-expert width in MoE layers
  kv_lora_rank: 192       # MLA KV compression rank R
  q_lora_rank: 0          # 0 = no query compression at 422M
  qk_nope_head_dim: 48    # Content QK per head
  qk_rope_head_dim: 24    # RoPE QK per head (must sum to d_h=64)
  v_head_dim: 64          # Value dim per head
  max_seq_len: 2048       # MLA cache size + dataset window
  mtp_depth: 1            # One auxiliary head (predict t+2)
  mtp_loss_weight: 0.3    # λ in total loss
  attn_impl: sdpa         # PyTorch SDPA (FlashAttention backend)
  moe_dispatch: stacked   # Safe default; triton_grouped needs env var
  weight_tying: true      # embed.weight IS head.weight

training:
  micro_batch_size: 8
  gradient_accumulation_steps: 4   # Effective batch = 32 sequences
  total_steps: 512000            # Optimiser steps (not micro-steps)
  lr: 8.0e-4                     # Overwritten by μP if mup_lr: true
  nan_guard: true                # AGENTS.md: never disable without consent
```

### Mapping YAML → TrainingConfig

`training/pretrain.py:main()` reads nested YAML and constructs `TrainingConfig`:

| YAML path | TrainingConfig field |
|---|---|
| `training.micro_batch_size` | `batch_size` |
| `training.save_interval` | `save_every` |
| `training.log_interval` | `log_every` |
| `training.grad_clip` | `max_grad_norm` |
| `model.mtp_loss_weight` | `mtp_weight` |
| `data.train_data_path` | `data_path` |

The **entire** YAML dict is stored in `model_config` so `Transformer` can unwrap `model` section internally.

---

## 1650 Smoke Config — When to Use

`configs/pretrain_1650_2m.yaml` exists for **structural validation** on 4 GB GPUs:

- Same code paths (MLA, MoE, MTP) at `dim=64`
- `compile: false`, `grad_checkpoint: false` — fits without optimisations
- `vocab_size: 50257` (GPT-2) — different tokenizer family

**Do not** tune production hyperparameters on 1650 config — loss dynamics differ at 2M scale.

---

## Config Validation Checklist

Before launching a 15-hour run:

- [ ] `len(tokenizer) == model.vocab_size`
- [ ] `qk_nope_head_dim + qk_rope_head_dim == dim / n_heads`
- [ ] `data/pretrain_chinchilla/shard_*.bin` exists
- [ ] `micro_batch_size × grad_accum × seq_len` matches expected tokens/step
- [ ] If `attn_impl: triton` or `moe_dispatch: triton_grouped`, set `ENABLE_TRITON_KERNELS=1`
- [ ] `moe_inter_dim > 256` with triton_grouped → expect stacked fallback at runtime

<!-- docs:verified 2026-07-31 · 5a880d2 -->
