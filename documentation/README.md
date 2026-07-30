# DeepSeek-v3-Lite — Documentation Index

Educational technical references for every component of this project. Each doc follows the same structure as [MLA.md](MLA.md) and [moe.md](moe.md): theory, math, implementation walkthrough, appendices, FAQ, and invariants.

**New here?** Start with [foundations.md](foundations.md) → [getting_started.md](getting_started.md) → [architecture.md](architecture.md).

---

## Learning Path

| Phase | Read | Learn |
|---|---|---|
| 0. Foundations | [foundations.md](foundations.md) | Transformer basics, RoPE, Chinchilla, μP |
| 1. Overview | [getting_started.md](getting_started.md) | What this project is, key numbers, smoke tests |
| 2. Big picture | [architecture.md](architecture.md) | How all components connect |
| 3. Attention | [MLA.md](MLA.md) | Low-rank KV compression, absorption trick, RoPE |
| 4. MoE | [moe.md](moe.md) | Aux-loss-free routing, expert dispatch |
| 5. MTP | [mtp.md](mtp.md) | Multi-token prediction, speculative decode |
| 6. Wiring | [transformer.md](transformer.md) | Layer stack, generation, weight tying |
| 7. Training | [training.md](training.md) | Pretrain loop, μP, NaN guard |
| 8. Data | [data_pipeline.md](data_pipeline.md) | 8.4B-token corpus, tokenizer, shards |
| 9. Inference | [inference.md](inference.md) | KV-cache decode, speculative decoding |
| 10. Ops | [configs.md](configs.md), [scripts.md](scripts.md), [utils.md](utils.md) | YAML, launch, checkpoints |
| 11. Quality | [testing.md](testing.md) | Test corpus as oracle |
| 12. Advanced | [triton_kernels.md](triton_kernels.md) | Fused MoE + MLA kernels |
---


## Foundations

| File | Topics |
|------|--------|
| [foundations.md](foundations.md) | Language modeling, RMSNorm, attention, RoPE, SwiGLU, MoE intuition, Chinchilla, μP, BF16 |

## Component Docs

| File | Component(s) | Source |
|------|--------------|--------|
| [MLA.md](MLA.md) | Multi-Head Latent Attention | `models/mla.py`, `models/mla_triton.py` |
| [moe.md](moe.md) | AuxLossFreeGate + DeepSeekMoE | `models/moe.py`, `models/moe_triton.py` |
| [mtp.md](mtp.md) | Multi-Token Prediction + speculative decoder | `models/mtp.py`, `inference/speculative.py` |
| [transformer.md](transformer.md) | Top-level Transformer wiring | `models/transformer.py` |
| [training.md](training.md) | Pretrain loop, μP, NaN guard | `training/pretrain.py` |
| [data_pipeline.md](data_pipeline.md) | Data mixture + tokenizer | `data/prepare_data.py` |
| [inference.md](inference.md) | Generate + speculative decoder | `inference/` |
| [utils.md](utils.md) | Checkpoint, memory, logging | `utils/` |
| [triton_kernels.md](triton_kernels.md) | Fused Triton kernel design + roadmap | `models/*_triton.py` |

---

## Operations Docs

| File | Purpose |
|------|---------|
| [getting_started.md](getting_started.md) | Onboarding, commands, pitfalls |
| [architecture.md](architecture.md) | System diagram, data flows, file map |
| [configs.md](configs.md) | YAML key reference (422M + 1650) |
| [scripts.md](scripts.md) | launch_a100, microbench, step_time |
| [testing.md](testing.md) | Test suite guide + load-bearing tests |

---

## Configs

| Config | Purpose |
|---|---|
| `configs/pretrain_a100_422m.yaml` | Canonical Chinchilla-optimal recipe, ~422M params, 1× A100 80GB SXM |
| `configs/pretrain_1650_2m.yaml` | ~2M-param tiny config for GTX 1650 4GB smoke test |

See [configs.md](configs.md) for every key.

---

## MLA reference

**[MLA.md](MLA.md)** is the single canonical MLA doc: paper-grounded theory, absorption algebra, SDPA / manual / Triton paths, and a `models/mla.py` walkthrough. If prose and code disagree, **`models/mla.py` wins**.

---

## Load-bearing invariants (do not break)

| Invariant | Doc |
|---|---|
| AuxLossFreeGate bias is a **buffer**, not a Parameter | [moe.md](moe.md) |
| MLA absorption trick — SDPA vs manual paths | [MLA.md](MLA.md) |
| μP LR scaling after MTP-wrap param count | [training.md](training.md) |
| NaN guard with checkpoint rollback | [training.md](training.md) |
| Vocab 100,018 — embedding dim must match | [data_pipeline.md](data_pipeline.md) |
| Weight tying — head.weight = embed.weight | [transformer.md](transformer.md) |
| `ENABLE_TRITON_KERNELS=1` for Triton paths | [moe.md](moe.md), [triton_kernels.md](triton_kernels.md) |
| `test_mla_triton.py` not yet added (MLA Triton gap) | [triton_kernels.md](triton_kernels.md), [testing.md](testing.md) |
| `use_cache=False` during training | [training.md](training.md), [inference.md](inference.md) |

---

## Doc size reference

| Doc | ~Lines | Status |
|---|---|---|
| MLA.md | 1,424 | Comprehensive |
| moe.md | 924 | Comprehensive |
| training.md | 819 | Comprehensive |
| foundations.md | 768 | Comprehensive |
| triton_kernels.md | 738 | Comprehensive |
| mtp.md | 650 | Comprehensive |
| transformer.md | 618 | Comprehensive |
| data_pipeline.md | 617 | Comprehensive |
| utils.md | 575 | Comprehensive |
| architecture.md | 565 | Comprehensive |
| testing.md | 514 | Comprehensive |
| inference.md | 507 | Comprehensive |
| scripts.md | 476 | Comprehensive |
| getting_started.md | 455 | Comprehensive |
| configs.md | 430 | Comprehensive |
| **Total** | **10,201** | |
