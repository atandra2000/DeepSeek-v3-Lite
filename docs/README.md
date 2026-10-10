# How to read this book

This portal is a book about building a DeepSeek-V3-style transformer from scratch in raw PyTorch. It has 25 chapters in four parts. Read it in order, or jump to the part you need.

Every citation names a symbol (`models/mla.py:MultiHeadLatentAttention`), never a line number, so the references survive edits. `tests/test_doc_refs.py` resolves each one and fails CI when a citation goes stale.

## The parts

| Part | What it covers |
|------|----------------|
| **Part I: Foundations** | What the model is: DeepSeek lineage, the 18-layer topology, and the parameter and memory budgets. |
| **Part II: The data** | How 8.0B tokens become a sharded, memory-mapped stream, and the DeepSeek shim that feeds it. |
| **Part III: The training run** | MLA, DeepSeekMoE, MTP, the DualPipe paper-spec, and the kernels and ops that keep one step honest. |
| **Part IV: Reference** | Inference, the operating guides, and the symbol-anchored API references. Open these when you need a signature or a config key, not a narrative. |

## Where to start

- **New to the model.** Read Part I in order. [Foundations and topography](concepts/foundations.md) is the longest chapter, and it assumes no prior reading.
- **You want to run it.** Start with the [Overview](../README.md) for hardware and configuration, then read [Getting started](guides/getting-started.md) and [The training pipeline](training.md).
- **You want the numerics.** Read [MLA and mixed precision](concepts/attention-and-precision.md) for the absorption trick and the KV-cache arithmetic, or [DeepSeekMoE and MTP](concepts/moe-mtp.md) for the bias-feedback loop and the length-alignment algebra.
- **It broke.** [G1 — Debugging playbook](guides/G1_debugging_playbook.md) first, then [G5 — Checkpoint ops](guides/G5_checkpoint_ops.md) for resume and recovery.

## Visual tour

The [Interactive Visual Systems Guide](diagrams/deepseek_visual_guide.html) is four verified Archify maps: model architecture, optimizations and memory, data pipeline, and training loop. Each map carries tensor shapes, memory arithmetic, and source links. The [verification receipts](diagrams/RECEIPTS.md) record what was verified for each map.

## Which source file is documented where

| Module | Chapter | Key symbols |
|--------|---------|-------------|
| `models/transformer.py` | [Foundations and topography](concepts/foundations.md) (topology and budgets), [R2 — Transformer API](references/R2_transformer_api.md) (full walkthrough) | `models/transformer.py:Transformer`, `models/transformer.py:Transformer.generate` |
| `models/mla.py` | [MLA and mixed precision](concepts/attention-and-precision.md) (absorption theory), [R3 — MLA API](references/R3_mla_api.md) (every option) | `models/mla.py:MultiHeadLatentAttention`, `models/mla_triton.py:triton_mla_attention` |
| `models/moe.py` | [DeepSeekMoE and MTP](concepts/moe-mtp.md) (bias-feedback theory), [R4 — MoE API](references/R4_moe_api.md) (gate and dispatch) | `models/moe.py:AuxLossFreeGate`, `models/moe.py:DeepSeekMoE`, `models/moe_triton.py:triton_grouped_moe_dispatch` |
| `models/mtp.py` | [DeepSeekMoE and MTP](concepts/moe-mtp.md) (MTP loss math), [R5 — MTP API](references/R5_mtp_api.md) (heads and alignment) | `models/mtp.py:MultiTokenPrediction`, `inference/speculative.py:SpeculativeDecoder` |
| `training/pretrain.py` | [The training pipeline](training.md) (full walkthrough), [R7 — Training API](references/R7_training_api.md) (config and trainer) | `training/pretrain.py:TrainingConfig`, `training/pretrain.py:Pretrainer.train_step` |
| `inference/generate.py`, `inference/speculative.py` | [Inference and speculative decoding](inference.md) (decode paths), [R9 — Inference API](references/R9_inference_api.md) | `inference/generate.py:generate_interactive`, `inference/speculative.py:SpeculativeDecoder` |
| `data/prepare_data.py` | [The data pipeline](concepts/data-pipeline.md) (shim and stages) | `data/prepare_data.py:main` |
| `utils/checkpoint.py` | [G5 — Checkpoint ops](guides/G5_checkpoint_ops.md) (atomic layout), [R8 — Utils API](references/R8_utils_api.md) (contract) | `utils/checkpoint.py:CheckpointManager` |
| `models/mla_triton.py`, `models/moe_triton.py` | [Operations and Triton kernels](concepts/kernels-and-ops.md) (kernel math), [G3 — Triton development](guides/G3_triton_development.md) (the recipe), [R6 — Triton API](references/R6_triton_api.md) (contracts) | `models/mla_triton.py:triton_mla_attention`, `models/moe_triton.py:triton_grouped_moe_dispatch` |
| `tests/*` | [Operations and Triton kernels](concepts/kernels-and-ops.md) (the suite and the CI ladder) | `tests/test_doc_refs.py:test_doc_anchors_resolve` |

The repository README is Chapter 1 of this book, not a separate page.