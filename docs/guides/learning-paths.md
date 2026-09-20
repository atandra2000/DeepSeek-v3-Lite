# Learning Paths — How to Read the DeepSeek-v3-Lite Docs

> Audience: all levels. This guide is the navigational meta layer for the
> documentation tree — it teaches no model topic itself; it tells you which
> doc to read next depending on what you already know and what you want
> from the codebase. The linear 14-chapter curriculum in
> [getting-started.md](getting-started.md) is the alternative presentation
> of the same corpus; the three paths below assume prior steps and end with
> a working mental model of the whole stack.

The docs are organized into three tracks: **concepts**
([concepts/](../concepts/foundations.md), theory and architecture from
first principles), **references** ([references/](../references/R1_config_schema.md),
symbol-anchored API walkthroughs), and **guides** (this folder — operations
and how-to), plus the two pipeline docs [training.md](../training.md) and
[inference.md](../inference.md). Every citation to code uses symbol anchors
(`models/mla.py:MultiHeadLatentAttention` style), verified by
`tests/test_doc_refs.py`.

---

## Beginner path — What this model is and how it works

Who this is for: you know basic PyTorch and what next-token prediction is,
and want to understand the DeepSeek-V3 architecture from the ground up. No
prior MLA or MoE background required.

| Step | Doc | What you will know after |
|------|-----|--------------------------|
| 1 | [getting-started.md](getting-started.md) | How the repo is laid out, how to run the 1650 smoke config on a laptop, and what the canonical 422M numbers are. |
| 2 | [foundations.md](../concepts/foundations.md) (§lineage + task) | The DeepSeek V1→V2→V3 lineage, the causal-LM objective, and why RMSNorm/SwiGLU/RoPE are the shared substrate. |
| 3 | [foundations.md](../concepts/foundations.md) (§topology) | The full 18-layer topology (2 dense + 16 MoE), the parameter budget, and the config→code routing table. |
| 4 | [attention-and-precision.md](../concepts/attention-and-precision.md) (MLA half) | Multi-Head Latent Attention: low-rank KV compression (`kv_lora_rank` = 192), the matrix-absorption trick, decoupled RoPE, and why the KV cache shrinks. |
| 5 | [moe-mtp.md](../concepts/moe-mtp.md) (MoE half) | Fine-grained experts + shared expert, and the auxiliary-loss-free bias feedback loop that replaces the balance loss. |
| 6 | [moe-mtp.md](../concepts/moe-mtp.md) (MTP half) | The depth-1 Multi-Token Prediction head, its loss weighting, and speculative-decoding theory. |
| 7 | [R2 — Transformer API](../references/R2_transformer_api.md) + [R3 — MLA API](../references/R3_mla_api.md) | The code tour: every block in `models/transformer.py:Transformer` and `models/mla.py:MultiHeadLatentAttention` with shape contracts. |

## Intermediate path — Train it and understand the numerics

Who this is for: you have read the beginner path and want to run, tune, or
resume training — including the μP learning-rate contract and the data
pipeline.

| Step | Doc | What you will know after |
|------|-----|--------------------------|
| 1 | [foundations.md](../concepts/foundations.md) (config section) | Every config key's meaning and where it is consumed. |
| 2 | [R1 — Config Schema](../references/R1_config_schema.md) | The full YAML schema: every key, default, 1650-variant, and its reader symbol. |
| 3 | [training.md](../training.md) | The applied pretrain loop: AdamW, gradient accumulation, warmup→cosine schedule (`training/pretrain.py:make_warmup_cosine_lambda`), μP scaling, NaN guard, atomic checkpointing. |
| 4 | [data-pipeline.md](../concepts/data-pipeline.md) | How the 8.0B-token corpus is prepared once in `shared_data/`, tokenized by the shim, packed into mmap shards, and consumed by `PretrainDataset`. |
| 5 | [R7 — Training API](../references/R7_training_api.md) | `training/pretrain.py:Pretrainer` internals — step budget, accumulation boundary, checkpoint writer. |
| 6 | [G2 — μP & LR Tuning](G2_mup_and_lr_tuning.md) | How to transfer the reference LR across widths and run an honest LR sweep. |

## Expert path — Operate, optimize, and serve

Who this is for: you want the kernels, the failure modes, the measurement
discipline, and the serving path.

| Step | Doc | What you will know after |
|------|-----|--------------------------|
| 1 | [kernels-and-ops.md](../concepts/kernels-and-ops.md) (ops half) | The real test suite, atomic safetensors checkpoint system, VRAM budget, and CI walkthrough. |
| 2 | [R6 — Triton API](../references/R6_triton_api.md) | Both kernels (fused MLA attention, grouped-GEMM MoE), the double-opt-in guard (`models/_triton_dispatch.py:enforce_triton_env_var`), and the dim caps. |
| 3 | [G3 — Triton Development](G3_triton_development.md) | How to write/extend a kernel while keeping the pure-PyTorch reference test green. |
| 4 | [G1 — Debugging Playbook](G1_debugging_playbook.md) | NaN triage, shape errors, Triton fallback diagnosis, cache bugs. |
| 5 | [G4 — Benchmarking](G4_benchmarking.md) | How to measure VRAM / throughput / MFU honestly, without claiming unmeasured numbers. |
| 6 | [G5 — Checkpoint Ops](G5_checkpoint_ops.md) | Save / load / resume / disaster-recovery procedures for the atomic checkpoint set. |
| 7 | [inference.md](../inference.md) + [R9 — Inference API](../references/R9_inference_api.md) | Autoregressive decode, MLA KV decompression, sampling, speculative decoding, CLI. |
| 8 | [parallelism.md](../concepts/parallelism.md) | DualPipe / 1F1B / all-to-all theory — paper-spec context, not implemented here. |

---

## What each track is for

- **Concepts** build the mental model; they are self-contained and can be
  read without the code open.
- **References** are the code-keyed counterparts; keep the cited file open
  beside them.
- **Guides** assume the concepts and give procedures; `contributing.md`
  documents the doc contract and both gates
  (`tests/test_doc_refs.py`, `scripts/check_docs.py`).