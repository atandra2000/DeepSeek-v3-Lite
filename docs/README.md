# DeepSeek-v3-Lite — Documentation Index

> Navigation map for the DeepSeek-v3-Lite documentation: `concepts/` (theory + architecture), `references/` (symbol-anchored API docs), `guides/` (how-to / ops), plus the top-level pipeline docs `training.md` and `inference.md`. Every code-symbol citation (path + class/method) is machine-verified by `tests/test_doc_refs.py`; prose hygiene and relative links are checked by `scripts/check_docs.py` — both run in CI.

## Quick Reference

A faithful, from-scratch, raw-PyTorch reproduction of the **DeepSeek-V3 architecture** at Chinchilla-optimal scale — **~412 M total / ~185 M active parameters per token** (411.6M deduped; 418.7M with MTP) — built to be fully inspectable (no HuggingFace Trainer, no Lightning). It implements the four architectural pillars of DeepSeek-V3 at a scale that fits on one A100 80 GB:

1. **Multi-Head Latent Attention (MLA)** — low-rank KV compression with decoupled RoPE and the matrix-absorption trick.
2. **DeepSeekMoE with auxiliary-loss-free load balancing** — fine-grained routed experts + a shared expert, balanced by a control-theory bias update rather than an auxiliary loss.
3. **Multi-Token Prediction (MTP)** — a depth-1 auxiliary prediction head that densifies the training signal and enables speculative decoding at inference.
4. **μP learning-rate scaling** — maximal-update parameterization so the LR transfers across model widths.

FP8 mixed precision and DualPipe bidirectional pipeline parallelism are DeepSeek-V3 *paper* techniques documented for completeness in [MLA & Mixed Precision](concepts/attention-and-precision.md) and [DualPipe Parallelism](concepts/parallelism.md); they are **not implemented** here because the target is single-GPU BF16 training.

| Parameter / Feature | Canonical Value |
|---|---|
| **Total / Active params** | **~422 M nominal / 411.6 M deduped (weight-tied) · ~185 M active per token** |
| **Compute target** | 8.4 B tokens (Chinchilla-optimal for 422 M) · 512 000 steps |
| **Topology** | 18 layers = **2 dense + 16 MoE** · `d_model = 768` |
| **Attention heads** | 12 query heads · `d_head_nope = 48` · `d_head_rope = 24` · `d_v = 64` |
| **MLA compression** | KV LoRA rank `d_c = 192` · decoupled RoPE `d_R = 24` · `q_lora_rank = 0` (full-rank Q projection) |
| **MoE layout** | 20 routed experts · **4 activated per token** · 1 shared expert · `moe_inter_dim = 384` |
| **Dense FFN** | SwiGLU · `inter_dim = 1536` (layers 0–1) |
| **MTP** | depth 1 · loss weight 0.3 · shared embedding + output head with main model |
| **Precision** | **BF16** autocast · FP32 AdamW master weights · TF32 matmul on CUDA |
| **Attention impl** | `sdpa` (default) · optional fused `triton` kernel, opt-in via `ENABLE_TRITON_KERNELS=1` |
| **MoE dispatch** | `stacked` (default) · optional fused `triton_grouped` grouped-GEMM, opt-in |
| **Optimizer / schedule** | AdamW (β = 0.9 / 0.95, wd = 0.1) · 2000-step linear warmup → cosine decay to 5% · grad clip 1.0 |
| **μP LR** | reference LR 6.0e-4 @ 757 M params → scaled `lr_ref · (n_ref / n) ^ 0.5` |
| **Context** | `max_seq_len = 2048` · `rope_theta = 10000` · `rope_factor = 1.0` (YaRN off at train; decode-time scaling only) |
| **Tokenizer** | `deepseek-ai/deepseek-coder-v2-lite` · vocab 100 018 (EOS 100 017, PAD 100 016) |
| **Weight tying** | `true` — LM head shares the embedding table |
| **Checkpointing** | atomic safetensors (`model_step_N.safetensors` + `optim_step_N.pt` + `meta_step_N.json`) · gradient checkpointing on |
| **Hardware target** | 1 × A100 80 GB SXM (single-GPU, no distributed) |

> A second toy config `configs/pretrain_1650_2m.yaml` (dim=64, 4 layers, 4 experts, 1 active, GPT-2 50 257 vocab) exists for local smoke tests on a laptop GPU — it preserves every MLA/MoE/MTP invariant at miniature scale.

## Concepts (`concepts/`)

Theory and architecture, consolidated from the original spine chapters 01–13.

| Doc | Covers | Status |
|---|---|---|
| [Foundations & Architecture](concepts/foundations.md) | DeepSeek lineage (V1 → V2 → V3), causal LM objective, RMSNorm, SwiGLU, attention, RoPE, KV caching, Chinchilla, μP, mixed precision; full 18-layer topology, parameter budget, memory budget, file map, config→code routing; portfolio comparison vs GPT-OSS-Lite, LLaMA-3-Lite, Mamba-3-Lite | implemented |
| [MLA & Mixed Precision](concepts/attention-and-precision.md) | MLA: low-rank KV compression, absorption trick, decoupled RoPE, KV-cache lifecycle, Triton path; FP8 E4M3/E5M2 scheme, tile/block scaling | implemented / FP8 paper-spec |
| [DeepSeekMoE & MTP](concepts/moe-mtp.md) | Fine-grained experts, shared expert, aux-loss-free bias feedback loop; MTP depth-1 loss, length-alignment algebra, shared head, speculative-decoding theory | implemented |
| [DualPipe Parallelism](concepts/parallelism.md) | Pipeline bubbles, GPipe, 1F1B, interleaved schedules, all-to-all, bidirectional overlap | paper-spec, not implemented |
| [Data Pipeline](concepts/data-pipeline.md) | Shared 8.0B-token universal pipeline, DeepSeek shim, tokenizer deep dive, mixture, shard format, `PretrainDataset` consumption | implemented |
| [Operations, Testing & Triton Kernels](concepts/kernels-and-ops.md) | Real pytest suite, atomic checkpoint system, VRAM budget, CI walkthrough, doc↔code gate; fused MLA attention + grouped-GEMM MoE Triton kernels, double-opt-in guard, register-pressure math | implemented (Triton opt-in) |

## References (`references/`)

Terse, symbol-anchored references — every signature, shape contract, default, and caller. The anchor gate (`tests/test_doc_refs.py`) verifies every code-symbol citation (file path + class/method) resolves.

| Doc | Covers |
|---|---|
| [R1 — Config Schema](references/R1_config_schema.md) | Every YAML key, default, and its reader |
| [R2 — Transformer API](references/R2_transformer_api.md) | `models/transformer.py` — Transformer, blocks, generate |
| [R3 — MLA API](references/R3_mla_api.md) | `models/mla.py` — attention paths, cache contract |
| [R4 — MoE API](references/R4_moe_api.md) | `models/moe.py` — gate, experts, dispatch |
| [R5 — MTP API](references/R5_mtp_api.md) | `models/mtp.py` — blocks, loss math, alignment |
| [R6 — Triton API](references/R6_triton_api.md) | Both kernels, dispatch gate, dim caps |
| [R7 — Training API](references/R7_training_api.md) | `training/pretrain.py` — config, dataset, trainer |
| [R8 — Utils API](references/R8_utils_api.md) | Checkpoints, logging, memory estimator |
| [R9 — Inference API](references/R9_inference_api.md) | Generation, speculative decoding |

## Guides (`guides/`)

Procedural, checklist-driven operating manuals.

| Doc | Use when… |
|---|---|
| [Getting Started](guides/getting-started.md) | New to the repo: install, smoke tests, first run, learning path |
| [G1 — Debugging Playbook](guides/G1_debugging_playbook.md) | NaN, shape errors, Triton fallback, cache bugs |
| [G2 — μP & LR Tuning](guides/G2_mup_and_lr_tuning.md) | Adjusting LR / μP reference, running an LR sweep |
| [G3 — Triton Development](guides/G3_triton_development.md) | Writing or extending a Triton kernel |
| [G4 — Benchmarking](guides/G4_benchmarking.md) | Measuring VRAM / throughput / MFU honestly |
| [G5 — Checkpoint Ops](guides/G5_checkpoint_ops.md) | Save / load / resume / disaster recovery |
| [Contributing](guides/contributing.md) | Doc contract, gates, test & code conventions |

## Pipeline Docs

| Doc | Covers |
|---|---|
| [Training Pipeline](training.md) | Pretrain loop, AdamW, gradient accumulation, LR scheduler, μP scaling, `PretrainDataset`, NaN guard, checkpointing, YAML reference; data pipeline (shim, tokenizer, mixture, shards) |
| [Inference & Serving](inference.md) | Autoregressive decode, KV cache prefill/decode, sampling theory, speculative decoding, CLI |

## Configs

- `configs/pretrain_a100_422m.yaml` — canonical 422M A100 recipe (full key-by-key reference in [R1](references/R1_config_schema.md) and [Training Pipeline](training.md) §Part B).
- `configs/pretrain_1650_2m.yaml` — ~2M-param GTX 1650 4GB smoke-test config; same MLA/MoE/MTP invariants at miniature scale, GPT-2 vocab.

## Doc size reference

| Doc | ~Lines | Status |
|---|---|---|
| foundations.md | 3,666 | Comprehensive |
| training.md | 2,823 | Comprehensive |
| moe-mtp.md | 2,439 | Comprehensive |
| attention-and-precision.md | 2,383 | Comprehensive |
| kernels-and-ops.md | 1,364 | Comprehensive |
| inference.md | 1,007 | Comprehensive |
| data-pipeline.md | 716 | Comprehensive |
| R2_transformer_api.md | 473 | Comprehensive |
| R4_moe_api.md | 453 | Comprehensive |
| R5_mtp_api.md | 452 | Comprehensive |
| getting-started.md | 381 | Comprehensive |
| G1_debugging_playbook.md | 360 | Comprehensive |
| R6_triton_api.md | 343 | Comprehensive |
| parallelism.md | 338 | Comprehensive |
| R8_utils_api.md | 324 | Comprehensive |
| R1_config_schema.md | 304 | Comprehensive |
| R7_training_api.md | 285 | Comprehensive |
| G5_checkpoint_ops.md | 279 | Comprehensive |
| R9_inference_api.md | 269 | Comprehensive |
| G2_mup_and_lr_tuning.md | 266 | Comprehensive |
| G3_triton_development.md | 258 | Comprehensive |
| G4_benchmarking.md | 257 | Comprehensive |
| R3_mla_api.md | 250 | Comprehensive |
| contributing.md | 245 | Comprehensive |
| **Total** | **19,935** | |

## References

- [Training Pipeline](training.md) — loop, μP, NaN guard, YAML reference
- [Inference & Serving](inference.md) — decode, KV cache, speculative decoding
- `tests/test_doc_refs.py` — machine-verified doc↔code anchor gate
- `scripts/check_docs.py` — prose hygiene, link/path lint, size table
