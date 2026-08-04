# DeepSeek-v3-Lite — Documentation Index

> **Start here** after [[Docs/00_Getting_Started]]. Textbook-style reference for a pure-PyTorch DeepSeek-V3 LLM. Project postcard: [[Reference]].

---

## Learning Path

| Step | Chapter | Learner Outcome |
|---|---|---|
| 00 | [[Docs/00_Getting_Started\|00 Getting Started]] | Project overview, installation, smoke tests |
| 01 | [[Docs/01_Foundations\|01 Foundations]] | DeepSeek lineage (V1 $\rightarrow$ V2 $\rightarrow$ V3) |
| 02 | [[Docs/02_Model_Architecture\|02 Model Architecture]] | Full model topology, parameter budget, specs |
| 03 | [[Docs/03_Multi_Head_Latent_Attention\|03 MLA]] | Low-rank KV compression, decoupled RoPE |
| 04 | [[Docs/04_DeepSeekMoE\|04 DeepSeekMoE]] | Fine-grained expert routing, shared experts |
| 05 | [[Docs/05_Multi_Token_Prediction\|05 MTP]] | Multi-token prediction depth & speculative decoding |
| 06 | [[Docs/06_FP8_Mixed_Precision\|06 FP8 Mixed Precision]] | E4M3/E5M2 quantization & block scaling |
| 07 | [[Docs/07_DualPipe_Parallelism\|07 DualPipe]] | Bidirectional pipeline parallelism & overlap |
| 08 | [[Docs/08_Training_Pipeline\|08 Training]] | Loop, AdamW, cosine schedule, loss balancing |
| 09 | [[Docs/09_Data_Pipeline\|09 Data Pipeline]] | Dataset mix, tokenization, binary mmap |
| 10 | [[Docs/10_Inference_and_Serving\|10 Inference]] | Autoregressive sampling, MLA KV decompression |
| 11 | [[Docs/11_Operations_and_Testing\|11 Operations]] | Pytest suite, checkpoints, VRAM budget |
| 12 | [[Docs/12_Triton_Kernels\|12 Triton Kernels]] | Fused FP8 GEMM & MoE dispatch Triton kernels |
| 13 | [[Docs/13_Portfolio_Comparison\|13 Portfolio Comparison]] | Architecture comparison vs LLaMA-3, Mamba-3, HyMo |

---

## Learner Routing

| Question | Read First |
|---|---|
| I'm new — what is this? | [[Docs/00_Getting_Started]] |
| Model evolution (V1 $\rightarrow$ V2 $\rightarrow$ V3) | [[Docs/01_Foundations]] |
| How does MLA compress KV cache? | [[Docs/03_Multi_Head_Latent_Attention]] |
| How does DeepSeekMoE route tokens? | [[Docs/04_DeepSeekMoE]] |
| How does Multi-Token Prediction (MTP) work? | [[Docs/05_Multi_Token_Prediction]] |
| FP8 quantization & block scaling | [[Docs/06_FP8_Mixed_Precision]] |
| DualPipe pipeline parallelism | [[Docs/07_DualPipe_Parallelism]] |
| Train loop & loss balancing | [[Docs/08_Training_Pipeline]] |
| Dataset mixture & tokenization | [[Docs/09_Data_Pipeline]] |
| Autoregressive decode & serving | [[Docs/10_Inference_and_Serving]] |
| Tests & checkpointing | [[Docs/11_Operations_and_Testing]] |
| Triton kernels | [[Docs/12_Triton_Kernels]] |
| Portfolio comparison | [[Docs/13_Portfolio_Comparison]] |

---

## API Reference (`docs/reference/`)

Terse, symbol-anchored references — every signature, shape contract, default, and caller. The anchor gate (`tests/test_doc_refs.py`) verifies every code-symbol citation (file path + class/method) resolves.

| Doc | Covers |
|---|---|
| [[reference/R1_config_schema\|R1 Config Schema]] | Every YAML key, default, and its reader |
| [[reference/R2_transformer_api\|R2 Transformer API]] | `models/transformer.py` — Transformer, blocks, generate |
| [[reference/R3_mla_api\|R3 MLA API]] | `models/mla.py` — attention paths, cache contract |
| [[reference/R4_moe_api\|R4 MoE API]] | `models/moe.py` — gate, experts, dispatch |
| [[reference/R5_mtp_api\|R5 MTP API]] | `models/mtp.py` — blocks, loss math, alignment |
| [[reference/R6_triton_api\|R6 Triton API]] | Both kernels, dispatch gate, dim caps |
| [[reference/R7_training_api\|R7 Training API]] | `training/pretrain.py` — config, dataset, trainer |
| [[reference/R8_utils_api\|R8 Utils API]] | Checkpoints, logging, memory estimator |
| [[reference/R9_inference_api\|R9 Inference API]] | Generation, speculative decoding |

## Practical Guides (`docs/guides/`)

Procedural, checklist-driven operating manuals.

| Doc | Use when… |
|---|---|
| [[guides/G1_debugging_playbook\|G1 Debugging Playbook]] | NaN, shape errors, Triton fallback, cache bugs |
| [[guides/G2_mup_and_lr_tuning\|G2 μP & LR Tuning]] | Adjusting LR / μP reference, running an LR sweep |
| [[guides/G3_triton_development\|G3 Triton Development]] | Writing or extending a Triton kernel |
| [[guides/G4_benchmarking\|G4 Benchmarking]] | Measuring VRAM / throughput / MFU honestly |
| [[guides/G5_checkpoint_ops\|G5 Checkpoint Ops]] | Save / load / resume / disaster recovery |
| [[guides/G6_contributing\|G6 Contributing]] | Doc contract, gates, test & code conventions |

---

## Meta

| Resource | Location |
|---|---|
| Expansion plan & audit | [[docs_expansion_plan]] (planning artifact, not part of the learning path) |
| Project Reference | [[Reference]] |

<!-- docs:verified 2026-08-04 · 59aeef3 -->

---

## Doc size reference

| Doc | ~Lines | Status |
|---|---|---|
| 03_Multi_Head_Latent_Attention.md | 2,101 | Comprehensive |
| 08_Training_Pipeline.md | 2,097 | Comprehensive |
| 02_Model_Architecture.md | 1,995 | Comprehensive |
| 01_Foundations.md | 1,443 | Comprehensive |
| 04_DeepSeekMoE.md | 1,417 | Comprehensive |
| 05_Multi_Token_Prediction.md | 1,014 | Comprehensive |
| 10_Inference_and_Serving.md | 1,008 | Comprehensive |
| 12_Triton_Kernels.md | 852 | Comprehensive |
| 09_Data_Pipeline.md | 719 | Comprehensive |
| 11_Operations_and_Testing.md | 512 | Comprehensive |
| reference/R2_transformer_api.md | 463 | Comprehensive |
| reference/R4_moe_api.md | 455 | Comprehensive |
| reference/R5_mtp_api.md | 443 | Comprehensive |
| 00_Getting_Started.md | 376 | Comprehensive |
| guides/G1_debugging_playbook.md | 362 | Comprehensive |
| reference/R6_triton_api.md | 345 | Comprehensive |
| 07_DualPipe_Parallelism.md | 336 | Comprehensive |
| reference/R8_utils_api.md | 319 | Comprehensive |
| reference/R1_config_schema.md | 307 | Comprehensive |
| reference/R7_training_api.md | 287 | Comprehensive |
| docs_expansion_plan.md | 284 | Comprehensive |
| 06_FP8_Mixed_Precision.md | 282 | Comprehensive |
| guides/G5_checkpoint_ops.md | 273 | Comprehensive |
| reference/R9_inference_api.md | 271 | Comprehensive |
| guides/G2_mup_and_lr_tuning.md | 261 | Comprehensive |
| guides/G3_triton_development.md | 251 | Comprehensive |
| guides/G4_benchmarking.md | 251 | Comprehensive |
| reference/R3_mla_api.md | 245 | Comprehensive |
| guides/G6_contributing.md | 240 | Comprehensive |
| 13_Portfolio_Comparison.md | 229 | Comprehensive |
| **Total** | **19,438** | |
