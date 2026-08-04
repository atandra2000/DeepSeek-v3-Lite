# DeepSeek-v3-Lite — Quick Reference

> **Status:** Architecture (MLA, DeepSeekMoE, MTP, aux-loss-free balancing, μP LR scaling, atomic safetensors checkpointing) is **implemented and CPU smoke-tested**. A full single-GPU pre-training run has **not yet been executed** (`checkpoints/` is empty). **FP8 mixed precision and DualPipe pipeline parallelism are paper-spec only — no FP8/DualPipe code is in this repo** (see [[Docs/06_FP8_Mixed_Precision]], [[Docs/07_DualPipe_Parallelism]]). This repo trains in **BF16 on a single GPU**.

---

## Project Overview

A from-scratch, raw-PyTorch reproduction of the **DeepSeek-V3 architecture** at Chinchilla-optimal scale — **~412 M total / ~185 M active parameters per token** (411.6M deduped; 418.7M with MTP) — built to be fully inspectable (no HuggingFace Trainer, no Lightning). It implements the four architectural pillars of DeepSeek-V3 at a scale that fits on one A100 80 GB:

1. **Multi-Head Latent Attention (MLA)** — low-rank KV compression with decoupled RoPE and the matrix-absorption trick.
2. **DeepSeekMoE with auxiliary-loss-free load balancing** — fine-grained routed experts + a shared expert, balanced by a control-theory bias update rather than an auxiliary loss.
3. **Multi-Token Prediction (MTP)** — a depth-1 auxiliary prediction head that densifies the training signal and enables speculative decoding at inference.
4. **μP learning-rate scaling** — maximal-update parameterization so the LR transfers across model widths.

FP8 mixed precision and DualPipe bidirectional pipeline parallelism are DeepSeek-V3 *paper* techniques documented for completeness in [[Docs/06_FP8_Mixed_Precision]] and [[Docs/07_DualPipe_Parallelism]]; they are **not implemented** here because the target is single-GPU BF16 training.

---

## Technical Specifications (canonical — `configs/pretrain_a100_422m.yaml`)

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

> [!NOTE] A second toy config `configs/pretrain_1650_2m.yaml` (dim=64, 4 layers, 4 experts, 1 active, GPT-2 50 257 vocab) exists for local smoke tests on a laptop GPU — it preserves every MLA/MoE/MTP invariant at miniature scale.

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/atandra2000/DeepSeek-v3-Lite.git
cd DeepSeek-v3-Lite

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run CPU smoke tests (< 2 s, no CUDA/Triton required)
python3 -m pytest tests/ -v

# 4. (Optional) Prepare the 8.4B-token shared corpus → data/pretrain_chinchilla/shard_*.bin
python3 data/prepare_data.py --stage pretrain

# 5. Launch pre-training on 1 × A100
python3 training/pretrain.py --config configs/pretrain_a100_422m.yaml

# 6. Interactive generation (after a checkpoint exists)
python3 -m inference.generate --config configs/pretrain_a100_422m.yaml \
       --checkpoint checkpoints/pretrain_a100 --use_speculative
```

Entry points: **`training/pretrain.py`** (train), **`inference/generate.py`** (decode). There is no `src/train.py`.

---

## Learning Path

| Step | Chapter | Focus |
|---|---|---|
| 00 | [[Docs/00_Getting_Started]] | Onboarding, canonical numbers, smoke tests |
| 01 | [[Docs/01_Foundations]] | DeepSeek lineage (V1 → V2 → V3) & design philosophy |
| 02 | [[Docs/02_Model_Architecture]] | Full topology, tensor shapes, parameter budget |
| 03 | [[Docs/03_Multi_Head_Latent_Attention]] | MLA: low-rank KV compression, matrix absorption, decoupled RoPE |
| 04 | [[Docs/04_DeepSeekMoE]] | Fine-grained experts, shared expert, aux-loss-free bias updates |
| 05 | [[Docs/05_Multi_Token_Prediction]] | MTP depth, training loss, speculative decoding |
| 06 | [[Docs/06_FP8_Mixed_Precision]] | E4M3/E5M2, tile/block scaling — **paper-spec, not implemented** |
| 07 | [[Docs/07_DualPipe_Parallelism]] | Bidirectional pipeline parallelism — **paper-spec, not implemented** |
| 08 | [[Docs/08_Training_Pipeline]] | Pretrain loop, AdamW, μP, NaN guard, checkpointing |
| 09 | [[Docs/09_Data_Pipeline]] | Shared universal pipeline, tokenizer shim, uint32 mmap shards |
| 10 | [[Docs/10_Inference_and_Serving]] | Autoregressive decode, MLA KV decompression, speculative decode |
| 11 | [[Docs/11_Operations_and_Testing]] | Real test suite, checkpoint format, VRAM budget |
| 12 | [[Docs/12_Triton_Kernels]] | Real fused kernels: MLA attention + grouped MoE GEMM (BF16) |
| 13 | [[Docs/13_Portfolio_Comparison]] | Architecture vs GPT-OSS-Lite, LLaMA-3-Lite, Mamba-3-Lite |

---

## Load-Bearing Invariants

> **Do not break these** — every one is asserted by a test or enforced in code.

- **MLA ranks:** `kv_lora_rank = 192`, `qk_rope_head_dim = 24`, `qk_nope_head_dim = 48`, `v_head_dim = 64`. The KV cache stores only the compressed latent `c_t^{KV} ∈ ℝ^192` + the decoupled RoPE key `∈ ℝ^24` per token — **not** full per-head K/V.
- **DeepSeekMoE layout:** 20 routed experts (top-4 selected) + 1 always-active shared expert, per MoE layer.
- **Auxiliary-loss-free balancing:** the gate's `bias` buffer is updated out-of-band by `update_gate_bias` (a control update, `b_i ± speed`), never added to the loss. The task gradient stays pure.
- **Weight tying:** `head.weight IS embed.weight` (one tensor). Checkpoint save dedups it; `load_state_dict(strict=False)` restores via `embed`.
- **Atomic checkpoints:** all three files (`model_*.safetensors`, `optim_*.pt`, `meta_*.json`) written via `tempfile` → `os.replace`; a checkpoint is only "complete" when all three exist.

> [!WARNING] **Paper-spec invariants (DeepSeek-V3, NOT this repo):** FP8 `128×128` block-wise weight scaling and E4M3/E5M2 mixed precision are properties of the published DeepSeek-V3, not of this reproduction. This repo is BF16 throughout. See [[Docs/06_FP8_Mixed_Precision]] §"Status in this repo".

---

## File Map

| Path | Role |
|---|---|
| `models/transformer.py` | `Transformer`, `TransformerBlock`, `SwiGLUFFN`, `count_parameters` |
| `models/mla.py` | `MultiHeadLatentAttention` (SDPA + Triton paths, KV cache, RoPE) |
| `models/moe.py` | `DeepSeekMoE`, `AuxLossFreeGate`, `Expert` (stacked + Triton dispatch) |
| `models/mtp.py` | `MTPBlock`, `MTPModule`, `MultiTokenPrediction` wrapper |
| `models/mla_triton.py` | Fused MLA attention Triton kernel + PyTorch reference |
| `models/moe_triton.py` | Fused grouped-GEMM MoE dispatch Triton kernel + reference |
| `models/_triton_dispatch.py` | `enforce_triton_env_var` — force-back guard |
| `training/pretrain.py` | `Pretrainer`, `PretrainDataset`, `TrainingConfig`, `main()` |
| `inference/generate.py` | Interactive generation CLI + tokenizer load |
| `inference/speculative.py` | `SpeculativeDecoder` (MTP draft → verify → accept) |
| `utils/checkpoint.py` | `CheckpointManager` (atomic safetensors) |
| `utils/memory.py` | VRAM budget estimator (params/optim/KV/activations) |
| `utils/logging.py` | Training logger |
| `data/prepare_data.py` | Thin shim over `LLM/shared_data/` universal pipeline |
| `configs/pretrain_a100_422m.yaml` | Canonical 422 M config |
| `tests/` | 7 test files + `conftest.py` (CPU-only small configs) |

---

## License

Distributed under the Apache 2.0 License. See `LICENSE` for details.