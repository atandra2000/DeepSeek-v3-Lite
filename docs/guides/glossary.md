# Glossary — DeepSeek-v3-Lite

> Notation, component terms, and the config keys that appear throughout the
> documentation. Config semantics live in [R1 — Config Schema](../references/R1_config_schema.md);
> this page is the quick lookup. Every code citation uses symbol anchors
> verified by `tests/test_doc_refs.py`.

---

## Notation

| Symbol | Meaning | Canonical value |
|--------|---------|-----------------|
| `dim` (`d_model`) | Residual-stream width | 768 |
| `n_layers` | Decoder blocks | 18 = 2 dense + 16 MoE |
| `n_heads` | Query heads (`n_q`) | 12 |
| `qk_nope_head_dim` | Per-head non-positional QK width | 48 |
| `qk_rope_head_dim` (`d_R`) | Decoupled RoPE width per head | 24 |
| `v_head_dim` | Value width per head | 64 |
| `kv_lora_rank` (`d_c`) | Compressed K/V latent rank | 192 |
| `q_lora_rank` | Q LoRA rank (0 = full-rank `wq`) | 0 |
| `n_routed_experts` / `n_activated_experts` | Routed / per-token active experts | 20 / 4 |
| `moe_inter_dim` | Routed-expert FFN width | 384 |
| `inter_dim` | Dense-layer FFN width | 1536 |
| `max_seq_len` | Training context window | 2048 |
| `vocab_size` | Tokenizer rows (incl. byte-fallback) | 100 018 |
| `[B, T, d]` | Batch × sequence × hidden tensor shape | — |

## Multi-Head Latent Attention (MLA)

| Term | Definition | Where implemented |
|------|------------|-------------------|
| **MLA** | Attention whose K and V are projections of a low-rank shared latent, shrinking the KV cache. | `models/mla.py:MultiHeadLatentAttention` |
| **Latent compression** | The `[B, T, d_c]` cached latent replaces per-head K/V (`d_c` = 192 ≪ 12 × 64). | `models/mla.py:MultiHeadLatentAttention.forward` |
| **Matrix absorption** | Folding `wkv_b` into the attention algebra so per-head K/V are never materialized on the decode path. | theory in [attention-and-precision.md](../concepts/attention-and-precision.md) |
| **Decoupled RoPE** | RoPE applied to a separate `qk_rope_head_dim` = 24 slice because position-dependent keys cannot be absorbed into the latent. | `models/mla.py:MultiHeadLatentAttention._extend_rope` |
| **YaRN** | Context extension via `rope_factor` > 1.0 with attention-scale `mscale`; dormant at canonical `rope_factor` = 1.0. | `models/mla.py:MultiHeadLatentAttention` |
| **`attn_impl`** | Backend selector: `"sdpa"` (default), `"manual"`, or `"triton"` (fused kernel). | `models/mla.py:MultiHeadLatentAttention.forward` |

## DeepSeekMoE

| Term | Definition | Where implemented |
|------|------------|-------------------|
| **DeepSeekMoE** | Fine-grained routed experts + one always-on shared expert, replacing the dense FFN in MoE layers. | `models/moe.py:DeepSeekMoE` |
| **Aux-loss-free balancing** | Load balance via a per-expert bias updated out-of-band every `bias_update_every` steps — no auxiliary loss term. | `models/moe.py:AuxLossFreeGate` |
| **Routed / shared expert** | Top-4 routed experts per token (`models/moe.py:Expert`); 1 shared expert always active. | `models/moe.py:DeepSeekMoE` |
| **Stacked dispatch** | Default dispatch: experts re-stacked into a single bmm per forward — never cached across optimizer steps. | `models/moe.py:DeepSeekMoE.forward` |
| **`moe_dispatch`** | `"stacked"` (default) or `"triton_grouped"` (fused grouped-GEMM, opt-in). | `models/moe.py:DeepSeekMoE.__init__` |

## Multi-Token Prediction (MTP)

| Term | Definition | Where implemented |
|------|------------|-------------------|
| **MTP** | Depth-1 auxiliary head predicting token *t+1* from a shared trunk; densifies training signal. | `models/mtp.py:MultiTokenPrediction` |
| **MTP loss weight** | Scalar weight of the auxiliary prediction loss (0.3 canonical). | `models/mtp.py:MultiTokenPrediction.compute_loss` |
| **Shared head** | MTP reuses the main model's embedding and LM head. | `models/mtp.py:MTPBlock` |
| **Speculative decoding** | MTP drafts tokens, the main model verifies — ~0.8 acceptance, up to 2× throughput on smoke tests. | `inference/speculative.py` |

## Training and precision

| Term | Definition | Where implemented |
|------|------------|-------------------|
| **μP LR scaling** | Transfer the reference LR `6.0e-4` @ 757 M across widths via `lr_ref · (n_ref / n)^0.5` — 8.07e-4 @ 418.7 M with MTP. | [G2](G2_mup_and_lr_tuning.md), [training.md](../training.md) |
| **Warmup→cosine schedule** | 2000-step linear ramp, cosine decay to `min_lr_ratio` = 0.05. | `training/pretrain.py:make_warmup_cosine_lambda` |
| **Atomic checkpoint** | `model_step_N.safetensors` + `optim_step_N.pt` + `meta_step_N.json` written so a crash never leaves a torn checkpoint. | [G5](G5_checkpoint_ops.md), [kernels-and-ops.md](../concepts/kernels-and-ops.md) |
| **NaN guard** | Loss finite-check with checkpoint rollback. | [training.md](../training.md), [G1](G1_debugging_playbook.md) |
| **BF16 autocast + FP32 master weights** | Compute in BF16 autocast; AdamW state stays FP32; matmuls on TF32 for CUDA. | [foundations.md](../concepts/foundations.md) (mixed-precision section) |
| **Double opt-in** | Triton runs only with a per-config key (`attn_impl: "triton"` / `moe_dispatch: "triton_grouped"`) **and** `ENABLE_TRITON_KERNELS=1`; otherwise one startup warning and PyTorch fallback. | `models/_triton_dispatch.py:enforce_triton_env_var` |
| **`HAS_TRITON`** | Import-time flag set `False` when `triton` is missing; keeps the repo importable on macOS/CPU. | `models/mla_triton.py`, `models/moe_triton.py` |

## Data pipeline

| Term | Definition |
|------|------------|
| **8.0B-token universal corpus** | Prepared once in workspace `shared_data/` (7-source mixture; see [data-pipeline.md](../concepts/data-pipeline.md)). |
| **Shim** | `data/prepare_data.py` — per-project entry that imports the workspace pipeline (not vendored). |
| **mmap shard** | Packed binary shard read via memory map by `training/pretrain.py:PretrainDataset`. |
| **Tokenizer** | `deepseek-ai/deepseek-coder-v2-lite`, vocab 100 018; EOS id 100 017, PAD id 100 016. |
| **1650 smoke config** | `configs/pretrain_1650_2m.yaml` — same MLA/MoE/MTP invariants at ~2M params for laptop GPUs. |

## Acronyms

| Acronym | Expansion |
|---------|-----------|
| MLA | Multi-Head Latent Attention |
| MoE | Mixture of Experts |
| MTP | Multi-Token Prediction |
| FFN | Feed-Forward Network |
| RoPE | Rotary Position Embedding |
| KV cache | Key/Value cache |
| SDPA | Scaled-Dot-Product Attention (`torch.nn.functional.scaled_dot_product_attention`) |
| FA2 | Flash Attention 2 (recompute-backward pattern) |
| MFU | Model FLOPs Utilization |
| μP | Maximal Update Parameterization |
| YaRN | Yet another RoPE extension (context-length scaling) |
| FP8 E4M3 / E5M2 | 8-bit float formats (paper-spec in this repo, not implemented) |