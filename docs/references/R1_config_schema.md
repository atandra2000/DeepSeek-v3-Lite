# DeepSeek-v3-Lite — R1 Config Schema Reference

> Reference chapter · Part of the DeepSeek-V3-Lite API reference (R1–R9). Companion learning chapters: [08 Training Pipeline](../training.md) (training loop semantics), [09 Data Pipeline](../concepts/data-pipeline.md) (data keys), [11 Operations and Testing](../concepts/kernels-and-ops.md) (fixture configs), [G2 mup and lr tuning](../guides/G2_mup_and_lr_tuning.md) (μP LR).

## 1. What this is

The training entry point is a single YAML file: `--config configs/pretrain_a100_422m.yaml` (the canonical 411.6M run) or `configs/pretrain_1650_2m.yaml` (the ~2M smoke run). This reference documents **every key** in both files — type, effective default when absent, the code that reads it (symbol-anchored), and the interactions that make a key inert or overridden. Two configs exist; there is no config-schema class — the schema *is* the union of `.get(...)`/`[...]` accesses in the readers listed below.

**Loaders.** `training/pretrain.py:main` does `yaml.safe_load`, splits the dict into `training` and `data` sections, and maps the training keys onto a `training/pretrain.py:TrainingConfig` dataclass. The *whole* YAML dict is passed through as `model_config`, and both `models/transformer.py:Transformer.__init__` and `models/mtp.py:MultiTokenPrediction.__init__` call `config.get("model", config)`, so a config with a bare `model:` section and one with the keys at top level are both valid. `models/_triton_dispatch.py:enforce_triton_env_var` mutates the same dict **in place** inside `Transformer.__init__` (before any layers are built), which is why the Triton-dispatch keys below must be read from the model-config dict, not from the file.

## 2. The two files, side by side

| Section | Key | Type | `pretrain_a100_422m.yaml` | `pretrain_1650_2m.yaml` |
|---|---|---|---|---|
| model | `vocab_size` | int | 100018 | 50257 (GPT-2) |
| model | `dim` | int | 768 | 64 |
| model | `n_layers` | int | 18 | 4 |
| model | `n_heads` | int | 12 | 4 |
| model | `n_dense_layers` | int | 2 | 2 |
| model | `n_routed_experts` | int | 20 | 4 |
| model | `n_shared_experts` | int | 1 | 1 |
| model | `n_activated_experts` | int | 4 | 1 |
| model | `inter_dim` | int | 1536 | 128 |
| model | `moe_inter_dim` | int | 384 | 32 |
| model | `kv_lora_rank` | int | 192 | 16 |
| model | `q_lora_rank` | int | 0 | 0 |
| model | `qk_nope_head_dim` | int | 48 | 16 |
| model | `qk_rope_head_dim` | int | 24 | 16 |
| model | `v_head_dim` | int | 64 | 16 |
| model | `max_seq_len` | int | 2048 | 128 |
| model | `rope_theta` | float | 10000 | 10000 |
| model | `rope_factor` | float | 1.0 | 1.0 |
| model | `mscale` | float | 1.0 | 1.0 |
| model | `mtp_depth` | int | 1 | 1 |
| model | `mtp_loss_weight` | float | 0.3 | 0.3 |
| model | `dtype` | str | `bf16` | `bf16` |
| model | `attn_impl` | str | `"sdpa"` | `"sdpa"` |
| model | `moe_dispatch` | str | `"stacked"` | `"stacked"` |
| model | `weight_tying` | bool | `true` | `true` |
| training | `micro_batch_size` | int | 8 | 2 |
| training | `gradient_accumulation_steps` | int | 4 | 2 |
| training | `total_steps` | int | 512000 | 10000 |
| training | `warmup_steps` | int | 2000 | 50 |
| training | `lr` | float | 8.0e-4 | 1.0e-3 |
| training | `min_lr_ratio` | float | 0.05 | 0.05 |
| training | `weight_decay` | float | 0.1 | 0.1 |
| training | `beta1` | float | 0.9 | 0.9 |
| training | `beta2` | float | 0.95 | 0.95 |
| training | `grad_clip` | float | 1.0 | 1.0 |
| training | `grad_checkpoint` | bool | `true` | `false` |
| training | `compile` | bool | `true` | `false` |
| training | `save_interval` | int | 4000 | 100 |
| training | `log_interval` | int | 50 | 10 |
| training | `mup_lr` | bool | `true` | `false` |
| training | `mup_lr_reference` | float | 6.0e-4 | — (absent) |
| training | `mup_lr_reference_params` | int | 757226496 | — (absent) |
| training | `nan_guard` | bool | `true` | `true` |
| training | `nan_guard_max_consecutive` | int | 5 | 5 |
| training | `log_per_component_params` | bool | `true` | `true` |
| training | `bias_update_speed` | float | 0.001 | 0.001 |
| training | `bias_update_every` | int | 1 | 1 |
| training | `save_dir` | str | `"checkpoints/pretrain_a100"` | `"checkpoints/pretrain_1650_2m"` |
| data | `train_data_path` | str | `"data/pretrain_chinchilla"` | `"data/pretrain_chinchilla"` |
| data | `tokenizer_path` | str | `"deepseek-ai/deepseek-coder-v2-lite"` | `"gpt2"` |

## 3. Model section

### 3.1 Shape keys (required — no default; `config[key]` raises `KeyError`)

| Key | Type | Canonical | Smoke | Read by |
|---|---|---|---|---|
| `vocab_size` | int | 100018 | 50257 | `models/transformer.py:Transformer.__init__` (embedding + LM head rows); `models/mtp.py:MTPModule.__init__`; `training/pretrain.py:main` → `TrainingConfig.vocab_size` → `training/pretrain.py:PretrainDataset.__init__` (token-id bound) |
| `dim` | int | 768 | 64 | `models/transformer.py:TransformerBlock.__init__`, `models/transformer.py:SwiGLUFFN.__init__`, `models/mla.py:MultiHeadLatentAttention.__init__`, `models/moe.py:DeepSeekMoE.__init__`, `models/mtp.py:MTPBlock.__init__`, `models/mtp.py:MTPModule.__init__` |
| `n_layers` | int | 18 | 4 | `models/transformer.py:Transformer.__init__` (block count) |
| `n_heads` | int | 12 | 4 | `models/mla.py:MultiHeadLatentAttention.__init__`; `models/mtp.py:MTPBlock.__init__` (`nn.MultiheadAttention`) |
| `n_dense_layers` | int | 2 | 2 | `models/transformer.py:TransformerBlock.__init__` — layers with `layer_id < n_dense_layers` get `SwiGLUFFN`, the rest get `DeepSeekMoE` |
| `n_routed_experts` | int | 20 | 4 | `models/moe.py:DeepSeekMoE.__init__`; `models/moe.py:AuxLossFreeGate.__init__` (gate rows + bias buffer) |
| `n_shared_experts` | int | 1 | 1 | `models/moe.py:DeepSeekMoE.__init__` |
| `n_activated_experts` | int | 4 | 1 | `models/moe.py:AuxLossFreeGate.__init__` → `topk` in `models/moe.py:AuxLossFreeGate.forward` |
| `inter_dim` | int | 1536 | 128 | `models/transformer.py:SwiGLUFFN.__init__` (dense FFN width); `models/mtp.py:MTPBlock.__init__` |
| `moe_inter_dim` | int | 384 | 32 | `models/moe.py:DeepSeekMoE.__init__` (expert width). Also gates the Triton grouped path: `triton_grouped` requires `moe_inter_dim ≤ 256` **and** `dim ≤ 256` — see §3.3 |
| `kv_lora_rank` | int | 192 | 16 | `models/mla.py:MultiHeadLatentAttention.__init__` (compressed K/V rank) |
| `q_lora_rank` | int | 0 | 0 | `models/mla.py:MultiHeadLatentAttention.__init__` — `0` selects the direct `wq` projection; `> 0` selects the `wq_a → q_norm → wq_b` LoRA path |
| `qk_nope_head_dim` | int | 48 | 16 | `models/mla.py:MultiHeadLatentAttention.__init__` (per-head non-positional QK dim; `qk_head_dim = qk_nope + qk_rope`) |
| `qk_rope_head_dim` | int | 24 | 16 | `models/mla.py:MultiHeadLatentAttention.__init__`, `models/mla.py:MultiHeadLatentAttention._extend_rope` (RoPE frequency ladder) |
| `v_head_dim` | int | 64 | 16 | `models/mla.py:MultiHeadLatentAttention.__init__` |
| `max_seq_len` | int | 2048 | 128 | `models/transformer.py:Transformer.__init__`; `models/mla.py:MultiHeadLatentAttention.__init__` (KV-cache bound, RoPE growth cap in `_extend_rope`); `training/pretrain.py:main` → `TrainingConfig.max_seq_len` → `PretrainDataset` window size |
| `rope_theta` | float | 10000 | 10000 | `models/mla.py:MultiHeadLatentAttention.__init__` → `_extend_rope` (base of the inverse-frequency schedule) |

**Purpose (one line each).** `vocab_size` = token-id universe, must equal `len(tokenizer)` (the DeepSeek tokenizer is 100,018 with `byte_fallback`, see [09 Data Pipeline](../concepts/data-pipeline.md)); `dim` = residual-stream width; `n_layers` = total blocks; `n_heads` = attention heads; `n_dense_layers` = how many leading blocks use dense FFNs; `n_routed_experts` / `n_shared_experts` / `n_activated_experts` = MoE fan-out; `inter_dim` vs `moe_inter_dim` = dense vs expert FFN width; `kv_lora_rank` / `q_lora_rank` = MLA compression ranks; `qk_nope_head_dim` / `qk_rope_head_dim` / `v_head_dim` = per-head MLA dims; `max_seq_len` = context window and cache cap; `rope_theta` = RoPE base.

### 3.2 Scaled / gated keys (defaulted with `.get`)

| Key | Type | Default when absent | Canonical | Smoke | Read by |
|---|---|---|---|---|---|
| `rope_factor` | float | `1.0` | 1.0 | 1.0 | `models/mla.py:MultiHeadLatentAttention.__init__` — YaRN extension factor |
| `mscale` | float | `1.0` | 1.0 | 1.0 | `models/mla.py:MultiHeadLatentAttention.__init__` — attention softmax scale |
| `mtp_depth` | int | `0` (trainer) / `1` (`MultiTokenPrediction`) | 1 | 1 | `training/pretrain.py:Pretrainer.__init__`; `models/mtp.py:MultiTokenPrediction.__init__` |
| `mtp_loss_weight` | float | `0.0` (trainer) / `0.3` (MTP module) | 0.3 | 0.3 | `training/pretrain.py:main` → `TrainingConfig.mtp_weight`; `models/mtp.py:MultiTokenPrediction.compute_loss` |
| `attn_impl` | str | `"sdpa"` | `"sdpa"` | `"sdpa"` | `models/mla.py:MultiHeadLatentAttention.__init__`, `models/mla.py:MultiHeadLatentAttention.forward`; `models/_triton_dispatch.py:enforce_triton_env_var` |
| `moe_dispatch` | str | `"stacked"` | `"stacked"` | `"stacked"` | `models/moe.py:DeepSeekMoE.__init__`, `models/moe.py:DeepSeekMoE.forward`; `models/_triton_dispatch.py:enforce_triton_env_var` |
| `weight_tying` | bool | `False` | `true` | `true` | `models/transformer.py:Transformer.__init__`; `training/pretrain.py:Pretrainer.save_checkpoint` (drops the duplicate `head.weight` from the state dict) |

**Interactions.**

- **`moe_dispatch="triton_grouped"`** routes routed-expert compute to
  `models/moe_triton.py:triton_grouped_moe_dispatch` via `models/moe.py:DeepSeekMoE._routed_forward_triton`. It is gated twice: (a) `models/_triton_dispatch.py:enforce_triton_env_var` force-backs it to `"stacked"` (with one warning) unless `ENABLE_TRITON_KERNELS=1`; (b) at canonical dims (`moe_inter_dim=384`, `dim=768`), `triton_grouped_moe_dispatch` raises `ValueError` (BLOCK_I/BLOCK_D exceed the 256 register cap) and `DeepSeekMoE.forward` falls back to `"stacked"` with a one-time warning. **Consequence: the Triton MoE path is unreachable in the canonical config** — it only runs at smoke-config dims (`moe_inter_dim=32`, `dim=64`), and only with the env var set.
- **`attn_impl="triton"`** routes the MLA core to
  `models/mla_triton.py:triton_mla_attention` via `models/mla.py:MultiHeadLatentAttention._forward_triton`. Same dispatch guard: force-backed to `"sdpa"` without `ENABLE_TRITON_KERNELS=1`; on CPU/Mac the Triton import raises `ImportError` and `models/mla.py:MultiHeadLatentAttention.forward` falls back to `"sdpa"` (one-time, sticky). Canonical dims are valid for this kernel.
- **`rope_factor > 1.0`** switches on YaRN rescaling in
  `models/mla.py:MultiHeadLatentAttention.__init__`: `mscale = 0.1 * mscale_raw * log(rope_factor) + 1.0`.
- **`mscale` is doubly inert** at canonical settings: with `rope_factor = 1.0` the
  formula above is skipped (mscale stays raw), and `softmax_scale = qk_head_dim**-0.5 * mscale**2` is only used when `max_seq_len > 4096` — the canonical 2048 and smoke 128 both take `softmax_scale = qk_head_dim**-0.5`.
- **`mtp_depth` and `mtp_loss_weight` gate each other.** The MTP wrapper is built in
  `training/pretrain.py:Pretrainer.__init__` only when `mtp_depth > 0 and config.mtp_weight > 0.0`. `mtp_depth: 1` with `mtp_loss_weight: 0.0` silently disables MTP (params drop from 418.7M to 411.6M); `mtp_depth: 0` disables it regardless of the weight. When active, `MultiTokenPrediction.compute_loss` adds `mtp_weight * mtp_loss` to the main loss. `MultiTokenPrediction.__init__` reads the keys as `config["mtp_depth"]` / `config.get("mtp_loss_weight", 0.3)` when the keys are present at top level of the config it receives (it is handed the full YAML dict, so the `model:` section matches), else falls back to an optional `mtp:` block (`depth`, `weight` — absent from both configs).
- **`weight_tying: true`** aliases `head.weight = embed.weight` in
  `Transformer.__init__`; `Pretrainer.save_checkpoint` then omits `head.weight` from the safetensors state (≈`vocab_size × dim × 4` bytes saved), and `training/pretrain.py:Pretrainer.load_checkpoint` restores it through the shared `embed.weight` (load uses `strict=False`).

## 4. Training section

All keys are read by `training/pretrain.py:main` and mapped onto `training/pretrain.py:TrainingConfig` fields (field name in parentheses); the loop then consumes the dataclass. Defaults below are the `t.get(key, <default>)` values in `main` — note several differ from the dataclass defaults, because `main` always passes an explicit value.

| Key → `TrainingConfig` field | Type | Default | Canonical | Smoke | Consumed by |
|---|---|---|---|---|---|
| `micro_batch_size` → `batch_size` | int | `8` | 8 | 2 | `training/pretrain.py:Pretrainer.train` (`DataLoader(..., batch_size=..., num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=8, drop_last=True)`) |
| `gradient_accumulation_steps` | int | `4` | 4 | 2 | `training/pretrain.py:Pretrainer.train_step` (`is_opt_step = (micro_step+1) % ... == 0`); scheduler horizon |
| `total_steps` → `max_steps` | int | `20000` | 512000 | 10000 | `training/pretrain.py:Pretrainer.train` loop budget; scheduler horizon (§4.1) |
| `warmup_steps` | int | `2000` | 2000 | 50 | `training/pretrain.py:make_warmup_cosine_lambda` (linear ramp) |
| `lr` | float | `2.2e-4` | 8.0e-4 | 1.0e-3 | `training/pretrain.py:Pretrainer.__init__` (AdamW); **overridden when `mup_lr: true`** (§4.1) |
| `min_lr_ratio` | float | `0.1` | 0.05 | 0.05 | `make_warmup_cosine_lambda` (cosine floor) |
| `weight_decay` | float | `0.1` | 0.1 | 0.1 | `Pretrainer.__init__` — applied **only to `dim ≥ 2` params** (embeddings, all weights); 1-D params (norms, gate bias) get 0 |
| `beta1` | float | `0.9` | 0.9 | 0.9 | `Pretrainer.__init__` (`AdamW(..., betas=(beta1, beta2), fused=cuda)`) |
| `beta2` | float | `0.95` | 0.95 | 0.95 | same |
| `grad_clip` → `max_grad_norm` | float | `1.0` | 1.0 | 1.0 | `training/pretrain.py:Pretrainer.train_step` (`nn.utils.clip_grad_norm_` on optimizer steps) |
| `grad_checkpoint` | bool | `True` | `true` | `false` | `main` (`and not args.no_checkpoint`) → `Transformer(use_checkpoint=...)` → `models/transformer.py:Transformer._run_layers` — active **only while `model.training`** |
| `compile` → `compile_model` | bool | `True` | `true` | `false` | `main` (`and not args.no_compile`) → `torch.compile(..., mode=os.environ.get("TORCH_COMPILE_MODE", "max-autotune"), fullgraph=False)`; skipped if `torch.compile` is absent |
| `save_interval` → `save_every` | int | `1000` | 4000 | 100 | `Pretrainer.train` (`global_step % save_every == 0` → `training/pretrain.py:Pretrainer.save_checkpoint`) |
| `log_interval` → `log_every` | int | `100` | 50 | 10 | `utils/logging.py:init_logging`; `Pretrainer.train` metric logging |
| `mup_lr` | bool | `False` | `true` | `false` | `Pretrainer.__init__` — switches on μP LR scaling (§4.1) |
| `mup_lr_reference` | float | `6.0e-4` | 6.0e-4 | — | `Pretrainer.__init__` |
| `mup_lr_reference_params` | int | `757226496` | 757226496 | — | `Pretrainer.__init__` (the external μP anchor — an unverifiable reference-model count, kept as-is) |
| `nan_guard` | bool | `False` | `true` | `true` | `training/pretrain.py:Pretrainer.train_step` (skip backward on NaN/Inf); `Pretrainer.train` streak + rollback logic |
| `nan_guard_max_consecutive` | int | `5` | 5 | 5 | `Pretrainer.train` — after N consecutive guard hits, restore latest checkpoint (or `RuntimeError` if none) |
| `log_per_component_params` | bool | `True` | `true` | `true` | `Pretrainer.__init__` (per-component param table via `models/transformer.py:count_parameters`) |
| `bias_update_speed` | float | `0.001` | 0.001 | 0.001 | `training/pretrain.py:Pretrainer._update_moe_bias` → `models/moe.py:AuxLossFreeGate.update_bias` |
| `bias_update_every` | int | `10` | 1 | 1 | `Pretrainer.train_step` — bias update every N **optimizer** steps |
| `save_dir` → `checkpoint_dir` | str | `"checkpoints/pretrain"` | `"checkpoints/pretrain_a100"` | `"checkpoints/pretrain_1650_2m"` | `main` (`or args.checkpoint_dir`) → `CheckpointManager(config.checkpoint_dir)` |

**Purpose (one line each).** `micro_batch_size` = tokens-per-forward batch;  `gradient_accumulation_steps` = micro-batches per optimizer step; `total_steps` = micro-batch budget (global-step loop bound); `warmup_steps`/`lr`/`min_lr_ratio` = schedule; `weight_decay`/`beta1`/`beta2`/`grad_clip` = optimizer; `grad_checkpoint` = activation recompute; `compile` = `torch.compile`; `save_interval`/`log_interval` = cadence; `mup_lr*` = μP LR rescaling; `nan_guard*` = NaN rollback;  `log_per_component_params` = startup param audit; `bias_update*` = aux-loss-free MoE gate-bias schedule; `save_dir` = checkpoint root.

### 4.1 Interactions that change semantics

- **`mup_lr: true` discards `lr`.** In `training/pretrain.py:Pretrainer.__init__`:
  `new_lr = mup_lr_reference * (mup_lr_reference_params / total_params) ** 0.5`, then `config.lr = new_lr`. With the canonical config the effective LR is $6.0\times10^{-4}\sqrt{757\,226\,496 / N}$, i.e. **8.14e-4** at $N = 411\,632\,256$ (base) and **8.07e-4** at $N = 418\,713\,984$ (with MTP) — the YAML's `lr: 8.0e-4` is never used in the canonical run. The smoke config sets `mup_lr: false`, so its `lr: 1.0e-3` is used verbatim. The two `mup_lr_reference*` keys are therefore inert in the smoke config (present in neither, and not consulted).
- **`total_steps` is in micro-batch space; the cosine horizon is in optimizer-step
  space.** `Pretrainer.__init__` computes `opt_steps = max(1, max_steps // gradient_accumulation_steps)` and feeds that to `make_warmup_cosine_lambda`; the loop budget `global_step < max_steps` counts micro-batches. Canonical: 512,000 micro-batches → 128,000 optimizer steps; warmup 2,000 opt-steps ≈ 1.6% of the arc. Changing `gradient_accumulation_steps` therefore changes the LR trajectory, not just the update cadence.
- **`grad_checkpoint` is a training-time-only switch** (`if self.use_checkpoint and
  self.training` in `models/transformer.py:Transformer._run_layers`) — irrelevant for inference, which never passes it.
- **`bias_update_every` counts optimizer steps**, so with
  `gradient_accumulation_steps: 4` the canonical gate bias is touched every 4 micro steps (`_opt_steps % bias_update_every == 0` in `Pretrainer.train_step`), i.e. once per optimizer step at the canonical setting of 1.

## 5. Data section

| Key | Type | Default | Canonical | Smoke | Read by |
|---|---|---|---|---|---|
| `train_data_path` | str | `"data/pretrain_data.bin"` | `"data/pretrain_chinchilla"` | `"data/pretrain_chinchilla"` | `training/pretrain.py:main` (`or args.data_path`) → `TrainingConfig.data_path` → `training/pretrain.py:PretrainDataset.__init__` — a **directory** selects the sharded layout (`shard_*.bin`, mmap'd, `PretrainDataset._locate` bisect), a **file** the single-tensor layout |
| `tokenizer_path` | str | `"deepseek-ai/deepseek-coder-v2-lite"` | `"deepseek-ai/deepseek-coder-v2-lite"` | `"gpt2"` | **Only `inference/generate.py:main`** (`AutoTokenizer.from_pretrained`). **Inert in training** — `Pretrainer` never touches it; the data pipeline fixes the tokenizer in `data/prepare_data.py` (see [09 Data Pipeline](../concepts/data-pipeline.md)) |

The tokenizer choice must agree with `vocab_size`: the canonical pair is `deepseek-coder-v2-lite` (100,018 rows, `byte_fallback`), the smoke pair is `gpt2` (50,257 rows, no HF auth). `training/pretrain.py:main` also reads two *model-section* keys into `TrainingConfig` (`vocab_size`, `max_seq_len`) for dataset sizing; the model code reads them again from the dict. There is no check that the two agree — that is a run-configuration invariant (test-guarded; see [11 Operations and Testing](../concepts/kernels-and-ops.md)).

## 6. Dispatch, env vars, and CLI overrides

**Triton dispatch** — `models/_triton_dispatch.py:enforce_triton_env_var`, called from `models/transformer.py:Transformer.__init__`, rewrites the model dict in place:

| Config value | Rewritten to | Unless |
|---|---|---|
| `attn_impl: "triton"` | `"sdpa"` | `ENABLE_TRITON_KERNELS=1` |
| `moe_dispatch: "triton_grouped"` | `"stacked"` | `ENABLE_TRITON_KERNELS=1` |

A single warning is logged listing every forced key. Both canonical and smoke configs set the PyTorch defaults, so the guard is a no-op for them.

**Other env vars.** `TORCH_COMPILE_MODE` (default `"max-autotune"`) selects the `torch.compile` mode in `Pretrainer.__init__`. `ENABLE_TRITON_KERNELS` is the only Triton gate; without it the fused kernels are unreachable regardless of config.

**CLI overrides** (`training/pretrain.py:main`): `--data-path` / `--checkpoint-dir` override `data.train_data_path` / `training.save_dir`; `--resume <step>` loads a checkpoint before training; `--no-checkpoint` / `--no-compile` force-disable `grad_checkpoint` / `compile` even when the YAML says `true` (`and not args.…`).

## 7. Keys read from the same dict but absent from both configs

Optional keys the readers fall back to, not present in either file:

| Key | Default | Reader |
|---|---|---|
| `seed` | `42` | `training/pretrain.py:main` → `TrainingConfig.seed` → `Pretrainer.__init__` (`torch.manual_seed`) and `Pretrainer.train` (DataLoader shuffle generator) |
| `route_scale` | `1.0` | `models/moe.py:AuxLossFreeGate.__init__` → `.forward` (post-normalization route weight scale) |
| `bias_upper_threshold` | `0.10` | `models/moe.py:AuxLossFreeGate.__init__` → `update_bias` |
| `bias_lower_threshold` | `0.10` | `models/moe.py:AuxLossFreeGate.__init__` → `update_bias` |
| `mtp:` block (`depth`, `weight`) | `1`, `0.3` | `models/mtp.py:MultiTokenPrediction.__init__` — fallback schema when `mtp_depth` is absent from the dict |
| `model` wrapper itself | (required) | `models/transformer.py:Transformer.__init__` / `models/mtp.py:MultiTokenPrediction.__init__` (`config.get("model", config)`) — top-level keys also work |

## 8. Inert-key summary

- **`dtype`** — read by **no code** in the repo. `training/pretrain.py:Pretrainer.__init__`
  hardcodes `self.amp_dtype = torch.bfloat16`; the key is documentation only.
- **`tokenizer_path`** — training never reads it; only `inference/generate.py:main`.
- **`mscale`** — inert at `rope_factor ≤ 1.0` **and** at `max_seq_len ≤ 4096` (both configs).
- **`mup_lr_reference` / `mup_lr_reference_params`** — inert when `mup_lr: false` (smoke config).
- **`lr`** — inert in the canonical config because `mup_lr: true` recomputes it.
- **`moe_dispatch: "triton_grouped"`** — unreachable at canonical dims (register cap);
  would also be force-backed to `stacked` without `ENABLE_TRITON_KERNELS=1`.
- **`attn_impl: "triton"`** — force-backed to `sdpa` without `ENABLE_TRITON_KERNELS=1`;
  on CPU/Mac always falls back to `sdpa`.
- **`mtp_loss_weight: 0.0` with `mtp_depth ≥ 1`** — MTP wrapper not built (silently).

## References
- Model-shape consequences: [02 Model Architecture](../concepts/foundations.md) (budgets), [03 Multi Head Latent Attention](../concepts/attention-and-precision.md) (MLA dims, mscale), [04 DeepSeekMoE](../concepts/moe-mtp.md) (gate + dispatch), [05 Multi Token Prediction](../concepts/moe-mtp.md) (MTP loss).
- Loop semantics behind the training keys: [08 Training Pipeline](../training.md); μP math: [G2 mup and lr tuning](../guides/G2_mup_and_lr_tuning.md); scheduler closed form: `training/pretrain.py:make_warmup_cosine_lambda`.
- Data keys: [09 Data Pipeline](../concepts/data-pipeline.md); Triton dispatch details: [12 Triton Kernels](../concepts/kernels-and-ops.md) and [R6 triton api](../references/R6_triton_api.md).
- Fixture configs used by the test suite (same schema, tiny dims): [11 Operations and Testing](../concepts/kernels-and-ops.md).
- The dataclass itself: [R7 training api](../references/R7_training_api.md) (`TrainingConfig`, `Pretrainer`, `PretrainDataset`).

