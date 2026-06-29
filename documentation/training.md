# Training — `training/pretrain.py`

## Stack

- **TF32** — `torch.backends.cuda.matmul.allow_tf32 = True`,
  `cudnn.allow_tf32 = True`, `set_float32_matmul_precision("high")`,
  `cudnn.benchmark = True`. Enabled only when CUDA is available.
- **`torch.compile(mode="max-autotune")`** — applied to the
  `training_model` (which may be the MTP wrapper). `fullgraph=False`.
  Mode overridable via `TORCH_COMPILE_MODE` env var. `--no-compile`
  disables.
- **FlashAttention-2** — via `F.scaled_dot_product_attention` in MLA's
  SDPA path.
- **BF16 autocast** — `torch.amp.autocast("cuda", dtype=torch.bfloat16)`.
  No `GradScaler` (BF16 on Ampere/Blackwell — see workspace rule 15).
- **FP32 AdamW master** — `AdamW(..., fused=True)`, betas=(0.9, 0.95),
  wd=0.1. Params split by `dim() >= 2` (decay) vs `< 2` (no decay).
  Deduplicates shared/tied params by `id(p)`.
- **Gradient checkpointing** — `use_checkpoint=True` on the
  `Transformer`; `_run_layers` wraps each layer in
  `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` when
  `self.training`.

## μP LR scaling

```
new_lr = mup_lr_reference * (mup_lr_reference_params / total) ** 0.5
```

- Reference: `mup_lr_reference=6.0e-4` @ `mup_lr_reference_params=757,226,496`
  (the ~757M reference model).
- `total` is counted **after** MTP-wrap (when MTP is enabled, `total`
  includes the MTP heads, which inflates the reference slightly — see
  `CONTEXT.md` open question 3).
- For the 422M config this yields **~8.07e-4**.
- `test_mup_lr_scaling` verifies the formula against `count_parameters`.

## Scheduler

`make_warmup_cosine_lambda(warmup_steps, total_steps, min_lr_ratio)`:
linear warmup → cosine decay to `min_lr_ratio` (default 0.1). Bound by
`min_lr_ratio` beyond `total_steps`. Tests verify monotonicity and key
points (step 0, mid-warmup, end-of-warmup, mid-cosine, end-of-cosine,
beyond total).

## NaN guard

`nan_guard: bool` + `nan_guard_max_consecutive: int = 5`.

- Each `train_step` checks `torch.isnan(loss).any() or torch.isinf(loss).any()`.
- On NaN/Inf: skip backward, `zero_grad(set_to_none=True)`, return `None`,
  increment `nan_guard_streak`.
- After 5 consecutive NaN/Inf: restore the latest checkpoint via
  `load_checkpoint(latest)` (resumes scheduler + optimizer + step
  counter), reset the streak. If no checkpoint exists, raise
  `RuntimeError`.
- Non-NaN steps reset the streak to 0.
- **Never disable without explicit user consent** (`AGENTS.md` hard rule 5).

## MoE bias updates

`_update_moe_bias()` iterates `raw_model.moe_layers()` and calls
`moe.update_gate_bias(speed=config.bias_update_speed)` every
`bias_update_every` optimizer steps (config: 1; dataclass default 10 —
YAML wins). See [moe.md](moe.md) for the bias mechanism.

## PretrainDataset

Packed-token dataset, two layouts:

- **single-file** — `torch.load(data_path, weights_only=True)`, windows
  of `max_seq_len+1` sliced as `(x=chunk[:-1], y=chunk[1:])`.
- **sharded** — `shard_*.bin` files; `_locate` is a binary search over
  cumulative `shard_offsets`; `_load_shard` keeps an LRU cache of size 2
  (small — may thrash on limited RAM); cross-shard windows stitched via
  `.tolist()` (slow at scale — see `CONTEXT.md` §8).

Final partial chunk is dropped (not padded) — `test_final_sample_truncated`.

## MTP training path

When `mtp_depth > 0 and mtp_weight > 0`:

- `MultiTokenPrediction(config.model_config, raw_model)` wraps the
  Transformer; `training_model = mtp_model`.
- `train_step`: `main_logits, mtp_pairs = self.model(tokens, start_pos=0)`
  then `compute_loss`. Only `total_loss` is divided by
  `gradient_accumulation_steps`.
- Checkpoints save MTP weights under the `mtp.` prefix
  (`mtp_modules.*` keys) in the same safetensors file; optimizer state
  is **intentionally skipped** for MTP (see `CONTEXT.md`).

## Checkpointing

`CheckpointManager(config.checkpoint_dir)` — atomic safetensors +
`optim_step_N.pt` + `meta_step_N.json`. See [utils.md](utils.md).

`save_checkpoint(step)` writes model + MTP-prefixed keys + meta
(scheduler state, opt_steps, tag, full config, `has_mtp` flag).
`load_checkpoint(step)` restores model, optional MTP, scheduler,
opt_steps, and returns the resumed step.