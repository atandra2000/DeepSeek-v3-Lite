# DeepSeek-v3-Lite — R7 Training API Reference

> The single-GPU BF16 pre-training entry point: `TrainingConfig` (the runtime view of the YAML), `PretrainDataset` (packed-token windows over single-file or sharded mmap'd data), `Pretrainer` (model/optimizer/scheduler wiring, micro-step loop, NaN guard, checkpoints), and `main` (CLI).

**Source:** `training/pretrain.py` — the only training driver in the repo. No `torch.distributed`, no pipeline stages; one process, one device (`torch.device("cuda" if torch.cuda.is_available() else "cpu")`).

**Locked numbers (see [the docs README](../README.md)):** deduped params **411.6M** base ($411\,632\,256$) / **418.7M** with MTP ($418\,713\,984$); μP LR $6.0\times 10^{-4}\times\sqrt{757\,226\,496/N}$ → **8.14e-4** base / **8.07e-4** with MTP; "422M" is only the config *filename* (`configs/pretrain_a100_422m.yaml`). **No GPU training run has ever executed** — all wall-clock/MFU figures elsewhere are estimates.

**Siblings:** [R1 config schema](R1_config_schema.md) (every YAML key), [R2 transformer API](R2_transformer_api.md), [R4 MoE API](R4_moe_api.md), [R5 MTP API](R5_mtp_api.md), [R8 utils API](R8_utils_api.md) (CheckpointManager, TrainingLogger). Narrative walkthroughs: [08 Training Pipeline](../training.md), [09 Data Pipeline](../concepts/data-pipeline.md), [11 Operations & Testing](../concepts/kernels-and-ops.md).

---

## Symbol index (all public symbols of the module)

| Symbol | Kind | One-line purpose |
|---|---|---|
| `training/pretrain.py:make_warmup_cosine_lambda` | function | Factory for the `LambdaLR` step function: linear warmup + cosine decay to `min_lr_ratio`. |
| `training/pretrain.py:TrainingConfig` | `@dataclass` | 29-field runtime training configuration; the dataclass defaults are a smoke profile, YAML overrides them. |
| `training/pretrain.py:PretrainDataset` | `Dataset` | Packed-token next-token-pair dataset over a flat uint32 tensor (single file or `shard_*.bin` directory). |
| `training/pretrain.py:Pretrainer` | class | The whole training system: construction wires model/optimizer/scheduler/checkpoints; `train` runs the loop. |
| `training/pretrain.py:main` | function | CLI: parse YAML → build `TrainingConfig` (CLI > YAML > default) → construct → optional resume → `train()`. |

---

## 1. `make_warmup_cosine_lambda`

```python
def make_warmup_cosine_lambda(warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1):
    def lr_lambda(step: int) -> float: ...
    return lr_lambda
```

**Contract.** Returns a pure function `lr_lambda(step) -> float`:

- $0 \le \text{step} < \text{warmup\_steps}$: $\text{step} / \max(1, \text{warmup\_steps})$ — linear ramp from 0 to 1.
- $\text{step} \ge \text{total\_steps}$: returns exactly `min_lr_ratio` (floor, never zero).
- otherwise: $\text{min\_lr\_ratio} + (1 - \text{min\_lr\_ratio})\cdot \frac{1}{2}\big(1 + \cos(\pi \cdot \frac{\text{step}-\text{warmup\_steps}}{\max(1,\text{total\_steps}-\text{warmup\_steps})})\big)$ — cosine decay from 1 to `min_lr_ratio`.

**Step space.** `step` counts **optimizer steps**, not micro-steps: `Pretrainer` calls `self.scheduler.step()` only inside the `is_opt_step` branch of `training/pretrain.py:Pretrainer.train_step`. Both `warmup_steps` and `total_steps` are therefore interpreted in optimizer-step space (see §5).

**Callers.** Constructed once in `training/pretrain.py:Pretrainer.__init__` with `warmup_steps=config.warmup_steps`, `total_steps=opt_steps` (the horizon fix, §5), `min_lr_ratio=config.min_lr_ratio`; wrapped in `torch.optim.lr_scheduler.LambdaLR`. Pinned by `tests/test_training.py` (`TestWarmupCosineScheduler`, `TestSchedulerBoundary`) — value checks at key points, monotonic warmup, monotonic decay, and the `total_steps` boundary returning exactly `min_lr_ratio`.

---

## 2. `TrainingConfig`

Plain `@dataclass` (`field`/`asdict` imported), 29 fields. It is the *runtime* view: `main()` copies YAML values into it, `Pretrainer.__init__` reads it, and nothing in the training loop touches YAML again — with one deliberate exception, `model_config`, which carries the **entire raw YAML dict** so `Transformer` can unwrap its own `model:` section (`models/transformer.py:Transformer`).

`dataclasses.asdict(config)` serializes it into checkpoint `extra_meta` (see `training/pretrain.py:Pretrainer.save_checkpoint`); `TestTrainingConfig.test_serializable` guards this.

| Field | Type | Dataclass default | YAML source in `main()` | Notes |
|---|---|---|---|---|
| `TrainingConfig.model_config` | `dict` | `{}` | whole `yaml_cfg` (model+training+data sections) | passed verbatim to `Transformer` and `MultiTokenPrediction` |
| `TrainingConfig.data_path` | `str` | `"data/pretrain_data.bin"` | `--data-path` or `data.train_data_path` | canonical: `"data/pretrain_chinchilla"` (sharded dir) |
| `TrainingConfig.checkpoint_dir` | `str` | `"checkpoints/pretrain"` | `--checkpoint-dir` or `training.save_dir` | canonical: `"checkpoints/pretrain_a100"` |
| `TrainingConfig.vocab_size` | `int` | `100018` | `model.vocab_size` | |
| `TrainingConfig.max_seq_len` | `int` | `4096` | `model.max_seq_len` | canonical: 2048 |
| `TrainingConfig.batch_size` | `int` | `8` | `training.micro_batch_size` | per-micro-step batch; **not** the effective opt-step batch |
| `TrainingConfig.gradient_accumulation_steps` | `int` | `4` | `training.gradient_accumulation_steps` | micro-steps per optimizer step; effective opt batch = `batch_size × gradient_accumulation_steps × max_seq_len` |
| `TrainingConfig.max_steps` | `int` | `20000` | `training.total_steps` | **micro-step** loop budget (canonical: 512,000) |
| `TrainingConfig.warmup_steps` | `int` | `2000` | `training.warmup_steps` | in **optimizer-step** space |
| `TrainingConfig.lr` | `float` | `2.2e-4` | `training.lr` | overwritten in place by μP scaling when `mup_lr` is true |
| `TrainingConfig.min_lr_ratio` | `float` | `0.1` | `training.min_lr_ratio` | cosine floor; canonical 0.05 |
| `TrainingConfig.weight_decay` | `float` | `0.1` | `training.weight_decay` | applied only to `dim() >= 2` params |
| `TrainingConfig.beta1` | `float` | `0.9` | — (dataclass default only) | `main()` does **not** map `training.beta1`; the key was removed from the YAMLs in the cleanup — the dataclass default applies |
| `TrainingConfig.beta2` | `float` | `0.95` | — (dataclass default only) | same caveat as `beta1` |
| `TrainingConfig.max_grad_norm` | `float` | `1.0` | `training.grad_clip` | `clip_grad_norm_` once per optimizer step |
| `TrainingConfig.mtp_weight` | `float` | `0.0` | `training.mtp_loss_weight` | MTP active iff `> 0.0` **and** `model.mtp_depth > 0`; canonical 0.3 |
| `TrainingConfig.bias_update_speed` | `float` | `0.001` | `training.bias_update_speed` | MoE gate-bias nudging step |
| `TrainingConfig.bias_update_every` | `int` | `10` | `training.bias_update_every` | cadence in **optimizer** steps; canonical 1 |
| `TrainingConfig.grad_checkpoint` | `bool` | `True` | `training.grad_checkpoint` **and not** `--no-checkpoint` | `use_checkpoint` for `Transformer` |
| `TrainingConfig.compile_model` | `bool` | `True` | `training.compile` **and not** `--no-compile` | `torch.compile(..., mode=TORCH_COMPILE_MODE, fullgraph=False)` |
| `TrainingConfig.save_every` | `int` | `1000` | `training.save_interval` | micro-step cadence; canonical 4000 |
| `TrainingConfig.log_every` | `int` | `100` | `training.log_interval` | micro-step cadence; canonical 50 |
| `TrainingConfig.nan_guard` | `bool` | `False` | `training.nan_guard` | NaN/Inf skip + rollback state machine |
| `TrainingConfig.nan_guard_max_consecutive` | `int` | `5` | `training.nan_guard_max_consecutive` | consecutive non-finite micro-steps before rollback/abort |
| `TrainingConfig.mup_lr` | `bool` | `False` | `training.mup_lr` | canonical `true` |
| `TrainingConfig.mup_lr_reference` | `float` | `6.0e-4` | `training.mup_lr_reference` | reference LR @ reference param count |
| `TrainingConfig.mup_lr_reference_params` | `int` | `757226496` | `training.mup_lr_reference_params` | μP reference anchor (external/unverifiable, kept as-is) |
| `TrainingConfig.log_per_component_params` | `bool` | `True` | `training.log_per_component_params` | per-component param breakdown at init |
| `TrainingConfig.seed` | `int` | `42` | `training.seed` | seeds `torch.manual_seed` **and** the DataLoader shuffle generator |

**Callers.** Constructed by `training/pretrain.py:main`; consumed by `training/pretrain.py:Pretrainer.__init__`; serialized via `asdict` in `training/pretrain.py:Pretrainer.save_checkpoint`; exercised directly by `tests/test_training.py::TestTrainingConfig` (defaults, from-YAML-dict construction, `asdict` serializability) and `scripts/e2e_test_gpu.py`.

---

## 3. `PretrainDataset`

`torch.utils.data.Dataset` over a flat token stream — never tokenizes, never pads, never samples within a window. Sample $i$ is the contiguous window starting at token $i \times S$ ($S = \text{max\_seq\_len}$). Storage dtype is `uint32` (4 B/token); the cast to `int64` happens in `train_step`, not here.

### 3.1 `PretrainDataset.__init__(data_path: str, max_seq_len: int, vocab_size: int)`

- Raises `FileNotFoundError` if `data_path` does not exist (message directs to `python data/prepare_data.py`).
- `layout = "sharded"` if `data_path` is a directory, else `"single"`.
  - **Sharded:** globs `shard_*.bin` (sorted; empty glob → `FileNotFoundError`), `torch.load(p, weights_only=True, map_location="cpu", mmap=True)` per shard, computes `shard_sizes` (numel each), `shard_offsets` (cumulative start index of each shard), `_total_tokens = sum(shard_sizes)`.
  - **Single:** `torch.load(data_path, weights_only=True, map_location="cpu", mmap=True)`, `_total_tokens = data.numel()`.
- `_n_samples = (total_tokens - 1) // max_seq_len` — each sample needs `max_seq_len + 1` tokens (one for the target shift); the final `total - n_samples × S` tail tokens are unreachable.
- `vocab_size` is stored but **not used** (no masking in the packed corpus); it exists for interface symmetry.

### 3.2 `PretrainDataset._locate(global_idx: int) -> Tuple[int, int]`

Maps a global token index to `(shard_idx, offset_within_shard)` in $O(\log K)$ for $K$ shards:

```python
lo = bisect.bisect_right(self.shard_offsets, global_idx) - 1
return lo, global_idx - self.shard_offsets[lo]
```

**Contract.** `global_idx < 0` or `global_idx >= _total_tokens` raises `IndexError` (message: `"global_idx N out of range [0, M)"`); negative indices are rejected explicitly, and `_total_tokens` is rejected because the last valid token position is `_total_tokens - 1`. Only meaningful on the sharded layout (the single-file branch of `__getitem__` never calls it); `TestPretrainDataset` pins the out-of-range behavior.

### 3.3 `PretrainDataset.__len__() -> int`

Returns `_n_samples`.

### 3.4 `PretrainDataset.__getitem__(idx: int) -> Tuple[torch.Tensor, torch.Tensor]`

Reads the window `[idx*S, idx*S + S]` (length `S + 1`):

- **Single layout:** `self.data[start : start + needed].clone()`.
- **Sharded layout:** walks `_locate(cursor)` from `start`, appending `shard[offset : offset + take]` per shard where `take = min(needed - cursor_pos, shard_sizes[shard_idx] - offset_in_shard)`, then `torch.cat(pieces)` (a single-piece window is cloned, not concatenated) — a window that straddles a shard boundary is stitched on the fly.
- Returns `chunk[:-1], chunk[1:]` — inputs `tokens[t : t+S]`, targets `tokens[t+1 : t+S+1]`.

**Callers.** `training/pretrain.py:Pretrainer.train` via `DataLoader(dataset, batch_size=config.batch_size, shuffle=True, generator=g, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=8, drop_last=True)`. Tests: `tests/test_training.py::TestPretrainDataset` (single-file and sharded fixtures, `_locate` errors); `scripts/e2e_test_gpu.py` imports it directly.

---

## 4. `Pretrainer`

### 4.1 `Pretrainer.__init__(config: TrainingConfig)`

Construction pipeline, top to bottom:

1. **Seed** — `torch.manual_seed(config.seed)` before model construction (reproducible init + data order).
2. **Device & CUDA flags** — `cuda` if available else `cpu` (CPU prints a smoke-test warning). On CUDA: `allow_tf32 = True` for both matmul and cudnn, `set_float32_matmul_precision("high")`, `cudnn.benchmark = True`.
3. **Logging** — `init_logging(config.log_every, seq_len=config.max_seq_len, batch_size=config.batch_size)` (`utils/logging.py:init_logging`); `self.logger = get_logger()` (`utils/logging.py:get_logger`).
4. **Model** — `Transformer(config.model_config, use_checkpoint=config.grad_checkpoint).to(self.device)` (`models/transformer.py:Transformer`); param count via `count_parameters` (`models/transformer.py:count_parameters`), logged as total/trainable. If `config.log_per_component_params`, calls `_log_per_component_params`.
5. **MTP wrap** — `mtp_depth = config.model_config.get("model", config.model_config).get("mtp_depth", 0)`; if `mtp_depth > 0 and config.mtp_weight > 0.0`, wraps in `MultiTokenPrediction(config.model_config, raw_model)` (`models/mtp.py:MultiTokenPrediction`), logs its params, and `self.mtp_wrapper` is set. The training model is the wrapper when present, else the raw `Transformer`.
6. **torch.compile** — if `config.compile_model and hasattr(torch, "compile")`: `mode = os.environ.get("TORCH_COMPILE_MODE", "max-autotune")`, `torch.compile(training_model, mode=mode, fullgraph=False)`. `self.raw_model` stays the **uncompiled** base `Transformer` (pinned by `TestPretrainerConstruction.test_raw_model_is_uncompiled`).
7. **μP LR scaling** — if `config.mup_lr`: `new_lr = config.mup_lr_reference * (config.mup_lr_reference_params / total) ** 0.5` where `total` is the post-MTP param count; `config.lr = new_lr` in place. With the locked constants: 8.14e-4 base / 8.07e-4 with MTP. (`TestMuPLRBoundary` pins the formula and the no-scaling-when-false path.)
8. **Optimizer** — dedups params by tensor id (`id(p)`), splits into `decay_params` (`dim() >= 2`, weight `config.weight_decay`) and `no_decay_params` (`dim() < 2`, weight 0.0); `AdamW([...], lr=config.lr, betas=(config.beta1, config.beta2), fused=torch.cuda.is_available())`. Tied `head.weight`/`embed.weight` share storage, so dedup prevents double-counting (`TestPretrainerConstruction.test_optimizer_deduplicates`).
9. **Scheduler** — **horizon fix**: `opt_steps = max(1, config.max_steps // config.gradient_accumulation_steps)`; `make_warmup_cosine_lambda(warmup_steps=config.warmup_steps, total_steps=opt_steps, min_lr_ratio=config.min_lr_ratio)` wrapped in `LambdaLR`. The loop budget `max_steps` is in micro-steps; the cosine horizon must be the optimizer-step count or the canonical run would traverse only a fraction of the cosine arc and never reach `min_lr_ratio` (pinned by `TestPretrainerConstruction.test_scheduler_horizon_is_optimizer_steps`).
10. **AMP dtype** — `self.amp_dtype = torch.bfloat16` (BF16 autocast; FP32 master weights live in AdamW).
11. **Checkpoints** — `self.ckpt_manager = CheckpointManager(config.checkpoint_dir)` (`utils/checkpoint.py:CheckpointManager`); `self._opt_steps = 0`.

### 4.2 `Pretrainer._log(msg: str)` — `staticmethod`

`print(msg)` — the trainer's only output channel besides `TrainingLogger`; no logging config needed.

### 4.3 `Pretrainer._amp_context()`

```python
return autocast(self.device.type, dtype=self.amp_dtype)
```

The single BF16 context used for every forward (and its backward). CPU runs use the same bf16 autocast path (`device.type == "cpu"`).

### 4.4 `Pretrainer._update_moe_bias()`

```python
for moe in self.raw_model.moe_layers():
    moe.update_gate_bias(speed=self.config.bias_update_speed)
```

Iterates `models/transformer.py:Transformer.moe_layers` (generator of the `DeepSeekMoE` FFNs, layers 2–17 canonical) and nudges each gate bias via `models/moe.py:DeepSeekMoE.update_gate_bias`. Operates on `raw_model`, never the compiled/MTP wrapper. Scheduling lives in `train_step` (every `bias_update_every` optimizer steps).

### 4.5 `Pretrainer._moe_balance_metric() -> torch.Tensor`

```python
losses = [moe.get_load_balance_loss() for moe in self.raw_model.moe_layers()]
if not losses:
    return torch.tensor(0.0, device=self.device)
return torch.stack(losses).sum()
```

Sum over MoE layers of `models/moe.py:DeepSeekMoE.get_load_balance_loss`. Returns an **on-device tensor** — `.item()` is deferred to the logger path to avoid a per-micro-step GPU sync. Computed inside the autocast region of `train_step`. Guarded by `tests/test_training.py::TestMoEBalanceMetric`.

### 4.6 `Pretrainer._log_per_component_params(model)`

Walks `model.named_parameters()`, skips any tensor `id` already seen (counts the tied head once), buckets by name substrings: `embed` → `embedding`; `head` → `lm_head`; `.attn.` with `wq/wkv_a/wkv_b/wo/q_norm/kv_norm` → `mla_attn`; `attn_norm`/`ffn_norm`/`.norm.weight` suffix → `rmsnorm`; `.experts.` with `w1/w2/w3` → `moe_routed_experts`; `shared_experts` → `moe_shared_experts`; `.ffn.w` → `dense_swiglu`; `.gate.` → `moe_gate`; else `other`. Logs each bucket with count and percent, then TOTAL in millions. Known cosmetic quirk (.concepts/foundations.md)): the final `norm.weight` (11 chars) cannot end with `".norm.weight"` (12), so it lands in `other`.

### 4.7 `Pretrainer.train_step(tokens, targets, micro_step) -> Optional[Dict[str, Optional[float]]]`

The entire training math, one micro-step:

1. **Opt-step test** — `is_opt_step = (micro_step + 1) % self.config.gradient_accumulation_steps == 0`.
2. **dtype cast** — `uint32 → torch.long` for `tokens`/`targets` when needed (nn.Embedding and F.cross_entropy require Long indices; done at the boundary so the dataset stays compact).
3. **Forward (inside `_amp_context()`)**:
   - **MTP path** (`self.mtp_wrapper is not None`): `main_logits, mtp_pairs = self.model(tokens)`; `total_loss, main_loss, mtp_loss = self.mtp_wrapper.compute_loss(main_logits, targets, mtp_pairs)` (`models/mtp.py:MultiTokenPrediction.compute_loss`); `loss = total_loss / gradient_accumulation_steps`.
   - **Main path**: `logits = self.model(tokens, start_pos=0, use_cache=False)`; `main_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)`; `loss = main_loss / gradient_accumulation_steps`.
   - `balance_loss = self._moe_balance_metric()`.
4. **NaN guard** — if `config.nan_guard` and the loss is NaN/Inf: log, `optimizer.zero_grad(set_to_none=True)`, return `None` (the loop treats `None` as "skip this micro-step").
5. **Backward** — `loss.backward()`.
6. **Optimizer step** (only `is_opt_step`): `nn.utils.clip_grad_norm_(self.model.parameters(), config.max_grad_norm)` → `optimizer.step()` → `scheduler.step()` → `optimizer.zero_grad(set_to_none=True)` → `self._opt_steps += 1` → bias update if `self._opt_steps % config.bias_update_every == 0`.
7. **Return** `{"loss": main_loss.detach(), "mtp_loss": mtp_loss.detach() or None, "balance_loss": balance_loss}` — detached tensors, **not** floats (the annotation is stale); `.item()` happens once per log step in `train`.

Guarded by `tests/test_training.py::TestTrainStep` (forward/backward, gradient coverage) and `TestNanGuardRollback`.

### 4.8 `Pretrainer.save_checkpoint(step: int, tag: str = "")`

- Serializes `self.raw_model` (never the compiled wrapper).
- **Weight-tying dedup:** if `raw_model.weight_tying`, drops `head.weight` from the state dict — it *is* `embed.weight` (shared storage); keeping it would double the embedding bytes (~154 MB at canonical vocab/dim, BF16) and risk drift. `load_state_dict(strict=False)` tolerates the missing key.
- **MTP:** appends `mtp.{k}: v` for keys starting with `mtp_modules.` from the (unwrapped, `_orig_mod`-aware) MTP state — shared embed/head are already in the trunk state.
- **extra_meta:** `{"scheduler": scheduler.state_dict(), "opt_steps": _opt_steps, "tag": tag or f"step_{step}", "config": asdict(config), "has_mtp": mtp_wrapper is not None}`.
- Delegates to `utils/checkpoint.py:CheckpointManager.save` (atomic `.tmp` → rename; see [R8](R8_utils_api.md)). Logs `Checkpoint saved at step N`.

Pinned by `TestCheckpointRoundtrip` (weights + MTP keys round-trip) and `TestCheckpointOptStepsRoundtrip` (`_opt_steps` survives).

### 4.9 `Pretrainer.load_checkpoint(step: int) -> int`

- Trunk + optimizer restored by `utils/checkpoint.py:CheckpointManager.load(self.raw_model, step, device=str(self.device), optimizer=self.optimizer, strict=False)` — `strict=False` is required (no `head.weight`); missing optimizer `.pt` only warns (fresh optimizer state).
- If `has_mtp` in meta and the MTP wrapper exists: re-reads the same safetensors, strips the `mtp.` prefix, `load_state_dict(mtp_state, strict=False)` on the unwrapped MTP module.
- Restores `scheduler` and `_opt_steps` from meta.
- Returns `meta.get("step", step)` — the actual resumed micro-step — and logs it.

### 4.10 `Pretrainer._find_latest_checkpoint() -> Optional[int]`

`self.ckpt_manager.latest_step()` (`utils/checkpoint.py:CheckpointManager.latest_step`) — the highest *complete* step (all three of `model_step_N.safetensors`, `optim_step_N.pt`, `meta_step_N.json` present).

### 4.11 `Pretrainer.train()`

The loop:

1. Builds `PretrainDataset(config.data_path, config.max_seq_len, config.vocab_size)` and a `DataLoader` (see §3.4 for flags; `shuffle=True` with a `torch.Generator` seeded by `config.seed`). **Resume does not restore the sampler RNG** — token order differs across a restart, benign at corpus scale (documented in the code).
2. **Auto-resume:** `latest = self._find_latest_checkpoint()`; on hit, `global_step = self.load_checkpoint(latest)` (failure logs a warning and continues from 0).
3. Loop `while global_step < config.max_steps:` over `tqdm(loader)` — a fresh pass over the loader re-shuffles (generator state advances); `global_step` counts **micro-steps** and is passed straight to `train_step` as `micro_step`.
   - `tokens/targets → .to(self.device, non_blocking=True)`; `metrics = self.train_step(tokens, targets, global_step)`.
   - `None` (NaN guard fired): streak++, rollback to latest checkpoint or `RuntimeError("NaN/Inf with no checkpoint to restore from")` at `nan_guard_max_consecutive`; streak reset on any finite step.
   - Log every `log_every` micro-steps: `lr = scheduler.get_last_lr()[0]`, one `.item()` per tensor (balance_loss, optional mtp_loss, main loss), via `self.logger.log(global_step, ...)`.
   - Save every `save_every` micro-steps (`global_step > 0`).
4. Final `self.save_checkpoint(global_step, tag="final")` and `Training complete.`

---

## 5. Step-counter semantics (micro-step vs opt-step)

Two counters exist, and they advance at different rates:

| Counter | Increments | Bounds | Checkpointed? |
|---|---|---|---|
| `global_step` (loop-local) | once per micro-batch in `train` | `[0, max_steps)` | re-derived from meta on resume; not stored |
| `Pretrainer._opt_steps` | once per `is_opt_step` in `train_step` | `[0, max_steps // gradient_accumulation_steps]` | **yes** (in `extra_meta`, restored by `load_checkpoint`) |

`is_opt_step = (micro_step + 1) % gradient_accumulation_steps == 0` — the optimizer steps after every `gradient_accumulation_steps`-th micro-step, and only then do clipping, `optimizer.step()`, `scheduler.step()`, `zero_grad()`, and the MoE bias-update cadence (`_opt_steps % bias_update_every == 0`) run.

**Scheduler horizon (recent fix).** `LambdaLR` advances only on optimizer steps, so the cosine's `total_steps` must be the optimizer-step count: `opt_steps = max(1, max_steps // gradient_accumulation_steps)`. With the canonical config (512,000 micro-steps ÷ 4 = 128,000 opt-steps), the schedule now completes its full arc and ends at `min_lr_ratio`. Using the micro-step budget as the horizon would leave the LR at ~peak and never reach the floor. `warmup_steps` lives in the same (optimizer-step) space. Pinned by `test_scheduler_horizon_is_optimizer_steps`, which asserts the run ends exactly at `lr × min_lr_ratio` and strictly below the old micro-step-horizon value.

---

## 6. `main() -> None` — CLI entry point

```bash
python3 training/pretrain.py [--config CONFIG] [--data-path PATH] [--checkpoint-dir DIR]
                            [--resume STEP] [--no-checkpoint] [--no-compile]
```

| Flag | Default | Effect |
|---|---|---|
| `--config` | `configs/pretrain_a100_422m.yaml` | YAML with `model:` / `training:` / `data:` sections |
| `--data-path` | YAML `data.train_data_path` | overrides `TrainingConfig.data_path` |
| `--checkpoint-dir` | YAML `training.save_dir` | overrides `TrainingConfig.checkpoint_dir` |
| `--resume STEP` | none | explicit `trainer.load_checkpoint(int(args.resume))` *before* `train()` (which then auto-resumes the same step) — for when you don't want the latest |
| `--no-checkpoint` | off | forces `grad_checkpoint=False` (`and not` with YAML) |
| `--no-compile` | off | forces `compile_model=False` (`and not` with YAML) |

`yaml.safe_load` → `t = yaml_cfg.get("training", {})`, `d = yaml_cfg.get("data", {})` → `TrainingConfig` with precedence **CLI > YAML > dataclass default** (pattern: `args.data_path or d.get("train_data_path", "...")`; booleans combine via `yaml_value and not cli_flag`). **`beta1`/`beta2` are not read from YAML** — the `training.beta1`/`training.beta2` keys in both shipped configs are currently inert; the dataclass defaults (0.9/0.95) apply. Then: `Pretrainer(config)` → optional `load_checkpoint(args.resume)` → `trainer.train()`.

Guarded by `tests/test_training.py::TestConfigFromYAML` (parses a fixture YAML through the same field mapping) and `TestMuPLRBoundary`.

---

## References
**CLI / scripts**
- `python3 training/pretrain.py --config configs/pretrain_a100_422m.yaml` — documented in [00 Getting Started](../guides/getting-started.md), and `scripts/launch_a100.sh` (nohup launch with `--data-path`/`--checkpoint-dir`/`--resume`).
- `scripts/e2e_test_gpu.py` — imports `PretrainDataset, Pretrainer, TrainingConfig`, constructs a real `Pretrainer` on GPU (with `fused=False` AdamW patch), reports peak VRAM after init.

**Tests** (`tests/test_training.py` is the primary consumer; `tests/test_utils.py::TestCheckpointManagerMTP` replicates the `save_checkpoint` combined-state layout)
- `TestTrainingConfig` — defaults, YAML-dict construction, `asdict` serializability.
- `TestWarmupCosineScheduler` / `TestSchedulerBoundary` — scheduler values, monotonicity, boundary floor.
- `TestPretrainDataset` — single/sharded layouts, `_locate` bounds.
- `TestPretrainerConstruction` — CPU construction, uncompiled `raw_model`, MTP param inclusion, optimizer dedup, μP LR scaling, scheduler horizon fix.
- `TestCheckpointRoundtrip` / `TestCheckpointOptStepsRoundtrip` — weights, MTP keys, `_opt_steps` round-trip.
- `TestTrainStep` / `TestMoEBalanceMetric` / `TestNanGuardRollback` — forward/backward, balance metric, NaN state machine.

---

