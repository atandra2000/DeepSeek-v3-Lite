# R8 — utils API Reference

> **60-second summary.** The `utils/` package is the training infrastructure that is not the model: `utils/checkpoint.py` owns the atomic three-file checkpoint triplet (`CheckpointManager`), `utils/logging.py` owns the rolling-window console/WandB logger (`TrainingLogger` plus the `init_logging`/`get_logger` process-global pair), and `utils/memory.py` owns the CPU-arithmetic VRAM budget estimator (`estimate_model_memory_gb`, `assert_fits_in_available_gpu`). All three are pure-PyTorch, CPU-runnable, and exercised directly by `tests/test_utils.py`; the training loop reaches them through `training/pretrain.py:Pretrainer`, and inference through `inference/generate.py:main`.
>
> **Scope.** Every public symbol of the three modules, with exact signatures, defaults, shape/atomicity contracts, and callers. Reference style: one-line purpose per entry, no tutorial prose (see [[Docs/08_Training_Pipeline|T8]] for the loop walkthrough, [[Docs/11_Operations_and_Testing|T11]] for the ops view, and `../guides/G5_checkpoint_ops.md` for the recovery playbook).
>
> **Anchor convention.** Symbols are cited as `utils/<file>.py:Symbol` (path prefix + `Class.method` or module-level name). No line anchors. JIT kernels are never cited — the utils layer has none.

---

## 1. CheckpointManager — `utils/checkpoint.py`

One class, one job: save and restore training state **atomically**, with **shared-tensor dedup** (weight tying) and **step discovery** that only ever returns *complete* checkpoints. `utils/checkpoint.py:CheckpointManager` is the only public symbol of the module. Files per step `N`:

| File | Contents | Writer |
|---|---|---|
| `model_step_N.safetensors` | Model weights (deduped, contiguous) | `utils/checkpoint.py:CheckpointManager._atomic_save_safetensors` |
| `optim_step_N.pt` | Optimizer state (`torch.save`) | `utils/checkpoint.py:CheckpointManager._atomic_save_torch` |
| `meta_step_N.json` | Resume contract (step, scheduler, opt_steps, config, …) | `utils/checkpoint.py:CheckpointManager._atomic_save_json` |

### 1.1 Constructor

```python
def __init__(self, save_dir: str):
```

- **Behavior.** `self.save_dir = Path(save_dir)`; `mkdir(parents=True, exist_ok=True)` — the directory is created on construction, so a fresh run needs no pre-flight.
- **Callers.** `training/pretrain.py:Pretrainer.__init__` (`self.ckpt_manager = CheckpointManager(config.checkpoint_dir)`); `inference/generate.py:main` (directory form of `--checkpoint`); `scripts/e2e_test_gpu.py`; `tests/test_utils.py` (fixture `tmp_ckpt_dir`).

### 1.2 `save`

```python
def save(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int,
         extra_meta: Optional[dict] = None, state_dict: Optional[dict] = None) -> None:
```

| Param | Default | Contract |
|---|---|---|
| `model` | — | Only used for `state_dict()` when `state_dict` is `None`; never moved or mutated. |
| `optimizer` | — | `optimizer.state_dict()` is written to `optim_step_N.pt` unconditionally. |
| `step` | — | The step number; **always** wins in the meta file. |
| `extra_meta` | `None` | Arbitrary JSON-serializable dict merged into `meta_step_N.json`; a caller-supplied `"step"` key is **dropped** (`meta = {"step": step, **{k: v for k, v in (extra_meta or {}).items() if k != "step"}}`). |
| `state_dict` | `None` | Pre-built state to write instead of `model.state_dict()` — the hook `Pretrainer` uses to inject `mtp.*` keys and drop `head.weight`. |

- **Write order.** (1) safetensors, (2) optimizer `.pt`, (3) meta `.json` — each individually atomic, the triplet **not** transactional (see §1.6).
- **Logs.** `[checkpoint] saved step %d → %s`.
- **Callers.** `training/pretrain.py:Pretrainer.save_checkpoint` is the only production caller: it pre-filters `head.weight` when `weight_tying` is on, merges `mtp.<key>` MTP weights, and passes `extra_meta={"scheduler": …, "opt_steps": …, "tag": …, "config": …, "has_mtp": …}`. Tests call it directly.

### 1.3 `load`

```python
def load(self, model: torch.nn.Module, step: int, device: str = "cuda",
         optimizer: Optional[torch.optim.Optimizer] = None, strict: bool = True) -> dict
```

| Param | Default | Contract |
|---|---|---|
| `model` | — | Must already be constructed (tying re-established); weights copied in via `load_state_dict(weights, strict=False)` — **always** non-strict internally. |
| `step` | — | Exact step; missing `model_step_N.safetensors` → `FileNotFoundError` listing `_list_steps()`. |
| `device` | `"cuda"` | Placement for `load_file(..., device=device)` and `torch.load(..., map_location=device)`. CPU callers must pass `device="cpu"` (see §5). |
| `optimizer` | `None` | If given, `optim_step_N.pt` is loaded with `weights_only=True`; a missing `.pt` only **warns** ("optimizer will start from scratch") — never fails. |
| `strict` | `True` | If `True`, any missing or unexpected key raises `RuntimeError` (first five keys in the message); if `False`, the same audit is a `logger.warning`. |

- **Missing/unexpected audit.** `missing, unexpected = model.load_state_dict(weights, strict=False)`; both lists are checked. With weight tying, `head.weight` is *expected* to be missing — the shared storage is restored through `embed.weight` (see §1.7).
- **Return.** The parsed `meta_step_N.json` dict (the resume contract: `scheduler`, `opt_steps`, `config`, `has_mtp`, `tag`, `step`); if the meta file is absent, `{"step": step}`.
- **Callers.** `training/pretrain.py:Pretrainer.load_checkpoint` (`device=str(self.device)`, `optimizer=self.optimizer`, `strict=False`); `inference/generate.py:main` (`strict=False`); `scripts/e2e_test_gpu.py`; `tests/test_utils.py`.

### 1.4 `latest_step`

```python
def latest_step(self) -> Optional[int]:
```

- **Behavior.** `steps = self._list_steps()`; returns the first step in **descending** order for which `_checkpoint_complete(s)` is true; `None` if no complete checkpoint exists. A torn step (crash between the three writes) is invisible to resume.
- **Callers.** `training/pretrain.py:Pretrainer._find_latest_checkpoint` (auto-resume at `train()` start and NaN-guard rollback); `inference/generate.py:main` (directory-form `--checkpoint`); `scripts/e2e_test_gpu.py`; `tests/test_utils.py`.

### 1.5 Atomic write machinery

```python
import contextlib
@contextlib.contextmanager
def _atomic_write(self, path: Path, suffix: str):
```

- **Guarantee.** `tempfile.mkstemp(dir=self.save_dir, suffix=suffix)` → close fd → `yield tmp` → on success `os.replace(tmp, path)`. `os.replace` is POSIX-atomic on the same filesystem (the temp file lives in the same directory), so a crash mid-write leaves either the old complete file or nothing — never a half-written file under the final name. On **any** exception the temp file is unlinked before re-raising.
- **Quirk.** `import contextlib` sits *inside* the class body, so `contextlib` is a class attribute; the decorator is applied at class-definition time. Cosmetic, but it means `CheckpointManager.contextlib` exists.
- **Callers.** Only the three `_atomic_save_*` helpers; the crash path is pinned by `tests/test_utils.py` (`test_atomic_save_crash_recovery` patches `save_file` to raise "disk full" and asserts no `.tmp`/partial file survives).

```python
def _atomic_save_safetensors(self, state: dict, path: Path) -> None:
```

- **Dedup.** Iterates `state.items()`; a key whose tensor `data_ptr()` was already seen is **skipped**; survivors are stored as `v.contiguous()` (safetensors requires contiguous storage). The duplicate key is dropped; `load_state_dict(strict=False)` restores it through the surviving shared storage.
- **Write.** `save_file(deduped, tmp)` inside `_atomic_write(path, ".safetensors.tmp")`.

```python
def _atomic_save_torch(self, obj, path: Path) -> None:
def _atomic_save_json(self, obj: dict, path: Path) -> None:
```

- `_atomic_save_torch`: `torch.save(obj, tmp)` inside `_atomic_write(path, ".pt.tmp")`.
- `_atomic_save_json`: `json.dump(obj, f, indent=2, default=str)` inside `_atomic_write(path, ".json.tmp")`. `default=str` means non-JSON values (e.g. a stray tensor) serialize as their string repr instead of failing — see §5.

### 1.6 Step discovery

```python
def _list_steps(self) -> list:
def _checkpoint_complete(self, step: int) -> bool:
```

- `utils/checkpoint.py:CheckpointManager._list_steps`: globs `model_step_*.safetensors`, parses `int(p.stem.split("_")[-1])`, silently skips unparseable names. Returns an **unsorted** list.
- `utils/checkpoint.py:CheckpointManager._checkpoint_complete(step)`: `all(...)` of the three file names existing for that step. This is the only definition of "resumable" in the repo.

### 1.7 Shared-tensor dedup semantics (the contract)

1. **Two layers of dedup.** `training/pretrain.py:Pretrainer.save_checkpoint` drops the `head.weight` key outright when `weight_tying` is on; `_atomic_save_safetensors` dedups **by `data_ptr()`** as a second line of defense that would also catch any other accidentally shared storage. The file therefore contains exactly one copy of the tied embedding.
2. **Restore.** `models/transformer.py:Transformer.__init__` re-establishes the tie (`self.head.weight = self.embed.weight` — one `Parameter` object) *before* `load` runs, so `load_state_dict(strict=False)` copies into `embed.weight`, which *is* the head's storage. `head.weight` shows up in the missing-keys audit and is expected: a warning under `strict=False`, an error under `strict=True`.
3. **MTP keys.** MTP weights travel in the same safetensors file under the `mtp.` prefix (`mtp.mtp_modules.*`); `training/pretrain.py:Pretrainer.load_checkpoint` strips the prefix and restores them into the MTP module with `strict=False`. See [[Docs/05_Multi_Token_Prediction|T5]] and `../reference/R5_mtp_api.md`.

---

## 2. TrainingLogger — `utils/logging.py`

Step-driven console logger with an optional WandB mirror (`utils/logging.py:TrainingLogger`). Prints a rolling-window summary every `log_interval` steps; the window is the losses accumulated since the last logged step.

### 2.1 Constructor

```python
def __init__(self, log_interval: int = 10, seq_len: int = 1024, batch_size: int = 1):
```

`utils/logging.py:TrainingLogger.__init__` stores the three cadence parameters and initializes the window/clock state:

| Param | Default | Contract |
|---|---|---|
| `log_interval` | `10` | Print cadence in optimizer steps; `log()` early-returns unless `step % log_interval == 0`. |
| `seq_len` | `1024` | Tokens per sequence — must match `model.max_seq_len` or the tps line is wrong. |
| `batch_size` | `1` | Micro-batch size — **included in the tps arithmetic** (see §2.2). |

- **WandB opt-in.** If `WANDB_PROJECT` is set in the environment, `wandb.init(project=…, name=os.environ.get("WANDB_RUN_NAME"), reinit=True)` runs at construction; `ImportError` prints `[logging] wandb not installed -- skipping WandB integration` and continues. The decision is frozen at construction time — changing the env var mid-run has no effect.
- **Callers.** `utils/logging.py:init_logging`, `utils/logging.py:get_logger` (lazy default), and tests.

### 2.2 `log`

```python
def log(self, step: int, loss: float, metrics: Optional[Dict[str, float]] = None, lr: float = 0.0) -> None:
```

| Param | Default | Contract |
|---|---|---|
| `step` | — | Optimizer step; the cadence test is `step % self.log_interval != 0` → early return (window keeps accumulating). |
| `loss` | — | Appended to `_loss_window`; averaged over the window on a logged step. |
| `metrics` | `None` | Extra floats printed as `k=v` (`.4f`) and forwarded to WandB as `train/{k}`. |
| `lr` | `0.0` | Printed as `lr={lr:.2e}`; forwarded as `train/lr`. |

- **Throughput line (batch_size included).** `tokens_per_sec = (log_interval * seq_len * batch_size) / elapsed`, where `elapsed = max(time.time() - self._step_start, 1e-6)`. The window is exactly `log_interval` micro-batches of `batch_size × seq_len` tokens each — the logger assumes every interval consumed exactly that many tokens.
- **PPL.** `ppl = torch.tensor(avg_loss).exp().item()` — nats → perplexity; see [[Docs/01_Foundations|T1]] for the loss/PPL relationship.
- **Output.** `step=… | loss=… | ppl=… | lr=… | tps=…` plus one `k=v` per metric. WandB receives `train/loss`, `train/ppl`, `train/lr`, `train/tokens_per_sec` (+ `train/{k}`), logged with `step=step`.
- **State reset.** On a logged step, `_loss_window = []` and `_step_start = time.time()`.
- **Callers.** `training/pretrain.py:Pretrainer.train` — once per optimizer step, with `lr=lr` and `metrics=log_metrics` (e.g. `mtp_loss`, `balance_loss`). This is the only host round-trip in the loop: `.item()` is called once per log step, not per micro-step.

### 2.3 Module-level factory pair

```python
_logger: Optional[TrainingLogger] = None

def init_logging(log_interval: int = 10, seq_len: int = 1024, batch_size: int = 1) -> None:
def get_logger() -> TrainingLogger:
```

- `utils/logging.py:init_logging` replaces the process-global `_logger` with a fresh `TrainingLogger(log_interval=…, seq_len=…, batch_size=…)`. **Caller:** `training/pretrain.py:Pretrainer.__init__` — `init_logging(config.log_every, seq_len=config.max_seq_len, batch_size=config.batch_size)`.
- `utils/logging.py:get_logger` returns the global, lazily constructing a default-parameter `TrainingLogger()` if `init_logging` was never called. **Caller:** `training/pretrain.py:Pretrainer.__init__` — `self.logger = get_logger()`.
- The pair exists so the logger's construction-time needs (`seq_len`, `batch_size`) are satisfied before the loop starts, while the loop itself only holds a handle.

---

## 3. Memory estimator — `utils/memory.py`

CPU-arithmetic VRAM budget estimator (no GPU needed) used by `scripts/microbench_a100.py` and the docs' memory sections. Component bytes (BF16, AdamW FP32): params = 2×N, optim = 12×N, KV = batch·seq·(kv_lora+qk_rope)·2, activations = 24×B·S·D·L (grad-ckpt) or 36× (without). **No GPU training run has ever executed — every figure this module produces is an estimate, not a measurement** (`.benchmarks/` is empty).

### 3.1 Module constant

```python
STATIC_PYTORCH_OVERHEAD_GB = 13.7
```

`utils/memory.py:STATIC_PYTORCH_OVERHEAD_GB` — approx peak overhead from CUDA context + NCCL + caching allocator (A100 80 GB, PyTorch 2.x); empirically ≤ 17% of device total. Consumed by `utils/memory.py:_detect_overhead_gb`.

### 3.2 Private arithmetic helpers

```python
def _deduped_numel(model: nn.Module) -> int:
```

`utils/memory.py:_deduped_numel` — total parameter count with shared tensors counted **once by `id(p)`** (Python object identity — the tied head is the *same `Parameter` object* as the embedding, so it contributes zero). This is why the estimator agrees with `models/transformer.py:count_parameters` (411\,632\,256 for the canonical config). Note the difference from checkpoint dedup, which keys on `data_ptr()`.

```python
def _parameter_bytes(model: nn.Module) -> int:
def _optimiser_bytes(model: nn.Module) -> int:
```

- `utils/memory.py:_parameter_bytes`: `_deduped_numel(model) * 2` — BF16 working weights.
- `utils/memory.py:_optimiser_bytes`: `_deduped_numel(model) * 12` — FP32 master (4) + first moment (4) + second moment (4). In this repo the FP32 parameter *is* the master (no separate copy), so the per-param total is 2 + 12 = 14 bytes.

```python
def _kv_cache_bytes(model: nn.Module, seq_len: int, batch_size: int, dtype_bytes: int = 2) -> int:
```

`utils/memory.py:_kv_cache_bytes` — sums over `model.layers`: per layer, `batch_size * seq_len * (kv_lora_rank + qk_rope_head_dim) * dtype_bytes`; layers without an `attn` attribute (or missing dims) contribute 0. The cache stores the **compressed latent** (192 + 24 = 216 floats/token/layer), independent of head count — the MLA win (see [[Docs/03_Multi_Head_Latent_Attention|T3]]).

```python
def _activation_bytes(
    seq_len: int, batch_size: int, hidden_dim: int, n_layers: int,
    grad_checkpoint: bool, dtype_bytes: int = 2,
) -> int:
```

`utils/memory.py:_activation_bytes` — `factor = 24 if grad_checkpoint else 36`; returns `factor * batch_size * seq_len * hidden_dim * n_layers * dtype_bytes`. The 24/36 constants mirror the PaLM formula (PaLM Appendix A).

```python
def _infer_dim_n_layers(model: nn.Module) -> tuple[int, int]:
```

`utils/memory.py:_infer_dim_n_layers` — `(hidden_dim, n_layers)` for a `Transformer`-shaped model: `model.embed.embedding_dim` and `len(model.layers)` (a `nn.ModuleList`); `(0, 0)` for stubs. `n_layers` counts **all** blocks (2 dense + 16 MoE = 18), matching the config's `n_layers: 18`.

```python
def _detect_overhead_gb() -> float:
```

`utils/memory.py:_detect_overhead_gb` — CPU: `2.0`. CUDA: `min(STATIC_PYTORCH_OVERHEAD_GB, max(2.0, total_gb * 0.17))` where `total_gb` is `torch.cuda.get_device_properties(0).total_memory / 1024**3`. A bound, not a measurement.

### 3.3 `estimate_model_memory_gb`

```python
def estimate_model_memory_gb(
    model: nn.Module, seq_len: int, batch_size: int,
    grad_checkpoint: bool = True, overhead_gb: float | None = None,
    dtype_bytes: int = 2, inference: bool = False,
) -> float:
```

| Param | Default | Contract |
|---|---|---|
| `model` | — | Any `nn.Module`; dims inferred via `_infer_dim_n_layers`, params via `_deduped_numel`. |
| `seq_len` / `batch_size` | — | Drive KV-cache and activation terms. |
| `grad_checkpoint` | `True` | 24× vs 36× activation factor. |
| `overhead_gb` | `None` | `None` → autodetect via `_detect_overhead_gb()`. |
| `dtype_bytes` | `2` | BF16. |
| `inference` | `False` | `True` drops the optimizer (12×N) and activation terms — inference carries no AdamW state and forward activations are dominated by the KV cache. |

- **Formula.** `bytes_total = _parameter_bytes(model) + _kv_cache_bytes(model, seq_len, batch_size, dtype_bytes)`; if not `inference`, add `_optimiser_bytes(model) + _activation_bytes(seq_len, batch_size, hidden_dim, n_layers, grad_checkpoint, dtype_bytes)`. Return `(bytes_total / 1024**3) + overhead_gb`.
- **Canonical-config numbers** (batch 8, seq 2048, 18 layers, BF16, 411\,632\,256 deduped params; component subtotal verified by running the estimator on the real model, 2026-08-04 — still estimates): parameters 0.77 GiB, AdamW 4.60 GiB, KV cache 0.12 GiB, activations (grad-ckpt) 10.13 GiB → subtotal ~15.6 GiB; + overhead 2 (CPU) – 13.6 (CUDA) → **~17.6 CPU / ~29.2 CUDA GiB**. With MTP (418\,713\,984 params) the subtotal is ~15.7 GiB. `inference=True` → ~0.89 GiB. Full table in [[Docs/11_Operations_and_Testing|T11 §5]].
- **Callers.** `scripts/microbench_a100.py` (training-mode estimate); docs chapters 02/10/11/12; `tests/test_utils.py` (`TestMemoryEstimation` pins each component's byte arithmetic).

### 3.4 `assert_fits_in_available_gpu`

```python
def assert_fits_in_available_gpu(estimate_gb: float, safety_margin_gb: float = 0.0) -> None:
```

- **Behavior.** No-op on CPU. On CUDA, raises `RuntimeError` if `estimate_gb > available_gb - safety_margin_gb` (message includes all three numbers, 2-dp). `available_gb` = device-0 total memory in GiB.
- **Callers.** `scripts/microbench_a100.py` (`safety_margin_gb=2.0`) so a misconfigured batch size fails loud before the run; `tests/test_utils.py`. The CPU no-op is what lets the estimator be unit-tested everywhere.

---

## 4. Callers map (quick reference)

| Symbol | Production callers |
|---|---|
| `utils/checkpoint.py:CheckpointManager.__init__` | `training/pretrain.py:Pretrainer.__init__`, `inference/generate.py:main`, `scripts/e2e_test_gpu.py` |
| `utils/checkpoint.py:CheckpointManager.save` | `training/pretrain.py:Pretrainer.save_checkpoint` |
| `utils/checkpoint.py:CheckpointManager.load` | `training/pretrain.py:Pretrainer.load_checkpoint`, `inference/generate.py:main`, `scripts/e2e_test_gpu.py` |
| `utils/checkpoint.py:CheckpointManager.latest_step` | `training/pretrain.py:Pretrainer._find_latest_checkpoint`, `inference/generate.py:main`, `scripts/e2e_test_gpu.py` |
| `utils/checkpoint.py:CheckpointManager._atomic_write` / `_atomic_save_safetensors` / `_atomic_save_torch` / `_atomic_save_json` | internal to `save`; crash path pinned by `tests/test_utils.py` |
| `utils/logging.py:init_logging` / `get_logger` | `training/pretrain.py:Pretrainer.__init__` |
| `utils/logging.py:TrainingLogger.log` | `training/pretrain.py:Pretrainer.train` (once per optim step) |
| `utils/memory.py:estimate_model_memory_gb` | `scripts/microbench_a100.py`, docs, `tests/test_utils.py` |
| `utils/memory.py:assert_fits_in_available_gpu` | `scripts/microbench_a100.py` (margin 2.0), `tests/test_utils.py` |

Sibling references: `../reference/R7_training_api.md` (Pretrainer wiring), `../reference/R2_transformer_api.md` (weight tying / `count_parameters`), `../reference/R9_inference_api.md` (checkpoint resolution in `generate`), `../guides/G5_checkpoint_ops.md` (recovery playbook).

---

## 5. Pitfalls

- **`load` defaults to `device="cuda"`.** On a CPU-only box, `CheckpointManager.load(model, step)` without `device="cpu"` fails at `load_file`. `Pretrainer` and `generate.py` always pass an explicit device; direct callers must too.
- **`strict=True` + weight tying = error.** The checkpoint legitimately lacks `head.weight`; loading with the default `strict=True` raises `RuntimeError` on the missing key. Production paths always pass `strict=False`; the audit then reports the missing key as a warning.
- **The triplet is not transactional.** Each file is atomic, but a crash between the three writes leaves an incomplete step. `latest_step`/`_checkpoint_complete` are the only safe resume selectors — never resume from a bare `model_step_N.safetensors` you found by globbing.
- **`_list_steps` parses stems.** A file named `model_step_abc.safetensors` is silently skipped; a step whose safetensors is missing is invisible even if the other two files exist.
- **`json.dump(..., default=str)` stringifies non-JSON values.** `Pretrainer`'s meta is JSON-safe only because `LambdaLR.state_dict()` holds floats/ints; a stray tensor in `extra_meta` would silently serialize as its repr and corrupt the resume contract on the next `load`.
- **`torch.load(..., weights_only=True)`** (in `load`) refuses pickled classes — optimizer files must be plain `torch.save` dicts of tensors. A hand-crafted `.pt` with custom objects fails loudly, which is the point.
- **Dedup keys differ between modules.** Checkpoint dedup uses `data_ptr()` (storage identity); the memory estimator uses `id(p)` (Parameter object identity). Both catch the tied head, but they are not interchangeable notions.
- **tps assumes full intervals.** `tokens_per_sec` divides `log_interval × seq_len × batch_size` by wall time; a stalled or skipped micro-batch inflates the window's elapsed time and under-reports throughput. It is a rolling average, not a peak.
- **WandB is frozen at construction.** `init_logging` decides once whether to attach WandB; setting `WANDB_PROJECT` later does nothing.
- **Memory figures are estimates.** No GPU run has executed; `estimate_model_memory_gb` output and the docs' budget tables are arithmetic, not `torch.cuda.max_memory_allocated()` measurements.

---

## 6. Check your understanding

**Q1. Why can `CheckpointManager.load` restore `head.weight` when the file has no `head.weight` key?**

The saver dropped the duplicate (caller filter + `_atomic_save_safetensors` `data_ptr` dedup). `Transformer.__init__` re-establishes the tie before loading, so `load_state_dict(strict=False)` copies into `embed.weight`, which is the head's storage; `head.weight` appears in the missing-keys audit and is expected.

**Q2. A crash happens after `model_step_4000.safetensors` and `optim_step_4000.pt` are written but before `meta_step_4000.json`. What does `latest_step()` return?**

Not 4000 — `_checkpoint_complete(4000)` is false (meta missing), so `latest_step()` skips it and returns the next-lower complete step (or `None`). The torn step is invisible to resume.

**Q3. Why does `estimate_model_memory_gb(..., inference=True)` return ~0.89 GiB while training mode returns ~29.2 GiB?**

`inference=True` drops the optimizer term (12×N ≈ 4.60 GiB) and the activation term (24×B·S·D·L ≈ 10.13 GiB), leaving weights (0.77) + KV cache (0.12) + overhead — inference carries no AdamW state and forward activations are dominated by the KV cache.

**Q4. `TrainingLogger.log` is called with `step=5` and `log_interval=10`. What happens?**

The loss is appended to `_loss_window` and the method returns early (`5 % 10 != 0`). Nothing prints, no WandB write, no window reset — the window keeps accumulating until a step divisible by 10.

---

<!-- docs:verified 2026-08-04 · 59aeef3 -->
