# G5 — Checkpoint Operations: Save, Load, Resume & Recovery

> **Canonical** for DeepSeek-v3-Lite's checkpoints: the three-file-per-step layout, the atomic-write mechanics that make a checkpoint crash-safe, shared-tensor dedup under weight tying, the two resume paths, and the disaster-recovery playbook. Procedural guide — the surrounding theory lives in [[Docs/08_Training_Pipeline]] and [[Docs/11_Operations_and_Testing]].

> **Status:** the checkpoint machinery is implemented and covered by the CPU test suite (196 nodes: 186 pass + 10 GPU-gated skips, see [[Docs/11_Operations_and_Testing]]), but **no GPU training run has ever executed** — `checkpoints/` is empty, so every file-size figure below is a derived estimate, not a measurement.

**Depends on:** [[Docs/08_Training_Pipeline]] · [[Docs/11_Operations_and_Testing]] · **Read next:** [[guides/G1_debugging_playbook]] (NaN rollback), [[guides/G6_contributing]] (doc/test expectations)

---

## 1. 60-Second Summary

A checkpoint at step $N$ is **three files in one directory**: `model_step_N.safetensors` (BF16 weights), `optim_step_N.pt` (FP32 AdamW state), and `meta_step_N.json` (scheduler state, step counters, the full `TrainingConfig`, and an `has_mtp` flag). Each file is written atomically — temp file in the same directory, then `os.replace` — so no reader ever sees a half-written file. The weight file stores **one copy of every shared tensor**: with weight tying on, `head.weight` is dropped (it *is* `embed.weight`), and on load `strict=False` restores it through the surviving storage. The training loop auto-resumes from the newest **complete** step (all three files present); a crash between the three writes leaves an incomplete set that is silently skipped. This guide walks the save → load → resume → recovery path end to end, and ends with the inference-loading nuance that matters when you serve a checkpoint.

## 2. Why Checkpoints Are Three Files

A training step produces three things that must come back on resume: the model weights, the optimizer's per-parameter moments, and the "where are we" bookkeeping (step counters, LR-schedule position). They have different formats and different failure profiles:

| Concern | Weights | Optimizer | Meta |
|---|---|---|---|
| Format | safetensors (no pickle, header with shapes) | `torch.save` pickle (required by `Optimizer.state_dict`) | JSON (human-readable, diffable) |
| Size | ~0.77 GiB base | ~4.6 GiB FP32 | a few KB |
| Dtype | BF16 | FP32 | text |

`CheckpointManager.save` (`utils/checkpoint.py:CheckpointManager.save`) writes **weights first, optimizer second, meta last**. That order is load-bearing: the meta file is the "commit" record, so a step is only considered complete when all three exist (see §4). It also means an interrupted save can only orphan the *early* files of a step — never leave a meta file claiming a step whose weights vanished.

The split also mirrors how the pieces decay in a disaster: weights are the crown jewels (recoverable from a `.pt`/HuggingFace export), optimizer state is the biggest but most disposable (losing it costs training progress, not correctness), and meta is tiny but **irreplaceable** — the LR schedule's position lives only there (§8.3).

## 3. The File Format

### 3.1 `model_step_N.safetensors` — weights only

Keys are the raw `Transformer` state-dict names with two transformations applied in `training/pretrain.py:Pretrainer.save_checkpoint`:

1. `head.weight` is dropped when `model.weight_tying: true` (canonical config) — §5.
2. MTP weights are renamed with an `mtp.` prefix and only the `mtp_modules.*` subset is kept:

```python
if self.mtp_wrapper is not None:
    mtp_mod = self.mtp_wrapper
    orig = getattr(mtp_mod, "_orig_mod", mtp_mod)
    mtp_state = {f"mtp.{k}": v for k, v in orig.state_dict().items() if k.startswith("mtp_modules.")}
    state.update(mtp_state)
```

So an MTP checkpoint contains keys like `mtp.mtp_modules.0.block.w1.weight` and `mtp.mtp_modules.0.norm.weight`, but **not** `mtp.embed.weight` or `mtp.mtp_modules.0.output_head.weight` — both alias main-model tensors and are excluded (filter) or deduplicated (§5). `tests/test_training.py` asserts the `mtp.` prefix exists after a save (`test_checkpoint_safetensors_mtp_prefix`).

### 3.2 `optim_step_N.pt` — optimizer state

The full `AdamW.state_dict()`: FP32 master weights, `exp_avg`, `exp_avg_sq`, and `param_groups`. The optimizer is constructed with **deduplicated parameter ids** in `training/pretrain.py:Pretrainer.__init__` (the `seen` set skips the shared head/embed pair), so the tied tensor's moments are stored once. Loaded with `torch.load(..., map_location=device, weights_only=True)` inside `utils/checkpoint.py:CheckpointManager.load`.

### 3.3 `meta_step_N.json` — everything else

Assembled by `Pretrainer.save_checkpoint` and written by `CheckpointManager._atomic_save_json` (`utils/checkpoint.py:CheckpointManager._atomic_save_json`). Contents:

```python
extra_meta = {"scheduler": self.scheduler.state_dict(), "opt_steps": self._opt_steps,
              "tag": tag or f"step_{step}", "config": asdict(self.config), "has_mtp": self.mtp_wrapper is not None}
```

- `step` — the micro-step (global_step) counter; forced from the save argument in `CheckpointManager.save` (`{"step": step, **{k: v for k, v in (extra_meta or {}).items() if k != "step"}}`), so a caller-supplied `step` key can never corrupt it.
- `scheduler` — the `LambdaLR` state dict: `base_lrs`, `last_epoch`, `_step_count`, `_last_lr`. It round-trips through JSON correctly because `LambdaLR.load_state_dict` pops `lr_lambdas` and keeps the live closures — `json.dump(..., default=str)` stringifies anything non-JSON (a latent foot-gun if scheduler state ever grows a tensor field; today it is all ints/floats/lists).
- `opt_steps` — the optimizer-step counter (`self._opt_steps`). This is what makes the LR schedule resume *mid-arc*: the cosine horizon is `max_steps // gradient_accumulation_steps` in opt-step space (see [[Docs/08_Training_Pipeline]] and [[guides/G2_mup_and_lr_tuning]]), and both this counter and the scheduler state travel in the JSON.
- `config` — `dataclasses.asdict(self.config)`, i.e. the entire `TrainingConfig` including the raw YAML `model_config`. A checkpoint is self-describing: you can rebuild the exact training setup from `meta_step_N.json` alone.
- `has_mtp` — whether the run trained with an MTP wrapper; the loader uses it to decide whether to restore `mtp.` keys.
- `tag` — `""` (empty) for periodic saves, `"final"` for the end-of-run save. **The filename does not change** — a final checkpoint is still `model_step_512000.safetensors`; only the meta distinguishes it. `_find_latest_checkpoint` therefore cannot tell a final from a periodic save.

### 3.4 What it costs (derived estimates)

With canonical numbers (411.6M deduped params, 2 B/param BF16 weights, 12 B/param FP32 AdamW state = 4 B master + 4 B `exp_avg` + 4 B `exp_avg_sq`):

| File | Size | Notes |
|---|---|---|
| `model_step_N.safetensors` | ~0.77 GiB (+ ~14 MB with MTP) | deduped count × 2 B; head stored once via embed |
| `optim_step_N.pt` | ~4.6 GiB (+ ~85 MB with MTP) | 12 B/param; largest file, slowest write |
| `meta_step_N.json` | a few KB | instant |

[INFERENCE] — derived from arithmetic, consistent with the memory budget in [[Docs/11_Operations_and_Testing]] §5.2; no checkpoint has been produced by a real run yet. At `save_interval: 4000` over 512,000 steps that is ~127 periodic checkpoints ≈ ~0.7 TB of disk if never pruned — retention is not wired in (see §10).

## 4. Atomicity Mechanics

`CheckpointManager._atomic_write` (`utils/checkpoint.py:CheckpointManager._atomic_write`) is the single chokepoint every writer goes through:

```python
import contextlib
@contextlib.contextmanager
def _atomic_write(self, path: Path, suffix: str):
    fd, tmp = tempfile.mkstemp(dir=self.save_dir, suffix=suffix)
    os.close(fd)
    try:
        yield tmp
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
```

Three properties, all deliberate:

1. **Same-directory temp file.** `tempfile.mkstemp(dir=self.save_dir, ...)` creates `*.safetensors.tmp` / `*.pt.tmp` / `*.json.tmp` next to the target. `os.replace` is only atomic on the *same filesystem* — a temp file in `/tmp` on another mount would make the rename non-atomic (or cross-device). Writing in the checkpoint dir guarantees the rename is a single metadata operation.
2. **Torn writes never reach the final name.** A reader (another process, or the next boot) either sees the old file or the new file, never a truncated middle. A hard kill mid-write leaves an orphaned `*.tmp`; the final name is untouched.
3. **Error cleanup.** An exception inside the block unlinks the temp before re-raising — no litter on the common failure paths. Only `kill -9`-style deaths orphan temps, and those are harmless: `_list_steps` globs only `model_step_*.safetensors`, and orphaned `model_step_N.safetensors.tmp` doesn't match.

**The three-file set is *not* atomic as a group.** Each file is atomic individually; the *step* is atomic only in the logical sense that `_checkpoint_complete` requires all three, and `latest_step` refuses incomplete steps:

```python
def latest_step(self) -> Optional[int]:
    steps = self._list_steps()
    return next((s for s in sorted(steps, reverse=True) if self._checkpoint_complete(s)), None)
```

(`utils/checkpoint.py:CheckpointManager.latest_step` / `utils/checkpoint.py:CheckpointManager._checkpoint_complete`.) Because meta is written last, a crash between files 1 and 3 leaves a step that `latest_step` skips — the run resumes from the previous complete step. The orphaned weight file is dead weight on disk but never selected.

## 5. Shared-Tensor Dedup and Weight Tying

Weight tying (`models/transformer.py:Transformer.__init__`) makes the LM head and the embedding **the same `Parameter`**:

```python
self.weight_tying = model_cfg.get("weight_tying", False)
self.head = nn.Linear(model_cfg["dim"], model_cfg["vocab_size"], bias=False)
if self.weight_tying:
    self.head.weight = self.embed.weight
```

So a naive `state_dict()` contains two keys over one storage: `head.weight` (100\,018 × 768 ≈ 153.6 MB of BF16) is a duplicate. Two layers of defense remove it:

1. **`Pretrainer.save_checkpoint` drops the key** before the manager sees it.
2. **`CheckpointManager._atomic_save_safetensors` (`utils/checkpoint.py:CheckpointManager._atomic_save_safetensors`) dedups by storage pointer** — a genuine dedup, not a name convention:

```python
seen_ptrs: set = set()
deduped: dict = {}
for k, v in state.items():
    if v.data_ptr() in seen_ptrs:
        continue
    seen_ptrs.add(v.data_ptr())
    deduped[k] = v.contiguous()
```

The first key in dict order wins; `head.weight` (registered last) loses to `embed.weight`. The same pointer check silently collapses the MTP aliases: `mtp.mtp_modules.0.output_head.weight` shares the main head's storage and `MultiTokenPrediction.__init__` (`models/mtp.py:MultiTokenPrediction.__init__`) registers `embed` and the shared head via `set_output_head` (`models/mtp.py:MTPModule.set_output_head`), so every one of those keys is a duplicate and never hits the file.

**Restore works because aliasing is structural, not serialized.** On load, `CheckpointManager.load` calls `model.load_state_dict(weights, strict=False)`; `head.weight` is reported missing (a logged warning, see §10) but the load copies into `embed.weight` *in place* — and since `head.weight` *is* `embed.weight` (same storage), the head is updated through the survivor. No code ever reconstructs the tie; the tie reconstructs it. This is asserted by `tests/test_utils.py` (load restores `embed.weight` and `head.weight.data_ptr()` equality) and documented as the expected warning in [[Docs/10_Inference_and_Serving]].

## 6. Save Flow

Checklist (what actually happens, in order):

1. `train()` hits `global_step % save_every == 0` (canonical `save_interval: 4000`) → `save_checkpoint(global_step)` (`training/pretrain.py:Pretrainer.save_checkpoint`).
2. Snapshot `self.raw_model` — **the unwrapped `Transformer`, not the `torch.compile`d or MTP-wrapped `self.model`**. Compile wrappers would emit `_orig_mod.`-prefixed keys; MTP keys are merged explicitly instead. Note the `_orig_mod` unwrap is applied to the MTP wrapper when it was compiled.
3. Drop `head.weight` (tying), merge `mtp.*` keys (MTP only).
4. Build `extra_meta` (§3.3) and call `self.ckpt_manager.save(...)`.
5. `CheckpointManager.save` writes safetensors → `.pt` → `.json`, each through `_atomic_write`.

The end-of-run save reuses the same path with `tag="final"` (`self.save_checkpoint(global_step, tag="final")` inside `training/pretrain.py:Pretrainer.train`).

**Cost note:** the safetensors write serializes the full deduped weight dict (≈0.77 GiB) and the optimizer write ≈4.6 GiB — on an A100's NVMe these are seconds, not minutes, but they happen on the training loop's critical path, synchronously. There is no async/background save: the loop stalls while the files land. [INFERENCE] — no run has measured this yet.

## 7. Load and Resume Flow

### 7.1 The resume decision tree

```
Launch training
├─ checkpoint dir empty            → fresh run, global_step = 0
└─ complete step(s) exist          → AUTO-RESUME: load newest complete step
     └─ load fails (corrupt/missing) → [warn] Could not load checkpoint: …
        └─ continue as fresh run
```

`Pretrainer.train` (`training/pretrain.py:Pretrainer.train`) does this unconditionally:

```python
global_step = 0
latest = self._find_latest_checkpoint()
if latest is not None:
    try:
        global_step = self.load_checkpoint(latest)
    except Exception as exc:
        self._log(f"[warn] Could not load checkpoint: {exc}")
```

`_find_latest_checkpoint` (`training/pretrain.py:Pretrainer._find_latest_checkpoint`) is a one-line delegate to `latest_step()`. **Operational consequence:** to start a genuinely fresh run in a directory that already has checkpoints, point `--checkpoint-dir` at a new directory (or move the old one aside) — a bare relaunch resumes.

`Pretrainer.load_checkpoint` (`training/pretrain.py:Pretrainer.load_checkpoint`) then:

1. `ckpt_manager.load(self.raw_model, step, device=..., optimizer=self.optimizer, strict=False)` — weights into the unwrapped model, optimizer state into `self.optimizer` (if the `.pt` exists).
2. If `self.mtp_wrapper is not None` **and** `meta["has_mtp"]` — reload the `mtp.`-prefixed keys (stripped of the prefix) into the unwrapped MTP module with `strict=False`.
3. `scheduler.load_state_dict(meta["scheduler"])` and `self._opt_steps = meta["opt_steps"]` — schedule position restored.
4. Return `meta["step"]` → becomes `global_step`.

### 7.2 `--resume N`: the ordering nuance

`training/pretrain.py:main` applies an explicit step **before** `train()` runs:

```python
trainer = Pretrainer(config)
if args.resume is not None:
    trainer.load_checkpoint(int(args.resume))
trainer.train()
```

Then `train()` *again* resolves `latest_step()` and loads it. The effective resume point is therefore **the newest complete step**, whatever `--resume` said — `--resume N` is decisive only when $N$ is that newest complete step (the normal case right after a crash at $N$). If $N$ names an incomplete or missing weight file, `CheckpointManager.load` raises `FileNotFoundError` immediately (with the available steps listed). In practice: after a crash, relaunch the same command with `--resume <step>` and you resume from the newest intact checkpoint; the flag documents intent, the completeness check guarantees integrity.

Resume is not lossless by design: the DataLoader is rebuilt with a fresh seeded generator (`torch.Generator().manual_seed(self.config.seed)`), so **data order differs across a restart** — the code comments call this benign at this scale; exact resume would need sampler checkpointing, which does not exist.

## 8. Disaster Recovery

Ranked by severity:

### 8.1 Missing optimizer file (`optim_step_N.pt`)

`CheckpointManager.load` treats it as a warning, not an error:

```
[checkpoint] no optimiser state at <path> — optimizer will start from scratch
```

Weights, scheduler, and `opt_steps` still load — the **LR schedule continues correctly** even though AdamW's moments are zero. The run loses the optimizer's adaptation (a few hundred steps of momentum history) but not its position. This is the cheapest disaster and the designed degradation.

### 8.2 Corrupt or truncated `model_step_N.safetensors`

`load_file` raises (safetensors verifies header/size metadata, so truncation is caught, not silently misread). Because each file is written atomically, this only happens through real disk corruption or a hand-edited file. Handling depends on the entry point:

- **Auto-resume in `train()`:** the `try/except` catches it → `[warn] Could not load checkpoint: ...` → fresh start from step 0 with the existing weights untouched. Note this *silently discards* the corrupted step's siblings (they are skipped because the step is incomplete).
- **Explicit `--resume N`:** the exception propagates and the run aborts — loud, which is what you want for an explicit request.

Recovery: the previous complete step is still intact (atomicity guarantees it). Resume with `--resume <previous>` or let auto-resume pick it up — and delete the corrupt step's three files so `latest_step` stops considering them.

### 8.3 Missing or corrupt `meta_step_N.json` — the silent hazard

If the meta file is absent, `CheckpointManager.load` substitutes `{"step": step}` — weights and optimizer load, but **scheduler state and `opt_steps` are not restored**. The run continues at `global_step = N` with a *fresh* `LambdaLR` — i.e. the LR schedule restarts its warmup at step 0 while training thinks it is at step $N$. The warmup lambda at step 0 is 0, so the effective LR collapses toward 0 and recovery is not obvious from the loss curve. This is why completeness requires all three files: meta is the only carrier of the schedule position. If you must reconstruct, take `max_steps // gradient_accumulation_steps` as the horizon and re-derive the position from `global_step` ([[guides/G2_mup_and_lr_tuning]]).

### 8.4 NaN guard rollback

`train_step` (`training/pretrain.py:Pretrainer.train_step`) returns `None` on a NaN/Inf total loss after zeroing gradients; after `nan_guard_max_consecutive: 5` consecutive fires, `train()` restores the latest complete checkpoint, resets `global_step` to its step, and clears the streak — or raises `RuntimeError("NaN/Inf with no checkpoint to restore from")` if none exists. Rollback is the same `load_checkpoint` path as resume, so weights + optimizer + schedule all rewind together. The DataLoader is **not** rewound and the partial accumulation window's gradients are discarded — the guard saves the model state, not the data stream. See [[guides/G1_debugging_playbook]] for the full state machine and `tests/test_training.py` (`TestNanGuardRollback`) for the rollback test.

## 9. Inference Loading: the Weight-Tying Nuance

`inference/generate.py:main` (`inference/generate.py:main`) accepts `--checkpoint` as either a directory (→ `latest_step()`) or a file path (→ the trailing integer of the stem, e.g. `model_step_4000.safetensors` → 4000), then:

```python
model = Transformer(model_cfg).to(args.device)
model.eval()
...
ckpt_mgr.load(model, step, device=args.device, strict=False)
```

The restore chain works **only because the inference model is built from the same config** — canonical `weight_tying: true` means `head.weight` aliases `embed.weight` again, so the missing-key restore of §5 applies. The nuance: **the checkpoint file never contains `head.weight`, period.** Any consumer that builds a non-tied model (`weight_tying: false`) gets a silently untrained head — `strict=False` logs a missing-key warning and continues. There is no way to recover the head from the file; the tie is the only carrier. Same contract for the MTP draft head: the wrapper re-aliases `output_head` to `model.head` via `set_output_head` *before* loading `mtp.` keys, and `mtp_module.load_state_dict(mtp_state, strict=False)` tolerates the absent shared head (see [[Docs/10_Inference_and_Serving]] §MTP, which covers the `[warn] No MTP weights in checkpoint` case). `Transformer.__init__` also re-derives the mask/cache infrastructure from the config, so an inference config must match the training config's `max_seq_len` or the pre-allocated cache differs from what the checkpoint assumed — benign for weights, relevant for KV-cache sizing ([[Docs/10_Inference_and_Serving]]).

## 10. Pitfalls Checklist

- **`head.weight` missing on load is expected, not a bug.** The `[checkpoint] 1 missing key(s): ['head.weight']` warning fires on every tied-model load (via `strict=False` in both `Pretrainer.load_checkpoint` and `inference/generate.py:main`). Verify the tie instead: `model.head.weight.data_ptr() == model.embed.weight.data_ptr()`.
- **`--no-checkpoint` disables gradient checkpointing, not checkpoint saving.** The flag feeds `grad_checkpoint` (activation re-computation) in `training/pretrain.py:main`; it has nothing to do with `save_interval`. Checkpoint files are always written.
- **`latest_step()` is a completeness filter, not a freshness scan.** It globs `model_step_*.safetensors` and requires all three files; a step whose meta or optimizer file was deleted (or whose save was interrupted) is invisible to auto-resume.
- **Meta is the only home of the schedule position.** Deleting `meta_step_N.json` silently restarts the LR warmup (§8.3). Never hand-prune a checkpoint to "just the weights."
- **Final checkpoints look identical to periodic ones on disk.** `tag="final"` lives only in the JSON; retention tooling must read meta, not filenames.
- **The three files are not a transaction.** Crash between writes → orphaned `model_step_N.safetensors` + possibly `optim_step_N.pt`; harmless (skipped) but they eat disk. Retention is unwired (the code comment says "Only callers were tests; training loop uses save + latest_step. Add back when retention is wired in") — at `save_interval: 4000` a full 512k-step run writes ~127 steps × ~5.4 GiB ≈ ~0.7 TB. Plan disk or add pruning before a real run.
- **Temps can litter on hard kills.** `*.safetensors.tmp` / `*.pt.tmp` / `*.json.tmp` orphans are ignored by discovery but never auto-cleaned.
- **Non-tied consumers get a dead head.** Loading a tied checkpoint into `weight_tying: false` silently leaves `head.weight` at init — check the config, not the log.

## 11. Check Your Understanding

1. A save is interrupted by `kill -9` after `model_step_4000.safetensors` lands but before `optim_step_4000.pt`. What happens on relaunch? *(Answer: `latest_step` finds 4000 incomplete → skips it → resumes from the previous complete step, e.g. 0. The orphaned weight file is never selected.)*
2. `meta_step_8000.json` is deleted by accident; the other two files survive. What does auto-resume do, and what is the danger? *(Answer: 8000 is no longer "complete" so it is skipped; resuming `--resume 8000` loads weights + optimizer but restores neither scheduler state nor `opt_steps`, restarting the LR warmup at `global_step = 8000` — a near-zero LR that is invisible in the loss curve for a while.)*
3. Why does loading a tied checkpoint report `head.weight` missing yet still produce a correctly tied, correctly trained head? *(Answer: the checkpoint stores the embedding once; `head.weight` is the same storage as `embed.weight` (set in `Transformer.__init__`), so the in-place copy into `embed.weight` writes through to the head. `strict=False` is what tolerates the cosmetic missing key.)*
4. You launch `python training/pretrain.py --config configs/pretrain_a100_422m.yaml --resume 4000` in a directory whose newest complete step is 12000. Which step does training actually continue from, and why? *(Answer: 12000 — `main` loads 4000, then `train()` unconditionally auto-resumes the newest complete step. `--resume` is decisive only when it names the newest complete step.)*

---

<!-- docs:verified 2026-08-04 · 59aeef3 -->
