# Utilities — Checkpoint, Memory, Logging

## A Comprehensive Technical Reference

> **Covers**: `utils/checkpoint.py`, `utils/memory.py`, `utils/logging.py` — production infrastructure for training and inference.

---

## Table of Contents

1. [Abstract](#abstract)
2. [CheckpointManager](#checkpointmanager)
3. [Atomic Write Protocol](#atomic-write-protocol)
4. [Shared-Tensor Dedup](#shared-tensor-dedup)
5. [MTP Checkpoint Roundtrip](#mtp-checkpoint-roundtrip)
6. [Memory Estimation](#memory-estimation)
7. [TrainingLogger](#traininglogger)
8. [Appendix A — Worked memory example](#appendix-a--worked-memory-example)
9. [Appendix B — FAQ](#appendix-b--faq)
10. [Appendix C — Glossary](#appendix-c--glossary)
11. [References](#references)

---

## Abstract

Three utility modules support the training and inference lifecycle:

| Module | Role |
|---|---|
| `checkpoint.py` | Atomic safetensors save/load with step discovery |
| `memory.py` | VRAM budget estimation (PaLM-style formulas) |
| `logging.py` | Console + optional WandB metrics |

All are CPU-friendly and tested without GPU.

---

## CheckpointManager

### Files per step `N`

```
checkpoints/pretrain_a100/
  model_step_N.safetensors   # model weights (+ mtp.* keys)
  optim_step_N.pt            # AdamW state_dict
  meta_step_N.json           # scheduler, opt_steps, config, has_mtp
```

### API

```python
mgr = CheckpointManager("checkpoints/pretrain_a100")

# Save
mgr.save(model, optimizer, step=4000, extra_meta={...}, state_dict=optional)

# Load
meta = mgr.load(model, step=4000, device="cuda", optimizer=opt, strict=False)

# Discovery
step = mgr.latest_step()  # highest N with all 3 files present
```

### Step discovery rules

- `_list_steps()` — parses `model_step_*.safetensors` stems
- `_checkpoint_complete(N)` — requires **all three** files
- `latest_step()` — highest complete step, or `None`

Incomplete checkpoints (e.g., crash mid-save) are ignored.

### Load behaviour

| `strict` | Missing keys | Unexpected keys |
|---|---|---|
| `True` | Raise | Raise |
| `False` | Log warning, continue | Log warning, continue |

Training and inference use `strict=False` (weight tying leaves `head.weight` "missing").

Optimizer load is optional — missing `optim_step_N.pt` logs warning, starts fresh.

---

## Atomic Write Protocol

Every save uses temp-file → rename:

```python
fd, tmp = tempfile.mkstemp(dir=save_dir, suffix=".safetensors.tmp")
save_file(deduped_state, tmp)
os.replace(tmp, final_path)   # atomic on POSIX
```

On failure: temp file unlinked, exception re-raised. No partial checkpoints.

`test_atomicity_temp_file_cleaned` verifies no `.tmp` files remain.

**Why not pickle for model weights?** Safetensors is mmap-friendly, cross-language, and avoids arbitrary code execution. Optimiser state still uses `torch.save` (contains non-tensor objects).

---

## Shared-Tensor Dedup

Weight tying makes `head.weight` and `embed.weight` the **same storage**:

```python
for k, v in state.items():
    ptr = v.data_ptr()
    if ptr in seen_ptrs:
        deduped[k] = v.contiguous().clone()  # safetensors rejects duplicate ptrs
    else:
        seen_ptrs.add(ptr)
        deduped[k] = v.contiguous()
```

On save, `Pretrainer` **drops** `head.weight` entirely (redundant with embed).

On load, `head.weight` is "missing" — the tied tensor is restored via `embed.weight`.

---

## MTP Checkpoint Roundtrip

Save path (`Pretrainer.save_checkpoint`):

```python
mtp_state = {f"mtp.{k}": v for k, v in mtp_wrapper.state_dict().items()
             if k.startswith("mtp_modules.")}
state.update(mtp_state)
extra_meta["has_mtp"] = True
```

Load path:

```python
if meta.get("has_mtp"):
    mtp_state = {k.removeprefix("mtp."): v for k, v in weights.items() if k.startswith("mtp.")}
    mtp_wrapper.load_state_dict(mtp_state, strict=False)
```

MTP params are in the **same AdamW optimiser** as the main model (via `MultiTokenPrediction.parameters()`). No separate optim state.

---

## Memory Estimation

### PaLM-style activation model

Activation memory dominates training VRAM for long sequences. This repo uses the PaLM heuristic:

$$
M_{\text{act}} = f \cdot B \cdot S \cdot D \cdot L \cdot \text{bytes}
$$

where $f = 24$ with gradient checkpointing (recompute ~1/3 of layers in backward), $f = 36$ without. The factor accounts for storing inputs to matmuls, attention softmax buffers, and MoE routing temporaries.

**Why not exact?** PyTorch allocator fragmentation, `torch.compile` workspaces, and MoE stacked-weight refresh are not modelled — always leave 15–20% headroom.

`estimate_model_memory_gb(model, seq_len, batch_size, grad_checkpoint=True)`

### Component formulas (BF16 training)

| Component | Bytes | Formula |
|---|---|---|
| Params | 2 × N | BF16 weights |
| Optimiser | 12 × N | FP32 master + m + v |
| KV cache | Σ B·S·(R+d_rope)·2 | Per MLA layer (training: usually 0) |
| Activations | factor·B·S·D·L·2 | factor=24 (ckpt) or 36 (no ckpt) |
| Overhead | min(13.7, 0.17·total) GB | CUDA context |

Where:
- N = deduped parameter count
- B = batch size, S = seq len, D = dim, L = n_layers
- R = kv_lora_rank (192), d_rope = qk_rope_head_dim (24)

### `assert_fits_in_available_gpu(estimate_gb, margin=2.0)`

No-op on CPU. On CUDA, raises if estimate exceeds `total_vram - margin`.

Used by `scripts/microbench_a100.py` as pre-flight check.

### Inference mode

`inference=True` drops optimiser and activation bytes — KV cache dominates.

---

## TrainingLogger

```python
init_logging(log_interval=50, seq_len=2048)
logger = get_logger()
logger.log(step, loss, lr=lr, metrics={"balance_loss": 1.2, "mtp_loss": 3.4})
```

### Console output (every `log_interval` steps)

```
step=    100 | loss=3.2145 | ppl=24.89 | lr=4.00e-05 | tps=125,440 | balance_loss=1.1234
```

- **ppl** = exp(avg_loss) over the log window
- **tps** = `log_interval × seq_len / elapsed_seconds`

### WandB integration

Set environment variables:

```bash
export WANDB_PROJECT=deepseek-v3-lite-a100
export WANDB_RUN_NAME=422m-chinchilla
```

If `wandb` not installed, prints warning and continues console-only.

---

## Appendix A — Worked memory example

422M config, B=8, S=2048, grad_checkpoint=True:

```
N ≈ 422M params (deduped)
Params:      422M × 2 B  = 0.84 GB
Optim:       422M × 12 B = 5.06 GB
Activations: 24 × 8 × 2048 × 768 × 18 × 2 = 10.9 GB
KV (train):  0 (use_cache=False)
Overhead:    min(13.7, 0.17 × 17) ≈ 2.9 GB → capped logic gives ~13.6 GB
Total ≈ 30.5 GB estimated, ~35 GB measured peak
```

---

## Appendix B — FAQ

**Q: Why three files per checkpoint?** Separates concerns: weights (safetensors), optim (torch), metadata (JSON). Allows inference-only load of safetensors.

**Q: Can I load a checkpoint on CPU?** Yes — `device="cpu"` in `load()`.

**Q: What if I only have model_step_N.safetensors?** `latest_step()` returns None (incomplete). Inference can still load weights directly via safetensors.

---

## Appendix C — Glossary

| Term | Meaning |
|---|---|
| `data_ptr` | Tensor storage address for dedup |
| `extra_meta` | JSON metadata beyond step number |
| `has_mtp` | Flag in meta for MTP weight restore |
| `STATIC_PYTORCH_OVERHEAD_GB` | 13.7 GB CUDA context estimate |
| `factor 24/36` | PaLM activation memory multiplier |

---



---

## Checkpoint Recovery Scenarios

### Crash during save

Atomic write protocol guarantees: either all three files exist (complete step) or none do (incomplete step ignored by `latest_step()`).

```
model_step_4000.safetensors.tmp  → deleted on failure
model_step_4000.safetensors      → only appears after os.replace succeeds
```

### Resume after NaN guard

`Pretrainer.train()` on 5 consecutive NaN/Inf micro-steps:
1. Calls `latest_step()`
2. `load_checkpoint(latest)` — restores weights, AdamW state, scheduler, `_opt_steps`
3. Resets `nan_guard_streak`

**What is NOT restored:** DataLoader position / shuffle order.

### Inference-only load

```python
from safetensors.torch import load_file
state = load_file("checkpoints/pretrain_a100/model_step_4000.safetensors")
model.load_state_dict(state, strict=False)
```

No `optim_step_*.pt` needed. MTP keys prefixed `mtp.` — load separately if using speculative decode.

---

## Memory Estimation — Derivation

### Parameter memory

BF16 weights: $2$ bytes × $N$ params.

### Optimiser memory (AdamW)

Per parameter:
- FP32 master weight: 4 bytes
- First moment $m$: 4 bytes  
- Second moment $v$: 4 bytes

Total: $12$ bytes × $N$ (fused AdamW may pack differently but budget uses 12).

### Activation memory (PaLM heuristic)

From PaLM paper (Chowdhery et al., 2022):

$$
M_{\text{act}} = f \cdot B \cdot S \cdot D \cdot L \cdot \text{bytes}
$$

$f = 24$ with gradient checkpointing (recompute ~2/3 of layers in backward).
$f = 36$ without checkpointing.

**Why linear in $L$?** Each layer stores activations for backward; checkpointing trades compute for memory by not storing all layers.

### KV cache (inference)

Per MLA layer:

```python
bytes = batch * seq * (kv_lora_rank + qk_rope_head_dim) * dtype_bytes
```

Summed over all layers in `_kv_cache_bytes`.

### Overhead

`STATIC_PYTORCH_OVERHEAD_GB = 13.7` — CUDA context, cuBLAS workspace, allocator pools. Capped at `min(13.7, 0.17 * total_vram)`.

---

## TrainingLogger — Metric Definitions

| Metric | Formula | Notes |
|---|---|---|
| `loss` | Rolling mean over `log_interval` micro-steps | Raw CE (main head only in log line) |
| `ppl` | `exp(avg_loss)` | Natural-log base |
| `lr` | `scheduler.get_last_lr()[0]` | After warmup+cosine |
| `tps` | `log_interval * seq_len / elapsed` | **Micro-steps** not optim steps |
| `balance_loss` | Sum of MoE `get_load_balance_loss()` | Logging only — not in task loss |
| `mtp_loss` | Auxiliary CE | When MTP enabled |

**GPU sync discipline:** `.item()` called once per log interval, not per micro-step — avoids 3-4 forced device syncs per step.

---

## WandB Integration Details

Activation requires **only** `WANDB_PROJECT` env var:

```python
wandb_project = os.environ.get("WANDB_PROJECT")
if wandb_project:
    wandb.init(project=wandb_project, name=os.environ.get("WANDB_RUN_NAME"), reinit=True)
```

Logged keys: `train/loss`, `train/ppl`, `train/lr`, `train/tokens_per_sec`, plus any `metrics` dict keys prefixed `train/`.

**Offline training:** Omit `WANDB_PROJECT` — console-only logging, no import error.




## Appendix — Checkpoint File Sizes (422M)

| File | Approximate size |
|---|---|
| `model_step_N.safetensors` | ~850 MB (BF16 weights, deduped head) |
| `optim_step_N.pt` | ~5 GB (FP32 AdamW state) |
| `meta_step_N.json` | < 100 KB |

128 checkpoints @ 4000-step interval ≈ 750 GB total — plan disk space or implement retention (not wired in training loop).

---

## FAQ (Extended)

**Q: Why safetensors for model but pickle for optimizer?**
A: Optimizer state contains Python objects and nested dicts. Safetensors is tensor-only.

**Q: Can I convert checkpoint to HuggingFace format?**
A: Not implemented in this repo. Manual key mapping required (MLA/MoE structure differs).

**Q: Memory estimate wrong on RTX 4090?**
A: `STATIC_PYTORCH_OVERHEAD_GB` tuned for A100. Consumer GPUs use `min(13.7, 0.17*total)` cap.



## CheckpointManager — Deep Dive

Three-file checkpoint triplet per step (see [training.md](training.md) §Checkpoint Format).

### Deduplication on save

```python
for k, v in state.items():
    ptr = v.data_ptr()
    if ptr in seen_ptrs: continue  # weight tying duplicate
```

Prevents writing `embed.weight` twice when `head.weight` shares storage.

### `latest_step()` completeness check

A step is "complete" only if **all three** files exist:

- `model_step_N.safetensors`
- `optim_step_N.pt`
- `meta_step_N.json`

Partial writes from killed jobs are ignored — training resumes from last complete triplet.

### Strict vs loose load

Training uses `strict=False` on trunk load because tied `head.weight` key is intentionally omitted from safetensors. Missing **unexpected** keys in MTP load are logged, not fatal.

---

## `estimate_model_memory_gb` — Formula

From `utils/memory.py` (PaLM-style heuristic):

$$
\text{GB} \approx \frac{N_{params} \times \text{bytes/param}}{10^9}
$$

Training multiplier ~24× params accounts for optimizer state (2×), gradients, activations (checkpointed), and fragmentation. Use before launching long runs on unfamiliar GPUs.

---

## TrainingLogger — Metric Definitions

Logged each `log_every` micro-steps:

| Metric | Definition |
|---|---|
| `loss` / `ce_loss` | Main next-token cross-entropy (detached) |
| `mtp_loss` | Auxiliary MTP CE when enabled |
| `balance_loss` | Sum of MoE load-balance **metrics** (not in loss) |
| `ppl` | `exp(ce_loss)` |
| `lr` | Current scheduler LR |
| `tps` | Tokens/sec since last log window |

**Perf note:** `.item()` called once per log interval — not per micro-step.

---

## WandB Integration (Optional)

When `WANDB_PROJECT` is set, `init_logging` may attach a WandB backend (see `utils/logging.py`). Hyperparameters and scalars mirror stdout metrics. Not required for local training.

---

## Recovery Playbook

| Scenario | Action |
|---|---|
| Corrupt safetensors | Delete incomplete step; resume from `latest_step()-1` |
| Optim state mismatch | Delete `optim_step_N.pt`; warm-restart optimizer (LR schedule still in meta) |
| OOM mid-save | Retry with smaller `micro_batch_size`; checkpoint every `save_every` |
| Wrong `has_mtp` flag | Re-save from known-good weights or load trunk only |



## References

- `utils/checkpoint.py`, `utils/memory.py`, `utils/logging.py`
- `tests/test_utils.py` — checkpoint and memory tests
- [training.md](training.md) — save/load integration
- `scripts/microbench_a100.py` — empirical validation

## CheckpointManager — Deep Dive

### `_atomic_save_safetensors` deduplication

When `weight_tying: true`, `embed.weight` and `head.weight` share the same storage (`data_ptr()`). The save path deduplicates by pointer:

```python
seen_ptrs = set()
for k, v in state.items():
    ptr = v.data_ptr()
    if ptr in seen_ptrs:
        continue  # skip duplicate tensor
    seen_ptrs.add(ptr)
    deduped[k] = v
```

This prevents writing 768×100018 weights twice (~600 MB saved per checkpoint).

### `_checkpoint_complete` semantics

A step is **complete** only when all three files exist:

```
model_step_N.safetensors
optim_step_N.pt
meta_step_N.json
```

Crash during `optim_step_N.pt` write → `latest_step()` skips step N → resume from N-1.

### MTP load path in Pretrainer

```python
mtp_state = {k.removeprefix("mtp."): v for k, v in state.items() if k.startswith("mtp.")}
mtp_orig.load_state_dict(mtp_state, strict=False)
```

Keys in checkpoint: `mtp.mtp_modules.0.block.norm_h.weight`, etc.

---

## Memory Estimation — Formula Derivation

`estimate_model_memory_gb` sums five terms:

| Term | Formula | 422M approx |
|---|---|---|
| Parameters | $2 \times N_{\text{params}}$ bytes (BF16) | 0.84 GB |
| Optimiser | $12 \times N_{\text{params}}$ bytes (AdamW FP32) | 5.0 GB |
| KV cache | $B \cdot S \cdot L \cdot (R + d_{\text{rope}}) \cdot 2$ | small at train (no cache) |
| Activations | $24 \cdot B \cdot S \cdot d \cdot L \cdot 2$ (PaLM factor) | ~22 GB |
| Overhead | `min(13.7, 0.17 × VRAM)` GB | ~13.7 GB |

**PaLM factor 24:** empirical peak activation multiplier for transformer training with checkpointing.

### `assert_fits_in_available_gpu`

No-op on CPU. On CUDA:

```python
if estimate_gb > total_gb - safety_margin_gb:
    raise RuntimeError(...)
```

`microbench_a100.py` uses `margin=2.0` GB.

---

## TrainingLogger — Metric Definitions

Printed every `log_interval` micro-steps:

```
step=   4000 | loss=2.8541 | ppl=17.28 | lr=7.92e-04 | tps=128,000 | balance_loss=0.0012
```

| Field | Definition |
|---|---|
| `loss` | Rolling mean CE (nats) over last `log_interval` steps |
| `ppl` | $\exp(\text{loss})$ |
| `lr` | Current scheduler LR |
| `tps` | `log_interval × seq_len / elapsed` (approx tokens/sec) |
| `balance_loss` | Sum of MoE load-balance metrics (logging only) |
| `mtp_loss` | Present when MTP enabled |

**WandB:** Set `WANDB_PROJECT` env var; metrics logged as `train/loss`, `train/ppl`, etc.

---

## Recovery Playbook

| Scenario | Action |
|---|---|
| Corrupt checkpoint at step N | Delete incomplete trio; resume from `latest_step()` |
| OOM mid-training | Lower `micro_batch_size`; resume from last complete step |
| Missing `optim_step_N.pt` | Load weights only; optimiser restarts (LR schedule restored from meta) |
| `head.weight` missing on load | Expected with weight tying — use `strict=False` |
| MTP keys missing | `has_mtp: false` in meta or train without speculative decode |

<!-- docs:verified 2026-07-31 · 88cb863 -->
