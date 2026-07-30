# Scripts — Launch, Benchmark, and Smoke Tests

## A Comprehensive Technical Reference

> **Covers**: `scripts/` — operational tooling for GPU validation and production training launch.

---

## Table of Contents

1. [Overview](#overview)
2. [launch_a100.sh](#launch_a100sh)
3. [microbench_a100.py](#microbench_a100py)
4. [step_time_a100.py](#step_time_a100py)
5. [smoke_forward.py](#smoke_forwardpy)
6. [e2e_test_gpu.py](#e2e_test_gpupy)
7. [build_small_pretrain_data.py](#build_small_pretrain_datapy)
8. [check_docs.py](#check_docspy)
9. [Recommended Workflow](#recommended-workflow)
10. [Environment Variables](#environment-variables)

---

## Overview

Scripts are **observability instruments** — they answer three questions before you commit GPU-days:

1. **Does it build?** (`smoke_forward.py`) — graph construction, shape correctness
2. **Does it fit?** (`microbench_a100.py`) — VRAM budget vs hardware
3. **Is it fast enough?** (`step_time_a100.py`) — MFU, tokens/sec

Each script mirrors a phase of the [getting_started.md](getting_started.md) mental model (CPU tests → GPU smoke → production).

| Script | Purpose | Requires GPU |
|---|---|---|
| `launch_a100.sh` | Production training launch (nohup) | Yes (A100) |
| `microbench_a100.py` | VRAM estimate vs measured peak | Yes |
| `step_time_a100.py` | ms/step, tokens/sec, MFU | Yes |
| `smoke_forward.py` | Single forward+backward sanity | Yes |
| `e2e_test_gpu.py` | End-to-end GPU test suite | Yes |
| `build_small_pretrain_data.py` | Tiny dataset for local dev | No |
| `check_docs.py` | Lint `documentation/` links, paths, stale patterns | No |

---

## check_docs.py

Validates documentation quality (also runs in CI):

```bash
python scripts/check_docs.py              # lint only
python scripts/check_docs.py --update-sizes --stamp-footers   # refresh README line counts + verification stamps
```

Checks: control characters, stale math/status patterns, internal `.md` links, and backtick-quoted repo paths. Generated data dirs and planned Triton files are allowlisted.

---

## launch_a100.sh

**Purpose:** Pre-flight checks + background training launch.

### Pre-flight

1. CUDA available, VRAM ≥ 75 GB
2. `data/pretrain_chinchilla/shard_*.bin` exists
3. Creates `checkpoints/pretrain_a100/`

### Environment

```bash
export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT=deepseek-v3-lite-a100
export TORCH_COMPILE_MODE=max-autotune
export TOKENIZERS_PARALLELISM=false
```

### Launch

```bash
nohup python -u training/pretrain.py \
  --config configs/pretrain_a100_422m.yaml \
  --data-path data/pretrain_chinchilla \
  --checkpoint-dir checkpoints/pretrain_a100 \
  > checkpoints/pretrain_a100/train.log 2>&1 &
```

### Monitor

```bash
tail -f checkpoints/pretrain_a100/train.log
nvidia-smi
```

### Resume

```bash
python training/pretrain.py --config configs/pretrain_a100_422m.yaml --resume 4000
```

---

## microbench_a100.py

**Purpose:** Validate memory budget before committing to a 15-hour run.

```bash
python scripts/microbench_a100.py
```

**What it does:**

1. Builds 422M `Transformer` with grad_checkpoint=True
2. Prints analytical estimate from `estimate_model_memory_gb`
3. Runs forward + backward on random data
4. Reports `torch.cuda.max_memory_allocated()`
5. Calls `assert_fits_in_available_gpu(estimate, margin=2.0)`

**Expected output (A100 80GB):**

```
estimated peak   = ~30.5 GB
measured peak    = ~33-35 GB
```

If measured > total - 8 GB → warning to reduce batch or seq len.

---

## step_time_a100.py

### MFU — what it measures

**Model FLOPs Utilisation (MFU)** compares achieved throughput to peak hardware FLOPs:

$$
\mathrm{MFU} = \frac{\text{achieved FLOPs/sec}}{\text{peak BF16 Tensor Core FLOPs/sec}}
$$

For 422M on A100 (312 TFLOPS peak BF16), MFU 35% ≈ 109 TFLOPS sustained. Low MFU usually means:
- Memory-bound MoE dispatch (stacked path copies weights)
- Activation recomputation from grad checkpointing
- Small batch under-utilising SMs

**Purpose:** Measure training throughput and Model FLOPs Utilisation (MFU).

```bash
python scripts/step_time_a100.py --steps 20 --warmup 5
```

**Metrics:**

- ms/step (median)
- tokens/sec = `batch × seq / ms`
- MFU = `actual_flops / (peak_tflops × time)`

**Target:** 35–40% MFU on A100 80GB for the 422M config.

**Flags:**

| Flag | Default | Effect |
|---|---|---|
| `--steps` | 20 | Timed iterations |
| `--warmup` | 5 | Warmup before timing |
| `--no-compile` | off | Disable torch.compile |
| `--compile-mode` | max-autotune | Compile mode |
| `--peak-tflops` | 312 | A100 BF16 peak |

---

## smoke_forward.py

Quick sanity: build model, one forward pass, print output shape. Used for CI GPU runners and manual smoke.

---

## e2e_test_gpu.py

End-to-end GPU validation: model construction, forward, backward, optional compile. Broader than `smoke_forward.py`.

---

## build_small_pretrain_data.py

Creates a tiny token file for local development without running the full 8B pipeline. Useful for debugging `PretrainDataset` and training loop on laptop.

---

## Recommended Workflow

```
1. pytest tests/ -q                    # CPU correctness
2. python scripts/smoke_forward.py     # GPU builds
3. python scripts/microbench_a100.py   # VRAM headroom
4. python scripts/step_time_a100.py    # throughput / MFU
5. python3 data/prepare_data.py ...    # full data (once)
6. bash scripts/launch_a100.sh         # production run
```

---

## Environment Variables

| Variable | Used by | Purpose |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | All GPU scripts | GPU selection |
| `TORCH_COMPILE_MODE` | pretrain, step_time | Compile mode |
| `ENABLE_TRITON_KERNELS` | Model init | Allow Triton paths |
| `WANDB_PROJECT` | launch, logging | Experiment tracking |
| `WANDB_RUN_NAME` | launch, logging | Run name |

---



---

## launch_a100.sh — Deep Dive

### Shell safety

```bash
set -euo pipefail
```

- `-e`: exit on first command failure
- `-u`: error on unset variables
- `-o pipefail`: pipeline fails if any stage fails

### Pre-flight Python check

The embedded `python -c` block verifies:
1. `torch.cuda.is_available()`
2. VRAM ≥ 75 GB (allows headroom below 80 GB nominal)
3. Prints GPU name, PyTorch version, CUDA version

**Why 75 GB threshold?** Some A100 instances report ~79 GB usable; 75 GB catches misconfigured VMs while allowing real A100s.

### Data directory validation

```bash
[[ ! -d "$DATA_DIR" ]] || [[ -z "$(ls -A "$DATA_DIR"/shard_*.bin 2>/dev/null)" ]]
```

Requires at least one `shard_*.bin` — empty directory fails fast with actionable error pointing to `data/prepare_data.py`.

### Background launch pattern

```bash
nohup python -u training/pretrain.py ... > "$LOG_FILE" 2>&1 &
```

| Flag | Purpose |
|---|---|
| `nohup` | Survives terminal disconnect |
| `python -u` | Unbuffered stdout (log appears immediately) |
| `2>&1` | Merge stderr into log |
| `&` | Background process |

**PID tracking:** Script prints PID for `ps -p $PID` monitoring.

### Resume (manual)

`launch_a100.sh` does not auto-resume. Use:

```bash
python training/pretrain.py --config configs/pretrain_a100_422m.yaml --resume 4000
```

`Pretrainer.train()` also auto-resumes from `latest_step()` if checkpoints exist and `--resume` is omitted.

---

## microbench_a100.py — Deep Dive

Full source: `scripts/microbench_a100.py` (45 lines).

### Execution flow

1. Load `configs/pretrain_a100_422m.yaml`
2. Build `Transformer(cfg, use_checkpoint=True).cuda()`
3. Print analytical estimate via `estimate_model_memory_gb`
4. `assert_fits_in_available_gpu(est, margin=2.0)` — raises if estimate > VRAM - 2GB
5. Random forward + backward on `(bs, seq)` tensor
6. Compare `torch.cuda.max_memory_allocated()` vs estimate

### Interpreting delta

| delta vs estimate | Action |
|---|---|
| < 20% | Normal — allocator fragmentation, compile workspace |
| 20-35% | Acceptable — monitor during long run |
| > 35% | Investigate — wrong batch size, leak, or estimate bug |

### Warning thresholds

- `measured > total - 8 GB`: **WARNING** — reduce `micro_batch_size` or `max_seq_len`
- `measured > 70% total`: **NOTICE** — comfortable headroom

**Does not test:** MTP wrapper, `torch.compile`, or full `Pretrainer` — only bare Transformer. Real training uses ~5-10% more VRAM (optimizer state, MTP, compile).

---

## step_time_a100.py — Deep Dive

### FLOP estimate

```python
flops = 6 * n_nonembed * seq * bs
```

Classic Chinchilla approximation: 6× params × tokens per step (forward + backward through matmul layers).

### MFU calculation

```python
tflops_per_s = flops / dt / 1e12
mfu = tflops_per_s / args.peak_tflops * 100
```

Default `peak_tflops=312` — A100 SXM BF16 Tensor Core nominal peak.

### MFU interpretation guide

| MFU | Diagnosis |
|---|---|
| < 25% | MoE Python overhead, compile disabled, or TF32 off |
| 25-35% | Normal for stacked MoE dispatch on A100 |
| 35-45% | Target range for this codebase |
| > 50% | Unusual — verify FLOP count or check for measurement bug |

### What is NOT measured

- DataLoader overhead (synthetic random tokens on GPU)
- MTP auxiliary loss path
- MoE bias updates
- Checkpoint I/O

Add ~10-15% wall time in real training for these.

### Flags reference

```bash
python scripts/step_time_a100.py \
  --steps 50 --warmup 10 \
  --no-compile \
  --compile-mode reduce-overhead \
  --peak-tflops 312
```

---

## smoke_forward.py

Minimal GPU sanity: constructs model, runs one forward, prints output shape. Used when you only need to verify CUDA + model construction without backward or memory profiling.

**When to use:** After changing `Transformer.__init__` or config schema — faster than full pytest GPU suite.

---

## e2e_test_gpu.py

Broader than smoke_forward:
- Forward + backward
- Optional `torch.compile` path
- Gradient norm check

Run before first production launch if you modified training loop or compile settings.

---

## build_small_pretrain_data.py

Creates a tiny token file for local `PretrainDataset` debugging without 8B-token pipeline.

**Use case:** Laptop debugging of training loop, NaN guard, checkpoint roundtrip — not for quality evaluation.

---

## Troubleshooting Guide

| Symptom | Script to run | Likely fix |
|---|---|---|
| CUDA OOM at step 0 | `microbench_a100.py` | Halve `micro_batch_size` |
| MFU < 20% | `step_time_a100.py --no-compile` vs default | Enable compile; check TF32 |
| Model won't construct | `smoke_forward.py` | Config/schema mismatch |
| Training dies after 1h | Check `train.log` | NaN guard rollback — reduce LR |
| Triton silently disabled | `echo $ENABLE_TRITON_KERNELS` | Set env var or use stacked |

---

## Appendix — Wall Clock Budget (422M)

| Phase | Duration |
|---|---|
| Data prep (once) | Hours-days (bandwidth bound) |
| microbench + step_time | ~5 min |
| Training (512K micro-steps) | 13-15 h on A100 |
| Checkpoint every 4000 steps | ~128 saves |

**Note:** `total_steps` in YAML counts **micro-steps** (each DataLoader batch). Optimizer steps = micro-steps / grad_accum = 128,000. At ~0.4s per optim step (post-compile), wall time matches 13-15h.



## Appendix — Script Source Map

| File | Lines | Entry point |
|---|---|---|
| `launch_a100.sh` | 91 | `bash scripts/launch_a100.sh` |
| `microbench_a100.py` | 45 | `python scripts/microbench_a100.py` |
| `step_time_a100.py` | 79 | `python scripts/step_time_a100.py` |
| `smoke_forward.py` | ~30 | `python scripts/smoke_forward.py` |
| `e2e_test_gpu.py` | varies | `python scripts/e2e_test_gpu.py` |
| `build_small_pretrain_data.py` | varies | `python scripts/build_small_pretrain_data.py` |

All Python scripts insert project root into `sys.path` for import portability.

---

## FAQ

**Q: Can I run microbench on RTX 4090?**
A: Yes — adjust expectations. 24GB VRAM will OOM at full 422M batch; reduce batch in YAML first.

**Q: step_time without compile is much slower — is that normal?**
A: Yes. `torch.compile` is load-bearing for 35%+ MFU on this codebase.

**Q: launch_a100.sh on multi-GPU machine?**
A: Set `CUDA_VISIBLE_DEVICES=0` (default). Multi-GPU training is not implemented.


## References

- [training.md](training.md) — what launch_a100 runs
- [utils.md](utils.md) — memory estimation formulas
- [getting_started.md](getting_started.md) — quick start path

## smoke_forward.py — Deep Dive

Builds the 422M `Transformer` from YAML, runs one forward pass on random `input_ids`, prints output shape.

**Use when:** Verifying CUDA + PyTorch install before data prep or training.

**Does not test:** MTP wrapper, compile, DataLoader, checkpoint I/O.

---

## e2e_test_gpu.py — Deep Dive

Broader GPU validation than `smoke_forward.py`:

- Forward + backward
- Optional `torch.compile` smoke
- Gradient norm sanity

Run before first production launch if you modified `training/pretrain.py`.

---

## build_small_pretrain_data.py

Creates a tiny uint32 token file for local `PretrainDataset` debugging without 8B download.

```bash
python scripts/build_small_pretrain_data.py
python training/pretrain.py --config configs/pretrain_1650_2m.yaml --data-path <output>
```

---

## Wall-Clock Budget — 422M Run

| Phase | Duration | Notes |
|---|---|---|
| Data prep (first time) | 4–24 h | Bandwidth + CPU bound |
| `torch.compile` warmup | 5–15 min | First steps slow |
| Training 512k steps | 13–15 h | @ 270–320 ms/step |
| Checkpoint saves | ~2 min each | Every 4000 steps |

**Disk:** 128 checkpoints × ~6 GB ≈ 750 GB if no retention — plan storage or prune old steps manually.

---

## Troubleshooting Scripts

| Script output | Meaning | Action |
|---|---|---|
| microbench WARNING > 72 GB | OOM risk | Lower `micro_batch_size` |
| step_time MFU < 20% | Memory bound | Try Triton MoE (with env var) |
| smoke_forward shape mismatch | Config / code drift | Run `pytest tests/test_models.py` |
| launch_a100 no shards | Data not prepared | `python3 data/prepare_data.py --stage pretrain` |

<!-- docs:verified 2026-07-31 · 88cb863 -->
