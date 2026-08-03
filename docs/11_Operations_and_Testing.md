# 11 — Operations, Verification & VRAM Budgeting

> **Canonical** for DeepSeek-v3-Lite's real test suite, atomic safetensors checkpointing, and memory budgeting. Educational textbook chapter.

> System verification, the actual pytest suite, atomic checkpoint mechanics, and the VRAM accounting that lets ~422 M params train on one A100 80 GB. **Status:** architecture is implemented and CPU smoke-tested; **no full GPU training run has been executed yet** — `checkpoints/` is empty, so every number below is a budget/estimate, not a measurement.

**Depends on:** [[Docs/08_Training_Pipeline]] · **Read next:** [[Docs/12_Triton_Kernels]]

---

## 1. The Test Suite — what actually exists

The repo ships **7 test files / 189 test functions**, all CPU-only, all running on small configs from `tests/conftest.py`. No CUDA, no Triton, no network — the suite is green on a laptop. This is deliberate: the project's design goal is *CPU-testability* (see [[Docs/02_Model_Architecture]] §Design Goals), so every correctness invariant has a test that compares a fast path against an explicit reference without leaving the CPU.

```bash
# Full suite (< 2 s on a modern laptop, no GPU required)
python3 -m pytest tests/ -v

# CI runs the same thing (see .github/workflows/ci.yml):
python -m pytest tests/ -q --tb=short
```

### 1.1 The seven files

| File | Tests | What it verifies |
|---|---|---|
| `tests/test_models.py` | 81 | `Transformer` / `MLA` / `SwiGLUFFN` / `Expert` / `DeepSeekMoE` / `AuxLossFreeGate` / `TransformerBlock` / `MTPBlock` / `MTPModule` / `MultiTokenPrediction` / `generate` / `count_parameters` — shapes, weight tying, residual streams, gradient flow through MTP |
| `tests/test_training.py` | 40 | `TrainingConfig`, warmup-cosine scheduler, `PretrainDataset` (single + sharded, cross-shard `_locate`), `Pretrainer` construction, checkpoint round-trip, `train_step`, MoE balance metric, YAML parsing, scheduler boundaries, μP LR boundary, NaN-guard rollback |
| `tests/test_utils.py` | 26 | `CheckpointManager` save/load (incl. MTP keys, shared-tensor dedup), `MemoryEstimation` (params/optim/KV/activation byte arithmetic) |
| `tests/test_moe_triton.py` | 16 | `grouped_moe_pytorch` reference, Triton import surface, dispatch wiring, and the GPU kernel (skipped without CUDA) |
| `tests/test_inference.py` | 13 | `Transformer.generate` (temperature/top-k/top-p/eos/`max_seq_len`), `SpeculativeDecoder` accept/reject, inference helpers |
| `tests/test_force_back.py` | 8 | `enforce_triton_env_var` — every Triton dispatch key is force-backed to its PyTorch default unless `ENABLE_TRITON_KERNELS=1` |
| `tests/test_mla_triton.py` | 5 | `mla_attention_reference` pure-PyTorch path, Triton import, and the GPU kernel (skipped without CUDA) |

### 1.2 The fixture layer (`tests/conftest.py`)

Tests do not build the 422 M model — they use two miniature configs that preserve every invariant at toy scale:

- **`cfg`** — an "82 M"-shape config (dim=640, 2 layers, 10 heads, 8 routed/2 active experts, `kv_lora_rank=128`). Used by training/checkpoint tests that need realistic module diversity.
- **`small_cfg`** — a 64-dim, 2-layer, 4-expert config. Used by component tests that need speed above fidelity.
- **`training_cfg`** — a full YAML-shaped dict (model + training + data sections) parsed by the same code path as `main()`, pointing at `/tmp` paths.
- **`tokens` / `targets`** — random token IDs in vocab range; `tmp_ckpt_dir` / `tmp_data_file` / `tmp_shard_dir` — ephemeral on-disk fixtures cleaned up post-test.

`★ Insight ─────────────────────────────────────`
The fixture configs are the reason the suite is fast and portable: they exercise the **real** `Transformer`/`Pretrainer`/`CheckpointManager` code, just at a scale where a forward pass is microseconds. A bug in MLA's matrix absorption or in sharded-dataset `_locate` will fire on `small_cfg` exactly as it would on the 422 M config — the shapes are smaller, the control flow is identical. This is why "CPU-testable" is a design goal, not a convenience.
`─────────────────────────────────────────────────`

### 1.3 What the suite guards (the invariants with teeth)

| Invariant | Test anchor |
|---|---|
| MLA KV compression matches uncompressed attention | `TestMLA*` (reference vs. SDPA path) |
| Aux-loss-free bias update changes routing without touching the loss | `TestAuxLossFreeGate`, `TestMoEBalanceMetric` |
| MTP depth-1 produces a length-aligned `(logits, targets)` pair and a combined loss | `TestMultiTokenPrediction`, `TestMTPGradientFlow` |
| Weight tying: `head.weight IS embed.weight` (one tensor, counted once) | `TestCountParameters`, `TestEmbedding` |
| Atomic checkpoint round-trip restores model + optim + scheduler + `opt_steps` | `TestCheckpointRoundtrip`, `TestCheckpointOptStepsRoundtrip` |
| NaN guard rolls back to the last complete checkpoint after N consecutive NaNs | `TestNanGuardRollback` |
| Triton paths never silently activate without `ENABLE_TRITON_KERNELS=1` | `TestEnforceTritonEnvVar` (8 cases) |
| Sharded dataset spans shard boundaries correctly | `TestPretrainDataset._locate` |

---

## 2. Atomic Checkpoint System

Checkpoints are managed by `utils/checkpoint.py:CheckpointManager`. Three files per step, all written **atomically**, only ever read back if **all three** are present.

### 2.1 The three-file layout

```
checkpoints/pretrain_a100/
├── model_step_004000.safetensors   # weights  (safetensors, deduped shared tensors)
├── optim_step_004000.pt             # AdamW state (FP32 master + m + v)  — torch.save
└── meta_step_004000.json            # step, scheduler state_dict, opt_steps, config, has_mtp
```

> [!NOTE] `optim_step_N.pt` uses `torch.save` (not pickle, not safetensors) — optimizer state contains tensors in nested Python dicts that safetensors cannot serialize directly. Weights are always safetensors; this split follows the workspace rule "no `pickle` checkpoints — `torch.save` or `safetensors`."

### 2.2 Atomicity — the `tempfile` → `os.replace` dance

Every file is written to a `tempfile.mkstemp` temp path in the same directory, fully written, then `os.replace`'d onto the final name. `os.replace` is POSIX-atomic on the same filesystem, so a crash mid-write leaves either the old complete file or nothing — never a half-written file.

```python
@contextlib.contextmanager
def _atomic_write(self, path, suffix):
    fd, tmp = tempfile.mkstemp(dir=self.save_dir, suffix=suffix)
    os.close(fd)
    try:
        yield tmp
        os.replace(tmp, path)      # atomic rename
    except Exception:
        os.unlink(tmp); raise      # clean up the temp on failure
```

`★ Insight ─────────────────────────────────────`
Writing the temp file in the *same directory* as the target is load-bearing: `os.replace` is only atomic within a single filesystem mount. A naive `tempfile.NamedTemporaryFile()` defaults to `/tmp`, which on many systems is a separate tmpfs mount — renaming across mounts degrades to a copy+delete (non-atomic). Forcing `dir=self.save_dir` keeps the rename on one mount and the atomicity guarantee holds.
`─────────────────────────────────────────────────`

### 2.3 Shared-tensor dedup on save

With `weight_tying=true`, `head.weight` and `embed.weight` are the **same tensor**. `CheckpointManager._atomic_save_safetensors` dedups by `data_ptr()` — the first occurrence is stored; duplicates are `.contiguous().clone()`-d only if they share a pointer (safetensors rejects duplicate storage). On load, `load_state_dict(strict=False)` leaves `head.weight` missing, but the shared tensor is restored via `embed`, and tying is re-established in `Transformer.__init__` before load.

### 2.4 "Complete checkpoint" — the only safe thing to resume from

```python
def _checkpoint_complete(self, step):
    return all((self.save_dir / n).exists() for n in [
        f"model_step_{step}.safetensors", f"optim_step_{step}.pt", f"meta_step_{step}.json"])
```

`latest_step()` returns the highest step whose **all three** files exist. A step with a missing `meta_step_N.json` (e.g. crash after writing `model` + `optim` but before `meta`) is invisible to resume — the NaN guard and the `--resume` path will never pick it. This is the invariant that makes crash recovery safe.

### 2.5 What gets saved (the full RNG/optim/scheduler state)

Per the workspace atomic-checkpoint rule, a checkpoint carries the **full** state needed for bit-identical resume:

- **Weights** (deduped) → safetensors.
- **Optimizer state** (`AdamW.state_dict()` = FP32 master + first moment `m` + second moment `v`, per param group) → `.pt`.
- **Scheduler** (`LambdaLR.state_dict()`) + **`opt_steps`** (the true optimizer-step counter, distinct from the micro-step `global_step`) + **config** + **`has_mtp`** flag → `meta.json`.
- **MTP module weights** are prefixed `mtp.` and merged into the same safetensors file (see `Pretrainer.save_checkpoint`).

> [!WARNING] Resume does **not** restore the `DataLoader` sampler RNG, so token order differs across a restart. This is benign at Chinchilla scale (samples are seen ~uniformly); exact resume would require checkpointing the sampler generator. This is documented in `training/pretrain.py:Pretrainer.train`.

---

## 3. VRAM Memory Budget (1 × A100 80 GB)

`utils/memory.py` is a CPU-arithmetic estimator (no GPU needed) used by `scripts/microbench_a100.py`. It sums four components plus an autodetected PyTorch overhead. All numbers below are computed for the canonical 422 M config (`batch=8`, `seq=2048`, 18 layers, BF16).

### 3.1 The four-component formula

| Component | Formula | Bytes/param | 422 M value |
|---|---|---|---|
| **Parameters** (BF16) | `N × 2` | 2 | `422 M × 2 = 844 MB ≈ 0.84 GB` |
| **AdamW state** (FP32 master + m + v) | `N × 12` | 12 | `422 M × 12 = 5.06 GB` |
| **MLA KV cache** (per layer) | `B × S × (d_c + d_R) × 2` × L | — | `8 × 2048 × (192+24) × 2 × 18 = 127 MB ≈ 0.13 GB` |
| **Activations** (grad-ckpt, PaLM formula) | `24 × B × S × D × L × 2` | — | `24 × 8 × 2048 × 768 × 18 × 2 = 10.1 GB` |

`★ Insight ─────────────────────────────────────`
The KV cache line is where MLA earns its keep. A **standard MHA** at this config would cache `B × S × n_heads × (d_nope + d_rope + d_v) × L × 2 = 8 × 2048 × 12 × (48+24+64) × 18 × 2 ≈ 6.0 GB`. MLA caches only the **compressed latent** `(d_c + d_R) = 216` floats/token/layer regardless of head count — **~0.13 GB, a ~46× reduction.** This is why the old "0.75 GB" figure in earlier notes was wrong: it used uncompressed head dimensions, not the latent. The reduction is the entire point of MLA (see [[Docs/03_Multi_Head_Latent_Attention]]).
`─────────────────────────────────────────────────`

The 24× (with grad-ckpt) / 36× (without) activation constants are the PaLM Appendix-A approximation. Gradient checkpointing drops the factor to 24 by recomputing layer activations in the backward pass — trading ~30% compute for the memory headroom that lets micro-batch 8 fit.

### 3.2 Overhead and the headline

`_detect_overhead_gb()` returns the PyTorch + CUDA context + caching-allocator overhead: `min(13.7 GB, 0.17 × total_device_GB)` on CUDA (empirically ≤17% of device total on A100 80 GB), or 2 GB on CPU. This is a *bound*, not a measurement.

| | GB |
|---|---|
| Parameters | 0.84 |
| AdamW optimizer | 5.06 |
| MLA KV cache | 0.13 |
| Activations (grad-ckpt) | 10.1 |
| **Component subtotal** | **~16.1** |
| PyTorch overhead (env-dependent) | ~2 – 13.6 |
| **Estimated peak** | **~18 – 30 GB** |

Either way, well within the 80 GB A100 — leaving room to raise micro-batch or disable grad checkpointing if throughput demands it. `estimate_model_memory_gb(..., inference=True)` drops the optimiser (12×N) and activation terms, since inference carries no AdamW state and forward activations are dominated by the KV cache.

### 3.3 The guard rail

```python
def assert_fits_in_available_gpu(estimate_gb, safety_margin_gb=0.0):
    # No-op on CPU. On CUDA, raises RuntimeError if estimate > available - margin.
```

Wired into the benchmark scripts so a misconfigured batch size fails loud and early rather than OOMing 20 minutes into a run.

---

## 4. Operational Status

| Capability | State |
|---|---|
| Architecture (MLA / MoE / MTP / aux-loss-free / μP / NaN guard) | ✅ implemented |
| CPU smoke test suite (189 tests) | ✅ passing |
| Atomic safetensors checkpointing | ✅ implemented |
| VRAM estimator | ✅ implemented |
| Fused Triton kernels (MLA attention, grouped MoE GEMM) | ✅ implemented, opt-in, BF16 |
| CI (`.github/workflows/ci.yml`) | ✅ runs `pytest tests/ -q` |
| Full GPU pre-training run (512 000 steps) | ⏳ **not yet executed** (`checkpoints/` empty) |
| FP8 mixed precision | ❌ paper-spec only — see [[Docs/06_FP8_Mixed_Precision]] |
| DualPipe pipeline parallelism | ❌ paper-spec only — see [[Docs/07_DualPipe_Parallelism]] |

> **Next:** [[Docs/12_Triton_Kernels]] — the real fused kernels: MLA attention and grouped-GEMM MoE dispatch (BF16, not FP8).