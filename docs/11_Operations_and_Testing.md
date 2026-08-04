# 11 — Operations, Verification & VRAM Budgeting

> **Canonical** for DeepSeek-v3-Lite's real test suite, atomic safetensors checkpointing, and memory budgeting. Educational textbook chapter.

> System verification, the actual pytest suite, atomic checkpoint mechanics, and the VRAM accounting that lets ~412 M params (411.6M deduped base / 418.7M with MTP) train on one A100 80 GB. **Status:** architecture is implemented and CPU smoke-tested; **no full GPU training run has been executed yet** — `checkpoints/` is empty, so every memory/latency figure below is a budget/estimate, not a measurement.

**Depends on:** [[Docs/08_Training_Pipeline]] · **Read next:** [[Docs/12_Triton_Kernels]]

---

## 1. 60-Second Summary

This chapter is the operations manual for DeepSeek-v3-Lite: how the repo proves it works, how it saves its state, how much memory it needs, and what to do when something breaks. The repo's verification strategy is deliberately CPU-first — a full pytest suite (~199 test nodes, all runnable without a GPU) exercises the real `Transformer`/`Pretrainer`/`CheckpointManager` code at toy scale, so correctness invariants fail loudly on a laptop instead of after GPU-days. Checkpoints are three files per step (`safetensors` weights, `torch.save` optimizer state, JSON metadata), each written atomically, only ever resumed from when all three exist. A pure-arithmetic estimator (`utils/memory.py`) puts the canonical 411.6M-parameter run at ~15.6 GiB of components plus ~13.6 GiB of PyTorch overhead — comfortably inside one A100 80 GB — and every one of these claims is machine-checked by a CI pipeline that also verifies the docs' code anchors resolve.

## 2. Why Verification Is a Design Goal

The failure mode this design answers is the classic research-repo trap: a model "works" because somebody ran it once on a GPU, and nothing guards the code against the next edit. In this repo the stakes are higher than usual because the architecture is load-bearing and unusual — MLA's low-rank KV compression, the aux-loss-free MoE gate with its learned bias, depth-1 MTP with a shared head, weight tying, μP-scaled learning rates. Any one of those subsystems can be subtly wrong (see §8 for the recipes), and a wrong-but-non-crashing implementation is worse than a crash: it trains fine and produces silently bad models.

The design goal is therefore *CPU-testability* (see [[Docs/02_Model_Architecture]] §Design Goals): every correctness invariant has a test that compares a fast or optimized path against an explicit reference implementation, and every test runs on a small config in milliseconds. The suite is the proof that the architecture matches its spec; the doc↔code alignment gate (§7) is the proof that the documentation matches the code. Neither requires a GPU, so both run in CI on every push.

## 3. The Test Suite — what actually exists

The repo ships **8 test files / 199 test nodes** at the time of writing (measured with `pytest --collect-only`; the count grows as tests are added — treat "the pytest suite" as the source of truth). All CPU-runnable, all running on small configs from `tests/conftest.py`. On a laptop without CUDA/Triton, 10 nodes skip (9 require CUDA+Triton for the fused kernels, 1 requires Triton installed) and the rest pass. No CUDA, no Triton, no network — the suite is green on a laptop.

```bash
# Full suite (< 20 s on a laptop, no GPU required)
python3 -m pytest tests/ -v

# CI runs the same thing (see §6):
python -m pytest tests/ -q --tb=short
```

### 3.1 The eight files

| File | Tests | What it verifies |
|---|---|---|
| `tests/test_models.py` | 83 | `Transformer` / `MLA` / `SwiGLUFFN` / `Expert` / `DeepSeekMoE` / `AuxLossFreeGate` / `TransformerBlock` / `MTPBlock` / `MTPModule` / `MultiTokenPrediction` / `generate` / `count_parameters` — shapes, weight tying, residual streams, chunked-prefill parity, gradient flow through MTP |
| `tests/test_training.py` | 41 | `TrainingConfig`, warmup-cosine scheduler, `PretrainDataset` (single + sharded, cross-shard `_locate`), `Pretrainer` construction, checkpoint round-trip, `train_step`, MoE balance metric, YAML parsing, scheduler boundaries, μP LR boundary, NaN-guard rollback |
| `tests/test_utils.py` | 27 | `CheckpointManager` save/load (incl. MTP keys, shared-tensor dedup, crash recovery), `MemoryEstimation` (params/optim/KV/activation byte arithmetic) |
| `tests/test_moe_triton.py` | 16 | `grouped_moe_pytorch` reference, Triton import surface, dispatch wiring, and the GPU kernel (skipped without CUDA) |
| `tests/test_inference.py` | 14 | `Transformer.generate` (temperature/top-k/top-p/eos/`max_seq_len`), `SpeculativeDecoder` accept/reject, inference helpers |
| `tests/test_force_back.py` | 8 | `enforce_triton_env_var` — every Triton dispatch key is force-backed to its PyTorch default unless `ENABLE_TRITON_KERNELS=1` |
| `tests/test_mla_triton.py` | 9 | `mla_attention_reference` pure-PyTorch path (incl. the causal `q_start` contract), Triton import, and the GPU kernel (skipped without CUDA) |
| `tests/test_doc_refs.py` | 1 | The doc↔code alignment gate — every `<path>.py:Symbol` anchor in `docs/` resolves; line anchors and JIT-symbol cites are banned (§7) |

The per-file detail matters because the files guard different layers of the stack:

- **`tests/test_models.py`** is the architectural conformance suite. `TestTransformer` checks forward shapes (`tests/test_models.py:TestTransformer.test_forward_shape`), the `(logits, hidden)` contract of `forward_with_hidden`, KV-cache population and `reset_cache`, dense-vs-MoE layer ordering, and — the sharpest test in the file — `test_chunked_prefill_matches_full_forward`, which asserts that a cached mid-sequence prefill (`start_pos=8`) produces bit-close logits to a full uncached forward. `TestEmbedding` asserts `head.weight IS embed.weight` (same storage). `TestMLA` / `TestDeepSeekMoE` / `TestMTPBlock` cover the components; `TestCountParameters` checks the deduped count; `TestMTPGradientFlow` proves MTP loss backprops into main-model weights.
- **`tests/test_training.py`** guards the loop: `TestPretrainDataset` covers single-file and sharded layouts including the cross-shard window (`test_sharded_cross_boundary`) and the `_locate` boundary/out-of-range contract; `TestWarmupCosineScheduler` and `TestSchedulerBoundary` pin the LR schedule; `TestCheckpointRoundtrip` proves save→load restores weights (with and without MTP); `TestCheckpointOptStepsRoundtrip` pins the `opt_steps` counter in metadata; `TestNanGuardRollback` simulates NaN steps and asserts rollback; `TestMuPLRBoundary` pins the μP scaling law.
- **`tests/test_utils.py`** guards the infrastructure: `TestCheckpointManagerSaveLoad` (weights round-trip), `TestCheckpointManagerMTP` (MTP-prefixed keys land in safetensors), `TestMemoryEstimation` (byte-arithmetic of each budget component), and `TestCheckpointManagerAdditional.test_atomic_save_crash_recovery`, which patches `save_file` to raise "disk full" mid-save and asserts **no** `.tmp` or half-written `.safetensors` survives (§4.2).
- **`tests/test_moe_triton.py`** and **`tests/test_mla_triton.py`** protect the Triton paths: a pure-PyTorch reference (`grouped_moe_pytorch`, `mla_attention_reference`) that must match the naive loop, an import surface that fails cleanly without Triton, dispatch wiring, and — inside the `@gpu_required` classes — kernel-vs-reference numerics, gradcheck, and the MLA `q_start` causal contract. `TestMoeTritonDispatchWiring` also pins the fallback: a `triton_grouped` config degrades to `stacked` cleanly on CPU.
- **`tests/test_force_back.py`** guards the master switch: `TestEnforceTritonEnvVar` asserts `attn_impl='triton'` → `'sdpa'` and `moe_dispatch='triton_grouped'` → `'stacked'` whenever `ENABLE_TRITON_KERNELS` is not `1`, with a single construction-time warning — never per-layer.
- **`tests/test_inference.py`** guards the serving path: `TestModelGenerate` (sampling params and `eos` termination), `TestSpeculativeDecoder` (accept/reject behavior of `inference/speculative.py:SpeculativeDecoder.generate_step`), `TestInferenceHelpers` (`load_config` YAML parsing).
- **`tests/test_doc_refs.py`** is not a unit test of the model at all — it is the machine-readable contract between prose and code (§7).

### 3.2 The fixture layer (`tests/conftest.py`)

Tests do not build the 412 M model — they use two miniature configs that preserve every invariant at toy scale:

- **`cfg`** — a "82M"-shape config: `dim=640`, `n_layers=2` (1 dense + 1 MoE), `n_heads=10`, `n_routed_experts=8` / `n_activated_experts=2` / `n_shared_experts=1`, `inter_dim=1280`, `moe_inter_dim=320`, `kv_lora_rank=128`, `qk_nope_head_dim=48`, `qk_rope_head_dim=16`, `v_head_dim=64`, `max_seq_len=128`, `mtp_depth=1`, `mtp_loss_weight=0.3`, `weight_tying=true`. Used by training/checkpoint tests that need realistic module diversity (it is the smallest config where the MLA head-count and rank structure still look like the real thing).
- **`small_cfg`** — a 64-dim, 2-layer, 4-expert, `kv_lora_rank=16` config with `max_seq_len=64` and `vocab_size=1024`. Used by component tests that need speed above fidelity (a forward pass is microseconds).
- **`training_cfg`** — a full YAML-shaped dict (model + training + data sections) parsed by the same code path as `main()`, pointing at `/tmp` paths. `micro_batch_size=2`, `gradient_accumulation_steps=2`, `total_steps=10`, `warmup_steps=2`, `save_interval=5`.
- **`tokens` / `targets`** — random token IDs in vocab range; **`tmp_ckpt_dir` / `tmp_data_file` / `tmp_shard_dir`** — ephemeral on-disk fixtures cleaned up post-test. `tmp_shard_dir` writes three `shard_{i:05d}.bin` files of 128 tokens each so `PretrainDataset` sharded-path tests have real boundaries to cross.

Both config fixtures are `scope="session"`, so every test in the run shares one constructed model config dict — the per-test cost is construction of the tiny model, not the config. When a training test needs a `TrainingConfig` object, the helper `_build_training_config(cfg, tmp_ckpt_dir)` in `tests/test_training.py` fills in loop parameters (`batch_size=2`, `gradient_accumulation_steps=2`, `max_steps=4`, `nan_guard=False`, `fused=False` for CPU AdamW).

`★ Insight ─────────────────────────────────────`
The fixture configs are the reason the suite is fast and portable: they exercise the **real** `Transformer`/`Pretrainer`/`CheckpointManager` code, just at a scale where a forward pass is microseconds. A bug in MLA's matrix absorption or in sharded-dataset `_locate` will fire on `small_cfg` exactly as it would on the 412 M config — the shapes are smaller, the control flow is identical. This is why "CPU-testable" is a design goal, not a convenience.
`─────────────────────────────────────────────────`

### 3.3 What the suite guards (the invariants with teeth)

| Invariant | Test anchor |
|---|---|
| MLA KV compression matches uncompressed attention | `TestMLA*` (reference vs. SDPA path) |
| Aux-loss-free bias update changes routing without touching the loss | `TestAuxLossFreeGate`, `TestMoEBalanceMetric` |
| MTP depth-1 produces a length-aligned `(logits, targets)` pair and a combined loss | `TestMultiTokenPrediction`, `TestMTPGradientFlow` |
| Weight tying: `head.weight IS embed.weight` (one tensor, counted once) | `TestCountParameters`, `TestEmbedding` |
| Cached chunked prefill equals a full forward (the causal-mask contract) | `TestTransformer.test_chunked_prefill_matches_full_forward` |
| Atomic checkpoint round-trip restores model + optim + scheduler + `opt_steps` | `TestCheckpointRoundtrip`, `TestCheckpointOptStepsRoundtrip` |
| Crash mid-save leaves no `.tmp` or partial checkpoint behind | `tests/test_utils.py:TestCheckpointManagerAdditional.test_atomic_save_crash_recovery` |
| NaN guard rolls back to the last complete checkpoint after N consecutive NaNs | `TestNanGuardRollback` |
| Triton paths never silently activate without `ENABLE_TRITON_KERNELS=1` | `TestEnforceTritonEnvVar` (8 cases) |
| Triton dispatch degrades to the stacked path cleanly when unavailable | `TestMoeTritonDispatchWiring` |
| Sharded dataset spans shard boundaries correctly | `TestPretrainDataset` (`test_sharded_cross_boundary`, `test_locate_edge_case`) |
| Every `<path>.py:Symbol` anchor in `docs/` resolves; no line anchors | `tests/test_doc_refs.py:test_doc_anchors_resolve` |

Two of these invariants deserve emphasis because they are the ones most likely to regress silently. The chunked-prefill test pins the causal-mask contract: the mask is causal by *global* position, with `kv_len = end_pos` when a cache is present (`models/transformer.py:Transformer._build_causal_mask`). A change that makes the mask causal only *within* the current chunk passes all uncached tests and leaks future tokens in the cached path; the chunked-prefill test exists precisely to catch that. And the force-back test pins the ops safety rule: Triton code paths must be opt-in through one environment variable, never discovered by accident.

---

## 4. Atomic Checkpoint System

Checkpoints are managed by `utils/checkpoint.py:CheckpointManager`. Three files per step, all written **atomically**, only ever read back if **all three** are present.

### 4.1 The three-file layout

```
checkpoints/pretrain_a100/
├── model_step_004000.safetensors   # weights  (safetensors, deduped shared tensors)
├── optim_step_004000.pt             # AdamW state (FP32 master + m + v)  — torch.save
└── meta_step_004000.json            # step, scheduler state_dict, opt_steps, config, has_mtp
```

> [!NOTE] `optim_step_N.pt` uses `torch.save` (the workspace-sanctioned format alongside safetensors — "no `pickle` checkpoints", where "pickle" means raw `pickle.dump` of arbitrary objects). Optimizer state contains tensors in nested Python dicts that safetensors cannot serialize directly, and it must round-trip *exactly* for bit-identical resume, so `torch.save` is the right tool. For the same reason the load side uses `torch.load(..., weights_only=True)` (see §4.5) — a checkpoint is data, not executable code. Weights are always safetensors; this split follows the workspace rule.

The write order is load-bearing: **weights → optimizer → metadata**. `meta_step_N.json` is the commit record — the last file written. A crash before it lands means the step is incomplete and therefore invisible to resume (§4.4); a crash after it lands means all three files are complete and the step is resumable.

### 4.2 Atomicity — the `tempfile` → `os.replace` dance

Every file is written to a `tempfile.mkstemp` temp path in the same directory, fully written, then `os.replace`'d onto the final name. `os.replace` is POSIX-atomic on the same filesystem, so a crash mid-write leaves either the old complete file or nothing — never a half-written file. The machinery lives in `utils/checkpoint.py:CheckpointManager._atomic_write`:

```python
import contextlib
@contextlib.contextmanager
def _atomic_write(self, path: Path, suffix: str):
    fd, tmp = tempfile.mkstemp(dir=self.save_dir, suffix=suffix)
    os.close(fd)
    try:
        yield tmp
        os.replace(tmp, path)      # atomic rename
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
```

`★ Insight ─────────────────────────────────────`
Writing the temp file in the *same directory* as the target is load-bearing: `os.replace` is only atomic within a single filesystem mount. A naive `tempfile.NamedTemporaryFile()` defaults to `/tmp`, which on many systems is a separate tmpfs mount — renaming across mounts degrades to a copy+delete (non-atomic). Forcing `dir=self.save_dir` keeps the rename on one mount and the atomicity guarantee holds.
`─────────────────────────────────────────────────`

The exception path matters as much as the rename: on *any* failure the temp file is unlinked before the exception re-raises. `tests/test_utils.py:TestCheckpointManagerAdditional.test_atomic_save_crash_recovery` proves it end-to-end by patching `safetensors.torch.save_file` to raise `RuntimeError("disk full")` and asserting the directory afterwards contains no `.safetensors` and no `.tmp` file.

### 4.3 Shared-tensor dedup on save

With `weight_tying=true`, `head.weight` and `embed.weight` are the **same tensor** — and `model.state_dict()` lists it twice. safetensors rejects duplicate storage, so saving must dedup. Two layers handle this:

1. `training/pretrain.py:Pretrainer.save_checkpoint` drops the duplicate key *explicitly* before serializing (saving ~vocab × dim × 2 bytes ≈ 154 MB at the canonical config):
   ```python
   if getattr(model_to_save, "weight_tying", False):
       state = {k: v for k, v in state.items() if k != "head.weight"}
   ```
2. `utils/checkpoint.py:CheckpointManager._atomic_save_safetensors` dedups *by pointer* as a second line of defense (it would also catch any other accidentally shared tensor):
   ```python
   seen_ptrs: set = set()
   deduped: dict = {}
   for k, v in state.items():
       if v.data_ptr() in seen_ptrs:
           continue
       seen_ptrs.add(v.data_ptr())
       deduped[k] = v.contiguous()
   ```
   The first occurrence of each storage pointer is kept, stored contiguous; later duplicates are **skipped entirely** (the surviving `embed.weight` restores the shared storage on load). The comment in the source is precise about the invariant: "a shared tensor (weight tying) must appear once; the duplicate key is dropped; `load_state_dict(strict=False)` restores it through the surviving shared storage."

On load, `head.weight` is reported among the missing keys, which is expected. Because `Transformer.__init__` re-establishes the tie (`head.weight = embed.weight` — one `Parameter` object) *before* loading, copying into `embed.weight` updates the head automatically. `strict=False` plus the missing/unexpected audit in `CheckpointManager.load` turns this benign case into a warning, not an error.

### 4.4 "Complete checkpoint" — the only safe thing to resume from

```python
def _checkpoint_complete(self, step: int) -> bool:
    return all((self.save_dir / n).exists() for n in [
        f"model_step_{step}.safetensors", f"optim_step_{step}.pt", f"meta_step_{step}.json"])
```

`utils/checkpoint.py:CheckpointManager.latest_step()` scans the directory for `model_step_*.safetensors` names (extracting the integer step from the stem), sorts descending, and returns the highest step whose **all three** files exist. A step with a missing `meta_step_N.json` (e.g. crash after writing `model` + `optim` but before `meta`) is invisible to resume — the NaN guard and the `--resume` path will never pick it. This is the invariant that makes crash recovery safe, and it is guarded by `TestCheckpointManagerAdditional.test_latest_step_skips_partial_checkpoints`, which deletes one meta file and asserts `latest_step()` returns the other step.

### 4.5 What gets saved (the full RNG/optim/scheduler state)

Per the workspace atomic-checkpoint rule, a checkpoint carries the **full** state needed for bit-identical resume:

- **Weights** (deduped) → safetensors. MTP module weights are prefixed `mtp.` and merged into the same safetensors file (`Pretrainer.save_checkpoint` collects keys starting with `mtp_modules.` from the MTP wrapper's state dict and re-prefixes them `mtp.`).
- **Optimizer state** (`AdamW.state_dict()` = FP32 master + first moment `m` + second moment `v`, per param group) → `.pt`.
- **Scheduler** (`LambdaLR.state_dict()`) + **`opt_steps`** (the true optimizer-step counter, distinct from the micro-step `global_step`) + **`tag`** + **config** + **`has_mtp`** flag → `meta.json`.

`utils/checkpoint.py:CheckpointManager.load` restores in dependency order: weights first (`load_state_dict(weights, strict=False)`), then the optimizer (only if a path exists — a missing `.pt` logs a warning and the optimizer starts fresh rather than failing), then reads `meta.json` and returns it to the caller. Two hardening details: `torch.load(..., weights_only=True)` refuses pickle gadget classes in the optimizer file, and the `strict=True` default raises `RuntimeError` with the first five missing/unexpected keys — `training/pretrain.py:Pretrainer.load_checkpoint` deliberately passes `strict=False` because the tied-head missing key is expected, then restores MTP weights and scheduler/`opt_steps` from the returned meta dict.

> [!WARNING] Resume does **not** restore the `DataLoader` sampler RNG, so token order differs across a restart. This is benign at Chinchilla scale (samples are seen ~uniformly); exact resume would require checkpointing the sampler generator. This is documented in `training/pretrain.py:Pretrainer.train`.

---

## 5. VRAM Memory Budget (1 × A100 80 GB)

`utils/memory.py` is a CPU-arithmetic estimator (no GPU needed) used by `scripts/microbench_a100.py`. It sums four components plus an autodetected PyTorch overhead. All numbers below are computed for the **canonical config** (`batch=8`, `seq=2048`, 18 layers, BF16, 411,632,256 deduped base parameters) — and the component subtotal was verified by *running* `estimate_model_memory_gb` on the real constructed model (2026-08-04). They remain **estimates**: no GPU training run has executed, so nothing below has been measured against `torch.cuda.max_memory_allocated()` yet.

### 5.1 The four-component formula

`utils/memory.py:estimate_model_memory_gb` computes, in order: parameter bytes, KV-cache bytes, and (training only) optimizer + activation bytes, then adds overhead.

| Component | Formula | Bytes | GiB (1024³) |
|---|---|---|---|
| **Parameters** (BF16) | `N × 2` via `utils/memory.py:_deduped_numel` | 823\,264\,512 | 0.77 |
| **AdamW state** (FP32 master + m + v) | `N × 12` | 4\,939\,587\,072 | 4.60 |
| **MLA KV cache** (per layer) | `B × S × (kv_lora_rank + qk_rope_head_dim) × dtype_bytes` × L | 127\,401\,984 | 0.12 |
| **Activations** (grad-ckpt, PaLM formula) | `24 × B × S × D × L × dtype_bytes` | 10\,871\,635\,968 | 10.13 |
| **Component subtotal** | — | — | **15.61** |

The estimator's arithmetic is small and worth reading once:

```python
def _optimiser_bytes(model: nn.Module) -> int:
    """AdamW state: FP32 master copy (4) + first moment (4) + second moment (4) = 12 bytes/param."""
    return _deduped_numel(model) * 12

def _kv_cache_bytes(model: nn.Module, seq_len: int, batch_size: int, dtype_bytes: int = 2) -> int:
    """Total MLA KV-cache bytes: `batch·seq·(kv_lora_rank + qk_rope_head_dim)·dtype_bytes` summed over layers."""
    total = 0
    layers = list(model.layers) if hasattr(model, "layers") else []
    for layer in layers:
        attn = getattr(layer, "attn", None)
        if attn is None:
            continue
        kv_lora_rank = getattr(attn, "kv_lora_rank", 0)
        qk_rope_head_dim = getattr(attn, "qk_rope_head_dim", 0)
        per_layer = batch_size * seq_len * (kv_lora_rank + qk_rope_head_dim) * dtype_bytes
        total += per_layer
    return total
```

Three things are deliberately encoded here. **First**, `_deduped_numel` counts shared tensors once by `id(p)` — the tied head contributes zero extra — which is exactly why the estimator agrees with `count_parameters` (411\,632\,256). **Second**, the KV cache stores the *compressed latent* per token: `kv_lora_rank + qk_rope_head_dim = 192 + 24 = 216` floats/token/layer, independent of head count — the whole point of MLA. **Third**, the activation factor `utils/memory.py:_activation_bytes` is `24×` with gradient checkpointing and `36×` without: the PaLM Appendix-A constants. Gradient checkpointing trades ~30% recompute for the memory headroom that lets micro-batch 8 fit; disabling it raises the activation term from 10.13 to 15.19 GiB and the subtotal from 15.61 to 20.67 GiB (also verified by running the estimator with `grad_checkpoint=False`).

`★ Insight ─────────────────────────────────────`
The KV cache line is where MLA earns its keep. A **standard MHA** at this config would cache K+V at `2 × n_heads × d_head = 2 × 12 × 64 = 1536` floats/token/layer → `8 × 2048 × 1536 × 18 × 2 ≈ 0.90 GB`. MLA caches only the **compressed latent** `(d_c + d_R) = 216` floats/token/layer regardless of head count — **~0.12 GB, a ~7.1× reduction.** This is why the old "0.75 GB" figure in earlier notes was wrong: it used uncompressed head dimensions, not the latent. The reduction is the entire point of MLA (see [[Docs/03_Multi_Head_Latent_Attention]]).
`─────────────────────────────────────────────────`

### 5.2 Overhead and the headline

`utils/memory.py:_detect_overhead_gb` returns the PyTorch + CUDA context + caching-allocator overhead: `min(13.7, 0.17 × total_device_GB)` on CUDA (empirically ≤17% of device total on A100 80 GB), or 2 GB on CPU. This is a *bound*, not a measurement.

| | GiB |
|---|---|
| Parameters | 0.77 |
| AdamW optimizer | 4.60 |
| MLA KV cache | 0.12 |
| Activations (grad-ckpt) | 10.13 |
| **Component subtotal** | **~15.6** |
| PyTorch overhead (env-dependent) | 2 (CPU) – 13.6 (CUDA) |
| **Estimated peak** | **~17.6 CPU / ~29.2 CUDA** |

With MTP attached the model is 418\,713\,984 params: weights + optimizer grow by ~7.08M × 14 bytes ≈ 0.09 GiB, for a subtotal of ~15.7 GiB. Either way, well within the 80 GB A100 — the `launch_a100.sh` pre-flight independently estimates "~30-35 GB / 80 GB", leaving room to raise micro-batch or disable grad checkpointing if throughput demands it. `estimate_model_memory_gb(..., inference=True)` drops the optimizer (12×N) and activation terms — inference carries no AdamW state and forward activations are dominated by the KV cache — leaving weights + KV + overhead ≈ **0.89 GiB** for the full model at seq 2048, batch 8 (the KV cache dominates at long sequences; see [[Docs/10_Inference_and_Serving]] for the decode lifecycle).

For scale intuition: the *active* parameters per token are only ~185M (2 dense layers + 4 of 20 routed experts + shared + embeddings), while the optimizer and gradient memory scale with the full 411.6M — which is exactly why a single A100 fits where a naive dense 411.6M-param model with the same batch would be comparable, but the *compute* per token is far lower (see [[Docs/04_DeepSeekMoE]]).

### 5.3 The guard rail

```python
def assert_fits_in_available_gpu(estimate_gb: float, safety_margin_gb: float = 0.0) -> None:
    """No-op on CPU. On CUDA, raise RuntimeError if estimate > available - margin."""
    if not torch.cuda.is_available():
        return
    available_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if estimate_gb > available_gb - safety_margin_gb:
        raise RuntimeError(
            f"estimate {estimate_gb:.2f} GB exceeds available {available_gb:.2f} GB "
            f"minus safety margin {safety_margin_gb:.2f} GB"
        )
```

`utils/memory.py:assert_fits_in_available_gpu` is wired into the benchmark scripts so a misconfigured batch size fails loud and early rather than OOMing 20 minutes into a run. It is a no-op on CPU, which is what lets the estimator itself be unit-tested everywhere.

### 5.4 What the estimator deliberately does *not* count

Reading the four terms tells you what the number is not:

- **No MTP activations.** `_infer_dim_n_layers` reads `model.embed` and `model.layers`; a separate MTP wrapper's depth-1 block adds one extra transformer-ish block whose activations are small but nonzero. Budget ~0.1–0.2 GiB for it at this batch.
- **No `torch.compile` workspace.** The canonical config sets `compile: true` with `TORCH_COMPILE_MODE=max-autotune`; compile-time CUDA graphs and autotune scratch are *extra* memory on top of the overhead bound. This is the most likely reason a real run lands above the 15.6 GiB subtotal.
- **No fragmentation / allocator slack beyond the 17% bound.** The overhead constant is a bound, not a measurement (no GPU run has executed).
- **No gradient memory as a separate term.** The PaLM 24× activation constant already includes the backward-pass overhead; the optimizer term covers the FP32 state but not the (transient) gradient tensors themselves.

All of these are reasons to treat the headline as a floor with a generous overhead, not a precise prediction — the honest summary is "comfortably inside 80 GB with room to spare."

---

## 6. CI Pipeline Walkthrough (`.github/workflows/ci.yml`)

CI runs on every push/PR to `main` on an `ubuntu-latest` runner. The pipeline is deliberately a *layered* defense: each step catches a different class of failure, and the whole thing is CPU-only (PyTorch is installed with the `cpu` index — no CUDA wheels, no Triton).

```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
jobs:
  smoke:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install torch --index-url https://download.pytorch.org/whl/cpu
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Import checks
        run: python -c "import models.transformer; import models.mla; import models.moe; import models.mtp"
      - name: Check documentation
        run: python scripts/check_docs.py
      - name: Check doc-code symbol anchors
        run: python -m pytest tests/test_doc_refs.py -q --tb=short
      - name: Run test suite
        run: python -m pytest tests/ -q --tb=short
      - name: Model forward smoke test
        run: python scripts/smoke_forward.py
```

Reading the steps as a failure-detection ladder:

1. **Install** — CPU-only torch + `requirements.txt`. This step itself is a check: it proves the dependency manifest resolves on a clean machine.
2. **Import checks** — importing `models.transformer`, `models.mla`, `models.moe`, `models.mtp` catches syntax errors, broken relative imports, and construction-time module-level failures in seconds, before any test runs. Note that these imports must succeed *without* Triton installed — the `if HAS_TRITON:` guards in the kernel modules exist for exactly this reason.
3. **`scripts/check_docs.py`** — lints the docs themselves: control characters, stale patterns (e.g. the banned curly-brace LaTeX thousand-separator form), internal `.md` links, and backtick-quoted repo paths. Since the expansion it globs `docs/**` recursively (`rglob("*.md")`), so `reference/` and `guides/` are linted too. This is the *prose hygiene* gate.
4. **`tests/test_doc_refs.py`** — the *semantic* docs gate: every `<path>.py:Symbol` anchor in the docs must resolve against the real code (§7). This is the newest step, added with the documentation expansion.
5. **The pytest suite** — the 199-node suite from §3. Fast (under 20 s on a laptop), so it is never worth skipping.
6. **`scripts/smoke_forward.py`** — builds the canonical 411.6M-param `Transformer` from `configs/pretrain_a100_422m.yaml` (with `max_seq_len` shrunk to 16) and runs one forward pass, asserting the logits shape `(2, 16, vocab_size)`. This is the closest CI comes to the real model: it proves the *full-size* architecture instantiates and executes, not just the toy fixtures.

The ladder has a deliberate property: steps 3–6 all run even when step 5 fails, so a push cannot hide doc drift behind a test failure. And because nothing needs a GPU, the entire pipeline costs seconds and can be reproduced locally with the same commands.

## 7. The Doc↔Code Alignment Gate (`tests/test_doc_refs.py`)

Prose rots. A paragraph that says "the mask is built by `_build_causal_mask`" is only as true as the symbol it names — and for years this repo's docs had anchors that silently pointed at nonexistent symbols, or worse, line numbers that drifted with every edit. The alignment gate makes documentation a *machine-checked contract*.

`tests/test_doc_refs.py:test_doc_anchors_resolve` scans every markdown file under `docs/**` (plus the root README/AGENTS/SKILLS/CONTEXT) and applies three rules:

- **Symbol anchors must resolve.** The scanner extracts anchor pairs of the form `<path>.py:<symbol>` (a path that may include `/` and `-`, a dot, `py`, a colon, then a dotted symbol name) with a single regex. Each module is imported as a package (`models/mla.py` → `importlib.import_module("models.mla")` so relative imports resolve), and the gate walks a `hasattr` chain through the symbol: `models/transformer.py:Transformer.generate` must be an attribute of an attribute of the module. Class-level resolution has teeth — an anchor naming the gate's `bias` buffer fails, because `bias` is an *instance* attribute registered in `__init__`, not a class-level symbol. The resolvable form for instance state is a method such as `models/moe.py:AuxLossFreeGate.update_bias`.
- **Line anchors are banned outright.** A second regex rejects a `.py` path immediately followed by a colon and a multi-digit number — numeric cites of the `path.py:<line>` form, with the terse justification "they rot; symbols do not". A line number is meaningless after the next refactor; a symbol name is a stable API contract.
- **JIT symbols are banned.** `_mla_flash_fwd_kernel`, `_grouped_moe_fwd_kernel`, `_TritonMlaAttentionFunction`, and friends are defined under `if HAS_TRITON:` and cannot resolve on a Triton-less CI box. Docs must cite the always-defined host wrappers (`triton_mla_attention`, `triton_grouped_moe_dispatch`) instead. A small allowlist (`ALLOW_MISSING_FILES`) covers paper-spec files like `training/loss_triton.py` that are planned but do not exist yet.

Run it the same way CI does:

```bash
python -m pytest tests/test_doc_refs.py -q --tb=short   # as a test
python tests/test_doc_refs.py                            # or as a script
```

A failure lists the offending doc, line, and reason (`missing-symbol: X`, `line anchor banned`, `JIT symbol (cite host wrapper)`, `missing-file`). The practical effect for writers: **every anchor in a doc is a compile-time reference** — if you rename a method, the gate tells you which prose claims to update, and if you cite a symbol that doesn't exist, CI turns red before the docs mislead anyone.

`★ Insight ─────────────────────────────────────`
Why symbols and not line numbers? A line anchor says "the interesting code is *here*" — it is true for exactly one commit. A symbol anchor says "this thing exists and has this name" — it stays true until the name changes, and when the name *does* change, the gate fires so the prose is fixed in the same commit. The gate converts documentation from a liability that decays into an asset that is checked on every push.
`─────────────────────────────────────────────────`

## 8. Debugging Recipes

This section is the operations playbook: the three failure classes this architecture actually produces, how they present, and the fastest way to confirm each one.

### 8.1 NaN guard: silent divergence → checkpoint rollback

**Symptom.** A training log that shows `loss = nan` (or `inf`), followed — if the guard is on — by a rollback line and training continuing from an old step; or a hard abort `RuntimeError: NaN/Inf with no checkpoint to restore from`.

**Mechanism.** The check lives in `training/pretrain.py:Pretrainer.train_step`, *before* `loss.backward()`:

```python
if self.config.nan_guard and (torch.isnan(loss).any().item() or torch.isinf(loss).any().item()):
    self._log(f"[nan-guard] NaN/Inf at micro_step={micro_step}, opt_steps={self._opt_steps}. Skipping backward.")
    self.optimizer.zero_grad(set_to_none=True)
    return None
```

A `None` return skips the optimizer step entirely — the bad gradient never touches the weights — and the training loop counts the streak:

```python
if metrics is None:
    nan_guard_streak += 1
    if nan_guard_streak >= self.config.nan_guard_max_consecutive:
        latest = self._find_latest_checkpoint()
        if latest is not None:
            global_step = self.load_checkpoint(latest)
        else:
            raise RuntimeError("NaN/Inf with no checkpoint to restore from")
        nan_guard_streak = 0
    continue
nan_guard_streak = 0
```

**Recipe.** The canonical config sets `nan_guard: true` and `nan_guard_max_consecutive: 5`. To verify the mechanism on CPU: build a `Pretrainer` with `nan_guard=True`, save a checkpoint at step 0, then feed inputs that force an exploding loss (e.g. logits with an extreme scale); `tests/test_training.py:TestNanGuardRollback.test_consecutive_nan_triggers_rollback` does exactly this and asserts the rollback. Key facts for diagnosis: (1) the guard checks the *total* loss (main + MTP), so a NaN that appears only in `mtp_loss` still triggers it; (2) rollback restores weights, optimizer state, scheduler, and `opt_steps` (§4.5) — but *not* the DataLoader sampler RNG, so the resumed token order differs; (3) the guard is a tripwire, not a fix — a persistent NaN streak (≥5 at the canonical config) means a real numerical bug, usually an LR that is too large under μP, an uninitialized scale, or a BF16 underflow path. See [[Docs/08_Training_Pipeline]] for the μP LR math and [[Docs/01_Foundations]] for BF16 numerics.

### 8.2 Shape errors: the causal-mask and dtype contracts

**Symptom.** `RuntimeError: The size of tensor a (…) must match the size of tensor b (…)` inside `F.scaled_dot_product_attention`, or logits that are right in prefill but wrong when a KV cache is involved, or `IndexError` on an `Embedding` lookup.

**The three contracts to check, in order:**

1. **Token dtype.** `nn.Embedding` and `F.cross_entropy` require Long indices, but `PretrainDataset` stores tokens as uint32 (4 bytes vs 8 for memory efficiency on sharded mmap data). The cast happens at the boundary: `Transformer.forward` and `Pretrainer.train_step` both contain `if tokens.dtype != torch.long: tokens = tokens.to(torch.long)`. If you bypass either entry point (raw module call, custom loop), cast yourself.
2. **The causal-mask shape.** `models/transformer.py:Transformer._build_causal_mask(seqlen, kv_len, start_pos, device)` returns an additive mask of shape `(1, 1, S_q, S_kv)` that is causal by **global** position: position `i` in the query attends to keys `0 … start_pos + i`. The caller passes `kv_len = end_pos` when a cache is present (`end_pos = start_pos + seqlen`), else `seqlen`. Violations present as SDPA shape errors when `S_kv` is wrong, or — far more insidiously — as *future leakage* in the cached path when the mask is causal only within the current chunk (this exact bug is what `TestTransformer.test_chunked_prefill_matches_full_forward` pins: a cached mid-sequence prefill must equal a full forward bit-for-bit at `atol=1e-5`).
3. **The cross-entropy reshape.** `train_step` flattens logits to `(-1, vocab_size)` and targets to `(-1)` with `ignore_index=-100`. The classic failure is forgetting `ignore_index`, which turns the padded/masked positions into real gradient contributions; the classic *shape* failure is feeding `(B, S)` targets where the model emitted `(B, S, V)` and vice versa. The MTP path adds a second alignment: MTP depth-1 pairs produce length-`seqlen - d - 1` targets, and the wrapper's `compute_loss` is responsible for that slicing — a length mismatch there means the loss shape asserts in `F.cross_entropy`, not in the model.

**Recipe.** Reproduce any mask suspicion with the chunked-prefill test on `small_cfg`: run `python -m pytest tests/test_models.py -k chunked -v` with both `attn_impl="sdpa"` and `"manual"`. If it fails, print the mask and check the `(S_q, S_kv)` block against `start_pos`: for a 16-token sequence with `start_pos=8`, the lower-right `8×16` block must be causal with the diagonal at global position 8. The single-token decode path (`seqlen=1`) bypasses the mask entirely — a bug that only shows at `seqlen>1` prefill is a mask bug, not an attention bug.

### 8.3 Triton fallback warnings: two different animals

This repo has **two** independent "Triton not active" mechanisms, and confusing them is the most common ops mistake:

1. **The env-var force-back (construction time).** `models/_triton_dispatch.py:enforce_triton_env_var` runs during model construction. If the config says `attn_impl: "triton"` or `moe_dispatch: "triton_grouped"` but `ENABLE_TRITON_KERNELS != "1"`, the keys are rewritten to `sdpa`/`stacked` *in the config dict* and one warning is logged:
   ```
   [warn] Triton dispatch keys set without ENABLE_TRITON_KERNELS=1; forcing attn_impl='triton' -> 'sdpa'. Set ENABLE_TRITON_KERNELS=1 to enable the fused Triton paths.
   ```
   This is a *policy* guard (AGENTS rule: Triton is opt-in), tested by `tests/test_force_back.py:TestEnforceTritonEnvVar`. If you set the env var but still see this warning, check the env at the *process* level — `ENABLE_TRITON_KERNELS=1 python training/pretrain.py …` must prefix the launch, and a `launch_a100.sh`-style script that exports it *after* the Python process starts does not count.
2. **The runtime fallback (forward time).** Even with the env var set, `models/moe.py:DeepSeekMoE.forward` can fall back per-call when the Triton path is unavailable — and at the canonical config it *always* does, because the grouped-MoE kernel is capped at register dims `I, D ≤ 256` while the canonical `moe_inter_dim=384` and `dim=768` exceed the cap:
   ```
   [moe] triton_grouped unavailable (ValueError: …); falling back to 'stacked' for this model.
   ```
   The fallback prints once per model (guarded by `self._triton_fallback_warned`) and then silently uses the stacked path — correct results, no fused speedup. The MLA kernel has no such cap at canonical dims (register dims R=192, D_nope=48, D_rope=24, D_v=64 all fit; sequence length is tiled), so the MLA fused path *can* activate at the canonical config if the env var is set — see [[Docs/12_Triton_Kernels]] for the register-budget math and the honest fallback design.

**Recipe.** To know which path a run actually executed: (1) set `ENABLE_TRITON_KERNELS=1`; (2) grep the log for the construction-time warning (absent = force-back passed); (3) grep for `[moe] triton_grouped unavailable` — if present, the MoE dispatch is stacked *by design* at these dims, and the benchmark numbers in `step_time_a100.py` reflect the stacked path, not the fused kernel. `TestMoeTritonDispatchWiring.test_triton_path_falls_back_cleanly_on_cpu` and `test_kernel_call_raises_value_error_on_too_large_dim` pin both halves of this contract.

### 8.4 General discipline

- **Run one file, not the world.** `python -m pytest tests/test_utils.py -v` is ~1–2 s; the full suite is under 20 s. There is never an excuse to skip the relevant file before claiming a change works.
- **The suite is the spec.** When behavior and docs disagree, the suite wins; when the suite and your mental model disagree, the suite is usually right — the invariants table (§3.3) names the test that pins each contract.
- **A red doc gate is a docs bug, not a CI nuisance.** If `tests/test_doc_refs.py` fails after an edit, you renamed or mis-cited a symbol — fix the anchor, never the gate.

---

## 9. Scripts & Utilities Reference

`scripts/` are observability instruments — they answer three questions before you commit GPU-days: *Does it build?* (`smoke_forward.py`), *Does it fit?* (`microbench_a100.py`), *Is it fast enough?* (`step_time_a100.py`).

| Script | Purpose | Requires GPU |
|---|---|---|
| `launch_a100.sh` | Pre-flight checks + production training launch (nohup) | Yes (A100) |
| `microbench_a100.py` | VRAM estimate vs measured peak | Yes |
| `step_time_a100.py` | ms/step, tokens/sec, MFU | Yes |
| `smoke_forward.py` | Single forward-pass sanity on the canonical config | No (CPU-safe, runs in CI) |
| `e2e_test_gpu.py` | End-to-end GPU test suite (1650 2M config) | Yes |
| `build_small_pretrain_data.py` | Tiny dataset for local dev (no 8B download) | No |
| `check_docs.py` | Lint `docs/` links, paths, stale patterns; refresh size table + `docs:verified` stamps (runs in CI) | No |

### `launch_a100.sh`

Pre-flight: asserts CUDA + ≥75 GB VRAM and `data/pretrain_chinchilla/shard_*.bin` present, creates `checkpoints/pretrain_a100/`, sets env (`CUDA_VISIBLE_DEVICES`, `TORCH_COMPILE_MODE=max-autotune`, `TOKENIZERS_PARALLELISM=false`, `WANDB_PROJECT`/`WANDB_RUN_NAME`), prints its own estimate ("peak VRAM ~30-35 GB / 80 GB", "wall ~30-45 h at 35-40% MFU" — both estimates, no GPU run yet), then launches `nohup python -u training/pretrain.py --config configs/pretrain_a100_422m.yaml --data-path data/pretrain_chinchilla --checkpoint-dir checkpoints/pretrain_a100 > checkpoints/pretrain_a100/train.log 2>&1 &` and tails the first 50 lines after a 30 s warmup.

Monitor: `tail -f checkpoints/pretrain_a100/train.log`. Resume: `python training/pretrain.py --config configs/pretrain_a100_422m.yaml --resume 4000` (or just relaunch — `Pretrainer.train` auto-resumes from `latest_step()` when a complete checkpoint exists, see §4.4).

### `microbench_a100.py`

Builds the canonical model with grad checkpointing, prints the analytical `estimate_model_memory_gb` figure (§5), runs one forward+backward, and reports `torch.cuda.max_memory_allocated()`. Calls `assert_fits_in_available_gpu(estimate, margin=2.0)` so a misconfigured batch size fails loud here, not 20 minutes into a run. **No A100 run has executed yet — both figures are estimates until it does.**

### `step_time_a100.py`

Measures median ms/step, tokens/sec, and **MFU** = achieved FLOPs/s ÷ peak BF16 FLOPs/s (312 TFLOPS on A100). Target for the canonical 412M config is 35–40% — an estimate, not a measurement.

```bash
python scripts/step_time_a100.py --steps 20 --warmup 5
```

Flags: `--steps` (20), `--warmup` (5), `--no-compile`, `--compile-mode` (max-autotune), `--peak-tflops` (312). Low MFU (< 25%) usually means memory-bound MoE dispatch (the `stacked` path re-stacks expert weights every forward — and recall from §8.3 that at canonical dims the fused path cannot activate), grad-checkpoint recompute, or a batch too small to saturate the SMs.

### `smoke_forward.py` / `e2e_test_gpu.py` / `build_small_pretrain_data.py`

- `smoke_forward.py` — builds the canonical config (411.6M params), runs one forward on tiny input, asserts the logits shape. CPU-only; this is the CI forward smoke test (§6).
- `e2e_test_gpu.py` — broader GPU validation on `configs/pretrain_1650_2m.yaml`: data shim → model forward/backward → checkpoint roundtrip → generation + speculative decode, sized to fit a 4 GB GPU.
- `build_small_pretrain_data.py` — creates a tiny token `.bin` for local development without running the multi-hundred-GB pipeline.

### `check_docs.py`

Lints the docs: control characters, stale math/status patterns, internal `.md` links, and backtick-quoted repo paths — and since the expansion it walks `docs/**` recursively, so subfolder docs (`reference/`, `guides/`) are covered too. It also refreshes the size table and `docs:verified` stamps:

```bash
python scripts/check_docs.py                                # lint only
python scripts/check_docs.py --update-sizes --stamp-footers # refresh README sizes + stamps
```

The two doc gates are complementary: `check_docs.py` checks *prose hygiene* (links, paths, stale numbers, LaTeX), `tests/test_doc_refs.py` checks *semantic truth* (every symbol anchor resolves, no line anchors, no JIT cites).

### TrainingLogger (`utils/logging.py:TrainingLogger`)

Rolling-window console logger (`step=… | loss=… | ppl=… | lr=… | tps=…`) with optional WandB via `WANDB_PROJECT` / `WANDB_RUN_NAME`. Metrics: `train/loss`, `train/ppl`, `train/lr`, `train/tokens_per_sec`, `train/mtp_loss`, `train/balance_loss`. One operational note: `Pretrainer.train` calls `.item()` on the loss **once per log step**, not per micro-step — logging is the only allowed host round-trip, so a per-micro-step `.item()` would add 3–4 forced GPU syncs per step (see `train_step` in [[Docs/08_Training_Pipeline]]).

---

## 10. Operational Status

| Capability | State |
|---|---|
| Architecture (MLA / MoE / MTP / aux-loss-free / μP / NaN guard) | ✅ implemented |
| CPU test suite (199 nodes at time of writing; ~190 CPU-runnable, remainder environment-gated) | ✅ passing |
| Atomic safetensors checkpointing | ✅ implemented |
| VRAM estimator | ✅ implemented |
| Fused Triton kernels (MLA attention, grouped MoE GEMM) | ✅ implemented, opt-in, BF16 (MoE capped at I, D ≤ 256 — canonical config falls back to stacked) |
| CI (`.github/workflows/ci.yml`) | ✅ runs docs lint + doc-anchor gate + `pytest tests/ -q` + forward smoke |
| Doc↔code alignment gate (`tests/test_doc_refs.py`) | ✅ implemented, wired into CI |
| Full GPU pre-training run (512 000 steps) | ⏳ **not yet executed** (`checkpoints/` empty) |
| FP8 mixed precision | ❌ paper-spec only — see [[Docs/06_FP8_Mixed_Precision]] |
| DualPipe pipeline parallelism | ❌ paper-spec only — see [[Docs/07_DualPipe_Parallelism]] |

> **Next:** [[Docs/12_Triton_Kernels]] — the real fused kernels: MLA attention and grouped-GEMM MoE dispatch (BF16, not FP8).

---

## 11. Check Your Understanding

**Q1. Why can `CheckpointManager.load` restore `head.weight` even though the checkpoint file does not contain a `head.weight` key?**

The checkpoint dropped the duplicate `head.weight` at save time (`Pretrainer.save_checkpoint` filters it; `_atomic_save_safetensors` would also dedup it by pointer). `Transformer.__init__` re-establishes the tie before loading — `head.weight` and `embed.weight` are one `Parameter` object — so `load_state_dict(strict=False)` copies into `embed.weight`, which *is* the head's storage. `head.weight` appears in the missing-keys audit and is expected (a warning under `strict=False`, an error only under `strict=True`).

**Q2. A checkpoint directory contains `model_step_10.safetensors` and `optim_step_10.pt` but no `meta_step_10.json`. Will `--resume` pick step 10?**

No. `latest_step()` only returns steps where all three files exist (`_checkpoint_complete`). Step 10 is invisible to resume — by design: the meta file is written last, so its absence means the save was interrupted and the step must not be trusted. `TestCheckpointManagerAdditional.test_latest_step_skips_partial_checkpoints` pins exactly this.

**Q3. You set `ENABLE_TRITON_KERNELS=1`, yet the log still shows `[moe] triton_grouped unavailable … falling back to 'stacked'`. Is the environment variable broken?**

No — these are two different mechanisms. The env var only clears the *policy* force-back (`enforce_triton_env_var`). The runtime fallback fires because the grouped-MoE kernel is register-capped at `I, D ≤ 256`, and the canonical config's `moe_inter_dim=384` / `dim=768` exceed it, so `triton_grouped_moe_dispatch` raises `ValueError` and `DeepSeekMoE.forward` falls back with a one-time warning. At canonical dims the stacked path is the expected behavior; the MLA fused path can still activate (its register dims all fit).

**Q4. Why does the doc-refs gate reject an anchor that names the gate's `bias` buffer, even though `bias` clearly exists at runtime?**

The gate resolves symbols with a `hasattr` chain against the *class* (and module), not against any instance. `bias` is registered per-instance in `__init__` (`self.register_buffer("bias", …)`), so a class-level lookup finds nothing. Cite what the gate can resolve instead — e.g. `models/moe.py:AuxLossFreeGate.update_bias`, a real class-level method. The rule of thumb: anchor classes and methods, never instance attributes or buffers.

<!-- docs:verified 2026-08-04 · 59aeef3 -->
