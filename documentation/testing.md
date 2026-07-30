# Testing — Test Corpus as a Learning Tool

## A Comprehensive Technical Reference

> **Covers**: The `tests/` suite — how to use it to verify correctness and learn the system's invariants.

---

## Table of Contents

1. [Philosophy](#philosophy)
2. [Running Tests](#running-tests)
3. [Fixtures (conftest.py)](#fixtures-conftestpy)
4. [test_models.py](#test_modelspy)
5. [test_training.py](#test_trainingpy)
6. [test_inference.py](#test_inferencepy)
7. [test_utils.py](#test_utilspy)
8. [test_moe_triton.py](#test_moe_tritonpy)
9. [test_mla_triton.py](#test_mla_tritonpy)
10. [test_force_back.py](#test_force_backpy)
11. [Load-Bearing Tests](#load-bearing-tests)
12. [Adding New Tests](#adding-new-tests)
13. [CI](#ci)

---

## Philosophy

Every test in this repo is designed to run on **CPU without CUDA or Triton**. This is intentional:

- Mac developers can verify correctness locally
- CI runs without GPU runners
- Triton paths compare against PyTorch references on CPU

GPU-specific tests are gated with `@pytest.mark.gpu` and auto-skipped without CUDA.

**Tests are documentation.** When in doubt about an invariant, search `tests/` for the behaviour.

### Tests as epistemology

In a from-scratch LLM repo, **tests are the only machine-checked specification**. Papers describe intent; code drifts; tests catch drift.

| Test category | What it proves | Example |
|---|---|---|
| Shape | Graph wiring | `test_forward_shape` |
| Equivalence | Two paths compute same math | `test_sdpa_and_manual_agree` |
| Invariant | Architectural rule | `test_bias_not_in_parameters` |
| Roundtrip | Persistence | `test_checkpoint_mtp_roundtrip` |
| Guard | Safety mechanism | `test_nan_guard_rollback` |

When extending the model, add a test **before** updating docs — the test is the contract.

---

## Running Tests

```bash
# Full suite (CPU)
python -m pytest tests/ -q

# Specific module
python -m pytest tests/test_models.py -v

# Keyword filter
python -m pytest tests/ -k "MoE or MTP or bias"

# With coverage (optional)
python -m pytest tests/ --cov=models --cov=training
```

Expected: all tests pass on a fresh clone with `torch` installed (~1–3 min on CPU).

### Suite overview (2026-07-31)

| File | Tests | Primary coverage |
|---|---|---|
| `test_models.py` | 81 | MLA, MoE, MTP, transformer shapes & invariants |
| `test_training.py` | 40 | Pretrain loop, μP, NaN guard, checkpoints |
| `test_utils.py` | 26 | Checkpoint I/O, memory estimates, logging |
| `test_moe_triton.py` | 16 | MoE Triton vs PyTorch reference (`@pytest.mark.gpu` for kernel) |
| `test_inference.py` | 13 | Generate, speculative decode |
| `test_mla_triton.py` | 5 | MLA reference + import surface; GPU Triton vs reference |
| `test_force_back.py` | 8 | `ENABLE_TRITON_KERNELS` force-back guard |
| **Total** | **189** | |

**Remaining gap:** full-model `test_sdpa_and_triton_agree` (end-to-end `attn_impl: triton`) is not yet in the suite — track in [triton_kernels.md](triton_kernels.md#status-2026-07-30).

---

## Fixtures (conftest.py)

| Fixture | Purpose | Key dims |
|---|---|---|
| `cfg` | Medium model (2 layers) | dim=640, 8 experts |
| `small_cfg` | Tiny model for fast tests | dim=64, vocab=1024, 4 experts |
| `nested_cfg` | Nested YAML shape | `{"model": {...}}` |
| `training_cfg` | Pretrainer smoke | Minimal training config |
| `tokens`, `targets` | Random int64 tensors | (2, 16) |
| `tmp_ckpt_dir` | Temp checkpoint dir | Auto-cleaned |
| `tmp_data_file` | Single-file dataset | uint32 tokens |
| `tmp_shard_dir` | Sharded dataset | 2 shards |

`device` fixture: always CPU in CI.

---

## test_models.py

**Largest file (~900 lines).** Covers every model component.

| Class | What it verifies |
|---|---|
| `TestEmbedding` | Embedding shape, init |
| `TestTransformer` | Construction, forward shape, nested config |
| `TestMLA` | SDPA vs manual agreement, cache, RoPE |
| `TestSwiGLUFFN` | FFN shape, non-linearity |
| `TestExpert` | Single MoE expert |
| `TestDeepSeekMoE` | Forward, gate shapes, bias update, balance loss |
| `TestAuxLossFreeGate` | **bias is buffer, in state_dict** |
| `TestTransformerBlock` | Dense vs MoE block selection |
| `TestMTPBlock/Module/Prediction` | MTP shapes, shared head, short seq |
| `TestGeneration` | generate(), sampling, cache reset |
| `TestCountParameters` | Weight tying dedup |

**Key learning test:**

```python
test_sdpa_and_manual_agree  # MLA absorption equivalence
test_bias_not_in_parameters # MoE bias invariant
test_shared_head_mtp        # MTP head sharing
```

---

## test_training.py

| Class | What it verifies |
|---|---|
| `TestTrainingConfig` | Dataclass defaults |
| `TestWarmupCosineScheduler` | LR curve shape |
| `TestPretrainDataset` | Single + sharded layouts |
| `TestPretrainerConstruction` | μP LR, optim dedup, MTP wrap |
| `TestCheckpointRoundtrip` | Save/load with MTP |
| `TestTrainStep` | Forward+backward smoke |
| `TestMoEBalanceMetric` | balance_loss logging |
| `TestNanGuardRollback` | NaN recovery |
| `TestConfigFromYAML` | YAML → TrainingConfig |

---

## test_inference.py

| Class | What it verifies |
|---|---|
| `TestModelGenerate` | KV-cache generation |
| `TestSpeculativeDecoder` | Accept/reject, cache coherence |
| `TestInferenceHelpers` | generate_tokens wrapper |

---

## test_utils.py

| Class | What it verifies |
|---|---|
| `TestCheckpointManagerSaveLoad` | Atomic save, step discovery |
| `TestCheckpointManagerMTP` | mtp. prefix roundtrip |
| `TestMemoryEstimation` | Formula components |
| `TestCheckpointManagerAdditional` | Crash recovery, incomplete steps |

---

## test_moe_triton.py

| Class | What it verifies |
|---|---|
| `TestGroupedMoePytorchReference` | Reference kernel correctness |
| `TestMoeTritonImportSurface` | HAS_TRITON gating |
| `TestMoeTritonDispatchWiring` | stacked vs triton_grouped fallback |
| `TestMoeTritonKernelGPU` | Triton ≈ reference on GPU (`@pytest.mark.gpu`) |

CPU tests always pass. GPU tests skipped on Mac.

---

## test_mla_triton.py

| Class | What it verifies |
|---|---|
| `TestMlaAttentionReference` | PyTorch reference shapes + decode step |
| `TestMlaTritonImport` | `HAS_TRITON` gating, ImportError without triton |
| `TestMlaTritonKernelGPU` | Triton ≈ reference on GPU (`@pytest.mark.gpu`) |

---

## test_force_back.py

Verifies `ENABLE_TRITON_KERNELS` env-var guard:

- Without env var: `triton_grouped` → `stacked`, `triton` attn → `sdpa`
- With `ENABLE_TRITON_KERNELS=1`: keys preserved
- Single warning at construction, not per-layer

---

## Load-Bearing Tests

**Never break these** — they guard architectural invariants:

| Test | Invariant |
|---|---|
| `test_bias_not_in_parameters` | MoE bias is buffer |
| `test_bias_in_state_dict` | Bias persists in checkpoints |
| `test_sdpa_and_manual_agree` | MLA math correctness |
| `test_update_bias_sign_rule` | Over/under-load bias direction |
| `test_stacked_weights_refresh_after_optimizer_step` | No stale MoE weights |
| `test_count_with_weight_tying` | Param dedup |
| `test_force_back_moe_dispatch_when_env_var_missing` | Triton guard |
| `test_nan_guard_rollback` | Training stability |

---

## Adding New Tests

Rules from AGENTS.md:

1. **CPU-runnable by default** — PyTorch reference for any Triton path
2. **GPU tests** — `@pytest.mark.gpu`, compare Triton vs reference at `atol=1e-2` BF16
3. **Triton kernels** — also need `gradcheck` on float32 tiny config
4. Place in `tests/test_<component>.py` or extend existing file
5. Use `small_cfg` fixture for speed

---

## CI

`.github/workflows/ci.yml` — CPU smoke on push.

**Known issue:** CI may reference missing `configs.pretrain_a100_422m` Python module (only YAML exists). Local `pytest tests/` is the authoritative correctness gate.

---



---

## Test-Driven Learning Exercises

Use tests as a **curriculum** — read the test, predict the assertion, then read the implementation.

### Exercise 1 — MLA absorption equivalence

```bash
python -m pytest tests/test_models.py -k sdpa_and_manual_agree -v
```

**Learn:** SDPA path expands compressed KV; manual path absorbs projections. They must agree within tolerance.

### Exercise 2 — MoE bias invariant

```bash
python -m pytest tests/test_models.py -k bias_not_in_parameters -v
```

**Learn:** `AuxLossFreeGate.bias` is `register_buffer`, not `nn.Parameter` — excluded from AdamW and weight decay.

### Exercise 3 — μP LR formula

```bash
python -m pytest tests/test_training.py -k mup_lr_scaling -v
```

**Learn:** LR scales as sqrt(ref_params / actual_params) after MTP wrap.

### Exercise 4 — NaN guard rollback

```bash
python -m pytest tests/test_training.py -k nan_guard -v
```

**Learn:** 5 consecutive bad micro-steps trigger checkpoint restore.

### Exercise 5 — Triton env guard

```bash
python -m pytest tests/test_force_back.py -v
```

**Learn:** Without `ENABLE_TRITON_KERNELS=1`, Triton config keys are force-backed at construction.

---

## Fixture Deep Dive (`conftest.py`)

### `small_cfg` — the workhorse

```python
dim=64, vocab_size=1024, n_layers=2, n_routed_experts=4
```

Small enough for instant CPU tests; preserves MLA/MoE/MTP structure.

### `cfg` — medium model

`dim=640`, 8 experts — used for shape tests closer to production aspect ratios.

### `tmp_shard_dir`

Two-shard layout testing `PretrainDataset._locate` cross-shard stitching.

---

## pytest Markers

| Marker | Meaning |
|---|---|
| `@pytest.mark.gpu` | Requires CUDA; skipped on Mac CI |

GPU tests compare Triton vs PyTorch reference at `atol=1e-2` BF16.

---

## Mapping Tests to Architecture Docs

| Test file | Primary doc |
|---|---|
| `test_models.py` | [MLA.md](MLA.md), [moe.md](moe.md), [transformer.md](transformer.md) |
| `test_training.py` | [training.md](training.md) |
| `test_inference.py` | [inference.md](inference.md) |
| `test_utils.py` | [utils.md](utils.md) |
| `test_moe_triton.py` | [triton_kernels.md](triton_kernels.md) |
| `test_force_back.py` | [configs.md](configs.md) §Triton |

---

## When a Test Fails — Diagnostic Tree

```
pytest failure
  ├─ shape mismatch → check config dims vs implementation
  ├─ MLA sdpa/manual disagree → absorption bug in mla.py
  ├─ MoE bias in parameters → violated buffer invariant
  ├─ checkpoint roundtrip → weight tying or mtp. prefix
  └─ nan_guard → training instability; check LR, data, dtype
```





## Complete Test Inventory

| File | Class / test | Asserts |
|---|---|---|
| `test_models.py` | `TestTransformerForward` | Output shapes, no NaN |
| `test_models.py` | `TestMLAPaths` | SDPA vs manual agreement |
| `test_models.py` | `TestMoE` | Routing, expert output shape |
| `test_models.py` | `TestMTP` | Shared head, loss finite |
| `test_models.py` | `TestWeightTying` | `head.weight is embed.weight` |
| `test_training.py` | `test_mup_lr_scaling` | μP formula |
| `test_training.py` | `test_nan_guard_rollback` | Recovery after NaN streak |
| `test_training.py` | `test_checkpoint_roundtrip` | Save/load weights |
| `test_training.py` | `test_moe_bias_update_during_training` | Bias buffer moves |
| `test_inference.py` | `TestGenerate` | Greedy decode deterministic |
| `test_inference.py` | `TestSpeculativeDecoder` | Accept/reject path |
| `test_utils.py` | `TestCheckpointManager` | Atomic save, latest_step |
| `test_force_back.py` | Triton env guard | Force-back without env var |
| `test_moe_triton.py` | GPU kernel | Triton vs stacked (CUDA only) |

Run `python -m pytest tests/ --collect-only -q` in a checkout to see the live list — names above are representative; grep `def test_` when adding docs.

---

## How to Write a Load-Bearing Test

Template for a new invariant:

```python
def test_my_invariant(small_cfg):
    model = Transformer(small_cfg)
    # exercise code path
    assert condition, "actionable message linking to doc section"
```

**Rules:**
1. Use `small_cfg` unless testing scale-specific behaviour
2. No CUDA required unless marked `@pytest.mark.gpu`
3. Prefer numerical agreement (two paths) over snapshot tests
4. Name tests after the invariant, not the implementation detail

---

## CI Expectations

- Full CPU suite completes in minutes on MacBook-class hardware
- GPU tests skip gracefully without CUDA
- No network I/O in unit tests (tokenizer fixtures are local or mocked)

---

## Regression Stories (Documented Bugs Tests Prevent)

| Bug class | Test that catches it |
|---|---|
| MoE bias in `parameters()` | `test_bias_not_trainable` |
| Triton silent enable | `test_force_back.py` |
| MTP checkpoint prefix | `test_save_load_with_mtp` |
| uint32 embed crash | training integration casts at boundary |
| KV cache in training | `use_cache=False` in train_step |



## Coverage Map (What Tests Do NOT Cover)

| Area | Gap | Mitigation |
|---|---|---|
| Full 8B data pipeline | Too slow for CI | Manual `prepare_data.py` run |
| 13h training convergence | Too expensive | Monitor loss in production |
| Triton MLA kernel | Not implemented | N/A |
| Multi-GPU | Not implemented | N/A |
| Chat quality / benchmarks | Out of scope | External eval harness |

---

## Running Subsets

```bash
# Fast smoke (< 30s)
pytest tests/test_models.py::TestAuxLossFreeGate -q

# Training loop only
pytest tests/test_training.py -q

# Everything except GPU Triton
pytest tests/ -q --ignore=tests/test_moe_triton.py::TestMoeTritonKernelGPU
```


## References

- `tests/conftest.py` — fixtures
- [getting_started.md](getting_started.md) — quick test commands
- Component docs: [moe.md](moe.md), [MLA.md](MLA.md), [mtp.md](mtp.md)

## Complete Test Inventory

| File | Classes | Focus |
|---|---|---|
| `test_models.py` | ~15 classes | MLA, MoE, MTP, Transformer shapes |
| `test_training.py` | 6 classes | Dataset, Pretrainer, μP, NaN guard |
| `test_inference.py` | 3 classes | generate(), SpeculativeDecoder |
| `test_utils.py` | 5 classes | Checkpoint, memory estimation |
| `test_force_back.py` | 1 class | Triton env-var guard |
| `test_moe_triton.py` | GPU + CPU | Triton vs stacked equivalence |

**Total:** ~86 `def test_*` functions (run `rg -c "def test_" tests/` to verify).

---

## Load-Bearing Test Template

When adding a new invariant, follow this pattern from `test_moe_triton.py`:

```python
def test_new_invariant(self, small_cfg, device):
  model = build_model(small_cfg).to(device)
  # 1. Exercise the code path
  out = model(input_ids)
  # 2. Assert the invariant (shape, value, or property)
  assert not any("gate.bias" in n for n, p in model.named_parameters())
```

**Rules:**
- Use `small_cfg` for speed (dim=64)
- No CUDA required unless `@pytest.mark.gpu`
- Compare against reference implementation when possible

---

## Regression Stories — What Tests Caught

| Bug class | Test that locks it | Doc |
|---|---|---|
| MLA absorption drift | `test_sdpa_and_manual_agree` | [MLA.md](MLA.md) |
| MoE bias in AdamW | `test_bias_not_in_parameters` | [moe.md](moe.md) |
| MTP alignment off-by-one | `test_forward_short_sequence` | [mtp.md](mtp.md) |
| Triton silent enable | `test_force_back_*` | [configs.md](configs.md) |
| Checkpoint MTP prefix | `test_mtp_weights_roundtrip` | [utils.md](utils.md) |
| μP LR formula | `test_mup_lr_scaling` | [training.md](training.md) |

---

## CI Recommendations

```bash
# Fast gate (< 3 min)
python -m pytest tests/ -q --ignore=tests/test_moe_triton.py::TestMoeTritonKernelGPU

# Full gate (GPU runner)
python -m pytest tests/ -q
```

Add new tests to the fast gate unless they require CUDA or Triton.

## test_models.py — Class Inventory

Major test classes (grep `class Test` in file):

| Class | Verifies |
|---|---|
| `TestTransformer` | Construction, forward shape, nested config unwrap |
| `TestMLA` | SDPA vs manual, cache read/write, RoPE |
| `TestDeepSeekMoE` | Routing, expert shapes, bias buffer |
| `TestMTPModule` | Shared head, alignment, short sequences |
| `TestMultiTokenPrediction` | Loss computation, depth>0 |
| `TestAuxLossFreeGate` | Bias is buffer, update_gate_bias |

**Equivalence tests** are the highest value — they catch math bugs that shape tests miss.

---

## test_training.py — Key Tests

| Test | Invariant |
|---|---|
| `test_mup_lr_scaling` | $\eta_{\text{new}} = \eta_{\text{ref}} \sqrt{N_{\text{ref}}/N}$ |
| `test_nan_guard_rollback` | 5 NaN steps → checkpoint restore |
| `test_sharded_cross_boundary` | `__getitem__` stitches shards |
| `test_optimizer_deduplicates` | Tied weights → one Adam state |
| `test_construction_with_mtp` | MTP wrap changes param count |

---

## Writing a New Test — Checklist

1. Pick `small_cfg` unless you need production aspect ratios (`cfg`)
2. Use `device` fixture (CPU in CI)
3. Assert **property**, not implementation detail
4. Name test `test_<behaviour>_<condition>`
5. Add one-line docstring referencing doc section if non-obvious
6. Run `pytest tests/test_yourfile.py -v` before committing

<!-- docs:verified 2026-07-31 · 5a880d2 -->
