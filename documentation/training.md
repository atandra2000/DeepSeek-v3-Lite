# Training — Pretrain Loop, μP, NaN Guard

## A Comprehensive Technical Reference

> **Covers**: `training/pretrain.py` — the full single-GPU BF16 training stack for DeepSeek-V3-Lite.

> **Read this if** you're debugging NaN loss, μP LR, checkpoints, or the train loop. **Skip if** you only need launch commands → [scripts.md](scripts.md).

---

## Table of Contents

1. [Abstract](#abstract)
2. [Training Stack Overview](#training-stack-overview)
3. [TrainingConfig](#trainingconfig)
4. [Pretrainer Lifecycle](#pretrainer-lifecycle)
5. [μP Learning Rate Scaling](#μp-learning-rate-scaling)
6. [LR Scheduler](#lr-scheduler)
7. [PretrainDataset](#pretraindataset)
8. [train_step](#train_step)
9. [MTP Training Path](#mtp-training-path)
10. [MoE Bias Updates](#moe-bias-updates)
11. [NaN Guard](#nan-guard)
12. [Checkpointing](#checkpointing)
13. [CLI Reference](#cli-reference)
14. [Appendix A — Train loop state diagram](#appendix-a--train-loop-state-diagram)
15. [Appendix B — FAQ](#appendix-b--faq)
16. [Appendix C — Glossary](#appendix-c--glossary)
17. [Load-Bearing Invariants](#load-bearing-invariants)
18. [Implementation Checklist](#implementation-checklist)

---

## Abstract

The training system is a **single-GPU, from-scratch PyTorch pretrainer** targeting Chinchilla-optimal training: ~422M parameters on 8.4B tokens. It combines BF16 autocast, FP32 AdamW master weights, `torch.compile(max-autotune)`, gradient checkpointing, μP LR scaling, aux-loss-free MoE bias updates, MTP auxiliary loss, and a NaN guard with automatic checkpoint rollback.

No HuggingFace Trainer. No distributed. One A100 80GB.

### The training problem, formally

Given token sequences $\mathbf{x} = (x_1, \ldots, x_T)$ from the data distribution, we minimise:

$$
\min_\theta \; \mathbb{E}_{\mathbf{x} \sim \mathcal{D}}\left[ -\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t}) + \lambda \sum_{d=1}^{D} \mathcal{L}_{\text{MTP}}^{(d)} \right]
$$

with $\lambda = 0.3$, $D = 1$ at 422M config. The expectation is approximated by stochastic mini-batches over sharded uint32 corpora.

**Single-GPU constraint:** No data parallelism. Effective batch size = `micro_batch_size × gradient_accumulation_steps` sequences.

**Prerequisites:** [foundations.md](foundations.md) §9 (loss), §12 (Chinchilla), §13 (μP).



### Design philosophy — no framework magic

`Pretrainer` is ~440 lines of explicit PyTorch. Every decision is visible:

| Concern | HuggingFace Trainer hides | This repo exposes |
|---|---|---|
| LR schedule | Callback | `make_warmup_cosine_lambda` + `LambdaLR` |
| Mixed precision | `fp16/bf16` flag | `autocast("cuda", dtype=bfloat16)` |
| Gradient accum | `gradient_accumulation_steps` | `loss / grad_accum` in `train_step` |
| MoE balancing | Custom callback | `update_gate_bias` after optim step |
| Checkpoint | `save_steps` | `CheckpointManager` atomic writes |

**Pedagogical goal:** When loss diverges, you can set breakpoints in `train_step` and read every tensor — no opaque trainer state machine.

---

## AdamW — Full Update Rule

Adam (Kingma & Ba, 2015) maintains per-parameter first and second moment estimates:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

Bias-corrected:

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$

**AdamW** (Loshchilov & Hutter, 2019) decouples weight decay from the gradient:

$$
\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t 
\right)
$$

Note: decay is applied to $\theta_t$ directly, not added to $g_t$.

**This repo:** $\beta_1=0.9$, $\beta_2=0.95$, $\lambda=0.1$ on `dim >= 2` params only. Norm scales and MoE bias (`dim < 2`) get $\lambda=0$.

```python
decay_params = [p for p in all_params if p.dim() >= 2]
no_decay_params = [p for p in all_params if p.dim() < 2]
```

**Why exclude 1D params from decay?** Weight decay on LayerNorm/RMSNorm $\gamma$ and router bias destabilises training — standard LLM recipe (GPT-3, LLaMA).

**Fused AdamW:** `fused=True` when CUDA available — single kernel for update, FP32 master weights internally.

---

## Gradient Accumulation — Mathematics

Effective batch size:

$$
B_{\text{eff}} = B_{\text{micro}} \times G_{\text{accum}}
$$

Gradients accumulate over $G_{\text{accum}}$ micro-steps before `optimizer.step()`:

$$

\nabla_\theta \mathcal{L}_{\text{total}} = \frac{1}{G} \sum_{g=1}^{G} 
\nabla_\theta \mathcal{L}_g
$$

**Implementation:** `loss = total_loss / gradient_accumulation_steps` before `backward()`. PyTorch accumulates gradients in `.grad` across micro-steps; `zero_grad(set_to_none=True)` clears after optim step.

**422M:** $B_{\text{micro}}=8$, $G=4$ → $B_{\text{eff}}=32$ sequences. Each optim step sees $32 \times 2048 = 65536$ tokens.

**Micro-step vs optim-step counters:**
- `global_step` in `train()` — counts every micro-batch (used for logging cadence)
- `_opt_steps` — counts actual `optimizer.step()` calls (used for MoE bias updates, scheduler)

---

## torch.compile — What It Does Here

When `compile_model=True`:

```python
training_model = torch.compile(training_model, mode="max-autotune", fullgraph=False)
```

**TorchInductor** traces the forward graph, fuses elementwise ops, and autotunes CUDA kernels. `max-autotune` spends more compile time benchmarking kernel variants.

**Trade-offs:**
| Benefit | Cost |
|---|---|
| 10-20% faster step time | 5-15 min compile on first run |
| Fused RMSNorm+matmul patterns | Opaque stack traces on error |
| Less Python overhead in MoE loop | Higher peak VRAM during compile |

**`fullgraph=False`:** Allows graph breaks (Python MoE dispatch, dynamic shapes) — required for this codebase.

**Env:** `TORCH_COMPILE_MODE=max-autotune` set in `launch_a100.sh`.

**1650 config:** `compile: false` — 4GB VRAM cannot absorb compile workspace.

---

## DataLoader Design

```python
loader = DataLoader(
    dataset, batch_size=8, shuffle=True, generator=g,
    num_workers=8, pin_memory=True,
    persistent_workers=True, prefetch_factor=8, drop_last=True,
)
```

| Flag | Rationale |
|---|---|
| `shuffle=True` + seeded `Generator` | Different order each epoch; reproducible given same seed |
| `num_workers=8` | Parallel `__getitem__` while GPU trains |
| `pin_memory=True` | Faster H2D copy via page-locked host memory |
| `persistent_workers=True` | Avoid worker respawn overhead each epoch |
| `prefetch_factor=8` | Pipeline 8 batches per worker ahead of GPU |
| `drop_last=True` | Avoid partial batch shape mismatch at epoch end |

**Resume caveat:** Sampler RNG is **not** checkpointed — resume changes data order (benign at 8.4B scale).

---

## Memory Timeline — One train_step

```
1. DataLoader delivers (tokens, targets) uint32 on CPU
2. .to(device, non_blocking=True) → GPU int64 cast in train_step
3. autocast forward:
     embed → 18 layers (checkpoint recompute in backward) → logits
     MTP: auxiliary heads if enabled
4. loss.backward() — activation recompute from checkpoints
5. clip_grad_norm_(max=1.0)
6. optimizer.step() — FP32 master update
7. scheduler.step()
8. zero_grad(set_to_none=True)
9. MoE bias update (reads _last_indices from last forward)
```

Peak memory occurs during backward recompute — typically layer 9-12 middle of stack. `estimate_model_memory_gb` uses PaLM factor 24 for this.

---

## Chinchilla Epoch Analysis

Unique corpus: ~8.4B tokens. Samples per epoch:

$$
N_{\text{samples}} = \frac{N_{\text{tokens}} - 1}{\text{max_seq_len}} \approx \frac{8.4 \times 10^9}{2048} \approx 4.1 \times 10^6
$$

With $B_{\text{eff}}=32$: $\approx 128000$ optim steps per epoch.

At 512,000 total optim steps: **~4 epochs** over the corpus (with shuffle reshuffling boundaries each epoch).

Total token **exposures**: $512000 \times 65536 \approx 33.5$B — intentional over-training relative to unique tokens improves convergence on rare mixture sources (code, math).


---

## Training Stack Overview

| Layer | Technology | Purpose |
|---|---|---|
| Matmul | TF32 (`allow_tf32=True`) | ~8× faster FP32 accum on Ampere |
| Compile | `torch.compile(mode=max-autotune)` | Kernel fusion, ~10–20% speedup |
| Attention | FA2 via `F.scaled_dot_product_attention` | Memory-efficient attention |
| Precision | BF16 autocast | No GradScaler needed |
| Optimiser | AdamW fused, FP32 master | Stable updates in low precision |
| Memory | Gradient checkpointing | ~3× activation savings |
| MoE balance | Out-of-band bias updates | No aux loss in task gradient |
| Stability | NaN guard + ckpt rollback | Auto-recover from divergence |

---

## TrainingConfig

`@dataclass` mapping YAML `training:` section:

| Field | 422M YAML | Description |
|---|---|---|
| `micro_batch_size` | 8 | Tokens per GPU per micro-step |
| `gradient_accumulation_steps` | 4 | Micro-steps per optimizer step |
| `total_steps` | 512,000 | Optimizer steps (not micro-steps) |
| `warmup_steps` | 2,000 | Linear LR warmup |
| `lr` | 8.0e-4 | Base LR (μP-adjusted at init) |
| `min_lr_ratio` | 0.05 | Floor as fraction of peak LR |
| `weight_decay` | 0.1 | On dim≥2 params only |
| `grad_clip` | 1.0 | Global norm clip |
| `grad_checkpoint` | true | Activation checkpointing |
| `compile` | true | torch.compile |
| `mtp_weight` | 0.3 | From `mtp_loss_weight` in model section |
| `bias_update_every` | 1 | MoE bias cadence |
| `nan_guard` | true | Enable rollback |
| `mup_lr` | true | μP scaling |

**Effective batch size:** `8 × 4 = 32` sequences per optimizer step.

**Tokens per optimizer step:** `32 × 2048 = 65,536`.

**Total training tokens:** `512,000 × 65,536 ≈ 33.5B` micro-token exposures (with reshuffling; Chinchilla target is 8.4B unique tokens over multiple epochs).

---

## Pretrainer Lifecycle

### `__init__`

```
1. Seed RNG (config.seed)
2. Enable TF32 + cudnn.benchmark (CUDA only)
3. init_logging(log_interval, seq_len)
4. Build Transformer(use_checkpoint=grad_checkpoint)
5. count_parameters → log breakdown
6. If mtp_depth > 0 and mtp_weight > 0:
     wrap with MultiTokenPrediction
7. torch.compile(training_model) if enabled
8. μP LR scaling if mup_lr
9. AdamW with param dedup (decay split by dim)
10. LambdaLR scheduler (warmup cosine)
11. CheckpointManager
```

### `train()`

```
1. PretrainDataset + DataLoader (shuffle, 8 workers, pin_memory)
2. Auto-resume from latest checkpoint if present
3. Loop: for each micro-step:
     train_step → log → save → nan_guard check
4. Final checkpoint with tag="final"
```

### Optimiser param groups

```python
decay_params = [p for p in all_params if p.dim() >= 2]    # weights
no_decay_params = [p for p in all_params if p.dim() < 2]  # bias, norm γ
```

Shared/tied params deduplicated by `id(p)` before grouping.

---

## μP Learning Rate Scaling

**Maximal Update Parametrisation (μP)** transfers hyperparameters across model scales:

```
new_lr = mup_lr_reference × (mup_lr_reference_params / total_params)^0.5
```

422M config:
- Reference: `6.0e-4` at `757,226,496` params (~757M reference model)
- Total counted **after** MTP wrap (includes MTP head params)
- Result: **~8.07e-4**

**Why sqrt?** μP theory: optimal LR scales inversely with sqrt(width) for width-dependent init schemes. The reference model was tuned at 757M; this formula transfers that tuning to 422M.

`test_mup_lr_scaling` verifies the computation.

---

## LR Scheduler

```python
make_warmup_cosine_lambda(warmup_steps=2000, total_steps=512000, min_lr_ratio=0.05)
```

| Phase | Steps | LR multiplier |
|---|---|---|
| Warmup | 0 → 2000 | linear 0 → 1 |
| Cosine | 2000 → 512000 | cosine 1 → 0.05 |
| Floor | > 512000 | 0.05 |

Bound to `LambdaLR` — stepped once per optimizer step (not micro-step).

---

## PretrainDataset

Packed contiguous token windows from pre-tokenised data.

### Single-file layout

```
data/pretrain_data.bin  →  torch tensor (N,) uint32
sample i: tokens = data[i*S : i*S+S], targets = data[i*S+1 : i*S+S+1]
```

### Sharded layout

```
data/pretrain_chinchilla/
  shard_000.bin  (50M tokens each)
  shard_001.bin
  ...
```

- `_locate(global_idx)` — binary search over `shard_offsets`
- Cross-shard windows: stitch pieces with `torch.cat`
- `mmap=True` for memory efficiency
- Final partial chunk **dropped** (not padded)

### Dtype boundary

Dataset stores `uint32` (4 bytes/token). `train_step` casts to `int64` before `nn.Embedding` and `cross_entropy`.

---

## train_step

```python
def train_step(tokens, targets, micro_step) -> metrics | None
```

1. Cast tokens/targets to long
2. BF16 autocast forward:
   - With MTP: `model(tokens)` → `compute_loss`
   - Without: `model(tokens, use_cache=False)` → CE
3. Compute `balance_loss` (logging only, on device)
4. NaN check → skip backward if bad
5. `loss / grad_accum` → `backward()`
6. On optimizer step boundary:
   - `clip_grad_norm_(max=1.0)`
   - `optimizer.step()`, `scheduler.step()`, `zero_grad`
   - MoE bias update if `opt_steps % bias_update_every == 0`
7. Return `{loss, mtp_loss, balance_loss}`

**Micro-step vs optimizer-step:** `global_step` in `train()` counts micro-steps. Optimizer steps tracked in `_opt_steps`.

---

## MTP Training Path

Enabled when `mtp_depth > 0` AND `mtp_weight > 0`:

```python
mtp_model = MultiTokenPrediction(config.model_config, raw_model)
training_model = torch.compile(mtp_model)
```

Forward: `(main_logits, mtp_pairs) = model(tokens)`

Loss: `total = main_loss + 0.3 * mean(mtp_depth_losses)`

Only `total / grad_accum` is backpropped. Main and MTP share trunk gradients through the shared embed/head.

---

## MoE Bias Updates

```python
if self._opt_steps % self.config.bias_update_every == 0:
    for moe in self.raw_model.moe_layers():
        moe.update_gate_bias(speed=self.config.bias_update_speed)
```

Uses routing counts from the **last forward's** `_last_indices`. With `bias_update_every=1`, every optimizer step updates bias based on that step's routing distribution.

See [moe.md](moe.md) for the full bias mechanism.

---

## NaN Guard

```python
if nan_guard and (isnan(loss) or isinf(loss)):
    skip backward, increment streak
    if streak >= 5:
        load latest checkpoint, reset streak
```

**Never disable without explicit user consent** (AGENTS.md hard rule).

Rollback restores: model weights, optimizer state, scheduler state, `opt_steps` counter.

If no checkpoint exists and NaN persists → `RuntimeError`.

---

## Checkpointing

```python
save_checkpoint(step):
  state = raw_model.state_dict()
  drop head.weight if weight_tying
  inject mtp.* keys if MTP enabled
  meta = {scheduler, opt_steps, config, has_mtp}
  ckpt_manager.save(...)
```

Auto-resume on `train()` start via `latest_step()`.

See [utils.md](utils.md) for atomic write details.

---

## CLI Reference

```bash
python training/pretrain.py \
  --config configs/pretrain_a100_422m.yaml \
  --data-path data/pretrain_chinchilla \
  --checkpoint-dir checkpoints/pretrain_a100 \
  --resume 4000 \
  --no-compile \
  --no-checkpoint
```

| Flag | Effect |
|---|---|
| `--config` | YAML path (default: 422M) |
| `--data-path` | Override `training.data_path` |
| `--checkpoint-dir` | Override save directory |
| `--resume N` | Load step N before training |
| `--no-compile` | Disable torch.compile |
| `--no-checkpoint` | Disable gradient checkpointing |

---


## Pretrainer.__init__ — Line-by-Line Walkthrough

`Pretrainer.__init__` (`training/pretrain.py:130-198`) is the single place where hardware, model, optimiser, and checkpoint subsystems are wired. Reading it top-to-bottom is the fastest way to understand the training stack.

| Line block | What happens | Why it matters |
|---|---|---|
| `torch.manual_seed` | Seeds CPU RNG before model init | Reproducible weight init + DataLoader shuffle |
| TF32 + cudnn flags | Enables fast matmul on Ampere | ~8× FP32 accum throughput; no accuracy change at BF16 |
| `init_logging` | Sets up `TrainingLogger` cadence | `log_every` controls GPU sync frequency |
| `Transformer(..., use_checkpoint=...)` | Builds **raw** trunk | Checkpointing toggled here, not in YAML model section |
| `count_parameters` | Deduped param tally | μP denominator; logged at startup |
| MTP wrap | `MultiTokenPrediction(raw_model)` if depth>0 and weight>0 | Changes `training_model` and **total** param count |
| `torch.compile` | Wraps `training_model` only | `raw_model` stays eager for save/bias |
| μP LR | `lr *= sqrt(N_ref / N_total)` | Applied **after** MTP wrap |
| AdamW param groups | `dim>=2` decay, `dim<2` no decay | Keeps RMSNorm γ and MoE bias stable |
| `LambdaLR` scheduler | Warmup+cosine lambda | Steps on **optim** steps, not micro-steps |
| `CheckpointManager` | Points at `save_dir` | Atomic safetensors writes |

**Critical split:** `self.model` may be compiled + MTP-wrapped; `self.raw_model` is always the bare `Transformer` used for checkpoint I/O and MoE bias updates.

---

## train_step — Pseudocode with Tensor Shapes

Inputs: `tokens (B, S)`, `targets (B, S)` with `B=8`, `S=2048` at 422M.

```
1. Cast tokens/targets uint32 → int64 if needed
2. autocast(BF16):
     if MTP:
         main_logits (B,S,V), mtp_pairs = model(tokens)
         total_loss, main_loss, mtp_loss = mtp_wrapper.compute_loss(...)
     else:
         logits (B,S,V) = model(tokens, use_cache=False)
         main_loss = CE(logits, targets)
3. balance_loss = sum(moe.get_load_balance_loss())  # tensor, no .item() yet
4. loss = total_loss / grad_accum
5. if nan_guard and isnan(loss): return None
6. loss.backward()
7. if (micro_step+1) % grad_accum == 0:
       clip_grad_norm_(max=1.0)
       optimizer.step(); scheduler.step(); zero_grad()
       opt_steps += 1
       if opt_steps % bias_update_every == 0: update_gate_bias()
8. return {loss tensors for logging}
```

**Shape invariants:**
- `V = vocab_size = 100_018`
- CE flattens to `(B*S, V)` vs `(B*S,)`
- MTP path never enables KV cache (`use_cache=False` inside trunk forward)

---

## Checkpoint Format — File-by-File

Each save at step `N` produces three files under `checkpoints/pretrain/`:

| File | Format | Contents |
|---|---|---|
| `model_step_N.safetensors` | SafeTensors | All `Transformer` weights; `mtp.mtp_modules.*` if MTP; **drops** duplicate `head.weight` when tied |
| `optim_step_N.pt` | PyTorch pickle | AdamW moments (FP32 master) |
| `meta_step_N.json` | JSON | `step`, `opt_steps`, `scheduler` state, `config` snapshot, `has_mtp`, `tag` |

**Atomic write pattern:** write temp file in same directory → `os.replace` — crash mid-write never leaves a torn checkpoint.

**Load path (`load_checkpoint`):**
1. `CheckpointManager.load(raw_model, step)` restores trunk weights (`strict=False` tolerates missing tied head)
2. If `has_mtp`, strip `mtp.` prefix and load into `mtp_wrapper`
3. Restore scheduler + `opt_steps` from meta

**Resume semantics:** `global_step` in `train()` loop equals micro-step counter after resume; DataLoader shuffle order is **not** restored.

---

## NaN Guard — State Machine

```
                    ┌─────────────┐
                    │  train_step │
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              │ loss finite?            │
              └────────────┬────────────┘
                     yes   │   no
                      ▼    │    ▼
                 streak=0  │  streak++
                      │    │    │
                      │    │    ▼
                      │    │ streak >= 5?
                      │    │    yes → load latest ckpt, reset streak
                      │    │    no  → skip backward, continue
                      ▼    ▼
                   normal path
```

**Why 5 consecutive?** Single bad micro-batches happen from MoE routing spikes; five in a row signals true divergence (LR too high, corrupt shard, dtype bug).

**422M default:** `nan_guard: true`, `nan_guard_max_consecutive: 5` in YAML.

---

## Worked Example — One Optimiser Step at 422M

**Config recap:** `micro_batch=8`, `grad_accum=4`, `seq_len=2048`, `lr≈8.07e-4` (μP).

| Quantity | Value |
|---|---|
| Tokens per optim step | $8 \times 4 \times 2048 = 65536$ |
| Main CE targets | $65536$ cross-entropy terms |
| MTP auxiliary targets | $\approx 65536 - 8 \times 4 \times 2$ (tail trim) |
| MoE layers updated | 16 (`layers 2–17`) |
| Experts evaluated/token | 5 (4 routed + 1 shared) |
| Grad clip | global norm capped at 1.0 |
| MoE bias update | every `bias_update_every=1` optim steps |

**Logging at step 100:** one `.item()` sync for CE (+ optional MTP), balance_loss tensor deferred to log interval — avoids 3–4 GPU stalls per micro-step.

---

## torch.compile Interaction with MTP and MoE

`torch.compile(training_model)` traces the **outer** forward including MTP heads. Graph breaks occur at:
- Python MoE `stacked` dispatch loop (per-expert `forward`)
- Dynamic `seqlen` if ever changed mid-run (not in this trainer — fixed 2048)

**Debugging tip:** pass `--no-compile` to get Python stack traces on failure. Re-enable compile only after the eager run is stable.

**Saved weights:** always extracted from `raw_model` / `mtp_wrapper._orig_mod` — compiled wrapper state is never checkpointed.



## Appendix A — Train loop state diagram

```
START
  │
  ├─ load checkpoint? ──yes──► restore weights/optim/sched/opt_steps
  │
  ▼
EPOCH LOOP (shuffle with seeded Generator)
  │
  ▼
MICRO-STEP: fetch batch → train_step
  │
  ├─ NaN? ──yes──► streak++ ──≥5?──► rollback checkpoint
  │
  ├─ log_every? ──► print loss/ppl/lr/tps/balance_loss
  │
  ├─ save_every? ──► atomic checkpoint
  │
  └─ global_step++ until max_steps
  │
  ▼
FINAL checkpoint (tag="final")
```

---

## Appendix B — FAQ

**Q: Why 512K steps with batch 32?** Provides multiple epochs over 8.4B tokens with shuffling. Exact epoch count depends on shard size.

**Q: Why no GradScaler?** BF16 has sufficient dynamic range on Ampere/Blackwell for this model scale.

**Q: Does compile work on CPU?** Disabled automatically when CUDA unavailable; smoke tests run eager.

**Q: Resume exact data order?** No — DataLoader shuffle RNG is not checkpointed. Benign at this scale.

---

## Appendix C — Glossary

| Term | Meaning |
|---|---|
| `micro_step` | One forward+backward (may not trigger optimiser) |
| `opt_steps` / `_opt_steps` | Optimizer step counter |
| `grad_accum` | Micro-steps accumulated before optimiser step |
| `nan_guard_streak` | Consecutive NaN/Inf micro-steps |
| `raw_model` | Uncompiled Transformer (for save/bias update) |
| `training_model` | Possibly compiled MTP wrapper |

---

## Load-Bearing Invariants

1. **`use_cache=False` in train_step** — training never writes KV cache.
2. **μP counts post-MTP total** — slight LR inflation vs bare Transformer.
3. **YAML `bias_update_every` wins over dataclass default (10)**.
4. **uint32→int64 cast at train boundary** — not in dataset.
5. **NaN guard never disabled by default** in 422M YAML.
6. **`raw_model` for checkpoint/bias** — not the compiled wrapper.

---

## Implementation Checklist

- [ ] `pytest tests/test_training.py` passes
- [ ] `test_mup_lr_scaling` — μP formula
- [ ] `test_nan_guard_rollback` — recovery path
- [ ] `test_checkpoint_roundtrip` — save/load integrity
- [ ] `test_moe_bias_update_during_training` — bias moves

---

## References

- [training/pretrain.py](../training/pretrain.py)
- [moe.md](moe.md) — bias mechanism
- [mtp.md](mtp.md) — auxiliary loss
- [utils.md](utils.md) — checkpoint format
- [configs.md](configs.md) — YAML reference

## `train()` Main Loop — Pseudocode

```python
global_step = resume_from_checkpoint_or_0()
nan_guard_streak = 0
while global_step < max_steps:
    for tokens, targets in DataLoader:
        tokens, targets = tokens.to(device), targets.to(device)
        metrics = train_step(tokens, targets, global_step)
        if metrics is None:          # NaN path
            handle_nan_guard()
            continue
        if global_step % log_every == 0:
            logger.log(global_step, loss, lr, metrics)
        if global_step % save_every == 0 and global_step > 0:
            save_checkpoint(global_step)
        global_step += 1
save_checkpoint(global_step, tag="final")
```

**Note:** `global_step` counts **micro-steps**; `_opt_steps` counts optimizer steps. MoE bias uses `_opt_steps`.

---

## Optimiser Param Groups — Why Split?

```python
decay_params = [p for p in all_params if p.dim() >= 2]
no_decay_params = [p for p in all_params if p.dim() < 2]
```

| Param type | dim | Weight decay | Examples |
|---|---|---|---|
| Matrices | ≥ 2 | 0.1 | Linear weights, embeddings |
| Vectors | 1 | 0.0 | RMSNorm γ, biases |
| Scalars | 0 | 0.0 | (none in this repo) |

MoE `gate.bias` is a **buffer** — not in either group.

---

## `compile_model` Interaction

```python
training_model = torch.compile(training_model, mode=compile_mode, fullgraph=False)
self.raw_model = raw_model  # NOT compiled
```

| Operation | Uses |
|---|---|
| `train_step` forward/backward | `self.model` (compiled) |
| `save_checkpoint` | `self.raw_model` |
| `update_gate_bias` | `self.raw_model.moe_layers()` |
| `load_checkpoint` | `self.raw_model` + separate MTP load |

**Why split?** `torch.compile` wraps module; `state_dict` keys may gain `_orig_mod.` prefix — saving `raw_model` avoids this.

---

## Hyperparameter Sensitivity (Empirical)

| Knob | Too high | Too low |
|---|---|---|
| `lr` | NaN guard fires | Underfitting, slow convergence |
| `grad_clip` | Rarely binding | Occasional spikes destabilise MoE |
| `mtp_weight` | MTP dominates CE | Weak draft, low speculative acceptance |
| `bias_update_speed` | Router oscillates | Expert collapse |
| `warmup_steps` | Wasted compute | Early instability |

---

## Worked Example — Micro-step vs Optim Step

`grad_accum=4`, `global_step` 0–3:

| global_step | backward? | optimizer.step? | _opt_steps |
|---|---|---|---|
| 0 | yes | no | 0 |
| 1 | yes | no | 0 |
| 2 | yes | no | 0 |
| 3 | yes | **yes** | 1 |

Loss scaled by `1/4` each micro-step so gradient magnitude matches batch of 32 sequences.

## Logging and Monitoring

### Console output format

```
step=   4050 | loss=2.8123 | ppl=16.64 | lr=7.91e-04 | tps=131072 | balance_loss=0.0008 | mtp_loss=2.9451
```

| Metric | Healthy trend | Warning sign |
|---|---|---|
| `loss` | Down over 10k steps | Flat or rising after 50k |
| `ppl` | $\exp(\text{loss})$ decreasing | Stuck > 100 mid-run |
| `lr` | Warmup then cosine | Stuck at 0 |
| `tps` | Stable after compile warmup | Drops > 30% suddenly |
| `balance_loss` | Small, stable | Spikes → routing instability |
| `mtp_loss` | Tracks main loss | >> main loss → reduce `mtp_weight` |

### WandB integration

Set before launch:

```bash
export WANDB_PROJECT=deepseek-v3-lite-a100
export WANDB_RUN_NAME=422m-run1
```

Metrics: `train/loss`, `train/ppl`, `train/lr`, `train/tokens_per_sec`, `train/mtp_loss`, `train/balance_loss`.

---

## Resume and Fault Tolerance

**Auto-resume:** `train()` calls `latest_step()` on startup if checkpoints exist.

**Manual resume:**

```bash
python training/pretrain.py --config configs/pretrain_a100_422m.yaml --resume 40000
```

**Not restored:** DataLoader shuffle order (benign at 8.4B scale).

**Restored:** Model weights, AdamW state, scheduler, `opt_steps`, MTP weights if `has_mtp: true`.

<!-- docs:verified 2026-07-31 · 5a880d2 -->
