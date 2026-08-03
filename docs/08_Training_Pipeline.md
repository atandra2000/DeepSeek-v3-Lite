# Training — Pretrain Loop, μP, NaN Guard

> **Canonical** for pretrain loop, μP, NaN guard, and YAML reference.

> **Read this if** you're debugging NaN loss, μP LR, checkpoints, or the train loop. **Skip if** you only need launch commands → [[Docs/02_Model_Architecture|Model Architecture]].

**Depends on:** [[Docs/02_Model_Architecture|Model Architecture]], [[Docs/04_DeepSeekMoE|MoE]] · **Read next:** [[Docs/09_Data_Pipeline|Data Prep]], [[Docs/10_Inference_and_Serving|Inference]]

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

**Prerequisites:** [[Docs/01_Foundations|Foundations]] §9 (loss), §12 (Chinchilla), §13 (μP).



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

See [[Docs/04_DeepSeekMoE|MoE]] for the full bias mechanism.

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

See [[Docs/08_Training_Pipeline|configs]] for atomic write details.

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
- [[Docs/04_DeepSeekMoE|MoE]] — bias mechanism
- [[Docs/05_Multi_Token_Prediction|MTP]] — auxiliary loss
- [[Docs/08_Training_Pipeline|configs]] — checkpoint format
- [[Docs/08_Training_Pipeline|training]] — YAML reference

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

---

## Part B — Configuration Reference

> Absorbed from the former configs encyclopedia. Single canonical YAML reference for this project.

> **Purpose:** Textbook-style reference for every YAML key in `configs/`, with theory for *why* each hyperparameter exists and *where* code consumes it.

> **Read this if** you're tuning hyperparameters or adding config keys. **Skip if** you're learning architecture theory → [[Docs/02_Model_Architecture|Architecture]].
---

## Table of Contents

1. [Overview](#overview)
2. [pretrain_a100_422m.yaml — Canonical Recipe](#pretrain_a100_422myaml--canonical-recipe)
3. [pretrain_1650_2m.yaml — Smoke Test](#pretrain_1650_2myaml--smoke-test)
4. [Model Keys](#model-keys)
5. [Training Keys](#training-keys)
6. [Data Keys](#data-keys)
7. [Triton Dispatch Keys](#triton-dispatch-keys)
8. [Config Nesting](#config-nesting)
9. [Appendix — Quick comparison table](#appendix--quick-comparison-table)

---

## Overview

Configs are YAML files with three top-level sections:

```yaml
model:      # Architecture — consumed by Transformer, MLA, MoE, MTP
training:   # Loop hyperparameters — consumed by Pretrainer
data:       # Paths — consumed by Pretrainer and inference/generate.py
```

`Pretrainer` reads the full YAML into `TrainingConfig.model_config` (the entire dict). `Transformer` unwraps `config.get("model", config)`.

### Why YAML instead of Python dataclasses?

Hyperparameters are **experiment surface area** — you will sweep LR, batch size, and MoE width without touching code. YAML provides:

1. **Git-diffable recipes** — `pretrain_a100_422m.yaml` is a reproducible artifact
2. **No import side effects** — changing config does not re-run module-level code
3. **Nested sections** — `model` / `training` / `data` mirror mental model

`Pretrainer.main()` maps YAML → `TrainingConfig` dataclass at runtime (`training/pretrain.py:main`). Tests may use flat dicts via `conftest.py` fixtures — both shapes work via `config.get("model", config)` unwrapping.


---

## pretrain_a100_422m.yaml — Canonical Recipe

**Target:** 1× A100 80GB, ~422M params, 8.4B Chinchilla tokens, 13–15 h wall.

| Section | Highlights |
|---|---|
| model | 18 layers, vocab 100018, dim 768, 16 MoE layers |
| training | 512K steps, μP LR, nan_guard, compile, grad_checkpoint |
| data | `data/pretrain_chinchilla`, deepseek tokenizer |

---

## pretrain_1650_2m.yaml — Smoke Test

**Target:** GTX 1650 4GB end-to-end validation.

| Difference from 422M | Value |
|---|---|
| vocab | 50257 (GPT-2) |
| dim | 64 |
| n_layers | 4 (2 dense + 2 MoE) |
| n_routed_experts | 4 |
| n_activated_experts | 1 |
| max_seq_len | 128 |
| compile | false |
| grad_checkpoint | false |
| mup_lr | false |

Preserves all architectural invariants (MLA, aux-loss-free MoE, MTP) at tiny scale.

---

## Model Keys

| Key | 422M | Read by | Description |
|---|---|---|---|
| `vocab_size` | 100018 | Embedding, LM head | Must match tokenizer |
| `dim` | 768 | All layers | Hidden dimension |
| `n_layers` | 18 | Transformer | Total transformer blocks |
| `n_heads` | 12 | MLA, MTP MHA | Attention heads |
| `n_dense_layers` | 2 | TransformerBlock | Dense before MoE |
| `n_routed_experts` | 20 | DeepSeekMoE | Routed expert count |
| `n_shared_experts` | 1 | DeepSeekMoE | Always-on experts |
| `n_activated_experts` | 4 | AuxLossFreeGate | Top-k routing |
| `inter_dim` | 1536 | SwiGLUFFN | Dense FFN width |
| `moe_inter_dim` | 384 | Expert | MoE expert width |
| `kv_lora_rank` | 192 | MLA | KV compression dim |
| `q_lora_rank` | 0 | MLA | Query compression (off) |
| `qk_nope_head_dim` | 48 | MLA | Content QK dim/head |
| `qk_rope_head_dim` | 24 | MLA | RoPE QK dim/head |
| `v_head_dim` | 64 | MLA | Value dim/head |
| `max_seq_len` | 2048 | MLA cache, dataset | Training sequence length |
| `rope_theta` | 10000 | MLA | RoPE base frequency |
| `rope_factor` | 1.0 | MLA | YaRN scale (1.0=off) |
| `mscale` | 1.0 | MLA | Attention temp scale |
| `mtp_depth` | 1 | MultiTokenPrediction | MTP heads |
| `mtp_loss_weight` | 0.3 | Pretrainer, compute_loss | λ for MTP loss |
| `dtype` | bf16 | Documentation | Training precision |
| `attn_impl` | sdpa | MLA | sdpa / manual / triton |
| `moe_dispatch` | stacked | DeepSeekMoE | stacked / triton_grouped |
| `weight_tying` | true | Transformer | Share embed/head |


### Dimensional constraints (422M)

These keys are not independent — violating a constraint crashes at first forward or silently mis-trains:

| Constraint | Rule | Example violation |
|---|---|---|
| Head divisibility | $d \bmod H = 0$ | dim=770, n_heads=12 |
| MLA head dims | $q_{\mathrm{k,nope}} + q_{\mathrm{k,rope}} = d_h$ | 48+24=72, 768/12=64 — **must match** |
| Vocab = embed rows | `vocab_size` = tokenizer len | 100000 vs 100018 |
| MoE width | experts use `moe_inter_dim` | dense layers use `inter_dim` |
| Triton MoE limit | `moe_inter_dim ≤ 256` for triton_grouped | 384 → auto-fallback stacked |

---

## Training Keys

| Key | 422M | Description |
|---|---|---|
| `micro_batch_size` | 8 | Per-GPU batch |
| `gradient_accumulation_steps` | 4 | Micro-steps per optim step |
| `total_steps` | 512000 | Optimiser steps |
| `warmup_steps` | 2000 | LR warmup |
| `lr` | 8.0e-4 | Base LR (μP-adjusted) |
| `min_lr_ratio` | 0.05 | Cosine floor |
| `weight_decay` | 0.1 | AdamW WD |
| `beta1`, `beta2` | 0.9, 0.95 | AdamW betas |
| `grad_clip` | 1.0 | Global norm clip |
| `grad_checkpoint` | true | Activation checkpointing |
| `compile` | true | torch.compile |
| `save_interval` | 4000 | Steps between checkpoints |
| `log_interval` | 50 | Steps between log lines |
| `mup_lr` | true | Enable μP scaling |
| `mup_lr_reference` | 6.0e-4 | Reference LR |
| `mup_lr_reference_params` | 757226496 | Reference param count |
| `nan_guard` | true | NaN rollback |
| `nan_guard_max_consecutive` | 5 | Streak before rollback |
| `bias_update_speed` | 0.001 | MoE bias step size |
| `bias_update_every` | 1 | Optim steps between bias updates |
| `save_dir` | checkpoints/pretrain_a100 | Checkpoint directory |

### Training budget arithmetic

Derive total token exposures from YAML:

$$
N_{\text{tokens/step}} = B_{\text{micro}} \times G_{\text{accum}} \times S = 8 \times 4 \times 2048 = 65536
$$

$$
N_{\text{total exposures}} = N_{\text{tokens/step}} \times T_{\text{steps}} = 65536 \times 512000 \approx 33.5 \times 10^9
$$

Unique corpus size ≈ 8.4B → **~4 epochs** over the mixture (with shuffle each epoch). Chinchilla-optimal **unique** tokens ≈ $20 \times 422\text{M} \approx 8.4$B — the corpus size matches; multi-epoch training is intentional for a single-GPU budget.


---

## Data Keys

| Key | 422M | Description |
|---|---|---|
| `train_data_path` | data/pretrain_chinchilla | Shard directory or .bin file |
| `tokenizer_path` | deepseek-ai/deepseek-coder-v2-lite | HF tokenizer for inference |

---

## Triton Dispatch Keys

| Key | Triton value | PyTorch default | Env guard |
|---|---|---|---|
| `attn_impl` | triton | sdpa | `ENABLE_TRITON_KERNELS=1` |
| `moe_dispatch` | triton_grouped | stacked | `ENABLE_TRITON_KERNELS=1` |

Without env var, `enforce_triton_env_var` force-backs at `Transformer.__init__` with a single warning.

### μP keys — full derivation

When `mup_lr: true`:

```python
new_lr = mup_lr_reference * (mup_lr_reference_params / total_params) ** 0.5
```

| Key | 422M value | Meaning |
|---|---|---|
| `mup_lr_reference` | 6.0e-4 | LR known-good on reference model |
| `mup_lr_reference_params` | 757,226,496 | Param count of reference (~DeepSeek-V3 class) |
| `lr` (YAML) | 8.0e-4 | Overwritten at init if μP enabled |

**Intuition:** Wider models have more parameters contributing to each update; sqrt scaling keeps effective update magnitude stable. See [[Docs/01_Foundations|Foundations]] §13.


**422M note:** `moe_inter_dim=384` and `dim=768` exceed Triton MoE kernel limit (256). Falls back to stacked at runtime.

---

## Config Nesting

Both forms work:

```yaml
# Nested (canonical YAML)
model:
  dim: 768
training:
  lr: 8e-4

# Flat (tests)
dim: 768
lr: 8e-4
```

`Transformer`: `config.get("model", config)`
`Pretrainer`: passes full YAML as `model_config`; sub-components unwrap as needed.

---

## Appendix — Quick comparison table

| | 422M A100 | 1650 2M |
|---|---|---|
| Params | ~422M | ~2M |
| VRAM | ~35 GB | <4 GB |
| Vocab | 100018 | 50257 |
| MoE layers | 16 | 2 |
| Experts | 20 routed, top-4 | 4 routed, top-1 |
| Seq len | 2048 | 128 |
| Compile | yes | no |
| Purpose | Production training | Smoke / CI |

## Worked example — tracing a config key

**Question:** What happens when you set `model.mtp_depth: 1` and `model.mtp_loss_weight: 0.3`?

1. `Pretrainer.__init__` reads `mtp_depth` from nested model config
2. If `mtp_depth > 0` and `mtp_weight > 0`, wraps `Transformer` in `MultiTokenPrediction`
3. `train_step` calls `mtp_wrapper.forward` → `compute_loss` with λ=0.3
4. μP scaling uses **post-wrap** param count
5. Checkpoint saves `mtp.mtp_modules.*` keys with `has_mtp: true` in meta

Trace in: `training/pretrain.py:155-163`, `models/mtp.py:67-117`.


---



---

## Hyperparameter Rationale (422M)

### Why `dim=768`, `n_layers=18`?

Chinchilla-optimal sizing: for ~422M params, depth and width balance FLOPs and memory. 18 layers fits in A100 VRAM with MoE at batch 8 × seq 2048.

### Why `n_dense_layers=2`?

DeepSeek-V3 uses early dense layers for stable low-level features before sparse routing. Matches paper schedule.

### Why `n_routed_experts=20`, `n_activated_experts=4`?

20 experts × 384 dim stores capacity; top-4 activates ~20% of routed compute. +1 shared expert always runs (5 FFN evaluations per MoE token).

### Why `lr=8e-4` (μP-adjusted)?

Base YAML value; overwritten at init to ~8.07e-4 when `mup_lr: true`. Tuned for stable BF16 training with grad clip 1.0.

### Why `warmup_steps=2000`?

~0.4% of total micro-steps. Sufficient for Adam moment estimates to stabilise without wasting compute.

### Why `min_lr_ratio=0.05`?

Floor LR = 5% of peak during cosine tail — prevents LR hitting zero while still allowing late-stage fine adjustment.

### Why `grad_clip=1.0`?

Standard for LLM pretraining. MoE routing spikes can produce large gradients; clip prevents single-step destabilisation.

### Why `mtp_loss_weight=0.3`?

DeepSeek-V3 uses λ ∈ [0.1, 0.3]. 0.3 gives meaningful auxiliary signal without overpowering main CE loss.

### Why `bias_update_speed=0.001`?

Small steps on router bias buffer — load balancing is slow control loop, not gradient descent.

---

## Sensitivity Guide

| If you change... | Watch for... |
|---|---|
| `micro_batch_size` ↑ | OOM — run microbench first |
| `max_seq_len` ↑ | Quadratic attention memory in training |
| `n_activated_experts` ↑ | More FFN compute, routing diversity |
| `lr` ↑ | NaN guard triggers — watch loss |
| `mup_lr: false` | May need manual LR retuning |
| `compile: false` | ~15-25% slower steps |




## CLI Overrides vs YAML

`training/pretrain.py` CLI flags override YAML at runtime:

| CLI flag | Overrides |
|---|---|
| `--data-path` | `data.train_data_path` |
| `--checkpoint-dir` | `training.save_dir` |
| `--no-compile` | `training.compile: false` |
| `--no-checkpoint` | `training.grad_checkpoint: false` |
| `--resume N` | Load step N before training |

Model architecture keys have **no CLI override** — edit YAML or use test fixtures.

---

## Environment + Config Interaction

| Env var | Config key | Interaction |
|---|---|---|
| `ENABLE_TRITON_KERNELS` | `attn_impl`, `moe_dispatch` | Env required for triton values |
| `TORCH_COMPILE_MODE` | `training.compile` | Only when compile true |
| `WANDB_PROJECT` | — | Logging only, not in YAML |

---

## FAQ

**Q: Can I train 422M config on 24GB GPU?**
A: No without major changes — halve batch and seq len, possibly disable MTP/compile.

**Q: Why two configs?**
A: 422M = production research. 1650 2M = structural smoke test on 4GB.

**Q: Where is `seed` in YAML?**
A: Not in 422M YAML — defaults to 42 in `TrainingConfig`. Add `seed: N` under `training:` to override.


## References

- `configs/pretrain_a100_422m.yaml`
- `configs/pretrain_1650_2m.yaml`
- `tests/conftest.py` — `cfg`, `small_cfg` fixtures
- [[Docs/08_Training_Pipeline|training]] — how YAML maps to `TrainingConfig`

## Full YAML Walkthrough — 422M Config

Annotated excerpt from `configs/pretrain_a100_422m.yaml`:

```yaml
model:
  vocab_size: 100018      # MUST match DeepSeek tokenizer len()
  dim: 768                # Hidden width d; divisibility: d % n_heads == 0
  n_layers: 18            # 2 dense + 16 MoE
  n_heads: 12             # MLA heads; qk_head_dim = 48 nope + 24 rope = 72, v_head_dim = 64 (see [[Docs/03_Multi_Head_Latent_Attention|MLA]])
  n_dense_layers: 2       # Layers 0-1 use SwiGLU; 2-17 use MoE
  n_routed_experts: 20    # Router selects top-k from these
  n_shared_experts: 1     # Always executed (not routed)
  n_activated_experts: 4  # Top-k per token (routed only)
  inter_dim: 1536         # Dense FFN width (layers 0-1)
  moe_inter_dim: 384      # Per-expert width in MoE layers
  kv_lora_rank: 192       # MLA KV compression rank R
  q_lora_rank: 0          # 0 = no query compression at 422M
  qk_nope_head_dim: 48    # Content QK per head
  qk_rope_head_dim: 24    # RoPE QK per head (must sum to d_h=64)
  v_head_dim: 64          # Value dim per head
  max_seq_len: 2048       # MLA cache size + dataset window
  mtp_depth: 1            # One auxiliary head (predict t+2)
  mtp_loss_weight: 0.3    # λ in total loss
  attn_impl: sdpa         # PyTorch SDPA (FlashAttention backend)
  moe_dispatch: stacked   # Safe default; triton_grouped needs env var
  weight_tying: true      # embed.weight IS head.weight

training:
  micro_batch_size: 8
  gradient_accumulation_steps: 4   # Effective batch = 32 sequences
  total_steps: 512000            # Optimiser steps (not micro-steps)
  lr: 8.0e-4                     # Overwritten by μP if mup_lr: true
  nan_guard: true                # AGENTS.md: never disable without consent
```

### Mapping YAML → TrainingConfig

`training/pretrain.py:main()` reads nested YAML and constructs `TrainingConfig`:

| YAML path | TrainingConfig field |
|---|---|
| `training.micro_batch_size` | `batch_size` |
| `training.save_interval` | `save_every` |
| `training.log_interval` | `log_every` |
| `training.grad_clip` | `max_grad_norm` |
| `model.mtp_loss_weight` | `mtp_weight` |
| `data.train_data_path` | `data_path` |

The **entire** YAML dict is stored in `model_config` so `Transformer` can unwrap `model` section internally.

---

## 1650 Smoke Config — When to Use

`configs/pretrain_1650_2m.yaml` exists for **structural validation** on 4 GB GPUs:

- Same code paths (MLA, MoE, MTP) at `dim=64`
- `compile: false`, `grad_checkpoint: false` — fits without optimisations
- `vocab_size: 50257` (GPT-2) — different tokenizer family

**Do not** tune production hyperparameters on 1650 config — loss dynamics differ at 2M scale.

---

## Config Validation Checklist

Before launching a 15-hour run:

- [ ] `len(tokenizer) == model.vocab_size`
- [ ] `qk_nope_head_dim + qk_rope_head_dim == dim / n_heads`
- [ ] `data/pretrain_chinchilla/shard_*.bin` exists
- [ ] `micro_batch_size × grad_accum × seq_len` matches expected tokens/step
- [ ] If `attn_impl: triton` or `moe_dispatch: triton_grouped`, set `ENABLE_TRITON_KERNELS=1`
- [ ] `moe_inter_dim > 256` with triton_grouped → expect stacked fallback at runtime

<!-- docs:verified 2026-08-01 · e8553c4 -->


---

# Operations — Scripts, Utilities, and Testing

> **Canonical** for launch scripts, utilities, and the test suite.

## Table of contents

- [Part A — Scripts](#part-a--scripts)
- [Part B — Utilities](#part-b--utilities)
- [Part C — Testing](#part-c--testing)

---

## Part A — Scripts

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

Each script mirrors a phase of the [[Docs/00_Getting_Started|Getting Started]] mental model (CPU tests → GPU smoke → production).

| Script | Purpose | Requires GPU |
|---|---|---|
| `launch_a100.sh` | Production training launch (nohup) | Yes (A100) |
| `microbench_a100.py` | VRAM estimate vs measured peak | Yes |
| `step_time_a100.py` | ms/step, tokens/sec, MFU | Yes |
| `smoke_forward.py` | Single forward+backward sanity | Yes |
| `e2e_test_gpu.py` | End-to-end GPU test suite | Yes |
| `build_small_pretrain_data.py` | Tiny dataset for local dev | No |
| `check_docs.py` | Lint `Docs/` links, paths, stale patterns | No |

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

- [[Docs/08_Training_Pipeline|Training]] — what launch_a100 runs
- [[Docs/11_Operations_and_Testing|Operations]] — memory estimation formulas
- [[Docs/00_Getting_Started|Getting Started]] — quick start path

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

<!-- docs:verified 2026-07-31 · 5a880d2 -->

---

## Part B — Utilities

## A Comprehensive Technical Reference

> **Covers**: `utils/checkpoint.py`, `utils/memory.py`, `utils/logging.py` — production infrastructure for training and inference.

> **Read this if** you're debugging checkpoints, VRAM estimates, or WandB logging. **Skip if** you're changing model math → component docs.

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

Three-file checkpoint triplet per step (see [[Docs/08_Training_Pipeline|Training]] §Checkpoint Format).

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
- [[Docs/08_Training_Pipeline|Training]] — save/load integration
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

<!-- docs:verified 2026-07-31 · 5a880d2 -->

---

## Part C — Testing

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

**Remaining gap:** full-model `test_sdpa_and_triton_agree` (end-to-end `attn_impl: triton`) is not yet in the suite — track in [[Docs/12_Triton_Kernels|triton kernels]].

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
| `test_models.py` | [[Docs/03_Multi_Head_Latent_Attention|MLA]], [[Docs/04_DeepSeekMoE|MoE]], [[Docs/02_Model_Architecture|Transformer]] |
| `test_training.py` | [[Docs/08_Training_Pipeline|Training]] |
| `test_inference.py` | [[Docs/10_Inference_and_Serving|Inference]] |
| `test_utils.py` | [[Docs/11_Operations_and_Testing|Operations]] |
| `test_moe_triton.py` | [[Docs/12_Triton_Kernels|triton kernels]] |
| `test_force_back.py` | [[Docs/08_Training_Pipeline|Training]] §Triton |

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
- [[Docs/00_Getting_Started|Getting Started]] — quick test commands
- Component docs: [[Docs/04_DeepSeekMoE|MoE]], [[Docs/03_Multi_Head_Latent_Attention|MLA]], [[Docs/05_Multi_Token_Prediction|MTP]]

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
| MLA absorption drift | `test_sdpa_and_manual_agree` | [[Docs/03_Multi_Head_Latent_Attention|MLA]] |
| MoE bias in AdamW | `test_bias_not_in_parameters` | [[Docs/04_DeepSeekMoE|MoE]] |
| MTP alignment off-by-one | `test_forward_short_sequence` | [[Docs/05_Multi_Token_Prediction|MTP]] |
| Triton silent enable | `test_force_back_*` | [[Docs/08_Training_Pipeline|Training]] |
| Checkpoint MTP prefix | `test_mtp_weights_roundtrip` | [[Docs/11_Operations_and_Testing|Operations]] |
| μP LR formula | `test_mup_lr_scaling` | [[Docs/08_Training_Pipeline|Training]] |

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

<!-- docs:verified 2026-08-01 · e8553c4 -->

## Appendix D — Technical Walkthroughs & Implementation Notes

### μP LR Scaling Derivation

The μP transfer rule (Yang et al., 2022) lets you tune LR on a small model and scale it to a larger one without re-running a sweep.

**Formula:**

$$
\eta_{\text{target}} = \eta_{\text{ref}} \times \sqrt{\frac{N_{\text{ref}}}{N_{\text{target}}}}
$$

where $N$ = total parameter count (including MTP params when MTP wrap is active).

**Worked example (422M config):**

- Reference: $\eta_{\text{ref}} = 6 \times 10^{-4}$ at $N_{\text{ref}} = 757226496$
- Target: $N_{\text{target}} \approx 422000000$

$$
\eta = 6 \times 10^{-4} \times \sqrt{\frac{757226496}{422000000}} = 6 \times 10^{-4} \times 1.339 \approx 8.03 \times 10^{-4}
$$

The YAML sets `lr: 8.0e-4` but μP overrides to ~8.03e-4 at init.

**Why wider models get lower LR:** Wider models average gradients over more parameters per layer → each parameter's update is smaller → lower LR maintains the same effective update magnitude. The $\sqrt{\cdot}$ scaling is the optimal balance point that keeps "update per parameter" constant across scales.

**MTP edge case:** If MTP is enabled, `total` includes MTP head parameters, making the denominator larger and the scaled LR slightly lower. This is a known limitation — the MTP params don't participate in the same μP theory as the trunk.

**Implementation order:** μP scaling runs **after** MTP wrap (step 8 in `__init__`), so it correctly counts the post-wrap parameter total.

```python
if config.mup_lr:
    new_lr = config.mup_lr_reference * (config.mup_lr_reference_params / total) ** 0.5
    config.lr = new_lr
```

### Sharded Dataset Loading

For large corpora, data is split into binary shard files (`shard_000.bin`, `shard_001.bin`, …). The dataset uses three mechanisms to serve arbitrary indices efficiently:

**Binary-search shard lookup:**

```python
def _locate(self, global_idx):
    lo, hi = 0, len(self.shard_offsets) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if self.shard_offsets[mid] <= global_idx:
            lo = mid
        else:
            hi = mid - 1
    return lo, global_idx - self.shard_offsets[lo]
```

This finds which shard a global index belongs to in $O(\log N)$ time. Each shard tracks its cumulative token offset in `shard_offsets`.

**LRU shard cache (capacity = 2):**

```python
def _load_shard(self, shard_idx):
    if shard_idx in self._shard_cache:
        return self._shard_cache[shard_idx]
    t = torch.load(self.shard_paths[shard_idx], weights_only=True, map_location="cpu")
    self._shard_cache[shard_idx] = t
    self._shard_cache_order.append(shard_idx)
    while len(self._shard_cache_order) > 2:  # evict oldest
        evict = self._shard_cache_order.pop(0)
        self._shard_cache.pop(evict, None)
    return t
```

Only **2 shards** stay in memory at once — small by design to avoid OOM on machines with limited RAM. With more memory, increasing this reduces disk I/O.

**Cross-shard stitching:**

When a training window spans two shards, tokens are collected sequentially using `.tolist()` to bridge the boundary, then packed into a tensor. This is slower than `torch.cat` (used in GPT-OSS-Lite) but correct and memory-safe — no temporary concatenation tensor is allocated.

### NaN Guard Mechanism

The NaN guard operates at three levels to protect training from divergence:

1. **Skip the step:** If loss is NaN or Inf, skip backward and clear gradients — the optimizer is not updated.
2. **Count consecutive NaNs:** A `nan_guard_streak` counter increments on each bad step and resets to 0 on any finite loss.
3. **Rollback after 5 consecutive NaNs:** Restore from the last good checkpoint (weights, optimizer state, scheduler, `_opt_steps` counter).

```python
if nan_guard_streak >= config.nan_guard_max_consecutive:
    latest = self._find_latest_checkpoint()
    if latest is not None:
        global_step = self.load_checkpoint(latest)
    else:
        raise RuntimeError("NaN with no checkpoint to restore")
    nan_guard_streak = 0
```

**Why 5?** Sporadic NaNs (1–4 in a row) happen from transient numerical instabilities (rare gradient spikes from MoE routing). The optimizer recovers naturally. Five consecutive NaNs signal true divergence — LR too high, corrupt shard, or dtype bug — requiring rollback.

**Resume behavior after rollback:** Weights, AdamW moments, scheduler state, and `_opt_steps` are all restored from the checkpoint. The DataLoader shuffle order is **not** restored (sampler RNG is not checkpointed) — benign at 8.4B token scale. `global_step` in the `train()` loop resumes from the restored micro-step counter.

**If no checkpoint exists:** Raises `RuntimeError` immediately. The NaN guard is never disabled by default in the 422M YAML (`nan_guard: true`).

### YAML Config Reference

The canonical config (`configs/pretrain_a100_422m.yaml`) has three top-level sections:

```yaml
model:      # Architecture — consumed by Transformer, MLA, MoE, MTP
training:   # Loop hyperparameters — consumed by Pretrainer
data:       # Paths — consumed by Pretrainer and inference/generate.py
```

**Key model parameters (422M):**

| Parameter | Value | Description |
|---|---|---|
| `vocab_size` | 100018 | DeepSeek-Coder-V2-Lite tokenizer length |
| `dim` | 768 | Hidden dimension |
| `n_layers` | 18 | 2 dense + 16 MoE |
| `n_heads` | 12 | MLA attention heads |
| `n_routed_experts` | 20 | MoE routed experts |
| `n_activated_experts` | 4 | Top-4 routing per token |
| `kv_lora_rank` | 192 | KV compression latent dim |
| `qk_nope_head_dim` | 48 | Content key/query per head |
| `qk_rope_head_dim` | 24 | Positional key/query per head |
| `v_head_dim` | 64 | Value per head |
| `max_seq_len` | 2048 | Training sequence length |
| `mtp_depth` | 1 | MTP auxiliary heads |
| `mtp_loss_weight` | 0.3 | λ for MTP loss |

**Key training parameters:**

| Parameter | Value | Description |
|---|---|---|
| `micro_batch_size` | 8 | Per-GPU batch |
| `gradient_accumulation_steps` | 4 | Effective batch = 32 |
| `total_steps` | 512000 | Micro-steps (128K optimizer steps) |
| `warmup_steps` | 2000 | Linear LR warmup |
| `lr` | 8.0e-4 | Overwritten by μP |
| `min_lr_ratio` | 0.05 | Cosine floor |
| `weight_decay` | 0.1 | On dim≥2 params only |
| `grad_clip` | 1.0 | Global norm clip |
| `compile` | true | torch.compile |
| `mup_lr` | true | Enable μP scaling |
| `nan_guard` | true | NaN rollback |

**Key data parameters:**

| Parameter | Value | Description |
|---|---|---|
| `train_data_path` | data/pretrain_chinchilla | Shard directory or .bin file |
| `tokenizer_path` | deepseek-ai/deepseek-coder-v2-lite | HF tokenizer |

**Token budget math:**

$$
\text{tokens/optimizer-step} = 8 \times 4 \times 2048 = 65536
$$

$$
\text{total tokens} = 128000 \times 65536 \approx 8.4\text{B unique} \approx 20 \text{ tokens/param (Chinchilla optimal)}
$$

**Hardware knobs** (set in `Pretrainer.__init__`):

```python
torch.backends.cuda.matmul.allow_tf32 = True   # ~8× faster FP32 accum on Ampere
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True           # auto-select best algorithm
```

**`torch.compile`:** `mode="max-autotune"`, `fullgraph=False` — fuses ops and autotunes CUDA kernels; graph breaks allowed for MoE dispatch and Python control flow.

### Checkpoint Management

Each save at step `N` produces three atomic files:

| File | Format | Contents |
|---|---|---|
| `model_step_N.safetensors` | SafeTensors | Transformer weights + `mtp.*` keys |
| `optim_step_N.pt` | PyTorch pickle | AdamW moments (FP32 master) |
| `meta_step_N.json` | JSON | step, opt_steps, scheduler, config, has_mtp, tag |

**Atomic write protocol:**

```python
fd, tmp = tempfile.mkstemp(dir=save_dir, suffix=".safetensors.tmp")
save_file(deduped_state, tmp)
os.replace(tmp, final_path)   # atomic on POSIX — no torn checkpoints
```

On failure: temp file is unlinked, exception re-raised. A checkpoint is only considered complete if **all three** files exist — `latest_step()` ignores partial writes from killed jobs.

**Tied weight deduplication:**

```python
seen_ptrs = set()
for k, v in state.items():
    ptr = v.data_ptr()
    if ptr in seen_ptrs:
        deduped[k] = v.contiguous().clone()  # clone tied weights
    else:
        seen_ptrs.add(ptr)
        deduped[k] = v.contiguous()
```

When `weight_tying: true`, `embed.weight` and `head.weight` share the same `data_ptr()`. safetensors cannot store the same tensor pointer twice — the second occurrence is cloned. `Pretrainer` also drops `head.weight` entirely on save (redundant with embed).

**MTP weight integration:**

```python
# Saving — merge into same file:
mtp_state = {f"mtp.{k}": v for k, v in orig.state_dict().items() if k.startswith("mtp_modules.")}
state.update(mtp_state)

# Loading — strip prefix:
mtp_state = {k.removeprefix("mtp."): v for k, v in state.items() if k.startswith("mtp.")}
mtp_wrapper.load_state_dict(mtp_state, strict=False)
```

MTP weights are saved with `mtp.` prefix in the **same file** as the main model — one file per checkpoint, fully atomic. The `has_mtp` flag in `meta_step_N.json` tells the loader whether to look for MTP keys.

**Metadata structure:**

```json
{
    "step": 4000,
    "scheduler": { "last_epoch": 1000, "_step_count": [1000] },
    "opt_steps": 1000,
    "tag": "step_4000",
    "config": { ... },
    "has_mtp": true
}
```

The scheduler state is saved in JSON (not a separate .pt file), simplifying the checkpoint layout. `_opt_steps` tracks actual optimizer steps (not micro-steps).

**Completeness check:**

```python
def _checkpoint_complete(self, step):
    return all((self.save_dir / n).exists() for n in [
        f"model_step_{step}.safetensors",
        f"optim_step_{step}.pt",
        f"meta_step_{step}.json"
    ])
```
