# DeepSeek-v3-Lite — Training Pipeline

> **Canonical** for pretrain loop, μP, NaN guard, and YAML reference.

> **Read this if** you're debugging NaN loss, μP LR, checkpoints, or the train loop. **Skip if** you only need launch commands → [Model Architecture](concepts/foundations.md).

**Depends on:** [Foundations & Architecture](concepts/foundations.md), [DeepSeekMoE & MTP](concepts/moe-mtp.md) · **Read next:** [Data Pipeline](concepts/data-pipeline.md), [Inference](inference.md)

---

## Table of Contents

1. [Abstract](#abstract)
2. [Training Stack Overview](#training-stack-overview)
3. [TrainingConfig Field by Field](#trainingconfig-field-by-field)
4. [Pretrainer Lifecycle](#pretrainer-lifecycle)
5. [AdamW — Full Update Rule](#adamw--full-update-rule)
6. [Gradient Accumulation — Mathematics](#gradient-accumulation--mathematics)
7. [LR Scheduler](#lr-scheduler)
8. [μP Learning Rate Scaling](#μp-learning-rate-scaling)
9. [PretrainDataset](#pretraindataset)
10. [DataLoader Design](#dataloader-design)
11. [train_step](#train_step)
12. [MTP Training Path](#mtp-training-path)
13. [MoE Bias Updates](#moe-bias-updates)
14. [NaN Guard](#nan-guard)
15. [Checkpointing](#checkpointing)
16. [torch.compile — What It Does Here](#torchcompile--what-it-does-here)
17. [Memory Timeline — One train_step](#memory-timeline--one-train_step)
18. [Chinchilla Epoch Analysis](#chinchilla-epoch-analysis)
19. [CLI Reference](#cli-reference)
20. [Pretrainer.__init__ — Line-by-Line Walkthrough](#pretrainer__init__--line-by-line-walkthrough)
21. [train_step — Pseudocode with Tensor Shapes](#train_step--pseudocode-with-tensor-shapes)
22. [Checkpoint Format — File-by-File](#checkpoint-format--file-by-file)
23. [NaN Guard — State Machine](#nan-guard--state-machine)
24. [Worked Example — One Optimiser Step at the 422m Config](#worked-example--one-optimiser-step-at-the-422m-config)
25. [torch.compile Interaction with MTP and MoE](#torchcompile-interaction-with-mtp-and-moe)
26. [Appendix A — Train loop state diagram](#appendix-a--train-loop-state-diagram)
27. [Appendix B — FAQ](#appendix-b--faq)
28. [Appendix C — Glossary](#appendix-c--glossary)
29. [Load-Bearing Invariants](#load-bearing-invariants)
30. [Implementation Checklist](#implementation-checklist)
31. [References](#references)
32. [train() Main Loop — Pseudocode](#train-main-loop--pseudocode)
33. [Optimiser Param Groups — Why Split?](#optimiser-param-groups--why-split)
34. [compile_model Interaction](#compile_model-interaction)
35. [Hyperparameter Sensitivity (Empirical)](#hyperparameter-sensitivity-empirical)
36. [Worked Example — Micro-step vs Optim Step](#worked-example--micro-step-vs-optim-step)
37. [Logging and Monitoring](#logging-and-monitoring)
38. [Resume and Fault Tolerance](#resume-and-fault-tolerance)
39. [Part B — Configuration Reference](#part-b--configuration-reference)

---

## Abstract

The training system is a **single-GPU, from-scratch PyTorch pretrainer** targeting Chinchilla-optimal training: ~411.6M parameters (418.7M with MTP) on 8.4B tokens. It combines BF16 autocast, FP32 AdamW master weights, `torch.compile(max-autotune)`, gradient checkpointing, μP LR scaling, aux-loss-free MoE bias updates, MTP auxiliary loss, and a NaN guard with automatic checkpoint rollback.

No HuggingFace Trainer. No distributed. One A100 80GB.

### 60-Second Summary

Pretraining here is a plain Python `while` loop over sharded, pre-tokenised data. Each micro-step loads one `(B, S)` batch of `uint32` token ids, casts them to `long`, runs the trunk under BF16 autocast, divides the loss by the gradient-accumulation count, and calls `backward()`. Only every `gradient_accumulation_steps`-th micro-step clips, steps the FP32 AdamW optimizer, advances the warmup-cosine scheduler, and nudges the MoE routing biases. A NaN guard watches every loss; five consecutive non-finite losses roll the whole training state (weights, optimizer, scheduler, step counter) back to the newest checkpoint. The learning rate is not a hand-tuned constant: μP scales `lr` by `sqrt(reference_params / actual_params)`, turning `6.0e-4` at 757.2M reference params into **8.138e-4** for the 411.6M base model and **8.069e-4** with the MTP wrapper. Everything a 15-hour run does is spelled out in `training/pretrain.py` — there is no trainer framework layer to fight.

### The training problem, formally

Given token sequences $\mathbf{x} = (x_1, \ldots, x_T)$ from the data distribution, we minimise:

$$
\min_\theta \; \mathbb{E}_{\mathbf{x} \sim \mathcal{D}}\left[ -\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t}) + \lambda \sum_{d=1}^{D} \mathcal{L}_{\text{MTP}}^{(d)} \right]
$$

with $\lambda = 0.3$, $D = 1$ at the canonical config (`configs/pretrain_a100_422m.yaml` — the filename is historical; the deduped model is 411,632,256 parameters, see [Model Architecture](concepts/foundations.md) for the budget). The expectation is approximated by stochastic mini-batches over sharded uint32 corpora.

**Single-GPU constraint:** No data parallelism. Effective batch size = `micro_batch_size × gradient_accumulation_steps` sequences.

**Prerequisites:** [Foundations](concepts/foundations.md) §9 (loss), §12 (Chinchilla), §13 (μP).

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

Each row of this table becomes a section below, in the order the training system actually touches them: configuration first, then the `Pretrainer` object, then the optimizer/scheduler math, then data, then the per-step mechanics.

---

## TrainingConfig Field by Field

`TrainingConfig` (`training/pretrain.py:TrainingConfig`) is a plain `@dataclass` — 29 fields, no magic. It is the *runtime* view of training hyperparameters: `main()` reads the YAML file and copies values into the dataclass, `Pretrainer.__init__` reads the dataclass, and nothing in the training loop ever touches YAML again. The one deliberate exception is `model_config`, which carries the **entire** raw YAML dict so that downstream model code can unwrap the `model:` section itself (`models/transformer.py:Transformer.__init__` does `config.get("model", config)`).

### The dataclass

```python
@dataclass
class TrainingConfig:
    model_config: dict = field(default_factory=dict)
    data_path: str = "data/pretrain_data.bin"
    checkpoint_dir: str = "checkpoints/pretrain"
    vocab_size: int = 100018
    max_seq_len: int = 4096
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_steps: int = 20000
    warmup_steps: int = 2000
    lr: float = 2.2e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    max_grad_norm: float = 1.0
    mtp_weight: float = 0.0
    bias_update_speed: float = 0.001
    bias_update_every: int = 10
    grad_checkpoint: bool = True
    compile_model: bool = True
    save_every: int = 1000
    log_every: int = 100
    nan_guard: bool = False
    nan_guard_max_consecutive: int = 5
    mup_lr: bool = False
    mup_lr_reference: float = 6.0e-4
    mup_lr_reference_params: int = 757226496
    log_per_component_params: bool = True
    seed: int = 42
```

Notice the defaults are **not** the 422M recipe. The defaults are a conservative smoke profile (`max_steps=20000`, `nan_guard=False`, `mup_lr=False`, `bias_update_every=10`); the canonical values come from YAML. The mapping below is exact — it is the code in `training/pretrain.py:main`.

### YAML to field mapping

| YAML key (422M value) | Field | Type | Who consumes it |
|---|---|---|---|
| *(whole file)* | `model_config` | dict | `Transformer`, `MultiTokenPrediction` |
| `data.train_data_path` (`data/pretrain_chinchilla`) | `data_path` | str | `PretrainDataset` |
| `training.save_dir` (`checkpoints/pretrain_a100`) | `checkpoint_dir` | str | `CheckpointManager` |
| `model.vocab_size` (`100018`) | `vocab_size` | int | dataset construction (informational) |
| `model.max_seq_len` (`2048`) | `max_seq_len` | int | `PretrainDataset`, logger |
| `training.micro_batch_size` (`8`) | `batch_size` | int | `DataLoader` batch size, logger |
| `training.gradient_accumulation_steps` (`4`) | `gradient_accumulation_steps` | int | `train_step` |
| `training.total_steps` (`512000`) | `max_steps` | int | scheduler, `train()` loop bound |
| `training.warmup_steps` (`2000`) | `warmup_steps` | int | `make_warmup_cosine_lambda` |
| `training.lr` (`8.0e-4`) | `lr` | float | AdamW (possibly overwritten by μP) |
| `training.min_lr_ratio` (`0.05`) | `min_lr_ratio` | float | scheduler floor |
| `training.weight_decay` (`0.1`) | `weight_decay` | float | AdamW decay group |
| `training.grad_clip` (`1.0`) | `max_grad_norm` | float | `clip_grad_norm_` |
| `model.mtp_loss_weight` (`0.3`) | `mtp_weight` | float | MTP engagement + loss |
| `training.bias_update_speed` (`0.001`) | `bias_update_speed` | float | `update_gate_bias` |
| `training.bias_update_every` (`1`) | `bias_update_every` | int | bias cadence |
| `training.grad_checkpoint` (`true`) | `grad_checkpoint` | bool | `Transformer(use_checkpoint=...)` |
| `training.compile` (`true`) | `compile_model` | bool | `torch.compile` gate |
| `training.save_interval` (`4000`) | `save_every` | int | checkpoint cadence |
| `training.log_interval` (`50`) | `log_every` | int | logging cadence |
| `training.nan_guard` (`true`) | `nan_guard` | bool | NaN skip/rollback |
| `training.nan_guard_max_consecutive` (`5`) | `nan_guard_max_consecutive` | int | rollback threshold |
| `training.mup_lr` (`true`) | `mup_lr` | bool | μP scaling |
| `training.mup_lr_reference` (`6.0e-4`) | `mup_lr_reference` | float | μP reference LR |
| `training.mup_lr_reference_params` (`757226496`) | `mup_lr_reference_params` | int | μP reference count |
| `training.log_per_component_params` (`true`) | `log_per_component_params` | bool | startup breakdown |
| `training.seed` (absent → `42`) | `seed` | int | RNG seeding |

Three name mismatches are worth memorising because they appear everywhere in docs and logs: YAML `micro_batch_size` → field `batch_size`, YAML `total_steps` → field `max_steps`, YAML `grad_clip` → field `max_grad_norm`. And `mtp_weight` lives under the **model** section in YAML but lands in the **training** dataclass.

### Field-by-field tour

**`model_config`** is the only non-scalar field: the entire parsed YAML dict. `Transformer.__init__` unwraps `model:` from it; `MultiTokenPrediction` reads `mtp_depth` / `mtp_loss_weight` from it; `Pretrainer.__init__` reads `mtp_depth` out of it via the two-level `config.model_config.get("model", config.model_config).get("mtp_depth", 0)` dance — the inner `.get("model", ...)` makes the same code work for flat test dicts and nested YAML.

**`data_path` / `checkpoint_dir`** are the only two fields that can also be overridden from the command line (`--data-path`, `--checkpoint-dir`). `data_path` may point at a single `.bin` tensor file *or* a directory of `shard_*.bin` files; `PretrainDataset.__init__` decides by `os.path.isdir`. `checkpoint_dir` is created (mkdir -p) by `CheckpointManager.__init__`.

**`vocab_size` / `max_seq_len`** mirror the model section so the dataset and logger can size themselves. `max_seq_len` drives both the dataset window and the `TrainingLogger` tokens-per-second arithmetic. If they disagree with the model config, nothing crashes — but windows and throughput numbers come out wrong, so keep them in lockstep with `model.max_seq_len`.

**`batch_size` (micro) and `gradient_accumulation_steps`** jointly define the effective batch. The `DataLoader` is given `batch_size` (the micro size); the accumulation factor lives only in `train_step`'s boundary test `(micro_step + 1) % gradient_accumulation_steps == 0` and the loss division. Nothing else in the codebase needs to know the product.

**`max_steps`** is the **micro-step budget** of the `train()` loop: `global_step` counts micro-batches, and the loop terminates when `global_step` reaches `max_steps`. At 422M that is 512,000 micro-batches = 128,000 optimizer steps (÷ `gradient_accumulation_steps` = 4) ≈ one pass over the 8.4B-token corpus — the Chinchilla-optimal budget. This is a genuine naming trap: `total_steps`, `save_interval`, and `log_interval` are all in micro-step units, but the LR scheduler's `total_steps` argument is in *scheduler* steps — one per optimizer step — so the schedule horizon and the loop bound are in different units (consequence in [LR Scheduler](#lr-scheduler)).

**`warmup_steps`, `lr`, `min_lr_ratio`** feed the scheduler. `lr` is the *base* LR that `LambdaLR` multiplies by the warmup-cosine curve — and, when `mup_lr=True`, it is silently overwritten at `Pretrainer.__init__` time *before* the optimizer is built (so the optimizer never sees the YAML value). More in [μP Learning Rate Scaling](#μp-learning-rate-scaling).

**`weight_decay`, `beta1`, `beta2`** are passed straight to AdamW. Decay is applied only to the `dim >= 2` parameter group (see [AdamW — Full Update Rule](#adamw--full-update-rule)); the betas are the standard LLM pair 0.9 / 0.95.

**`max_grad_norm`** is the global-norm clip bound in `train_step`.

**`mtp_weight`** gates the whole MTP machinery: `Pretrainer.__init__` wraps the model in `MultiTokenPrediction` only when `mtp_depth > 0` **and** `mtp_weight > 0.0`. Setting `mtp_weight: 0` in YAML quietly disables the auxiliary path (depth is ignored).

**`bias_update_speed` / `bias_update_every`** control the MoE router-bias control loop: how far the bias moves per update and how many optimizer steps between updates. `bias_update_every=1` in the 422M YAML means "every optimizer step" — the dataclass default of 10 is for smoke runs.

**`grad_checkpoint` / `compile_model`** are booleans that reach into model construction (`use_checkpoint`) and wrapping (`torch.compile`). Both can be force-disabled from the CLI with `--no-checkpoint` / `--no-compile` — the YAML value is `and`-ed with the CLI flags in `main()`.

**`save_every` / `log_every`** are the two cadences of the run: checkpoints at multiples of 4000 optimizer steps, log lines at multiples of 50.

**`nan_guard` / `nan_guard_max_consecutive`** arm the divergence safety net. Note the dataclass default is `False` — the 422M YAML explicitly sets `true` (AGENTS.md treats disabling it as a deliberate, reviewed decision).

**`mup_lr` / `mup_lr_reference` / `mup_lr_reference_params`** are the μP recipe: the reference model's tuned LR and parameter count. The reference (757,226,496 params @ 6.0e-4) is an external anchor from the wider DeepSeek-V3 family — it is *not* measurable in this repo.

**`log_per_component_params`** turns on the startup parameter breakdown (embedding vs MLA vs experts …) — cheap, and the single best sanity check that the architecture you built matches the architecture you intended.

**`seed`** seeds the CPU RNG before model construction (`torch.manual_seed(config.seed)`) and the DataLoader shuffle generator. It is not in the 422M YAML, so it defaults to 42.

### Effective batch size arithmetic

The two numbers that every other budget derives from:

$$
B_{\text{eff}} = B_{\text{micro}} \times G_{\text{accum}} = 8 \times 4 = 32 \text{ sequences / optimizer step}
$$

$$
T_{\text{micro-step}} = B_{\text{micro}} \times S = 8 \times 2048 = 16\,384 \text{ tokens / micro-step}
$$

$$
T_{\text{opt-step}} = B_{\text{eff}} \times S = 32 \times 2048 = 65\,536 \text{ tokens / optimizer step}
$$

Total tokens over the run (512,000 micro-steps):

$$
N_{\text{tokens}} = 512\,000 \times 16\,384 \approx 8.39 \times 10^9 \approx \text{one pass over the 8.4B corpus}
$$

Optimizer steps over the run: $512\,000 / 4 = 128\,000$. The 8.4B-token corpus is the Chinchilla-optimal budget for 411.6M parameters; the run ends ~0.14% short of a full epoch because one epoch is 512,695 micro-batches (see [Chinchilla Epoch Analysis](#chinchilla-epoch-analysis)).
---

## Pretrainer Lifecycle

`Pretrainer` (`training/pretrain.py:Pretrainer`) is the whole training system in one class: construction wires hardware flags, model, optimizer, scheduler, and checkpoints; `train()` owns the data loop; `train_step` owns the per-micro-batch math. The class deliberately keeps **no** state that `train_step` cannot see — the only persistent counters are `_opt_steps` (checkpointed) and the implicit `global_step` (re-derived on resume).

### `__init__` — the eleven steps

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

### `__init__` deep dive

Everything below is the actual body of `training/pretrain.py:Pretrainer.__init__`, read top to bottom.

#### Seeding

```python
# Seed before model construction for reproducible init + data order.
torch.manual_seed(config.seed)
```

`torch.manual_seed` seeds the CPU RNG used by weight initialisation (`nn.init.normal_`, `torch.empty`) and, transitively, the CUDA RNG for the current device. It runs **before** `Transformer(...)` so a given `seed` reproduces the exact weight initialisation. Two things it does *not* cover:

- **DataLoader shuffle order** — that uses a separate `torch.Generator` created in `train()` (`g = torch.Generator().manual_seed(self.config.seed)`), deliberately independent so the training loop can reseed it per epoch without touching model init.
- **Run-to-run jitter on CUDA** — `torch.compile`, cuBLAS heuristics and nondeterministic kernels mean two identical-seed runs on the same GPU can still differ in the last ULPs. Seed reproducibility here is "same initial weights and same data order", not bit-exact runs.

#### Device and numerics flags

```python
self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("[warn] CUDA not available — running on CPU (smoke-testing only).")
else:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
```

- **TF32** is a reduced-precision FP32 mode on Ampere+ GPUs: FP32 inputs with a 10-bit mantissa (19 bits total) instead of 23. cuBLAS TF32 GEMMs run ~8× the FP32 rate. Two flags: `torch.backends.cuda.matmul.allow_tf32` covers `torch.mm`/`torch.bmm`/`nn.Linear` in plain FP32; `torch.set_float32_matmul_precision("high")` is the newer umbrella that enables the same for eager and compiled paths. Why is it safe here? The model's heavy compute runs in **BF16** under autocast, so TF32 only affects stray FP32 matmuls; the numerically sensitive FP32 spots (RMSNorm, the loss, the AdamW update) are elementwise/reductions, not matmuls — cuBLAS TF32 never touches them.
- **`cudnn.benchmark`** lets cuDNN autotune kernel choices; this repo has no convolutions, so it is mostly inert — it is set for uniformity with the rest of the stack.
- **CPU path:** the warning is literal. On a laptop the loop runs eagerly in FP32/BF16 CPU matmuls and is meant for smoke tests only (the 1650 config, `torch.compile=False`).

#### Logging setup

```python
init_logging(config.log_every, seq_len=config.max_seq_len, batch_size=config.batch_size)
self.logger = get_logger()
```

`utils/logging.py:init_logging` constructs the process-global `TrainingLogger`; `get_logger` hands it back. The logger needs `seq_len` and `batch_size` up front because its throughput line is `tokens_per_sec = (log_interval × seq_len × batch_size) / elapsed` — i.e. it assumes every logged interval consumed exactly `log_interval` full micro-batches.

#### Model construction

```python
# AGENTS rule #7: default-config run must never silently switch to a Triton path.
# Guard already ran in Transformer.__init__; this is a no-op.
raw_model = Transformer(config.model_config, use_checkpoint=config.grad_checkpoint).to(self.device)
total, trainable = count_parameters(raw_model)
self._log(f"Parameters: {total:,} total / {trainable:,} trainable")
```

`use_checkpoint` is threaded into `Transformer` and becomes the switch inside `models/transformer.py:Transformer._run_layers` that decides between `torch.utils.checkpoint.checkpoint(...)` and a plain layer call. The "no-op" comment refers to `enforce_triton_env_var` (the `models/_triton_dispatch.py` guard that forces `attn_impl="sdpa"`/`moe_dispatch="stacked"` unless `ENABLE_TRITON_KERNELS=1`): it already ran inside `Transformer.__init__`, so calling it again here would be redundant — the comment documents *why* it is not called.

#### Parameter accounting

`count_parameters` (`models/transformer.py:count_parameters`) deduplicates by tensor `id` — with weight tying, `head.weight` *is* `embed.weight`, and counting both would double-count ~76.8M. Measured on a CPU instantiation of the canonical config (2026-08-04): **411,632,256 total, 411,632,256 trainable** (every parameter trains).

With `log_per_component_params: true`, `Pretrainer._log_per_component_params` prints a name-based breakdown. The real numbers for the canonical config:

| Component | Params | Share |
|---|---|---|
| MoE routed experts (16 layers × 20 × SwiGLU 768→384→768) | 283,115,520 | 68.78% |
| Embedding (100,018 × 768; head tied, counted once) | 76,813,824 | 18.66% |
| MLA attention (18 layers: wq, wkv_a, wkv_b, wo, q_norm, kv_norm) | 30,195,072 | 7.34% |
| Shared experts (16 layers × 1) | 14,155,776 | 3.44% |
| Dense SwiGLU (2 layers) | 7,077,888 | 1.72% |
| MoE gate weights (16 × 20 × 768) | 245,760 | 0.06% |
| RMSNorm γ (18 layers × 2 + final) | 27,648 | 0.01% |
| other (final `norm.weight`) | 768 | 0.00% |

Two instructive facts hide in this table. First, **routed experts are 2/3 of the model** — the sparse part dominates the budget, which is exactly why DeepSeek-style MoE economics work: you pay ~20% of expert FLOPs per token but own all 20 experts' weights. Second, **the embedding is a sixth of the model** at 100,018 vocabulary — and it is also the LM head (tied), so the logit projection is *free* in parameter count (it is not free in compute: the head is a 768→100,018 GEMM every token, ~77M MACs/token, more than the whole 18-layer dense forward at ~59M MACs/token [derived: 2·768·1536 per dense layer]). The classifier is string-based and heuristic — a parameter whose name matches `.experts.` with `w1/w2/w3` lands in routed experts even if some future refactor moves it — so treat the breakdown as a debugging aid, not a contract.

#### MTP wrapping

```python
self.mtp_wrapper: Optional[MultiTokenPrediction] = None
mtp_depth = config.model_config.get("model", config.model_config).get("mtp_depth", 0)
if mtp_depth > 0 and config.mtp_weight > 0.0:
    mtp_model = MultiTokenPrediction(config.model_config, raw_model).to(self.device)
    mtp_total, mtp_trainable = count_parameters(mtp_model)
    self._log(f"MTP enabled (depth={mtp_depth}, weight={config.mtp_weight}): {mtp_total:,} total / {mtp_trainable:,} trainable")
    total = mtp_total
    training_model: nn.Module = mtp_model
    self.mtp_wrapper = mtp_model
else:
    training_model = raw_model
```

Both conditions must hold: `mtp_depth > 0` (architecture has heads) **and** `mtp_weight > 0.0` (the loss is actually used). When engaged, the training model becomes the wrapper, `total` switches to 418,713,984, and `self.mtp_wrapper` is set — `train_step` branches on that pointer. The delta is 7,081,728 parameters: exactly one `MTPBlock` (fusion projection 1536→768, `nn.MultiheadAttention` 768-wide, SwiGLU 768→1536→768, four RMSNorms) plus its output norm — the embedding and LM head are **shared** with the trunk, so MTP adds no new embedding or head parameters (verified against `models/mtp.py:MultiTokenPrediction.__init__`, which reuses `main_model.embed` and `main_model.head`).

#### torch.compile

```python
if config.compile_model and hasattr(torch, "compile"):
    compile_mode = os.environ.get("TORCH_COMPILE_MODE", "max-autotune")
    self._log(f"Compiling model with torch.compile (mode={compile_mode})...")
    training_model = torch.compile(training_model, mode=compile_mode, fullgraph=False)
```

The compile mode comes from the environment (`TORCH_COMPILE_MODE`, defaulting to `max-autotune`; `scripts/launch_a100.sh` exports it). `fullgraph=False` tolerates the Python MoE dispatch loop. Only `training_model` is compiled — `self.raw_model` stays eager.

#### μP LR adjustment

```python
if config.mup_lr:
    new_lr = config.mup_lr_reference * (config.mup_lr_reference_params / total) ** 0.5
    self._log(f"µP LR scaling: {config.lr:.2e} → {new_lr:.2e} (ref {config.mup_lr_reference:.2e} @ {config.mup_lr_reference_params:,} params)")
    config.lr = new_lr
```

Full derivation in [μP Learning Rate Scaling](#μp-learning-rate-scaling). Note `total` here is the **post-MTP** count when the wrapper is engaged.

#### Optimizer construction

```python
seen = set()
all_params = []
for p in self.model.parameters():
    pid = id(p)
    if pid not in seen:
        seen.add(pid)
        all_params.append(p)
decay_params = [p for p in all_params if p.dim() >= 2]
no_decay_params = [p for p in all_params if p.dim() < 2]
self.optimizer = AdamW([
    {"params": decay_params, "weight_decay": config.weight_decay},
    {"params": no_decay_params, "weight_decay": 0.0},
], lr=config.lr, betas=(config.beta1, config.beta2), fused=torch.cuda.is_available())
```

`id(p)` dedup handles both weight tying (`head.weight` shares storage with `embed.weight`) and the MTP wrapper (`mtp_modules` share the trunk's embed/head tensors). Two param groups: decay on `dim >= 2` matrices, none on vectors. `fused=True` on CUDA selects the fused AdamW kernel. See [AdamW — Full Update Rule](#adamw--full-update-rule).

#### Scheduler construction

```python
# The LR schedule lives in optimizer-step space: `max_steps` is a micro-step
# budget, so the cosine horizon is max_steps // gradient_accumulation_steps.
opt_steps = max(1, config.max_steps // config.gradient_accumulation_steps)
lr_lambda = make_warmup_cosine_lambda(warmup_steps=config.warmup_steps, total_steps=opt_steps, min_lr_ratio=config.min_lr_ratio)
self.scheduler = LambdaLR(self.optimizer, lr_lambda)
```

`LambdaLR` re-scales the optimizer's LR by `lr_lambda(step)` where `step` counts `scheduler.step()` calls — one per optimizer step, issued from `train_step`. Closed form in [LR Scheduler](#lr-scheduler). The horizon division means the canonical run's cosine arc spans its full 128,000 optimizer steps and does reach the 5% floor (pinned by `tests/test_training.py:TestPretrainerConstruction.test_scheduler_horizon_is_optimizer_steps`).

#### The tail

```python
self.amp_dtype = torch.bfloat16
self.ckpt_manager = CheckpointManager(config.checkpoint_dir)
self._opt_steps: int = 0
```

`amp_dtype` is the autocast precision for the whole run; `CheckpointManager` creates the save directory; `_opt_steps` is the only training counter that survives in the object — and it is checkpointed.

**Critical split (repeated everywhere):** `self.model` may be the compiled MTP wrapper; `self.raw_model` is always the bare eager `Transformer`, used for checkpoint I/O, MoE bias updates, and the balance metric.

---

## AdamW — Full Update Rule

### From SGD to Adam

Plain SGD follows the gradient: $\theta_{t+1} = \theta_t - \eta g_t$. Two failures motivate the Adam family: **ill-conditioning** (directions with tiny curvature need bigger steps than directions with huge curvature, but SGD gives every coordinate the same step) and **noisy gradients** (mini-batch noise makes single-sample estimates jumpy).

**Momentum** smooths the noise: keep an exponential moving average of gradients,

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t,
$$

which damps oscillation and accelerates along consistent directions. **RMSProp-style scaling** fixes conditioning: keep a moving average of *squared* gradients,

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2,
$$

and divide each coordinate's step by $\sqrt{v_t}$. Coordinates with persistently large gradients (steep curvature) get small steps; flat coordinates get large ones. **Adam** combines the two:

$$
\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon},
$$

where $\hat{m}_t = m_t/(1-\beta_1^t)$ and $\hat{v}_t = v_t/(1-\beta_2^t)$ are **bias corrections**. They matter because both averages start at zero: at small $t$ the estimates are dragged toward zero, so dividing by $1-\beta^t$ (which grows from ~0 toward 1) restores the true scale. Without correction, early steps are systematically too small.

**AdamW** (Loshchilov & Hutter, 2019) decouples weight decay from the gradient:

$$
\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)
$$

Note: decay is applied to $\theta_t$ directly, not added to $g_t$. That distinction is the entire point of the "W": in classic Adam (Loshchilov & Hutter's earlier AdamL2), decay enters $g_t$, then interacts with the per-coordinate $\sqrt{\hat v_t}$ scaling, so heavily-updated coordinates get *more* effective decay — a coupling that fights the optimizer's own normalization. AdamW keeps decay uniform across coordinates.

**This repo:** $\beta_1=0.9$, $\beta_2=0.95$, $\lambda=0.1$ on `dim >= 2` params only. Norm scales and MoE bias (`dim < 2`) get $\lambda=0$.

### Why FP32 master weights

The model parameters are stored **FP32**; autocast never mutates them. During the forward pass, `autocast(dtype=bfloat16)` casts each Linear/Embedding's inputs and weights to BF16 *transiently* — the fp32 tensors are untouched. Gradients computed by `backward()` land in FP32 `.grad` buffers, and the AdamW update runs entirely in FP32 on the FP32 parameters.

Why this design instead of keeping parameters in BF16? Two reasons:

1. **BF16 has an 8-bit exponent** (same range as FP32, max ~3.4e38, min normal ~1.2e-38) but only **7 mantissa bits** (~3 decimal digits). A parameter update like `θ += 1e-4` applied to a value ~0.1 is representable in FP32 but would be swallowed by BF16 rounding (relative precision 2^-8 ≈ 0.4% of the value — an update of 0.1% of the value is below the representable granularity). FP32 master weights keep every update; the BF16 cast happens only for compute, where the forward pass's precision requirements are much looser.
2. **The AdamW state is FP32 anyway**: `exp_avg` and `exp_avg_sq` are FP32 tensors. The fused AdamW kernel reads FP32 params + FP32 moments and writes FP32 params — no precision conversion anywhere in the update path.

So "FP32 master weights" is not an extra copy here (unlike classic fp16 mixed precision, which keeps a separate fp32 master because fp16 params genuinely cannot hold small updates). The fp32 parameter **is** the master. The memory estimator counts it in the optimizer bucket: 2 bytes/param for working weights + 12 bytes/param for optimizer state (fp32 master + two fp32 moments) = 14 bytes/param total (`utils/memory.py:estimate_model_memory_gb`).

### The constructor in this repo

```python
decay_params = [p for p in all_params if p.dim() >= 2]
no_decay_params = [p for p in all_params if p.dim() < 2]
self.optimizer = AdamW([
    {"params": decay_params, "weight_decay": config.weight_decay},
    {"params": no_decay_params, "weight_decay": 0.0},
], lr=config.lr, betas=(config.beta1, config.beta2), fused=torch.cuda.is_available())
```

`lr=config.lr` is the μP-scaled value when `mup_lr` is on (see [μP Learning Rate Scaling](#μp-learning-rate-scaling)). `epsilon` uses PyTorch's default `1e-8`. `fused=True` on CUDA: the update is one vectorized kernel over all parameters (`m`, `v`, `θ` all resident) instead of per-tensor Python kernel launches — measurably faster and the standard choice for single-GPU LLM training.

### Why exclude 1D parameters from decay?

Weight decay on LayerNorm/RMSNorm $\gamma$ and router bias destabilises training — standard LLM recipe (GPT-3, LLaMA). Decay shrinks parameters toward zero every step; for a scale vector that is a *multiplicative* gain (RMSNorm γ multiplies the normalized activation), decay acts as a slow annealing of the whole residual-stream scale. Empirically this fights the norm's learned scale and slows convergence. For the router bias (a control variable, see [MoE Bias Updates](#moe-bias-updates)), decay would fight the load-balancing controller. Matrices — Linear weights, embeddings, projections — are the parameters where decay's regularisation is wanted. The split rule `dim >= 2` is a coarse proxy that happens to be exact for this architecture: every matrix is 2D, every scale/bias is 1D, and there are no scalar parameters (verified: `dim < 2` catches only RMSNorm γ, gate bias is a *buffer* outside the optimizer, and the final norm weight).

**Fused AdamW:** `fused=True` when CUDA available — single kernel for update, FP32 master weights internally.

---

## Gradient Accumulation — Mathematics

Effective batch size:

$$
B_{\text{eff}} = B_{\text{micro}} \times G_{\text{accum}}
$$

Gradients accumulate over $G_{\text{accum}}$ micro-steps before `optimizer.step()`:

$$
\nabla_\theta \mathcal{L}_{\text{total}} = \frac{1}{G} \sum_{g=1}^{G} \nabla_\theta \mathcal{L}_g
$$

**Implementation:** `loss = total_loss / gradient_accumulation_steps` before `backward()`. PyTorch accumulates gradients in `.grad` across micro-steps; `zero_grad(set_to_none=True)` clears after optim step.

**Why divide by G rather than summing?** The optimizer must see the *average* gradient over the effective batch, not the sum — otherwise the update magnitude would grow with G and the LR would silently stop meaning the same thing across configs. Dividing each micro-loss by G makes each `.grad` the running average of the window, so one optimizer step at accumulation G is (up to data ordering and batch-normalization effects) the same step as a single batch of size $B_{\text{eff}}$.

**Why accumulate at all instead of just using a bigger `batch_size`?** Memory: activations for one micro-batch of 8 × 2048 dominate the peak, and the backward pass recomputes them (gradient checkpointing). A direct batch of 32 would need ~4× the activation memory. Accumulation buys the optimizer's signal quality (batch 32 gradient) at the memory cost of batch 8, paying only extra forward/backward passes (which are compute-bound anyway).

**422M:** $B_{\text{micro}}=8$, $G=4$ → $B_{\text{eff}}=32$ sequences. Each optim step sees $32 \times 2048 = 65536$ tokens.

**Micro-step vs optim-step counters:**
- `global_step` in `train()` — counts every micro-batch (used for logging cadence)
- `_opt_steps` — counts actual `optimizer.step()` calls (used for MoE bias updates, scheduler)

The loop bound is `global_step < max_steps` with `global_step` counting micro-batches, so a run performs `max_steps` micro-batches and `max_steps / G` optimizer steps — 512,000 micro-batches → 128,000 optimizer steps at 422M (see [TrainingConfig Field by Field](#trainingconfig-field-by-field)).

Two subtleties that bite in practice:

1. **The division is applied to the full training loss**, which for the MTP path is `main_loss + λ·mtp_loss` — the *combined* loss is divided by G, so both heads' gradients get the same 1/G scale. The balance metric is *not* divided (it is never backpropagated — see [train_step](#train_step)).
2. **A NaN skip discards the whole accumulation window.** `train_step` returns `None` and calls `zero_grad(set_to_none=True)` when the loss is non-finite. If the NaN lands at micro-step 2 of a 4-step window, the gradients already accumulated from micro-steps 0–1 are wiped along with it — the optimizer simply never sees that partial window. This is deliberate (a window containing an Inf gradient is unusable), but it means NaN-guard rollbacks can "lose" up to G-1 clean micro-steps of gradient signal.

---

## LR Scheduler

### The closed form

`training/pretrain.py:make_warmup_cosine_lambda` returns a pure function of the optimizer-step count:

```python
def make_warmup_cosine_lambda(warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        if step >= total_steps:
            return min_lr_ratio
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda
```

In closed form, with $w$ = warmup, $T$ = total, $r$ = min ratio:

$$
\lambda(s) = \begin{cases} s / w & 0 \le s < w \\[4pt] r + (1-r)\,\dfrac{1+\cos\big(\pi \cdot \frac{s-w}{T-w}\big)}{2} & w \le s < T \\[4pt] r & s \ge T \end{cases}
$$

The actual LR seen by the optimizer is $\eta(s) = \eta_{\text{base}} \cdot \lambda(s)$, because `LambdaLR` multiplies the optimizer's base LR by `lr_lambda(step)`.

**Continuity is exact at every seam** (a property the tests pin down):
- At $s = w$: warmup branch gives $w/w = 1$; cosine branch has progress 0, so $r + (1-r)\cdot(1+\cos 0)/2 = r + (1-r) = 1$. ✓
- At $s = T$: cosine gives $r + (1-r)\cdot(1+\cos\pi)/2 = r + (1-r)\cdot 0 = r$; the $\ge T$ branch also returns $r$. ✓
- The cosine is symmetric: progress 0 → 1, progress 0.5 → $r + (1-r)/2$ (the midpoint multiplier is the average of peak and floor — with $r=0.05$ that is 0.525), progress 1 → $r$.

For the canonical config the key points are (all in **optimizer-step** space, $s$ = `scheduler.step()` count):

| $s$ | $\lambda(s)$ | actual LR (with MTP base 8.069e-4) |
|---|---|---|
| 0 | 0 | 0 |
| 2000 (end of warmup) | 1.0 | 8.069e-4 |
| 64,000 (cosine midpoint) | 0.525 | 4.24e-4 |
| 128,000 (run end = horizon) | 0.05 | 4.03e-5 |
| ≥ 128,001 | 0.05 (clamped) | 4.03e-5 |

**The horizon is auto-scaled to optimizer steps.** `train_step` advances the scheduler once per *optimizer* step, but `max_steps` is a *micro*-step loop budget; the scheduler is therefore constructed with `total_steps = max_steps // gradient_accumulation_steps` (see [Scheduler construction](#scheduler-construction)). For the canonical run that is 512,000 ÷ 4 = 128,000 — exactly the number of optimizer steps the loop performs — so the cosine arc completes and the LR reaches the 5% floor at the final step. Warmup's 2,000 steps are also in optimizer-step space (1.6% of the run). If you change `max_steps` or `gradient_accumulation_steps`, the horizon follows automatically; only `warmup_steps` needs manual rescaling for a different run length.

### Why warmup, cosine, and a floor?

- **Warmup exists because Adam's early steps are miscalibrated.** Both moment estimates start at zero; for the first handful of steps $\hat v_t$ is tiny, so $\eta / \sqrt{\hat v_t}$ would be huge. Linear warmup from 0 lets the moments fill in before the LR reaches full strength. 2000 steps ≈ 1.6% of the run's 128,000 optimizer steps — the standard order of magnitude for LLMs.
- **Cosine decay** anneals smoothly: unlike step decay (sudden cliffs) or linear decay (constant slope), cosine's slope is ~0 at both ends and steepest mid-decay, spending more time at both high and low LR — empirically good for the loss landscape of a transformer.
- **The floor `min_lr_ratio=0.05`** keeps the tail from collapsing to exactly zero: zero LR means no further learning *and* no gradient signal to guide the final phase; 5% of peak (≈ 4e-5) still lets the model settle. It also bounds the effective number of "useful" steps, which interacts with the total-step budget.

### Binding and off-by-one

The scheduler is stepped exactly once per optimizer step, from `train_step`:

```python
self.optimizer.step()
self.scheduler.step()
```

Because `scheduler.step()` fires *after* `optimizer.step()`, optimizer step $t$ executes with LR $\eta \cdot \lambda(t-1)$. In particular the **very first optimizer step runs at $\lambda(0) = 0$** — a literal no-op (weight decay is also scaled by $\eta$, so nothing moves). This is benign because warmup starts at zero anyway, but it means the effective schedule is shifted by one step: optimizer step $k$ uses $\lambda(k-1)$. Logging uses `scheduler.get_last_lr()[0]`, which after $k$ scheduler steps reports $\lambda(k)$ — so the first log line at micro-step 0 reports `lr=0.00e+00`.

Edge cases handled by `max(1, ...)`: `warmup_steps=0` (cosine from step 0, λ(0)=1 — covered by `test_no_warmup`) and `total_steps == warmup_steps` (no cosine room; everything past warmup clamps to $r$). Tests: `tests/test_training.py` `TestWarmupCosineScheduler.test_values_at_key_points`, `test_monotonic_warmup`, `test_cosine_decay`, plus the boundary tests `test_before_warmup_is_zero`, `test_at_warmup_end_is_one`, `test_at_total_steps_is_min_ratio`, `test_past_total_steps_clamps_to_min`.

---

## μP Learning Rate Scaling

**Maximal Update Parametrisation (μP)** transfers hyperparameters across model scales:

```
new_lr = mup_lr_reference × (mup_lr_reference_params / total_params)^0.5
```

422M config:
- Reference: `6.0e-4` at `757,226,496` params (~757M reference model)
- Total counted **after** MTP wrap (includes MTP head params): 418.7M
- Result: **~8.07e-4**

**Why sqrt?** μP theory: optimal LR scales inversely with sqrt(width) for width-dependent init schemes. The reference model was tuned at 757M; this formula transfers that tuning to this repo's 411.6M base / 418.7M with MTP.

`test_mup_lr_scaling` verifies the computation.

### The real numbers

The code that does this, verbatim from `training/pretrain.py:Pretrainer.__init__`:

```python
if config.mup_lr:
    new_lr = config.mup_lr_reference * (config.mup_lr_reference_params / total) ** 0.5
    self._log(f"µP LR scaling: {config.lr:.2e} → {new_lr:.2e} (ref {config.mup_lr_reference:.2e} @ {config.mup_lr_reference_params:,} params)")
    config.lr = new_lr
```

With the locked constants the arithmetic is:

$$
\eta_{\text{base}} = 6.0\times10^{-4} \times \sqrt{\frac{757\,226\,496}{411\,632\,256}}
= 6.0\times10^{-4} \times \sqrt{1.8396}
= 6.0\times10^{-4} \times 1.3563
= 8.138\times10^{-4}
$$

$$
\eta_{\text{mtp}} = 6.0\times10^{-4} \times \sqrt{\frac{757\,226\,496}{418\,713\,984}}
= 6.0\times10^{-4} \times \sqrt{1.8085}
= 6.0\times10^{-4} \times 1.3448
= 8.069\times10^{-4}
$$

So: **8.138e-4 for the bare 411.6M model, 8.069e-4 with the MTP wrapper**. Both are "~8.1e-4" — close to the YAML's nominal 8.0e-4, which is why the config's `lr: 8.0e-4` looks "already right". It is a coincidence of the two reference anchors; the formula, not the YAML value, is what trains.

**Two properties worth internalising:**

1. **The YAML `lr` key is dead config when `mup_lr: true`.** `new_lr` depends only on `mup_lr_reference`, `mup_lr_reference_params`, and `total`. The YAML `lr` appears *only* in the log message's "before" slot. If you tune `lr` in YAML expecting it to matter while `mup_lr: true`, nothing changes — edit `mup_lr_reference` instead, or set `mup_lr: false`.
2. **The denominator is the deduped post-MTP count.** Enabling MTP *lowers* the LR (8.138 → 8.069e-4) because the wrapper adds 7.08M parameters. The μP argument is that the update per parameter should stay at the reference scale; more parameters at the same LR would overshoot.

### Why sqrt — the intuition

The reference model (757.2M params) was tuned to LR 6.0e-4 — someone swept it, or it inherits a known-good family value. μP asks: *what LR should a differently-sized model of the same family use so that the training dynamics match?* The answer for standard (Pytorch-default) init and Adam-family optimizers is that per-parameter update magnitudes are width-dependent: as the hidden width $d$ grows, the typical activation magnitude and the typical gradient magnitude each scale, and the combination means the safe LR scales like $1/\sqrt{\text{width}}$ in the "maximal update" regime — the regime where every parameter keeps learning at the same rate regardless of scale. Since parameter count $N$ grows roughly quadratically in width for a fixed-depth transformer (the dense layers contribute $O(d^2)$ each), $\sqrt{N_{\text{ref}}/N}$ is the width-ratio proxy:

$$
\frac{\eta_{\text{new}}}{\eta_{\text{ref}}} \approx \frac{d_{\text{ref}}}{d_{\text{new}}} \approx \sqrt{\frac{N_{\text{ref}}}{N_{\text{new}}}}.
$$

That is the entire formula: **scale LR by the inverse square root of the parameter-count ratio**. It is a transfer recipe, not a guarantee — the reference anchor (757,226,496 @ 6.0e-4) is external to this repo, and the honest reading is "8.07e-4 is our best transfer estimate; the warmup, clipping, and NaN guard are the safety net if the transfer is imperfect."

### Testing

- `tests/test_training.py::TestPretrainerConstruction::test_mup_lr_scaling` — builds a small `Pretrainer`, enables `mup_lr`, asserts the optimizer LR equals `mup_lr_reference × sqrt(ref/total)`.
- `test_factor_one_when_total_equals_reference` — when the model *is* the reference size, the factor is 1.0 and the LR is unchanged (the formula's identity check).
- `test_mup_disabled_no_scaling` — with `mup_lr: false`, the YAML LR passes through untouched.
---

## PretrainDataset

Packed contiguous token windows from pre-tokenised data. `PretrainDataset` (`training/pretrain.py:PretrainDataset`) is a `torch.utils.data.Dataset` over a flat token stream: it never tokenizes, never pads, and never randomizes within a sample — sample $i$ is simply the contiguous window starting at token $i \times S$.

### Single-file layout

```
data/pretrain_data.bin  →  torch tensor (N,) uint32
sample i: tokens = data[i*S : i*S+S], targets = data[i*S+1 : i*S+S+1]
```

### Sharded layout

```
data/pretrain_chinchilla/
  shard_000.bin  (≈50M tokens each — exact sizes vary by preparation)
  shard_001.bin
  ...
```

- `_locate(global_idx)` — binary search over `shard_offsets`
- Cross-shard windows: stitch pieces with `torch.cat`
- `mmap=True` for memory efficiency
- Final partial chunk **dropped** (not padded)

### Dtype boundary

Dataset stores `uint32` (4 bytes/token). `train_step` casts to `int64` before `nn.Embedding` and `cross_entropy`.

### Construction

```python
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Pre-training data not found: {data_path}\nRun `python data/prepare_data.py` first.")
self.layout = "sharded" if os.path.isdir(data_path) else "single"
if self.layout == "sharded":
    self.shard_paths = [str(p) for p in sorted(Path(data_path).glob("shard_*.bin"))]
    if not self.shard_paths:
        raise FileNotFoundError(f"No `shard_*.bin` files in {data_path}")
    self.shards = [torch.load(p, weights_only=True, map_location="cpu", mmap=True) for p in self.shard_paths]
    self.shard_sizes = [s.numel() for s in self.shards]
    self.shard_offsets = []
    running = 0
    for s in self.shard_sizes:
        self.shard_offsets.append(running)
        running += s
    self._total_tokens = sum(self.shard_sizes)
else:
    self.data = torch.load(data_path, weights_only=True, map_location="cpu", mmap=True)
    self._total_tokens = self.data.numel()
self._n_samples = (self._total_tokens - 1) // self.max_seq_len
```

`mmap=True` means shards are never fully resident: pages are faulted in on access and the OS evicts them, so a 30+ GB corpus can be streamed from a single process. `weights_only=True` rejects pickle gadgets. Shard files are matched by the literal `shard_*.bin` glob — a directory containing other files is fine, but a shard named `part_0.bin` will be silently ignored. The layout is decided by `os.path.isdir`, so the *same* `data_path` contract serves both forms. The full corpus is never concatenated: only `shard_offsets` (cumulative token counts) is materialised.

### The bisect math — `_locate`

```python
def _locate(self, global_idx: int) -> Tuple[int, int]:
    """Map a global token index to (shard_idx, offset_within_shard). ..."""
    if global_idx < 0 or global_idx >= self._total_tokens:
        raise IndexError(
            f"global_idx {global_idx} out of range [0, {self._total_tokens})"
        )
    import bisect
    lo = bisect.bisect_right(self.shard_offsets, global_idx) - 1
    return lo, global_idx - self.shard_offsets[lo]
```

`shard_offsets` is the list of starting indices: `[0, |s₀|, |s₀|+|s₁|, …]`. For a global index $g$, the containing shard is the **last** shard whose start $\le g$:

$$
\text{shard}(g) = \max\{k : \text{offsets}[k] \le g\}, \qquad \text{offset}(g) = g - \text{offsets}[k].
$$

`bisect.bisect_right(offsets, g) - 1` is exactly that: `bisect_right` returns the insertion point *after* any equal element, so when $g$ equals a shard boundary, the result points one past the boundary shard — minus one lands on the next shard, which is correct because the boundary token belongs to the shard whose start it is. (`bisect_left` would mis-assign boundary indices to the previous shard.) Worked example, shards of 100 / 50 / 75 tokens → offsets `[0, 100, 150]`:

| $g$ | `bisect_right` | shard | offset |
|---|---|---|---|
| 0 | 1 | 0 | 0 |
| 99 | 1 | 0 | 99 |
| 100 | 2 | 1 | 0 |
| 149 | 2 | 1 | 49 |
| 150 | 3 | 2 | 0 |
| 224 | 3 | 2 | 74 |

Each lookup is $O(\log K)$ in the number of shards (Python's C bisect), so per-sample cost is negligible even for thousands of shards. Out-of-range indices raise `IndexError` — the check `global_idx >= self._total_tokens` matters because `_total_tokens - 1` is the last valid token (the targets need the token *after* the window).

### `__getitem__` walkthrough

```python
def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    start = idx * self.max_seq_len
    needed = self.max_seq_len + 1
    if self.layout == "single":
        chunk = self.data[start: start + needed].clone()
    else:
        pieces = []
        cursor = start
        cursor_pos = 0
        while cursor_pos < needed:
            shard_idx, offset_in_shard = self._locate(cursor)
            shard = self.shards[shard_idx]
            take = min(needed - cursor_pos, self.shard_sizes[shard_idx] - offset_in_shard)
            pieces.append(shard[offset_in_shard: offset_in_shard + take])
            cursor += take
            cursor_pos += take
        chunk = torch.cat(pieces) if len(pieces) > 1 else pieces[0].clone()
    return chunk[:-1], chunk[1:]
```

Step by step:

1. `start = idx * max_seq_len` — windows tile the stream with stride $S$; sample $i$ owns tokens $[iS, iS+S)$.
2. `needed = S + 1` — one extra token so `chunk[:-1]` (inputs) and `chunk[1:]` (targets) both have length $S$ and the targets are the inputs shifted by one. This is the whole "next-token prediction" contract: sample `i` predicts `x[iS+1 : iS+S+1]` from `x[iS : iS+S]`.
3. Single-file: one slice + `.clone()`. The clone is not cosmetic — the slice is a *view* into the mmap'd tensor, and views into mmap memory are exactly what you do not want crossing process boundaries or outliving the mapping; the copy makes each sample a standalone tensor.
4. Sharded: a `while` loop that can cross any number of shard boundaries. Each iteration locates the shard containing the current `cursor`, takes as many tokens as fit in that shard (`min(needed - cursor_pos, shard_size - offset_in_shard)`), appends the slice, and advances. `torch.cat` joins the pieces; a window that happens to fit in one shard still gets `.clone()`d.
5. Return `chunk[:-1], chunk[1:]` — two views of the same cloned buffer; the DataLoader collates them into the `(B, S)` tokens and targets tensors that `train_step` receives.

**Length and the dropped tail:** `_n_samples = (total_tokens - 1) // S`. The final sample starts at `(_n_samples - 1) * S` and needs `S + 1` tokens, which is guaranteed to exist by the floor; whatever remains after that (fewer than $S$ tokens, since the next window would not fit) is **dropped, not padded** — padding would inject fake tokens into the training distribution. The drop is guarded by `tests/test_training.py::TestPretrainDataset::test_final_sample_truncated`. One consequence: corpus position and sample index are a *bijection* — there is no per-sample random offset, so every token (except the last $S$ of the stream) is seen exactly once per epoch, and the only randomness in data order comes from the DataLoader shuffle of sample indices.

Other guards: `test_single_file`, `test_single_file_shift` (targets are inputs shifted by one), `test_sharded_dataset`, `test_sharded_cross_boundary` (a window spanning ≥ 2 shards), `test_missing_file_raises`, `test_locate_edge_case`, `test_locate_out_of_range_raises`.

---

## DataLoader Design

```python
loader = DataLoader(
    dataset, batch_size=8, shuffle=True, generator=g,
    num_workers=8, pin_memory=True,
    persistent_workers=True, prefetch_factor=8, drop_last=True,
)
```

(This is `training/pretrain.py:Pretrainer.train` with the config fields substituted; `batch_size` is `config.batch_size` — the micro-batch.)

| Flag | Rationale |
|---|---|
| `shuffle=True` + seeded `Generator` | Different order each epoch; reproducible given same seed |
| `num_workers=8` | Parallel `__getitem__` while GPU trains |
| `pin_memory=True` | Faster H2D copy via page-locked host memory |
| `persistent_workers=True` | Avoid worker respawn overhead each epoch |
| `prefetch_factor=8` | Pipeline 8 batches per worker ahead of GPU |
| `drop_last=True` | Avoid partial batch shape mismatch at epoch end |

### The seeded shuffle

```python
g = torch.Generator().manual_seed(self.config.seed)
```

The generator is created once, before the epoch loop, with the *same* `config.seed` used for weight init. DataLoader's shuffle sampler draws permutation indices from `g`, so: same seed → same first-epoch order; each subsequent epoch draws a fresh permutation (the sampler does not reset `g`, so epoch 2's order differs from epoch 1's — that is the point, no more identical sequential order every epoch).

### Worker semantics

With `num_workers=8`, `__getitem__` runs in 8 child processes; each worker holds its own copy of the `Dataset` object (fork semantics), so the mmap'd shards are shared at the OS page level — 8 workers do not multiply corpus memory. `persistent_workers=True` keeps those processes alive across epochs instead of respawning, which matters here because each worker re-warms its page cache. `prefetch_factor=8` fills 8 batches per worker ahead of the GPU's demand, which is what keeps a 15-hour run from ever idling on data. `pin_memory=True` makes the H2D copies in `train()` (`tokens.to(self.device, non_blocking=True)`) overlap with compute.

`drop_last=True` matters for a subtle reason beyond shape hygiene: `train_step` divides by `gradient_accumulation_steps` and the boundary test requires *exactly* `G` micro-batches per optimizer window. A partial final batch would both break the `(B,S)` shape contract and shift the accumulation cadence.

### Resume caveat

Sampler RNG is **not** checkpointed — resume changes data order (benign at 8.4B scale). More precisely: after a restart, the *same* `config.seed` re-creates `g`, so the new run replays the same epoch-1 order from the beginning, while `global_step` resumes mid-run. The result is that a resumed run sees a re-shuffled corpus from epoch 1 onward — no sample is skipped, the order is simply different from what a never-interrupted run would have seen.

---

## train_step

`train_step` (`training/pretrain.py:Pretrainer.train_step`) is the entire training math, in one function:

```python
def train_step(self, tokens: torch.Tensor, targets: torch.Tensor, micro_step: int) -> Optional[Dict[str, Optional[float]]]:
    is_opt_step = (micro_step + 1) % self.config.gradient_accumulation_steps == 0
    # Cast uint32 → int64 at the boundary. PretrainDataset stores tokens as
    # uint32 for memory efficiency (4 bytes vs 8), but `nn.Embedding` and
    # `F.cross_entropy` both require Long indices. Doing the cast here keeps
    # the dataset's storage compact and the training path dtype-correct.
    if tokens.dtype != torch.long:
        tokens = tokens.to(torch.long)
    if targets.dtype != torch.long:
        targets = targets.to(torch.long)
    with self._amp_context():
        if self.mtp_wrapper is not None:
            main_logits, mtp_pairs = self.model(tokens)
            total_loss, main_loss, mtp_loss = self.mtp_wrapper.compute_loss(main_logits, targets, mtp_pairs)
            # Defer host round-trip to the logger (avoids per-step GPU sync).
            _ce_loss_val = main_loss.detach()
            _mtp_loss_val = mtp_loss.detach() if mtp_pairs else None
            loss = total_loss / self.config.gradient_accumulation_steps
        else:
            logits = self.model(tokens, start_pos=0, use_cache=False)
            main_loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
            _ce_loss_val = main_loss.detach()
            _mtp_loss_val = None
            loss = main_loss / self.config.gradient_accumulation_steps
        balance_loss = self._moe_balance_metric()

    if self.config.nan_guard and (torch.isnan(loss).any().item() or torch.isinf(loss).any().item()):
        self._log(f"[nan-guard] NaN/Inf at micro_step={micro_step}, opt_steps={self._opt_steps}. Skipping backward.")
        self.optimizer.zero_grad(set_to_none=True)
        return None

    loss.backward()
    if is_opt_step:
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self._opt_steps += 1
        if self._opt_steps % self.config.bias_update_every == 0:
            self._update_moe_bias()

    return {"loss": _ce_loss_val, "mtp_loss": _mtp_loss_val, "balance_loss": balance_loss}
```

### The optimizer-step boundary

```python
is_opt_step = (micro_step + 1) % self.config.gradient_accumulation_steps == 0
```

With $G = 4$, `is_opt_step` is true at `micro_step ∈ {3, 7, 11, …}` — the *last* micro-step of each window. Everything below the boundary line (clip, step, scheduler, zero, bias) runs only then. This is the single source of truth for the micro/opt distinction: `global_step` (the loop counter, passed in as `micro_step`) counts windows' micro-steps; `_opt_steps` counts how many times the boundary was crossed. For a worked table see [Worked Example — Micro-step vs Optim Step](#worked-example--micro-step-vs-optim-step).

### The uint32 → int64 cast

The dataset stores `uint32` (4 bytes/token) so a 33 GB corpus of token ids costs 33 GB, not 66. `nn.Embedding` and `F.cross_entropy` both require `torch.long` (int64) indices — the cast happens here, once, at the boundary between the data path and the compute path, after the batch has moved to the GPU in `train()`. The comment in the source is explicit about why the cast lives here and not in the dataset: keeping storage compact *and* the training path dtype-correct.

### The autocast forward

`self._amp_context()` is `autocast(self.device.type, dtype=torch.bfloat16)`. Inside it:

- **MTP branch:** `self.model(tokens)` runs the whole `MultiTokenPrediction.forward` — trunk *and* heads — under one autocast scope; `compute_loss` returns `(total_loss, main_loss, mtp_loss)` and the *combined* loss is divided by $G$.
- **Standard branch:** `self.model(tokens, start_pos=0, use_cache=False)` — `use_cache=False` is load-bearing: training never writes a KV cache (invariant #1; see also [Memory Timeline](#memory-timeline--one-train_step)). The cross-entropy flattens logits to `(B·S, V)` against targets `(B·S,)` with `ignore_index=-100` — `-100` is the PyTorch convention for "mask this position out of the loss"; the current dataset never emits it (no padding), so it is a forward-compatible contract for padded batches rather than an active mask.
- **What runs in what precision:** under autocast(bf16), the matmuls (Linear, Embedding, SDPA) execute in BF16 with FP32 accumulation inside the kernels; the normalization layers and the cross-entropy loss run in FP32 per PyTorch's autocast policy. The model's stored parameters stay FP32 — autocast casts transiently, it never mutates them (see [Why FP32 master weights](#why-fp32-master-weights)).
- **`balance_loss`** is computed *inside* the autocast scope but is log-only: it is a sum over MoE layers of `get_load_balance_loss()`, never backpropagated, and `.item()` is deferred to the logger path to avoid a per-micro-step GPU sync.

### NaN skip

When `nan_guard` is on and the divided loss is NaN or Inf, `train_step` logs, zeroes *all* gradients (discarding the partial window — see [Gradient Accumulation](#gradient-accumulation--mathematics)), and returns `None`. The caller (`train()`) treats `None` as "this micro-step contributed nothing" and feeds the streak counter. Note the check happens **before** `loss.backward()`, so a poisoned batch never contaminates parameters — that is what makes rollback (see [NaN Guard](#nan-guard)) a recovery, not a repair.

### Backward, clip, step

`loss.backward()` accumulates into `.grad` (FP32) across the window. On the boundary:

- `clip_grad_norm_(self.model.parameters(), max_grad_norm)` computes the global norm $||g||_2 = \sqrt{\sum_i ||g_i||_2^2}$ over all parameter gradients and rescales them by $\min(1, \text{max\_norm}/||g||_2)$ — so clipping only ever *shrinks* gradients, never grows them, and a single exploding expert gradient cannot drag the whole step off (MoE routing spikes are the classic source; see [Hyperparameter Sensitivity](#hyperparameter-sensitivity-empirical)).
- `optimizer.step()` applies AdamW (FP32), `scheduler.step()` advances the warmup-cosine curve, `zero_grad(set_to_none=True)` frees the gradient buffers (set_to_none is faster than zeroing and releases memory), `_opt_steps += 1`, and the MoE bias controller fires on its cadence.

### The return value and deferred sync

The dict's `"loss"` key is **`main_loss`** (the trunk cross-entropy), *not* `total_loss` — the MTP penalty is reported separately under `"mtp_loss"`. All three values are GPU tensors (`detach()`ed); the only `.item()` calls happen in `train()`'s logging branch (`global_step % log_every == 0`), so a micro-step that is not logged performs **zero** host-device synchronizations for metrics. The balance loss `.item()` is likewise deferred to the same branch. This is a deliberate, measured choice: forced syncs stall the pipeline 3–4 times per micro-step if done naively.

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

### Forward and length alignment

The wrapper's forward (`models/mtp.py:MultiTokenPrediction.forward`) runs the trunk via `forward_with_hidden`, which returns both the logits **and** the pre-norm trunk hidden state `h` (the raw residual-stream output — MTP blocks apply their own norms, so the *pre*-norm `h` is what they consume). Then, per depth $d$ (0-indexed):

$$
\text{usable}_d = S - d - 2
$$

For depth 0 with $S = 2048$: usable = 2046. The three slices are

- `h_in = h[:, :usable]` — the last 2046 trunk hidden states,
- `emb_in = embed(tokens[:, 1 : usable+1])` — the *next* token's embedding (the token the MTP head should condition on, i.e. $x_{t+1}$),
- `tgt = tokens[:, 2 : usable+2]` — the target two positions ahead ($x_{t+2}$).

The MTP block fuses `[norm_h(h_t), norm_e(emb_{t+1})]` through a projection, runs causal self-attention (its own `nn.MultiheadAttention`, not MLA, with an internal cached triangular mask via `MTPBlock._get_causal_mask`) and a SwiGLU, and the head predicts $x_{t+2}$. Length alignment is exact: each depth trims two positions (one for the conditioning token, one for the target shift), so `usable` shrinks by one per additional depth — `usable = seq_len - d - 2` — and the code `break`s out if usable ≤ 0. The tail trim means the MTP loss supervises $8 \times 2046 = 16\,368$ positions per micro-step (vs 16,384 for the main loss), or 65,472 per optimizer step.

### compute_loss

```python
def compute_loss(self, main_logits, targets, mtp_pairs=None):
    """Returns (total_loss, main_loss, mtp_loss). MTP loss is mean across depths."""
    main_loss = F.cross_entropy(main_logits.reshape(-1, main_logits.size(-1)), targets.reshape(-1), ignore_index=-100)
    if not mtp_pairs:
        return main_loss, main_loss, main_loss.new_zeros(())
    depth_losses: List[torch.Tensor] = []
    for logits, tgt in mtp_pairs:
        if tgt.numel() == 0:
            continue
        depth_losses.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-100))
    mtp_loss = torch.stack(depth_losses).mean() if depth_losses else main_loss.new_zeros(())
    return main_loss + self.mtp_weight * mtp_loss, main_loss, mtp_loss
```

(`models/mtp.py:MultiTokenPrediction.compute_loss`.) Each depth's CE is computed on its aligned slice; `mtp_loss` is the **mean across depths** (so with $D=1$ it is just that depth's CE); the total is `main + λ·mtp` with λ = `mtp_weight` = 0.3. The trunk receives gradient from both terms: the main CE through the LM head, and the MTP term through the fusion block reading `h` — which is why MTP is a *training* regularizer (it sharpens trunk representations for next-token prediction one step ahead) and why its weight must be small enough not to overpower the main objective. The shared `embed` and `head` accumulate gradients from both paths (they are the same tensors), which is exactly why the optimizer dedup by `id(p)` matters — without it, the shared tensors would appear twice in the optimizer and get double-updated. Full MTP theory in [MTP](concepts/moe-mtp.md).

---

## MoE Bias Updates

```python
if self._opt_steps % self.config.bias_update_every == 0:
    for moe in self.raw_model.moe_layers():
        moe.update_gate_bias(speed=self.config.bias_update_speed)
```

Uses routing counts from the **last forward's** `_last_indices`. With `bias_update_every=1`, every optimizer step updates bias based on that step's routing distribution.

See [MoE](concepts/moe-mtp.md) for the full bias mechanism.

### The control loop

This is the aux-loss-free load balancer from DeepSeek-V3 §2.3.3, implemented as a **discrete-time controller** rather than a loss term. The plant is the router; the measured output is the per-expert token count over the last forward; the actuator is the gate's bias vector; the controller is `AuxLossFreeGate.update_bias` (`models/moe.py:AuxLossFreeGate.update_bias`):

```python
@torch.no_grad()
def update_bias(self, counts: torch.Tensor, speed: float = 0.001) -> None:
    counts = counts.float()
    avg = counts.mean()
    self.bias[counts > avg * (1.0 + self.bias_upper)] -= speed
    self.bias[counts < avg * (1.0 - self.bias_lower)] += speed
```

With the canonical `bias_upper = bias_lower = 0.10`:

- an expert routed more than **10% above** the average count gets its bias **decreased** by `speed` (its sigmoid scores drop, so it is selected less);
- an expert routed more than **10% below** average gets its bias **increased**;
- experts within the ±10% deadband are untouched (no dithering).

Because `bias` is added to the gate *scores before* top-k selection but the routed **weights** come from the bias-free sigmoid scores, the bias steers *which* experts run without corrupting the mixing weights — and because the whole update is `@torch.no_grad()` on a **buffer** (not a parameter), no gradient ever flows through it and AdamW never sees it (it is also excluded from weight decay by construction — it is not even in a param group).

### Timing

`Pretrainer._update_moe_bias` (`training/pretrain.py:Pretrainer._update_moe_bias`) iterates `self.raw_model.moe_layers()` (`models/transformer.py:Transformer.moe_layers`, a generator yielding the `DeepSeekMoE` modules of layers 2–17) and calls `DeepSeekMoE.update_gate_bias` (`models/moe.py:DeepSeekMoE.update_gate_bias`), which bin-counts `self._last_indices` — the routing decisions of the **last forward**, recorded in `DeepSeekMoE.forward` as `self._last_weights = weights.detach(); self._last_indices = indices.detach()`.

The cadence is: after the optimizer step, when `_opt_steps % bias_update_every == 0`. With the canonical `bias_update_every: 1` the bias therefore reacts to the routing distribution of the last micro-step of each accumulation window — a one-window delay, which is fine for a controller whose whole job is slow correction (speed 0.001, i.e. 0.1% of the bias scale per update; the bias lives in a sane range because it is nudged, not optimized). Control-theory reading: proportional control on imbalance with a deadband, deliberately *slower* than gradient descent so it does not fight the router's learned specialization. `bias_update_every: 10` (the dataclass default) slows the loop 10× for smoke runs.

The **balance metric** (`models/moe.py:DeepSeekMoE.get_load_balance_loss`) is the observable of the same system: with $f_e$ = empirical expert fraction (from counts) and $P_e$ = mean routing probability per expert,

$$
\mathcal{L}_{\text{bal}} = E \sum_{e=1}^{E} f_e P_e
$$

It is logged (`balance_loss` in train_step's return) as the monitoring signal for routing health — a value creeping up means the gate is concentrating, and the bias controller should be pushing back.

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

### The skip path

The guard has two layers. Layer one is inside `train_step` (see [train_step](#train_step)): on a non-finite *divided* loss it logs, zeroes gradients, returns `None`. Layer two is the streak logic in `train()` — the state machine:

```python
nan_guard_streak = 0
while global_step < self.config.max_steps:
    for tokens, targets in tqdm(loader):
        ...
        metrics = self.train_step(tokens, targets, global_step)
        if metrics is None:
            nan_guard_streak += 1
            if nan_guard_streak >= self.config.nan_guard_max_consecutive:
                latest = self._find_latest_checkpoint()
                if latest is not None:
                    self._log(f"[nan-guard] {nan_guard_streak} consecutive NaN/Inf — restoring checkpoint step {latest}.")
                    global_step = self.load_checkpoint(latest)
                else:
                    self._log("[nan-guard] No checkpoint to restore from. Aborting.")
                    raise RuntimeError("NaN/Inf with no checkpoint to restore from")
                nan_guard_streak = 0
            continue
        nan_guard_streak = 0
        ...
```

(`training/pretrain.py:Pretrainer.train`.) States: **armed** (streak < 5 — skip and keep going), **triggered** (streak ≥ 5 — roll back), **reset** (streak = 0 — either after a successful step or after a rollback), and **abort** (no checkpoint exists — `RuntimeError`). Every finite-loss micro-step resets the streak to 0, so the threshold means *consecutive* failures.

**Why 5 consecutive?** Single bad micro-batches happen from MoE routing spikes; five in a row signals true divergence (LR too high, corrupt shard, dtype bug).

**422M default:** `nan_guard: true`, `nan_guard_max_consecutive: 5` in YAML.

### Rollback semantics

`load_checkpoint(latest)` restores model weights, AdamW state, scheduler state, and `_opt_steps` (see [Checkpointing](#checkpointing)), then returns the checkpoint's step, which becomes the new `global_step`. Two properties are worth understanding precisely:

1. **The step counter rewinds; the data stream does not.** `load_checkpoint` does not recreate the DataLoader, so iteration continues from wherever the loader was — the batches that caused the divergence are *not* replayed (the loader is ahead of the counter by the rollback distance). The run therefore consumes up to `(diverged_steps − checkpoint_step)` extra micro-batches over its nominal budget before `global_step` catches back up to `max_steps`. At 8.4B-token scale with rollbacks measured in a handful of steps this is a rounding error on corpus exposure.
2. **Rollback targets the latest *complete* checkpoint**, not necessarily the most recent save. `CheckpointManager.latest_step` walks saved steps newest-first and returns the first whose three files all exist (see [Checkpointing](#checkpointing)) — a torn checkpoint (crash mid-save) is skipped, so the guard can never roll back onto a half-written state.

Tests: `test_train_step_returns_none_on_nan` (skip path returns `None` without stepping) and `test_consecutive_nan_triggers_rollback` (streak → load latest checkpoint, reset streak).

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

### save_checkpoint

```python
def save_checkpoint(self, step: int, tag: str = "") -> None:
    model_to_save = self.raw_model
    state = model_to_save.state_dict()
    # Weight tying: head.weight IS embed.weight (same tensor). Dropping the
    # duplicate saves ~vocab×dim×4B per checkpoint; load_state_dict(strict=False)
    # leaves head.weight missing, but the shared tensor is restored via embed.
    if getattr(model_to_save, "weight_tying", False):
        state = {k: v for k, v in state.items() if k != "head.weight"}
    if self.mtp_wrapper is not None:
        mtp_mod = self.mtp_wrapper
        orig = getattr(mtp_mod, "_orig_mod", mtp_mod)
        mtp_state = {f"mtp.{k}": v for k, v in orig.state_dict().items() if k.startswith("mtp_modules.")}
        state.update(mtp_state)
    extra_meta = {"scheduler": self.scheduler.state_dict(), "opt_steps": self._opt_steps,
                  "tag": tag or f"step_{step}", "config": asdict(self.config), "has_mtp": self.mtp_wrapper is not None}
    self.ckpt_manager.save(model_to_save, self.optimizer, step, extra_meta=extra_meta, state_dict=state)
```

(`training/pretrain.py:Pretrainer.save_checkpoint`.) Three details carry the weight:

- **`head.weight` is dropped** when tying is on — it is the same tensor as `embed.weight`, so keeping it would double the checkpoint's embedding bytes (~307 MB at 100,018 × 768 × 4 B) and, worse, store two copies that could *drift apart* if someone ever loaded them independently. `load_state_dict(strict=False)` tolerates the missing key; the shared storage is restored through `embed.weight`. The same dedup exists a second time inside `CheckpointManager._atomic_save_safetensors` (`utils/checkpoint.py:CheckpointManager._atomic_save_safetensors`), which skips any tensor whose `data_ptr()` was already seen — belt and suspenders for any future sharing.
- **MTP weights ride along with a `mtp.` prefix**, and only `mtp_modules.*` keys — the shared embed/head are already in the trunk state, and prefixing avoids any key collision with the trunk's own names. The `_orig_mod` unwrap handles the compiled wrapper (whose `state_dict` would otherwise carry `_orig_mod.` prefixes).
- **The meta payload is the full resume contract**: scheduler state (so the LR curve continues where it left off), `_opt_steps` (so the MoE bias cadence and any bias_update_every logic resume), the `config` snapshot (so a resumed run can detect that its YAML changed), and `has_mtp` (so the loader knows to look for `mtp.*` keys).

The file-by-file layout and atomic write mechanics are in [Checkpoint Format — File-by-File](#checkpoint-format--file-by-file).

### The load path

```python
def load_checkpoint(self, step: int) -> int:
    from safetensors.torch import load_file
    meta = self.ckpt_manager.load(self.raw_model, step, device=str(self.device), optimizer=self.optimizer, strict=False)
    if self.mtp_wrapper is not None and meta.get("has_mtp", False):
        weight_path = self.ckpt_manager.save_dir / f"model_step_{step}.safetensors"
        if weight_path.exists():
            state = load_file(str(weight_path), device=str(self.device))
            mtp_state = {k.removeprefix("mtp."): v for k, v in state.items() if k.startswith("mtp.")}
            if mtp_state:
                mtp_orig = getattr(self.mtp_wrapper, "_orig_mod", self.mtp_wrapper)
                mtp_orig.load_state_dict(mtp_state, strict=False)
                self._log(f"MTP weights restored ({len(mtp_state)} keys)")
    if "scheduler" in meta:
        self.scheduler.load_state_dict(meta["scheduler"])
    if "opt_steps" in meta:
        self._opt_steps = meta["opt_steps"]
    resumed_step = meta.get("step", step)
    self._log(f"Resumed from step {resumed_step}")
    return resumed_step
```

(`training/pretrain.py:Pretrainer.load_checkpoint`.) The trunk is restored by `CheckpointManager.load` (`utils/checkpoint.py:CheckpointManager.load`) with `strict=False` — required, since `head.weight` is absent — and the optimizer state is pulled from `optim_step_N.pt` *if it exists* (its absence only warns: the optimizer restarts from scratch, which is a silent-but-survivable degradation worth knowing about if you ever hand-copy checkpoint files). MTP weights are then restored **from the same safetensors file**, stripping the `mtp.` prefix back off — the loader never reads a separate file. Finally scheduler and `_opt_steps` come from meta, and the *resumed* step is returned to the caller so the loop counter can jump.

### Auto-resume and latest_step

```python
latest = self._find_latest_checkpoint()
if latest is not None:
    try:
        global_step = self.load_checkpoint(latest)
    except Exception as exc:
        self._log(f"[warn] Could not load checkpoint: {exc}")
```

`train()` auto-resumes from the newest complete checkpoint at startup; a failed load only warns and starts from 0 (so a corrupt checkpoint does not brick the run — but it *does* silently abandon the saved state, which is exactly when you want the checkpoint-ops playbook, G5). `_find_latest_checkpoint` delegates to `CheckpointManager.latest_step` (`utils/checkpoint.py:CheckpointManager.latest_step`), which sorts the steps parsed from `model_step_*.safetensors` filenames descending and returns the first that `_checkpoint_complete` — all three of `model_step_N.safetensors`, `optim_step_N.pt`, `meta_step_N.json` present. Completeness, not recency, is the criterion: a crash between the three atomic writes leaves a partial step that is invisible to resume.

**Restored:** model weights, AdamW state, scheduler, `_opt_steps`, MTP weights (when `has_mtp`). **Not restored:** DataLoader shuffle order (see [DataLoader Design](#dataloader-design)) and the per-epoch generator state. Also note the manual path: `python training/pretrain.py --resume 40000` calls `trainer.load_checkpoint(int(args.resume))` explicitly before `train()`, which then finds the same checkpoint and resumes identically — the flag exists for when you *don't* want the latest step.
---

## torch.compile — What It Does Here

When `compile_model=True`:

```python
training_model = torch.compile(training_model, mode="max-autotune", fullgraph=False)
```

**TorchInductor** traces the forward graph, fuses elementwise ops, and autotunes CUDA kernels. `max-autotune` spends more compile time benchmarking kernel variants.

**What actually changes:** the eager Python-level execution of the trunk is replaced by a lowered, fused Triton-kernel program. Elementwise chains (RMSNorm, SiLU, residual adds) collapse into single kernels; matmul scheduling is tuned per shape; the per-layer Python loop overhead disappears. The *math* is identical — that is the contract `torch.compile` promises, and the tests that compare compiled vs eager agree on outputs.

**Trade-offs:**
| Benefit | Cost |
|---|---|
| 10-20% faster step time | 5-15 min compile on first run |
| Fused RMSNorm+matmul patterns | Opaque stack traces on error |
| Less Python overhead in MoE loop | Higher peak VRAM during compile |

**`fullgraph=False`:** Allows graph breaks (Python MoE dispatch, dynamic shapes) — required for this codebase. Each graph break is a boundary where compiled regions hand off to eager execution; the MoE `stacked` dispatch loop (per-expert Python forward) is the main break. `fullgraph=True` would refuse to compile this model at all.

**Env:** `TORCH_COMPILE_MODE=max-autotune` set in `launch_a100.sh` — the code reads `os.environ.get("TORCH_COMPILE_MODE", "max-autotune")`, so the env var overrides the default at runtime without touching YAML.

**1650 config:** `compile: false` — 4GB VRAM cannot absorb compile workspace.

**Saved weights are always read from `raw_model`/`mtp_wrapper._orig_mod`** — the compiled wrapper's `state_dict` may carry `_orig_mod.` prefixes and is never checkpointed. See [compile_model Interaction](#compile_model-interaction).

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

### What the estimator counts

`utils/memory.py:estimate_model_memory_gb` sums five buckets ([derived] from the module docstring and PaLM Appendix A; no GPU run has executed, so treat every figure as an estimate):

| Bucket | Formula | 422M value |
|---|---|---|
| Weights (working set) | 2 bytes × N | 411.6M × 2 B ≈ 0.8 GB |
| Optimizer (AdamW FP32) | 12 bytes × N (fp32 master + m + v) | 411.6M × 12 B ≈ 4.9 GB |
| Activations (grad ckpt) | 24 × B·S·D·L × 2 B | 24 × 8·2048·768·18 × 2 B ≈ 10.1 GB |
| Activations (no ckpt) | 36 × B·S·D·L × 2 B | ≈ 15.2 GB |
| Logits (transient, not in estimator) | B·S·V × 2 B | 8·2048·100018 × 2 B ≈ 3.1 GB |
| Static PyTorch/CUDA overhead | `_detect_overhead_gb()` | 13.7 GB (A100 80GB cap) or 17% of device |

The 24/36 constants mirror the PaLM formula (PaLM Appendix A): activations dominate the training footprint, which is exactly why `grad_checkpoint: true` is on by default — it trades ~1/3 of activation memory for recompute compute. Note the estimator does **not** include the transient logits tensor (B·S·V is 3+ GB at 100,018 vocab — the single largest activation, which is why `head` is tied and why the loss is computed *before* any logits-shaped buffer is retained: `logits.reshape(-1, V)` reuses the same storage). Summing with overhead lands in the ~30–35 GB range for training — comfortably inside 80 GB, which is what makes the 1650's 4 GB the real constraint and the "~35 GB" planning figure in Part B. Training never allocates a KV cache (`use_cache=False`), so the KV-cache bucket in the estimator exists for inference only.

---

## Chinchilla Epoch Analysis

Unique corpus: ~8.4B tokens. Samples per epoch:

$$
N_{\text{samples}} = \frac{N_{\text{tokens}} - 1}{\text{max_seq_len}} \approx \frac{8.4 \times 10^9}{2048} \approx 4.1 \times 10^6
$$

(More precisely: 8,399,999,999 usable tokens → 4,101,562 samples.)

**Micro-batches per epoch:** 4,101,562 / 8 = 512,695. **Optimizer steps per epoch:** 512,695 / 4 = 128,174.

**The canonical run is one epoch, not four.** `max_steps = 512,000` is a *micro-step* budget, so the run consumes 512,000 micro-batches → 128,000 optimizer steps → 8.39B tokens. That is ≈ 0.999 epochs: it ends 695 micro-batches (0.14%) short of completing the first pass, having exposed the corpus essentially once. That is exactly the Chinchilla-optimal budget — 20 × 411.6M ≈ 8.2B unique tokens, and the corpus is 8.4B. (Historical note: earlier versions of this doc claimed "~4 epochs / 33.5B exposures" by treating `total_steps` as optimizer steps; the loop counts micro-steps — see [TrainingConfig Field by Field](#trainingconfig-field-by-field) — so the correct figure is one pass, not four.)

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
| `--data-path` | Override `data.train_data_path` |
| `--checkpoint-dir` | Override `training.save_dir` |
| `--resume N` | Load step N before training |
| `--no-compile` | Disable torch.compile |
| `--no-checkpoint` | Disable gradient checkpointing |

### main() wiring

`training/pretrain.py:main` parses the YAML, then builds `TrainingConfig` with precedence **CLI > YAML > dataclass default**. The pattern is `args.data_path or d.get("train_data_path", "data/pretrain_data.bin")` — a CLI flag wins when given; otherwise the YAML section is consulted; otherwise the dataclass default stands. `--no-compile` / `--no-checkpoint` combine with YAML via `and not`: the feature is on only if YAML says on *and* the CLI did not switch it off. `--resume N` runs *after* construction — `trainer.load_checkpoint(int(args.resume))` — so it overrides the auto-resume that `train()` would otherwise perform at startup. Model architecture keys have no CLI override — edit YAML or use test fixtures.

---

## Pretrainer.__init__ — Line-by-Line Walkthrough

`Pretrainer.__init__` (`training/pretrain.py:Pretrainer.__init__`) is the single place where hardware, model, optimiser, and checkpoint subsystems are wired. Reading it top-to-bottom is the fastest way to understand the training stack. (The expanded version of each row lives in [Pretrainer Lifecycle](#pretrainer-lifecycle).)

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

**Atomic write pattern:** write temp file in same directory → `os.replace` — crash mid-write never leaves a torn checkpoint. Concretely, `CheckpointManager._atomic_write` (`utils/checkpoint.py:CheckpointManager._atomic_write`) is a context manager that `tempfile.mkstemp`s in the save dir, yields the temp path, and `os.replace`s it onto the final name on success (unlinking the temp on failure). `os.replace` is atomic on POSIX — a reader either sees the old file or the new one, never a partial write. The safetensors writer additionally dedups shared tensors by `data_ptr()` and calls `.contiguous()` before `save_file`.

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

## Worked Example — One Optimiser Step at the 422m Config

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

---

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

**Q: Why 512K steps?** `total_steps` counts micro-batches: 512,000 × 16,384 tokens = 8.39B ≈ one Chinchilla-optimal pass over the 8.4B corpus (128,000 optimizer steps at grad_accum 4). For multi-epoch training, raise `total_steps` and scale `warmup_steps`/`save_interval` with it.

**Q: Why no GradScaler?** BF16 has sufficient dynamic range on Ampere/Blackwell for this model scale.

**Q: Does compile work on CPU?** Disabled automatically when CUDA unavailable; smoke tests run eager.

**Q: Resume exact data order?** No — DataLoader shuffle RNG is not checkpointed. Benign at this scale.

**Q: Why does the first optimizer step do nothing?** `scheduler.step()` fires after `optimizer.step()`, so optimizer step 1 runs at λ(0) = 0 (warmup starts at zero). Benign — see [LR Scheduler](#lr-scheduler).

**Q: Why is the log's first line `lr=0.00e+00`?** `get_last_lr()` reports λ(0) before any scheduler step. Expected.

**Q: Does `--resume 40000` differ from auto-resume?** It forces step 40000 even when a newer checkpoint exists; auto-resume always takes `latest_step()`.

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
| `global_step` | Micro-step counter in `train()`; the loop bound |
| `is_opt_step` | `(micro_step+1) % grad_accum == 0` — window boundary |
| `λ(s)` | Scheduler multiplier at optimizer step `s` |
| `model_config` | The full YAML dict stored in `TrainingConfig` |

---

## Load-Bearing Invariants

1. **`use_cache=False` in train_step** — training never writes KV cache.
2. **μP counts post-MTP total** — slight LR inflation vs bare Transformer.
3. **YAML `bias_update_every` wins over dataclass default (10)**.
4. **uint32→int64 cast at train boundary** — not in dataset.
5. **NaN guard never disabled by default** in 422M YAML.
6. **`raw_model` for checkpoint/bias** — not the compiled wrapper.
7. **`max_steps` is a micro-step budget** — the loop runs `max_steps` micro-batches = `max_steps / grad_accum` optimizer steps; the scheduler horizon is auto-scaled to optimizer steps (`max_steps // grad_accum`), so the cosine arc always spans the run.
8. **The YAML `lr` key is inert under `mup_lr: true`** — the μP formula, not YAML, sets the LR.

---

## Implementation Checklist

- [ ] `pytest tests/test_training.py` passes
- [ ] `test_mup_lr_scaling` — μP formula (`tests/test_training.py::TestPretrainerConstruction`)
- [ ] `test_consecutive_nan_triggers_rollback` — recovery path
- [ ] `TestCheckpointRoundtrip.test_save_load` — save/load integrity
- [ ] `test_moe_bias_update_during_training` — bias moves
- [ ] `TestWarmupCosineScheduler` — schedule continuity at seams

---

- [training/pretrain.py](../training/pretrain.py)
- [MoE](concepts/moe-mtp.md) — bias mechanism
- [MTP](concepts/moe-mtp.md) — auxiliary loss
- [configs](training.md) — checkpoint format
- [training](training.md) — YAML reference

## train() Main Loop — Pseudocode

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

## compile_model Interaction

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

**How the line is produced:** `utils/logging.py:TrainingLogger.log` appends each micro-step's loss to a rolling window; when `step % log_interval == 0` it prints the window average, `ppl = exp(avg_loss)`, `lr = scheduler.get_last_lr()[0]` (passed in by `train()`), and `tokens_per_sec = (log_interval × seq_len × batch_size) / elapsed` — the elapsed wall time of the *last* `log_interval` micro-steps, so `tps` is a rolling throughput meter, not a device counter. The window is then cleared and the clock restarts. This is the only place per-step `.item()` calls happen.

### WandB integration

Set before launch:

```bash
export WANDB_PROJECT=deepseek-v3-lite-a100
export WANDB_RUN_NAME=422m-run1
```

`TrainingLogger.__init__` checks `WANDB_PROJECT`; if set *and* `wandb` is importable, it initialises a run (name from `WANDB_RUN_NAME`) and every log line additionally pushes `train/loss`, `train/ppl`, `train/lr`, `train/tokens_per_sec`, plus any extra metrics (`train/mtp_loss`, `train/balance_loss`). Missing `wandb` degrades to a one-line console notice — logging never crashes training.

---

## Resume and Fault Tolerance

**Auto-resume:** `train()` calls `latest_step()` on startup if checkpoints exist.

**Manual resume:**

```bash
python training/pretrain.py --config configs/pretrain_a100_422m.yaml --resume 40000
```

**Not restored:** DataLoader shuffle order (benign at 8.4B scale).

**Restored:** Model weights, AdamW state, scheduler, `opt_steps`, MTP weights if `has_mtp: true`.

---

## Part B — Configuration Reference

> Absorbed from the former configs encyclopedia. Single canonical YAML reference for this project.

> **Purpose:** Textbook-style reference for every YAML key in `configs/`, with theory for *why* each hyperparameter exists and *where* code consumes it.

> **Read this if** you're tuning hyperparameters or adding config keys. **Skip if** you're learning architecture theory → [Architecture](concepts/foundations.md).
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

**Target:** 1× A100 80GB, ~412M params (411.6M deduped — the "422m" filename is historical nominal), 8.4B Chinchilla tokens, ~30–45 h wall (estimated; the old 13–15 h target implied >70% MFU).

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
| `attn_impl` | sdpa | MLA | sdpa / manual / triton |
| `moe_dispatch` | stacked | DeepSeekMoE | stacked / triton_grouped |
| `weight_tying` | true | Transformer | Share embed/head |

### Dimensional constraints (422M)

These keys are not independent — violating a constraint crashes at first forward or silently mis-trains:

| Constraint | Rule | Example violation |
|---|---|---|
| Head divisibility | $d \bmod H = 0$ | dim=770, n_heads=12 |
| MLA head dims | `v_head_dim = dim / n_heads`; `qk_head_dim = qk_nope + qk_rope` | v_head_dim=64 = 768/12; qk_head_dim=72 = 48+24 — the two are independent |
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

Derive the token budget from YAML:

$$
N_{\text{tokens/micro-step}} = B_{\text{micro}} \times S = 8 \times 2048 = 16384
$$

$$
N_{\text{tokens/opt-step}} = B_{\text{micro}} \times G_{\text{accum}} \times S = 8 \times 4 \times 2048 = 65536
$$

$$
N_{\text{total tokens}} = 16384 \times T_{\text{steps}} = 16384 \times 512000 \approx 8.4 \times 10^9
$$

`total_steps` counts **micro-steps**, so the run is one pass over the corpus (8.39B tokens ≈ 8.4B), and the optimizer performs $512000 / 4 = 128000$ steps. Unique corpus size ≈ 8.4B → **≈ 1 epoch**. Chinchilla-optimal **unique** tokens ≈ $20 \times 411.6\text{M} \approx 8.2$B — the corpus matches the target.

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

**Intuition:** Wider models have more parameters contributing to each update; sqrt scaling keeps effective update magnitude stable. See [Foundations](concepts/foundations.md) §13.

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

`Transformer`: `config.get("model", config)` `Pretrainer`: passes full YAML as `model_config`; sub-components unwrap as needed.

---

## Appendix — Quick comparison table

| | 422M A100 | 1650 2M |
|---|---|---|
| Params | ~412M (411.6M deduped) | ~2M |
| VRAM | ~35 GB (estimate) | <4 GB |
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

Trace in: `training/pretrain.py:Pretrainer.train_step`, `models/mtp.py:MultiTokenPrediction.forward`.

---

---

## Hyperparameter Rationale (422M)

### Why `dim=768`, `n_layers=18`?

Chinchilla-optimal sizing: for ~412M params, depth and width balance FLOPs and memory. 18 layers fits in A100 VRAM with MoE at batch 8 × seq 2048.

### Why `n_dense_layers=2`?

DeepSeek-V3 uses early dense layers for stable low-level features before sparse routing. Matches paper schedule.

### Why `n_routed_experts=20`, `n_activated_experts=4`?

20 experts × 384 dim stores capacity; top-4 activates ~20% of routed compute. +1 shared expert always runs (5 FFN evaluations per MoE token).

### Why `lr=8e-4` (μP-adjusted)?

Base YAML value; overwritten at init to ~8.07e-4 when `mup_lr: true`. Tuned for stable BF16 training with grad clip 1.0.

### Why `warmup_steps=2000`?

≈1.6% of the run's 128,000 optimizer steps (warmup counts scheduler steps, one per optimizer step). Sufficient for Adam moment estimates to stabilise without wasting compute.

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

**Q: Can I train 422M config on 24GB GPU?** A: No without major changes — halve batch and seq len, possibly disable MTP/compile.

**Q: Why two configs?** A: 422M = production research. 1650 2M = structural smoke test on 4GB.

**Q: Where is `seed` in YAML?** A: Not in 422M YAML — defaults to 42 in `TrainingConfig`. Add `seed: N` under `training:` to override.

- `configs/pretrain_a100_422m.yaml`
- `configs/pretrain_1650_2m.yaml`
- `tests/conftest.py` — `cfg`, `small_cfg` fixtures
- [training](training.md) — how YAML maps to `TrainingConfig`

## Full YAML Walkthrough — 422M Config

Annotated excerpt from `configs/pretrain_a100_422m.yaml`:

```yaml
model:
  vocab_size: 100018      # MUST match DeepSeek tokenizer len()
  dim: 768                # Hidden width d; divisibility: d % n_heads == 0
  n_layers: 18            # 2 dense + 16 MoE
  n_heads: 12             # MLA heads; qk_head_dim = 48 nope + 24 rope = 72, v_head_dim = 64 (see [MLA](concepts/attention-and-precision.md))
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

## Check Your Understanding

1. **Counter semantics.** `global_step` counts micro-steps and `max_steps` is the micro-step budget. If `gradient_accumulation_steps = 4` and `max_steps = 512000`, how many micro-batches does the run consume, and what is `_opt_steps` when the loop exits? *(Answer: 512,000 micro-batches; `_opt_steps` reaches 128,000 — one optimizer step per 4 micro-steps, and the final window ends exactly at the loop bound.)*

2. **μP arithmetic.** Reproduce the LR computation for the bare 411.6M model from the locked constants. *(Answer: 6.0e-4 × sqrt(757226496/411632256) = 8.138e-4.)* What single edit makes the YAML `lr` key actually take effect? *(Answer: set `mup_lr: false` — under `mup_lr: true` the LR is the formula's output.)*

3. **The first optimizer step.** Why is optimizer step #1 a no-op, and is that a bug? *(Answer: `scheduler.step()` runs after `optimizer.step()`, so step #1 executes at λ(0) = 0 — every term in the AdamW update is scaled by η = 0. Benign: warmup is zero at step 0 by construction, so the schedule is just shifted by one step.)*

4. **NaN recovery.** A NaN fires at micro-step 2 of a 4-step window with a checkpoint at step 4000 and current `global_step = 4007`. What happens to the gradients of micro-steps 0–1 of the current window, and what does `global_step` become after rollback? *(Answer: the partial window's accumulated gradients are discarded (`zero_grad(set_to_none=True)`), and rollback sets `global_step` to 4000 — the DataLoader position is not rewound, so the lost steps' data is not replayed.)*

---

## Data Pipeline — Tokenization, Sharding, and Loading

> **Read this if** you're preparing or validating training data. **Skip if** shards already exist → [Training](training.md).

**Depends on:** [Training](training.md) · **Read next:** [Inference](inference.md)

---

## Table of Contents

1. [60-Second Summary](#60-second-summary)
2. [Why a Shared Pipeline?](#why-a-shared-pipeline)
3. [The DeepSeek Shim](#the-deepseek-shim)
4. [Tokenizer Deep Dive](#tokenizer-deep-dive)
5. [Pipeline Stages](#pipeline-stages)
6. [Data Mixture](#data-mixture)
7. [Shard Format](#shard-format)
8. [PretrainDataset Consumption](#pretraindataset-consumption)
9. [CLI Reference](#cli-reference)
10. [Environment Variables](#environment-variables)
11. [Operational Runbook](#operational-runbook)
12. [Pipeline Failure Modes](#pipeline-failure-modes)
13. [Learning Exercises](#learning-exercises)
14. [Check Your Understanding](#check-your-understanding)
15. [Appendix A — FAQ](#appendix-a--faq)
16. [Appendix B — Glossary](#appendix-b--glossary)
17. [References](#references)

---

## 60-Second Summary

DeepSeek-V3-Lite does **not** vendor its own data pipeline. `data/prepare_data.py` is a thin shim that configures the **universal 8.0B-token LLM pipeline** in `LLM/shared_data/` with DeepSeek-specific tokenizer settings (`vocab_size=100,018`, EOS=100,017, PAD=100,016) and writes a project-local `data/data_config.yaml`. The pipeline then runs four crash-isolated stages — **download → clean → tokenize → pack** — and the output is a directory of memory-mapped `uint32` token shards (`shard_*.bin`, 50M tokens each, ~200 MB) plus a `manifest.json` recording provenance, checksums, and per-source token counts.

Training consumes those shards directly through `PretrainDataset` in `training/pretrain.py`: it memory-maps every shard with `torch.load(..., mmap=True)`, computes sample windows of `max_seq_len + 1` contiguous tokens, and serves next-token pairs `(chunk[:-1], chunk[1:])`. A binary search (`PretrainDataset._locate`) maps any global token index to `(shard_idx, offset)` in $O(\log K)$ for $K$ shards, which is what lets a single window stitch tokens across a shard boundary. The whole 8.0B-token corpus sits on disk as ~32 GB of uint32 tokens; it is never fully resident in RAM.

Everything in this chapter is verified against the current sources (`data/prepare_data.py`, `training/pretrain.py`, `LLM/shared_data/`). No full-scale run has ever executed (no GPU training run, no completed data-prep run) — sizes, shard counts, and wall-clock figures are **estimates** or arithmetic, labeled as such.

---

## Why a Shared Pipeline?

The sibling projects under `LLM/` (per `LLM/shared_data/README.md`: Mamba-2-Lite, GPT-OSS-Lite, HyMo, DeepSeek-V3-Lite, LLaMA-3-Lite) all target a ~400–500M-param model trained on roughly the same corpus budget. Without a shared pipeline, each project would independently:

- Re-download FineWeb-Edu (~1.3 TB at full scale) and re-run cleaning — weeks of duplicated bandwidth and compute.
- Drift apart on the *meaning* of a source: "fineweb-edu" with different configs, different filters, different weights makes cross-project loss comparisons meaningless.
- Invent its own shard format (`torch.save` vs raw uint32 vs JSONL), so training recipes could not be ported between projects without rewriting the dataset.

The shared pipeline fixes all three with one decision: **the text corpus, cleaning, dedup, shard format, and manifest schema are shared; only the tokenizer differs per project** (each project has a different vocab — GPT-2 BPE, LLaMA-3 BPE, a custom 64K BPE, or DeepSeek-Coder). This gives:

- **One place to fix bugs** in download/clean/dedup, benefiting every project.
- **Fair cross-project comparisons** — same text, different token IDs.
- **Bit-identical shards except for token values** — the EOS positions, dtype, shard boundaries, and per-source mix are identical across projects using the same corpus; projects sharing a tokenizer produce byte-identical shards.

The cost of sharing is the shim: each project's `data/prepare_data.py` is a ~60-line wrapper that injects its tokenizer settings and delegates to `shared_data.prepare_data.run_pipeline(...)`.

---

## The DeepSeek Shim

`data/prepare_data.py` does three things:

1. **Path injection** — adds the project root and the `CoreProjects/` root to `sys.path` so both `shared_data` and the project's own packages are importable.
2. **Config materialisation** — writes `data/data_config.yaml` with DeepSeek overrides by copying the universal config and patching the tokenizer block (`data/prepare_data.py:_ensure_deepseek_data_config`):

```python
DEEPSEEK_TOKENIZER_NAME = "deepseek-coder-v2-lite"
DEEPSEEK_VOCAB_SIZE = 100_018
DEEPSEEK_EOS_TOKEN_ID = 100_017
DEEPSEEK_PAD_TOKEN_ID = 100_016
```

```python
cfg = load_yaml(UNIVERSAL_DATA_CONFIG_PATH)
cfg["pipeline"]["tokenizer"]["name"] = DEEPSEEK_TOKENIZER_NAME
cfg["pipeline"]["tokenizer"]["vocab_size"] = DEEPSEEK_VOCAB_SIZE
cfg["pipeline"]["tokenizer"]["eos_token_id"] = DEEPSEEK_EOS_TOKEN_ID
cfg["pipeline"]["tokenizer"]["pad_token_id"] = DEEPSEEK_PAD_TOKEN_ID
cfg["_generator"] = "DeepSeek-v3-Lite/data/prepare_data.py"
cfg["_tokenizer_family"] = "deepseek-coder-v2"
```

3. **Delegation** — `data/prepare_data.py:main` parses `--stage`, `--mixture`, `--data-config`, `--data-root`, `--source`, and the four `--skip-*` flags, then calls `shared_data.prepare_data.run_pipeline(...)` with the DeepSeek data config. The shim also prints the corpus budget it expects (`data/prepare_data.py:_apply_deepseek_defaults` reads `UNIVERSAL_TOTAL_TOKENS = 8_000_000_000` from `shared_data/config.py`):

```bash
python3 data/prepare_data.py --stage pretrain
```

The resulting `data/data_config.yaml` is small and fully self-describing (it records `_generator` and `_tokenizer_family`). The sharding block stays at the universal defaults — 50,000,000 tokens per shard, `uint32`, target 8.0B tokens — while the DeepSeek overrides only touch the tokenizer block.

```yaml
pipeline:
  tokenizer:
    name: deepseek-coder-v2-lite
    path: null
    vocab_size: 100018
    eos_token_id: 100017
    pad_token_id: 100016
    add_eos: true
  sharding:
    shard_size_tokens: 50000000
    dtype: uint32
    target_total_tokens: 8000000000
  # dedup / quality / tokenize / pack / seed — universal defaults
```

**Cross-check at startup:** `run_pipeline` reads both `mixture.yaml` and `data_config.yaml` and compares `mixture.total_tokens` against `pipeline.sharding.target_total_tokens`. If they disagree it logs a warning and uses the **mixture's** value — the mixture is authoritative. Keeping the two in sync is a deliberate, cheap guard against silently building a corpus of the wrong size.

---

## Tokenizer Deep Dive

| Property | Value | Why it matters |
|---|---|---|
| Name | `deepseek-ai/deepseek-coder-v2-lite` | Public HF tokenizer (DeepSeek-Coder-V2-Lite family) |
| `vocab_size` | **100,018** | Must match `model.vocab_size` and the embedding rows exactly |
| `eos_token_id` | 100,017 | Appended at every document boundary (the packer relies on it) |
| `pad_token_id` | 100,016 | Padding (rarely used in packed pretraining) |
| `byte_fallback` | yes | BPE falls back to raw byte tokens instead of `<unk>` — handles arbitrary bytes, no OOV ever |

### Why 100,018 and not a round number?

The tokenizer is the *same one DeepSeek shipped for DeepSeek-Coder-V2-Lite*, a byte-level BPE whose trained merges plus special tokens total 100,018 ids. The number is not a hyperparameter you choose; it is a property of the tokenizer. Two consequences:

- The embedding matrix in `models/transformer.py:Transformer.__init__` is `nn.Embedding(vocab_size, dim)` with `vocab_size` read from the config — the 422M config (`configs/pretrain_a100_422m.yaml`) sets `vocab_size: 100018`. If the two ever disagree, every embedding lookup for ids $\geq$ the config's vocab silently misaligns or crashes with an index error.
- The 1650 smoke config (`configs/pretrain_1650_2m.yaml`) deliberately uses `vocab_size: 50257` with the GPT-2 tokenizer — the architecture features are identical but the tokenizer needs no HF authentication, which is what makes CI runs self-contained.

### byte_fallback: the mechanism

`byte_fallback: yes` is the HF tokenizers setting that changes what happens when the BPE model has **no merge path** for the bytes it is scanning. A tokenizer without it emits the `<unk>` token (id 0 in many vocabularies), which destroys information — any unseen byte sequence, any rare emoji, any exotic script, collapses to one id. With byte_fallback enabled, the encoder instead emits one token per raw byte from a reserved range of byte tokens, so:

- **Encoding is total**: every possible UTF-8 string maps to a sequence of in-vocab ids. There is no out-of-vocabulary input, ever.
- **Round-tripping works**: decoding a byte-fallback sequence reproduces the original bytes exactly.
- **The model can still learn the byte-level grammar** of rare content instead of being blind to it.

This is unusual compared to GPT-2/LLaMA-style tokenizers (byte-level BPE where bytes *are* the bottom of the merge ladder) and is one reason the DeepSeek tokenizer is worth keeping: the model never sees `<unk>`, so no training signal is silently discarded. The practical upshot: the embedding table must have exactly 100,018 rows, and token IDs can legitimately be any value in `[0, 100018)` — do not assume they are all printable Unicode.

### Special tokens

EOS (100,017) is the workhorse of the pipeline: every document gets exactly one trailing EOS (the tokenizer config sets `add_eos: true`, and the pack stage's writer also appends EOS explicitly, de-duplicating a document that already ends in EOS). The `PretrainDataset` consumer never needs to *find* document boundaries — the model simply learns to predict EOS, and the next window's first token is the first token of the next document. PAD (100,016) is defined but essentially unused in packed pretraining: windows are never padded, they are truncated/dropped (see [Sample Count Formula](#sample-count-formula)).

### Verification

The tokenizer itself is external to the repo, but the contract is checkable:

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-v2-lite")
assert len(tok) == 100018
assert tok.eos_token_id == 100017
assert tok.pad_token_id == 100016
```

Changing the tokenizer without regenerating shards **and** updating `vocab_size` in the model config breaks training at the embedding lookup on step 0 — see [Pipeline Failure Modes](#pipeline-failure-modes).

---

## Pipeline Stages

`shared_data.prepare_data.run_pipeline(...)` orchestrates four subprocess stages. Each stage runs as `python -m shared_data.scripts.<stage>` via `subprocess`, which is what makes them **crash-isolated**: a failure (or OOM, or disk-full) in tokenize leaves the clean JSONL untouched, and a partial stage never corrupts the output of a later one. The orchestrator checks the mixture/data-config files exist, cross-checks the token totals, seeds all RNGs (`seed: 42`), and stops at the first failing stage with a nonzero exit code.

```
   ┌──────────────┐
   │ HuggingFace  │  all 7 mixture sources streamed via load_dataset
   └──────┬───────┘
          │ stage 1: shared_data.scripts.download_raw
          ▼
   data/raw/<source>/data.jsonl          raw text, one JSON object per line
          │ stage 2: shared_data.scripts.clean
          ▼
   data/clean/<source>/data.jsonl        quality-filtered + dedup'd text
          │ stage 3: shared_data.scripts.tokenize
          ▼
   data/tokens/<source>/data.bin         per-source uint32 token streams
          │ stage 4: shared_data.scripts.pack_shards
          ▼
   data/shards/shard_00000.bin …         packed 50M-token shards
   data/shards/manifest.json             provenance + checksums + stats
```

All paths resolve under a **data root**: `$LLM_DATA_ROOT` if set, else `<cwd>/data` (see [Environment Variables](#environment-variables)). Every stage is **resumable** — see [Resumption Semantics](#resumption-semantics).

### Stage 1 — Download

`shared_data/scripts/download_raw.py` streams each source from HuggingFace with `load_dataset(..., streaming=True)` and appends JSON objects `{"text": ...}` to `data/raw/<source_id>/data.jsonl`.

Two details matter:

- **Budget truncation happens here first.** Each source has a target of `int(total_tokens * weight)` tokens; the downloader converts that to characters with a fixed estimate `CHARS_PER_TOKEN = 4.0` and stops when `n_chars >= target_chars`. Downloading is thus *truncated, not sampled*: the first documents of the HF split fill the source's budget. This is the first of three places the 8.0B budget is enforced (download → tokenize → pack).
- **Multi-field documents.** Some sources combine two columns. OpenMath concatenates `problem` and `generated_solution` with a separator so the model sees full worked derivations (`_build_text` in `download_raw.py`):

```python
extra_field = spec.get("extra_text_field")
if extra_field:
    extra = row.get(extra_field, "") or ""
    sep = spec.get("extra_separator", "\n\n")
    if extra:
        text = f"{text}{sep}{extra}" if text else extra
```

Progress (`n_processed`, `n_chars`) is flushed to the per-source download state file every 50,000 docs, which is what makes resumption cheap.

### Stage 2 — Clean

`shared_data/scripts/clean.py` runs two passes over each source's raw JSONL: a **quality filter** and an **exact-match dedup**.

The quality filter (`shared_data/quality_filter.py`) is a chain of cheap heuristics, each returning keep/drop:

| Filter | Rule (defaults from `data_config.yaml` → `pipeline.quality`) |
|---|---|
| Length | keep `min_chars <= len(text) <= max_chars` — per-source bounds from `mixture.yaml` (e.g. arxiv keeps 500–500,000 chars, capping one-doc-per-shard pathologies) |
| Empty | drop empty strings |
| Unique chars | drop if `len(set(text)) < 0.05 * len(text)` |
| Digit ratio | drop if digits > 50% of chars — **disabled for `lang: python`** sources, where digits are legitimate |
| Punctuation ratio | drop if punctuation > 50% |
| Whitespace ratio | drop if whitespace > 50% |
| Language hint | cheap ASCII-letter-ratio heuristic (`lang: en` / `lang: python` / `null`) — not a real language ID |

`QualityFilter.apply(text)` returns the (possibly whitespace-normalised) kept text or `None`, and `FilterStats` accumulates per-reason drop counts that end up in the manifest.

Dedup (`shared_data/dedup.py` — `DedupFilter`) is **per-source, exact SHA-256**: a document is dropped iff its whitespace-normalised SHA-256 hash has been seen before in *that source* (`sha256_text` normalises with `" ".join(text.split())` before hashing, so near-identical formatting still collides). Note this is *not* a cross-source union — each source keeps its own seen-set, so the same text may survive in two sources. The `minhash` near-duplicate strategy and the bloom-filter knobs (`n_hash_buckets`, `bloom_capacity_per_bucket`, `bloom_error_rate`) are documented as reserved in `data_config.yaml`; the implemented `sha256` path uses a plain in-memory set. Cleaned documents are written to `data/clean/<source>/data.jsonl` with a 16-hex-char content hash prefix (`{"id": sha16, "text": ...}`) so downstream stages can reference provenance.

### Stage 3 — Tokenize

`shared_data/scripts/tokenize.py` encodes each clean document to token ids with the project tokenizer (for us: deepseek-coder-v2-lite) and writes a **per-source token stream** — a binary file with a tiny header followed by raw uint32 token ids, EOS-separated (see [Byte-Level Shard Spec](#byte-level-shard-spec) for the exact header).

The producer/consumer design keeps the GPU-less stage CPU-bound in the right way:

```python
# Producer thread: tokenise ahead of the writer.
q: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=prefetch)
stop = threading.Event()

def producer() -> None:
    for text in doc_iter:
        if not text:
            continue
        ids = tokenizer.encode(text)
        if not ids:
            continue
        validate_tokens(np.asarray(ids, dtype="uint32"), ...)
        q.put((ids, len(ids) + 1))
    q.put(None)

t = threading.Thread(target=producer, daemon=True)
t.start()
```

The main thread pops `(ids, n)` and calls `stream.write_doc(ids)`, which appends the ids **plus one EOS** to the stream and advances the counters. `validate_tokens` enforces the hard upper bound `token_id < vocab_size + 256` — if the tokenizer ever emitted something wildly out of range, the stage refuses to write a corrupt stream rather than silently proceeding. The per-source token budget (`total_tokens * weight`) is enforced *here* as the exact truncation point: `if n_tokens >= target_tokens: stop.set(); break`. `add_special_tokens: false` means the only special token in the stream is the manually appended EOS.

### Stage 4 — Pack

`shared_data/scripts/pack_shards.py` interleaves the per-source token streams **round-robin at the document level** (`interleave_sources` yields `(source_id, doc)` cycling through sources) and feeds them to `ShardWriter`, which packs 50M tokens per shard. Because each source's stream was already truncated to its budget at tokenize time, the round-robin preserves the global mixture without any re-blending at pack time. (The per-source `remaining` budgets threaded through `interleave_sources` are not enforced there — truncation happened upstream in stage 3.)

`ShardWriter` gives three guarantees:

1. **Document atomicity:** a document (with its EOS) is never split across shards — `cross_document_boundary_ok: false` in the config, and `ShardWriter.add` raises if a single document + EOS exceeds `shard_size_tokens` (50M), which the per-source `max_chars` caps make impossible in practice.
2. **Shard atomicity:** a shard is written to a `.tmp` sibling, flushed, `fsync`'d, then moved into place with `os.replace`:

```python
payload = self._buf[: self._buf_pos]
raw_bytes = payload.tobytes()
with open(tmp_path, "wb") as f:
    f.write(raw_bytes)
    f.flush()
    os.fsync(f.fileno())
os.replace(tmp_path, shard_path)
```

A SIGKILL mid-write can leave only a `.tmp` file, never a half-written `shard_*.bin` that a reader would happily mmap.
3. **Verification before manifest:** after packing, `verify_shard` re-reads every shard and checks byte alignment, token count, `max_token < vocab_size + 256`, and the EOS count; the `Manifest` is then built *from the shard files actually on disk* (authoritative), validated (`manifest.validate`), and saved next to the shards as `manifest.json`.

### Resumption Semantics

Every stage is designed to be re-runnable; nothing is "all or nothing".

- **Download:** state `data/state/download_<source>.json` records `n_processed`/`n_chars`. On re-run, rows before `n_processed` are skipped and the JSONL is reopened in append mode — no re-download of the first N docs.
- **Clean:** state `data/state/clean_<source>.json` records processed/kept/dropped counters. On re-run the first `n_processed` input rows are skipped and output is appended (output is already correct; the input is re-read from the start). The dedup seen-set is rebuilt in memory each run — the shared_data design docs describe a persisted seen-set (`dedup_<source>.json`), but the current `DedupFilter` implementation does not persist it [verified 2026-08-04], so a crash mid-clean re-emits rows that were already emitted (they may be re-dropped as duplicates only within the same run).
- **Tokenize:** state `data/state/tokenize_<source>.json` gains a `"complete": true` marker when a source finishes; `_should_retokenize` skips completed sources on re-run (the stream file is reused and its token count recomputed from the file size). A partially-tokenised source resumes from `n_processed` docs.
- **Pack:** `ShardWriter` flushes complete shards atomically and the manifest is rebuilt from disk, so re-running after a crash re-packs from the beginning but every re-written shard is byte-identical. The `resume=True` argument passed by `pack_shards.py` and the `shard_writer_state.json` file described in the shared_data docs are **not implemented** in the current `ShardWriter` — its constructor has no `resume` parameter, and passing one raises `TypeError` [measured 2026-08-04 on the workspace copy]. Treat pack as restartable, not incrementally resumable, in the current code.
- **Stage-level skip:** the `--skip-*` flags bypass whole stages; `--source NAME` limits a run to one source (useful for debugging a single mixture component).

---

## Data Mixture

The canonical mixture lives in `LLM/shared_data/config/mixture.yaml`. **It is the only source of truth** — the table below is quoted from it as of 2026-08-04 and may drift; always re-read the YAML before trusting any copy (including this doc).

### Mixture weights math

The corpus target is `total_tokens: 8_000_000_000` (8.0B). Each source declares a weight $w_i$; the per-source token budget is simply

$$
T_i = w_i \cdot T, \qquad T = \sum_i T_i = 8.0 \times 10^9
$$

| Source (`id`) | HF dataset | Weight $w_i$ | Budget $T_i$ |
|---|---|---|---|
| `fineweb-edu` | `HuggingFaceFW/fineweb-edu` (sample-10BT) | 0.40 | 3.20B |
| `dclm-baseline` | `mlfoundations/dclm-baseline-1.0` | 0.15 | 1.20B |
| `the-stack-v2-python` | `bigcode/the-stack-v2` (Python) | 0.15 | 1.20B |
| `the-stack-v2-jupyter` | `bigcode/the-stack-v2` (JupyterNotebook) | 0.05 | 0.40B |
| `openmath` | `nvidia/OpenMathInstruct-2` | 0.10 | 0.80B |
| `arxiv` | `cdv/arxiv-classification` | 0.10 | 0.80B |
| `cosmopedia` | `HuggingFaceTB/cosmopedia` | 0.05 | 0.40B |
| **Σ** | | **1.00** | **8.00B** |

Grouped: web backbone 0.55, code 0.20, math 0.10, long-form scientific prose 0.10, synthetic instruction prose 0.05.

The weights are tuned to the modern "data diet" consensus: a quality-gated web backbone, a strong code+math block (30% — the single strongest lever for reasoning in small models), and long documents from arxiv for long-context coherence. The YAML documents that weights must sum to 1.0; the *pipeline* only cross-checks the token totals between mixture and data config (a warning), so treat the weight sum as a documented invariant rather than a runtime-enforced one.

Because the budget is enforced by **truncating each source's stream at $T_i$ tokens**, the mixture is exact up to the last partial document: under-represented sources still contribute their full budget (arxiv's 0.10 → 0.80B tokens — plenty for a ~412M model). No sampling or re-blending happens at train time; the packer's round-robin interleave is what makes the local mix (the stream inside one shard) look like the global mix instead of "blocks of one source".

### Token budget: 8.0B canonical vs "~8.4B" nominal

The canonical, pipeline-enforced target is **8.0B tokens** (`mixture.yaml` `total_tokens`, `UNIVERSAL_TOTAL_TOKENS` in `shared_data/config.py`, and the shim-generated `data/data_config.yaml` all agree). The "~8.4B" phrasing that appears in repo scripts and older docs is the *Chinchilla rationale* for that number: roughly 20 tokens per parameter times 418.7M parameters (with MTP) gives

$$
20 \times 4.187 \times 10^8 \approx 8.4 \times 10^9,
$$

so the realized round budget of 8.0B sits right at the compute-optimal recommendation. When you see "8.4B" in this repo, it means "the 8.0B canonical target, justified by ~20 tokens/param" — the YAML value 8,000,000,000 is what the pipeline actually produces.

---

## Shard Format

### Why uint32 on disk, int64 in training

| Stage | dtype | Reason |
|---|---|---|
| Disk shards | uint32 | 4 bytes/token; vocab 100,018 < $2^{32}$ |
| `PretrainDataset` storage | uint32 mmap | Memory-efficient random access |
| Model input | int64 | `nn.Embedding` and `F.cross_entropy` require Long indices |

Keeping tokens as uint32 until the model boundary halves disk and page-cache footprint: 8.0B tokens at 4 B/token = 32 GB on disk versus ~64 GB for int64. The cast to int64 happens once, at the boundary (see [The uint32 → int64 Cast Boundary](#the-uint32--int64-cast-boundary)).

### Byte-Level Shard Spec

The **universal pipeline** (`LLM/shared_data/shard_writer.py`) writes shards as **raw little-endian uint32 arrays — no torch header, no per-sample headers**:

```
shard_00000.bin
  byte 0      token[0]     (uint32 LE)
  byte 4      token[1]
  ...
  byte 4*(n-1) token[n-1]
```

- dtype: `uint32`, platform-native little-endian (the writer uses `arr.tobytes()` on an `np.uint32` array).
- size: 50M tokens/shard → exactly 200 MB (`50e6 × 4 B`).
- content: a flat token stream; documents are separated by single EOS tokens (id 100,017). There are **no per-document headers** — the EOS token *is* the boundary.
- naming: `shard_{index:05d}.bin` (`shard_00000.bin`, `shard_00001.bin`, …), zero-padded so lexicographic order = numeric order.
- the `PretrainDataset` consumer globs `shard_*.bin`, so any zero-padding (e.g. `shard_0000.bin` from `scripts/build_small_pretrain_data.py`) works.

**Two writers, two on-disk formats — keep them straight:**

1. The universal pipeline's `ShardWriter` writes the raw-byte format above; its reader (`ShardDataset` in `shared_data/dataset.py`) opens them with `np.memmap(..., mode="r")`.
2. `scripts/build_small_pretrain_data.py` (the self-contained smoke-test path, no `shared_data` needed) writes a **torch-serialised** 1D `torch.uint32` tensor via `torch.save(tensor, out_path)` — and this repo's `PretrainDataset` loads shards with `torch.load(p, weights_only=True, map_location="cpu", mmap=True)`, which requires the torch-serialised form.

A raw-byte shard from the universal pipeline will **not** load through `PretrainDataset.torch.load(..., mmap=True)` — `torch.load` rejects files not written by `torch.save` [measured 2026-08-04: `RuntimeError: mmap can only be used with files saved with torch.save(...)`]. Since the full-scale corpus has never been produced, this seam is unexercised; converting raw shards (e.g. `torch.save(torch.from_numpy(np.memmap(...)))`) or extending `PretrainDataset` to numpy-memmap raw shards is an open integration item if the universal output is pointed at `data/pretrain_chinchilla/` [INFERENCE — no conversion helper exists in the repo].

The per-source intermediate streams (stage 3) have a slightly richer layout (`TokenStream` in `shared_data/shard_writer.py`): an 8-byte header `<II` = (version=1, eos_token_id) followed by the same raw uint32 body, EOS-separated. `read_token_stream` parses the header, memory-maps the body, finds EOS positions with `np.where(body == eos_id)`, and yields each document as a `np.ndarray` copy.

### manifest.json

The manifest (`Manifest` in `shared_data/manifest.py`) is written next to the shards (default `manifest.json` inside the shards directory) and is the machine-readable record of what the corpus is:

| Field | Meaning |
|---|---|
| `version`, `created_utc` | schema version 1.0.0, creation timestamp |
| `vocab_size`, `eos_token_id`, `pad_token_id`, `tokenizer_name` | the tokenizer contract the shards were encoded with |
| `dtype`, `shard_size_tokens` | storage contract (`uint32`, 50M) |
| `total_tokens`, `shard_count`, `shards_dir` | corpus totals |
| `shards[]` | per shard: `index`, `path`, `n_tokens`, `sha256`, `n_eos` |
| `sources{}` | per source: `target_tokens`, `actual_tokens`, `n_docs`, `n_dedup_dropped`, `shard_count` |
| `config_hash`, `mixture_hash` | SHA-256 of the exact `data_config.yaml` and `mixture.yaml` used — changing either invalidates the corpus fingerprint |

`Manifest.validate` enforces sanity invariants before save: positive vocab/totals, `eos_token_id` within `[0, vocab_size + 256)`, `len(shards) == shard_count`, and $\sum_{\text{sources}} \text{actual\_tokens} \approx \text{total\_tokens}$ (within 0.1%).

### Storage Budget

| Item | Size | Notes |
|---|---|---|
| Raw downloads | ~500 GB+ | cached under the data root (`data/raw/<source>/`) |
| Cleaned text | ~200 GB | JSONL under `<data-root>/clean/` |
| Token streams | ~32 GB | per-source uint32 under `<data-root>/tokens/` |
| Packed shards | **32 GB** | 8.0B × 4 B — `8.0e9 × 4 = 3.2e10` bytes |
| Shard count | **160** | `8.0e9 / 5.0e7 = 160` |

All figures except the arithmetic are estimates — no full-scale run has executed. Raw-text stages dominate disk; the training-relevant artifact (shards) is a tidy 32 GB, mmap-friendly, and shared across sibling projects (only token IDs differ per tokenizer).

---

## PretrainDataset Consumption

The training-side consumer is `PretrainDataset` (`training/pretrain.py:PretrainDataset.__init__`). It is a `torch.utils.data.Dataset` over a **flat token stream** — there are no per-sample records anywhere; samples are *windows* over the packed tokens.

### Layout Detection

`PretrainDataset` accepts either a single `.bin` file or a directory of shards, decided by `os.path.isdir(data_path)`:

```python
self.layout = "sharded" if os.path.isdir(data_path) else "single"
if self.layout == "sharded":
    self.shard_paths = [str(p) for p in sorted(Path(data_path).glob("shard_*.bin"))]
    if not self.shard_paths:
        raise FileNotFoundError(f"No `shard_*.bin` files in {data_path}")
    self.shards = [torch.load(p, weights_only=True, map_location="cpu", mmap=True) for p in self.shard_paths]
    self.shard_sizes = [s.numel() for s in self.shards]
    self.shard_offsets = []
    running = 0
    for s in self.shard_sizes:
        self.shard_offsets.append(running)
        running += s
    self._total_tokens = sum(self.shard_sizes)
else:
    self.data = torch.load(data_path, weights_only=True, map_location="cpu", mmap=True)
    self._total_tokens = self.data.numel()
self._n_samples = (self._total_tokens - 1) // self.max_seq_len
```

Key mechanics:

- **mmap, not load:** `torch.load(..., mmap=True)` maps each shard lazily; the OS demand-pages tokens as they are read, and read-only pages are shared across `num_workers=8` DataLoader workers. The full 32 GB corpus is never resident in RAM.
- **Cumulative offsets:** `shard_offsets[i]` is the global token index where shard `i` begins. With $K$ shards this is a sorted array of length $K$, which is exactly what the bisect in `_locate` needs.
- **Total token count** is the sum of shard sizes — the consumer never reads the manifest; the shard files themselves are authoritative.
- **A missing path** raises `FileNotFoundError("Pre-training data not found: ... Run `python data/prepare_data.py` first.")` — the standard first-run error, also hit when a config points at a directory that only contains a manifest.

### Sample Count Formula

$$
N_{\text{samples}} = \left\lfloor \frac{N_{\text{tokens}} - 1}{S} \right\rfloor, \qquad S = \text{max\_seq\_len}
$$

The `-1` accounts for the extra target token every window needs: window `idx` reads `S + 1` tokens at `start = idx * S`, returning input `chunk[:-1]` (positions 0..S-1) and shifted target `chunk[1:]` (positions 1..S). At position $t$ the model predicts $x_{t+1}$ from $x_{\le t}$ — classic teacher-forced next-token prediction.

The final `(N mod S)`-ish tail that cannot fill a full window is **dropped, not padded**. Padding would inject fake targets and is deliberately avoided; `drop_last=True` on the DataLoader is a *second*, orthogonal drop — it drops partial **batches**, not partial samples.

For the 422M config ($S = 2048$):

$$
N_{\text{samples}} \approx \frac{8.0 \times 10^9}{2048} \approx 3.9 \times 10^6 \text{ samples/epoch}
$$

With `micro_batch_size 8 × 2048 × gradient_accumulation 4` = 65,536 tokens/opt-step and 512,000 steps, training exposes 512,000 × 65,536 ≈ 33.5B tokens ≈ **4.2 epochs** over the 8.0B corpus. Repeated epochs are intentional — rare sources (arxiv, math) benefit from re-exposure, and the seeded per-epoch reshuffle (`shuffle=True` with a fixed-seed `torch.Generator`) means each epoch presents a different order. Note that resume does not restore the sampler RNG, so post-restart order differs — documented as benign at this scale in `Pretrainer.train`.

### Packed-Stream Semantics

Because shards are flat token streams with no per-sample records, every windowing property follows from two facts: windows are **contiguous** (`start = idx * S`) and **non-overlapping** (window `idx` occupies `[idx·S, idx·S + S + 1)`).

**Next-token alignment.** For `chunk = data[start : start + S + 1]`:

```
tokens  = chunk[:-1]   # positions 0..S-1
targets = chunk[1:]    # positions 1..S   (shifted by 1)
```

At position $t$ the model predicts $x_{t+1}$ from context $x_{\le t}$ — teacher forcing, where targets are ground-truth tokens, never model predictions. The shifted-pair construction is what makes a single contiguous slice yield both inputs and targets with zero duplication of storage.

**Cross-document boundaries.** Documents are concatenated with exactly one EOS between them. A window may span any number of document boundaries:

```
... doc_A ... EOS | doc_B ...
          ^^^^^^^^^ window boundary
```

The model learns to predict the first token of `doc_B` after seeing EOS — standard GPT-style pretraining. There is no special "document start" token beyond EOS, and no cross-document masking: the causal mask (`Transformer._build_causal_mask`) is purely positional, so context flows across document boundaries exactly as it does within one. This is a deliberate simplicity: EOS is the only structure the data carries, and the model internalises "EOS ⇒ new document" from data alone.

**Window count.** The flat, contiguous, non-overlapping layout is what makes the sample count a closed form rather than a scan: every window except the last has full length, so

$$
N_{\text{samples}} = \left\lfloor \frac{N_{\text{tokens}} - 1}{S} \right\rfloor
$$

is exact. The same formula is used for both layouts (`_n_samples` is computed once in `__init__`), which is why the single-file and sharded paths yield identical epochs for the same token count.

### Cross-Shard Windows

A window that would cross a shard boundary is stitched on the fly (`training/pretrain.py:PretrainDataset.__getitem__`):

```python
pieces = []
cursor = start
cursor_pos = 0
while cursor_pos < needed:
    shard_idx, offset_in_shard = self._locate(cursor)
    shard = self.shards[shard_idx]
    take = min(needed - cursor_pos, self.shard_sizes[shard_idx] - offset_in_shard)
    pieces.append(shard[offset_in_shard: offset_in_shard + take])
    cursor += take
    cursor_pos += take
chunk = torch.cat(pieces) if len(pieces) > 1 else pieces[0].clone()
return chunk[:-1], chunk[1:]
```

`take` is clamped by the shard's remaining tokens, so the loop advances to the next shard exactly when needed. In practice a window can span **at most two shards** (windows are $S+1 \le 50$M tokens, shards are 50M), so the `torch.cat` cost is a one-time concatenation of two mmap slices. Single-file layout skips all of this with one slice: `chunk = self.data[start: start + needed].clone()`.

### `_locate`: The Bisect Math

`training/pretrain.py:PretrainDataset._locate` maps a global token index to `(shard_idx, offset_within_shard)` in $O(\log K)$:

```python
def _locate(self, global_idx: int) -> Tuple[int, int]:
    """Map a global token index to (shard_idx, offset_within_shard). ..."""
    if global_idx < 0 or global_idx >= self._total_tokens:
        raise IndexError(
            f"global_idx {global_idx} out of range [0, {self._total_tokens})"
        )
    import bisect
    lo = bisect.bisect_right(self.shard_offsets, global_idx) - 1
    return lo, global_idx - self.shard_offsets[lo]
```

Why `bisect_right` minus one? `shard_offsets` is strictly increasing with `shard_offsets[0] = 0`. For any valid `global_idx`:

- `bisect_right(shard_offsets, global_idx)` returns the insertion point *after* all elements `<= global_idx`. Since `global_idx >= 0 = shard_offsets[0]`, the insertion point is `>= 1`, so `lo >= 0` — the shard index never goes negative.
- Because the insertion point is after *all* equal elements, a `global_idx` that is **exactly** `shard_offsets[k]` (the first token of shard `k`) maps to `lo = k`, offset 0 — the boundary token belongs to the next shard, which is the correct indexing (token `shard_offsets[k]` is physically shard `k`'s first element).
- For `global_idx` strictly inside shard `k` (i.e. `shard_offsets[k] < global_idx < shard_offsets[k+1]`), the insertion point is `k+1`, so `lo = k` and the offset `global_idx - shard_offsets[k]` is in `[1, shard_sizes[k])` — strictly inside the shard.

The explicit bounds check matters: with `bisect` alone, a corrupt index like `-5` or `N_tokens + 3` would silently wrap into a plausible-looking `(shard, offset)` and produce wrong training data. `IndexError` makes the corruption loud. This is exactly the contract guarded by `tests/test_training.py::TestPretrainDataset::test_locate_out_of_range_raises` and `test_locate_edge_case`, plus `test_sharded_cross_boundary` for the stitching loop.

### The uint32 → int64 Cast Boundary

The model boundary is `training/pretrain.py:Pretrainer.train_step`, which casts before entering autocast:

```python
if tokens.dtype != torch.long:
    tokens = tokens.to(torch.long)
if targets.dtype != torch.long:
    targets = targets.to(torch.long)
```

`nn.Embedding` indexes with Long, and `F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), ignore_index=-100)` requires Long targets. `models/transformer.py:Transformer.forward` defensively casts again at its own entry (`if tokens.dtype != torch.long: tokens = tokens.to(torch.long)`), so even a caller that forgets the boundary cannot crash the embedding — the uint32 representation never leaks past the model door. This dual guard is deliberate: the dataset stays compact, the training path stays dtype-correct, and the cost is one upcast of a (8, 2048) batch per micro-step — negligible against the forward/backward compute.

---

## CLI Reference

```bash
python3 data/prepare_data.py --stage pretrain \
    [--mixture PATH] \
    [--data-config PATH] \
    [--data-root PATH] \
    [--source NAME] \
    [--skip-download] \
    [--skip-clean] \
    [--skip-tokenize] \
    [--skip-pack]
```

| Flag | Effect |
|---|---|
| `--stage pretrain` | the only stage (the shim is a pretrain-only entry point) |
| `--mixture PATH` | override the mixture YAML (default: `LLM/shared_data/config/mixture.yaml`) |
| `--data-config PATH` | override the pipeline config (default: the generated `data/data_config.yaml`) |
| `--data-root PATH` | override the data root (default: `$LLM_DATA_ROOT` or `<cwd>/data`) |
| `--source NAME` | process a single mixture source by `id` (e.g. `--source fineweb-edu`) |
| `--skip-download` | re-run from cached raw data |
| `--skip-clean` | re-tokenize only |
| `--skip-tokenize` | re-pack only |
| `--skip-pack` | stop after tokenize |

Every stage is also independently invokable as `python -m shared_data.scripts.<download_raw|clean|tokenize|pack_shards>` from the `LLM/` root — the escape hatch for debugging one stage.

---

## Environment Variables

| Variable | Effect |
|---|---|
| `LLM_DATA_ROOT` | Shared cache root for all LLM projects; overrides the `<cwd>/data` default for `raw/`, `clean/`, `tokens/`, `shards/`, `state/`, `manifest.json` |
| `HF_HOME` / `HF_DATASETS_CACHE` | HuggingFace download cache (raw downloads and the tokenizer) |
| `TOKENIZERS_PARALLELISM=false` | Set by the launch script to silence HF fork warnings when the DataLoader spawns workers |
| `ENABLE_TRITON_KERNELS=1` | Required only if the config requests Triton attention/MoE — unrelated to data, but set alongside in `scripts/launch_a100.sh` |

---

## Operational Runbook

### First-time full pipeline

```bash
export LLM_DATA_ROOT=/path/to/shared_data   # optional; default is <cwd>/data
python3 data/prepare_data.py --stage pretrain
```

Stages run download → clean → tokenize → pack, each as an isolated subprocess. Expect hours to days depending on bandwidth and CPU cores (bandwidth-bound at stage 1, CPU-bound at stage 3). Output lands under the data root: `shards/shard_*.bin` + `shards/manifest.json`. The training configs (`configs/pretrain_a100_422m.yaml` → `data.train_data_path`, and the `scripts/launch_a100.sh` pre-flight) expect the shards at `data/pretrain_chinchilla/` — point them there (or symlink) once packing completes.

### Incremental reruns

| Goal | Flags |
|---|---|
| Re-download one source | `--source NAME` (no skip flags) |
| Re-tokenize after a tokenizer change | `--skip-download --skip-clean` |
| Re-pack only | `--skip-download --skip-clean --skip-tokenize` |
| Resume after a crash | re-run with no flags — per-stage state files skip completed work |

### CI / dev path (no shared_data, no HF auth)

```bash
python3 scripts/build_small_pretrain_data.py   # writes data/pretrain_chinchilla/shard_0000.bin
python3 training/pretrain.py --config configs/pretrain_1650_2m.yaml --no-compile
```

`build_small_pretrain_data.py` streams a small text source (fineweb-edu/wikitext, with a deterministic synthetic fallback), tokenizes with the requested tokenizer (`gpt2` for the 1650 config), and writes a single torch-serialised uint32 shard — byte-compatible with `PretrainDataset`. Never require 8.0B tokens in unit tests; the test suite builds tiny fixture shards instead.

### Output verification

```bash
ls data/pretrain_chinchilla/shard_*.bin | wc -l   # expect 160 shards at the 8.0B target
python -c "
import torch; from pathlib import Path
p = sorted(Path('data/pretrain_chinchilla').glob('shard_*.bin'))[0]
t = torch.load(p, weights_only=True, mmap=True)
print(t.dtype, t.numel())
"
# torch.uint32 50000000
```

---

## Pipeline Failure Modes

| Failure | Symptom | Fix |
|---|---|---|
| Wrong `vocab_size` in the model config | Embedding crash at step 0 | Regenerate `data_config.yaml` and the shards; keep `model.vocab_size` == `DEEPSEEK_VOCAB_SIZE` |
| Tokenizer changed without re-tokenizing | Ids out of range or garbage text | `--skip-download --skip-clean`, re-tokenize + re-pack |
| HF auth for the tokenizer/dataset | Download or tokenize stage fails | `huggingface-cli login`, or use the 1650 config (GPT-2) |
| Incomplete download | Missing source in manifest | Re-run without `--skip-download` (state file resumes) |
| Dedup too aggressive | Tiny corpus / huge `n_dropped_dedup` | Check clean-stage logs; per-source SHA-256 only drops exact normalised duplicates |
| Disk full during pack | Partial shard or failed `os.replace` | Delete `shard_*.bin.tmp` leftovers, free space, re-pack (shards are atomic) |
| Mixture/data-config token totals disagree | `WARNING: mixture.total_tokens != ...` at startup | Fix one of them; the mixture's value wins |
| Stale manifest | SHA mismatch or wrong totals | Re-run the pack stage; the manifest is rebuilt from disk |
| Raw shards from the universal pipeline fed to `PretrainDataset` | `RuntimeError: mmap can only be used with files saved with torch.save` | Convert to torch-serialised form, or use the universal `ShardDataset` reader |
| No shards in the configured dir | `ERROR: no shard_*.bin files in data/pretrain_chinchilla` (launch script) | Run the pipeline, or `build_small_pretrain_data.py` for smoke tests |

---

## Learning Exercises

1. **Token count audit:** Load one shard, count EOS tokens (id 100,017), and estimate documents per 50M-token shard: `(t == 100017).sum()`.
2. **Vocab spot-check:** Decode tokens around 100,000–100,017 — observe the special-token range and byte_fallback behaviour.
3. **Window boundary:** Find a sample index where `idx * max_seq_len` crosses a shard boundary; verify `__getitem__` stitches correctly and matches a manual concatenation.
4. **`_locate` invariants:** For random `global_idx`, assert `shard_offsets[lo] <= global_idx < shard_offsets[lo] + shard_sizes[lo]`; assert `_locate(0) == (0, 0)` and `_locate(total-1)` lands in the last shard.
5. **Sample-count edge case:** With `N` tokens and window `S`, confirm `len(ds) == (N - 1) // S` and that the final partial window is dropped, not padded.
6. **Mixture ablation:** Train a few hundred steps on a single source (e.g. `--source fineweb-edu` then re-pack), compare loss to the full mixture.
7. **Manifest audit:** Verify each shard's SHA-256 in `manifest.json` against the on-disk file; check `config_hash`/`mixture_hash` change when either YAML changes.
8. **Compression ratio:** Tokenize 1 MB of raw text with the DeepSeek tokenizer — compute bytes/token and compare against the 4.0 char/token download estimate.
9. **Resumption drill:** Kill a tokenize run mid-source (Ctrl+C), re-run with no flags, and confirm the state file skips completed docs and the stream is not re-written from zero.
10. **Cast boundary:** Feed `PretrainDataset` output straight into `Transformer.forward` and confirm the defensive `to(torch.long)` in `models/transformer.py:Transformer.forward` accepts the uint32 batch unchanged (dtype-correct output).

---

## Check Your Understanding

**Q1: Why does `PretrainDataset._locate` use `bisect_right` instead of `bisect_left`?** A: `bisect_right` returns the insertion point *after* all elements `<= global_idx`. For a `global_idx` exactly equal to `shard_offsets[k]` — the first token of shard `k` — the insertion point is `k+1`, so `lo = k` and the offset is 0: the boundary token maps to the shard that physically holds it. `bisect_left` would put it in the *previous* shard with an offset equal to that shard's size — one past the end — and the slice `shard[offset:offset+take]` would silently be empty. The off-by-one is the whole point.

**Q2: The corpus target is 8.0B tokens, but repo scripts say "~8.4B". Which is right?** A: 8.0B — it is the value in `mixture.yaml` (`total_tokens`), `UNIVERSAL_TOTAL_TOKENS` in `shared_data/config.py`, and the generated `data/data_config.yaml`, and it is what the pipeline enforces. "~8.4B" is the Chinchilla rationale (~20 tokens/param × 418.7M params with MTP ≈ 8.4B) quoted loosely; the realized budget is 8.0B → 160 shards of 50M tokens → 32 GB.

**Q3: A window of `S + 1 = 2049` tokens starts at global index 49,999,999 (the last token of shard 0). How many shards does it touch, and what does `_locate` return for the first `cursor`?** A: Two shards. `_locate(49_999_999)` returns `(0, 49_999_999)` — `shard_offsets[0] = 0`, and `bisect_right([0, 50_000_000], 49_999_999) - 1 = 0`. `take = min(2049, 50_000_000 - 49_999_999) = 1`, so one token is taken from shard 0; the next `_locate(50_000_000)` returns `(1, 0)` (boundary token belongs to shard 1) and the remaining 2048 tokens come from shard 1.

**Q4: Why is the final partial window dropped instead of padded with PAD (100,016)?** A: Padding would inject fake next-token targets — the model would be trained to predict PAD at the tail of every epoch, and the effective data distribution would include synthetic continuations. Dropping the tail wastes at most `S` tokens per 8.0B and keeps every training pair real. This is also why `add_special_tokens: false` + manual EOS is used: the only synthetic token in the stream is the document-boundary EOS, which is a genuine, learnable event.

---

## Appendix A — FAQ

**Q: Can I use the GPT-2 tokenizer?** Only for the 1650 smoke config (`vocab_size=50257`). The 422M config (`configs/pretrain_a100_422m.yaml`) requires the DeepSeek tokenizer (`vocab_size=100018`).

**Q: Where is the data guide?** The data guide is this document (the standalone data redirect was folded into this file). The universal pipeline's own README is at `LLM/shared_data/README.md`.

**Q: How long does full prep take?** Hours to days depending on bandwidth and CPU cores (download is bandwidth-bound; tokenize is CPU-bound). Use the `--skip-*` flags and `--source NAME` for incremental reruns. The shared_data README's stage wall-clock table (~8 h once, ~13 h for five projects sharing the clean output) is an estimate — no full run has been executed.

**Q: Does training read `manifest.json`?** No. `PretrainDataset` derives everything (shard list, sizes, offsets, totals) from the shard files themselves. The manifest is the audit/provenance record and the entry point for the universal `ShardDataset` reader.

**Q: What happens at a document boundary inside a window?** Nothing special — the stream is flat, documents are separated by EOS, and a window may span any number of document boundaries. The model learns to predict the first token of the next document after EOS, which is standard GPT-style pretraining. There is no document-start token beyond EOS and no masking of cross-document attention.

---

## Appendix B — Glossary

| Term | Meaning |
|---|---|
| `mixture.yaml` | Canonical source list + weights for the 8.0B corpus (`LLM/shared_data/config/mixture.yaml`) |
| `data_config.yaml` | Per-project tokenizer + pipeline settings, generated by the shim |
| data root | `$LLM_DATA_ROOT` or `<cwd>/data`; holds `raw/`, `clean/`, `tokens/`, `shards/`, `state/` |
| `manifest.json` | Shard metadata, per-source stats, config/mixture hashes |
| `TokenStream` | Per-source binary token stream: 8-byte header + raw uint32, EOS-separated |
| `ShardWriter` | Atomic packer: 50M-token raw uint32 shards, tmp + `os.replace` |
| `PretrainDataset` | This repo's training consumer: mmap'd shards, windowed samples, bisect `_locate` |
| uint32 | 4-byte token storage dtype on disk and in the dataset |
| int64 | Long dtype required by `nn.Embedding` / `F.cross_entropy`; cast at the model boundary |
| byte_fallback | BPE fallback to raw byte tokens; encoding is total, no `<unk>` ever |
| EOS (100,017) | Document boundary token, appended by tokenize and pack |
| PAD (100,016) | Padding token, defined but unused in packed pretraining |

---

- `data/prepare_data.py` — the DeepSeek shim (`_ensure_deepseek_data_config`, `main`)
- `data/data_config.yaml` — generated project config (DeepSeek overrides)
- `training/pretrain.py` — `PretrainDataset` (`__init__`, `_locate`, `__getitem__`) and `Pretrainer.train_step`
- `models/transformer.py` — `Transformer.__init__` (embedding rows) and `Transformer.forward` (defensive cast)
- `scripts/build_small_pretrain_data.py` — torch-serialised single-shard builder for smoke tests
- `LLM/shared_data/README.md` — universal pipeline design doc
- `LLM/shared_data/config/mixture.yaml` — canonical mixture (the only source of truth for weights)
- `LLM/shared_data/config/data_config.yaml` — universal pipeline knobs
- `LLM/shared_data/shard_writer.py`, `manifest.py`, `dedup.py`, `quality_filter.py`, `dataset.py` — pack, manifest, dedup, filter, reader implementations
- [Training](training.md) — the full training loop that consumes `PretrainDataset`
- [Inference](inference.md) — tokenizer reuse at inference time (chat template, `generate`)

---

## Appendix D — Data Pipeline Quick Start

The universal 8.0B-token pipeline implementation lives in the workspace sibling `LLM/shared_data/` (imported by `data/prepare_data.py` via `sys.path`; not vendored here). See `LLM/shared_data/README.md` for mixture, tokenization, and sharding details.

```bash
python3 data/prepare_data.py --stage pretrain
```
## References

- [training/pretrain.py](../training/pretrain.py)
- [MoE](concepts/moe-mtp.md) — bias mechanism
- [MTP](concepts/moe-mtp.md) — auxiliary loss
- [configs](training.md) — checkpoint format
- [training](training.md) — YAML reference
- `configs/pretrain_a100_422m.yaml`
- `configs/pretrain_1650_2m.yaml`
- `tests/conftest.py` — `cfg`, `small_cfg` fixtures
- [training](training.md) — how YAML maps to `TrainingConfig`
- `data/prepare_data.py` — the DeepSeek shim (`_ensure_deepseek_data_config`, `main`)
- `data/data_config.yaml` — generated project config (DeepSeek overrides)
- `training/pretrain.py` — `PretrainDataset` (`__init__`, `_locate`, `__getitem__`) and `Pretrainer.train_step`
- `models/transformer.py` — `Transformer.__init__` (embedding rows) and `Transformer.forward` (defensive cast)
- `scripts/build_small_pretrain_data.py` — torch-serialised single-shard builder for smoke tests
- `LLM/shared_data/README.md` — universal pipeline design doc
- `LLM/shared_data/config/mixture.yaml` — canonical mixture (the only source of truth for weights)
- `LLM/shared_data/config/data_config.yaml` — universal pipeline knobs
- `LLM/shared_data/shard_writer.py`, `manifest.py`, `dedup.py`, `quality_filter.py`, `dataset.py` — pack, manifest, dedup, filter, reader implementations
- [Training](training.md) — the full training loop that consumes `PretrainDataset`
- [Inference](inference.md) — tokenizer reuse at inference time (chat template, `generate`)
