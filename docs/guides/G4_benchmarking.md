# G4 — Benchmarking DeepSeek-v3-Lite on a GPU

> **Status:** guide · **Applies to:** anyone who quotes a throughput, VRAM, or MFU number
> **Depends on:** [[Docs/11_Operations_and_Testing]], [[Docs/08_Training_Pipeline]] · **Read next:** [G2 — μP & LR tuning](G2_mup_and_lr_tuning.md), [G3 — Triton development](G3_triton_development.md)

## 60-Second Summary

The repo ships three GPU benchmark scripts plus a built-in training-log throughput meter, but **no GPU run has ever executed** — every wall-clock, VRAM-peak, and MFU figure in this repo's prose is an estimate. This guide explains what each tool actually measures, shows the arithmetic you can verify on CPU today, lists what a real A100 benchmark must capture to be trustworthy, and sets the labeling rules for publishing results back into the docs. The run order on a fresh A100 is: `scripts/e2e_test_gpu.py` (does the whole pipeline work?) → `scripts/microbench_a100.py` (does it fit, and is the estimator honest?) → `scripts/step_time_a100.py` (how fast, what MFU?) → a real `training/pretrain.py` run (steady-state tps from the logger).

## 1. The Honest Baseline: No A100 Run Has Executed

### 1.1 Why this section exists

Every number in a benchmark report must carry one of three labels: `[MEASURED]` (read off a real device), `[DERIVED]` (computed from a formula — reproducible by anyone), or `[ESTIMATE]` (a guess with an unverified model behind it). The `.benchmarks/` directory at the repo root exists and is **empty** (verified 2026-08-04), so the current state of this repo is: **zero measured numbers**. Any prose that reads like a result is an estimate or a target, and the moment a real run produces numbers, the old estimates must be replaced — never mixed in silently.

### 1.2 The figures currently in the repo, and their true status

| Figure | Where it lives | Status |
|---|---|---|
| "~30–45 h wall time" | `configs/pretrain_a100_422m.yaml` header comment; `scripts/launch_a100.sh` (corrected 2026-08-04 from the arithmetically impossible 13–15 h, see §1.3) | `[ESTIMATE]` — derived, unverified |
| "~30–35 GB peak VRAM / 80 GB" | `scripts/launch_a100.sh` | `[ESTIMATE]` — consistent with the estimator math in §3.3 |
| "30–45% MFU" | `scripts/step_time_a100.py` docstring and closing thresholds | `[TARGET]`, not a result |
| "MFU > 30%" (prereq gate) | `scripts/launch_a100.sh` header comment | gate the microbench must satisfy before a launch |
| 411.6M base / 418.7M with MTP params | docs, configs, this guide | `[COUNTED]` — deterministic via `models/transformer.py:count_parameters` |

Note the config header comment itself says the run is "targeted" at those figures; nothing in the repo claims a measured value. Keep it that way until a real run lands.

### 1.3 A reconciliation you can do on CPU today

The 13–15 h claim deserves a sanity check before anyone books GPU time. All inputs below are verified from `configs/pretrain_a100_422m.yaml`:

- The scheduler horizon is `max_steps // gradient_accumulation_steps` opt-steps (computed in `Pretrainer.__init__`; `training/pretrain.py:Pretrainer.train_step` is where opt-steps are actually counted via `_opt_steps`): $512\,000 / 4 = 128\,000$.
- Tokens per schedule: $128\,000 \times 4 \times 8 \times 2048 = 8.39 \times 10^9$ — this matches the config's "8.4B Chinchilla-optimal tokens" comment, good.
- FLOPs per micro-step (see §4.2): $6 \times 334\,818\,432 \times 2048 \times 8 \approx 32.9 \times 10^{12}$.
- At 40% MFU of the A100 SXM BF16 peak (312 TFLOPS): $32.9 \times 10^{12} / (0.40 \times 312 \times 10^{12}) \approx 0.264$ s/step $\rightarrow$ 62.1k tok/s $\rightarrow$ **37.5 h**. At 35% MFU: **42.9 h**.

The old 13–15 h claim implied $8.39 \times 10^9 / 14\,\text{h} \approx 172.6$k tok/s $\rightarrow$ 0.095 s/step $\rightarrow$ ~347 TFLOPS $\rightarrow$ **111% MFU — impossible on A100**. It contradicted the very MFU assumption it cited. On 2026-08-04 the repo's wall-clock claims were corrected to **~30–45 h**, which brackets the two CPU-side reconciliations above (37.5 h @ 40% MFU, 42.9 h @ 35% MFU). The first real benchmark must still reconcile the residual spread: batch size, the logits GEMM, and grad-checkpoint recompute all move the number.

### 1.4 Known defect in the benchmark tooling itself (fixed 2026-08-04)

`scripts/microbench_a100.py:main` previously called `count_parameters(m)` without importing it — the first GPU invocation died with `NameError` before measuring anything. The import is now present:

```python
from models.transformer import Transformer, count_parameters
```

This defect is the canonical example of why this guide exists: a benchmark script nobody has run can look complete and still be broken.

## 2. Choosing a Benchmark — Decision Tree

| Question you are answering | Tool | GPU needed |
|---|---|---|
| Does the whole train→checkpoint→generate pipeline work on this GPU? | `scripts/e2e_test_gpu.py` | any CUDA GPU ≥ 4 GB |
| Will the canonical 411.6M config fit, and is the memory estimator honest? | `scripts/microbench_a100.py` | A100 80 GB (or the target card) |
| How fast is one training step, and what MFU do we get? | `scripts/step_time_a100.py` | A100 80 GB |
| What is real steady-state throughput with loss + MTP + logging? | `TrainingLogger.log` tps meter during a `training/pretrain.py` run | any training GPU |
| Eager vs `torch.compile`, stacked vs Triton dispatch? | `step_time_a100.py` with `--no-compile` / config flags; see [G3 — Triton development](G3_triton_development.md) | A100 |

Decision tree:

```
Is the GPU new to this repo? ──yes──► e2e_test_gpu.py (pipeline proof)
        │ no
Canonical config fits & estimator sane? ──no──► microbench_a100.py ──► fix config/safety margins
        │ yes
Need ms/step + MFU? ──yes──► step_time_a100.py ──► report [MEASURED]
        │ no
Need real-training throughput? ──yes──► training/pretrain.py + logger tps
```

## 3. Microbench — VRAM Estimate vs Measured

### 3.1 What it measures

`scripts/microbench_a100.py:main` builds the full canonical model (`Transformer(cfg, use_checkpoint=True)` on CUDA), prints the deduped parameter count, runs the memory estimator, asserts the estimate fits with a safety margin, then executes one real forward+backward while `torch.cuda.max_memory_allocated()` tracks the **measured** peak. Output: estimated GB, measured GB, percentage delta, and a graded verdict (headroom / comfortable / dangerously close). It answers one question: *"does the estimator lie, and by how much?"* — it does **not** measure speed.

### 3.2 `count_parameters` dedup

Both the microbench and the estimator rely on deduplicated counts. Naive `sum(p.numel() for p in model.parameters())` double-counts the tied output head, because `self.head.weight = self.embed.weight` makes the same tensor appear under two modules. `models/transformer.py:count_parameters` fixes this by tracking tensor `id`s:

```python
def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """(total, trainable) — deduplicated by tensor id (shared weights counted once)."""
    seen = set()
    total = 0
    ...
    for p in model.parameters():
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        ...
```

Verified on CPU (2026-08-04): $411\,632\,256$ total and trainable — the tied head counted once, matching the locked constant. The naive sum would report $488\,446\,080$ (off by $76\,813\,824 = 100\,018 \times 768$).

### 3.3 The estimator, component by component

`utils/memory.py:estimate_model_memory_gb` sums five terms. Running it on CPU against the canonical config (bs=8, seq=2048) gives these `[DERIVED]` numbers, reproducible by anyone:

| Component | Formula (source) | GB |
|---|---|---|
| Weights (BF16) | $411\,632\,256 \times 2$ B | 0.82 |
| Optimizer (fp32 AdamW) | $411\,632\,256 \times 12$ B (master + m + v) | 4.94 |
| KV cache (MLA) | $8 \times 2048 \times (192 + 24) \times 2$ B $\times$ 18 layers | 0.13 |
| Activations, grad-checkpointed | `_activation_bytes` heuristic $\times$ 8·2048·768·18·2 B | 10.87 |
| Activations, no checkpoint | same, larger factor | 16.31 |
| PyTorch/CUDA overhead | `_detect_overhead_gb`: CPU 2.0; CUDA min(13.7, 17% of VRAM) | 2.0 / 13.6 |

Sum with CUDA overhead: $0.82 + 4.94 + 0.13 + 10.87 + 13.6 \approx$ **30.4 GB** on an A100 80 GB — consistent with the "~30–35 GB" claim in `scripts/launch_a100.sh`. The 13.7 GB static overhead line is the single biggest term after activations; treat it as the estimator's weakest assumption, which is exactly why the measured-vs-estimated delta matters.

### 3.4 Safety margin

`scripts/microbench_a100.py:main` passes `safety_margin_gb=2.0` to `utils/memory.py:assert_fits_in_available_gpu`, which raises `RuntimeError` if the estimate exceeds free VRAM minus the margin. After measuring, the script warns when the measured peak sits within 8 GB of capacity ("consider halving micro_batch_size or seq_len"), prints a notice above 70% of VRAM, and otherwise reports headroom.

### 3.5 Running it

```bash
# after the §1.4 import fix; A100 80 GB
python3 scripts/microbench_a100.py
```

Expected output shape: `estimated peak`, `measured peak`, `delta vs estimate = X%`. A delta above ~20% should trigger an investigation of the activation-factor and overhead assumptions in `utils/memory.py`, not a silent acceptance.

## 4. Step-Time — ms/Step, Tokens/s, MFU

### 4.1 What it measures

`scripts/step_time_a100.py:main` builds the canonical model, sets TF32 flags, creates a fused AdamW, optionally wraps the model in `torch.compile`, runs `--warmup` steps, then times `--steps` synthetic steps (randint input → forward → `sum().backward()` → `optimizer.step()`). It prints mean **ms/step**, **tok/s**, achieved TFLOPS, and **MFU** against the configurable peak (`--peak-tflops`, default 312.0 = A100 SXM BF16 dense).

### 4.2 The MFU formula

The script counts non-embedding params as `n_p − vocab·dim` when weight tying is on (it sums raw `parameters()`, which double-counts the head — the subtraction removes the duplicate, and both benchmark scripts are consistent with `count_parameters` on this point): $n_{nonembed} = 411\,632\,256 - 76\,813\,824 = 334\,818\,432$. Then

$$\text{FLOPs/step} = 6 \times n_{nonembed} \times seq \times bs \approx 32.9 \times 10^{12}$$

(the 6 covers the 2× fwd + 4× bwd multiply-accumulates), and

$$\text{MFU} = \frac{\text{FLOPs/step} / \text{step\_time}}{\text{peak\_tflops}}.$$

The docstring's "validates ~30–45% MFU" is a **target for MoE-on-A100 BF16**, not a measured result. The script's own grading: MFU < 25% → investigate (suspected MoE Python-loop overhead, missing compile, missing TF32); 25–35% → workable but improvable; ≥ 35% → "expected 30–45% range".

### 4.3 Flags you must know

- `--no-compile` — disables `torch.compile`. **Watch for silent fallback**: compile is wrapped in try/except; a failure prints `torch.compile: FAILED (...) continuing without` and the run proceeds eager. Always grep the output for `torch.compile: enabled` before trusting an MFU number.
- `--compile-mode` (default `max-autotune`) — the expensive-but-fastest mode; first-call compilation can take minutes, which is why warmup exists.
- `--warmup 5` / `--steps 20` — the script reports the **mean** over the timed window. For publication, use `--steps 50` and report the median plus spread (see §7).

TF32 note: this script sets `torch.backends.cuda.matmul.allow_tf32 = True`, `torch.backends.cudnn.allow_tf32 = True`, and `cudnn.benchmark = True`. TF32 changes numerics — a TF32 benchmark's loss curve is not directly comparable to a CPU fp32 reference run. Record the flags next to every number you publish.

### 4.4 Running it

```bash
python3 scripts/step_time_a100.py --steps 50 --warmup 10          # compiled, max-autotune
python3 scripts/step_time_a100.py --no-compile --steps 50         # eager baseline for comparison
```

## 5. E2E GPU Smoke — the 1650 Run

### 5.1 What it verifies

`scripts/e2e_test_gpu.py:run_e2e` runs seven gates on the tiny `configs/pretrain_1650_2m.yaml` (dim 64, 4 layers, ~2M params, but all architecture features: MLA, aux-loss-free MoE, MTP depth 1):

1. data pipeline — `PretrainDataset` loads, sample shape/shift invariant `x[1:] == y[:-1]`, batched GPU transfer;
2. model construction — `Transformer` on GPU, forward logits `(2, 16, vocab_size)`, finite;
3+4. training — a real `Pretrainer` runs `--train-steps` (default 20) micro-steps with the NaN guard and MTP loss, loss must stay finite and not explode;
5. checkpoint round-trip — save, reload, finite logits;
6. autoregressive generation with KV cache;
7. speculative MTP decoding via `SpeculativeDecoder`.

`scripts/e2e_test_gpu.py:Results.add` records each case; `scripts/e2e_test_gpu.py:main` exits non-zero if any case failed. The explicit "Peak VRAM < 4GB" case is the 1650 fit gate.

### 5.2 The deliberate degradations — and why it is not a speed benchmark

Two things are forcibly turned off inside `run_e2e`: `config.compile_model = False` ("1650: torch.compile can OOM or hang") and fused AdamW is monkeypatched to `fused=False` (the fused CUDA path may be absent on the test host). The e2e test is a **correctness** smoke on constrained hardware; its ms/step is meaningless as a performance number. Do not quote it as throughput.

### 5.3 Prerequisites and running

The script hardcodes the dataset path `data/pretrain_chinchilla`, which does **not** exist in the repo checkout right now — prepare it first, then run:

```bash
python3 data/prepare_data.py --stage pretrain
python3 scripts/e2e_test_gpu.py --config configs/pretrain_1650_2m.yaml --train-steps 20
```

## 6. The TrainingLogger tps Meter — Steady-State Throughput

### 6.1 Formula

`utils/logging.py:TrainingLogger.log` maintains a rolling loss window and, every `log_interval` micro-steps, prints `step=… loss=… ppl=… lr=… tps=…` where

$$\text{tps} = \frac{\text{log\_interval} \times seq\_len \times batch\_size}{\text{elapsed}}.$$

`batch_size` here is `config.batch_size` — which `training/pretrain.py` populates from `micro_batch_size`, not the accumulated batch. So the meter reports **micro-step stream** tokens per second: with the canonical config, 50 × 2048 × 8 = 819,200 tokens per window. `ppl` is $\exp(\text{avg\_loss})$ over the window. WandB logging is opt-in via the `WANDB_PROJECT` environment variable (which `scripts/launch_a100.sh` sets).

### 6.2 What the number includes

The window wall-time includes data loading, the optimizer step, and — deliberately — only **one** `.item()` GPU-sync per log window (see the comment at the log call site in `training/pretrain.py`): the meter is a genuine steady-state figure, not a microbenchmark. Checkpoint saves happen on a separate branch and can stretch one window; a tps outlier right after a save is expected. It is a rolling window that resets every log (`self._step_start = time.time()`), so it is never a cumulative average.

### 6.3 From tps to MFU

$$\text{MFU} = \frac{6 \times 334\,818\,432 \times \text{tps}}{312 \times 10^{12}} \times 100$$

This is the only conversion you need to compare logger tps against `step_time_a100.py` output — but expect real-training tps to sit below the synthetic step-time figure, because the real loop adds loss computation (including MTP, +7.1M params) and checkpoint/save traffic that the microbench omits.

## 7. What a Real Benchmark Must Capture

Checklist for the first genuine A100 run — every item is either handled by a script or must be added by the operator:

- [ ] **Pipeline proof first**: `e2e_test_gpu.py` green on the target card before any perf run.
- [ ] **Pre-warm**: discard ≥ 10 steps; verify `torch.compile: enabled` in the log (a silent eager fallback invalidates the number).
- [ ] **Compile amortization**: record compile time separately; time only post-compile steps. `max-autotune` first call can take minutes.
- [ ] **TF32 flags recorded**: note `allow_tf32` settings next to every number; never mix TF32 and non-TF32 runs in one comparison without a label.
- [ ] **Memory peak vs steady**: call `torch.cuda.reset_peak_memory_stats()` after init, then report both the window peak and the steady level (caching-allocator growth can push peak far above steady; the microbench's measured peak is the number the fit decision should use).
- [ ] **Median, not just mean**: `step_time_a100.py` prints the mean; re-time with `--steps 50` and report p50/p95 from the raw per-step samples.
- [ ] **Environment block**: `torch.__version__`, CUDA version, driver, GPU name, clocks (`nvidia-smi`), cuDNN version — attach it to the results file.
- [ ] **Token reconciliation**: recompute total tokens from `total_steps // grad_accum × grad_accum × bs × seq` and compare with the data size (see §1.3 — the current config's claims disagree with each other).
- [ ] **Results file**: write JSON to `.benchmarks/<date>_a100.json` (the dir exists and is empty — this is what it is for): config hash, env block, flags, per-step samples, peak/steady memory, tps, MFU.
- [ ] **Correctness sidecar**: one short `training/pretrain.py` run with the same flags, confirming loss decreases — a fast-but-diverging run is not a benchmark result.

## 8. Reporting Numbers Back Into the Docs

Rules, in order of importance:

1. **Label everything**: `[MEASURED 2026-08-04 A100-80GB]` with the §7 environment block, or `[ESTIMATE]`, or `[DERIVED]` with the formula. Never a bare number.
2. **Replace, don't mix**: when measured data exists, the old estimate in the same location is deleted, not annotated. Two numbers for one figure in one place is how stale claims survive.
3. **Every perf claim cites its script**: anchor the tool that produced it (`scripts/step_time_a100.py:main` for MFU, `scripts/microbench_a100.py:main` for VRAM, `utils/logging.py:TrainingLogger.log` for tps). No script, no claim.
4. **Parameter counts stay `[COUNTED]`** via `models/transformer.py:count_parameters` — 411.6M / 418.7M is not a benchmark and never gets relabeled.

Where the numbers live today, and what to touch after the first run: the "13–15 h / 35–40% MFU" comment in `configs/pretrain_a100_422m.yaml` and the `scripts/launch_a100.sh` echoes (both `[ESTIMATE]`, §1.3 says they are wrong); the VRAM budget section of `docs/11_Operations_and_Testing.md` (the estimator walkthrough lives in `../reference/R8_utils_api.md`); the budget tables in `docs/02_Model_Architecture.md`; and this guide's §1.2 table.

## 9. Pitfalls (Consolidated)

| Trap | Symptom | Defense |
|---|---|---|
| Unrun benchmark script | `NameError: count_parameters` in `microbench_a100.py` | apply the §1.4 import fix; treat every script as suspect until run once |
| Silent eager fallback | output has no `torch.compile: enabled` line | grep the log; `--no-compile` runs are labeled |
| Mean instead of median | one slow step (dataloader, save) inflates ms/step | report p50/p95 from `--steps 50` samples |
| Quoting e2e ms/step as throughput | 1650 run is correctness-only, compile off, fused off | only `step_time_a100.py` or logger tps are throughput |
| TF32/non-TF32 mixing | loss curves differ from fp32 references | record `allow_tf32` next to every number |
| Counting the tied head | params reported 488.4M instead of 411.6M | always `count_parameters`, never raw `parameters()` |
| Estimating instead of measuring | "fits" but OOMs at step 17 | trust `measured peak`, not `estimated peak` |
| MTP omitted | microbench/step-time build base-only; training adds 7.1M params | label which model the number is for |

## 10. Check Your Understanding

1. **Q:** Why is the "13–15 h" comment untrustworthy even before any GPU time? **A:** At bs=8, seq=2048 and 8.39B scheduled tokens it implies ~111% MFU of the A100 BF16 peak — mathematically impossible at the 35–40% MFU it claims (§1.3).
2. **Q:** `sum(p.numel() for p in model.parameters())` on the canonical config reports ~488M params. Where does the error come from? **A:** The tied output head shares the embedding tensor, so `parameters()` yields it twice; `count_parameters` dedups by tensor id (§3.2).
3. **Q:** You measure 62,000 tok/s on the logger during a real run. What is the MFU? **A:** $6 \times 334\,818\,432 \times 62\,000 / (312 \times 10^{12}) \approx 40\%$, before accounting for MTP's extra FLOPs — and only if TF32 + compile were on and recorded (§6.3).

<!-- docs:verified 2026-08-04 · 59aeef3 -->
