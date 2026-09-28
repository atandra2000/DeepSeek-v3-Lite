# DeepSeek-v3-Lite — μP LR Scaling

> **Audience:** expert. Assumes LR tuning and AdamW are familiar; the scaling theory is built here.
>
> **Read this if** you want the theory behind `lr_ref · (n_ref / n)^0.5` and an honest account of what this repo does and does not implement. **Skip if** you just want to run an LR sweep → [G2 — μP & LR Tuning](../guides/G2_mup_and_lr_tuning.md); for the full chapter → [Foundations](foundations.md) §13.

**Depends on:** optimizer basics · **Read next:** [G2 — μP & LR Tuning](../guides/G2_mup_and_lr_tuning.md), [Training Pipeline](../training.md)

---

## 1. The problem: the best LR is width-dependent

Scale a transformer's width while keeping the tuned LR and training destabilizes or under-trains: the optimal LR changes with size, so every width would need its own expensive sweep. μP (maximal-update parameterization, Yang et al. 2021) removes that cost by making the LR **transfer**: tune once on a cheap reference model, reuse on the large target.

## 2. Why width breaks naive tuning (built from scratch)

At init, a hidden preactivation `y = Wx` sums `n` width-scaled terms. With `O(1)` weights, `‖y‖ ~ √n`, so weights are conventionally initialized `O(1/√n)` to keep activations width-invariant. But the *update* to the function scales differently: one SGD/AdamW step perturbs the preactivation by `√n · η · ∂L/∂y`, which **grows with width** — the same η produces larger functional change in a wider model. Full μP fixes this with two coordinated rules: hidden-weight LRs scale as `1/n` while the output layer keeps a width-invariant LR (and initializations are adjusted to match). The result is a *feature-learning* regime whose update magnitude is width-invariant — the transfer property.

## 3. What this repo implements — and what it does not

This repo does **not** implement full μP (no per-tensor LR groups, no width-dependent init). It implements the **count-based transfer rule** that DeepSeek-V3 reports: one global LR, scaled by total parameter count to the power `−0.5`.

```python
# illustrative — the override, training/pretrain.py:Pretrainer.__init__ (runs before AdamW is built)
if config.mup_lr:
    new_lr = config.mup_lr_reference * (config.mup_lr_reference_params / total) ** 0.5
    config.lr = new_lr     # 'total' counts MTP too when the wrapper is enabled
```

The three keys live on `training/pretrain.py:TrainingConfig` — `mup_lr` (bool, default `false`; the canonical config sets `true`, the 1650 smoke config `false` because scaling only matters across widths), `mup_lr_reference: 6.0e-4`, and `mup_lr_reference_params: 757226496` (the reference run's parameter count). Because the override happens in `training/pretrain.py:Pretrainer.__init__` *before* the optimizer is constructed, AdamW and the warmup→cosine scheduler (`training/pretrain.py:make_warmup_cosine_lambda`) both inherit the scaled LR — there is no separate μP-aware scheduler.

## 4. The numbers, derived

With `lr_ref = 6.0e-4` at `n_ref = 757,226,496`:

- At **411.6 M** (base model, weight-tied): `6.0e-4 · (757226496 / 411.6e6)^0.5 ≈ 8.14e-4`
- At **418.7 M** (with the MTP wrapper enabled, which is what the trainer counts): `6.0e-4 · (757226496 / 418.7e6)^0.5 ≈ 8.07e-4`

Both are **derived arithmetic**, not measurements — the same values stated in [Foundations](foundations.md) §13. The direction is the tell: *fewer* parameters than the reference → *higher* LR, compensating the weaker √n functional-update growth.

## 5. Honest limits of the count-based rule

Tagging what the simplification costs [INFERENCE, no sweep has been run here]:

- **A single exponent for every tensor.** Full μP scales hidden layers by `1/n` and leaves the output layer alone; a global `(n_ref/n)^0.5` is a compromise that will mis-scale at least one component at large width gaps. At ~412 M vs a 757 M reference (a 1.8× gap) the error is modest.
- **Parameter count is a proxy for width.** MTP modules, MoE expert counts, and vocabulary size all move `n` without moving every layer's width uniformly.
- **Init is untouched.** μP's transfer guarantee assumes matching width-dependent initializations; this repo keeps its single init scheme.

## 6. Where to use this

Treat `mup_lr_reference[_params]` as the anchor of the transfer law: change them together, or reset `mup_lr: false` and tune `lr` directly. The operational procedure — grid, warmup sanity, divergence thresholds — is [G2 — μP & LR Tuning](../guides/G2_mup_and_lr_tuning.md); the schedule interaction (`opt_steps` horizon after gradient-accumulation division) is in [training.md](../training.md).

<!-- docs:verified 2026-09-21 · d9a5de4 -->
