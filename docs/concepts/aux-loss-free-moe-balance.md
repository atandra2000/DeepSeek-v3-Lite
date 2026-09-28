# DeepSeek-v3-Lite — Aux-Loss-Free MoE Balance

> **Audience:** intermediate. Builds its own prerequisites: top-k MoE routing is defined here.
>
> **Read this if** you want to understand the bias feedback loop — the control system that replaces the load-balancing loss. **Skip if** you want dispatch layouts and Triton paths → [DeepSeekMoE & MTP](moe-mtp.md), [R4 — MoE API](../references/R4_moe_api.md).

**Depends on:** nothing (routing defined below) · **Read next:** [DeepSeekMoE & MTP](moe-mtp.md) (full treatment), [MLA: Latent Attention](mla-latent-attention.md)

---

## 1. Prerequisite in one paragraph

Each MoE layer holds 20 routed experts (`models/moe.py:Expert` — a SwiGLU FFN of width 384) plus 1 shared expert that every token passes through. A router scores all experts per token and picks the **top-4**; the chosen experts run, and their outputs are blended by normalized routing weights. Nothing forces usage to spread across experts — if the router early on favors expert 7, gradient descent *reinforces* expert 7, and unused experts stop learning. Unbalanced routing wastes capacity and compute.

## 2. The classic fix and its cost

The classic fix is an **auxiliary load-balancing loss**: add a term to the training loss that penalizes uneven expert usage. It works, but it mixes objectives — the balance gradient flows into the router *and* into the experts/attention through the shared loss, trading task quality for balance. DeepSeek-V3's answer: **keep the loss clean; steer routing with a bias instead.**

## 3. The mechanism: selection uses the bias, weights do not

`models/moe.py:AuxLossFreeGate` keeps a per-expert **bias buffer** `b` (registered as a buffer, not a parameter — it never receives gradients). The split rule is the whole idea:

```python
# illustrative — the selection/weighting split, models/moe.py:AuxLossFreeGate.forward
scores = F.linear(x, self.weight).sigmoid()            # unbiased per-expert scores s_i
biased = scores + self.bias.to(scores.dtype)           # s_i + b_i — selection ONLY
indices = biased.topk(self.topk, dim=-1)[1]            # top-4 chosen by the biased score
weights = scores.gather(1, indices)                    # routing weight = unbiased s_i
```

The bias decides **who gets picked**; the unbiased sigmoid score decides **how much each pick counts**. The model's function is therefore computed exactly as an unbalanced router would compute it — the bias exists purely to reshuffle the top-4 set.

## 4. The update rule

After `bias_update_every` optimizer steps (`training/pretrain.py:TrainingConfig` — 1 in the canonical config, default 10), `training/pretrain.py:Pretrainer._update_moe_bias` calls `models/moe.py:DeepSeekMoE.update_gate_bias`, which counts how many tokens each expert saw in the latest forward and nudges the bias with a **deadband rule** (`models/moe.py:AuxLossFreeGate.update_bias`, `@torch.no_grad`):

```python
# illustrative — the deadband rule, verbatim logic
avg = counts.mean()
self.bias[counts > avg * (1.0 + self.bias_upper)] -= speed   # over-used  → 0.001 step down
self.bias[counts < avg * (1.0 - self.bias_lower)] += speed   # under-used → 0.001 step up
```

Both thresholds default to 0.10, so experts within ±10% of the mean token share are left alone; only outliers move, by `bias_update_speed = 0.001` per update. It is a **controller**, not a loss: an integral controller whose setpoint is uniform utilization, acting only through routing selection.

## 5. Why this beats the auxiliary loss

- **Zero interference.** The update runs under `no_grad`, out-of-band. The task gradient never sees a balance term, so attention and experts learn exactly the language-modeling objective.
- **Weight purity.** Because routing *weights* stay unbiased, the loss landscape the optimizer sees is unchanged; only the discrete top-4 choice is steered.
- **Self-correcting.** Lowering a hot expert's bias lets it lose tokens next step; counts fall; the deadband stops the correction — no oscillation as long as `speed` stays small.

## 6. Honesty note: the "loss" that is only a metric

The repo still implements the auxiliary-loss *formula* — `models/moe.py:DeepSeekMoE.get_load_balance_loss` computes the standard `Σ fᵢ·Pᵢ` balance number from the latest forward — but it is **logged, never added to the loss**. It exists so the pretrain loop can print how balanced routing actually is; see [training.md](../training.md). That distinction (diagnostic vs objective) is easy to miss and load-bearing for the "aux-loss-free" claim.

<!-- docs:verified 2026-09-21 · d9a5de4 -->
