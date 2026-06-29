# MoE — AuxLossFreeGate + DeepSeekMoE

Source: `models/moe.py`.

## AuxLossFreeGate (DeepSeek-V3 §2.3.3)

The gate routes each token to `n_activated_experts` (=4) of
`n_routed_experts` (=20) experts **without an auxiliary loss**. Load
balancing is maintained by a per-expert **bias** added to the sigmoid
scores:

```
scores   = sigmoid(x @ W_gate)        # (T, n_routed_experts)
biased   = scores + bias              # bias is a buffer, not a Parameter
indices  = biased.topk(topk).indices
weights  = scores.gather(indices)     # raw sigmoid scores, NOT biased
weights  = (weights / weights.sum()) * route_scale
```

### Bias-update mechanism (load-bearing)

`self.bias` is registered as a **buffer** (`register_buffer`), not a
`Parameter`:

- It receives **no autograd updates** — gradient never flows into it.
- It **is** persisted in `state_dict()` (buffers survive
  save/load), so checkpointed runs keep the learned bias.
- `test_bias_not_in_parameters` and `test_bias_in_state_dict` enforce
  these invariants.

`update_bias(counts, speed)` is `@torch.no_grad()` and run out-of-band
every `bias_update_every` optimizer steps (config: 1):

```
avg = counts.mean()
bias[counts > avg * (1 + upper)] -= speed   # over-loaded → demote
bias[counts < avg * (1 - lower)] += speed   # under-loaded → promote
```

Default `speed=0.001`, `upper=lower=0.10`. Because the bias shifts the
**topk selection** (via `biased`) but the routing **weights** come from
the raw sigmoid scores, no gradient term contaminates the task loss.
Replacing this with a standard auxiliary load-balancing loss **silently
breaks MoE balance** (see `AGENTS.md` hard rule 2).

### Group routing (optional)

When `n_expert_groups > 1`, the gate first picks `group_topk` top groups
per token via a masked topk, then routes within the surviving groups.
The 422M config uses `n_groups=1` (single group).

## Expert

A SwiGLU expert: `w2(silu(w1(x)) * w3(x))`. Shared by routed and shared
experts.

## DeepSeekMoE

- **20 routed experts** (top-4 per token) + **1 shared expert** (always
  active, no routing). The README mentions 2 shared experts but the
  config/code build 1 — see `CONTEXT.md` known issues.
- Two dispatch modes:
  - **`stacked` (default, `use_grouped="stacked"`)** — builds
    `_stacked_w1/w2/w3` lazily on first forward (one tensor per
    projection stacked across experts), then loops over experts with
    `bmm` against the stacked weight slices. One Python trip per layer.
  - **`grouped`** — reference path using `self.experts[e_id](...)` per
    chunk. `test_stacked_and_grouped_agree` verifies the two agree.
- Routing order is computed once: `argsort(flat_idx)` →
  `sorted_token_ids`, `sorted_weights`, `sorted_expert_id`; per-expert
  slices are read from these via `expert_offsets` (cumsum of
  `bincount`).
- `_last_weights` / `_last_indices` are stashed (detached) after each
  forward for `update_gate_bias` and balance/metric queries.
- `get_load_balance_loss()` returns the classic `(f·P)·N` balance
  surrogate (used for logging only; `balance_loss_alpha=0.0` in the
  aux-loss-free framework).
- `get_routing_stats()` returns `{counts, load, mean_weight, utilisation}`.