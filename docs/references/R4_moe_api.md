# DeepSeek-v3-Lite — R4 MoE API Reference

The sparse Mixture-of-Experts module: an auxiliary-loss-free routing gate
(`AuxLossFreeGate`), a SwiGLU expert FFN (`Expert`), and the composite
(`DeepSeekMoE`) that routes tokens, dispatches to routed experts, adds the
shared-expert output, and reports the load-balance metric + bias-update state.
It implements DeepSeek-V3's per-token sigmoid-gated top-k routing (V3 §2.3.3),
single-GPU, BF16-first. Two dispatch modes exist — a pure-PyTorch per-expert
loop (`stacked`) and a Triton grouped-GEMM path (`triton_grouped`) with an
automatic one-shot fallback. See [DeepSeekMoE & MTP](../concepts/moe-mtp.md) for the tutorial
treatment; this file is the API contract.

## Module map

| Symbol | Kind | One-line purpose |
|---|---|---|
| `models/moe.py:AuxLossFreeGate` | class | Sigmoid top-k router with learnable `weight`, non-trainable fp32 `bias` buffer. |
| `models/moe.py:AuxLossFreeGate.__init__` | method | Read gate config keys; init weight (std 0.006), register bias buffer. |
| `models/moe.py:AuxLossFreeGate.update_bias` | method | In-place, no-grad bias shift toward equal load (band-triggered ±`speed`). |
| `models/moe.py:AuxLossFreeGate.forward` | method | Return `(weights, indices)` — top-k over biased scores, normalized unbiased weights. |
| `models/moe.py:Expert` | class | Single SwiGLU expert `W2(silu(W1(x)) * W3(x))`. |
| `models/moe.py:Expert.__init__` | method | Three bias-free linears: `w1`, `w3` (`dim→inter_dim`), `w2` (`inter_dim→dim`). |
| `models/moe.py:Expert.forward` | method | Apply SwiGLU; `(…, dim) → (…, dim)`. |
| `models/moe.py:DeepSeekMoE` | class | Routed + shared experts, dispatch, balance metric, bias-update entry. |
| `models/moe.py:DeepSeekMoE.__init__` | method | Build gate, expert lists, plain-attribute weight stacks and snapshots. |
| `models/moe.py:DeepSeekMoE.forward` | method | Flatten → route → dispatch → add shared output → restore shape. |
| `models/moe.py:DeepSeekMoE._routed_forward_stacked` | method | Per-expert Python-loop routed forward; default AND fallback. |
| `models/moe.py:DeepSeekMoE._routed_forward_triton` | method | Sorted grouped-GEMM routed forward via `models/moe_triton.py:triton_grouped_moe_dispatch`. |
| `models/moe.py:DeepSeekMoE._shared_forward` | method | Batched shared-expert forward, 1 bmm per SwiGLU projection. |
| `models/moe.py:DeepSeekMoE.get_load_balance_loss` | method | Aux-loss-free balance metric from the last forward's detached snapshots. |
| `models/moe.py:DeepSeekMoE.update_gate_bias` | method | Feed last-forward expert counts to `AuxLossFreeGate.update_bias`. |

## 1. Config keys consumed

All `config` args are plain `dict` reads (no dataclass). Keys marked "required"
raise `KeyError` if absent; the rest fall back to the default shown.

| Key | Default | Read by | Canonical (`configs/pretrain_a100_422m.yaml`) | Smoke (`configs/pretrain_1650_2m.yaml`) |
|---|---|---|---|---|
| `dim` | — (required) | gate, MoE | 768 | 256 |
| `n_routed_experts` | — (required) | gate, MoE | 20 | 4 |
| `n_activated_experts` | — (required) | gate (`topk`) | 4 | 1 |
| `n_shared_experts` | — (required) | MoE | 1 | 1 |
| `moe_inter_dim` | — (required) | MoE | 384 | 32 |
| `moe_dispatch` | `"stacked"` | MoE | `"stacked"` | `"stacked"` |
| `route_scale` | `1.0` | gate | (absent) | (absent) |
| `bias_upper_threshold` | `0.10` | gate | (absent) | (absent) |
| `bias_lower_threshold` | `0.10` | gate | (absent) | (absent) |

Training-loop keys (read by `training/pretrain.py`, not the module): `bias_update_speed`
(default `0.001`) and `bias_update_every` (default `1`) — see §7.

## 2. `AuxLossFreeGate`

Anchor: `models/moe.py:AuxLossFreeGate`. Router: per-token sigmoid scores, top-k
over bias-shifted scores, weights taken from the *unbiased* scores.

### `__init__(self, config: dict)`

Anchor: `models/moe.py:AuxLossFreeGate.__init__`.

```python
self.dim = config["dim"]
self.topk = config["n_activated_experts"]
self.n_routed_experts = config["n_routed_experts"]
self.route_scale = config.get("route_scale", 1.0)
self.bias_upper_threshold = config.get("bias_upper_threshold", 0.10)
self.bias_lower_threshold = config.get("bias_lower_threshold", 0.10)
self.weight = nn.Parameter(torch.empty(self.n_routed_experts, self.dim))
nn.init.normal_(self.weight, std=0.006)
self.register_buffer("bias", torch.zeros(self.n_routed_experts, dtype=torch.float32))
```

State contract:

| Attribute | Type | Role |
|---|---|---|
| `weight` | `nn.Parameter` `(E, dim)`, init `Normal(0, 0.006)` | Learned routing logits — gets gradients. Same init std as the rest of the model (see [Foundations & Architecture](../concepts/foundations.md)). |
| `bias` | `register_buffer` `(E,)` **fp32** | Load-balancing bias. NOT a `Parameter`: excluded from `named_parameters()`, the optimizer, and `nn.utils.clip_grad_norm_`; updated in place under `torch.no_grad()` only. Lives in `state_dict` (buffers are serialized), so it round-trips checkpoints. |

The fp32 dtype is deliberate: the bias is updated by tiny `±speed` (0.001)
increments; a BF16 buffer would quantize most updates to zero.

### `update_bias(self, counts: torch.Tensor, speed: float = 0.001) -> None`

Anchor: `models/moe.py:AuxLossFreeGate.update_bias`. `@torch.no_grad`, in-place.

```python
counts = counts.float()
avg = counts.mean()
self.bias[counts > avg * (1.0 + self.bias_upper)] -= speed
self.bias[counts < avg * (1.0 - self.bias_lower)] += speed
```

- `counts`: per-expert token counts, `(E,)`, int or float, any device.
- Band-triggered control law: an expert over the upper band (`avg·(1+ε_upper)`)
  is *penalized* (`−speed`); an expert under the lower band (`avg·(1−ε_lower)`)
  is *boosted* (`+speed`); experts inside the band are untouched. No gradient —
  pure in-place buffer arithmetic.
- The boolean masks must share a device with `bias`; the caller keeps `counts`
  on the model device (see `DeepSeekMoE.update_gate_bias`).
- Returns `None`.

### `forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]`

Anchor: `models/moe.py:AuxLossFreeGate.forward`.

```python
T = x.size(0)
scores = F.linear(x, self.weight).sigmoid()
biased = scores + self.bias.to(scores.dtype)
indices = biased.topk(self.topk, dim=-1)[1]
weights = scores.gather(1, indices)
weights = (weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-10) * self.route_scale).to(x.dtype)
return weights, indices
```

- Input `x`: `(T, dim)` — the flattened hidden state.
- `scores`: `(T, E)` sigmoid probabilities (per-expert independent, not
  softmax — several experts can be simultaneously "on" for one token).
- `indices`: `(T, topk)` int64, `torch.topk` over **biased** scores — the bias
  only re-ranks selection; it never enters the weight.
- `weights`: `(T, topk)` in `x.dtype`. Gathered from **unbiased** `scores`,
  then renormalized to sum `route_scale` per token (`Σ = 0` protected by
  `clamp(min=1e-10)`). With defaults every token's routed weights sum to 1.
- Gradient flows to `weight` through both the sigmoid scores and the gather.

Who calls it: `models/moe.py:DeepSeekMoE.forward` (once per forward, on the
flattened `(T, dim)` tensor).

## 3. `Expert`

Anchor: `models/moe.py:Expert`. One SwiGLU expert; no routing logic.

```python
class Expert(nn.Module):
    """Single SwiGLU expert: W2(silu(W1(x)) * W3(x))."""
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

- `w1`, `w3`: `(inter_dim, dim)` weight matrices (out, in), no bias.
- `w2`: `(dim, inter_dim)`, no bias.
- Forward: `(…, dim) → (…, dim)`; gate/up projections share the input, elementwise
  `silu(gate)·up`, then down-project. Same shape contract as the dense
  `SwiGLUFFN` (see `./R2_transformer_api.md`).
- Called only by `DeepSeekMoE` — never directly by the training loop.

## 4. `DeepSeekMoE`

Anchor: `models/moe.py:DeepSeekMoE`. Composite: gate + `n_routed_experts` routed
experts + `n_shared_experts` shared experts.

### `__init__(self, config: dict)`

Anchor: `models/moe.py:DeepSeekMoE.__init__`.

```python
self.dim = config["dim"]
self.n_routed_experts = config["n_routed_experts"]
self.n_shared_experts = config["n_shared_experts"]
self.moe_inter_dim = config["moe_inter_dim"]
self.moe_dispatch = config.get("moe_dispatch", "stacked")
self.gate = AuxLossFreeGate(config)
self.experts = nn.ModuleList([Expert(self.dim, self.moe_inter_dim) for _ in range(self.n_routed_experts)])
self.shared_experts = nn.ModuleList([Expert(self.dim, self.moe_inter_dim) for _ in range(self.n_shared_experts)])
self._stacked_w1: Optional[torch.Tensor] = None
self._stacked_w2: Optional[torch.Tensor] = None
self._stacked_w3: Optional[torch.Tensor] = None
self._shared_w1: Optional[torch.Tensor] = None
self._shared_w2: Optional[torch.Tensor] = None
self._shared_w3: Optional[torch.Tensor] = None
self._last_weights: Optional[torch.Tensor] = None
self._last_indices: Optional[torch.Tensor] = None
```

State contract — three distinct storage classes:

| Attribute | Class | Gradient | Serialized | Notes |
|---|---|---|---|---|
| `gate.weight` | `Parameter` | yes | yes | via the `gate` submodule |
| `gate.bias` | **buffer** | no | yes | §2 bias contract |
| `experts[*].w{1,2,3}.weight` | `Parameter` | yes | yes | routed experts |
| `shared_experts[*].w{1,2,3}.weight` | `Parameter` | yes | yes | shared experts |
| `_stacked_w{1,2,3}` | **plain attribute** | no (views of params, but the *stack* is a new tensor) | **no** | rebuilt every forward — see the re-stack rule below |
| `_shared_w{1,2,3}` | **plain attribute** | no | **no** | rebuilt every forward |
| `_last_weights`, `_last_indices` | **plain attribute** | no (`detach()`ed) | **no** | snapshots for metric + bias update |

**Re-stack-every-forward rule (MUST NOT cache `_stacked_w*` / `_shared_w*` across
optimizer steps).** The comment in `forward` is normative:

> Re-stack every forward: caching across steps leaves stale copies after
> optimizer.step() (experts would be frozen at init values).

`torch.stack([ex.w1.weight …])` produces a **new tensor** — a snapshot, not a
view. If it were cached, `optimizer.step()` would mutate the underlying expert
parameters while the stale stacked copy would still point at pre-update memory,
so the MoE would train with frozen-at-init expert weights while the optimizer
"moves" parameters nobody reads. The stacks are also cast to
`device=flat.device, dtype=flat.dtype` each time, so the module stays portable
across dtype/device moves. They are deliberately **not** `register_buffer`:
buffers would be swept into `state_dict` and persist across steps.

### `forward(self, x: torch.Tensor) -> torch.Tensor`

Anchor: `models/moe.py:DeepSeekMoE.forward`.

```python
shape = x.shape
flat = x.view(-1, self.dim)
T = flat.size(0)
weights, indices = self.gate(flat)
self._last_weights = weights.detach()
self._last_indices = indices.detach()
E, I, D = self.n_routed_experts, self.moe_inter_dim, self.dim
# Re-stack every forward: …
self._stacked_w1 = torch.stack([ex.w1.weight for ex in self.experts], dim=0).to(device=flat.device, dtype=flat.dtype)
self._stacked_w2 = torch.stack([ex.w2.weight for ex in self.experts], dim=0).to(device=flat.device, dtype=flat.dtype)
self._stacked_w3 = torch.stack([ex.w3.weight for ex in self.experts], dim=0).to(device=flat.device, dtype=flat.dtype)
dispatch = self.moe_dispatch
if dispatch == "triton_grouped":
    try:
        y_routed = self._routed_forward_triton(flat, indices, weights)
    except (ImportError, ValueError) as exc:
        if not getattr(self, "_triton_fallback_warned", False):
            print(f"[moe] triton_grouped unavailable ({type(exc).__name__}: {exc}); "
                  f"falling back to 'stacked' for this model.")
            self._triton_fallback_warned = True
        y_routed = self._routed_forward_stacked(flat, indices, weights)
else:
    y_routed = self._routed_forward_stacked(flat, indices, weights)
y = y_routed
if self.shared_experts:
    y = y + self._shared_forward(flat)
return y.view(shape)
```

- Input `x`: `(…, dim)` (any leading batch/seq shape) → flattened to `(T, dim)`.
- Output: `(…, dim)`, shape restored via `view`.
- `_last_weights` / `_last_indices` are **detached snapshots** recorded on every
  forward, before dispatch — the balance metric and bias update consume these,
  never the live tensors (which carry gate gradients).
- Stacks are rebuilt unconditionally every forward, regardless of dispatch mode
  (the Triton path reads `_stacked_w*` too).
- Shared-expert output is added when `n_shared_experts > 0` (always true in both
  shipped configs); the `if self.shared_experts:` guard skips the branch for a
  hypothetical zero-shared config, in which case `y = y_routed`.

Who calls it: `models/transformer.py:TransformerBlock.forward` — `x = x + self.ffn(self.ffn_norm(x))` (pre-norm residual). The block constructs it at `models/transformer.py:TransformerBlock.__init__` as `DeepSeekMoE(config)` for `layer_id >= n_dense_layers` (2 dense + 16 MoE layers at canonical dims); `Transformer.moe_layers()` (`models/transformer.py:Transformer.moe_layers`) is the generator the training loop iterates.

### `_routed_forward_stacked(self, flat, indices, weights) -> torch.Tensor`

Anchor: `models/moe.py:DeepSeekMoE._routed_forward_stacked`. "Original per-expert
Python loop. Always available; used as both the default and the auto-fallback
for the Triton path."

- Inputs: `flat (T, dim)`, `indices (T, topk)` int64, `weights (T, topk)`.
- Sort-based gather: flatten routes → `argsort` by expert → per-expert contiguous
  chunks via `bincount` + `cumsum` offsets; empty experts are skipped.
- Per-expert compute: `expert_in @ _stacked_w1[e].t()` (gate), `@ _stacked_w3[e].t()`
  (up), `silu(gate)·up`, `@ _stacked_w2[e].t()` (down).
- Scatter: `y_routed.index_add(0, chunk_tokens, out * chunk_weights.unsqueeze(-1))`
  — tokens hitting the same expert *and position* accumulate, which is exact for
  top-k overlap.
- Returns `(T, dim)`. Gradient flows through the matmuls into the stacked weights
  (which alias the expert parameters) and through `chunk_weights` into the gate.

### `_routed_forward_triton(self, flat, indices, weights) -> torch.Tensor`

Anchor: `models/moe.py:DeepSeekMoE._routed_forward_triton`. Triton grouped-GEMM
SwiGLU path; same sort/scatter structure as the stacked loop, but the per-expert
loop becomes one fused kernel launch.

```python
from .moe_triton import triton_grouped_moe_dispatch
…
x_sorted = flat[sorted_token_ids].contiguous()
y_sorted = triton_grouped_moe_dispatch(
    x_sorted=x_sorted,
    w1=self._stacked_w1,
    w2=self._stacked_w2,
    w3=self._stacked_w3,
    sorted_weights=sorted_weights_1d,
    expert_offsets=expert_offsets,
)
y_routed = torch.zeros_like(flat)
y_routed.index_add_(0, sorted_token_ids, y_sorted)
```

- Same inputs/shapes as the stacked variant; returns `(T, dim)`.
- `weights`/`indices` carry grad to the gate — unlike the detached `_last_*`
  snapshots used only for the balance metric / bias update. The gate-weight
  multiply is applied **outside** the autograd `Function` in
  `models/moe_triton.py:triton_grouped_moe_dispatch` (`out * sorted_weights.unsqueeze(-1)`),
  precisely so the gate receives a gradient.
- Raises `ImportError` (triton not installed — CPU/Mac) or `ValueError`
  (`moe_inter_dim` or `dim` exceeds the 256-register kernel cap). Both are
  caught by `forward()` and fall back to the stacked path.
- **Register-cap limitation (measured repo fact):** at canonical dims
  (`dim=768`, `moe_inter_dim=384`) both `I` and `D` exceed 256, so
  `triton_grouped` **cannot run** the canonical config — it always raises
  `ValueError` and lands on the stacked fallback with the one-time warning.
  The kernel is only valid at smoke-config dims (`dim=256`, `moe_inter_dim=32`).
  See [R6 — Triton API](./R6_triton_api.md) and [Kernels & Ops](../concepts/kernels-and-ops.md) for the register budget
  and the latent `dh`-accumulation constraint in the `bwd_dw` kernel.

### `_shared_forward(self, flat: torch.Tensor) -> torch.Tensor`

Anchor: `models/moe.py:DeepSeekMoE._shared_forward`. "Batched shared-expert
forward. Stacks weights lazily so 1 bmm per SwiGLU projection."

- `flat (T, dim)` → `(T, dim)`; if `n_shared_experts == 0`, returns
  `torch.zeros_like(flat)` (no-op).
- Re-stacks `_shared_w{1,2,3}` **every forward** — same staleness rule as the
  routed path.
- Batches all shared experts in one `torch.bmm` per projection:
  `flat.unsqueeze(0).expand(E, -1, -1)` against the stacked transposed weights,
  then `silu(gate)·up`, down-project, `out.sum(dim=0)` — one row per token,
  summed over the `E` shared experts.
- Gradient flows into `shared_experts[*]` parameters via the stacks.

### `get_load_balance_loss(self) -> torch.Tensor`

Anchor: `models/moe.py:DeepSeekMoE.get_load_balance_loss`. Aux-loss-free balance
metric (DeepSeek-V3 §2.3.3), computed **without gradient** from the detached
last-forward snapshots:

```python
weights = self._last_weights; indices = self._last_indices
T = weights.size(0)
counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).float()
f = counts / counts.sum().clamp(min=1e-10)
one_hot = F.one_hot(indices.flatten(), num_classes=self.n_routed_experts).float()
P = (one_hot * weights.flatten().unsqueeze(-1)).view(T, -1, self.n_routed_experts).sum(dim=1).mean(dim=0)
return (f * P).sum() * self.n_routed_experts
```

- `f_i` = fraction of routed token-slots landing on expert $i$;
  `P_i` = mean normalized gate weight expert $i$ receives across tokens.
  $L_{\text{bal}} = E \sum_i f_i P_i \in [0, E]$; minimum at perfect balance
  ($f_i = P_i = 1/E$ for all $i$ → $L = 1$).
- Returns a 0-dim scalar tensor. If no forward has run yet (`_last_*` are
  `None`), returns `torch.tensor(0.0, device=self.gate.weight.device)`.
- **Metric only** — computed from `detach()`ed tensors, never added to the
  training loss (the aux-loss-free design replaces the auxiliary loss with the
  bias update; see [DeepSeekMoE & MTP](../concepts/moe-mtp.md) §load-balancing).

Who calls it: `training/pretrain.py:Pretrainer._moe_balance_metric` — sums over
all `moe_layers()`; `training/pretrain.py:Pretrainer.train_step` records it as
the `"balance_loss"` log key (`.item()` deferred to the logger to avoid a
per-step host sync). It is **not** part of `loss.backward()`.

### `update_gate_bias(self, speed: float = 0.001) -> None`

Anchor: `models/moe.py:DeepSeekMoE.update_gate_bias`.

```python
if self._last_indices is None:
    return
counts = torch.bincount(self._last_indices.flatten(), minlength=self.n_routed_experts)
self.gate.update_bias(counts, speed=speed)
```

- No-op until the first forward populated `_last_indices`.
- Counts stay on the model device (comment: boolean indexing in
  `update_bias` requires the mask and `self.bias` to share a device).
- Returns `None`; mutates `gate.bias` in place.

## 5. Dispatch modes & fallback

| Mode (`moe_dispatch`) | Path | Availability | Canonical config |
|---|---|---|---|
| `"stacked"` (default) | `_routed_forward_stacked` | always (pure PyTorch) | ✔ used |
| `"triton_grouped"` | `_routed_forward_triton` → `models/moe_triton.py:triton_grouped_moe_dispatch` | CUDA + triton installed **and** `dim, moe_inter_dim ≤ 256` | ✘ `ValueError` → one-shot warning → stacked |

Fallback contract: `forward()` catches `(ImportError, ValueError)` around the
Triton path, prints `[moe] triton_grouped unavailable (…); falling back to
'stacked' for this model.` exactly once (guarded by the
`_triton_fallback_warned` flag — set as a plain attribute, not in `__init__`),
and silently runs stacked for all subsequent calls. `moe_dispatch` itself is
never rewritten; the fallback is per-call, not a config mutation. Any other
value of `moe_dispatch` also runs stacked (the `else` branch).

## 6. Who calls what (call graph)

- `models/transformer.py:TransformerBlock.__init__` — instantiates
  `DeepSeekMoE(config)` for MoE layers.
- `models/transformer.py:TransformerBlock.forward` — pre-norm residual call
  `x = x + self.ffn(self.ffn_norm(x))`.
- `models/transformer.py:Transformer.moe_layers` — generator over MoE layers,
  consumed by the trainer.
- `training/pretrain.py:Pretrainer._update_moe_bias` — `moe.update_gate_bias(speed=self.config.bias_update_speed)` for every MoE layer.
- `training/pretrain.py:Pretrainer._moe_balance_metric` — `torch.stack(losses).sum()` over `get_load_balance_loss()`; `0.0` when no MoE layers.
- `training/pretrain.py:Pretrainer.train_step` — computes the balance metric
  inside the amp context; after `optimizer.step()` / `scheduler.step()` /
  `zero_grad(set_to_none=True)`, when `self._opt_steps % bias_update_every == 0`,
  calls `_update_moe_bias()`. **Timing contract:** the bias always moves *after*
  an optimizer step, based on counts from the forward that produced that step's
  gradients. Because the bias is a buffer (not a parameter), the optimizer never
  sees or decays it; weight decay and momentum apply only to `gate.weight`.
- `models/moe_triton.py:triton_grouped_moe_dispatch` — host wrapper invoked
  solely by `_routed_forward_triton`.

## 7. Pitfalls (terse)

1. **Caching `_stacked_w*` / `_shared_w*` across optimizer steps freezes the
   experts at init values** — the stacks are snapshots, not views; the
   re-stack-every-forward rule in `DeepSeekMoE.forward` / `_shared_forward` is
   load-bearing. Never hoist the stacks into `__init__` or cache them on a
   second forward.
2. **`gate.bias` is a buffer, not a `Parameter`** — it must not appear in
   `optimizer.param_groups`, is ignored by `clip_grad_norm_`, and must stay
   fp32 (BF16 would swallow the `±0.001` increments). Checkpoints round-trip it
   via `state_dict`.
3. **Top-k is over biased scores, weights come from unbiased scores** — the bias
   changes *which* experts are selected, never the *magnitude* of their weights.
4. **`triton_grouped` cannot run the canonical config** (`dim=768`,
   `moe_inter_dim=384` > 256-register cap) — expect the one-time fallback
   warning; the ≥1.5× kernel speedup target is structurally blocked at canonical
   dims. Treat any `moe_dispatch="triton_grouped"` canonical-config benchmark
   claim as suspect; see `./R6_triton_api.md`.
5. **`get_load_balance_loss()` is metric-only** — it is computed from detached
   snapshots and is never added to the loss; wiring it into `loss` would be a
   silent no-op on gradients (and a double-count of the bias update's job).
6. **`balance_loss` needs an on-device scalar** — `Pretrainer._moe_balance_metric`
   returns a stacked tensor; `.item()` is deferred to the logging path to avoid
   per-step host↔device sync stalls.
7. **Dtype discipline**: gate weights/`weights` end in `x.dtype` (BF16 under
   autocast) while `bias` stays fp32 and is cast per-forward with
   `bias.to(scores.dtype)` — keep the fp32 buffer untouched.
8. **Empty experts are legal** — the stacked loop `continue`s on `cnt == 0`;
   `torch.bincount(..., minlength=E)` keeps offsets aligned, so a zero-count
   expert never corrupts the offsets.
9. **`indices` are not sorted per token** — routing order is arbitrary; both
   dispatch paths re-sort by expert and scatter with `index_add`/`index_add_`
   (in-place for Triton), which is what makes token overlap correct.

## References
- Tutorial: [DeepSeekMoE & MTP](../concepts/moe-mtp.md) (gate math, balance derivation, Triton design),
  [Kernels & Ops](../concepts/kernels-and-ops.md) (kernel walkthrough), [Training Pipeline](../training.md)
  (bias-update timing in `train_step`).
- Sibling references: `./R2_transformer_api.md` (`TransformerBlock`, `SwiGLUFFN`,
  `Transformer.moe_layers`), `./R6_triton_api.md` (both kernels + dispatch
  contract), `./R7_training_api.md` (`TrainingConfig` keys incl.
  `bias_update_speed` / `bias_update_every`).
- Guides: `./../guides/G1_debugging_playbook.md` (shape errors, Triton fallback),
  `./../guides/G3_triton_development.md` (register budget), `./../guides/G4_benchmarking.md`.

