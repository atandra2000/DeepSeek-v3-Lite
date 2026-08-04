# R6 — Triton Kernel API Reference

Scope: the two opt-in fused kernels — fused MLA attention (`models/mla_triton.py`) and fused grouped-GEMM SwiGLU MoE dispatch (`models/moe_triton.py`) — plus the master environment-variable gate (`models/_triton_dispatch.py`) and the two callers that own fallback behavior (`models/mla.py`, `models/moe.py`).

Both kernels are **opt-in**: a config may request `attn_impl: "triton"` / `moe_dispatch: "triton_grouped"`, but unless `ENABLE_TRITON_KERNELS=1` is set, `models/_triton_dispatch.py:enforce_triton_env_var` rewrites those keys to their PyTorch defaults (`"sdpa"` / `"stacked"`) at model construction, with one warning. At runtime each caller additionally falls back if Triton is missing or a register-budget dim exceeds 256. This file is the API contract; the design walkthrough lives in `../12_Triton_Kernels.md`, and the MLA / MoE layer contracts in `./R3_mla_api.md` / `./R4_moe_api.md`.

> Honesty note: Triton is Linux + CUDA only. No GPU training run has ever executed in this repo (`.benchmarks/` is empty), so every performance figure elsewhere is an estimate — this reference makes no throughput claims.

## Module map

| Symbol | Kind | One-line purpose |
|---|---|---|
| `models/_triton_dispatch.py:_DISPATCH` | module dict | Single force-back table: Triton dispatch key → PyTorch default. |
| `models/_triton_dispatch.py:enforce_triton_env_var` | function | Master gate; mutates `model_cfg` in place unless `ENABLE_TRITON_KERNELS=1`. |
| `models/mla_triton.py:mla_attention_reference` | function | Pure-PyTorch MLA forward; CPU-test reference and backward re-compute. |
| `models/mla_triton.py:triton_mla_attention` | function | Public MLA kernel host wrapper; autograd via the internal `Function`; raises `ImportError` without Triton. |
| `models/mla_triton.py:_check_mla_dim_limits` | function | Hard-fail if `R`/`D_nope`/`D_rope`/`D_v` exceeds 256 (register budget). |
| `models/mla_triton.py:_next_pow2` | function | Ceiling power-of-two for block sizes (both triton modules have one). |
| `models/moe_triton.py:grouped_moe_pytorch` | function | Pure-PyTorch grouped SwiGLU forward over the sorted layout; CPU tests. |
| `models/moe_triton.py:triton_grouped_moe_dispatch` | function | Public MoE kernel host wrapper; gate-weight multiply applied outside the `Function`. |
| `models/moe_triton.py:_check_dim_limits` | function | Hard-fail if `I` or `D` exceeds 256 (register budget). |
| `models/mla.py:MultiHeadLatentAttention._forward_triton` | method | MLA caller: layout rearrange + `q_start` causality semantics. |
| `models/mla.py:MultiHeadLatentAttention.forward` | method | MLA dispatch point; catches kernel errors, falls back to SDPA. |
| `models/moe.py:DeepSeekMoE._routed_forward_triton` | method | MoE caller: sort/gather, expert offsets, scatter-back. |
| `models/moe.py:DeepSeekMoE.forward` | method | MoE dispatch point; catches kernel errors, falls back to stacked. |
| `models/transformer.py:Transformer.__init__` | method | Construction-time call site of `enforce_triton_env_var`. |

The Triton JIT kernels (`_mla_flash_fwd_kernel`, `_grouped_moe_fwd_kernel`, `_grouped_moe_bwd_dx_kernel`, `_grouped_moe_bwd_dw_kernel`) and the internal autograd `Function`s (`_TritonMlaAttentionFunction`, `_TritonGroupedMoeFunction`) are defined only under `if HAS_TRITON:` and are intentionally **not** anchorable symbols — cite the host wrappers above. They are described by bare name below.

## 1. The dispatch gate (`models/_triton_dispatch.py`)

### 1.1 `_DISPATCH`

Anchor: `models/_triton_dispatch.py:_DISPATCH`. One tuple-keyed table, verbatim:

```python
_DISPATCH = {
    ("attn_impl",    "triton"):         "sdpa",
    ("moe_dispatch", "triton_grouped"): "stacked",
}
```

Contract: any config that carries the Triton value for one of these `(key, triton_value)` pairs is force-backed to the paired PyTorch default when the master env var is unset. This replaced an earlier two-dict (`_TRITON_DISPATCH_KEYS` + `_PYTORCH_DEFAULTS`) arrangement that had to be kept in lockstep by a test; the table is the single source of truth now. `tests/test_force_back.py` locks every row.

### 1.2 `enforce_triton_env_var(model_cfg, log)`

Anchor: `models/_triton_dispatch.py:enforce_triton_env_var`.

```python
def enforce_triton_env_var(model_cfg: dict, log: Callable[[str], None]) -> None:
```

Behavior, in order:

1. `os.environ.get("ENABLE_TRITON_KERNELS", "0") == "1"` → **no-op** (opt-in granted; config left as-is).
2. Otherwise, iterate `_DISPATCH`; for every `(key, triton_val)` the config carries, rewrite `model_cfg[key] = pytorch_val` **in place** and record `"key='triton_val' -> 'pytorch_val'"`.
3. If anything was forced, call `log(...)` exactly once with a combined message: `"[warn] Triton dispatch keys set without ENABLE_TRITON_KERNELS=1; forcing attn_impl='triton' -> 'sdpa'. Set ENABLE_TRITON_KERNELS=1 to enable the fused Triton paths."` — never per-layer, never per-forward.

The `log` parameter is a plain callable (`print` in production, `list.append` in tests). Non-Triton keys and unrecognized values are never touched — e.g. `attn_impl: "manual"` passes through untouched (locked by `tests/test_force_back.py`).

**Who calls it:** `models/transformer.py:Transformer.__init__` at model construction (`enforce_triton_env_var(model_cfg, print)` after the `config.get("model", config)` unwrap). `training/pretrain.py:Pretrainer.__init__` documents the guard as already run ("this is a no-op" comment) — `Transformer` construction inside it does the real work, so any entry point that builds a `Transformer` is covered.

**State effects:** the rewrite persists in the config dict. Downstream readers (`MultiHeadLatentAttention.__init__` reading `attn_impl`, `DeepSeekMoE.__init__` reading `moe_dispatch`) therefore see the PyTorch default for the whole model lifetime.

## 2. Shared dtype / device contract

- **Device:** CUDA only. Triton cannot run on CPU/Mac; both modules set `HAS_TRITON = False` on import failure, and both public wrappers then raise `ImportError` with a message pointing to `attn_impl='sdpa'` / `moe_dispatch='stacked'`. The callers catch that (see §3.5 / §4.6), so a CPU/Mac run is fully functional on the PyTorch paths.
- **Input dtype:** BF16 at the canonical config (`dtype: bf16`); the kernels accept whatever dtype the tensors carry as long as operands match. Both fwd kernels require bf16·bf16 (or fp16·fp16) `tl.dot` operands.
- **Accumulators:** fp32 throughout — MLA online-softmax state (`m_i`, `l_i`, `acc`) and score tiles; MoE `gate_acc`/`up_acc` (SwiGLU contraction), the output GEMM, and every `dw*_local` gradient tile.
- **Output casting:** stores cast back to the destination pointer's element type (`acc.to(out_ptr.dtype.element_ty)`, `out_acc.to(y_ptr.dtype.element_ty)`, `dx.to(dx_ptr.dtype.element_ty)`). MoE `dw` buffers are allocated fp32 and cast to `w.dtype` only when returned from `backward`.
- **Non-contiguity:** strides are passed explicitly for every tensor; only `models/mla.py:MultiHeadLatentAttention._forward_triton` and `models/moe.py:DeepSeekMoE._routed_forward_triton` call `.contiguous()` at their gather/permute seams.

## 3. Fused MLA attention (`models/mla_triton.py`)

### 3.1 `mla_attention_reference`

Anchor: `models/mla_triton.py:mla_attention_reference`.

```python
def mla_attention_reference(
    q_nope: torch.Tensor,        # (B, H, S_q, D_nope)
    q_pe: torch.Tensor,          # (B, H, S_q, D_rope)
    ctx_kv: torch.Tensor,        # (B, S_kv, R)
    ctx_pe: torch.Tensor,        # (B, S_kv, D_rope)
    wkv_b_k: torch.Tensor,       # (H, D_nope, R)
    wkv_b_v: torch.Tensor,       # (H, D_v, R)
    softmax_scale: float,
    is_causal: bool = False,
    q_start: int = 0,
) -> torch.Tensor:  # (B, S_q, H, D_v)
```

Pure-PyTorch, self-contained (no KV cache, no pre-applied RoPE — `q_pe`/`ctx_pe` are already rotated). Materializes `K_nope = einsum("bsr,hdr->bhsd", ctx_kv, wkv_b_k)` and `V = einsum("bsr,hdr->bhsd", ctx_kv, wkv_b_v)`, concatenates with the rope halves, computes `scores = einsum("bhqd,bhkd->bhqk", Q, K) * softmax_scale`, softmaxes over `k`, and returns `(B, S_q, H, D_v)`.

Causality is global-position based: when `is_causal`, query position `q_start + q_idx` is masked against `k_idx` (so a cached mid-sequence prefill stays causal). When `not is_causal` but `S_q == S_kv`, a `triu(diagonal=1)` of `-inf` is applied instead. Note the kernel's causal rule (§3.2) always uses the `q_start` form — the `S_q == S_kv` shortcut exists only in the reference.

**Who calls it:** `tests/test_mla_triton.py` (CPU-testable ground truth incl. the causal `q_start` contract) and `_TritonMlaAttentionFunction.backward` as the re-compute forward (§3.3).

### 3.2 `triton_mla_attention`

Anchor: `models/mla_triton.py:triton_mla_attention`.

```python
def triton_mla_attention(
    q_nope: torch.Tensor,        # (B, H, S_q, D_nope)
    q_pe: torch.Tensor,          # (B, H, S_q, D_rope)
    ctx_kv: torch.Tensor,        # (B, S_kv, R)
    ctx_pe: torch.Tensor,        # (B, S_kv, D_rope)
    wkv_b_k: torch.Tensor,       # (H, D_nope, R)
    wkv_b_v: torch.Tensor,       # (H, D_v, R)
    softmax_scale: float,
    is_causal: bool = False,
    q_start: int = 0,
) -> torch.Tensor:  # (B, S_q, H, D_v)
```

Raises `ImportError` immediately if `HAS_TRITON` is false; otherwise `_TritonMlaAttentionFunction.apply(q_nope, q_pe, ctx_kv, ctx_pe, wkv_b_k, wkv_b_v, softmax_scale, is_causal, q_start)`. The fused kernel materializes `K_nope`/`V` inside the inner KV-block loop (per-head `w_k`/`w_v` tiles held in registers), applies RoPE-free `q_pe·k_pe` scoring, FA2 online softmax (`m_i`, `l_i`, `acc` in fp32), and the causal test `s_q_off + q_start >= k_off` when `is_causal` (a `tl.constexpr`), plus the `k_mask` padding mask. Softmax scale is applied as `(dot_nope + dot_rope) * softmax_scale`.

Launch contract (inside `_TritonMlaAttentionFunction.forward`):

| Parameter | Value | Notes |
|---|---|---|
| Grid | `(B, H, ceil(S_q / 64))` | one program per (batch, head, query-block) |
| `BLOCK_Q` | 64 | tiles `S_q` — unbounded |
| `BLOCK_N` | 64 | tiles `S_kv` — unbounded |
| `BLOCK_R` | `_next_pow2(R)` | canonical 256 (R = 192) |
| `BLOCK_D_NOPE` | `_next_pow2(D_nope)` | canonical 64 (48) |
| `BLOCK_D_ROPE` | `_next_pow2(D_rope)` | canonical 32 (24) |
| `BLOCK_D_V` | `_next_pow2(D_v)` | canonical 64 (64) |
| `num_warps` / `num_stages` | 4 / 2 | fixed; no autotune |

Register caps (anchor: `models/mla_triton.py:_check_mla_dim_limits`): `R`, `D_nope`, `D_rope`, `D_v` must each be ≤ 256 or the function raises `ValueError` before launch; `S_q`/`S_kv` are tiled and need no cap. **Canonical dims all fit** (192/48/24/64 ≤ 256), so `attn_impl="triton"` is launchable at the canonical config.

### 3.3 Autograd contract (`_TritonMlaAttentionFunction`)

Bare-name symbol; not anchorable. Forward takes 9 arguments (6 tensors + `softmax_scale`, `is_causal`, `q_start`) and:

- runs `_check_mla_dim_limits`, picks blocks via `models/mla_triton.py:_next_pow2`, allocates `out = torch.empty(B, S_q, H, D_v, ...)`, launches `_mla_flash_fwd_kernel[(B, H, n_q_blocks)]` with explicit strides;
- saves for backward: `q_nope, q_pe, ctx_kv, ctx_pe, wkv_b_k, wkv_b_v` (`ctx.save_for_backward`) plus scalars `softmax_scale`, `is_causal`, `q_start` and the six shape constants.

Backward is a **v1 re-compute stub**: it re-runs `mla_attention_reference` on the saved tensors and lets `torch.autograd.grad` produce the grads (`allow_unused=True`). Correct (same math as the kernel), but not a fused backward; the docstring flags it. Return tuple is exactly 9 entries, in forward order: the 6 tensor grads, each `None`-guarded with `torch.zeros_like` (a `None` grad for a non-`None` input is an autograd error), then `None, None, None` for the three scalars. Get the count wrong and the first training backward dies with "function returned an invalid number of gradient tensors".

### 3.4 Caller: `MultiHeadLatentAttention._forward_triton`

Anchor: `models/mla.py:MultiHeadLatentAttention._forward_triton`.

```python
def _forward_triton(self, q_nope, q_pe, ctx_kv, ctx_pe, wkv_b_k, wkv_b_v,
                    bsz: int, seqlen: int, h: int,
                    mask: Optional[torch.Tensor], start_pos: int, use_cache: bool) -> torch.Tensor:
```

Imports `triton_mla_attention` lazily (import inside the method — that is the `ImportError` source the caller catches). Responsibilities:

1. **Layout:** permutes `q_nope`/`q_pe` from layer layout `(B, S_q, H, ·)` to kernel layout `(B, H, S_q, ·)` via `.permute(0, 2, 1, 3).contiguous()`.
2. **Causality:** `is_causal = mask is not None` (the layer only receives a mask when causal); `q_start = start_pos if (is_causal and use_cache) else 0` — the cached-chunk-prefill offset; `0` for cache-free prefill.
3. **Scale:** passes `self.softmax_scale` (canonical: `qk_head_dim ** -0.5 = 72 ** -0.5`, `mscale = 1.0` since `max_seq_len = 2048 ≤ 4096` and `rope_factor = 1.0`).
4. **Output:** flattens `(B, S_q, H, D_v)` to `(B, S_q, H*D_v)` and applies `self.wo`.

### 3.5 Fallback: `MultiHeadLatentAttention.forward`

Anchor: `models/mla.py:MultiHeadLatentAttention.forward`. The dispatch point wraps `self._forward_triton(...)` in:

```python
try:
    return self._forward_triton(...)
except (ImportError, ValueError) as exc:
    if not getattr(self, "_triton_fallback_warned", False):
        print(f"[mla] triton attn_impl unavailable ({type(exc).__name__}: {exc}); "
              f"falling back to 'sdpa' for this model.")
        self._triton_fallback_warned = True
    self.attn_impl = "sdpa"
```

Fallback semantics: catches `ImportError` (no Triton) and `ValueError` (register cap); warns **once per model instance** (sticky `_triton_fallback_warned` flag); and **permanently rewrites `self.attn_impl = "sdpa"`** — every later forward takes the SDPA branch (`F.scaled_dot_product_attention` over the fused `K_nope`/`V` bmm) without retrying the kernel. The SDPA branch is not just a CPU path — it is the intended CUDA fallback too.

## 4. Fused grouped-GEMM MoE dispatch (`models/moe_triton.py`)

### 4.1 `grouped_moe_pytorch`

Anchor: `models/moe_triton.py:grouped_moe_pytorch`.

```python
def grouped_moe_pytorch(
    x_sorted: torch.Tensor,         # (T, D)
    w1: torch.Tensor,               # (E, I, D)
    w2: torch.Tensor,               # (E, D, I)
    w3: torch.Tensor,               # (E, I, D)
    expert_offsets: torch.Tensor,   # (E+1,) INT64 cumsum boundaries
    sorted_weights: torch.Tensor,   # (T,)
) -> torch.Tensor:  # y_sorted (T, D)
```

Pure-PyTorch grouped GEMM: same arithmetic as the `stacked` path but routed through the sorted-token layout — for each expert `e`, slice `x_sorted[offsets[e]:offsets[e+1]]`, compute `gate = chunk @ w1[e].t()`, `up = chunk @ w3[e].t()`, `h = silu(gate) * up`, `out = h @ w2[e].t()`, scale by `sorted_weights`. The gate-weight multiply is **inside** the reference. Used by `tests/test_moe_triton.py` as the CPU ground truth.

### 4.2 `triton_grouped_moe_dispatch`

Anchor: `models/moe_triton.py:triton_grouped_moe_dispatch`.

```python
def triton_grouped_moe_dispatch(
    x_sorted: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    sorted_weights: torch.Tensor,
    expert_offsets: torch.Tensor,
) -> torch.Tensor:  # y_sorted (T, D)
```

Raises `ImportError` if `HAS_TRITON` is false; `ValueError` if `_check_dim_limits(I, D)` fails. Contract differences from the reference:

```python
out = _TritonGroupedMoeFunction.apply(x_sorted, w1, w2, w3, expert_offsets)
return out * sorted_weights.unsqueeze(-1)
```

The kernel returns the **unweighted** expert output; the `sorted_weights` multiply happens **outside** the autograd `Function` so the routing gate receives a gradient. Returns `y_sorted (T, D)`.

Launch contract (inside `_TritonGroupedMoeFunction.forward`):

| Parameter | Value | Notes |
|---|---|---|
| Grid (fwd) | `(E, ceil(T / 32))` | one program per (expert, token-block) |
| `BLOCK_T` | 32 | tiles the token dim — unbounded |
| `BLOCK_D` | `_next_pow2(D)` | canonical 1024 → **illegal** (see §4.3) |
| `BLOCK_I` | `_next_pow2(I)` | canonical 512 → **illegal** |
| `num_warps` / `num_stages` | 4 / 2 | fixed |

The fwd kernel loops `d_start` over `BLOCK_D` chunks accumulating `gate_acc`/`up_acc` in fp32 (`tl.dot(..., acc=...)`), computes `h = sigmoid(gate_acc) * gate_acc * up_acc`, casts to the input dtype, and finishes with a single full-`I` GEMM. `BLOCK_I` is documented as "full I" — the out-GEMM is not `I`-tiled.

### 4.3 Register cap: `_check_dim_limits` and the canonical config

Anchor: `models/moe_triton.py:_check_dim_limits`.

```python
def _check_dim_limits(I: int, D: int) -> None:
    if I > 256 or D > 256:
        raise ValueError(
            f"triton_grouped_moe_dispatch: BLOCK_I=ceil_pow2({I}) and "
            f"BLOCK_D=ceil_pow2({D}) must each be ≤ 256. Got I={I}, D={D}. "
            "For larger dims, fall back to `moe_dispatch='stacked'`."
        )
```

**Measured consequence:** the canonical config (`moe_inter_dim = 384`, `dim = 768`) exceeds the cap on both axes, so `triton_grouped_moe_dispatch` always raises `ValueError` at canonical dims and `DeepSeekMoE.forward` falls back to stacked. This is the *expected* behavior at the canonical config — a structural limit, not a bug. The smoke config (`moe_inter_dim = 32`, `dim = 256`) fits. Contrast with MLA, whose canonical dims (192/48/24/64) all fit. Both `_check_*` functions share the same root cause: a CUDA thread has 255 registers; with `num_warps=4` a `(32, 256)` fp32 accumulator already costs 64 registers/thread and the MoE fwd kernel needs two. See `../guides/G3_triton_development.md` for the budget math.

### 4.4 Autograd contract (`_TritonGroupedMoeFunction`)

Bare-name symbol; not anchorable. Forward takes 5 tensors, runs `_check_dim_limits`, allocates `y_sorted (T, D)`, launches `_grouped_moe_fwd_kernel[grid]`. Saves `x_sorted, w1, w2, w3, expert_offsets` plus `T, D, I, E, BLOCK_T, BLOCK_D, BLOCK_I`.

Backward (fully fused, unlike MLA's re-compute stub) launches two kernels:

- `_grouped_moe_bwd_dx_kernel[(E, ceil(T/32))]` — re-computes `gate_acc`/`up_acc` from saved tensors, then `dh = dy@w2`, `dsilu = dh*up`, `dgate_pre = dsilu·silu'(g)` with `silu'(g) = sig(g)(1 + g(1 - sig(g)))`, `dup = dsilu·silu(g)`, `dx = dgate_pre@w1 + dup@w3`. Note `dy` arrives already gate-scaled (the outside multiply), so no weight load is needed here.
- `_grouped_moe_bwd_dw_kernel[(E,)]` — one program per expert, **no atomics**; each program owns its `(I, D)` tiles and loops `i_start`/`d_start`/`t_start`, accumulating `dw1_local`/`dw3_local` (fp32 `(BLOCK_I, BLOCK_D)`) and `dw2_local` (fp32 `(BLOCK_D, BLOCK_I)`), then stores directly.

`dw1/dw2/dw3` are allocated fp32 and cast to `w.dtype` in the return tuple: `(dx, dw1.to(w1.dtype), dw2.to(w2.dtype), dw3.to(w3.dtype), None)` — 5 entries for 5 forward inputs. The `to()` casts make the gradient buffers match the optimizer's parameter dtype (bf16) while the contraction itself stays fp32.

Latent constraint (honest note): in `_grouped_moe_bwd_dw_kernel`, `dh = tl.dot(dy_tile, w2_tile)` does **not** accumulate across `D`-blocks (`dh` is recomputed per tile, never `acc=`-chained). At legal dims the `d_start` loop is single-iteration so this is invisible; it would become a correctness bug if `D`-tiling were ever added. Currently unreachable because the 256 cap makes `BLOCK_D ≥ D` always single-iteration.

### 4.5 Caller: `DeepSeekMoE._routed_forward_triton`

Anchor: `models/moe.py:DeepSeekMoE._routed_forward_triton`.

```python
def _routed_forward_triton(self, flat: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
```

`flat (T, dim)`; `indices`/`weights` come straight from `self.gate(flat)` and **carry grad to the gate** (unlike the detached `_last_*` snapshots used only for the balance metric / bias update). Steps:

1. `flat_idx = indices.reshape(-1)`, `flat_w = weights.reshape(-1)`; `order = argsort(flat_idx)`; `sorted_token_ids = token_id[order]`; `sorted_weights_1d = flat_w[order]`.
2. `expert_counts = bincount(flat_idx, minlength=E)`; `expert_offsets = cat([zeros(1), cumsum(0)[:-1]])` — the `(E+1,)` INT64 cumsum boundaries.
3. `x_sorted = flat[sorted_token_ids].contiguous()` — differentiable gather.
4. Call `triton_grouped_moe_dispatch(x_sorted, w1=self._stacked_w1, w2=self._stacked_w2, w3=self._stacked_w3, sorted_weights=sorted_weights_1d, expert_offsets=expert_offsets)`. The `_stacked_*` tensors are re-stacked every forward from `expert.w{1,2,3}.weight` (same staleness rationale as the stacked path) and cast to `flat.dtype`/device — BF16 weights passed straight to the kernel since the backward casts `dw` back to `w.dtype`.
5. `y_routed = torch.zeros_like(flat)`; `y_routed.index_add_(0, sorted_token_ids, y_sorted)` — in-place scatter back to original token positions (differentiable; gradients route back through the gather and the outside multiply).

### 4.6 Fallback: `DeepSeekMoE.forward`

Anchor: `models/moe.py:DeepSeekMoE.forward`. Dispatch point:

```python
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
```

**Asymmetry vs MLA:** the MoE fallback is per-call — `self.moe_dispatch` is **not** rewritten, so a later forward retries the kernel (only the warning is one-time). The MLA fallback rewrites `self.attn_impl = "sdpa"` permanently. Consequence at canonical dims: every MoE forward pays the sort/gather/offset overhead once, then hits the `ValueError` and re-runs the stacked loop — retry cost is real but small relative to the FFN compute; `models/mla.py`'s sticky rewrite avoids any analogous retry.

## 5. Register budget and tiling summary

| Kernel | Dim | Tiled? | Block | Cap |
|---|---|---|---|---|
| MLA | `S_q` | yes | `BLOCK_Q = 64` | none (grid axis 2) |
| MLA | `S_kv` | yes | `BLOCK_N = 64` | none (loop) |
| MLA | `R` | no | `next_pow2(R)` = 256 | ≤ 256 |
| MLA | `D_nope` / `D_rope` / `D_v` | no | 64 / 32 / 64 | ≤ 256 each |
| MoE | `T` | yes | `BLOCK_T = 32` | none (grid axis 1) |
| MoE | `I` | no (fwd out-GEMM single-shot) | `next_pow2(I)` | ≤ 256 |
| MoE | `D` | loop structure exists (fwd/`bwd_dx` `acc=`; `bwd_dw` plain) | `next_pow2(D)` | ≤ 256 |

`models/mla_triton.py:_next_pow2` and `models/moe_triton.py:_next_pow2` are the shared block-sizing helpers (`1 << (n-1).bit_length()` for `n > 1`). Neither kernel is autotuned — blocks, `num_warps=4`, `num_stages=2` are compile-time constants chosen at the launch site; `../guides/G3_triton_development.md` shows the autotune pattern the kernels deliberately don't use.

## 6. Path selection matrix

| Config requests | `ENABLE_TRITON_KERNELS` | Dims fit? | Actual path |
|---|---|---|---|
| `attn_impl: "triton"` | unset | — | construction force-back → `"sdpa"` (one warning) |
| `attn_impl: "triton"` | `1` | canonical (all ≤ 256) | fused kernel (§3) |
| `attn_impl: "triton"` | `1` | any dim > 256 | `ValueError` → SDPA, sticky `attn_impl="sdpa"` |
| `moe_dispatch: "triton_grouped"` | unset | — | construction force-back → `"stacked"` (one warning) |
| `moe_dispatch: "triton_grouped"` | `1` | smoke (32, 256) | fused kernel (§4) |
| `moe_dispatch: "triton_grouped"` | `1` | canonical (384, 768) | `ValueError` per forward → stacked fallback (expected) |
| Triton keys on CPU/Mac | either | — | `ImportError` → PyTorch path (MLA sticky, MoE per-call) |

Canonical config ships `attn_impl: "sdpa"` + `moe_dispatch: "stacked"` (`configs/pretrain_a100_422m.yaml`); Triton keys are the opt-in override.

## 7. Pitfalls (terse)

1. **Config mutation is in place and permanent.** `enforce_triton_env_var` rewrites the dict the caller passed; tests that share a config object across cases must copy it (see `tests/test_force_back.py` env-var restore pattern).
2. **`ENABLE_TRITON_KERNELS` must be exactly `"1"`** — any other value (including `"0"`, unset, or `"true"`) means force-back.
3. **Warning-once flags are per-model-instance attributes**, not module state: `_triton_fallback_warned` on the MLA layer and on the MoE module. Reconstructing a model re-arms the warning.
4. **Fallback asymmetry:** MLA retires the kernel permanently (`attn_impl = "sdpa"`); MoE retries every forward and only silences the warning. Do not rely on the MoE path "sticking" — check `moe_dispatch` if you expect silence.
5. **Canonical-config MoE Triton is structurally impossible** — `moe_inter_dim=384` and `dim=768` both exceed 256; do not file the `ValueError` as a bug. The ≥1.5× MoE speedup benchmark target is blocked at canonical dims. MLA Triton is fine at canonical dims.
6. **MLA backward is a re-run, not a fused kernel** — memory traffic ≈ forward; correctness identical to `mla_attention_reference` by construction. The MoE backward *is* fused (`bwd_dx` + `bwd_dw`).
7. **`q_start` must equal the query block's global start position** for cached chunked prefill; passing `0` with a non-empty cache silently re-derives a wrong (too-lenient) causal mask. `models/mla.py:MultiHeadLatentAttention._forward_triton` computes it correctly; callers that bypass it must replicate `start_pos if (is_causal and use_cache) else 0`.
8. **Autograd tuple arity is load-bearing** — 9 entries for MLA (6 grads + 3 `None`), 5 for MoE. Miss one and the first backward raises.
9. **`bwd_dw` `dh` does not accumulate across `D`-blocks** — latent defect, unreachable while the 256 cap holds; revisit before any D-tiling change.
10. **Kernels JIT-compile on first launch** (seconds of wall time); there is no `triton.compile` pre-warm in this repo. Combined with no GPU run ever executed, all kernel timing claims are estimates.

## 8. Cross-links

- Tutorials: `../12_Triton_Kernels.md` (kernel-by-kernel design, register budget, autotune), `../03_Multi_Head_Latent_Attention.md` (MLA math, `q_start` cache lifecycle), `../04_DeepSeekMoE.md` (gate math, stacked-vs-grouped layouts, canonical-cap framing).
- Sibling references: `./R3_mla_api.md` (layer contract: `softmax_scale`, `mscale`, cache), `./R4_moe_api.md` (gate/stacked path, `_stacked_w*` contract), `./R2_transformer_api.md` (construction-time guard, mask building), `./R7_training_api.md` (training entry point, `Pretrainer.__init__`).
- Guides: `./../guides/G1_debugging_playbook.md` (Triton-fallback symptoms), `./../guides/G3_triton_development.md` (extending kernels, register math), `./../guides/G4_benchmarking.md` (what is/isn't measured).

<!-- docs:verified 2026-08-04 · 59aeef3 -->
