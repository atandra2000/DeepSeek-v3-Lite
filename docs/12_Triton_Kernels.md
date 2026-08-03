# 12 — Custom Triton Hardware Kernels

> **Canonical** for DeepSeek-v3-Lite's real fused Triton kernels: fused MLA attention and fused grouped-GEMM MoE dispatch. Educational textbook chapter with full from-scratch explanations.

> How two custom Triton kernels turn the PyTorch reference paths into single-launch fused GPU programs, and the double-opt-in guard that keeps the CPU test suite green. **Both kernels are BF16** — there is no FP8 GEMM in this repo (FP8 is paper-spec, see [[Docs/06_FP8_Mixed_Precision]]).

**Depends on:** [[Docs/03_Multi_Head_Latent_Attention]], [[Docs/04_DeepSeekMoE]], [[Docs/06_FP8_Mixed_Precision]] · **Read next:** [[Docs/13_Portfolio_Comparison]]

**Source files:** `models/mla_triton.py`, `models/moe_triton.py`, `models/_triton_dispatch.py`

---

## 0. Status in this repo

| Kernel | File | Status | Precision |
|---|---|---|---|
| Fused MLA attention (FA2-style) | `models/mla_triton.py` | ✅ implemented, opt-in | BF16 |
| Fused grouped-GEMM MoE SwiGLU dispatch | `models/moe_triton.py` | ✅ implemented, opt-in | BF16 |
| Triton force-back guard | `models/_triton_dispatch.py` | ✅ always-on | — |
| FP8 GEMM | — | ❌ not implemented (paper-spec) | — |

The kernels are **opt-in**: a config may request `attn_impl: triton` or `moe_dispatch: triton_grouped`, but unless the environment variable `ENABLE_TRITON_KERNELS=1` is set, `_triton_dispatch.enforce_triton_env_var` silently rewrites those keys to their PyTorch defaults (`sdpa`, `stacked`) with a single warning. This is what keeps the 189-test CPU suite green — Triton is Linux+CUDA-only, and the test box has neither.

---

## 1. Why custom kernels at all?

The PyTorch reference paths (`mla.py` SDPA path, `moe.py` stacked path) are correct and readable, but they make tensor round-trips through HBM that a fused kernel can avoid:

- **MLA:** the SDPA path materialises full per-head `K_nope` and `V` by a batched matmul of the compressed latent `ctx_kv` with `wkv_b`, writes them to HBM, then immediately reads them back for attention. The fused kernel keeps `wkv_b_k` / `wkv_b_v` in registers and produces `K_nope`/`V` **inside** the K-block loop, so they never touch HBM.
- **MoE:** the stacked path is a Python `for e in range(E)` loop, one matmul per expert per weight, with a sort + `index_add` scatter. The fused grouped-GEMM kernel launches once over the sorted-token layout and does all experts' SwiGLU in one kernel.

The payoff is fewer kernel launches and less HBM traffic — the two things that dominate latency at these shapes. The cost is a 256-element register-budget cap and a backward pass that (for MLA) is currently a correctness-preserving stub.

---

## 2. The double-opt-in guard (`models/_triton_dispatch.py`)

Before any kernel runs, `Transformer.__init__` calls `enforce_triton_env_var(model_cfg, print)`. The guard is a one-table dispatch:

```python
_DISPATCH = {
    ("attn_impl",    "triton"):         "sdpa",
    ("moe_dispatch", "triton_grouped"): "stacked",
}

def enforce_triton_env_var(model_cfg, log):
    if os.environ.get("ENABLE_TRITON_KERNELS", "0") == "1":
        return                       # opt-in granted: leave config as-is
    forced = []
    for (key, triton_val), pytorch_val in _DISPATCH.items():
        if model_cfg.get(key) == triton_val:
            model_cfg[key] = pytorch_val
            forced.append(f"{key}='{triton_val}' -> '{pytorch_val}'")
    if forced:
        log("[warn] Triton dispatch keys set without ENABLE_TRITON_KERNELS=1; "
            f"forcing {', '.join(forced)}.")
```

`★ Insight ─────────────────────────────────────`
This guard exists because of a workspace rule (AGENTS #7): *a default-config run must never silently switch to a Triton path.* Without it, a config checked in with `attn_impl: triton` would either crash (no Triton on Mac) or silently take a different code path than the CPU tests verified — two failure modes that are painful to debug because they only appear on certain machines. Forcing back to the PyTorch default unless the env var is explicitly set makes "same config, same code path" hold everywhere by default. The single source-of-truth `_DISPATCH` table replaced an earlier pair of dicts enforced in lockstep by a dedicated test (`tests/test_force_back.py`, 8 cases).
`─────────────────────────────────────────────────`

There is a **second** layer of defense at the call site: even if the guard is bypassed (env var set but Triton missing at runtime), `MultiHeadLatentAttention.forward` and `DeepSeekMoE.forward` catch `ImportError`/`ValueError` from the kernel import and fall back to the PyTorch path with a one-shot warning (`_triton_fallback_warned`). So there are two independent safety nets: config-time force-back and runtime fallback.

---

## 3. Fused MLA Attention Kernel (`models/mla_triton.py`)

### 3.1 The reference path — what the kernel replaces

`mla_attention_reference` is the pure-PyTorch path used by CPU tests. It is the clearest statement of the MLA arithmetic:

```python
# ctx_kv: (B, S_kv, R=192)  — the compressed latent, cached
# wkv_b_k: (H, D_nope=48, R)  wkv_b_v: (H, D_v=64, R)  — per-head up-projections
K_nope = torch.einsum("bsr,hdr->bhsd", ctx_kv, wkv_b_k)   # materialise K_nope
V      = torch.einsum("bsr,hdr->bhsd", ctx_kv, wkv_b_v)   # materialise V
K_rope = ctx_pe.unsqueeze(1)                              # decoupled RoPE key (shared across heads)
Q_full = torch.cat([q_nope, q_pe], dim=-1)
K_full = torch.cat([K_nope, K_rope.expand(B, H, S_kv, D_rope)], dim=-1)
scores = torch.einsum("bhqd,bhkd->bhqk", Q_full, K_full) * softmax_scale
# + causal mask, softmax, @ V ...
```

The two `einsum`s are the HBM round-trip the kernel eliminates: `K_nope` and `V` are computed, written to HBM, then read back for the score and output matmuls. At `R=192`, `D_nope=48`, `D_v=64`, `H=12`, `S=2048` that is `B·H·S·(D_nope+D_v)·2 ≈ 2.3 GB` of avoidable traffic per layer.

### 3.2 The fused kernel — FlashAttention-2 style

`_mla_flash_fwd_kernel` is one program per `(batch, head, query-block)`. The structure is FA2's online softmax:

```
load q_nope_tile, q_pe_tile           # (BLOCK_Q, D_nope) and (BLOCK_Q, D_rope)
load w_k, w_v into registers           # (D_nope, R) and (D_v, R) — per-head, stay in registers
m_i = -inf, l_i = 0, acc = 0          # online-softmax accumulators
for kv_start in range(0, S_kv, BLOCK_N):
    kv_tile = load ctx_kv[kv_block]    # (BLOCK_N, R) — the latent, from HBM
    k_nope = dot(kv_tile, w_k.T)       # materialise K_nope IN REGISTERS (no HBM write)
    v_tile = dot(kv_tile, w_v.T)       # materialise V IN REGISTERS
    k_pe  = load ctx_pe[kv_block]       # (BLOCK_N, D_rope) — RoPE key, from HBM
    s_block = (dot(q_nope_tile, k_nope.T) + dot(q_pe_tile, k_pe.T)) * softmax_scale
    # online softmax update (m_i, l_i, acc) with rescale
m_i, l_i, acc updated; acc /= l_i
store out[q_block]
```

The key move: `wkv_b_k` and `wkv_b_v` are loaded **once per program** and held in registers across the entire K-block loop. `K_nope` and `V` are produced by `tl.dot(kv_tile, tl.trans(w_k))` **inside** the loop — they live in registers, never in HBM. The only HBM traffic for keys/values is the compressed `ctx_kv` (R=192 floats/token) and the small `ctx_pe` (D_rope=24 floats/token), which is exactly the MLA cache savings made into a kernel savings.

`★ Insight ─────────────────────────────────────`
The fused kernel is the *inference-time* form of the matrix-absorption trick (see [[Docs/03_Multi_Head_Latent_Attention]]). At inference, MLA absorbs the K/V up-projection into the query so attention runs directly on the latent. The kernel does the complementary thing for training (where you need gradients to flow to `wkv_b`): it keeps `wkv_b` resident and re-materialises `K_nope`/`V` per K-block, so the forward never writes them to HBM but the backward still has the graph it needs. Same savings, different mechanism.
`─────────────────────────────────────────────────`

### 3.3 The register-budget cap

`_check_mla_dim_limits` hard-fails if `R`, `D_nope`, `D_rope`, or `D_v` exceeds **256**:

```python
for name, val in [("R", R), ("D_nope", D_nope), ("D_rope", D_rope), ("D_v", D_v)]:
    if val > 256:
        raise ValueError(f"triton_mla_attention: {name}={val} exceeds the 256 cap. ...")
```

The canonical 422 M config (`R=192, D_nope=48, D_rope=24, D_v=64`) is well under the cap. The cap exists because `w_k`/`w_v` and the score tile all live in registers simultaneously — at `R=192, D_nope=48` the `w_k` tile alone is `48×192` floats, and the BLOCK_Q×BLOCK_N score tile adds more. 256 is the empirical register-budget ceiling before spills dominate. This is also why `moe_dispatch='triton_grouped'` auto-falls-back to `stacked` when `moe_inter_dim > 256` (the gate in `DeepSeekMoE.forward`).

### 3.4 The backward pass — a deliberate stub

The forward is a real fused kernel; the **backward is a stub** that re-runs the reference forward and lets PyTorch autograd compute gradients:

```python
@staticmethod
def backward(ctx, *grad_outputs):
    dout = grad_outputs[0]
    q_nope, q_pe, ctx_kv, ctx_pe, wkv_b_k, wkv_b_v = ctx.saved_tensors
    out_ref = mla_attention_reference(...)          # recompute reference
    grads = torch.autograd.grad(out_ref, [...], grad_outputs=dout, allow_unused=True)
    return (g if g is not None else torch.zeros_like(t) for t in [...])  # + None, None
```

This is **correct but not optimal**: the recompute is the PyTorch reference, so gradients match the SDPA path exactly (asserted by `tests/test_mla_triton.py`), but it forfeits the fused-kernel speedup in the backward. The comment marks it as a v1 stub with the upgrade path (a fused recompute-backward) documented. For training throughput this is the obvious next optimization; for correctness it is a no-op.

### 3.5 Autograd wiring — `torch.autograd.Function`

`_TritonMlaAttentionFunction.apply(...)` is the seam between the kernel and PyTorch autograd. Forward saves `q_nope, q_pe, ctx_kv, ctx_pe, wkv_b_k, wkv_b_v` for backward (`ctx.save_for_backward`) plus the scalar shapes/scale. The public entry `triton_mla_attention(...)` raises `ImportError` with a helpful message if `triton` is not installed — caught by `mla.py`'s try/except to trigger the SDPA fallback.

---

## 4. Fused Grouped-GEMM MoE Kernel (`models/moe_triton.py`)

### 4.1 The reference path — the Python loop the kernel replaces

`DeepSeekMoE._routed_forward_stacked` (the default `stacked` path) is, after the gate and a sort-by-expert-id:

```python
# tokens sorted so each expert's tokens are contiguous
for e in range(E):                       # E = 20 routed experts
    chunk = x_sorted[start:end]           # this expert's tokens
    h = silu(chunk @ w1[e].T) * (chunk @ w3[e].T)   # SwiGLU: 2 GEMMs
    out = h @ w2[e].T                              # 1 GEMM
    y_sorted[start:end] = out * weights[start:end]
```

That is **3 GEMMs × 20 experts = 60 small GEMM launches per MoE layer per step**, each launch paying the fixed kernel-overhead tax. At 16 MoE layers × many steps, launch overhead dominates wall time.

### 4.2 The fused grouped-GEMM kernel

`triton_grouped_moe_dispatch` is the single-launch replacement. The call site (`DeepSeekMoE._routed_forward_triton`) does the same sort, then:

```python
x_sorted = flat[sorted_token_ids].contiguous()      # (T, D) expert-contiguous
y_sorted = triton_grouped_moe_dispatch(
    x_sorted, w1=self._stacked_w1, w2=self._stacked_w2, w3=self._stacked_w3,
    sorted_weights=sorted_weights_1d, expert_offsets=expert_offsets)
y_routed = torch.zeros_like(flat)
y_routed.index_add_(0, sorted_token_ids, y_sorted)  # scatter back to original order
```

`★ Insight ─────────────────────────────────────`
The kernel works on the **sorted-token layout**: a single `(T, D)` tensor where all tokens routed to expert 0 come first, then expert 1's, etc., delimited by `expert_offsets` (the cumsum boundaries). This is the trick that makes a *grouped* GEMM possible — instead of 20 separate `chunk @ w[e]` calls, the kernel walks the sorted tensor block by block, switching which expert's `w1/w2/w3` slice it uses at each boundary. One launch, all experts. The weights carry gradient back to the gate (the `weights`/`indices` tensors are *not* detached — only the `_last_*` snapshots used for the balance metric are).
`─────────────────────────────────────────────────`

The reference `grouped_moe_pytorch` is the same arithmetic routed through the sorted layout, used by CPU tests so the kernel's numerics are verified without a GPU.

### 4.3 The autograd path

`moe_triton.py` exposes a `torch.autograd.Function` whose forward calls the grouped-GEMM kernel and whose backward casts the gradient `dw` to the weight dtype. The kernel autograd path passes BF16 weights directly (comment: "Kernel autograd backward casts dw to w.dtype; pass BF16 weights directly"), so the SwiGLU routed path keeps its gradient flow through the gate — the load-balancing signal survives the fused dispatch.

---

## 5. Re-stacking: a correctness subtlety both kernels share

Both `DeepSeekMoE.forward` and `_shared_forward` **re-stack the expert weights every forward**:

```python
# Re-stack every forward: caching across steps leaves stale copies
# after optimizer.step() (experts would be frozen at init values).
self._stacked_w1 = torch.stack([ex.w1.weight for ex in self.experts], dim=0)...
```

The comment is the lesson: stacking once and caching would capture the *initial* weights; after `optimizer.step()` the live `nn.Linear` weights move but the cached stack would not, silently freezing the experts. Re-stacking every forward is a few-millisecond tax that prevents a catastrophic correctness bug. This is the kind of invariant that's invisible in the forward pass but would make a training run diverge with no obvious cause — and it is exactly what the "never cache optimizer-adjacent tensors across steps" discipline catches.

---

## 6. When to use which path

| Path | When | Why |
|---|---|---|
| `sdpa` / `stacked` (default) | CPU dev, tests, any default-config run | correct, readable, no Triton dependency |
| `triton` / `triton_grouped` | A100/H100 training with `ENABLE_TRITON_KERNELS=1` | fewer launches, less HBM traffic, higher MFU |

The choice is per-config and reversible at runtime — the fallback catches make it impossible to land in a broken state: a missing `triton` package or a dim over the cap drops to the PyTorch path with one warning, not a crash.

> **Next:** [[Docs/13_Portfolio_Comparison]] — positioning DeepSeek-V3-Lite against GPT-OSS-Lite, LLaMA-3-Lite, and Mamba-3-Lite.