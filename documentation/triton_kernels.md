# Triton Kernel Optimization Plan — DeepSeek-v3-Lite

> **Status:** Design proposal, not yet implemented.
> **Audience:** Atandra (review), future-self (implementation).
> **Companion docs:** `MLA.md`, `moe.md`, `mtp.md`, `AGENTS.md §Hard rules`.

---

## 0. Why this plan exists

The 422M DeepSeek-v3-Lite reimplementation is currently **pure PyTorch + `torch.compile(max-autotune)` + FA2-SDPA**. HyMo already ships a custom Triton kernel (`gdn_triton.py`) for its Gated Delta Net recurrence. The rest of the portfolio (LLaMA-3-Lite, Mamba-3-Lite, GPT-OSS-Lite) and **DeepSeek** still rely on vendor kernels only.

This plan identifies the **highest-ROI custom Triton kernels for DeepSeek-v3-Lite training**, ordered by expected speedup, implementation cost, and risk to numerical fidelity. It follows the HyMo integration pattern (optional import, pure-PyTorch fallback, autograd `Function` wrapper) so that the existing test suite and `torch.compile` path keep working.

**Constraint summary (from AGENTS.md):**

- Raw PyTorch first; Triton only where vendor kernels leave measurable headroom.
- Preserve the AuxLossFreeGate bias mechanism — never replace it with an aux loss.
- NaN guard must remain on; new kernels must not silently produce NaNs.
- `BF16` on Ampere/Blackwell; no `GradScaler` needed.
- Single-GPU, 1×A100 80GB SXM; sequence length 2048; micro-batch 8; grad-accum 4.
- Heavy GPU benchmarks live in `scripts/microbench_a100.py` — that's where we validate the win.

---

## 1. Where time actually goes in the current model

Before writing kernels, we need a baseline. Quick analysis of the 422M model from `configs/pretrain_a100_422m.yaml` (single 1×A100 80GB, `max_seq_len=2048`, `micro_batch_size=8`, BF16 + grad-checkpointing):

| Component (per-layer) | FLOPs/token (BF16 MACs) | Memory traffic (B/token) | Notes |
|---|---|---|---|
| `wq` (768→768) | ~1.2M | 2.4 KB | pointwise, mem-bound |
| `wkv_a` (768→216 = 192+24) | ~0.33M | 0.7 KB | pointwise, mem-bound |
| **`wkv_b` materialise K_nope & V (12×(48+64) from 192)** | ~0.13M | 4.4 KB | **the MLA materialisation** |
| `wo` (768→768) | ~1.2M | 2.4 KB | pointwise |
| **SDPA attention (concat Q/K, FA2)** | ~6.0M (4×S=2048) | heavy | FA2 already very fast |
| **RMSNorm × 3 (`attn_norm`, `ffn_norm`, `.norm`)** | trivial | 7.7 KB/token | launch-overhead bound |
| Dense SwiGLU (2 layers × 1536) | ~4.7M | small | simple, well-fused |
| **MoE gate (sigmoid+topk+bias)** | trivial | 0.06 KB | Python overhead heavy |
| **MoE dispatch loop (20 experts × 384)** | ~1.2M (active) | dominant | **per-expert Python trip** |
| MoE shared expert (1× SwiGLU) | ~0.6M | small | already batched |
| MTP block (depth=1) | ~doubled attn/ffn | doubled cache | only on main path |

**Per-step estimate (16 MoE + 2 dense = 18 layers, S=2048, B=8, ~8.4 B tokens / 512k steps):**

- MLA materialisation + SDPA: ~30-35% of step time
- MoE dispatch loop (the per-expert `for e in range(E):`): **~25-30% of step time** — 20 expert launches per layer, 16 MoE layers = 320 expert GEMM launches per step
- RMSNorm / pointwise / bias updates: ~5-10%
- Embedding + LM head: ~5-10% (with weight tying the GEMM is the largest single matmul in the model)
- Optimizer (AdamW fused, FP32 master): ~10-15% (parameter count, not GEMM)

**Targeted win ranking** (based on this profile + HyMo's measured 1.6-1.8× kernel speedup on equivalent kernels):

| Rank | Kernel | Est. step-time saving | Difficulty | Risk |
|---|---|---|---|---|
| 1 | **Fused grouped GEMM for MoE dispatch** (replace 20-expert loop) | **20-30%** | High (block-sparse or sorted-token layout) | Medium |
| 2 | **Fused MLA materialise+RoPE+attention** (one Triton kernel, no `bmm`-`cat`-`permute` chain) | 10-15% | High | Medium |
| 3 | **Fused gate (sigmoid+bias+topk)** | 2-4% | Low | Low |
| 4 | **Fused SwiGLU (gated FFN, dense+expert both)** | 2-4% | Low (vendor fallback exists) | Low |
| 5 | **Fused RMSNorm (forward+residual)** | 1-3% | Low | Very low |
| 6 | **Fused chunked cross-entropy** (loss reduction) | 3-8% at large vocab | Medium | Low (math is trivial) |
| 7 | **Fused MTP-projector + cross-attention** | 1-2% | Medium | Low |

**Cumulative realistic win: 30-45% step-time reduction** if all 7 land cleanly. The first three deliver ~80% of that.

---

## 2. Integration pattern (lifted from HyMo's `gdn_triton.py`)

HyMo already established the right convention. We will copy it exactly so the codebase stays uniform.

**File layout:**

```
LLM/DeepSeek-v3-Lite/
├── models/
│   ├── mla.py            # adds `attn_impl: "triton"` mode
│   ├── mla_triton.py     # NEW — fused MLA materialise+RoPE+attn kernel
│   ├── moe.py            # adds `dispatch: "triton_grouped"` mode
│   ├── moe_triton.py     # NEW — fused grouped-GEMM MoE kernel
│   ├── moe_gate_triton.py# NEW — fused sigmoid+bias+topk kernel
│   ├── norm_triton.py    # NEW — fused RMSNorm(+residual) kernel
│   ├── swiglu_triton.py  # NEW — fused SwiGLU kernel (dense + expert)
│   ├── mtp.py            # unchanged (uses standard SDPA → no immediate win)
│   └── transformer.py    # routes attn_impl and moe_dispatch via config
├── training/
│   └── pretrain.py       # env-var gate: ENABLE_TRITON_KERNELS=1
├── tests/
│   ├── test_mla_triton.py
│   ├── test_moe_triton.py
│   ├── test_moe_gate_triton.py
│   ├── test_norm_triton.py
│   └── test_swiglu_triton.py
└── pyproject.toml        # adds triton>=3.0.0; sys_platform=='linux'
```

**Wrapper convention (HyMo-style, mandatory):**

```python
# models/moe_triton.py
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def triton_grouped_moe_dispatch(x, w1, w2, w3, sorted_token_ids, expert_offsets, ...):
    if not HAS_TRITON:
        raise ImportError("Triton is required for triton_grouped_moe_dispatch. "
                          "Install with: pip install triton")
    # fall through to kernel
```

**Routing convention:**

- New config keys: `attn_impl: "triton"`, `moe_dispatch: "triton_grouped"`, `gate_impl: "triton"`, `norm_impl: "triton"`, `swiglu_impl: "triton"`.
- Old keys retain their values and the existing pure-PyTorch paths stay in place (regression safety net).
- Single env-var master switch: `ENABLE_TRITON_KERNELS=0` (default) → all triton paths off; `=1` → opt in per-config. (Lets tests run on CPU/Mac without triton.)
- `torch.compile(max-autotune)` is left as the **outer** wrapper. Triton kernels are plain `@triton.jit` functions called from inside `nn.Module` forward, so `torch.compile` will trace around them and avoid double-overlapping.

**Test conventions (from HyMo AGENTS.md §Testing rules, adapted):**

1. Every triton kernel ships with a PyTorch reference in the same file.
2. Default tests (CPU, no triton) exercise only the reference path. The triton path is gated behind `@pytest.mark.gpu` (auto-skipped, like HyMo's `@pytest.mark.heavy`).
3. Numerical agreement test: `assert torch.allclose(triton_out, ref_out, atol=…, rtol=…)` — tolerances per kernel (RMSNorm ≈ 1e-5, MoE ≈ 1e-3, MLA ≈ 1e-3 because of bf16 accumulation order).
4. Shape + dtype test, NaN-finite test, deterministic-seed test, gradient-check test.
5. A new `scripts/microbench_a100_triton.py` runs forward+backward for N=50 iters and reports median ms/step, peak VRAM, and **speedup vs the pure-PyTorch baseline** — the same harness structure as `scripts/microbench_a100.py` so the comparison is apples-to-apples.

---

## 3. Kernel 1 — Fused MoE Grouped GEMM (the big one)

**File:** `models/moe_triton.py`
**Replaces:** the `for e in range(E): … index_add …` loop in `models/moe.py:95-108` (20 sequential launches per layer × 16 MoE layers).

### 3.1 Problem statement

Current code:

```python
# in moe.py forward
for e in range(E):                         # E=20
    cnt = counts_cpu[e]                    # host round-trip per iter
    if cnt == 0: continue
    chunk_tokens = sorted_token_ids[start:end]
    expert_in = flat[chunk_tokens]         # gather, BF16
    gate = expert_in @ self._stacked_w1[e].t()  # (cnt, dim) × (inter_dim, dim)  — sparse GEMM
    up   = expert_in @ self._stacked_w3[e].t()
    h    = F.silu(gate) * up
    out  = h @ self._stacked_w2[e].t()
    y_routed = y_routed.index_add(0, chunk_tokens, out * chunk_weights)
```

Cost per layer: **3 small GEMMs × 20 experts = 60 GEMM launches**, plus 20 host syncs from `tolist()`. On A100, each launch is 8-15 µs of CPU overhead, so 20 experts × 3 GEMMs × 16 layers ≈ **1,000 kernel launches per step just for routed experts**, before the GPU has even started.

### 3.2 Triton design: `grouped_gemm_swiglu` kernel

We follow the well-known **grouped GEMM** pattern (CUTLASS-style, single kernel over a sorted-token layout). One kernel launch handles all 20 experts. Each program tiles over one (expert, block-of-tokens) pair.

**Inputs:**

- `x_sorted: (T, dim)` — token activations sorted by expert ID. Index `sorted_token_ids` maps sorted→original position. T = topk × batch × seq (8×2048×4 ≈ 65k tokens for B=8, S=2048, topk=4).
- `w1_stacked: (E, inter_dim, dim)` — `[E, 384, 768]` BF16.
- `w2_stacked: (E, dim, inter_dim)` — `[E, 768, 384]` BF16.
- `w3_stacked: (E, inter_dim, dim)` — same shape as `w1_stacked`.
- `expert_offsets: (E+1,)` — cumsum boundary indices, INT32.
- `sorted_weights: (T, 1)` — gate weights, BF16.

**Kernel grid:** `(E, ceil(max_tokens_per_expert / BLOCK_T))`. Tiling in dim & inter_dim with `BLOCK_D=64`, `BLOCK_I=64` (or autotune).

**Pseudocode (BLOCK_T along token axis, BLOCK_D along input dim, BLOCK_I along inter dim):**

```python
@triton.jit
def grouped_swiglu_kernel(
    x_ptr, w1_ptr, w2_ptr, w3_ptr,
    out_ptr, weights_ptr, expert_offsets_ptr,
    stride_x_t, stride_x_d,
    stride_w1_e, stride_w1_i, stride_w1_d,
    stride_w3_e, stride_w3_i, stride_w3_d,
    stride_w2_e, stride_w2_d, stride_w2_i,
    stride_out_t, stride_out_d,
    T, dim, inter_dim,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    e = tl.program_id(0)
    pid_t = tl.program_id(1)

    start = tl.load(expert_offsets_ptr + e)
    end   = tl.load(expert_offsets_ptr + e + 1)
    n_tokens_e = end - start

    # Skip empty experts
    if n_tokens_e == 0:
        return

    t_off = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t_off < n_tokens_e

    # Load BLOCK_T tokens (rows of x)
    d_idx = tl.arange(0, BLOCK_D)
    x_block = tl.load(
        x_ptr + (start + t_off)[:, None] * stride_x_t + d_idx[None, :] * stride_x_d,
        mask=t_mask[:, None], other=0.0,
    )                                                       # (BLOCK_T, BLOCK_D)

    # Accumulator for gate and up
    i_idx = tl.arange(0, BLOCK_I)
    gate_acc = tl.zeros((BLOCK_T, BLOCK_I), dtype=tl.float32)
    up_acc   = tl.zeros((BLOCK_T, BLOCK_I), dtype=tl.float32)

    # Loop over dim in BLOCK_D tiles
    for d_start in range(0, dim, BLOCK_D):
        d_off = d_start + d_idx
        d_mask = d_off < dim
        x_tile = tl.load(
            x_ptr + (start + t_off)[:, None] * stride_x_t + d_off[None, :] * stride_x_d,
            mask=t_mask[:, None] & d_mask[None, :], other=0.0,
        )                                                   # (BLOCK_T, BLOCK_D)

        w1_tile = tl.load(
            w1_ptr + e * stride_w1_e + i_idx[:, None] * stride_w1_i + d_off[None, :] * stride_w1_d,
            mask=d_mask[None, :], other=0.0,
        )                                                   # (BLOCK_I, BLOCK_D)
        w3_tile = tl.load(
            w3_ptr + e * stride_w3_e + i_idx[:, None] * stride_w3_i + d_off[None, :] * stride_w3_d,
            mask=d_mask[None, :], other=0.0,
        )

        gate_acc += tl.dot(x_tile, tl.trans(w1_tile))       # (BLOCK_T, BLOCK_I)
        up_acc   += tl.dot(x_tile, tl.trans(w3_tile))

    # SwiGLU
    h = tl.sigmoid(gate_acc) * gate_acc * up_acc            # silu(g)*u, fp32

    # Down projection: (BLOCK_T, BLOCK_I) × (dim, BLOCK_I) → (BLOCK_T, dim)
    out_acc = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)
    for i_start in range(0, inter_dim, BLOCK_I):
        i_off = i_start + i_idx
        i_mask = i_off < inter_dim
        h_tile = tl.load(
            h_ptr + (start + t_off)[:, None] * stride_h_t + i_off[None, :] * stride_h_i,
            mask=t_mask[:, None] & i_mask[None, :], other=0.0,
        )                                                   # reuse h from above — actually re-load or keep in registers
        w2_tile = tl.load(
            w2_ptr + e * stride_w2_e + d_idx[:, None] * stride_w2_d + i_off[None, :] * stride_w2_i,
            mask=i_mask[None, :], other=0.0,
        )                                                   # (BLOCK_D, BLOCK_I)
        out_acc += tl.dot(h_tile, tl.trans(w2_tile))

    # Weight & scatter-add
    w = tl.load(weights_ptr + start + t_off, mask=t_mask, other=0.0)
    out_acc = (out_acc * w[:, None]).to(tl.bfloat16)

    tl.atomic_add(
        out_ptr + (start + t_off)[:, None] * stride_out_t + d_idx[None, :] * stride_out_d,
        out_acc, mask=t_mask[:, None],
    )
```

**Output:**

- `y_sorted: (T, dim)` — same shape as `x_sorted`. The original Python wrapper then scatters with `sorted_token_ids` to undo the sort, exactly matching the current code's behaviour (a single `index_add` over T is much cheaper than 20 of them).

### 3.3 Forward + autograd Function

Wrap the kernel in a `torch.autograd.Function` (HyMo pattern: `TritonGDNFunction`). The backward pass needs three additional kernels:

- `grouped_swiglu_dgate_kernel` — gradient w.r.t. `w1`, `w3`, `x`.
- `grouped_swiglu_dw2_kernel` — gradient w.r.t. `w2`, `x`.
- `grouped_swiglu_dx_kernel` — gradient w.r.t. `x` from up-stream.

The cleanest design is to follow the **flash-attention "save-lhs / re-compute"** pattern: forward stores `x_sorted`, `gate_pre_silu`, `h_post_silu`, and `sorted_weights`; backward recomputes what it can. This trades ~3× activation memory for not having to materialise the full (T, dim, inter_dim) intermediate in HBM.

### 3.4 Shared expert

The shared expert is one SwiGLU applied to **all** T tokens (no routing). That is just a single dense FFN, much smaller than the routed expert workload. We use the same Triton **fused SwiGLU** kernel from §3.5, called once. It already exists in the dense path; we can share the implementation.

### 3.5 Validation

- `test_grouped_gemm_swiglu_matches_reference`: random (T, dim, inter_dim, E) with 1-2 non-empty experts → reference `einsum` vs triton. `atol=1e-3` for BF16, `rtol=1e-3`.
- `test_grouped_gemm_empty_experts`: experts with 0 tokens don't NaN.
- `test_grouped_gemm_gradient_check`: `torch.autograd.gradcheck` on a tiny config (dim=32, inter_dim=32, E=4, T=8) in float32.
- `test_moe_full_path_agrees`: integrate into `DeepSeekMoE.forward`, share weights with the `stacked` path, assert `torch.allclose(y_triton, y_stacked, atol=1e-2)`. This is a tightened version of the existing `test_stacked_and_grouped_agree`.

### 3.6 Expected impact

HyMo's GDN kernel reported ~1.7× over the pure-PyTorch loop. For MoE, the per-launch overhead is even worse (more launches per layer), so a **1.8-2.5× speedup on the MoE path is realistic**. Given MoE is 25-30% of step time, the overall step-time win is **~15-20%**.

### 3.7 Risk register

- **Risk 1: Atomic-add contention** on the scatter. If a single output row receives contributions from many experts, atomic adds serialise. Mitigation: `index_add` over the sorted layout is already correct; we just collect all expert outputs into `y_sorted` (one row per sorted token), then a single `index_add_` from sorted to original position outside the kernel — avoids contention.
- **Risk 2: Numerical drift** from BF16 reduction order. Mitigation: use fp32 accumulators inside the kernel (mandatory), and assert tolerance against the reference within `atol=1e-2` (BF16 noise floor).
- **Risk 3: Compilation time** on first call (autotune over BLOCK_T/BLOCK_D/BLOCK_I). Mitigation: pre-warm in `__init__` with a single dummy forward.

---

## 4. Kernel 2 — Fused MLA materialise+RoPE+attention

**File:** `models/mla_triton.py`
**Replaces:** the chunky `bmm → cat → permute → SDPA → permute → wo` sequence in `models/mla.py:118-143`.

### 4.1 Problem statement

Current SDPA path:

```python
# lines 127-143 in models/mla.py
ctx_kv_bmm = ctx_kv.reshape(bsz * seqlen_k, self.kv_lora_rank).unsqueeze(0).expand(h, -1, -1)
wkv_b_kv = torch.cat([wkv_b_k, wkv_b_v], dim=1)
KV_nope_h = torch.bmm(ctx_kv_bmm, wkv_b_kv.transpose(-1, -2))   # bmm #1
K_nope_h, V_h = KV_nope_h.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
K_nope = K_nope_h.reshape(h, bsz, seqlen_k, self.qk_nope_head_dim).permute(...).contiguous()
V     = V_h.reshape(h, bsz, seqlen_k, self.v_head_dim).permute(...).contiguous()
Q_nope = q_nope.transpose(1, 2)
Q_rope = q_pe.transpose(1, 2)
K_rope = ctx_pe.unsqueeze(1).expand(-1, h, -1, -1)
attn = F.scaled_dot_product_attention(...)                       # FA2
return self.wo(attn.transpose(...).flatten(2))
```

Pain points:

1. The `bmm` materialises `K_nope` and `V` as **(B, H, S, 48)** and **(B, H, S, 64)** in HBM — that's 2 × 8 × 12 × 2048 × ~60 = ~24 MB at B=8. It's only a single GEMM but the **subsequent permute+contiguous** chain copies this twice.
2. The RoPE K (`ctx_pe`) is broadcast-expanded to (B, H, S, 24) and concatenated — 3 separate HBM allocations.
3. The SDPA result is permuted back and copied again before `wo`.

This is the **classic Flash-Decoding-style fusion opportunity**: keep K_nope, V, K_rope in registers inside one kernel, fuse with the RoPE rotation, and write the final per-token `wo(input)` result directly.

### 4.2 Triton design: `mla_flash_attn` kernel

We adapt the FlashAttention-2 forward kernel (Dao-AI's open reference) to the MLA specifics:

- **Q content (`q_nope`)** is per-head, per-token: (B, H, S, D_nope=48).
- **Q rope (`q_pe`)** is per-head, per-token: (B, H, S, D_rope=24), already RoPE-rotated upstream.
- **K_nope** is `wkv_b_k @ c_kv` per (head, key-token): (B, H, S, D_nope=48).
- **V** is `wkv_b_v @ c_kv` per (head, key-token): (B, H, S, D_v=64).
- **K_rope** is the cached RoPE key: (B, S, D_rope=24), broadcast to heads.

The score is `Q_nope · K_nope + Q_rope · K_rope`, scaled by `1/√(qk_head_dim)`, softmax, then weighted sum over `V`.

**Key fusion:** the K_nope and V materialisation is **fused into the QK loop**. Each program is one (B, H, Q_block). Inside the inner loop over K-blocks:

```python
# K-block tile
# 1. Load ctx_kv (B, H_block, K_block, kv_lora_rank)  — small (192)
# 2. On-the-fly matmul: K_nope_tile = ctx_kv @ wkv_b_k[h, :, :].T  (Triton tl.dot, in registers)
# 3. On-the-fly matmul: V_tile      = ctx_kv @ wkv_b_v[h, :, :].T
# 4. Load ctx_pe_tile (B, 1, K_block, 24), broadcast over H.
# 5. Accumulate partial scores
# 6. Online softmax (FA2 algorithm)
# 7. Accumulate partial V-weighted sum
```

**Important:** the `wkv_b` weights are **loaded into SMEM once per (B, H_block)** and reused for every K-block iteration. This is the big win over the current Python code: 2 full HBM roundtrips saved per attention call.

**Output:** directly write `wo @ attn_out` (B, S, dim) by also loading `wo` into the kernel — or, simpler, return (B, H, S, D_v) and let the existing `wo` matmul run as before. The first version saves another HBM roundtrip on the attention output but requires `wo` to fit in SMEM (dim × H×D_v = 768 × 768 = ~590k elements ≈ 1.2 MB in BF16 — fits, but tight). **Start with the simpler version** and upgrade if profiling shows it.

### 4.3 Pre-compute the `wkv_b` reshaping

The current code reshapes `wkv_b.weight` on every forward (line 118). Cache it: store `wkv_b_k`, `wkv_b_v` as buffers/parameters, built once in `__init__` and updated only if weights are re-initialised. This is a free 5% win and unblocks the kernel.

### 4.4 RoPE helpers

The RoPE application (`_apply_rope` in mla.py:54-58) does complex multiply via `view_as_complex`. We can fuse it into the kernel: pass `freqs_cis` as a kernel argument and apply rotation on the fly inside the same kernel that loads `q_pe` and `ctx_pe`. This removes two separate kernels' worth of HBM traffic.

### 4.5 Decoding path

For prefill (S = 2048, many tokens) the FlashAttention-style block-tiled kernel wins. For decoding (S = 1, growing context up to 2048) the **flash-decoding split-K** pattern wins. We provide both; the runtime dispatches based on `seqlen_q == 1`.

### 4.6 Validation

- `test_mla_triton_matches_sdpa`: random (B, H, S, D) inputs; build a tiny `MultiHeadLatentAttention` with both impls, share weights; assert logits match within `atol=1e-2` (BF16).
- `test_mla_triton_gradient_check`: float32, tiny config.
- `test_mla_triton_with_cache`: end-to-end with `use_cache=True`, verify cache contents.
- `test_mla_triton_long_sequence`: S=2048, check no NaN, check wall-time vs SDPA.

### 4.7 Expected impact

FA2 already saturates the GEMM-bound part of attention. Our win is **all the surrounding memory traffic** (K_nope/V materialisation, RoPE I/O, the permute-contiguous chain). Empirically (from PyTorch profiler, the `.contiguous()` calls each cost 0.3-0.5 ms at S=2048):

- 2 saved `.contiguous()` on K_nope/V: ~0.6-1.0 ms
- 1 saved materialisation+permute of K_nope/V: ~0.4-0.6 ms
- RoPE fusion: ~0.2-0.4 ms
- Combined: **~1.5-2.5 ms per layer** × 18 layers = 27-45 ms/step.
- At a baseline of ~250 ms/step (rough), that is **10-15% step-time reduction**.

### 4.8 Risk register

- **Risk 1: correctness of online softmax in BF16** — we use fp32 accumulators for the softmax denominator and the value sum (FA2 reference is the gold standard).
- **Risk 2: q_lora_rank != 0** — the design above assumes `q_lora_rank=0` (no query compression, the 422M config). The kernel can be extended to support `q_lora_rank>0` by pre-computing the compressed `c_q` upstream and treating it as a virtual Q; this is straightforward but unverified in this plan. **Defer if no rush.**
- **Risk 3: kernel autotune compile time** — use a small grid of `triton.autotune` configs (5-8) keyed on `(S, H, D_nope, D_rope, D_v)`; pre-warm at `__init__` with dummy inputs.

---

## 5. Kernel 3 — Fused gate (sigmoid + bias-add + topk + renormalise)

**File:** `models/moe_gate_triton.py`
**Replaces:** `models/moe.py:28-35` (`AuxLossFreeGate.forward`).

### 5.1 Problem statement

Current gate (called every MoE layer forward, 16× per step):

```python
scores = F.linear(x, self.weight).sigmoid()   # (T, E) — 1 matmul + 1 elementwise
biased = scores + self.bias.to(scores.dtype)  # 1 broadcast-add
indices = biased.topk(self.topk, dim=-1)[1]  # 1 sort-like op
weights = scores.gather(1, indices)           # 1 gather
weights = (weights / weights.sum(...).clamp(min=1e-10) * route_scale).to(x.dtype)  # reduce+mul+cast
```

Five separate kernel launches, each at (T, E) = (8×2048, 20) ≈ 320k elements — completely launch-bound, not compute-bound. T_topk sort alone is slow for E=20.

### 5.2 Triton design

One kernel that does the whole pipeline:

```python
@triton.jit
def gate_fwd_kernel(
    x_ptr,         # (T, dim)
    w_ptr,         # (E, dim)
    bias_ptr,      # (E,)
    out_w_ptr,     # (T, topk)
    out_i_ptr,     # (T, topk)  — INT32
    T, dim, E, topk,
    route_scale,
    BLOCK_T: tl.constexpr,   # rows of x to process per program
    BLOCK_E: tl.constexpr,   # = next_pow2(E), 32 for E=20
):
    pid = tl.program_id(0)
    t_off = pid * BLOCK_T + tl.arange(0, BLOCK_T)         # (BLOCK_T,)
    t_mask = t_off < T
    e_idx = tl.arange(0, BLOCK_E)                         # (BLOCK_E,)
    e_mask = e_idx < E

    # Load x tile and all of w
    # For BLOCK_T rows, compute scores = x @ w.T → (BLOCK_T, BLOCK_E)
    # ... in fp32
    s = tl.sigmoid(scores)                                # (BLOCK_T, BLOCK_E)
    biased = s + tl.load(bias_ptr + e_idx, mask=e_mask, other=0.0)

    # Top-k via repeated argmax (E=20, topk=4 → ~80 ops, fine in a kernel)
    # Or use Triton's tl.argmax / tl.sort if available
    # For each of topk slots:
    #   best_idx = argmax(biased)
    #   write best_idx
    #   set biased[best_idx] = -inf
    # Then gather s at topk indices → out_w
    # Normalise: out_w = out_w / sum(out_w) * route_scale
```

Triton 3.0+ has `tl.argmax` for small inner dims; for E=20 the explicit loop is fine.

### 5.3 Expected impact

At ~30 µs per launch × 5 launches × 16 MoE layers = ~2.4 ms/step. The fused kernel is one launch of ~50 µs, total 0.8 ms/step. **Step-time saving: 1.5-2.5 ms (~1-2%).** Small but easy.

### 5.4 Risk register

- The bias-update mechanism stays outside the kernel (`AuxLossFreeGate.update_bias` runs in Python, separately). **No change to AGENTS.md hard rule #3.**
- Tie-breaking in argmax may differ from `torch.topk` exactly; we don't care as long as the routing distribution is statistically close. We won't try to bitwise-match `torch.topk` — we just need the gradient to remain valid.

---

## 6. Kernel 4 — Fused SwiGLU (dense + shared-expert FFN)

**File:** `models/swiglu_triton.py`
**Replaces:** `SwiGLUFFN.forward` in `models/transformer.py:17` and the shared-expert path in `models/moe.py:114-131`.

### 6.1 Problem statement

```python
return self.w2(F.silu(self.w1(x)) * self.w3(x))  # 3 GEMMs + silu + mul
```

Each is a small matmul: 768×1536 (dense) or 768×384 (expert). At B=8, S=2048, dense: 8×2048×768×1536 = 19.3 GFLOPS per matmul × 3 = 58 GFLOPS per dense layer. Two dense layers × 18 layers / step = manageable. The cost is more about the **silu + multiply** memory traffic and launch overhead.

### 6.2 Triton design

A straightforward `fused_swiglu` kernel:

```python
@triton.jit
def swiglu_fwd_kernel(
    x_ptr, w1_ptr, w2_ptr, w3_ptr, out_ptr,
    T, dim, inter_dim,
    BLOCK_T, BLOCK_D, BLOCK_I,
):
    pid = tl.program_id(0)
    t_off = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    # ... (load x_tile, do w1 dot, w3 dot, silu, mul, w2 dot, write out)
```

Standard pattern; reference: `triton/python/triton/ops/ffn.py` in upstream Triton. Add backward kernels for `dx`, `dw1`, `dw2`, `dw3` (or skip backward and let autograd recompute via a custom Function that stores intermediates).

### 6.3 Expected impact

Dense path is 2 of 18 layers, ~5% of total. Shared expert is 1 SwiGLU on all 65k tokens, ~2% of total. **Step-time saving: 2-4%.** Worth doing because the kernel is **the same one** we need for the MoE grouped kernel — write it once, call it from both places.

---

## 7. Kernel 5 — Fused RMSNorm (+ residual add)

**File:** `models/norm_triton.py`
**Replaces:** `nn.RMSNorm` and the residual additions around it (3-4 per layer).

### 7.1 Problem statement

Three RMSNorms per layer × 18 layers = 54 per step. Each is `(B, S, 768)` reduction, completely bandwidth-bound. Each launches a `mean+rsqrt+mul` chain. vLLM and xFormers both ship fused RMSNorm Triton kernels we can adapt (vLLM's is ~1.5× faster than `nn.RMSNorm` at our shape).

### 7.2 Triton design

Standard fused RMSNorm. Optionally fold the residual add into the same kernel: instead of `x = x + attn(rms(x)); x = x + ffn(rms(x))`, we write `x = x + attn(rms(x_in))` and have the kernel also add the residual stream in the same pass. The current code already has `x + self.attn(...)` as a single line — folding it is straightforward.

### 7.3 Expected impact

~30 µs × 54 = 1.6 ms/step. Fused ≈ 0.5 ms/step. **Saving: 1-2%.** Lowest priority of the seven; do this last or skip.

---

## 8. Kernel 6 — Fused chunked cross-entropy

**File:** `training/loss_triton.py` (or inline in `models/transformer.py`)
**Replaces:** the per-step `F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))`.

### 8.1 Problem statement

`vocab_size=100,018`. The cross-entropy is the single largest matmul in the model (output of the tied LM head). It materialises `(B*S, 100018) = 8 × 2048 × 100,018 = 1.6 billion` floats in BF16 = **3.2 GB of activations per step** at B=8, S=2048. This is the dominant activation memory consumer and a major wall-time hit.

### 8.2 Triton design

Fused **log-softmax + NLL + chunked reduction** kernel, à la Liger-Kernel. The matmul (LM head) is left to `torch.matmul` (or `F.linear` with the tied embedding — `mm.t()`), but the **log-softmax + gather + mean** over `vocab_size` is done in one Triton kernel that processes vocab in chunks of `BLOCK_V=8192`:

```python
@triton.jit
def chunked_ce_fwd_kernel(
    logits_ptr, targets_ptr, out_loss_ptr,
    BT, vocab,                                  # BT = B*S
    ignore_index, BLOCK_BT, BLOCK_V,
):
    pid = tl.program_id(0)                      # one program per row of BT
    row = pid * BLOCK_BT + tl.arange(0, BLOCK_BT)
    row_mask = row < BT
    target = tl.load(targets_ptr + row, mask=row_mask, other=ignore_index)

    # 1. Find max across vocab (numerical stability)
    m = -float("inf")
    for v_start in range(0, vocab, BLOCK_V):
        v = v_start + tl.arange(0, BLOCK_V)
        v_mask = v < vocab
        logits_chunk = tl.load(logits_ptr + row[:, None] * vocab + v[None, :],
                                mask=row_mask[:, None] & v_mask[None, :], other=-float("inf"))
        m = tl.maximum(m, tl.max(logits_chunk, axis=1))

    # 2. Sum exp(logits - m) for log-sum-exp
    s = tl.zeros((BLOCK_BT,), dtype=tl.float32)
    for v_start in range(0, vocab, BLOCK_V):
        v = v_start + tl.arange(0, BLOCK_V)
        v_mask = v < vocab
        logits_chunk = tl.load(...)
        s += tl.sum(tl.exp(logits_chunk - m[:, None]), axis=1)

    # 3. Gather target logits and compute -log_softmax
    log_z = m + tl.log(s)
    target_logits = ...  # gather (use a separate small gather kernel or fold into the loop)
    loss = log_z - target_logits
    loss = tl.where(target == ignore_index, 0.0, loss)
    tl.store(out_loss_ptr + row, loss, mask=row_mask)
```

Then a single `loss.mean()` in PyTorch.

### 8.3 Expected impact

Activation memory: 3.2 GB → **~32 MB** (only one row of the logits in flight at a time). Wall-time: torch's `cross_entropy` is heavily optimised but still launches ~5 kernels. Fused version: 1 kernel + 1 mean. **Step-time saving: 3-8%, memory saving 1-2 GB.**

### 8.4 Risk register

- NaN-guard interaction: the fused kernel must write 0 for `ignore_index` rows, so `loss.mean()` on a batch with all-ignored rows doesn't NaN. Already handled above.

---

## 9. Kernel 7 — Fused MTP-projector (deferred)

**File:** inside `models/mtp.py`.
**Status:** **Lowest priority.** MTP depth=1 doubles the attention+FFN cost only on the MTP path, but the MTP attention uses `nn.MultiheadAttention` (not MLA), and the win from a custom kernel is bounded. Defer until the other six land and we have a clear bottleneck profile.

---

## 10. Implementation roadmap (sequenced)

Two-week plan with two implementation phases. All GPU work happens on the A100 box; CPU/Mac is for code authoring + unit tests.

### Phase A — Foundations + low-risk wins (5-7 days)

| Day | Task | Verifies |
|---|---|---|
| A1 | Add `triton>=3.0.0; sys_platform=='linux'` to `pyproject.toml`; document fallback path in `models/__init__.py`. | `pip install -e .` works on Linux; `HAS_TRITON=False` on Mac. |
| A2 | Implement `models/norm_triton.py` (RMSNorm + residual). | `test_norm_triton.py` passes on CPU. |
| A3 | Implement `models/swiglu_triton.py` (dense + shared-expert path). | `test_swiglu_triton.py` passes on CPU. |
| A4 | Implement `models/moe_gate_triton.py` (fused sigmoid+bias+topk). | `test_moe_gate_triton.py` passes on CPU. |
| A5 | A100 validation of A2-A4: per-kernel benchmark vs PyTorch, plus full `microbench_a100.py` with each enabled individually. | Each saves the predicted %; nothing regresses. |
| A6 | Wire `ENABLE_TRITON_KERNELS=1` env-var gate in `training/pretrain.py`; document in `SKILLS.md` and `AGENTS.md` (note added in changelog, not in the hard-rules list). | Default path (env=0) identical to current behaviour, including the existing 28-test suite. |

**Phase A expected cumulative win: 4-8% step time, zero regressions.**

### Phase B — The two big wins (5-7 days)

| Day | Task | Verifies |
|---|---|---|
| B1 | Implement `models/moe_triton.py` (grouped GEMM + SwiGLU). | `test_moe_triton.py` passes on CPU; MoE path matches `stacked` reference within `atol=1e-2` on A100. |
| B2 | Implement backward for MoE kernel (or use re-compute pattern). | `torch.autograd.gradcheck` passes on float32 tiny config. |
| B3 | Wire into `DeepSeekMoE.forward` behind `moe_dispatch: "triton_grouped"`. End-to-end: full-model forward+backward produces identical loss to `stacked` within bf16 tolerance. | `tests/test_models.py` still passes; new `test_moe_full_path_agrees` passes. |
| B4 | A100 benchmark: 50-step forward+backward median, MoE only vs full model. | Speedup ≥ 1.5× on MoE path; ≥ 1.2× overall. |
| B5 | Implement `models/mla_triton.py` (FA2-style fused kernel with on-the-fly K_nope/V materialisation). | `test_mla_triton.py` passes. |
| B6 | Implement MLA forward + backward (autograd Function). | Gradient check on float32 tiny. |
| B7 | Wire into `MultiHeadLatentAttention` behind `attn_impl: "triton"`. End-to-end loss agreement test. | `test_sdpa_and_triton_agree` passes within `atol=1e-2`. |
| B8 | A100 benchmark: full `microbench_a100.py` with all kernels on. | Total step-time reduction ≥ 25% vs baseline. |

**Phase B expected cumulative win: 25-40% step time on top of Phase A.**

### Phase C — Polish (3-4 days, optional)

| Day | Task |
|---|---|
| C1 | Implement `training/loss_triton.py` (fused chunked cross-entropy). |
| C2 | MTP-block Triton attention (deferred to here only if profiling shows it matters). |
| C3 | A100 sweep: BLOCK_T, BLOCK_D, BLOCK_I, num_warps, num_stages for every kernel. |
| C4 | End-to-end 1000-step smoke training run; verify convergence (loss curve within noise of baseline). |
| C5 | Update `documentation/MLA.md`, `documentation/moe.md`, `documentation/training.md` with new sections. |
| C6 | Add `scripts/microbench_a100_triton.py` and a comparison table to `documentation/triton_kernels.md`. |

---

## 11. Hard constraints and non-goals

### 11.1 Will not do

- **No `torch.compile` rewriting of these kernels.** Triton kernels are opaque to `torch.compile`; we run them in `nn.Module.forward` and let `compile` trace around. Replacing the Triton with `torch.compile`'s codegen is the same fight we'd lose anyway.
- **No replacing the AuxLossFreeGate mechanism with an aux loss.** The bias-update in `update_bias` stays in Python (out-of-band, every step). Only the forward path of the gate is fused.
- **No MoE expert-parallel or tensor-parallel.** The repo is single-GPU by design (`AGENTS.md §Hard rules`, training/pretrain.py:122).
- **No speculative-decoding changes.** `inference/speculative.py` is unaffected.
- **No changes to checkpointing format.** Triton kernels are stateless wrappers — the underlying `nn.Parameter`s don't change shape, dtype, or names. The bias buffer still lives on the gate.

### 11.2 Will not break

- The 28 existing tests in `tests/` keep passing with `ENABLE_TRITON_KERNELS=0` (the default).
- The MLA SDPA path and the manual path stay in the file.
- The MoE `stacked` and `grouped` paths stay in the file.
- `torch.compile(max-autotune)` continues to wrap the model; triton kernels are opaque to it.
- The NaN guard, μP LR scaling, and `bias_update_every` cadence are unaffected.

### 11.3 Will require

- A100 box (or any CUDA SM_80+) for the GPU tests and benchmarks. The Triton kernels are pure GPU code; no CPU fallback is viable for the hot paths.
- Updating `pyproject.toml` to declare `triton>=3.0.0; sys_platform=='linux'` (HyMo's pattern).
- Adding a new `pytest` marker `gpu` (alongside HyMo's `heavy`) and a `conftest.py` auto-skip rule.
- One new doc (`documentation/triton_kernels.md`, this file) and entries in `documentation/MLA.md`, `documentation/moe.md`, and `documentation/training.md`.

### 11.4 Two-layered opt-in: dispatch contract

The sanctioned Triton paths are gated on **both** a per-kernel config key
and the master `ENABLE_TRITON_KERNELS=1` env-var. The implementation
of the env-var guard lives in `models/_triton_dispatch.py`:

| Per-kernel config key | Triton value | PyTorch default | File |
|---|---|---|---|
| `attn_impl`     | `"triton"`         | `"sdpa"`    | `models/mla.py:MultiHeadLatentAttention.__init__` |
| `moe_dispatch`  | `"triton_grouped"` | `"stacked"` | `models/moe.py:DeepSeekMoE.__init__` |

`enforce_triton_env_var(model_cfg, log)` is the single function that
checks the env-var and force-backs any Triton value to its PyTorch
default. It is called from two sites so the guard fires no matter how
the model is built:

- `models/transformer.py:Transformer.__init__` — covers model
  construction in tests, inference scripts, and any direct
  `Transformer(cfg)` use.
- `training/pretrain.py:Pretrainer.__init__` — covers the training
  entry point, including `python -m training.pretrain --config ...`.

A misconfigured run (Triton keys in YAML but no env-var) surfaces as
**one** startup warning listing all forced keys, not one warning per
MoE/MLA layer when the kernel first fires. This matches the Mamba-3
and LLaMA-3 force-back pattern.

Test coverage: `tests/test_force_back.py` — 11 CPU tests covering
single-key and dual-key force-back, env-var pass-through, PyTorch
defaults preserved, `attn_impl='manual'` not affected, and an
integration test that builds a `Transformer` and asserts the
per-layer module reads the rewritten value.

---

## 12. Open questions for Atandra (decide before Phase B)

1. **Kernel-1 grouped-GEMM design**: full re-implementation in Triton (Phase B1), or wrap an existing library like `grouped_gemm` (CUTLASS Python bindings) and call it from `nn.Module.forward`? The custom route is more code, more learning value, and ~10% faster in our experience. The library route is 1-2 days vs 4-5. *Default: custom route, but flag if you'd rather ship the library first.*
2. **MLA kernel scope**: prefill-only (S=2048) or also flash-decoding (S=1, growing)? *Default: prefill only first; decode as a follow-up if profile shows decode-bound.*
3. **Loss-fusion scope**: do we also need a fused `cross_entropy` for the **MTP** predictions? The MTP heads call `F.cross_entropy` in a Python loop. *Default: yes, fuse both in one kernel that handles a list of (logits, targets) pairs.*
4. **Backward strategy for MoE kernel**: re-compute (FA2-style, low HBM, ~30% more compute) vs save intermediates (high HBM, simpler code). *Default: re-compute, since MoE already activates only ~20% of experts per token and HBM is the constraint at our 422M scale.*
5. **Master switch location**: env-var (`ENABLE_TRITON_KERNELS=1`) or config key (`model.triton_kernels: true`)? *Default: env-var, because the per-kernel switches are config keys and we want a single kill-switch for "ship without any triton".*

---

## 13. Success criteria

The plan is successful when, after Phase B, on the canonical 422M config and a 200-step warmup + 50-step timed window on 1×A100 80GB:

| Metric | Baseline (current) | Target | Stretch |
|---|---|---|---|
| Median step time (ms) | ~250 | **≤ 175** | ≤ 150 |
| Peak VRAM (GB) | 60-65 | ≤ 60 (slight drop from loss fusion) | ≤ 55 |
| Tokens/sec | ~16k | **≥ 23k** | ≥ 27k |
| MFU | 35-40% | **≥ 50%** | ≥ 60% |
| Loss curve over 1k steps | (baseline) | within ±2% | within ±1% |
| Test pass count | 28 | **28 + new (28+) = 56+** | 60+ |

The 50% MFU target is the headline number: it's the gap between "the model works" and "the model runs at a competitive fraction of the GPU's peak".

---

## 14. References

- **HyMo GDN kernel (the model for this plan):** `LLM/HyMo/src/hymo/models/gdn_triton.py` — Triton 3.0+ kernel with autograd `Function`, pure-PyTorch fallback gated on `HAS_TRITON`.
- **HyMo dependency declaration:** `LLM/HyMo/pyproject.toml:49` — `triton>=3.0.0; sys_platform=='linux'`.
- **FlashAttention-2 reference (Triton):** https://github.com/Dao-AILab/flash-attention — the public Triton FA2 fwd/bwd kernel.
- **Liger-Kernel (chunked CE reference):** https://github.com/linkedin/Liger-Kernel — fused linear+CE pattern.
- **vLLM RMSNorm:** https://github.com/vllm-project/vllm — fused RMSNorm+residual.
- **DeepSeek-V3 paper:** arXiv:2412.19437 — §2.1.2 (MLA), §2.3.3 (aux-loss-free MoE).
- **vLLM MLA implementation** (DeepSeek-specific kernel patterns): https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/attention/MLA — our main reference for the flash-decoding split-K pattern.
- **Project docs (authoritative):** `documentation/MLA.md`, `documentation/moe.md`, `documentation/mtp.md`.
- **Project hard rules:** `AGENTS.md §1, §Hard rules`, `AGENTS.md §2.13 (deepseek-v3-engineer subagent)`.
