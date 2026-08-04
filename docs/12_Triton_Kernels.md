# 12 — Custom Triton Hardware Kernels

> **Canonical** for DeepSeek-v3-Lite's real fused Triton kernels: fused MLA attention and fused grouped-GEMM MoE dispatch. Educational textbook chapter with full from-scratch explanations.

> How two custom Triton kernels turn the PyTorch reference paths into single-launch fused GPU programs, and the double-opt-in guard that keeps the CPU test suite green. **Both kernels are BF16** — there is no FP8 GEMM in this repo (FP8 is paper-spec, see [[Docs/06_FP8_Mixed_Precision]]).

**Depends on:** [[Docs/03_Multi_Head_Latent_Attention]], [[Docs/04_DeepSeekMoE]], [[Docs/06_FP8_Mixed_Precision]] · **Read next:** [[Docs/13_Portfolio_Comparison]]

**Source files:** `models/mla_triton.py`, `models/moe_triton.py`, `models/mla.py`, `models/moe.py`, `models/_triton_dispatch.py`, `scripts/microbench_a100.py`

---

## 0. Status in this repo

| Kernel | File | Status | Precision |
|---|---|---|---|
| Fused MLA attention (FA2-style) | `models/mla_triton.py` | ✅ implemented, opt-in | BF16 |
| Fused grouped-GEMM MoE SwiGLU dispatch | `models/moe_triton.py` | ✅ implemented, opt-in | BF16 |
| Triton force-back guard | `models/_triton_dispatch.py` | ✅ always-on | — |
| FP8 GEMM | — | ❌ not implemented (paper-spec) | — |

The kernels are **opt-in**: a config may request `attn_impl: triton` or `moe_dispatch: triton_grouped`, but unless the environment variable `ENABLE_TRITON_KERNELS=1` is set, `models/_triton_dispatch.py:enforce_triton_env_var` silently rewrites those keys to their PyTorch defaults (`sdpa`, `stacked`) with a single warning. This is what keeps the pytest suite green (189 passed + 10 GPU-gated skips on a laptop) — Triton is Linux+CUDA-only, and the test box has neither.

### Which kernel runs at which config?

The two kernels are not usable on the same configs. The MoE grouped-GEMM kernel hard-caps at `dim, moe_inter_dim ≤ 256` (§5.7), which excludes the canonical config:

| Config | MLA kernel (`attn_impl: triton`) | MoE kernel (`moe_dispatch: triton_grouped`) |
|---|---|---|
| **Canonical 411.6M** (`pretrain_a100_422m.yaml`) — `kv_lora_rank=192`, head dims 48/24/64 | ✅ usable (all dims < 256) | ❌ **falls back to `stacked`** — `dim=768`, `moe_inter_dim=384` exceed the 256 register cap |
| **1650 2M** (`pretrain_1650_2m.yaml`) — `dim=64`, `moe_inter_dim=32` | ✅ usable | ✅ usable |

Consequence: on the canonical A100 run **only the MLA Triton kernel can be exercised**. The MoE grouped-GEMM kernel is validated only at smoke scale, so the AGENTS.md `≥1.5× MoE speedup` benchmark target applies to the 1650 config — at the canonical config the MoE path is `stacked` regardless of `moe_dispatch`. Every speed figure in this chapter is therefore an **estimate** (`.benchmarks/` is empty; no GPU run has executed yet — see §9).

---

## 1. Why custom kernels at all?

The PyTorch reference paths (`mla.py` SDPA path, `moe.py` stacked path) are correct and readable, but they make tensor round-trips through HBM that a fused kernel can avoid:

- **MLA:** the SDPA path materialises full per-head `K_nope` and `V` by a batched matmul of the compressed latent `ctx_kv` with `wkv_b`, writes them to HBM, then immediately reads them back for attention. The fused kernel keeps `wkv_b_k` / `wkv_b_v` in registers and produces `K_nope`/`V` **inside** the K-block loop, so they never touch HBM.
- **MoE:** the stacked path is a Python `for e in range(E)` loop, one matmul per expert per weight, with a sort + `index_add` scatter. The fused grouped-GEMM kernel launches once over the sorted-token layout and does all experts' SwiGLU in one kernel.

The payoff is fewer kernel launches and less HBM traffic — the two things that dominate latency at these shapes. The cost is a 256-element register-budget cap and a backward pass that (for MLA) is currently a correctness-preserving stub.

To put a number on the MLA traffic: at the canonical config (`R=192`, `D_nope=48`, `D_v=64`, `H=12`, `S=2048`, `micro_batch_size=8`), the two materialisation einsums produce `B·H·S·(D_nope+D_v) = 8·12·2048·112 ≈ 22` million elements ≈ 44 MB per layer counting the write *and* the re-read (22 MB each way). Across the 18 layers that is ≈ 0.8 GB of avoidable HBM traffic per forward pass — and because training runs with grad checkpointing (`use_checkpoint=True`), the whole forward re-runs in the backward pass, so it is paid **twice per step**. This is exactly the traffic the fused kernel eliminates.

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

## 3. Triton from zero

Before we walk through either kernel, we need the minimal mental model of what a Triton program *is*. Everything in §4 and §5 is built from the six ideas in this section — if you already know Triton, skim to §3.5 (online softmax) and §3.6.

### 3.1 The execution model: a kernel is a grid of independent programs

A GPU executes a kernel as a **grid** of *program instances* (the CUDA name is *thread blocks* / CTAs; Triton calls them programs). The grid is a 1-, 2-, or 3-D array; the launch site states its shape:

```python
_mla_flash_fwd_kernel[(B, H, n_q_blocks)](...)   # a 3-D grid
_grouped_moe_fwd_kernel[(E, n_blocks_per_expert)](...)  # a 2-D grid
```

Inside the kernel, `tl.program_id(0)`, `tl.program_id(1)`, `tl.program_id(2)` read the program's own coordinate along each grid axis. Every program instance runs the *same* code on *different* data; the only thing that distinguishes them is these ids (plus the pointer offsets they compute from them). This is the whole trick of GPU programming: **parallelism = partitioning data across programs, then writing one kernel that knows its slice.**

Two things are deliberately *not* inside the kernel body:

- The grid shape itself. It is a Python-side decision at the launch site, and the kernel can only see it through `program_id` bounds.
- Any Python data structure. A `@triton.jit` function is compiled (by the `triton` package, which lowers it to PTX via LLVM) into a static GPU program. You can still write `for i in range(...)` loops, but the loop bounds must be computable at trace time or be scalar runtime values — you cannot build Python lists of tensors inside a kernel, and `tl.arange` is a language-level construct (§3.2).

Below the program level, each program is executed by a set of **warps** — 32-thread SIMT groups. `num_warps=4` at the launch site means 4 warps = 128 threads cooperate on each program's tiles. Register pressure (§7) is measured per thread, so `num_warps` directly controls how many registers each tile costs.

### 3.2 `tl.arange` and tiles: thinking in blocks, not scalars

The unit of work in a Triton kernel is a **tile**: a small dense 2-D block of a tensor (say 64 rows × 64 columns). Instead of writing a loop that touches one element at a time, you load, compute on, and store whole tiles. Two primitives build tiles:

```python
r_idx = tl.arange(0, BLOCK_R)          # 1-D tensor [0, 1, ..., BLOCK_R-1]
s_q_off = q_blk * BLOCK_Q + tl.arange(0, BLOCK_Q)   # absolute row indices of this program's query block
```

`tl.arange(0, N)` produces a 1-D tensor of `N` consecutive integers. Two hard rules:

1. **`N` must be a power of two.** Triton requires it; every kernel in this repo therefore calls `_next_pow2` on its dims before choosing block sizes (see `models/mla_triton.py:_next_pow2` and `models/moe_triton.py:_next_pow2`).
2. You cannot index a tile; you **build 2-D tiles by broadcasting** 1-D index tensors against each other:

```python
kv_2d = k_off[:, None] * stride_kv_s + r_idx[None, :] * stride_kv_r
```

`k_off[:, None]` has shape `(BLOCK_N, 1)`, `r_idx[None, :]` has shape `(1, BLOCK_R)`, and the sum has shape `(BLOCK_N, BLOCK_R)` — the standard NumPy-style broadcasting rule. The `[:, None]` / `[None, :]` idiom is the single most common pattern in these files: **`[:, None]` owns rows, `[None, :]` owns columns.**

### 3.3 `tl.load` / `tl.store` and masks: the boundary contract

A tile load computes a tensor of pointers (base pointer + the offset tensor from §3.2) and reads one element per pointer:

```python
kv_tile = tl.load(
    ctx_kv_ptr + b_id * stride_kv_b + k_off[:, None] * stride_kv_s + r_idx[None, :] * stride_kv_r,
    mask=k_mask[:, None], other=0.0,
)
```

Three things matter here:

- **Strides are explicit.** The kernel is written for *any* memory layout; `stride_kv_b`, `stride_kv_s`, `stride_kv_r` are the strides of `ctx_kv` along batch, sequence, and rank, passed from the host as integers. This is why the host wrapper passes `ctx_kv.stride(0), ctx_kv.stride(1), ctx_kv.stride(2)` at the launch site.
- **`mask=` turns out-of-range reads into `other=` fills instead of faults.** A tile that overhangs the tensor boundary reads garbage addresses — the mask forces those lanes to read `other=0.0`. The kernel then *must* make sure masked lanes never contribute to the result. In the MLA kernel, padded key positions get scores of `-inf` (§3.5), so they contribute probability mass zero — the zeroed `v_tile` rows they multiply are harmless.
- **A masked `tl.store` leaves the destination untouched** at masked lanes. The MLA kernel allocates `out = torch.empty(...)` and stores with `mask=s_q_mask`; rows beyond `S_q` are never written, which is fine because nothing ever reads them.

### 3.4 `tl.dot`: the matmul instruction and its constraints

`tl.dot(a, b)` computes a matrix product `a @ b` in one instruction (lowered to NVIDIA's MMA tensor-core units). Shape contract: `a` is `(M, K)`, `b` is `(K, N)`, result is `(M, N)` — the contraction dimension is the middle one, and Triton will happily multiply `tl.trans` views to get the layout you need (`tl.trans` is free-ish: it is a register-layout change, not a memory round-trip).

The constraints that shape every block-size decision in this repo:

1. **All three dimensions must be ≥ 16.** The tensor cores operate on 16×16×16 MMA fragments; smaller tiles are padded wastefully or rejected. This is why `configs/pretrain_1650_2m.yaml` carries the comments `kv_lora_rank: 16 # must be >= 16 for Triton's tl.dot` and `qk_nope_head_dim: 16 # must be >= 16` — the smoke config was sized to stay legal.
2. **Operands must share a dtype.** The MLA kernel computes `tl.dot(p.to(v_tile.dtype), v_tile)` — the attention probabilities `p` are FP32 but `v_tile` is BF16, so `p` is cast *to BF16* before the dot (a deliberate accuracy tradeoff; the dot's accumulator is still FP32).
3. **The accumulator is FP32.** `tl.dot(a, b)` returns FP32 by default, and `tl.dot(a, b, acc=acc)` accumulates into a caller-supplied FP32 tile. This is the bf16×bf16→fp32 GEMM that the repo's "BF16 autocast, FP32 accumulators" discipline (§8 of [[Docs/08_Training_Pipeline]]) extends all the way down into the kernels. The MoE kernel's inner loop is built on this: `gate_acc = tl.dot(x_tile, tl.trans(w1_tile), acc=gate_acc)` chains K-blocks into one fp32 accumulation.
4. **The `acc=` chain is how you tile a large GEMM.** To multiply a `(BLOCK_T, D)` input by a `(D, I)` weight when `D` exceeds the tile size, loop over D-blocks and accumulate (see §5.3). Without `acc=`, each block's dot would round to fp32 separately and you'd lose the fused accumulation.

### 3.5 Online softmax: the single-pass rescaling trick

Full softmax needs *two* passes over the row (find the max, then exponentiate-and-normalise). FlashAttention's insight — and this kernel's — is that you can fold the max and normaliser into a running state so a **single pass over K** suffices. This is what makes the "stream K-blocks one at a time" loop of §4 possible at all.

The math. Row $i$ of the score matrix has values $s_{i,1}, s_{i,2}, \dots$. After seeing a prefix of blocks, keep three quantities:

- $m_i = \max_j s_{ij}$ — the running max (initially $-\infty$),
- $\ell_i = \sum_j e^{s_{ij} - m_i}$ — the running normaliser (initially 0),
- $o_i = \sum_j e^{s_{ij} - m_i} v_j$ — the running weighted sum of values (initially 0).

The final answer is $o_i / \ell_i$ — and note the exponent is *already* shifted by $m_i$, so both numerator and denominator are numerically tame. When a new block with scores $s'$ arrives, its own max is $m' = \max_j s'_j$. The new global max is $m_{\text{new}} = \max(m_i, m')$, and every previously accumulated term must be re-expressed relative to it. Define the rescale factor

$$\alpha = e^{m_i - m_{\text{new}}} \in (0, 1],$$

then $e^{s - m_i} = \alpha \cdot e^{s - m_{\text{new}}}$, so

$$\ell_{\text{new}} = \ell_i \cdot \alpha + \sum_j e^{s'_j - m_{\text{new}}}, \qquad o_{\text{new}} = o_i \cdot \alpha + \sum_j e^{s'_j - m_{\text{new}}} v'_j, \qquad m_{\text{new}} = \max(m_i, m').$$

Each step subtracts the *latest* max before exponentiating, so no intermediate ever overflows — the classic numerically-stable softmax, but with the shift updated incrementally instead of computed in a pre-pass. The kernel implements exactly these four lines (see §4.4).

### 3.6 Putting it together: the shape of a fused kernel

Here is the skeleton every kernel in this chapter follows — grid over output tiles, load a working set into registers, stream the reduction dimension with online rescaling, store:

```python
# illustrative — the structure of _mla_flash_fwd_kernel, minus MLA specifics
@triton.jit
def flash_attn_skeleton(q_ptr, k_ptr, v_ptr, o_ptr, S, softmax_scale, BLOCK_Q: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)                       # one program per query block
    q_off = pid * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_tile = tl.load(q_ptr + q_off[:, None] * S + tl.arange(0, 64)[None, :])  # (BLOCK_Q, 64)
    m_i = tl.full((BLOCK_Q,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_Q, 64), dtype=tl.float32)
    for n_start in range(0, S, BLOCK_N):         # stream key blocks
        n_off = n_start + tl.arange(0, BLOCK_N)
        k_tile = tl.load(k_ptr + n_off[:, None] * 64 + tl.arange(0, 64)[None, :])
        s = tl.dot(q_tile, tl.trans(k_tile)) * softmax_scale     # (BLOCK_Q, BLOCK_N)
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)                    # §3.5 online update
        acc = acc * alpha[:, None] + tl.dot(p.to(k_tile.dtype), v_tile)
        m_i = m_new
    tl.store(o_ptr + q_off[:, None] * 64 + tl.arange(0, 64)[None, :], acc / l_i[:, None])
```

That is the entire engine. The MLA kernel is this skeleton plus two jobs: materialising `K_nope`/`V` *inside* the loop from the compressed latent (§4.4), and the causal `q_start` masking (§4.5).

`★ Pitfall ─────────────────────────────────────`
`tl.arange` with a non-power-of-2 length fails at compile time with a cryptic error. The repo's universal answer is `_next_pow2`: `BLOCK_R = _next_pow2(R)`, `BLOCK_D = _next_pow2(D)`, and so on — every block size in `models/mla_triton.py` and `models/moe_triton.py` is a power of two computed from a real dimension, and the masked loads of §3.3 absorb the difference between `BLOCK_*` and the true dim. When you extend these kernels, reach for `_next_pow2` first and `mask=` second.
`─────────────────────────────────────────────────`

## 4. Fused MLA Attention Kernel (`models/mla_triton.py`)

### 4.1 The reference path — what the kernel replaces

`models/mla_triton.py:mla_attention_reference` is the pure-PyTorch path used by CPU tests. It is the clearest statement of the MLA arithmetic:

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

The two `einsum`s are the HBM round-trip the kernel eliminates: `K_nope` and `V` are computed, written to HBM, then read back for the score and output matmuls. The *production* SDPA branch inside `models/mla.py:MultiHeadLatentAttention.forward` avoids the four-tensor `cat` by fusing the two up-projections into one batched matmul — same arithmetic, one `torch.cat([wkv_b_k, wkv_b_v], dim=1)`:

```python
ctx_kv_bmm = ctx_kv.reshape(bsz * seqlen_k, self.kv_lora_rank).unsqueeze(0).expand(h, -1, -1)
# Fused: one bmm over ctx_kv produces K_nope and V together.
wkv_b_kv = torch.cat([wkv_b_k, wkv_b_v], dim=1)
KV_nope_h = torch.bmm(ctx_kv_bmm, wkv_b_kv.transpose(-1, -2))
K_nope_h, V_h = KV_nope_h.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
```

Either way, `K_nope` and `V` are materialised as full `(B, H, S_kv, ·)` tensors in HBM before the attention reads them. The fused kernel's whole reason to exist is to skip exactly that.

### 4.2 The fused kernel — FlashAttention-2 style

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

### 4.3 Grid and launch — who runs which tile

The host side of the kernel is `_TritonMlaAttentionFunction.forward`, which chooses the grid, the block sizes, and the launch configuration:

```python
_check_mla_dim_limits(R, D_nope, D_rope, D_v)

BLOCK_Q = 64
BLOCK_N = 64
BLOCK_R = _next_pow2(R)
BLOCK_D_NOPE = _next_pow2(D_nope)
BLOCK_D_ROPE = _next_pow2(D_rope)
BLOCK_D_V = _next_pow2(D_v)
n_q_blocks = (S_q + BLOCK_Q - 1) // BLOCK_Q

out = torch.empty(B, S_q, H, D_v, dtype=q_nope.dtype, device=q_nope.device)

_mla_flash_fwd_kernel[(B, H, n_q_blocks)](
    ...
    BLOCK_Q=BLOCK_Q, BLOCK_N=BLOCK_N,
    num_warps=4, num_stages=2,
)
```

Read this against §3.1: the grid is 3-D with shape `(B, H, n_q_blocks)`, so `program_id(0)` selects the batch element, `program_id(1)` the head, and `program_id(2)` the query block — exactly the three ids the kernel reads at the top (`b_id`, `h_id`, `q_blk`). Query tiling is the only axis that scales with sequence length: at the canonical config, `B=8`, `H=12`, `S_q=2048` gives a grid of `8 × 12 × 32 = 3072` programs, each handling 64 queries against the *whole* key context in `S_kv / 64` iterations.

Two details worth noting at the launch site:

- **`num_stages=2`** asks the compiler to software-pipeline the K-block loop — prefetch the next `ctx_kv` block into shared memory while the current one is being consumed. For a streaming loop like this, pipelining hides HBM latency and is usually worth 20–40% (an estimate, not a measurement — see §9).
- **Block sizes are powers of two ≥ the real dims** (`BLOCK_R = 256` for `R=192`), and the *masks* inside the kernel (not the grid) absorb the padding. The grid is sized off the real `S_q`; the tiles are sized off the padded dims.

### 4.4 Kernel internals — the loop, annotated

Here is the actual body of `_mla_flash_fwd_kernel` with the pieces from §3 called out.

First, the program's own slice of the queries and the per-head up-projections (kept in registers for the whole loop):

```python
s_q_off = q_blk * BLOCK_Q + tl.arange(0, BLOCK_Q)
s_q_mask = s_q_off < S_q
r_idx = tl.arange(0, BLOCK_R)
d_nope_idx = tl.arange(0, BLOCK_D_NOPE)
...
q_nope_tile = tl.load(
    q_nope_ptr + b_id * stride_qn_b + h_id * stride_qn_h
    + s_q_off[:, None] * stride_qn_s + d_nope_idx[None, :] * stride_qn_d,
    mask=s_q_mask[:, None], other=0.0,
)
...
w_k = tl.load(
    wkv_b_k_ptr + h_id * stride_wk_h
    + d_nope_idx[:, None] * stride_wk_n + r_idx[None, :] * stride_wk_r,
    mask=(d_nope_idx[:, None] < D_nope), other=0.0,
)
w_v = tl.load(
    wkv_b_v_ptr + h_id * stride_wv_h
    + d_v_idx[:, None] * stride_wv_v + r_idx[None, :] * stride_wv_r,
    mask=(d_v_idx[:, None] < D_v), other=0.0,
)
```

Note that `q_nope_tile` and `q_pe_tile` are loaded with `s_q_mask` (rows beyond `S_q` get zeros — they will never be stored), while `w_k`/`w_v` are loaded once and stay live across every iteration of the loop below. Then the streaming loop:

```python
for kv_start in range(0, S_kv, BLOCK_N):
    k_off = kv_start + n_idx
    k_mask = k_off < S_kv

    kv_tile = tl.load(
        ctx_kv_ptr + b_id * stride_kv_b
        + k_off[:, None] * stride_kv_s + r_idx[None, :] * stride_kv_r,
        mask=k_mask[:, None], other=0.0,
    )
    k_nope = tl.dot(kv_tile, tl.trans(w_k))
    v_tile = tl.dot(kv_tile, tl.trans(w_v))

    k_pe = tl.load(
        ctx_pe_ptr + b_id * stride_pe_b
        + k_off[:, None] * stride_pe_s + d_rope_idx[None, :] * stride_pe_d,
        mask=k_mask[:, None], other=0.0,
    )

    s_block = (tl.dot(q_nope_tile, tl.trans(k_nope))
               + tl.dot(q_pe_tile, tl.trans(k_pe))) * softmax_scale

    if is_causal:
        causal_mask = (s_q_off[:, None] + q_start >= k_off[None, :])
        s_block = tl.where(causal_mask, s_block, float("-inf"))
    s_block = tl.where(k_mask[None, :], s_block, float("-inf"))

    m_new = tl.maximum(m_i, tl.max(s_block, axis=1))
    alpha = tl.exp(m_i - m_new)
    p = tl.exp(s_block - m_new[:, None])
    l_i = l_i * alpha + tl.sum(p, axis=1)
    acc = acc * alpha[:, None] + tl.dot(p.to(v_tile.dtype), v_tile)
    m_i = m_new

acc = acc / l_i[:, None]
```

Line by line:

1. **`kv_tile = tl.load(...)`** — the compressed latent block `(BLOCK_N, BLOCK_R)` from HBM. This is the *only* K/V data the loop reads from memory (plus the tiny `k_pe` below). Masked lanes (past the end of `S_kv`) load zeros.
2. **`k_nope = tl.dot(kv_tile, tl.trans(w_k))`** — the materialisation, in registers (§3.4: `(64, 256) @ (256, 64) → (64, 64)`; the fp32 accumulator is implicit). Same for `v_tile` with `w_v`. Nothing touches HBM.
3. **`k_pe`** — the RoPE key comes from `ctx_pe`, which is already rotated (`_apply_rope` ran in `MultiHeadLatentAttention.forward` before the kernel is called). "RoPE pre-applied" means the kernel does *not* compute rotations — it just loads the rotated keys. The query side is the same: `q_pe` arrives rotated.
4. **`s_block`** — the score tile: content scores from `q_nope·k_nopeᵀ` plus rope scores from `q_pe·k_peᵀ`, scaled by `softmax_scale`. Splitting the score into two dots is what keeps `D_nope` and `D_rope` as separate small contraction dims instead of one padded 64-wide cat.
5. **The two `tl.where` lines** — masked scores become `-inf`: `causal_mask` enforces *global-position* causality (see §4.5), `k_mask` zeroes out the padded key lanes from step 1. After this, `p = exp(s - m_new)` is exactly zero on every forbidden position, so those rows of `v_tile` (which are zeros anyway) contribute nothing to `acc`.
6. **The four accumulator lines** — verbatim §3.5: rescale the running state by `alpha = exp(m_i - m_new)`, fold in the new block, update the max. `p` is cast to `v_tile.dtype` (BF16) only because `tl.dot` requires matching operand dtypes; the multiply-accumulate itself stays FP32.
7. **`acc / l_i[:, None]`** — the final normalisation, then the store under `s_q_mask`.

### 4.5 `q_start` — the causality contract for cached chunks

The kernel's `is_causal` flag is not the same contract as "square causal matrix". It is called with `is_causal=True` whenever a mask exists, and it compares **global** positions:

```python
if is_causal:
    causal_mask = (s_q_off[:, None] + q_start >= k_off[None, :])
    s_block = tl.where(causal_mask, s_block, float("-inf"))
```

`q_start` is the offset of this query block inside the full KV context. The host computes it in `models/mla.py:MultiHeadLatentAttention._forward_triton`:

```python
# Causal within the current block: queries are offset by `start_pos`
# in the KV context, so pass it as q_start (0 for cache-free).
is_causal = mask is not None
q_start = start_pos if (is_causal and use_cache) else 0
out = triton_mla_attention(
    q_nope=q_nope_k, q_pe=q_pe_k, ctx_kv=ctx_kv, ctx_pe=ctx_pe,
    wkv_b_k=wkv_b_k, wkv_b_v=wkv_b_v,
    softmax_scale=self.softmax_scale,
    is_causal=is_causal, q_start=q_start,
)
```

Why is this needed? With a KV cache, `ctx_kv` spans positions `0 … end_pos-1` but the queries in this call only cover `start_pos … end_pos-1` (a decoded token or a mid-sequence prefill chunk). A *local* causal mask over the block would let query position `start_pos` attend key positions `> start_pos` that exist in the cache — reading the future. The `+ q_start` shift makes the comparison `q_global ≥ k_global`, which is exactly the semantics of the PyTorch-side mask built by `models/transformer.py:Transformer._build_causal_mask`:

```python
q = torch.arange(seqlen, device=device)[:, None] + start_pos
k = torch.arange(kv_len, device=device)[None, :]
mask = torch.where(q >= k, torch.zeros((), device=device),
                   torch.full((), float("-inf"), device=device))
```

Same arithmetic, two implementations: the torch version materialises a `(1, 1, S_q, S_kv)` tensor (cached per `(seqlen, kv_len, start_pos, device)`), the kernel version evaluates the comparison per tile. This contract was the subject of a hard bug: before the `q_start` parameter existed, a cached mid-sequence prefill either crashed (SDPA path) or silently attended future tokens (Triton path) — the fix landed 2026-08-04 and is guarded by `tests/test_mla_triton.py` (GPU-gated) and the reference test `TestMlaAttentionReference.test_output_shape` (CPU).

### 4.6 The register-budget cap

`models/mla_triton.py:_check_mla_dim_limits` hard-fails if `R`, `D_nope`, `D_rope`, or `D_v` exceeds **256**:

```python
for name, val in [("R", R), ("D_nope", D_nope), ("D_rope", D_rope), ("D_v", D_v)]:
    if val > 256:
        raise ValueError(f"triton_mla_attention: {name}={val} exceeds the 256 cap. ...")
```

The canonical config (`R=192, D_nope=48, D_rope=24, D_v=64`) is well under the cap. The cap exists because `w_k`/`w_v` and the score tile all live in registers simultaneously — at `R=192, D_nope=48` the `w_k` tile alone is `48×192` floats, and the BLOCK_Q×BLOCK_N score tile adds more. 256 is the empirical register-budget ceiling before spills dominate (§7 does the per-thread arithmetic). This is also why `moe_dispatch='triton_grouped'` auto-falls-back to `stacked` when `moe_inter_dim > 256` (the gate in `DeepSeekMoE.forward`).

### 4.7 The backward pass — a deliberate stub

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

This is **correct but not optimal**: the recompute is the PyTorch reference, so gradients match the SDPA path exactly (asserted by `tests/test_mla_triton.py`), but it forfeits the fused-kernel speedup in the backward. The comment marks it as a v1 stub with the upgrade path (a fused recompute-backward) documented. For training throughput this is the obvious next optimization; for correctness it is a no-op. (The full re-compute pattern — why it is legal, what it costs — is §6.)

### 4.8 Autograd wiring — `torch.autograd.Function`

`_TritonMlaAttentionFunction.apply(...)` is the seam between the kernel and PyTorch autograd. Forward saves `q_nope, q_pe, ctx_kv, ctx_pe, wkv_b_k, wkv_b_v` for backward (`ctx.save_for_backward`) plus the scalar shapes/scale. The public entry `models/mla_triton.py:triton_mla_attention` raises `ImportError` with a helpful message if `triton` is not installed — caught by `mla.py`'s try/except to trigger the SDPA fallback.

`★ Pitfall ─────────────────────────────────────`
The backward's return tuple must have exactly one entry per forward input, in order. The MLA forward takes 9 arguments (6 tensors + `softmax_scale`, `is_causal`, `q_start`), so backward returns 6 grads (each `None`-guarded with `torch.zeros_like`, because a `None` grad for a non-None input is an autograd error) followed by three `None`s. Get the count wrong and training dies at the first backward with a "function returned an invalid number of gradient tensors" error — one of the most common custom-operator bugs, and one this repo carefully avoids in both kernels (§6).
`─────────────────────────────────────────────────`

## 5. Fused Grouped-GEMM MoE Kernel (`models/moe_triton.py`)

### 5.1 The reference path — the Python loop the kernel replaces

`models/moe.py:DeepSeekMoE._routed_forward_stacked` (the default `stacked` path) is, after the gate and a sort-by-expert-id:

```python
for e in range(E):                       # E = 20 routed experts
    chunk = x_sorted[start:end]           # this expert's tokens
    h = silu(chunk @ w1[e].T) * (chunk @ w3[e].T)   # SwiGLU: 2 GEMMs
    out = h @ w2[e].T                              # 1 GEMM
    y_sorted[start:end] = out * weights[start:end]
```

That is **3 GEMMs × 20 experts = 60 small GEMM launches per MoE layer per step**, each launch paying the fixed kernel-overhead tax. At 16 MoE layers × many steps, launch overhead dominates wall time.

The per-expert arithmetic itself is the standard SwiGLU block (`Expert.forward`): gate $g = x W_1^T$, up $u = x W_3^T$, hidden $h = \mathrm{silu}(g) \odot u$, output $y = h W_2^T$. The gate here is the *expert* SwiGLU gate — do not confuse it with the routing gate `AuxLossFreeGate`; the routing weights multiply the output (§5.4).

### 5.2 The fused grouped-GEMM kernel

`models/moe_triton.py:triton_grouped_moe_dispatch` is the single-launch replacement. The call site (`models/moe.py:DeepSeekMoE._routed_forward_triton`) does the same sort, then:

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

The reference `models/moe_triton.py:grouped_moe_pytorch` is the same arithmetic routed through the sorted layout, used by CPU tests so the kernel's numerics are verified without a GPU. `expert_offsets` is an `(E+1,)` INT64 tensor whose entries are the cumulative token counts per expert (the `start`/`end` of each expert's contiguous slice); the kernel loads `start`/`end` per program directly from it.

### 5.3 The forward kernel — one program per (expert, token-block)

`_grouped_moe_fwd_kernel` runs with grid `(E, n_blocks_per_expert)` where `n_blocks_per_expert = ceil(T / 32)` (computed from `BLOCK_T = 32` at the launch site in `_TritonGroupedMoeFunction.forward`). Each program owns one expert `e` and one block of `BLOCK_T` tokens of that expert's slice:

```python
e = tl.program_id(0)
pid_t = tl.program_id(1)

start = tl.load(offsets_ptr + e)
end = tl.load(offsets_ptr + e + 1)
n_tokens_e = end - start
if n_tokens_e == 0:
    return                      # empty expert: no work, early out

t_off = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
t_mask = t_off < n_tokens_e
d_idx = tl.arange(0, BLOCK_D)
i_idx = tl.arange(0, BLOCK_I)

# gate, up: tile over D, accumulate fp32
gate_acc = tl.zeros((BLOCK_T, BLOCK_I), dtype=tl.float32)
up_acc = tl.zeros((BLOCK_T, BLOCK_I), dtype=tl.float32)
for d_start in range(0, D, BLOCK_D):
    d_off = d_start + d_idx
    d_mask = d_off < D
    x_tile = tl.load(
        x_ptr + (start + t_off)[:, None] * stride_x_t + d_off[None, :] * stride_x_d,
        mask=t_mask[:, None] & d_mask[None, :], other=0.0,
    )
    w1_tile = tl.load(
        w1_ptr + e * stride_w1_e + i_idx[:, None] * stride_w1_i + d_off[None, :] * stride_w1_d,
        mask=d_mask[None, :], other=0.0,
    )
    w3_tile = tl.load(
        w3_ptr + e * stride_w3_e + i_idx[:, None] * stride_w3_i + d_off[None, :] * stride_w3_d,
        mask=d_mask[None, :], other=0.0,
    )
    gate_acc = tl.dot(x_tile, tl.trans(w1_tile), acc=gate_acc)
    up_acc = tl.dot(x_tile, tl.trans(w3_tile), acc=up_acc)

h = tl.sigmoid(gate_acc) * gate_acc * up_acc
```

Three things to notice:

- **The D-loop is the §3.4 `acc=` pattern.** `x_tile` is `(BLOCK_T, BLOCK_D)`, `w1_tile` is `(BLOCK_I, BLOCK_D)`, and each `tl.dot(x_tile, tl.trans(w1_tile), acc=gate_acc)` accumulates one D-block's contribution into the FP32 `gate_acc` tile. So the forward *does* tile over `D` — the cap is not about the loop, it is about the tile shapes (§5.7).
- **`h = tl.sigmoid(gate_acc) * gate_acc * up_acc`** is silu without a separate `exp`: $\mathrm{silu}(g) = g \cdot \sigma(g)$, computed from the sigmoid. No `tl.exp` needed, and the product is element-wise.
- **The token mask is only applied on loads and stores.** `gate_acc`'s padded token rows (beyond `n_tokens_e`) accumulate zeros from masked loads, so the padded rows of `h` are garbage-free: zero inputs → zero outputs. The store below re-applies `t_mask`.

Then the output GEMM and store:

```python
w2_tile = tl.load(
    w2_ptr + e * stride_w2_e + d_idx[:, None] * stride_w2_d + i_idx[None, :] * stride_w2_i,
    mask=(d_idx[:, None] < D) & (i_idx[None, :] < I), other=0.0,
)
# `h` is fp32 (from gate_acc/up_acc SwiGLU accumulators). Cast to the
# input dtype so the final bf16·bf16 dot has matching operand dtypes.
h_typed = h.to(x_ptr.dtype.element_ty)
out_acc = tl.dot(h_typed, tl.trans(w2_tile))

tl.store(
    y_ptr + (start + t_off)[:, None] * stride_y_t + d_idx[None, :] * stride_y_d,
    out_acc.to(y_ptr.dtype.element_ty),
    mask=t_mask[:, None] & (d_idx[None, :] < D),
)
```

`w2_tile` is `(BLOCK_D, BLOCK_I)` loaded as one tile — legal because `BLOCK_I` is the *full* I (the cap guarantees `BLOCK_I ≤ 256`), so the output GEMM is a single `(BLOCK_T, BLOCK_I) @ (BLOCK_I, BLOCK_D)` dot. Note the explicit `h_typed` cast: `h` is FP32 (it grew out of the FP32 accumulators), and `tl.dot` demands matching operand dtypes, so `h` is rounded to BF16 right here — the one rounding step the SwiGLU path performs beyond the reference's own bf16 matmuls. `out_acc` accumulates in FP32, then is rounded back to BF16 at the store.

### 5.4 The gate-weight-multiply-outside design

The kernel stores the **unweighted** expert output. The routing weight is applied *after* the autograd boundary, back in Python:

```python
def triton_grouped_moe_dispatch(x_sorted, w1, w2, w3, sorted_weights, expert_offsets):
    ...
    out = _TritonGroupedMoeFunction.apply(x_sorted, w1, w2, w3, expert_offsets)
    return out * sorted_weights.unsqueeze(-1)
```

and the kernel's own docstring states the contract: *"Returns UNWEIGHTED out = h @ w2^T; the gate-weight multiply is done outside the autograd Function so the gate receives a gradient."* Why does this matter? The routing weights come from `models/moe.py:AuxLossFreeGate.forward`:

```python
scores = F.linear(x, self.weight).sigmoid()
biased = scores + self.bias.to(scores.dtype)
indices = biased.topk(self.topk, dim=-1)[1]       # argmax — non-differentiable
weights = scores.gather(1, indices)
weights = (weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-10) * self.route_scale).to(x.dtype)
```

`scores` is a differentiable function of the gate's `weight` parameter — that is the only route by which the load-balancing gradient (and hence the aux-loss-free training signal) reaches the gate. `indices` is an `argmax` (integer, no gradient). The routed output must therefore be `out * weights`, with `weights` a live differentiable tensor. If the multiply happened *inside* the `Function` and the kernel swallowed `sorted_weights` as a plain input, the gate gradient would die at the boundary. By applying the multiply outside, autograd sees `mul(weight, kernel_out)` and the gradient flows: `∂/∂gate_weight` gets `scores' · (…routing math…) · out`. The backward that the kernel *does* implement (`dx`, `dw1/2/3`) therefore only sees the already-scaled `dy` and never touches the routing weights.

### 5.5 The backward dx kernel — recompute, then chain rule

`_grouped_moe_bwd_dx_kernel` computes `dx = ∂y/∂x` for one `(expert, token-block)` program, with the same grid as the forward. It needs `dy` (already scaled by the gate weight — see §5.4), `x`, and the weights, and it must re-derive the intermediate activations because they were not saved:

```python
# Re-compute gate_acc, up_acc
gate_acc = tl.zeros((BLOCK_T, BLOCK_I), dtype=tl.float32)
up_acc = tl.zeros((BLOCK_T, BLOCK_I), dtype=tl.float32)
for d_start in range(0, D, BLOCK_D):
    ...                                        # identical D-loop to the forward

sig_g = tl.sigmoid(gate_acc)
silu_g = sig_g * gate_acc
h = silu_g * up_acc

dy_tile = tl.load(...).to(tl.float32)          # (BLOCK_T, BLOCK_D), fp32
w2_tile = tl.load(...)                          # (BLOCK_D, BLOCK_I)
dh = tl.dot(dy_tile, w2_tile)

# silu'(g) = sig(g) * (1 + g*(1 - sig(g)))
dsilu = dh * up_acc
dgate_pre = dsilu * sig_g * (1.0 + gate_acc * (1.0 - sig_g))
dup = dsilu * silu_g
```

The chain rule here is worth checking by hand. With $h = \mathrm{silu}(g) \odot u$, we get $\partial h / \partial u = \mathrm{silu}(g)$ and $\partial h / \partial g = \mathrm{silu}'(g) \odot u$, where $\mathrm{silu}'(g) = \sigma(g)\bigl(1 + g(1 - \sigma(g))\bigr)$ — the code's comment. Since `dh = dy @ w2` is the gradient of the loss w.r.t. $h$, the code computes `dsilu = dh * up_acc` (the $u$-contribution to the silu input gradient), then `dgate_pre = dsilu · silu'(g)` and `dup = dsilu · silu(g)`. Finally:

```python
d_b = tl.arange(0, BLOCK_D)
w1_re = tl.load(
    w1_ptr + e * stride_w1_e + i_idx[:, None] * stride_w1_i + d_b[None, :],
    mask=(i_idx[:, None] < I) & (d_b[None, :] < D), other=0.0,
)
w3_re = tl.load(...)                             # same shape for w3
dx = tl.dot(dgate_pre, w1_re) + tl.dot(dup, w3_re)
```

$dx = dg \, W_1 + du \, W_3$ (note: no transpose — the dots are `(BLOCK_T, BLOCK_I) @ (BLOCK_I, BLOCK_D)`, the "reverse" GEMM of the forward's `W^T` layout).

`★ Pitfall ─────────────────────────────────────`
`w1_re`/`w3_re` are loaded with `+ d_b[None, :]` — **no `stride_w1_d` term**. This hard-codes a unit stride along the last (D) axis, i.e. row-major contiguous weights. It is safe today only because `DeepSeekMoE.forward` re-stacks the experts with `torch.stack(...)` every forward, which always yields contiguous tensors (§10). If anyone ever passed a transposed or sliced weight tensor to `triton_grouped_moe_dispatch`, this load would silently read wrong memory. The forward kernel and the dw kernel carry the full stride terms; the dx kernel does not — worth fixing when the kernel is next touched.
`─────────────────────────────────────────────────`

### 5.6 The backward dw kernel — one program per expert, no atomics

`_grouped_moe_bwd_dw_kernel` computes `dw1`, `dw2`, `dw3` with grid `(E,)` — one program owns an expert's entire gradient tensor. It loops over I-blocks, D-blocks, and token-blocks in nested `for` loops, accumulating into local FP32 tiles:

```python
for i_start in range(0, I, BLOCK_I):
    for d_start in range(0, D, BLOCK_D):
        dw1_local = tl.zeros((BLOCK_I, BLOCK_D), dtype=tl.float32)
        dw3_local = tl.zeros((BLOCK_I, BLOCK_D), dtype=tl.float32)
        dw2_local = tl.zeros((BLOCK_D, BLOCK_I), dtype=tl.float32)
        for t_start in range(0, n_tokens_e, BLOCK_T):
            ...   # re-compute gate_acc, up_acc, h for this token block (full D loop)
            dy_tile = tl.load(...).to(tl.float32)       # (BLOCK_T, BLOCK_D)
            w2_tile = tl.load(...)                       # (BLOCK_D, BLOCK_I)
            dh = tl.dot(dy_tile, w2_tile)
            dsilu = dh * up_acc
            dgate_pre = dsilu * sig_g * (1.0 + gate_acc * (1.0 - sig_g))
            dup = dsilu * silu_g
            x_tile = tl.load(...)                        # (BLOCK_T, BLOCK_D)
            ...
            dw1_local += tl.dot(tl.trans(dgate_masked), x_tile)   # (I, D) += (I,T)@(T,D)
            dw3_local += tl.dot(tl.trans(dup_masked), x_tile)
            dw2_local += tl.dot(dy_t, h_masked)                   # (D, I) += (D,T)@(T,I)
        tl.store(dw1_ptr + ..., dw1_local, mask=mask_1d)   # one store per (i,d) block
        tl.store(dw3_ptr + ..., dw3_local, mask=mask_1d)
        tl.store(dw2_ptr + ..., dw2_local, mask=mask_2d)
```

The design note in the source is explicit: **"One program per expert. No atomics: each program owns its (I, D) gradient tile and writes directly after the token loop."** Atomics (scatter-add) are the usual nightmare of gradient accumulation over a routed layout; the sorted layout makes each expert's gradient a private dense accumulation, so the kernel writes each `(i, d)` block exactly once. The mask gymnastics around the dots (`i_mask[None, :]`, `t_i_mask`, `d_t_mask`) exist because the padded lanes of `dgate_pre`, `h`, and `dy_tile` are garbage-free *zeros* — the masked loads guarantee it — and the dots must not let any padded lane contribute. The host allocates the dw tensors as **FP32** (`torch.zeros(E, I, D, dtype=torch.float32)`) and the backward returns them cast to the weight dtype:

```python
return (dx, dw1.to(w1.dtype), dw2.to(w2.dtype), dw3.to(w3.dtype), None)
```

so the kernels accumulate in full FP32 and only the final return rounds to BF16 — consistent with the repo-wide FP32-accumulator discipline.

### 5.7 The `D ≤ 256` cap — and why I-tiling / dh-accumulation is the open constraint

`models/moe_triton.py:_check_dim_limits` guards the launch:

```python
def _check_dim_limits(I: int, D: int) -> None:
    """Hard-fail if I or D exceeds the 256 register-pressure cap."""
    if I > 256 or D > 256:
        raise ValueError(
            f"triton_grouped_moe_dispatch: BLOCK_I=ceil_pow2({I}) and "
            f"BLOCK_D=ceil_pow2({D}) must each be ≤ 256. Got I={I}, D={D}. "
            "For larger dims, fall back to `moe_dispatch='stacked'`."
        )
```

The canonical config has `I = moe_inter_dim = 384` and `D = dim = 768`, so `BLOCK_I = 512` and `BLOCK_D = 1024` — both blow the cap and `models/moe.py:DeepSeekMoE.forward`'s `except ValueError` drops to `stacked`. The smoke config (`I=32, D=64`) is comfortably inside.

Why are the *tile* sizes the binding constraint rather than the loops?

- **Forward / dx:** `i_idx = tl.arange(0, BLOCK_I)` covers the *full* I in one tile — `gate_acc`, `up_acc`, `h`, `dgate_pre`, `dup` are all `(BLOCK_T, BLOCK_I)` — and the weight tiles are `(BLOCK_I, BLOCK_D)`. With `BLOCK_I = 512` and `BLOCK_D = 1024`, a single `w1_tile` load would be `512 × 1024 = 524,288` elements: ~4,096 registers per thread at `num_warps=4` (§7). The forward already tiles D with an accumulator loop, but it has no I-loop; lifting the cap means adding an outer `for i_start` loop that accumulates `gate_acc`/`up_acc` across I-blocks and, in dx, produces `dgate_pre`/`dup` per I-tile.
- **dw:** the dw kernel *already* loops over both I and D, so tiling alone is not the issue — but its `dh` is computed from a single D-block: `dh = tl.dot(dy_tile, w2_tile)` where `dy_tile`/`w2_tile` cover only the current `d_start` block. The true gradient needs `dh = dy @ w2` summed over **all** D: `dgate_pre[t, i] = Σ_d dy[t, d] · w2[d, i]`. With D split across blocks, `dh` must be an accumulator over the D-loop (`tl.zeros` + `acc=` chained dots), and it currently is *not*. [INFERENCE: this is a latent numerical bug, not a crash — with `D ≤ BLOCK_D` (single D-block, as today) the single dot is exact, so every currently reachable run is correct. If the cap were lifted and D-tiling enabled without fixing `dh`, the gradients would silently be partial sums.]

So the open constraint for running this kernel at the canonical config is a *three-part* change: an I-tiling loop in fwd/dx, a D-accumulating `dh` in dw, and (bonus) the unit-D-stride fix from §5.5. None of it is algorithmically hard — it is the classic register-budget engineering — but it is real work with correctness risk, which is why the repo ships the honest fallback instead.

## 6. The `autograd.Function` re-compute pattern

Both kernels are wrapped in `torch.autograd.Function` subclasses, and both use the same memory-vs-time strategy in the backward: **recompute the forward instead of saving its outputs**. This section is the pattern itself; §4.7 and §5.5–5.6 show the two concrete instantiations.

The generic shape of a custom kernel's autograd contract:

```python
class _CustomFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *inputs):
        out = custom_kernel(*inputs)          # the fused launch
        ctx.save_for_backward(*inputs)        # save INPUTS, not activations
        ctx.scalar_meta = ...                 # any non-tensor metadata
        return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        inputs = ctx.saved_tensors
        # re-derive what the forward computed, then autograd through it
        ref = reference_forward(*inputs)
        grads = torch.autograd.grad(ref, inputs, grad_outputs=grad_outputs[0])
        return tuple(g if g is not None else torch.zeros_like(t) for t in inputs) + (None,)*n_scalars
```

The design choices, and why:

- **Save inputs, not activations.** Saving the attention weights $P$ (`BLOCK_Q × BLOCK_N` per program — the whole `(B, H, S_q, S_kv)` tensor) or the SwiGLU hidden `h` would blow up activation memory: for MLA at the canonical shape that is `B·H·S_q·S_kv` fp32 elements, tens of gigabytes. Saving the *inputs* (the compressed `ctx_kv`, the per-head `wkv_b`, the weight stack) costs `B·S_kv·R` plus weights — the MLA cache size itself. The price is recomputation: the backward re-runs the reference forward, trading FLOPs for HBM.
- **The recompute is the PyTorch reference, not the kernel.** For MLA, `backward` calls `mla_attention_reference` and `torch.autograd.grad` over it. This is what makes the stub *correct by construction*: the gradient of the kernel's forward equals the gradient of the reference (the kernel computes the same arithmetic, modulo BF16 rounding), so no per-op derivative math needs to be hand-derived or validated. `tests/test_mla_triton.py` asserts the numerics agree with the SDPA path.
- **`None`-grad discipline.** Autograd requires one return value per forward input. Any input whose gradient is `None` (e.g. an unused branch under `allow_unused=True`) must be returned as `torch.zeros_like(t)`, because returning `None` for a tensor input is an error. The MLA backward returns 9 values (6 tensor grads + `None, None, None` for the three scalar args); the MoE backward returns 5 (`dx`, three dw casts, and `None` for `expert_offsets`, which is an INT64 tensor that never takes a gradient).
- **The MoE variant is the "real" re-compute pattern.** Rather than falling back to a Python reference, its backward re-runs the *fused* kernels (`_grouped_moe_bwd_dx_kernel` and `_grouped_moe_bwd_dw_kernel`) with the activations recomputed in-register (§5.5–5.6). This is the pattern's endgame: recompute inside the kernel, in registers, where the data already lives.

When is the re-compute pattern the right call? The tradeoff is recompute FLOPs against activation bytes. For attention, the saved `P` matrix grows quadratically in sequence length while the recompute grows linearly (each K-block re-materialised once) — re-compute wins decisively at long context. For the MoE, the activations are `(T, I)` per expert — small at smoke scale — but saving them would force a second HBM round-trip; recompute-in-registers avoids it entirely. The repo's choice of *saving inputs + recomputing* is the standard answer for both.

`★ Insight ─────────────────────────────────────`
There is a subtle interplay between the re-compute pattern and grad checkpointing (`use_checkpoint=True` in `Transformer.__init__`). The MLA kernel backward re-runs `mla_attention_reference` inside autograd; if the *layer* forward was also checkpointed, the checkpoint re-runs the whole layer forward (including the kernel launch) *and then* the kernel's backward re-runs the reference — two recomputes for the price of one. Still correct (checkpointing just re-executes the graph), still memory-cheap, but a reminder that "recompute" costs stack. A fused MLA backward (§4.7's documented upgrade path) would remove the second one.
`─────────────────────────────────────────────────`

## 7. Register-pressure math: where the 256 cap comes from

Every cap in this chapter — `_check_mla_dim_limits` (R, D_nope, D_rope, D_v ≤ 256) and `_check_dim_limits` (I, D ≤ 256) — is a register-budget decision. Here is the arithmetic that motivates it. All figures in this section are **estimates** ([INFERENCE] — no profiler has run on these kernels; see §9): the numbers are per-thread tile elements, which is the right order-of-magnitude proxy for register allocation, not the compiler's actual allocation.

Hardware facts: on Volta-and-newer NVIDIA GPUs (including the A100), each thread has **255 addressable 32-bit registers** (the compiler may spill beyond that to local memory, which is L1/L2-backed and *much* slower than registers), and each SM has 64K registers total. A program launched with `num_warps=4` runs 128 threads per CTA, so a tile of $N$ elements costs roughly $N/128$ registers per thread.

**MLA kernel, canonical config** (`BLOCK_Q=64, BLOCK_N=64, BLOCK_R=256, num_warps=4`):

| Resident tile | Shape | Elements | ≈ regs/thread |
|---|---|---|---|
| `w_k` (kept across loop) | 64 × 256 | 16,384 | 128 |
| `w_v` (kept across loop) | 64 × 256 | 16,384 | 128 |
| `kv_tile` (per iteration) | 64 × 256 | 16,384 | 128 |
| `k_nope`, `v_tile` | 64 × 64 each | 8,192 | 64 |
| `q_nope_tile`, `q_pe_tile` | 64×64 + 64×32 | 6,144 | 48 |
| `s_block`, `p` | 64 × 64 each | 8,192 | 64 |
| `acc` | 64 × 64 | 4,096 | 32 |
| `k_pe` | 64 × 32 | 2,048 | 16 |
| **Total** | | | **≈ 600** |

≈ 600 registers/thread against a 255-register ceiling: the compiler *will* spill at canonical dims, and spilled tiles round-trip through local memory — the very HBM traffic the kernel exists to avoid, now on the wrong side of L1. The `R ≤ 256` cap exists precisely to bound `w_k`, `w_v`, and `kv_tile` (the three biggest rows); at `R=192` those three alone are 384 regs/thread. [INFERENCE: this suggests the MLA kernel at canonical dims is register-heavy even under the cap — spilling is plausible but unmeasured; a first GPU run should check `pynvml`/`ncu` occupancy and spill counts before trusting the speedup.]

**MoE forward, smoke config** (`BLOCK_T=32, BLOCK_I=32, BLOCK_D=64, num_warps=4`):

| Resident tile | Shape | Elements | ≈ regs/thread |
|---|---|---|---|
| `x_tile`, `w1_tile`, `w3_tile` | 32×64 each | 6,144 | 48 |
| `gate_acc`, `up_acc` | 32×32 each | 2,048 | 16 |
| `h`, `out_acc` | 32×64, 32×64 | 4,096 | 32 |
| `w2_tile` | 64 × 32 | 2,048 | 16 |
| **Total** | | | **≈ 112** |

Comfortable — and exactly why the smoke config runs the kernel while the canonical one cannot. Now force the canonical dims: `BLOCK_I = 512`, `BLOCK_D = 1024` make a single `w1_tile` `512 × 1024 = 524,288` elements ≈ 4,096 regs/thread — more than the *entire* SM register file. That is the concrete meaning of the cap: it is not a policy, it is the tile arithmetic (which is why the fix is I-tiling + D-accumulation, §5.7, not a bigger cap).

Two consequences for kernel authors:

1. **Block sizes trade against resident tiles.** Bigger `BLOCK_Q`/`BLOCK_N` means fewer loop iterations and better data reuse, but the `acc`, `s_block`, `p`, and `kv_tile` rows all grow linearly. `BLOCK_Q=64`/`BLOCK_N=64` in the MLA kernel is a middle point; 128-wide blocks would halve the iterations and double the pressure.
2. **`num_warps` is a pressure dial too.** More warps = more threads to spread a tile across = fewer registers per thread, at the cost of more synchronization. Raising `num_warps` from 4 to 8 roughly halves the per-thread numbers in both tables — the first lever to pull if a kernel spills.

## 8. Autotune guidance

Neither kernel currently uses `@triton.autotune` — the launch sites hard-code `BLOCK_Q/BLOCK_N = 64`, `BLOCK_T = 32`, `num_warps=4`, `num_stages=2`. That is a deliberate choice for a single-model repo with fixed canonical shapes (autotuning would pay a compile-time sweep on every new shape for a kernel that runs at exactly two configurations). But the *method* is worth knowing, and both kernels are structured to accept it.

The standard pattern:

```python
# illustrative — how to autotune a launch like _mla_flash_fwd_kernel
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_Q": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 64, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
    ],
    key=["S_q", "S_kv", "R", "D_nope", "D_rope", "D_v"],   # cache key per shape
)
@triton.jit
def _mla_flash_fwd_kernel(...):
    ...
```

Rules of thumb, mapped onto these kernels:

- **`key=` must cover every shape-dependent dimension.** Autotune benchmarks each config once per distinct key tuple and caches the winner (on disk under `~/.triton/cache`). If `S_q`/`S_kv` are absent from the key, a long-context run silently reuses a short-context config. For the MLA kernel the honest key is `["S_q", "S_kv", "R", "D_nope", "D_rope", "D_v"]`; for the MoE kernel `["T", "D", "I", "E"]` (token count matters: `n_blocks_per_expert` changes the grid).
- **Bounded config lists, not grids.** Each extra config is a compile + benchmark at first use; 4–8 candidates is the sweet spot. Since the two real configurations here are fixed (canonical and 1650), a hand-picked pair per shape beats a sweep.
- **`num_stages` pipelines the streaming loops.** The MLA K-loop and the MoE D-loop both stream HBM; `num_stages=2` prefetches one block ahead. Higher stages hide more latency but consume shared memory linearly — at `BLOCK_R=256` the MLA kernel has little shared-memory headroom, so 2 is a sane default.
- **`num_warps` interacts with register pressure (§7).** Tuning order that respects the cap arithmetic: first fix block sizes so the resident-tile table fits, then sweep `num_warps`/`num_stages`, then revisit block sizes.
- **Watch the first-call tax.** `@triton.autotune` benchmarks on the *first* invocation per shape — that latency lands inside the first training step unless you warm the kernel up. A warmup call with the real shapes (or a `TRITON_CACHE_DIR` shared across runs) keeps step-time benchmarks honest (§9).
- **Autotune belongs on the host wrapper, not inside `forward`.** Both `triton_mla_attention` and `triton_grouped_moe_dispatch` are thin Python entry points — the right place for the `@triton.autotune` decoration is the kernel function itself, so the Function's `forward` keeps launching the same `(grid)(...)` call with tuned constants resolved at trace time.

## 9. Benchmark methodology — what is measured, what is estimated

Honest status first: **no GPU run of this repo has executed yet.** `.benchmarks/` is empty; the A100 figures in AGENTS.md and in this chapter are budgets and estimates. The kernel speedup targets (`≥1.5×` MoE, per AGENTS.md) are *aspirations*, not measurements, and the MoE one is structurally blocked at the canonical config anyway (§5.7 — the kernel never runs there). This section documents the tooling that exists and the methodology that will make the numbers trustworthy when the first A100 session happens.

**What exists today.** `scripts/microbench_a100.py:main` measures **peak VRAM**, not speed. It builds the canonical model with `use_checkpoint=True`, prints the estimator's prediction, runs one forward + backward, and compares:

```python
est = estimate_model_memory_gb(m, seq_len=seq, batch_size=bs, grad_checkpoint=True)
assert_fits_in_available_gpu(est, safety_margin_gb=2.0)
print("Running forward + backward ...")
torch.cuda.reset_peak_memory_stats()
x = torch.randint(0, cfg["model"]["vocab_size"], (bs, seq), device="cuda")
y = m(x)
y.sum().backward()
measured = torch.cuda.max_memory_allocated() / 1024**3
```

`utils/memory.py:estimate_model_memory_gb` sums weights (BF16, 2 B/param), optimizer state (FP32 AdamW: 12 B/param), the MLA KV cache, activation blocks (halved by grad checkpointing), and a context-overhead term; `utils/memory.py:assert_fits_in_available_gpu` raises if the estimate exceeds `available − margin`. The script then reports `delta vs estimate` and warns if the peak is within 8 GB of the 80 GB ceiling. So the repo's one GPU script answers "does the 411.6M model fit on an A100?" — it does not time anything.

**Throughput instrumentation.** The training loop reports `tokens_per_sec` in `utils/logging.py:TrainingLogger.log`:

```python
elapsed = max(time.time() - self._step_start, 1e-6)
tokens_per_sec = (self.log_interval * self.seq_len * self.batch_size) / elapsed
```

That is wall-clock time over a rolling window of `log_interval` steps (forward + backward + optimizer + data loading all included), reported every `log_interval` steps — a good end-to-end number, deliberately not a kernel number.

**What a proper kernel benchmark needs (methodology, not yet executed):**

1. **Synchronize before timing.** `torch.cuda.synchronize()` before `time.perf_counter()`; kernel launches are async and the first `stop` would otherwise measure almost nothing. The repo's `.item()` discipline (deferred to the logger to avoid per-step syncs) already respects this in spirit.
2. **Warmup + median.** Run each configuration several times, discard the first (compile/cache-fill), take the median — `triton.testing.do_bench` implements exactly this and is the standard tool for kernel-level comparisons.
3. **Compare like for like.** The honest MLA benchmark is `triton_mla_attention` vs the `sdpa` branch of `MultiHeadLatentAttention.forward` on identical `q_nope/q_pe/ctx_kv/ctx_pe/wkv_b` inputs, *including* the reference-backward cost of the stub (§4.7) — training throughput pays it every step. The honest MoE benchmark is the fused dispatch vs `_routed_forward_stacked` at the *1650* config (the canonical one can't run the kernel).
4. **Count launches, not just FLOPs.** The stacked MoE path issues 60 small GEMMs per layer per step plus sort/index ops; the fused path issues 3 kernel launches (fwd + dx + dw). At these shapes launch overhead and HBM traffic dominate — report launch counts alongside wall time.
5. **Report MFU only with a reference FLOP count.** The canonical config's per-step FLOPs are dominated by the 16 MoE layers' `3·T·I·D·2` per activated expert; without a measured time, MFU claims are arithmetic, not evidence.
6. **Isolate the compiler cache.** First-call compile time (seconds per kernel) must not pollute steady-state numbers; warm up or set `TRITON_CACHE_DIR`.

When the first A100 run lands, the honest deliverable is a table: per-path step time, tokens/sec, peak VRAM (measured vs the §4 estimate), kernel launch counts, and the MLA kernel's spill/occupancy report (which would also resolve the §7 [INFERENCE] about register pressure at canonical dims).

## 10. Re-stacking: a correctness subtlety both kernels share

Both `DeepSeekMoE.forward` and `_shared_forward` **re-stack the expert weights every forward**:

```python
# Re-stack every forward: caching across steps leaves stale copies
# after optimizer.step() (experts would be frozen at init values).
self._stacked_w1 = torch.stack([ex.w1.weight for ex in self.experts], dim=0)...
```

The comment is the lesson: stacking once and caching would capture the *initial* weights; after `optimizer.step()` the live `nn.Linear` weights move but the cached stack would not, silently freezing the experts. Re-stacking every forward is a few-millisecond tax that prevents a catastrophic correctness bug. This is the kind of invariant that's invisible in the forward pass but would make a training run diverge with no obvious cause — and it is exactly what the "never cache optimizer-adjacent tensors across steps" discipline catches. The fused kernel inherits the same invariant through `_routed_forward_triton`'s `w1=self._stacked_w1, ...`, and the dx kernel's unit-stride assumption (§5.5) depends on `torch.stack` producing contiguous tensors.

## 11. When to use which path

| Path | When | Why |
|---|---|---|
| `sdpa` / `stacked` (default) | CPU dev, tests, any default-config run | correct, readable, no Triton dependency |
| `triton` / `triton_grouped` | A100/H100 training with `ENABLE_TRITON_KERNELS=1` | fewer launches, less HBM traffic, higher MFU |

The choice is per-config and reversible at runtime — the fallback catches make it impossible to land in a broken state: a missing `triton` package or a dim over the cap drops to the PyTorch path with one warning, not a crash. And remember the full decision chain from §0: `triton` for MLA works at the canonical config; `triton_grouped` for MoE only at the 1650 smoke config; both only with the env var; both only on Linux+CUDA.

## 12. Check your understanding

1. **Why does the MLA kernel load `wkv_b_k`/`wkv_b_v` once outside the K-block loop, and what would change if they were loaded inside?** They are per-head up-projections that every K-block needs; loading once keeps them in registers across all iterations, which is the entire point (materialising `K_nope`/`V` in registers, never in HBM). Loaded inside the loop they would re-fetch `2 × 64 × 256` elements per iteration from HBM — the avoidable traffic the kernel exists to remove (§4.2, §4.4).
2. **The kernel's causal test is `s_q_off + q_start >= k_off`. When is `q_start` nonzero, and what goes wrong if it were always 0?** `q_start = start_pos` whenever a KV cache is in use and a mask exists (cached decode or mid-sequence prefill), 0 otherwise. With `q_start = 0` on a cached run, query positions near the cache end would attend keys *after* them in global time that already sit in the cache — a silent future-leak (§4.5, `MultiHeadLatentAttention._forward_triton`).
3. **The MoE forward kernel stores the unweighted expert output; the routing-weight multiply happens in `triton_grouped_moe_dispatch`. Why not inside the kernel?** Because `sorted_weights` must remain a differentiable node on the autograd graph: `out * weights` outside the `Function` lets `∂L/∂weights` flow back through `AuxLossFreeGate.forward`'s sigmoid scores. If the multiply were inside the kernel as a plain input, the gate would receive no gradient and the aux-loss-free load balancing would never train (§5.4).
4. **The dw kernel computes `dh = tl.dot(dy_tile, w2_tile)` once per D-block and never accumulates it. When is this a bug?** Today it is never a bug: `_check_dim_limits` guarantees `D ≤ 256`, so `BLOCK_D = next_pow2(D)` covers all of D and the single dot is the full `dy @ w2`. It becomes a silent-gradient bug the moment someone tiles D (canonical `D=768` requires it) without making `dh` an accumulator over the D-loop (§5.7).

> **Next:** [[Docs/13_Portfolio_Comparison]] — positioning DeepSeek-V3-Lite against GPT-OSS-Lite, LLaMA-3-Lite, and Mamba-3-Lite.




<!-- docs:verified 2026-08-04 · 59aeef3 -->
