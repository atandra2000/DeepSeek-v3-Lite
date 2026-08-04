# G3 — Triton Kernel Development Guide

> How to write, extend, gate, test, and benchmark custom Triton kernels in this repo. Covers the contract every kernel must satisfy, the register-budget math behind the 256-element caps, the two sanctioned kernels' known extension points, and the end-to-end recipe for adding a new one.

**Depends on:** [[Docs/12_Triton_Kernels]] (the tutorial), [[Docs/04_DeepSeekMoE]], [[Docs/03_Multi_Head_Latent_Attention]] · **Read next:** [[Docs/guides/G4_benchmarking]]

**Source files:** `models/mla_triton.py`, `models/moe_triton.py`, `models/_triton_dispatch.py`, `models/mla.py`, `models/moe.py`, `models/transformer.py`, `tests/test_mla_triton.py`, `tests/test_moe_triton.py`, `scripts/microbench_a100.py`, `scripts/step_time_a100.py`

---

## 0. 60-second summary

This repo has exactly two sanctioned custom Triton kernels: fused MLA attention (`models/mla_triton.py:triton_mla_attention`) and fused grouped-GEMM MoE dispatch (`models/moe_triton.py:triton_grouped_moe_dispatch`). Both are **opt-in** (config key + `ENABLE_TRITON_KERNELS=1`), both ship a pure-PyTorch reference for CPU tests, both wrap the kernel in a `torch.autograd.Function` with FP32 accumulators, and both are gated by a 256-element register cap. Before any other component gets a kernel, AGENTS.md hard rule 1 requires updating AGENTS.md and `docs/12_Triton_Kernels.md` — and rule 2 requires a measured ≥1.5× speedup over the raw-PyTorch path before it may become a default. This guide is the procedural companion to that contract: what the rules are, where the caps come from, how to extend the two existing kernels, and the step-by-step recipe for a third.

## 1. Why this guide exists

Custom kernels are the most expensive code to get wrong in this repo: a fused kernel's failure modes are silent (wrong registers, spilled tiles, a forgotten mask lane), it only runs on Linux+CUDA, and the CPU test suite can never execute it. The repo already encodes the answer to "what does a correct kernel look like" in AGENTS.md and in the two existing kernel files — but that knowledge is scattered across a contract file, a tutorial chapter, and 875 lines of kernel code. This guide collects the rules into one procedural checklist, shows the arithmetic that makes a kernel legal or illegal, and gives a copy-paste path for adding the third kernel.

Everything in here is checkable against the repo: every rule cites the file that enforces it, every dim is the real config dim, and every speed figure is labeled an **estimate** because no GPU run has executed yet (`.benchmarks/` is empty).

## 2. Should this kernel exist at all? — decision tree

The repo's default is raw PyTorch. Before writing any kernel, walk this tree:

```
Is the component one of the two sanctioned hot paths (MLA materialisation,
MoE routed dispatch)? ──no──> STOP. Adding a third requires updating
                              AGENTS.md hard-rule-1 list + docs/12 first.
        │ yes
        ▼
Is it on a real hot path (runs per layer per step, dominates HBM traffic
or launch count)? ──no──> Keep raw PyTorch. Correctness > micro-tuning.
        │ yes
        ▼
Can you write a pure-PyTorch reference for it (same arithmetic, CPU-runnable)?
──no──> STOP. Without a reference you cannot test the kernel.
        │ yes
        ▼
Do the tile dims fit the register budget (§4) at the target config?
──no──> Either tile the reduction dims properly (see the MoE §5.2
        extension plan) or stay on the PyTorch path.
        │ yes
        ▼
Can you demonstrate ≥1.5× speedup over the raw-PyTorch path (§7)?
──no──> Do not enable by default (AGENTS.md rule 2). Keep it opt-in.
        │ yes
        ▼
Write it, gate it (ENABLE_TRITON_KERNELS), test it (§3.7), document it.
```

Two hard constraints from AGENTS.md that cut this tree short most of the time:

- **Rule 1:** "Bulk of the codebase (RMSNorm, SwiGLU, embeddings, LM head, loss, gate, MLA SDPA, MTP, inference) stays raw PyTorch. … No other component gets a custom kernel without updating this file and `docs/12_Triton_Kernels.md`."
- **Rule 2:** the ≥1.5× bar — "below that, do not enable by default."

## 3. The kernel contract, rule by rule

This is the checklist every kernel file in this repo must satisfy. It is enforced partly by tests (`tests/test_mla_triton.py`, `tests/test_moe_triton.py`, `tests/test_force_back.py`) and partly by review.

### 3.1 Optional import — never fail at module load

Both kernel modules wrap `import triton` in `try/except ImportError` and set a module-level `HAS_TRITON` flag on failure:

```python
try:
    import triton  # type: ignore[import-not-found]
    import triton.language as tl  # type: ignore[import-not-found]
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
```

Consequences: the module imports cleanly on a CPU/Mac box (`tests/test_mla_triton.py:TestMlaTritonImport.test_module_imports_without_triton` asserts `HAS_TRITON == ("triton" in sys.modules)`), and the **host wrapper** raises a helpful `ImportError` only if actually called. `models/mla_triton.py:triton_mla_attention` does exactly this — "Install with `pip install triton` (Linux + CUDA only). For CPU/Mac, use `attn_impl='sdpa'` in your config." The call-site fallback (§3.6) catches that `ImportError`.

### 3.2 A pure-PyTorch reference lives in the same file

`models/mla_triton.py:mla_attention_reference` and `models/moe_triton.py:grouped_moe_pytorch` are module-level (outside `if HAS_TRITON:`), CPU-runnable, and compute the same arithmetic as the kernel. They are the ground truth the CPU tests check, and (for MLA) even the autograd backward. The MoE reference is deliberately the `stacked`-style per-expert loop routed through the sorted-token layout — "Same arithmetic as the `stacked` path but routed through the sorted-token layout; used by CPU tests." A new kernel without a reference is not a kernel in this repo; it is a hypothesis.

### 3.3 The `torch.autograd.Function` wrapper — save inputs, re-compute

The kernel is never called directly from model code. `DeepSeekMoE._routed_forward_triton` calls the host wrapper `models/moe_triton.py:triton_grouped_moe_dispatch`, which applies `_TritonGroupedMoeFunction` and then multiplies by the routing weights *outside* the Function:

```python
out = _TritonGroupedMoeFunction.apply(x_sorted, w1, w2, w3, expert_offsets)
return out * sorted_weights.unsqueeze(-1)
```

That last line is a contract, not an optimization: the gate must receive a gradient, which requires `sorted_weights` to stay on the autograd graph (see [[Docs/04_DeepSeekMoE]] for the aux-loss-free mechanism).

The Function itself follows the FA2 re-compute pattern:

- `forward` saves **only the inputs** (`ctx.save_for_backward(...)`) plus scalar metadata — never the attention matrix or the SwiGLU hidden `h`, which are quadratic/large.
- `backward` re-derives the forward instead of saving activations. The MLA backward is a correctness-first stub: it re-runs `mla_attention_reference` and differentiates it with `torch.autograd.grad`. The MoE backward is the full pattern: it re-runs the fused `dx`/`dw` kernels with activations recomputed in registers.
- **None-grad discipline:** every tensor input must return a gradient — `None` is only legal for scalar/int args. The MLA backward returns six tensor grads plus `None, None, None` for the three scalars; the MoE backward returns `dx`, the three `dw`s, and `None` for `expert_offsets` (INT64, never gradients).

### 3.4 FP32 accumulators, BF16 I/O

"BF16 autocast, FP32 AdamW" ([[Docs/08_Training_Pipeline]]) extends all the way down: `tl.dot` accumulates in FP32, and the kernels only round at the boundary. The MoE forward builds `gate_acc`/`up_acc` as `tl.zeros(..., dtype=tl.float32)` chained through `tl.dot(x_tile, tl.trans(w1_tile), acc=gate_acc)`; the dw kernel accumulates into FP32 locals and the backward casts once on return (`dw1.to(w1.dtype)`). Where a dot needs matching operand dtypes (Triton requires it), the cast is explicit and deliberate: `tl.dot(p.to(v_tile.dtype), v_tile)` in MLA, `h_typed = h.to(x_ptr.dtype.element_ty)` in MoE — a documented accuracy tradeoff, not an accident.

### 3.5 Block sizes, autotune, pre-warm

Block sizes are `tl.constexpr` and computed from real dims with `models/mla_triton.py:_next_pow2` (Triton requires power-of-two `tl.arange` lengths). The AGENTS.md contract adds: "tuned with `@triton.autotune` over a small grid of (block size, num_warps, num_stages) configs. Pre-warm at `__init__` to amortise the first-call compile cost."

Current state, honestly: **neither kernel uses `@triton.autotune` today.** The launch sites hard-code `BLOCK_Q=BLOCK_N=64` (MLA), `BLOCK_T=32` (MoE), `num_warps=4`, `num_stages=2` — a deliberate choice for a single-model repo with two fixed shapes, per `docs/12_Triton_Kernels.md` §8. Both kernels are structured to accept autotune without rework: when you add it, (a) put the decorator on the `@triton.jit` kernel, not inside the Function, (b) make `key=` cover every shape-dependent dimension (for MLA `["S_q", "S_kv", "R", "D_nope", "D_rope", "D_v"]`; for MoE `["T", "D", "I", "E"]`), and (c) keep 4–8 hand-picked configs, not a sweep. Pre-warm: there is no kernel warmup call in the repo yet — the first real step pays the compile tax. `scripts/step_time_a100.py:main`'s `--warmup` steps absorb it for benchmarks, but a `__init__`-time warmup call with the real shapes is the contract's intent and still open work.

### 3.6 Double opt-in: `ENABLE_TRITON_KERNELS` and no silent fallback

The master gate is `models/_triton_dispatch.py:enforce_triton_env_var`, called from `models/transformer.py:Transformer.__init__` (and therefore from `Pretrainer.__init__`, which builds the model). It is a one-table dispatch: `("attn_impl", "triton") → "sdpa"` and `("moe_dispatch", "triton_grouped") → "stacked"`. Unless `ENABLE_TRITON_KERNELS=1`, any Triton dispatch key in the config is force-backed to its PyTorch default with a single startup warning — never 32 per-layer warnings, never an error.

Opt-in therefore requires two things: the env var *and* a config key (`attn_impl: "triton"` or `moe_dispatch: "triton_grouped"` in YAML). On a Linux+CUDA box:

```bash
ENABLE_TRITON_KERNELS=1 python3 scripts/e2e_test_gpu.py          # GPU end-to-end smoke
ENABLE_TRITON_KERNELS=1 python3 training/pretrain.py --config configs/pretrain_1650_2m.yaml
```

There is a **second** safety net at runtime: if the env var is set but the kernel still can't run (triton missing, or a dim over the cap), `models/moe.py:DeepSeekMoE.forward` and `models/mla.py:MultiHeadLatentAttention.forward` catch `ImportError`/`ValueError` and fall back to the PyTorch path with a one-shot `_triton_fallback_warned` message. Two independent guards, both required: a default-config run must never silently take the Triton path (AGENTS.md rule 7), and a broken Triton run must never crash a config that asked for it (the fallback).

### 3.7 The unit test file requirements

AGENTS.md: every new kernel "**must** add a unit test file under `tests/test_<kernel>_triton.py` that (a) compares the Triton path against the PyTorch reference within `atol=1e-2` for BF16, (b) runs `torch.autograd.gradcheck` on a float32 tiny config, and (c) runs a shape / dtype / NaN-finite test on a random input." Plus hard rule 8: tests must run on CPU (via the pure-PyTorch reference) without `triton` installed, with GPU-only behavior gated behind a skip.

The two existing files implement this as:

| Clause | MLA (`tests/test_mla_triton.py`) | MoE (`tests/test_moe_triton.py`) |
|---|---|---|
| (a) BF16 vs reference @ `atol=1e-2` | `TestMlaTritonKernelGPU.test_triton_matches_reference_bf16` | `TestMoeTritonKernelGPU.test_forward_matches_pytorch_bf16` |
| (b) gradient check, fp32 tiny | (covered indirectly by full-model + reference tests; the stub backward is autograd-verified by construction) | `TestMoeTritonKernelGPU.test_autograd_gradcheck_tiny` (grad-presence per input) + `test_backward_matches_pytorch_tiny` (reference-gradient comparison @ `atol=1e-3`) |
| (c) shape / NaN-finite | `TestMlaAttentionReference.test_output_shape` (`torch.isfinite(out).all()`) | `TestMoeTritonKernelGPU.test_forward_with_empty_experts` (`torch.isfinite(y).all()`) |
| GPU gating | `pytest.mark.skipif(not (HAS_TRITON and torch.cuda.is_available()))` | same |

The suite is 199 tests (189 pass + 10 GPU-gated skips) on a CPU laptop; the Triton GPU tests are the bulk of the skips. `tests/test_moe_triton.py:TestMoeTritonKernelGPU.test_forward_422m_config_shape` also locks the cap behavior: at `D=768, I=384` the wrapper must raise `ValueError` matching `"256"`. And `tests/test_mla_triton.py:TestMlaTritonKernelGPU.test_full_model_sdpa_and_triton_agree` builds a full `Transformer` with both dispatch keys under `ENABLE_TRITON_KERNELS=1` and asserts the logits agree at `atol=2e-2` — the end-to-end numerics contract.

### 3.8 The ≥1.5× speedup bar

AGENTS.md rule 2: "For the two sanctioned Triton paths, target ≥ 1.5× speedup over the raw-PyTorch path in `scripts/microbench_a100.py`; below that, do not enable by default." Two honest caveats:

- `scripts/microbench_a100.py:main` measures **peak VRAM**, not speed — the ≥1.5× bar is therefore a policy target whose measurement harness does not exist yet. §7 below is the methodology that will produce it.
- The MoE bar is **structurally blocked at the canonical config**: `moe_inter_dim=384` and `dim=768` exceed the 256 cap, so `moe_dispatch: "triton_grouped"` always falls back to `stacked` there. The bar is only reachable at smoke-config dims (`configs/pretrain_1650_2m.yaml`, `I=32, D=64`) until the §5.2 extension lands. The MLA kernel, by contrast, is usable at the canonical config (`kv_lora_rank=192`, `qk_nope_head_dim=48`, `qk_rope_head_dim=24`, `v_head_dim=64` — all under the cap). No GPU run has measured either kernel; every speed figure in this guide and [[Docs/12_Triton_Kernels]] is an estimate.

## 4. Register-budget math: where the 256 caps come from

Both caps — `models/mla_triton.py:_check_mla_dim_limits` (R, D_nope, D_rope, D_v ≤ 256) and `models/moe_triton.py:_check_dim_limits` (I, D ≤ 256) — are tile-arithmetic decisions, not policy. The reasoning (all figures `[INFERENCE]` — no profiler has run; see [[Docs/12_Triton_Kernels]] §7):

- A thread has **255 addressable 32-bit registers**; beyond that the compiler spills to local memory (L1/L2-backed, much slower).
- A `num_warps=4` program is 128 threads, so a tile of $N$ elements costs roughly $N/128$ registers per thread.

**MLA at canonical dims** (`BLOCK_Q=64, BLOCK_N=64, BLOCK_R=256, num_warps=4`): the two per-head up-projections held across the K-loop (`w_k` and `w_v`, 64×256 each) are ~128 registers/thread each before anything else; the full resident set sums to ≈600 regs/thread — over the 255 ceiling, so spilling is plausible. The cap exists to bound exactly those three big rows. S_q/S_kv are **not** capped because they are tiled (`BLOCK_Q`/`BLOCK_N`); only the register-resident dims are.

**MoE at smoke dims** (`BLOCK_T=32, BLOCK_I=32, BLOCK_D=64`): the resident set is ≈112 regs/thread — comfortable, which is why the smoke config runs it. Force the canonical dims and a single `w1_tile` becomes 512 × 1024 = 524,288 elements ≈ 4,096 regs/thread — more than the entire SM register file. That is the concrete meaning of the cap.

Two levers when a kernel spills: shrink block sizes (fewer resident elements, more loop iterations) or raise `num_warps` (more threads to spread a tile across). The order that respects the arithmetic: fix block sizes so the resident table fits, then sweep `num_warps`/`num_stages`.

## 5. Extending the two sanctioned kernels

### 5.1 MLA kernel (`models/mla_triton.py`)

What works today: the kernel fuses K_nope/V materialisation, RoPE, and online-softmax attention in one program per `(B, H, query-block)`; `q_start` keeps cached mid-sequence prefill causal (`q_start = start_pos if (is_causal and use_cache) else 0`, set in `models/mla.py:MultiHeadLatentAttention._forward_triton`); the backward is the reference-recompute stub.

Open work, in order of value:

1. **Fused backward.** The v1 stub re-runs `mla_attention_reference` inside `backward` — correct by construction, but it costs a full Python reference forward + autograd every step, and it stacks with grad-checkpoint recompute (see the `★ Insight` in [[Docs/12_Triton_Kernels]] §6). A fused `dx`-style backward (mirroring the MoE pattern) removes the second recompute.
2. **Autotune + pre-warm** (§3.5), then measure ([[Docs/guides/G4_benchmarking]]).
3. **Watch shared memory if you raise `num_stages`:** the K-loop pipeline prefetches `ctx_kv` blocks; at `BLOCK_R=256` there is little shared-memory headroom.

### 5.2 MoE grouped kernel (`models/moe_triton.py`) — the canonical-config path

The known blocker: `_check_dim_limits` raises at `I=384, D=768`, so the canonical config always runs `stacked`. Lifting the cap is a three-part change, all documented in [[Docs/12_Triton_Kernels]] §5.7:

1. **I-tiling in fwd/dx.** Today `i_idx = tl.arange(0, BLOCK_I)` covers the *full* I in one tile. Add an outer `for i_start in range(0, I, BLOCK_I)` loop and accumulate `gate_acc`/`up_acc` across I-blocks (they are already fp32 `acc=` chains).
2. **`dh` must become a D-accumulator in the dw kernel.** This is the latent bug to fix while you're there: `_grouped_moe_bwd_dw_kernel` computes `dh = tl.dot(dy_tile, w2_tile)` once per D-block and never accumulates. The true gradient is `dh = dy @ w2` summed over *all* D. Today it is exact only because `D ≤ BLOCK_D` guarantees a single D-block; the moment D-tiling is enabled without an `acc=` chain on `dh`, the weight gradients silently become partial sums. `[INFERENCE]` — unreachable today, a correctness landmine tomorrow.
3. **Unit-stride fix in the dx kernel.** `w1_re`/`w3_re` are loaded with `+ d_b[None, :]` and no `stride_w1_d` term — they assume row-major contiguous weights. Safe today because `DeepSeekMoE.forward` re-stacks experts with `torch.stack` every forward (contiguous by construction), but wrong for any sliced/transposed input.

Also remember the two invariants that must survive any rewrite: the gate-weight multiply stays outside the Function (§3.3), and the expert weights are re-stacked every forward — never cached across optimizer steps.

## 6. Adding a new kernel end-to-end — the recipe

Follow this order; each step leaves the repo in a working state.

**Step 1 — write the reference first.** Module-level pure-PyTorch function in the new file (mirror `grouped_moe_pytorch`): same arithmetic as the PyTorch path you are fusing, CPU-runnable, deterministic. This is your test oracle.

**Step 2 — write the kernel.** `@triton.jit` function; grid over output tiles via `tl.program_id`; block sizes from `_next_pow2`; FP32 `tl.zeros` accumulators chained with `acc=`; masked `tl.load`/`tl.store` with `other=0.0` (and `-inf` for causal/score lanes so masked rows contribute zero probability mass); unit-stride assumptions only where you also assert contiguity. Check the dims against §4 before launching.

**Step 3 — wrap in `torch.autograd.Function`.** `forward` launches the kernel, saves *inputs* plus scalar metadata, returns the output tensor. `backward` either re-runs the reference under `torch.autograd.grad` (fast to ship, correct by construction — the MLA pattern) or re-runs fused backward kernels with in-register recompute (the MoE pattern). Return one value per forward input; `torch.zeros_like` for unused tensor inputs, `None` for scalars. `# illustrative` skeleton:

```python
class _CustomFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, scale):
        out = _custom_fwd_kernel[(grid)](x, w, scale, BLOCK_N=_next_pow2(n), num_warps=4, num_stages=2)
        ctx.save_for_backward(x, w)
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, w = ctx.saved_tensors
        ref = custom_reference(x, w, ctx.scale)
        grads = torch.autograd.grad(ref, (x, w), grad_outputs=grad_out, allow_unused=True)
        return (grads[0] if grads[0] is not None else torch.zeros_like(x),
                grads[1] if grads[1] is not None else torch.zeros_like(w), None)
```

**Step 4 — host wrapper.** Guard on `HAS_TRITON`; raise the actionable `ImportError` ("install with `pip install triton`…") when called without it; run any dimension check (`_check_*_dim_limits`) before the launch. Keep the wrapper thin — this is the public API that `tests/test_doc_refs.py` requires you to anchor (never the `if HAS_TRITON:`-block classes).

**Step 5 — wire into the module.** In the `nn.Module`, try the wrapper, catch `(ImportError, ValueError)`, fall back to the PyTorch path with a one-shot warning — mirror `models/moe.py:DeepSeekMoE.forward` and `MultiHeadLatentAttention.forward` exactly (including the `_triton_fallback_warned` flag), and add the new dispatch key to the `_DISPATCH` table in `models/_triton_dispatch.py:enforce_triton_env_var` so the env-var guard covers it.

**Step 6 — test file.** `tests/test_<kernel>_triton.py` with: reference-vs-kernel comparisons at `atol=1e-2` (BF16, GPU-gated), `torch.autograd.gradcheck` (or at minimum per-input grad-presence + reference-gradient comparison) on a tiny fp32 config, a shape/NaN-finite test, an import-surface test (no-triton path), and the `ValueError` cap test. GPU tests behind `pytest.mark.skipif(not (HAS_TRITON and torch.cuda.is_available()))` so the 199-node CPU suite stays green. Add the full-model agree test (`test_sdpa_and_triton_agree` pattern) once the module wiring exists.

**Step 7 — docs and contract.** Update AGENTS.md (rule 1 requires the sanctioned list to grow) and `docs/12_Triton_Kernels.md`; the API reference lives in `reference/R6_triton_api.md`. Every cited symbol must resolve under `tests/test_doc_refs.py` — run `python3 tests/test_doc_refs.py` before finishing.

**Step 8 — benchmark before defaulting.** §7 methodology, ≥1.5× bar, and only then consider making the key default-on. Until then it stays opt-in.

## 7. Benchmarking protocol

The repo's existing GPU scripts: `scripts/microbench_a100.py:main` (VRAM: builds the canonical model, runs one fwd+bwd, compares measured peak against `utils/memory.py:estimate_model_memory_gb`) and `scripts/step_time_a100.py:main` (end-to-end ms/step, tokens/s, MFU vs a `--peak-tflops` argument; defaults `--steps 20 --warmup 5`, TF32 on, `torch.compile(mode="max-autotune")` by default, AdamW `fused=True`). Step-level throughput is also logged live by `utils/logging.py:TrainingLogger.log` (`tokens_per_sec` over a `log_interval` rolling window — deliberately a whole-step number, not a kernel number).

**Kernel-level comparisons — the honest protocol** (from [[Docs/12_Triton_Kernels]] §9; none of it has run yet):

1. **Synchronize before timing** — `torch.cuda.synchronize()` around `time.perf_counter()`; kernel launches are async.
2. **Warm up, then median** — discard the first call (compile + cache fill); `triton.testing.do_bench` implements exactly this.
3. **Compare like for like, including backward.** MLA: `triton_mla_attention` vs the `sdpa` branch of `MultiHeadLatentAttention.forward` on identical tensors, *including* the reference-backward cost of the stub — training pays it every step. MoE: the fused dispatch vs `DeepSeekMoE._routed_forward_stacked` at the **1650** config (canonical cannot run the kernel).
4. **Count launches, not just FLOPs.** The stacked MoE path issues ~60 small GEMMs per layer per step plus sort/`index_add`; the fused path is 3 launches (fwd + dx + dw). At these shapes launch overhead and HBM traffic dominate — report launch counts beside wall time.
5. **MFU only with a reference FLOP count.** The canonical per-step FLOPs are dominated by the 16 MoE layers' `3·T·I·D·2` per activated expert; without a measured time, MFU is arithmetic, not evidence.
6. **Isolate the compiler cache** — warm up or point `TRITON_CACHE_DIR` somewhere persistent so first-call compile seconds don't pollute steady state.

The honest deliverable when the first A100 run lands: a table of per-path step time, tokens/sec, measured vs estimated peak VRAM, launch counts, and (for MLA) `ncu` spill/occupancy numbers — which would also resolve the §4 `[INFERENCE]` about register pressure at canonical dims. Until then, label every number an estimate.

## 8. Pitfalls

- **`tl.arange` lengths must be powers of two.** Reach for `models/mla_triton.py:_next_pow2` first; absorb the padding with `mask=` second.
- **Masked lanes must not contribute.** Padded score lanes get `-inf`; padded data lanes get `0.0`. A `tl.store` with a mask leaves the destination untouched — allocate `torch.empty` only if every lane is either written or never read.
- **`tl.dot` operand dtypes must match, and dims must be ≥ 16.** The smoke config carries the comments `kv_lora_rank: 16 # must be >= 16 for Triton's tl.dot` — a new kernel inherits that floor.
- **Register pressure is per-thread.** A tile that "fits" at `num_warps=4` may spill at 2 and be wasteful at 8; use §4 to sanity-check before profiling.
- **Never put the routing weights inside the MoE kernel** — the gate would lose its gradient (§3.3).
- **Never cache stacked expert weights across optimizer steps** — `DeepSeekMoE.forward` re-stacks every forward for exactly this reason.
- **The re-compute pattern stacks with grad checkpointing** — a checkpointed layer forward re-runs the kernel, then the Function backward re-runs the reference. Correct, but count the recompute cost before promising speedups.
- **Autotune `key=` must cover every shape-dependent dim**, or a long-context run silently reuses a short-context config.
- **A silent fallback is a bug.** The only legal fallbacks are the two guarded ones (§3.6); a kernel that throws mid-run must surface an error, never quietly route around itself.
- **Anchors, not line numbers.** Docs cite `models/moe_triton.py:triton_grouped_moe_dispatch`, never a `file.py`-plus-line-offset citation; JIT symbols under `if HAS_TRITON:` are banned anchors. `tests/test_doc_refs.py` enforces this.

## 9. Check your understanding

1. **Why does the MoE kernel's gate-weight multiply live outside the `autograd.Function`, and what breaks if you move it inside?** Because `sorted_weights` must stay on the autograd graph so `∂L/∂weights` flows to `AuxLossFreeGate`; inside the Function it would be an opaque input and the aux-loss-free balancing would never train (§3.3).
2. **The MLA kernel is legal at the canonical config; the MoE kernel is not. Why the difference?** MLA's register-resident dims (R=192, D_nope=48, D_rope=24, D_v=64) are all under the 256 cap and S is tiled; MoE's `BLOCK_I=512`/`BLOCK_D=1024` tiles would need ≈4,096 registers per thread — more than the SM has (§4). The fix is I-tiling plus a D-accumulating `dh` (§5.2).
3. **You add a third kernel and skip the `ValueError` cap test. What breaks?** Nothing on the laptop — but the `199`-node CPU suite no longer documents the kernel's boundary, and a future config change could silently route the kernel into a register-spill regime on GPU with no test noticing (§3.7).
4. **`ENABLE_TRITON_KERNELS=1` is set and triton is installed, yet a run still executes the PyTorch path. Name two legal ways that happens.** The config doesn't request the Triton dispatch key (both must be present), or the kernel raised `ValueError`/`ImportError` at runtime and the module's one-shot fallback engaged (§3.6). Both are by design — a config checked in without the env var must never take the Triton path, and a run that requested it must never crash silently.

> **Next:** [[Docs/guides/G4_benchmarking]] — microbench/step_time/MFU methodology.

<!-- docs:verified 2026-08-04 · 59aeef3 -->
