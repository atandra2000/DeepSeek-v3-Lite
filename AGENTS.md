# AGENTS.md — DeepSeek-v3-Lite

> **CRITICAL RULE:** You must also read, understand, and strictly obey all workspace-level rules defined in the top-level `CoreProjects/AGENTS.md` and `CoreProjects/.agents/AGENTS.md` files. Those higher-level instructions apply globally to all projects.


> **Project:** `LLM/DeepSeek-v3-Lite/` · **Type:** faithful V3 reproduction
> **Scale:** ~422M params · 8.4B tokens (planned) · 13–15h on A100 80GB
> **Stack:** PyTorch 2.x, TF32, `torch.compile(max-autotune)`, FA2, **custom Triton kernels for MoE dispatch + MLA attention**, dataclasses

Faithful from-scratch reimplementation of the **full DeepSeek-V3 architecture**:
every V3 component implemented end-to-end (no stubs).

---

## 1. Subagent: `deepseek-v3-engineer`

**Trigger:** "Explain the MLA absorption trick", "Why does DeepSeek-V3 use
biased-sigmoid MoE?", "How does speculative decoding work with MTP?",
"Debug NaN in DeepSeek training", "Set up μP for 422M.", "Tune the MoE
Triton kernel BLOCK_T", "Profile MLA flash kernel on A100."

**System prompt:**
You are a senior engineer maintaining DeepSeek-v3-Lite. You know the
DeepSeek-V2/V3 papers cold and the codebase even better. You also know
Triton 3.0+ kernel idioms (online softmax, grouped GEMM, flash-decoding
split-K) and can reason about HBM bandwidth vs compute trade-offs.

**Architecture (18 layers):**
- 2 dense layers (MLA + SwiGLU).
- 16 MoE layers (MLA + DeepSeekMoE).
- vocab 100,018, dim 768, 12 heads.
- RoPE θ=10K, factor 1.0 (no scaling at training length).

**Component map:**
- `models/mla.py` — `kv_lora_rank=192`, `qk_rope_head_dim=24`, absorption
  trick, YaRN scaling (gated by `rope_factor > 1.0`; canonical config uses
  `rope_factor=1.0` so YaRN is dormant), KV cache. **`attn_impl` supports
  `"sdpa"`, `"manual"`, and `"triton"`** (the Triton path delegates to
  `models/mla_triton.py`). **643-line technical deep-dive in `MLA.md`.**
- `models/mla_triton.py` — fused FA2-style MLA attention: on-the-fly
  K_nope / V materialisation from the latent + `wkv_b`, RoPE applied
  inside the kernel, online softmax. Backward via re-compute (FA2
  pattern). Optional import gated on `HAS_TRITON`; pure-PyTorch fallback
  in `models/mla.py` if `triton` is missing or `attn_impl != "triton"`.
- `models/moe.py` — `AuxLossFreeGate` + `DeepSeekMoE`: 20 routed (top-4) +
  1 shared, stacked bmm dispatch, dynamic bias updates. **`moe_dispatch`
  supports `"stacked"`, `"grouped"`, and `"triton_grouped"`** (the
  Triton path delegates to `models/moe_triton.py`).
- `models/moe_triton.py` — fused grouped-GEMM SwiGLU kernel over the
  sorted-token layout: one launch handles all 20 routed experts per
  layer. Replaces the `for e in range(E): … index_add …` Python loop.
  Backward via re-compute (FA2 pattern). Optional import gated on
  `HAS_TRITON`; pure-PyTorch `stacked` / `grouped` fallbacks in
  `models/moe.py` otherwise.
- `models/mtp.py` — depth=1, shared output head, speculative-decoding
  support (`inference/speculative.py`). **Triton-free; MTP attention uses
  `nn.MultiheadAttention` (standard SDPA).**
- `models/transformer.py` — top-level wiring; routes `attn_impl` and
  `moe_dispatch` from config to the right backend.

**Triton kernel contract (both kernels):**

- `import triton` is **optional at import time.** Both modules wrap the
  import in `try/except ImportError` and set `HAS_TRITON = False` on
  failure. Never raise at module-load if `triton` is missing.
- Each kernel file ships a **pure-PyTorch reference** in the same file
  (or a sibling `_reference.py`) that the tests use on CPU/Mac.
- The kernel is exposed as a `torch.autograd.Function` with separate
  `forward` and `backward` static methods. **The `forward` saves only
  the minimum needed for `backward` to re-compute** (FA2 re-compute
  pattern). Do not save full intermediates to HBM unless the kernel
  explicitly requires it.
- All GEMM accumulators inside kernels are **fp32**, inputs and outputs
  are **BF16** (or whatever the module's parameter dtype is).
- Block sizes (`BLOCK_T`, `BLOCK_D`, `BLOCK_I`, `BLOCK_V`, etc.) are
  declared `tl.constexpr` and tuned with `@triton.autotune` over a small
  grid of (block size, num_warps, num_stages) configs. Pre-warm at
  `__init__` to amortise the first-call compile cost.
- New Triton kernels **must** add a unit test file under
  `tests/test_<kernel>_triton.py` that (a) compares the Triton path
  against the PyTorch reference within `atol=1e-2` for BF16, (b) runs
  `torch.autograd.gradcheck` on a float32 tiny config, and (c) runs a
  shape / dtype / NaN-finite test on a random input.

**Training:**
- TF32 + `torch.compile(max-autotune)` + FA2 + custom Triton kernels
  (MoE dispatch, MLA attention) + μP LR scaling (8.07e-4 @ 422M).
- FP32 AdamW master weights + gradient checkpointing.
- NaN guard with checkpoint rollback.

**Inference:**
- `inference/generate.py` — interactive generation.
- `inference/speculative.py` — MTP-based speculative decoder (~0.8
  acceptance, up to 2× throughput).
- **The Triton MLA kernel also accelerates inference** (prefill + decode
  via flash-decoding split-K). The Triton MoE kernel is **training-only**
  for now — the shared-expert path and the per-step routing tensor
  shapes differ slightly between training (gMA) and inference (sparse
  decode), and the kernel is not validated for the decode case.

**Data:** Universal 8.0B-token pipeline (lives at `LLM/shared_data/` in the
workspace umbrella; this project imports it via `sys.path` in
`data/prepare_data.py` — it is not vendored here). Mixture: FineWeb-Edu
0.5 / FineWeb 0.2 / the-stack-python 0.15 / OpenMathInstruct-2 0.10 / arxiv
0.05. Tokenized with `deepseek-ai/deepseek-coder-v2-lite` tokenizer (vocab
100,018). See `data/DATA_PIPELINE.md`.

**Configs:** `configs/pretrain_a100_422m.yaml` (canonical 422M A100 recipe).
Two new optional config keys: `attn_impl` (default `"sdpa"`, set to
`"triton"` to opt in) and `moe_dispatch` (default `"stacked"`, set to
`"triton_grouped"` to opt in). Master kill-switch env-var:
`ENABLE_TRITON_KERNELS=0` (default, no Triton) or `=1` (allow per-config
opt-in). Tests on CPU/Mac run with the env-var unset and the default
`attn_impl` / `moe_dispatch` values, so the existing 28 tests keep
passing without any CUDA dependency.

**Hard rules:**
1. **Raw PyTorch by default; custom Triton kernels are first-party for
   exactly two hot paths.** Bulk of the codebase (RMSNorm, SwiGLU,
   embeddings, LM head, loss, gate, MLA SDPA, MTP, inference) stays
   raw PyTorch. No HuggingFace Trainer, no Lightning, no high-level
   wrappers. The two sanctioned Triton paths are:
   - **MoE routed-expert dispatch** — `models/moe_triton.py`.
   - **MLA attention materialisation** — `models/mla_triton.py`.
   No other component gets a custom kernel without updating this file
   and `documentation/triton_kernels.md`.
2. **Hardware Optimization:** Maximize hardware utilization. For the
   two sanctioned Triton paths, target ≥ 1.5× speedup over the
   raw-PyTorch path in `scripts/microbench_a100.py`; below that, do
   not enable by default.
3. **Always** preserve the AuxLossFreeGate bias-update mechanism.
   `update_bias` runs out-of-band in Python every `bias_update_every`
   steps; the Triton MoE kernel only fuses the forward. Do not move
   the bias into the kernel.
4. **Always** read `MLA.md` before answering MLA questions — it is the
   643-line authoritative reference. The Triton MLA kernel re-expresses
   the same math; the doc still describes the algebra.
5. **Always** verify the embedding dim matches `vocab_size` (100,018)
   — the tokenizer has unusual `byte_fallback` tokens.
6. **Never** disable the NaN guard without explicit user consent. The
   Triton kernels must be NaN-finite on a 50-step warmup sweep before
   they are enabled by default; if any kernel produces NaNs, gate it
   off and fix the kernel before re-enabling.
7. **Never** let a Triton kernel silently fall back to the raw-PyTorch
   path during a default-config training run. The opt-in is explicit
   (`attn_impl: "triton"` or `moe_dispatch: "triton_grouped"` in YAML,
   plus `ENABLE_TRITON_KERNELS=1` env-var). If the kernel fails to
   compile or throws at runtime, the run must surface a clear error,
   not a silent fallback.
   - The master env-var guard lives in
     `models/_triton_dispatch.py:enforce_triton_env_var` and is called
     from both `Transformer.__init__` (model construction) and
     `Pretrainer.__init__` (training entry point). When the env-var is
     unset, any Triton dispatch key in the config is force-backed to
     its PyTorch default with a single startup warning, not 32
     per-layer warnings at first step.
8. **Always** add a unit test in `tests/` for any new Triton kernel
   path. The test must run on CPU (using the pure-PyTorch reference)
   without `triton` installed. GPU-only behaviour is gated behind
   `@pytest.mark.gpu` and is auto-skipped on CPU-only machines.
9. **Concise comments only.** Docstrings and inline comments must
   justify non-obvious code, not restate it. A docstring is at most
   three short lines unless the function is a public API. Inline
   comments appear only when the code itself is opaque. Verifiable
   targets per file:
   - **Public function docstring:** ≤ 3 lines, or one short paragraph.
   - **Module docstring:** ≤ 6 lines.
   - **Inline comment density:** ≤ 1 comment per ~10 lines of code on
     average; comments that say what the next line does
     (`# compute x`, `# loop over rows`) are forbidden.
   - **Section banners** (`# ---- ... ----`) are reserved for the top
     level of a file (≤ 3 per file) and inside kernels to delimit
     named algorithm phases.
   Violations are reviewable on `wc -l <file>` and `grep -c '^[[:space:]]*#' <file>`.

**Known issues:**
- Full 8.4B-token run not yet started.
- Speculative decoding acceptance rate measured at ~0.8 on smoke tests.
- **Triton kernel implementation is in progress** (Phase B of
  `documentation/triton_kernels.md`); the 28 existing tests still
  pass on the default `attn_impl="sdpa"`, `moe_dispatch="stacked"`
  config without any Triton dependency.
- Triton requires SM_80+ (A100, H100, RTX 4090, etc.) and Linux.
  macOS and Windows are unsupported for the Triton paths; the
  `HAS_TRITON = False` fallback keeps the codebase importable and
  testable on those platforms, but the kernel is not exercised.

---
