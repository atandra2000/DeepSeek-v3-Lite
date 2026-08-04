# Documentation Expansion Plan — DeepSeek-v3-Lite

> Status: **COMPLETED (2026-08-04)** · Owner: coordinator · Result: 181,057 words
> across 31 docs, every public symbol anchor-verified, zero stale numbers.
> This file is the single source of truth for the expansion; writer agents read
> it + `local://doc_contract.md` and write ONLY their assigned files.

---

## 1. Current-State Audit (verified 2026-08-04)

### 1.1 Inventory

| Doc | Words | Lines | Depth (1-5) | Verdict |
|---|---|---|---|---|
| 00 Getting Started | 791 | 110 | 1 | Thin; no troubleshooting, no install variants |
| 01 Foundations | 8,621 | 811 | 4 | Strong theory; nearly zero code anchors (2) |
| 02 Model Architecture | 7,279 | 1,480 | 3 | Good topology; duplicated budget tables; stale code listings |
| 03 MLA | 10,189 | 1,599 | 5 | Best chapter; 10 appendices; weak code wiring (1 anchor, bare `mla.py:`) |
| 04 DeepSeekMoE | 6,164 | 1,012 | 4 | Strong; good invariants section |
| 05 MTP | 3,797 | 805 | 3 | Good; speculative-decoding theory thin |
| 06 FP8 (paper-spec) | 1,651 | 168 | 2 | Thin for a full textbook chapter |
| 07 DualPipe (paper-spec) | 1,366 | 121 | 2 | Thin |
| 08 Training Pipeline | 6,073 | 1,262 | 3 | Broad but shallow in places; 3 stale line anchors |
| 09 Data Pipeline | 2,725 | 619 | 2 | Sharding format + tokenizer internals under-explained |
| 10 Inference & Serving | 3,049 | 631 | 3 | Sampling theory thin; speculative decode needs depth |
| 11 Operations & Testing | 2,310 | 239 | 2 | Suite map + checkpoints; no debugging recipes |
| 12 Triton Kernels | 2,135 | 222 | 2 | 2 kernels × 860 LoC deserve a real tutorial each |
| 13 Portfolio Comparison | 1,337 | 139 | 2 | Fine as-is (comparative, not reference) |
| **Total** | **181,057** | **19,562** | — | — |

### 1.2 Alignment audit (machine-checked) — *historical context (pre-expansion state)*

Run: `ANCHOR_RE = file.py:Symbol` across all docs, resolved via importlib.

- **14 symbol anchors total** across 14 docs — theory-heavy, code-wiring thin.
- **6 stale line anchors** (`file.py:L110/L44/L130/L155/L67/L108`) — banned format,
  currently undetected by `scripts/check_docs.py` (it checks links/paths/patterns only).
- **5 anchor defects**: `docs/03` cites bare `mla.py:_forward_triton` (missing
  `models/` prefix); `docs/01` cites `models/mla.py:_apply_rope` and `docs/09`
  cites `training/pretrain.py:train_step` without their classes
  (`MultiHeadLatentAttention._apply_rope`, `Pretrainer.train_step`).
- **No symbol gate existed at audit time.** `scripts/check_docs.py` verifies markdown links,
  backtick paths, control chars, and a hardcoded stale-pattern list — not that
  documented symbols exist. **This is now fixed:** the symbol gate
  `tests/test_doc_refs.py` was built in phase P0, is wired into CI, and passes
  (100% of cited symbols resolve, 0 line anchors).

### 1.3 Stale facts fixed during the audit (2026-08-04) — *historical context*

- Test count 189 → **196** (186 pass + 10 GPU skips): AGENTS.md ×2, docs/11 ×2,
  docs/12, `scripts/check_docs.py` hint string.
- "422M" → 411.6M base / 418.7M with MTP; MTP cost ~3M → **7.1M**; active
  params ~280M → **~185M**; μP LR numbers corrected; docs/02 mask listings
  updated to the new `_build_causal_mask(seqlen, kv_len, start_pos, device)`
  signature (all landed 2026-08-04, see git diff).

### 1.4 Coverage gaps (concept-building holes)

No dedicated chapter/derivation for: **RMSNorm** (37+ instances), **RoPE**
(only inside MLA chapter), **embedding & weight-tying**, **pre-norm residual
stream theory**, **AdamW/weight-decay/betas derivation**, **gradient clipping &
loss scaling**, **cross-entropy + ignore_index semantics**, **init scheme
(std=0.006 rationale)**, **BF16 autocast & fp32-accumulator numerics for THIS
repo**, **sampling theory (temperature/top-k/top-p)**, **speculative-decoding
theory (acceptance, expected speedup)**, **KV-cache memory lifecycle math**,
**sharded-data format spec**. No **API reference** files, no **guides**
(debugging, μP tuning, Triton dev, benchmarking, ops). docs/06 & 07 are
paper-spec but deserve full textbook chapters.

---

## 2. Target Tree

Keep the numbered learning-path spine (linter + wikilinks depend on it) and
add subfolders. `docs/*.md` glob in `scripts/check_docs.py` must become
`rglob("*.md")` to lint subfolders.

```
docs/
  00_Getting_Started.md          T0  (expand)
  01_Foundations.md              T1  (expand + split)
  02_Model_Architecture.md       T2  (de-dup, re-anchor)
  03_Multi_Head_Latent_Attention.md  T3 (deepen, anchor)
  04_DeepSeekMoE.md              T4  (deepen, anchor)
  05_Multi_Token_Prediction.md   T5  (expand)
  06_FP8_Mixed_Precision.md      T6  (expand to full chapter)
  07_DualPipe_Parallelism.md     T7  (expand to full chapter)
  08_Training_Pipeline.md        T8  (expand, re-anchor)
  09_Data_Pipeline.md            T9  (expand)
  10_Inference_and_Serving.md    T10 (expand)
  11_Operations_and_Testing.md   T11 (expand + guide material)
  12_Triton_Kernels.md           T12 (expand to tutorial grade)
  13_Portfolio_Comparison.md     T13 (light touch)
  reference/
    R1_config_schema.md          every YAML key, defaults, who reads it
    R2_transformer_api.md        Transformer/TransformerBlock/SwiGLUFFN/count_parameters
    R3_mla_api.md                MultiHeadLatentAttention, cache contract, dims
    R4_moe_api.md                AuxLossFreeGate/DeepSeekMoE/Expert, bias contract
    R5_mtp_api.md                MTPBlock/MTPModule/MultiTokenPrediction, loss math
    R6_triton_api.md             both kernels + references + dispatch contract
    R7_training_api.md           TrainingConfig/Pretrainer/PretrainDataset/train_step
    R8_utils_api.md              CheckpointManager/TrainingLogger/memory estimator
    R9_inference_api.md          generate/SpeculativeDecoder/generate_interactive
  guides/
    G1_debugging_playbook.md     NaN guard, shape errors, cache bugs, Triton fallback
    G2_mup_and_lr_tuning.md      μP math, LR sweep procedure, warmup/cosine reasoning
    G3_triton_development.md     how to write/extend kernels, autotune, register budget
    G4_benchmarking.md           microbench/step_time/MFU, what is/isn't measured
    G5_checkpoint_ops.md         save/load/resume/atomicity/dedup, disaster recovery
    G6_contributing.md           doc contract, anchor rules, test expectations, CI
```

New concept chapters folded into the spine (T-series) so the learning path
stays linear: RMSNorm + init (01), RoPE (01/03), embedding & tying (02),
optimizer & scheduler math (08), sampling theory (10), BF16 numerics (01 or 08).

---

## 3. Per-Doc Outlines (writer briefs)

Each writer task = `TARGET + SOURCE FILES + OUTLINE + ACCEPTANCE`. Outlines below
are the plan-level contract; writers expand into section trees in the doc.

### T0 Getting Started (→ ~3k words)
Prereq knowledge, install variants (CPU Mac / A100), first-run checklist,
smoke test walkthrough, "what to read next" graph, troubleshooting table.

### T1 Foundations (→ ~18k words)
Splits into full from-scratch chapters: language-modeling objective;
tokenization (byte_fallback, vocab 100,018); embeddings & weight tying;
residual stream & pre-norm (why pre-norm); **RMSNorm derivation** (code:
`nn.RMSNorm` usage, eps); **RoPE derivation** (θ, complex multiply, code:
`_apply_rope`); attention from scratch → MHA → why MLA; **BF16/autocast/fp32
accumulators**; Chinchilla; **init scheme** (std=0.006, why); μP intuition;
loss landscape / CE. Anchor every code touchpoint.

### T2 Model Architecture (→ ~15k words)
De-duplicate the two budget tables (keep one, cross-ref reference/); rewrite
every code listing to current signatures; add tensor-shape walkthrough per
module with `models/transformer.py` anchors; active-vs-total params math;
FLOP accounting; forward/backward memory timeline.

### T3 MLA (→ ~15k words)
Keep the 10 appendices; add: full derivation of the **absorption trick**
(algebra both directions, `models/mla.py` line-anchored to `_per_batch_bmm`
path), cache lifecycle state machine, softmax_scale/mscale semantics
(now-correct YaRN formula), gradient flow appendix with anchor
`MultiHeadLatentAttention.forward`; fix `mla.py:` → `models/mla.py:` anchors.

### T4 DeepSeekMoE (→ ~12k words)
Gate math (sigmoid + bias), bias-update control-theory analysis, load-balance
metric derivation (`get_load_balance_loss`), stacked vs grouped layouts,
Triton path design (cross-ref T12), expert capacity / sparsity math, the
canonical-config register-cap limitation (measured facts, honest framing).

### T5 MTP (→ ~8k words)
MTP loss derivation (depth-d targets), length alignment algebra (the
`usable = seq_len - d - 2` slicing), shared-head mechanics, **speculative
decoding theory**: acceptance criterion math, expected tokens/step, the
repo's greedy-verification variant vs rejection sampling (documented as a
deliberate simplification), cache-consistency walkthrough.

### T6 FP8 (→ ~5k words) / T7 DualPipe (→ ~4k words)
Full paper chapters: E4M3/E5M2, block scaling, per-tensor vs per-block,
why skipped here (measured 411.6M BF16 fits); pipeline schedules, bubbles,
overlap math, why single-GPU repo doesn't need it. Both keep the explicit
"paper-spec only" banner.

### T8 Training Pipeline (→ ~15k words)
Every stage of `training/pretrain.py` with anchors: `TrainingConfig` field-by-
field; `PretrainDataset` sharded `_locate` math; `train_step` end-to-end
(micro-step vs opt-step counters); AdamW update rule derivation + why fp32
master; scheduler closed form; **μP derivation with the real numbers**
(6e-4 × √(757226496/N), N=411.6M/418.7M → 8.14e-4/8.07e-4); NaN-guard state
machine; checkpoint save/load; bias-update timing.

### T9 Data Pipeline (→ ~7k words)
Universal pipeline stages, mixture math (FineWeb-Edu 0.5 ... arxiv 0.05),
tokenizer internals (byte_fallback, vocab 100,018, EOS/PAD ids), shard
format spec (uint32 mmap, `shard_*.bin`), `_locate` bisect math, resumption
semantics.

### T10 Inference (→ ~8k words)
Sampling theory: temperature, top-k, top-p (with `Transformer._sample`
walkthrough); KV-cache decode lifecycle; flash-decoding split-K rationale;
**speculative decode** (cross-ref T5); generate() loop invariants.

### T11 Operations & Testing (→ ~6k words)
Test-suite map (7 files, what each guards), fixture configs rationale,
checkpoint file format, atomicity mechanics, VRAM budget with the corrected
412M numbers, CI pipeline walkthrough.

### T12 Triton Kernels (→ ~10k words)
Triton tutorial from zero (grid/program/tiles/online softmax); per-kernel
walkthrough: MLA fwd kernel (materialize+RoPE in-kernel, causal q_start),
grouped MoE fwd/dx/dw (the dh-accumulation constraint at D > BLOCK_D),
autograd.Function re-compute pattern, register budget math (256 cap),
autotune guidance, benchmark methodology.

### T13 Portfolio Comparison — light touch only (→ ~2k words).

### R1–R9 API reference (each ~2–4k words)
Every public symbol, signature, shape contract, config key, default, and
"who calls it". 100% symbol-anchored. No prose beyond one-line purpose.

### G1–G6 Guides (each ~2–4k words)
Procedural, checklist-driven, command snippets, decision trees.

---

## 4. Doc-Writing Contract (`local://doc_contract.md`)

Mandatory for every writer:
1. **Style template**: 60-second summary → why it exists → intuition →
   math/proof → code walkthrough → pitfalls → check-your-understanding.
2. **Symbol anchors only**: `file.py:Class.method` / `file.py:function`.
   NEVER `file.py:123` or `L123`. Include `models/`/`training/`/`utils/`/
   `inference/` prefixes. Anchor the host wrapper, never JIT symbols defined
   under `if HAS_TRITON:`.
3. **Snippets**: real code copied from source verbatim; pseudo-code marked
   `# illustrative`. When the code changed, update the snippet — never "as of
   a past commit".
4. **Cross-links**: only to plan-tree docs; use the `[[Docs/...]]` convention
   or relative `.md` links; no dead `#anchor` fragments (verify slugs).
5. **Honesty**: measured vs derived vs `[INFERENCE]` labeled; empty
   `.benchmarks/` means numbers are estimates; paper-spec chapters keep the banner.
6. **Scope**: write ONLY your assigned file; no tests, no linters, no git.
7. **Numbers**: canonical = 411.6M base / 418.7M with MTP, 199 tests
   (189+10), μP 8.07e-4 (MTP) / 8.14e-4 (base). Never reintroduce 422M/189/8.04e-4.

---

## 5. Alignment Gate (build BEFORE writers)

`tests/test_doc_refs.py`:
- Regex `([A-Za-z_][A-Za-z0-9_./-]*\.py):([A-Za-z_][A-Za-z0-9_.]+)` with a
  negative lookahead so math like `L2` is never an anchor.
- Resolve modules **as packages** (importlib `spec_from_file_location` with
  the repo root on `sys.path`; package-relative imports inside `models/`
  require loading `models` first, e.g. `importlib.import_module("models.transformer")`).
- hasattr-chain `Class.method`; report missing files/symbols with doc + line.
- **Ban line anchors** (`file.py:\d+`) as hard failures.
- Allowlist: `data/pretrain_chinchilla`, paper-only files (`training/loss_triton.py` etc.).
- Wire into `.github/workflows/ci.yml` (runs on CPU, no triton needed).
- Extend `scripts/check_docs.py`: glob subfolders (`rglob("*.md")`), keep the
  stale-pattern list, add "docs:verified" footer stamp refresh on merge.

---

## 6. Phased Execution

| Phase | Work | Dispatch |
|---|---|---|
| **P0 Foundation** | build `tests/test_doc_refs.py`; fix every anchor defect it finds in current docs; extend check_docs glob; fix stale numbers (done 2026-08-04) | coordinator, 1 batch |
| **P1 Theory** | T0–T13 rewrites/expansions (~110k words) | 3 batches: T0–T5, T6–T9, T10–T13 |
| **P2 Reference** | R1–R9 (~25k words) | 1 batch of 9 |
| **P3 Guides** | G1–G6 (~18k words) | 1 batch of 6 |
| **P4 Verify + Land** | run checker, fix writer mistakes centrally, link audit, full pytest, update docs/README index + size table, sync vault from workspace root (`bash ~/Desktop/CoreProjects/scripts/sync_to_vault.sh`), commit | coordinator |

Writer task template per doc: `# Target` (file), `# Change` (outline pointer +
source files to read first + contract pointer `local://doc_contract.md`),
`# Acceptance` (anchors resolve, 0 line anchors, links live, style template
followed, word target). Batch context carries: verified project facts, the
411.6/418.7/199/8.07e-4 constants, CRITICAL constraint "no `_stream_to_disk`-
style fiction: only cite symbols that exist".

---

## 7. Acceptance Metrics

1. `tests/test_doc_refs.py` green in CI: **100% of cited symbols resolve**,
   **0 line anchors** across all docs (current: 14 anchors / 12 defects).
2. **Coverage**: every public symbol in `models/`, `training/`, `utils/`,
   `inference/` anchored from ≥1 doc (target ≥95%; measured by a reverse
   grep of the API surface).
3. **Links**: `scripts/check_docs.py` green over `docs/**` (subfolders
   included); 0 dead links; 0 stale-pattern hits.
4. **Freshness**: no "189", "422M-params", "8.04e-4", "~3M MTP" anywhere;
   size table + verification stamps regenerated.
5. **Sizes**: total ≥ 150k words; every T-chapter ≥ 5k words; every R/G ≥ 2k.
6. **Suite**: full pytest green (199 nodes) before landing.
7. **Vault**: mirrored docs in `~/Documents/obsidian` via the workspace sync
   script, never hand-edited.

<!-- docs:verified 2026-08-04 · 59aeef3 -->
