# DeepSeek-v3-Lite — Documentation & Codebase Audit

> **Scope.** Full audit of docs↔code alignment, verified against the working
> tree on 2026-09-21. Every claim below was checked by running the command or
> inspecting the cited source — nothing asserted from memory. This audit is
> the fourth portfolio audit performed against the LLaMA-3-Lite docs standard
> (four-track taxonomy + machine gates); it records the state after the
> learning-paths/glossary/coverage-gate upgrade.

---

## Verification runs (2026-09-21)

| Command | Result |
|---|---|
| `python3 tests/test_doc_refs.py` | OK — 34 files scanned, all path+symbol anchors resolve, 0 line anchors |
| `python3 scripts/check_docs.py` | OK — 31 files linted (links, backtick paths, stale patterns) |
| `python3 scripts/check_docs.py --coverage` | OK — 13 coverage files, **0 uncited public symbols** |
| `python3 -m pytest tests/ -q` | **200 passed, 10 skipped** (GPU-gated), ~20 s on macOS CPU |
| AST census | 31 public top-level symbols across 13 modules; 156 distinct file↔symbol citations in docs |

---

## 1. State of the docs — what is already excellent

- **Corpus:** 33 markdown files under `docs/`, ~196 000 words (measured
  `wc -w` 2026-09-21; per-track table in [README.md](README.md)). The
  `concepts/` track alone is ~102 000 words across 6 self-contained docs.
- **Structure:** four-track taxonomy fully present — `concepts/` (6),
  `references/` (9, R1–R9), `guides/` (9), pipeline docs `training.md` +
  `inference.md`, nav map `README.md`, plus `diagrams/RECEIPTS.md` and the
  `superpowers/` plan archive.
- **Machine gates:** `tests/test_doc_refs.py` (anchor resolution, line-anchor
  ban, JIT-symbol ban) and `scripts/check_docs.py` (link/path/stale-pattern
  lint) both ran in CI before this audit; this audit **added** the
  `--coverage` mode to `scripts/check_docs.py` and wired it into pytest as
  `tests/test_doc_refs.py:test_doc_symbol_coverage`, closing the last gap
  vs the LLaMA-3-Lite / DiffusionGemma gate set.
- **Symbol coverage:** every public symbol in the 13 coverage modules
  (`models/`, `training/pretrain.py`, `data/prepare_data.py`,
  `inference/`, `utils/`) is cited at least once — zero gaps (AST-verified
  by the new coverage gate).
- **Honesty discipline:** the nav map separates implemented pillars from
  paper-spec techniques (FP8, DualPipe explicitly **not implemented**);
  μP numbers tagged with their accounting (8.07e-4 @ 418.7 M with MTP);
  the 1650 smoke config is labeled as such everywhere.
- **Navigation:** two complementary entries — the 14-chapter linear
  curriculum in [guides/getting-started.md](guides/getting-started.md) §9
  and the new three-tier audience paths in
  [guides/learning-paths.md](guides/learning-paths.md).

## 2. Findings table — docs↔code misalignment

| ID | Severity | Finding | Status |
|---|---|---|---|
| A1 | low | `models/mtp.py:MTPModule` was the only public symbol never cited in docs | **fixed 2026-09-21** — cited in [glossary.md](guides/glossary.md) and [R1](references/R1_config_schema.md) already covered `MTPBlock`/`MultiTokenPrediction` |
| A2 | low | `check_docs.py` lacked a `--coverage` mode (the only gate gap vs DiffusionGemma's script) | **fixed 2026-09-21** — mode added, wired into pytest |
| A3 | low | no audience-routed learning paths (only the linear 14-chapter curriculum) | **fixed 2026-09-21** — `guides/learning-paths.md` added |
| A4 | low | no glossary; config-key semantics only in R1 | **fixed 2026-09-21** — `guides/glossary.md` added (notation, MLA/MoE/MTP terms, acronyms) |
| A5 | info | `docs/superpowers/` plans reference retired file names (`G6_contributing.md` excluded from anchor scan by design) | accepted — archive documents, intentionally not scanned |
| A6 | info | canonical config has no full pretrain run yet (8.4 B tokens pending); all A100 timings are estimates | accepted — docs correctly mark them estimated/unverified |

## 3. From-scratch codebase explanation (condensed map)

The repo implements the four DeepSeek-V3 pillars end-to-end in raw PyTorch:

- **MLA** (`models/mla.py:MultiHeadLatentAttention`): K/V live in a 192-dim
  shared latent (`kv_lora_rank`); the matrix-absorption trick folds `wkv_b`
  out of the decode path; RoPE is decoupled onto a 24-dim per-head slice
  (`models/mla.py:MultiHeadLatentAttention._extend_rope`). Backends:
  `sdpa` (default) / `manual` / `triton` (`models/mla_triton.py`, FA2-style
  recompute).
- **DeepSeekMoE** (`models/moe.py:DeepSeekMoE`): 20 routed experts, top-4
  per token (`models/moe.py:Expert`), 1 shared expert; balance enforced by
  `models/moe.py:AuxLossFreeGate` — a per-expert bias updated out-of-band
  every `bias_update_every` steps, no auxiliary loss. Dispatch: `stacked`
  bmm (default) or opt-in grouped-GEMM kernel (`models/moe_triton.py`).
- **MTP** (`models/mtp.py:MultiTokenPrediction`): depth-1 auxiliary head,
  shared embedding + LM head, loss weight 0.3
  (`models/mtp.py:MultiTokenPrediction.compute_loss`); powers the
  speculative decoder `inference/speculative.py`.
- **Trunk** (`models/transformer.py:Transformer`): 18 layers = 2 dense
  (SwiGLU, `models/transformer.py:SwiGLUFFN`) + 16 MoE; weight tying;
  Triton opt-in guarded by `models/_triton_dispatch.py:enforce_triton_env_var`.
- **Training** (`training/pretrain.py:Pretrainer`): AdamW + μP-scaled LR +
  warmup→cosine (`training/pretrain.py:make_warmup_cosine_lambda`), grad
  accumulation, NaN guard with rollback, atomic safetensors checkpoints
  (`utils/checkpoint.py`).
- **Data:** workspace `shared_data/` pipeline consumed via the
  `data/prepare_data.py` shim; mmap shards read by
  `training/pretrain.py:PretrainDataset`.

## 4. Modification plan (priority order)

1. ~~Add `--coverage` gate + close symbol gaps~~ — **done** (A1, A2).
2. ~~Add learning-paths + glossary~~ — **done** (A3, A4).
3. ~~Write this audit~~ — **done**.
4. **Optional next:** port `docs_html/` premium build when the portfolio
   retheme lands; keep `--stamp-footers` cadence per commit.
5. **On the 8.4 B run:** after the first A100 run, replace every
   "estimated" timing in README/nav map with measured numbers and update
   the audit's verification table.

## 5. Acceptance criteria for "audit complete"

- [x] `tests/test_doc_refs.py` green (anchor resolution + coverage via pytest).
- [x] `scripts/check_docs.py` and `--coverage` green.
- [x] Every public symbol in the 13 coverage modules cited ≥ 1× (AST-verified).
- [x] Four-track taxonomy complete: concepts / references / guides (incl.
      learning-paths + glossary) / training.md + inference.md.
- [x] Nav map carries a measured, dated size table.
- [x] No unmeasured headline presented as measured (estimates marked).
