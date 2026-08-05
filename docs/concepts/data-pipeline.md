# DeepSeek-v3-Lite — Data Pipeline

> **Read this if** you're preparing or validating training data. **Skip if** shards already exist → [Training](../training.md).

**Depends on:** [Training](../training.md) · **Read next:** [Inference](../inference.md)

---

## Table of Contents

1. [60-Second Summary](#60-second-summary)
2. [Why a Shared Pipeline?](#why-a-shared-pipeline)
3. [The DeepSeek Shim](#the-deepseek-shim)
4. [Tokenizer Deep Dive](#tokenizer-deep-dive)
5. [Pipeline Stages](#pipeline-stages)
6. [Data Mixture](#data-mixture)
7. [Shard Format](#shard-format)
8. [PretrainDataset Consumption](#pretraindataset-consumption)
9. [CLI Reference](#cli-reference)
10. [Environment Variables](#environment-variables)
11. [Operational Runbook](#operational-runbook)
12. [Pipeline Failure Modes](#pipeline-failure-modes)
13. [Learning Exercises](#learning-exercises)
14. [Check Your Understanding](#check-your-understanding)
15. [Appendix A — FAQ](#appendix-a--faq)
16. [Appendix B — Glossary](#appendix-b--glossary)
17. [References](#references)

---

## 60-Second Summary

DeepSeek-V3-Lite does **not** vendor its own data pipeline. `data/prepare_data.py` is a thin shim that configures the **universal 8.0B-token LLM pipeline** in `LLM/shared_data/` with DeepSeek-specific tokenizer settings (`vocab_size=100,018`, EOS=100,017, PAD=100,016) and writes a project-local `data/data_config.yaml`. The pipeline then runs four crash-isolated stages — **download → clean → tokenize → pack** — and the output is a directory of memory-mapped `uint32` token shards (`shard_*.bin`, 50M tokens each, ~200 MB) plus a `manifest.json` recording provenance, checksums, and per-source token counts.

Training consumes those shards directly through `PretrainDataset` in `training/pretrain.py`: it memory-maps every shard with `torch.load(..., mmap=True)`, computes sample windows of `max_seq_len + 1` contiguous tokens, and serves next-token pairs `(chunk[:-1], chunk[1:])`. A binary search (`PretrainDataset._locate`) maps any global token index to `(shard_idx, offset)` in $O(\log K)$ for $K$ shards, which is what lets a single window stitch tokens across a shard boundary. The whole 8.0B-token corpus sits on disk as ~32 GB of uint32 tokens; it is never fully resident in RAM.

Everything in this chapter is verified against the current sources (`data/prepare_data.py`, `training/pretrain.py`, `LLM/shared_data/`). No full-scale run has ever executed (no GPU training run, no completed data-prep run) — sizes, shard counts, and wall-clock figures are **estimates** or arithmetic, labeled as such.

---

## Why a Shared Pipeline?

The sibling projects under `LLM/` (per `LLM/shared_data/README.md`: Mamba-2-Lite, GPT-OSS-Lite, HyMo, DeepSeek-V3-Lite, LLaMA-3-Lite) all target a ~400–500M-param model trained on roughly the same corpus budget. Without a shared pipeline, each project would independently:

- Re-download FineWeb-Edu (~1.3 TB at full scale) and re-run cleaning — weeks of duplicated bandwidth and compute.
- Drift apart on the *meaning* of a source: "fineweb-edu" with different configs, different filters, different weights makes cross-project loss comparisons meaningless.
- Invent its own shard format (`torch.save` vs raw uint32 vs JSONL), so training recipes could not be ported between projects without rewriting the dataset.

The shared pipeline fixes all three with one decision: **the text corpus, cleaning, dedup, shard format, and manifest schema are shared; only the tokenizer differs per project** (each project has a different vocab — GPT-2 BPE, LLaMA-3 BPE, a custom 64K BPE, or DeepSeek-Coder). This gives:

- **One place to fix bugs** in download/clean/dedup, benefiting every project.
- **Fair cross-project comparisons** — same text, different token IDs.
- **Bit-identical shards except for token values** — the EOS positions, dtype, shard boundaries, and per-source mix are identical across projects using the same corpus; projects sharing a tokenizer produce byte-identical shards.

The cost of sharing is the shim: each project's `data/prepare_data.py` is a ~60-line wrapper that injects its tokenizer settings and delegates to `shared_data.prepare_data.run_pipeline(...)`.

---

## The DeepSeek Shim

`data/prepare_data.py` does three things:

1. **Path injection** — adds the project root and the `CoreProjects/` root to `sys.path` so both `shared_data` and the project's own packages are importable.
2. **Config materialisation** — writes `data/data_config.yaml` with DeepSeek overrides by copying the universal config and patching the tokenizer block (`data/prepare_data.py:_ensure_deepseek_data_config`):

```python
DEEPSEEK_TOKENIZER_NAME = "deepseek-coder-v2-lite"
DEEPSEEK_VOCAB_SIZE = 100_018
DEEPSEEK_EOS_TOKEN_ID = 100_017
DEEPSEEK_PAD_TOKEN_ID = 100_016
```

```python
cfg = load_yaml(UNIVERSAL_DATA_CONFIG_PATH)
cfg["pipeline"]["tokenizer"]["name"] = DEEPSEEK_TOKENIZER_NAME
cfg["pipeline"]["tokenizer"]["vocab_size"] = DEEPSEEK_VOCAB_SIZE
cfg["pipeline"]["tokenizer"]["eos_token_id"] = DEEPSEEK_EOS_TOKEN_ID
cfg["pipeline"]["tokenizer"]["pad_token_id"] = DEEPSEEK_PAD_TOKEN_ID
cfg["_generator"] = "DeepSeek-v3-Lite/data/prepare_data.py"
cfg["_tokenizer_family"] = "deepseek-coder-v2"
```

3. **Delegation** — `data/prepare_data.py:main` parses `--stage`, `--mixture`, `--data-config`, `--data-root`, `--source`, and the four `--skip-*` flags, then calls `shared_data.prepare_data.run_pipeline(...)` with the DeepSeek data config. The shim also prints the corpus budget it expects (`data/prepare_data.py:_apply_deepseek_defaults` reads `UNIVERSAL_TOTAL_TOKENS = 8_000_000_000` from `shared_data/config.py`):

```bash
python3 data/prepare_data.py --stage pretrain
```

The resulting `data/data_config.yaml` is small and fully self-describing (it records `_generator` and `_tokenizer_family`). The sharding block stays at the universal defaults — 50,000,000 tokens per shard, `uint32`, target 8.0B tokens — while the DeepSeek overrides only touch the tokenizer block.

```yaml
pipeline:
  tokenizer:
    name: deepseek-coder-v2-lite
    path: null
    vocab_size: 100018
    eos_token_id: 100017
    pad_token_id: 100016
    add_eos: true
  sharding:
    shard_size_tokens: 50000000
    dtype: uint32
    target_total_tokens: 8000000000
  # dedup / quality / tokenize / pack / seed — universal defaults
```

**Cross-check at startup:** `run_pipeline` reads both `mixture.yaml` and `data_config.yaml` and compares `mixture.total_tokens` against `pipeline.sharding.target_total_tokens`. If they disagree it logs a warning and uses the **mixture's** value — the mixture is authoritative. Keeping the two in sync is a deliberate, cheap guard against silently building a corpus of the wrong size.

---

## Tokenizer Deep Dive

| Property | Value | Why it matters |
|---|---|---|
| Name | `deepseek-ai/deepseek-coder-v2-lite` | Public HF tokenizer (DeepSeek-Coder-V2-Lite family) |
| `vocab_size` | **100,018** | Must match `model.vocab_size` and the embedding rows exactly |
| `eos_token_id` | 100,017 | Appended at every document boundary (the packer relies on it) |
| `pad_token_id` | 100,016 | Padding (rarely used in packed pretraining) |
| `byte_fallback` | yes | BPE falls back to raw byte tokens instead of `<unk>` — handles arbitrary bytes, no OOV ever |

### Why 100,018 and not a round number?

The tokenizer is the *same one DeepSeek shipped for DeepSeek-Coder-V2-Lite*, a byte-level BPE whose trained merges plus special tokens total 100,018 ids. The number is not a hyperparameter you choose; it is a property of the tokenizer. Two consequences:

- The embedding matrix in `models/transformer.py:Transformer.__init__` is `nn.Embedding(vocab_size, dim)` with `vocab_size` read from the config — the 422M config (`configs/pretrain_a100_422m.yaml`) sets `vocab_size: 100018`. If the two ever disagree, every embedding lookup for ids $\geq$ the config's vocab silently misaligns or crashes with an index error.
- The 1650 smoke config (`configs/pretrain_1650_2m.yaml`) deliberately uses `vocab_size: 50257` with the GPT-2 tokenizer — the architecture features are identical but the tokenizer needs no HF authentication, which is what makes CI runs self-contained.

### byte_fallback: the mechanism

`byte_fallback: yes` is the HF tokenizers setting that changes what happens when the BPE model has **no merge path** for the bytes it is scanning. A tokenizer without it emits the `<unk>` token (id 0 in many vocabularies), which destroys information — any unseen byte sequence, any rare emoji, any exotic script, collapses to one id. With byte_fallback enabled, the encoder instead emits one token per raw byte from a reserved range of byte tokens, so:

- **Encoding is total**: every possible UTF-8 string maps to a sequence of in-vocab ids. There is no out-of-vocabulary input, ever.
- **Round-tripping works**: decoding a byte-fallback sequence reproduces the original bytes exactly.
- **The model can still learn the byte-level grammar** of rare content instead of being blind to it.

This is unusual compared to GPT-2/LLaMA-style tokenizers (byte-level BPE where bytes *are* the bottom of the merge ladder) and is one reason the DeepSeek tokenizer is worth keeping: the model never sees `<unk>`, so no training signal is silently discarded. The practical upshot: the embedding table must have exactly 100,018 rows, and token IDs can legitimately be any value in `[0, 100018)` — do not assume they are all printable Unicode.

### Special tokens

EOS (100,017) is the workhorse of the pipeline: every document gets exactly one trailing EOS (the tokenizer config sets `add_eos: true`, and the pack stage's writer also appends EOS explicitly, de-duplicating a document that already ends in EOS). The `PretrainDataset` consumer never needs to *find* document boundaries — the model simply learns to predict EOS, and the next window's first token is the first token of the next document. PAD (100,016) is defined but essentially unused in packed pretraining: windows are never padded, they are truncated/dropped (see [Sample Count Formula](#sample-count-formula)).

### Verification

The tokenizer itself is external to the repo, but the contract is checkable:

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-v2-lite")
assert len(tok) == 100018
assert tok.eos_token_id == 100017
assert tok.pad_token_id == 100016
```

Changing the tokenizer without regenerating shards **and** updating `vocab_size` in the model config breaks training at the embedding lookup on step 0 — see [Pipeline Failure Modes](#pipeline-failure-modes).

---

## Pipeline Stages

`shared_data.prepare_data.run_pipeline(...)` orchestrates four subprocess stages. Each stage runs as `python -m shared_data.scripts.<stage>` via `subprocess`, which is what makes them **crash-isolated**: a failure (or OOM, or disk-full) in tokenize leaves the clean JSONL untouched, and a partial stage never corrupts the output of a later one. The orchestrator checks the mixture/data-config files exist, cross-checks the token totals, seeds all RNGs (`seed: 42`), and stops at the first failing stage with a nonzero exit code.

```
   ┌──────────────┐
   │ HuggingFace  │  all 7 mixture sources streamed via load_dataset
   └──────┬───────┘
          │ stage 1: shared_data.scripts.download_raw
          ▼
   data/raw/<source>/data.jsonl          raw text, one JSON object per line
          │ stage 2: shared_data.scripts.clean
          ▼
   data/clean/<source>/data.jsonl        quality-filtered + dedup'd text
          │ stage 3: shared_data.scripts.tokenize
          ▼
   data/tokens/<source>/data.bin         per-source uint32 token streams
          │ stage 4: shared_data.scripts.pack_shards
          ▼
   data/shards/shard_00000.bin …         packed 50M-token shards
   data/shards/manifest.json             provenance + checksums + stats
```

All paths resolve under a **data root**: `$LLM_DATA_ROOT` if set, else `<cwd>/data` (see [Environment Variables](#environment-variables)). Every stage is **resumable** — see [Resumption Semantics](#resumption-semantics).

### Stage 1 — Download

`shared_data/scripts/download_raw.py` streams each source from HuggingFace with `load_dataset(..., streaming=True)` and appends JSON objects `{"text": ...}` to `data/raw/<source_id>/data.jsonl`.

Two details matter:

- **Budget truncation happens here first.** Each source has a target of `int(total_tokens * weight)` tokens; the downloader converts that to characters with a fixed estimate `CHARS_PER_TOKEN = 4.0` and stops when `n_chars >= target_chars`. Downloading is thus *truncated, not sampled*: the first documents of the HF split fill the source's budget. This is the first of three places the 8.0B budget is enforced (download → tokenize → pack).
- **Multi-field documents.** Some sources combine two columns. OpenMath concatenates `problem` and `generated_solution` with a separator so the model sees full worked derivations (`_build_text` in `download_raw.py`):

```python
extra_field = spec.get("extra_text_field")
if extra_field:
    extra = row.get(extra_field, "") or ""
    sep = spec.get("extra_separator", "\n\n")
    if extra:
        text = f"{text}{sep}{extra}" if text else extra
```

Progress (`n_processed`, `n_chars`) is flushed to the per-source download state file every 50,000 docs, which is what makes resumption cheap.

### Stage 2 — Clean

`shared_data/scripts/clean.py` runs two passes over each source's raw JSONL: a **quality filter** and an **exact-match dedup**.

The quality filter (`shared_data/quality_filter.py`) is a chain of cheap heuristics, each returning keep/drop:

| Filter | Rule (defaults from `data_config.yaml` → `pipeline.quality`) |
|---|---|
| Length | keep `min_chars <= len(text) <= max_chars` — per-source bounds from `mixture.yaml` (e.g. arxiv keeps 500–500,000 chars, capping one-doc-per-shard pathologies) |
| Empty | drop empty strings |
| Unique chars | drop if `len(set(text)) < 0.05 * len(text)` |
| Digit ratio | drop if digits > 50% of chars — **disabled for `lang: python`** sources, where digits are legitimate |
| Punctuation ratio | drop if punctuation > 50% |
| Whitespace ratio | drop if whitespace > 50% |
| Language hint | cheap ASCII-letter-ratio heuristic (`lang: en` / `lang: python` / `null`) — not a real language ID |

`QualityFilter.apply(text)` returns the (possibly whitespace-normalised) kept text or `None`, and `FilterStats` accumulates per-reason drop counts that end up in the manifest.

Dedup (`shared_data/dedup.py` — `DedupFilter`) is **per-source, exact SHA-256**: a document is dropped iff its whitespace-normalised SHA-256 hash has been seen before in *that source* (`sha256_text` normalises with `" ".join(text.split())` before hashing, so near-identical formatting still collides). Note this is *not* a cross-source union — each source keeps its own seen-set, so the same text may survive in two sources. The `minhash` near-duplicate strategy and the bloom-filter knobs (`n_hash_buckets`, `bloom_capacity_per_bucket`, `bloom_error_rate`) are documented as reserved in `data_config.yaml`; the implemented `sha256` path uses a plain in-memory set. Cleaned documents are written to `data/clean/<source>/data.jsonl` with a 16-hex-char content hash prefix (`{"id": sha16, "text": ...}`) so downstream stages can reference provenance.

### Stage 3 — Tokenize

`shared_data/scripts/tokenize.py` encodes each clean document to token ids with the project tokenizer (for us: deepseek-coder-v2-lite) and writes a **per-source token stream** — a binary file with a tiny header followed by raw uint32 token ids, EOS-separated (see [Byte-Level Shard Spec](#byte-level-shard-spec) for the exact header).

The producer/consumer design keeps the GPU-less stage CPU-bound in the right way:

```python
# Producer thread: tokenise ahead of the writer.
q: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=prefetch)
stop = threading.Event()

def producer() -> None:
    for text in doc_iter:
        if not text:
            continue
        ids = tokenizer.encode(text)
        if not ids:
            continue
        validate_tokens(np.asarray(ids, dtype="uint32"), ...)
        q.put((ids, len(ids) + 1))
    q.put(None)

t = threading.Thread(target=producer, daemon=True)
t.start()
```

The main thread pops `(ids, n)` and calls `stream.write_doc(ids)`, which appends the ids **plus one EOS** to the stream and advances the counters. `validate_tokens` enforces the hard upper bound `token_id < vocab_size + 256` — if the tokenizer ever emitted something wildly out of range, the stage refuses to write a corrupt stream rather than silently proceeding. The per-source token budget (`total_tokens * weight`) is enforced *here* as the exact truncation point: `if n_tokens >= target_tokens: stop.set(); break`. `add_special_tokens: false` means the only special token in the stream is the manually appended EOS.

### Stage 4 — Pack

`shared_data/scripts/pack_shards.py` interleaves the per-source token streams **round-robin at the document level** (`interleave_sources` yields `(source_id, doc)` cycling through sources) and feeds them to `ShardWriter`, which packs 50M tokens per shard. Because each source's stream was already truncated to its budget at tokenize time, the round-robin preserves the global mixture without any re-blending at pack time. (The per-source `remaining` budgets threaded through `interleave_sources` are not enforced there — truncation happened upstream in stage 3.)

`ShardWriter` gives three guarantees:

1. **Document atomicity:** a document (with its EOS) is never split across shards — `cross_document_boundary_ok: false` in the config, and `ShardWriter.add` raises if a single document + EOS exceeds `shard_size_tokens` (50M), which the per-source `max_chars` caps make impossible in practice.
2. **Shard atomicity:** a shard is written to a `.tmp` sibling, flushed, `fsync`'d, then moved into place with `os.replace`:

```python
payload = self._buf[: self._buf_pos]
raw_bytes = payload.tobytes()
with open(tmp_path, "wb") as f:
    f.write(raw_bytes)
    f.flush()
    os.fsync(f.fileno())
os.replace(tmp_path, shard_path)
```

A SIGKILL mid-write can leave only a `.tmp` file, never a half-written `shard_*.bin` that a reader would happily mmap.
3. **Verification before manifest:** after packing, `verify_shard` re-reads every shard and checks byte alignment, token count, `max_token < vocab_size + 256`, and the EOS count; the `Manifest` is then built *from the shard files actually on disk* (authoritative), validated (`manifest.validate`), and saved next to the shards as `manifest.json`.

### Resumption Semantics

Every stage is designed to be re-runnable; nothing is "all or nothing".

- **Download:** state `data/state/download_<source>.json` records `n_processed`/`n_chars`. On re-run, rows before `n_processed` are skipped and the JSONL is reopened in append mode — no re-download of the first N docs.
- **Clean:** state `data/state/clean_<source>.json` records processed/kept/dropped counters. On re-run the first `n_processed` input rows are skipped and output is appended (output is already correct; the input is re-read from the start). The dedup seen-set is rebuilt in memory each run — the shared_data design docs describe a persisted seen-set (`dedup_<source>.json`), but the current `DedupFilter` implementation does not persist it [verified 2026-08-04], so a crash mid-clean re-emits rows that were already emitted (they may be re-dropped as duplicates only within the same run).
- **Tokenize:** state `data/state/tokenize_<source>.json` gains a `"complete": true` marker when a source finishes; `_should_retokenize` skips completed sources on re-run (the stream file is reused and its token count recomputed from the file size). A partially-tokenised source resumes from `n_processed` docs.
- **Pack:** `ShardWriter` flushes complete shards atomically and the manifest is rebuilt from disk, so re-running after a crash re-packs from the beginning but every re-written shard is byte-identical. The `resume=True` argument passed by `pack_shards.py` and the `shard_writer_state.json` file described in the shared_data docs are **not implemented** in the current `ShardWriter` — its constructor has no `resume` parameter, and passing one raises `TypeError` [measured 2026-08-04 on the workspace copy]. Treat pack as restartable, not incrementally resumable, in the current code.
- **Stage-level skip:** the `--skip-*` flags bypass whole stages; `--source NAME` limits a run to one source (useful for debugging a single mixture component).

---

## Data Mixture

The canonical mixture lives in `LLM/shared_data/config/mixture.yaml`. **It is the only source of truth** — the table below is quoted from it as of 2026-08-04 and may drift; always re-read the YAML before trusting any copy (including this doc).

### Mixture weights math

The corpus target is `total_tokens: 8_000_000_000` (8.0B). Each source declares a weight $w_i$; the per-source token budget is simply

$$
T_i = w_i \cdot T, \qquad T = \sum_i T_i = 8.0 \times 10^9
$$

| Source (`id`) | HF dataset | Weight $w_i$ | Budget $T_i$ |
|---|---|---|---|
| `fineweb-edu` | `HuggingFaceFW/fineweb-edu` (sample-10BT) | 0.40 | 3.20B |
| `dclm-baseline` | `mlfoundations/dclm-baseline-1.0` | 0.15 | 1.20B |
| `the-stack-v2-python` | `bigcode/the-stack-v2` (Python) | 0.15 | 1.20B |
| `the-stack-v2-jupyter` | `bigcode/the-stack-v2` (JupyterNotebook) | 0.05 | 0.40B |
| `openmath` | `nvidia/OpenMathInstruct-2` | 0.10 | 0.80B |
| `arxiv` | `cdv/arxiv-classification` | 0.10 | 0.80B |
| `cosmopedia` | `HuggingFaceTB/cosmopedia` | 0.05 | 0.40B |
| **Σ** | | **1.00** | **8.00B** |

Grouped: web backbone 0.55, code 0.20, math 0.10, long-form scientific prose 0.10, synthetic instruction prose 0.05.

The weights are tuned to the modern "data diet" consensus: a quality-gated web backbone, a strong code+math block (30% — the single strongest lever for reasoning in small models), and long documents from arxiv for long-context coherence. The YAML documents that weights must sum to 1.0; the *pipeline* only cross-checks the token totals between mixture and data config (a warning), so treat the weight sum as a documented invariant rather than a runtime-enforced one.

Because the budget is enforced by **truncating each source's stream at $T_i$ tokens**, the mixture is exact up to the last partial document: under-represented sources still contribute their full budget (arxiv's 0.10 → 0.80B tokens — plenty for a ~412M model). No sampling or re-blending happens at train time; the packer's round-robin interleave is what makes the local mix (the stream inside one shard) look like the global mix instead of "blocks of one source".

### Token budget: 8.0B canonical vs "~8.4B" nominal

The canonical, pipeline-enforced target is **8.0B tokens** (`mixture.yaml` `total_tokens`, `UNIVERSAL_TOTAL_TOKENS` in `shared_data/config.py`, and the shim-generated `data/data_config.yaml` all agree). The "~8.4B" phrasing that appears in repo scripts and older docs is the *Chinchilla rationale* for that number: roughly 20 tokens per parameter times 418.7M parameters (with MTP) gives

$$
20 \times 4.187 \times 10^8 \approx 8.4 \times 10^9,
$$

so the realized round budget of 8.0B sits right at the compute-optimal recommendation. When you see "8.4B" in this repo, it means "the 8.0B canonical target, justified by ~20 tokens/param" — the YAML value 8,000,000,000 is what the pipeline actually produces.

---

## Shard Format

### Why uint32 on disk, int64 in training

| Stage | dtype | Reason |
|---|---|---|
| Disk shards | uint32 | 4 bytes/token; vocab 100,018 < $2^{32}$ |
| `PretrainDataset` storage | uint32 mmap | Memory-efficient random access |
| Model input | int64 | `nn.Embedding` and `F.cross_entropy` require Long indices |

Keeping tokens as uint32 until the model boundary halves disk and page-cache footprint: 8.0B tokens at 4 B/token = 32 GB on disk versus ~64 GB for int64. The cast to int64 happens once, at the boundary (see [The uint32 → int64 Cast Boundary](#the-uint32--int64-cast-boundary)).

### Byte-Level Shard Spec

The **universal pipeline** (`LLM/shared_data/shard_writer.py`) writes shards as **raw little-endian uint32 arrays — no torch header, no per-sample headers**:

```
shard_00000.bin
  byte 0      token[0]     (uint32 LE)
  byte 4      token[1]
  ...
  byte 4*(n-1) token[n-1]
```

- dtype: `uint32`, platform-native little-endian (the writer uses `arr.tobytes()` on an `np.uint32` array).
- size: 50M tokens/shard → exactly 200 MB (`50e6 × 4 B`).
- content: a flat token stream; documents are separated by single EOS tokens (id 100,017). There are **no per-document headers** — the EOS token *is* the boundary.
- naming: `shard_{index:05d}.bin` (`shard_00000.bin`, `shard_00001.bin`, …), zero-padded so lexicographic order = numeric order.
- the `PretrainDataset` consumer globs `shard_*.bin`, so any zero-padding (e.g. `shard_0000.bin` from `scripts/build_small_pretrain_data.py`) works.

**Two writers, two on-disk formats — keep them straight:**

1. The universal pipeline's `ShardWriter` writes the raw-byte format above; its reader (`ShardDataset` in `shared_data/dataset.py`) opens them with `np.memmap(..., mode="r")`.
2. `scripts/build_small_pretrain_data.py` (the self-contained smoke-test path, no `shared_data` needed) writes a **torch-serialised** 1D `torch.uint32` tensor via `torch.save(tensor, out_path)` — and this repo's `PretrainDataset` loads shards with `torch.load(p, weights_only=True, map_location="cpu", mmap=True)`, which requires the torch-serialised form.

A raw-byte shard from the universal pipeline will **not** load through `PretrainDataset.torch.load(..., mmap=True)` — `torch.load` rejects files not written by `torch.save` [measured 2026-08-04: `RuntimeError: mmap can only be used with files saved with torch.save(...)`]. Since the full-scale corpus has never been produced, this seam is unexercised; converting raw shards (e.g. `torch.save(torch.from_numpy(np.memmap(...)))`) or extending `PretrainDataset` to numpy-memmap raw shards is an open integration item if the universal output is pointed at `data/pretrain_chinchilla/` [INFERENCE — no conversion helper exists in the repo].

The per-source intermediate streams (stage 3) have a slightly richer layout (`TokenStream` in `shared_data/shard_writer.py`): an 8-byte header `<II` = (version=1, eos_token_id) followed by the same raw uint32 body, EOS-separated. `read_token_stream` parses the header, memory-maps the body, finds EOS positions with `np.where(body == eos_id)`, and yields each document as a `np.ndarray` copy.

### manifest.json

The manifest (`Manifest` in `shared_data/manifest.py`) is written next to the shards (default `manifest.json` inside the shards directory) and is the machine-readable record of what the corpus is:

| Field | Meaning |
|---|---|
| `version`, `created_utc` | schema version 1.0.0, creation timestamp |
| `vocab_size`, `eos_token_id`, `pad_token_id`, `tokenizer_name` | the tokenizer contract the shards were encoded with |
| `dtype`, `shard_size_tokens` | storage contract (`uint32`, 50M) |
| `total_tokens`, `shard_count`, `shards_dir` | corpus totals |
| `shards[]` | per shard: `index`, `path`, `n_tokens`, `sha256`, `n_eos` |
| `sources{}` | per source: `target_tokens`, `actual_tokens`, `n_docs`, `n_dedup_dropped`, `shard_count` |
| `config_hash`, `mixture_hash` | SHA-256 of the exact `data_config.yaml` and `mixture.yaml` used — changing either invalidates the corpus fingerprint |

`Manifest.validate` enforces sanity invariants before save: positive vocab/totals, `eos_token_id` within `[0, vocab_size + 256)`, `len(shards) == shard_count`, and $\sum_{\text{sources}} \text{actual\_tokens} \approx \text{total\_tokens}$ (within 0.1%).

### Storage Budget

| Item | Size | Notes |
|---|---|---|
| Raw downloads | ~500 GB+ | cached under the data root (`data/raw/<source>/`) |
| Cleaned text | ~200 GB | JSONL under `<data-root>/clean/` |
| Token streams | ~32 GB | per-source uint32 under `<data-root>/tokens/` |
| Packed shards | **32 GB** | 8.0B × 4 B — `8.0e9 × 4 = 3.2e10` bytes |
| Shard count | **160** | `8.0e9 / 5.0e7 = 160` |

All figures except the arithmetic are estimates — no full-scale run has executed. Raw-text stages dominate disk; the training-relevant artifact (shards) is a tidy 32 GB, mmap-friendly, and shared across sibling projects (only token IDs differ per tokenizer).

---

## PretrainDataset Consumption

The training-side consumer is `PretrainDataset` (`training/pretrain.py:PretrainDataset.__init__`). It is a `torch.utils.data.Dataset` over a **flat token stream** — there are no per-sample records anywhere; samples are *windows* over the packed tokens.

### Layout Detection

`PretrainDataset` accepts either a single `.bin` file or a directory of shards, decided by `os.path.isdir(data_path)`:

```python
self.layout = "sharded" if os.path.isdir(data_path) else "single"
if self.layout == "sharded":
    self.shard_paths = [str(p) for p in sorted(Path(data_path).glob("shard_*.bin"))]
    if not self.shard_paths:
        raise FileNotFoundError(f"No `shard_*.bin` files in {data_path}")
    self.shards = [torch.load(p, weights_only=True, map_location="cpu", mmap=True) for p in self.shard_paths]
    self.shard_sizes = [s.numel() for s in self.shards]
    self.shard_offsets = []
    running = 0
    for s in self.shard_sizes:
        self.shard_offsets.append(running)
        running += s
    self._total_tokens = sum(self.shard_sizes)
else:
    self.data = torch.load(data_path, weights_only=True, map_location="cpu", mmap=True)
    self._total_tokens = self.data.numel()
self._n_samples = (self._total_tokens - 1) // self.max_seq_len
```

Key mechanics:

- **mmap, not load:** `torch.load(..., mmap=True)` maps each shard lazily; the OS demand-pages tokens as they are read, and read-only pages are shared across `num_workers=8` DataLoader workers. The full 32 GB corpus is never resident in RAM.
- **Cumulative offsets:** `shard_offsets[i]` is the global token index where shard `i` begins. With $K$ shards this is a sorted array of length $K$, which is exactly what the bisect in `_locate` needs.
- **Total token count** is the sum of shard sizes — the consumer never reads the manifest; the shard files themselves are authoritative.
- **A missing path** raises `FileNotFoundError("Pre-training data not found: ... Run `python data/prepare_data.py` first.")` — the standard first-run error, also hit when a config points at a directory that only contains a manifest.

### Sample Count Formula

$$
N_{\text{samples}} = \left\lfloor \frac{N_{\text{tokens}} - 1}{S} \right\rfloor, \qquad S = \text{max\_seq\_len}
$$

The `-1` accounts for the extra target token every window needs: window `idx` reads `S + 1` tokens at `start = idx * S`, returning input `chunk[:-1]` (positions 0..S-1) and shifted target `chunk[1:]` (positions 1..S). At position $t$ the model predicts $x_{t+1}$ from $x_{\le t}$ — classic teacher-forced next-token prediction.

The final `(N mod S)`-ish tail that cannot fill a full window is **dropped, not padded**. Padding would inject fake targets and is deliberately avoided; `drop_last=True` on the DataLoader is a *second*, orthogonal drop — it drops partial **batches**, not partial samples.

For the 422M config ($S = 2048$):

$$
N_{\text{samples}} \approx \frac{8.0 \times 10^9}{2048} \approx 3.9 \times 10^6 \text{ samples/epoch}
$$

With `micro_batch_size 8 × 2048 × gradient_accumulation 4` = 65,536 tokens/opt-step and 512,000 steps, training exposes 512,000 × 65,536 ≈ 33.5B tokens ≈ **4.2 epochs** over the 8.0B corpus. Repeated epochs are intentional — rare sources (arxiv, math) benefit from re-exposure, and the seeded per-epoch reshuffle (`shuffle=True` with a fixed-seed `torch.Generator`) means each epoch presents a different order. Note that resume does not restore the sampler RNG, so post-restart order differs — documented as benign at this scale in `Pretrainer.train`.

### Packed-Stream Semantics

Because shards are flat token streams with no per-sample records, every windowing property follows from two facts: windows are **contiguous** (`start = idx * S`) and **non-overlapping** (window `idx` occupies `[idx·S, idx·S + S + 1)`).

**Next-token alignment.** For `chunk = data[start : start + S + 1]`:

```
tokens  = chunk[:-1]   # positions 0..S-1
targets = chunk[1:]    # positions 1..S   (shifted by 1)
```

At position $t$ the model predicts $x_{t+1}$ from context $x_{\le t}$ — teacher forcing, where targets are ground-truth tokens, never model predictions. The shifted-pair construction is what makes a single contiguous slice yield both inputs and targets with zero duplication of storage.

**Cross-document boundaries.** Documents are concatenated with exactly one EOS between them. A window may span any number of document boundaries:

```
... doc_A ... EOS | doc_B ...
          ^^^^^^^^^ window boundary
```

The model learns to predict the first token of `doc_B` after seeing EOS — standard GPT-style pretraining. There is no special "document start" token beyond EOS, and no cross-document masking: the causal mask (`Transformer._build_causal_mask`) is purely positional, so context flows across document boundaries exactly as it does within one. This is a deliberate simplicity: EOS is the only structure the data carries, and the model internalises "EOS ⇒ new document" from data alone.

**Window count.** The flat, contiguous, non-overlapping layout is what makes the sample count a closed form rather than a scan: every window except the last has full length, so

$$
N_{\text{samples}} = \left\lfloor \frac{N_{\text{tokens}} - 1}{S} \right\rfloor
$$

is exact. The same formula is used for both layouts (`_n_samples` is computed once in `__init__`), which is why the single-file and sharded paths yield identical epochs for the same token count.

### Cross-Shard Windows

A window that would cross a shard boundary is stitched on the fly (`training/pretrain.py:PretrainDataset.__getitem__`):

```python
pieces = []
cursor = start
cursor_pos = 0
while cursor_pos < needed:
    shard_idx, offset_in_shard = self._locate(cursor)
    shard = self.shards[shard_idx]
    take = min(needed - cursor_pos, self.shard_sizes[shard_idx] - offset_in_shard)
    pieces.append(shard[offset_in_shard: offset_in_shard + take])
    cursor += take
    cursor_pos += take
chunk = torch.cat(pieces) if len(pieces) > 1 else pieces[0].clone()
return chunk[:-1], chunk[1:]
```

`take` is clamped by the shard's remaining tokens, so the loop advances to the next shard exactly when needed. In practice a window can span **at most two shards** (windows are $S+1 \le 50$M tokens, shards are 50M), so the `torch.cat` cost is a one-time concatenation of two mmap slices. Single-file layout skips all of this with one slice: `chunk = self.data[start: start + needed].clone()`.

### `_locate`: The Bisect Math

`training/pretrain.py:PretrainDataset._locate` maps a global token index to `(shard_idx, offset_within_shard)` in $O(\log K)$:

```python
def _locate(self, global_idx: int) -> Tuple[int, int]:
    """Map a global token index to (shard_idx, offset_within_shard). ..."""
    if global_idx < 0 or global_idx >= self._total_tokens:
        raise IndexError(
            f"global_idx {global_idx} out of range [0, {self._total_tokens})"
        )
    import bisect
    lo = bisect.bisect_right(self.shard_offsets, global_idx) - 1
    return lo, global_idx - self.shard_offsets[lo]
```

Why `bisect_right` minus one? `shard_offsets` is strictly increasing with `shard_offsets[0] = 0`. For any valid `global_idx`:

- `bisect_right(shard_offsets, global_idx)` returns the insertion point *after* all elements `<= global_idx`. Since `global_idx >= 0 = shard_offsets[0]`, the insertion point is `>= 1`, so `lo >= 0` — the shard index never goes negative.
- Because the insertion point is after *all* equal elements, a `global_idx` that is **exactly** `shard_offsets[k]` (the first token of shard `k`) maps to `lo = k`, offset 0 — the boundary token belongs to the next shard, which is the correct indexing (token `shard_offsets[k]` is physically shard `k`'s first element).
- For `global_idx` strictly inside shard `k` (i.e. `shard_offsets[k] < global_idx < shard_offsets[k+1]`), the insertion point is `k+1`, so `lo = k` and the offset `global_idx - shard_offsets[k]` is in `[1, shard_sizes[k])` — strictly inside the shard.

The explicit bounds check matters: with `bisect` alone, a corrupt index like `-5` or `N_tokens + 3` would silently wrap into a plausible-looking `(shard, offset)` and produce wrong training data. `IndexError` makes the corruption loud. This is exactly the contract guarded by `tests/test_training.py::TestPretrainDataset::test_locate_out_of_range_raises` and `test_locate_edge_case`, plus `test_sharded_cross_boundary` for the stitching loop.

### The uint32 → int64 Cast Boundary

The model boundary is `training/pretrain.py:Pretrainer.train_step`, which casts before entering autocast:

```python
if tokens.dtype != torch.long:
    tokens = tokens.to(torch.long)
if targets.dtype != torch.long:
    targets = targets.to(torch.long)
```

`nn.Embedding` indexes with Long, and `F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), ignore_index=-100)` requires Long targets. `models/transformer.py:Transformer.forward` defensively casts again at its own entry (`if tokens.dtype != torch.long: tokens = tokens.to(torch.long)`), so even a caller that forgets the boundary cannot crash the embedding — the uint32 representation never leaks past the model door. This dual guard is deliberate: the dataset stays compact, the training path stays dtype-correct, and the cost is one upcast of a (8, 2048) batch per micro-step — negligible against the forward/backward compute.

---

## CLI Reference

```bash
python3 data/prepare_data.py --stage pretrain \
    [--mixture PATH] \
    [--data-config PATH] \
    [--data-root PATH] \
    [--source NAME] \
    [--skip-download] \
    [--skip-clean] \
    [--skip-tokenize] \
    [--skip-pack]
```

| Flag | Effect |
|---|---|
| `--stage pretrain` | the only stage (the shim is a pretrain-only entry point) |
| `--mixture PATH` | override the mixture YAML (default: `LLM/shared_data/config/mixture.yaml`) |
| `--data-config PATH` | override the pipeline config (default: the generated `data/data_config.yaml`) |
| `--data-root PATH` | override the data root (default: `$LLM_DATA_ROOT` or `<cwd>/data`) |
| `--source NAME` | process a single mixture source by `id` (e.g. `--source fineweb-edu`) |
| `--skip-download` | re-run from cached raw data |
| `--skip-clean` | re-tokenize only |
| `--skip-tokenize` | re-pack only |
| `--skip-pack` | stop after tokenize |

Every stage is also independently invokable as `python -m shared_data.scripts.<download_raw|clean|tokenize|pack_shards>` from the `LLM/` root — the escape hatch for debugging one stage.

---

## Environment Variables

| Variable | Effect |
|---|---|
| `LLM_DATA_ROOT` | Shared cache root for all LLM projects; overrides the `<cwd>/data` default for `raw/`, `clean/`, `tokens/`, `shards/`, `state/`, `manifest.json` |
| `HF_HOME` / `HF_DATASETS_CACHE` | HuggingFace download cache (raw downloads and the tokenizer) |
| `TOKENIZERS_PARALLELISM=false` | Set by the launch script to silence HF fork warnings when the DataLoader spawns workers |
| `ENABLE_TRITON_KERNELS=1` | Required only if the config requests Triton attention/MoE — unrelated to data, but set alongside in `scripts/launch_a100.sh` |

---

## Operational Runbook

### First-time full pipeline

```bash
export LLM_DATA_ROOT=/path/to/shared_data   # optional; default is <cwd>/data
python3 data/prepare_data.py --stage pretrain
```

Stages run download → clean → tokenize → pack, each as an isolated subprocess. Expect hours to days depending on bandwidth and CPU cores (bandwidth-bound at stage 1, CPU-bound at stage 3). Output lands under the data root: `shards/shard_*.bin` + `shards/manifest.json`. The training configs (`configs/pretrain_a100_422m.yaml` → `data.train_data_path`, and the `scripts/launch_a100.sh` pre-flight) expect the shards at `data/pretrain_chinchilla/` — point them there (or symlink) once packing completes.

### Incremental reruns

| Goal | Flags |
|---|---|
| Re-download one source | `--source NAME` (no skip flags) |
| Re-tokenize after a tokenizer change | `--skip-download --skip-clean` |
| Re-pack only | `--skip-download --skip-clean --skip-tokenize` |
| Resume after a crash | re-run with no flags — per-stage state files skip completed work |

### CI / dev path (no shared_data, no HF auth)

```bash
python3 scripts/build_small_pretrain_data.py   # writes data/pretrain_chinchilla/shard_0000.bin
python3 training/pretrain.py --config configs/pretrain_1650_2m.yaml --no-compile
```

`build_small_pretrain_data.py` streams a small text source (fineweb-edu/wikitext, with a deterministic synthetic fallback), tokenizes with the requested tokenizer (`gpt2` for the 1650 config), and writes a single torch-serialised uint32 shard — byte-compatible with `PretrainDataset`. Never require 8.0B tokens in unit tests; the test suite builds tiny fixture shards instead.

### Output verification

```bash
ls data/pretrain_chinchilla/shard_*.bin | wc -l   # expect 160 shards at the 8.0B target
python -c "
import torch; from pathlib import Path
p = sorted(Path('data/pretrain_chinchilla').glob('shard_*.bin'))[0]
t = torch.load(p, weights_only=True, mmap=True)
print(t.dtype, t.numel())
"
# torch.uint32 50000000
```

---

## Pipeline Failure Modes

| Failure | Symptom | Fix |
|---|---|---|
| Wrong `vocab_size` in the model config | Embedding crash at step 0 | Regenerate `data_config.yaml` and the shards; keep `model.vocab_size` == `DEEPSEEK_VOCAB_SIZE` |
| Tokenizer changed without re-tokenizing | Ids out of range or garbage text | `--skip-download --skip-clean`, re-tokenize + re-pack |
| HF auth for the tokenizer/dataset | Download or tokenize stage fails | `huggingface-cli login`, or use the 1650 config (GPT-2) |
| Incomplete download | Missing source in manifest | Re-run without `--skip-download` (state file resumes) |
| Dedup too aggressive | Tiny corpus / huge `n_dropped_dedup` | Check clean-stage logs; per-source SHA-256 only drops exact normalised duplicates |
| Disk full during pack | Partial shard or failed `os.replace` | Delete `shard_*.bin.tmp` leftovers, free space, re-pack (shards are atomic) |
| Mixture/data-config token totals disagree | `WARNING: mixture.total_tokens != ...` at startup | Fix one of them; the mixture's value wins |
| Stale manifest | SHA mismatch or wrong totals | Re-run the pack stage; the manifest is rebuilt from disk |
| Raw shards from the universal pipeline fed to `PretrainDataset` | `RuntimeError: mmap can only be used with files saved with torch.save` | Convert to torch-serialised form, or use the universal `ShardDataset` reader |
| No shards in the configured dir | `ERROR: no shard_*.bin files in data/pretrain_chinchilla` (launch script) | Run the pipeline, or `build_small_pretrain_data.py` for smoke tests |

---

## Learning Exercises

1. **Token count audit:** Load one shard, count EOS tokens (id 100,017), and estimate documents per 50M-token shard: `(t == 100017).sum()`.
2. **Vocab spot-check:** Decode tokens around 100,000–100,017 — observe the special-token range and byte_fallback behaviour.
3. **Window boundary:** Find a sample index where `idx * max_seq_len` crosses a shard boundary; verify `__getitem__` stitches correctly and matches a manual concatenation.
4. **`_locate` invariants:** For random `global_idx`, assert `shard_offsets[lo] <= global_idx < shard_offsets[lo] + shard_sizes[lo]`; assert `_locate(0) == (0, 0)` and `_locate(total-1)` lands in the last shard.
5. **Sample-count edge case:** With `N` tokens and window `S`, confirm `len(ds) == (N - 1) // S` and that the final partial window is dropped, not padded.
6. **Mixture ablation:** Train a few hundred steps on a single source (e.g. `--source fineweb-edu` then re-pack), compare loss to the full mixture.
7. **Manifest audit:** Verify each shard's SHA-256 in `manifest.json` against the on-disk file; check `config_hash`/`mixture_hash` change when either YAML changes.
8. **Compression ratio:** Tokenize 1 MB of raw text with the DeepSeek tokenizer — compute bytes/token and compare against the 4.0 char/token download estimate.
9. **Resumption drill:** Kill a tokenize run mid-source (Ctrl+C), re-run with no flags, and confirm the state file skips completed docs and the stream is not re-written from zero.
10. **Cast boundary:** Feed `PretrainDataset` output straight into `Transformer.forward` and confirm the defensive `to(torch.long)` in `models/transformer.py:Transformer.forward` accepts the uint32 batch unchanged (dtype-correct output).

---

## Check Your Understanding

**Q1: Why does `PretrainDataset._locate` use `bisect_right` instead of `bisect_left`?** A: `bisect_right` returns the insertion point *after* all elements `<= global_idx`. For a `global_idx` exactly equal to `shard_offsets[k]` — the first token of shard `k` — the insertion point is `k+1`, so `lo = k` and the offset is 0: the boundary token maps to the shard that physically holds it. `bisect_left` would put it in the *previous* shard with an offset equal to that shard's size — one past the end — and the slice `shard[offset:offset+take]` would silently be empty. The off-by-one is the whole point.

**Q2: The corpus target is 8.0B tokens, but repo scripts say "~8.4B". Which is right?** A: 8.0B — it is the value in `mixture.yaml` (`total_tokens`), `UNIVERSAL_TOTAL_TOKENS` in `shared_data/config.py`, and the generated `data/data_config.yaml`, and it is what the pipeline enforces. "~8.4B" is the Chinchilla rationale (~20 tokens/param × 418.7M params with MTP ≈ 8.4B) quoted loosely; the realized budget is 8.0B → 160 shards of 50M tokens → 32 GB.

**Q3: A window of `S + 1 = 2049` tokens starts at global index 49,999,999 (the last token of shard 0). How many shards does it touch, and what does `_locate` return for the first `cursor`?** A: Two shards. `_locate(49_999_999)` returns `(0, 49_999_999)` — `shard_offsets[0] = 0`, and `bisect_right([0, 50_000_000], 49_999_999) - 1 = 0`. `take = min(2049, 50_000_000 - 49_999_999) = 1`, so one token is taken from shard 0; the next `_locate(50_000_000)` returns `(1, 0)` (boundary token belongs to shard 1) and the remaining 2048 tokens come from shard 1.

**Q4: Why is the final partial window dropped instead of padded with PAD (100,016)?** A: Padding would inject fake next-token targets — the model would be trained to predict PAD at the tail of every epoch, and the effective data distribution would include synthetic continuations. Dropping the tail wastes at most `S` tokens per 8.0B and keeps every training pair real. This is also why `add_special_tokens: false` + manual EOS is used: the only synthetic token in the stream is the document-boundary EOS, which is a genuine, learnable event.

---

## Appendix A — FAQ

**Q: Can I use the GPT-2 tokenizer?** Only for the 1650 smoke config (`vocab_size=50257`). The 422M config (`configs/pretrain_a100_422m.yaml`) requires the DeepSeek tokenizer (`vocab_size=100018`).

**Q: Where is the data guide?** The data guide is this document (the standalone data redirect was folded into [Training](../training.md)). The universal pipeline's own README is at `LLM/shared_data/README.md`.

**Q: How long does full prep take?** Hours to days depending on bandwidth and CPU cores (download is bandwidth-bound; tokenize is CPU-bound). Use the `--skip-*` flags and `--source NAME` for incremental reruns. The shared_data README's stage wall-clock table (~8 h once, ~13 h for five projects sharing the clean output) is an estimate — no full run has been executed.

**Q: Does training read `manifest.json`?** No. `PretrainDataset` derives everything (shard list, sizes, offsets, totals) from the shard files themselves. The manifest is the audit/provenance record and the entry point for the universal `ShardDataset` reader.

**Q: What happens at a document boundary inside a window?** Nothing special — the stream is flat, documents are separated by EOS, and a window may span any number of document boundaries. The model learns to predict the first token of the next document after EOS, which is standard GPT-style pretraining. There is no document-start token beyond EOS and no masking of cross-document attention.

---

## Appendix B — Glossary

| Term | Meaning |
|---|---|
| `mixture.yaml` | Canonical source list + weights for the 8.0B corpus (`LLM/shared_data/config/mixture.yaml`) |
| `data_config.yaml` | Per-project tokenizer + pipeline settings, generated by the shim |
| data root | `$LLM_DATA_ROOT` or `<cwd>/data`; holds `raw/`, `clean/`, `tokens/`, `shards/`, `state/` |
| `manifest.json` | Shard metadata, per-source stats, config/mixture hashes |
| `TokenStream` | Per-source binary token stream: 8-byte header + raw uint32, EOS-separated |
| `ShardWriter` | Atomic packer: 50M-token raw uint32 shards, tmp + `os.replace` |
| `PretrainDataset` | This repo's training consumer: mmap'd shards, windowed samples, bisect `_locate` |
| uint32 | 4-byte token storage dtype on disk and in the dataset |
| int64 | Long dtype required by `nn.Embedding` / `F.cross_entropy`; cast at the model boundary |
| byte_fallback | BPE fallback to raw byte tokens; encoding is total, no `<unk>` ever |
| EOS (100,017) | Document boundary token, appended by tokenize and pack |
| PAD (100,016) | Padding token, defined but unused in packed pretraining |

---

## References

- `data/prepare_data.py` — the DeepSeek shim (`_ensure_deepseek_data_config`, `main`)
- `data/data_config.yaml` — generated project config (DeepSeek overrides)
- `training/pretrain.py` — `PretrainDataset` (`__init__`, `_locate`, `__getitem__`) and `Pretrainer.train_step`
- `models/transformer.py` — `Transformer.__init__` (embedding rows) and `Transformer.forward` (defensive cast)
- `scripts/build_small_pretrain_data.py` — torch-serialised single-shard builder for smoke tests
- `LLM/shared_data/README.md` — universal pipeline design doc
- `LLM/shared_data/config/mixture.yaml` — canonical mixture (the only source of truth for weights)
- `LLM/shared_data/config/data_config.yaml` — universal pipeline knobs
- `LLM/shared_data/shard_writer.py`, `manifest.py`, `dedup.py`, `quality_filter.py`, `dataset.py` — pack, manifest, dedup, filter, reader implementations
- [Training](../training.md) — the full training loop that consumes `PretrainDataset`
- [Inference](../inference.md) — tokenizer reuse at inference time (chat template, `generate`)
