# scripts/ — the 5 stage scripts

> See [`../README.md`](../README.md) §2 (the 5-stage pipeline diagram),
> §4 (usage), §11 (performance notes) for the authoritative description.

Each script is a CLI entrypoint. They are invoked either directly
(`python -m shared_data.scripts.download_raw`) or via the orchestrator
(`shared_data.prepare_data.run_pipeline`, see
[`prepare_data.md`](prepare_data.md)).

| Script | Stage | Input → Output |
|--------|-------|---------------|
| `download_raw.py`    | 1 | HuggingFace → `data/raw/<source>/data.jsonl` |
| `clean.py`          | 2 | `data/raw/<source>/data.jsonl` → `data/clean/<source>/data.jsonl` (quality + SHA-256 dedup) |
| `tokenize.py`       | 3 | `data/clean/<source>/data.jsonl` → `data/tokens/<source>/data.bin` (uint32, EOS-separated) |
| `pack_shards.py`    | 4 | `data/tokens/<source>/data.bin` (round-robin) → `data/shards/shard_NNNNN.bin` + `data/manifest.json` |
| `train_tokenizer.py`| (opt) | trains a custom BPE on `data/clean/**` → `data/tokenizer/` (FusionLLM only) |

All stages are independently resumable via `state/<stage>_<source>.json`
(see [`common.md`](common.md) — state persistence).

## download_raw.py

For each source in `mixture.yaml`:

1. Open the dataset with `load_dataset(name, config, split=split,
   streaming=True)`. Streaming avoids downloading the full dataset
   upfront (FineWeb-Edu alone is 1.3 TB — full download would OOM most
   machines).
2. Concatenate `text_field` (+ optional `extra_text_field` joined by
   `extra_separator`).
3. Write one JSONL record per line to `data/raw/<source>/data.jsonl`.

The streaming dataset is **resumable**: we record the last-seen row
index in `state/download_<source>.json`. Re-running picks up where we
left off, appending to the JSONL. Stop after `target_chars =
target_tokens * chars_per_token` (default `chars_per_token=4.0`).

**Why one JSONL per source?**

- Resume is per-source (if one source's download stalls, the others
  keep progressing).
- The dedup pass is per-source (faster + smaller working set).
- Tokenisation is per-source (we can attribute tokens in the manifest).

`datasets` is a heavy import (~5 s). We defer it until `download_source`
is actually called so the rest of the pipeline can be imported without
the dep installed (important for unit tests).

## clean.py

For each source:

1. Read `data/raw/<source>/data.jsonl` (one document per line).
2. Apply `QualityFilter` (length, char ratio, language hint — see
   [`quality_filter.md`](quality_filter.md)). Drop rejected docs and
   record reasons in `FilterStats`.
3. Apply `Deduper` (SHA-256 → 256 hash buckets → write unique docs to
   `data/clean/<source>/data.jsonl` — see [`dedup.md`](dedup.md)).

Both sub-stages are independently resumable via
`state/clean_<source>.json`. State is checkpointed every 100 000 docs.

**Why filter and dedup before tokenisation?**

- We never spend tokens on documents we'll throw away.
- The dedup hash files are SHA-256 of normalised text — cheap.
- We can compute the *exact* dedup ratio per source (recorded in the
  manifest) without re-hashing on each pipeline run.

## tokenize.py

For each source:

1. Read `data/clean/<source>/data.jsonl` (one document per line).
2. Tokenize into `data/tokens/<source>/data.bin` via `TokenStream` (see
   [`shard_writer.md`](shard_writer.md)). EOS is appended after every
   document.

The stage is resumable via `state/tokenize_<source>.json`. If a
previous run marked `complete=True`, the existing token stream is
reused (skipped).

`load_tokenizer` tries an HF `AutoTokenizer` from `tokenizer.path`; if
`transformers` is not installed or the path is bad, it falls back to
`_TiktokenFallback` (tiktoken `cl100k_base`). The fallback **changes
EOS semantics** (cl100k_base EOT is 100257, not 128009) and warns the
user — it's only meant for tests / quick smoke runs.

## pack_shards.py

Reads `data/tokens/<source>/data.bin` for each source and writes a
sequence of uniform-sized shards under `data/shards/shard_NNNNN.bin`.

**Round-robin packing** — sources are interleaved one-document-at-a-time
into the buffer (`interleave_sources`). This prevents the first N shards
being only FineWeb-Edu (the 50% source) and the last being only arxiv
(the 5% source), which would cause the loss curve to "drift" as the
source mix shifts during training.

**Resumability** — per-shard state is saved after each successful write
(`state/pack_shards.json`). Re-running the script will skip
already-written shards (their content is unchanged between runs because
the inputs are immutable).

**Manifest** — after all shards are written, `data/manifest.json` is
generated (see [`manifest.md`](manifest.md)) with per-source token
counts, shard SHA-256s, EOS token id, vocab size, and hash of
`mixture.yaml` + `data_config.yaml` (for reproducibility). The manifest
is validated before being saved. Existing shards on disk are also
folded into the manifest (with empty `sha256` / `n_eos` — those are only
populated for shards written this run).

## train_tokenizer.py

Optional stage used by projects that don't ship with a pre-trained
tokenizer (e.g. FusionLLM's custom 64K BPE). The output goes into
`data/tokenizer/` and is picked up by the tokenize stage via
`data_config.yaml:tokenizer.path`.

The training reads from `data/clean/<source>/data.jsonl` (already
quality-filtered + dedup'd by the clean stage) so we never train on
junk. The vocabulary is sized via `--vocab-size` (default 64 000) and
special tokens are configured via the same flags as the rest of the
pipeline:

| special token        | id |
|----------------------|----|
| `""` (EOS)           | 0  |
| `<\|startoftext\|>` (BOS) | 1  |
| `<\|pad\|>` (PAD)         | 2  |
| `<\|unk\|>` (UNK)         | 3  |