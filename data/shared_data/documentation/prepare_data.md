# prepare_data.py — the orchestrator

> See [`../README.md`](../README.md) §4 (usage) for the authoritative
> per-project and cross-project usage examples.

This is the single entrypoint for the universal pipeline. It runs the
four-stage pipeline (download → clean → tokenize → pack) end-to-end,
writing the output under `data/` of the *invoking project* (or under
`$LLM_DATA_ROOT` if set globally to share one cache across all 5 — see
[`common.md`](common.md) for the path-resolution precedence).

## Stage summary

1. `download_raw`   HF → `data/raw/<source>/data.jsonl`
2. `clean`          `data/raw` → `data/clean` (quality + SHA-256 dedup)
3. `tokenize`       `data/clean` → `data/tokens` (uint32 stream)
4. `pack_shards`    `data/tokens` → `data/shards/shard_NNNNN.bin` +
   `manifest.json`

Each stage is invoked as a **subprocess** (`_run_module`) so that a
stage crash (e.g. OOM during tokenisation) doesn't lose intermediate
progress in the orchestrator process.

## run_pipeline

`run_pipeline(...)` is the programmatic entry point. Key arguments:

- `mixture_path` / `data_config_path` — default to the canonical
  `UNIVERSAL_MIXTURE_PATH` / `UNIVERSAL_DATA_CONFIG_PATH` shipped with
  the package (see [`../config.py`](../config.py)).
- `source` — restrict to a single source id (debugging).
- `skip_download` / `skip_clean` / `skip_tokenize` / `skip_pack` —
  per-stage skip flags.
- `skip_train_tokenizer` — off by default (most projects use an HF
  tokenizer); FusionLLM sets `--train-tokenizer` to also run
  `train_tokenizer` before download.
- `data_root` — override `DATA_ROOT` (calls `set_data_root`).

## Idempotency & resume

Each stage is **idempotent and resumable** — re-running picks up where
it left off thanks to per-stage state files in `data/state/` (see
[`common.md`](common.md) — state persistence).

The orchestrator also validates that `mixture.total_tokens` agrees with
`data_config.pipeline.sharding.target_total_tokens` (warns if they
differ, uses the mixture's value).

## Cross-project sharing

The corpus target is fixed at 8.0 B tokens (Chinchilla-optimal for
~500M-param models). The same mixture and the same processing stages are
run regardless of which project's `data/prepare_data.py` shim invokes
the pipeline. The output (raw/clean/tokens/shards/manifest) is
bit-identical across the 5 projects, so any project can mmap the shards
produced by any other.

If `$LLM_DATA_ROOT` is set to a single shared directory and the pipeline
is run once from any project, all 5 projects can train on the same
corpus without re-downloading or re-tokenising (see `../README.md` §4.2
for the wall-clock savings table — ~27 h saved on a single A100).

## CLI

`main()` is the CLI entry point. Flags:

- `--stage pretrain` (only stage currently supported).
- `--mixture` / `--data-config` — override the canonical configs.
- `--data-root` — override `DATA_ROOT`.
- `--source` — restrict to a single source id.
- `--skip-download` / `--skip-clean` / `--skip-tokenize` / `--skip-pack`.
- `--train-tokenizer` — also train a custom BPE before download
  (FusionLLM uses this).