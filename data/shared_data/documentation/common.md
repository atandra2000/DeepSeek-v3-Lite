# common.py — notes

> See [`../README.md`](../README.md) §3 (directory layout), §10 (validation &
> invariants), §11 (performance notes) for the authoritative description.

## Data-root resolution precedence

`_resolve_data_root()` picks the project's `data/` root in this order:

1. `$LLM_DATA_ROOT` — explicit global override (single shared cache
   across all 5 projects).
2. `$LLM_PROJECT_ROOT/data` — per-project override.
3. `$PWD/data` — the current working directory's `data/` folder (default).

All five LLM projects invoke the pipeline from their own project
directory, so the default of `$PWD/data` matches user expectations: each
project keeps its own `data/` tree. Users who want a single shared cache
set `LLM_DATA_ROOT=/path/to/uni` once and all 5 projects read from the
same shards (saves ~27 h of duplicate download/clean work; see
`../README.md` §4.2 for the wall-clock table).

`set_data_root(path)` is used by the per-project shims
(`data/prepare_data.py`) so the pipeline writes to the *invoking
project's* data directory rather than wherever the Python process was
launched.

## Directory layout (auto-created on first write)

```
data/
  raw/        downloaded JSONL from HuggingFace (one file per source)
  clean/      quality-filtered + dedup'd JSONL (one file per source)
  tokens/     flat uint32 token streams per source (binary, EOS-separated)
  shards/     packed, training-ready shard_NNNNN.bin files
  state/      resumable state files per pipeline stage
  config/     YAML configs (mixture.yaml, data_config.yaml)
  manifest.json   final provenance written by pack_shards
```

Pipeline stages: `download → clean → tokenize → pack → manifest`.

The pipeline is **idempotent** at every stage. Re-running picks up where
it left off thanks to per-stage state files in `data/state/`.

## Atomic IO (POSIX rename)

`atomic_write_bytes` / `atomic_write_json` write to a sibling temp file
then `os.replace()` it into place. A crash mid-write leaves either the
old file or the new one — never a half-written one. This is the same
discipline applied to model checkpoints (see workspace AGENTS.md §9).

`_json_default` handles numpy/torch scalars (via `.item()` / `.tolist()`).

## SHA-256 hashing & bucket assignment

- `sha256_bytes(data)` — hex SHA-256 of a byte string.
- `sha256_text(text)` — SHA-256 of a UTF-8 string with whitespace
  normalisation (`" ".join(text.split())`). Whitespace-only differences
  across documents do not cause false duplicates to evade dedup. Case
  and punctuation are preserved (semantically meaningful).
- `hash_to_bucket(sha, n_buckets)` — `int(sha[:8], 16) % n_buckets`.
  First 8 hex chars (32 bits) mod `n_buckets`.

SHA-256 is the only hash used (no `hash()` randomness) — determinism across
runs is guaranteed.

## State persistence (resumability)

- `load_state(stage)` — reads `data/state/<stage>.json`; returns `{}` if
  absent or corrupt (corrupt state logs a warning and starts fresh).
- `save_state(stage, state)` — atomically writes the per-stage state.
- `clear_state(stage)` — wipes the state for a stage (used when restarting
  from scratch).

## Vocabulary / EOS conventions

The pipeline is **vocab-agnostic** at the token-stream layer (it stores
uint32 IDs and validates against `vocab_size` only). The defaults here
match the LLaMA-3 BPE used by GPT-OSS-Lite and LLaMA-3-Lite:

- `DEFAULT_VOCAB_SIZE = 128_000`   (LLaMA-3 BPE)
- `DEFAULT_EOS_TOKEN_ID = 128_009`  (LLaMA-3 `<|eot_id|>`)
- `DEFAULT_PAD_TOKEN_ID = 128_002` (reserved in LLaMA-3)

Each project using a different tokenizer (Mamba-2: GPT-2,
DeepSeek-v3: deepseek-coder-v2, FusionLLM: 64K custom) overrides these via
`data_config.yaml` and the orchestrator's `--vocab-size` / `--eos-token-id`
flags.

## Logging

A single shared `shared_data` logger writes to stderr with a `[data]`
prefix. Child modules call `get_logger(__name__)`. `log(msg)` prepends a
timestamp and uses `_logger.log(level, ...)` so it won't interleave with
tqdm progress bars. Level is controlled by `$LLM_DATA_LOG` (default INFO).

## Reproducibility

`seed_everything(seed)` seeds Python `random`, NumPy, and PyTorch (CPU +
CUDA). Called by the orchestrator with `seed = cfg["pipeline"]["seed"]`
(default 42).

## Iterator helpers

- `iter_jsonl(path)` — yields dicts from a JSONL file; skips malformed
  lines with a warning (never raises on a single bad line).
- `write_jsonl(path, records)` — writes JSONL (UTF-8, `ensure_ascii=False`);
  returns the count written.