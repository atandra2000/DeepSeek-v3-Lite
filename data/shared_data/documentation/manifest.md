# manifest.py — notes

> See [`../README.md`](../README.md) §6 (the manifest) for the
> authoritative description and a full JSON example.

A *manifest* is a JSON file at `data/manifest.json` that records the full
provenance of the training corpus: which sources contributed, how many
tokens each produced, the tokenizer used, and SHA-256 fingerprints of
every shard. This is critical for:

- **Reproducibility** — `manifest.json` + `mixture.yaml` + `data_config.yaml`
  fully describe how the corpus was built. A future rerun with the same
  inputs produces bit-identical shards.
- **Cross-project sharing** — each of the 5 LLM projects reads the same
  `manifest.json` to know the EOS id, total token count, vocab size,
  dtype, and per-source mix — guaranteeing all 5 train on identical data.
- **Validation** — `Manifest.validate()` enforces shard count, total
  tokens, EOS coverage, and per-source mix.
- **Dataset alignment** — `shard_reader.py` reads the manifest to know
  the EOS token id and the list of shards.

## Manifest schema (v1.0.0)

```json
{
  "version": "1.0.0",
  "created_utc": "2026-06-29T01:23:45Z",
  "vocab_size": 128000,
  "eos_token_id": 128009,
  "pad_token_id": 128002,
  "tokenizer_name": "llama3",
  "dtype": "uint32",
  "shard_size_tokens": 50000000,
  "total_tokens": 8000000000,
  "shard_count": 161,
  "shards_dir": "data/shards",
  "shards": [
    {"index": 0, "path": "shards/shard_00000.bin",
     "n_tokens": 50000000, "sha256": "...", "n_eos": 12345},
    ...
  ],
  "sources": {
    "fineweb-edu":      {"target_tokens": 4000000000,
                         "actual_tokens": 3998234567,
                         "n_docs": 12345678,
                         "n_dedup_dropped": 23456},
    ...
  },
  "config_hash": "...",   # SHA-256 of merged config dict
  "mixture_hash": "...",  # SHA-256 of mixture.yaml contents
}
```

## Dataclasses

- **`ShardInfo`** — `index`, `path` (relative to `DATA_ROOT`), `n_tokens`,
  `sha256` (hex SHA-256 of the shard's raw bytes), `n_eos` (count of EOS
  tokens in the shard, for sanity).
- **`SourceInfo`** — `target_tokens` (planned: `mixture.weight ×
  total_tokens`), `actual_tokens` (produced), `n_docs` (documents that
  contributed), `n_dedup_dropped` (documents removed by dedup),
  `shard_count` (default 0; how many of the global shards this source spans).
- **`Manifest`** — the top-level container; see schema above. `to_dict`
  serialises, `save(path)` atomically writes (auto-stamps `created_utc`),
  `load(path)` deserialises (with tolerant defaults for forward-compat),
  `exists(path)` is a presence check, `validate(strict=True)` returns a
  list of issue strings.

## Validation invariants (`Manifest.validate`)

- `vocab_size > 0`
- `0 <= eos_token_id < vocab_size + 256` (the reserved special-token range)
- `shard_count > 0`
- `total_tokens > 0`
- `len(shards) == shard_count`
- `|Σ source.actual_tokens - total_tokens| <= max(1, total_tokens // 1000)`

The manifest is validated before it's saved. If anything is off, the
pipeline aborts. This is the same fail-fast discipline applied to model
weights — silent corruption is the enemy of reproducibility.

## Provenance hashes

- `hash_yaml(path) -> str` — SHA-256 of a YAML file's raw text. Used to
  stamp `manifest.mixture_hash`.
- `hash_config(cfg) -> str` — SHA-256 of a config dict with stable key
  ordering (`json.dumps(..., sort_keys=True, default=str)`). Used to
  stamp `manifest.config_hash`.

These hashes let a downstream reader refuse to load a corpus whose
mixture or config has drifted from the canonical recipe.