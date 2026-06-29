# shard_reader.py — notes

> See [`../README.md`](../README.md) §7 (shard format), §9 (how to
> consume shards), §10 (validation & invariants) for the authoritative
> description.

This module is the **consumer-side** companion to `shard_writer.py`
(see [`shard_writer.md`](shard_writer.md)). Every one of the 5 LLM
projects calls `open_shard_dataset(...)` or `open_shard_memmaps(...)` to
load the prepared corpus for training.

The reader is **vocab-agnostic** at the bytes level: it just mmap's
`shard_NNNNN.bin` as a `uint32` array and slices it. Each project's own
dataset wrapper interprets the IDs through its own tokenizer.

## Why a separate reader module?

- The writer writes to the project that invoked it (via
  `shared_data.common.set_data_root`).
- The reader must work from any project that imports it.

## API surface

- `load_manifest(manifest_path=None) -> Manifest` — reads
  `data/manifest.json` (raises `FileNotFoundError` with a helpful
  "run the pipeline first" message if missing).
- `shard_paths(manifest) -> List[Path]` — absolute paths to every shard
  in index order.
- `shard_total_tokens(manifest) -> int` — total tokens across all shards.
- `open_shard_memmaps(manifest=None, *, dtype="uint32") -> List[ShardMemmap]`
  — opens every shard as a memmap'd uint32 array. The whole 8B-token
  corpus (32 GB at uint32) lives on disk; we mmap it lazily so RAM usage
  is essentially zero. Missing shards log a warning and are skipped.
- `open_shard_memmap(manifest=None, *, dtype="uint32") -> np.memmap` —
  opens the first shard only (prefer `open_shard_memmaps` for training;
  this is mainly for diagnostics / quick `tokens[start:end]` reads).
- `iter_shards(manifest=None, *, dtype="uint32") -> Iterator[ShardMemmap]`
  — yields each shard in order.

## ShardMemmap

`ShardMemmap` is a dataclass: `index`, `path`, `n_tokens`, `mmap: np.memmap`.

## ShardDataset

`ShardDataset` is a simple `Dataset`-like iterator over packed shards.
Returns `(input, target)` numpy arrays of shape `(seq_len,)` each,
produced by sliding a window across the concatenated shards. EOS tokens
serve as natural document boundaries — the caller decides whether to
reset state at EOS.

| field | type | meaning |
|-------|------|---------|
| `seq_len`    | int | window length (excludes the +1 target shift) |
| `eos_id`     | int | token id used as document separator |
| `vocab_size` | int | sanity-check upper bound on token ids (defensive) |
| `stride`     | int | window stride; defaults to `seq_len` (non-overlapping) |

- `open(manifest)` — bind the dataset to a manifest (opens all shards).
- `reset()` — rewind to the first shard.
- `n_chunks` — `sum(s.n_tokens) // (seq_len + 1)`.
- `get_chunk(idx) -> (input, target)` — returns the `idx`-th
  `(seq_len+1)`-token chunk. `input = chunk[:-1]`, `target = chunk[1:]`.
  Fast path: chunk fits in one shard. Slow path (`_gather_chunk`):
  chunk straddles a shard boundary — gather from multiple shards (rare).

## open_shard_dataset

`open_shard_dataset(seq_len, eos_id, vocab_size, *, stride=0,
manifest=None) -> ShardDataset` — convenience constructor. Loads
`data/manifest.json` (or accepts a pre-loaded one), opens the shards as
memmaps, and returns a ready-to-use `ShardDataset`. This is the
recommended entry point for cross-project training scripts; see
`LLaMA-3-Lite/dataset.py::UniversalShardDataset` for a working reference.