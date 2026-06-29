# dedup.py — notes

> See [`../README.md`](../README.md) §11 (performance notes — "256 hash
> buckets", "Bloom filters") for the authoritative description.

## Why hash sharding?

A naive dedup holds a Python `set` of every document hash seen so far.
At 8 B tokens (roughly 200 M documents at ~40 tokens/doc on average),
that set alone is ~12 GB of Python objects — too much for a typical
data-prep machine.

The approach:

1. Hash every document (SHA-256 of normalised text — see
   [`common.md`](common.md)).
2. Bucket by `hash_to_bucket(sha, n_buckets)` (modulo `n_buckets`).
3. Persist hashes to per-bucket files (`hash_buckets/00000`, ...).
4. For each bucket, load just *that bucket's* hashes (≤ 12 MB / 256
   buckets → fits comfortably in RAM) and dedup against a local set.

This makes dedup **O(N) time, O(N / n_buckets) memory**, deterministic
across runs (SHA-256 is bit-stable), and trivially resumable (each
bucket's output is independent).

## BloomFilter

`BloomFilter` is a classic bloom filter using SHA-256 slices into k
integer indices. Deterministic across Python versions (no `hash()`
randomness).

Sizing follows the standard formulas:

```
m = -n * ln(p) / (ln(2)^2)         # bits
k = (m / n) * ln(2)                # number of hash functions
```

where `n` = expected capacity, `p` = target false-positive rate. `m` is
rounded up to the next power of two so the bit-mask `& (m_bits - 1)` works.

Each bucket also has a 200k-capacity Bloom filter (0.1% false-positive
rate) for fast "have I seen this?" checks in pass 1. The exact set is
recovered from the on-disk bucket files in pass 2.

`add(item_hash)` returns True if likely already present (might be a false
positive), False if definitely new. `__contains__` is a non-mutating
membership test. `save` / `load` use `atomic_write_bytes` + pickle.

## Deduper — two-pass, constant-memory

`Deduper` is a two-pass, constant-memory dedup of an iterable of
`(id, text)` records.

**Pass 1 — `collect(records, state)`:** hash every record and write its
hash to the appropriate bucket. A per-bucket `BloomFilter` (when
`use_bloom=True`) provides a fast "have I seen this?" pre-filter. State
is checkpointed every 200 000 docs (`save_state`), so a crash resumes
exactly where it left off. After pass 1, blooms are saved to
`bloom_NNNNN.pkl` for inspection.

**Pass 2 — `write_unique(records, out_path, state)`:** re-reads the source
records and writes only those whose hash is unique in their bucket.
Records must be iterated in the same order as `collect`. The on-disk
bucket files are loaded into per-bucket `set`s (each ≤ ~50 MB for 256
buckets / 200 M docs). A per-bucket `seen_in_bucket` set tracks
in-bucket duplicates separately.

Both passes are independently resumable via `state.json`.

## Defaults

| Parameter | Default | Notes |
|-----------|---------|-------|
| `n_buckets`            | 256               | 256 → ≤ 12 MB / bucket working set |
| `bloom_capacity_per_bucket` | 200 000       | per-bucket Bloom capacity |
| `bloom_error_rate`     | 0.001             | 0.1% false-positive rate |
| `use_bloom`            | `True`            | set `False` to skip Bloom pre-filter |