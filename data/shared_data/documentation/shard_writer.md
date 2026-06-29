# shard_writer.py — notes

> See [`../README.md`](../README.md) §7 (shard format), §10 (validation &
> invariants), §11 (performance notes) for the authoritative description.

This is where documents become training-ready tensors. A *shard* is a
flat contiguous binary buffer of token IDs (`uint32` by default — 4
bytes/token, so a 50M-token shard is ~190 MB). The training script mmaps
these directly via `np.memmap(..., mode="r")` for zero-copy slicing.

## Hard guarantees

1. **Atomicity** — shards are written to `.tmp` and renamed via
   `os.replace` on success. A crash mid-write leaves either the old shard
   or the new one — never a half-written one. Critical for the NaN-guard
   in any of the 5 training scripts to roll back to a known-good state.
2. **EOS-separated** — every document boundary is marked with an EOS
   token. We never split a document across shards
   (`cross_document_boundary_ok` defaults to `False`). A training
   window can safely cross an EOS — it's just a regular token — without
   leaking semantic context between unrelated documents.
3. **Determinism** — given the same input tokens and config, the output
   is bit-exact across runs. No random padding, no metadata that varies.
4. **Dtype-aware** — `select_token_dtype(vocab_size)` picks `uint8` /
   `uint16` / `uint32` / `uint64` based on vocab size. The shard's numpy
   dtype is encoded in the manifest so the training script knows how to
   load it.
5. **Verification** — `verify_shard(...)` re-reads the shard and confirms
   exact token count, EOS count, and that no value exceeds
   `vocab_size + 256` (the reserved special-token range). The
   verification SHA-256 is stored in the manifest.

## Why per-source streams then pack?

We tokenize each source independently (`TokenStream`) so we can:

- attribute tokens per source in the manifest (validation of the mix),
- resume per source if one source's download is slower,
- parallelise tokenisation across machines if desired.

Then `pack_shards` (see [`scripts.md`](scripts.md)) reads per-source
token streams in round-robin and writes shards. Round-robin ensures each
shard sees every source (rather than the first N shards being only
FineWeb-Edu), which keeps training loss curves smooth.

## TokenStream

`TokenStream` is an append-only writer for a per-source token stream.
It stores raw bytes (little-endian). We deliberately avoid an in-RAM
list of all tokens — for 1.2 B code tokens, a list-of-ints is ~10 GB; a
binary file is on-disk and mmap'd at read time.

**Header format** (`HEADER_FMT = "<II"`, `HEADER_SIZE = 8` bytes):

| offset | field          | type   | meaning            |
|--------|----------------|--------|--------------------|
| 0      | `version`      | uint32 | stream format ver  |
| 4      | `eos_token_id` | uint32 | EOS token id       |

Body: `uint32` little-endian token IDs, EOS-separated (one EOS after every
document). `write_doc(tokens)` strips a trailing EOS if present, validates
tokens against `vocab_size`, writes the array, then writes a single EOS.
`n_written` / `n_eos` are read-only properties. Context-manager friendly
(`__enter__` / `__exit__` calls `close()`).

`read_token_stream(path, *, mmap=True)` is the reader: validates the
header, then yields `np.ndarray` documents by splitting on EOS. Raises
if the body size isn't a multiple of 4 bytes or if no EOS is found
(corrupt stream).

## ShardWriter

`ShardWriter` fills a fixed `shard_size_tokens` buffer and flushes to
disk via `_flush()`:

1. Write `shard_NNNNN.bin.tmp`.
2. `os.fsync` the file.
3. `os.replace(tmp, shard_NNNNN.bin)` — atomic rename.
4. Compute SHA-256 of the raw bytes and `n_eos = count_nonzero(payload == eos)`.
5. Append a `ShardMeta` to `self._shards`.

`add(doc)` validates, splits oversized docs when
`cross_document_boundary_ok=True`, and delegates to `_add_internal`
which appends the doc + an EOS to the buffer, flushing first if the doc
would overflow the current shard. `finalize()` flushes the partial
buffer and marks the writer as finalized.

## ShardMeta

`ShardMeta` is the per-shard verification metadata:
`index`, `path` (relative to `output_dir.parent`), `n_tokens`,
`sha256`, `n_eos`. Used to build the manifest's `shards` list.

## verify_shard

`verify_shard(shard_path, *, expected_tokens, expected_dtype, vocab_size,
eos_token_id) -> dict` re-reads the shard and returns:

```python
{"ok": True, "actual_tokens": int, "max_token": int,
 "n_eos": int, "sha256": str}
```

Raises on file-size misalignment, token-count mismatch, or
out-of-vocab tokens (`max_token >= vocab_size + 256`).