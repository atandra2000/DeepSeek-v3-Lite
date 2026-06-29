"""SHA-256 document deduplication with hash sharding (shared pipeline)."""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Iterable, Optional

from shared_data.common import (
    STATE_ROOT,
    atomic_write_bytes,
    hash_to_bucket,
    human_count,
    log,
    save_state,
    sha256_text,
)


class BloomFilter:
    """Classic bloom filter using SHA-256 slices into k integer indices."""

    def __init__(self, capacity: int, error_rate: float = 0.001):
        import math
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        if not (0 < error_rate < 1):
            raise ValueError(f"error_rate must be in (0, 1), got {error_rate}")
        self.capacity = capacity
        self.error_rate = error_rate
        m_raw = -capacity * math.log(error_rate) / (math.log(2) ** 2)
        m_bits = 1 << (int(m_raw).bit_length())  # next power of 2
        k = max(1, round((m_bits / capacity) * math.log(2)))
        self.m_bits = m_bits
        self.k = k
        self.bitmap = bytearray(m_bits // 8)
        self._added = 0

    def _indices(self, item_hash: str) -> Iterable[int]:
        """Deterministically derive k distinct indices from the 256-bit hash."""
        h = bytes.fromhex(item_hash)
        n_chunks = len(h) // 4
        chunks = [
            int.from_bytes(h[i * 4:(i + 1) * 4], "big") for i in range(n_chunks)
        ]
        for i in range(self.k):
            yield chunks[i % n_chunks] & (self.m_bits - 1)

    def add(self, item_hash: str) -> bool:
        """Add a hash. Returns True if likely already present, False if definitely new."""
        present = True
        for idx in self._indices(item_hash):
            byte = idx >> 3
            mask = 1 << (idx & 7)
            if not (self.bitmap[byte] & mask):
                present = False
                self.bitmap[byte] |= mask
        if not present:
            self._added += 1
        return present

    def __contains__(self, item_hash: str) -> bool:
        for idx in self._indices(item_hash):
            byte = idx >> 3
            mask = 1 << (idx & 7)
            if not (self.bitmap[byte] & mask):
                return False
        return True

    @property
    def n_added(self) -> int:
        return self._added

    def save(self, path: Path) -> None:
        atomic_write_bytes(
            path,
            pickle.dumps({
                "capacity": self.capacity,
                "error_rate": self.error_rate,
                "m_bits": self.m_bits,
                "k": self.k,
                "bitmap": bytes(self.bitmap),
                "added": self._added,
            }),
        )

    @classmethod
    def load(cls, path: Path) -> "BloomFilter":
        d = pickle.loads(Path(path).read_bytes())
        bf = cls(capacity=d["capacity"], error_rate=d["error_rate"])
        bf.m_bits = d["m_bits"]
        bf.k = d["k"]
        bf.bitmap = bytearray(d["bitmap"])
        bf._added = d["added"]
        return bf


class Deduper:
    """Two-pass, constant-memory dedup of an iterable of (id, text) records."""

    def __init__(
        self,
        source_id: str,
        n_buckets: int = 256,
        bloom_capacity_per_bucket: int = 200_000,
        bloom_error_rate: float = 0.001,
        use_bloom: bool = True,
    ):
        self.source_id = source_id
        self.n_buckets = n_buckets
        self.bloom_capacity = bloom_capacity_per_bucket
        self.bloom_error_rate = bloom_error_rate
        self.use_bloom = use_bloom

        self.workdir = STATE_ROOT / "dedup" / source_id
        self.hash_dir = self.workdir / "hash_buckets"
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.hash_dir.mkdir(parents=True, exist_ok=True)

    def collect(self, records: Iterable[tuple[str, str]], state: Optional[dict] = None) -> dict:
        """Hash every record and write its hash to the appropriate bucket."""
        state = dict(state or {})
        n_processed = int(state.get("n_processed", 0))
        n_unique = int(state.get("n_unique", 0))
        n_duplicate = int(state.get("n_duplicate", 0))

        blooms = [
            BloomFilter(self.bloom_capacity, self.bloom_error_rate)
            for _ in range(self.n_buckets)
        ] if self.use_bloom else [None] * self.n_buckets

        bucket_files = [
            open(self.hash_dir / f"{b:05d}", "a", encoding="utf-8")
            for b in range(self.n_buckets)
        ]
        try:
            for i, (doc_id, text) in enumerate(records):
                if i < n_processed:
                    continue
                sha = sha256_text(text)
                bucket = hash_to_bucket(sha, self.n_buckets)
                bf = blooms[bucket]
                if bf is None or not bf.add(sha):
                    n_unique += 1
                else:
                    n_duplicate += 1
                bucket_files[bucket].write(sha + "\n")
                n_processed += 1
                if n_processed % 200_000 == 0:
                    log(
                        f"dedup[{self.source_id}] "
                        f"{n_processed:,} docs processed "
                        f"({n_unique:,} unique, {n_duplicate:,} dup)",
                    )
                    state["n_processed"] = n_processed
                    state["n_unique"] = n_unique
                    state["n_duplicate"] = n_duplicate
                    save_state(f"dedup_{self.source_id}", state)
                    for f in bucket_files:
                        f.flush()
        finally:
            for f in bucket_files:
                f.close()

        state["n_processed"] = n_processed
        state["n_unique"] = n_unique
        state["n_duplicate"] = n_duplicate
        save_state(f"dedup_{self.source_id}", state)
        if self.use_bloom:
            for b, bf in enumerate(blooms):
                bf.save(self.workdir / f"bloom_{b:05d}.pkl")
        return state

    def write_unique(
        self,
        records: Iterable[tuple[str, str]],
        out_path: Path,
        state: Optional[dict] = None,
    ) -> dict:
        """Re-read records, write only those whose hash is unique in their bucket."""
        state = dict(state or {})
        n_processed = int(state.get("write_n_processed", 0))
        n_kept = int(state.get("write_n_kept", 0))
        n_dropped = int(state.get("write_n_dropped", 0))

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        bucket_sets = []
        for b in range(self.n_buckets):
            p = self.hash_dir / f"{b:05d}"
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    bucket_sets.append(set(f.read().splitlines()))
            else:
                bucket_sets.append(set())

        seen_in_bucket = [set() for _ in range(self.n_buckets)]
        in_bucket_dup = [0] * self.n_buckets

        try:
            with open(out_path, "a", encoding="utf-8") as out_f:
                for i, (doc_id, text) in enumerate(records):
                    if i < n_processed:
                        continue
                    sha = sha256_text(text)
                    bucket = hash_to_bucket(sha, self.n_buckets)
                    bucket_seen = bucket_sets[bucket]
                    bucket_local = seen_in_bucket[bucket]

                    if sha in bucket_local:
                        n_dropped += 1
                        in_bucket_dup[bucket] += 1
                    elif sha in bucket_seen:
                        bucket_local.add(sha)
                        out_f.write(_format_record(doc_id, text) + "\n")
                        n_kept += 1
                    else:
                        bucket_local.add(sha)
                        out_f.write(_format_record(doc_id, text) + "\n")
                        n_kept += 1
                    n_processed += 1

                    if n_processed % 200_000 == 0:
                        log(
                            f"dedup[write:{self.source_id}] "
                            f"{n_processed:,} docs, {n_kept:,} kept, "
                            f"{n_dropped:,} dropped",
                        )
                        state["write_n_processed"] = n_processed
                        state["write_n_kept"] = n_kept
                        state["write_n_dropped"] = n_dropped
                        save_state(f"dedup_{self.source_id}", state)
        finally:
            pass

        state["write_n_processed"] = n_processed
        state["write_n_kept"] = n_kept
        state["write_n_dropped"] = n_dropped
        state["in_bucket_dup"] = in_bucket_dup
        save_state(f"dedup_{self.source_id}", state)
        return state


def _format_record(doc_id: str, text: str) -> str:
    """JSON-encode a single record for the dedup-output JSONL file."""
    import json
    return json.dumps({"id": doc_id, "text": text}, ensure_ascii=False)


__all__ = ["BloomFilter", "Deduper"]