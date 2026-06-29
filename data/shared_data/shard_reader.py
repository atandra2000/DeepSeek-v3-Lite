"""Cross-project shard reader for the shared LLM data pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np

from shared_data.common import (
    DATA_ROOT,
    MANIFEST_PATH,
    SHARDS_ROOT,
    get_logger,
    human_bytes,
)
from shared_data.manifest import Manifest


log = get_logger("shard_reader")


def load_manifest(manifest_path: Optional[Path] = None) -> Manifest:
    """Load ``data/manifest.json`` for the current data root (raises if missing)."""
    path = Path(manifest_path) if manifest_path else MANIFEST_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"No manifest at {path}. Run the data pipeline first:\n"
            f"  python data/prepare_data.py --stage pretrain"
        )
    return Manifest.load(path)


def shard_paths(manifest: Manifest) -> List[Path]:
    """Return the absolute path to every shard, in index order."""
    base = DATA_ROOT
    return [base / s.path for s in manifest.shards]


def shard_total_tokens(manifest: Manifest) -> int:
    """Total tokens across all shards."""
    return int(manifest.total_tokens)


@dataclass
class ShardMemmap:
    """Memory-mapped view of a single training shard."""
    index: int
    path: Path
    n_tokens: int
    mmap: np.memmap


def open_shard_memmaps(
    manifest: Optional[Manifest] = None,
    *,
    dtype: str = "uint32",
) -> List[ShardMemmap]:
    """Open every shard in the manifest as a memmap'd uint32 array."""
    if manifest is None:
        manifest = load_manifest()
    np_dtype = np.dtype(dtype)
    out: List[ShardMemmap] = []
    for s in manifest.shards:
        path = DATA_ROOT / s.path
        if not path.exists():
            log(f"WARNING: shard missing at {path}")
            continue
        m = np.memmap(path, dtype=np_dtype, mode="r")
        out.append(ShardMemmap(
            index=s.index,
            path=path,
            n_tokens=int(m.shape[0]),
            mmap=m,
        ))
    return out


def open_shard_memmap(
    manifest: Optional[Manifest] = None,
    *,
    dtype: str = "uint32",
) -> np.memmap:
    """Open the first shard as a memmap (prefer ``open_shard_memmaps`` for training)."""
    shs = open_shard_memmaps(manifest, dtype=dtype)
    if not shs:
        raise FileNotFoundError("No shards available")
    return shs[0].mmap


def iter_shards(
    manifest: Optional[Manifest] = None,
    *,
    dtype: str = "uint32",
) -> Iterator[ShardMemmap]:
    """Yield each shard as a memmap, in order."""
    yield from open_shard_memmaps(manifest, dtype=dtype)


@dataclass
class ShardDataset:
    """Simple ``Dataset``-like iterator over packed shards."""
    seq_len: int
    eos_id: int
    vocab_size: int
    stride: int = 0
    _shards: Optional[List[ShardMemmap]] = None
    _cur_shard: int = 0
    _cur_pos: int = 0
    _manifest: Optional[Manifest] = None

    def __post_init__(self) -> None:
        if self.stride <= 0:
            self.stride = self.seq_len

    def open(self, manifest: Manifest) -> None:
        """Bind the dataset to a manifest."""
        self._manifest = manifest
        self._shards = open_shard_memmaps(manifest)
        self._cur_shard = 0
        self._cur_pos = 0

    def reset(self) -> None:
        self._cur_shard = 0
        self._cur_pos = 0

    @property
    def n_chunks(self) -> int:
        if self._shards is None:
            return 0
        total = sum(s.n_tokens for s in self._shards)
        return max(0, total // (self.seq_len + 1))

    def __len__(self) -> int:
        return self.n_chunks

    def get_chunk(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return the ``idx``-th ``(seq_len+1)``-token chunk (input=chunk[:-1], target=chunk[1:])."""
        if self._shards is None:
            raise RuntimeError("Call open(manifest) before get_chunk()")
        offset = idx * (self.seq_len + 1)
        running = 0
        for s in self._shards:
            if running + s.n_tokens > offset:
                local_off = offset - running
                if local_off + self.seq_len + 1 <= s.n_tokens:
                    chunk = np.array(
                        s.mmap[local_off: local_off + self.seq_len + 1],
                        copy=True,
                    )
                    return chunk[:-1], chunk[1:]
                break
            running += s.n_tokens
        return self._gather_chunk(offset)

    def _gather_chunk(self, offset: int) -> Tuple[np.ndarray, np.ndarray]:
        """Gather a chunk that straddles shard boundaries (rare)."""
        total_needed = self.seq_len + 1
        out = np.empty(total_needed, dtype=np.uint32)
        filled = 0
        running = 0
        for s in self._shards:
            if filled >= total_needed:
                break
            if running + s.n_tokens > offset:
                local_off = max(0, offset - running)
                take = min(s.n_tokens - local_off, total_needed - filled)
                out[filled:filled + take] = s.mmap[local_off: local_off + take]
                filled += take
                offset = running + s.n_tokens
            running += s.n_tokens
        return out[:-1], out[1:]


def open_shard_dataset(
    seq_len: int,
    eos_id: int,
    vocab_size: int,
    *,
    stride: int = 0,
    manifest: Optional[Manifest] = None,
) -> ShardDataset:
    """Construct a ready-to-use :class:`ShardDataset` bound to ``data/manifest.json``."""
    if manifest is None:
        manifest = load_manifest()
    ds = ShardDataset(
        seq_len=seq_len,
        eos_id=eos_id,
        vocab_size=vocab_size,
        stride=stride or seq_len,
    )
    ds.open(manifest)
    return ds


__all__ = [
    "ShardMemmap", "ShardDataset",
    "load_manifest", "shard_paths", "shard_total_tokens",
    "open_shard_memmap", "open_shard_memmaps",
    "iter_shards", "open_shard_dataset",
]