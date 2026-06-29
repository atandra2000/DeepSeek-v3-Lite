"""Atomic shard writer for packed token streams (shared pipeline)."""
from __future__ import annotations

import hashlib
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np

from shared_data.common import (
    DEFAULT_EOS_TOKEN_ID,
    DEFAULT_VOCAB_SIZE,
    atomic_write_bytes,
    human_count,
    log,
)


def select_token_dtype(vocab_size: int) -> np.dtype:
    """Pick uint8 / uint16 / uint32 / uint64 based on vocab size."""
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be > 0, got {vocab_size}")
    if vocab_size <= 0xFF:
        return np.dtype("uint8")
    if vocab_size <= 0xFFFF:
        return np.dtype("uint16")
    if vocab_size <= 0xFFFFFFFF:
        return np.dtype("uint32")
    raise ValueError(f"vocab_size {vocab_size:,} exceeds uint64 capacity")


def validate_tokens(
    tokens: np.ndarray,
    *,
    vocab_size: int,
    label: str = "",
    eos_token_id: int = 0,
) -> None:
    """Verify all tokens are within [0, vocab_size + 256). Raises on failure."""
    if tokens.size == 0:
        return
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be > 0, got {vocab_size}")
    mx = int(tokens.max())
    max_allowed = vocab_size + 256
    if mx >= max_allowed:
        raise ValueError(
            f"{label}: token id {mx} >= vocab_size + 256 ({max_allowed:,}). "
            "Tokeniser output is wildly out of range — refusing to write "
            "a corrupt shard."
        )


class TokenStream:
    """Append-only writer for a per-source token stream (raw uint32, EOS-separated)."""

    HEADER_FMT = "<II"  # version (uint32), eos_token_id (uint32)
    HEADER_SIZE = struct.calcsize(HEADER_FMT)

    def __init__(self, path: Path, eos_token_id: int, *, vocab_size: int):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.eos_token_id = int(eos_token_id)
        self.vocab_size = int(vocab_size)
        self._n_written = 0
        self._n_eos = 0
        self._fp = open(self.path, "wb")
        self._fp.write(struct.pack(self.HEADER_FMT, 1, self.eos_token_id))
        self._fp.flush()

    @property
    def n_written(self) -> int:
        return self._n_written

    @property
    def n_eos(self) -> int:
        return self._n_eos

    def write_doc(self, tokens: List[int]) -> None:
        """Write a single document's tokens, followed by an EOS."""
        if not tokens:
            return
        if tokens[-1] == self.eos_token_id:
            tokens = tokens[:-1]
        if not tokens:
            return
        arr = np.asarray(tokens, dtype=np.uint32)
        validate_tokens(arr, vocab_size=self.vocab_size, label=str(self.path))
        self._fp.write(arr.tobytes())
        self._fp.write(struct.pack("<I", self.eos_token_id))
        self._n_written += len(tokens) + 1
        self._n_eos += 1

    def close(self) -> None:
        if not self._fp.closed:
            self._fp.flush()
            self._fp.close()

    def __enter__(self) -> "TokenStream":
        return self

    def __exit__(self, *args) -> None:
        self.close()


def read_token_stream(
    path: Path,
    *,
    mmap: bool = True,
) -> Iterator[np.ndarray]:
    """Yield documents (np.ndarray of token IDs) from a per-source stream."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Token stream not found: {p}")
    with open(p, "rb") as f:
        header = f.read(TokenStream.HEADER_SIZE)
    if len(header) < TokenStream.HEADER_SIZE:
        raise ValueError(f"Truncated header in {p}")
    version, eos_id = struct.unpack(TokenStream.HEADER_FMT, header)
    if version != 1:
        raise ValueError(f"Unsupported stream version {version} in {p}")

    body_bytes = p.stat().st_size - TokenStream.HEADER_SIZE
    if body_bytes == 0:
        return
    if body_bytes % 4 != 0:
        raise ValueError(
            f"Token stream body size ({body_bytes}) is not a multiple of 4 bytes "
            f"in {p}"
        )

    if mmap:
        body = np.memmap(p, dtype=np.uint32, mode="r",
                         offset=TokenStream.HEADER_SIZE, shape=(body_bytes // 4,))
    else:
        body = np.frombuffer(
            open(p, "rb").read()[TokenStream.HEADER_SIZE:],
            dtype=np.uint32,
        )

    is_eos = (body == eos_id)
    eos_indices = np.where(is_eos)[0]
    if eos_indices.size == 0:
        raise ValueError(f"No EOS tokens found in {p} — stream is corrupt")

    start = 0
    for eos_pos in eos_indices:
        if eos_pos > start:
            yield body[start:eos_pos].copy()
        start = eos_pos + 1


@dataclass
class ShardMeta:
    """Per-shard verification metadata."""
    index: int
    path: str
    n_tokens: int
    sha256: str
    n_eos: int


class ShardWriter:
    """Write packed training shards from one or more token streams."""

    def __init__(
        self,
        output_dir: Path,
        shard_size_tokens: int,
        dtype: np.dtype,
        eos_token_id: int,
        vocab_size: int,
        *,
        cross_document_boundary_ok: bool = False,
    ):
        if shard_size_tokens <= 0:
            raise ValueError(f"shard_size_tokens must be > 0, got {shard_size_tokens}")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size_tokens = int(shard_size_tokens)
        self.dtype = np.dtype(dtype)
        self.eos_token_id = int(eos_token_id)
        self.vocab_size = int(vocab_size)
        self.cross_document_boundary_ok = cross_document_boundary_ok

        self._buf = np.empty(self.shard_size_tokens, dtype=self.dtype)
        self._buf_pos = 0
        self._shard_index = 0
        self._total_tokens = 0
        self._total_eos = 0
        self._shards: List[ShardMeta] = []
        self._finalized = False
        self._owns_handle = True

    def _flush(self) -> None:
        if self._buf_pos == 0:
            return

        shard_path = self.output_dir / f"shard_{self._shard_index:05d}.bin"
        tmp_path = self.output_dir / f"shard_{self._shard_index:05d}.bin.tmp"
        payload = self._buf[: self._buf_pos]
        raw_bytes = payload.tobytes()

        with open(tmp_path, "wb") as f:
            f.write(raw_bytes)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, shard_path)

        sha = hashlib.sha256(raw_bytes).hexdigest()
        n_eos = int(np.count_nonzero(payload == self.eos_token_id))

        self._shards.append(ShardMeta(
            index=self._shard_index,
            path=str(shard_path.relative_to(self.output_dir.parent)),
            n_tokens=int(self._buf_pos),
            sha256=sha,
            n_eos=n_eos,
        ))
        log(
            f"shard {self._shard_index:05d}  "
            f"{self._buf_pos:,} tokens  EOS={n_eos:,}  sha={sha[:12]}…"
        )
        self._shard_index += 1
        self._buf_pos = 0

    def add(self, doc: np.ndarray) -> None:
        if self._finalized:
            raise RuntimeError("Cannot add after finalize()")
        n = doc.size
        if n == 0:
            return
        validate_tokens(doc, vocab_size=self.vocab_size,
                        label="shard_writer.add",
                        eos_token_id=self.eos_token_id)
        if n > self.shard_size_tokens:
            if self.cross_document_boundary_ok:
                piece_size = self.shard_size_tokens - 1
                for start in range(0, n, piece_size):
                    piece = doc[start:start + piece_size]
                    if piece.size == 0:
                        continue
                    self.add(piece)
                return
            else:
                raise ValueError(
                    f"Document of {n:,} tokens exceeds shard_size_tokens "
                    f"({self.shard_size_tokens:,}) and "
                    f"cross_document_boundary_ok is False."
                )
        self._add_internal(doc)

    def _add_internal(self, doc: np.ndarray) -> None:
        n = doc.size
        needed = n + 1
        if needed > self.shard_size_tokens:
            if self.cross_document_boundary_ok:
                self.add(doc)
                return
            raise ValueError(
                f"Document + EOS ({needed} tokens) exceeds shard_size_tokens "
                f"({self.shard_size_tokens:,})."
            )
        if self._buf_pos + needed > self.shard_size_tokens:
            self._flush()
        self._buf[self._buf_pos: self._buf_pos + n] = doc
        self._buf[self._buf_pos + n] = self.eos_token_id
        self._buf_pos += needed
        self._total_tokens += needed
        self._total_eos += 1

    def finalize(self) -> List[ShardMeta]:
        if self._finalized:
            return self._shards
        self._flush()
        self._finalized = True
        log(
            f"pack complete: {self._shard_index} shards, "
            f"{self._total_tokens:,} tokens, {self._total_eos:,} EOS"
        )
        return self._shards

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, *args) -> None:
        self.finalize()


def verify_shard(
    shard_path: Path,
    *,
    expected_tokens: int,
    expected_dtype: np.dtype,
    vocab_size: int,
    eos_token_id: int,
) -> dict:
    """Re-read ``shard_path`` and verify count, dtype, vocab bounds, EOS count."""
    shard_path = Path(shard_path)
    if not shard_path.exists():
        raise FileNotFoundError(shard_path)

    itemsize = np.dtype(expected_dtype).itemsize
    file_size = shard_path.stat().st_size
    if file_size % itemsize != 0:
        raise ValueError(
            f"{shard_path}: file size {file_size} not aligned to {itemsize} bytes"
        )
    n_tokens = file_size // itemsize
    if n_tokens != expected_tokens:
        raise ValueError(
            f"{shard_path}: expected {expected_tokens} tokens, got {n_tokens}"
        )

    arr = np.memmap(shard_path, dtype=expected_dtype, mode="r")
    max_token = int(arr.max()) if n_tokens > 0 else 0
    if max_token >= vocab_size + 256:
        raise ValueError(
            f"{shard_path}: token id {max_token} >= vocab_size + 256 "
            f"({vocab_size + 256:,})"
        )
    n_eos = int(np.count_nonzero(arr == eos_token_id))

    sha = hashlib.sha256(shard_path.read_bytes()).hexdigest()
    return {
        "ok": True,
        "actual_tokens": n_tokens,
        "max_token": max_token,
        "n_eos": n_eos,
        "sha256": sha,
    }


__all__ = [
    "select_token_dtype", "validate_tokens",
    "TokenStream", "read_token_stream",
    "ShardMeta", "ShardWriter",
    "verify_shard",
]