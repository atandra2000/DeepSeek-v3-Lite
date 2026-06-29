"""Manifest I/O for the shared LLM data pipeline."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from shared_data.common import (
    DEFAULT_EOS_TOKEN_ID,
    DEFAULT_VOCAB_SIZE,
    atomic_write_json,
    read_json,
)


MANIFEST_VERSION = "1.0.0"


@dataclass
class ShardInfo:
    index: int
    path: str
    n_tokens: int
    sha256: str
    n_eos: int


@dataclass
class SourceInfo:
    target_tokens: int
    actual_tokens: int
    n_docs: int
    n_dedup_dropped: int
    shard_count: int = 0


@dataclass
class Manifest:
    version: str = MANIFEST_VERSION
    created_utc: str = ""
    vocab_size: int = DEFAULT_VOCAB_SIZE
    eos_token_id: int = DEFAULT_EOS_TOKEN_ID
    pad_token_id: int = 128_002
    tokenizer_name: str = "llama3"
    dtype: str = "uint32"
    shard_size_tokens: int = 50_000_000
    total_tokens: int = 0
    shard_count: int = 0
    shards_dir: str = "data/shards"
    shards: List[ShardInfo] = field(default_factory=list)
    sources: Dict[str, SourceInfo] = field(default_factory=dict)
    config_hash: str = ""
    mixture_hash: str = ""

    def to_dict(self) -> dict:
        d = {
            "version": self.version,
            "created_utc": self.created_utc,
            "vocab_size": self.vocab_size,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "tokenizer_name": self.tokenizer_name,
            "dtype": self.dtype,
            "shard_size_tokens": self.shard_size_tokens,
            "total_tokens": self.total_tokens,
            "shard_count": self.shard_count,
            "shards_dir": self.shards_dir,
            "shards": [asdict(s) for s in self.shards],
            "sources": {k: asdict(v) for k, v in self.sources.items()},
            "config_hash": self.config_hash,
            "mixture_hash": self.mixture_hash,
        }
        return d

    def save(self, path: Path) -> None:
        if not self.created_utc:
            self.created_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        atomic_write_json(Path(path), self.to_dict())

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        d = read_json(Path(path))
        return cls(
            version=d.get("version", MANIFEST_VERSION),
            created_utc=d.get("created_utc", ""),
            vocab_size=d.get("vocab_size", DEFAULT_VOCAB_SIZE),
            eos_token_id=d.get("eos_token_id", DEFAULT_EOS_TOKEN_ID),
            pad_token_id=d.get("pad_token_id", 128_002),
            tokenizer_name=d.get("tokenizer_name", "llama3"),
            dtype=d.get("dtype", "uint32"),
            shard_size_tokens=d.get("shard_size_tokens", 50_000_000),
            total_tokens=d.get("total_tokens", 0),
            shard_count=d.get("shard_count", 0),
            shards_dir=d.get("shards_dir", "data/shards"),
            shards=[ShardInfo(**s) for s in d.get("shards", [])],
            sources={k: SourceInfo(**v) for k, v in d.get("sources", {}).items()},
            config_hash=d.get("config_hash", ""),
            mixture_hash=d.get("mixture_hash", ""),
        )

    @classmethod
    def exists(cls, path: Path) -> bool:
        return Path(path).exists()

    def validate(self, *, strict: bool = True) -> List[str]:
        """Sanity-check the manifest. Returns a list of issue strings."""
        issues: List[str] = []
        if self.vocab_size <= 0:
            issues.append(f"vocab_size must be positive, got {self.vocab_size}")
        if not (0 <= self.eos_token_id < self.vocab_size + 256):
            issues.append(
                f"eos_token_id ({self.eos_token_id}) must be in [0, "
                f"{self.vocab_size + 256}) (vocab + reserved special tokens)"
            )
        if self.shard_count <= 0:
            issues.append(f"shard_count must be positive, got {self.shard_count}")
        if self.total_tokens <= 0:
            issues.append(f"total_tokens must be positive, got {self.total_tokens}")
        if len(self.shards) != self.shard_count:
            issues.append(
                f"len(shards) ({len(self.shards)}) != shard_count ({self.shard_count})"
            )
        if self.sources:
            src_sum = sum(s.actual_tokens for s in self.sources.values())
            if abs(src_sum - self.total_tokens) > max(1, self.total_tokens // 1000):
                issues.append(
                    f"Σ source.actual_tokens ({src_sum}) != total_tokens ({self.total_tokens})"
                )
        return issues


def hash_yaml(path: Path) -> str:
    """SHA-256 of a YAML file's raw text."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hash_config(cfg: dict) -> str:
    """SHA-256 of a config dict, with stable key ordering."""
    text = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = [
    "MANIFEST_VERSION", "ShardInfo", "SourceInfo", "Manifest",
    "hash_yaml", "hash_config",
]