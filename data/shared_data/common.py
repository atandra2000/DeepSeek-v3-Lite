"""Shared constants and IO helpers for the universal LLM data pipeline."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Optional


# .../LLM/shared_data/common.py → .../LLM/
PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # .../LLM/
PROJECT_ROOT = PACKAGE_ROOT


def _resolve_data_root() -> Path:
    """Resolve the project's data root (LLM_DATA_ROOT > LLM_PROJECT_ROOT/data > $PWD/data)."""
    env = os.environ
    if env.get("LLM_DATA_ROOT"):
        return Path(env["LLM_DATA_ROOT"]).resolve()
    if env.get("LLM_PROJECT_ROOT"):
        return Path(env["LLM_PROJECT_ROOT"]).resolve() / "data"
    return Path.cwd() / "data"


DATA_ROOT = _resolve_data_root()
RAW_ROOT = DATA_ROOT / "raw"
CLEAN_ROOT = DATA_ROOT / "clean"
TOKENS_ROOT = DATA_ROOT / "tokens"
SHARDS_ROOT = DATA_ROOT / "shards"
STATE_ROOT = DATA_ROOT / "state"
CONFIG_ROOT = DATA_ROOT / "config"
MANIFEST_PATH = DATA_ROOT / "manifest.json"


def set_data_root(path: Path) -> None:
    """Programmatically set the data root and re-derive all sub-roots."""
    global DATA_ROOT, RAW_ROOT, CLEAN_ROOT, TOKENS_ROOT, SHARDS_ROOT
    global STATE_ROOT, CONFIG_ROOT, MANIFEST_PATH
    DATA_ROOT = Path(path).resolve()
    RAW_ROOT = DATA_ROOT / "raw"
    CLEAN_ROOT = DATA_ROOT / "clean"
    TOKENS_ROOT = DATA_ROOT / "tokens"
    SHARDS_ROOT = DATA_ROOT / "shards"
    STATE_ROOT = DATA_ROOT / "state"
    CONFIG_ROOT = DATA_ROOT / "config"
    MANIFEST_PATH = DATA_ROOT / "manifest.json"


# Vocabulary / EOS conventions (LLaMA-3 BPE — universal default).
DEFAULT_VOCAB_SIZE = 128_000       # LLaMA-3 BPE
DEFAULT_EOS_TOKEN_ID = 128_009     # LLaMA-3  <|eot_id|>
DEFAULT_PAD_TOKEN_ID = 128_002     # reserved in LLaMA-3


_logger = logging.getLogger("shared_data")
if not _logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("[data] %(message)s"))
    _logger.addHandler(handler)
    _logger.setLevel(os.environ.get("LLM_DATA_LOG", "INFO"))
    _logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a child logger of the shared ``shared_data`` logger."""
    return _logger.getChild(name)


def log(msg: str, *, level: int = logging.INFO) -> None:
    """Write a timestamped message that won't interleave with tqdm bars."""
    ts = time.strftime("%H:%M:%S")
    _logger.log(level, f"{ts} {msg}")


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomically write ``data`` to ``path`` via a sibling temp file + rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=path.suffix + ".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_json(path: Path, obj: Any, *, indent: int = 2) -> None:
    """Atomically serialise ``obj`` as JSON (numpy/torch-aware encoder)."""
    text = json.dumps(obj, indent=indent, default=_json_default)
    atomic_write_bytes(path, text.encode("utf-8"))


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_default(obj: Any) -> Any:
    """JSON encoder for numpy/torch scalars."""
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except (ValueError, TypeError):
            pass
    if hasattr(obj, "tolist") and callable(obj.tolist):
        try:
            return obj.tolist()
        except Exception:
            pass
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def load_state(stage: str) -> dict:
    """Load the per-stage state file, returning an empty dict if absent."""
    p = STATE_ROOT / f"{stage}.json"
    if not p.exists():
        return {}
    try:
        return read_json(p)
    except (json.JSONDecodeError, OSError) as e:
        log(f"corrupt state at {p}: {e}; starting fresh", level=logging.WARNING)
        return {}


def save_state(stage: str, state: dict) -> None:
    """Atomically write the per-stage state (enables resume after a crash)."""
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(STATE_ROOT / f"{stage}.json", state)


def clear_state(stage: str) -> None:
    """Wipe the state for ``stage``."""
    p = STATE_ROOT / f"{stage}.json"
    if p.exists():
        p.unlink()


def sha256_bytes(data: bytes) -> str:
    """Hex-encoded SHA-256 of a byte string."""
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str, *, encoding: str = "utf-8") -> str:
    """SHA-256 of a UTF-8 string with whitespace normalisation."""
    normalised = " ".join(text.split())
    return hashlib.sha256(normalised.encode(encoding)).hexdigest()


def hash_to_bucket(sha: str, n_buckets: int) -> int:
    """Map a hex SHA-256 to one of ``n_buckets`` shards (first 8 hex chars mod n)."""
    return int(sha[:8], 16) % n_buckets


def seed_everything(seed: int) -> None:
    """Seed Python + NumPy + PyTorch RNGs for reproducible data preparation."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_yaml(path: Path) -> dict:
    """Load a YAML config (PyYAML is required)."""
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs() -> None:
    """Create the full directory layout. Idempotent."""
    for p in (RAW_ROOT, CLEAN_ROOT, TOKENS_ROOT, SHARDS_ROOT, STATE_ROOT, CONFIG_ROOT):
        p.mkdir(parents=True, exist_ok=True)


def human_bytes(n: int) -> str:
    """Pretty-print a byte count (e.g. 16106127360 → '15.0 GiB')."""
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PiB"


def human_count(n) -> str:
    """Pretty-print an integer count with thousands separators."""
    if isinstance(n, int):
        return f"{n:,}"
    return f"{int(n):,}"


def iter_jsonl(path: Path) -> Iterable[dict]:
    """Yield dicts from a JSONL file. Skips malformed lines with a warning."""
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError as e:
                log(f"skip malformed line {line_no} in {path.name}: {e}",
                    level=logging.WARNING)
                continue


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    """Write JSONL records to ``path``. Returns the count written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


__all__ = [
    "DATA_ROOT", "RAW_ROOT", "CLEAN_ROOT", "TOKENS_ROOT", "SHARDS_ROOT",
    "STATE_ROOT", "CONFIG_ROOT", "MANIFEST_PATH", "PACKAGE_ROOT", "PROJECT_ROOT",
    "set_data_root",
    "DEFAULT_VOCAB_SIZE", "DEFAULT_EOS_TOKEN_ID", "DEFAULT_PAD_TOKEN_ID",
    "get_logger", "log",
    "atomic_write_bytes", "atomic_write_json", "read_json",
    "load_state", "save_state", "clear_state",
    "sha256_bytes", "sha256_text", "hash_to_bucket",
    "load_yaml", "seed_everything", "ensure_dirs", "human_bytes", "human_count",
    "iter_jsonl", "write_jsonl",
]