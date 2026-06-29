"""Stage 3: tokenize clean JSONL → per-source uint32 token streams."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from shared_data.common import (
    CLEAN_ROOT,
    ensure_dirs,
    get_logger,
    iter_jsonl,
    load_state,
    load_yaml,
    log,
    save_state,
)
from shared_data.shard_writer import TokenStream, validate_tokens


logger = get_logger("tokenize")


class _TiktokenFallback:
    """Tokenizer wrapper exposing ``encode(text)``; used when HF tokenizer is unavailable."""

    def __init__(self, eos_token_id: int, vocab_size: int):
        import tiktoken
        self._enc = tiktoken.get_encoding("cl100k_base")
        self.eos_token_id = eos_token_id
        self.vocab_size = vocab_size
        self._warned = False

    def encode(self, text: str) -> List[int]:
        if not self._warned:
            log(
                f"WARNING: using tiktoken cl100k_base fallback (vocab="
                f"{self.vocab_size_actual:,}). Token IDs will not match "
                f"LLaMA-3 BPE; downstream EOS={self.eos_token_id} may be "
                f"out-of-vocab. Train a real tokenizer with "
                f"`python -m shared_data.scripts.train_tokenizer`.",
            )
            self._warned = True
        return self._enc.encode(text)

    @property
    def vocab_size_actual(self) -> int:
        return self._enc.max_token_value + 1


def load_tokenizer(
    name: str = "llama3",
    path: Optional[str] = None,
    eos_token_id: int = 128_009,
    vocab_size: int = 128_000,
):
    """Load an HF tokenizer; fall back to tiktoken if not available."""
    if path is not None and Path(path).exists():
        try:
            from transformers import AutoTokenizer
            log(f"loading HF tokenizer from {path}")
            tok = AutoTokenizer.from_pretrained(path)
            class _HFWrapper:
                def __init__(self, hf):
                    self._hf = hf
                def encode(self, text: str) -> List[int]:
                    return self._hf.encode(text, add_special_tokens=False)
                @property
                def vocab_size_actual(self) -> int:
                    return self._hf.vocab_size
            return _HFWrapper(tok)
        except ImportError:
            log("transformers not installed; using tiktoken fallback",
                level="WARNING")
        except Exception as e:
            log(f"failed to load HF tokenizer from {path}: {e}; using tiktoken fallback",
                level="WARNING")
    return _TiktokenFallback(eos_token_id=eos_token_id, vocab_size=vocab_size)


def tokenize_source(
    spec: dict,
    *,
    target_tokens: int,
    tokenizer,
    eos_token_id: int,
    vocab_size: int,
) -> dict:
    """Tokenize one source into a per-source binary stream."""
    source_id = spec["id"]
    clean_path = CLEAN_ROOT / source_id / "data.jsonl"
    from shared_data.common import TOKENS_ROOT
    token_path = TOKENS_ROOT / source_id / "data.bin"

    if not clean_path.exists():
        log(f"[{source_id}] no clean data at {clean_path}; skipping")
        return {"n_tokens": 0, "n_docs": 0, "n_eos": 0}

    if not _should_retokenize(source_id):
        from shared_data.shard_writer import TokenStream as _TS
        n_tokens = (token_path.stat().st_size - _TS.HEADER_SIZE) // 4
        n_docs = sum(1 for _ in iter_jsonl(clean_path))
        log(f"[{source_id}] reusing existing token stream: {n_tokens:,} tokens")
        return {"n_tokens": n_tokens, "n_docs": n_docs, "n_eos": n_docs}

    state = load_state(f"tokenize_{source_id}")
    n_processed = int(state.get("n_processed", 0))
    n_tokens = int(state.get("n_tokens", 0))
    n_docs = int(state.get("n_docs", 0))

    log(f"[{source_id}] tokenising → {token_path} (target {target_tokens:,} tokens)")

    with TokenStream(token_path, eos_token_id=eos_token_id, vocab_size=vocab_size) as stream:
        for i, rec in enumerate(iter_jsonl(clean_path)):
            if i < n_processed:
                continue
            text = rec.get("text", "")
            if not text:
                continue
            ids = tokenizer.encode(text)
            if not ids:
                continue
            import numpy as np
            validate_tokens(
                np.asarray(ids, dtype="uint32"),
                vocab_size=vocab_size,
                label=f"tokenize[{source_id}]",
            )
            stream.write_doc(ids)
            n_tokens += len(ids) + 1
            n_docs += 1
            n_processed += 1

            if n_tokens >= target_tokens:
                log(f"[{source_id}] hit token target after {n_docs:,} docs")
                break
            if n_processed % 50_000 == 0:
                state.update({"n_processed": n_processed,
                               "n_tokens": n_tokens, "n_docs": n_docs})
                save_state(f"tokenize_{source_id}", state)

    state.update({"n_processed": n_processed, "n_tokens": n_tokens,
                  "n_docs": n_docs, "complete": True})
    save_state(f"tokenize_{source_id}", state)
    log(f"[{source_id}] tokenize complete: {n_tokens:,} tokens, {n_docs:,} docs")
    return {"n_tokens": n_tokens, "n_docs": n_docs, "n_eos": n_docs}


def _should_retokenize(source_id: str) -> bool:
    s = load_state(f"tokenize_{source_id}")
    return not bool(s.get("complete", False))


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 3: tokenise")
    parser.add_argument("--mixture", required=True)
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--source", default=None)
    args = parser.parse_args(argv)

    ensure_dirs()
    mix = load_yaml(Path(args.mixture))
    cfg = load_yaml(Path(args.data_config))
    total_tokens = mix["mixture"]["total_tokens"]
    pipeline_cfg = cfg["pipeline"]
    tok_cfg = pipeline_cfg["tokenizer"]

    tokenizer = load_tokenizer(
        name=tok_cfg.get("name", "llama3"),
        path=tok_cfg.get("path"),
        eos_token_id=int(tok_cfg["eos_token_id"]),
        vocab_size=int(tok_cfg["vocab_size"]),
    )

    for spec in mix["mixture"]["sources"]:
        if args.source and spec["id"] != args.source:
            continue
        target_tokens = int(total_tokens * spec["weight"])
        try:
            tokenize_source(
                spec,
                target_tokens=target_tokens,
                tokenizer=tokenizer,
                eos_token_id=int(tok_cfg["eos_token_id"]),
                vocab_size=int(tok_cfg["vocab_size"]),
            )
        except Exception as e:
            logger.error("[%s] FAILED: %s: %s", spec["id"], type(e).__name__, e)
            continue

    log("tokenize: all requested sources done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())