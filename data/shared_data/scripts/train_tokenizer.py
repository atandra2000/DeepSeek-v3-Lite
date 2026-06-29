"""Train a custom BPE tokenizer on the cleaned corpus."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator, List, Optional

from shared_data.common import (
    CLEAN_ROOT,
    get_logger,
    iter_jsonl,
    log,
)


logger = get_logger("train_tokenizer")


def _iter_clean_texts(max_docs_per_source: Optional[int] = None) -> Iterator[str]:
    """Yield text strings from every ``data/clean/<source>/data.jsonl``."""
    if not CLEAN_ROOT.exists():
        log(f"WARNING: {CLEAN_ROOT} does not exist; the clean stage must run first")
        return
    for source_dir in sorted(CLEAN_ROOT.iterdir()):
        clean_path = source_dir / "data.jsonl"
        if not clean_path.exists():
            continue
        log(f"reading {clean_path}")
        for i, rec in enumerate(iter_jsonl(clean_path)):
            text = rec.get("text", "")
            if text:
                yield text
            if max_docs_per_source is not None and i + 1 >= max_docs_per_source:
                break


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Train a custom BPE tokenizer")
    parser.add_argument("--vocab-size", type=int, default=64_000)
    parser.add_argument("--output", default="data/tokenizer/custom-bpe")
    parser.add_argument("--max-docs-per-source", type=int, default=None,
                        help="Cap each source at this many docs (for speed)")
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args(argv)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
    except ImportError:
        log("ERROR: `tokenizers` is required. pip install tokenizers",
            level="ERROR")
        return 1

    log(f"training BPE: vocab_size={args.vocab_size}, "
        f"min_frequency={args.min_frequency}")
    log(f"output: {out_dir}")

    tok = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=[
            "",              # EOS  → id 0
            "<|startoftext|>",  # BOS  → id 1
            "<|pad|>",         # PAD  → id 2
            "<|unk|>",         # UNK  → id 3
        ],
    )

    texts = _iter_clean_texts(max_docs_per_source=args.max_docs_per_source)
    tok.train_from_iterator(texts, trainer=trainer)

    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    tok.save(str(out_dir / "tokenizer.json"))
    log(f"tokenizer saved to {out_dir / 'tokenizer.json'}")

    vocab = tok.get_vocab()
    log(f"final vocab size: {len(vocab):,}")

    return 0


if __name__ == "__main__":
    sys.exit(main())