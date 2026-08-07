#!/usr/bin/env python
"""Build a small pretrain shard for the GTX 1650 E2E test.

Self-contained replacement for the `shared_data` pipeline the macOS dev box uses.
Streams a text source (HF fineweb-edu/wikitext, or a deterministic synthetic
fallback), tokenizes with the requested tokenizer, and writes a single
`shard_0000.bin` of uint32 tokens to `data/pretrain_chinchilla/`. Output is
byte-compatible with `PretrainDataset`.
"""
import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _ensure_data_dir() -> Path:
    out_dir = _project_root() / "data" / "pretrain_chinchilla"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _load_text_source(name: str, num_docs: int):
    """Yield text from a small HF dataset, falling back to synthetic on failure."""
    try:
        from datasets import load_dataset
        if name == "fineweb-edu":
            ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                              split="train", streaming=True)
        elif name == "wikitext":
            ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
        else:
            raise ValueError(f"Unknown source {name!r}")
        return (item["text"] for item in ds)
    except Exception as exc:
        print(f"[warn] Could not load {name} ({type(exc).__name__}: {str(exc)[:120]})")
        print("[warn] Falling back to a synthetic text source.")
        return _synthetic_source(num_docs)


def _synthetic_source(num_docs: int):
    """Deterministic synthetic text — reproducible byte-for-byte."""
    import hashlib
    # Topic-mixed paragraphs: not linguistically realistic, but give a stable
    # token distribution across runs.
    seeds = [
        "the quick brown fox jumps over the lazy dog. ",
        "deep learning has revolutionized natural language processing in the last decade. ",
        "transformer models use self-attention to capture long-range dependencies in sequences. ",
        "mixture of experts allows sparse computation by routing tokens to specialized sub-networks. ",
        "multi-head latent attention reduces the kv cache by absorbing key projections into the query. ",
        "aux-loss-free routing uses bias updates instead of auxiliary losses to balance expert load. ",
        "the chinchilla scaling law suggests twenty tokens per parameter for compute-optimal training. ",
        "speculative decoding accepts draft tokens when the main model agrees with their probability. ",
    ]
    def gen():
        """Yield deterministic topic-mixed documents."""
        for i in range(num_docs):
            h = hashlib.sha256(f"doc-{i}".encode()).digest()
            n_paragraphs = 4 + (h[0] % 5)
            chunks = []
            for j in range(n_paragraphs):
                idx = h[(j + 1) % len(h)] % len(seeds)
                chunks.append(seeds[idx] * (1 + (h[(j + 2) % len(h)] % 3)))
            yield "".join(chunks)
    return gen()


def build_shard(
    out_dir: Path,
    tokenizer_name: str = "gpt2",
    source: str = "fineweb-edu",
    target_tokens: int = 200_000,
    min_chars: int = 200,
) -> Path:
    """Tokenize until we have >= target_tokens valid tokens; write a single shard."""
    out_path = out_dir / "shard_0000.bin"
    print(f"[data] loading tokenizer {tokenizer_name!r}...")
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    if tok.eos_token_id is None:
        # gpt2 has eos; safety net for tokenizers that don't.
        tok.eos_token_id = tok.vocab_size - 1
    print(f"[data] tokenizer vocab={tok.vocab_size} eos={tok.eos_token_id} pad={tok.pad_token_id}")

    print(f"[data] streaming text from {source!r}...")
    text_iter = _load_text_source(source, num_docs=10_000)

    all_ids: list[int] = []
    docs_seen = 0
    docs_kept = 0
    for text in text_iter:
        docs_seen += 1
        if len(text) < min_chars:
            continue
        ids = tok.encode(text, add_special_tokens=False)
        if not ids:
            continue
        all_ids.extend(ids)
        all_ids.append(tok.eos_token_id)
        docs_kept += 1
        if len(all_ids) >= target_tokens:
            break
        if docs_seen % 500 == 0:
            print(f"[data]   scanned {docs_seen} docs, kept {docs_kept}, "
                  f"tokens so far: {len(all_ids):,}/{target_tokens:,}")

    if not all_ids:
        raise RuntimeError("No tokens produced — text source returned nothing")

    # Truncate to exact target_tokens for predictable file size.
    tensor = torch.tensor(all_ids[:target_tokens], dtype=torch.uint32)
    print(f"[data] writing {tensor.numel():,} tokens -> {out_path}")
    torch.save(tensor, out_path)
    print(f"[data] done. docs_seen={docs_seen} docs_kept={docs_kept} "
          f"tokens={tensor.numel():,}")
    return out_path


def main() -> int:
    """Parse shard options and build the requested token file."""
    parser = argparse.ArgumentParser(description="Build a small pretrain shard")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--source", default="fineweb-edu",
                        choices=["fineweb-edu", "wikitext", "synthetic"])
    parser.add_argument("--target-tokens", type=int, default=200_000)
    parser.add_argument("--out-dir", default=None,
                        help="Override output dir (default: data/pretrain_chinchilla)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else _ensure_data_dir()
    build_shard(
        out_dir=out_dir,
        tokenizer_name=args.tokenizer,
        source=args.source,
        target_tokens=args.target_tokens,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
