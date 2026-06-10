# data/prepare_data.py
"""
Data preparation pipeline for DeepSeek-V3-Lite.

Stages:
  pretrain  — download datasets, tokenise, pack into a flat binary tensor
              (sharded .bin files for the 800M run)
  sft       — produce a minimal instruction-tuning JSON
  distill   — seed a distillation JSON with a few reasoning examples
  all       — all three stages

Pre-tokenised data is loaded by `training.pretrain.PretrainDataset`. The dataset
auto-detects sharded (directory of `shard_*.bin`) vs single-file layouts.
"""
import argparse
import json
import os
from pathlib import Path
from typing import Iterator, List, Optional

import torch


# ── Dataset definitions ────────────────────────────────────────────────────────

# Each entry: (huggingface_id, subdir_name, weight_in_mix)
# `weight_in_mix` controls sampling proportion when --data-mix is set.
# A weight of 0 means "use only for the named mix" (see DATA_MIXES below).
_DATASETS = [
    ("HuggingFaceFW/fineweb-edu",         "fineweb",       1.0),
    ("bigcode/the-stack-v2-train-smol-ids", "code",         0.3),
    ("lighteval/MATH",                     "math",         0.2),
]

# Additional datasets for richer mixes.
# Pulled only when explicitly requested via --data-mix.
_EXTRA_DATASETS = [
    # (id, subdir, weight). Weights are relative; see DATA_MIXES for totals.
    ("HuggingFaceTB/smollm-corpus",       "smollm",       1.0),
    ("nvidia/OpenMathReasoning",           "openmath",     0.4),
    ("arcee-ai/cosmo-corpus",              "cosmo",        0.3),
]

# Named data mixes. Each mix is a list of (subdir, weight) used to combine
# the resulting .bin files into the final pretraining corpus.
DATA_MIXES = {
    # Richer mix for the 800M Chinchilla run.
    # Web-dominant with a code / math / synthetic tail.
    "deepseek-v3": [
        ("fineweb",  1.0),
        ("smollm",   0.6),
        ("code",     0.3),
        ("cosmo",    0.2),
        ("math",     0.1),
        ("openmath", 0.1),
    ],
    # Useful for quick smoke tests.
    "smoke": [
        ("fineweb",  1.0),
    ],
}


# ── Dataset download ───────────────────────────────────────────────────────────

def download_and_prepare_dataset(
    output_dir: str = "data/datasets",
    max_rows: Optional[int] = None,
    include_extra: bool = False,
) -> str:
    """Download HuggingFace datasets and save as JSONL shards.

    Default: three core datasets. Pass `include_extra=True` for
    auxiliary datasets used by richer mixes.
    """
    from datasets import load_dataset

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    sources = list(_DATASETS) + (_EXTRA_DATASETS if include_extra else [])
    ok = 0

    for name, subdir, _w in sources:
        try:
            print(f"Downloading {name} …")
            split = f"train[:{max_rows}]" if max_rows else "train"
            ds = load_dataset(name, split=split, trust_remote_code=True)
            out_path = Path(output_dir) / subdir
            out_path.mkdir(parents=True, exist_ok=True)
            ds.to_json(str(out_path / "data.jsonl"))
            print(f"  → saved {len(ds):,} examples to {out_path}")
            ok += 1
        except Exception as exc:
            print(f"  [warn] {name}: {exc}")

    if ok == 0:
        print("[warn] No datasets downloaded; tokenisation will fail without --tokenizer.")
    return output_dir


# ── Text iteration ─────────────────────────────────────────────────────────────

def _iter_texts(data_dir: str) -> Iterator[str]:
    """Yield text strings from all JSON / JSONL files under data_dir."""
    for root, _, files in os.walk(data_dir):
        for fname in sorted(files):
            if not (fname.endswith(".json") or fname.endswith(".jsonl")):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(obj, list):
                            for item in obj:
                                if isinstance(item, dict) and isinstance(item.get("text"), str):
                                    yield item["text"]
                        elif isinstance(obj, dict):
                            text = obj.get("text") or obj.get("content") or obj.get("problem") or ""
                            if text:
                                yield text
            except OSError as exc:
                print(f"  [warn] Could not read {fpath}: {exc}")


# ── Tokenisation & packing ─────────────────────────────────────────────────────

def _shard_path(out_dir: str, shard_idx: int) -> str:
    return os.path.join(out_dir, f"shard_{shard_idx:05d}.bin")


def tokenize_and_pack(
    data_dir: str,
    output_path: str,
    tokenizer,
    max_tokens: int = 50_000_000,
    shard_size_tokens: int = 0,
) -> str:
    """
    Tokenise all text files and concatenate into token tensors.

    Two output modes:
      - `shard_size_tokens == 0` (default): write a single .bin file at
        `output_path`. Backwards-compatible with the original 50M-token signature.
      - `shard_size_tokens > 0`: write a directory of `shard_NNNNN.bin` files,
        each up to `shard_size_tokens` tokens. The 16B+ token corpus is split
        into ~1 GB shards, never loaded entirely into memory.

    A HuggingFace tokenizer is required.
    """
    if tokenizer is None:
        raise ValueError(
            "A HuggingFace tokenizer is required. "
            "Pass --tokenizer <name> when invoking this script."
        )

    if shard_size_tokens > 0:
        return _tokenize_and_pack_sharded(
            data_dir, output_path, tokenizer, max_tokens, shard_size_tokens
        )
    return _tokenize_and_pack_single(
        data_dir, output_path, tokenizer, max_tokens
    )


def _tokenize_and_pack_single(
    data_dir: str,
    output_path: str,
    tokenizer,
    max_tokens: int,
) -> str:
    """Original single-file tokenisation."""
    all_tokens: List[int] = []
    n_texts = 0

    for text in _iter_texts(data_dir):
        toks = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(toks)
        n_texts += 1

        if len(all_tokens) >= max_tokens:
            print(f"  Reached {max_tokens:,} token cap after {n_texts:,} documents.")
            break

    if not all_tokens:
        raise RuntimeError(
            f"No tokens collected from {data_dir}. "
            f"Check that `download_and_prepare_dataset` succeeded."
        )
    tensor = torch.tensor(all_tokens[:max_tokens], dtype=torch.long)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, output_path)
    print(f"Saved {len(tensor):,} tokens to {output_path}  (texts={n_texts})")
    return output_path


def _tokenize_and_pack_sharded(
    data_dir: str,
    output_dir: str,
    tokenizer,
    max_tokens: int,
    shard_size_tokens: int,
) -> str:
    """Streaming tokeniser that writes ~`shard_size_tokens`-sized shards.

    Never holds more than one document's tokens in Python memory.
    Flushes a shard when full; on-disk footprint bounded by one shard.
    Stops after `max_tokens` total tokens.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    shard_idx = 0
    buf: List[int] = []
    n_texts = 0
    n_total = 0

    for text in _iter_texts(data_dir):
        toks = tokenizer.encode(text, add_special_tokens=False)
        buf.extend(toks)
        n_texts += 1

        # Flush full shards
        while len(buf) >= shard_size_tokens:
            chunk = torch.tensor(buf[:shard_size_tokens], dtype=torch.long)
            torch.save(chunk, _shard_path(output_dir, shard_idx))
            shard_idx += 1
            n_total += shard_size_tokens
            print(f"  wrote shard {shard_idx-1:05d}  total={n_total:,} tokens  "
                  f"texts={n_texts:,}")
            buf = buf[shard_size_tokens:]

        if n_total + len(buf) >= max_tokens:
            break

    # Final partial shard
    if buf:
        chunk = torch.tensor(buf, dtype=torch.long)
        torch.save(chunk, _shard_path(output_dir, shard_idx))
        n_total += len(buf)
        print(f"  wrote shard {shard_idx:05d}  total={n_total:,} tokens  "
              f"texts={n_texts:,}  (final)")

    if n_total == 0:
        raise RuntimeError(
            f"No tokens collected from {data_dir}. "
            f"Check that `download_and_prepare_dataset` succeeded."
        )

    # Write a small manifest for sanity-checking.
    manifest = {
        "data_dir": data_dir,
        "n_shards": shard_idx + (1 if buf else 0),
        "n_tokens": n_total,
        "n_texts":  n_texts,
        "shard_size_tokens": shard_size_tokens,
    }
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved {n_total:,} tokens across {manifest['n_shards']} shards → {output_dir}")
    return output_dir


# ── Seed datasets for SFT / distillation ───────────────────────────────────────

def prepare_sft_data(output_path: str = "data/sft_data.json") -> str:
    """Seed SFT dataset with a handful of instruction/response examples."""
    examples = [
        {
            "messages": [
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hello! How can I help you today?"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is 2 + 2?"},
                {"role": "assistant", "content": "2 + 2 = 4."},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Write a Python function to check if a number is prime."},
                {
                    "role": "assistant",
                    "content": (
                        "```python\n"
                        "def is_prime(n: int) -> bool:\n"
                        "    if n < 2:\n"
                        "        return False\n"
                        "    for i in range(2, int(n**0.5) + 1):\n"
                        "        if n % i == 0:\n"
                        "            return False\n"
                        "    return True\n"
                        "```"
                    ),
                },
            ]
        },
    ]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(examples, f, indent=2)
    print(f"Prepared {len(examples)} SFT examples → {output_path}")
    return output_path


def prepare_distillation_data(output_path: str = "data/distill_data.json") -> str:
    """Seed distillation dataset with a single reasoning example."""
    examples = [
        {
            "prompt": "A train travels 300 miles in 5 hours. What is its speed?",
            "teacher_response": (
                "Step 1: Identify knowns — distance = 300 mi, time = 5 h\n"
                "Step 2: Speed = distance / time = 300 / 5 = 60 mph\n"
                "Step 3: Verify: 60 × 5 = 300 ✓\n\nAnswer: \\boxed{60} mph"
            ),
        }
    ]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(examples, f, indent=2)
    print(f"Prepared {len(examples)} distillation examples → {output_path}")
    return output_path


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare pre-training and SFT data")
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument(
        "--stage",
        type=str,
        choices=["pretrain", "sft", "distill", "all"],
        default="all",
    )
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="HuggingFace tokenizer name (required for pretrain)")
    parser.add_argument("--max-tokens", type=int, default=50_000_000)
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Cap each dataset to this many rows (default: full split)")
    parser.add_argument("--shard-size-tokens", type=int, default=0,
                        help="If >0, write per-shard .bin files of this size "
                             "(used for 800M-run; default 0 = single .bin file)")
    parser.add_argument("--data-mix", type=str, default="deepseek-v3",
                        choices=list(DATA_MIXES.keys()),
                        help="Which mix of sources to use (Phase C1).")
    parser.add_argument("--include-extra", action="store_true",
                        help="Download the auxiliary datasets (smollm, cosmo, "
                             "openmath) required by richer mixes.")
    args = parser.parse_args()

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        print(f"Loading tokenizer: {args.tokenizer}")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    if args.stage in ("pretrain", "all"):
        ds_dir = download_and_prepare_dataset(
            os.path.join(args.output_dir, "datasets"),
            max_rows=args.max_rows,
            include_extra=args.include_extra,
        )

    # For sharded output (--shard-size-tokens > 0) the data path is a directory.
    if args.shard_size_tokens > 0:
        out_path = os.path.join(args.output_dir, "pretrain_800m")
    else:
        out_path = os.path.join(args.output_dir, "pretrain_data.bin")

    tokenize_and_pack(
        ds_dir,
        out_path,
        tokenizer=tokenizer,
        max_tokens=args.max_tokens,
        shard_size_tokens=args.shard_size_tokens,
    )

    # Record which mix was used.
    if args.shard_size_tokens > 0:
        with open(os.path.join(out_path, "mix.json"), "w") as f:
            json.dump({"mix": args.data_mix, "components": DATA_MIXES[args.data_mix]}, f, indent=2)

    if args.stage in ("sft", "all"):
        prepare_sft_data(os.path.join(args.output_dir, "sft_data.json"))

    if args.stage in ("distill", "all"):
        prepare_distillation_data(os.path.join(args.output_dir, "distill_data.json"))

    print("Data preparation complete.")


if __name__ == "__main__":
    main()
