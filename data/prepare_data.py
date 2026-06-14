"""Data preparation: download datasets, tokenise, pack into flat binary tensors (sharded or single-file)."""
import argparse, json, os
from pathlib import Path
from typing import Iterator, List, Optional
import torch

_DATASETS = [
    ("HuggingFaceFW/fineweb-edu", "fineweb", 1.0),
    ("bigcode/the-stack-v2-train-smol-ids", "code", 0.3),
    ("lighteval/MATH", "math", 0.2),
]
_EXTRA_DATASETS = [
    ("HuggingFaceTB/smollm-corpus", "smollm", 1.0),
    ("nvidia/OpenMathReasoning", "openmath", 0.4),
    ("arcee-ai/cosmo-corpus", "cosmo", 0.3),
]
DATA_MIXES = {
    "deepseek-v3": [("fineweb", 1.0), ("smollm", 0.6), ("code", 0.3), ("cosmo", 0.2), ("math", 0.1), ("openmath", 0.1)],
    "code-82m": [("code", 1.5), ("fineweb", 1.0), ("math", 0.2)],
    "smoke": [("fineweb", 1.0)],
}


def download_and_prepare_dataset(output_dir: str = "data/datasets", max_rows: Optional[int] = None, include_extra: bool = False) -> str:
    from datasets import load_dataset
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    sources = list(_DATASETS) + (_EXTRA_DATASETS if include_extra else [])
    ok = 0
    for name, subdir, _w in sources:
        try:
            print(f"Downloading {name} ...")
            split = f"train[:{max_rows}]" if max_rows else "train"
            ds = load_dataset(name, split=split, trust_remote_code=True)
            out_path = Path(output_dir) / subdir
            out_path.mkdir(parents=True, exist_ok=True)
            ds.to_json(str(out_path / "data.jsonl"))
            print(f"  -> saved {len(ds):,} examples to {out_path}")
            ok += 1
        except Exception as exc:
            print(f"  [warn] {name}: {exc}")
    if ok == 0:
        print("[warn] No datasets downloaded; tokenisation will fail without --tokenizer.")
    return output_dir


def _iter_texts(data_dir: str) -> Iterator[str]:
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


def _shard_path(out_dir: str, shard_idx: int) -> str:
    return os.path.join(out_dir, f"shard_{shard_idx:05d}.bin")


def tokenize_and_pack(data_dir: str, output_path: str, tokenizer, max_tokens: int = 50_000_000, shard_size_tokens: int = 0) -> str:
    if tokenizer is None:
        raise ValueError("A HuggingFace tokenizer is required. Pass --tokenizer.")
    if shard_size_tokens > 0:
        return _tokenize_and_pack_sharded(data_dir, output_path, tokenizer, max_tokens, shard_size_tokens)
    return _tokenize_and_pack_single(data_dir, output_path, tokenizer, max_tokens)


def _tokenize_and_pack_single(data_dir: str, output_path: str, tokenizer, max_tokens: int) -> str:
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
        raise RuntimeError(f"No tokens collected from {data_dir}. Check that download_and_prepare_dataset succeeded.")
    tensor = torch.tensor(all_tokens[:max_tokens], dtype=torch.long)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, output_path)
    print(f"Saved {len(tensor):,} tokens to {output_path}  (texts={n_texts})")
    return output_path


def _tokenize_and_pack_sharded(data_dir: str, output_dir: str, tokenizer, max_tokens: int, shard_size_tokens: int) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    shard_idx = 0
    buf: List[int] = []
    n_texts = 0
    n_total = 0
    for text in _iter_texts(data_dir):
        toks = tokenizer.encode(text, add_special_tokens=False)
        buf.extend(toks)
        n_texts += 1
        while len(buf) >= shard_size_tokens:
            chunk = torch.tensor(buf[:shard_size_tokens], dtype=torch.long)
            torch.save(chunk, _shard_path(output_dir, shard_idx))
            shard_idx += 1
            n_total += shard_size_tokens
            print(f"  wrote shard {shard_idx-1:05d}  total={n_total:,} tokens  texts={n_texts:,}")
            buf = buf[shard_size_tokens:]
        if n_total + len(buf) >= max_tokens:
            break
    if buf:
        chunk = torch.tensor(buf, dtype=torch.long)
        torch.save(chunk, _shard_path(output_dir, shard_idx))
        n_total += len(buf)
        print(f"  wrote shard {shard_idx:05d}  total={n_total:,} tokens  texts={n_texts:,}  (final)")
    if n_total == 0:
        raise RuntimeError(f"No tokens collected from {data_dir}. Check that download_and_prepare_dataset succeeded.")
    manifest = {"data_dir": data_dir, "n_shards": shard_idx + (1 if buf else 0), "n_tokens": n_total, "n_texts": n_texts, "shard_size_tokens": shard_size_tokens}
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved {n_total:,} tokens across {manifest['n_shards']} shards -> {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Prepare pre-training data")
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--stage", type=str, choices=["pretrain"], default="pretrain")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=50_000_000)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--shard-size-tokens", type=int, default=0)
    parser.add_argument("--data-mix", type=str, default="deepseek-v3", choices=list(DATA_MIXES.keys()))
    parser.add_argument("--include-extra", action="store_true")
    args = parser.parse_args()
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        print(f"Loading tokenizer: {args.tokenizer}")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    ds_dir = download_and_prepare_dataset(os.path.join(args.output_dir, "datasets"), max_rows=args.max_rows, include_extra=args.include_extra)
    out_path = os.path.join(args.output_dir, f"pretrain_{args.data_mix}") if args.shard_size_tokens > 0 else os.path.join(args.output_dir, "pretrain_data.bin")
    tokenize_and_pack(ds_dir, out_path, tokenizer=tokenizer, max_tokens=args.max_tokens, shard_size_tokens=args.shard_size_tokens)
    if args.shard_size_tokens > 0:
        with open(os.path.join(out_path, "mix.json"), "w") as f:
            json.dump({"mix": args.data_mix, "components": DATA_MIXES[args.data_mix]}, f, indent=2)
    print("Data preparation complete.")


if __name__ == "__main__":
    main()
