"""Stage 4: pack per-source token streams into training shards."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np

from shared_data.common import (
    MANIFEST_PATH,
    SHARDS_ROOT,
    TOKENS_ROOT,
    ensure_dirs,
    get_logger,
    human_bytes,
    load_state,
    load_yaml,
    log,
    save_state,
)
from shared_data.manifest import Manifest, ShardInfo, SourceInfo, hash_config, hash_yaml
from shared_data.shard_writer import (
    ShardWriter,
    select_token_dtype,
    verify_shard,
    read_token_stream,
)


logger = get_logger("pack_shards")


def interleave_sources(
    source_ids: List[str],
    *,
    target_per_source: Dict[str, int],
) -> Iterator[tuple[str, np.ndarray]]:
    """Yield ``(source_id, doc)`` in round-robin order across sources."""
    streams: Dict[str, tuple[Iterator[np.ndarray], int]] = {}
    for sid in source_ids:
        path = TOKENS_ROOT / sid / "data.bin"
        if not path.exists():
            log(f"[pack] source {sid} has no token stream at {path}; skipping")
            continue
        streams[sid] = (read_token_stream(path), int(target_per_source.get(sid, 0)))

    if not streams:
        raise FileNotFoundError(f"No token streams found under {TOKENS_ROOT}")

    while streams:
        for sid in list(streams.keys()):
            it, remaining = streams[sid]
            try:
                doc = next(it)
            except StopIteration:
                del streams[sid]
                continue
            yield sid, doc


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 4: pack shards + write manifest")
    parser.add_argument("--mixture", required=True)
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--shards-dir", default=str(SHARDS_ROOT))
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args(argv)

    ensure_dirs()
    mix = load_yaml(Path(args.mixture))
    cfg = load_yaml(Path(args.data_config))

    total_tokens = int(mix["mixture"]["total_tokens"])
    sources = mix["mixture"]["sources"]
    pipeline_cfg = cfg["pipeline"]
    tok_cfg = pipeline_cfg["tokenizer"]
    sh_cfg = pipeline_cfg["sharding"]
    pack_cfg = pipeline_cfg["pack"]

    vocab_size = int(tok_cfg["vocab_size"])
    eos_token_id = int(tok_cfg["eos_token_id"])
    shard_size = int(sh_cfg["shard_size_tokens"])
    target_dtype = np.dtype(sh_cfg.get("dtype", "uint32"))
    actual_dtype = select_token_dtype(vocab_size)
    if target_dtype != actual_dtype:
        log(f"WARNING: requested dtype {target_dtype} doesn't match vocab-derived "
            f"{actual_dtype}; using {actual_dtype}", level="WARNING")
        target_dtype = actual_dtype

    log(f"packing → {args.shards_dir}  shard_size={shard_size:,}  "
        f"vocab={vocab_size:,}  eos={eos_token_id}  dtype={target_dtype}")

    target_per_source: Dict[str, int] = {
        s["id"]: int(total_tokens * s["weight"]) for s in sources
    }

    shards_dir = Path(args.shards_dir)
    shards_dir.mkdir(parents=True, exist_ok=True)
    existing_shards = sorted(shards_dir.glob("shard_*.bin"))
    start_shard_index = len(existing_shards)
    if start_shard_index > 0:
        log(f"resuming from shard index {start_shard_index} "
            f"({len(existing_shards)} existing shards, "
            f"{human_bytes(sum(s.stat().st_size for s in existing_shards))})")

    actual_per_source: Dict[str, int] = {s["id"]: 0 for s in sources}
    docs_per_source: Dict[str, int] = {s["id"]: 0 for s in sources}

    source_ids = [s["id"] for s in sources]
    writer = ShardWriter(
        output_dir=shards_dir,
        shard_size_tokens=shard_size,
        dtype=target_dtype,
        eos_token_id=eos_token_id,
        vocab_size=vocab_size,
        cross_document_boundary_ok=pack_cfg.get("cross_document_boundary_ok", False),
    )

    tokens_written = sum(s.stat().st_size // target_dtype.itemsize for s in existing_shards)
    log(f"starting token count: {tokens_written:,} / target {total_tokens:,}")

    try:
        for source_id, doc in interleave_sources(source_ids, target_per_source=target_per_source):
            writer.add(doc)
            actual_per_source[source_id] += doc.size + 1
            docs_per_source[source_id] += 1
            tokens_written = writer._total_tokens + sum(
                s.stat().st_size // target_dtype.itemsize for s in existing_shards
            )
            if tokens_written >= total_tokens:
                log(f"hit total target {total_tokens:,}; finalising")
                break
            if writer._shard_index > start_shard_index and writer._buf_pos == 0:
                state = {
                    "shard_index": writer._shard_index,
                    "tokens_written": tokens_written,
                    "actual_per_source": actual_per_source,
                    "docs_per_source": docs_per_source,
                }
                save_state("pack_shards", state)
                start_shard_index = writer._shard_index
                existing_shards = sorted(shards_dir.glob("shard_*.bin"))
    finally:
        shards = writer.finalize()

    log(f"wrote {len(shards)} new shards; total shards: "
        f"{len(shards) + len(existing_shards) - len(shards)}")

    if not args.no_verify:
        log("verifying shards...")
        for shard in shards:
            shard_path = shards_dir / Path(shard.path).name
            try:
                verify_shard(
                    shard_path,
                    expected_tokens=shard.n_tokens,
                    expected_dtype=target_dtype,
                    vocab_size=vocab_size,
                    eos_token_id=eos_token_id,
                )
            except Exception as e:
                logger.error("verify failed for %s: %s", shard_path, e)
                return 1

    manifest = Manifest(
        vocab_size=vocab_size,
        eos_token_id=eos_token_id,
        pad_token_id=int(tok_cfg["pad_token_id"]),
        tokenizer_name=tok_cfg.get("name", "llama3"),
        dtype=target_dtype.name,
        shard_size_tokens=shard_size,
        total_tokens=writer._total_tokens,
        shard_count=writer._shard_index,
        shards_dir=str(shards_dir.relative_to(shards_dir.parent.parent)),
    )
    for shard_path in sorted(shards_dir.glob("shard_*.bin")):
        idx = int(shard_path.stem.split("_")[-1])
        if any(s.index == idx for s in manifest.shards):
            continue
        manifest.shards.append(ShardInfo(
            index=idx,
            path=str(shard_path.relative_to(shards_dir.parent.parent)),
            n_tokens=shard_path.stat().st_size // target_dtype.itemsize,
            sha256="",
            n_eos=0,
        ))

    for s in sources:
        manifest.sources[s["id"]] = SourceInfo(
            target_tokens=target_per_source[s["id"]],
            actual_tokens=actual_per_source[s["id"]],
            n_docs=docs_per_source[s["id"]],
            n_dedup_dropped=0,
            shard_count=0,
        )
    manifest.config_hash = hash_config(cfg)
    manifest.mixture_hash = hash_yaml(Path(args.mixture))

    issues = manifest.validate(strict=True)
    if issues:
        for issue in issues:
            logger.error("manifest issue: %s", issue)
        return 2

    manifest.save(Path(args.manifest))
    log(f"manifest saved to {args.manifest}")
    log(f"summary: {manifest.shard_count} shards, "
        f"{manifest.total_tokens:,} tokens, "
        f"{sum(s.n_docs for s in manifest.sources.values()):,} docs")
    return 0


if __name__ == "__main__":
    sys.exit(main())