"""Stage 2: clean raw JSONL → quality-filtered + dedup'd clean JSONL."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from shared_data.common import (
    CLEAN_ROOT,
    RAW_ROOT,
    ensure_dirs,
    get_logger,
    human_count,
    iter_jsonl,
    load_state,
    load_yaml,
    log,
    save_state,
)
from shared_data.dedup import Deduper
from shared_data.quality_filter import FilterStats, QualityFilter


logger = get_logger("clean")


def clean_source(
    spec: dict,
    *,
    quality_cfg: dict,
    dedup_cfg: dict,
) -> dict:
    """Filter + dedup one source. Returns a stats dict for the manifest."""
    source_id = spec["id"]
    raw_path = RAW_ROOT / source_id / "data.jsonl"
    clean_path = CLEAN_ROOT / source_id / "data.jsonl"

    if not raw_path.exists():
        log(f"[{source_id}] no raw data at {raw_path}; skipping")
        return {
            "n_seen": 0, "n_kept": 0, "n_dropped_filter": 0,
            "n_dropped_dedup": 0,
        }

    state = load_state(f"clean_{source_id}")
    n_processed = int(state.get("n_processed", 0))
    n_kept = int(state.get("n_kept", 0))
    n_dropped_filter = int(state.get("n_dropped_filter", 0))
    n_dropped_dedup = int(state.get("n_dropped_dedup", 0))

    qf = QualityFilter(
        min_chars=spec.get("min_chars", 200),
        max_chars=spec.get("max_chars", 200_000),
        lang=spec.get("lang"),
        drop_empty=quality_cfg.get("drop_empty", True),
        min_unique_chars_ratio=quality_cfg.get("min_unique_chars_ratio", 0.05),
        max_digit_ratio=quality_cfg.get("max_digit_ratio", 0.50)
            if spec.get("lang") != "python" else None,
        max_punct_ratio=quality_cfg.get("max_punct_ratio", 0.50),
        max_whitespace_ratio=quality_cfg.get("max_whitespace_ratio", 0.50),
    )
    stats = FilterStats()
    stats.n_seen = n_processed
    stats.n_kept = n_kept
    stats.n_dropped = n_dropped_filter + n_dropped_dedup
    stats.reasons = __import__("collections").Counter(state.get("reasons", {}))

    dedup = None
    if dedup_cfg.get("enabled", True):
        dedup = Deduper(
            source_id=source_id,
            n_buckets=int(dedup_cfg.get("n_hash_buckets", 256)),
            bloom_capacity_per_bucket=int(dedup_cfg.get("bloom_capacity_per_bucket", 200_000)),
            bloom_error_rate=float(dedup_cfg.get("bloom_error_rate", 0.001)),
        )

    clean_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(clean_path, "a", encoding="utf-8")

    log(f"[{source_id}] cleaning + dedup → {clean_path}")

    skipped = 0
    for rec in iter_jsonl(raw_path):
        if skipped < n_processed:
            skipped += 1
            continue
        text = rec.get("text", "")
        kept = qf.apply(text)
        if kept is None:
            stats.n_dropped_filter += 1
            stats.reasons["quality"] += 1
        else:
            from shared_data.common import sha256_text, hash_to_bucket
            sha = sha256_text(kept)
            bucket = hash_to_bucket(sha, dedup.n_buckets if dedup else 1)
            if not hasattr(clean_source, f"_seen_{source_id}"):
                setattr(clean_source, f"_seen_{source_id}", set())
            seen_set = getattr(clean_source, f"_seen_{source_id}")
            if sha in seen_set:
                stats.n_dropped_dedup += 1
                stats.reasons["duplicate"] += 1
            else:
                seen_set.add(sha)
                out_f.write(json.dumps({"id": sha[:16], "text": kept},
                                       ensure_ascii=False) + "\n")
                stats.n_kept += 1
        stats.n_seen += 1
        n_processed = stats.n_seen
        n_kept = stats.n_kept
        n_dropped_filter = stats.n_dropped_filter
        n_dropped_dedup = stats.n_dropped_dedup

        if n_processed % 100_000 == 0:
            log(f"[{source_id}] clean: {n_processed:,} docs, "
                f"{n_kept:,} kept, {n_dropped_filter + n_dropped_dedup:,} dropped")
            state.update({
                "n_processed": n_processed,
                "n_kept": n_kept,
                "n_dropped_filter": n_dropped_filter,
                "n_dropped_dedup": n_dropped_dedup,
                "reasons": dict(stats.reasons),
            })
            save_state(f"clean_{source_id}", state)
            out_f.flush()

    out_f.flush()
    out_f.close()
    state.update({
        "n_processed": n_processed,
        "n_kept": n_kept,
        "n_dropped_filter": n_dropped_filter,
        "n_dropped_dedup": n_dropped_dedup,
        "reasons": dict(stats.reasons),
    })
    save_state(f"clean_{source_id}", state)
    log(f"[{source_id}] clean complete:\n{stats.summary()}")
    return {
        "n_seen": stats.n_seen,
        "n_kept": stats.n_kept,
        "n_dropped_filter": stats.n_dropped_filter,
        "n_dropped_dedup": stats.n_dropped_dedup,
    }


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 2: clean + dedup")
    parser.add_argument("--mixture", required=True)
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--source", default=None)
    args = parser.parse_args(argv)

    ensure_dirs()
    mix = load_yaml(Path(args.mixture))
    cfg = load_yaml(Path(args.data_config))
    quality_cfg = cfg["pipeline"]["quality"]
    dedup_cfg = cfg["pipeline"]["dedup"]

    for spec in mix["mixture"]["sources"]:
        if args.source and spec["id"] != args.source:
            continue
        try:
            clean_source(spec, quality_cfg=quality_cfg, dedup_cfg=dedup_cfg)
        except Exception as e:
            logger.error("[%s] FAILED: %s: %s", spec["id"], type(e).__name__, e)
            continue

    log("clean: all requested sources done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())