"""Stage 1: download raw text from HuggingFace datasets → JSONL."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator, Optional

from shared_data.common import (
    RAW_ROOT,
    ensure_dirs,
    get_logger,
    human_count,
    load_state,
    load_yaml,
    log,
    save_state,
)


logger = get_logger("download")


def _build_text(row: dict, spec: dict) -> str:
    """Combine the primary text_field with any extra fields per the spec."""
    text = row.get(spec["text_field"], "") or ""
    extra_field = spec.get("extra_text_field")
    if extra_field:
        extra = row.get(extra_field, "") or ""
        sep = spec.get("extra_separator", "\n\n")
        if extra:
            text = f"{text}{sep}{extra}" if text else extra
    return text


def download_source(
    spec: dict,
    *,
    target_tokens: int,
    state: Optional[dict] = None,
    streaming: bool = True,
    bytes_per_doc_estimate: int = 1500,
    chars_per_token: float = 4.0,
) -> dict:
    """Download ``spec`` (a single source from mixture.yaml) to JSONL. Resumable."""
    from datasets import load_dataset  # heavy import

    state = dict(state or {})
    n_processed = int(state.get("n_processed", 0))
    n_chars = int(state.get("n_chars", 0))

    source_id = spec["id"]
    out_dir = RAW_ROOT / source_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "data.jsonl"
    out_f = open(out_path, "a", encoding="utf-8")

    ds_name = spec["dataset"]
    ds_config = spec.get("config")
    split = spec.get("split", "train")
    log(f"[{source_id}] streaming {ds_name}/{ds_config or '-'} split={split}")

    try:
        ds = load_dataset(
            ds_name,
            ds_config,
            split=split,
            streaming=streaming,
            trust_remote_code=True,
        )

        target_chars = int(target_tokens * chars_per_token)
        log_every = 50_000

        for i, row in enumerate(ds):
            if i < n_processed:
                continue
            text = _build_text(row, spec)
            if not text:
                continue
            rec = {"text": text}
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_chars += len(text)
            n_processed += 1

            if n_processed % log_every == 0:
                est_tokens = n_chars / chars_per_token
                log(
                    f"[{source_id}] {n_processed:,} docs, "
                    f"{n_chars:,} chars (~{est_tokens:,.0f} tokens)"
                )
                state["n_processed"] = n_processed
                state["n_chars"] = n_chars
                save_state(f"download_{source_id}", state)
                out_f.flush()

            if n_chars >= target_chars:
                log(
                    f"[{source_id}] hit target {target_tokens:,} tokens "
                    f"after {n_processed:,} docs"
                )
                break
    finally:
        out_f.flush()
        out_f.close()

    state["n_processed"] = n_processed
    state["n_chars"] = n_chars
    state["path"] = str(out_path)
    save_state(f"download_{source_id}", state)
    log(
        f"[{source_id}] done: {n_processed:,} docs → {out_path} "
        f"({human_count(n_chars)} chars)"
    )
    return state


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 1: download raw text")
    parser.add_argument("--mixture", required=True)
    parser.add_argument("--source", default=None,
                        help="Download only this source id (default: all)")
    parser.add_argument("--no-streaming", action="store_true",
                        help="Disable HF streaming (downloads the full dataset upfront)")
    args = parser.parse_args(argv)

    ensure_dirs()
    mix = load_yaml(Path(args.mixture))
    total_tokens = mix["mixture"]["total_tokens"]
    sources = mix["mixture"]["sources"]

    for spec in sources:
        if args.source and spec["id"] != args.source:
            continue
        target_tokens = int(total_tokens * spec["weight"])
        state = load_state(f"download_{spec['id']}")
        try:
            download_source(
                spec,
                target_tokens=target_tokens,
                state=state,
                streaming=not args.no_streaming,
            )
        except Exception as e:
            logger.error("[%s] FAILED: %s: %s", spec["id"], type(e).__name__, e)
            continue

    log("download: all requested sources done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())