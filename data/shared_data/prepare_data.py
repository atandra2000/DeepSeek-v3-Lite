"""Universal LLM data pipeline orchestrator (shared by all 5 projects)."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from shared_data.common import (
    CONFIG_ROOT,
    DATA_ROOT,
    RAW_ROOT,
    ensure_dirs,
    get_logger,
    load_yaml,
    log,
    seed_everything,
)
from shared_data.config import (
    PIPELINE_VERSION,
    UNIVERSAL_DATA_CONFIG_PATH,
    UNIVERSAL_MIXTURE_PATH,
    UNIVERSAL_TOTAL_TOKENS,
)


logger = get_logger("prepare_data")


def run_pipeline(
    *,
    mixture_path: Optional[Path] = None,
    data_config_path: Optional[Path] = None,
    source: Optional[str] = None,
    skip_download: bool = False,
    skip_clean: bool = False,
    skip_tokenize: bool = False,
    skip_pack: bool = False,
    skip_train_tokenizer: bool = True,
    data_root: Optional[Path] = None,
) -> int:
    """Run the requested stages (each invoked as a subprocess for crash isolation)."""
    if data_root is not None:
        from shared_data.common import set_data_root
        set_data_root(data_root)
    mixture_path = Path(mixture_path or UNIVERSAL_MIXTURE_PATH)
    data_config_path = Path(data_config_path or UNIVERSAL_DATA_CONFIG_PATH)

    if not mixture_path.exists():
        log(f"ERROR: mixture.yaml not found at {mixture_path}", level="ERROR")
        return 2
    if not data_config_path.exists():
        log(f"ERROR: data_config.yaml not found at {data_config_path}",
            level="ERROR")
        return 2

    mix = load_yaml(mixture_path)
    cfg = load_yaml(data_config_path)
    declared_total = int(mix["mixture"]["total_tokens"])
    pipeline_total = int(cfg["pipeline"]["sharding"]["target_total_tokens"])
    if declared_total != pipeline_total:
        log(
            f"WARNING: mixture.total_tokens ({declared_total:,}) != "
            f"data_config.pipeline.sharding.target_total_tokens "
            f"({pipeline_total:,}). Using the mixture's value.",
            level="WARNING",
        )
    log(f"pipeline version: {PIPELINE_VERSION}")
    log(f"corpus target: {declared_total:,} tokens")
    log(f"data root: {DATA_ROOT}")

    seed = int(cfg["pipeline"].get("seed", 42))
    seed_everything(seed)

    ensure_dirs()
    rc = 0

    if not skip_train_tokenizer:
        rc = _run_module(
            "shared_data.scripts.train_tokenizer",
            ["--output", str(DATA_ROOT / "tokenizer" / "custom-bpe")],
        )
        if rc != 0:
            return rc

    if not skip_download:
        rc = _run_module(
            "shared_data.scripts.download_raw",
            ["--mixture", str(mixture_path),
             *(["--source", source] if source else [])],
        )
        if rc != 0:
            return rc

    if not skip_clean:
        rc = _run_module(
            "shared_data.scripts.clean",
            ["--mixture", str(mixture_path),
             "--data-config", str(data_config_path),
             *(["--source", source] if source else [])],
        )
        if rc != 0:
            return rc

    if not skip_tokenize:
        rc = _run_module(
            "shared_data.scripts.tokenize",
            ["--mixture", str(mixture_path),
             "--data-config", str(data_config_path),
             *(["--source", source] if source else [])],
        )
        if rc != 0:
            return rc

    if not skip_pack:
        rc = _run_module(
            "shared_data.scripts.pack_shards",
            ["--mixture", str(mixture_path),
             "--data-config", str(data_config_path)],
        )
        if rc != 0:
            return rc

    log(f"all stages complete. manifest at {DATA_ROOT / 'manifest.json'}")
    return rc


def _run_module(module: str, argv: List[str]) -> int:
    """Invoke ``python -m <module> <argv>`` as a subprocess."""
    cmd = [sys.executable, "-m", module, *argv]
    log(f"running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except Exception as e:
        logger.error("failed to invoke %s: %s", module, e)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Universal LLM data preparation (shared across all 5 projects)"
    )
    parser.add_argument("--stage", choices=["pretrain"], default="pretrain")
    parser.add_argument("--mixture",
                        default=str(UNIVERSAL_MIXTURE_PATH))
    parser.add_argument("--data-config",
                        default=str(UNIVERSAL_DATA_CONFIG_PATH))
    parser.add_argument("--data-root", default=None,
                        help="Override DATA_ROOT (default: $LLM_DATA_ROOT or $PWD/data)")
    parser.add_argument("--source", default=None,
                        help="Restrict to a single source id (default: all)")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-clean", action="store_true")
    parser.add_argument("--skip-tokenize", action="store_true")
    parser.add_argument("--skip-pack", action="store_true")
    parser.add_argument("--train-tokenizer", action="store_true",
                        help="Also train a custom BPE before download (FusionLLM uses this)")
    args = parser.parse_args()

    rc = run_pipeline(
        mixture_path=Path(args.mixture),
        data_config_path=Path(args.data_config),
        source=args.source,
        skip_download=args.skip_download,
        skip_clean=args.skip_clean,
        skip_tokenize=args.skip_tokenize,
        skip_pack=args.skip_pack,
        skip_train_tokenizer=not args.train_tokenizer,
        data_root=Path(args.data_root) if args.data_root else None,
    )
    return rc


if __name__ == "__main__":
    sys.exit(main())