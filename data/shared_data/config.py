"""Universal pipeline configuration constants."""
from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent        # .../LLM/shared_data/
LLM_ROOT = PACKAGE_ROOT.parent                         # .../LLM/

UNIVERSAL_MIXTURE_PATH = PACKAGE_ROOT / "config" / "mixture.yaml"
UNIVERSAL_DATA_CONFIG_PATH = PACKAGE_ROOT / "config" / "data_config.yaml"

UNIVERSAL_TOTAL_TOKENS: int = 8_000_000_000

PIPELINE_VERSION: str = "1.0.0"


def load_universal_mixture() -> dict:
    """Load the canonical mixture YAML."""
    from shared_data.common import load_yaml
    return load_yaml(UNIVERSAL_MIXTURE_PATH)


def load_universal_data_config() -> dict:
    """Load the canonical data_config YAML."""
    from shared_data.common import load_yaml
    return load_yaml(UNIVERSAL_DATA_CONFIG_PATH)


DEFAULT_SHARD_SIZE_TOKENS: int = 50_000_000

SRC_FINEWEB_EDU = "fineweb-edu"
SRC_FINEWEB = "fineweb"
SRC_STACK_PYTHON = "the-stack-python"
SRC_OPENMATH = "openmath"
SRC_ARXIV = "arxiv"

ALL_SOURCES = (SRC_FINEWEB_EDU, SRC_FINEWEB, SRC_STACK_PYTHON, SRC_OPENMATH, SRC_ARXIV)


__all__ = [
    "PACKAGE_ROOT", "LLM_ROOT",
    "UNIVERSAL_MIXTURE_PATH", "UNIVERSAL_DATA_CONFIG_PATH",
    "UNIVERSAL_TOTAL_TOKENS", "PIPELINE_VERSION",
    "DEFAULT_SHARD_SIZE_TOKENS",
    "SRC_FINEWEB_EDU", "SRC_FINEWEB", "SRC_STACK_PYTHON",
    "SRC_OPENMATH", "SRC_ARXIV", "ALL_SOURCES",
    "load_universal_mixture", "load_universal_data_config",
]