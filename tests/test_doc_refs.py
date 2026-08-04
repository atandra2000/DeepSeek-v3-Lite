"""Machine-checked doc<->code alignment gate (docs/docs_expansion_plan.md section 5).

Scans docs/**/*.md and the root README/AGENTS/SKILLS/CONTEXT/Reference for
`file.py:Symbol` anchors and resolves each against the repo:

  - missing file          -> fail (wrong prefix, e.g. bare `mla.py:`)
  - missing symbol        -> fail (use `file.py:Class.method` when it is a method)
  - line anchors `file.py:123` -> fail (they rot; symbols do not)
  - JIT symbols defined under `if HAS_TRITON:` -> fail (cite the host wrapper)

Run: python -m pytest tests/test_doc_refs.py   (CPU-only, no triton needed)
"""
from __future__ import annotations

import importlib
import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SYMBOL_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_./-]*\.py):([A-Za-z_][A-Za-z0-9_.]+)")
LINE_ANCHOR_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_./-]*\.py):(\d{2,})\b")

# JIT kernels/classes defined under `if HAS_TRITON:` — never resolvable on a
# triton-less CI box. Writers must cite the always-defined host wrappers
# (triton_mla_attention, triton_grouped_moe_dispatch) instead.
JIT_SYMBOLS = {
    "_mla_flash_fwd_kernel",
    "_grouped_moe_fwd_kernel",
    "_grouped_moe_bwd_dx_kernel",
    "_grouped_moe_bwd_dw_kernel",
    "_TritonMlaAttentionFunction",
    "_TritonGroupedMoeFunction",
}

# Paper-spec / planned files allowed to be absent (kept in sync with
# scripts/check_docs.py ALLOW_MISSING_PATHS).
ALLOW_MISSING_FILES = {
    "training/loss_triton.py",
    "training/pretrain_distributed.py",
    "scripts/microbench_a100_triton.py",
    "models/moe_gate_triton.py",
    "models/swiglu_triton.py",
    "models/norm_triton.py",
    "models/__init__.py",
    "tests/test_doc_refs.py",
}

_mod_cache: dict[str, object] = {}


def iter_docs() -> list[Path]:
    # Meta-documents that QUOTE anchor formats as examples (the expansion plan,
    # the contributing guide) are link-linted but not symbol-scanned.
    excluded = {"docs_expansion_plan.md", "G6_contributing.md"}
    docs = [p for p in sorted(ROOT.glob("docs/**/*.md")) if p.name not in excluded]
    for name in ("README.md", "AGENTS.md", "SKILLS.md", "CONTEXT.md", "Reference.md"):
        p = ROOT / name
        if p.exists():
            docs.append(p)
    return docs


def _module_for(path_str: str):
    """Import `models/mla.py` as package `models.mla` (relative imports resolve)."""
    if path_str in _mod_cache:
        return _mod_cache[path_str]
    rel = Path(path_str)
    if not (ROOT / rel).exists():
        return None
    if rel.suffix != ".py":
        return None
    parts = rel.with_suffix("").parts
    if parts[0] in ("models", "training", "utils", "inference", "data", "tests", "scripts"):
        try:
            mod = importlib.import_module(".".join(parts))
        except ModuleNotFoundError:
            # Not a package (e.g. tests/ has no __init__.py) — spec-load below.
            mod = None
        except Exception as exc:  # pragma: no cover - surfaces import breakage
            mod = f"import-error: {type(exc).__name__}: {exc}"
        if mod is not None:
            _mod_cache[path_str] = mod
            return mod
    spec = importlib.util.spec_from_file_location(path_str.replace("/", "_")[:-3], ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # pragma: no cover
        mod = f"import-error: {type(exc).__name__}: {exc}"
    _mod_cache[path_str] = mod
    return mod


def _resolve_symbol(file_str: str, symbol: str) -> str:
    if file_str in ALLOW_MISSING_FILES:
        return "ok-allowed"
    mod = _module_for(file_str)
    if mod is None:
        return "missing-file"
    if isinstance(mod, str):
        return mod
    obj = mod
    for part in symbol.split("."):
        if not hasattr(obj, part):
            return f"missing-symbol: {symbol}"
        obj = getattr(obj, part)
    return "ok"


def collect_issues() -> list[tuple[str, int, str]]:
    issues: list[tuple[str, int, str]] = []
    for doc in iter_docs():
        text = doc.read_text(encoding="utf-8")
        # Line anchors first (banned); then symbols, skipping spans already
        # reported as line anchors.
        line_spans = [m.span() for m in LINE_ANCHOR_RE.finditer(text)]
        for m in LINE_ANCHOR_RE.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            issues.append((doc.name, line, f"line anchor banned: {m.group(1)}:{m.group(2)}"))
        for m in SYMBOL_RE.finditer(text):
            if any(m.start() == s and m.end() == e for s, e in line_spans):
                continue
            file_str, symbol = m.group(1), m.group(2)
            if symbol in JIT_SYMBOLS:
                line = text.count("\n", 0, m.start()) + 1
                issues.append((doc.name, line, f"JIT symbol (cite host wrapper): {file_str}:{symbol}"))
                continue
            status = _resolve_symbol(file_str, symbol)
            if status != "ok" and status != "ok-allowed":
                line = text.count("\n", 0, m.start()) + 1
                issues.append((doc.name, line, f"{file_str}:{symbol} -> {status}"))
    return issues


def test_doc_anchors_resolve() -> None:
    issues = collect_issues()
    assert not issues, "doc<->code anchor defects:\n" + "\n".join(
        f"  {doc}:{line}: {msg}" for doc, line, msg in sorted(issues)
    )


if __name__ == "__main__":
    issues = collect_issues()
    if issues:
        for doc, line, msg in sorted(issues):
            print(f"{doc}:{line}: {msg}")
        raise SystemExit(1)
    print(f"test_doc_refs: OK ({len(iter_docs())} files, all anchors resolve)")
