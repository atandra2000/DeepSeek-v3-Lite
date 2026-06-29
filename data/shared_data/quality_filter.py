"""Per-document quality filters for the shared data pipeline."""
from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from shared_data.common import log


def length_filter(text: str, *, min_chars: int, max_chars: int) -> bool:
    """Keep documents within the [min_chars, max_chars] character range."""
    n = len(text)
    return min_chars <= n <= max_chars


def unique_chars_filter(text: str, *, min_ratio: float = 0.05) -> bool:
    """Reject documents where unique characters < min_ratio * total chars."""
    if not text:
        return False
    return len(set(text)) >= min_ratio * len(text)


def digit_ratio_filter(text: str, *, max_ratio: float = 0.50) -> bool:
    """Reject documents where digits make up > max_ratio of the chars."""
    if not text:
        return False
    digits = sum(c.isdigit() for c in text)
    return (digits / len(text)) <= max_ratio


def punctuation_filter(text: str, *, max_ratio: float = 0.50) -> bool:
    """Reject documents that are mostly punctuation."""
    if not text:
        return False
    punct = sum(1 for c in text if unicodedata.category(c).startswith("P"))
    return (punct / len(text)) <= max_ratio


def whitespace_filter(text: str, *, max_ratio: float = 0.50) -> bool:
    """Reject documents that are mostly whitespace."""
    if not text:
        return False
    ws = sum(c.isspace() for c in text)
    return (ws / len(text)) <= max_ratio


def language_hint_filter(text: str, *, lang: Optional[str] = None) -> bool:
    """Cheap language check by ASCII-letter ratio. ``None`` disables the check."""
    if lang is None:
        return True
    if not text:
        return False

    sample = text[:5000]
    if not sample:
        return False

    if lang.lower() in ("en", "english"):
        ascii_letters = sum(c.isascii() and c.isalpha() for c in sample)
        ratio = ascii_letters / len(sample)
        if ratio < 0.5:
            return False
        lower = sample.lower()
        common_bigrams = ("th", "he", "in", "er", "an", "re", "on", "at")
        return any(bg in lower for bg in common_bigrams)

    return True


@dataclass
class FilterStats:
    """Counters that survive across an entire clean pass."""
    n_seen: int = 0
    n_kept: int = 0
    n_dropped: int = 0
    reasons: Counter = field(default_factory=Counter)

    def record_drop(self, reason: str) -> None:
        self.n_dropped += 1
        self.reasons[reason] += 1

    def keep_ratio(self) -> float:
        return self.n_kept / max(1, self.n_seen)

    def summary(self) -> str:
        lines = [
            f"seen:    {self.n_seen:,}",
            f"kept:    {self.n_kept:,}  ({self.keep_ratio():.1%})",
            f"dropped: {self.n_dropped:,}",
            "reasons:",
        ]
        for reason, count in self.reasons.most_common():
            pct = 100.0 * count / max(1, self.n_seen)
            lines.append(f"  - {reason:24s} {count:>10,}  ({pct:5.2f}%)")
        return "\n".join(lines)


class QualityFilter:
    """Composite filter: run all enabled sub-filters and track rejections."""

    def __init__(
        self,
        min_chars: int,
        max_chars: int,
        lang: Optional[str] = None,
        *,
        drop_empty: bool = True,
        min_unique_chars_ratio: float = 0.05,
        max_digit_ratio: Optional[float] = 0.50,
        max_punct_ratio: float = 0.50,
        max_whitespace_ratio: float = 0.50,
    ):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.lang = lang
        self.drop_empty = drop_empty
        self.min_unique_chars_ratio = min_unique_chars_ratio
        self.max_digit_ratio = max_digit_ratio
        self.max_punct_ratio = max_punct_ratio
        self.max_whitespace_ratio = max_whitespace_ratio

    def apply(self, text: str) -> Optional[str]:
        """Run all filters; return ``text`` if kept, ``None`` if rejected."""
        if self.drop_empty and not text:
            return None
        if not length_filter(text, min_chars=self.min_chars, max_chars=self.max_chars):
            return None
        if not unique_chars_filter(text, min_ratio=self.min_unique_chars_ratio):
            return None
        if self.max_digit_ratio is not None:
            if not digit_ratio_filter(text, max_ratio=self.max_digit_ratio):
                return None
        if not punctuation_filter(text, max_ratio=self.max_punct_ratio):
            return None
        if not whitespace_filter(text, max_ratio=self.max_whitespace_ratio):
            return None
        if not language_hint_filter(text, lang=self.lang):
            return None
        return text


__all__ = [
    "length_filter", "unique_chars_filter", "digit_ratio_filter",
    "punctuation_filter", "whitespace_filter", "language_hint_filter",
    "FilterStats", "QualityFilter",
]