# quality_filter.py — notes

> See [`../README.md`](../README.md) §1 (the mixture table) and §2
> (pipeline diagram) for context.

These heuristics run AFTER dedup and BEFORE tokenisation. They are
intentionally cheap (no external models, no language IDs) so the
pipeline stays offline-friendly and CPU-fast.

Each filter is a function `(text: str) -> bool` returning `True` to KEEP
the document. The reason for failure (if any) is captured in the
`rejection_reasons` counter so we can debug the data mix after the fact.

## The 6 heuristics

1. **`length_filter(text, *, min_chars, max_chars)`** — min/max character
   bounds (per source). Drops documents outside `[min_chars, max_chars]`.
2. **`unique_chars_filter(text, *, min_ratio=0.05)`** — reject low-diversity
   junk (`"aaaaaaaaaa..."`). Keeps when `len(set(text)) >= min_ratio * len(text)`.
3. **`digit_ratio_filter(text, *, max_ratio=0.50)`** — reject pure-number
   dumps. Pure-number dumps tend to be data tables / logs / IDs — not
   useful for pretraining. **NOT applied to code corpora** — the pipeline
   sets `max_digit_ratio=None` for `the-stack-python`.
4. **`punctuation_filter(text, *, max_ratio=0.50)`** — reject
   punctuation-only junk. Uses `unicodedata.category(c).startswith("P")`.
5. **`whitespace_filter(text, *, max_ratio=0.50)`** — reject documents
   that are mostly whitespace (newlines, tabs, spaces).
6. **`language_hint_filter(text, *, lang=None)`** — cheap byte-ratio
   heuristic for English. `None` disables the check.
   - `lang in {"en", "english"}`: ASCII-letter ratio must be ≥ 0.5 AND a
     common English bigram (`"th"`, `"he"`, `"in"`, `"er"`, `"an"`,
     `"re"`, `"on"`, `"at"`) must appear at least once in the first
     5 000 chars. This rejects random Latin-1-looking noise without
     needing a real language ID.
   - For other languages (`"python"`, `"zh"`): pass-through.

## FilterStats

`FilterStats` is a dataclass of counters that survive across an entire
clean pass: `n_seen`, `n_kept`, `n_dropped`, `reasons: Counter`.
`record_drop(reason)` bumps `n_dropped` and the reason counter.
`keep_ratio()` returns `n_kept / max(1, n_seen)`. `summary()` returns a
pretty multi-line breakdown used by `scripts/clean.py` for per-source
logging.

## QualityFilter (composite)

`QualityFilter` runs all enabled sub-filters in order and returns
`text` (kept) or `None` (rejected). `max_digit_ratio=None` disables the
digit filter (used for code corpora). The `apply(text)` method is the
single entry point used by `scripts/clean.py`.