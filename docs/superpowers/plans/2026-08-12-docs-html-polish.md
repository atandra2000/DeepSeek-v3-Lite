# DeepSeek-v3-Lite Documentation Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refine the generated static documentation portal while preserving DeepSeek-v3-Lite's MLA/MoE research-blueprint identity and Markdown source workflow.

**Architecture:** Keep `scripts/build_docs_html.py` as the single generator and `assets/style.css` as the source stylesheet. Improve the portal's complete documentation coverage and interaction consistency, then regenerate the ignored `docs_html/` artifact.

**Tech Stack:** Python standard library, HTML, CSS, vanilla browser JavaScript.

## Global Constraints

- Keep the Markdown documents as the sole content source.
- Add no runtime or build dependency.
- Keep `docs_html/` fully static and locally navigable.
- Preserve existing uncommitted work unless directly required by this polish.

---

### Task 1: Align portal coverage and chrome

**Files:**
- Modify: `scripts/build_docs_html.py`
- Modify: `assets/style.css`

- [x] Add every generated documentation entry to the portal categories, retaining concise DeepSeek-specific tags and descriptions.
- [x] Normalize portal theme labels and accessible controls to the established GPT-OSS interaction pattern.
- [x] Retain the MLA bottleneck motif, DeepSeek measurements, and blueprint palette.

### Task 2: Regenerate and verify static output

**Files:**
- Generate: `docs_html/index.html` and documentation pages

- [x] Run `python3 scripts/build_docs_html.py`.
- [x] Assert the portal contains all generated pages and validate every local HTML link target.
- [x] Run `python3 scripts/check_docs.py`.
