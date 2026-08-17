# Design — Premium Polish of the docs_html Documentation Portal

**Date:** 2026-08-18
**Status:** Approved in brainstorming; pending user review of this spec
**Scope:** `assets/style.css`, `scripts/build_docs_html.py`, new `assets/portal.js`, regenerated `docs_html/`

## Goal

Elevate the static documentation portal to a premium, minimalist, code-centric
standard: monospace typography throughout, a DeepSeek-specific ASCII pixel
animation on the landing page, a loading animation on every navigation, and a
set of interactive ASCII widgets — without altering any documentation text.

## Locked decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Visual direction | "Blueprint, refined" — evolve existing ink/paper/ember theme, no repaint |
| Typography | Monospace for body AND headings (IBM Plex Mono; serif dropped) |
| Hero animation | Wordmark scramble-decode (ASCII, prototype C) |
| Loading animation | Weight-load ASCII progress bar (prototype B), shown on EVERY page load |
| Widgets | Expandable code, 18-layer explorer, MoE playground, MLA toggle, progress strip |
| Rejected | Keyboard navigation widget |

## Hard constraints

1. Markdown files under `docs/` (plus README/AGENTS/SKILLS) are the sole
   content source and MUST NOT be edited. No textual content or structural
   hierarchy changes.
2. `docs_html/` remains a generated, gitignored artifact. All changes live in
   `assets/style.css`, `assets/portal.js` (new), `scripts/build_docs_html.py`.
3. No runtime or build dependencies. Vanilla JS only, stdlib Python only.
4. Portal stays fully static and locally navigable (`file://` safe: no
   fetch(), no modules).
5. `python3 scripts/check_docs.py` and `tests/test_doc_refs.py` must pass.

## 1. Theme and typography

- Keep all existing design tokens (ink/paper surfaces, ruler hairlines,
  ember accent, rule slate). **Dark theme only** (2026-08-18 amendment):
  remove the `[data-theme="light"]` token blocks, the theme-toggle
  button, and the toggle script from every generated page.
- Swap prose face: `--font-prose` becomes `--font-mono` (or the prose rules
  are pointed at the mono stack). Remove IBM Plex Serif from the Google
  Fonts link in `build_docs_html.py`; keep IBM Plex Mono weights.
- Mono body readability tuning: body size 15px, line-height 1.7,
  slightly negative letter-spacing on headings only.
- All mono-everywhere rules already covering chrome remain; extend to
  `.markdown-body p/li/blockquote`, `.hero-subtitle`, `.card-desc`,
  `.callout-body`, `.math-block` captions.
- KaTeX math blocks keep their rendered output (KaTeX fonts) — math is the
  one sanctioned non-mono surface.

## 2. Landing hero — ASCII scramble-decode

- Replace the SVG bottleneck figure (`.hero-figure` contents) with an ASCII
  stage: `<pre class="ascii-stage">` running the wordmark scramble-decode:
  - Target line: `DEEPSEEK-V3-LITE`; glyph pool `░▒▓/\\|=+*·#<>`.
  - Characters lock left→right over ~1.6 s, unlocked cells cycle glyphs.
  - After lock, a second line types with a block cursor:
    `411.6M · MLA · MoE · MTP · μP`.
  - Runs once per page load, then holds the final frame (no infinite loop).
- The visible `<h1 class="hero-title">` becomes `sr-only` (kept for
  accessibility/SEO); the ASCII stage is `aria-hidden="true"` with an
  `aria-label` describing it.
- Coordinate strip (`FIG · 00 / MLP-DEPTH 18 / …`) and the datasheet grid
  stay exactly as-is, below the stage.
- `prefers-reduced-motion`: render the final frame immediately, no cycling.

## 3. Loading overlay — every navigation

- Injected on ALL generated pages (index + every doc page).
- Markup: full-screen overlay `#boot-overlay` (ink background, centered):
  wordmark line + `loading weights [▓▓▓▓░░░░░░░░] NN%` (12-cell bar).
- Behaviour:
  - Fills 0→100% over ~500 ms (ease-out), percentage rendered as integer.
  - Then ~200 ms fade-out; overlay removed from DOM.
  - Skippable: any click or keypress completes instantly.
  - `prefers-reduced-motion`: overlay never shown.
- Anti-flash + fail-safe: a ~15-line inline `<script>` in `<head>` runs
  before first paint. It (a) does nothing if `prefers-reduced-motion`
  matches or JS is disabled, (b) otherwise adds `html.booting` (CSS hides
  page content except the overlay), and (c) arms a 1.2 s watchdog that
  removes `.booting` unconditionally, so a portal.js failure can never
  strand the page. Normal path: portal.js (defer) runs the ~700 ms
  animation and clears the class itself.
- Shown on every page load by design (static navigation = full page load),
  no sessionStorage suppression.

## 4. Widgets (all vanilla JS in portal.js, containers injected by build script)

All widgets degrade gracefully: with JS disabled, containers render their
static fallback content (final-frame ASCII art) — no blank boxes.

### 4.1 Expandable code snippets (all doc pages)
- Any `.code-wrapper pre` with more than 16 lines collapses to the first 12.
- Seam button below: `── show N more lines ▾` / `── collapse ▴` (mono, dim,
  hairline top border). No layout jump wider than the block itself.

### 4.2 18-layer stack explorer (`index.html`, inserted after the datasheet)
- Static ASCII stack (final frame of the "layer stack" prototype): 18 rows,
  `00`/`01` dense (MLA + SwiGLU), `02–17` MoE (MLA + 20+1 experts, top-4),
  with a `···` compression for middle layers.
- Each row is focusable; hover/focus updates an adjacent spec panel:
  layer index, block type, key dims (dim 768, kv_lora_rank 192,
  qk_rope_head_dim 24, heads 12, moe_inter_dim 384, 20 routed + 1 shared,
  top-4), per-type parameter note. Values are constants mirroring
  `configs/pretrain_a100_422m.yaml`.

### 4.3 MoE routing playground (`docs/concepts/moe-mtp.html`)
- 4×5 pixel grid of the 20 routed experts + a pinned shared-expert row.
- Button `▸ route token`: picks 4 distinct experts at random, lights them
  (`▓▓` in ember), prints normalized routing weights summing to 1, shared
  expert always lit. Status line: `shared + 4 routed active`.

### 4.4 MLA absorption toggle (`docs/concepts/attention-and-precision.html`)
- Toggle `[ standard ] / [ absorbed ]` swaps a `<pre>` diagram between the
  materialised KV path and the absorbed latent path, each annotated with
  the cache/compression deltas (768→192+24 latent, per-token KV savings).
  Numbers consistent with `docs/concepts/attention-and-precision.md`.

### 4.5 Reading-progress strip (all doc pages, not index)
- Sticky 1-line strip under the header: `[▓▓▓░░░░░░░] 42% · §3 Heading`.
- rAF-throttled scroll listener; section name from the last heading above
  the viewport (existing `.heading-anchor` ids / TOC entries).

## 5. JS architecture

- New `assets/portal.js` (single file, ~400–600 lines), copied to
  `docs_html/assets/portal.js` by the build script, exactly like
  `style.css`. Loaded with `defer`.
- Build script changes: copy portal.js; inject `<script src>` + inline boot
  script; inject widget container divs at fixed anchors (index hero /
  after datasheet; the two concept pages, identified by rel path); replace
  hero SVG with the ASCII stage; make hero h1 sr-only; update font link.
- No widget state persists; everything recomputes on load.

## 6. Motion and accessibility

- All animations use `requestAnimationFrame`; loop-style animations pause
  off-screen via `IntersectionObserver`.
- Every animation gated behind `prefers-reduced-motion: no-preference`
  (CSS and JS sides).
- Focus-visible rings, `aria-hidden` on decorative ASCII, sr-only headings,
  contrast of the existing token system all preserved.

## 7. Verification

1. `python3 scripts/build_docs_html.py` — regenerate portal.
2. `python3 scripts/check_docs.py` — links + stale-pattern lint.
3. `python3 -m pytest tests/test_doc_refs.py -q` — doc↔code anchors intact.
4. Manual browser pass: index (hero decode, layer explorer, overlay),
   moe-mtp.html, attention-and-precision.html, one long reference page
   (expandable code + progress strip); dark theme only; reduced
   motion; JS-disabled render.

## Out of scope

- Keyboard navigation widget (rejected).
- Any edit to Markdown documentation content.
- New documentation pages or nav restructuring.
- Phosphor-terminal or graphite-zinc repaints (rejected directions).
