# Premium docs_html Portal Polish — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Polish the generated docs portal to a premium, minimalist, code-centric standard: mono typography everywhere, an ASCII wordmark scramble-decode hero, a weight-load boot overlay on every page, and five ASCII widgets — without touching any Markdown content.

**Architecture:** `docs_html/` is a gitignored build artifact. All changes live in the sources of truth: `assets/style.css` (tokens + new component styles), new `assets/portal.js` (all runtime behaviour), and `scripts/build_docs_html.py` (asset copying, head/body injection, hero replacement, widget containers). `python3 scripts/build_docs_html.py` regenerates the portal.

**Tech Stack:** Python stdlib, HTML, CSS, vanilla browser JS. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-18-docs-html-premium-design.md`

## Global Constraints

- NEVER edit Markdown files (`docs/**`, `README.md`, `AGENTS.md`, `SKILLS.md`).
- All JS must be `file://`-safe: no `fetch()`, no ES modules, no top-level `await`.
- All animations gate on `prefers-reduced-motion`.
- Follow AGENTS.md rule 9: concise comments only.
- Commit steps are included; if the user has asked not to commit, skip the `git commit` steps but keep the rest.

---

### Task 1: Failing build-output contract tests

**Files:**
- Create: `tests/test_build_docs_html.py`
- Modify: `.gitignore` (append one line)

- [ ] **Step 1: Add `.superpowers/` to `.gitignore`**

Append to `.gitignore`:

```
.superpowers/
```

- [ ] **Step 2: Write the contract tests**

Create `tests/test_build_docs_html.py`:

```python
"""Build-output contract tests for the docs_html portal.

Runs the real generator once (module scope) and asserts the premium-polish
wiring: portal.js asset, boot overlay, ASCII hero, mono-only fonts, widget
containers. Markdown sources are never modified by the build.
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "scripts" / "build_docs_html.py"
OUT = ROOT / "docs_html"


@pytest.fixture(scope="module")
def built():
    subprocess.run([sys.executable, str(BUILD)], check=True, cwd=ROOT)
    return OUT


def read(rel: str) -> str:
    return (OUT / rel).read_text(encoding="utf-8")


def test_portal_js_asset_copied(built):
    assert (OUT / "assets" / "portal.js").is_file()


def test_doc_page_boot_wiring(built):
    html = read("README.html")
    assert 'id="boot-overlay"' in html
    assert "booting" in html
    assert '<script defer src="./assets/portal.js"></script>' in html


def test_nested_page_rel_prefix(built):
    html = read("docs/concepts/moe-mtp.html")
    assert 'src="../../assets/portal.js"' in html


def test_font_link_mono_only(built):
    html = read("README.html")
    assert "IBM+Plex+Mono" in html
    assert "IBM+Plex+Serif" not in html


def test_index_hero_ascii_stage(built):
    html = read("index.html")
    assert 'id="hero-decode"' in html
    assert 'data-title="DEEPSEEK-V3-LITE"' in html
    assert "bottleneck-svg" not in html
    assert "hero-title sr-only" in html


def test_index_layer_explorer_container(built):
    html = read("index.html")
    assert 'id="widget-layer-stack"' in html
    assert "ascii-stack" in html


def test_moe_playground_container(built):
    assert 'id="widget-moe-routing"' in read("docs/concepts/moe-mtp.html")


def test_mla_toggle_container(built):
    assert 'id="widget-mla-absorb"' in read("docs/concepts/attention-and-precision.html")


def test_widgets_not_on_other_pages(built):
    html = read("docs/concepts/parallelism.html")
    assert "widget-moe-routing" not in html
    assert "widget-mla-absorb" not in html
    assert "widget-layer-stack" not in html
```

- [ ] **Step 3: Run to verify failure**

Run: `python3 -m pytest tests/test_build_docs_html.py -q`
Expected: multiple FAILs (missing `portal.js`, missing markers). `test_font_link_mono_only` and any container test fail; asset test fails.

- [ ] **Step 4: Commit**

```bash
git add .gitignore tests/test_build_docs_html.py
git commit -m "test: add build-output contract tests for docs portal polish"
```

---

### Task 2: CSS — mono typography + new component styles

**Files:**
- Modify: `assets/style.css`

- [ ] **Step 1: Point prose at mono, tune body metrics, and strip the light theme**

In `assets/style.css`, apply these edits:

Edit A — type tokens (end of section 1):

```css
    /* Type families */
    --font-prose:    'IBM Plex Serif', Charter, 'Iowan Old Style', Georgia, serif;
    --font-mono:     'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
```

replace with:

```css
    /* Type families — the portal is mono end-to-end; --font-prose is an
       alias kept so the existing prose rules keep resolving. */
    --font-mono:     'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    --font-prose:    var(--font-mono);
```

Edit B — body metrics:

```css
    font-family: var(--font-prose);
    font-size: 16px;
    line-height: 1.62;
```

replace with:

```css
    font-family: var(--font-prose);
    font-size: 15px;
    line-height: 1.7;
```

Edit C — markdown body size:

```css
.markdown-body { font-size: 1.04rem; }
```

replace with:

```css
.markdown-body { font-size: 1rem; }
```

Edit D — dark theme only: delete the ENTIRE `[data-theme="light"] { ... }`
 token block (section 1) and the `[data-theme="light"] body::before { ... }`
 block (section 2). Then purge `.theme-toggle` from the selector pairs
 `.theme-toggle, .mobile-toggle` → `.mobile-toggle` and
 `.theme-toggle:hover, .mobile-toggle:hover` → `.mobile-toggle:hover`,
 and remove `.theme-toggle` from the mono-font selector list in section 3.
 Keep the `data-theme="dark"` attribute on `<html>` (harmless).

- [ ] **Step 2: Append the new component styles**

Append at the END of `assets/style.css` (after the `prefers-reduced-motion: reduce` block):

```css
/* =====================================================================
   12. PREMIUM POLISH — boot overlay, ASCII stage, widgets, seam, strip
   ===================================================================== */

.sr-only {
    position: absolute;
    width: 1px; height: 1px;
    padding: 0; margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
}

/* ----- Boot (loading) overlay ---------------------------------------- */
#boot-overlay {
    position: fixed;
    inset: 0;
    z-index: 100;
    background: var(--ink);
    display: flex;
    align-items: center;
    justify-content: center;
    transition: opacity 0.2s ease;
}
#boot-overlay.boot-fading { opacity: 0; pointer-events: none; }
.boot-inner { font-family: var(--font-mono); text-align: left; }
.boot-wordmark {
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    color: var(--ink-text);
    margin-bottom: 0.9rem;
}
.boot-line { font-size: 0.78rem; color: var(--note); }
.boot-bar { color: var(--rule-mark); }

/* While booting, hide the page chrome (overlay covers it anyway, but this
   prevents a flash at the edges on slow paints). */
html.booting body > *:not(#boot-overlay) { visibility: hidden; }

/* ----- Hero ASCII stage ------------------------------------------------ */
.ascii-stage {
    margin: 0;
    padding: 1.6rem 1rem;
    font-family: var(--font-mono);
    font-size: clamp(0.9rem, 2.6vw, 1.5rem);
    font-weight: 700;
    letter-spacing: 0.08em;
    line-height: 1.6;
    color: var(--ink-text);
    white-space: pre;
    overflow-x: auto;
}

/* ----- ASCII widgets (shared chrome) ----------------------------------- */
.ascii-widget {
    margin: 2.2rem 0;
    border: 1px solid var(--ruler);
    background: var(--paper);
}
.ascii-widget-head {
    font-family: var(--font-mono);
    font-size: 0.66rem;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--rule-mark);
    border-bottom: 1px solid var(--ruler);
    padding: 0.55rem 1rem;
}
.ascii-widget-head span { color: var(--faint); font-weight: 400; letter-spacing: 0.06em; text-transform: none; }
.ascii-widget pre {
    margin: 0;
    padding: 1rem 1.2rem;
    font-family: var(--font-mono);
    font-size: 0.8rem;
    line-height: 1.55;
    color: var(--dim);
    overflow-x: auto;
}

.ascii-widget-grid {
    display: grid;
    grid-template-columns: minmax(0, auto) minmax(0, 1fr);
    gap: 1px;
    background: var(--ruler);
}
.ascii-widget-grid > pre { background: var(--paper); }

.stack-row { display: inline; cursor: pointer; outline: none; }
.stack-row:hover, .stack-row:focus-visible, .stack-row.active {
    color: var(--ember);
    background: var(--ember-tint);
}
.ascii-panel { color: var(--note); min-width: 260px; }

.moe-fire, .mla-toggle-bar button {
    background: transparent;
    border: 1px solid var(--ruler-strong);
    color: var(--dim);
    font-family: var(--font-mono);
    font-size: 0.74rem;
    letter-spacing: 0.06em;
    padding: 5px 12px;
    margin: 0 1rem 1rem 1.2rem;
    cursor: pointer;
    transition: color 0.15s ease, border-color 0.15s ease, background 0.15s ease;
}
.moe-fire:hover, .mla-toggle-bar button:hover {
    color: var(--ink-text);
    border-color: var(--rule-mark);
    background: var(--rule-tint);
}
.mla-toggle-bar button[aria-pressed="true"] {
    color: var(--ember);
    border-color: var(--ember);
    background: var(--ember-tint);
}
.moe-status {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--note);
    padding: 0 1.2rem 0.8rem;
}

/* ----- Expandable code seam --------------------------------------------- */
.code-wrapper.collapsed pre { max-height: 15.5rem; overflow: hidden; }
.code-wrapper.collapsed::after {
    content: "";
    position: absolute;
    left: 0; right: 0; bottom: 2rem;
    height: 3rem;
    background: linear-gradient(to bottom, transparent, var(--code-paper));
    pointer-events: none;
}
.code-seam {
    display: block;
    width: 100%;
    background: var(--paper-2);
    border: none;
    border-top: 1px solid var(--code-rule);
    color: var(--note);
    font-family: var(--font-mono);
    font-size: 0.7rem;
    letter-spacing: 0.06em;
    padding: 7px 14px;
    text-align: left;
    cursor: pointer;
    transition: color 0.15s ease, background 0.15s ease;
}
.code-seam:hover { color: var(--ember); background: var(--paper-3); }

/* ----- Reading progress strip ------------------------------------------- */
.progress-strip {
    position: fixed;
    top: 50px;
    left: 0; right: 0;
    z-index: 40;
    font-family: var(--font-mono);
    font-size: 0.64rem;
    letter-spacing: 0.06em;
    color: var(--note);
    background: var(--header-bg);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border-bottom: 1px solid var(--ruler);
    padding: 3px 1.6rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    pointer-events: none;
}

@media (max-width: 880px) {
    .ascii-widget-grid { grid-template-columns: 1fr; }
    .ascii-panel { min-width: 0; border-top: 1px solid var(--ruler); }
}
```

- [ ] **Step 3: Sanity-check the CSS parses**

Run: `python3 -c "open('assets/style.css').read(); print('ok')"`
Expected: `ok` (no syntax tooling in repo; visual check braces balance).

- [ ] **Step 4: Commit**

```bash
git add assets/style.css
git commit -m "style: mono-everywhere typography + boot/hero/widget styles for portal"
```

---

### Task 3: Create `assets/portal.js`

**Files:**
- Create: `assets/portal.js`

- [ ] **Step 1: Write the complete runtime**

Create `assets/portal.js` with exactly this content:

```js
/* DeepSeek-v3-Lite portal runtime: boot overlay, hero decode, ASCII
   widgets. Vanilla JS, no dependencies, file://-safe. */
(function () {
    'use strict';

    var REDUCED = window.matchMedia &&
        window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    /* ---- boot overlay ------------------------------------------------ */

    function finishBoot() {
        document.documentElement.classList.remove('booting');
        var ov = document.getElementById('boot-overlay');
        if (ov && ov.parentNode) ov.parentNode.removeChild(ov);
    }

    function runBoot() {
        var ov = document.getElementById('boot-overlay');
        if (!ov || REDUCED) { finishBoot(); return; }
        var bar = ov.querySelector('.boot-bar');
        var t0 = null, FILL = 500, done = false;
        function skip() { if (!done) { done = true; finishBoot(); } }
        document.addEventListener('click', skip, { once: true });
        document.addEventListener('keydown', skip, { once: true });
        function frame(t) {
            if (done) return;
            if (t0 === null) t0 = t;
            var p = Math.min(1, (t - t0) / FILL);
            var e = 1 - Math.pow(1 - p, 3);
            var n = Math.round(e * 12);
            if (bar) {
                bar.textContent = '[' + '\u2593'.repeat(n) +
                    '\u2591'.repeat(12 - n) + '] ' + Math.round(e * 100) + '%';
            }
            if (p >= 1) {
                done = true;
                ov.classList.add('boot-fading');
                setTimeout(finishBoot, 220);
                return;
            }
            requestAnimationFrame(frame);
        }
        requestAnimationFrame(frame);
    }

    /* ---- hero scramble-decode ----------------------------------------- */

    var GLYPHS = '\u2591\u2592\u2593/\\|=+*\u00b7#<>';

    function glyph() { return GLYPHS[Math.floor(Math.random() * GLYPHS.length)]; }

    function runHeroDecode() {
        var pre = document.getElementById('hero-decode');
        if (!pre) return;
        var title = pre.getAttribute('data-title') || 'DEEPSEEK-V3-LITE';
        var sub = pre.getAttribute('data-sub') || '';
        if (REDUCED) { pre.textContent = '  ' + title + '\n\n  ' + sub; return; }
        var t0 = null, LOCK = 1600, TYPE_SPEED = 28;
        function frame(t) {
            if (t0 === null) t0 = t;
            var el = t - t0;
            var locked = Math.min(title.length, Math.floor(el / LOCK * title.length));
            var line = '';
            for (var i = 0; i < title.length; i++) line += i < locked ? title[i] : glyph();
            if (locked >= title.length) {
                var k = Math.min(sub.length, Math.floor((el - LOCK) / TYPE_SPEED));
                pre.textContent = '  ' + line + '\n\n  ' + sub.slice(0, k) +
                    (k < sub.length ? '\u2588' : '');
                if (k >= sub.length) return;
            } else {
                pre.textContent = '  ' + line;
            }
            requestAnimationFrame(frame);
        }
        requestAnimationFrame(frame);
    }

    /* ---- expandable code blocks ---------------------------------------- */

    var CODE_MAX = 16, CODE_SHOWN = 12;

    function initExpandableCode() {
        document.querySelectorAll('.markdown-body .code-wrapper').forEach(function (wrap) {
            var pre = wrap.querySelector('pre');
            if (!pre) return;
            var lines = pre.textContent.split('\n');
            if (lines.length <= CODE_MAX) return;
            var hidden = lines.length - CODE_SHOWN;
            wrap.classList.add('collapsible', 'collapsed');
            var btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'code-seam';
            btn.setAttribute('aria-expanded', 'false');
            btn.textContent = '\u2500\u2500 show ' + hidden + ' more lines \u25be';
            btn.addEventListener('click', function () {
                var open = wrap.classList.toggle('collapsed') === false;
                btn.setAttribute('aria-expanded', String(open));
                btn.textContent = open ? '\u2500\u2500 collapse \u25b4'
                    : ('\u2500\u2500 show ' + hidden + ' more lines \u25be');
            });
            wrap.appendChild(btn);
        });
    }

    /* ---- 18-layer stack explorer ---------------------------------------- */

    function layerSpec(i) {
        var dense = i < 2;
        return [
            'layer      ' + String(i).padStart(2, '0') + ' / 17',
            'attention  MLA \u00b7 kv_lora_rank 192 \u00b7 rope_head 24',
            dense ? 'ffn        SwiGLU \u00b7 inter_dim 1536'
                  : 'ffn        DeepSeekMoE \u00b7 20 routed (top-4) + 1 shared \u00b7 inter 384',
            dense ? 'role       dense warmup layer' : 'role       routed MoE layer',
            'source     configs/pretrain_a100_422m.yaml'
        ].join('\n');
    }

    function stackRowText(i) {
        var dense = i < 2;
        var inner = (' ' + String(i).padStart(2, '0') + ' \u2502 MLA \u25b8 ' +
            (dense ? 'SwiGLU \u00b7 dense' : 'MoE 20+1 \u00b7 top-4')).padEnd(36);
        return '\u2502' + inner + '\u2502';
    }

    function initLayerExplorer() {
        var box = document.getElementById('widget-layer-stack');
        if (!box) return;
        var pre = box.querySelector('pre.ascii-stack');
        var panel = box.querySelector('.ascii-panel');
        if (!pre) return;
        /* Rebuild the static fallback art as focusable rows. */
        pre.textContent = '';
        pre.appendChild(document.createTextNode(
            '\u250c' + '\u2500'.repeat(36) + '\u2510\n'));
        for (var i = 17; i >= 0; i--) {
            var row = document.createElement('span');
            row.className = 'stack-row';
            row.tabIndex = 0;
            row.setAttribute('data-layer', String(i));
            row.textContent = stackRowText(i);
            pre.appendChild(row);
            pre.appendChild(document.createTextNode('\n'));
        }
        pre.appendChild(document.createTextNode(
            '\u2514' + '\u2500'.repeat(36) + '\u2518'));
        if (!panel) return;
        function select(ev) {
            var idx = parseInt(ev.currentTarget.getAttribute('data-layer'), 10);
            panel.textContent = layerSpec(idx);
            pre.querySelectorAll('.stack-row.active').forEach(function (r) {
                r.classList.remove('active');
            });
            ev.currentTarget.classList.add('active');
        }
        pre.querySelectorAll('.stack-row').forEach(function (row) {
            row.addEventListener('mouseenter', select);
            row.addEventListener('focus', select);
            row.addEventListener('click', select);
        });
    }

    /* ---- MoE routing playground ------------------------------------------ */

    function initMoePlayground() {
        var box = document.getElementById('widget-moe-routing');
        if (!box) return;
        var grid = box.querySelector('pre.moe-grid');
        if (!grid) return;
        var status = document.createElement('div');
        status.className = 'moe-status';
        status.textContent = 'shared + 4 routed active';
        var btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'moe-fire';
        btn.textContent = '\u25b8 route token';
        box.appendChild(status);
        box.appendChild(btn);
        function render(active, weights) {
            var s = 'gate(x) \u25b8 top-4 of 20 routed\n';
            for (var r = 0; r < 4; r++) {
                for (var c = 0; c < 5; c++) {
                    var idx = r * 5 + c;
                    s += (active.indexOf(idx) >= 0 ? '\u2593\u2593' : '\u2591\u2591') +
                        String(idx).padStart(2, '0') + ' ';
                }
                s += '\n';
            }
            s += 'shared  \u2593\u2593sh (always on)';
            grid.textContent = s;
            if (weights) {
                var parts = active.map(function (e, k) {
                    return 'e' + String(e).padStart(2, '0') + '=' + weights[k].toFixed(2);
                });
                status.textContent = 'shared + 4 routed active \u00b7 ' + parts.join(' ');
            }
        }
        function route() {
            var act = [];
            while (act.length < 4) {
                var e = Math.floor(Math.random() * 20);
                if (act.indexOf(e) < 0) act.push(e);
            }
            var logits = act.map(function () { return Math.random() * 2; });
            var m = Math.max.apply(null, logits);
            var ex = logits.map(function (v) { return Math.exp(v - m); });
            var sum = ex.reduce(function (a, b) { return a + b; }, 0);
            render(act, ex.map(function (v) { return v / sum; }));
        }
        btn.addEventListener('click', route);
        route();
    }

    /* ---- MLA absorption toggle --------------------------------------------- */

    var MLA_STANDARD = [
        'standard attention (materialised KV)',
        '',
        ' h[768] \u2500 W_q \u2500\u25b8 q [12 \u00d7 72]  \u2510',
        ' h[768] \u2500 W_k \u2500\u25b8 k [12 \u00d7 48]  \u251c\u2500\u25b8 attn(q, k, v) \u25b8 out',
        ' h[768] \u2500 W_v \u2500\u25b8 v [12 \u00d7 64]  \u2518',
        '',
        ' KV cache / token : 12 \u00d7 (48 + 64) = 1,344 dims'
    ].join('\n');

    var MLA_ABSORBED = [
        'absorbed attention (latent-compressed)',
        '',
        ' h[768] \u2500 W_D \u2500\u25b8 c [192] (+ rope 24) \u2500\u25b8 cached',
        ' c \u2500\u25b8 W_U \u00b7 W_q absorbed into q\u2032 \u2500\u25b8 attn \u25b8 out',
        '',
        ' KV cache / token : 192 + 24 = 216 dims',
        ' compression      : 1,344 \u25b8 216 (\u2248 6.2\u00d7)'
    ].join('\n');

    function initMlaToggle() {
        var box = document.getElementById('widget-mla-absorb');
        if (!box) return;
        var pre = box.querySelector('pre.mla-figure');
        if (!pre) return;
        var bar = document.createElement('div');
        bar.className = 'mla-toggle-bar';
        var bStd = document.createElement('button');
        bStd.type = 'button';
        bStd.textContent = '[ standard ]';
        var bAbs = document.createElement('button');
        bAbs.type = 'button';
        bAbs.textContent = '[ absorbed ]';
        bar.appendChild(bStd);
        bar.appendChild(bAbs);
        box.insertBefore(bar, pre);
        function show(absorbed) {
            pre.textContent = absorbed ? MLA_ABSORBED : MLA_STANDARD;
            bStd.setAttribute('aria-pressed', String(!absorbed));
            bAbs.setAttribute('aria-pressed', String(absorbed));
        }
        bStd.addEventListener('click', function () { show(false); });
        bAbs.addEventListener('click', function () { show(true); });
        show(true);
    }

    /* ---- reading progress strip --------------------------------------------- */

    function initProgressStrip() {
        if (!document.querySelector('.markdown-body')) return;
        var strip = document.createElement('div');
        strip.className = 'progress-strip';
        strip.setAttribute('aria-hidden', 'true');
        document.body.appendChild(strip);
        var heads = Array.prototype.slice.call(
            document.querySelectorAll('.markdown-body h2, .markdown-body h3'));
        var ticking = false;
        function update() {
            ticking = false;
            var doc = document.documentElement;
            var max = doc.scrollHeight - window.innerHeight;
            var p = max > 0 ? Math.min(1, Math.max(0, window.scrollY / max)) : 1;
            var n = Math.round(p * 10);
            var label = '';
            for (var i = 0; i < heads.length; i++) {
                if (heads[i].getBoundingClientRect().top <= 90) {
                    label = heads[i].textContent.replace(/#+\s*$/, '').trim();
                }
            }
            strip.textContent = '[' + '\u2593'.repeat(n) + '\u2591'.repeat(10 - n) +
                '] ' + Math.round(p * 100) + '%' + (label ? ' \u00b7 ' + label : '');
        }
        window.addEventListener('scroll', function () {
            if (!ticking) { ticking = true; requestAnimationFrame(update); }
        }, { passive: true });
        update();
    }

    /* ---- init ---------------------------------------------------------------- */

    function init() {
        runBoot();
        runHeroDecode();
        initExpandableCode();
        initLayerExplorer();
        initMoePlayground();
        initMlaToggle();
        initProgressStrip();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
```

- [ ] **Step 2: Verify it parses**

Run: `node --check assets/portal.js` (if node is unavailable, skip — browser pass in Task 5 covers it).
Expected: no output (valid syntax).

- [ ] **Step 3: Commit**

```bash
git add assets/portal.js
git commit -m "feat: portal.js runtime — boot overlay, hero decode, ASCII widgets"
```

---

### Task 4: Build-script wiring

**Files:**
- Modify: `scripts/build_docs_html.py`

- [ ] **Step 0: Extend the contract tests for dark-only output**

Append to `tests/test_build_docs_html.py`:

```python
def test_dark_theme_only(built):
    html = read("README.html")
    assert "toggleTheme" not in html
    assert "theme-toggle" not in html
    css = (OUT / "assets" / "style.css").read_text(encoding="utf-8")
    assert '[data-theme="light"]' not in css
```

Run: `python3 -m pytest tests/test_build_docs_html.py::test_dark_theme_only -q`
Expected: FAIL (toggle still present) — the correct TDD state.

- [ ] **Step 1: Add constants and the stack-art generator**

Insert after the `DOC_FILES` list (after the closing `]` on line ~56), before `def slugify`:

```python
# Premium-polish assets: mono-only font link, boot overlay, widget containers.
FONT_LINK = ('<link href="https://fonts.googleapis.com/css2?'
             'family=IBM+Plex+Mono:ital,wght@0,400;0,500;0,600;0,700;1,400'
             '&display=swap" rel="stylesheet">')

BOOT_OVERLAY_HTML = (
    '<div id="boot-overlay" aria-hidden="true">'
    '<div class="boot-inner">'
    '<div class="boot-wordmark">DEEPSEEK-V3-LITE</div>'
    '<div class="boot-line">loading weights '
    '<span class="boot-bar">[\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591] 0%</span>'
    '</div></div></div>'
)

# Marks html.booting before first paint; a watchdog ALWAYS clears it (and
# the overlay) so a portal.js failure can never strand the page — under
# reduced motion the removal fires immediately instead.
BOOT_SCRIPT = """<script>
(function () {
    var reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (!reduced) document.documentElement.classList.add('booting');
    document.addEventListener('DOMContentLoaded', function () {
        setTimeout(function () {
            document.documentElement.classList.remove('booting');
            var ov = document.getElementById('boot-overlay');
            if (ov && ov.parentNode) ov.parentNode.removeChild(ov);
        }, reduced ? 0 : 1200);
    });
})();
</script>"""


def layer_stack_ascii() -> str:
    """Static 18-layer stack art; portal.js rebuilds it interactively."""
    rows = ["\u250c" + "\u2500" * 36 + "\u2510"]
    for i in range(17, -1, -1):
        kind = "SwiGLU \u00b7 dense" if i < 2 else "MoE 20+1 \u00b7 top-4"
        inner = f" {i:02d} \u2502 MLA \u25b8 {kind}".ljust(36)
        rows.append("\u2502" + inner + "\u2502")
    rows.append("\u2514" + "\u2500" * 36 + "\u2518")
    return "\n".join(rows)


LAYER_STACK_WIDGET = f"""<div class="ascii-widget" id="widget-layer-stack">
    <div class="ascii-widget-head">FIG \u00b7 01 / 18-LAYER STACK <span>\u2014 hover or focus a layer</span></div>
    <div class="ascii-widget-grid">
        <pre class="ascii-stack" aria-label="18-layer stack: 2 dense layers, 16 MoE layers">{layer_stack_ascii()}</pre>
        <pre class="ascii-panel" aria-live="polite">hover / focus a layer \u25b8 spec readout</pre>
    </div>
</div>"""

MOE_WIDGET = """<div class="ascii-widget" id="widget-moe-routing">
    <div class="ascii-widget-head">FIG / GATE \u25b8 TOP-4 OF 20 <span>\u2014 press route token</span></div>
    <pre class="moe-grid" aria-label="Expert grid: 20 routed experts, 4 active per token, 1 shared always on">gate(x) \u25b8 top-4 of 20 routed
\u2591\u259100 \u2591\u259101 \u2593\u259302 \u2591\u259103 \u2591\u259104 
\u2591\u259105 \u2591\u259106 \u2593\u259307 \u2591\u259108 \u2591\u259109 
\u2591\u259110 \u2591\u259111 \u2591\u259112 \u2593\u259313 \u2591\u259114 
\u2591\u259115 \u2591\u259116 \u2591\u259117 \u2593\u259318 \u2591\u259119 
shared  \u2593\u2593sh (always on)</pre>
</div>"""

MLA_WIDGET = """<div class="ascii-widget" id="widget-mla-absorb">
    <div class="ascii-widget-head">FIG / MLA \u2014 STANDARD vs ABSORBED <span>\u2014 toggle the path</span></div>
    <pre class="mla-figure" aria-label="MLA absorption comparison: materialised KV versus latent-compressed path">absorbed attention (latent-compressed)

 h[768] \u2500 W_D \u2500\u25b8 c [192] (+ rope 24) \u2500\u25b8 cached
 c \u2500\u25b8 W_U \u00b7 W_q absorbed into q\u2032 \u2500\u25b8 attn \u25b8 out

 KV cache / token : 192 + 24 = 216 dims
 compression      : 1,344 \u25b8 216 (\u2248 6.2\u00d7)</pre>
</div>"""

# rel path -> widget injected between the article and the footer nav.
WIDGET_CONTAINERS = {
    "docs/concepts/moe-mtp.md": MOE_WIDGET,
    "docs/concepts/attention-and-precision.md": MLA_WIDGET,
}
```

- [ ] **Step 2: Wire `generate_html_page`**

In `generate_html_page`, four edits:

(a) Font link — replace:

```python
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Mono:ital,wght@0,400;0,500;0,600;1,400&display=swap" rel="stylesheet">
```

with `{FONT_LINK}` (the f-string interpolates the constant).

(a2) Dark-only — delete the header line
`<button class="theme-toggle" onclick="toggleTheme()" id="themeToggleBtn" aria-label="Toggle Theme">🌙 Dark</button>`
from the template, and delete the entire `// Theme Toggle` function block
plus the `const savedTheme = ...` block from the inline script (keep
copyCode, toggleSidebar, filterNav, and the hljs/KaTeX init).

(b) Boot script — replace:

```python
    <link rel="stylesheet" href="{rel_prefix}assets/style.css">
</head>
```

with:

```python
    <link rel="stylesheet" href="{rel_prefix}assets/style.css">
    {BOOT_SCRIPT}
</head>
```

(c) Overlay — replace `<body>` (the one in this template) with:

```python
<body>
    {BOOT_OVERLAY_HTML}
```

(d) Widget container — replace:

```python
                </article>

                <div class="doc-footer-nav">
```

with:

```python
                </article>

                {widget_html}

                <div class="doc-footer-nav">
```

and define `widget_html = WIDGET_CONTAINERS.get(rel_path, "")` next to the other locals (e.g. right after `toc_html = build_toc_html(toc_items)`).

(e) portal.js — replace the template's closing:

```python
    </script>
</body>
</html>
"""
```

(the end of the `page_html` f-string) with:

```python
    </script>
    <script defer src="{rel_prefix}assets/portal.js"></script>
</body>
</html>
"""
```

- [ ] **Step 3: Wire `generate_index_portal`**

In `generate_index_portal`, five edits:

(a) Font link — replace the same IBM Plex Serif+Mono `<link>` with `{FONT_LINK}`.

(a2) Dark-only — delete the index template's `<button class="theme-toggle" ...>`
header line and the entire `toggleTheme()` function plus `const savedTheme = ...`
block from its inline script (keep toggleSidebar, filterNav).

(b) Boot script + overlay — replace:

```python
    <link rel="stylesheet" href="assets/style.css">
</head>
<body>
```

with:

```python
    <link rel="stylesheet" href="assets/style.css">
    {BOOT_SCRIPT}
</head>
<body>
    {BOOT_OVERLAY_HTML}
```

(c) Hero — replace the `<h1 class="hero-title">…</h1>` line with:

```python
                    <h1 class="hero-title sr-only">DeepSeek-v3-Lite — documentation portal</h1>
```

(d) Hero figure — replace the ENTIRE `<div class="hero-figure" aria-hidden="true"> … </div>` block (SVG included) with:

```python
                    <div class="hero-figure">
                        <pre id="hero-decode" class="ascii-stage" role="img"
                             data-title="DEEPSEEK-V3-LITE"
                             data-sub="411.6M \u00b7 MLA \u00b7 MoE \u00b7 MTP \u00b7 \u03bcP"
                             aria-label="Animated ASCII wordmark decoding into DEEPSEEK-V3-LITE">  DEEPSEEK-V3-LITE

  411.6M \u00b7 MLA \u00b7 MoE \u00b7 MTP \u00b7 \u03bcP</pre>
                    </div>
```

(e) Layer explorer + portal.js — replace:

```python
                    </div>
                </div>

                <div class="portal-content">
```

(the end of `.spec-sheet` + `.hero-banner`, before `.portal-content`) with:

```python
                    </div>
                    {LAYER_STACK_WIDGET}
                </div>

                <div class="portal-content">
```

and replace the index template's final:

```python
    </script>
</body>
</html>
"""
```

with:

```python
    </script>
    <script defer src="assets/portal.js"></script>
</body>
</html>
"""
```

- [ ] **Step 4: Copy portal.js in `generate_css`**

In `generate_css`, replace:

```python
    src_css = WORKSPACE_DIR / "assets" / "style.css"
    shutil.copyfile(src_css, assets_dir / "style.css")
```

with:

```python
    src_css = WORKSPACE_DIR / "assets" / "style.css"
    shutil.copyfile(src_css, assets_dir / "style.css")
    src_js = WORKSPACE_DIR / "assets" / "portal.js"
    shutil.copyfile(src_js, assets_dir / "portal.js")
```

- [ ] **Step 4b: Delete hero CSS made dead by the ASCII hero swap**

In `assets/style.css`, delete these now-dead blocks (the SVG hero markup
they style was removed in Step 3):

- the `.hero-title-em` and `.hero-title-em-accent` rules (section 5, ~L607-614)
- the entire `.bottleneck-svg` rule group (~L634-685, including
  `.ruler line/text`, `.stream`, `.latent-axis`, `.axis`, `.node`, `.spark`)
- the `@keyframes spark-pulse` and `@keyframes flow` blocks
- any `.hero-figure` rule group that no longer has markup consumers
  (check `.hero-figure`, `.hero-figure .lbl`, `.hero-figure .lbl-key`);
  also remove `.hero-figure .lbl` / `.hero-figure .lbl-key` from the
  section-3 mono selector list if they appear there

Then re-check: `grep -n "bottleneck" assets/style.css` → no matches,
and braces still balance.

- [ ] **Step 5: Run the build and the contract tests**

Run: `python3 scripts/build_docs_html.py && python3 -m pytest tests/test_build_docs_html.py -q`
Expected: build prints "Documentation build complete!", all 10 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/build_docs_html.py assets/style.css
git commit -m "feat: wire boot overlay, ASCII hero, and widgets into docs build"
```

---

### Task 5: Regenerate, full verification, browser pass

**Files:**
- Regenerate: `docs_html/` (gitignored artifact)

- [ ] **Step 1: Full test suite + docs lint**

Run: `python3 -m pytest tests/ -q`
Expected: 198 existing results unchanged (189 pass + 10 GPU skips, minus any pre-existing variations) PLUS 9 new build tests passing.

Run: `python3 scripts/check_docs.py`
Expected: no errors (links + stale-pattern lint).

Run: `python3 -m pytest tests/test_doc_refs.py -q`
Expected: PASS (doc↔code anchors intact — Markdown untouched).

- [ ] **Step 2: Serve and manually verify in the browser**

Run: `python3 -m http.server 8811 --directory docs_html` (background), open `http://localhost:8811`. Check:
- Every page: boot overlay fills 0→100% then lifts (~0.7 s); click skips it.
- `index.html`: wordmark scramble-decodes, sub-line types, holds final frame; layer-stack explorer updates the spec panel on hover/focus; datasheet + coords intact.
- `docs/concepts/moe-mtp.html`: route-token button lights exactly 4 experts + shared, prints weights.
- `docs/concepts/attention-and-precision.html`: `[ standard ]` / `[ absorbed ]` toggle swaps diagrams.
- A long reference page (e.g. `docs/references/R2_transformer_api.html`): code blocks >16 lines collapse with the seam button; progress strip tracks scroll + section.
- No theme-toggle button anywhere; pages are dark-only.
- With OS "reduce motion" enabled: no boot overlay, hero renders final frame instantly.

- [ ] **Step 3: Stop the server and commit nothing further**

`docs_html/` is gitignored — verify with `git status` that only intended source files changed (Markdown files must be clean).

---

## Self-Review Notes (author)

- Spec §1 (theme/type + dark-only amendment) → Tasks 2 and 4; §2 (hero) → Tasks 3–4; §3 (boot overlay) → Tasks 2–4; §4.1–4.5 widgets → Tasks 2–4; §5 (JS arch) → Tasks 3–4; §6 (motion/a11y) → `REDUCED` guards in Task 3 + CSS media query in Task 2; §7 (verification) → Task 5.
- Consistency: `widget-layer-stack` / `widget-moe-routing` / `widget-mla-absorb` ids, `ascii-stack` / `ascii-panel` / `moe-grid` / `mla-figure` classes, `hero-decode`, `boot-overlay`, `.booting` are identical across CSS (Task 2), JS (Task 3), build script (Task 4), and tests (Task 1).
- Expansion uses CSS `max-height` clipping (not text slicing) so highlight.js spans survive collapsing.
- `WIDGET_CONTAINERS` keys are `.md` rel paths (matching `DOC_FILES`), not `.html`.
