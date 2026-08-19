#!/usr/bin/env python3
"""
DeepSeek-v3-Lite Documentation Generator
Converts project markdown files into a responsive, beautifully-styled HTML documentation portal
with full LaTeX math (KaTeX) and syntax highlighting support.
Output directory: docs_html/ (ignored by git).
"""

import os
import re
import html
import subprocess
from functools import lru_cache
from pathlib import Path

# Paths
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = WORKSPACE_DIR / "docs_html"

DOC_FILES = [
    # (relative_path_from_root, category, display_title)
    ("README.md", "Core", "Project Overview (README)"),
    ("AGENTS.md", "Core", "AGENTS & System Architecture"),
    ("SKILLS.md", "Core", "Skills Reference"),
    ("docs/README.md", "Core", "Documentation Index"),
    ("docs/training.md", "Core", "Training Architecture & Pipeline"),
    ("docs/inference.md", "Core", "Inference & Speculative Decoding"),
    
    # Concepts
    ("docs/concepts/foundations.md", "Concepts", "Foundations & Architecture"),
    ("docs/concepts/attention-and-precision.md", "Concepts", "MLA & Mixed Precision"),
    ("docs/concepts/moe-mtp.md", "Concepts", "DeepSeekMoE & MTP"),
    ("docs/concepts/parallelism.md", "Concepts", "DualPipe Parallelism"),
    ("docs/concepts/data-pipeline.md", "Concepts", "Data Pipeline"),
    ("docs/concepts/kernels-and-ops.md", "Concepts", "Operations & Triton Kernels"),
    
    # Guides
    ("docs/guides/getting-started.md", "Guides", "Getting Started"),
    ("docs/guides/G1_debugging_playbook.md", "Guides", "G1 — Debugging Playbook"),
    ("docs/guides/G2_mup_and_lr_tuning.md", "Guides", "G2 — μP & LR Tuning"),
    ("docs/guides/G3_triton_development.md", "Guides", "G3 — Triton Development"),
    ("docs/guides/G4_benchmarking.md", "Guides", "G4 — Benchmarking"),
    ("docs/guides/G5_checkpoint_ops.md", "Guides", "G5 — Checkpoint Ops"),
    ("docs/guides/contributing.md", "Guides", "Contributing Guide"),
    
    # References
    ("docs/references/R1_config_schema.md", "References", "R1 — Config Schema"),
    ("docs/references/R2_transformer_api.md", "References", "R2 — Transformer API"),
    ("docs/references/R3_mla_api.md", "References", "R3 — MLA API"),
    ("docs/references/R4_moe_api.md", "References", "R4 — MoE API"),
    ("docs/references/R5_mtp_api.md", "References", "R5 — MTP API"),
    ("docs/references/R6_triton_api.md", "References", "R6 — Triton API"),
    ("docs/references/R7_training_api.md", "References", "R7 — Training API"),
    ("docs/references/R8_utils_api.md", "References", "R8 — Utils API"),
    ("docs/references/R9_inference_api.md", "References", "R9 — Inference API"),
]

# Premium-polish assets: mono-only font link, boot overlay, widget containers.
FONT_LINK = ('<link href="https://fonts.googleapis.com/css2?'
             'family=IBM+Plex+Mono:ital,wght@0,400;0,500;0,600;0,700;1,400'
             '&family=JetBrains+Mono:ital,wght@0,400;0,500;0,600;0,700;1,400'
             '&display=swap" rel="stylesheet">')

BOOT_OVERLAY_HTML = (
    '<div id="boot-overlay" aria-hidden="true">'
    '<div class="boot-inner">'
    '<div class="boot-wordmark">DEEPSEEK-V3-LITE</div>'
    '<div class="boot-line">loading weights '
    '<span class="boot-bar">[░░░░░░░░░░░░] 0%</span>'
    '</div></div></div>'
)

BOOT_SCRIPT = """<script>
(function () {
    var reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (!reduced) document.documentElement.classList.add('booting');
    document.addEventListener('DOMContentLoaded', function () {
        setTimeout(function () {
            document.documentElement.classList.remove('booting');
            var ov = document.getElementById('boot-overlay');
            if (ov && ov.parentNode) ov.parentNode.removeChild(ov);
        }, reduced ? 0 : 350);
    });
})();
</script>"""

# Shared <head> for every generated page.
HEAD_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <!-- Fonts — one voice for the whole portal -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    {font_link}
{extra_head}
    <!-- CSS Stylesheet -->
    <link rel="stylesheet" href="{rel_prefix}assets/style.css">
    {boot_script}
</head>
"""

DOC_EXTRA_HEAD = """    <!-- Highlight.js for Syntax Highlighting -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <!-- KaTeX for LaTeX Math -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>"""


def layer_stack_ascii() -> str:
    """Compact 18-layer stack architectural overview with exact monospace alignment."""
    return """\u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510
\u2502 L02-L17 (x16) \u2502 MLA -> DeepSeekMoE 20+1 (top-4)  \u2502
\u2502 L00-L01 (x02) \u2502 MLA -> SwiGLU Dense (warmup)     \u2502
\u251c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2524
\u2502 TOTAL 18 L    \u2502 dim 768 · 12 heads · μP scaling  \u2502
\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518"""


LAYER_STACK_WIDGET = f"""<div class="ascii-widget" id="widget-layer-stack">
    <div class="ascii-widget-head">FIG \u00b7 01 / 18-LAYER STACK <span>\u2014 click or hover layer group</span></div>
    <div class="ascii-widget-grid">
        <pre class="ascii-stack" aria-label="18-layer stack: 2 dense layers, 16 MoE layers">{layer_stack_ascii()}</pre>
        <pre class="ascii-panel" aria-live="polite">STAGE       : Layers 02 - 17 (16 MoE blocks)
ATTENTION   : MLA (kv_lora_rank=192, qk_rope=24)
FFN ROUTING : DeepSeekMoE (20 routed top-4 + 1 shared)
INTER_DIM   : 384 / expert (active: 5 x 384 = 1,920)
ROUTING BIAS: Aux-loss-free gating with dynamic delta_b
SOURCE      : models/moe.py · configs/pretrain_a100_422m.yaml</pre>
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
 compression      : 1,536 \u25b8 216 (\u2248 7.1\u00d7)</pre>
</div>"""

# rel path -> widget injected between the article and the footer nav.
WIDGET_CONTAINERS = {
    "docs/concepts/moe-mtp.md": MOE_WIDGET,
    "docs/concepts/attention-and-precision.md": MLA_WIDGET,
}


def slugify(text: str) -> str:
    """Generate clean HTML id for headings."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'\s', '-', text)
    return text.strip('-') or "heading"


@lru_cache(maxsize=1)
def github_base_url() -> str:
    """Derive the GitHub blob base (https://github.com/<owner>/<repo>/blob/<branch>)."""
    try:
        out = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True, text=True, check=True, cwd=WORKSPACE_DIR,
        ).stdout.strip()
        out = out.replace("git@github.com:", "https://github.com/").removesuffix(".git")
        if not out.startswith("https://github.com/"):
            return ""
    except Exception:
        return ""
    try:
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True, text=True, check=True, cwd=WORKSPACE_DIR,
        ).stdout.strip()
    except Exception:
        return ""
    return f"{out}/blob/{branch}" if branch else ""


def fix_md_links(content: str, src_rel_path: str) -> str:
    """Rewrite relative markdown links for the HTML build."""
    repo_base = github_base_url()
    src_dir = WORKSPACE_DIR / Path(src_rel_path).parent

    def link_replacer(match):
        label = match.group(1)
        url = match.group(2)
        if url.startswith(("http://", "https://", "mailto:", "#")):
            return f"[{label}]({url})"
        path_part, _, anchor = url.partition("#")
        if path_part.endswith(".md"):
            target = path_part[:-3] + ".html"
            if anchor:
                target += "#" + anchor
            return f"[{label}]({target})"
        if repo_base and not path_part.startswith("/"):
            repo_rel = (src_dir / path_part).resolve().relative_to(WORKSPACE_DIR)
            return f"[{label}]({repo_base}/{repo_rel})"
        return f"[{label}]({url})"

    return re.sub(r'\[([^\]]+)\]\(([^)]+)\)', link_replacer, content)


def parse_markdown_to_html(md_text: str, src_rel_path: str) -> tuple[str, list[dict]]:
    """
    Statically convert markdown to rich HTML structure with full LaTeX & Math protection.
    Returns (html_content, toc_items).
    """
    md_text = fix_md_links(md_text, src_rel_path)
    
    # -------------------------------------------------------------
    # STEP 1: Protect Code Blocks & Inline Code
    # -------------------------------------------------------------
    code_blocks = []
    def store_code_block(m):
        code_blocks.append(m.group(0))
        return f"\n\n___CODEBLOCK_{len(code_blocks)-1}___\n\n"
    
    md_text = re.sub(r'```[\s\S]*?```', store_code_block, md_text)
    
    inline_codes = []
    def store_inline_code(m):
        inline_codes.append(m.group(0))
        return f"___INLINECODE_{len(inline_codes)-1}___"
    
    md_text = re.sub(r'`[^`\n]+`', store_inline_code, md_text)

    # -------------------------------------------------------------
    # STEP 2: Protect LaTeX Math Blocks & Inline Math
    # -------------------------------------------------------------
    display_maths = []
    def store_display_math(m):
        inner = m.group(1).strip()
        safe_math = html.escape(inner, quote=False)
        display_maths.append(f'<div class="math-block">$$\n{safe_math}\n$$</div>')
        return f"\n\n___DISPLAYMATH_{len(display_maths)-1}___\n\n"
    
    # Match $$ ... $$ (display math)
    md_text = re.sub(r'\$\$([\s\S]+?)\$\$', store_display_math, md_text)
    # Also match \[ ... \]
    md_text = re.sub(r'\\\[([\s\S]+?)\\\]', store_display_math, md_text)
    
    inline_maths = []
    def store_inline_math(m):
        inner = m.group(1).strip()
        safe_math = html.escape(inner, quote=False)
        inline_maths.append(f'<span class="math-inline">${safe_math}$</span>')
        return f"___INLINEMATH_{len(inline_maths)-1}___"
    
    # Match $ ... $ (inline math)
    md_text = re.sub(r'(?<!\$)\$([^\$\n]+?)\$(?!\$)', store_inline_math, md_text)
    # Also match \( ... \)
    md_text = re.sub(r'\\\(([\s\S]+?)\\\)', store_inline_math, md_text)

    # -------------------------------------------------------------
    # STEP 3: Parse Document Structure Line by Line
    # -------------------------------------------------------------
    toc = []
    lines = md_text.splitlines()
    html_lines = []
    
    in_table = False
    table_headers = []
    table_rows = []
    
    list_stack = []
    h1_seen = False

    in_blockquote = False
    blockquote_type = "normal"
    blockquote_lines = []

    def close_li():
        nonlocal list_stack
        if list_stack and list_stack[-1]['li_open']:
            html_lines.append("</li>")
            list_stack[-1]['li_open'] = False

    def flush_list():
        nonlocal list_stack
        while list_stack:
            close_li()
            html_lines.append(f"</{list_stack[-1]['tag']}>")
            list_stack.pop()

    def flush_blockquote():
        nonlocal in_blockquote, blockquote_type, blockquote_lines
        if in_blockquote:
            content = "<br>".join(blockquote_lines)
            if blockquote_type != "normal":
                title = blockquote_type.upper()
                icon = {"NOTE": "ℹ️", "TIP": "💡", "IMPORTANT": "📌", "WARNING": "⚠️", "CAUTION": "🚨"}.get(title, "ℹ️")
                html_lines.append(
                    f'<div class="callout callout-{blockquote_type.lower()}">'
                    f'<div class="callout-header"><span class="callout-icon">{icon}</span><span class="callout-title">{title}</span></div>'
                    f'<div class="callout-body">{content}</div>'
                    f'</div>'
                )
            else:
                html_lines.append(f'<blockquote>{content}</blockquote>')
            in_blockquote = False
            blockquote_type = "normal"
            blockquote_lines = []

    def flush_table():
        nonlocal in_table, table_headers, table_rows
        if in_table:
            th_html = "".join(f"<th>{h}</th>" for h in table_headers)
            tr_html = ""
            for row in table_rows:
                td_html = "".join(f"<td>{c}</td>" for c in row)
                tr_html += f"<tr>{td_html}</tr>"
            html_lines.append(
                f'<div class="table-container"><table class="doc-table">'
                f'<thead><tr>{th_html}</tr></thead>'
                f'<tbody>{tr_html}</tbody>'
                f'</table></div>'
            )
            in_table = False
            table_headers = []
            table_rows = []

    def escape_preserving_entities(text: str) -> str:
        text = re.sub(r'&(?!(?:[a-zA-Z][a-zA-Z0-9]*|#[0-9]+|#x[0-9a-fA-F]+);)', '&amp;', text)
        return text.replace('<', '&lt;').replace('>', '&gt;')

    def render_inline_formatting(text: str) -> str:
        text = escape_preserving_entities(text)
        text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', text)
        text = re.sub(r'~~([^~]+)~~', r'<del>\1</del>', text)
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" class="doc-link">\1</a>', text)
        return text

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("___DISPLAYMATH_"):
            flush_table()
            flush_list()
            flush_blockquote()
            html_lines.append(stripped)
            i += 1
            continue

        if stripped.startswith("___CODEBLOCK_"):
            flush_table()
            flush_list()
            flush_blockquote()
            html_lines.append(stripped)
            i += 1
            continue

        if not stripped:
            flush_table()
            flush_list()
            flush_blockquote()
            i += 1
            continue

        if stripped.startswith(">"):
            flush_table()
            flush_list()
            bq_content = stripped.lstrip(">").strip()
            
            callout_match = re.match(r'^\[\!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]', bq_content, re.IGNORECASE)
            if callout_match:
                in_blockquote = True
                blockquote_type = callout_match.group(1).upper()
                remaining = bq_content[callout_match.end():].strip()
                if remaining:
                    blockquote_lines.append(render_inline_formatting(remaining))
            else:
                if not in_blockquote:
                    in_blockquote = True
                    blockquote_type = "normal"
                if bq_content:
                    blockquote_lines.append(render_inline_formatting(bq_content))
            i += 1
            continue

        if re.match(r'^(---|\*\*\*|___)\s*$', stripped):
            flush_table()
            flush_list()
            flush_blockquote()
            html_lines.append("<hr class='doc-hr'>")
            i += 1
            continue

        heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
        if heading_match:
            flush_table()
            flush_list()
            flush_blockquote()
            level = len(heading_match.group(1))
            heading_text_raw = heading_match.group(2).strip()
            
            clean_title = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', heading_text_raw)
            clean_title = re.sub(r'`([^`]+)`', r'\1', clean_title)
            clean_title = re.sub(r'___INLINECODE_\d+___', '', clean_title)
            clean_title = re.sub(r'___INLINEMATH_\d+___', '', clean_title)
            heading_id = slugify(clean_title)
            
            rendered_heading = render_inline_formatting(heading_text_raw)
            
            if level in (2, 3):
                toc.append({
                    'level': level,
                    'title': clean_title,
                    'id': heading_id
                })

            if level == 1 and not h1_seen:
                h1_seen = True
                html_lines.append(f'<span class="doc-anchor" id="{heading_id}"></span>')
            else:
                html_lines.append(
                    f'<h{level} id="{heading_id}" class="heading-anchor">'
                    f'{rendered_heading}'
                    f'<a href="#{heading_id}" class="anchor-link" aria-label="Link to section">#</a>'
                    f'</h{level}>'
                )
            i += 1
            continue

        if "|" in line and i + 1 < len(lines) and re.match(r'^\s*\|?\s*:?---', lines[i + 1].strip()):
            flush_list()
            flush_blockquote()
            in_table = True
            headers_raw = [c.strip() for c in line.strip().strip("|").split("|")]
            table_headers = [render_inline_formatting(h) for h in headers_raw]
            i += 2
            
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                cells_raw = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                table_rows.append([render_inline_formatting(c) for c in cells_raw])
                i += 1
            flush_table()
            continue

        ul_match = re.match(r'^[\*\-]\s+(.+)$', stripped)
        ol_match = re.match(r'^\d+\.\s+(.+)$', stripped)
        if ul_match or ol_match:
            flush_table()
            flush_blockquote()
            tag = 'ul' if ul_match else 'ol'
            item_text = (ul_match or ol_match).group(1).strip()
            indent = len(line) - len(line.lstrip(' '))

            while list_stack and indent < list_stack[-1]['indent']:
                close_li()
                html_lines.append(f"</{list_stack[-1]['tag']}>")
                list_stack.pop()
            if list_stack and list_stack[-1]['indent'] == indent and list_stack[-1]['tag'] != tag:
                close_li()
                html_lines.append(f"</{list_stack[-1]['tag']}>")
                list_stack.pop()
            if not list_stack or list_stack[-1]['indent'] != indent:
                list_stack.append({'indent': indent, 'tag': tag, 'li_open': False})
                html_lines.append(f'<{tag} class="doc-list">')
            else:
                close_li()

            task_match = re.match(r'^\[([ xX])\]\s+(.+)$', item_text)
            if task_match:
                checked = 'checked' if task_match.group(1).lower() == 'x' else ''
                item_content = render_inline_formatting(task_match.group(2))
                html_lines.append(f'<li class="task-item"><input type="checkbox" disabled {checked}> {item_content}')
            else:
                html_lines.append(f'<li>{render_inline_formatting(item_text)}')
            list_stack[-1]['li_open'] = True
            i += 1
            continue

        if list_stack and list_stack[-1]['li_open'] and line[:1] in (' ', '\t'):
            html_lines.append("<br> " + render_inline_formatting(stripped))
            i += 1
            continue

        flush_table()
        flush_list()
        flush_blockquote()
        html_lines.append(f'<p>{render_inline_formatting(stripped)}</p>')
        i += 1

    flush_table()
    flush_list()
    flush_blockquote()

    full_html = "\n".join(html_lines)

    # -------------------------------------------------------------
    # STEP 4: Restore Protected Tokens
    # -------------------------------------------------------------
    for idx, math_html in enumerate(inline_maths):
        full_html = full_html.replace(f"___INLINEMATH_{idx}___", math_html)

    for idx, raw_code in enumerate(inline_codes):
        code_content = raw_code[1:-1]
        code_html = f'<code class="inline-code">{html.escape(code_content)}</code>'
        full_html = full_html.replace(f"___INLINECODE_{idx}___", code_html)

    for idx, math_html in enumerate(display_maths):
        full_html = full_html.replace(f"___DISPLAYMATH_{idx}___", math_html)

    for idx, raw_block in enumerate(code_blocks):
        lines_b = raw_block.splitlines()
        first_line = lines_b[0].strip()
        code_lang = first_line.lstrip("```").strip().lower()
        code_content = "\n".join(lines_b[1:-1])
        escaped_content = html.escape(code_content)
        
        lang_attr = f' class="language-{code_lang}"' if code_lang else ''
        data_lang = code_lang if code_lang else 'code'

        n_lines = len(lines_b) - 2
        collapsed = n_lines > 14
        wrapper_cls = 'code-wrapper collapsed' if collapsed else 'code-wrapper'
        expand_btn = (
            f'<button class="expand-btn" data-label="expand ▾ · {n_lines} lines" '
            f'onclick="toggleCode(this)">expand ▾ · {n_lines} lines</button>'
            if collapsed else ''
        )

        block_html = (
            f'<div class="{wrapper_cls}" data-lines="{n_lines}">'
            f'<div class="code-header">'
            f'<span class="code-lang">{data_lang}</span>'
            f'<span class="code-actions">{expand_btn}'
            f'<button class="copy-btn" onclick="copyCode(this)">Copy</button></span>'
            f'</div>'
            f'<pre><code{lang_attr}>{escaped_content}</code></pre>'
            f'</div>'
        )
        full_html = full_html.replace(f"___CODEBLOCK_{idx}___", block_html)

    return full_html, toc


def compute_rel_prefix(target_rel_path: str) -> str:
    """Calculate relative path back to root docs_html directory."""
    parts = Path(target_rel_path).parts
    if len(parts) <= 1:
        return "./"
    return "../" * (len(parts) - 1)


def build_sidebar_html(current_rel_path: str, rel_prefix: str) -> str:
    """Build the navigation sidebar HTML."""
    sidebar_sections = {
        "Core": [],
        "Concepts": [],
        "Guides": [],
        "References": []
    }
    
    for rel_path, category, display_title in DOC_FILES:
        target_html_rel = rel_path.replace(".md", ".html")
        href = rel_prefix + target_html_rel
        is_active = (rel_path == current_rel_path)
        active_cls = "active" if is_active else ""
        sidebar_sections[category].append(
            f'<li class="nav-item"><a href="{href}" class="nav-link {active_cls}">{display_title}</a></li>'
        )
        
    html_out = ['<div class="sidebar-search"><input type="text" id="navSearch" placeholder="Search docs..." onkeyup="filterNav()"></div>']
    
    for cat_name, items in sidebar_sections.items():
        if items:
            html_out.append(f'<div class="nav-group">')
            html_out.append(f'<div class="nav-group-title">{cat_name}</div>')
            html_out.append(f'<ul class="nav-list">{"".join(items)}</ul>')
            html_out.append(f'</div>')
            
    return "\n".join(html_out)


def build_toc_html(toc_items: list[dict]) -> str:
    """Build the right sidebar table of contents."""
    if not toc_items:
        return '<div class="toc-empty">No section headings</div>'
    
    toc_links = []
    for item in toc_items:
        indent_cls = "toc-h3" if item['level'] == 3 else "toc-h2"
        toc_links.append(f'<li class="{indent_cls}"><a href="#{item["id"]}" class="toc-link">{item["title"]}</a></li>')
        
    return f'<ul class="toc-list">{"".join(toc_links)}</ul>'


def generate_html_page(rel_path: str, category: str, display_title: str):
    """Generate single HTML file for a markdown document."""
    src_file = WORKSPACE_DIR / rel_path
    if not src_file.exists():
        print(f"Warning: {src_file} does not exist, skipping.")
        return

    md_text = src_file.read_text(encoding="utf-8")
    
    word_count = len(md_text.split())
    reading_time = max(1, round(word_count / 200))
    
    html_body, toc_items = parse_markdown_to_html(md_text, rel_path)
    
    rel_prefix = compute_rel_prefix(rel_path)
    sidebar_html = build_sidebar_html(rel_path, rel_prefix)
    toc_html = build_toc_html(toc_items)
    widget_html = WIDGET_CONTAINERS.get(rel_path, "")
    
    current_idx = next((i for i, df in enumerate(DOC_FILES) if df[0] == rel_path), 0)
    prev_doc = DOC_FILES[current_idx - 1] if current_idx > 0 else None
    next_doc = DOC_FILES[current_idx + 1] if current_idx < len(DOC_FILES) - 1 else None
    
    prev_html = ""
    if prev_doc:
        prev_href = rel_prefix + prev_doc[0].replace(".md", ".html")
        prev_html = f'<a href="{prev_href}" class="nav-card prev-card"><span class="card-label">← Previous</span><span class="card-title">{prev_doc[2]}</span></a>'
        
    next_html = ""
    if next_doc:
        next_href = rel_prefix + next_doc[0].replace(".md", ".html")
        next_html = f'<a href="{next_href}" class="nav-card next-card"><span class="card-label">Next →</span><span class="card-title">{next_doc[2]}</span></a>'
        
    page_html = HEAD_TEMPLATE.format(
        rel_prefix=rel_prefix,
        title=f"{display_title} | DeepSeek-v3-Lite Documentation",
        extra_head=DOC_EXTRA_HEAD,
        font_link=FONT_LINK,
        boot_script=BOOT_SCRIPT,
    ) + f"""<body>
    {BOOT_OVERLAY_HTML}
    <!-- Top Header -->
    <header class="site-header">
        <div class="header-left">
            <button class="mobile-toggle" onclick="toggleSidebar()" aria-label="Toggle Sidebar">☰</button>
            <a href="{rel_prefix}index.html" class="brand-logo">
                <span class="brand-name">DeepSeek-v3-Lite</span>
                <span class="brand-badge">Docs</span>
            </a>
        </div>
        <div class="header-right">
            <a href="{rel_prefix}index.html" class="header-link">Portal Home</a>
            <a href="{rel_prefix}README.html" class="header-link">GitHub README</a>
        </div>
    </header>

    <div class="app-layout">
        <!-- Left Sidebar Navigation -->
        <aside class="sidebar" id="sidebar">
            <div class="sidebar-inner">
                {sidebar_html}
            </div>
        </aside>

        <!-- Main Content Area -->
        <main class="main-content">
            <div class="content-container">
                <div class="breadcrumb">
                    <a href="{rel_prefix}index.html">Docs</a> &gt; <span>{category}</span> &gt; <span class="current">{display_title}</span>
                </div>
                
                <div class="doc-header">
                    <h1 class="doc-title">{display_title}</h1>
                    <div class="doc-meta">
                        <span class="meta-item"><span class="meta-label">source</span><span class="meta-val">{rel_path}</span></span>
                        <span class="meta-item"><span class="meta-label">words</span><span class="meta-val">{word_count:,}</span></span>
                        <span class="meta-item"><span class="meta-label">read</span><span class="meta-val">~{reading_time} min</span></span>
                    </div>
                </div>

                <article class="markdown-body" id="articleBody">
                    {html_body}
                </article>

                {widget_html}

                <div class="doc-footer-nav">
                    {prev_html}
                    {next_html}
                </div>
            </div>
        </main>

        <!-- Right Sidebar Table of Contents -->
        <aside class="toc-sidebar">
            <div class="toc-inner">
                <div class="toc-title">On This Page</div>
                {toc_html}
            </div>
        </aside>
    </div>

    <!-- Scripts -->
    <script>
        document.addEventListener("DOMContentLoaded", function() {{
            if (window.hljs) {{
                hljs.highlightAll();
            }}
            if (window.renderMathInElement) {{
                renderMathInElement(document.body, {{
                    delimiters: [
                        {{left: '$$', right: '$$', display: true}},
                        {{left: '\\\\[', right: '\\\\]', display: true}},
                        {{left: '\\\\(', right: '\\\\)', display: false}},
                        {{left: '$', right: '$', display: false}}
                    ],
                    ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
                    throwOnError: false
                }});
            }}
        }});
    </script>
    <script defer src="{rel_prefix}assets/portal.js"></script>
</body>
</html>
"""

    out_file = OUTPUT_DIR / rel_path.replace(".md", ".html")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(page_html, encoding="utf-8")


def generate_index_portal():
    """Generate interactive index.html home portal."""
    sidebar_html = build_sidebar_html("index.html", "./")
    
    categories = {
        ("CORE", "Core Architecture"): [
            ("README.html", "README", "Project Overview", "Chinchilla-scale 412M parameter PyTorch implementation of DeepSeek-V3."),
            ("AGENTS.html", "AGENTS", "System Architecture", "Codebase contracts, Triton rules, μP scaling & architectural limits."),
            ("SKILLS.html", "SKILLS", "Skills Map", "Specialized agent tools, scripts, and domain competencies."),
            ("docs/README.html", "DOCS", "Documentation Index", "A map of the concepts, guides, and API references in this portal."),
            ("docs/training.html", "CORE", "Training Pipeline", "BF16 training loop, loss scaling, checkpointing & NaN safety."),
            ("docs/inference.html", "CORE", "Inference & Speculative", "MTP speculative decoder, KV caching & latency optimizations.")
        ],
        ("CONCEPTS", "Architecture & Concepts"): [
            ("docs/concepts/foundations.html", "C1", "Foundations & Topography", "DeepSeek lineage, 18-layer layout, parameter & memory budgets."),
            ("docs/concepts/attention-and-precision.html", "C2", "MLA & Mixed Precision", "Multi-Head Latent Attention, matrix absorption, FP8 scheme."),
            ("docs/concepts/moe-mtp.html", "C3", "DeepSeekMoE & MTP", "Auxiliary-loss-free load balancing, 20 routed + 1 shared expert, MTP."),
            ("docs/concepts/parallelism.html", "C4", "DualPipe Parallelism", "Bidirectional pipeline scheduling, overlap, and distributed execution."),
            ("docs/concepts/data-pipeline.html", "C5", "Data Pipeline", "Pretraining data stages, validation, packing, and reproducibility."),
            ("docs/concepts/kernels-and-ops.html", "C6", "Operations & Triton", "Triton grouped GEMM, MLA fused kernel, CI testing & invariants.")
        ],
        ("GUIDES", "Guides & Playbooks"): [
            ("docs/guides/getting-started.html", "G0", "Getting Started", "Quickstart installation, synthetic pretraining & smoke testing."),
            ("docs/guides/G1_debugging_playbook.html", "G1", "Debugging Playbook", "Troubleshooting NaNs, CUDA OOMs, and loss divergence."),
            ("docs/guides/G2_mup_and_lr_tuning.html", "G2", "μP & LR Tuning", "Maximal Update Parameterization setup & hyperparameter transfer."),
            ("docs/guides/G3_triton_development.html", "G3", "Triton Development", "Writing, profiling, and benchmarking custom Triton GPU kernels."),
            ("docs/guides/G4_benchmarking.html", "G4", "Benchmarking", "Repeatable performance measurement and result interpretation."),
            ("docs/guides/G5_checkpoint_ops.html", "G5", "Checkpoint Operations", "Save, resume, inspect, and safely manage training state."),
            ("docs/guides/contributing.html", "G6", "Contributing", "Documentation conventions, development workflow, and project contribution."),
        ],
        ("REFS", "API References"): [
            ("docs/references/R1_config_schema.html", "R1", "Config Schema", "Complete YAML specification for model & training configs."),
            ("docs/references/R2_transformer_api.html", "R2", "Transformer API", "Model initialization, forward pass, and weight wiring."),
            ("docs/references/R3_mla_api.html", "R3", "MLA API", "Multi-Head Latent Attention layer implementation & options."),
            ("docs/references/R4_moe_api.html", "R4", "MoE API", "AuxLossFreeGate and DeepSeekMoE expert routing engine."),
            ("docs/references/R5_mtp_api.html", "R5", "MTP API", "Multi-token prediction heads and speculative decoding interfaces."),
            ("docs/references/R6_triton_api.html", "R6", "Triton API", "Kernel contracts, fused operators, and development seams."),
            ("docs/references/R7_training_api.html", "R7", "Training API", "Trainer entry points, loss plumbing, and checkpoint controls."),
            ("docs/references/R8_utils_api.html", "R8", "Utilities API", "Shared helpers, diagnostics, and common utilities."),
            ("docs/references/R9_inference_api.html", "R9", "Inference API", "Generation, caches, and inference-time interfaces.")
        ]
    }

    portal_cards_html = ""
    for (cat_tag, cat_title), items in categories.items():
        cards = ""
        for href, tag, title, desc in items:
            cards += f"""
            <a href="{href}" class="portal-card">
                <span class="card-tag">{tag}</span>
                <div class="card-body">
                    <h3 class="card-heading">{title}</h3>
                    <p class="card-desc">{desc}</p>
                </div>
            </a>
            """
        portal_cards_html += f"""
        <section class="portal-section">
            <header class="portal-section-head">
                <span class="portal-section-mark">§ {cat_tag.lower()}</span>
                <h2 class="portal-section-title">{cat_title}</h2>
                <span class="portal-section-meta">{len(items)} entries</span>
            </header>
            <div class="portal-grid">{cards}</div>
        </section>
        """

    index_html = HEAD_TEMPLATE.format(
        title="DeepSeek-v3-Lite Documentation Portal",
        extra_head="",
        rel_prefix="./",
        font_link=FONT_LINK,
        boot_script=BOOT_SCRIPT,
    ) + f"""<body class="index-portal">
    {BOOT_OVERLAY_HTML}
    <!-- Top Header -->
    <header class="site-header">
        <div class="header-left">
            <button class="mobile-toggle" onclick="toggleSidebar()" aria-label="Toggle Sidebar">☰</button>
            <a href="index.html" class="brand-logo">
                <span class="brand-name">DeepSeek-v3-Lite</span>
                <span class="brand-badge">Documentation</span>
            </a>
        </div>
        <div class="header-right">
            <a href="README.html" class="header-link">GitHub README</a>
        </div>
    </header>

    <div class="app-layout">
        <!-- Left Sidebar Navigation -->
        <aside class="sidebar" id="sidebar">
            <div class="sidebar-inner">
                {sidebar_html}
            </div>
        </aside>

        <!-- Main Portal Content -->
        <main class="main-content">
            <div class="content-container">
                <div class="hero-banner">
                    <div class="hero-margin-ticks" aria-hidden="true"></div>

                    <div class="hero-coords" aria-hidden="true">
                        <span class="coord">FIG · A0</span>
                        <span class="coord-sep">/</span>
                        <span class="coord">PARAM 411.6M</span>
                        <span class="coord-sep">/</span>
                        <span class="coord">MLA 192·24</span>
                        <span class="coord-sep">/</span>
                        <span class="coord">MoE 20+1 TOP-4</span>
                        <span class="coord-sep">/</span>
                        <span class="coord">DEPTH 18</span>
                        <span class="coord-sep">/</span>
                        <span class="coord">MTP 1</span>
                        <span class="coord-sep">/</span>
                        <span class="coord">PREC BF16</span>
                    </div>
                    <h1 class="hero-title">DeepSeek<span class="hero-title-em">-v3</span><span class="hero-title-em-accent">-Lite</span><span class="hero-title sr-only"> — documentation portal</span></h1>
                    <p class="hero-subtitle">412M Chinchilla-scale PyTorch reproduction of DeepSeek-V3 &mdash; Multi-Head Latent Attention (MLA), auxiliary-loss-free DeepSeekMoE (20+1 experts), Multi-Token Prediction (MTP), and &mu;P width scaling.</p>

                    <div class="deepseek-telemetry-ribbon" aria-label="Key DeepSeek Architectural Metrics">
                        <div class="telemetry-card terra">
                            <div class="tc-badge"><span class="tc-badge-dot"></span> MLA COMPRESSION</div>
                            <div class="tc-val">7.1&times; <span class="unit">KV CUT</span></div>
                            <div class="tc-desc">1,536d &rarr; 216d per token (192d latent + 24d decoupled RoPE)</div>
                        </div>
                        <div class="telemetry-card olive">
                            <div class="tc-badge"><span class="tc-badge-dot"></span> ACTIVE COMPUTE</div>
                            <div class="tc-val">23.8% <span class="unit">FLOPs</span></div>
                            <div class="tc-desc">5 of 21 experts active (4 routed + 1 shared &times; 384d = 1,920d)</div>
                        </div>
                        <div class="telemetry-card gold">
                            <div class="tc-badge"><span class="tc-badge-dot"></span> AUXILIARY LOSS</div>
                            <div class="tc-val">0.000 <span class="unit">&alpha; = 0</span></div>
                            <div class="tc-desc">Zero loss penalty &middot; Out-of-band dynamic bias leveling &Delta;b</div>
                        </div>
                        <div class="telemetry-card ink">
                            <div class="tc-badge"><span class="tc-badge-dot"></span> MTP SPEEDUP</div>
                            <div class="tc-val">~1.8&times; <span class="unit">TOKENS/s</span></div>
                            <div class="tc-desc">Multi-Token Prediction depth-1 self-speculative draft decoding</div>
                        </div>
                    </div>

                    <div class="hero-figure" id="hero-decode" data-title="DEEPSEEK-V3-LITE" data-sub="411.6M &middot; MLA 192+24 &middot; MoE 20+1 &middot; MTP &middot; &mu;P" role="region" aria-label="Figure A0: Multi-Head Latent Attention Manifold &amp; DeepSeekMoE 20+1 Dynamic Biased Routing Field">
                        <div class="hero-figure-header">
                            <div class="fig-badge">
                                <span class="fig-dot" aria-hidden="true"></span>
                                <span class="fig-tag">FIG. A0</span>
                                <span class="fig-sep" aria-hidden="true">/</span>
                                <span class="fig-title">MLA MANIFOLD &middot; DEEPSEEKMOE 20+1</span>
                                <span class="fig-dim">12 HEADS &middot; 192d LATENT &middot; 7.1&times; KV CUT</span>
                            </div>
                            <div class="fig-controls">
                                <button class="fig-btn active" data-mode="mla" type="button" title="Multi-Head Latent Attention (192d KV + 24d RoPE)">MLA &middot; LATENT</button>
                                <button class="fig-btn" data-mode="moe" type="button" title="DeepSeekMoE 20+1 Biased-Sigmoid Routing">MoE &middot; 20+1 ROUTER</button>
                                <button class="fig-btn" data-mode="mtp" type="button" title="Multi-Token Prediction (Depth-1 Speculative Tree)">MTP &middot; DRAFT</button>
                                <button class="fig-btn" data-mode="dualpipe" type="button" title="DualPipe 1F1B Bidirectional Overlap">DUALPIPE &middot; FLOW</button>
                                <button class="fig-btn fig-btn-icon" id="heroSpeedBtn" type="button" title="Simulation Speed" aria-label="Speed: 1x">1&times;</button>
                                <button class="fig-btn fig-btn-icon" id="heroPauseBtn" type="button" title="Pause / Resume" aria-label="Pause simulation">&#10074;&#10074;</button>
                            </div>
                        </div>

                        <div class="fig-canvas-wrap">
                            <canvas id="heroStateCanvas" class="hero-canvas" aria-hidden="true"></canvas>
                            <div class="fig-overlay-hud" aria-hidden="true">
                                <div class="hud-corner top-left">
                                    <span class="hud-lbl">SIMULATION MODE</span>
                                    <span class="hud-val" id="hudModeLabel">MLA &middot; LATENT PROJECTION</span>
                                </div>
                                <div class="hud-corner top-right">
                                    <span class="hud-lbl">KV CACHE REDUCTION</span>
                                    <span class="hud-val" id="hudKVCut">7.1&times; <span class="unit">(1536 &rarr; 216 dims)</span></span>
                                </div>
                                <div class="hud-corner bottom-left">
                                    <span class="hud-lbl">EXPERT DISPATCH</span>
                                    <span class="hud-val" id="hudMoEStatus">20 ROUTED <span class="num">(TOP-4)</span> + 1 SHARED</span>
                                </div>
                                <div class="hud-corner bottom-right">
                                    <span class="hud-lbl">DUALPIPE OVERLAP</span>
                                    <span class="hud-val" id="hudOverlap">100% <span class="unit">COMM / COMP OVERLAP</span></span>
                                </div>
                                <div class="hud-probe" id="heroProbeHUD" style="opacity: 0;">
                                    <div class="probe-card">
                                        <span class="probe-tag" id="probeTag">PROBE &middot; HEAD #04</span>
                                        <span class="probe-val" id="probeCoords">q'_t = W_q h_t &middot; W_U [192d]</span>
                                        <span class="probe-sub" id="probeDecay">Decoupled RoPE k_t [24d] &middot; &theta;=10K</span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div class="hero-figure-footer">
                            <div class="fig-formula" id="heroFormulaBar">
                                <span class="formula-sym">q'<sub class="f-sub">t,i</sub></span>
                                <span class="formula-op">=</span>
                                <span class="formula-term term-a">W<sub class="f-sub">q</sub> h<sub class="f-sub">t</sub> &middot; W<sub class="f-sub">U</sub></span>
                                <span class="formula-dot">&middot;</span>
                                <span class="formula-sym">score</span>
                                <span class="formula-op">=</span>
                                <span class="formula-term term-b">q'<sub class="f-sub">t,i</sub> c<sub class="f-sub">j</sub><sup class="f-sub">KV&top;</sup> + q<sub class="f-sub">t,i</sub><sup class="f-sub">R</sup> k<sub class="f-sub">j</sub><sup class="f-sub">R&top;</sup></span>
                            </div>
                            <div class="fig-legend">
                                <span class="legend-item"><span class="legend-swatch terra"></span> <span class="legend-label">192d Compressed Latent c<sub class="f-sub">t</sub></span></span>
                                <span class="legend-item"><span class="legend-swatch olive"></span> <span class="legend-label">24d Decoupled RoPE k<sub class="f-sub">t</sub></span></span>
                                <span class="legend-item"><span class="legend-swatch core"></span> <span class="legend-label">Top-4 Routed + 1 Shared Expert</span></span>
                                <span class="legend-hint">CLICK: SHOCKWAVE &middot; HOVER: PROBE HUD</span>
                            </div>
                        </div>
                    </div>

                    <div class="mechanism-section" aria-label="Interactive DeepSeek Core Mechanisms">
                        <div class="mechanism-section-head">
                            <span class="mechanism-section-title">§ DEEPSEEK CORE MECHANISMS &middot; INTERACTIVE BENCHMARK LABS</span>
                        </div>
                        <div class="mechanism-grid">
                            <!-- Card 1: MLA VRAM Calculator -->
                            <div class="mechanism-card card-mla" id="mchCardMla">
                                <div class="mechanism-card-head">
                                    <span class="mch-tag">01 &middot; MLA COMPRESSION</span>
                                    <span class="mch-title">VRAM Footprint Analyzer</span>
                                </div>
                                <div class="mechanism-card-body">
                                    <p class="mch-explainer">Compare KV cache memory between standard Multi-Head Attention (1,536 dims/token) and DeepSeek-V3 MLA (216 dims/token) across context lengths.</p>
                                    <div class="mch-control-row">
                                        <span>Context Length:</span>
                                        <strong id="mlaContextLabel">32,768 tokens</strong>
                                    </div>
                                    <input type="range" id="mlaContextSlider" class="mch-slider" min="4096" max="131072" step="4096" value="32768" aria-label="Context length in tokens">
                                    <div class="mch-stat-box">
                                        <div class="msb-row">
                                            <span>Standard MHA (12 &times; 64):</span>
                                            <span id="statMhaVram">1.81 GB</span>
                                        </div>
                                        <div class="msb-row">
                                            <span>DeepSeek MLA (192 + 24):</span>
                                            <span id="statMlaVram">0.25 GB</span>
                                        </div>
                                        <div class="msb-row highlight">
                                            <span>VRAM Saved per Worker:</span>
                                            <span id="statMlaSaved">1.56 GB (7.1&times; cut)</span>
                                        </div>
                                    </div>
                                    <button type="button" class="mch-action-btn" id="mlaAbsorbAnimBtn">▸ Observe Matrix Absorption W_U &middot; W_q</button>
                                </div>
                            </div>

                            <!-- Card 2: DeepSeekMoE Router Playground -->
                            <div class="mechanism-card card-moe" id="mchCardMoe">
                                <div class="mechanism-card-head">
                                    <span class="mch-tag">02 &middot; DeepSeekMoE</span>
                                    <span class="mch-title">Aux-Loss-Free Router (20+1)</span>
                                </div>
                                <div class="mechanism-card-body">
                                    <p class="mch-explainer">Simulate 20 fine-grained routed experts (384d each) + 1 shared expert. Top-4 routed per token with dynamic load leveling biases &Delta;b.</p>
                                    <div class="moe-mini-grid" id="moeMiniGrid">
                                        <!-- 20 mini cells generated via JS -->
                                    </div>
                                    <div class="mch-stat-box">
                                        <div class="msb-row">
                                            <span>Active Experts:</span>
                                            <span id="moeActiveList">#02, #07, #13, #18 + shared</span>
                                        </div>
                                        <div class="msb-row">
                                            <span>Auxiliary Loss Penalty:</span>
                                            <span>0.0000 (&alpha;=0)</span>
                                        </div>
                                    </div>
                                    <button type="button" class="mch-action-btn" id="moeRouteBatchBtn">▸ Route Token Batch &amp; Update &Delta;b</button>
                                </div>
                            </div>

                            <!-- Card 3: MTP Speculative Tree -->
                            <div class="mechanism-card card-mtp" id="mchCardMtp">
                                <div class="mechanism-card-head">
                                    <span class="mch-tag">03 &middot; MTP DECODER</span>
                                    <span class="mch-title">Multi-Token Speculative Tree</span>
                                </div>
                                <div class="mechanism-card-body">
                                    <p class="mch-explainer">Dual-token prediction at depth 1. Main model predicts token t while auxiliary module drafts candidate token t+1 with shared embeddings.</p>
                                    <div class="mtp-tree-box" id="mtpTreeDisplay">
                                        <div class="mtp-branch">
                                            <span class="mtp-badge draft">MAIN HEAD</span>
                                            <span>Token t &rarr; "architectures" (p = 0.99)</span>
                                        </div>
                                        <div class="mtp-branch">
                                            <span class="mtp-badge acc">MTP DRAFT</span>
                                            <span>Token t+1 &rarr; "converge" (p = 0.88 &ge; 0.80)</span>
                                        </div>
                                    </div>
                                    <div class="mch-stat-box">
                                        <div class="msb-row">
                                            <span>Speculative Verification:</span>
                                            <span id="mtpStatusText">ACCEPTED (2 Tokens / Step)</span>
                                        </div>
                                        <div class="msb-row">
                                            <span>Throughput Acceleration:</span>
                                            <span id="mtpSpeedText">2.00&times; effective rate</span>
                                        </div>
                                    </div>
                                    <button type="button" class="mch-action-btn" id="mtpVerifyStepBtn">▸ Step Speculative Verification</button>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="pass-widget" id="passWidget" role="region" aria-label="Figure A1: Full Training Step Pipeline — Forward Activations, Autograd Backward Pass, Gradient Checkpointing Re-computation, and AdamW Step">
                        <div class="pass-widget-header">
                            <div class="pass-badge">
                                <span class="pass-dot" aria-hidden="true"></span>
                                <span class="pass-tag">FIG. A1</span>
                                <span class="pass-sep" aria-hidden="true">/</span>
                                <span class="pass-title">FULL TRAINING STEP PIPELINE</span>
                                <span class="pass-dim">FORWARD &middot; AUTOGRAD BACKWARD &middot; ADAMW</span>
                            </div>
                            <div class="pass-controls">
                                <button class="pass-btn active" data-phase="cycle" type="button" title="Full Forward / Backward / AdamW Cycle">AUTO CYCLE</button>
                                <button class="pass-btn" data-phase="forward" type="button" title="Inspect Forward Activation Flow">FORWARD</button>
                                <button class="pass-btn" data-phase="backward" type="button" title="Inspect Backward Gradient Flow">BACKWARD</button>
                                <button class="pass-btn pass-btn-icon" id="passPauseBtn" type="button" title="Pause / Resume Pipeline" aria-label="Pause pipeline">&#10074;&#10074;</button>
                            </div>
                        </div>

                        <div class="pass-canvas-wrap">
                            <canvas id="passDiagramCanvas" class="pass-canvas" aria-hidden="true"></canvas>
                            <div class="pass-overlay-hud" aria-hidden="true">
                                <div class="pass-hud-chip top-left">
                                    <span class="ph-lbl">CURRENT PHASE</span>
                                    <span class="ph-val" id="phCurrentPhase">FORWARD ACTIVATIONS</span>
                                </div>
                                <div class="pass-hud-chip top-right">
                                    <span class="ph-lbl">MEMORY CONTRACT</span>
                                    <span class="ph-val" id="phStepMetrics">RE-COMPUTE ACTIVATIONS &middot; 7.1&times; KV SAVED</span>
                                </div>
                                <div class="pass-stage-tooltip" id="passStageTooltip" style="opacity: 0;">
                                    <div class="st-card">
                                        <span class="st-tag" id="stTag">STAGE 02: MULTI-HEAD LATENT ATTN</span>
                                        <span class="st-op" id="stOp">c_t = W_D &middot; h_t [192d] &middot; q'_t = W_q h_t &middot; W_U</span>
                                        <span class="st-shape" id="stShape">Latent KV: [B, 4096, 192] bf16 + RoPE [B, 4096, 24]</span>
                                        <span class="st-desc" id="stDesc">Multi-Head Latent Attention with matrix absorption (7.1× KV reduction)</span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div class="pass-widget-footer">
                            <div class="pass-status-ticker">
                                <span class="ticker-beacon" id="passTickerBeacon">&bull;</span>
                                <span class="ticker-text" id="passTickerText">FORWARD &middot; Tokens x<sub class="f-sub">t</sub> &rarr; Fused Embed &rarr; 18&times; (MLA &middot; DeepSeekMoE 20+1) &rarr; MTP &rarr; Chunked CE</span>
                            </div>
                            <div class="pass-legend">
                                <span class="legend-item"><span class="legend-swatch terra"></span> <span class="legend-label">Forward Activations</span></span>
                                <span class="legend-item"><span class="legend-swatch olive"></span> <span class="legend-label">Backward Gradients (&part;&ell;/&part;&theta;)</span></span>
                                <span class="legend-item"><span class="legend-swatch gold"></span> <span class="legend-label">AdamW Update (&Delta;&theta;)</span></span>
                                <span class="legend-hint">CLICK STAGE: INSPECT &middot; HOVER: TENSOR SHAPES</span>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="portal-content">
                    {portal_cards_html}
                </div>
            </div>
        </main>
    </div>

    <!-- Scripts -->
    <script defer src="assets/portal.js"></script>
</body>
</html>
"""

    out_file = OUTPUT_DIR / "index.html"
    out_file.write_text(index_html, encoding="utf-8")


def generate_css():
    """Copy assets/style.css and assets/portal.js to docs_html/assets/."""
    import shutil
    assets_dir = OUTPUT_DIR / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    src_css = WORKSPACE_DIR / "assets" / "style.css"
    shutil.copyfile(src_css, assets_dir / "style.css")
    src_js = WORKSPACE_DIR / "assets" / "portal.js"
    shutil.copyfile(src_js, assets_dir / "portal.js")


def main():
    print("Building DeepSeek-v3-Lite HTML Documentation...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    generate_css()
    
    for rel_path, category, display_title in DOC_FILES:
        print(f"Generating: {rel_path} -> docs_html/{rel_path.replace('.md', '.html')}")
        generate_html_page(rel_path, category, display_title)
        
    generate_index_portal()
    print("\nDocumentation build complete!")
    print(f"HTML Portal location: {OUTPUT_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
