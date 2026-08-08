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


def slugify(text: str) -> str:
    """Generate clean HTML id for headings.

    Matches the anchor convention the docs were authored against (GitHub-style):
    lowercase, keep word chars + underscore + hyphen, drop everything else,
    and turn each space into a single hyphen (no run collapsing). An em-dash
    surrounded by spaces thus yields ``--`` (e.g. ``Data Flow — Training`` ->
    ``data-flow--training``), and code identifiers keep their underscores
    (``## train_step`` -> ``train_step``).
    """
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
    """Rewrite relative markdown links for the HTML build.

    - ``.md`` links (with optional ``#anchor``) -> ``.html`` twin.
    - non-``.md`` repo-relative links (``configs/…``, ``training/…``) -> GitHub
      blob URL (resolved against the source file's repo-relative dir), since the
      file isn't shipped inside ``docs_html/``.
    """
    repo_base = github_base_url()
    src_dir = Path(src_rel_path).parent

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
        # Escape minimal HTML entities (< and >) so browser doesn't interpret them as tags,
        # but keep all backslashes and LaTeX symbols untouched!
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
    
    in_list = False
    list_type = None
    
    in_blockquote = False
    blockquote_type = "normal"
    blockquote_lines = []

    def flush_list():
        nonlocal in_list, list_type
        if in_list:
            html_lines.append(f"</{list_type}>")
            in_list = False
            list_type = None

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

    def render_inline_formatting(text: str) -> str:
        # Escape html chars for safety (except placeholders)
        # Note: < and > in prose get escaped
        text = html.escape(text, quote=False)
        
        # Bold **text**
        text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
        # Italic *text*
        text = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', text)
        # Strikethrough ~~text~~
        text = re.sub(r'~~([^~]+)~~', r'<del>\1</del>', text)
        # Links [text](url)
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" class="doc-link">\1</a>', text)

        return text

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Placeholders for display math or code blocks on their own line
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

        # Empty line
        if not stripped:
            flush_table()
            flush_list()
            flush_blockquote()
            i += 1
            continue

        # Blockquote or Callout
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

        # Horizontal Rule
        if re.match(r'^(---|\*\*\*|___)\s*$', stripped):
            flush_table()
            flush_list()
            flush_blockquote()
            html_lines.append("<hr class='doc-hr'>")
            i += 1
            continue

        # Headings (# to ######)
        heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
        if heading_match:
            flush_table()
            flush_list()
            flush_blockquote()
            level = len(heading_match.group(1))
            heading_text_raw = heading_match.group(2).strip()
            
            # Clean heading text for id/slug
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
                
            html_lines.append(
                f'<h{level} id="{heading_id}" class="heading-anchor">'
                f'{rendered_heading}'
                f'<a href="#{heading_id}" class="anchor-link" aria-label="Link to section">#</a>'
                f'</h{level}>'
            )
            i += 1
            continue

        # Markdown Table Detection
        if "|" in line and i + 1 < len(lines) and re.match(r'^\s*\|?\s*:?---', lines[i + 1].strip()):
            flush_list()
            flush_blockquote()
            in_table = True
            headers_raw = [c.strip() for c in line.strip().strip("|").split("|")]
            table_headers = [render_inline_formatting(h) for h in headers_raw]
            i += 2  # skip separator line
            
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                cells_raw = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                table_rows.append([render_inline_formatting(c) for c in cells_raw])
                i += 1
            flush_table()
            continue

        # Lists (unordered - or *, ordered 1.)
        ul_match = re.match(r'^[\*\-]\s+(.+)$', stripped)
        ol_match = re.match(r'^\d+\.\s+(.+)$', stripped)
        if ul_match or ol_match:
            flush_table()
            flush_blockquote()
            target_type = 'ul' if ul_match else 'ol'
            item_text = (ul_match or ol_match).group(1).strip()
            
            if not in_list or list_type != target_type:
                flush_list()
                in_list = True
                list_type = target_type
                html_lines.append(f'<{list_type} class="doc-list">')
                
            task_match = re.match(r'^\[([ xX])\]\s+(.+)$', item_text)
            if task_match:
                checked = 'checked' if task_match.group(1).lower() == 'x' else ''
                item_content = render_inline_formatting(task_match.group(2))
                html_lines.append(f'<li class="task-item"><input type="checkbox" disabled {checked}> {item_content}</li>')
            else:
                html_lines.append(f'<li>{render_inline_formatting(item_text)}</li>')
            i += 1
            continue

        # Standard Paragraph
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
    # Restore inline math
    for idx, math_html in enumerate(inline_maths):
        full_html = full_html.replace(f"___INLINEMATH_{idx}___", math_html)

    # Restore inline code
    for idx, raw_code in enumerate(inline_codes):
        # Extract content between `...`
        code_content = raw_code[1:-1]
        code_html = f'<code class="inline-code">{html.escape(code_content)}</code>'
        full_html = full_html.replace(f"___INLINECODE_{idx}___", code_html)

    # Restore display math
    for idx, math_html in enumerate(display_maths):
        full_html = full_html.replace(f"___DISPLAYMATH_{idx}___", math_html)

    # Restore code blocks
    for idx, raw_block in enumerate(code_blocks):
        lines_b = raw_block.splitlines()
        first_line = lines_b[0].strip()
        code_lang = first_line.lstrip("```").strip().lower()
        code_content = "\n".join(lines_b[1:-1])
        escaped_content = html.escape(code_content)
        
        lang_attr = f' class="language-{code_lang}"' if code_lang else ''
        data_lang = code_lang if code_lang else 'code'
        
        block_html = (
            f'<div class="code-wrapper">'
            f'<div class="code-header">'
            f'<span class="code-lang">{data_lang}</span>'
            f'<button class="copy-btn" onclick="copyCode(this)">Copy</button>'
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
        
    page_html = f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{display_title} | DeepSeek-v3-Lite Documentation</title>
    <!-- Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <!-- Highlight.js for Syntax Highlighting -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css" id="highlight-theme">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <!-- KaTeX for LaTeX Math -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>
    <!-- CSS Stylesheet -->
    <link rel="stylesheet" href="{rel_prefix}assets/style.css">
</head>
<body>
    <!-- Top Header -->
    <header class="site-header">
        <div class="header-left">
            <button class="mobile-toggle" onclick="toggleSidebar()" aria-label="Toggle Sidebar">☰</button>
            <a href="{rel_prefix}index.html" class="brand-logo">
                <span class="logo-icon">⚡</span>
                <span class="brand-name">DeepSeek-v3-Lite</span>
                <span class="brand-badge">Docs</span>
            </a>
        </div>
        <div class="header-right">
            <a href="{rel_prefix}index.html" class="header-link">Portal Home</a>
            <a href="{rel_prefix}README.html" class="header-link">GitHub README</a>
            <button class="theme-toggle" onclick="toggleTheme()" id="themeToggleBtn" aria-label="Toggle Theme">🌙 Dark</button>
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
                        <span class="meta-item">📁 {rel_path}</span>
                        <span class="meta-item">📝 {word_count:,} words</span>
                        <span class="meta-item">⏱️ ~{reading_time} min read</span>
                    </div>
                </div>

                <article class="markdown-body" id="articleBody">
                    {html_body}
                </article>

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
        // Copy Code Functionality
        function copyCode(btn) {{
            const wrapper = btn.closest('.code-wrapper');
            const code = wrapper.querySelector('code').innerText;
            navigator.clipboard.writeText(code).then(() => {{
                btn.innerText = 'Copied!';
                btn.classList.add('copied');
                setTimeout(() => {{
                    btn.innerText = 'Copy';
                    btn.classList.remove('copied');
                }}, 2000);
            }});
        }}

        // Theme Toggle
        function toggleTheme() {{
            const htmlEl = document.documentElement;
            const themeBtn = document.getElementById('themeToggleBtn');
            const hlTheme = document.getElementById('highlight-theme');
            
            if (htmlEl.getAttribute('data-theme') === 'dark') {{
                htmlEl.setAttribute('data-theme', 'light');
                themeBtn.innerText = '☀️ Light';
                hlTheme.href = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css';
                localStorage.setItem('theme', 'light');
            }} else {{
                htmlEl.setAttribute('data-theme', 'dark');
                themeBtn.innerText = '🌙 Dark';
                hlTheme.href = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css';
                localStorage.setItem('theme', 'dark');
            }}
        }}

        const savedTheme = localStorage.getItem('theme');
        if (savedTheme === 'light') {{
            document.documentElement.setAttribute('data-theme', 'light');
            document.getElementById('themeToggleBtn').innerText = '☀️ Light';
            document.getElementById('highlight-theme').href = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css';
        }}

        function toggleSidebar() {{
            document.getElementById('sidebar').classList.toggle('open');
        }}

        function filterNav() {{
            const query = document.getElementById('navSearch').value.toLowerCase();
            const items = document.querySelectorAll('.nav-item');
            items.forEach(item => {{
                const text = item.innerText.toLowerCase();
                item.style.display = text.includes(query) ? 'block' : 'none';
            }});
        }}

        // Initialize Highlight.js & KaTeX
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
        "Core Architecture": [
            ("README.html", "Project Overview", "Chinchilla-scale 412M parameter PyTorch implementation of DeepSeek-V3."),
            ("AGENTS.html", "AGENTS & System Architecture", "Codebase contracts, Triton rules, μP scaling & architectural limits."),
            ("SKILLS.html", "Skills Map", "Specialized agent tools, scripts, and domain competencies."),
            ("docs/training.html", "Training Pipeline", "BF16 training loop, loss scaling, checkpointing & NaN safety."),
            ("docs/inference.html", "Inference & Speculative", "MTP speculative decoder, KV caching & latency optimizations.")
        ],
        "Architecture & Concepts": [
            ("docs/concepts/foundations.html", "Foundations & Topography", "DeepSeek lineage, 18-layer layout, parameter & memory budgets."),
            ("docs/concepts/attention-and-precision.html", "MLA & Mixed Precision", "Multi-Head Latent Attention, matrix absorption, FP8 scheme."),
            ("docs/concepts/moe-mtp.html", "DeepSeekMoE & MTP", "Auxiliary-loss-free load balancing, 20 routed + 1 shared expert, MTP."),
            ("docs/concepts/kernels-and-ops.html", "Operations & Triton", "Triton grouped GEMM, MLA fused kernel, CI testing & invariants.")
        ],
        "Guides & Playbooks": [
            ("docs/guides/getting-started.html", "Getting Started", "Quickstart installation, synthetic pretraining & smoke testing."),
            ("docs/guides/G1_debugging_playbook.html", "G1 — Debugging Playbook", "Troubleshooting NaNs, CUDA OOMs, and loss divergence."),
            ("docs/guides/G2_mup_and_lr_tuning.html", "G2 — μP & LR Tuning", "Maximal Update Parameterization setup & hyperparameter transfer."),
            ("docs/guides/G3_triton_development.html", "G3 — Triton Development", "Writing, profiling, and benchmarking custom Triton GPU kernels.")
        ],
        "API References": [
            ("docs/references/R1_config_schema.html", "R1 — Config Schema", "Complete YAML specification for model & training configs."),
            ("docs/references/R2_transformer_api.html", "R2 — Transformer API", "Model initialization, forward pass, and weight wiring."),
            ("docs/references/R3_mla_api.html", "R3 — MLA API", "Multi-Head Latent Attention layer implementation & options."),
            ("docs/references/R4_moe_api.html", "R4 — MoE API", "AuxLossFreeGate and DeepSeekMoE expert routing engine.")
        ]
    }
    
    portal_cards_html = ""
    for cat_title, items in categories.items():
        cards = ""
        for href, title, desc in items:
            cards += f"""
            <a href="{href}" class="portal-card">
                <div class="card-icon">📄</div>
                <div class="card-body">
                    <h3 class="card-heading">{title}</h3>
                    <p class="card-desc">{desc}</p>
                </div>
            </a>
            """
        portal_cards_html += f"""
        <section class="portal-section">
            <h2 class="portal-category-title">{cat_title}</h2>
            <div class="portal-grid">{cards}</div>
        </section>
        """

    index_html = f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DeepSeek-v3-Lite Documentation Portal</title>
    <!-- Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <!-- CSS Stylesheet -->
    <link rel="stylesheet" href="assets/style.css">
</head>
<body>
    <!-- Top Header -->
    <header class="site-header">
        <div class="header-left">
            <button class="mobile-toggle" onclick="toggleSidebar()" aria-label="Toggle Sidebar">☰</button>
            <a href="index.html" class="brand-logo">
                <span class="logo-icon">⚡</span>
                <span class="brand-name">DeepSeek-v3-Lite</span>
                <span class="brand-badge">Documentation</span>
            </a>
        </div>
        <div class="header-right">
            <a href="README.html" class="header-link">GitHub README</a>
            <button class="theme-toggle" onclick="toggleTheme()" id="themeToggleBtn" aria-label="Toggle Theme">🌙 Dark</button>
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
                    <h1 class="hero-title">DeepSeek-v3-Lite</h1>
                    <p class="hero-subtitle">Faithful, from-scratch PyTorch reimplementation of DeepSeek-V3 (412M parameters, MLA, Aux-Loss-Free MoE, MTP, μP scaling).</p>
                    <div class="hero-stats">
                        <div class="stat-pill"><span class="stat-num">411.6M</span> Base Params</div>
                        <div class="stat-pill"><span class="stat-num">18</span> Layers (2 Dense + 16 MoE)</div>
                        <div class="stat-pill"><span class="stat-num">20+1</span> Experts (Top-4)</div>
                        <div class="stat-pill"><span class="stat-num">BF16</span> Autocast + μP</div>
                    </div>
                </div>

                <div class="portal-content">
                    {portal_cards_html}
                </div>
            </div>
        </main>
    </div>

    <!-- Scripts -->
    <script>
        function toggleTheme() {{
            const htmlEl = document.documentElement;
            const themeBtn = document.getElementById('themeToggleBtn');
            if (htmlEl.getAttribute('data-theme') === 'dark') {{
                htmlEl.setAttribute('data-theme', 'light');
                themeBtn.innerText = '☀️ Light';
                localStorage.setItem('theme', 'light');
            }} else {{
                htmlEl.setAttribute('data-theme', 'dark');
                themeBtn.innerText = '🌙 Dark';
                localStorage.setItem('theme', 'dark');
            }}
        }}

        const savedTheme = localStorage.getItem('theme');
        if (savedTheme === 'light') {{
            document.documentElement.setAttribute('data-theme', 'light');
            document.getElementById('themeToggleBtn').innerText = '☀️ Light';
        }}

        function toggleSidebar() {{
            document.getElementById('sidebar').classList.toggle('open');
        }}

        function filterNav() {{
            const query = document.getElementById('navSearch').value.toLowerCase();
            const items = document.querySelectorAll('.nav-item');
            items.forEach(item => {{
                const text = item.innerText.toLowerCase();
                item.style.display = text.includes(query) ? 'block' : 'none';
            }});
        }}
    </script>
</body>
</html>
"""

    out_file = OUTPUT_DIR / "index.html"
    out_file.write_text(index_html, encoding="utf-8")


def generate_css():
    """Create docs_html/assets/style.css with modern design system."""
    assets_dir = OUTPUT_DIR / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    
    css_content = """/* Modern Documentation CSS Design System */
:root {
    --bg-main: #0d1117;
    --bg-surface: #161b22;
    --bg-surface-hover: #21262d;
    --border-color: #30363d;
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
    --text-muted: #6e7681;
    --accent-color: #10b981;
    --accent-hover: #059669;
    --accent-alpha: rgba(16, 185, 129, 0.12);
    --code-bg: #161b22;
    --header-bg: rgba(13, 17, 23, 0.85);
    --callout-note-bg: rgba(59, 130, 246, 0.1);
    --callout-note-border: #3b82f6;
    --callout-tip-bg: rgba(16, 185, 129, 0.1);
    --callout-tip-border: #10b981;
    --callout-warn-bg: rgba(245, 158, 11, 0.1);
    --callout-warn-border: #f59e0b;
    --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    --font-mono: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
}

[data-theme="light"] {
    --bg-main: #ffffff;
    --bg-surface: #f6f8fa;
    --bg-surface-hover: #eaeef2;
    --border-color: #d0d7de;
    --text-primary: #1f2328;
    --text-secondary: #656d76;
    --text-muted: #8c959f;
    --accent-color: #059669;
    --accent-hover: #047857;
    --accent-alpha: rgba(5, 150, 105, 0.1);
    --code-bg: #f6f8fa;
    --header-bg: rgba(255, 255, 255, 0.85);
    --callout-note-bg: rgba(59, 130, 246, 0.08);
    --callout-note-border: #2563eb;
    --callout-tip-bg: rgba(16, 185, 129, 0.08);
    --callout-tip-border: #059669;
    --callout-warn-bg: rgba(245, 158, 11, 0.08);
    --callout-warn-border: #d97706;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: var(--font-sans);
    background-color: var(--bg-main);
    color: var(--text-primary);
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
}

/* Site Header */
.site-header {
    position: sticky;
    top: 0;
    z-index: 100;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem 1.5rem;
    background: var(--header-bg);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border-color);
}

.brand-logo {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    text-decoration: none;
    color: var(--text-primary);
    font-weight: 700;
    font-size: 1.1rem;
}

.logo-icon { font-size: 1.3rem; }
.brand-badge {
    font-size: 0.75rem;
    padding: 2px 8px;
    border-radius: 12px;
    background: var(--accent-alpha);
    color: var(--accent-color);
    font-weight: 600;
}

.header-right { display: flex; align-items: center; gap: 1rem; }
.header-link {
    color: var(--text-secondary);
    text-decoration: none;
    font-size: 0.9rem;
    font-weight: 500;
    transition: color 0.2s;
}
.header-link:hover { color: var(--accent-color); }

.theme-toggle, .mobile-toggle {
    background: var(--bg-surface);
    border: 1px solid var(--border-color);
    color: var(--text-primary);
    padding: 6px 12px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.85rem;
    font-weight: 500;
    transition: all 0.2s;
}
.theme-toggle:hover, .mobile-toggle:hover { background: var(--bg-surface-hover); }

.mobile-toggle { display: none; }

/* Layout Grid */
.app-layout {
    display: grid;
    grid-template-columns: 280px 1fr 240px;
    max-width: 1600px;
    margin: 0 auto;
    min-height: calc(100vh - 57px);
}

/* Sidebar Navigation */
.sidebar {
    border-right: 1px solid var(--border-color);
    background: var(--bg-main);
    position: sticky;
    top: 57px;
    height: calc(100vh - 57px);
    overflow-y: auto;
}

.sidebar-inner { padding: 1.25rem 1rem; }

.sidebar-search input {
    width: 100%;
    padding: 8px 12px;
    background: var(--bg-surface);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    color: var(--text-primary);
    font-size: 0.85rem;
    margin-bottom: 1.25rem;
    outline: none;
}
.sidebar-search input:focus { border-color: var(--accent-color); }

.nav-group { margin-bottom: 1.25rem; }
.nav-group-title {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    margin-bottom: 0.5rem;
    padding-left: 8px;
}

.nav-list { list-style: none; }
.nav-item { margin-bottom: 2px; }

.nav-link {
    display: block;
    padding: 6px 10px;
    border-radius: 6px;
    color: var(--text-secondary);
    text-decoration: none;
    font-size: 0.88rem;
    font-weight: 400;
    transition: all 0.15s;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.nav-link:hover {
    background: var(--bg-surface-hover);
    color: var(--text-primary);
}

.nav-link.active {
    background: var(--accent-alpha);
    color: var(--accent-color);
    font-weight: 600;
}

/* Main Content Area */
.main-content {
    padding: 2.5rem 3rem;
    overflow-x: hidden;
}

.content-container { max-width: 900px; margin: 0 auto; }

.breadcrumb {
    font-size: 0.85rem;
    color: var(--text-muted);
    margin-bottom: 1rem;
}
.breadcrumb a { color: var(--text-secondary); text-decoration: none; }
.breadcrumb a:hover { color: var(--accent-color); }
.breadcrumb .current { color: var(--text-primary); font-weight: 500; }

.doc-header {
    border-bottom: 1px solid var(--border-color);
    padding-bottom: 1.25rem;
    margin-bottom: 2rem;
}

.doc-title {
    font-size: 2.25rem;
    font-weight: 800;
    letter-spacing: -0.025em;
    margin-bottom: 0.75rem;
}

.doc-meta {
    display: flex;
    gap: 1.25rem;
    font-size: 0.85rem;
    color: var(--text-muted);
}

/* Typography & Markdown Body */
.markdown-body p { margin-bottom: 1.25rem; font-size: 1rem; color: var(--text-primary); }

.markdown-body h1, .markdown-body h2, .markdown-body h3, .markdown-body h4 {
    color: var(--text-primary);
    font-weight: 700;
    line-height: 1.3;
    margin-top: 2rem;
    margin-bottom: 1rem;
    scroll-margin-top: 80px;
    position: relative;
}

.heading-anchor .anchor-link {
    opacity: 0;
    margin-left: 0.5rem;
    color: var(--text-muted);
    text-decoration: none;
    font-weight: 400;
    transition: opacity 0.2s;
}
.heading-anchor:hover .anchor-link { opacity: 1; }

.markdown-body h2 { font-size: 1.5rem; border-bottom: 1px solid var(--border-color); padding-bottom: 0.4rem; }
.markdown-body h3 { font-size: 1.25rem; }

.doc-link { color: var(--accent-color); text-decoration: none; font-weight: 500; }
.doc-link:hover { text-decoration: underline; }

.inline-code {
    background: var(--code-bg);
    border: 1px solid var(--border-color);
    padding: 2px 6px;
    border-radius: 4px;
    font-family: var(--font-mono);
    font-size: 0.88em;
    color: var(--text-primary);
}

.code-wrapper {
    background: var(--code-bg);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    margin: 1.5rem 0;
    overflow: hidden;
}

.code-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 14px;
    background: var(--bg-surface);
    border-bottom: 1px solid var(--border-color);
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--text-muted);
}

.copy-btn {
    background: transparent;
    border: 1px solid var(--border-color);
    color: var(--text-secondary);
    padding: 2px 8px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.75rem;
    transition: all 0.2s;
}
.copy-btn:hover { background: var(--bg-surface-hover); color: var(--text-primary); }
.copy-btn.copied { border-color: var(--accent-color); color: var(--accent-color); }

.code-wrapper pre { margin: 0; padding: 1rem; overflow-x: auto; font-family: var(--font-mono); font-size: 0.88rem; }

/* LaTeX Math Styling */
.math-block {
    overflow-x: auto;
    margin: 1.5rem 0;
    padding: 1rem;
    background: var(--bg-surface);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    text-align: center;
}

.math-inline {
    font-size: 1.02em;
    padding: 0 2px;
}

/* Tables */
.table-container { overflow-x: auto; margin: 1.5rem 0; }
.doc-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
    text-align: left;
}
.doc-table th {
    background: var(--bg-surface);
    padding: 10px 14px;
    font-weight: 600;
    border-bottom: 2px solid var(--border-color);
}
.doc-table td { padding: 10px 14px; border-bottom: 1px solid var(--border-color); }
.doc-table tr:hover { background: var(--bg-surface-hover); }

/* Callouts / Blockquotes */
.callout {
    padding: 1rem 1.25rem;
    border-left: 4px solid;
    border-radius: 0 8px 8px 0;
    margin: 1.5rem 0;
}
.callout-note { background: var(--callout-note-bg); border-color: var(--callout-note-border); }
.callout-tip { background: var(--callout-tip-bg); border-color: var(--callout-tip-border); }
.callout-warn, .callout-warning { background: var(--callout-warn-bg); border-color: var(--callout-warn-border); }
.callout-header { display: flex; align-items: center; gap: 0.5rem; font-weight: 700; font-size: 0.9rem; margin-bottom: 0.4rem; }

blockquote {
    border-left: 4px solid var(--border-color);
    padding: 0.5rem 1rem;
    color: var(--text-secondary);
    margin: 1.25rem 0;
    font-style: italic;
}

/* Lists */
.doc-list { padding-left: 1.5rem; margin-bottom: 1.25rem; }
.doc-list li { margin-bottom: 0.4rem; }

/* Footer Page Nav */
.doc-footer-nav {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-top: 4rem;
    padding-top: 2rem;
    border-top: 1px solid var(--border-color);
}

.nav-card {
    display: flex;
    flex-direction: column;
    padding: 1rem;
    background: var(--bg-surface);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    text-decoration: none;
    transition: all 0.2s;
}
.nav-card:hover { border-color: var(--accent-color); transform: translateY(-2px); }
.next-card { text-align: right; }
.card-label { font-size: 0.75rem; color: var(--text-muted); font-weight: 600; text-transform: uppercase; }
.card-title { font-size: 0.95rem; color: var(--text-primary); font-weight: 600; margin-top: 4px; }

/* TOC Sidebar */
.toc-sidebar {
    border-left: 1px solid var(--border-color);
    padding: 1.5rem 1rem;
    position: sticky;
    top: 57px;
    height: calc(100vh - 57px);
    overflow-y: auto;
}
.toc-title { font-size: 0.75rem; font-weight: 700; text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.75rem; }
.toc-list { list-style: none; }
.toc-list li { margin-bottom: 6px; }
.toc-link { color: var(--text-secondary); text-decoration: none; font-size: 0.82rem; transition: color 0.2s; }
.toc-link:hover { color: var(--accent-color); }
.toc-h3 { padding-left: 12px; }

/* Portal Dashboard Page */
.hero-banner {
    padding: 3rem 2rem;
    background: radial-gradient(circle at top right, var(--accent-alpha), transparent 60%);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    margin-bottom: 2.5rem;
}
.hero-title { font-size: 2.5rem; font-weight: 800; letter-spacing: -0.03em; margin-bottom: 0.5rem; }
.hero-subtitle { font-size: 1.1rem; color: var(--text-secondary); max-width: 700px; margin-bottom: 1.5rem; }
.hero-stats { display: flex; gap: 1rem; flex-wrap: wrap; }
.stat-pill { background: var(--bg-surface); border: 1px solid var(--border-color); padding: 6px 14px; border-radius: 20px; font-size: 0.85rem; font-weight: 500; }
.stat-num { color: var(--accent-color); font-weight: 700; }

.portal-section { margin-bottom: 2.5rem; }
.portal-category-title { font-size: 1.3rem; font-weight: 700; margin-bottom: 1rem; border-bottom: 1px solid var(--border-color); padding-bottom: 0.4rem; }
.portal-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 1rem; }

.portal-card {
    display: flex;
    gap: 1rem;
    padding: 1.25rem;
    background: var(--bg-surface);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    text-decoration: none;
    transition: all 0.2s;
}
.portal-card:hover { border-color: var(--accent-color); background: var(--bg-surface-hover); transform: translateY(-2px); }
.card-icon { font-size: 1.5rem; }
.card-heading { font-size: 1rem; font-weight: 600; color: var(--text-primary); margin-bottom: 4px; }
.card-desc { font-size: 0.83rem; color: var(--text-secondary); line-height: 1.4; }

/* Responsive adjustments */
@media (max-width: 1100px) {
    .app-layout { grid-template-columns: 240px 1fr; }
    .toc-sidebar { display: none; }
}

@media (max-width: 768px) {
    .app-layout { grid-template-columns: 1fr; }
    .sidebar { display: none; position: fixed; left: 0; top: 57px; width: 280px; z-index: 99; }
    .sidebar.open { display: block; }
    .mobile-toggle { display: block; }
    .main-content { padding: 1.5rem; }
}
"""
    (assets_dir / "style.css").write_text(css_content, encoding="utf-8")


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
