"""Build-output contract tests for the docs_html portal.

Runs the real generator once (module scope) and asserts the premium-polish
wiring: portal.js asset, boot overlay, ASCII hero, mono-only fonts, widget
containers. Markdown sources are never modified by the build.
"""

import re
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


def test_index_hero_stage(built):
    html = read("index.html")
    assert 'id="hero-decode"' in html
    assert 'id="heroStateCanvas"' in html
    assert 'data-title="DEEPSEEK-V3-LITE"' in html
    assert "bottleneck-svg" not in html
    assert "hero-title sr-only" in html


def test_index_pass_widget_container(built):
    html = read("index.html")
    assert 'id="passWidget"' in html
    assert 'id="passDiagramCanvas"' in html


def test_moe_playground_container(built):
    assert 'id="widget-moe-routing"' in read("docs/concepts/moe-mtp.html")


def test_mla_toggle_container(built):
    assert 'id="widget-mla-absorb"' in read("docs/concepts/attention-and-precision.html")


def test_widgets_not_on_other_pages(built):
    html = read("docs/concepts/parallelism.html")
    assert "widget-moe-routing" not in html
    assert "widget-mla-absorb" not in html
    assert "passWidget" not in html


def test_dark_theme_only(built):
    html = read("README.html")
    assert "toggleTheme" not in html
    assert "theme-toggle" not in html
    css = (OUT / "assets" / "style.css").read_text(encoding="utf-8")
    assert '[data-theme="light"]' not in css


# --- link / anchor integrity -------------------------------------------------
# `check_docs.py` validates anchors in the Markdown sources using GitHub slug
# rules. The HTML builder must emit the same ids, or every cross-page anchor
# silently 404s in the portal while the source docs still lint clean.

def _heading_ids(html: str) -> list[str]:
    return re.findall(r'<(?:h[1-6]|span)[^>]*id="([^"]+)"', html)


def _generated_pages() -> list[Path]:
    """Pages the markdown pass generates.

    Excludes the two non-generated kinds under docs_html: `index.html` (the
    portal index, not a page body) and the hand-authored diagram `.html`
    siblings copied verbatim from `docs/diagrams/`. Both are audited for dead
    links too, but they carry no heading ids we own.
    """
    copied = {p.name for p in (ROOT / "docs" / "diagrams").glob("*.html")}
    return sorted(
        p
        for p in OUT.rglob("*.html")
        if "assets" not in p.parts and p.name not in copied and p.name != "index.html"
    )


def test_no_duplicate_heading_ids(built):
    """Repeated headings get GitHub's -1/-2 suffix instead of colliding."""
    for page in _generated_pages():
        ids = _heading_ids(page.read_text(encoding="utf-8"))
        dupes = {i for i in ids if ids.count(i) > 1}
        assert not dupes, f"{page.name}: duplicate heading ids {sorted(dupes)}"


def test_no_dead_internal_anchors(built):
    """Every href="#x" resolves to an id on the same page."""
    for page in _generated_pages():
        html = page.read_text(encoding="utf-8")
        ids = set(_heading_ids(html))
        dead = {m for m in re.findall(r'href="#([^"]+)"', html) if m not in ids}
        assert not dead, f"{page.name}: dead anchors {sorted(dead)}"


def test_no_dead_local_page_links(built):
    """Every relative *.html href points at a file that exists."""
    for page in _generated_pages():
        for href in re.findall(r'href="([^"#:]+\.html)(?:#[^"]*)?"', page.read_text(encoding="utf-8")):
            assert (page.parent / href).exists(), f"{page.name}: dead link {href}"


def test_slugs_match_github_convention(built):
    """Em-dash headings keep one hyphen per space; inline code text survives."""
    ids = _heading_ids(read("docs/inference.html"))
    assert "standard-generation--generate-walkthrough" in ids
    ids_mla = _heading_ids(read("docs/concepts/attention-and-precision.html"))
    assert "appendix-e--why-wkv_b-has-shape-h-d_nope--d_v-r" in ids_mla


def test_all_markdown_docs_are_generated(built):
    """Every docs/**/*.md is reachable in the portal.

    Guards the dead-link class where a new doc is written and linked from
    docs/README.md but never added to DOC_FILES, so the nav and the link 404.
    """
    root = ROOT
    sources = {
        p.relative_to(root).as_posix()
        for p in (root / "docs").rglob("*.md")
        if "superpowers" not in p.parts
    }
    generated = {
        p.relative_to(OUT).with_suffix(".md").as_posix() for p in _generated_pages()
    }
    missing = sorted(sources - generated)
    assert not missing, f"markdown never generated into the portal: {missing}"


def test_standalone_diagrams_copied(built):
    """Hand-authored interactive diagrams land next to the generated pages."""
    src = sorted((ROOT / "docs" / "diagrams").glob("*.html"))
    assert src, "no standalone diagram html found"
    for diagram in src:
        assert (OUT / "docs" / "diagrams" / diagram.name).is_file()

