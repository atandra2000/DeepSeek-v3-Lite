"""
Generate premium architecture diagram for DeepSeek-V3-Lite.
Output: assets/architecture_overview.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── Palette ──────────────────────────────────────────────────────────────
BG      = "#0d1117"
CARD    = "#161b22"
BORDER  = "#30363d"
TEXT    = "#e6edf3"
MUTED   = "#8b949e"
BLUE    = "#58a6ff"
GREEN   = "#3fb950"
CORAL   = "#f78166"
PURPLE  = "#d2a8ff"
CYAN    = "#79c0ff"
SUBTLE  = "#1c2128"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "font.family": "sans-serif",
    "font.size": 11,
})

# ── Helpers ──────────────────────────────────────────────────────────────
CX = 5.0   # center of main flow
BW = 3.4   # main block width

def label_box(ax, x, y, w, h, color, title, sub="", lw=1.5):
    r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                        linewidth=lw, edgecolor=color, facecolor=CARD)
    ax.add_patch(r)
    ax.text(x + w / 2, y + h / 2 + 0.03, title, ha="center", va="center",
            color=color, fontsize=10, fontweight="bold")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.26, sub, ha="center", va="center",
                color=MUTED, fontsize=7.5)

def v_arrow(ax, y1, y2, color=MUTED, lw=1.3):
    ax.annotate("", xy=(CX, y2), xytext=(CX, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw))

def note_box(ax, x, y, w, h, color, lines, lw=1):
    r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                        linewidth=lw, edgecolor=color, facecolor=CARD)
    ax.add_patch(r)
    n = len(lines)
    for i, (txt, col, sz, wt) in enumerate(lines):
        ay = y + h - (i + 0.5) * h / n
        ax.text(x + w / 2, ay, txt, ha="center", va="center",
                color=col, fontsize=sz, fontweight=wt)

# ── Figure setup ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(9, 13))
ax.set_xlim(0, 9)
ax.set_ylim(0, 13)
ax.axis("off")

# ── Title ────────────────────────────────────────────────────────────────
ax.text(CX, 12.5, "DeepSeek-V3-Lite", ha="center", va="center",
        fontsize=19, fontweight="bold", color=TEXT)
ax.text(CX, 12.05, "757M Chinchilla-Scale Reproduction  ·  Single A100 80GB",
        ha="center", va="center", fontsize=10.5, color=MUTED)

# ── Main flow blocks with explicit (x, y, w, h) ─────────────────────────
# All blocks centered at CX with width BW.
bx = CX - BW / 2   # block left x (constant)

blocks = [
    # (y, h, color, title, sub)
    (10.6, 0.65, BLUE,  "Input Tokens",    "vocab_size = 14,336"),
    (9.35, 0.65, CYAN,  "Embedding",        "14,336 × 1,024"),
    (7.85, 0.80, GREEN, "MLA + SwiGLU",     "Dense Block ×2  ·  Layer 0-1"),
    (5.35, 1.70, CORAL, "MLA + DeepSeekMoE","MoE Block ×22  ·  Layer 2-23  ·  top-2 routing"),
    (4.10, 0.65, TEXT,  "RMSNorm",          "eps = 1e-6"),
    (2.85, 0.65, BLUE,  "Linear Head",      "14,336 → logits"),
    (1.90, 0.60, GREEN, "Output Logits",    "(B, S, 14,336)"),
]

for y, h, c, t, s in blocks:
    label_box(ax, bx, y, BW, h, c, t, s)

# Arrows between consecutive blocks (from bottom of one to top of next)
# v_arrow(y_from, y_to) where y_from > y_to (downward in figure coords)
for i in range(len(blocks) - 1):
    y_from = blocks[i][0] + blocks[i][1]   # bottom of current
    y_to   = blocks[i + 1][0] + blocks[i + 1][1]  # top of next = y + h
    v_arrow(ax, y_from, y_to)

# ── MoE detail mini-diagram (inside MoE block) ──────────────────────────
moe_y, moe_h = blocks[3][0], blocks[3][1]  # (5.35, 1.70)
moe_cx = CX + 0.05
moe_cy = moe_y + moe_h / 2

mini_items = [
    (moe_cx - 1.4, moe_cy + 0.25, 0.65, 0.35, CORAL,  "Gate",    5.5),
    (moe_cx + 0.6, moe_cy + 0.25, 1.15, 0.35, PURPLE, "1 Shared", 5.5),
    (moe_cx + 0.6, moe_cy - 0.25, 1.15, 0.35, CORAL,  "Top-2/16", 5.5),
]
for xi, yi, wi, hi, ci, ti, si in mini_items:
    rr = FancyBboxPatch((xi - wi / 2, yi - hi / 2), wi, hi,
                         boxstyle="round,pad=0.04", linewidth=0.8,
                         edgecolor=ci, facecolor=SUBTLE, alpha=0.7)
    ax.add_patch(rr)
    ax.text(xi, yi, ti, ha="center", va="center", color=ci, fontsize=si,
            fontweight="bold")

# Mini arrows inside MoE block
ax.annotate("", xy=(moe_cx - 1.4 + 0.33, moe_cy + 0.25),
            xytext=(moe_cx + 0.6 - 0.58, moe_cy + 0.25),
            arrowprops=dict(arrowstyle="->", color=CORAL, lw=0.6))
ax.annotate("", xy=(moe_cx - 1.4 + 0.33, moe_cy - 0.25),
            xytext=(moe_cx + 0.6 - 0.58, moe_cy - 0.25),
            arrowprops=dict(arrowstyle="->", color=CORAL, lw=0.6))

# Section bracket labels (left of flow)
bracket_x = bx - 0.35
for y_bot, h, label, col in [
    (7.85, 0.80, "Dense ×2", GREEN),
    (5.35, 1.70, "MoE ×22",  CORAL),
]:
    y_top = y_bot + h
    ax.plot([bracket_x, bracket_x], [y_bot, y_top], color=col, lw=0.8, solid_capstyle="round")
    ax.plot([bracket_x, bracket_x + 0.12], [y_top, y_top], color=col, lw=0.8)
    ax.plot([bracket_x, bracket_x + 0.12], [y_bot, y_bot], color=col, lw=0.8)
    ax.text(bracket_x - 0.08, (y_bot + y_top) / 2, label,
            ha="right", va="center", color=col, fontsize=8, fontweight="bold",
            rotation=90)

# ── Left panel: MTP ──────────────────────────────────────────────────────
mtp_x = 0.4
mtp_w = 2.0
mtp_h = 1.5
mtp_y = 4.35

# Connection line from dense-to-MoE gap
conn_y = blocks[2][0] + blocks[2][1] + 0.05  # just below dense bottom
ax.plot([bx, mtp_x + mtp_w], [conn_y, conn_y], color=PURPLE, lw=1, ls="--", alpha=0.5)
ax.plot([mtp_x + mtp_w, mtp_x + mtp_w], [conn_y, mtp_y + mtp_h],
        color=PURPLE, lw=1, ls="--", alpha=0.5)
ax.plot([mtp_x + mtp_w, mtp_x + mtp_w - 0.08], [mtp_y + mtp_h, mtp_y + mtp_h],
        color=PURPLE, lw=1, alpha=0.7)

note_box(ax, mtp_x, mtp_y, mtp_w, mtp_h, PURPLE, [
    ("Multi-Token Prediction", PURPLE, 9, "bold"),
    ("Depth = 1", MUTED, 7.5, "normal"),
    ("Shared output head", MUTED, 7, "normal"),
    ("Predicts token t+2", MUTED, 7, "normal"),
], lw=1.2)

ax.text(mtp_x, mtp_y + mtp_h + 0.12, "MTP", ha="center", va="center",
        color=PURPLE, fontsize=7.5, fontweight="bold")

# ── Right panel: MoE callout ────────────────────────────────────────────
rd_x = 6.8
rd_w = 1.9
rd_y = blocks[3][0] + 0.15
rd_h = blocks[3][1] - 0.3

# Connection from MoE block
r_conn_y = blocks[3][0] + blocks[3][1] / 2
ax.plot([bx + BW, rd_x], [r_conn_y, r_conn_y], color=CORAL, lw=1, ls="--", alpha=0.4)
ax.plot([rd_x, rd_x], [r_conn_y, rd_y + rd_h], color=CORAL, lw=1, ls="--", alpha=0.4)
ax.plot([rd_x, rd_x + 0.08], [rd_y + rd_h, rd_y + rd_h], color=CORAL, lw=1, alpha=0.6)

note_box(ax, rd_x, rd_y, rd_w, rd_h, CORAL, [
    ("DeepSeekMoE", CORAL, 8.5, "bold"),
    ("1 shared expert", PURPLE, 7, "normal"),
    ("16 routed experts", CORAL, 7, "normal"),
    ("Top-2 per token", CORAL, 7, "normal"),
    ("Aux-loss-free bias", MUTED, 6.5, "normal"),
], lw=1)

# ── Bottom left: Key Specifications ──────────────────────────────────────
specs_x = 0.4
specs_y = 0.15
specs_w = 4.0
specs_h = 1.5

note_box(ax, specs_x, specs_y, specs_w, specs_h, BORDER, [
    ("Key Specifications", TEXT, 9, "bold"),
    ("", MUTED, 2, "normal"),
], lw=1)

metrics = [
    ("Parameters", "757M"),
    ("Layers", "2 dense + 22 MoE"),
    ("KV cache", "256-dim latent per layer"),
    ("Precision", "BF16 + FP32 master"),
    ("Hardware", "1× A100 80GB"),
]
n_met = len(metrics)
for i, (k, v) in enumerate(metrics):
    row_y = specs_y + specs_h - 0.55 - i * 0.22
    ax.text(specs_x + 0.3, row_y, k, ha="left", va="center",
            color=MUTED, fontsize=7)
    ax.text(specs_x + specs_w - 0.3, row_y, v, ha="right", va="center",
            color=TEXT, fontsize=7, fontweight="bold")
    if i < n_met - 1:
        ax.plot([specs_x + 0.3, specs_x + specs_w - 0.3],
                [row_y - 0.11, row_y - 0.11], color=SUBTLE, lw=0.5)

# ── Bottom right: Training Pipeline ──────────────────────────────────────
pipe_x = 4.6
pipe_y = 0.15
pipe_w = 4.0
pipe_h = 1.5

note_box(ax, pipe_x, pipe_y, pipe_w, pipe_h, BORDER, [
    ("Training Pipeline", TEXT, 9, "bold"),
    ("", MUTED, 2, "normal"),
], lw=1)

stages = [
    ("Pre-train", "15.1B tokens · Chinchilla-20", BLUE),
    ("SFT", "Sample-isolation masking", GREEN),
    ("GRPO", "Group size 4 · PPO clip 0.2", CORAL),
    ("Distill", "KL(T=2) + CE · frozen teacher", PURPLE),
]
for i, (name, desc, col) in enumerate(stages):
    row_y = pipe_y + pipe_h - 0.55 - i * 0.30
    ax.text(pipe_x + 0.3, row_y + 0.04, "●", ha="left", va="center",
            color=col, fontsize=6.5)
    ax.text(pipe_x + 0.55, row_y + 0.04, name, ha="left", va="center",
            color=col, fontsize=7, fontweight="bold")
    ax.text(pipe_x + 0.55, row_y - 0.13, desc, ha="left", va="center",
            color=MUTED, fontsize=6)
    if i < len(stages) - 1:
        ax.plot([pipe_x + 0.3, pipe_x + pipe_w - 0.3],
                [row_y - 0.20, row_y - 0.20], color=SUBTLE, lw=0.5)

# ── Footer ────────────────────────────────────────────────────────────────
ax.text(CX, 0.02, "github.com/atandra2000/DeepSeek-V3-Lite",
        ha="center", va="center", color=MUTED, fontsize=7.5)

# ── Save ──────────────────────────────────────────────────────────────────
out = "assets/architecture_overview.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
print(f"Saved → {out}")
