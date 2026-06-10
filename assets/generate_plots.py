"""
Generate premium architecture diagram for DeepSeek-V3-Lite.
Output: assets/architecture_overview.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
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
ORANGE  = "#ffa657"
CYAN    = "#79c0ff"
SUBTLE  = "#21262d"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "font.family": "sans-serif",
    "font.size": 11,
})

# ── helpers ──────────────────────────────────────────────────────────────
def box(ax, x, y, w, h, color, label, sub="", lw=1.5, alpha=1.0):
    r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                        linewidth=lw, edgecolor=color, facecolor=CARD, alpha=alpha)
    ax.add_patch(r)
    ax.text(x + w/2, y + h/2 + 0.04, label, ha="center", va="center",
            color=color, fontsize=10, fontweight="bold")
    if sub:
        ax.text(x + w/2, y + h/2 - 0.28, sub, ha="center", va="center",
                color=MUTED, fontsize=7.5)

def arrow(ax, x1, y1, x2, y2, color=MUTED, lw=1.2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw))

def note(ax, x, y, text, color=MUTED, size=7.5, ha="left"):
    ax.text(x, y, text, ha=ha, va="center", color=color, fontsize=size)

# ── figure ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(12, 16))
ax.set_xlim(0, 12)
ax.set_ylim(0, 18)
ax.axis("off")

# ── Title ────────────────────────────────────────────────────────────────
ax.text(6, 17.2, "DeepSeek-V3-Lite", ha="center", va="center",
        fontsize=20, fontweight="bold", color=TEXT)
ax.text(6, 16.6, "757M Chinchilla-Scale Reproduction · Single A100 80GB",
        ha="center", va="center", fontsize=11, color=MUTED)

# ── Section: Input ──────────────────────────────────────────────────────
box(ax, 4.5, 15.0, 3, 0.7, BLUE, "Input Tokens", "vocab_size = 14,336")
arrow(ax, 6, 14.95, 6, 14.2)

# ── Section: Embedding ─────────────────────────────────────────────────
box(ax, 4.2, 13.3, 3.6, 0.8, CYAN, "Embedding", "14,336 × 1,024")
arrow(ax, 6, 13.25, 6, 12.5)

# ── Section: Dense Layers ──────────────────────────────────────────────
ax.text(1.5, 12.8, "Dense Block ×2", ha="center", va="center",
        fontsize=8.5, fontweight="bold", color=GREEN, rotation=90)

box(ax, 3.0, 11.3, 6, 1.0, GREEN, "MLA + SwiGLU", "Layers 0-1")
arrow(ax, 6, 11.25, 6, 10.5)

# ── Section: MoE Layers ────────────────────────────────────────────────
ax.text(1.5, 10.0, "MoE Block ×22", ha="center", va="center",
        fontsize=8.5, fontweight="bold", color=CORAL, rotation=90)

# MoE layer main block
box(ax, 3.0, 8.7, 6, 1.5, CORAL, "MLA + DeepSeekMoE", "Layers 2-23 (top-2 routing)")
arrow(ax, 6, 8.65, 6, 7.9)

# ── MoE detail callout ─────────────────────────────────────────────────
# Shared expert
r = FancyBboxPatch((9.5, 9.4), 2.2, 0.55, boxstyle="round,pad=0.06",
                    linewidth=1, edgecolor=PURPLE, facecolor=CARD)
ax.add_patch(r)
ax.text(10.6, 9.68, "Shared Expert", ha="center", va="center",
        color=PURPLE, fontsize=8, fontweight="bold")
ax.text(10.6, 9.45, "always active", ha="center", va="center",
        color=MUTED, fontsize=6.5)
arrow(ax, 9.0, 9.6, 9.5, 9.6, color=PURPLE, lw=1)

# Routed experts
for i in range(4):
    rx = 9.5 + i * 0.55
    c = CORAL if i < 2 else MUTED
    lw = 1.2 if i < 2 else 0.5
    al = 0.9 if i < 2 else 0.4
    rr = FancyBboxPatch((rx, 8.7), 0.45, 0.45, boxstyle="round,pad=0.04",
                         linewidth=lw, edgecolor=c, facecolor=CARD, alpha=al)
    ax.add_patch(rr)
    ax.text(rx + 0.23, 8.93, f"E{i+1}" if i < 2 else "…",
            ha="center", va="center", color=c, fontsize=7)
arrow(ax, 9.0, 9.0, 9.5, 9.0, color=CORAL, lw=1)

note(ax, 9.5, 8.45, "16 routed experts, top-2 per token", size=7)

# ── Section: Output ────────────────────────────────────────────────────
box(ax, 4.2, 6.7, 3.6, 0.8, TEXT, "RMSNorm", "eps = 1e-6")
arrow(ax, 6, 6.65, 6, 5.9)

box(ax, 4.2, 5.0, 3.6, 0.8, BLUE, "Linear Head", "14,336 → logits")
arrow(ax, 6, 4.95, 6, 4.2)

box(ax, 4.5, 3.3, 3, 0.7, GREEN, "Output Logits", "(B, S, 14,336)")

# ── MTP side panel ─────────────────────────────────────────────────────
# Vertical bar indicating MTP branch
ax.plot([2.2, 2.2], [10.8, 4.5], color=PURPLE, lw=1.5, ls="--", alpha=0.6)
ax.plot([2.2, 2.8], [5.8, 5.8], color=PURPLE, lw=1.5, ls="--", alpha=0.6)

# MTP block
r = FancyBboxPatch((0.3, 5.0), 2.2, 1.5, boxstyle="round,pad=0.1",
                    linewidth=1.5, edgecolor=PURPLE, facecolor=CARD)
ax.add_patch(r)
ax.text(1.4, 6.0, "MTP Module", ha="center", va="center",
        color=PURPLE, fontsize=9, fontweight="bold")
ax.text(1.4, 5.55, "Depth = 1", ha="center", va="center",
        color=MUTED, fontsize=7.5)
ax.text(1.4, 5.15, "Shared output head", ha="center", va="center",
        color=MUTED, fontsize=7)
ax.text(1.4, 4.8, "Predicts token t+2", ha="center", va="center",
        color=MUTED, fontsize=7)

note(ax, 0.3, 6.7, "Multi-Token Prediction", size=8, color=PURPLE, ha="left")

# ── Speculative decoding note ──────────────────────────────────────────
r = FancyBboxPatch((8.5, 3.0), 3.2, 1.0, boxstyle="round,pad=0.08",
                    linewidth=1, edgecolor=CYAN, facecolor=CARD)
ax.add_patch(r)
ax.text(10.1, 3.7, "Speculative Decoding", ha="center", va="center",
        color=CYAN, fontsize=8.5, fontweight="bold")
ax.text(10.1, 3.25, "MTP draft · acceptance ≥ 0.8", ha="center", va="center",
        color=MUTED, fontsize=7)

note(ax, 8.5, 4.2, "Inference", size=8, color=CYAN, ha="left")

# ── Key metrics panel ──────────────────────────────────────────────────
metrics = [
    ("Parameters", "757M"),
    ("Layers", "2 dense + 22 MoE"),
    ("KV cache / layer", "256-dim latent"),
    ("Precision", "BF16 + FP32 master"),
    ("Hardware", "1× A100 80GB"),
    ("Est. wall time", "~9 days"),
]

r = FancyBboxPatch((0.25, 0.3), 5.5, 2.2, boxstyle="round,pad=0.12",
                    linewidth=1, edgecolor=BORDER, facecolor=CARD)
ax.add_patch(r)
ax.text(0.5, 2.2, "Key Specifications", ha="left", va="center",
        color=TEXT, fontsize=9, fontweight="bold")

for i, (k, v) in enumerate(metrics):
    y = 1.8 - i * 0.3
    ax.text(0.5, y, k, ha="left", va="center", color=MUTED, fontsize=7.5)
    ax.text(3.8, y, v, ha="left", va="center", color=TEXT, fontsize=7.5,
            fontweight="bold")

# ── Training stages panel ──────────────────────────────────────────────
stages = [
    ("Pre-train", "15.1B tokens, Chinchilla-20", BLUE),
    ("SFT", "Sample-isolation masking", GREEN),
    ("GRPO", "Group size 4, PPO clip ε=0.2", CORAL),
    ("Distill", "KL(T=2) + CE, frozen teacher", PURPLE),
]

r = FancyBboxPatch((6.25, 0.3), 5.5, 2.2, boxstyle="round,pad=0.12",
                    linewidth=1, edgecolor=BORDER, facecolor=CARD)
ax.add_patch(r)
ax.text(6.5, 2.2, "Training Pipeline", ha="left", va="center",
        color=TEXT, fontsize=9, fontweight="bold")

for i, (name, desc, col) in enumerate(stages):
    y = 1.8 - i * 0.3
    ax.text(6.5, y, "●", ha="left", va="center", color=col, fontsize=8)
    ax.text(6.9, y, name, ha="left", va="center", color=col, fontsize=7.5,
            fontweight="bold")
    ax.text(6.9, y - 0.18, desc, ha="left", va="center", color=MUTED,
            fontsize=6.5)

# ── Footer ─────────────────────────────────────────────────────────────
ax.text(6, 0.05, "github.com/atandra2000/DeepSeek-V3-Lite",
        ha="center", va="center", color=MUTED, fontsize=8)

# ── Save ────────────────────────────────────────────────────────────────
out = "assets/architecture_overview.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
print(f"Saved → {out}")
