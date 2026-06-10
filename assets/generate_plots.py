"""
Architecture overview chart for DeepSeek-V3-Lite.
6-panel dark-themed figure — no training required.
Run: python assets/generate_plots.py
Output: assets/architecture_overview.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np

# ── theme ─────────────────────────────────────────────────────────────────────
BG      = "#0d1117"
PANEL   = "#161b22"
BORDER  = "#30363d"
ACCENT1 = "#58a6ff"   # blue
ACCENT2 = "#f78166"   # coral
ACCENT3 = "#3fb950"   # green
ACCENT4 = "#d2a8ff"   # lavender
ACCENT5 = "#ffa657"   # orange
ACCENT6 = "#79c0ff"   # light blue
TEXT    = "#e6edf3"
MUTED   = "#8b949e"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   TEXT,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "text.color":        TEXT,
    "grid.color":        BORDER,
    "grid.alpha":        0.5,
    "font.family":       "monospace",
    "font.size":         9,
})

fig = plt.figure(figsize=(20, 14), facecolor=BG)
fig.suptitle("DeepSeek-V3-Lite  ·  Chinchilla-Scale Architecture  ·  ~800M params",
             fontsize=18, fontweight="bold", color=TEXT, y=0.97)

gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32,
                      left=0.06, right=0.97, top=0.92, bottom=0.05)

axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
for ax in axes:
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)

# ── helper ────────────────────────────────────────────────────────────────────
def title(ax, t):
    ax.set_title(t, color=TEXT, fontsize=10, fontweight="bold", pad=8)

# ══════════════════════════════════════════════════════════════════════════════
# Panel 0 — Parameter / component breakdown
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[0]
title(ax, "Model Parameter Budget")

components = [
    "Token Embedding\n(14 336 × 1024)",
    "MLA (× 24 layers)",
    "MoE FFN\n(16 routed + 1 shared, × 23)",
    "Dense FFN\n(layer 0)",
    "MTP Module\n(depth = 1)",
    "Output Head\n(shared)",
]
params_B = [0.7, 0.5, 2.3, 0.3, 0.3, 0.0]   # ~757M total
colors   = [ACCENT1, ACCENT2, ACCENT3, ACCENT4, ACCENT5, ACCENT6]

y = np.arange(len(components))
bars = ax.barh(y, params_B, color=colors, height=0.55, edgecolor=BORDER)
for bar, v in zip(bars, params_B):
    if v > 0:
        ax.text(v + 0.15, bar.get_y() + bar.get_height()/2,
                f"{v:.2f}B", va="center", color=TEXT, fontsize=8)

ax.set_yticks(y)
ax.set_yticklabels(components, fontsize=7.5)
ax.set_xlabel("Parameters (millions)", color=MUTED)
ax.set_xlim(0, 12)
ax.grid(axis="x", ls="--")
ax.text(0.97, 0.04, "Total active  ≈ 757.0M", ha="right", va="bottom",
        transform=ax.transAxes, color=ACCENT3, fontsize=8,
        fontweight="bold")

# ══════════════════════════════════════════════════════════════════════════════
# Panel 1 — MLA KV-cache compression
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[1]
title(ax, "Multi-Head Latent Attention  —  KV Cache Compression")

seq_lens = np.array([512, 1024, 2048, 4096, 8192])
n_heads, head_dim = 16, 128
kv_rank = 512

standard_MB = seq_lens * n_heads * head_dim * 2 * 2 / 1e6        # BF16 each KV
mla_MB      = seq_lens * kv_rank * 2 / 1e6                       # latent c_kv

ax.fill_between(seq_lens, standard_MB, alpha=0.25, color=ACCENT2)
ax.fill_between(seq_lens, mla_MB,      alpha=0.25, color=ACCENT3)
ax.plot(seq_lens, standard_MB, color=ACCENT2, lw=2, label="MHA standard KV cache")
ax.plot(seq_lens, mla_MB,      color=ACCENT3, lw=2, label="MLA compressed  (kv_rank=512)")

ax.set_xlabel("Sequence length (tokens)", color=MUTED)
ax.set_ylabel("KV cache  (MB / layer)", color=MUTED)
ax.set_xscale("log", base=2)
ax.set_xticks(seq_lens)
ax.set_xticklabels([str(s) for s in seq_lens])
ax.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)
ax.grid(ls="--")

ratio = standard_MB[-1] / mla_MB[-1]
ax.text(0.97, 0.97, f"Compression  ≈ {ratio:.0f}×  at 8 k ctx",
        ha="right", va="top", transform=ax.transAxes,
        color=ACCENT3, fontsize=8, fontweight="bold")

# ══════════════════════════════════════════════════════════════════════════════
# Panel 2 — MoE routing diagram
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[2]
title(ax, "DeepSeekMoE  —  Expert Routing (per token)")
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis("off")

# Token box
trect = mpatches.FancyBboxPatch((3.5, 6.3), 3, 0.9,
    boxstyle="round,pad=0.1", linewidth=1.5,
    edgecolor=ACCENT1, facecolor=PANEL)
ax.add_patch(trect)
ax.text(5, 6.75, "Input Token  x", ha="center", va="center",
        color=ACCENT1, fontsize=9, fontweight="bold")

# Gate
grect = mpatches.FancyBboxPatch((3.8, 4.9), 2.4, 0.8,
    boxstyle="round,pad=0.1", linewidth=1.5,
    edgecolor=ACCENT5, facecolor=PANEL)
ax.add_patch(grect)
ax.text(5, 5.3, "AuxLossFree Gate\n(biased sigmoid)", ha="center", va="center",
        color=ACCENT5, fontsize=7.5)
ax.annotate("", xy=(5, 5.7), xytext=(5, 6.3),
            arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.4))

# Shared experts
for i, x in enumerate([1.5, 2.8]):
    r = mpatches.FancyBboxPatch((x-0.5, 2.2), 1.1, 0.9,
        boxstyle="round,pad=0.08", linewidth=1,
        edgecolor=ACCENT4, facecolor=PANEL)
    ax.add_patch(r)
    ax.text(x+0.05, 2.65, f"Shared\nExp {i+1}", ha="center", va="center",
            color=ACCENT4, fontsize=7)
    ax.annotate("", xy=(x+0.05, 3.1), xytext=(4.5, 4.9),
                arrowprops=dict(arrowstyle="->", color=ACCENT4, lw=1, alpha=0.7))

# Routed experts (top-6 highlighted)
routed_x = np.linspace(3.8, 9.2, 8)
for i, rx in enumerate(routed_x):
    active = i < 6
    color  = ACCENT2 if active else MUTED
    lw     = 1.5 if active else 0.6
    r = mpatches.FancyBboxPatch((rx-0.32, 2.2), 0.7, 0.9,
        boxstyle="round,pad=0.06", linewidth=lw,
        edgecolor=color, facecolor=PANEL)
    ax.add_patch(r)
    ax.text(rx+0.03, 2.65, f"R{i+1}" if i<6 else f"··", ha="center", va="center",
            color=color, fontsize=6.5)
    if active:
        ax.annotate("", xy=(rx+0.03, 3.1), xytext=(5.5, 4.9),
                    arrowprops=dict(arrowstyle="->", color=ACCENT2, lw=0.9, alpha=0.6))

ax.text(0.5, 1.9, "Shared (always):", color=ACCENT4, fontsize=7.5, style="italic")
ax.text(3.8, 1.9, "Routed (top-6 / 64 activated):", color=ACCENT2, fontsize=7.5, style="italic")

# Aggregate
arect = mpatches.FancyBboxPatch((3.5, 0.5), 3, 0.85,
    boxstyle="round,pad=0.1", linewidth=1.5,
    edgecolor=ACCENT3, facecolor=PANEL)
ax.add_patch(arect)
ax.text(5, 0.93, "Sum  →  FFN output", ha="center", va="center",
        color=ACCENT3, fontsize=8.5)
ax.annotate("", xy=(5, 1.4), xytext=(5, 2.2),
            arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.4))

# ══════════════════════════════════════════════════════════════════════════════
# Panel 3 — BF16 + SDPA + torch.compile pipeline
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[3]
title(ax, "BF16 Single-GPU Training Pipeline")
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis("off")

stages = [
    (5, 6.2, "BF16 Activations\n(forward input)", ACCENT1),
    (5, 4.6, "BF16 Linear\ncuBLAS SGEMM", ACCENT5),
    (5, 3.0, "F.scaled_dot_product_attention\n(Flash-Attn on A100)", ACCENT2),
    (5, 1.4, "BF16 Output  +  AdamW FP32 State\n→ BF16 master weights", ACCENT3),
]
for (x, y, label, col) in stages:
    r = mpatches.FancyBboxPatch((x-2.8, y-0.55), 5.6, 1.0,
        boxstyle="round,pad=0.1", linewidth=1.5,
        edgecolor=col, facecolor=PANEL)
    ax.add_patch(r)
    ax.text(x, y, label, ha="center", va="center",
            color=col, fontsize=8)

for yi in [5.65, 4.05, 2.45]:
    ax.annotate("", xy=(5, yi - 0.2), xytext=(5, yi),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.4))

ax.text(0.03, 0.03, "Memory  ≈ 3.7 GB (bs=32, seq=1024, grad-ckpt)\nThroughput target: >10k tok/s on A100",
        transform=ax.transAxes, va="bottom", color=MUTED, fontsize=7.5)

# ══════════════════════════════════════════════════════════════════════════════
# Panel 4 — Training pipeline stages
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[4]
title(ax, "Full Training Pipeline")

stages_info = [
    ("Pre-training",     "FineWeb-Edu + Stack v2 + MATH\nBF16  ·  SDPA  ·  MTP loss = 0.3",   ACCENT1),
    ("SFT",              "Chat templates  ·  sample-isolation mask\nCE loss on completions only",    ACCENT3),
    ("GRPO  (RL)",       "Group size 8  ·  PPO clip ε=0.2\nRule + model rewards  ·  KL=0.04",  ACCENT5),
    ("R1 Distillation",  "KL + CE  ·  temperature=0.7\nFrozen deepseek-r1-distill teacher",    ACCENT2),
    ("Speculative Dec.", "MTP draft head  ·  acceptance ≥ 0.8\n1 draft token per step",         ACCENT4),
]
y_pos = np.arange(len(stages_info))[::-1] * 1.15
for i, (name, desc, col) in enumerate(stages_info):
    ax.barh(y_pos[i], 1, color=col, alpha=0.85, height=0.6,
            left=0, edgecolor=BORDER)
    ax.text(1.08, y_pos[i], f"  {name}", va="center", color=col,
            fontsize=8.5, fontweight="bold")
    ax.text(1.08, y_pos[i] - 0.28, f"     {desc}", va="center",
            color=MUTED, fontsize=7)

ax.set_xlim(0, 8)
ax.set_ylim(-0.6, 5.5)
ax.axis("off")

arrows = [(y_pos[i]+0.3, y_pos[i+1]-0.3) for i in range(len(stages_info)-1)]
for y1, y2 in arrows:
    ax.annotate("", xy=(0.5, y2), xytext=(0.5, y1),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))

# ══════════════════════════════════════════════════════════════════════════════
# Panel 5 — Key hyperparameters table
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[5]
title(ax, "Key Hyperparameters")
ax.axis("off")

rows = [
    ("Architecture",     "24-layer (2 dense + 22 MoE)"),
    ("dim / heads",      "1024  / 16"),
    ("MoE experts",      "16 routed + 1 shared, top-2"),
    ("kv_lora_rank",     "256"),
    ("qk_rope_head_dim", "32  (decoupled RoPE)"),
    ("vocab size",       "14 336"),
    ("Max seq len",      "1 024"),
    ("Precision",        "BF16 forward + FP32 master"),
    ("Batch (effective)","4 micro × 8 grad_acc × 1 GPU = 32"),
    ("Learning rate",    "8.4e-5  (WarmupCosine)"),
    ("MTP weight",       "0.3"),
    ("Hardware",         "1 × NVIDIA A100 80GB SXM"),
    ("Total params",     "757.0M"),
    ("Status",           "Implementation  /  pre-training pending"),
]

col_labels = ["Parameter", "Value"]
col_x      = [0.02, 0.45]

ax.text(col_x[0], 0.97, col_labels[0], transform=ax.transAxes,
        color=ACCENT1, fontsize=8.5, fontweight="bold", va="top")
ax.text(col_x[1], 0.97, col_labels[1], transform=ax.transAxes,
        color=ACCENT1, fontsize=8.5, fontweight="bold", va="top")
ax.plot([0, 1], [0.935, 0.935], color=BORDER, lw=1,
        transform=ax.transAxes, clip_on=False)

for i, (k, v) in enumerate(rows):
    y = 0.895 - i * 0.059
    bg = PANEL if i % 2 == 0 else "#1c2128"
    ax.axhspan(y - 0.008, y + 0.045, color=bg,
               transform=ax.transAxes, zorder=0)
    col = ACCENT3 if k == "Status" else TEXT
    ax.text(col_x[0], y + 0.018, k, transform=ax.transAxes,
            color=MUTED, fontsize=7.8, va="center")
    ax.text(col_x[1], y + 0.018, v, transform=ax.transAxes,
            color=col, fontsize=7.8, va="center")

# ── footer ────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.005,
         "DeepSeek-V3-Lite  ·  Atandra Bharati  ·  github.com/atandra2000/DeepSeek-V3-Lite",
         ha="center", color=MUTED, fontsize=8)

out = "assets/architecture_overview.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved → {out}")
