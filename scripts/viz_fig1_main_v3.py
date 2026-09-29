"""EMNLP Figure 1 — v3.

Layout  :  [left col (a)]  [right col: (b) / (c) / (d) equal-height stacked]
Legend  :  removed — y-tick labels "Selected"/"Random" serve as inline legend
(b)/(c) :  both compact dot-strip (same height, same format)
(d)     :  compact 4-bar horizontal
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

mpl.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          7.5,
    "axes.titlesize":     7.5,
    "axes.labelsize":     7.0,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.labelsize":    6.5,
    "ytick.labelsize":    6.5,
})

# ── colours ───────────────────────────────────────────────────────────────────
C_FAIL = "#D0D0D0"
C_S0   = "#969696"
C_S1   = "#FDAE6B"
C_S2   = "#E6550D"
C_S3   = "#238B45"
C_TGT  = "#2166AC"
C_RND  = "#969696"
PII_COLORS = {
    "name":  "#1f77b4",
    "id":    "#ff7f0e",
    "phone": "#2ca02c",
    "email": "#d62728",
}

OUT = Path(__file__).resolve().parent.parent / "results" / "figures_final"

# ── data ──────────────────────────────────────────────────────────────────────
L16_TGT = np.array([-1.1526,-1.1545,-1.3370,-1.3462,-1.0781,
                    -1.1349,-1.0467,-0.6688,-0.6205])
L16_RND = np.array([-0.00030,-0.00035,-0.00041,-0.00038,-0.00074,
                    +0.00059,-0.00070,-0.00158,-0.00090])

G20_TGT = np.array([-2.5281,-2.5093,-2.4148,-2.3755,-3.0422,
                    -3.4591,-2.7601,-3.1522,-3.0023])
G20_RND = np.array([-7.5475,-6.7869,-7.4497,-7.3778,-7.0462,
                    -7.6683,-7.7953,-7.1344,-8.9078])

L8_PII = [("name", 4.57), ("id", 1.96), ("phone", 1.95), ("email", 1.97)]

# ── layout ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(6.9, 3.10))
gs = fig.add_gridspec(
    1, 2,
    width_ratios=[1.08, 1.85],
    left=0.06, right=0.99,
    bottom=0.09, top=0.93,
    wspace=0.44,
)
ax_a = fig.add_subplot(gs[0])

gs_r = gs[1].subgridspec(3, 1, hspace=1.40)
ax_b = fig.add_subplot(gs_r[0])
ax_c = fig.add_subplot(gs_r[1])
ax_d = fig.add_subplot(gs_r[2])

# ════════════════════════════════════════════════════════════════════════════
# PANEL (a) — stepwise filtering funnel (4-bar)
# ════════════════════════════════════════════════════════════════════════════
TOTAL = 36
stages = [
    (3, "All cells",                                              36, C_S0),
    (2, r"$\Delta_{lp}{\leq}{-0.3}$",                            17, C_S1),
    (1, r"$+\,\Delta_{br}{\geq}1.0$",                            14, C_S2),
    (0, r"$+\,\mathrm{sel}_{lp}\&\mathrm{sel}_{br}{\geq}5$",     1, C_S3),
]

for y, lbl, n, col in stages:
    ax_a.barh(y, TOTAL - n, left=n, height=0.52,
              color=C_FAIL, edgecolor="white", linewidth=0.5)
    ax_a.barh(y, n, height=0.52,
              color=col, edgecolor="white", linewidth=0.5)
    pct = n / TOTAL * 100
    pct_fmt = f"{pct:.1f}%" if pct < 5 else f"{pct:.0f}%"
    if n > 3:
        txt_col = "#333333" if col in (C_S0, C_S1) else "white"
        ax_a.text(n - 0.8, y, f"{n}  ({pct_fmt})",
                  va="center", ha="right", fontsize=6.5,
                  color=txt_col, fontweight="bold")
    else:
        ax_a.text(n + 0.8, y, f"{n}  ({pct_fmt})",
                  va="center", ha="left", fontsize=6.5,
                  color=C_S3, fontweight="bold")

ax_a.set_xlim(0, 36)
ax_a.set_ylim(-0.55, 3.55)
ax_a.set_xlabel("Number of cells  (out of 36)", fontsize=6.5)
ax_a.set_yticks([3, 2, 1, 0])
ax_a.set_yticklabels(
    ["All cells",
     r"$\Delta_{lp}{\leq}{-0.3}$",
     r"$+\,\Delta_{br}{\geq}1.0$",
     r"$+\,\mathrm{sel}{\geq}5$ (both)"],
    fontsize=6.2)
ax_a.xaxis.set_tick_params(labelsize=6.2)
ax_a.spines["left"].set_visible(False)
ax_a.tick_params(left=False)
ax_a.set_title("(a) Single-metric success collapses\nunder multi-metric evaluation",
               fontsize=7.2, pad=3)
ax_a.axvline(TOTAL, color="lightgray", linewidth=0.6, linestyle=":")

# ════════════════════════════════════════════════════════════════════════════
# helper: compact dot-strip
# ════════════════════════════════════════════════════════════════════════════
FS = dict(title=6.6, ytick=5.8, xlabel=5.8, annot=5.6)

def dot_strip(ax, tgt, rnd, title, annot,
              xlim=None, show_xlabel=False, annot_side="right", title_pad=2):
    RNG = np.random.default_rng(42)
    jit = lambda n: RNG.uniform(-0.09, 0.09, n)

    ax.scatter(tgt, np.ones(len(tgt))  + jit(len(tgt)),
               color=C_TGT, s=9, zorder=3, alpha=0.90, linewidths=0)
    ax.scatter(rnd, np.zeros(len(rnd)) + jit(len(rnd)),
               color=C_RND, s=9, zorder=3, alpha=0.90, linewidths=0)

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Random", "Selected"], fontsize=FS["ytick"])
    ax.set_ylim(-0.52, 1.52)
    if xlim:
        ax.set_xlim(xlim)
    ax.axvline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.35)
    if show_xlabel:
        ax.set_xlabel(r"$\Delta_{lp}$", fontsize=FS["xlabel"])
    ax.set_title(title, fontsize=FS["title"], pad=title_pad)

    ha = annot_side
    ax_x = 0.97 if ha == "right" else 0.03
    ax.text(ax_x, 0.97, annot,
            transform=ax.transAxes, fontsize=FS["annot"],
            va="top", ha=ha,
            bbox=dict(boxstyle="round,pad=0.22", fc="white",
                      ec="lightgray", alpha=0.9))
    ax.grid(axis="x", alpha=0.20, linestyle="--", linewidth=0.5)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)

# ════════════════════════════════════════════════════════════════════════════
# PANEL (b) — Llama L16/name  positive control
# ════════════════════════════════════════════════════════════════════════════
dot_strip(
    ax_b, L16_TGT, L16_RND,
    title="(b) Positive control: Llama L16 / name",
    annot="sel$_{lp}$ = 1,995\n(9/9 templates)",
    xlim=(-1.60, 0.22),
    show_xlabel=False,
    annot_side="right",
)

# ════════════════════════════════════════════════════════════════════════════
# PANEL (c) — Failure mode A: Gemma L20/name
# ════════════════════════════════════════════════════════════════════════════
dot_strip(
    ax_c, G20_TGT, G20_RND,
    title="(c) Failure mode A: Gemma L20 / name",
    annot="Random > selected\n(9/9 templates)",
    xlim=(-10.5, 0.6),
    show_xlabel=True,
    annot_side="left",
    title_pad=6,
)

# ════════════════════════════════════════════════════════════════════════════
# PANEL (d) — Failure mode B: Llama L8 feature 16363
# ════════════════════════════════════════════════════════════════════════════
pii_labels = [p for p, _ in L8_PII]
vals_bar   = np.array([v for _, v in L8_PII])
colors_bar = [PII_COLORS[p] for p in pii_labels]
y_pos = np.arange(len(L8_PII))

ax_d.barh(y_pos, vals_bar, height=0.52,
          color=colors_bar, alpha=0.88)
ax_d.set_yticks(y_pos)
ax_d.set_yticklabels(pii_labels, fontsize=5.8)
ax_d.set_xlabel(r"$|\Delta_{lp}|$  (selected feature)", fontsize=FS["xlabel"])
ax_d.spines["left"].set_visible(False)
ax_d.tick_params(left=False)
ax_d.set_title("(d) Failure mode B: Llama L8  (all 4 PII types → same feature)",
               fontsize=FS["title"], pad=6)

for i, v in enumerate(vals_bar):
    ax_d.text(v + 0.07, i, f"{v:.2f}",
              va="center", fontsize=5.0, color="dimgray")

ax_d.text(0.97, 0.97,
          "f=16363 for all PII;\nname: 2.3× larger",
          transform=ax_d.transAxes, fontsize=5.2,
          va="top", ha="right",
          bbox=dict(boxstyle="round,pad=0.20", fc="white",
                    ec="lightgray", alpha=0.9))
ax_d.set_xlim(0, 6.6)
ax_d.xaxis.set_tick_params(labelsize=5.8)

# ── save ──────────────────────────────────────────────────────────────────────
for ext in ["png", "pdf"]:
    out = OUT / f"fig1_main_diagnostic_v3.{ext}"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"saved: {out}")
plt.close()
