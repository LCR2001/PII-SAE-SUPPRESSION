"""
scripts/fig_tasksae_categorical.py

Task-SAE intervention 결과를 'intervention 효과의 실제 quality' 기준으로 시각화.

각 (model, layer, PII) cell에서 4가지 metric을 보여줌:
  • Δlp (logprob change, 음수=suppression)
  • Δbr (b_rank change, 양수=suppression)
  • sel_lp / sel_br (selectivity)
  • category: clean / br-only mismatch / lp-only / anti / noise

Layout: 2 rows × 2 cols
  Row 1: Gemma  Row 2: Llama
  Col 1: Δlp (with category color)
  Col 2: Δbr (with category color)

각 panel은 grouped bar chart (PII 4종 × layer 3개).
bar 색상 = category, 위에 sel_lp / sel_br 표시.

Output: results/figures_final/fig6_tasksae_intervention_quality.{png,pdf,svg}
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# (Δlp, Δbr, sel_lp, sel_br) from full grid aggregation
GEMMA = {
    "L9":  {"name":  (-3.675, +6.999, 10.63, 27.32),
            "id":    (-1.073, +5.452, 3.94, 5.10),
            "phone": (-0.929, +15.527, 3.15, 11.15),
            "email": (-0.717, +10.397, 12.01, 15.94)},
    "L20": {"name":  (-1.617, +2.954, 1.40, 2.60),
            "id":    (+0.007, +1.305, 0.62, 3.67),
            "phone": (-0.144, +4.311, 0.48, 4.34),
            "email": (+0.090, +2.300, 1.09, 1.24)},
    "L31": {"name":  (-2.954, +2.308, 1.07, 0.59),
            "id":    (-2.597, +7.597, 1.83, 1.20),
            "phone": (-2.179, +18.878, 2.77, 2.18),
            "email": (-3.392, +14.173, 4.78, 1.31)},
}
LLAMA = {
    "L8":  {"name":  (-1.346, +1.960, 0.80, 0.90),
            "id":    (-0.795, +2.529, 0.95, 0.46),
            "phone": (-0.838, +4.743, 0.56, 0.66),
            "email": (-0.228, +2.440, 0.24, 0.52)},
    "L16": {"name":  (-0.584, +0.912, 0.72, 1.11),
            "id":    (-0.319, -0.634, 8.47, 6.94),
            "phone": (-0.581, +3.016, 0.75, 0.80),
            "email": (-0.892, +3.386, 1.63, 1.21)},
    "L24": {"name":  (-0.032, +0.046, 7.46, 7.78),
            "id":    (-0.545, +0.785, 2.81, 20.00),
            "phone": (-0.337, +1.517, 3.68, 1.84),
            "email": (-0.295, +0.490, 2.50, 0.61)},
}
PCA_PEAK = {"Gemma": "L9", "Llama": "L24"}
PII_ORDER = ["name", "id", "phone", "email"]

# Category thresholds
LP_THRESH = -0.3
BR_THRESH = +1.0

CAT_COLORS = {
    "clean-strong":   "#2E7D32",  # 진한 녹색: clean both + sel≥5×
    "clean-weak":     "#A5D6A7",  # 옅은 녹색: clean both but sel<5× (magnitude artifact 의심)
    "br-only":        "#FBC02D",  # 노란색: br-only mismatch (lp NS, br suppress)
    "lp-only":        "#FB8C00",  # 주황: lp-only mismatch (lp suppress, br NS)
    "anti":           "#C62828",  # 빨간색: anti-suppression
    "noise":          "#BDBDBD",  # 회색: noise
}


def categorize(dlp, dbr, sel_lp, sel_br):
    """단순 threshold-based category (no CI here)."""
    lp_sup  = dlp <= LP_THRESH
    br_sup  = dbr >= BR_THRESH
    lp_anti = dlp >  +0.1
    br_anti = dbr <  -0.3
    if lp_anti or br_anti:
        return "anti"
    if lp_sup and br_sup:
        # Strong only if both sel ≥ 5×
        if (sel_lp is not None and sel_br is not None
                and sel_lp >= 5 and sel_br >= 5):
            return "clean-strong"
        return "clean-weak"
    if br_sup and not lp_sup:
        return "br-only"
    if lp_sup and not br_sup:
        return "lp-only"
    return "noise"


OUT = Path(__file__).resolve().parent.parent / "results" / "figures_final"
OUT.mkdir(exist_ok=True, parents=True)

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
})


def draw_panel(ax, data, metric, peak_layer, ylabel, ref_line, ref_label, ylim):
    """data: {layer: {pii: (dlp, dbr, sel_lp, sel_br)}}
    metric: "dlp" or "dbr"
    """
    layers = list(data.keys())
    n_layers = len(layers)
    n_piis = len(PII_ORDER)
    bar_width = 0.78 / n_piis
    x_positions = np.arange(n_layers)
    idx = 0 if metric == "dlp" else 1

    for i, pii in enumerate(PII_ORDER):
        vals = []
        colors = []
        sels = []
        for L in layers:
            dlp, dbr, sl, sb = data[L][pii]
            cat = categorize(dlp, dbr, sl, sb)
            vals.append(dlp if metric == "dlp" else dbr)
            colors.append(CAT_COLORS[cat])
            sels.append(sl if metric == "dlp" else sb)
        offsets = (i - (n_piis - 1) / 2) * bar_width
        bars = ax.bar(x_positions + offsets, vals, bar_width,
                       color=colors, edgecolor="black", linewidth=0.5)
        # Annotate selectivity above (or below if negative) each bar
        for x, v, s in zip(x_positions + offsets, vals, sels):
            label_y = v + (0.06 * abs(ylim[1] - ylim[0]) if v >= 0
                            else -0.06 * abs(ylim[1] - ylim[0]))
            va = "bottom" if v >= 0 else "top"
            # Color-code selectivity number
            sel_color = "#1B5E20" if s >= 5 else ("#FF6F00" if s >= 1 else "#B71C1C")
            ax.text(x, label_y, f"sel={s:.1f}×", ha="center", va=va,
                    fontsize=8.5, color=sel_color, rotation=0,
                    fontweight="bold")
        # PII label inside bar (larger now)
        for x, v in zip(x_positions + offsets, vals):
            inside_y = v * 0.5 if abs(v) > 0.5 * abs(ylim[1] - ylim[0]) * 0.05 else (
                ylim[0] + 0.04 * (ylim[1] - ylim[0]))
            ax.text(x, inside_y, pii[0].upper(),
                    ha="center", va="center", fontsize=8.5, color="white",
                    fontweight="bold", alpha=0.95)

    ax.axhline(ref_line, color="black", linestyle="--", linewidth=0.9,
               alpha=0.7, label=ref_label)
    ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.5)
    ax.set_ylim(*ylim)
    ax.set_xticks(x_positions)
    labels = [f"$\\bf{{{L}\\ (peak)}}$" if L == peak_layer else L for L in layers]
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)


fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5), constrained_layout=True)

draw_panel(axes[0, 0], GEMMA, "dlp", PCA_PEAK["Gemma"],
           ylabel=r"$\Delta_{\mathrm{lp}}$  (nat)",
           ref_line=LP_THRESH, ref_label=f"suppression threshold ({LP_THRESH:+.1f})",
           ylim=(-5.5, 1.5))
axes[0, 0].set_title("(a) Gemma — Δlp per (layer × PII)")

draw_panel(axes[0, 1], GEMMA, "dbr", PCA_PEAK["Gemma"],
           ylabel=r"$\Delta_{\mathrm{br}}$  (bits)",
           ref_line=BR_THRESH, ref_label=f"suppression threshold (+{BR_THRESH:.1f})",
           ylim=(-2.5, 25))
axes[0, 1].set_title("(b) Gemma — Δb_rank per (layer × PII)")

draw_panel(axes[1, 0], LLAMA, "dlp", PCA_PEAK["Llama"],
           ylabel=r"$\Delta_{\mathrm{lp}}$  (nat)",
           ref_line=LP_THRESH, ref_label=f"suppression threshold ({LP_THRESH:+.1f})",
           ylim=(-2.0, 1.0))
axes[1, 0].set_title("(c) Llama-Med42 — Δlp per (layer × PII)")

draw_panel(axes[1, 1], LLAMA, "dbr", PCA_PEAK["Llama"],
           ylabel=r"$\Delta_{\mathrm{br}}$  (bits)",
           ref_line=BR_THRESH, ref_label=f"suppression threshold (+{BR_THRESH:.1f})",
           ylim=(-2.5, 8))
axes[1, 1].set_title("(d) Llama-Med42 — Δb_rank per (layer × PII)")

for ax in axes[1]:
    ax.set_xlabel("Layer")

# Shared legend
cat_legend = [
    mpatches.Patch(color=CAT_COLORS["clean-strong"], label="✅ clean & sel ≥ 5×"),
    mpatches.Patch(color=CAT_COLORS["clean-weak"],   label="✅ clean but sel < 5×"),
    mpatches.Patch(color=CAT_COLORS["br-only"],      label="⚠️ br-only mismatch"),
    mpatches.Patch(color=CAT_COLORS["lp-only"],      label="⚠️ lp-only"),
    mpatches.Patch(color=CAT_COLORS["anti"],         label="🔴 ANTI-suppression"),
    mpatches.Patch(color=CAT_COLORS["noise"],        label="🔘 noise"),
]
fig.legend(handles=cat_legend, loc="upper center",
           bbox_to_anchor=(0.5, 1.045), ncol=6, frameon=False, fontsize=9.5)

fig.suptitle("Task-SAE Intervention Quality — Bar height = Δlp or Δbr value, color = category, 'sel=X×' = selectivity vs random direction",
             fontsize=12, fontweight="bold", y=1.075)

for ext in ("png", "pdf", "svg"):
    fig.savefig(OUT / f"fig6_tasksae_intervention_quality.{ext}",
                dpi=300 if ext == "png" else None, bbox_inches="tight")
plt.close(fig)
print(f"[save] {OUT}/fig6_tasksae_intervention_quality.{{png,pdf,svg}}")

# Also dump categorization table
print("\nCategorization summary:")
print(f"{'Model':<7} {'Layer':>5} {'PII':>6} {'Δlp':>9} {'Δbr':>9} {'sel_lp':>7} {'sel_br':>7} | category")
print("-"*90)
for tag, data in [("Gemma", GEMMA), ("Llama", LLAMA)]:
    for L, row in data.items():
        for pii in PII_ORDER:
            dlp, dbr, sl, sb = row[pii]
            cat = categorize(dlp, dbr, sl, sb)
            print(f"{tag:<7} {L:>5} {pii:>6} {dlp:>+9.3f} {dbr:>+9.3f} {sl:>7.2f} {sb:>7.2f} | {cat}")
