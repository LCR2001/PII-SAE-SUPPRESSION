"""Unified Figure 2 & 3 for all 3 models (Gemma + Llama-3.1-Inst + Qwen3-8B)."""
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

ROOT = Path(__file__).resolve().parent.parent / "results"
SUMMARY = ROOT / "new_x8_v2_full_grid_coherent.csv"
OUT = ROOT / "figures_unified_viz"
OUT.mkdir(exist_ok=True)

PII_ORDER = ["name", "id", "phone", "email"]
# (model, layer, short_label, color, marker)
CELLS = [
    ("gemma_2_9b", 9,  "Gemma L9",   "#1f77b4", "o"),
    ("gemma_2_9b", 20, "Gemma L20",  "#ff7f0e", "o"),
    ("gemma_2_9b", 31, "Gemma L31",  "#2ca02c", "o"),
    ("llama3_1_8b_instruct", 8, "Llama-3.1 L8", "#9467bd", "s"),
    ("qwen3_8b", 9,    "Qwen L9",    "#d62728", "^"),
]

rows = []
with open(SUMMARY) as f:
    for r in csv.DictReader(f):
        rows.append(dict(
            model=r["model"], layer=int(r["layer"]), pii=r["pii_type"],
            dlp_t=float(r["dlp_t"]), dbr_t=float(r["dbr_t"]),
            dlp_r=float(r["dlp_r"]), dbr_r=float(r["dbr_r"]),
            dlp_t_ci=float(r["dlp_t_ci"]), dbr_t_ci=float(r["dbr_t_ci"]),
            dlp_r_ci=float(r["dlp_r_ci"]), dbr_r_ci=float(r["dbr_r_ci"])))


def get_row(model, layer, pii):
    return next(x for x in rows if x["model"] == model and x["layer"] == layer and x["pii"] == pii)


# ============================================================
# FIGURE 2: bars (20 cells)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(22, 5.5))
labels = []
for m, L, lbl, _, _ in CELLS:
    for p in PII_ORDER:
        labels.append(f"{lbl}\n{p}")

for ax, metric, ytitle in zip(
    axes,
    [("dlp_t", "dlp_r", "dlp_t_ci", "dlp_r_ci"),
     ("dbr_t", "dbr_r", "dbr_t_ci", "dbr_r_ci")],
    ["|$\\Delta_{lp}$|  (log-prob change magnitude)",
     "|$\\Delta_{br}$|  (rank-cost change magnitude)"]
):
    t_key, r_key, t_ci, r_ci = metric
    t_vals, r_vals, t_cis, r_cis = [], [], [], []
    for m, L, _, _, _ in CELLS:
        for p in PII_ORDER:
            row = get_row(m, L, p)
            t_vals.append(abs(row[t_key]))
            r_vals.append(abs(row[r_key]))
            t_cis.append(row[t_ci])
            r_cis.append(row[r_ci])
    x = np.arange(20)
    w = 0.36
    ax.bar(x - w/2, t_vals, w, yerr=t_cis,
           label="target feature", color="#2E7D32",
           capsize=2.5, edgecolor="black", linewidth=0.4)
    ax.bar(x + w/2, r_vals, w, yerr=r_cis,
           label="random direction (3 seeds)", color="#C62828",
           capsize=2.5, edgecolor="black", linewidth=0.4)
    # Group dividers between models
    for k in [4, 8, 12, 16]:
        ax.axvline(k - 0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5, rotation=0)
    ax.set_ylabel(ytitle)
    ax.legend(loc="upper left", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
axes[0].set_title("All 3 models × 20 cells: target feature vs random direction (3 seeds)")
plt.tight_layout()
plt.savefig(OUT / "fig2_unified_magnitude.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved fig2_unified_magnitude.png")

# ============================================================
# FIGURE 3: scatter (20 cells)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
for ax, (tk, rk), title, show_legend in zip(
    axes,
    [("dlp_t", "dlp_r"), ("dbr_t", "dbr_r")],
    ["$\\Delta_{lp}$ magnitude", "$\\Delta_{br}$ magnitude"],
    [True, False]
):
    all_xs, all_ys = [], []
    for m, L, lbl, color, marker in CELLS:
        xs, ys, names = [], [], []
        for p in PII_ORDER:
            row = get_row(m, L, p)
            xs.append(abs(row[rk]))
            ys.append(abs(row[tk]))
            names.append(p)
        ax.scatter(xs, ys, s=110, color=color, label=lbl,
                   marker=marker, edgecolors="black", linewidth=0.7,
                   zorder=3, alpha=0.85)
        for xv, yv, n in zip(xs, ys, names):
            ax.annotate(n, (xv, yv), textcoords="offset points",
                        xytext=(6, 5), fontsize=7.5, alpha=0.8)
        all_xs += xs; all_ys += ys
    mx = max(max(all_xs), max(all_ys)) * 1.12
    ax.plot([0, mx], [0, mx], 'k--', alpha=0.6, linewidth=1.2,
            label="target = random (sel = 1)", zorder=1)
    ax.plot([0, mx/5], [0, mx], 'r:', alpha=0.6, linewidth=1.2,
            label="target = 5×random (clean-strong)", zorder=1)
    ax.set_xlabel(f"|{title}|  —  random direction")
    ax.set_ylabel(f"|{title}|  —  target feature")
    ax.set_xlim(0, mx); ax.set_ylim(0, mx)
    ax.set_aspect("equal", adjustable="box")
    if show_legend:
        ax.legend(loc="lower right", framealpha=0.95, fontsize=8,
                  bbox_to_anchor=(mx, mx * 0.02), bbox_transform=ax.transData)
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_title(title)
fig.suptitle("All 3 models × 20 cells: most points fall below y=x diagonal; none reach y=5x", y=1.02)
plt.tight_layout()
plt.savefig(OUT / "fig3_unified_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved fig3_unified_scatter.png")

# Also save log-scale variant for fig 3 to handle wide range
fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
for ax, (tk, rk), title in zip(
    axes,
    [("dlp_t", "dlp_r"), ("dbr_t", "dbr_r")],
    ["$\\Delta_{lp}$ magnitude", "$\\Delta_{br}$ magnitude"]
):
    all_xs, all_ys = [], []
    for m, L, lbl, color, marker in CELLS:
        xs, ys, names = [], [], []
        for p in PII_ORDER:
            row = get_row(m, L, p)
            xs.append(max(abs(row[rk]), 1e-3))
            ys.append(max(abs(row[tk]), 1e-3))
            names.append(p)
        ax.scatter(xs, ys, s=110, color=color, label=lbl,
                   marker=marker, edgecolors="black", linewidth=0.7,
                   zorder=3, alpha=0.85)
        for xv, yv, n in zip(xs, ys, names):
            ax.annotate(n, (xv, yv), textcoords="offset points",
                        xytext=(6, 5), fontsize=7.5, alpha=0.8)
        all_xs += xs; all_ys += ys
    mx = max(max(all_xs), max(all_ys)) * 1.4
    mn = min(min(all_xs), min(all_ys)) * 0.5
    ax.plot([mn, mx], [mn, mx], 'k--', alpha=0.6, linewidth=1.2,
            label="target = random (sel = 1)", zorder=1)
    ax.plot([mn, mx/5], [mn*5, mx], 'r:', alpha=0.6, linewidth=1.2,
            label="target = 5×random (clean-strong)", zorder=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(f"|{title}|  —  random direction  (log scale)")
    ax.set_ylabel(f"|{title}|  —  target feature  (log scale)")
    ax.set_xlim(mn, mx); ax.set_ylim(mn, mx)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left", framealpha=0.95, fontsize=8)
    ax.grid(alpha=0.3, linestyle="--", which="both")
    ax.set_title(title)
fig.suptitle("All 3 models × 20 cells (log scale) — covers small Gemma L9 and large L20 simultaneously", y=1.02)
plt.tight_layout()
plt.savefig(OUT / "fig3_unified_scatter_log.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved fig3_unified_scatter_log.png")
print(f"\nUnified figures saved to: {OUT}")
