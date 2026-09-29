"""
scripts/fig_tasksae_selectivity_grid.py

Fig 5 — Task-SAE selectivity grid: per (model, layer, PII type).
2 rows × 2 cols:
  Row 1: Gemma Task-SAE (L9, L20, L31)
  Row 2: Llama Task-SAE (L8, L16, L24)
  Col 1: sel_lp (logprob selectivity)
  Col 2: sel_br (b_rank selectivity)

Each panel: grouped bar chart (4 PII × 3 layers) with 5× threshold line.
PCA-peak layer (Gemma L9, Llama L24) bolded.

Output: results/figures_final/fig5_tasksae_selectivity_grid.{png,pdf,svg}
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Data from full grid aggregation (avg over 9 templates × n=50)
GEMMA = {
    "L9":  {"name": (10.63, 27.32), "id": (3.94, 5.10),  "phone": (3.15, 11.15), "email": (12.01, 15.94)},
    "L20": {"name": (1.40, 2.60),   "id": (0.62, 3.67),  "phone": (0.48, 4.34),  "email": (1.09, 1.24)},
    "L31": {"name": (1.07, 0.59),   "id": (1.83, 1.20),  "phone": (2.77, 2.18),  "email": (4.78, 1.31)},
}
LLAMA = {
    "L8":  {"name": (0.80, 0.90),  "id": (0.95, 0.46),  "phone": (0.56, 0.66), "email": (0.24, 0.52)},
    "L16": {"name": (0.72, 1.11),  "id": (8.47, 6.94),  "phone": (0.75, 0.80), "email": (1.63, 1.21)},
    "L24": {"name": (7.46, 7.78),  "id": (2.81, 20.00), "phone": (3.68, 1.84), "email": (2.50, 0.61)},
}
PCA_PEAK = {"Gemma": "L9", "Llama": "L24"}

PII_ORDER = ["name", "id", "phone", "email"]
PII_COLORS = {
    "name":  "#1565C0",
    "id":    "#2E7D32",
    "phone": "#F57C00",
    "email": "#C62828",
}

OUT = Path(__file__).resolve().parent.parent / "results" / "figures_final"
OUT.mkdir(exist_ok=True, parents=True)

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11.5,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
})


def draw_panel(ax, data, metric_idx, title, peak_layer):
    """data = {layer_name: {pii: (sel_lp, sel_br)}}.  metric_idx: 0=lp, 1=br."""
    layers = list(data.keys())
    n_layers = len(layers)
    n_piis = len(PII_ORDER)
    bar_width = 0.78 / n_piis
    x_positions = np.arange(n_layers)

    for i, pii in enumerate(PII_ORDER):
        vals = [data[L][pii][metric_idx] for L in layers]
        offsets = (i - (n_piis - 1) / 2) * bar_width
        bars = ax.bar(x_positions + offsets, vals, bar_width,
                       color=PII_COLORS[pii], edgecolor="black", linewidth=0.4,
                       label=pii)
        # numeric labels above each bar
        for x, v in zip(x_positions + offsets, vals):
            label_y = max(v * 1.08, 0.13)
            ax.text(x, label_y, f"{v:.1f}", ha="center", va="bottom",
                    fontsize=7.5, color="black")

    # Threshold line at 5×
    ax.axhline(5.0, color="#444444", linestyle="--", linewidth=1.0,
               alpha=0.8, zorder=1, label="5× threshold")
    # Parity line at 1×
    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.7,
               alpha=0.5, zorder=1)
    ax.set_yscale("log")
    ax.set_ylim(0.1, 200)
    ax.set_xticks(x_positions)
    # Bold PCA peak layer label
    labels = []
    for L in layers:
        if L == peak_layer:
            labels.append(f"$\\bf{{{L}\\ (PCA\\ peak)}}$")
        else:
            labels.append(L)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_axisbelow(True)


fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), constrained_layout=True)

draw_panel(axes[0, 0], GEMMA, 0, "(a) Gemma Task-SAE — sel(Δlp)", PCA_PEAK["Gemma"])
draw_panel(axes[0, 1], GEMMA, 1, "(b) Gemma Task-SAE — sel(Δb_rank)", PCA_PEAK["Gemma"])
draw_panel(axes[1, 0], LLAMA, 0, "(c) Llama-Med42 Task-SAE — sel(Δlp)", PCA_PEAK["Llama"])
draw_panel(axes[1, 1], LLAMA, 1, "(d) Llama-Med42 Task-SAE — sel(Δb_rank)", PCA_PEAK["Llama"])

# Common labels
for ax in axes[0]:
    ax.set_xlabel("")
for ax in axes[1]:
    ax.set_xlabel("Layer")
for ax in axes[:, 0]:
    ax.set_ylabel("Selectivity  (log scale)")
for ax in axes[:, 1]:
    ax.set_ylabel("")

# Single shared legend at top
handles, labels = axes[0, 0].get_legend_handles_labels()
# keep "name","id","phone","email","5× threshold" (skip duplicates)
seen = set()
unique = []
for h, l in zip(handles, labels):
    if l not in seen:
        seen.add(l); unique.append((h, l))
fig.legend([h for h, _ in unique], [l for _, l in unique],
           loc="upper center", bbox_to_anchor=(0.5, 1.04),
           ncol=5, frameon=False, fontsize=10)

fig.suptitle("Task-SAE Selectivity Grid — 4 PII × 3 layers × 2 models (avg over 9 templates, n=50)",
             fontsize=13, fontweight="bold", y=1.075)

for ext in ("png", "pdf", "svg"):
    fig.savefig(OUT / f"fig5_tasksae_selectivity_grid.{ext}",
                dpi=300 if ext == "png" else None, bbox_inches="tight")
plt.close(fig)
print(f"[save] {OUT}/fig5_tasksae_selectivity_grid.{{png,pdf,svg}}")
