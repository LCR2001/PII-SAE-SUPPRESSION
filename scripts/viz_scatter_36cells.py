"""36-cell scatter: |Δlp_target| vs |Δlp_random|  +  |Δbr_target| vs |Δbr_random|.

All 3 models × 3 layers × 4 PII = 36 cells, log scale.
Shape = model, Color = PII type.
clean-strong cell gets black border + bold annotation.
anti cells get gray 'x' marker overlay.
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

ROOT    = Path(__file__).resolve().parent.parent / "results"
CSV     = ROOT / "new_x8_v2_full_grid_coherent.csv"
OUT_DIR = ROOT / "figures_unified_viz"
OUT_DIR.mkdir(exist_ok=True)

PII_ORDER  = ["name", "id", "phone", "email"]
PII_COLORS = {
    "name":  "#1f77b4",   # blue
    "id":    "#ff7f0e",   # orange
    "phone": "#2ca02c",   # green
    "email": "#d62728",   # red
}

MODEL_INFO = {
    "gemma_2_9b":           ("o", "Gemma-2-9B",        [9, 20, 31]),
    "llama3_1_8b_instruct": ("s", "Llama-3.1-8B-Inst", [8, 16, 24]),
    "qwen3_8b":             ("^", "Qwen3-8B",           [9, 18, 27]),
}

# Load
rows = []
with open(CSV) as f:
    for r in csv.DictReader(f):
        rows.append(r)


def get(model, layer, pii, key):
    for r in rows:
        if r["model"] == model and int(r["layer"]) == layer and r["pii_type"] == pii:
            v = r[key]
            return float(v) if v not in ("", None) else None
    return None


def get_cat(model, layer, pii):
    for r in rows:
        if r["model"] == model and int(r["layer"]) == layer and r["pii_type"] == pii:
            return r["category"]
    return "noise"


# ── Figure ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))

METRICS = [
    ("dlp_t", "dlp_r", r"$|\Delta_\mathrm{lp}|$  target", r"$|\Delta_\mathrm{lp}|$  random",
     r"$\Delta_{lp}$ magnitude (log scale)"),
    ("dbr_t", "dbr_r", r"$|\Delta_\mathrm{br}|$  target", r"$|\Delta_\mathrm{br}|$  random",
     r"$\Delta_{br}$ magnitude (log scale)"),
]

for ax, (tk, rk, ylabel, xlabel, title) in zip(axes, METRICS):
    all_xs, all_ys = [], []

    for model, (marker, mlabel, layers) in MODEL_INFO.items():
        mshort = mlabel.split("-")[0]   # "Gemma" / "Llama" / "Qwen"
        for layer in layers:
            for pii in PII_ORDER:
                tv = get(model, layer, pii, tk)
                rv = get(model, layer, pii, rk)
                if tv is None or rv is None:
                    continue

                x = max(abs(rv), 1e-4)
                y = max(abs(tv), 1e-4)
                all_xs.append(x)
                all_ys.append(y)

                cat = get_cat(model, layer, pii)
                is_cs   = (cat == "clean-strong")
                is_anti = (cat == "anti")

                ec  = "black" if is_cs else ("none" if not is_anti else "#555555")
                lw  = 2.0     if is_cs else (0.5 if is_anti else 0.3)
                sz  = 160     if is_cs else 90
                alp = 0.95    if is_cs else (0.55 if is_anti else 0.80)

                ax.scatter(x, y,
                           s=sz,
                           color=PII_COLORS[pii],
                           marker=marker,
                           edgecolors=ec,
                           linewidth=lw,
                           zorder=4 if is_cs else (2 if is_anti else 3),
                           alpha=alp)

                # Annotate clean-strong
                if is_cs:
                    ax.annotate(
                        f"L{layer}/{pii}\n({mshort})",
                        (x, y),
                        textcoords="offset points",
                        xytext=(10, 5),
                        fontsize=8.5,
                        fontweight="bold",
                        color="black",
                        zorder=6,
                        arrowprops=dict(arrowstyle="-", color="black",
                                        lw=0.8, alpha=0.6),
                    )

    mn = min(all_xs + all_ys) * 0.35
    mx = max(all_xs + all_ys) * 3.0

    # sel = 1  (y = x)
    ax.plot([mn, mx], [mn, mx], color="dimgray", linestyle="--",
            linewidth=1.3, alpha=0.6, label="sel = 1  (target = random)", zorder=1)
    # sel = 5  (y = 5x)
    ax.plot([mn, mx / 5], [mn * 5, mx], color="#e6550d", linestyle="--",
            linewidth=1.5, alpha=0.75, label="sel = 5  (clean-strong threshold)", zorder=1)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(mn, mx)
    ax.set_ylim(mn, mx)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel + "  (log scale)", fontsize=10)
    ax.set_ylabel(ylabel + "  (log scale)", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(False)

# ── Legend ───────────────────────────────────────────────────────────────────
shape_handles = [
    Line2D([0], [0], marker=m, color="gray", linestyle="None",
           markersize=9, markeredgecolor="gray", label=lbl)
    for _, (m, lbl, _) in MODEL_INFO.items()
]
color_handles = [
    Line2D([0], [0], marker="o", color=PII_COLORS[p], linestyle="None",
           markersize=9, markeredgewidth=0, label=p)
    for p in PII_ORDER
]
cs_handle = Line2D([0], [0], marker="o", color="gray", linestyle="None",
                   markersize=11, markeredgecolor="black", markeredgewidth=2.0,
                   label="clean-strong ★")
ref_handles = [
    Line2D([0], [0], color="dimgray", linestyle="--", linewidth=1.3, label="sel = 1"),
    Line2D([0], [0], color="#e6550d", linestyle="--", linewidth=1.5, label="sel = 5 (CS threshold)"),
]

axes[1].legend(
    handles=shape_handles + color_handles + [cs_handle] + ref_handles,
    loc="lower right",
    framealpha=0.95,
    fontsize=8.5,
    title="Shape = model  |  Color = PII type",
    title_fontsize=8.5,
)

# Category count annotation on left panel
cats = {}
for r in rows:
    cats[r["category"]] = cats.get(r["category"], 0) + 1
cat_str = "\n".join([
    f"clean-strong: {cats.get('clean-strong',0)}/36",
    f"clean-weak:   {cats.get('clean-weak',0)}/36",
    f"anti:         {cats.get('anti',0)}/36",
    f"noise/other:  {36 - cats.get('clean-strong',0) - cats.get('clean-weak',0) - cats.get('anti',0)}/36",
])
axes[0].text(0.03, 0.97, cat_str, transform=axes[0].transAxes,
             fontsize=8, va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="lightgray", alpha=0.9))

fig.suptitle(
    "All 36 cells (3 models × 3 layers × 4 PII types) — coherence-matched random control\n"
    "Points above diagonal: target feature > random;  above dotted red: clean-strong (sel ≥ 5)",
    fontsize=11, y=1.01,
)

plt.tight_layout()
out = OUT_DIR / "fig3_scatter_36cells_log.png"
plt.savefig(out, dpi=180, bbox_inches="tight")
plt.close()
print(f"saved: {out}")
