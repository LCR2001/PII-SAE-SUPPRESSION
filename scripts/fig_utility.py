"""
scripts/fig_utility.py

Generate fig4_pubmedqa_utility — 4 hero conditions × PubMedQA accuracy
preservation (baseline vs hook-on) + off-target logprob shift magnitude.

Inputs: results/exp5_pubmedqa/{gemma,llama}_{task,pt}_*.csv
Output: results/figures_final/fig4_pubmedqa_utility.{png,pdf,svg}
"""

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "exp5_pubmedqa"
OUT = ROOT / "results" / "figures_final"

C_PT = "#9E9E9E"
C_TASK = "#1565C0"

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "errorbar.capsize": 2.5,
})

CONDITIONS = [
    dict(short="Gemma\ntask-SAE\nL9 f=20",
         file="gemma_task_L9_f20_alpham2.csv", sae="task", color=C_TASK),
    dict(short="Gemma\nPT SAE\nL10 f=2759",
         file="gemma_pt_L10_f2759_alpham2.csv", sae="pt",   color=C_PT),
    dict(short="Llama\ntask-SAE\nL24 f=1523",
         file="llama_task_L24_f1523_alpham2.csv", sae="task", color=C_TASK),
    dict(short="Llama\nPT SAE\nL25 f=22072",
         file="llama_pt_L25_f22072_alphap2.csv", sae="pt",   color=C_PT),
]


def summarise(df):
    n = len(df)
    base_acc = (df["base_pred"] == df["gt"]).mean()
    int_acc = (df["int_pred"] == df["gt"]).mean()
    p_base = base_acc; p_int = int_acc
    se_base = math.sqrt(p_base * (1 - p_base) / n)
    se_int = math.sqrt(p_int * (1 - p_int) / n)
    # off-target lp shift (mean |Δlp| across the three labels)
    dlp = pd.concat([
        (df["int_lp_yes"] - df["base_lp_yes"]).abs(),
        (df["int_lp_no"] - df["base_lp_no"]).abs(),
        (df["int_lp_maybe"] - df["base_lp_maybe"]).abs(),
    ])
    flipped = (df["base_pred"] != df["int_pred"]).mean()
    return dict(n=n, base_acc=base_acc, int_acc=int_acc,
                se_base=1.96 * se_base, se_int=1.96 * se_int,
                mean_dlp=float(dlp.mean()), flipped=float(flipped))


def main():
    data = []
    for c in CONDITIONS:
        df = pd.read_csv(SRC / c["file"])
        s = summarise(df)
        s.update(c)
        data.append(s)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)

    # Panel (a): PubMedQA accuracy preservation
    ax = axes[0]
    x = np.arange(len(data))
    width = 0.38
    base_vals = [d["base_acc"] * 100 for d in data]
    int_vals = [d["int_acc"] * 100 for d in data]
    base_err = [d["se_base"] * 100 for d in data]
    int_err = [d["se_int"] * 100 for d in data]
    ax.bar(x - width / 2, base_vals, width, yerr=base_err,
           color="#E0E0E0", edgecolor="black", linewidth=0.5,
           label="baseline (no hook)", capsize=3)
    ax.bar(x + width / 2, int_vals, width, yerr=int_err,
           color=[d["color"] for d in data], edgecolor="black", linewidth=0.5,
           label="hook on (intervention)", capsize=3)
    for i, d in enumerate(data):
        delta = (d["int_acc"] - d["base_acc"]) * 100
        ax.text(i, max(d["base_acc"], d["int_acc"]) * 100 + 4,
                f"Δ = {delta:+.1f}pp",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.axhline(50, color="black", linestyle=":", linewidth=0.7, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([d["short"] for d in data], fontsize=8.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("PubMedQA accuracy (%)")
    ax.set_title("(a) Utility preservation — PubMedQA (n=500)")
    ax.legend(loc="lower left", framealpha=0.92)
    ax.set_axisbelow(True)

    # Panel (b): off-target |Δlp| and prediction flip rate
    ax = axes[1]
    ax2 = ax.twinx()
    bars = ax.bar(x, [d["mean_dlp"] for d in data], width=0.55,
                  color=[d["color"] for d in data],
                  edgecolor="black", linewidth=0.5, alpha=0.85,
                  label="mean |Δlp| over {yes,no,maybe}")
    ax2.plot(x, [d["flipped"] * 100 for d in data], "o-",
             color="#D84315", markersize=8, linewidth=1.5,
             label="prediction flip rate (%)")
    max_dlp = max(d["mean_dlp"] for d in data)
    for i, d in enumerate(data):
        # Bar value annotation — inside large bars (white), above small bars (black)
        if d["mean_dlp"] > max_dlp * 0.3:
            ax.text(i, d["mean_dlp"] * 0.5, f"{d['mean_dlp']:.2f}",
                    ha="center", va="center", fontsize=9.5,
                    color="white", fontweight="bold")
        else:
            ax.text(i, d["mean_dlp"] + max_dlp * 0.03,
                    f"{d['mean_dlp']:.2f}",
                    ha="center", va="bottom", fontsize=9)
        # Flip-rate annotation — to the right of marker (avoids collision)
        ax2.annotate(f"{d['flipped']*100:.1f}%",
                     xy=(i, d["flipped"] * 100), xytext=(7, -2),
                     textcoords="offset points", ha="left", va="center",
                     color="#D84315", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([d["short"] for d in data], fontsize=8.5)
    ax.set_ylabel(r"mean $|\Delta_{\mathrm{lp}}|$  (nat)")
    ax2.set_ylabel("prediction flip rate (%)", color="#D84315")
    ax2.tick_params(axis="y", colors="#D84315")
    ax2.grid(False)
    ax.set_ylim(0, max(0.5, max(d["mean_dlp"] for d in data) * 1.4))
    ax2.set_ylim(0, max(5, max(d["flipped"] * 100 for d in data) * 2.0))
    ax.set_title("(b) Off-target intervention magnitude")
    ax.set_axisbelow(True)
    # combined legend
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2,
              loc="upper right", framealpha=0.92, fontsize=8.5)

    fig.suptitle("External Utility — PubMedQA (test, n=500) under hero PII interventions",
                 fontsize=13, fontweight="bold")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"fig4_pubmedqa_utility.{ext}",
                    dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {OUT}/fig4_pubmedqa_utility.{{png,pdf,svg}}")

    # Print summary table for caption / paper text
    print("\nSummary table:")
    print(f"  {'condition':<32} {'n':>4} {'base':>7} {'int':>7} {'Δacc':>7} "
          f"{'flipped':>8} {'|Δlp|':>7}")
    for d in data:
        s = d["short"].replace("\n", " ")
        print(f"  {s:<32} {d['n']:>4} {d['base_acc']*100:>6.1f}% "
              f"{d['int_acc']*100:>6.1f}% {(d['int_acc']-d['base_acc'])*100:>+6.1f} "
              f"{d['flipped']*100:>7.1f}% {d['mean_dlp']:>7.3f}")


if __name__ == "__main__":
    main()
