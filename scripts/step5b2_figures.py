"""
scripts/step5b2_figures.py

Generate 5 main paper figures from existing CSVs (no re-experiments).

Outputs: results/figures/{fig1..fig5}.{png,svg} + figure_captions.md
"""

import csv
import math
import os
import statistics as st
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#DDDDDD",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
})

NEW_BASE = str(Path(__file__).resolve().parent.parent / "results" / "new_paper_step5b2_multimetric") + "/"
V3_BASE  = str(Path(__file__).resolve().parent.parent / "results" / "v3_layer_baseline") + "/"
FIG_DIR  = str(Path(__file__).resolve().parent.parent / "results" / "figures") + "/"
os.makedirs(FIG_DIR, exist_ok=True)


def agg_csv(fp):
    """Return dict with mean/CI for delta_lp, delta_b_rank, and net@k."""
    if not os.path.exists(fp):
        return None
    rows = list(csv.DictReader(open(fp)))
    if not rows:
        return None
    out = {}
    for col in ("delta_lp", "delta_b_rank"):
        v = [float(r[col]) for r in rows]
        n = len(v); m = sum(v) / n
        sd = st.stdev(v) if n > 1 else 0
        se = sd / math.sqrt(n) if n > 1 else 0
        ci = 1.96 * se
        out[col] = (m, m - ci, m + ci)
    base_ranks = [int(r["first_rank_base"]) for r in rows]
    int_ranks = [int(r["first_rank_int"]) for r in rows]
    for k in (1, 10, 100):
        base_in = sum(1 for r in base_ranks if r <= k) / len(rows)
        int_in = sum(1 for r in int_ranks if r <= k) / len(rows)
        out[f"net@{k}"] = (int_in - base_in) * 100
    return out


def fp_for(base, model, layer, pii, tmpl, alpha_str, mode="target"):
    suffix = "" if mode == "target" else f"_{mode}"
    return f"{base}{model}_L{layer}_{pii}_{tmpl}_{alpha_str}{suffix}.csv"


def alpha_str_for(model):
    return "alpham2" if "gemma" in model else "alphap2"


def save_fig(fig, name, png_dpi=300):
    fig.savefig(os.path.join(FIG_DIR, f"{name}.png"), dpi=png_dpi, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, f"{name}.svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"  [save] {name}.png + .svg")


HERO_CONDS = [
    # (label, model, new_layer, v3_layer, pii, tmpl, color)
    ("G:name×T5",   "gemma_2_9b",    10, 40, "name",  "T5", "#1f77b4"),
    ("G:email×T1",  "gemma_2_9b",    10, 40, "email", "T1", "#1f77b4"),
    ("G:email×T4",  "gemma_2_9b",    10, 40, "email", "T4", "#1f77b4"),
    ("G:email×T5",  "gemma_2_9b",    10, 40, "email", "T5", "#1f77b4"),
    ("L:name×T0",   "llama3_med_8b", 25, 30, "name",  "T0", "#ff7f0e"),
    ("L:name×T1",   "llama3_med_8b", 25, 30, "name",  "T1", "#ff7f0e"),
    ("L:name×T5",   "llama3_med_8b", 25, 30, "name",  "T5", "#ff7f0e"),
    ("L:email×T8",  "llama3_med_8b", 25, 30, "email", "T8", "#ff7f0e"),
    ("L:id×T3 *",   "llama3_med_8b", 25, 30, "id",    "T3", "#d62728"),  # mismatch
]


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — V3 baseline vs new layer
# ─────────────────────────────────────────────────────────────────────────────

def fig1():
    print("\n[fig1] V3 baseline vs new layer ...")
    n_cond = len(HERO_CONDS)
    new_lp_means, new_lp_los, new_lp_his = [], [], []
    v3_lp_means, v3_lp_los, v3_lp_his = [], [], []
    new_br_means, new_br_los, new_br_his = [], [], []
    v3_br_means, v3_br_los, v3_br_his = [], [], []
    colors = []
    labels = []
    for lbl, model, new_l, v3_l, pii, t, color in HERO_CONDS:
        new = agg_csv(fp_for(NEW_BASE, model, new_l, pii, t, alpha_str_for(model)))
        v3  = agg_csv(fp_for(V3_BASE,  model, v3_l,  pii, t, alpha_str_for(model)))
        labels.append(lbl); colors.append(color)
        new_lp_means.append(new["delta_lp"][0]); new_lp_los.append(new["delta_lp"][1]); new_lp_his.append(new["delta_lp"][2])
        v3_lp_means.append(v3["delta_lp"][0]);   v3_lp_los.append(v3["delta_lp"][1]);   v3_lp_his.append(v3["delta_lp"][2])
        new_br_means.append(new["delta_b_rank"][0]); new_br_los.append(new["delta_b_rank"][1]); new_br_his.append(new["delta_b_rank"][2])
        v3_br_means.append(v3["delta_b_rank"][0]);   v3_br_los.append(v3["delta_b_rank"][1]);   v3_br_his.append(v3["delta_b_rank"][2])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(n_cond)
    width = 0.36

    # Panel A: Δ_lp
    ax = axes[0]
    new_err = [[m - lo for m, lo in zip(new_lp_means, new_lp_los)],
               [hi - m for m, hi in zip(new_lp_means, new_lp_his)]]
    v3_err = [[m - lo for m, lo in zip(v3_lp_means, v3_lp_los)],
              [hi - m for m, hi in zip(v3_lp_means, v3_lp_his)]]
    bars1 = ax.bar(x - width/2, new_lp_means, width, yerr=new_err,
                    label="New (L10/L25)", color=colors, edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width/2, v3_lp_means, width, yerr=v3_err,
                    label="V3 (L40/L30)", color="#cccccc", edgecolor="black", linewidth=0.5, alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel(r"$\Delta_{lp}$ (mean)")
    ax.set_title("(A) Logprob change at hero conditions")
    ax.legend(loc="lower left", framealpha=0.92)

    # Panel B: Δb_rank
    ax = axes[1]
    new_err = [[m - lo for m, lo in zip(new_br_means, new_br_los)],
               [hi - m for m, hi in zip(new_br_means, new_br_his)]]
    v3_err = [[m - lo for m, lo in zip(v3_br_means, v3_br_los)],
              [hi - m for m, hi in zip(v3_br_means, v3_br_his)]]
    ax.bar(x - width/2, new_br_means, width, yerr=new_err,
           label="New (L10/L25)", color=colors, edgecolor="black", linewidth=0.5)
    ax.bar(x + width/2, v3_br_means, width, yerr=v3_err,
           label="V3 (L40/L30)", color="#cccccc", edgecolor="black", linewidth=0.5, alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel(r"$\Delta b_{\rm rank}$ (bits)")
    ax.set_title(r"(B) $\Delta b_{\rm rank}$ at hero conditions")
    ax.legend(loc="upper left", framealpha=0.92)

    fig.suptitle("PCA-guided new layer dominates V3 final layer (same feature index)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    save_fig(fig, "fig1_v3_vs_new_layer")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Multi-metric heatmap (4 PII × 9 templates × 2 metrics × 2 models)
# ─────────────────────────────────────────────────────────────────────────────

def fig2():
    print("\n[fig2] Multi-metric heatmap ...")
    templates = ["T0","T1","T2","T3","T4","T5","T6","T7","T8"]
    piis = ["name","id","phone","email"]

    def fetch(model, layer, alpha_s):
        lp_grid = np.zeros((4, 9))
        br_grid = np.zeros((4, 9))
        sig_lp = np.zeros((4, 9), dtype=bool)
        sig_br = np.zeros((4, 9), dtype=bool)
        for i, p in enumerate(piis):
            for j, t in enumerate(templates):
                d = agg_csv(fp_for(NEW_BASE, model, layer, p, t, alpha_s))
                if d is None: continue
                lp_m, lp_lo, lp_hi = d["delta_lp"]
                br_m, br_lo, br_hi = d["delta_b_rank"]
                lp_grid[i, j] = lp_m
                br_grid[i, j] = br_m
                sig_lp[i, j] = (lp_hi < 0) or (lp_lo > 0)
                sig_br[i, j] = (br_hi < 0) or (br_lo > 0)
        return lp_grid, br_grid, sig_lp, sig_br

    g_lp, g_br, g_lp_sig, g_br_sig = fetch("gemma_2_9b", 10, "alpham2")
    l_lp, l_br, l_lp_sig, l_br_sig = fetch("llama3_med_8b", 25, "alphap2")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    def draw_heatmap(ax, grid, sig, title, cmap, vmin, vmax, fmt=".2f"):
        im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                v = grid[i, j]
                col = "white" if abs(v) > (vmax - vmin) * 0.3 + vmin else "black"
                txt = f"{v:{fmt}}"
                if sig[i, j]:
                    txt += "*"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=col)
        ax.set_xticks(range(9)); ax.set_xticklabels(templates)
        ax.set_yticks(range(4)); ax.set_yticklabels(piis)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
        # mismatch highlight: Llama id × T3
        if "Llama" in title and "rank" in title:
            ax.add_patch(Rectangle((2.5, 0.5), 1, 1, fill=False, edgecolor="black", linewidth=2.5))
        if "Llama" in title and "logprob" in title:
            ax.add_patch(Rectangle((2.5, 0.5), 1, 1, fill=False, edgecolor="black", linewidth=2.5))

    # Δ_lp: blue=suppress (negative), red=anti-suppress (positive)
    draw_heatmap(axes[0,0], g_lp, g_lp_sig, "Gemma L10 — logprob Δ", "RdBu", vmin=-5, vmax=1)
    draw_heatmap(axes[1,0], l_lp, l_lp_sig, "Llama L25 — logprob Δ", "RdBu", vmin=-5, vmax=1)
    # Δb_rank: blue=positive (rank ↑, good), red=negative (rank ↓, mismatch)
    draw_heatmap(axes[0,1], g_br, g_br_sig, r"Gemma L10 — $\Delta b_{\rm rank}$ (bits)",
                  "RdBu_r", vmin=-2, vmax=10)
    draw_heatmap(axes[1,1], l_br, l_br_sig, r"Llama L25 — $\Delta b_{\rm rank}$ (bits)",
                  "RdBu_r", vmin=-2, vmax=10)

    fig.suptitle("Cross-template multi-metric evaluation (val50; * = CI excludes 0)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_fig(fig, "fig2_multimetric_heatmap")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Llama × id × T3 mismatch
# ─────────────────────────────────────────────────────────────────────────────

def fig3():
    print("\n[fig3] Logprob-rank mismatch ...")
    model, layer_new, layer_v3, pii, t, asg = "llama3_med_8b", 25, 30, "id", "T3", "alphap2"
    hero = agg_csv(fp_for(NEW_BASE, model, layer_new, pii, t, asg))
    rand = agg_csv(fp_for(NEW_BASE, model, layer_new, pii, t, asg, "random_vector"))
    v3   = agg_csv(fp_for(V3_BASE,  model, layer_v3,  pii, t, asg))

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    groups = ["Hero\n(feat 22072\nL25, α=+2)", "Random\n(same |α|)", "V3 layer\n(L30, same\nfeat id)"]
    colors = ["#d62728", "#888888", "#cccccc"]

    def panel(ax, key, title, ylabel, ref0=True, units=""):
        means, lows, highs = [], [], []
        for d in (hero, rand, v3):
            m, lo, hi = d[key]
            means.append(m); lows.append(m - lo); highs.append(hi - m)
        bars = ax.bar(groups, means, color=colors, yerr=[lows, highs],
                       edgecolor="black", linewidth=0.5)
        if ref0:
            ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        # annotate values
        for bar, m in zip(bars, means):
            x = bar.get_x() + bar.get_width()/2
            y = m + (0.08 if m >= 0 else -0.18) * max(abs(min(means)), abs(max(means)), 0.5)
            ax.text(x, y, f"{m:+.2f}{units}", ha="center", fontsize=8)

    panel(axes[0], "delta_lp", "(A) Logprob Δ", r"$\Delta_{lp}$ (nat)")
    panel(axes[1], "delta_b_rank", "(B) Rank-cost Δ", r"$\Delta b_{\rm rank}$ (bits)")
    # highlight mismatch on panel B
    axes[1].annotate("MISMATCH:\nlogprob says suppress,\nrank says worse",
                      xy=(0, -1.45), xytext=(1.0, -2.6),
                      fontsize=9, color="#d62728", weight="bold",
                      arrowprops=dict(arrowstyle="->", color="#d62728"))

    # Net@10
    ax = axes[2]
    nets = [hero["net@10"], rand["net@10"], v3["net@10"]]
    bars = ax.bar(groups, nets, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("net Δ@10 (%)")
    ax.set_title(r"(C) Top-10 inclusion change")
    for bar, m in zip(bars, nets):
        ax.text(bar.get_x() + bar.get_width()/2, m + 0.3, f"{m:+.1f}%",
                ha="center", fontsize=8)
    ax.set_ylim(-5, 5)

    fig.suptitle("Llama × id × T3: feature-specific logprob-rank mismatch",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig3_logprob_rank_mismatch")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Selectivity scatter
# ─────────────────────────────────────────────────────────────────────────────

def fig4():
    print("\n[fig4] Selectivity scatter ...")
    fig, ax = plt.subplots(figsize=(7, 6))
    # selectivity zone shading
    ax.axhspan(5, 100, xmin=np.log10(5)/np.log10(100), xmax=1, color="#d4edda", alpha=0.35,
                label="feature-specific zone (≥5×)")

    sel_data = []
    for lbl, model, new_l, v3_l, pii, t, color in HERO_CONDS:
        hero = agg_csv(fp_for(NEW_BASE, model, new_l, pii, t, alpha_str_for(model)))
        rand = agg_csv(fp_for(NEW_BASE, model, new_l, pii, t, alpha_str_for(model), "random_vector"))
        if hero is None or rand is None:
            continue
        sel_lp = abs(hero["delta_lp"][0]) / max(abs(rand["delta_lp"][0]), 1e-3)
        sel_br = abs(hero["delta_b_rank"][0]) / max(abs(rand["delta_b_rank"][0]), 1e-3)
        is_mismatch = (lbl == "L:id×T3 *")
        marker = "*" if is_mismatch else ("s" if "G:" in lbl else "o")
        size = 220 if is_mismatch else 130
        ax.scatter(sel_lp, sel_br, s=size, marker=marker, color=color,
                    edgecolor="black", linewidth=0.6, zorder=5, alpha=0.92)
        # offset label
        dx, dy = 0.10, 0.05
        if "T8" in lbl: dx, dy = 0.15, -0.3
        if "T0" in lbl: dx, dy = -0.5, 0.15
        ax.annotate(lbl, (sel_lp, sel_br),
                    xytext=(sel_lp * (1 + dx), sel_br * (1 + dy)),
                    fontsize=8, color="black", zorder=6)
        sel_data.append((lbl, sel_lp, sel_br))

    ax.axhline(1, color="gray", linestyle=":", linewidth=0.8)
    ax.axvline(1, color="gray", linestyle=":", linewidth=0.8)
    ax.axhline(5, color="green", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.axvline(5, color="green", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.5, 100)
    ax.set_ylim(0.3, 30)
    ax.set_xlabel("Selectivity by Δ_lp  (|hero| / |random|, log scale)")
    ax.set_ylabel(r"Selectivity by $\Delta b_{\rm rank}$  (log scale)")
    ax.set_title("Hero vs random direction selectivity (val50)")
    # custom legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#1f77b4",
                markeredgecolor="black", markersize=10, label="Gemma L10 (feat 2759)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#ff7f0e",
                markeredgecolor="black", markersize=10, label="Llama L25 heroes"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#d62728",
                markeredgecolor="black", markersize=14, label="Llama L25 id T3 (MISMATCH)"),
    ]
    ax.legend(handles=handles, loc="lower right", framealpha=0.95)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "fig4_selectivity")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — PCA silhouette PT vs FT
# ─────────────────────────────────────────────────────────────────────────────

def fig5():
    print("\n[fig5] PCA silhouette PT vs FT ...")
    def load_sil(fp):
        rows = list(csv.DictReader(open(fp)))
        x = [int(r["layer"]) for r in rows]
        y = [float(r["silhouette"]) for r in rows]
        return x, y

    g_pt_x, g_pt_y = load_sil(str(Path(__file__).resolve().parent.parent / "results" / "gemma_2_9b/layer_silhouette_4way_PT.csv"))
    g_ft_x, g_ft_y = load_sil(str(Path(__file__).resolve().parent.parent / "results" / "gemma_2_9b/layer_silhouette_4way_FT.csv"))
    l_pt_x, l_pt_y = load_sil(str(Path(__file__).resolve().parent.parent / "results" / "llama3_med_8b/layer_silhouette_4way_PT.csv"))
    l_ft_x, l_ft_y = load_sil(str(Path(__file__).resolve().parent.parent / "results" / "llama3_med_8b/layer_silhouette_4way_FT.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    def panel(ax, pt_x, pt_y, ft_x, ft_y, title, ft_peak_x, pt_peak_x, v3_x):
        ax.plot(pt_x, pt_y, "o-", color="#888888", linewidth=1.5, markersize=6, label="PT (base)")
        ax.plot(ft_x, ft_y, "s-", color="#d62728", linewidth=1.8, markersize=7, label="FT (LoRA merged)")
        ax.axhline(0.3, color="black", linestyle="--", linewidth=0.6, alpha=0.7, label="threshold = 0.30")
        # markers
        ft_peak_y = ft_y[ft_x.index(ft_peak_x)]
        pt_peak_y = pt_y[pt_x.index(pt_peak_x)]
        v3_y = ft_y[ft_x.index(v3_x)] if v3_x in ft_x else None
        ax.scatter([ft_peak_x], [ft_peak_y], s=200, facecolor="white",
                    edgecolor="#d62728", linewidth=2.5, zorder=10)
        ax.annotate(f"FT peak\nL{ft_peak_x}\n(selected)",
                    (ft_peak_x, ft_peak_y), xytext=(ft_peak_x + 2, ft_peak_y + 0.04),
                    fontsize=8.5, color="#d62728",
                    arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2))
        ax.scatter([pt_peak_x], [pt_peak_y], s=120, facecolor="white",
                    edgecolor="#888888", linewidth=2, zorder=10)
        ax.annotate(f"PT peak\nL{pt_peak_x}",
                    (pt_peak_x, pt_peak_y), xytext=(pt_peak_x - 8, pt_peak_y - 0.05),
                    fontsize=8, color="#888888")
        if v3_y is not None:
            ax.scatter([v3_x], [v3_y], s=120, facecolor="white",
                        edgecolor="black", linewidth=1.5, zorder=10)
            ax.annotate(f"V3 layer\nL{v3_x}",
                        (v3_x, v3_y), xytext=(v3_x - 4, v3_y + 0.06),
                        fontsize=8, color="black",
                        arrowprops=dict(arrowstyle="->", color="black", lw=1))
        ax.set_xlabel("Layer")
        ax.set_ylabel("4-way silhouette score")
        ax.set_title(title)
        ax.set_ylim(0, 0.85)
        ax.legend(loc="lower left", framealpha=0.95)

    panel(axes[0], g_pt_x, g_pt_y, g_ft_x, g_ft_y, "Gemma-2-9B",
          ft_peak_x=10, pt_peak_x=20, v3_x=40)
    panel(axes[1], l_pt_x, l_pt_y, l_ft_x, l_ft_y, "Llama3-Med42-8B",
          ft_peak_x=25, pt_peak_x=15, v3_x=30)

    fig.suptitle("PCA-based layer selection: 4-way silhouette (name/id/phone/email) per layer",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    save_fig(fig, "fig5_pca_silhouette")


def write_captions():
    captions = """# Figure Captions

## Figure 1 — V3 baseline vs new (PCA-guided) layer
Bar comparison of hero conditions at the PCA-guided new layer (L10 for Gemma, L25 for Llama) vs the V3-paper SAE-final layer (L40 for Gemma, L30 for Llama), using the same feature index. (A) Mean Δ logprob over the gold PII span on val50 (n=50; error bars 95% CI). (B) Mean Δb_rank in bits. At the V3 layer the same feature index produces effects near zero — confirming the new layer's importance and that the SAE feature index is layer-specific. Supports the main claim that PCA-guided layer selection is decisive.

## Figure 2 — Multi-metric × cross-template heatmap
4 PII types × 9 templates (T0 = full training format, T1–T8 = OOD variants) heatmap of Δ logprob (left column) and Δb_rank (right column), Gemma top, Llama bottom. * marks conditions whose 95% CI excludes zero. Black box highlights the Llama × id × T3 logprob-rank mismatch: Δ_lp negative significant (left) coexists with negative significant Δb_rank (right). Supports the cross-template robustness analysis and the mismatch novelty.

## Figure 3 — Logprob-rank mismatch at Llama × id × T3
Three-panel comparison of (A) Δ_lp, (B) Δb_rank, (C) top-10 net Δ for the Llama × id × T3 condition under three regimes: hero feature 22072 at L25 (red), magnitude-matched random direction at L25 (gray), and the same feature index at the V3 layer L30 (light gray). Panel B shows the mismatch is feature-specific (hero −1.45 bits vs random ~0 vs V3 ~0); top-k is unaffected (panel C). Supports the main novelty claim.

## Figure 4 — Selectivity scatter (hero vs random)
Log-log scatter of selectivity (|hero| / |random|) computed on Δ_lp (x) and Δb_rank (y) for 8 hero conditions plus the Llama mismatch (star). Green region marks the ≥5× feature-specific zone. Gemma L10 conditions (feat 2759) cluster at 1.2–1.6× (magnitude artifact); Llama L25 heroes show 3–52× selectivity (true feature-specific). Supports the distributed-vs-collapsed representation comparison and the Gemma selectivity caveat.

## Figure 5 — PCA silhouette curves: PT vs FT per layer
4-way (name/id/phone/email) silhouette across layers using PT base activations (gray) and FT LoRA-merged activations (red). FT peak (L10 Gemma / L25 Llama) is the layer where PII-type discriminability sharply rises after fine-tuning. V3 layer (L40 Gemma / L30 Llama) is well past the FT peak. Layer selection methodology figure.
"""
    with open(os.path.join(FIG_DIR, "figure_captions.md"), "w", encoding="utf-8") as f:
        f.write(captions)
    print(f"  [save] figure_captions.md")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5()
    write_captions()
    print(f"\n[done] all figures saved to {FIG_DIR}")
