"""Appendix Figure A1 — α sensitivity v3.

Data  : x8_v2 task-SAE experiments (main paper setup)
Cells : (a) Llama-3.1-8B-Inst L16/name  f=32619  clean-strong
         (b) Gemma-2-9B        L20/name  f=20385  failure mode A
α     : {-1, -2, -3}  ×  templates {T1, T5}  ×  50 pairs (validation)
Note  : α=-2 is the main-paper setting — highlighted with ★ and shaded band.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

mpl.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          8.0,
    "axes.titlesize":     8.5,
    "axes.labelsize":     8.0,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.labelsize":    7.5,
    "ytick.labelsize":    7.5,
    "legend.fontsize":    7.5,
})

ROOT = Path(__file__).resolve().parent.parent / "results" / "exp1_alpha_sensitivity"
OUT  = Path(__file__).resolve().parent.parent / "results" / "figures_final"

# third cell — Qwen L18/name (clean-weak; crossover between α=-2 and -3)
qwen_dir    = ROOT / "x8v2_qwen_L18_name"
qwen_prefix = "qwen3_8b_L18_name"

C_TGT = "#2166AC"   # selected feature — blue
C_RND = "#969696"   # random direction — gray

ALPHAS   = [-1, -2, -3, -4, -5]
ALPHA_MAIN = -2          # main-paper α — highlighted
TEMPLATES  = ["T1", "T5"]


# ── data loading ──────────────────────────────────────────────────────────────

def load_delta_lp(data_dir: Path, prefix: str,
                  alpha: float, mode: str) -> list[float]:
    """Return per-pair Δlp values pooled across T1 and T5."""
    a_str    = f"m{abs(int(alpha))}"
    mode_tag = "" if mode == "target" else "_random_vector"
    vals = []
    for tpl in TEMPLATES:
        fname = data_dir / f"{prefix}_{tpl}_alpha{a_str}{mode_tag}.csv"
        if fname.exists():
            vals.extend(pd.read_csv(fname)["delta_lp"].tolist())
    return vals


def per_template_mean(data_dir: Path, prefix: str,
                      alpha: float, mode: str) -> list[float]:
    """Return per-template means (one value per template) for error estimation."""
    a_str    = f"m{abs(int(alpha))}"
    mode_tag = "" if mode == "target" else "_random_vector"
    means = []
    for tpl in TEMPLATES:
        fname = data_dir / f"{prefix}_{tpl}_alpha{a_str}{mode_tag}.csv"
        if fname.exists():
            means.append(pd.read_csv(fname)["delta_lp"].mean())
    return means


def build_series(data_dir: Path, prefix: str):
    """Return (means_tgt, sems_tgt, means_rnd, sems_rnd) for ALPHAS."""
    means_t, sems_t, means_r, sems_r = [], [], [], []
    for a in ALPHAS:
        # Use all-pair pool for mean; per-template for SEM
        pool_t = load_delta_lp(data_dir, prefix, a, "target")
        pool_r = load_delta_lp(data_dir, prefix, a, "random")
        tmeans_t = per_template_mean(data_dir, prefix, a, "target")
        tmeans_r = per_template_mean(data_dir, prefix, a, "random")
        means_t.append(np.mean(pool_t))
        means_r.append(np.mean(pool_r))
        # SEM across templates (n=2); use std of per-template means / sqrt(n)
        sems_t.append(np.std(tmeans_t) / np.sqrt(len(tmeans_t)) if len(tmeans_t) > 1 else 0.0)
        sems_r.append(np.std(tmeans_r) / np.sqrt(len(tmeans_r)) if len(tmeans_r) > 1 else 0.0)
    return (np.array(means_t), np.array(sems_t),
            np.array(means_r), np.array(sems_r))


def sel_lp_at_main(data_dir: Path, prefix: str) -> float | None:
    """Selectivity at α=ALPHA_MAIN, averaged over templates."""
    sels = []
    a_str = f"m{abs(int(ALPHA_MAIN))}"
    for tpl in TEMPLATES:
        ft = data_dir / f"{prefix}_{tpl}_alpha{a_str}.csv"
        fr = data_dir / f"{prefix}_{tpl}_alpha{a_str}_random_vector.csv"
        if ft.exists() and fr.exists():
            mt = pd.read_csv(ft)["delta_lp"].mean()
            mr = pd.read_csv(fr)["delta_lp"].mean()
            if abs(mr) > 1e-4:
                sels.append(abs(mt) / abs(mr))
    return float(np.mean(sels)) if sels else None


# ── load ──────────────────────────────────────────────────────────────────────

llama_dir    = ROOT / "x8v2_llama_inst_L16_name"
llama_prefix = "llama3_1_8b_instruct_L16_name"
gemma_dir    = ROOT / "x8v2_gemma_L20_name"
gemma_prefix = "gemma_2_9b_L20_name"

l_mt, l_st, l_mr, l_sr = build_series(llama_dir, llama_prefix)
g_mt, g_st, g_mr, g_sr = build_series(gemma_dir, gemma_prefix)
q_mt, q_st, q_mr, q_sr = build_series(qwen_dir,  qwen_prefix)

llama_sel = sel_lp_at_main(llama_dir, llama_prefix)
gemma_sel = sel_lp_at_main(gemma_dir, gemma_prefix)
qwen_sel  = sel_lp_at_main(qwen_dir,  qwen_prefix)


# ── plot ──────────────────────────────────────────────────────────────────────

fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(8.5, 2.7),
                                        gridspec_kw={"wspace": 0.45})
fig.subplots_adjust(left=0.07, right=0.98, bottom=0.20, top=0.78)
fig.suptitle(r"$\alpha$ sensitivity: diagnostic conclusions are robust across $\alpha$ magnitudes",
             fontsize=8.5, y=0.97)

x_pos = np.arange(len(ALPHAS))
main_idx = ALPHAS.index(ALPHA_MAIN)   # index of α=-2 in x_pos


def draw_panel(ax, mt, st, mr, sr, title, sel_str,
               ylim=None, show_ylabel=True, annot_side="right"):
    # α=-2 column highlight
    ax.axvspan(main_idx - 0.28, main_idx + 0.28,
               color="#FFF3CD", alpha=0.70, zorder=0)

    # Lines + error bands
    for means, sems, color, label, ls in [
        (mt, st, C_TGT, "Selected feature", "-"),
        (mr, sr, C_RND, "Random direction (3-seed mean)", "--"),
    ]:
        ax.plot(x_pos, means, color=color, linestyle=ls,
                linewidth=1.5, marker="o", markersize=5,
                zorder=3, label=label)
        ax.fill_between(x_pos, means - sems, means + sems,
                        color=color, alpha=0.18, zorder=2)

    # ★ marker at α=-2 for selected feature
    ax.plot(main_idx, mt[main_idx], marker="*", color=C_TGT,
            markersize=9, zorder=4, linestyle="None",
            markeredgecolor="white", markeredgewidth=0.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(a) for a in ALPHAS])
    ax.set_xlabel(r"$\alpha$", fontsize=8.0)
    if show_ylabel:
        ax.set_ylabel(r"$\Delta_{lp}$  (nat)", fontsize=7.5)
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":", alpha=0.4)
    ax.set_title(title, fontsize=8.0, pad=4)
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(axis="y", alpha=0.18, linestyle="--", linewidth=0.5)

    # α=-2 label at top of highlight band
    ax.text(main_idx, ax.get_ylim()[1] * 0.97 if ylim is None
            else ylim[1] * 0.97,
            r"$\alpha=-2$" + "\n(paper)",
            ha="center", va="top", fontsize=6.0,
            color="#8B6914", zorder=5)

    # Selectivity annotation box
    if sel_str:
        ax_x = 0.97 if annot_side == "right" else 0.03
        ax.text(ax_x, 0.04, sel_str,
                transform=ax.transAxes, fontsize=6.5,
                va="bottom", ha=annot_side,
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="lightgray", alpha=0.9))


# Panel (a) — Llama L16/name
draw_panel(
    ax_a, l_mt, l_st, l_mr, l_sr,
    title="(a) Positive control\nLlama L16 / name  (clean-strong)",
    sel_str=(f"sel$_{{lp}}$(α=−2) ≈ {llama_sel:.0f}×  [T1+T5]\n(9-template mean: 1,995×)"
             if llama_sel else None),
    show_ylabel=True,
    annot_side="left",
)

# Panel (b) — Gemma L20/name
draw_panel(
    ax_b, g_mt, g_st, g_mr, g_sr,
    title="(b) Failure mode A\nGemma L20 / name  (random-dominant)",
    sel_str=(f"sel$_{{lp}}$(α=−2) = {gemma_sel:.2f}×"
             if gemma_sel else None),
    show_ylabel=False,
    annot_side="right",
)

# Panel (c) — Qwen L18/name (clean-weak; crossover)
draw_panel(
    ax_c, q_mt, q_st, q_mr, q_sr,
    title="(c) Clean-weak\nQwen L18 / name  (α-dependent crossover)",
    sel_str=(f"sel$_{{lp}}$(α=−2) = {qwen_sel:.2f}×\ncrossover at α ≈ −2.5"
             if qwen_sel else None),
    show_ylabel=False,
    annot_side="right",
)

# ── shared legend ─────────────────────────────────────────────────────────────
h_tgt = Line2D([0], [0], color=C_TGT, linestyle="-",  linewidth=1.5,
               marker="o", markersize=5, label="Selected feature")
h_rnd = Line2D([0], [0], color=C_RND, linestyle="--", linewidth=1.5,
               marker="o", markersize=5, label="Random direction (3-seed mean)")
h_main = Line2D([0], [0], color=C_TGT, linestyle="None",
                marker="*", markersize=8, label=r"$\alpha=-2$ (main paper)")
fig.legend(handles=[h_tgt, h_rnd, h_main],
           loc="lower center", ncol=3, fontsize=7.5,
           bbox_to_anchor=(0.5, -0.05), frameon=False)

# ── save ─────────────────────────────────────────────────────────────────────
for ext in ["png", "pdf"]:
    out = OUT / f"figA1_alpha_sensitivity_v3.{ext}"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"saved: {out}")
plt.close()
