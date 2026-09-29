"""
scripts/figures_final.py

Generate paper's 5 final figures (3 main + 2 appendix) at results/figures_final/.

Design priorities:
  1. No text overlap (constrained_layout + explicit placement).
  2. Colorblind-friendly palette, log-scale where appropriate.
  3. Each figure saved as PNG (300 dpi) + PDF + SVG.

Figures:
  fig1_layer_selectivity      Exp 2 — Gemma L9/L20/L31 cross-template selectivity
  fig2_id_mismatch            Exp 4 — Llama L25 × id × 9 templates (Δ_lp + Δb_rank)
  fig3_artifact_comparison    Task-SAE Gemma L9 feat 20 vs Pre-trained L10 feat 2759
  figA1_alpha_sensitivity     Exp 1 — 4-panel (2×2)
  figA2_pca_silhouette        Copy/adapt existing fig5
"""

import math
import os
import shutil
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths & constants
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = RESULTS / "figures_final"
OUT.mkdir(exist_ok=True, parents=True)

TEMPLATES = ["T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]

# Colorblind-friendly palette
C_L9 = "#2E7D32"          # PCA peak (good layer)
C_L20 = "#F9A825"
C_L31 = "#C62828"
C_PT = "#9E9E9E"          # Pre-trained baseline
C_TASK = "#1565C0"        # Task-SAE
C_MISMATCH = "#B71C1C"    # Strong mismatch
C_NONMISMATCH = "#9E9E9E"
C_ANTI = "#8E24AA"
C_THRESHOLD = "#444444"
C_RAND = "#BDBDBD"

# rcParams — sans-serif, consistent font sizes
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
    "lines.linewidth": 1.4,
    "patch.linewidth": 0.4,
    "errorbar.capsize": 2.5,
})


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def agg(df, col):
    """Return (n, mean, ci95_lo, ci95_hi). Returns None if df empty."""
    if df is None or len(df) == 0:
        return None
    v = df[col].dropna().to_numpy(dtype=float)
    n = len(v)
    if n == 0:
        return None
    m = float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    ci = 1.96 * se
    return n, m, m - ci, m + ci


def load(path):
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p)


def selectivity(hero_mean, rand_mean, floor=0.01):
    """Selectivity = |hero| / max(|random|, floor)."""
    return abs(hero_mean) / max(abs(rand_mean), floor)


def sig_marker(lo, hi):
    return "*" if (lo > 0 or hi < 0) else ""


def save_all(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{ext}",
                    dpi=300 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  [save] {name}.png/pdf/svg")


# --------------------------------------------------------------------------- #
# Fig 1 — Layer Selectivity (Gemma L9 vs L20 vs L31 × name × 9 templates)
# --------------------------------------------------------------------------- #

def fig1_layer_selectivity():
    print("\n[fig1] layer selectivity (Gemma L9/L20/L31)")
    # Sources:
    #   L9  T1     -> new_sae_phase3/
    #   L9  T0,T2-T8 -> new_sae_phase3_xtemplate/
    #   L20 T0-T8  -> exp2_gemma_xtemplate/gemma_L20/
    #   L31 T0-T8  -> exp2_gemma_xtemplate/gemma_L31/
    def src_L9(t):
        if t == "T1":
            return (RESULTS / "new_sae_phase3" / f"gemma_2_9b_L9_name_{t}_alpham2.csv",
                    RESULTS / "new_sae_phase3" / f"gemma_2_9b_L9_name_{t}_alpham2_random_vector.csv")
        return (RESULTS / "new_sae_phase3_xtemplate" / f"gemma_2_9b_L9_name_{t}_alpham2.csv",
                RESULTS / "new_sae_phase3_xtemplate" / f"gemma_2_9b_L9_name_{t}_alpham2_random_vector.csv")

    def src(layer, t):
        d = RESULTS / "exp2_gemma_xtemplate" / f"gemma_L{layer}"
        return (d / f"gemma_2_9b_L{layer}_name_{t}_alpham2.csv",
                d / f"gemma_2_9b_L{layer}_name_{t}_alpham2_random_vector.csv")

    layers = [("L9 (PCA peak)", src_L9, C_L9),
              ("L20",          lambda t: src(20, t), C_L20),
              ("L31",          lambda t: src(31, t), C_L31)]

    sel_lp = {name: [] for name, _, _ in layers}
    sel_br = {name: [] for name, _, _ in layers}
    sig_lp = {name: [] for name, _, _ in layers}

    for t in TEMPLATES:
        for name, srcfn, _ in layers:
            hpath, rpath = srcfn(t)
            h = load(hpath); r = load(rpath)
            if h is None or r is None:
                sel_lp[name].append(np.nan); sel_br[name].append(np.nan)
                sig_lp[name].append(False); continue
            ah = agg(h, "delta_lp"); ar = agg(r, "delta_lp")
            ahb = agg(h, "delta_b_rank"); arb = agg(r, "delta_b_rank")
            if ah is None or ar is None:
                sel_lp[name].append(np.nan); sel_br[name].append(np.nan)
                sig_lp[name].append(False); continue
            _, mh, lo_h, hi_h = ah
            _, mr, _, _ = ar
            sel_lp[name].append(selectivity(mh, mr))
            sig_lp[name].append(lo_h > 0 or hi_h < 0)
            if ahb is not None and arb is not None:
                _, mhb, _, _ = ahb
                _, mrb, _, _ = arb
                sel_br[name].append(selectivity(mhb, mrb))
            else:
                sel_br[name].append(np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    x = np.arange(len(TEMPLATES))
    width = 0.27
    offsets = {layers[0][0]: -width, layers[1][0]: 0.0, layers[2][0]: width}

    for ax, dat, ylab, title in [
        (axes[0], sel_lp, "Selectivity (Δlogprob)", "(a) Logprob selectivity"),
        (axes[1], sel_br, "Selectivity (Δb_rank)",  "(b) b_rank selectivity"),
    ]:
        for name, _, col in layers:
            y = np.array(dat[name], dtype=float)
            y_plot = np.where(np.isnan(y), 0.05, y)
            ax.bar(x + offsets[name], y_plot, width=width,
                   color=col, edgecolor="black", linewidth=0.5,
                   label=name)
        ax.axhline(5.0, color=C_THRESHOLD, linestyle="--", linewidth=1.0,
                   alpha=0.7, label="working threshold (5×)")
        ax.axhline(1.0, color="black", linestyle=":", linewidth=0.7, alpha=0.5)
        ax.set_yscale("log")
        ax.set_ylim(0.05, 200)
        ax.set_xticks(x)
        ax.set_xticklabels(TEMPLATES)
        ax.set_xlabel("Template")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(True, which="both", axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    axes[0].legend(loc="upper right", framealpha=0.92,
                   ncol=1, bbox_to_anchor=(1.0, 1.02))
    fig.suptitle("Layer Selectivity — Gemma (task-SAE α=−2, n=50/template)",
                 fontsize=13, fontweight="bold")
    save_all(fig, "fig1_layer_selectivity")


# --------------------------------------------------------------------------- #
# Fig 2 — ID Mismatch Pattern (Llama L25 × id × 9 templates)
# --------------------------------------------------------------------------- #

def fig2_id_mismatch():
    print("\n[fig2] id mismatch (Llama L25 × id × T0..T8)")
    base = RESULTS / "new_paper_step5b2_multimetric"

    rows = []
    for t in TEMPLATES:
        hpath = base / f"llama3_med_8b_L25_id_{t}_alphap2.csv"
        rpath = base / f"llama3_med_8b_L25_id_{t}_alphap2_random_vector.csv"
        h = load(hpath)
        r = load(rpath)
        if h is None:
            continue
        a_lp = agg(h, "delta_lp")
        a_br = agg(h, "delta_b_rank")
        if a_lp is None or a_br is None:
            continue
        nh, m_lp, lo_lp, hi_lp = a_lp
        _,  m_br, lo_br, hi_br = a_br
        # significance
        sig_lp_neg = hi_lp < 0
        sig_lp_ns  = (lo_lp <= 0 <= hi_lp)
        sig_br_neg = hi_br < 0
        # MISMATCH: Δ_lp NS but Δb_rank significantly negative
        is_mismatch = sig_lp_ns and sig_br_neg
        rows.append(dict(t=t, m_lp=m_lp, lo_lp=lo_lp, hi_lp=hi_lp,
                          m_br=m_br, lo_br=lo_br, hi_br=hi_br,
                          mismatch=is_mismatch))

    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), constrained_layout=True)
    x = np.arange(len(df))

    for ax, mc, lc, hc, ylab, title in [
        (axes[0], "m_lp", "lo_lp", "hi_lp",
         r"$\Delta_{\mathrm{lp}}$  (nat)",
         "(a) Logprob change"),
        (axes[1], "m_br", "lo_br", "hi_br",
         r"$\Delta b\_rank$  (bits)",
         "(b) b_rank change"),
    ]:
        means = df[mc].to_numpy()
        lows = df[lc].to_numpy()
        highs = df[hc].to_numpy()
        err_lo = means - lows
        err_hi = highs - means
        cols = [C_MISMATCH if m else C_NONMISMATCH for m in df["mismatch"]]
        bars = ax.bar(x, means, yerr=[err_lo, err_hi],
                      color=cols, edgecolor="black", linewidth=0.5,
                      ecolor="black", capsize=3)
        ax.axhline(0.0, color="black", linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(df["t"])
        ax.set_xlabel("Template")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.set_axisbelow(True)

    # Legend handles
    from matplotlib.patches import Patch
    handles = [
        Patch(color=C_MISMATCH, label="mismatch template (Δlp NS, Δb_rank<0)"),
        Patch(color=C_NONMISMATCH, label="non-mismatch"),
    ]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False)

    # text box explanation
    n_mismatch = int(df["mismatch"].sum())
    txt = (f"Mismatch pattern: {n_mismatch}/9 templates show $\\Delta_{{lp}}$ "
           f"not significant but $\\Delta b\\_rank<0$ significant.\n"
           f"Logprob mean is preserved while the gold token's rank degrades — "
           f"a feature-specific, sub-mean effect on PII recoverability.")
    fig.text(0.5, -0.07, txt, ha="center", va="top", fontsize=9,
             bbox=dict(facecolor="#FFFDE7", edgecolor="#FBC02D",
                       linewidth=0.5, pad=4))

    fig.suptitle("ID Mismatch Pattern — Llama-Med L25 × id (α=+2, n=50/template)",
                 fontsize=13, fontweight="bold")
    save_all(fig, "fig2_id_mismatch")


# --------------------------------------------------------------------------- #
# Fig 3 — Magnitude Artifact (Task-SAE Gemma L9 vs Pre-trained Gemma L10)
# --------------------------------------------------------------------------- #

def fig3_artifact_comparison():
    print("\n[fig3] artifact comparison")
    pt_base = RESULTS / "new_paper_step5b2_multimetric"   # L10 feat 2759
    task_x = RESULTS / "new_sae_phase3_xtemplate"          # L9 feat 20 (T0,T2-T8)
    task_1 = RESULTS / "new_sae_phase3"                    # L9 feat 20 (T1)

    def task_src(t):
        if t == "T1":
            return (task_1 / f"gemma_2_9b_L9_name_{t}_alpham2.csv",
                    task_1 / f"gemma_2_9b_L9_name_{t}_alpham2_random_vector.csv")
        return (task_x / f"gemma_2_9b_L9_name_{t}_alpham2.csv",
                task_x / f"gemma_2_9b_L9_name_{t}_alpham2_random_vector.csv")

    def pt_src(t):
        return (pt_base / f"gemma_2_9b_L10_name_{t}_alpham2.csv",
                pt_base / f"gemma_2_9b_L10_name_{t}_alpham2_random_vector.csv")

    pt_lp, pt_br, pt_sel_lp, pt_sel_br = [], [], [], []
    tk_lp, tk_br, tk_sel_lp, tk_sel_br = [], [], [], []
    for t in TEMPLATES:
        ph, pr = pt_src(t)
        th, tr = task_src(t)
        # Pre-trained
        a = agg(load(ph), "delta_lp"); b = agg(load(pr), "delta_lp")
        ab = agg(load(ph), "delta_b_rank"); bb = agg(load(pr), "delta_b_rank")
        if a and b:
            pt_lp.append(abs(a[1]))
            pt_sel_lp.append(selectivity(a[1], b[1]))
        else:
            pt_lp.append(np.nan); pt_sel_lp.append(np.nan)
        if ab and bb:
            pt_br.append(abs(ab[1]))
            pt_sel_br.append(selectivity(ab[1], bb[1]))
        else:
            pt_br.append(np.nan); pt_sel_br.append(np.nan)

        # Task-SAE
        a = agg(load(th), "delta_lp"); b = agg(load(tr), "delta_lp")
        ab = agg(load(th), "delta_b_rank"); bb = agg(load(tr), "delta_b_rank")
        if a and b:
            tk_lp.append(abs(a[1]))
            tk_sel_lp.append(selectivity(a[1], b[1]))
        else:
            tk_lp.append(np.nan); tk_sel_lp.append(np.nan)
        if ab and bb:
            tk_br.append(abs(ab[1]))
            tk_sel_br.append(selectivity(ab[1], bb[1]))
        else:
            tk_br.append(np.nan); tk_sel_br.append(np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    x = np.arange(len(TEMPLATES))

    # Panel (a): hero |Δlp| comparison
    ax = axes[0]
    ax.plot(x, pt_lp, "o-", color=C_PT, label="Pre-trained SAE L10 (f=2759)",
            markersize=5, linewidth=1.5)
    ax.plot(x, tk_lp, "s-", color=C_TASK, label="Task-SAE L9 (f=20)",
            markersize=5, linewidth=1.5)
    ax.set_xticks(x); ax.set_xticklabels(TEMPLATES)
    ax.set_xlabel("Template")
    ax.set_ylabel(r"hero $|\Delta_{\mathrm{lp}}|$ (nat)")
    ax.set_title("(a) Hero magnitude")
    ax.legend(loc="upper left", framealpha=0.92)
    ax.set_axisbelow(True)

    # Panel (b): selectivity comparison (log scale)
    ax = axes[1]
    ax.plot(x, pt_sel_lp, "o-", color=C_PT, label="Pre-trained SAE",
            markersize=5, linewidth=1.5)
    ax.plot(x, tk_sel_lp, "s-", color=C_TASK, label="Task-SAE",
            markersize=5, linewidth=1.5)
    ax.axhline(5.0, color=C_THRESHOLD, linestyle="--", linewidth=1.0,
               alpha=0.7, label="working threshold (5×)")
    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.7, alpha=0.5)
    ax.set_yscale("log")
    ax.set_ylim(0.3, 100)
    ax.set_xticks(x); ax.set_xticklabels(TEMPLATES)
    ax.set_xlabel("Template")
    ax.set_ylabel(r"Selectivity ($|\Delta_{\mathrm{lp}}|$ hero / random)")
    ax.set_title("(b) Selectivity (magnitude-matched)")
    ax.legend(loc="upper left", framealpha=0.92)
    ax.set_axisbelow(True)

    # Annotate T8 dramatic ratio
    if not (np.isnan(pt_sel_lp[8]) or np.isnan(tk_sel_lp[8])):
        ratio = tk_sel_lp[8] / pt_sel_lp[8] if pt_sel_lp[8] > 0 else 0
        if ratio > 5:
            yT = max(tk_sel_lp[8], pt_sel_lp[8])
            axes[1].annotate(f"{ratio:.0f}× ratio",
                             xy=(8, tk_sel_lp[8]), xytext=(6.2, yT * 2.5),
                             fontsize=9, color=C_TASK, fontweight="bold",
                             arrowprops=dict(arrowstyle="->",
                                             color=C_TASK, lw=0.8))

    fig.suptitle("Magnitude Artifact Test — Task-SAE vs Pre-trained SAE (Gemma × name)",
                 fontsize=13, fontweight="bold")
    save_all(fig, "fig3_artifact_comparison")


# --------------------------------------------------------------------------- #
# Fig A1 — α sensitivity (4-panel 2×2)
# --------------------------------------------------------------------------- #

def figA1_alpha_sensitivity():
    print("\n[figA1] α sensitivity (4-panel)")
    panels = [
        dict(title="(a) Gemma PT L10 (f=2759) × name × T1",
             dir=RESULTS / "exp1_alpha_sensitivity" / "pt_gemma_L10",
             prefix="gemma_2_9b_L10_name_T1",
             alphas=[-1, -2, -3], color=C_PT),
        dict(title="(b) Task-SAE Gemma L9 (f=20) × name × T1",
             dir=RESULTS / "exp1_alpha_sensitivity" / "task_gemma_L9",
             prefix="gemma_2_9b_L9_name_T1",
             alphas=[-1, -2, -3], color=C_TASK),
        dict(title="(c) Llama PT L25 (f=22072) × name × T0",
             dir=RESULTS / "exp1_alpha_sensitivity" / "pt_llama_L25_name",
             prefix="llama3_med_8b_L25_name_T0",
             alphas=[1, 2, 3], color=C_L20),
        dict(title="(d) Llama PT L25 (f=22072) × id × T3 (mismatch)",
             dir=RESULTS / "exp1_alpha_sensitivity" / "pt_llama_L25_id",
             prefix="llama3_med_8b_L25_id_T3",
             alphas=[1, 2, 3], color=C_MISMATCH),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5), constrained_layout=True)
    for ax, pan in zip(axes.flat, panels):
        alphas = pan["alphas"]
        h_lp_m, h_lp_lo, h_lp_hi = [], [], []
        r_lp_m, r_lp_lo, r_lp_hi = [], [], []
        h_br_m = []; r_br_m = []
        for a in alphas:
            tag = f"alpha{('p' if a > 0 else 'm')}{abs(a)}"
            hp = pan["dir"] / f"{pan['prefix']}_{tag}.csv"
            rp = pan["dir"] / f"{pan['prefix']}_{tag}_random_vector.csv"
            ah = agg(load(hp), "delta_lp"); ar = agg(load(rp), "delta_lp")
            ahb = agg(load(hp), "delta_b_rank"); arb = agg(load(rp), "delta_b_rank")
            if ah is not None:
                _, m, lo, hi = ah
                h_lp_m.append(m); h_lp_lo.append(lo); h_lp_hi.append(hi)
            else:
                h_lp_m.append(np.nan); h_lp_lo.append(np.nan); h_lp_hi.append(np.nan)
            if ar is not None:
                _, m, lo, hi = ar
                r_lp_m.append(m); r_lp_lo.append(lo); r_lp_hi.append(hi)
            else:
                r_lp_m.append(np.nan); r_lp_lo.append(np.nan); r_lp_hi.append(np.nan)
            h_br_m.append(ahb[1] if ahb is not None else np.nan)
            r_br_m.append(arb[1] if arb is not None else np.nan)

        xs = np.array(alphas, dtype=float)
        ax.errorbar(xs, h_lp_m,
                    yerr=[np.array(h_lp_m) - np.array(h_lp_lo),
                          np.array(h_lp_hi) - np.array(h_lp_m)],
                    fmt="o-", color=pan["color"], markersize=6, capsize=3,
                    label="hero (target feature)")
        ax.errorbar(xs + 0.05, r_lp_m,
                    yerr=[np.array(r_lp_m) - np.array(r_lp_lo),
                          np.array(r_lp_hi) - np.array(r_lp_m)],
                    fmt="s--", color=C_RAND, markersize=5, capsize=3,
                    label="random vector (matched ||·||)")
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(r"$\Delta_{\mathrm{lp}}$  (nat)")
        ax.set_title(pan["title"], fontsize=11)
        ax.set_xticks(xs)
        ax.legend(loc="best", framealpha=0.92, fontsize=8)
        ax.set_axisbelow(True)

    fig.suptitle("α Sensitivity — Hero conditions robust across α magnitudes",
                 fontsize=13, fontweight="bold")
    save_all(fig, "figA1_alpha_sensitivity")


# --------------------------------------------------------------------------- #
# Fig A2 — PCA Silhouette (reuse existing)
# --------------------------------------------------------------------------- #

def figA2_pca_silhouette():
    print("\n[figA2] pca silhouette (copy from results/figures/fig5)")
    src_dir = RESULTS / "figures"
    for ext in ("png", "svg"):
        src = src_dir / f"fig5_pca_silhouette.{ext}"
        if src.exists():
            shutil.copy(src, OUT / f"figA2_pca_silhouette.{ext}")
            print(f"  [copy] figA2_pca_silhouette.{ext}")
    # Generate a PDF alongside the PNG if missing
    png = OUT / "figA2_pca_silhouette.png"
    pdf = OUT / "figA2_pca_silhouette.pdf"
    if png.exists() and not pdf.exists():
        import matplotlib.image as mpimg
        img = mpimg.imread(str(png))
        h, w = img.shape[:2]
        fig, ax = plt.subplots(figsize=(w / 200, h / 200))
        ax.imshow(img); ax.axis("off")
        fig.savefig(pdf, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        print(f"  [save] figA2_pca_silhouette.pdf (re-encoded from PNG)")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    fig1_layer_selectivity()
    fig2_id_mismatch()
    fig3_artifact_comparison()
    figA1_alpha_sensitivity()
    figA2_pca_silhouette()
    print(f"\n[done] All figures saved to {OUT}/")


if __name__ == "__main__":
    main()
