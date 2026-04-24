#!/usr/bin/env python3
"""
09_generate_figures.py — Generate all report figures from experiment JSON results.

Figures produced (saved to report/figures/):
  fig1_alpha_sensitivity.pdf  — NDCG@10 vs alpha for all datasets
  fig2_k_sensitivity.pdf      — NDCG@10 vs RRF k for all datasets
  fig3_oracle_gap.pdf         — Oracle vs best strategies per dataset
  fig4_query_bins.pdf         — Per-query alpha bin analysis (Strategy E)

Usage:
    python scripts/09_generate_figures.py
"""
import json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── Paths ──────────────────────────────────────────────────────────────────
TABLES = Path("report/tables")
FIGS   = Path("report/figures")
FIGS.mkdir(parents=True, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  9,
    "lines.linewidth":  2,
    "lines.markersize": 6,
    "axes.grid":        True,
    "grid.alpha":       0.35,
    "figure.dpi":       150,
})

DS_LABELS = {
    "msmarco":    "MS MARCO",
    "trec-covid": "TREC-COVID",
    "arguana":    "ArguAna",
    "fiqa":       "FiQA",
}
DS_COLORS = {
    "msmarco":    "#1f77b4",
    "trec-covid": "#d62728",
    "arguana":    "#2ca02c",
    "fiqa":       "#ff7f0e",
}
DS_MARKERS = {
    "msmarco":    "o",
    "trec-covid": "s",
    "arguana":    "^",
    "fiqa":       "D",
}

EVAL_DS = ["trec-covid", "arguana", "fiqa"]


# ══════════════════════════════════════════════════════════════════════════
# Figure 1 — Alpha Sensitivity
# ══════════════════════════════════════════════════════════════════════════
def fig1_alpha_sensitivity():
    e2 = json.loads((TABLES / "exp2_sensitivity.json").read_text())
    alpha_data = e2["alpha"]

    fig, ax = plt.subplots(figsize=(7, 4.2))

    for ds in ["msmarco", "trec-covid", "arguana", "fiqa"]:
        xs = sorted(float(k) for k in alpha_data[ds])
        ys = [alpha_data[ds][str(x)] for x in xs]
        ax.plot(xs, ys,
                label=DS_LABELS[ds],
                color=DS_COLORS[ds],
                marker=DS_MARKERS[ds],
                markevery=4)

    ax.set_xlabel("α  (0 = pure dense, 1 = pure BM25)")
    ax.set_ylabel("NDCG@10")
    ax.set_title("Strategy A — Linear Interpolation Sensitivity to α")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.set_xlim(-0.02, 1.02)

    fig.tight_layout()
    out = FIGS / "fig1_alpha_sensitivity.pdf"
    fig.savefig(out)
    fig.savefig(str(out).replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════
# Figure 2 — RRF k Sensitivity
# ══════════════════════════════════════════════════════════════════════════
def fig2_k_sensitivity():
    e2 = json.loads((TABLES / "exp2_sensitivity.json").read_text())
    k_data = e2.get("k", {})
    if not k_data:
        print("  [skip] No k-sensitivity data in exp2_sensitivity.json")
        return

    ks_all = sorted({int(k) for ds in k_data.values() for k in ds})
    x = np.arange(len(ks_all))
    width = 0.2
    offsets = np.linspace(-(len(EVAL_DS)-1)/2, (len(EVAL_DS)-1)/2, len(EVAL_DS)) * width

    fig, ax = plt.subplots(figsize=(7, 4.2))

    for i, ds in enumerate(EVAL_DS):
        if ds not in k_data:
            continue
        ys = [k_data[ds].get(str(k), 0.0) for k in ks_all]
        ax.bar(x + offsets[i], ys,
               width=width,
               label=DS_LABELS[ds],
               color=DS_COLORS[ds],
               alpha=0.85)

    ax.set_xlabel("RRF  k  parameter")
    ax.set_ylabel("NDCG@10")
    ax.set_title("Strategy B — RRF Sensitivity to k")
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks_all])
    ax.legend()
    ax.set_ylim(bottom=0.0)

    fig.tight_layout()
    out = FIGS / "fig2_k_sensitivity.pdf"
    fig.savefig(out)
    fig.savefig(str(out).replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════
# Figure 3 — Oracle Gap
# ══════════════════════════════════════════════════════════════════════════
def fig3_oracle_gap():
    e1 = json.loads((TABLES / "exp1_matrix.json").read_text())
    e4 = json.loads((TABLES / "exp4_oracle.json").read_text())

    strats = ["bm25_only", "dense_only", "A_linear", "B_rrf", "D_learned", "E_adaptive"]
    strat_labels = {
        "bm25_only":  "BM25",
        "dense_only": "Dense",
        "A_linear":   "A-Linear",
        "B_rrf":      "B-RRF",
        "D_learned":  "D-Learned",
        "E_adaptive": "E-Adaptive",
    }
    strat_colors = {
        "bm25_only":  "#aec7e8",
        "dense_only": "#ffbb78",
        "A_linear":   "#1f77b4",
        "B_rrf":      "#2ca02c",
        "D_learned":  "#9467bd",
        "E_adaptive": "#8c564b",
    }

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=False)

    for ax, ds in zip(axes, EVAL_DS):
        ds_results = e1["results"][ds]
        oracle_ndcg = e4[ds]["oracle_ndcg"]

        xs = list(range(len(strats)))
        ys = [ds_results[s]["ndcg_cut_10"] for s in strats]
        colors = [strat_colors[s] for s in strats]

        bars = ax.bar(xs, ys, color=colors, alpha=0.9, zorder=3)

        # Oracle line
        ax.axhline(oracle_ndcg, color="red", linestyle="--", linewidth=1.5,
                   label=f"Oracle ({oracle_ndcg:.3f})", zorder=4)

        # Value labels on bars
        for bar, val in zip(bars, ys):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

        ax.set_title(DS_LABELS[ds])
        ax.set_xticks(xs)
        ax.set_xticklabels([strat_labels[s] for s in strats], rotation=35, ha="right", fontsize=9)
        ax.set_ylabel("NDCG@10" if ax == axes[0] else "")
        ax.legend(fontsize=8)
        ymin = min(ys) * 0.92
        ymax = oracle_ndcg * 1.05
        ax.set_ylim(ymin, ymax)
        ax.grid(axis="y", alpha=0.35)

    fig.suptitle("Strategy Comparison vs Oracle Upper Bound", fontsize=13, y=1.01)
    fig.tight_layout()
    out = FIGS / "fig3_oracle_gap.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════
# Figure 4 — Query Bin Analysis (Strategy E)
# ══════════════════════════════════════════════════════════════════════════
def fig4_query_bins():
    e3 = json.loads((TABLES / "exp3_query_analysis.json").read_text())

    bin_order  = ["dense_dominant", "balanced", "sparse_dominant"]
    bin_labels = {"dense_dominant": "Dense-dom.", "balanced": "Balanced", "sparse_dominant": "Sparse-dom."}
    bin_colors = {"dense_dominant": "#4c72b0", "balanced": "#55a868", "sparse_dominant": "#c44e52"}

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=False)

    for ax, ds in zip(axes, EVAL_DS):
        ds_data = e3[ds]
        bins    = ds_data.get("bins", {})

        present = [b for b in bin_order if b in bins]
        xs      = np.arange(len(present))
        deltas  = [bins[b]["ndcg_delta"] for b in present]
        counts  = [bins[b]["count"]       for b in present]
        colors  = [bin_colors[b]          for b in present]

        bars = ax.bar(xs, deltas, color=colors, alpha=0.85, zorder=3)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="-")

        # Count labels
        for bar, cnt, delta in zip(bars, counts, deltas):
            ypos = delta + (0.002 if delta >= 0 else -0.004)
            ax.text(bar.get_x() + bar.get_width()/2, ypos,
                    f"n={cnt}", ha="center", va="bottom" if delta >= 0 else "top",
                    fontsize=8)

        ax.set_title(f"{DS_LABELS[ds]}\nstatic={ds_data['static_ndcg']:.4f}  adaptive={ds_data['adaptive_ndcg']:.4f}")
        ax.set_xticks(xs)
        ax.set_xticklabels([bin_labels[b] for b in present], fontsize=9)
        ax.set_ylabel("ΔNDCG@10 (adaptive − static)" if ax == axes[0] else "")
        ax.grid(axis="y", alpha=0.35)

    fig.suptitle("Strategy E — Per-Query NDCG Delta by Predicted Alpha Bin", fontsize=12, y=1.01)
    fig.tight_layout()
    out = FIGS / "fig4_query_bins.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating figures ...")
    fig1_alpha_sensitivity()
    fig2_k_sensitivity()
    fig3_oracle_gap()
    fig4_query_bins()
    print(f"\n✅  All figures saved to {FIGS}/")
