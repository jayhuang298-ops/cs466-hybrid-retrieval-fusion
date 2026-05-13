#!/usr/bin/env python3
"""
10_generate_tables.py — Generate LaTeX tables from experiment JSON results.

Tables produced (saved to report/tables/):
  table1_main_results.tex   — 7×4 strategy × dataset performance matrix
  table2_significance.tex   — Paired t-test results vs RRF baseline
  table3_oracle_gap.tex     — Oracle upper bound summary

Usage:
    python scripts/10_generate_tables.py
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TABLES = Path("report/tables")

DS_ORDER    = ["trec-covid", "arguana", "fiqa"]
DS_LABELS   = {"trec-covid": "TREC-COVID", "arguana": "ArguAna", "fiqa": "FiQA"}
STRAT_ORDER = ["bm25_only", "dense_only", "A_linear", "B_rrf", "C_conv", "D_learned", "E_adaptive"]
STRAT_LABELS = {
    "bm25_only":  "BM25 (sparse)",
    "dense_only": "Dense (BGE)",
    "A_linear":   "A – Linear Interp.",
    "B_rrf":      "B – RRF",
    "C_conv":     "C – Convex Rank",
    "D_learned":  "D – Learned (LogReg)",
    "E_adaptive": "E – Adaptive (RF)",
}
METRICS = ["ndcg_cut_10", "map", "recall_100"]
METRIC_LABELS = {"ndcg_cut_10": "NDCG@10", "map": "MAP", "recall_100": "R@100"}


def bold(x: str) -> str:
    return r"\textbf{" + x + "}"


def dagger(x: str) -> str:
    return x + r"$^\dagger$"


def fmt(v: float) -> str:
    return f"{v:.4f}"


# ══════════════════════════════════════════════════════════════════════════
# Table 1 — Main Results Matrix
# ══════════════════════════════════════════════════════════════════════════
def table1_main_results():
    e1 = json.loads((TABLES / "exp1_matrix.json").read_text())
    results = e1["results"]

    # Find best per (dataset, metric) among fusion strategies only
    fusion_strats = ["A_linear", "B_rrf", "C_conv", "D_learned", "E_adaptive"]
    best = {}
    for ds in DS_ORDER:
        best[ds] = {}
        for m in METRICS:
            best[ds][m] = max(results[ds][s][m] for s in fusion_strats)

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Zero-shot retrieval performance (NDCG@10 / MAP / R@100) across three BEIR datasets. "
                 r"Bold = best fusion strategy per column. "
                 r"$^\dagger$ = significantly better than B-RRF baseline ($p < 0.05$, paired $t$-test).}")
    lines.append(r"\label{tab:main_results}")

    # Column spec: strategy + 3 metrics × 3 datasets = 10 cols
    col_spec = "l" + "".join([" ccc" for _ in DS_ORDER])
    lines.append(r"\begin{tabular}{" + col_spec + r"}")
    lines.append(r"\toprule")

    # Header row 1 — dataset names spanning 3 cols each
    header1 = "Strategy"
    for ds in DS_ORDER:
        header1 += r" & \multicolumn{3}{c}{" + DS_LABELS[ds] + "}"
    lines.append(header1 + r" \\")

    # cmidrule under each dataset header
    cmidrules = []
    for i, _ in enumerate(DS_ORDER):
        start = 2 + i * 3
        end   = start + 2
        cmidrules.append(rf"\cmidrule(lr){{{start}-{end}}}")
    lines.append(" ".join(cmidrules))

    # Header row 2 — metric names
    header2 = ""
    for _ in DS_ORDER:
        for m in METRICS:
            header2 += " & " + METRIC_LABELS[m]
    lines.append(header2 + r" \\")
    lines.append(r"\midrule")

    # Load significance data if available
    sig_path = TABLES / "exp1_significance.json"
    sig = json.loads(sig_path.read_text()) if sig_path.exists() else {}

    # Data rows
    for s in STRAT_ORDER:
        row = STRAT_LABELS[s]
        for ds in DS_ORDER:
            for m in METRICS:
                val  = results[ds][s][m]
                cell = fmt(val)
                # Bold best fusion
                if s in fusion_strats and abs(val - best[ds][m]) < 1e-8:
                    cell = bold(cell)
                # Dagger if significantly better than RRF
                if sig.get(ds, {}).get(s, {}).get(m, {}).get("significant", False):
                    cell = dagger(cell)
                row += " & " + cell
        lines.append(row + r" \\")
        # Separator after baselines
        if s == "dense_only":
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = TABLES / "table1_main_results.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════
# Table 2 — Significance Tests
# ══════════════════════════════════════════════════════════════════════════
def table2_significance():
    """Run paired t-tests (Strategy vs RRF) and write LaTeX table."""
    e1 = json.loads((TABLES / "exp1_matrix.json").read_text())

    # We need per-query scores — re-run evaluate to get them
    try:
        from src.utils.config  import load_config, get_run_path
        from src.utils.io      import read_trec_run
        from src.eval.evaluator import evaluate, load_qrels_from_tsv
        from src.eval.significance import paired_ttest, is_significant
        from src.fusion.strategy_a_linear    import LinearFusion
        from src.fusion.strategy_b_rrf       import RRFFusion
        from src.fusion.strategy_c_conv_rank import ConvexRankFusion
        from src.fusion.strategy_d_learned   import LearnedFusion
        from src.fusion.strategy_e_adaptive  import AdaptiveFusion
    except ImportError as e:
        print(f"  [skip] Cannot import src modules: {e}")
        return

    cfg        = load_config()
    best_p     = e1["best_params"]
    trained    = Path(cfg["base"]["paths"]["trained"])
    DATASETS_MAP = {"trec-covid": "test", "arguana": "test", "fiqa": "test"}

    sig_data = {}
    rows = []

    compare_strats = ["A_linear", "C_conv", "D_learned", "E_adaptive"]
    strat_label = {
        "A_linear":   "A – Linear",
        "C_conv":     "C – Conv. Rank",
        "D_learned":  "D – Learned",
        "E_adaptive": "E – Adaptive",
    }

    for ds, split in DATASETS_MAP.items():
        bm25_r  = read_trec_run(get_run_path("bm25",  ds, cfg))
        dense_r = read_trec_run(get_run_path("dense", ds, cfg))
        qrels   = load_qrels_from_tsv(
            Path(cfg["base"]["paths"]["data"]) / ds / "qrels" / f"{split}.tsv")

        # RRF baseline per-query
        rrf_run   = RRFFusion().fuse(bm25_r, dense_r, k=best_p["k"])
        _, rrf_pq = evaluate(rrf_run, qrels)

        sig_data[ds] = {}
        for s in compare_strats:
            if s == "A_linear":
                run = LinearFusion().fuse(bm25_r, dense_r, alpha=best_p["alpha"])
            elif s == "C_conv":
                run = ConvexRankFusion().fuse(bm25_r, dense_r, beta=best_p["beta"])
            elif s == "D_learned":
                m = LearnedFusion(); m.load(trained / "strategyD_logreg.pkl")
                run = m.fuse(bm25_r, dense_r)
            else:
                import json as _json
                queries_path = Path(cfg["base"]["paths"]["data"]) / ds / "queries.jsonl"
                queries = {}
                with open(queries_path) as f:
                    for line in f:
                        obj = _json.loads(line)
                        qid = obj.get("_id") or obj.get("id")
                        if qid:
                            queries[qid] = obj.get("text", "")
                m = AdaptiveFusion(); m.load(trained / "strategyE_alpha_rf.pkl")
                run = m.fuse(bm25_r, dense_r, queries=queries)

            _, pq = evaluate(run, qrels)
            t, p, n = paired_ttest(pq, rrf_pq, metric="ndcg_cut_10")
            sig = is_significant(p)
            sig_data[ds][s] = {
                "ndcg_cut_10": {"t": t, "p": p, "n": n, "significant": sig}
            }
            rows.append((ds, s, t, p, n, sig))
            print(f"    [{ds}] {s:15s}  t={t:+.3f}  p={p:.4f}  {'*' if sig else ''}")

    # Save sig data for table1 use
    (TABLES / "exp1_significance.json").write_text(json.dumps(sig_data, indent=2))

    # LaTeX table
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Paired $t$-test results (NDCG@10) comparing each fusion strategy against "
                 r"B-RRF baseline. $n$ = number of queries. * = $p < 0.05$.}")
    lines.append(r"\label{tab:significance}")
    lines.append(r"\begin{tabular}{llrrrc}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & Strategy & $t$ & $p$ & $n$ & Sig. \\")
    lines.append(r"\midrule")

    prev_ds = None
    for ds, s, t, p, n, sig in rows:
        if prev_ds and prev_ds != ds:
            lines.append(r"\midrule")
        prev_ds = ds
        ds_cell = DS_LABELS[ds] if (ds, s) == next(
            (x[:2] for x in rows if x[0] == ds), (None, None)) else ""
        lines.append(
            f"{DS_LABELS[ds]} & {strat_label[s]} & {t:+.3f} & {p:.4f} & {n} & "
            + (r"$*$" if sig else "–") + r" \\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = TABLES / "table2_significance.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════
# Table 3 — Oracle Gap
# ══════════════════════════════════════════════════════════════════════════
def table3_oracle_gap():
    e1 = json.loads((TABLES / "exp1_matrix.json").read_text())
    e4 = json.loads((TABLES / "exp4_oracle.json").read_text())

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Oracle upper bound (per-query best $\alpha$, step=0.01) vs best practical "
                 r"strategy (D-Learned) on NDCG@10. Gap = oracle $-$ best practical.}")
    lines.append(r"\label{tab:oracle}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & Best Practical & D-Learned & Oracle & Gap \\")
    lines.append(r"\midrule")

    for ds in DS_ORDER:
        d_ndcg  = e1["results"][ds]["D_learned"]["ndcg_cut_10"]
        oracle  = e4[ds]["oracle_ndcg"]
        gap     = e4[ds]["gap"]
        best_st = e4[ds]["best_static_ndcg"]
        lines.append(
            f"{DS_LABELS[ds]} & {fmt(best_st)} & {fmt(d_ndcg)} & "
            f"{bold(fmt(oracle))} & {gap:+.4f} " + r"\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = TABLES / "table3_oracle_gap.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating LaTeX tables ...")
    table1_main_results()
    print("  Running significance tests (this takes ~2 min) ...")
    table2_significance()
    # Re-run table1 now that significance data exists
    print("  Updating table1 with significance markers ...")
    table1_main_results()
    table3_oracle_gap()
    print(f"\n✅  All tables saved to {TABLES}/")
