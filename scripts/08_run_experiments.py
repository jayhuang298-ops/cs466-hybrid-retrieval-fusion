#!/usr/bin/env python3
"""
08_run_experiments.py — Run all 4 experiments and produce tables + figures.

Experiments:
  Exp 1 — Full 7×4 strategy × dataset performance matrix
  Exp 2 — Alpha and k sensitivity curves
  Exp 3 — Query-level analysis of Strategy E predicted alphas
  Exp 4 — Oracle upper bound vs practical strategies

Usage:
    python scripts/08_run_experiments.py [--exp 1,2,3,4] [--force]
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, get_run_path
from src.utils.logging import get_logger
from src.utils.io import read_trec_run, write_trec_run
from src.eval.evaluator import evaluate, load_qrels_from_tsv
from src.eval.significance import paired_ttest, is_significant

log = get_logger("experiments", "logs/08_experiments.log")

DATASETS = [
    ("msmarco",    "dev"),
    ("trec-covid", "test"),
    ("arguana",    "test"),
    ("fiqa",       "test"),
]
EVAL_DATASETS  = ["trec-covid", "arguana", "fiqa"]   # cross-domain eval
TRAIN_DATASET  = "msmarco"


# ─────────────────────────────────────────────────────────────────────── #
# Helpers
# ─────────────────────────────────────────────────────────────────────── #

def get_qrels(name: str, split: str, cfg: dict) -> dict:
    qrels_path = Path(cfg["base"]["paths"]["data"]) / name / "qrels" / f"{split}.tsv"
    return load_qrels_from_tsv(qrels_path)


def load_runs(cfg: dict) -> dict:
    """Load all cached base runs. Returns {name: {retriever: run}}."""
    runs = {}
    for name, _ in DATASETS:
        runs[name] = {
            "bm25":  read_trec_run(get_run_path("bm25",  name, cfg)),
            "dense": read_trec_run(get_run_path("dense", name, cfg)),
        }
    return runs


def tune_param_on_msmarco(fuser_cls, param_name, grid, bm25, dense, qrels, **fixed):
    """Grid-search a single param on MS MARCO dev. Returns (best_param, scores_dict)."""
    from src.eval.evaluator import evaluate
    best_param, best_ndcg = grid[0], -1.0
    scores = {}
    for val in grid:
        kwargs = {param_name: val, **fixed}
        run    = fuser_cls().fuse(bm25, dense, **kwargs)
        agg, _ = evaluate(run, qrels)
        ndcg   = agg["ndcg_cut_10"]
        scores[val] = ndcg
        if ndcg > best_ndcg:
            best_ndcg, best_param = ndcg, val
    return best_param, scores


# ─────────────────────────────────────────────────────────────────────── #
# Experiment 1 — Full Strategy × Dataset Matrix
# ─────────────────────────────────────────────────────────────────────── #

def exp1_matrix(cfg: dict, runs: dict, force: bool = False):
    log.info("\n" + "="*60)
    log.info("Experiment 1: Full Strategy × Dataset Matrix")
    out_path = Path(cfg["base"]["paths"]["report"]) / "tables" / "exp1_matrix.json"
    if out_path.exists() and not force:
        log.info("  Already done — skipping (--force to redo).")
        return json.loads(out_path.read_text())

    from src.fusion.strategy_a_linear     import LinearFusion
    from src.fusion.strategy_b_rrf        import RRFFusion
    from src.fusion.strategy_c_conv_rank  import ConvexRankFusion
    from src.fusion.strategy_d_learned    import LearnedFusion
    from src.fusion.strategy_e_adaptive   import AdaptiveFusion

    trained_dir = Path(cfg["base"]["paths"]["trained"])
    fcfg        = cfg["fusion"]

    # ── Tune A, B, C on MS MARCO dev ──────────────────────────────────
    log.info("  Tuning hyperparams on MS MARCO dev ...")
    ms_qrels  = get_qrels("msmarco", "dev", cfg)
    ms_bm25   = runs["msmarco"]["bm25"]
    ms_dense  = runs["msmarco"]["dense"]

    best_alpha, alpha_curve = tune_param_on_msmarco(
        LinearFusion, "alpha",
        fcfg["strategy_a"]["alpha_grid"], ms_bm25, ms_dense, ms_qrels)
    best_k, k_curve = tune_param_on_msmarco(
        RRFFusion, "k",
        fcfg["strategy_b"]["k_grid"], ms_bm25, ms_dense, ms_qrels)
    best_beta, beta_curve = tune_param_on_msmarco(
        ConvexRankFusion, "beta",
        fcfg["strategy_c"]["beta_grid"], ms_bm25, ms_dense, ms_qrels)

    log.info(f"  Best alpha={best_alpha}  k={best_k}  beta={best_beta}")

    # Load trained models D & E
    model_d = LearnedFusion(trained_dir / "strategyD_logreg.pkl")
    model_e = AdaptiveFusion(trained_dir / "strategyE_alpha_rf.pkl")

    # Load queries for Strategy E
    all_queries = {}
    for name, split in DATASETS:
        queries_path = Path(cfg["base"]["paths"]["data"]) / name / "queries.jsonl"
        with open(queries_path) as f:
            for line in f:
                obj = json.loads(line)
                qid  = obj.get("_id") or obj.get("id")
                text = obj.get("text", "")
                if qid:
                    all_queries[qid] = text

    # ── Evaluate all strategies on all datasets ────────────────────────
    results = {}
    per_query_all = {}

    for name, split in DATASETS:
        bm25_r  = runs[name]["bm25"]
        dense_r = runs[name]["dense"]
        qrels   = get_qrels(name, split, cfg)
        queries = {qid: all_queries[qid] for qid in bm25_r if qid in all_queries}

        results[name]      = {}
        per_query_all[name] = {}

        strategies = {
            "bm25_only":  bm25_r,
            "dense_only": dense_r,
            "A_linear":   LinearFusion().fuse(bm25_r, dense_r, alpha=best_alpha),
            "B_rrf":      RRFFusion().fuse(bm25_r, dense_r, k=best_k),
            "C_conv":     ConvexRankFusion().fuse(bm25_r, dense_r, beta=best_beta),
            "D_learned":  model_d.fuse(bm25_r, dense_r),
            "E_adaptive": model_e.fuse(bm25_r, dense_r, queries=queries),
        }

        for strat_name, run in strategies.items():
            agg, per_q = evaluate(run, qrels)
            results[name][strat_name]       = agg
            per_query_all[name][strat_name] = per_q
            log.info(f"  [{name}] {strat_name:<14} "
                     f"NDCG@10={agg['ndcg_cut_10']:.4f}  "
                     f"MAP={agg['map']:.4f}  "
                     f"R@100={agg['recall_100']:.4f}")

    # ── Significance tests vs RRF (best static baseline per hypothesis) ─
    sig = {}
    for name, split in DATASETS:
        sig[name] = {}
        rrf_pq = per_query_all[name]["B_rrf"]
        for strat in ["A_linear", "C_conv", "D_learned", "E_adaptive"]:
            t, p, n = paired_ttest(per_query_all[name][strat], rrf_pq)
            sig[name][strat] = {"t": t, "p": p, "n": n,
                                 "significant": is_significant(p)}

    out = {
        "best_params": {"alpha": best_alpha, "k": best_k, "beta": best_beta},
        "results":     results,
        "significance": sig,
        "alpha_curve": {str(k): v for k, v in alpha_curve.items()},
        "k_curve":     {str(k): v for k, v in k_curve.items()},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    # Save per-query metrics for significance tests
    pq_path = Path(cfg["base"]["paths"]["cache"]) / "exp1_per_query.json"
    pq_path.write_text(json.dumps(per_query_all))

    log.info(f"  Exp1 saved → {out_path}")
    return out


# ─────────────────────────────────────────────────────────────────────── #
# Experiment 2 — Parameter Sensitivity
# ─────────────────────────────────────────────────────────────────────── #

def exp2_sensitivity(cfg: dict, runs: dict, force: bool = False):
    log.info("\n" + "="*60)
    log.info("Experiment 2: Parameter Sensitivity")
    out_path = Path(cfg["base"]["paths"]["report"]) / "tables" / "exp2_sensitivity.json"
    if out_path.exists() and not force:
        log.info("  Already done — skipping.")
        return json.loads(out_path.read_text())

    import numpy as np
    from src.fusion.strategy_a_linear import LinearFusion
    from src.fusion.strategy_b_rrf    import RRFFusion

    alpha_grid = np.arange(0, 1.01, 0.05).tolist()
    k_grid     = [10, 30, 60, 100]

    sensitivity = {"alpha": {}, "k": {}}

    for name, split in DATASETS:
        bm25_r = runs[name]["bm25"]
        dense_r = runs[name]["dense"]
        qrels   = get_qrels(name, split, cfg)

        # Alpha sweep
        sensitivity["alpha"][name] = {}
        for a in alpha_grid:
            run = LinearFusion().fuse(bm25_r, dense_r, alpha=a)
            agg, _ = evaluate(run, qrels)
            sensitivity["alpha"][name][round(a, 2)] = agg["ndcg_cut_10"]

        # k sweep
        sensitivity["k"][name] = {}
        for k in k_grid:
            run = RRFFusion().fuse(bm25_r, dense_r, k=k)
            agg, _ = evaluate(run, qrels)
            sensitivity["k"][name][k] = agg["ndcg_cut_10"]

        best_alpha = max(sensitivity["alpha"][name], key=sensitivity["alpha"][name].get)
        log.info(f"  [{name}] best_alpha={best_alpha}  "
                 f"NDCG={sensitivity['alpha'][name][best_alpha]:.4f}")

    out_path.write_text(json.dumps(sensitivity, indent=2))
    log.info(f"  Exp2 saved → {out_path}")
    return sensitivity


# ─────────────────────────────────────────────────────────────────────── #
# Experiment 3 — Query-Level Analysis of Strategy E
# ─────────────────────────────────────────────────────────────────────── #

def exp3_query_analysis(cfg: dict, runs: dict, force: bool = False):
    log.info("\n" + "="*60)
    log.info("Experiment 3: Query-Level Analysis (Strategy E)")
    out_path = Path(cfg["base"]["paths"]["report"]) / "tables" / "exp3_query_analysis.json"
    if out_path.exists() and not force:
        log.info("  Already done — skipping.")
        return json.loads(out_path.read_text())

    import numpy as np
    from src.fusion.strategy_a_linear   import LinearFusion
    from src.fusion.strategy_e_adaptive import AdaptiveFusion, extract_query_features

    trained_dir = Path(cfg["base"]["paths"]["trained"])
    model_e     = AdaptiveFusion(trained_dir / "strategyE_alpha_rf.pkl")

    # Best alpha from Exp1 (or recompute)
    exp1_path = Path(cfg["base"]["paths"]["report"]) / "tables" / "exp1_matrix.json"
    best_alpha = 0.1   # default fallback
    if exp1_path.exists():
        best_alpha = json.loads(exp1_path.read_text())["best_params"]["alpha"]

    # Load all queries
    all_queries = {}
    for name, split in DATASETS:
        queries_path = Path(cfg["base"]["paths"]["data"]) / name / "queries.jsonl"
        with open(queries_path) as f:
            for line in f:
                obj = json.loads(line)
                qid  = obj.get("_id") or obj.get("id")
                text = obj.get("text", "")
                if qid:
                    all_queries[qid] = text

    analysis = {}
    bin_defs = cfg["fusion"]["strategy_e"]["alpha_bins"]

    for name in EVAL_DATASETS:
        split   = dict(DATASETS)[name]
        bm25_r  = runs[name]["bm25"]
        dense_r = runs[name]["dense"]
        qrels   = get_qrels(name, split, cfg)
        queries = {qid: all_queries[qid] for qid in bm25_r if qid in all_queries}

        # Predict per-query alpha
        predicted = model_e.predict_alphas(queries, bm25_r)
        feats     = extract_query_features(queries, bm25_r)

        # Compute NDCG@10 delta vs static best alpha
        static_run   = LinearFusion().fuse(bm25_r, dense_r, alpha=best_alpha)
        static_agg, static_pq = evaluate(static_run, qrels)

        # Per-query adaptive run
        adaptive_run   = model_e.fuse(bm25_r, dense_r, queries=queries)
        _, adaptive_pq = evaluate(adaptive_run, qrels)

        # Bin queries
        bins = {
            "dense_dominant": [],
            "balanced":       [],
            "sparse_dominant": [],
        }
        for qid, alpha in predicted.items():
            lo, hi = bin_defs["dense_dominant"]
            if alpha < hi:
                bins["dense_dominant"].append(qid)
            lo, hi = bin_defs["balanced"]
            if lo <= alpha <= hi:
                bins["balanced"].append(qid)
            lo, hi = bin_defs["sparse_dominant"]
            if alpha > lo:
                bins["sparse_dominant"].append(qid)

        bin_stats = {}
        feat_names = ["q_len", "mean_idf", "stopword_ratio", "is_question", "bm25_score_std"]
        for bin_name, qids in bins.items():
            if not qids:
                continue
            # Feature means
            feat_means = np.mean(
                [feats[qid] for qid in qids if qid in feats], axis=0
            ).tolist() if any(qid in feats for qid in qids) else [0]*5

            # NDCG@10 delta: adaptive - static
            deltas = [
                adaptive_pq.get(qid, {}).get("ndcg_cut_10", 0.0) -
                static_pq.get(qid,   {}).get("ndcg_cut_10", 0.0)
                for qid in qids
            ]
            bin_stats[bin_name] = {
                "count":        len(qids),
                "mean_alpha":   float(np.mean([predicted[q] for q in qids])),
                "ndcg_delta":   float(np.mean(deltas)),
                "feat_means":   dict(zip(feat_names, feat_means)),
            }

        analysis[name] = {
            "static_ndcg":   static_agg["ndcg_cut_10"],
            "adaptive_ndcg": evaluate(adaptive_run, qrels)[0]["ndcg_cut_10"],
            "bins":          bin_stats,
        }
        log.info(f"  [{name}] static={static_agg['ndcg_cut_10']:.4f}  "
                 f"adaptive={analysis[name]['adaptive_ndcg']:.4f}")
        for bn, bs in bin_stats.items():
            log.info(f"    {bn}: n={bs['count']}  delta={bs['ndcg_delta']:+.4f}")

    out_path.write_text(json.dumps(analysis, indent=2))
    log.info(f"  Exp3 saved → {out_path}")
    return analysis


# ─────────────────────────────────────────────────────────────────────── #
# Experiment 4 — Oracle Upper Bound
# ─────────────────────────────────────────────────────────────────────── #

def exp4_oracle(cfg: dict, runs: dict, force: bool = False):
    log.info("\n" + "="*60)
    log.info("Experiment 4: Oracle Upper Bound")
    out_path = Path(cfg["base"]["paths"]["report"]) / "tables" / "exp4_oracle.json"
    if out_path.exists() and not force:
        log.info("  Already done — skipping.")
        return json.loads(out_path.read_text())

    import numpy as np
    from src.fusion.strategy_e_adaptive import compute_oracle_alphas
    from src.fusion.strategy_a_linear   import LinearFusion
    from src.fusion.strategy_b_rrf      import RRFFusion

    exp1_path = Path(cfg["base"]["paths"]["report"]) / "tables" / "exp1_matrix.json"
    best_params = {"alpha": 0.1, "k": 60}
    if exp1_path.exists():
        bp = json.loads(exp1_path.read_text())["best_params"]
        best_params["alpha"] = bp["alpha"]
        best_params["k"]     = bp["k"]

    oracle_results = {}

    for name in EVAL_DATASETS:
        split   = dict(DATASETS)[name]
        bm25_r  = runs[name]["bm25"]
        dense_r = runs[name]["dense"]
        qrels   = get_qrels(name, split, cfg)

        log.info(f"  [{name}] Computing oracle (step=0.01) ...")
        oracle_alphas = compute_oracle_alphas(
            bm25_r, dense_r, qrels, alpha_step=0.01
        )

        # Oracle run: use each query's oracle alpha
        fuser      = LinearFusion()
        oracle_run = {}
        for qid, a in oracle_alphas.items():
            result = fuser.fuse(
                {qid: bm25_r.get(qid, [])},
                {qid: dense_r.get(qid, [])},
                alpha=a,
            )
            oracle_run[qid] = result[qid]

        oracle_agg, _ = evaluate(oracle_run, qrels)

        # Best practical strategy (from Exp1)
        best_static_run = RRFFusion().fuse(bm25_r, dense_r, k=best_params["k"])
        static_agg, _   = evaluate(best_static_run, qrels)

        gap = oracle_agg["ndcg_cut_10"] - static_agg["ndcg_cut_10"]
        log.info(f"  [{name}] oracle={oracle_agg['ndcg_cut_10']:.4f}  "
                 f"best_static(RRF)={static_agg['ndcg_cut_10']:.4f}  "
                 f"gap={gap:+.4f}")

        oracle_results[name] = {
            "oracle_ndcg":        oracle_agg["ndcg_cut_10"],
            "best_static_ndcg":   static_agg["ndcg_cut_10"],
            "gap":                gap,
            "alpha_distribution": {
                "mean": float(np.mean(list(oracle_alphas.values()))),
                "std":  float(np.std(list(oracle_alphas.values()))),
                "p25":  float(np.percentile(list(oracle_alphas.values()), 25)),
                "p75":  float(np.percentile(list(oracle_alphas.values()), 75)),
            },
        }

    out_path.write_text(json.dumps(oracle_results, indent=2))
    log.info(f"  Exp4 saved → {out_path}")
    return oracle_results


# ─────────────────────────────────────────────────────────────────────── #
# Main
# ─────────────────────────────────────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp",   default="1,2,3,4",
                        help="Comma-separated experiment numbers to run")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if output exists")
    args = parser.parse_args()

    cfg  = load_config()
    exps = [int(x.strip()) for x in args.exp.split(",")]

    log.info("Loading base runs ...")
    runs = load_runs(cfg)

    if 1 in exps:
        exp1_matrix(cfg, runs, force=args.force)
    if 2 in exps:
        exp2_sensitivity(cfg, runs, force=args.force)
    if 3 in exps:
        exp3_query_analysis(cfg, runs, force=args.force)
    if 4 in exps:
        exp4_oracle(cfg, runs, force=args.force)

    log.info("\n✅  All experiments complete.")
    log.info(f"   Results in: {cfg['base']['paths']['report']}/tables/")


if __name__ == "__main__":
    main()
