#!/usr/bin/env python3
"""
05_run_baselines.py — Generate top-1000 BM25 and Dense baseline runs.

Outputs:
  runs/bm25/{dataset}.trec
  runs/dense/{dataset}.trec
  runs/baseline_metrics.json   ← NDCG@10, MAP, Recall@100 for all datasets

Usage:
    python scripts/05_run_baselines.py [--datasets all|...] [--force]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, get_index_path, get_run_path
from src.utils.logging import get_logger
from src.utils.io import write_trec_run, run_to_pytrec_format
from src.data.beir_loader import load_dataset
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever

log = get_logger("baselines", "logs/05_baselines.log")

ALL_DATASETS = ["arguana", "fiqa", "trec-covid", "msmarco"]

# Sanity-check thresholds: NDCG@10 must be at least this value
MIN_NDCG = {
    "bm25":  {"arguana": 0.35, "fiqa": 0.20, "trec-covid": 0.55, "msmarco": 0.18},
    "dense": {"arguana": 0.55, "fiqa": 0.30, "trec-covid": 0.65, "msmarco": 0.35},
}


def evaluate_run(run, qrels, metrics=None):
    """Run pytrec_eval and return metric dict."""
    import pytrec_eval
    if metrics is None:
        metrics = {"ndcg_cut_10", "map", "recall_100"}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
    per_query = evaluator.evaluate(run)
    agg = {}
    for metric in metrics:
        vals = [v[metric] for v in per_query.values() if metric in v]
        agg[metric] = sum(vals) / len(vals) if vals else 0.0
    return agg, per_query


def run_bm25(name, cfg, args) -> dict | None:
    run_path = get_run_path("bm25", name, cfg)
    if run_path.exists() and not args.force:
        log.info(f"  BM25 run exists — skipping (--force to redo).")
        from src.utils.io import read_trec_run
        return read_trec_run(run_path)

    index_dir = get_index_path("bm25", name, cfg)
    if not (index_dir / "index.properties").exists():
        raise FileNotFoundError(f"BM25 index missing: {index_dir}. Run 03 first.")

    ds_cfg  = cfg["datasets"]["datasets"][name]
    k1, b   = ds_cfg["bm25_params"]["k1"], ds_cfg["bm25_params"]["b"]
    retriever = BM25Retriever(index_dir, k1=k1, b=b)

    _, queries, _ = load_dataset(name, cfg=cfg)
    log.info(f"  BM25 searching {len(queries):,} queries ...")
    t0 = time.time()
    run = retriever.batch_search(queries, k=1000, threads=4)
    log.info(f"  Done in {time.time()-t0:.1f}s")

    write_trec_run(run, run_path, tag="bm25")
    log.info(f"  Saved → {run_path}")
    return run


def run_dense(name, cfg, args) -> dict | None:
    run_path = get_run_path("dense", name, cfg)
    if run_path.exists() and not args.force:
        log.info(f"  Dense run exists — skipping (--force to redo).")
        from src.utils.io import read_trec_run
        return read_trec_run(run_path)

    index_dir = get_index_path("dense", name, cfg)
    if not (index_dir / "index.faiss").exists():
        raise FileNotFoundError(f"FAISS index missing: {index_dir}. Run 04 first.")

    device      = cfg["base"]["device"]
    model_cache = cfg["base"]["paths"]["models"]
    retriever   = DenseRetriever(device=device, cache_folder=model_cache)

    index, docids = DenseRetriever.load_index(index_dir)
    _, queries, _ = load_dataset(name, cfg=cfg)

    qids   = list(queries.keys())
    qtexts = list(queries.values())
    log.info(f"  Encoding {len(qids):,} queries ...")
    q_embs = retriever.encode_queries(qtexts)

    log.info(f"  FAISS searching top-1000 ...")
    t0  = time.time()
    run = retriever.search(index, docids, q_embs, qids, k=1000)
    log.info(f"  Done in {time.time()-t0:.1f}s")

    write_trec_run(run, run_path, tag="dense")
    log.info(f"  Saved → {run_path}")
    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    targets = ALL_DATASETS if args.datasets == "all" else \
              [d.strip() for d in args.datasets.split(",")]

    all_metrics = {}

    for name in targets:
        log.info(f"\n{'='*55}")
        log.info(f"Dataset: {name}")

        _, _, qrels = load_dataset(name, cfg=cfg)
        qrels_pt = {qid: {did: int(r) for did, r in docs.items()}
                    for qid, docs in qrels.items()}

        metrics_entry = {}

        for retriever_name, run_fn in [("bm25", run_bm25), ("dense", run_dense)]:
            try:
                run = run_fn(name, cfg, args)
                run_pt = run_to_pytrec_format(run)
                agg, per_query = evaluate_run(run_pt, qrels_pt)

                ndcg = agg.get("ndcg_cut_10", 0.0)
                threshold = MIN_NDCG[retriever_name].get(name, 0.0)
                flag = "✅" if ndcg >= threshold else "⚠️  BELOW EXPECTED"

                log.info(
                    f"  [{retriever_name.upper()}] "
                    f"NDCG@10={ndcg:.4f} {flag}  "
                    f"MAP={agg.get('map', 0):.4f}  "
                    f"R@100={agg.get('recall_100', 0):.4f}"
                )
                metrics_entry[retriever_name] = {
                    "ndcg_cut_10": ndcg,
                    "map": agg.get("map", 0),
                    "recall_100": agg.get("recall_100", 0),
                    "per_query": per_query,
                }
            except Exception as e:
                log.error(f"  [{retriever_name.upper()}] FAILED: {e}", exc_info=True)
                metrics_entry[retriever_name] = None

        all_metrics[name] = metrics_entry

    # Save metrics (exclude per_query for the top-level summary file)
    summary = {
        ds: {
            r: {k: v for k, v in m.items() if k != "per_query"}
            for r, m in retrievers.items()
            if m is not None
        }
        for ds, retrievers in all_metrics.items()
    }
    metrics_path = Path(cfg["base"]["paths"]["report"]) / "tables" / "baseline_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(summary, indent=2))
    log.info(f"\nSummary saved → {metrics_path}")

    # Per-query metrics (needed for significance tests later)
    perq_path = Path(cfg["base"]["paths"]["cache"]) / "baseline_per_query.json"
    perq_path.parent.mkdir(parents=True, exist_ok=True)
    perq_out = {
        ds: {r: m["per_query"] for r, m in ret.items() if m}
        for ds, ret in all_metrics.items()
    }
    perq_path.write_text(json.dumps(perq_out))
    log.info(f"Per-query metrics saved → {perq_path}")

    log.info("\n✅  Baseline runs complete. Check ⚠️ warnings above vs expected ranges.")


if __name__ == "__main__":
    main()
