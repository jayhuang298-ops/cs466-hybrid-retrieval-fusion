#!/usr/bin/env python3
"""
06_train_fusion_models.py — Train Strategy D (LogReg) and Strategy E (Adaptive).

Both models are trained on MS MARCO dev data, then saved to models/trained/.

Usage:
    python scripts/06_train_fusion_models.py [--strategy d|e|all] [--force]
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, get_run_path
from src.utils.logging import get_logger
from src.utils.io import read_trec_run

log = get_logger("train_fusion", "logs/06_train_fusion.log")


def load_msmarco_qrels(cfg: dict) -> dict[str, dict[str, int]]:
    """Read MS MARCO dev qrels directly from TSV (fast, no corpus loading)."""
    qrels_path = Path(cfg["base"]["paths"]["data"]) / "msmarco" / "qrels" / "dev.tsv"
    qrels: dict[str, dict[str, int]] = {}
    with open(qrels_path) as f:
        next(f)   # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                qid, docid, rel = parts[0], parts[1], int(parts[2])
            elif len(parts) == 4:
                qid, docid, rel = parts[0], parts[2], int(parts[3])
            else:
                continue
            qrels.setdefault(qid, {})[docid] = rel
    log.info(f"qrels loaded: {len(qrels):,} queries")
    return qrels


def load_msmarco_queries(cfg: dict) -> dict[str, str]:
    """Read MS MARCO queries.jsonl."""
    queries_path = Path(cfg["base"]["paths"]["data"]) / "msmarco" / "queries.jsonl"
    queries = {}
    with open(queries_path) as f:
        for line in f:
            obj = json.loads(line)
            qid  = obj.get("_id") or obj.get("id")
            text = obj.get("text", "")
            if qid:
                queries[qid] = text
    log.info(f"queries loaded: {len(queries):,}")
    return queries


def train_strategy_d(bm25_run, dense_run, qrels, cfg, force=False):
    from src.fusion.strategy_d_learned import LearnedFusion
    model_path = Path(cfg["base"]["paths"]["trained"]) / "strategyD_logreg.pkl"
    if model_path.exists() and not force:
        log.info("Strategy D model already exists — skipping (--force to retrain).")
        return
    log.info("=== Training Strategy D (Learned LogReg) ===")
    model = LearnedFusion()
    model.train(bm25_run, dense_run, qrels)
    model.save(model_path)
    log.info(f"Strategy D saved → {model_path}")


def train_strategy_e(bm25_run, dense_run, qrels, queries, cfg, force=False):
    from src.fusion.strategy_e_adaptive import AdaptiveFusion
    import numpy as np

    for model_type in ["rf", "ridge"]:
        model_path = Path(cfg["base"]["paths"]["trained"]) / \
                     f"strategyE_alpha_{model_type}.pkl"
        if model_path.exists() and not force:
            log.info(f"Strategy E ({model_type}) already exists — skipping.")
            continue

        log.info(f"=== Training Strategy E — {model_type.upper()} ===")
        model = AdaptiveFusion()
        oracle = model.train(
            bm25_run, dense_run, qrels, queries,
            model_type=model_type,
        )
        model.save(model_path)

        # Save oracle alphas for Experiment 4
        oracle_path = Path(cfg["base"]["paths"]["cache"]) / "oracle_alphas_msmarco.json"
        oracle_path.parent.mkdir(parents=True, exist_ok=True)
        if not oracle_path.exists() or force:
            with open(oracle_path, "w") as f:
                json.dump(oracle, f)
            log.info(f"Oracle alphas saved → {oracle_path}")

        # Log alpha distribution
        vals = list(oracle.values())
        log.info(f"  Oracle alpha stats: mean={np.mean(vals):.3f}  "
                 f"std={np.std(vals):.3f}  "
                 f"dense-dom(<0.3)={sum(1 for v in vals if v<0.3)/len(vals)*100:.1f}%  "
                 f"sparse-dom(>0.7)={sum(1 for v in vals if v>0.7)/len(vals)*100:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="all", choices=["d", "e", "all"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config()

    log.info("Loading MS MARCO runs ...")
    bm25_run  = read_trec_run(get_run_path("bm25",  "msmarco", cfg))
    dense_run = read_trec_run(get_run_path("dense", "msmarco", cfg))
    log.info(f"  BM25:  {len(bm25_run):,} queries")
    log.info(f"  Dense: {len(dense_run):,} queries")

    qrels   = load_msmarco_qrels(cfg)
    queries = load_msmarco_queries(cfg)

    if args.strategy in ("d", "all"):
        train_strategy_d(bm25_run, dense_run, qrels, cfg, force=args.force)

    if args.strategy in ("e", "all"):
        train_strategy_e(bm25_run, dense_run, qrels, queries, cfg, force=args.force)

    log.info("\n✅  Fusion model training complete.")


if __name__ == "__main__":
    main()
