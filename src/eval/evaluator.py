"""Evaluation utilities wrapping pytrec_eval."""
import pytrec_eval
import numpy as np
from pathlib import Path
from src.utils.io import run_to_pytrec_format


METRICS = {"ndcg_cut_10", "map", "recall_100"}


def evaluate(
    run:   dict[str, list[tuple[str, float]]],
    qrels: dict[str, dict[str, int]],
    metrics: set[str] = METRICS,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """
    Evaluate a run against qrels.

    Returns:
        agg      : {metric: mean_value}
        per_query: {qid: {metric: value}}
    """
    run_pt = run_to_pytrec_format(run)
    ev     = pytrec_eval.RelevanceEvaluator(qrels, metrics)
    per_q  = ev.evaluate(run_pt)

    agg = {}
    for m in metrics:
        vals = [v[m] for v in per_q.values() if m in v]
        agg[m] = float(np.mean(vals)) if vals else 0.0

    return agg, per_q


def load_qrels_from_tsv(path: str | Path) -> dict[str, dict[str, int]]:
    """Read BEIR-style 3-col or TREC 4-col qrels TSV."""
    qrels: dict[str, dict[str, int]] = {}
    with open(path) as f:
        first = f.readline().strip()
        # Detect and skip header
        if not first[0].isdigit():
            pass   # header skipped
        else:
            # First line is data — re-parse it
            parts = first.split("\t")
            if len(parts) == 3:
                qrels.setdefault(parts[0], {})[parts[1]] = int(parts[2])
            elif len(parts) == 4:
                qrels.setdefault(parts[0], {})[parts[2]] = int(parts[3])

        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                qid, docid, rel = parts[0], parts[1], int(parts[2])
            elif len(parts) == 4:
                qid, docid, rel = parts[0], parts[2], int(parts[3])
            else:
                continue
            qrels.setdefault(qid, {})[docid] = rel
    return qrels
