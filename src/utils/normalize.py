"""Score and rank normalization utilities used across fusion strategies."""
import numpy as np


def minmax_normalize(scores: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Min-max normalize an array to [0, 1].
    All-equal arrays (range ≈ 0) map to 0.5 to avoid division by zero.
    """
    lo, hi = scores.min(), scores.max()
    if hi - lo < eps:
        return np.full_like(scores, 0.5, dtype=float)
    return (scores - lo) / (hi - lo)


def normalize_run_scores(
    run: dict[str, list[tuple[str, float]]],
    method: str = "minmax",
) -> dict[str, list[tuple[str, float]]]:
    """
    Normalize scores within each query's result list.

    Args:
        run:    {qid: [(docid, score), ...]}  (sorted by score desc)
        method: "minmax"
    Returns:
        Same structure with normalized scores; order preserved.
    """
    normed = {}
    for qid, hits in run.items():
        if not hits:
            normed[qid] = []
            continue
        docids = [d for d, _ in hits]
        raw    = np.array([s for _, s in hits], dtype=float)
        if method == "minmax":
            norm = minmax_normalize(raw)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        normed[qid] = list(zip(docids, norm.tolist()))
    return normed


def rank_normalize(
    run: dict[str, list[tuple[str, float]]],
    missing_value: float = 0.0,
    corpus_size: int | None = None,
) -> dict[str, dict[str, float]]:
    """
    Convert ranks to normalized [0, 1] scores.
    Rank 1 (best) → 1.0, rank N → 0.0 (or ~0 for large N).

    Args:
        run:           {qid: [(docid, score), ...]} sorted by descending score
        missing_value: value assigned to docs not in this list
        corpus_size:   if given, normalize over corpus_size instead of list length

    Returns:
        {qid: {docid: normalized_rank_score}}
    """
    result = {}
    for qid, hits in run.items():
        n = corpus_size if corpus_size else max(len(hits), 1)
        scores = {}
        for rank, (docid, _) in enumerate(hits, start=1):
            # rank 1 → 1 - (1-1)/(n-1) = 1.0 ;  rank n → 0.0
            scores[docid] = 1.0 - (rank - 1) / max(n - 1, 1)
        result[qid] = scores
    return result
