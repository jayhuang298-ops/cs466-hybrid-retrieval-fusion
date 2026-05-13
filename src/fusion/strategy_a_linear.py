"""
Strategy A — Score-Level Linear Interpolation.

score(d) = alpha * norm_BM25(d) + (1 - alpha) * norm_Dense(d)

Both scores are min-max normalized per query to [0, 1].
Missing documents (in only one list) receive score 0.0 after normalization.
"""
import numpy as np
from src.fusion.base import Fusion
from src.utils.normalize import minmax_normalize


class LinearFusion(Fusion):
    name = "strategy_a"

    def __init__(self, alpha: float = 0.5):
        """
        Args:
            alpha: Weight for BM25. 0.0 = dense only, 1.0 = BM25 only.
        """
        self.alpha = alpha

    def fuse(
        self,
        bm25_run:  dict[str, list[tuple[str, float]]],
        dense_run: dict[str, list[tuple[str, float]]],
        alpha: float | None = None,
        **kwargs,
    ) -> dict[str, list[tuple[str, float]]]:
        """
        Args:
            alpha: Override instance alpha for this call (used in grid search).
        """
        a = alpha if alpha is not None else self.alpha
        fused = {}

        for qid in set(bm25_run) | set(dense_run):
            b_dict = dict(bm25_run.get(qid, []))
            d_dict = dict(dense_run.get(qid, []))
            candidates = list(set(b_dict) | set(d_dict))

            b_scores = np.array([b_dict.get(c, 0.0) for c in candidates])
            d_scores = np.array([d_dict.get(c, 0.0) for c in candidates])

            # Min-max normalize within each query
            # (missing docs already have 0.0; normalization keeps them at 0)
            b_norm = minmax_normalize(b_scores)
            d_norm = minmax_normalize(d_scores)

            combined = a * b_norm + (1.0 - a) * d_norm
            order = np.argsort(-combined)
            fused[qid] = [(candidates[i], float(combined[i])) for i in order]

        return fused
