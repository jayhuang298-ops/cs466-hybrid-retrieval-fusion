"""
Strategy B — Reciprocal Rank Fusion (RRF).

score_RRF(d) = sum_i  1 / (k + rank_i(d))

Purely rank-based; score magnitudes are irrelevant.
"""
from collections import defaultdict
from src.fusion.base import Fusion


class RRFFusion(Fusion):
    name = "strategy_b"

    def __init__(self, k: int = 60):
        """
        Args:
            k: Smoothing constant. Typical values: 10, 30, 60, 100.
        """
        self.k = k

    def fuse(
        self,
        bm25_run:  dict[str, list[tuple[str, float]]],
        dense_run: dict[str, list[tuple[str, float]]],
        k: int | None = None,
        **kwargs,
    ) -> dict[str, list[tuple[str, float]]]:
        k_ = k if k is not None else self.k
        fused = {}

        for qid in set(bm25_run) | set(dense_run):
            scores: dict[str, float] = defaultdict(float)

            for rank, (docid, _) in enumerate(bm25_run.get(qid, []), start=1):
                scores[docid] += 1.0 / (k_ + rank)

            for rank, (docid, _) in enumerate(dense_run.get(qid, []), start=1):
                scores[docid] += 1.0 / (k_ + rank)

            fused[qid] = sorted(scores.items(), key=lambda x: -x[1])

        return fused
