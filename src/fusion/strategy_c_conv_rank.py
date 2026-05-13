"""
Strategy C — Convex Combination of Normalized Ranks.

rank_norm(d) = 1 - (rank(d) - 1) / (N - 1)   → rank 1 = 1.0, rank N = 0.0
score(d) = beta * rank_bm25_norm(d) + (1 - beta) * rank_dense_norm(d)

Missing documents receive rank N+1 (normalized to 0.0).
"""
import numpy as np
from src.fusion.base import Fusion


class ConvexRankFusion(Fusion):
    name = "strategy_c"

    def __init__(self, beta: float = 0.5):
        """
        Args:
            beta: Weight for BM25 ranks. 0.0 = dense only, 1.0 = BM25 only.
        """
        self.beta = beta

    def fuse(
        self,
        bm25_run:  dict[str, list[tuple[str, float]]],
        dense_run: dict[str, list[tuple[str, float]]],
        beta: float | None = None,
        **kwargs,
    ) -> dict[str, list[tuple[str, float]]]:
        b_ = beta if beta is not None else self.beta
        fused = {}

        for qid in set(bm25_run) | set(dense_run):
            b_list = bm25_run.get(qid, [])
            d_list = dense_run.get(qid, [])
            candidates = list({d for d, _ in b_list} | {d for d, _ in d_list})
            N = len(candidates)

            # Build rank lookup (1-indexed); missing → N+1
            b_rank = {d: r for r, (d, _) in enumerate(b_list, start=1)}
            d_rank = {d: r for r, (d, _) in enumerate(d_list, start=1)}

            denom = max(N, 1)  # avoid div-by-zero for single doc
            scores = []
            for doc in candidates:
                br = b_rank.get(doc, N + 1)
                dr = d_rank.get(doc, N + 1)
                # Normalize: rank 1 → 1.0, rank N+1 → 0.0
                bn = 1.0 - (br - 1) / denom
                dn = 1.0 - (dr - 1) / denom
                scores.append((doc, b_ * bn + (1.0 - b_) * dn))

            scores.sort(key=lambda x: -x[1])
            fused[qid] = scores

        return fused
