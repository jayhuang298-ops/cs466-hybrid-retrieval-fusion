"""Abstract base class for all fusion strategies."""
from abc import ABC, abstractmethod


class Fusion(ABC):
    """
    All fusion strategies share the same interface:

        fuse(bm25_run, dense_run, **kwargs) -> fused_run

    Input/output format:
        run = {qid: [(docid, score), ...]}   sorted by descending score
    """
    name: str = "base"

    @abstractmethod
    def fuse(
        self,
        bm25_run:  dict[str, list[tuple[str, float]]],
        dense_run: dict[str, list[tuple[str, float]]],
        **kwargs,
    ) -> dict[str, list[tuple[str, float]]]:
        ...

    @staticmethod
    def _all_candidates(
        bm25_run:  dict[str, list[tuple[str, float]]],
        dense_run: dict[str, list[tuple[str, float]]],
    ) -> dict[str, set[str]]:
        """Return union of docids per query from both runs."""
        qids = set(bm25_run) | set(dense_run)
        return {
            qid: {d for d, _ in bm25_run.get(qid, [])} |
                 {d for d, _ in dense_run.get(qid, [])}
            for qid in qids
        }
