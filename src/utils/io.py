"""TREC run file I/O and qrels utilities."""
from pathlib import Path


# ---------------------------------------------------------------------------
# TREC run format:  qid Q0 docid rank score tag
# ---------------------------------------------------------------------------

def write_trec_run(
    run: dict[str, list[tuple[str, float]]],
    path: str | Path,
    tag: str = "hybrid",
    max_docs: int = 1000,
) -> None:
    """Write a retrieval run to disk in TREC format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for qid, hits in run.items():
            for rank, (docid, score) in enumerate(hits[:max_docs], start=1):
                f.write(f"{qid} Q0 {docid} {rank} {score:.6f} {tag}\n")


def read_trec_run(path: str | Path) -> dict[str, list[tuple[str, float]]]:
    """
    Read a TREC run file.
    Returns {qid: [(docid, score), ...]} sorted by descending score.
    """
    run: dict[str, list[tuple[str, float]]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            qid, docid, score = parts[0], parts[2], float(parts[4])
            run.setdefault(qid, []).append((docid, score))
    # Ensure sorted by score descending
    for qid in run:
        run[qid].sort(key=lambda x: -x[1])
    return run


def read_qrels(path: str | Path) -> dict[str, dict[str, int]]:
    """
    Read a TREC qrels file.
    Returns {qid: {docid: relevance_grade}}.
    """
    qrels: dict[str, dict[str, int]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            qid, docid, rel = parts[0], parts[2], int(parts[3])
            qrels.setdefault(qid, {})[docid] = rel
    return qrels


def run_to_pytrec_format(
    run: dict[str, list[tuple[str, float]]]
) -> dict[str, dict[str, float]]:
    """Convert {qid: [(docid, score)]} → pytrec_eval input {qid: {docid: score}}."""
    return {qid: {docid: score for docid, score in hits} for qid, hits in run.items()}


def qrels_to_pytrec_format(
    qrels: dict[str, dict[str, int]]
) -> dict[str, dict[str, int]]:
    """Pass-through — qrels are already in pytrec_eval format."""
    return qrels
