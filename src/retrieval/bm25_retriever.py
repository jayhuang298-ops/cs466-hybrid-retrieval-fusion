"""BM25 retriever — thin Pyserini wrapper."""
import json
from pathlib import Path

from src.utils.logging import get_logger

log = get_logger("bm25_retriever")


def convert_corpus_to_pyserini(
    corpus: dict,
    output_dir: Path,
) -> Path:
    """
    Convert a BEIR corpus dict to Pyserini's JSON collection format.
    Each line: {"id": docid, "contents": "title text"}

    Returns the output directory path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "corpus.jsonl"
    if out_file.exists():
        log.info(f"  Pyserini corpus already exists at {out_file} — skipping conversion.")
        return output_dir

    log.info(f"  Converting {len(corpus):,} docs to Pyserini format → {out_file}")
    with open(out_file, "w") as f:
        for docid, doc in corpus.items():
            title = doc.get("title", "").strip()
            text  = doc.get("text",  "").strip()
            contents = (title + " " + text).strip() if title else text
            f.write(json.dumps({"id": docid, "contents": contents}) + "\n")
    return output_dir


def build_bm25_index(
    corpus_dir: Path,
    index_dir: Path,
    threads: int = 4,
) -> None:
    """Build a Lucene BM25 index via Pyserini."""
    import subprocess, sys
    if (index_dir / "index.properties").exists():
        log.info(f"  BM25 index already exists at {index_dir} — skipping.")
        return

    index_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"  Building BM25 index: {corpus_dir} → {index_dir}")
    cmd = [
        sys.executable, "-m", "pyserini.index.lucene",
        "--collection",  "JsonCollection",
        "--input",       str(corpus_dir),
        "--index",       str(index_dir),
        "--generator",   "DefaultLuceneDocumentGenerator",
        "--threads",     str(threads),
        "--storePositions",
        "--storeDocvectors",
        "--storeRaw",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Pyserini indexing failed:\n{result.stderr[-2000:]}"
        )
    log.info(f"  BM25 index built at {index_dir}")


class BM25Retriever:
    def __init__(self, index_path: str | Path, k1: float = 0.9, b: float = 0.4):
        from pyserini.search.lucene import LuceneSearcher
        self.index_path = str(index_path)
        self.searcher = LuceneSearcher(self.index_path)
        self.searcher.set_bm25(k1, b)
        log.info(f"BM25Retriever ready: {index_path} (k1={k1}, b={b})")

    def batch_search(
        self,
        queries: dict[str, str],
        k: int = 1000,
        threads: int = 4,
    ) -> dict[str, list[tuple[str, float]]]:
        """
        Returns {qid: [(docid, score), ...]} sorted by descending score.
        """
        qids  = list(queries.keys())
        texts = list(queries.values())
        raw = self.searcher.batch_search(texts, qids, k=k, threads=threads)
        return {
            qid: [(h.docid, h.score) for h in hits]
            for qid, hits in raw.items()
        }
