#!/usr/bin/env python3
"""
03_build_bm25_index.py — Build Lucene BM25 indexes for all 4 BEIR datasets.

Usage:
    python scripts/03_build_bm25_index.py [--datasets all|msmarco|...] [--force]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, get_dataset_path, get_index_path
from src.utils.logging import get_logger
from src.data.beir_loader import load_dataset
from src.retrieval.bm25_retriever import convert_corpus_to_pyserini, build_bm25_index

log = get_logger("build_bm25", "logs/03_build_bm25.log")

ALL_DATASETS = ["arguana", "fiqa", "trec-covid", "msmarco"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild indexes even if they exist.")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    cfg = load_config()
    targets = ALL_DATASETS if args.datasets == "all" else \
              [d.strip() for d in args.datasets.split(",")]

    results = {}
    for name in targets:
        log.info(f"\n{'='*50}")
        log.info(f"Dataset: {name}")
        try:
            # 1. Load corpus
            corpus, _, _ = load_dataset(name, cfg=cfg)

            # 2. Convert to Pyserini format
            pyserini_dir = get_dataset_path(name, cfg) / "pyserini_corpus"
            if args.force and pyserini_dir.exists():
                import shutil; shutil.rmtree(pyserini_dir)
            convert_corpus_to_pyserini(corpus, pyserini_dir)

            # 3. Build index
            index_dir = get_index_path("bm25", name, cfg)
            if args.force and index_dir.exists():
                import shutil; shutil.rmtree(index_dir)
            build_bm25_index(pyserini_dir, index_dir, threads=args.threads)

            results[name] = True
        except Exception as e:
            log.error(f"[{name}] FAILED: {e}", exc_info=True)
            results[name] = False

    log.info("\n=== BM25 Index Build Summary ===")
    for name, ok in results.items():
        log.info(f"  {name}: {'✅' if ok else '❌ FAILED'}")

    if not all(results.values()):
        sys.exit(1)
    log.info("\n✅  All BM25 indexes ready.")


if __name__ == "__main__":
    main()
