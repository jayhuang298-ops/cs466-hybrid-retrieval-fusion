#!/usr/bin/env python3
"""
04_encode_and_build_faiss.py — Encode corpora with BGE and build FAISS indexes.

MS MARCO (~8.84M docs) takes ~60-90 min on M4 MPS.
Smaller datasets finish in minutes.

Usage:
    python scripts/04_encode_and_build_faiss.py [--datasets all|msmarco|...] [--force]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, get_index_path
from src.utils.logging import get_logger
from src.data.beir_loader import load_dataset, get_corpus_texts
from src.retrieval.dense_retriever import DenseRetriever

log = get_logger("build_faiss", "logs/04_build_faiss.log")

# Build small corpora with exact search; MS MARCO gets IVF
IVF_THRESHOLD = 500_000
ALL_DATASETS  = ["arguana", "fiqa", "trec-covid", "msmarco"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    cfg = load_config()
    device = cfg["base"]["device"]          # "mps" on M4
    model_cache = cfg["base"]["paths"]["models"]

    targets = ALL_DATASETS if args.datasets == "all" else \
              [d.strip() for d in args.datasets.split(",")]

    retriever = DenseRetriever(
        model_name="BAAI/bge-base-en-v1.5",
        device=device,
        cache_folder=model_cache,
    )

    results = {}
    for name in targets:
        log.info(f"\n{'='*50}")
        log.info(f"Dataset: {name}")
        index_dir = get_index_path("dense", name, cfg)

        if (index_dir / "index.faiss").exists() and not args.force:
            log.info(f"  FAISS index already exists — skipping (use --force).")
            results[name] = True
            continue

        try:
            t0 = time.time()

            # 1. Load corpus
            corpus, _, _ = load_dataset(name, cfg=cfg)
            docids, texts = get_corpus_texts(corpus)
            log.info(f"  Corpus: {len(texts):,} documents")

            # 2. Encode
            log.info(f"  Encoding with BGE on {device} (batch={args.batch_size}) ...")
            embeddings = retriever.encode_corpus(
                texts,
                batch_size=args.batch_size,
                show_progress=True,
            )
            log.info(f"  Embeddings shape: {embeddings.shape}")

            # 3. Build FAISS index
            use_ivf = len(texts) > IVF_THRESHOLD
            index = DenseRetriever.build_faiss_index(embeddings, use_ivf=use_ivf)

            # 4. Save
            DenseRetriever.save_index(index, docids, index_dir)

            elapsed = time.time() - t0
            log.info(f"  Done in {elapsed/60:.1f} min")
            results[name] = True

        except Exception as e:
            log.error(f"[{name}] FAILED: {e}", exc_info=True)
            results[name] = False

    log.info("\n=== FAISS Index Build Summary ===")
    for name, ok in results.items():
        log.info(f"  {name}: {'✅' if ok else '❌ FAILED'}")

    if not all(results.values()):
        sys.exit(1)
    log.info("\n✅  All FAISS indexes ready.")


if __name__ == "__main__":
    main()
