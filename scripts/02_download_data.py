#!/usr/bin/env python3
"""
02_download_data.py — Download BEIR datasets and pre-cache BGE model.

Usage (from project root, with hybridir conda env active):
    python scripts/02_download_data.py [--datasets all|msmarco|trec-covid|arguana|fiqa]
                                       [--skip-model]
                                       [--force]

Download order (smallest → largest to fail fast):
    arguana (8.67K) → fiqa (57K) → trec-covid (171K) → msmarco (8.84M)
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Allow running from project root without installing as package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, get_dataset_path
from src.utils.logging import get_logger

log = get_logger("download", "logs/02_download.log")

# Download order: small first
DOWNLOAD_ORDER = ["arguana", "fiqa", "trec-covid", "msmarco"]

# Expected corpus sizes for validation (allow ±5%)
EXPECTED_CORPUS = {
    "arguana":   8674,
    "fiqa":      57638,
    "trec-covid": 171332,
    "msmarco":   8841823,
}

EXPECTED_QUERIES = {
    "arguana":   1406,
    "fiqa":      648,
    "trec-covid": 50,
    "msmarco":   6980,   # dev split
}


def _robust_download(url: str, zip_path: Path) -> bool:
    """
    Download url → zip_path using wget (resume-capable) if available,
    else fall back to requests with integrity check.
    Deletes the file and returns False if the result is not a valid zip.
    """
    import subprocess, zipfile, shutil, requests

    # Remove corrupt leftovers first
    if zip_path.exists():
        try:
            zipfile.ZipFile(zip_path).close()
            log.info(f"  Existing zip looks valid, skipping re-download.")
            return True
        except zipfile.BadZipFile:
            log.warning(f"  Removing corrupt file: {zip_path}")
            zip_path.unlink()

    if shutil.which("wget"):
        log.info(f"  Using wget (resume-capable) ...")
        result = subprocess.run(
            ["wget", "-c", "--tries=5", "--timeout=60", "-O", str(zip_path), url],
            check=False,
        )
        if result.returncode != 0:
            log.error(f"  wget failed (code {result.returncode})")
            if zip_path.exists():
                zip_path.unlink()
            return False
    else:
        log.info(f"  wget not found — using requests with streaming ...")
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
        except Exception as e:
            log.error(f"  requests download failed: {e}")
            if zip_path.exists():
                zip_path.unlink()
            return False

    # Validate
    try:
        zipfile.ZipFile(zip_path).close()
        return True
    except zipfile.BadZipFile as e:
        log.error(f"  Downloaded file is not a valid zip: {e}")
        zip_path.unlink()
        return False


def download_dataset(name: str, cfg: dict, force: bool = False) -> bool:
    """Download and unzip one BEIR dataset. Returns True on success."""
    from beir.datasets.data_loader import GenericDataLoader
    import zipfile, shutil

    out_dir = get_dataset_path(name, cfg)
    ds_cfg = cfg["datasets"]["datasets"][name]
    split = ds_cfg["split"]

    # Skip if already present and not forced
    corpus_file = out_dir / "corpus.jsonl"
    if corpus_file.exists() and not force:
        log.info(f"[{name}] Already downloaded — skipping (use --force to re-download).")
        return True

    url = cfg["datasets"]["base_url"].format(name=name)
    data_root = out_dir.parent     # data/beir/
    zip_path = data_root / f"{name}.zip"

    log.info(f"[{name}] Downloading from {url} ...")
    ok = _robust_download(url, zip_path)
    if not ok:
        return False

    # Unzip
    log.info(f"[{name}] Extracting ...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(str(data_root))
    except Exception as e:
        log.error(f"[{name}] Extraction failed: {e}")
        return False

    # Verify by loading
    log.info(f"[{name}] Verifying corpus / queries / qrels ...")
    try:
        corpus, queries, qrels = GenericDataLoader(str(out_dir)).load(split=split)
    except Exception as e:
        log.error(f"[{name}] Load verification failed: {e}")
        return False

    n_corpus  = len(corpus)
    n_queries = len(queries)
    n_qrels   = sum(len(v) for v in qrels.values())

    exp_c = EXPECTED_CORPUS[name]
    exp_q = EXPECTED_QUERIES[name]

    c_ok = abs(n_corpus  - exp_c) / exp_c < 0.05
    q_ok = abs(n_queries - exp_q) / exp_q < 0.05 if name != "msmarco" else n_queries > 0

    log.info(f"[{name}] corpus={n_corpus:,} (expected ~{exp_c:,}) {'✅' if c_ok else '❌'}")
    log.info(f"[{name}] queries={n_queries:,} (expected ~{exp_q:,}) {'✅' if q_ok else '❌'}")
    log.info(f"[{name}] qrel pairs={n_qrels:,}")

    if not (c_ok and q_ok):
        log.warning(f"[{name}] Size mismatch — check your download.")
        return False

    # Save stats for later reference
    stats = {
        "name": name, "split": split,
        "n_corpus": n_corpus, "n_queries": n_queries, "n_qrels": n_qrels,
    }
    stats_path = out_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    log.info(f"[{name}] Stats saved to {stats_path}")
    return True


def download_bge_model(cfg: dict, force: bool = False) -> bool:
    """Pre-download BGE-base-en-v1.5 to models/ cache."""
    from sentence_transformers import SentenceTransformer
    import torch

    model_dir = Path(cfg["base"]["paths"]["models"]) / "bge-base-en-v1.5"
    if model_dir.exists() and any(model_dir.iterdir()) and not force:
        log.info("[BGE] Model already cached — skipping (use --force to re-download).")
        return True

    device = cfg["base"]["device"]   # "mps" on M4
    log.info("[BGE] Downloading BAAI/bge-base-en-v1.5 ...")
    try:
        model = SentenceTransformer(
            "BAAI/bge-base-en-v1.5",
            device=device,
            cache_folder=str(model_dir.parent),
        )
        # Quick encode test
        test_emb = model.encode(["test sentence"], normalize_embeddings=True)
        assert test_emb.shape == (1, 768), f"Unexpected embedding shape: {test_emb.shape}"
        log.info(f"[BGE] Embedding shape: {test_emb.shape} ✅")
        log.info(f"[BGE] Device used: {device}")
        return True
    except Exception as e:
        log.error(f"[BGE] Download/test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download BEIR datasets and BGE model.")
    parser.add_argument(
        "--datasets", default="all",
        help="Comma-separated dataset names or 'all'. Default: all"
    )
    parser.add_argument("--skip-model", action="store_true",
                        help="Skip BGE model download.")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if files exist.")
    args = parser.parse_args()

    cfg = load_config()

    # Resolve which datasets to download
    if args.datasets.strip().lower() == "all":
        targets = DOWNLOAD_ORDER
    else:
        targets = [d.strip() for d in args.datasets.split(",")]
        unknown = set(targets) - set(DOWNLOAD_ORDER)
        if unknown:
            log.error(f"Unknown datasets: {unknown}. Valid: {DOWNLOAD_ORDER}")
            sys.exit(1)
        # Keep the optimal download order
        targets = [d for d in DOWNLOAD_ORDER if d in targets]

    log.info(f"Datasets to download: {targets}")
    results = {}

    # --- Datasets ---
    for name in targets:
        ok = download_dataset(name, cfg, force=args.force)
        results[name] = ok
        if not ok:
            log.error(f"[{name}] FAILED. Fix the error before continuing.")

    # --- BGE Model ---
    if not args.skip_model:
        ok = download_bge_model(cfg, force=args.force)
        results["bge-model"] = ok

    # --- Summary ---
    log.info("\n=== Download Summary ===")
    all_ok = True
    for item, ok in results.items():
        status = "✅" if ok else "❌ FAILED"
        log.info(f"  {item}: {status}")
        if not ok:
            all_ok = False

    if all_ok:
        log.info("\n✅  All downloads complete. Ready for §3 (indexing).")
    else:
        log.error("\n❌  Some downloads failed — review errors above before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
