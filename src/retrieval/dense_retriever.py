"""Dense retriever — BGE (Sentence-Transformers) + FAISS."""
import numpy as np
from pathlib import Path

from src.utils.logging import get_logger

log = get_logger("dense_retriever")

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class DenseRetriever:
    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: str = "mps",
        cache_folder: str | None = None,
    ):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(
            model_name, device=device, cache_folder=cache_folder
        )
        self.device = device
        log.info(f"DenseRetriever: {model_name} on {device}")

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_corpus(
        self,
        texts: list[str],
        batch_size: int = 256,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode corpus texts (no prefix). Returns float32 L2-normalized embeddings."""
        emb = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return emb.astype("float32")

    def encode_queries(
        self,
        queries: list[str],
        batch_size: int = 256,
    ) -> np.ndarray:
        """Encode queries WITH BGE prefix. Returns float32 L2-normalized embeddings."""
        prefixed = [QUERY_PREFIX + q for q in queries]
        emb = self.model.encode(
            prefixed,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return emb.astype("float32")

    # ------------------------------------------------------------------
    # Chunked encoding (for large corpora that don't fit in RAM)
    # ------------------------------------------------------------------

    def encode_corpus_chunked(
        self,
        texts: list[str],
        docids: list[str],
        chunk_dir: Path,
        chunk_size: int = 500_000,
        batch_size: int = 256,
    ) -> list[Path]:
        """
        Encode corpus in chunks of `chunk_size` docs, saving each chunk
        as a .npy file under chunk_dir.  Returns list of saved chunk paths.
        Skips chunks that already exist on disk (resume-friendly).
        """
        chunk_dir = Path(chunk_dir)
        chunk_dir.mkdir(parents=True, exist_ok=True)
        n = len(texts)
        chunk_paths = []

        for start in range(0, n, chunk_size):
            end       = min(start + chunk_size, n)
            chunk_idx = start // chunk_size
            emb_path  = chunk_dir / f"emb_{chunk_idx:04d}.npy"
            did_path  = chunk_dir / f"did_{chunk_idx:04d}.npy"

            if emb_path.exists() and did_path.exists():
                log.info(f"  Chunk {chunk_idx} already exists ({start:,}–{end:,}) — skipping.")
                chunk_paths.append(emb_path)
                continue

            log.info(f"  Encoding chunk {chunk_idx}: docs {start:,}–{end:,} ...")
            emb = self.encode_corpus(
                texts[start:end],
                batch_size=batch_size,
                show_progress=True,
            )
            np.save(str(emb_path), emb)
            np.save(str(did_path), np.array(docids[start:end]))
            log.info(f"  Saved chunk {chunk_idx} → {emb_path}")
            chunk_paths.append(emb_path)

        return chunk_paths

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    @staticmethod
    def build_faiss_index(
        embeddings: np.ndarray,
        use_ivf: bool = False,
        nlist: int = 4096,
        nprobe: int = 64,
    ):
        """
        Build a FAISS index from a single in-memory array.
        - Small corpora (<= 500K): IndexFlatIP (exact).
        - Large corpora (> 500K):  IndexIVFFlat (approximate).
        """
        import faiss
        d = embeddings.shape[1]
        n = embeddings.shape[0]

        auto_ivf = use_ivf or n > 500_000
        if auto_ivf:
            log.info(f"  Building IndexIVFFlat (n={n:,}, d={d}, nlist={nlist})")
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            log.info(f"  Training IVF quantizer on {min(n, 200_000):,} vectors ...")
            train_vecs = embeddings[:200_000] if n > 200_000 else embeddings
            index.train(train_vecs)
            index.nprobe = nprobe
        else:
            log.info(f"  Building IndexFlatIP (exact, n={n:,}, d={d})")
            index = faiss.IndexFlatIP(d)

        log.info(f"  Adding {n:,} vectors to index ...")
        index.add(embeddings)
        log.info(f"  Index total: {index.ntotal:,} vectors")
        return index

    @staticmethod
    def build_faiss_index_from_chunks(
        chunk_dir: Path,
        nlist: int = 4096,
        nprobe: int = 64,
        train_chunks: int = 1,
    ):
        """
        Build an IVFFlat FAISS index by loading embedding chunks one at a time.
        Memory usage = one chunk at a time (~1.5 GB per 500K docs).

        Args:
            chunk_dir:    directory with emb_*.npy files
            nlist:        IVF number of clusters
            nprobe:       search-time probe count
            train_chunks: how many chunks to use for IVF training
        """
        import faiss
        chunk_dir = Path(chunk_dir)
        emb_files = sorted(chunk_dir.glob("emb_*.npy"))
        if not emb_files:
            raise FileNotFoundError(f"No embedding chunks found in {chunk_dir}")

        # Peek at dimensionality
        sample = np.load(str(emb_files[0]), mmap_mode="r")
        d = sample.shape[1]
        log.info(f"  Building IVFFlat from {len(emb_files)} chunks, d={d}")

        # Train
        train_data = np.vstack([
            np.load(str(emb_files[i])) for i in range(min(train_chunks, len(emb_files)))
        ])
        train_vecs = train_data[:200_000]
        log.info(f"  Training IVF on {len(train_vecs):,} vectors ...")
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(train_vecs)
        index.nprobe = nprobe
        del train_data, train_vecs

        # Add chunks one by one
        all_docids = []
        for emb_path in emb_files:
            chunk_idx = emb_path.stem.split("_")[1]
            did_path  = chunk_dir / f"did_{chunk_idx}.npy"
            emb  = np.load(str(emb_path))
            dids = np.load(str(did_path), allow_pickle=True).tolist()
            log.info(f"  Adding chunk {chunk_idx}: {len(emb):,} vectors ...")
            index.add(emb)
            all_docids.extend(dids)
            del emb

        log.info(f"  Index total: {index.ntotal:,} vectors")
        return index, all_docids

    @staticmethod
    def save_index(index, docids: list[str], index_dir: Path) -> None:
        """Save FAISS index + docid mapping to disk."""
        import faiss
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_dir / "index.faiss"))
        np.save(str(index_dir / "docids.npy"), np.array(docids))
        log.info(f"  Saved FAISS index + docids to {index_dir}")

    @staticmethod
    def load_index(index_dir: Path):
        """Load FAISS index + docid mapping from disk."""
        import faiss
        index  = faiss.read_index(str(index_dir / "index.faiss"))
        docids = np.load(str(index_dir / "docids.npy"), allow_pickle=True).tolist()
        log.info(f"  Loaded FAISS index ({index.ntotal:,} vectors) from {index_dir}")
        return index, docids

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        index,
        docids: list[str],
        query_embeddings: np.ndarray,
        query_ids: list[str],
        k: int = 1000,
    ) -> dict[str, list[tuple[str, float]]]:
        """
        Run nearest-neighbour search.
        Returns {qid: [(docid, score), ...]} sorted by descending score.
        """
        scores, indices = index.search(query_embeddings, k)
        results = {}
        for i, qid in enumerate(query_ids):
            hits = []
            for j in range(k):
                idx = indices[i, j]
                if idx == -1:        # FAISS sentinel for no result
                    break
                hits.append((docids[idx], float(scores[i, j])))
            results[qid] = hits
        return results
