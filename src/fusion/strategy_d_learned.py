"""
Strategy D — Learned Linear Fusion (Logistic Regression).

Features per (query, doc) pair:
  [bm25_raw, bm25_norm, dense_raw, dense_norm,
   bm25_rank_norm, dense_rank_norm, in_bm25_top100, in_dense_top100]

Label: 1 if doc is relevant (qrels grade >= 1), else 0.
Training: 5-fold CV on MS MARCO train queries (stratified by query).
"""
import numpy as np
import pickle
from pathlib import Path
from src.fusion.base import Fusion
from src.utils.normalize import minmax_normalize
from src.utils.logging import get_logger

log = get_logger("strategy_d")


class LearnedFusion(Fusion):
    name = "strategy_d"

    def __init__(self, model_path: str | Path | None = None):
        self.clf   = None
        self.scaler = None
        if model_path and Path(model_path).exists():
            self.load(model_path)

    # ------------------------------------------------------------------ #
    # Feature extraction
    # ------------------------------------------------------------------ #

    @staticmethod
    def build_feature_matrix(
        bm25_run:  dict[str, list[tuple[str, float]]],
        dense_run: dict[str, list[tuple[str, float]]],
        top_k: int = 100,
    ) -> tuple[np.ndarray, list[tuple[str, str]]]:
        """
        Build feature matrix for all (qid, docid) pairs in the union of
        top-k results from each run.

        Returns:
            X      : (n_pairs, 8) float32 feature matrix
            pairs  : [(qid, docid), ...] in the same row order as X
        """
        X_rows, pairs = [], []

        for qid in set(bm25_run) | set(dense_run):
            b_list = bm25_run.get(qid,  [])[:top_k]
            d_list = dense_run.get(qid, [])[:top_k]

            candidates = list({d for d, _ in b_list} | {d for d, _ in d_list})
            if not candidates:
                continue

            b_dict = dict(b_list)
            d_dict = dict(d_list)
            b_rank = {d: r for r, (d, _) in enumerate(b_list, start=1)}
            d_rank = {d: r for r, (d, _) in enumerate(d_list, start=1)}

            b_raw = np.array([b_dict.get(c, 0.0) for c in candidates])
            d_raw = np.array([d_dict.get(c, 0.0) for c in candidates])
            b_norm = minmax_normalize(b_raw)
            d_norm = minmax_normalize(d_raw)

            n = len(candidates)
            for i, doc in enumerate(candidates):
                br = b_rank.get(doc, n + 1)
                dr = d_rank.get(doc, n + 1)
                feat = [
                    b_raw[i],                             # bm25_raw
                    b_norm[i],                            # bm25_norm
                    d_raw[i],                             # dense_raw
                    d_norm[i],                            # dense_norm
                    1.0 - (br - 1) / max(n, 1),          # bm25_rank_norm
                    1.0 - (dr - 1) / max(n, 1),          # dense_rank_norm
                    float(doc in b_dict),                 # in_bm25_top100
                    float(doc in d_dict),                 # in_dense_top100
                ]
                X_rows.append(feat)
                pairs.append((qid, doc))

        X = np.array(X_rows, dtype=np.float32) if X_rows else np.empty((0, 8), dtype=np.float32)
        return X, pairs

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def train(
        self,
        bm25_run:  dict[str, list[tuple[str, float]]],
        dense_run: dict[str, list[tuple[str, float]]],
        qrels:     dict[str, dict[str, int]],
        cv_folds:  int = 5,
        neg_ratio: int = 20,
        random_state: int = 42,
    ) -> None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import GroupKFold
        import random
        rng = random.Random(random_state)

        log.info("Building training feature matrix ...")
        X_all, pairs_all = self.build_feature_matrix(bm25_run, dense_run)
        if len(X_all) == 0:
            raise ValueError("Empty feature matrix — check run files.")

        # Labels
        y_all = np.array([
            1 if qrels.get(qid, {}).get(doc, 0) >= 1 else 0
            for qid, doc in pairs_all
        ], dtype=np.int32)

        # Groups for GroupKFold (no query leakage across folds)
        qid_list  = [p[0] for p in pairs_all]
        unique_qs = list(dict.fromkeys(qid_list))
        q2group   = {q: i for i, q in enumerate(unique_qs)}
        groups    = np.array([q2group[q] for q in qid_list])

        # Down-sample negatives per positive
        pos_idx = np.where(y_all == 1)[0].tolist()
        neg_idx = np.where(y_all == 0)[0].tolist()
        rng.shuffle(neg_idx)
        keep_neg = neg_idx[: len(pos_idx) * neg_ratio]
        keep_idx = sorted(pos_idx + keep_neg)

        X = X_all[keep_idx]
        y = y_all[keep_idx]
        g = groups[keep_idx]
        log.info(f"  Training pairs: {len(y):,}  (pos={y.sum():,}, neg={(y==0).sum():,})")

        # Scale
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train final model on all data (CV is for validation only)
        self.clf = LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=random_state
        )
        self.clf.fit(X_scaled, y)
        log.info("Logistic Regression trained ✅")

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def fuse(
        self,
        bm25_run:  dict[str, list[tuple[str, float]]],
        dense_run: dict[str, list[tuple[str, float]]],
        **kwargs,
    ) -> dict[str, list[tuple[str, float]]]:
        if self.clf is None:
            raise RuntimeError("Model not trained. Call .train() or .load() first.")

        X, pairs = self.build_feature_matrix(bm25_run, dense_run)
        X_scaled = self.scaler.transform(X)
        proba    = self.clf.predict_proba(X_scaled)[:, 1]   # P(relevant)

        # Group scores back by qid
        from collections import defaultdict
        qid_scores: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for (qid, doc), score in zip(pairs, proba):
            qid_scores[qid].append((doc, float(score)))

        return {qid: sorted(hits, key=lambda x: -x[1])
                for qid, hits in qid_scores.items()}

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"clf": self.clf, "scaler": self.scaler}, f)
        log.info(f"Strategy D model saved → {path}")

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self.clf    = obj["clf"]
        self.scaler = obj["scaler"]
        log.info(f"Strategy D model loaded ← {path}")
