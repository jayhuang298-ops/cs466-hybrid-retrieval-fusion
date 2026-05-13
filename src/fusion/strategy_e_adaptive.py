"""
Strategy E — Query-Adaptive Fusion (novel contribution).

For each query, predict the optimal linear interpolation weight alpha*
using a regression model trained on MS MARCO dev queries.

Training pipeline:
  1. For each training query, find oracle alpha that maximises NDCG@10.
  2. Extract 5 query features.
  3. Train Ridge / RandomForest regressor to predict alpha from features.

Inference:
  alpha* = regressor.predict(features(q))
  fused_score(d) = alpha* * norm_BM25(d) + (1-alpha*) * norm_Dense(d)
"""
import numpy as np
import pickle
from pathlib import Path
from src.fusion.base import Fusion
from src.fusion.strategy_a_linear import LinearFusion
from src.utils.logging import get_logger

log = get_logger("strategy_e")


# ─────────────────────────────────────────────────────────────────────── #
# Query feature extraction
# ─────────────────────────────────────────────────────────────────────── #

def extract_query_features(
    queries: dict[str, str],
    bm25_run: dict[str, list[tuple[str, float]]],
    corpus_idf: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """
    Extract 5 features per query.

    Features:
      0  q_len          — number of whitespace tokens
      1  mean_idf       — mean IDF of query terms (0 if corpus_idf not given)
      2  stopword_ratio — fraction of tokens that are English stopwords
      3  is_question    — 1 if starts with interrogative or ends with '?'
      4  bm25_score_std — std of BM25 scores in top-100 (query specificity)

    Returns:
        {qid: np.ndarray shape (5,)}
    """
    import re
    try:
        from nltk.corpus import stopwords
        SW = set(stopwords.words("english"))
    except Exception:
        SW = set()

    INTERROGATIVES = {"what", "who", "when", "where", "why", "how",
                      "is", "are", "do", "does", "can", "will", "which"}

    features = {}
    for qid, text in queries.items():
        tokens = text.lower().split()
        n = len(tokens)

        q_len          = float(n)
        mean_idf       = float(np.mean([corpus_idf.get(t, 0.0) for t in tokens])
                               if corpus_idf and n > 0 else 0.0)
        stopword_ratio = float(sum(1 for t in tokens if t in SW) / n) if n > 0 else 0.0
        is_question    = float(
            (tokens[0] in INTERROGATIVES if tokens else False) or text.strip().endswith("?")
        )

        # BM25 score dispersion in top-100
        hits = bm25_run.get(qid, [])[:100]
        if len(hits) > 1:
            bm25_score_std = float(np.std([s for _, s in hits]))
        else:
            bm25_score_std = 0.0

        features[qid] = np.array(
            [q_len, mean_idf, stopword_ratio, is_question, bm25_score_std],
            dtype=np.float32,
        )
    return features


def compute_corpus_idf(
    corpus: dict,
    min_df: int = 2,
) -> dict[str, float]:
    """Compute IDF for each term from a BEIR corpus dict."""
    from collections import Counter
    import math

    df: Counter = Counter()
    N = len(corpus)
    for doc in corpus.values():
        title = doc.get("title", "")
        text  = doc.get("text",  "")
        terms = set((title + " " + text).lower().split())
        df.update(terms)

    return {
        term: math.log((N + 1) / (count + 1)) + 1.0
        for term, count in df.items()
        if count >= min_df
    }


# ─────────────────────────────────────────────────────────────────────── #
# Oracle alpha computation
# ─────────────────────────────────────────────────────────────────────── #

def compute_oracle_alphas(
    bm25_run:   dict[str, list[tuple[str, float]]],
    dense_run:  dict[str, list[tuple[str, float]]],
    qrels:      dict[str, dict[str, int]],
    alpha_step: float = 0.05,
) -> dict[str, float]:
    """
    For each query, find the alpha in [0,1] that maximises NDCG@10.
    Returns {qid: oracle_alpha}.
    Queries with no relevant docs in either top-1000 are skipped.
    """
    import pytrec_eval

    alphas  = np.arange(0, 1.0 + alpha_step / 2, alpha_step)
    fuser   = LinearFusion()
    oracle  = {}

    # Pre-filter: only keep queries that have relevant docs in the runs
    valid_qids = [
        qid for qid in qrels
        if qid in bm25_run or qid in dense_run
    ]
    log.info(f"Computing oracle alphas for {len(valid_qids):,} queries ...")

    for qid in valid_qids:
        rel_docs = set(qrels[qid].keys())
        b_docs   = {d for d, _ in bm25_run.get(qid,  [])}
        d_docs   = {d for d, _ in dense_run.get(qid, [])}
        if not rel_docs & (b_docs | d_docs):
            continue   # no relevant docs retrievable

        qrel_single = {qid: qrels[qid]}
        best_a, best_ndcg = 0.0, -1.0

        for a in alphas:
            fused_single = fuser.fuse(
                {qid: bm25_run.get(qid,  [])},
                {qid: dense_run.get(qid, [])},
                alpha=float(a),
            )
            run_pt = {qid: {d: s for d, s in hits}
                      for qid, hits in fused_single.items()}
            ev     = pytrec_eval.RelevanceEvaluator(qrel_single, {"ndcg_cut_10"})
            score  = ev.evaluate(run_pt).get(qid, {}).get("ndcg_cut_10", 0.0)
            if score > best_ndcg:
                best_ndcg, best_a = score, float(a)

        oracle[qid] = best_a

    log.info(f"Oracle alphas computed: {len(oracle):,} queries")
    return oracle


# ─────────────────────────────────────────────────────────────────────── #
# Adaptive fusion model
# ─────────────────────────────────────────────────────────────────────── #

class AdaptiveFusion(Fusion):
    name = "strategy_e"

    def __init__(self, model_path: str | Path | None = None):
        self.regressor = None
        self.scaler    = None
        self.model_type: str = "rf"
        if model_path and Path(model_path).exists():
            self.load(model_path)

    def train(
        self,
        bm25_run:    dict[str, list[tuple[str, float]]],
        dense_run:   dict[str, list[tuple[str, float]]],
        qrels:       dict[str, dict[str, int]],
        queries:     dict[str, str],
        corpus_idf:  dict[str, float] | None = None,
        model_type:  str = "rf",       # "rf" or "ridge"
        alpha_step:  float = 0.05,
        random_state: int = 42,
    ) -> dict[str, float]:
        """
        Train the alpha predictor.

        Returns oracle alpha dict (useful for analysis / oracle experiment).
        """
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        self.model_type = model_type

        # Step 1: oracle alphas
        log.info("Step 1: Computing oracle alphas ...")
        oracle = compute_oracle_alphas(bm25_run, dense_run, qrels, alpha_step)

        # Step 2: query features
        log.info("Step 2: Extracting query features ...")
        feats = extract_query_features(queries, bm25_run, corpus_idf)

        # Align oracle ↔ features
        common_qids = [qid for qid in oracle if qid in feats]
        log.info(f"  Training examples: {len(common_qids):,}")

        X = np.stack([feats[qid] for qid in common_qids])
        y = np.array([oracle[qid] for qid in common_qids], dtype=np.float32)

        # Step 3: fit regressor
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        if model_type == "rf":
            self.regressor = RandomForestRegressor(
                n_estimators=300, max_depth=8, random_state=random_state, n_jobs=-1
            )
        else:
            self.regressor = Ridge(alpha=1.0)

        self.regressor.fit(X_scaled, y)
        log.info(f"Strategy E ({model_type}) trained ✅")
        return oracle

    def fuse(
        self,
        bm25_run:  dict[str, list[tuple[str, float]]],
        dense_run: dict[str, list[tuple[str, float]]],
        queries:   dict[str, str] | None = None,
        corpus_idf: dict[str, float] | None = None,
        **kwargs,
    ) -> dict[str, list[tuple[str, float]]]:
        if self.regressor is None:
            raise RuntimeError("Model not trained. Call .train() or .load() first.")
        if queries is None:
            raise ValueError("queries dict required for Strategy E inference.")

        feats = extract_query_features(queries, bm25_run, corpus_idf)
        fuser = LinearFusion()
        fused = {}

        for qid in set(bm25_run) | set(dense_run):
            if qid not in feats:
                # Fallback to alpha=0.5 if no features available
                alpha_pred = 0.5
            else:
                x = feats[qid].reshape(1, -1)
                alpha_pred = float(np.clip(
                    self.regressor.predict(self.scaler.transform(x))[0], 0.0, 1.0
                ))
            result = fuser.fuse(
                {qid: bm25_run.get(qid,  [])},
                {qid: dense_run.get(qid, [])},
                alpha=alpha_pred,
            )
            fused[qid] = result[qid]

        return fused

    def predict_alphas(
        self,
        queries:   dict[str, str],
        bm25_run:  dict[str, list[tuple[str, float]]],
        corpus_idf: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Return predicted alpha per query (for analysis in Experiment 3)."""
        feats = extract_query_features(queries, bm25_run, corpus_idf)
        result = {}
        for qid, feat in feats.items():
            x = feat.reshape(1, -1)
            alpha_pred = float(np.clip(
                self.regressor.predict(self.scaler.transform(x))[0], 0.0, 1.0
            ))
            result[qid] = alpha_pred
        return result

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "regressor":  self.regressor,
                "scaler":     self.scaler,
                "model_type": self.model_type,
            }, f)
        log.info(f"Strategy E model saved → {path}")

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self.regressor  = obj["regressor"]
        self.scaler     = obj["scaler"]
        self.model_type = obj.get("model_type", "rf")
        log.info(f"Strategy E model loaded ← {path}")
