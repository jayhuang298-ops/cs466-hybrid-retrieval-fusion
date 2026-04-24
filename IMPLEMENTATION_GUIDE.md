# Hybrid Retrieval Fusion — Implementation Guideline

> A step-by-step blueprint for implementing the proposal "Systematic Exploration of Hybrid Retrieval Fusion Strategies for Dense and Sparse Retrieval" (Jay Huang, JHU CS466, 2026).
>
> This document is designed to be handed to another coding agent. It specifies directory layout, environment setup, data and model preparation, implementation details for all five fusion strategies, the experiment harness, and the final analysis/reporting pipeline.

---

## 0. High-Level Architecture

```
final_project/
├── README.md
├── requirements.txt
├── environment.yml                 # optional conda env
├── configs/
│   ├── base.yaml                   # paths, seeds, device
│   ├── datasets.yaml               # BEIR dataset registry
│   └── fusion.yaml                 # per-strategy hyperparams
├── data/
│   ├── beir/                       # raw BEIR datasets
│   │   ├── msmarco/
│   │   ├── trec-covid/
│   │   ├── arguana/
│   │   └── fiqa/
│   └── cache/                      # processed artifacts
├── indexes/
│   ├── bm25/                       # Lucene indexes (one per dataset)
│   └── dense/                      # FAISS indexes (one per dataset)
├── runs/                           # raw retrieval runs (TREC format)
│   ├── bm25/{dataset}.trec
│   └── dense/{dataset}.trec
├── fused_runs/                     # fused runs per strategy × dataset
│   └── {strategy}/{dataset}.trec
├── models/                         # HuggingFace caches + trained fusion models
│   ├── bge-base-en-v1.5/
│   └── trained/
│       ├── strategyD_logreg.pkl
│       └── strategyE_alpha_regressor.pkl
├── src/
│   ├── __init__.py
│   ├── utils/
│   │   ├── io.py                   # TREC run I/O, qrels loading
│   │   ├── normalize.py            # min-max, z-score
│   │   └── logging.py
│   ├── data/
│   │   ├── beir_loader.py          # download + load BEIR
│   │   └── features.py             # query feature extraction for Strategy E
│   ├── retrieval/
│   │   ├── bm25_retriever.py       # Pyserini wrapper
│   │   ├── dense_retriever.py      # BGE + FAISS wrapper
│   │   └── runner.py               # produce top-1000 runs
│   ├── fusion/
│   │   ├── base.py                 # Fusion ABC
│   │   ├── strategy_a_linear.py
│   │   ├── strategy_b_rrf.py
│   │   ├── strategy_c_conv_rank.py
│   │   ├── strategy_d_learned.py
│   │   └── strategy_e_adaptive.py
│   ├── eval/
│   │   ├── evaluator.py            # pytrec_eval wrapper
│   │   └── significance.py         # paired t-test
│   └── experiments/
│       ├── exp1_matrix.py          # 5 × 4 strategy × dataset
│       ├── exp2_sensitivity.py     # α and k sweeps
│       ├── exp3_query_analysis.py  # bin queries by α*
│       └── exp4_oracle.py          # oracle upper bound
├── scripts/
│   ├── 01_setup_env.sh
│   ├── 02_download_data.py
│   ├── 03_build_bm25_index.py
│   ├── 04_encode_and_build_faiss.py
│   ├── 05_run_baselines.py
│   ├── 06_train_learned_fusion.py
│   ├── 07_train_adaptive_fusion.py
│   ├── 08_run_experiments.py
│   └── 09_generate_report.py
├── notebooks/
│   ├── 01_exploratory.ipynb
│   ├── 02_param_sensitivity.ipynb
│   └── 03_query_analysis.ipynb
└── report/
    ├── figures/
    ├── tables/
    └── main.tex
```

**Design principles**

1. **Retrieve once, fuse many times.** Run BM25 and BGE exactly once per dataset, persist top-1000 with raw scores. All fusion experiments operate on these cached runs — they are CPU-only and fast to iterate.
2. **Unified TREC run format.** Every run is stored as `qid Q0 docid rank score tag`. All fusion strategies produce and consume this format. Evaluation uses `pytrec_eval` against standard BEIR qrels.
3. **Deterministic pipeline.** Fix seeds (`PYTHONHASHSEED`, `numpy`, `torch`). Store all hyperparameters in YAML for reproducibility.

---

## 1. Environment Setup (Week 1)

### 1.1 System prerequisites

- Python 3.10 (Pyserini is sensitive to Python/Java versions)
- Java 21 (required by Pyserini / Lucene 9.x)
- CUDA 11.8+ with a GPU ≥ 16 GB VRAM (for encoding MS MARCO's 8.84M passages)
- ~150 GB free disk (MS MARCO corpus + FAISS index + BM25 index)

### 1.2 Conda environment

```bash
conda create -n hybridir python=3.10 -y
conda activate hybridir
conda install -c conda-forge openjdk=21 maven -y   # required by Pyserini
```

### 1.3 Python dependencies (`requirements.txt`)

```
# Core retrieval
pyserini==0.22.1
faiss-gpu==1.7.2        # or faiss-cpu if no GPU
sentence-transformers==2.7.0
transformers>=4.40.0
torch>=2.1.0

# BEIR and evaluation
beir==2.0.0
pytrec-eval==0.5

# ML / data
scikit-learn>=1.3.0
numpy>=1.24
pandas>=2.0
scipy>=1.11

# Utilities
tqdm
pyyaml
matplotlib
seaborn
jupyterlab
```

Install:

```bash
pip install -r requirements.txt
python -c "import pyserini; import faiss; import torch; print(torch.cuda.is_available())"
```

### 1.4 Environment sanity check (`scripts/01_setup_env.sh`)

- Prints Python, Java, CUDA versions.
- Runs a 5-query BM25 query against a toy Pyserini prebuilt index (`msmarco-v1-passage`) to confirm Lucene works.
- Encodes 3 sentences with BGE to confirm HuggingFace + GPU.

---

## 2. Data and Model Preparation (Week 1)

### 2.1 BEIR datasets

Use the BEIR helper for consistent loading. All four datasets are public.

```python
# scripts/02_download_data.py
from beir import util
from beir.datasets.data_loader import GenericDataLoader

DATASETS = ["msmarco", "trec-covid", "arguana", "fiqa"]
BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"

for name in DATASETS:
    path = util.download_and_unzip(BASE_URL.format(name=name), "data/beir")
    # For MS MARCO use split="dev"; others use split="test"
    split = "dev" if name == "msmarco" else "test"
    corpus, queries, qrels = GenericDataLoader(path).load(split=split)
    print(name, len(corpus), len(queries), len(qrels))
```

**Key notes**

- MS MARCO: use the `dev` split (6,980 queries) for development/training, as specified in the proposal.
- For MS MARCO training data for Strategies D and E, use `qrels/train.tsv` and a sampled subset (e.g., 50K queries) — full 500K is overkill and slow.
- Store qrels in `data/beir/{name}/qrels/{split}.tsv` (BEIR default).

### 2.2 Models

```bash
# Pre-download BGE-base-en-v1.5 to local cache
python -c "from sentence_transformers import SentenceTransformer; \
  SentenceTransformer('BAAI/bge-base-en-v1.5', cache_folder='models/')"
```

BGE instructs adding a query prefix for retrieval: `"Represent this sentence for searching relevant passages: "`. Encode queries WITH the prefix; encode the corpus WITHOUT.

### 2.3 Disk budget check

| Artifact | ~Size |
|---|---|
| MS MARCO corpus (raw) | 3 GB |
| MS MARCO BM25 index | 2.5 GB |
| MS MARCO FAISS index (fp32, 768-d, 8.84M) | ~27 GB |
| Other datasets (combined) | ~5 GB |
| Runs + cache | ~10 GB |

If VRAM is tight, encode MS MARCO in fp16 and use `IndexFlatIP` for small corpora, `IndexIVFFlat` for MS MARCO. Record recall@1000 vs. exact to confirm ANN loss is negligible.

---

## 3. Base Retrieval (Week 2)

### 3.1 BM25 index and retrieval

Use Pyserini's JSON ingestion:

```python
# scripts/03_build_bm25_index.py
# For each dataset:
# 1. Convert BEIR corpus.jsonl -> Pyserini jsonl (id, contents).
# 2. Run:
#    python -m pyserini.index.lucene \
#       --collection JsonCollection \
#       --input data/beir/{name}/pyserini_corpus \
#       --index indexes/bm25/{name} \
#       --generator DefaultLuceneDocumentGenerator \
#       --threads 8 --storePositions --storeDocvectors --storeRaw
```

Retrieval (`src/retrieval/bm25_retriever.py`):

```python
from pyserini.search.lucene import LuceneSearcher

class BM25Retriever:
    def __init__(self, index_path, k1=0.9, b=0.4):
        self.searcher = LuceneSearcher(index_path)
        self.searcher.set_bm25(k1, b)

    def batch_search(self, queries: dict[str, str], k=1000, threads=8):
        # returns {qid: [(docid, score), ...]}
        hits = self.searcher.batch_search(
            list(queries.values()), list(queries.keys()), k=k, threads=threads
        )
        return {qid: [(h.docid, h.score) for h in hits[qid]] for qid in queries}
```

### 3.2 Dense index and retrieval

```python
# src/retrieval/dense_retriever.py
import faiss, numpy as np, torch
from sentence_transformers import SentenceTransformer

class DenseRetriever:
    def __init__(self, model_name="BAAI/bge-base-en-v1.5", device="cuda"):
        self.model = SentenceTransformer(model_name, device=device)
        self.query_prefix = "Represent this sentence for searching relevant passages: "

    def encode_corpus(self, texts, batch_size=256, fp16=True):
        with torch.cuda.amp.autocast(enabled=fp16):
            emb = self.model.encode(
                texts, batch_size=batch_size,
                normalize_embeddings=True,       # cosine = inner product
                show_progress_bar=True,
                convert_to_numpy=True,
            )
        return emb.astype("float32")

    def encode_queries(self, queries):
        qs = [self.query_prefix + q for q in queries]
        return self.model.encode(qs, normalize_embeddings=True, convert_to_numpy=True).astype("float32")

    def build_index(self, embeddings, use_ivf=False):
        d = embeddings.shape[1]
        if use_ivf:
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, 4096, faiss.METRIC_INNER_PRODUCT)
            index.train(embeddings); index.add(embeddings); index.nprobe = 64
        else:
            index = faiss.IndexFlatIP(d); index.add(embeddings)
        return index

    def search(self, index, q_embs, k=1000):
        scores, idxs = index.search(q_embs, k)
        return scores, idxs
```

`scripts/04_encode_and_build_faiss.py` drives this for every dataset and persists the FAISS index + `docid_map.npy`.

### 3.3 Baseline runs

`scripts/05_run_baselines.py`:

- For each `(dataset, retriever) in [(d, r) for d in datasets for r in ['bm25', 'dense']]`, produce top-1000 in TREC format.
- **Always save raw scores** — Strategy A needs them, Strategy D needs them.
- Evaluate NDCG@10 / MAP / Recall@100 immediately; log to `runs/baseline_metrics.json`.

Expected sanity numbers (rough zero-shot benchmarks from BEIR paper / BGE reports):

| Dataset | BM25 NDCG@10 | BGE NDCG@10 |
|---|---|---|
| MS MARCO | ~0.228 | ~0.41 |
| TREC-COVID | ~0.656 | ~0.78 |
| ArguAna | ~0.414 | ~0.63 |
| FiQA | ~0.236 | ~0.40 |

If you see large deviations, **stop and debug** before moving on.

---

## 4. Fusion Strategies (Weeks 3–4)

### 4.1 Common interface (`src/fusion/base.py`)

```python
class Fusion:
    name: str
    def fuse(
        self,
        bm25_run: dict[str, list[tuple[str, float]]],
        dense_run: dict[str, list[tuple[str, float]]],
        *,
        queries: dict[str, str] | None = None,
        corpus_stats: dict | None = None,
    ) -> dict[str, list[tuple[str, float]]]:
        ...
```

All strategies return top-k fused results per query. The runner writes them as TREC runs.

### 4.2 Strategy A — Linear Interpolation

```python
def fuse(self, bm25_run, dense_run, alpha=0.5):
    fused = {}
    for qid in bm25_run:
        # Min-max normalize within the query's candidate pool.
        b = dict(bm25_run[qid]); d = dict(dense_run[qid])
        candidates = set(b) | set(d)
        b_vals = np.array([b.get(x, 0.0) for x in candidates])
        d_vals = np.array([d.get(x, 0.0) for x in candidates])
        b_norm = _minmax(b_vals); d_norm = _minmax(d_vals)
        scores = alpha * b_norm + (1 - alpha) * d_norm
        order = np.argsort(-scores)
        fused[qid] = [(list(candidates)[i], float(scores[i])) for i in order[:1000]]
    return fused
```

**Decisions to lock in**

- Missing-doc convention: if a document is in only one list, its missing score is 0 after min-max (documented in the report).
- α grid: `np.arange(0, 1.01, 0.1)` for main sweep, `np.arange(0, 1.01, 0.05)` for the sensitivity figure.
- α is tuned on MS MARCO dev only, then frozen for cross-domain evaluation.

### 4.3 Strategy B — RRF

```python
def fuse(self, bm25_run, dense_run, k=60):
    fused = {}
    for qid in bm25_run:
        scores = defaultdict(float)
        for rank, (docid, _) in enumerate(bm25_run[qid], start=1):
            scores[docid] += 1.0 / (k + rank)
        for rank, (docid, _) in enumerate(dense_run[qid], start=1):
            scores[docid] += 1.0 / (k + rank)
        fused[qid] = sorted(scores.items(), key=lambda x: -x[1])[:1000]
    return fused
```

Sweep `k ∈ {10, 30, 60, 100}`.

### 4.4 Strategy C — Convex Rank Combination

- Normalize ranks to [0,1] via `1 - (rank-1)/(N-1)` where N is the list length (so rank-1 → 1.0).
- `score = β * rank_bm25 + (1-β) * rank_dense`.
- Missing doc ⇒ 0.
- Grid-search `β ∈ {0, 0.1, …, 1.0}` on MS MARCO dev.

### 4.5 Strategy D — Learned Linear Fusion

**Training data construction (MS MARCO train):**

1. Sample 50K training queries.
2. For each query, get top-100 from BM25 and top-100 from BGE; take the union.
3. Feature vector per `(query, doc)`: `[bm25_raw, bm25_norm_within_q, dense_raw, dense_norm_within_q, bm25_rank_norm, dense_rank_norm, in_bm25_topk, in_dense_topk]`.
4. Label: 1 if `(qid, docid) ∈ qrels` with relevance ≥ 1, else 0.
5. Stratified/negative sampling: keep all positives; sample 20 negatives per positive.
6. Train `LogisticRegression(class_weight="balanced")`; 5-fold CV on query-id to avoid leakage.

**Inference:** at query time, score each candidate with `predict_proba` and rank.

Persist the fitted model + the `StandardScaler` to `models/trained/strategyD_logreg.pkl`.

### 4.6 Strategy E — Query-Adaptive Fusion (primary novel contribution)

**Step 1 — Oracle α per training query:**

```python
for qid in train_queries:
    best_alpha, best_ndcg = 0.0, -1
    for a in np.arange(0, 1.01, 0.05):
        ndcg = ndcg_at_10(linear_fuse(qid, a), qrels[qid])
        if ndcg > best_ndcg:
            best_ndcg, best_alpha = ndcg, a
    oracle[qid] = best_alpha
```

Filter out queries with no relevant docs in top-1000 of either system (target is undefined).

**Step 2 — Query features (`src/data/features.py`):**

| Feature | How |
|---|---|
| `q_len` | whitespace token count |
| `mean_idf` | mean IDF over query terms from the target corpus (precompute IDF from BM25 index postings) |
| `stopword_ratio` | fraction of tokens in NLTK English stopwords |
| `is_question` | leading `{what, who, when, where, why, how, is, are, do, does, can}` or ends with `?` |
| `bm25_score_std` | std of BM25 scores in its top-100 for this query |

Optional extras worth ablating: `max_bm25_score`, `bm25_dense_rank_corr` (Kendall τ between top-100 lists).

**Step 3 — Train the α predictor:**

- `sklearn.ensemble.RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42)` as the primary model.
- `Ridge(alpha=1.0)` as an interpretable baseline — report both.
- 5-fold CV on query-id; metric = RMSE against oracle α and downstream NDCG@10 after fusion.
- Clip predictions to `[0, 1]`.

**Step 4 — Inference:** predict α per test query, then apply Strategy A with that α.

Persist to `models/trained/strategyE_alpha_regressor.pkl`.

---

## 5. Experiment Harness (Week 4)

All experiments are thin scripts that consume cached runs + fusion modules.

### 5.1 Experiment 1 — Strategy × Dataset matrix

```
datasets × {BM25 only, BGE only, A, B, C, D, E} → NDCG@10 | MAP | Recall@100
```

Emit `report/tables/exp1_matrix.csv` and a Markdown version.

### 5.2 Experiment 2 — Parameter sensitivity

- Strategy A: NDCG@10 vs α curve per dataset (21 points each).
- Strategy B: NDCG@10 vs k (four points).
- Save PNGs to `report/figures/exp2_alpha_curve.png`, `exp2_rrf_k.png`.
- **Key observation to report:** does the argmax α agree across datasets? (This answers RQ2.)

### 5.3 Experiment 3 — Query-level analysis (Strategy E)

- Predict α* on each test dataset's queries.
- Bin into `[0, 0.3)`, `[0.3, 0.7]`, `(0.7, 1]`.
- For each bin, report feature means + NDCG@10 delta over static-α baseline.
- Produce 2 figures: stacked bar of bin sizes per dataset, and radar/heatmap of feature profiles per bin.

### 5.4 Experiment 4 — Oracle upper bound

- For each test query, exhaustive grid `α ∈ arange(0, 1.01, 0.01)` → oracle NDCG@10.
- Report: **Best static** < **Adaptive (Strategy E)** ≤ **Oracle** per dataset.
- The Oracle − Strategy E gap quantifies the ceiling for any linear-fusion extension (motivates H4).

---

## 6. Evaluation and Statistical Testing

Use `pytrec_eval` with BEIR qrels. Canonical metric names:

- `ndcg_cut_10`, `map`, `recall_100`.

Significance (`src/eval/significance.py`):

```python
from scipy.stats import ttest_rel
def paired_ttest(per_query_a, per_query_b):
    # Align on common qids.
    keys = sorted(set(per_query_a) & set(per_query_b))
    a = [per_query_a[k] for k in keys]; b = [per_query_b[k] for k in keys]
    t, p = ttest_rel(a, b)
    return t, p
```

- Always report per-query metric vectors so pairwise tests across ANY two systems are possible.
- Threshold: p < 0.05. Bold the best per column in tables; add a dagger for statistically significant improvements over the RRF baseline.

---

## 7. Reproducibility Checklist

- [ ] Global seed set (`set_seed(42)` in every script).
- [ ] All hyperparameters read from `configs/*.yaml` — no magic numbers in code.
- [ ] Commit lockfile: `pip freeze > requirements.lock.txt`.
- [ ] `make all` (or `scripts/08_run_experiments.py --all`) re-runs the full pipeline from raw data.
- [ ] Save per-query metrics as `.json` alongside aggregated metrics.
- [ ] Git-ignore `data/`, `indexes/`, `runs/`, `models/` — publish only code + trained small fusion models (< 1 MB).

---

## 8. Execution Order (aligns with proposal §6 timeline)

| Order | Script | Input | Output |
|---|---|---|---|
| 1 | `01_setup_env.sh` | — | verified env |
| 2 | `02_download_data.py` | — | `data/beir/*` |
| 3 | `03_build_bm25_index.py` | corpora | `indexes/bm25/*` |
| 4 | `04_encode_and_build_faiss.py` | corpora + BGE | `indexes/dense/*` |
| 5 | `05_run_baselines.py` | indexes | `runs/bm25/*`, `runs/dense/*`, baseline metrics |
| 6 | `06_train_learned_fusion.py` | MS MARCO train | `models/trained/strategyD_logreg.pkl` |
| 7 | `07_train_adaptive_fusion.py` | MS MARCO train + features | `models/trained/strategyE_alpha_regressor.pkl` |
| 8 | `08_run_experiments.py --exp 1,2,3,4` | everything above | tables + figures |
| 9 | `09_generate_report.py` | tables + figures | populated LaTeX includes |

Expected wall-clock on a single RTX 4090 / A5000:

- Encoding MS MARCO corpus: 3–5 h.
- All other encoding: < 30 min combined.
- BM25 indexing: < 30 min combined.
- Retrieval runs: < 10 min each dataset.
- All fusion + experiments: < 1 h (CPU only).
- **Total first-run**: ~6–8 h. Re-runs of fusion only: minutes.

---

## 9. Common Pitfalls

1. **BGE query prefix.** Forgetting it silently degrades NDCG by 3–5 points. Unit-test that the prefix is applied exactly once.
2. **Score normalization scope.** Normalize *within each query*, not globally — global min-max leaks across queries and destroys the signal.
3. **Rank ties in Strategy C.** Use stable sort; break ties by docid to keep runs deterministic.
4. **MS MARCO train leakage.** Strategies D and E must NEVER see MS MARCO dev queries. Filter by qid.
5. **Oracle α trivially high.** If Oracle NDCG@10 ≈ 1.0, check that you're recomputing NDCG using `qrels`, not self-labels.
6. **Missing documents in only one run.** Decide the convention once (treat missing as score=0 post-normalization, rank=N+1) and document it.
7. **Pyserini Java heap.** Export `JAVA_OPTS="-Xmx32g"` before indexing MS MARCO, otherwise it OOMs.
8. **FAISS vs. exact search for ArguAna.** ArguAna corpus is tiny (8.67K); just use `IndexFlatIP`. Don't bother with IVF.
9. **Statistical significance direction.** Paired t-test assumes query alignment — sort both per-query vectors by the same qid list before calling `ttest_rel`.

---

## 10. Mapping Back to Research Questions

| RQ | Answered by | Deliverable |
|---|---|---|
| RQ1 — how strategies compare across domains | Experiment 1 | 5×4 matrix table + significance markers |
| RQ2 — stability of fusion parameters | Experiment 2 | α-curve figure; argmax-α table per dataset |
| RQ3 — can query features drive adaptive fusion | Experiments 3 & 4 | Bin analysis + Oracle-gap chart |

Hypotheses H1–H4 map cleanly onto these outputs; the final report should explicitly cite the figure/table that confirms or refutes each.

---

## 11. Hand-off Instructions for the Coding Agent

When you (the implementing agent) pick this up:

1. Create the directory skeleton from §0 first — don't deviate from the names; later scripts import by path.
2. Build in the strict order of §8. Do not move to step N+1 until step N's sanity check passes.
3. After §3 completes, compare baselines to the expected numbers in §3.3. Any gap > 2 NDCG@10 points is a red flag.
4. Implement the `Fusion` ABC and Strategy B (RRF) first — it's parameter-insensitive and provides a fast end-to-end smoke test.
5. Keep every script idempotent: re-running should detect existing artifacts and skip unless `--force` is passed.
6. Log every experiment run with a timestamped config hash so partial results can be reproduced.
7. When uncertain about a design choice (e.g. normalization convention for missing docs), pick the option documented here, record the choice in `report/decisions.md`, and move on — don't block the pipeline on unanswered questions.

Good luck.
