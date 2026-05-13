# Appendix: System Implementation and Reproducibility Guide

This appendix provides a detailed account of the software environment, data preparation procedures, fusion strategy implementations, and experimental pipeline used in this project. The description is intended to be sufficiently complete for independent replication of the reported results.

---

## A.1 Repository Structure

The complete source code is organized as follows:

```
final_project/
├── configs/            # YAML configuration files (paths, seeds, hyperparameters)
├── data/beir/          # Raw BEIR datasets (excluded from version control)
├── indexes/            # BM25 Lucene and FAISS dense indexes (excluded from version control)
├── runs/               # Top-1000 BM25 and dense retrieval runs in TREC format
├── fused_runs/         # Fused ranked lists per strategy and dataset
├── models/trained/     # Serialized fusion model weights
├── src/
│   ├── data/           # Dataset loading and query feature extraction
│   ├── retrieval/      # BM25 and dense retrieval modules
│   ├── fusion/         # Implementations of Strategies A–E
│   └── eval/           # Metric computation and statistical testing
├── scripts/            # Numbered end-to-end pipeline scripts
├── notebooks/          # Google Colab notebook for GPU-based corpus encoding
└── report/             # Generated figures, LaTeX tables, and JSON result summaries
```

All hyperparameters are stored in `configs/*.yaml` and read at runtime. No numeric constants are hard-coded in the pipeline scripts.

The central architectural principle is *retrieve once, fuse many times*. BM25 and BGE retrieval runs are generated exactly once per dataset and persisted to disk. All five fusion strategies operate on these cached runs, which are CPU-only operations and can be iterated rapidly without re-running expensive indexing or embedding steps.

---

## A.2 Software Environment

**System requirements:**

- Python 3.10
- Java 21 (required by Pyserini / Lucene 9.x)
- CUDA-capable GPU with at least 16 GB VRAM (required only for BGE corpus encoding; all other steps run on CPU)
- Approximately 150 GB of available disk space

The Python environment is managed via conda:

```bash
conda create -n hybridir python=3.10 -y
conda activate hybridir
conda install -c conda-forge openjdk=21 maven -y
pip install -r requirements.txt
```

Core dependencies include `pyserini==0.22.1`, `faiss-gpu==1.7.2`, `sentence-transformers==2.7.0`, `beir==2.0.0`, `pytrec-eval==0.5`, `scikit-learn>=1.3.0`, and `scipy>=1.11`. A full dependency lockfile is provided as `requirements.lock.txt`.

Environment correctness can be verified by running:

```bash
python -c "import pyserini; import faiss; import torch; print(torch.cuda.is_available())"
```

For MS MARCO corpus encoding, which requires encoding 8.84 million passages, a Google Colab environment with a T4 GPU was used (see `notebooks/colab_compute.ipynb`). All other pipeline steps were executed on local CPU hardware.

---

## A.3 Datasets and Data Preparation

Four datasets from the BEIR benchmark were used. MS MARCO serves as the training and development corpus; the remaining three are used exclusively for zero-shot cross-domain evaluation.

**Table A.1: Dataset statistics.**

| Dataset    | Domain            | Role                 | Corpus Size        | Queries | Relevance |
| ---------- | ----------------- | -------------------- | ------------------ | ------- | --------- |
| MS MARCO   | Web / Q&A         | Training & tuning    | 500K (sub-sampled) | 6,980   | Binary    |
| TREC-COVID | Biomedical        | Zero-shot evaluation | 171,332            | 50      | Graded    |
| ArguAna    | Argument / Debate | Zero-shot evaluation | 8,674              | 1,406   | Binary    |
| FiQA       | Financial Q&A     | Zero-shot evaluation | 57,638             | 648     | Binary    |

The full MS MARCO corpus contains 8.84 million passages. Due to computational constraints, the corpus was reduced to 500,000 documents via reservoir sampling prior to BM25 indexing and BGE encoding. This sub-sampled corpus is used exclusively for training Strategies D and E; the evaluation datasets are processed in their entirety. MS MARCO uses the `dev` split for tuning; all other datasets use the `test` split.

Datasets are downloaded via the BEIR utility library (`scripts/02_download_data.py`) from the canonical BEIR distribution hosted at `https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/`.

---

## A.4 BM25 Indexing and Retrieval

BM25 indexes are constructed using Pyserini's Lucene backend (`scripts/03_build_bm25_index.py`). Each BEIR corpus is converted to Pyserini's JSON ingestion format and indexed with the following command:

```bash
python -m pyserini.index.lucene \
    --collection JsonCollection \
    --input data/beir/{dataset}/pyserini_corpus \
    --index indexes/bm25/{dataset} \
    --generator DefaultLuceneDocumentGenerator \
    --threads 8 --storePositions --storeDocvectors --storeRaw
```

BM25 parameters are fixed at `k1=0.9` and `b=0.4` throughout all experiments, consistent with the default BEIR evaluation settings. These parameters are not tuned on any dataset. When indexing MS MARCO, the Java heap must be explicitly extended by setting `JAVA_OPTS="-Xmx32g"` to prevent out-of-memory errors during index construction.

---

## A.5 Dense Encoding and FAISS Index Construction

Corpus embeddings are generated using the `BAAI/bge-base-en-v1.5` model via the `sentence-transformers` library (`scripts/04_encode_and_build_faiss.py`). Corpus documents are encoded without any prefix. Query embeddings are generated with the model-recommended prefix:

```
"Represent this sentence for searching relevant passages: "
```

This prefix is required by the BGE model for retrieval tasks and must be applied consistently. Omitting it silently degrades retrieval performance by approximately 3–5 NDCG@10 points.

All embeddings are L2-normalized, so inner product is equivalent to cosine similarity. For small corpora (ArguAna, FiQA, TREC-COVID), an exact `IndexFlatIP` index is used. For MS MARCO, an `IndexIVFFlat` index with 4,096 centroids and `nprobe=64` is used to reduce memory requirements. The MS MARCO FAISS index occupies approximately 27 GB on disk (fp32, 768-dimensional, 8.84M vectors).

---

## A.6 Baseline Retrieval Runs

Baseline retrieval runs are generated by `scripts/05_run_baselines.py`. For each dataset, the top 1,000 results are retrieved from both BM25 and BGE and stored in TREC run format (`qid Q0 docid rank score tag`). Raw relevance scores are preserved alongside ranks, as they are required by Strategies A and D.

The following baseline NDCG@10 values serve as sanity checks. Results deviating by more than two points from these targets indicate a configuration error and should be investigated before proceeding.

**Table A.2: Expected baseline NDCG@10 values.**

| Dataset        | BM25   | BGE-base-en-v1.5 |
| -------------- | ------ | ---------------- |
| MS MARCO (dev) | ~0.228 | ~0.410           |
| TREC-COVID     | ~0.656 | ~0.780           |
| ArguAna        | ~0.414 | ~0.630           |
| FiQA           | ~0.236 | ~0.400           |

---

## A.7 Fusion Strategy Implementations

All fusion strategies implement a common abstract interface that accepts the BM25 and dense runs as input and returns a fused ranked list. Fused runs are written to `fused_runs/{strategy}/{dataset}.trec`.

### A.7.1 Strategy A: Linear Score Interpolation

For each query, BM25 and dense scores are independently min-max normalized within the query's candidate pool. Documents absent from one ranked list are assigned a normalized score of zero. The fused score is computed as:

```
s_A(q, d) = (1 − α) × s̃_dense(q, d) + α × s̃_BM25(q, d)
```

The interpolation weight `α` is selected by grid search over `{0.0, 0.1, ..., 1.0}` on the MS MARCO development set, maximizing NDCG@10, and is then held fixed for all zero-shot evaluation datasets. Normalization is performed within each query independently; global normalization would introduce cross-query information leakage and is not used.

### A.7.2 Strategy B: Reciprocal Rank Fusion

RRF combines ranked lists using only rank positions, disregarding score magnitudes:

```
s_B(q, d) = Σ_{r ∈ {BM25, dense}}  1 / (k + rank_r(q, d))
```

The smoothing constant `k` is evaluated at `{10, 30, 60, 100}`. This method is robust to differences in score scale between the two retrievers but discards score confidence information.

### A.7.3 Strategy C: Convex Rank Fusion

Ranks from each retriever are normalized to [0, 1] such that rank 1 maps to 1.0, and then linearly interpolated:

```
s_C(q, d) = (1 − β) × r̃_dense(q, d) + β × r̃_BM25(q, d)
```

Documents absent from one list are assigned a normalized rank of zero. The weight `β` is tuned on MS MARCO dev via grid search over `{0.0, 0.1, ..., 1.0}`. A stable sort with document ID as a tiebreaker is used to ensure deterministic output across runs.

### A.7.4 Strategy D: Learned Logistic-Regression Fusion

A logistic regression classifier is trained on MS MARCO to predict binary document relevance from retrieval features. Training uses 50,000 queries sampled from the MS MARCO training split (disjoint from the development set used for tuning). For each query, the union of the top-100 BM25 results and top-100 dense results forms the candidate pool. Each (query, document) pair is represented by an eight-dimensional feature vector comprising raw BM25 score, normalized BM25 score, raw dense score, normalized dense score, normalized BM25 rank, normalized dense rank, and binary membership indicators for the BM25 top-k and dense top-k lists.

Relevance labels are derived from MS MARCO qrels (relevance ≥ 1 is treated as positive). To address class imbalance, all positive instances are retained and 20 negatives are sampled per positive. The model is trained with `class_weight="balanced"` and evaluated via five-fold cross-validation stratified on query IDs to prevent within-query leakage. Features are standardized using a `StandardScaler` fitted on the training data. The trained model and scaler are serialized to `models/trained/strategyD_logreg.pkl`. At inference time, documents are ranked by their predicted positive-class probability.

### A.7.5 Strategy E: Query-Adaptive Fusion

Strategy E extends Strategy A by replacing the global interpolation weight with a query-specific prediction. Training proceeds in three stages.

**Stage 1 — Oracle weight computation.** For each training query, Strategy A is applied with 101 candidate weights `α ∈ {0.00, 0.01, ..., 1.00}` and the weight maximizing NDCG@10 against the MS MARCO qrels is recorded as the oracle `α*(q)`. Queries for which no relevant document appears in the top-1,000 results of either system are excluded from training.

**Stage 2 — Query feature extraction.** Each query is represented by five features:

| Feature                       | Definition                                                   |
| ----------------------------- | ------------------------------------------------------------ |
| Query length                  | Number of whitespace-tokenized terms                         |
| Mean IDF                      | Mean inverse document frequency of query terms, computed from BM25 index posting statistics |
| Stopword ratio                | Fraction of query tokens appearing in the NLTK English stopword list |
| Question indicator            | Binary flag: query begins with an interrogative word or ends with `?` |
| BM25 score standard deviation | Standard deviation of BM25 scores across the top-100 retrieved documents |

**Stage 3 — Regressor training.** A `RandomForestRegressor` (300 estimators, maximum depth 8, random seed 42) is trained to predict `α*(q)` from the query feature vector. A `Ridge` regressor is also trained as an interpretable reference. Both models use five-fold cross-validation on query IDs. Predictions are clipped to [0, 1]. The trained regressor is serialized to `models/trained/strategyE_alpha_regressor.pkl`. At inference time, the predicted weight `α̂(q)` is passed to the Strategy A fusion formula.

---

## A.8 Evaluation Procedure

All retrieval metrics are computed using `pytrec_eval` against the official BEIR qrels. The primary metric is NDCG@10; MAP and Recall@100 are reported as supplementary metrics. Per-query metric vectors are saved as JSON files alongside aggregate results to support post-hoc significance analysis.

Statistical significance is assessed using two-sided paired t-tests (`scipy.stats.ttest_rel`) on per-query NDCG@10 vectors, with a significance threshold of `p < 0.05`. Prior to testing, both per-query vectors are aligned on the same sorted list of query IDs to ensure correct pairing.

---

## A.9 Execution Order and Estimated Runtime

The pipeline scripts are numbered and must be executed in the following order. Each step should be verified before proceeding.

| Step | Script                         | Purpose                                                      |
| ---- | ------------------------------ | ------------------------------------------------------------ |
| 1    | `01_setup_env.sh`              | Create and verify the conda environment                      |
| 2    | `02_download_data.py`          | Download all BEIR datasets                                   |
| 3    | `03_build_bm25_index.py`       | Construct Lucene BM25 indexes                                |
| 4    | `04_encode_and_build_faiss.py` | Encode corpora with BGE; build FAISS indexes                 |
| 5    | `05_run_baselines.py`          | Generate top-1000 BM25 and dense runs; verify sanity metrics |
| 6    | `06_train_fusion_models.py`    | Train Strategies D and E on MS MARCO                         |
| 7    | `08_run_experiments.py`        | Execute all four experiments                                 |
| 8    | `09_generate_figures.py`       | Produce all result figures                                   |
| 9    | `10_generate_tables.py`        | Produce all LaTeX tables                                     |

Estimated wall-clock times on a single NVIDIA RTX 4090: MS MARCO corpus encoding requires 3–5 hours; all remaining encoding and indexing steps require under one hour combined; all fusion experiments and evaluations run in under one hour on CPU. The complete pipeline requires approximately 6–8 hours on the first run. Subsequent re-runs of fusion and evaluation steps only require minutes, as retrieval runs are cached.
