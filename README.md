# Hybrid Retrieval Fusion: A Systematic Comparison of Five Fusion Strategies

**Course:** CS 466 — Information Retrieval, Johns Hopkins University  
**Author:** Jian Huang   Qingchen Li 
**Date:** April 2026

---

## Overview

This project systematically compares **five hybrid retrieval fusion strategies** that combine BM25 (sparse/lexical) and BGE-base-en-v1.5 (dense/semantic) retrievers across four BEIR benchmark datasets.

### Fusion Strategies

| ID | Strategy | Type |
|---|---|---|
| — | BM25 (Pyserini) | Baseline |
| — | Dense (BGE-base-en-v1.5) | Baseline |
| A | Linear Score Interpolation | Unsupervised |
| B | Reciprocal Rank Fusion (RRF) | Unsupervised |
| C | Convex Rank Fusion | Unsupervised |
| D | Learned Fusion (Logistic Regression) | Supervised |
| E | Query-Adaptive Fusion (Random Forest) | Supervised |

### Datasets (BEIR Benchmark)

| Dataset | Domain | Role | Corpus Size |
|---|---|---|---|
| MS MARCO | Web Q&A | Train & Tune | 500K (sub-sampled) |
| TREC-COVID | Biomedical | Zero-shot Eval | 171,332 |
| ArguAna | Argument/Debate | Zero-shot Eval | 8,674 |
| FiQA | Financial Q&A | Zero-shot Eval | 57,638 |

---

## Key Results (NDCG@10)

| Strategy | TREC-COVID | ArguAna | FiQA |
|---|---|---|---|
| BM25 | 0.5947 | 0.2999 | 0.2361 |
| Dense (BGE) | 0.7807 | 0.4558 | 0.4062 |
| A – Linear | 0.8233† | **0.4576**† | 0.4194† |
| B – RRF | 0.7653 | 0.4193 | 0.3857 |
| C – Convex Rank | 0.7807† | 0.4558† | 0.4062† |
| **D – Learned** | **0.8271**† | 0.4446† | **0.4194**† |
| E – Adaptive | 0.8152† | 0.4422† | 0.4130† |
| Oracle upper bound | 0.8863 | 0.4987 | 0.4928 |

† = significantly better than B-RRF baseline (paired t-test, p < 0.05)

---

## Repository Structure

```
.
├── configs/                  # YAML configuration files
│   ├── base.yaml             # Paths, device, retrieval settings
│   ├── datasets.yaml         # Dataset-specific settings
│   └── fusion.yaml           # Fusion hyperparameter grids
│
├── src/                      # Core library
│   ├── data/                 # BEIR data loader
│   ├── retrieval/            # BM25 (Pyserini) & Dense (FAISS) retrievers
│   ├── fusion/               # Strategies A–E
│   │   ├── strategy_a_linear.py
│   │   ├── strategy_b_rrf.py
│   │   ├── strategy_c_conv_rank.py
│   │   ├── strategy_d_learned.py
│   │   └── strategy_e_adaptive.py
│   ├── eval/                 # pytrec_eval wrapper + paired t-test
│   └── utils/                # Config, I/O, logging, normalization
│
├── scripts/                  # Numbered pipeline scripts
│   ├── 01_setup_env.sh       # Conda env setup (Python 3.11, FAISS, PyTorch MPS)
│   ├── 02_download_data.py   # BEIR dataset download
│   ├── 06_train_fusion_models.py  # Train Strategy D & E on MS MARCO
│   ├── 08_run_experiments.py      # Run all 4 experiments
│   ├── 09_generate_figures.py     # Generate PDF/PNG figures
│   ├── 10_generate_tables.py      # Generate LaTeX tables + significance tests
│   ├── 11_write_analysis.py       # Auto-generate findings.md
│   └── 12_generate_description_tables.py  # Dataset & model description tables
│
├── notebooks/
│   └── colab_compute.ipynb   # Google Colab notebook for GPU encoding (T4)
│
├── models/
│   └── trained/              # Trained fusion model weights (committed)
│       ├── strategyD_logreg.pkl
│       ├── strategyE_alpha_rf.pkl
│       └── strategyE_alpha_ridge.pkl
│
├── report/
│   ├── figures/              # fig1–fig4 (PDF + PNG)
│   ├── tables/               # LaTeX tables + JSON experiment results
│   └── findings.md           # Data-driven analysis writeup
│
├── requirements.txt
└── README.md
```

> **Note:** `data/`, `indexes/`, `runs/` are excluded from git (large files).  
> Download data with `python scripts/02_download_data.py`.  
> BM25/Dense runs were generated on Google Colab (T4 GPU) — see `notebooks/colab_compute.ipynb`.

---

## Quickstart

### 1. Environment Setup
```bash
bash scripts/01_setup_env.sh
conda activate hybridir
```

### 2. Download Data
```bash
python scripts/02_download_data.py
```

### 3. Generate BM25 + Dense Runs
Run `notebooks/colab_compute.ipynb` on Google Colab (free T4 GPU), then download the run files to `runs/`.

### 4. Train Fusion Models
```bash
python scripts/06_train_fusion_models.py --strategy all
```

### 5. Run All Experiments
```bash
python scripts/08_run_experiments.py
```

### 6. Generate Report Outputs
```bash
python scripts/09_generate_figures.py
python scripts/10_generate_tables.py
python scripts/11_write_analysis.py
python scripts/12_generate_description_tables.py
```

---

## Hardware Notes

- **Local (Apple M4):** CPU-only experiments (fusion, evaluation). Uses MPS for encoding if available.
- **Google Colab T4:** BM25 indexing + BGE dense encoding for all 4 datasets.
- **RAM:** MS MARCO corpus sub-sampled to 500K documents to fit within 12GB Colab RAM.

---

## AI Assistance Disclosure

This project used **Claude (Anthropic)** for code scaffolding, boilerplate utilities, debugging assistance, and report generation scripts. All research design, fusion strategy formulations, feature engineering, experimental methodology, and analytical conclusions are the author's own work. See [`AI_DISCLOSURE.md`](AI_DISCLOSURE.md) for full details.

---

## Dependencies

Key packages (see `requirements.txt` for full list):

```
beir
pyserini
faiss-cpu
torch
transformers
sentence-transformers
pytrec_eval
scikit-learn
scipy
matplotlib
numpy
```
