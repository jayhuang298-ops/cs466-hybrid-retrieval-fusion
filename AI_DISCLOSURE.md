# AI Assistance Disclosure

**Course:** CS 466 — Information Retrieval, Johns Hopkins University  
**Author:** Jay Huang  

---

## Statement

This project was completed with the assistance of **Claude (Anthropic)**, an AI assistant. In accordance with course policy on AI tool usage, the nature and extent of AI assistance is disclosed below.

---

## Scope of AI Assistance

### 1. Framework & Architecture Design
- High-level project structure (directory layout, module organization)
- Pipeline design: data download → indexing → retrieval → fusion → evaluation
- Interface design for the abstract `Fusion` base class and strategy implementations

### 2. Boilerplate & Utility Code
- Configuration loading utilities (`src/utils/config.py`)
- TREC run file I/O helpers (`src/utils/io.py`, `src/utils/normalize.py`)
- Logging setup (`src/utils/logging.py`)
- Dataset loader wrapper around the BEIR library (`src/data/beir_loader.py`)

### 3. Evaluation & Significance Testing
- `pytrec_eval` wrapper (`src/eval/evaluator.py`)
- Paired t-test helper (`src/eval/significance.py`)

### 4. Report Generation Scripts
- Figure generation script (`scripts/09_generate_figures.py`)
- LaTeX table generation scripts (`scripts/10_generate_tables.py`, `scripts/12_generate_description_tables.py`)
- Findings analysis script (`scripts/11_write_analysis.py`)

### 5. Debugging Assistance
- Diagnosing and fixing data pipeline bugs (empty corpus files, qrels parsing errors, FAISS index errors)
- Identifying Python syntax errors (f-string format specs, unpacking bugs)

---

## Author's Own Contributions

All core research ideas, experimental design, and analytical conclusions are the author's own:

- **Research questions** and hypothesis formulation
- **Fusion strategy design** — mathematical formulations for Strategies A–E
- **Feature engineering** for Strategy D (8 pairwise features) and Strategy E (5 query features)
- **Oracle alpha analysis** — the per-query grid-search upper bound methodology
- **Experimental design** — the 4-experiment structure (matrix, sensitivity, query analysis, oracle)
- **Interpretation of results** — all findings, conclusions, and discussion
- **Google Colab workflow** — GPU encoding pipeline adaptation for Apple M4 hardware constraints
- **MS MARCO sub-sampling strategy** — reservoir sampling approach for the 500K sub-corpus

---

## Tools Used

| Tool | Purpose |
|---|---|
| Claude (Anthropic) | Code scaffolding, debugging, report generation |
| GitHub Copilot | — (not used) |
| ChatGPT | — (not used) |

---

*This disclosure is provided voluntarily in the interest of academic transparency.*
