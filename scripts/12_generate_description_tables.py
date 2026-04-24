#!/usr/bin/env python3
"""
12_generate_description_tables.py — Generate dataset & model description tables.

Tables produced (saved to report/tables/):
  table_datasets.tex   — Dataset statistics and characteristics
  table_models.tex     — Fusion strategy descriptions

Usage:
    python scripts/12_generate_description_tables.py
"""
from pathlib import Path
import json

TABLES = Path("report/tables")
TABLES.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
# Table: Dataset Description
# ══════════════════════════════════════════════════════════════════════════
def table_datasets():
    datasets = [
        {
            "name":    "MS MARCO",
            "label":   "msmarco",
            "domain":  "Web / Q\\&A",
            "split":   "dev",
            "role":    "Train \\& Tune",
            "corpus":  "500K$^*$",
            "queries": "6,980",
            "qrels":   "7,437",
            "rel_scale": "Binary (0/1)",
            "note":    "Sub-sampled from 8.84M via reservoir sampling",
        },
        {
            "name":    "TREC-COVID",
            "label":   "trec-covid",
            "domain":  "Biomedical",
            "split":   "test",
            "role":    "Eval (zero-shot)",
            "corpus":  "171,332",
            "queries": "50",
            "qrels":   "66,336",
            "rel_scale": "Graded (0--2)",
            "note":    "COVID-19 literature retrieval; very dense judgments",
        },
        {
            "name":    "ArguAna",
            "label":   "arguana",
            "domain":  "Argument / Debate",
            "split":   "test",
            "role":    "Eval (zero-shot)",
            "corpus":  "8,674",
            "queries": "1,406",
            "qrels":   "1,406",
            "rel_scale": "Binary (0/1)",
            "note":    "Counter-argument retrieval; 1 relevant doc per query",
        },
        {
            "name":    "FiQA",
            "label":   "fiqa",
            "domain":  "Financial Q\\&A",
            "split":   "test",
            "role":    "Eval (zero-shot)",
            "corpus":  "57,638",
            "queries": "648",
            "qrels":   "1,706",
            "rel_scale": "Binary (0/1)",
            "note":    "Stack Exchange Finance; informal question style",
        },
    ]

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Dataset statistics. "
                 r"$^*$MS MARCO corpus sub-sampled to 500K documents via reservoir sampling "
                 r"for computational feasibility; used only for training and tuning, "
                 r"not for zero-shot evaluation.}")
    lines.append(r"\label{tab:datasets}")
    lines.append(r"\begin{tabular}{llllrrrl}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & Domain & Split & Role & \#Corpus & \#Queries & \#Qrels & Relevance \\")
    lines.append(r"\midrule")

    for d in datasets:
        lines.append(
            f"{d['name']} & {d['domain']} & {d['split']} & {d['role']} & "
            f"{d['corpus']} & {d['queries']} & {d['qrels']} & {d['rel_scale']} \\\\"
        )

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{8}{l}{\textit{Notes: "
                 r"TREC-COVID has very dense relevance judgments (avg.\ 1,327 per query). "
                 r"ArguAna has exactly one relevant document per query. "
                 r"FiQA uses informal financial language.}} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = TABLES / "table_datasets.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════
# Table: Model / Strategy Description
# ══════════════════════════════════════════════════════════════════════════
def table_models():
    models = [
        # ── Baselines ─────────────────────────────────────────────────────
        {
            "id":       "—",
            "name":     "BM25",
            "type":     "Baseline",
            "formula":  r"$\text{BM25}(q,d)$",
            "params":   r"$k_1{=}0.9,\ b{=}0.4$",
            "tunable":  "—",
            "notes":    "Pyserini/Lucene; sparse lexical retrieval",
        },
        {
            "id":       "—",
            "name":     "Dense (BGE)",
            "type":     "Baseline",
            "formula":  r"$\cos(\mathbf{q},\mathbf{d})$",
            "params":   r"768-d FAISS IP",
            "tunable":  "—",
            "notes":    r"\texttt{BAAI/bge-base-en-v1.5}; semantic retrieval",
        },
        # ── Fusion Strategies ─────────────────────────────────────────────
        {
            "id":       "A",
            "name":     "Linear Interpolation",
            "type":     "Unsupervised",
            "formula":  r"$(1{-}\alpha)\,\tilde{s}_\text{dense} + \alpha\,\tilde{s}_\text{BM25}$",
            "params":   r"$\alpha \in [0,1]$",
            "tunable":  r"$\alpha$",
            "notes":    "Min-max normalised per query; tuned on MS MARCO dev",
        },
        {
            "id":       "B",
            "name":     "Reciprocal Rank Fusion",
            "type":     "Unsupervised",
            "formula":  r"$\sum_{r} \frac{1}{k + r_i(d)}$",
            "params":   r"$k \in \{10,30,60,100\}$",
            "tunable":  r"$k$",
            "notes":    "Score-free; robust to score distribution differences",
        },
        {
            "id":       "C",
            "name":     "Convex Rank Fusion",
            "type":     "Unsupervised",
            "formula":  r"$(1{-}\beta)\,\tilde{r}_\text{dense} + \beta\,\tilde{r}_\text{BM25}$",
            "params":   r"$\beta \in [0,1]$",
            "tunable":  r"$\beta$",
            "notes":    "Rank-based interpolation; normalised rank scores",
        },
        {
            "id":       "D",
            "name":     "Learned Fusion (LogReg)",
            "type":     "Supervised",
            "formula":  r"$\sigma(\mathbf{w}^\top \phi(q,d))$",
            "params":   r"8 pairwise features",
            "tunable":  r"$\mathbf{w}$",
            "notes":    "Logistic Regression; trained on MS MARCO dev qrels",
        },
        {
            "id":       "E",
            "name":     "Query-Adaptive Fusion (RF)",
            "type":     "Supervised",
            "formula":  r"$\hat{\alpha}(q) = f_\theta(q)$",
            "params":   r"5 query features; RF $n{=}300$",
            "tunable":  r"$\theta$",
            "notes":    "Per-query $\alpha$ predicted by Random Forest; "
                        r"oracle $\alpha^*$ via grid-search on dev set",
        },
    ]

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Retrieval models and fusion strategies evaluated. "
                 r"Supervised strategies (D, E) are trained on MS MARCO dev and evaluated "
                 r"zero-shot on TREC-COVID, ArguAna, and FiQA.}")
    lines.append(r"\label{tab:models}")
    lines.append(r"\begin{tabular}{clllll}")
    lines.append(r"\toprule")
    lines.append(r"ID & Strategy & Type & Scoring Formula & Tunable Param & Notes \\")
    lines.append(r"\midrule")

    prev_type = None
    for m in models:
        if prev_type and prev_type != m["type"]:
            lines.append(r"\midrule")
        prev_type = m["type"]
        lines.append(
            f"{m['id']} & {m['name']} & {m['type']} & {m['formula']} & "
            f"{m['tunable']} & {m['notes']} \\\\"
        )

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{6}{l}{\textit{Query features for Strategy E: "
                 r"query length, mean IDF, stopword ratio, is-question flag, BM25 score std.}} \\")
    lines.append(r"\multicolumn{6}{l}{\textit{Features for Strategy D (pairwise): "
                 r"BM25/dense scores, rank diff, score diff, rank product, harmonic rank, "
                 r"score ratio, overlap.}} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = TABLES / "table_models.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating description tables ...")
    table_datasets()
    table_models()
    print("\n✅  Done.")
