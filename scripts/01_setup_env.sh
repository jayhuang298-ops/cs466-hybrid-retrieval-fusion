#!/usr/bin/env zsh
# =============================================================
# 01_setup_env.sh — One-time environment setup for M4 Mac
# Run from the project root:  zsh scripts/01_setup_env.sh
# =============================================================
set -e

echo "=== [1/7] Checking Homebrew ==="
if ! command -v brew &>/dev/null; then
  echo "ERROR: Homebrew not found. Install from https://brew.sh"; exit 1
fi
echo "Homebrew: $(brew --version | head -1)"

# -------------------------------------------------------------
echo "=== [2/7] Installing miniforge (conda for Apple Silicon) ==="
if ! command -v conda &>/dev/null; then
  brew install miniforge
  # Initialize conda for zsh
  conda init zsh
  echo ""
  echo "⚠️  conda init wrote to ~/.zshrc."
  echo "    Please CLOSE this terminal, open a NEW one, then re-run this script."
  echo "    (conda needs a fresh shell to activate properly)"
  exit 0
else
  echo "conda already available: $(conda --version)"
fi

# -------------------------------------------------------------
echo "=== [3/7] Installing Java 21 (required by Pyserini/Lucene) ==="
if ! /usr/libexec/java_home -v 21 &>/dev/null 2>&1; then
  brew install openjdk@21
  # Symlink so macOS java_home can find it
  sudo ln -sfn "$(brew --prefix openjdk@21)/libexec/openjdk.jdk" \
               /Library/Java/JavaVirtualMachines/openjdk-21.jdk 2>/dev/null || true
fi
export JAVA_HOME="$(brew --prefix openjdk@21)/libexec/openjdk.jdk/Contents/Home"
echo "JAVA_HOME=$JAVA_HOME"
java -version

# -------------------------------------------------------------
echo "=== [4/7] Creating conda environment 'hybridir' (Python 3.11) ==="
if conda env list | grep -q "^hybridir "; then
  echo "Environment 'hybridir' already exists — skipping creation."
else
  conda create -n hybridir python=3.11 -y
fi

# Activate inside this script
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hybridir

# -------------------------------------------------------------
echo "=== [5/7] Installing faiss-cpu via conda (ARM64 native) ==="
conda install -c conda-forge faiss-cpu -y

# -------------------------------------------------------------
echo "=== [6/7] Installing PyTorch (MPS backend) + pip packages ==="

# PyTorch for Apple Silicon — standard pip wheel includes MPS
pip install --upgrade pip
pip install torch torchvision torchaudio

# All other dependencies
pip install -r requirements.txt

# Download NLTK stopwords (needed for Strategy E query features)
python -c "import nltk; nltk.download('stopwords', quiet=True); print('NLTK stopwords: OK')"

# Persist JAVA_HOME in the conda env so it's always set on activation
mkdir -p "$(conda info --base)/envs/hybridir/etc/conda/activate.d"
cat > "$(conda info --base)/envs/hybridir/etc/conda/activate.d/java_home.sh" <<EOF
export JAVA_HOME="$(brew --prefix openjdk@21)/libexec/openjdk.jdk/Contents/Home"
export PATH="\$JAVA_HOME/bin:\$PATH"
EOF

# -------------------------------------------------------------
echo "=== [7/7] Sanity check ==="
python - <<'PYCHECK'
import sys, platform
print(f"Python: {sys.version}")
print(f"Platform: {platform.machine()}")   # should be arm64

import torch
print(f"PyTorch: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")

import faiss
print(f"FAISS: {faiss.__version__}")

import sentence_transformers
print(f"Sentence-Transformers: {sentence_transformers.__version__}")

from pyserini.search.lucene import LuceneSearcher
searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage-slim')
hits = searcher.search('what is information retrieval', k=3)
print(f"Pyserini BM25 smoke test: top hit docid={hits[0].docid}, score={hits[0].score:.4f}")

import sklearn, numpy, pandas, scipy
print(f"scikit-learn: {sklearn.__version__}")
print(f"numpy: {numpy.__version__}")
print(f"pandas: {pandas.__version__}")

from beir.datasets.data_loader import GenericDataLoader
import importlib.metadata
beir_ver = importlib.metadata.version("beir")
print(f"BEIR: {beir_ver}")

print("\n✅  All checks passed. Environment is ready.")
PYCHECK
