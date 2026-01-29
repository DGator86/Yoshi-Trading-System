#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
# Minimal deps for current repo
python -m pip install -U \
  pytest pyyaml pandas numpy scikit-learn pyarrow

echo
echo "✅ Venv ready. Activate with:"
echo "   source .venv/bin/activate"
