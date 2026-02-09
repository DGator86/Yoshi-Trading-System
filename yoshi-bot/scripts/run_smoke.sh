#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Prefer active venv python, fallback to python3
PY="${PYTHON:-python}"
if ! command -v "$PY" >/dev/null 2>&1; then
  PY="python3"
fi

"$PY" -m pytest -q
"$PY" scripts/run_experiment.py --config configs/experiment.yaml
