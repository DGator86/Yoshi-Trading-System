#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d .venv ] && source .venv/bin/activate || true
PYTHONPATH=./src python3 scripts/run_experiment.py \
  --config configs/experiment.yaml \
  --hparams configs/hparams.yaml
