#!/usr/bin/env bash
set -euo pipefail
python -m pytest -q
python scripts/run_experiment.py --config configs/experiment.yaml
