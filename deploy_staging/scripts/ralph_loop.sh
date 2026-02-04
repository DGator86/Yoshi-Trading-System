#!/usr/bin/env bash
set -euo pipefail

PHASE="${1:-A}"
MAX_ITERS="${2:-25}"

mkdir -p reports/latest
echo "$PHASE" > reports/latest/PHASE.txt

for i in $(seq 1 "$MAX_ITERS"); do
  echo "=== ITER $i / PHASE $PHASE ==="
  bash scripts/run_smoke.sh
  python -m gnosis.loop.summarize --report reports/latest/report.json --out reports/latest/last_report.md
  echo "ITER $i complete. Feed reports/latest/last_report.md + src/gnosis/loop/ralph_contract.md + configs/*.yaml to Claude for next patch."
done
