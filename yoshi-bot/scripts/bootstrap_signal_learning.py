#!/usr/bin/env python3
"""Bootstrap scanner learning from historical API data.

Example:
  python3 yoshi-bot/scripts/bootstrap_signal_learning.py \
    --symbols BTCUSDT,ETHUSDT --days 120 --timeframe 1h
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add yoshi-bot root to import path for `src.*` imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover - optional dependency fallback
    load_dotenv = None

from src.gnosis.execution.historical_learning import (  # type: ignore
    HistoricalBootstrapConfig,
    bootstrap_learning_from_api,
)
from src.gnosis.execution.signal_learning import KalshiSignalLearner  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bootstrap learning outcomes from historical crypto API data")
    p.add_argument("--symbols", type=str, default="BTCUSDT", help="Comma-separated symbols")
    p.add_argument("--days", type=int, default=90, help="Historical days to fetch")
    p.add_argument("--timeframe", type=str, default="1h", help="Candle timeframe")
    p.add_argument("--horizon-bars", type=int, default=1, help="Forward horizon bars")
    p.add_argument("--min-abs-edge", type=float, default=0.08, help="Minimum absolute synthetic edge")
    p.add_argument("--max-records", type=int, default=4000, help="Maximum bootstrap rows to append")

    p.add_argument("--state-path", type=str, default="data/signals/learning_state.json")
    p.add_argument("--outcomes-path", type=str, default="data/signals/signal_outcomes.jsonl")
    p.add_argument("--policy-path", type=str, default="data/signals/learned_policy.json")
    p.add_argument("--learning-min-samples", type=int, default=30)
    p.add_argument("--learning-lookback", type=int, default=300)
    p.add_argument("--base-yes-edge", type=float, default=0.10)
    p.add_argument("--base-no-edge", type=float, default=0.13)
    return p.parse_args()


def main() -> int:
    # Load local .env when script is run manually from shell.
    if load_dotenv is not None:
        try:
            load_dotenv(Path(__file__).resolve().parents[1] / ".env")
            load_dotenv()
        except Exception:
            pass

    args = parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        print("No symbols provided.")
        return 2

    learner = KalshiSignalLearner(
        state_path=args.state_path,
        outcomes_path=args.outcomes_path,
        policy_path=args.policy_path,
        min_samples=max(5, int(args.learning_min_samples)),
        lookback=max(20, int(args.learning_lookback)),
        base_yes_edge=max(0.01, float(args.base_yes_edge)),
        base_no_edge=max(0.01, float(args.base_no_edge)),
    )
    cfg = HistoricalBootstrapConfig(
        symbols=symbols,
        days=max(5, int(args.days)),
        timeframe=str(args.timeframe),
        horizon_bars=max(1, int(args.horizon_bars)),
        min_abs_edge=max(0.01, float(args.min_abs_edge)),
        max_records=max(100, int(args.max_records)),
    )
    report = bootstrap_learning_from_api(learner, cfg)
    print(json.dumps(report, indent=2, default=str))

    policy_path = Path(args.policy_path)
    if policy_path.exists():
        print(f"policy_path={policy_path.resolve()}")
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
