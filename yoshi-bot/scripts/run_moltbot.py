#!/usr/bin/env python3
"""Run Moltbot orchestration for a single forecast payload.

Usage:
    python scripts/run_moltbot.py --config configs/moltbot.yaml
    python scripts/run_moltbot.py --config configs/moltbot.yaml --forecast data/forecast.json
    python scripts/run_moltbot.py --config configs/moltbot.yaml --notify
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gnosis.execution import MoltbotOrchestrator, load_moltbot_config


DEFAULT_FORECAST = {
    "symbol": "BTCUSDT",
    "direction": "up",
    "confidence": 0.72,
    "q05": 64200,
    "q50": 66800,
    "q95": 70100,
}


def load_forecast(path: str | None) -> dict:
    if not path:
        return DEFAULT_FORECAST
    forecast_path = Path(path)
    if not forecast_path.exists():
        raise FileNotFoundError(f"Forecast file not found: {forecast_path}")
    with forecast_path.open() as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Moltbot orchestration")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/moltbot.yaml",
        help="Path to Moltbot YAML config",
    )
    parser.add_argument(
        "--forecast",
        type=str,
        default=None,
        help="Path to forecast JSON payload (optional)",
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        help="Send the trade plan to configured services",
    )
    parser.add_argument(
        "--use-stub",
        action="store_true",
        help="Force the stub AI client (no external API calls)",
    )
    args = parser.parse_args()

    config = load_moltbot_config(args.config)
    if args.use_stub:
        config.ai.provider = "stub"

    forecast = load_forecast(args.forecast)
    orchestrator = MoltbotOrchestrator(config)
    trade_plan = orchestrator.propose_trade(forecast)

    print("MOLT BOT TRADE PLAN")
    print("=" * 40)
    print(json.dumps(trade_plan, indent=2))

    if args.notify:
        responses = orchestrator.notify(trade_plan)
        print("\nNOTIFICATIONS")
        print("=" * 40)
        print(json.dumps(responses, indent=2))
    else:
        print("\nNotifications skipped. Use --notify to send.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
