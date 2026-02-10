#!/usr/bin/env python3
"""Run always-on ML/backtest/optimization supervisor."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gnosis.loop.continuous_learning import (  # noqa: E402
    ContinuousLearningConfig,
    ContinuousLearningSupervisor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuous learning supervisor")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/continuous_learning.yaml",
        help="Path to supervisor YAML config",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one scheduler tick and exit",
    )
    parser.add_argument(
        "--now",
        type=str,
        default=None,
        help="Override current timestamp for one-shot run (ISO-8601 UTC)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("run_continuous_learning")

    config = ContinuousLearningConfig.from_yaml(args.config)
    supervisor = ContinuousLearningSupervisor(config=config)

    if args.once:
        now_ts = pd.Timestamp(args.now, tz="UTC") if args.now else None
        out = supervisor.run_once(now_ts=now_ts)
        logger.info("One-shot tick: due=%s triggered=%s", out["due"], out["triggered"])
        return 0

    supervisor.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
