#!/usr/bin/env python3
"""Run backtest on model predictions.

Usage:
    python scripts/run_backtest.py --predictions reports/latest/predictions.parquet --out reports/backtest/
    python scripts/run_backtest.py --config configs/backtest.yaml --predictions reports/latest/predictions.parquet
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from gnosis.backtest import (
    BacktestConfig,
    BacktestRunner,
    ExecutionConfig,
    PositionConfig,
    SignalConfig,
    load_config_from_yaml,
)


def main():
    parser = argparse.ArgumentParser(description="Run backtest on predictions")
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to predictions.parquet",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/backtest.yaml"),
        help="Path to backtest config YAML",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: reports/<run_id>/)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed",
    )
    args = parser.parse_args()

    # Load config
    if args.config.exists():
        print(f"Loading config from {args.config}")
        config = load_config_from_yaml(args.config)
    else:
        print("Using default config")
        config = BacktestConfig(
            signal=SignalConfig(),
            position=PositionConfig(),
            execution=ExecutionConfig(),
        )

    # Override seed if provided
    if args.seed is not None:
        config.random_seed = args.seed

    # Determine output directory
    if args.out is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("reports") / f"backtest_{run_id}"
    else:
        out_dir = args.out

    # Load predictions
    print(f"Loading predictions from {args.predictions}")
    predictions_df = pd.read_parquet(args.predictions)
    print(f"  {len(predictions_df)} rows, {len(predictions_df['symbol'].unique())} symbols")

    # Run backtest
    print("Running backtest...")
    runner = BacktestRunner(config)
    result = runner.run(predictions_df)

    # Save results
    print(f"Saving results to {out_dir}/")
    runner.save_results(result, out_dir)

    # Print summary
    print("\n=== Backtest Results ===")
    print(f"Initial Capital: ${config.initial_capital:,.2f}")
    print(f"Final Equity:    ${result.stats['final_equity']:,.2f}")
    print(f"Total Return:    {result.stats['total_return_pct']:.2f}%")
    print(f"Sharpe Ratio:    {result.stats['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:    {result.stats['max_drawdown_pct']:.2f}%")
    print(f"Win Rate:        {result.stats['win_rate']:.1%}")
    print(f"Num Trades:      {result.stats['n_trades']}")
    print(f"Total Fees:      ${result.stats['total_fees']:.2f}")

    # Verify no lookahead
    if not result.trades_df.empty:
        lookahead_violations = result.trades_df[
            result.trades_df["fill_bar_idx"] <= result.trades_df["decision_bar_idx"]
        ]
        if len(lookahead_violations) > 0:
            print("\nWARNING: Lookahead violations detected!")
            print(lookahead_violations)
        else:
            print("\nNo lookahead violations detected (fill_bar_idx > decision_bar_idx for all trades)")

    print(f"\nOutputs saved to: {out_dir}/")
    print(f"  - trades.parquet")
    print(f"  - equity_curve.parquet")
    print(f"  - stats.json")


if __name__ == "__main__":
    main()
