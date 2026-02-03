#!/usr/bin/env python3
"""Evaluate Yoshi prediction accuracy across time horizons.

Usage:
    python scripts/evaluate_accuracy.py --predictions reports/latest/predictions.parquet

Output:
    ======================================================================
    YOSHI PREDICTION ACCURACY REPORT
    ======================================================================

    Best Accuracy: 72.3% at 0.5 days
    Accuracy Decay Rate: 8.2% per day

    Horizon    Time         Direction   Coverage90  MAE      RMSE     Samples
    ----------------------------------------------------------------------
      1 bars      0.1 days     72.3%        91.2%      0.23%    0.31%      448
      5 bars      0.5 days     65.1%        89.5%      0.45%    0.58%      440
     10 bars      1.0 days     58.7%        87.3%      0.78%    0.95%      430
     20 bars      2.0 days     52.4%        85.1%      1.12%    1.45%      410
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.gnosis.evaluation import PredictionEvaluator, evaluate_predictions


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Yoshi prediction accuracy"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions parquet file",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to bars/features data (default: use trades.parquet in same dir)",
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default="1,3,5,10,20",
        help="Comma-separated list of horizons to evaluate (default: 1,3,5,10,20)",
    )
    parser.add_argument(
        "--bars-per-day",
        type=float,
        default=10.0,
        help="Bars per day for time conversion (default: 10.0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for report (default: print to stdout)",
    )

    args = parser.parse_args()

    # Parse horizons
    horizons = [int(h.strip()) for h in args.horizons.split(",")]

    # Load predictions
    print(f"Loading predictions from {args.predictions}...")
    predictions_df = pd.read_parquet(args.predictions)
    print(f"  {len(predictions_df)} predictions loaded")

    # Load actuals (bars data with close prices)
    if args.data:
        data_path = args.data
    else:
        # Try to find trades.parquet in same directory
        pred_dir = Path(args.predictions).parent
        trades_path = pred_dir / "trades.parquet"
        if trades_path.exists():
            data_path = str(trades_path)
        else:
            # Use predictions themselves (must have close prices)
            data_path = args.predictions

    print(f"Loading actuals from {data_path}...")
    actuals_df = pd.read_parquet(data_path)

    # Check if we have what we need
    if "close" not in actuals_df.columns and "future_return" not in actuals_df.columns:
        print("ERROR: Actuals data must have 'close' or 'future_return' column")
        sys.exit(1)

    # Run evaluation
    print(f"Evaluating {len(horizons)} horizons: {horizons}")
    print()

    evaluator = PredictionEvaluator(
        horizons=horizons,
        bars_per_day=args.bars_per_day,
    )

    analysis = evaluator.evaluate_all_horizons(predictions_df, actuals_df)
    report = evaluator.generate_report(analysis)

    # Output
    print(report)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"\nReport saved to {args.output}")

    # Return exit code based on accuracy
    if analysis.horizons:
        best = max(analysis.horizons, key=lambda m: m.directional_accuracy)
        if best.directional_accuracy >= 0.55:
            print(f"\n✓ Yoshi predicts {best.directional_accuracy:.0%} accuracy at {best.horizon_time}")
            return 0
        else:
            print(f"\n✗ Accuracy below 55% - model needs improvement")
            return 1
    else:
        print("\n✗ No valid predictions to evaluate")
        return 1


if __name__ == "__main__":
    sys.exit(main())
