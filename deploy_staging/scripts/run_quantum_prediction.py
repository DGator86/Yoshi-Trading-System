#!/usr/bin/env python3
"""Run quantum-inspired price predictions with confidence intervals.

This treats price as a particle moving through steering fields:
- Funding rate pressure
- Order book gravity
- VWAP mean reversion
- Liquidation cascades
- Multi-scale momentum

Outputs probabilistic forecasts with confidence intervals.

Usage:
    python scripts/run_quantum_prediction.py --data data/parquet/prints.parquet
    python scripts/run_quantum_prediction.py --horizon 60 --simulations 50000
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gnosis.domains import DomainAggregator, compute_features
from gnosis.particle import QuantumPriceEngine, compute_quantum_features


def main():
    parser = argparse.ArgumentParser(description="Run quantum price prediction")
    parser.add_argument(
        "--data", type=str, default="data/parquet/prints.parquet",
        help="Path to prints parquet file"
    )
    parser.add_argument(
        "--horizon", type=int, default=60,
        help="Prediction horizon in minutes (default: 60)"
    )
    parser.add_argument(
        "--simulations", type=int, default=10000,
        help="Number of Monte Carlo simulations (default: 10000)"
    )
    parser.add_argument(
        "--n-trades", type=int, default=200,
        help="Trades per bar for aggregation (default: 200)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("YOSHI QUANTUM PRICE PREDICTION ENGINE")
    print("=" * 70)
    print()

    # Load data
    print(f"Loading data from {args.data}...")
    prints_df = pd.read_parquet(args.data)
    print(f"  {len(prints_df):,} prints loaded")

    # Aggregate to bars
    print(f"\nAggregating to {args.n_trades}-trade bars...")
    domain_cfg = {"domains": {"D0": {"n_trades": args.n_trades}}}
    aggregator = DomainAggregator(domain_cfg)
    bars_df = aggregator.aggregate(prints_df, "D0")
    print(f"  {len(bars_df)} bars created")

    # Compute basic features
    print("\nComputing features...")
    features_df = compute_features(bars_df, extended=True)

    # Compute quantum features
    features_df = compute_quantum_features(features_df)
    print(f"  {len(features_df.columns)} features computed")

    # Calculate order book proxy from buy/sell volume
    if "buy_volume" in features_df.columns and "sell_volume" in features_df.columns:
        bid_volume = features_df["buy_volume"].tail(50).sum()
        ask_volume = features_df["sell_volume"].tail(50).sum()
    else:
        bid_volume = 1.0
        ask_volume = 1.0

    # Calculate VWAP
    if "volume" in features_df.columns:
        vwap = (
            (features_df["close"] * features_df["volume"]).tail(50).sum() /
            features_df["volume"].tail(50).sum()
        )
    else:
        vwap = features_df["close"].tail(50).mean()

    # Initialize quantum engine
    print(f"\nInitializing quantum engine with {args.simulations:,} simulations...")
    engine = QuantumPriceEngine(
        n_simulations=args.simulations,
        random_seed=args.seed,
    )

    # Generate support/resistance levels from price distribution
    recent_prices = features_df["close"].tail(100)
    support_levels = [
        (recent_prices.quantile(0.1), 100),
        (recent_prices.quantile(0.25), 50),
        (recent_prices.min(), 200),
    ]
    resistance_levels = [
        (recent_prices.quantile(0.9), 100),
        (recent_prices.quantile(0.75), 50),
        (recent_prices.max(), 200),
    ]

    # Run prediction
    print(f"\nRunning prediction for {args.horizon} minute horizon...")
    result = engine.predict(
        df=features_df,
        horizon_minutes=args.horizon,
        funding_rate=0.0001,  # Placeholder - would come from exchange API
        bid_volume=bid_volume,
        ask_volume=ask_volume,
        vwap=vwap,
        support_levels=support_levels,
        resistance_levels=resistance_levels,
    )

    # Generate and print report
    report = engine.generate_report(result, args.horizon)
    print()
    print(report)

    # Additional analysis
    print("\nPREDICTION SUMMARY:")
    print("-" * 40)
    print(f"  Current:  ${result.current_price:,.2f}")
    print(f"  Predict:  ${result.point_estimate:,.2f}")
    print(f"  Change:   {result.expected_return_pct:+.2f}%")
    print()
    print(f"  90% CI:   ${result.confidence_intervals['90%'][0]:,.2f} - "
          f"${result.confidence_intervals['90%'][1]:,.2f}")
    print()

    # Accuracy evaluation against historical data
    if len(features_df) > args.horizon + 100:
        print("\nBACKTEST VALIDATION (last 100 predictions):")
        print("-" * 40)

        correct_direction = 0
        in_90_ci = 0
        in_68_ci = 0
        n_tests = 100

        for i in range(n_tests):
            # Use historical point as "current"
            idx = len(features_df) - args.horizon - n_tests + i
            if idx < 100:
                continue

            test_df = features_df.iloc[:idx].copy()
            actual_future = features_df["close"].iloc[idx + args.horizon // 10]  # Approximate

            # Quick prediction (fewer simulations for backtest)
            test_engine = QuantumPriceEngine(n_simulations=1000, random_seed=42 + i)
            test_result = test_engine.predict(
                df=test_df,
                horizon_minutes=args.horizon,
            )

            # Check direction
            predicted_direction = np.sign(test_result.point_estimate - test_result.current_price)
            actual_direction = np.sign(actual_future - test_result.current_price)
            if predicted_direction == actual_direction:
                correct_direction += 1

            # Check confidence intervals
            ci_90 = test_result.confidence_intervals["90%"]
            ci_68 = test_result.confidence_intervals["68%"]

            if ci_90[0] <= actual_future <= ci_90[1]:
                in_90_ci += 1
            if ci_68[0] <= actual_future <= ci_68[1]:
                in_68_ci += 1

        print(f"  Directional Accuracy:  {correct_direction/n_tests:.1%}")
        print(f"  68% CI Coverage:       {in_68_ci/n_tests:.1%} (target: 68%)")
        print(f"  90% CI Coverage:       {in_90_ci/n_tests:.1%} (target: 90%)")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
