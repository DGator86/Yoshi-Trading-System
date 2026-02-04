#!/usr/bin/env python3
"""Evaluate Yoshi using Price-Time Manifold with Supply/Demand Wavefunction Collapse.

This script:
1. Loads 1m OHLCV data or converts existing prints to OHLCV
2. Builds the price-time manifold treating each price as a quantum particle
3. Models supply/demand as wavefunctions that collapse at trade points
4. Evaluates prediction accuracy with t-minus and t-plus price deltas

Usage:
    python scripts/evaluate_manifold.py --data data/large_history/prints.parquet
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.gnosis.quantum import (
    PriceTimeManifold,
    aggregate_to_quantum_bars,
    compute_wavefunction_features,
)


def prints_to_ohlcv(prints_df: pd.DataFrame, bar_minutes: int = 1) -> pd.DataFrame:
    """Convert print-level data to OHLCV bars with order flow.
    
    Args:
        prints_df: DataFrame with columns [timestamp, symbol, price, quantity, side]
        bar_minutes: Bar size in minutes
        
    Returns:
        OHLCV DataFrame with buy/sell volume
    """
    df = prints_df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create bar boundaries
    df['bar_start'] = df['timestamp'].dt.floor(f'{bar_minutes}min')
    
    # Compute buy/sell volumes using vectorized operations for speed
    df['buy_qty'] = np.where(df['side'] == 'BUY', df['quantity'], 0)
    df['sell_qty'] = np.where(df['side'] == 'SELL', df['quantity'], 0)
    
    # Aggregate to OHLCV with order flow
    ohlcv = df.groupby(['symbol', 'bar_start']).agg({
        'price': ['first', 'max', 'min', 'last'],
        'quantity': 'sum',
        'buy_qty': 'sum',
        'sell_qty': 'sum',
    }).reset_index()
    
    # Flatten column names
    ohlcv.columns = [
        'symbol', 'timestamp', 'open', 'high', 'low', 'close',
        'volume', 'buy_volume', 'sell_volume'
    ]
    
    return ohlcv.sort_values(['symbol', 'timestamp']).reset_index(drop=True)


def generate_accuracy_report(
    metrics_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    avg_bar_duration_mins: float = 1.0,
) -> str:
    """Generate formatted accuracy report with t-minus/t-plus deltas.
    
    Args:
        metrics_df: DataFrame with accuracy metrics by horizon
        predictions_df: Raw predictions DataFrame
        avg_bar_duration_mins: Average duration of a bar in minutes
        
    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("YOSHI PRICE-TIME MANIFOLD ACCURACY REPORT")
    lines.append("Supply/Demand Wavefunction Collapse Analysis")
    lines.append("=" * 80)
    lines.append("")
    
    if metrics_df.empty:
        lines.append("No predictions generated - insufficient data")
        return "\n".join(lines)
    
    # Summary statistics
    best_horizon = metrics_df.loc[metrics_df['direction_accuracy'].idxmax()]
    total_samples = metrics_df['samples'].sum()
    
    lines.append(f"Total Predictions: {total_samples}")
    lines.append(f"Best Horizon: {int(best_horizon['horizon_bars'])} bars")
    lines.append(f"Best Direction Accuracy: {best_horizon['direction_accuracy']:.1%}")
    lines.append(f"Avg Bar Duration: {avg_bar_duration_mins:.1f} minutes")
    lines.append("")
    
    # T-minus / T-plus analysis
    lines.append("-" * 80)
    lines.append("T-MINUS / T-PLUS PRICE DELTA ANALYSIS")
    lines.append("-" * 80)
    lines.append("")
    lines.append("Horizon    Time Equiv.  Direction   MAE(Price)   RMSE(Price)  MAPE     Samples")
    lines.append("-" * 80)
    
    for _, row in metrics_df.iterrows():
        horizon = int(row['horizon_bars'])
        
        # Convert bars to human-readable time
        minutes = horizon * avg_bar_duration_mins
        if minutes < 60:
            time_str = f"{int(minutes)} min"
        elif minutes < 1440:
            time_str = f"{minutes/60:.1f} hrs"
        else:
            time_str = f"{minutes/1440:.1f} days"
        
        lines.append(
            f"  t+{horizon:<5}  {time_str:>12}  {row['direction_accuracy']:>8.1%}   "
            f"${row['mae']:>10,.2f}  ${row['rmse']:>10,.2f}  {row['mape']:>6.2%}   {int(row['samples']):>6}"
        )
    
    lines.append("")
    lines.append("=" * 80)
    
    # Regime breakdown
    if 'regime' in predictions_df.columns:
        lines.append("")
        lines.append("REGIME ANALYSIS")
        lines.append("-" * 40)
        
        for regime in predictions_df['regime'].unique():
            regime_df = predictions_df[predictions_df['regime'] == regime]
            accuracy = regime_df['direction_correct'].mean()
            count = len(regime_df)
            lines.append(f"  {regime:<20}: {accuracy:.1%} accuracy ({count} samples)")
    
    lines.append("")
    lines.append("=" * 80)
    
    # Interpretation
    lines.append("")
    lines.append("INTERPRETATION:")
    
    if best_horizon['direction_accuracy'] >= 0.7:
        lines.append("  [EXCELLENT] Wavefunction collapse model shows strong predictive power")
        lines.append(f"  Price particles collapse with {best_horizon['direction_accuracy']:.0%} directional accuracy")
    elif best_horizon['direction_accuracy'] >= 0.55:
        lines.append("  [GOOD] Model shows above-random prediction capability")
        lines.append("  Supply/demand wavefunction interference patterns detected")
    else:
        lines.append("  [NEEDS IMPROVEMENT] Prediction accuracy near random")
        lines.append("  Consider tuning wavefunction parameters or adding more data")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate predictions using Price-Time Manifold"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/large_history/prints.parquet",
        help="Path to prints or OHLCV parquet file",
    )
    parser.add_argument(
        "--vol-threshold",
        type=float,
        default=0.00975,
        help="Volatility threshold for Quantum Bars (default: 0.00975)",
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default="1,3,5,10,20",
        help="Comma-separated prediction horizons in bars",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/latest/manifold_accuracy.txt",
        help="Output file for report",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Filter to single symbol (default: all symbols)",
    )
    
    args = parser.parse_args()
    
    # Parse horizons
    horizons = [int(h.strip()) for h in args.horizons.split(",")]
    
    # Load data
    print(f"Loading data from {args.data}...")
    data_path = Path(args.data)
    
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        sys.exit(1)
    
    df = pd.read_parquet(data_path)
    print(f"  Loaded {len(df)} rows")
    
    # Check if this is prints or OHLCV data
    if 'open' in df.columns and 'high' in df.columns:
        ohlcv_1m = df
    else:
        print(f"  Converting prints to 1m OHLCV bars...")
        ohlcv_1m = prints_to_ohlcv(df, bar_minutes=1)
    
    # Filter by symbol if specified
    if args.symbol:
        ohlcv_1m = ohlcv_1m[ohlcv_1m['symbol'] == args.symbol]
    
    # Process each symbol
    all_predictions = []
    all_features = []
    total_bars_processed = 0
    total_time_span_mins = 0
    
    for symbol in ohlcv_1m['symbol'].unique():
        print(f"\nProcessing {symbol}...")
        symbol_1m = ohlcv_1m[ohlcv_1m['symbol'] == symbol].reset_index(drop=True)
        
        # Assemble 1m bars into Quantum Bars
        print(f"  Assembling 1m bars into Quantum Bars (threshold={args.vol_threshold})...")
        symbol_df = aggregate_to_quantum_bars(symbol_1m, vol_threshold_perc=args.vol_threshold)
        print(f"  Created {len(symbol_df)} Quantum Bars from {len(symbol_1m)} 1m bars")
        
        if len(symbol_df) < 20:
            print("  Warning: Insufficient bars for this symbol.")
            continue
            
        total_bars_processed += len(symbol_df)
        if len(symbol_1m) > 1:
            total_time_span_mins += (symbol_1m['timestamp'].max() - symbol_1m['timestamp'].min()).total_seconds() / 60
        
        # Build price-time manifold
        manifold = PriceTimeManifold(
            price_resolution=100,
            decay_rate=0.95,
            sigma_supply=0.02,
            sigma_demand=0.02,
        )
        
        print(f"  Building manifold from {len(symbol_df)} bars...")
        manifold.fit_from_1m_bars(symbol_df)
        
        # Generate predictions
        print(f"  Generating predictions for horizons: {horizons}")
        predictions = manifold.predict_collapse(horizons=horizons)
        
        if not predictions.empty:
            predictions['symbol'] = symbol
            all_predictions.append(predictions)
            print(f"  Generated {len(predictions)} predictions")
        
        # Extract wavefunction features
        features = compute_wavefunction_features(manifold)
        if not features.empty:
            features['symbol'] = symbol
            all_features.append(features)
    
    avg_bar_duration = total_time_span_mins / total_bars_processed if total_bars_processed > 0 else 1.0
    
    if not all_predictions:
        print("\nERROR: No predictions generated. Need more data.")
        sys.exit(1)
    
    # Combine all predictions
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    features_df = pd.concat(all_features, ignore_index=True) if all_features else pd.DataFrame()
    
    print(f"\nTotal predictions: {len(predictions_df)}")
    
    # Compute accuracy metrics
    print("Computing accuracy metrics...")
    metrics_list = []
    
    for horizon in horizons:
        horizon_df = predictions_df[predictions_df['horizon'] == horizon]
        if horizon_df.empty:
            continue
        
        metrics_list.append({
            'horizon_bars': horizon,
            'samples': len(horizon_df),
            'direction_accuracy': horizon_df['direction_correct'].mean(),
            'mae': horizon_df['price_error'].mean(),
            'rmse': np.sqrt((horizon_df['price_error'] ** 2).mean()),
            'mape': horizon_df['price_error_pct'].mean(),
        })
    metrics_df = pd.DataFrame(metrics_list)
    
    # Generate report
    report = generate_accuracy_report(metrics_df, predictions_df, avg_bar_duration_mins=avg_bar_duration)
    print("\n" + report)
    
    # Save outputs
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {output_path}")
    
    # Save predictions and features
    predictions_df.to_parquet(output_path.parent / "manifold_predictions.parquet", index=False)
    if not features_df.empty:
        features_df.to_parquet(output_path.parent / "wavefunction_features.parquet", index=False)
    
    print(f"Predictions saved to: {output_path.parent / 'manifold_predictions.parquet'}")
    
    if metrics_df.empty:
        return 1
    
    best_accuracy = metrics_df['direction_accuracy'].max()
    if best_accuracy >= 0.55:
        print(f"\n[PASS] Best accuracy: {best_accuracy:.1%}")
        return 0
    else:
        print(f"\n[FAIL] Best accuracy: {best_accuracy:.1%} (below 55%)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
