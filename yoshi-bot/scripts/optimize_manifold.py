#!/usr/bin/env python3
"""Hyperparameter Optimization for Price-Time Manifold.

Uses Optuna to find the best:
1. vol_threshold: Sensitivity of Quantum Bar creation
2. sigma_supply/demand: Width of wavefunction distributions
3. price_resolution: Grid granularity
4. decay_rate: Historical influence
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
from typing import List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gnosis.quantum import (
    PriceTimeManifold,
    aggregate_to_quantum_bars,
)

def prints_to_ohlcv(prints_df: pd.DataFrame, bar_minutes: int = 1) -> pd.DataFrame:
    """Convert print-level data to OHLCV bars with order flow."""
    df = prints_df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['bar_start'] = df['timestamp'].dt.floor(f'{bar_minutes}min')
    df['buy_qty'] = np.where(df['side'] == 'BUY', df['quantity'], 0)
    df['sell_qty'] = np.where(df['side'] == 'SELL', df['quantity'], 0)
    
    ohlcv = df.groupby(['symbol', 'bar_start']).agg({
        'price': ['first', 'max', 'min', 'last'],
        'quantity': 'sum',
        'buy_qty': 'sum',
        'sell_qty': 'sum',
    }).reset_index()
    
    ohlcv.columns = [
        'symbol', 'timestamp', 'open', 'high', 'low', 'close',
        'volume', 'buy_volume', 'sell_volume'
    ]
    return ohlcv.sort_values(['symbol', 'timestamp']).reset_index(drop=True)

class ManifoldObjective:
    def __init__(self, ohlcv_1m: pd.DataFrame, horizons: List[int]):
        self.ohlcv_1m = ohlcv_1m
        self.horizons = horizons
        
    def __call__(self, trial):
        # Sample hyperparameters
        vol_threshold = trial.suggest_float("vol_threshold", 0.0001, 0.01, log=True)
        sigma_supply = trial.suggest_float("sigma_supply", 0.005, 0.1, log=True)
        sigma_demand = trial.suggest_float("sigma_demand", 0.005, 0.1, log=True)
        decay_rate = trial.suggest_float("decay_rate", 0.5, 0.99)
        price_resolution = trial.suggest_int("price_resolution", 50, 300)
        
        all_correct = []
        
        for symbol in self.ohlcv_1m['symbol'].unique():
            symbol_1m = self.ohlcv_1m[self.ohlcv_1m['symbol'] == symbol].reset_index(drop=True)
            
            # 1. Aggregate Bars
            symbol_df = aggregate_to_quantum_bars(symbol_1m, vol_threshold_perc=vol_threshold)
            if len(symbol_df) < max(self.horizons) + 15:
                continue
            
            # 2. Fit Manifold
            manifold = PriceTimeManifold(
                price_resolution=price_resolution,
                decay_rate=decay_rate,
                sigma_supply=sigma_supply,
                sigma_demand=sigma_demand,
            )
            manifold.fit_from_1m_bars(symbol_df)
            
            # 3. Predict & Score
            predictions = manifold.predict_collapse(horizons=self.horizons)
            if not predictions.empty:
                # Target: maximize direction accuracy for the longest horizon (usually most difficult)
                # or mean accuracy across horizons. Let's use mean.
                all_correct.append(predictions['direction_correct'].mean())
        
        if not all_correct:
            return 0.0
            
        return np.mean(all_correct)

def main():
    parser = argparse.ArgumentParser(description="Optimize Price-Time Manifold Hyperparameters")
    parser.add_argument("--data", type=str, default="data/large_history/prints.parquet")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--horizons", type=str, default="1,3,5,10,20")
    args = parser.parse_args()
    
    horizons = [int(h.strip()) for h in args.horizons.split(",")]
    
    print(f"Loading data from {args.data}...")
    df = pd.read_parquet(args.data)
    
    if 'open' in df.columns and 'high' in df.columns:
        ohlcv_1m = df
    else:
        ohlcv_1m = prints_to_ohlcv(df)
        
    study = optuna.create_study(direction="maximize")
    objective = ManifoldObjective(ohlcv_1m, horizons)
    
    print(f"Starting optimization with {args.trials} trials...")
    study.optimize(objective, n_trials=args.trials)
    
    print("\n" + "="*50)
    print("BEST PARAMETERS FOUND:")
    print("="*50)
    for key, value in study.best_params.items():
        print(f"{key:>20}: {value}")
    print(f"{'Best Accuracy':>20}: {study.best_value:.4%}")
    print("="*50)

if __name__ == "__main__":
    main()
