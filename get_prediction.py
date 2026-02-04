import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, time as dtime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.getcwd())

from src.gnosis.quantum import PriceTimeManifold
from scripts.evaluate_manifold import prints_to_ohlcv

def get_10am_prediction():
    data_path = "data/large_history/prints.parquet"
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    # 1. Load data
    df = pd.read_parquet(data_path)
    symbol = "BTCUSDT"
    symbol_df = df[df['symbol'] == symbol].copy()
    
    if symbol_df.empty:
        print(f"Error: No data for {symbol}")
        return

    # 2. Process to 1m bars
    ohlcv_1m = prints_to_ohlcv(symbol_df, bar_minutes=1)
    current_p = ohlcv_1m['close'].iloc[-1]
    last_tick_time = ohlcv_1m['timestamp'].iloc[-1]
    
    # 3. Calculate horizon to 10:00 AM
    # Use 10:00 AM today
    now = datetime.now()
    target_time = datetime.combine(now.date(), dtime(10, 0, 0))
    
    # If it's already past 10am today, target 11am or tomorrow? 
    # The user asked specifically for 10am. 
    # If it's currently 09:40, 10:00 is correct.
    
    delta_minutes = int((target_time - last_tick_time).total_seconds() / 60)
    
    if delta_minutes <= 0:
        # If the data is close but slightly past, or if we are exactly at 10am
        delta_minutes = 1 

    print(f"Current Price: ${current_p:,.2f}")
    print(f"Last Data Timestamp: {last_tick_time}")
    print(f"Target Time: {target_time}")
    print(f"Horizon: {delta_minutes} minute(s)")

    # 4. Predict
    results = []
    timeframes = [5, 15, 30, 60]
    
    for tf in timeframes:
        tf_df = ohlcv_1m.resample(f'{tf}min', on='timestamp').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
            'volume': 'sum', 'buy_volume': 'sum', 'sell_volume': 'sum'
        }).dropna().reset_index()

        manifold = PriceTimeManifold()
        manifold.fit_from_1m_bars(tf_df)

        # Convert 1m delta to tf bars
        h_bars = max(1, delta_minutes // tf)
        
        # We'll use predict_probabilistic directly for more detail
        res = manifold.predict_probabilistic(len(manifold._states) - 1, [h_bars], n_sims=5000)
        sim_data = res[h_bars]
        results.append(sim_data)

    # 5. Aggregate Results
    avg_median = sum(r['median'] for r in results) / len(results)
    avg_std = sum(r['std'] for r in results) / len(results)
    avg_upper = sum(r['upper_90'] for r in results) / len(results)
    avg_lower = sum(r['lower_90'] for r in results) / len(results)
    
    # Confidence as a percentage of the price (relative tightness)
    # Or simply report the 90% confidence interval
    
    print("\n--- ENGINE PREDICTION FOR 10:00 AM ---")
    print(f"Predicted Median Price: ${avg_median:,.2f}")
    print(f"90% Confidence Interval: [${avg_lower:,.2f}, ${avg_upper:,.2f}]")
    print(f"Expected Volatility (Std Dev): ${avg_std:,.2f}")
    
    direction = "UP" if avg_median > current_p else "DOWN"
    move_pct = (avg_median - current_p) / current_p * 100
    print(f"Predicted Move: {direction} ({move_pct:.2f}%)")

if __name__ == "__main__":
    get_10am_prediction()
