#!/usr/bin/env python3
# pylint: disable=wrong-import-position
"""Kalshi Prediction Opportunity Scanner.

Uses the Price-Time Manifold with Monte Carlo to find mispriced 1-hour
prediction market opportunities on Kalshi across multi-scale regimes.
Supports continuous background running and Telegram alerts.
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path BEFORE importing local modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from src.gnosis.quantum import PriceTimeManifold  # noqa: E402
import src.gnosis.utils.notifications as notify  # noqa: E402
from src.gnosis.utils.kalshi_client import KalshiClient  # noqa: E402
from scripts.evaluate_manifold import prints_to_ohlcv  # noqa: E402

# Load environment variables
load_dotenv()
kalshi = KalshiClient()


def format_kalshi_report(opportunities):
    """Format the found opportunities into a clean report."""
    if not opportunities:
        return "No significant mispricings detected in current regime."

    lines = ["=" * 40,
             "🚀 YOSHI KALSHI ALERT",
             "=" * 40, ""]

    for opp in opportunities:
        lines.append(f"*{opp['symbol']}*")
        lines.append(f"Price: ${opp['current_p']:,.2f}")
        lines.append(f"Forecast: ${opp['forecast_p']:,.2f}")

        # Binary Option Logic
        lines.append("-" * 20)
        lines.append(f"Strike: *Above ${opp['strike']:,.2f}*")
        lines.append(f"Market Prob: {opp['market_prob']:.1%}")
        lines.append(f"Model Prob: {opp['model_prob']:.1%}")
        edge = opp['model_prob'] - opp['market_prob']
        lines.append(f"✅ *EDGE: {edge:+.1%}*")
        action = ("BUY YES" if edge > 0.05 else "BUY NO"
                  if edge < -0.05 else "NEUTRAL")
        lines.append(f"ACTION: `{action}`")
        lines.append("")

    lines.append("=" * 40)
    return "\n".join(lines)


def run_scan(symbol, data_path, edge_threshold=0.10):
    """Perform a single scan for opportunities using live Kalshi data."""

    try:
        df = pd.read_parquet(data_path)
    except (FileNotFoundError, IOError) as e:
        print(f"Error loading data: {e}")
        return []

    symbol_df = df[df['symbol'] == symbol].copy()
    if len(symbol_df) < 100:
        return []

    ohlcv_1m = prints_to_ohlcv(symbol_df, bar_minutes=1)
    current_p = ohlcv_1m['close'].iloc[-1]

    # 1. Fetch live Kalshi Bitcoin markets (Event: BTC - Bitcoin Hourly)
    # Note: Ticker depends on current date/time.
    # For now, we search for the 'BTC' event group.
    try:
        # BTC is the general Bitcoin event group
        event_data = kalshi.get_event_markets("BTC")
        if not event_data or 'markets' not in event_data:
            print("No active Kalshi Hourly Bitcoin markets found.")
            return []
        target_markets = event_data['markets']
    except (KeyError, RuntimeError) as e:
        print(f"Kalshi fetch error: {e}")
        return []

    timeframes = [5, 15, 30, 60]
    opportunities = []

    # 2. Iterate through Kalshi strikes and find edges
    for market in target_markets:
        # Example Kalshi 'Yes' price is in cents (0-100)
        market_prob = ((market['yes_bid'] + market['yes_ask']) / 2) / 100
        strike = float(market.get('strike_price', 0))
        if strike == 0:
            continue

        # Filter for strikes near current price
        if abs(strike - current_p) / current_p > 0.05:
            continue

        agg_probs = []
        medians = []

        for tf in timeframes:
            tf_df = ohlcv_1m.resample(f'{tf}min', on='timestamp').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
                'volume': 'sum', 'buy_volume': 'sum', 'sell_volume': 'sum'
            }).dropna().reset_index()

            manifold = PriceTimeManifold()
            manifold.fit_from_1m_bars(tf_df)

            h_bars = max(1, 60 // tf)
            res = manifold.predict_binary_market(strike, h_bars, n_sims=2000)
            agg_probs.append(res['prob'])
            medians.append(res['median'])

        final_prob = sum(agg_probs) / len(agg_probs)
        final_median = sum(medians) / len(medians)
        edge = final_prob - market_prob

        if abs(edge) >= edge_threshold:
            opportunities.append({
                'symbol': symbol,
                'current_p': current_p,
                'forecast_p': final_median,
                'market_prob': market_prob,
                'model_prob': final_prob,
                'strike': strike,
                'ticker': market['ticker']
            })

    return opportunities


def main():
    """Main entry point for the Kalshi Scanner."""
    parser = argparse.ArgumentParser(description="Kalshi Bot Scanner")
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=300,
                        help="Interval between scans in seconds")
    parser.add_argument("--threshold", type=float, default=0.10,
                        help="Edge threshold for alerts (e.g. 0.10 = 10%)")
    args = parser.parse_args()

    data_path = "data/large_history/prints.parquet"
    last_alert_time = 0
    cooldown = 3600  # 1 hour cooldown per alert

    print(f"Starting Kalshi Scanner for {args.symbol}...")
    if args.loop:
        print(f"Continuous mode active. Interval: {args.interval}s")

    while True:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Scanning...")
        opps = run_scan(args.symbol, data_path, args.threshold)
        if opps:
            print(f"Found {len(opps)} opportunities!")
            report = format_kalshi_report(opps)
            # Check cooldown
            if time.time() - last_alert_time > cooldown:
                print("Sending Telegram alert...")
                notify.send_telegram_alert_sync(report)
                last_alert_time = time.time()
            else:
                print("Alert found but currently in cooldown.")
            print(report)
        else:
            print("No significant edge found.")

        if not args.loop:
            break

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
