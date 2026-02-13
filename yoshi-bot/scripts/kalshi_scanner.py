#!/usr/bin/env python3
# pylint: disable=wrong-import-position
"""Kalshi Prediction Opportunity Scanner.

Uses the Price-Time Manifold with Monte Carlo to find mispriced 1-hour
prediction market opportunities on Kalshi across multi-scale regimes.
Supports continuous background running and Telegram alerts.
"""
import argparse
import asyncio
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# Add project root to path BEFORE importing local modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.evaluate_manifold import prints_to_ohlcv  # noqa: E402
from src.gnosis.quantum import PriceTimeManifold  # noqa: E402
from src.gnosis.particle.quantum import QuantumPriceEngine # For param hot-reload
from src.gnosis.utils.kalshi_client import KalshiClient  # noqa: E402
import src.gnosis.utils.notifications as notify  # noqa: E402
from src.gnosis.data.aggregator import DataSourceAggregator # Task 1
from src.gnosis.forecasting.modular_ensemble import (
    GatingInputs, GatingPolicyConfig, compute_module_weights
)

# Load environment variables
load_dotenv()
kalshi = KalshiClient()

# Set up logging
log_file = Path("scanner.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("kalshi_scanner")


def format_kalshi_report(opportunities):
    """Format the found opportunities into a clean report."""
    if not opportunities:
        return "No significant mispricings detected in current regime."

    lines = ["=" * 40,
             "YOSHI KALSHI ALERT",
             "=" * 40, ""]

    for opp in opportunities:
        lines.append(f"*{opp['symbol']}*")
        lines.append(f"Ticker: {opp.get('ticker', 'N/A')}")
        lines.append(f"Price: ${opp['current_p']:,.2f}")
        lines.append(f"Forecast: ${opp['forecast_p']:,.2f}")

        # Binary Option Logic
        lines.append("-" * 20)
        lines.append(f"Strike: *Above ${opp['strike']:,.2f}*")
        lines.append(f"Market Prob: {opp['market_prob']:.1%}")
        lines.append(f"Model Prob: {opp['model_prob']:.1%}")
        edge = opp['model_prob'] - opp['market_prob']
        lines.append(f"EDGE: {edge:+.1%}")
        confidence = opp.get('confidence', 0.5)
        lines.append(f"Confidence: {confidence:.0%}")
        action = ("BUY YES" if edge > 0.05 else "BUY NO"
                  if edge < -0.05 else "NEUTRAL")
        lines.append(f"ACTION: `{action}`")
        lines.append("")

    lines.append("=" * 40)
    return "\n".join(lines)


def run_scan(symbol, data_path=None, edge_threshold=0.10, live_ohlcv=None, ralph_params=None):
    """Perform a single scan for opportunities using live Kalshi data."""

    if live_ohlcv is not None:
        ohlcv_1m = live_ohlcv
    elif data_path:
        try:
            df = pd.read_parquet(data_path)
            symbol_df = df[df['symbol'] == symbol].copy()
            if len(symbol_df) < 100:
                print(f"Insufficient data for {symbol} in parquet.")
                return []
            ohlcv_1m = prints_to_ohlcv(symbol_df, bar_minutes=1)
        except (FileNotFoundError, IOError) as e:
            print(f"Error loading data: {e}")
            return []
    else:
        print("No data source provided to run_scan.")
        return []

    if ohlcv_1m.empty:
        return []

    current_p = ohlcv_1m['close'].iloc[-1]

    # 1. Check Exchange Status
    status = kalshi.get_exchange_status()
    if not status or not status.get('exchange_active'):
        print("Kalshi exchange is currently inactive.")
        return []

    # Map Symbol to Kalshi Series
    # Hourly series: KXBTC, KXETH
    series_map = {
        "BTCUSDT": "KXBTC",
        "ETHUSDT": "KXETH"
    }
    series_ticker = series_map.get(symbol, "KXBTC")

    # 2. Fetch live Kalshi markets for the series
    try:
        # Use series_ticker and status=open
        markets = kalshi.list_markets(
            limit=100,
            series_ticker=series_ticker,
            status="open"
        )

        # Relaxed filter: just need a yes_ask to know we can buy YES
        # or yes_bid to know we can buy NO (sell Yes)
        target_markets = [
            m for m in markets
            if (m.get('status') == 'active' and
                (m.get('yes_ask', 0) > 0 or m.get('yes_bid', 0) > 0))
        ]

        if not target_markets:
            print(f"No active {series_ticker} markets found.")
            return []

        print(f"Found {len(target_markets)} active {series_ticker} markets.")

    except (KeyError, RuntimeError) as e:
        print(f"Kalshi fetch error: {e}")
        return []

    timeframes = [5, 15, 30, 60]
    # Horizon-appropriate weights: longer timeframes matter more for 1-hour
    # predictions. 60m captures the full horizon directly, 5m is mostly noise.
    tf_weights = {5: 0.10, 15: 0.20, 30: 0.30, 60: 0.40}
    opportunities = []

    # 2. Iterate through Kalshi strikes and find edges
    for market in target_markets:
        # Calculate Market Probability
        # If yes_bid is 0, use 0. If yes_ask is missing, use 100
        y_bid = market.get('yes_bid', 0)
        y_ask = market.get('yes_ask', 100)

        # Midpoint of the spread for the market probability
        market_prob = ((y_bid + y_ask) / 2) / 100

        # Extract strike price
        strike = market.get('floor_strike')
        if strike is None:
            strike = market.get('strike_price')

        if strike is None:
            # Last resort: parse from ticker
            try:
                ticker = market.get('ticker', '')
                if '-T' in ticker:
                    strike = float(ticker.split('-T')[-1])
                elif '-B' in ticker:
                    strike = float(ticker.split('-B')[-1])
            except (ValueError, IndexError):
                continue

        if strike is None:
            continue

        strike = float(strike)

        # Filter for strikes near current price (within 10%)
        if abs(strike - current_p) / current_p > 0.10:
            continue

        agg_probs = []
        agg_weights = []
        medians = []
        median_weights = []

        for tf in timeframes:
            # Dynamically build aggregation dict based on available columns
            agg_dict = {
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
                'volume': 'sum'
            }
            if 'buy_volume' in ohlcv_1m.columns:
                agg_dict['buy_volume'] = 'sum'
            if 'sell_volume' in ohlcv_1m.columns:
                agg_dict['sell_volume'] = 'sum'

            tf_df = ohlcv_1m.resample(f'{tf}min', on='timestamp').agg(
                agg_dict
            ).dropna().reset_index()

            if len(tf_df) < 3:
                continue

            # Use Ralph-optimized params if available, otherwise defaults
            if ralph_params:
                manifold = PriceTimeManifold.from_params(ralph_params)
            else:
                manifold = PriceTimeManifold()
            manifold.fit_from_1m_bars(tf_df)

            h_bars = max(1, 60 // tf)
            res = manifold.predict_binary_market(strike, h_bars, n_sims=3000)

            w = tf_weights.get(tf, 0.25)
            agg_probs.append(res['prob'])
            agg_weights.append(w)
            medians.append(res['median'])
            median_weights.append(w)

        if not agg_probs:
            continue

        # Weighted average across timeframes
        total_w = sum(agg_weights)
        final_prob = sum(p * w for p, w in zip(agg_probs, agg_weights)) / total_w
        final_median = sum(m * w for m, w in zip(medians, median_weights)) / total_w
        edge = final_prob - market_prob

        # Compute ensemble confidence to scale the edge threshold.
        # Higher confidence = lower bar needed, lower confidence = stricter.
        try:
            gating_inputs = GatingInputs(
                regime_probs={},  # Populated by regime detector if available
                spread_bps=float(y_ask - y_bid) if y_ask > y_bid else 5.0,
                depth_norm=0.5,  # Default medium liquidity
                lfi=0.0,
                jump_probability=0.0,
            )
            _, confidence = compute_module_weights(gating_inputs)
        except Exception:
            confidence = 0.5

        # Scale edge threshold: confident predictions need less edge,
        # uncertain predictions need more.
        adjusted_threshold = edge_threshold * (1.5 - confidence)
        adjusted_threshold = max(adjusted_threshold, 0.05)  # Floor at 5%

        if abs(edge) >= adjusted_threshold:
            opportunities.append({
                'symbol': symbol,
                'current_p': current_p,
                'forecast_p': final_median,
                'market_prob': market_prob,
                'model_prob': final_prob,
                'strike': strike,
                'ticker': market['ticker'],
                'confidence': confidence,
                'adjusted_threshold': adjusted_threshold,
            })

    return opportunities


async def data_loop(aggregator: DataSourceAggregator, symbols: list[str], interval: int = 30):
    """Background task to fetch market data independently."""
    while True:
        try:
            # logger.info("Background data fetch triggered...")
            await aggregator.fetch_cycle(symbols)
        except Exception as e:
            logger.error(f"Data fetch failed: {e}")
        await asyncio.sleep(interval)


async def main():
    """Main entry point for the Kalshi Scanner."""
    parser = argparse.ArgumentParser(description="Kalshi Bot Scanner")
    parser.add_argument("--symbol", type=str, default="BTCUSDT",
                        help="Symbol to scan (e.g. BTCUSDT, ETHUSDT)")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=300,
                        help="Interval between scans in seconds")
    parser.add_argument("--threshold", type=float, default=0.10,
                        help="Edge threshold for alerts (e.g. 0.10 = 10%)")
    parser.add_argument("--live", action="store_true", default=True,
                        help="Fetch live data from APIs (default: True)")
    parser.add_argument("--exchange", type=str, default="kraken",
                        help="Deprecated: Exchange ID (now handled by aggregator)")
    parser.add_argument("--bridge", action="store_true",
                        help="Send opportunities to Trading Core API")
    args = parser.parse_args()

    data_path = "data/large_history/prints.parquet"
    last_alert_time = 0
    cooldown = 3600  # 1 hour cooldown per alert

    # Task 1: Initialize Aggregator
    aggregator = DataSourceAggregator()
    
    # Task 4: Initialize Quantum Engine for Param Hot-Reload
    quantum_engine = QuantumPriceEngine()

    print(f"Starting Kalshi Scanner for {args.symbol}...")
    if args.loop:
        print(f"Continuous mode active. Scan Interval: {args.interval}s")

    # Start independent data loop
    if args.live:
        asyncio.create_task(data_loop(aggregator, [args.symbol]))
        print("Background data aggregator started.")

    while True:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Scanning...")
        
        # Task 4: Check for Ralph updates. Capture returned params for
        # other engines (PriceTimeManifold, etc.) that need them.
        ralph_params = quantum_engine.maybe_reload_params()

        live_data = None
        if args.live:
            # Task 1: Use aggregator cache instead of direct fetch
            snapshot = aggregator.get_latest()
            if snapshot and args.symbol in snapshot.get("symbols", {}):
                s_data = snapshot["symbols"][args.symbol]
                
                # Convert list of dicts to DataFrame
                # s_data["ohlcv"] is list of dicts [timestamp, open, high, low, close, volume]
                try:
                    df = pd.DataFrame(s_data["ohlcv"])
                    if not df.empty:
                        # Ensure timestamp is datetime
                        if pd.api.types.is_numeric_dtype(df["timestamp"]):
                            df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s')
                        
                        # Ensure columns are present and correct type
                        cols = ["open", "high", "low", "close", "volume"]
                        for c in cols:
                            df[c] = df[c].astype(float)
                        
                        live_data = df
                        # print(f"Loaded {len(live_data)} bars from cache ({snapshot['source']})")
                except Exception as e:
                    logger.error(f"Error converting cache to DataFrame: {e}")

            else:
                if snapshot:
                    logger.warning(f"Symbol {args.symbol} not in cache.")
                else:
                    logger.warning("Cache empty or stale.")
        
        if live_data is None:
            if args.live:
                print("No live data available yet (waiting for aggregator). Using local fallback if allowed.")
            else:
                print("Live mode disabled. Using local parquet.")

        # Run Scan with Ralph's optimized parameters
        opps = run_scan(
            args.symbol,
            data_path if not args.live or live_data is None else None,
            args.threshold,
            live_ohlcv=live_data,
            ralph_params=ralph_params if ralph_params else None,
        )
        
        if opps:
            logger.info("Found %d opportunities!", len(opps))
            report = format_kalshi_report(opps)

            # Bridge to Trading Core
            if args.bridge:
                for opp in opps:
                    edge = opp['model_prob'] - opp['market_prob']
                    action = ("BUY_YES" if edge > 0.05 else "BUY_NO"
                              if edge < -0.05 else "NEUTRAL")

                    if action != "NEUTRAL":
                        payload = {
                            "symbol": opp['symbol'],
                            "ticker": opp['ticker'],
                            "action": action,
                            "strike": float(opp['strike']),
                            "market_prob": float(opp['market_prob']),
                            "model_prob": float(opp['model_prob']),
                            "edge": float(edge)
                        }
                        try:
                            # Note: requests is sync, blocking the loop briefly. 
                            # Ideally use aiohttp, but keeping legacy bridge logic for now 
                            # as user instructions focused on data layer.
                            resp = requests.post(
                                "http://127.0.0.1:8000/propose",
                                json=payload,
                                timeout=5
                            )
                            if resp.status_code == 200:
                                logger.info(
                                    "Bridged %s for %s to Trading Core",
                                    action, opp['symbol']
                                )
                            else:
                                logger.error(
                                    "Bridge failed: %d",
                                    resp.status_code
                                )
                        except Exception as e:  # pylint: disable=broad-except
                            logger.error("Bridge connection error: %s", e)

            # Check cooldown
            if time.time() - last_alert_time > cooldown:
                logger.info("Sending Telegram alert...")
                notify.send_telegram_alert_sync(report)
                last_alert_time = time.time()
            else:
                logger.info("Alert found but currently in cooldown.")
            print(report)
        else:
            logger.info("No significant edge found.")

        if not args.loop:
            break

        await asyncio.sleep(args.interval)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
