#!/usr/bin/env python3
# pylint: disable=wrong-import-position
"""Kalshi Prediction Opportunity Scanner.

Uses the Price-Time Manifold with Monte Carlo to find mispriced 1-hour
prediction market opportunities on Kalshi across multi-scale regimes.
Supports continuous background running and Telegram alerts.
"""
import argparse
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
from src.gnosis.ingest.ccxt_loader import fetch_live_ohlcv  # noqa: E402
from src.gnosis.ingest.coingecko import fetch_coingecko_prints  # noqa: E402
from src.gnosis.quantum import PriceTimeManifold  # noqa: E402
from src.gnosis.utils.kalshi_client import KalshiClient  # noqa: E402
import src.gnosis.utils.notifications as notify  # noqa: E402

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
        lines.append(f"Price: ${opp['current_p']:,.2f}")
        lines.append(f"Forecast: ${opp['forecast_p']:,.2f}")

        # Binary Option Logic
        lines.append("-" * 20)
        lines.append(f"Strike: *Above ${opp['strike']:,.2f}*")
        lines.append(f"Market Prob: {opp['market_prob']:.1%}")
        lines.append(f"Model Prob: {opp['model_prob']:.1%}")
        edge = opp['model_prob'] - opp['market_prob']
        lines.append(f"EDGE: {edge:+.1%}")
        action = ("BUY YES" if edge > 0.05 else "BUY NO"
                  if edge < -0.05 else "NEUTRAL")
        lines.append(f"ACTION: `{action}`")
        lines.append("")

    lines.append("=" * 40)
    return "\n".join(lines)


def run_scan(symbol, data_path=None, edge_threshold=0.10, live_ohlcv=None):
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
    opportunities = []

    # 2. Iterate through Kalshi strikes and find edges
    for market in target_markets:
        # Calculate Market Probability
        # If yes_bid is 0, use 0. If yes_ask is missing, use 100
        y_bid = market.get('yes_bid', 0)
        y_ask = market.get('yes_ask', 100)

        # Midpoint of the spread for the market probability
        market_prob = ((y_bid + y_ask) / 2) / 100

        # Extract strike price: check 'floor_strike', 'strike_price',
        # or parse from ticker if needed.
        strike = market.get('floor_strike')
        if strike is None:
            strike = market.get('strike_price')

        if strike is None:
            # Last resort: parse from ticker (e.g. KXBTC-...-T85000)
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
        medians = []

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
                        help="CCXT exchange ID (default: kraken)")
    parser.add_argument("--bridge", action="store_true",
                        help="Send opportunities to Trading Core API")
    args = parser.parse_args()

    data_path = "data/large_history/prints.parquet"
    last_alert_time = 0
    cooldown = 3600  # 1 hour cooldown per alert

    print(f"Starting Kalshi Scanner for {args.symbol}...")
    if args.loop:
        print(f"Continuous mode active. Interval: {args.interval}s")

    while True:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Scanning...")

        live_data = None
        if args.live:
            # Multi-source ingestion attempt
            sources = [args.exchange, 'coinbase', 'kraken']
            for src in sources:
                try:
                    print(f"Attempting live ingestion from {src}...")
                    live_data = fetch_live_ohlcv(
                        [args.symbol],
                        exchange=src,
                        timeframe='1m',
                        days=2
                    )
                    if not live_data.empty:
                        print(f"Successfully ingested data from {src}.")
                        break
                except Exception as e:  # pylint: disable=broad-except
                    print(f"Ingestion failed for {src}: {e}")

            # Fallback to CoinGecko if CCXT fails
            if live_data is None or live_data.empty:
                print("Falling back to CoinGecko...")
                try:
                    cg_data = fetch_coingecko_prints([args.symbol], days=2)
                    if not cg_data.empty:
                        live_data = prints_to_ohlcv(cg_data, bar_minutes=1)
                        print("Successfully ingested data from CoinGecko.")
                except Exception as e:  # pylint: disable=broad-except
                    print(f"CoinGecko fallback failed: {e}")

        if live_data is None:
            print("No live data available. Using local cache.")

        opps = run_scan(
            args.symbol,
            data_path if not args.live or live_data is None else None,
            args.threshold,
            live_ohlcv=live_data
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
                            "action": action,
                            "strike": float(opp['strike']),
                            "market_prob": float(opp['market_prob']),
                            "model_prob": float(opp['model_prob']),
                            "edge": float(edge)
                        }
                        try:
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

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
