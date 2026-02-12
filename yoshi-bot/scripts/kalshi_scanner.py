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
from dotenv import load_dotenv

# Add project root to path BEFORE importing local modules
sys.path.insert(0, str(Path(__file__).parent.parent))
# Add monorepo root for shared runtime schema imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.evaluate_manifold import prints_to_ohlcv  # noqa: E402
from src.gnosis.quantum import PriceTimeManifold  # noqa: E402
from src.gnosis.particle.quantum import QuantumPriceEngine
from src.gnosis.utils.kalshi_client import KalshiClient  # noqa: E402
import src.gnosis.utils.notifications as notify  # noqa: E402
from src.gnosis.ingest.data_aggregator import DataSourceAggregator
from src.gnosis.execution.signal_learning import KalshiSignalLearner
from shared.trading_signals import (  # noqa: E402
    SIGNAL_EVENTS_PATH_DEFAULT,
    append_event_jsonl,
    make_trade_signal,
    wrap_signal_event,
)

# Load environment variables
load_dotenv()
kalshi = KalshiClient()

# Set up logging
log_file = Path("logs/kalshi-scanner-runtime.log")
log_file.parent.mkdir(parents=True, exist_ok=True)
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
        action = opp.get("action", "NEUTRAL").replace("_", " ")
        ev = opp.get("ev_cents")
        if ev is not None:
            lines.append(f"EV: {float(ev):+.1f}c")
        lines.append(f"ACTION: `{action}`")
        lines.append("")

    lines.append("=" * 40)
    return "\n".join(lines)


def _safe_prob(x: float) -> float:
    return max(0.001, min(0.999, float(x)))


def _calc_side_ev_cents(prob_win: float, cost_cents: int) -> float:
    prob = _safe_prob(prob_win)
    cost = max(1, min(99, int(cost_cents)))
    profit = 100 - cost
    return (prob * profit) - ((1.0 - prob) * cost)


def run_scan(
    symbol,
    data_path=None,
    edge_threshold=0.10,
    live_ohlcv=None,
    min_ev_cents: float = 5.0,
    min_volume: float = 25.0,
    max_spread_cents: int = 15,
):
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
    manifold_by_tf: list[tuple[int, PriceTimeManifold]] = []

    # Build one manifold per timeframe and reuse across all strike checks.
    for tf in timeframes:
        agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        if "buy_volume" in ohlcv_1m.columns:
            agg_dict["buy_volume"] = "sum"
        if "sell_volume" in ohlcv_1m.columns:
            agg_dict["sell_volume"] = "sum"

        tf_df = ohlcv_1m.resample(f"{tf}min", on="timestamp").agg(agg_dict).dropna().reset_index()
        if tf_df.empty:
            continue
        manifold = PriceTimeManifold()
        manifold.fit_from_1m_bars(tf_df)
        manifold_by_tf.append((tf, manifold))

    if not manifold_by_tf:
        logger.warning("No valid timeframe manifolds for %s", symbol)
        return []

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
        medians = []

        for tf, manifold in manifold_by_tf:
            h_bars = max(1, 60 // tf)
            res = manifold.predict_binary_market(strike, h_bars, n_sims=2000)
            agg_probs.append(res["prob"])
            medians.append(res["median"])

        final_prob = sum(agg_probs) / len(agg_probs)
        final_median = sum(medians) / len(medians)
        edge = final_prob - market_prob

        action = None
        side_prob = 0.0
        side_market_prob = 0.0
        side_cost = 0
        side_bid = 0

        no_bid = int(market.get("no_bid", 0) or 0)
        no_ask = int(market.get("no_ask", 0) or 0)
        volume = float(market.get("volume", 0) or 0.0)

        if edge >= edge_threshold:
            action = "BUY_YES"
            side_prob = final_prob
            side_market_prob = market_prob
            side_cost = int(y_ask or round(market_prob * 100))
            side_bid = int(y_bid or 0)
        elif edge <= -edge_threshold:
            action = "BUY_NO"
            side_prob = 1.0 - final_prob
            side_market_prob = 1.0 - market_prob
            side_cost = int(no_ask or round((1.0 - market_prob) * 100))
            side_bid = int(no_bid or 0)

        if not action:
            continue

        side_cost = max(1, min(99, side_cost))
        ev_cents = _calc_side_ev_cents(prob_win=side_prob, cost_cents=side_cost)
        spread_cents = max(0, side_cost - side_bid) if side_bid > 0 else None
        side_edge_pct = (side_prob - side_market_prob) * 100.0

        # Value-only gating: positive expectancy and basic liquidity quality.
        if ev_cents < float(min_ev_cents):
            continue
        if volume < float(min_volume):
            continue
        if spread_cents is not None and spread_cents > int(max_spread_cents):
            continue

        opportunities.append({
            'symbol': symbol,
            'current_p': current_p,
            'forecast_p': final_median,
            'market_prob': market_prob,
            'model_prob': final_prob,
            'strike': strike,
            'ticker': market['ticker'],
            'close_time': market.get("close_time") or market.get("expiration_time") or "",
            'action': action,
            'ev_cents': round(ev_cents, 2),
            'side_cost_cents': side_cost,
            'side_edge_pct': round(side_edge_pct, 2),
            'volume': volume,
            'spread_cents': spread_cents,
            'yes_bid': int(y_bid or 0),
            'yes_ask': int(y_ask or 0),
            'no_bid': no_bid,
            'no_ask': no_ask,
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


def _snapshot_symbol_to_df(snapshot: dict | None, symbol: str) -> pd.DataFrame | None:
    """Convert cached aggregator symbol snapshot to OHLCV DataFrame."""
    if not snapshot or symbol not in snapshot.get("symbols", {}):
        return None
    s_data = snapshot["symbols"][symbol]
    try:
        df = pd.DataFrame(s_data["ohlcv"])
        if df.empty:
            return None
        if pd.api.types.is_numeric_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        cols = ["open", "high", "low", "close", "volume"]
        for c in cols:
            df[c] = df[c].astype(float)
        return df
    except Exception as e:  # pylint: disable=broad-except
        logger.error("Error converting cache to DataFrame: %s", e)
        return None


def emit_structured_signals(
    opportunities: list[dict],
    signal_path: str,
    *,
    learner: KalshiSignalLearner | None = None,
    fallback_edge: float = 0.10,
) -> int:
    """Emit scanner opportunities to JSONL events for yoshi-bridge."""
    emitted = 0
    for opp in opportunities:
        edge = float(opp["model_prob"] - opp["market_prob"])
        default_action = str(opp.get("action", "")).strip().upper()
        if default_action not in {"BUY_YES", "BUY_NO"}:
            default_action = (
                "BUY_YES"
                if edge >= fallback_edge
                else "BUY_NO"
                if edge <= -fallback_edge
                else "NEUTRAL"
            )
        action = (
            learner.classify_edge(edge=edge, fallback_edge=fallback_edge)
            if learner is not None
            else default_action
        )
        if action == "NEUTRAL":
            continue
        signal = make_trade_signal(
            symbol=opp["symbol"],
            ticker=opp["ticker"],
            action=action,
            strike=float(opp["strike"]),
            market_prob=float(opp["market_prob"]),
            model_prob=float(opp["model_prob"]),
            edge=edge,
            source="kalshi_scanner",
        )
        if learner is not None:
            accepted = learner.record_signal(
                {
                    "signal_id": signal.signal_id,
                    "ticker": signal.ticker,
                    "symbol": signal.symbol,
                    "action": signal.action,
                    "edge": signal.edge,
                    "market_prob": signal.market_prob,
                    "model_prob": signal.model_prob,
                    "strike": signal.strike,
                    "source": signal.source,
                    "created_at": signal.created_at,
                    "close_time": str(opp.get("close_time", "")),
                }
            )
            if not accepted:
                logger.info(
                    "signal_suppressed_duplicate ticker=%s action=%s edge=%.4f",
                    signal.ticker,
                    signal.action,
                    signal.edge,
                )
                continue
        event = wrap_signal_event(signal)
        append_event_jsonl(signal_path, event)
        emitted += 1
        logger.info(
            "signal_emitted signal_id=%s idempotency_key=%s symbol=%s action=%s edge=%.4f ev_cents=%s volume=%s",
            signal.signal_id,
            signal.idempotency_key,
            signal.symbol,
            signal.action,
            signal.edge,
            opp.get("ev_cents"),
            opp.get("volume"),
        )
    return emitted


async def main():
    """Main entry point for the Kalshi Scanner."""
    parser = argparse.ArgumentParser(description="Kalshi Bot Scanner")
    parser.add_argument("--symbol", type=str, default="BTCUSDT",
                        help="Symbol to scan (e.g. BTCUSDT, ETHUSDT)")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=60,
                        help="Interval between scans in seconds")
    parser.add_argument("--threshold", type=float, default=0.10,
                        help="Edge threshold for alerts (e.g. 0.10 = 10%)")
    parser.add_argument("--live", action="store_true", default=True,
                        help="Fetch live data from APIs (default: True)")
    parser.add_argument("--exchange", type=str, default="kraken",
                        help="Deprecated: Exchange ID (now handled by aggregator)")
    parser.add_argument("--bridge", action="store_true",
                        help="Emit structured signals for yoshi-bridge")
    parser.add_argument(
        "--signal-path",
        type=str,
        default=SIGNAL_EVENTS_PATH_DEFAULT,
        help=f"Structured signal JSONL path (default: {SIGNAL_EVENTS_PATH_DEFAULT})",
    )
    parser.add_argument(
        "--disable-learning",
        action="store_true",
        help="Disable adaptive backtesting/learning thresholds",
    )
    parser.add_argument(
        "--learning-state-path",
        type=str,
        default="data/signals/learning_state.json",
        help="Persistent pending-signal learning state path",
    )
    parser.add_argument(
        "--learning-outcomes-path",
        type=str,
        default="data/signals/signal_outcomes.jsonl",
        help="Resolved signal outcomes JSONL path",
    )
    parser.add_argument(
        "--learning-policy-path",
        type=str,
        default="data/signals/learned_policy.json",
        help="Published adaptive threshold policy path",
    )
    parser.add_argument(
        "--learning-min-samples",
        type=int,
        default=30,
        help="Minimum resolved samples before policy optimization",
    )
    parser.add_argument(
        "--learning-lookback",
        type=int,
        default=300,
        help="Resolved outcomes lookback size for threshold backtest",
    )
    parser.add_argument(
        "--learning-max-resolve-checks",
        type=int,
        default=25,
        help="Max pending contracts to poll for settlement each cycle",
    )
    parser.add_argument(
        "--no-side-buffer",
        type=float,
        default=0.03,
        help="Extra minimum edge buffer applied to BUY_NO (anti-noise)",
    )
    parser.add_argument(
        "--min-ev-cents",
        type=float,
        default=5.0,
        help="Minimum expected value in cents required to emit a trade signal",
    )
    parser.add_argument(
        "--min-market-volume",
        type=float,
        default=25.0,
        help="Minimum market volume required to emit a trade signal",
    )
    parser.add_argument(
        "--max-spread-cents",
        type=int,
        default=15,
        help="Maximum side spread (ask-bid) in cents allowed for signal emission",
    )
    args = parser.parse_args()

    data_path = "data/large_history/prints.parquet"
    last_alert_time = 0
    cooldown = 3600  # 1 hour cooldown per alert

    # Task 1: Initialize Aggregator
    aggregator = DataSourceAggregator()
    
    # Task 4: Initialize Quantum Engine for Param Hot-Reload
    quantum_engine = QuantumPriceEngine()
    learner: KalshiSignalLearner | None = None
    if not args.disable_learning:
        learner = KalshiSignalLearner(
            state_path=args.learning_state_path,
            outcomes_path=args.learning_outcomes_path,
            policy_path=args.learning_policy_path,
            min_samples=args.learning_min_samples,
            lookback=args.learning_lookback,
            base_yes_edge=max(0.01, float(args.threshold)),
            base_no_edge=max(0.01, float(args.threshold) + max(0.0, float(args.no_side_buffer))),
        )

    print(f"Starting Kalshi Scanner for {args.symbol}...")
    if args.loop:
        print(f"Continuous mode active. Scan Interval: {args.interval}s")

    # Start independent data loop
    if args.live:
        asyncio.create_task(data_loop(aggregator, [args.symbol]))
        print("Background data aggregator started.")
        # Prime cache immediately so first scan doesn't wait for background loop.
        try:
            await aggregator.fetch_cycle([args.symbol])
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Initial live data fetch failed: %s", e)

    while True:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Scanning...")

        if learner is not None:
            try:
                resolved = learner.resolve_pending(
                    kalshi.get_market,
                    max_checks=args.learning_max_resolve_checks,
                )
                if resolved:
                    pol = learner.policy
                    logger.info(
                        "learning_resolved count=%d n_resolved=%d yes_edge=%.3f no_edge=%.3f mode=%s",
                        resolved,
                        pol.n_resolved,
                        pol.min_edge_buy_yes,
                        pol.min_edge_buy_no,
                        pol.mode,
                    )
            except Exception as e:  # pylint: disable=broad-except
                logger.warning("Learning resolve cycle failed: %s", e)
        
        # Task 4: Check for Ralph updates
        quantum_engine.maybe_reload_params()

        live_data = None
        if args.live:
            # Task 1: Use aggregator cache instead of direct fetch
            snapshot = aggregator.get_latest()
            live_data = _snapshot_symbol_to_df(snapshot, args.symbol)
            if live_data is None:
                try:
                    # One-shot refresh attempt before giving up this cycle.
                    await aggregator.fetch_cycle([args.symbol])
                    live_data = _snapshot_symbol_to_df(aggregator.get_latest(), args.symbol)
                except Exception as e:  # pylint: disable=broad-except
                    logger.warning("Live refresh failed for %s: %s", args.symbol, e)
        
        if live_data is None:
            if args.live:
                logger.info("No live cache for %s yet; skipping this cycle.", args.symbol)
                if not args.loop:
                    break
                await asyncio.sleep(min(args.interval, 15))
                continue
            else:
                print("Live mode disabled. Using local parquet.")

        # Run Scan
        opps = run_scan(
            args.symbol,
            data_path if not args.live else None,
            args.threshold,
            live_ohlcv=live_data,
            min_ev_cents=args.min_ev_cents,
            min_volume=args.min_market_volume,
            max_spread_cents=args.max_spread_cents,
        )
        
        if opps:
            logger.info("Found %d opportunities!", len(opps))
            yes_edge = float(args.threshold)
            no_edge = float(args.threshold)
            if learner is not None:
                yes_edge, no_edge = learner.effective_thresholds(fallback_edge=float(args.threshold))
                logger.info(
                    "learning_thresholds yes=%.3f no=%.3f pending=%d mode=%s",
                    yes_edge,
                    no_edge,
                    learner.pending_count,
                    learner.policy.mode,
                )
            report = format_kalshi_report(opps)

            # Emit structured events for yoshi-bridge (single ingestion path)
            if args.bridge:
                emitted = emit_structured_signals(
                    opps,
                    args.signal_path,
                    learner=learner,
                    fallback_edge=float(args.threshold),
                )
                logger.info("bridge_emit_complete count=%d path=%s", emitted, args.signal_path)

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
