"""Historical bootstrap for Kalshi signal learning.

Builds resolved learning outcomes from historical crypto OHLCV data fetched
through the multi-provider API layer (Binance public, CoinGecko, CoinMarketCap,
and optional CoinAPI when configured).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import timezone
from typing import Any

import pandas as pd

from .signal_learning import KalshiSignalLearner
from ..ingest.providers.unified import UnifiedDataFetcher


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float(default)


def _sigmoid(x: float) -> float:
    z = max(-60.0, min(60.0, float(x)))
    return 1.0 / (1.0 + math.exp(-z))


def _normalize_symbol_for_series(symbol: str) -> str:
    s = str(symbol).upper().strip()
    if "/" in s:
        s = s.replace("/", "")
    if not s.endswith("USDT"):
        if s in {"BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "BNB"}:
            s = f"{s}USDT"
    return s


@dataclass
class HistoricalBootstrapConfig:
    """Configuration for historical learning bootstrap."""

    symbols: list[str] = field(default_factory=lambda: ["BTCUSDT"])
    days: int = 90
    timeframe: str = "1h"
    horizon_bars: int = 1
    strike_bps_grid: tuple[int, ...] = (-180, -120, -90, -60, -30, 0, 30, 60, 90, 120, 180)
    min_abs_edge: float = 0.08
    max_records: int = 4000
    max_records_per_symbol: int = 1800


def build_historical_outcomes(
    ohlcv: pd.DataFrame,
    *,
    symbol: str,
    cfg: HistoricalBootstrapConfig,
) -> list[dict[str, Any]]:
    """Create resolved pseudo-Kalshi outcomes from historical bars."""
    if ohlcv.empty:
        return []
    df = ohlcv.copy()
    required_cols = {"timestamp", "close"}
    if not required_cols.issubset(set(df.columns)):
        return []

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["timestamp", "close"])
    if len(df) < max(80, cfg.horizon_bars + 20):
        return []

    # Rolling volatility proxy (hourly-style); clamp to realistic bands.
    rets = (df["close"].astype(float).pct_change()).fillna(0.0)
    vol = rets.rolling(24, min_periods=8).std().fillna(rets.std() or 0.01).clip(lower=0.003, upper=0.08)
    mom_fast = rets.rolling(3, min_periods=1).mean().fillna(0.0)
    mom_slow = rets.rolling(12, min_periods=1).mean().fillna(0.0)

    normalized_symbol = _normalize_symbol_for_series(symbol)
    series = "KXBTC" if normalized_symbol.startswith("BTC") else "KXETH" if normalized_symbol.startswith("ETH") else "KXCRYPTO"
    outcomes: list[dict[str, Any]] = []

    last_i = len(df) - int(cfg.horizon_bars) - 1
    for i in range(24, max(last_i, 24)):
        cur = _safe_float(df.iloc[i]["close"], 0.0)
        fut = _safe_float(df.iloc[i + int(cfg.horizon_bars)]["close"], 0.0)
        if cur <= 0.0 or fut <= 0.0:
            continue

        v = _safe_float(vol.iloc[i], 0.01)
        m = (0.6 * _safe_float(mom_fast.iloc[i], 0.0)) + (0.4 * _safe_float(mom_slow.iloc[i], 0.0))
        # Modest forecasted drift from momentum to avoid extreme pseudo-confidence.
        pred = cur * (1.0 + max(-0.03, min(0.03, m * 5.0)))

        ts_open = df.iloc[i]["timestamp"]
        ts_close = df.iloc[i + int(cfg.horizon_bars)]["timestamp"]
        if pd.isna(ts_open) or pd.isna(ts_close):
            continue

        for bps in cfg.strike_bps_grid:
            strike = cur * (1.0 + (float(bps) / 10_000.0))

            # Market probability is a softer distance model.
            z_mkt = (cur - strike) / max(cur * max(v * 1.35, 0.004), 1e-9)
            market_yes = _sigmoid(0.9 * z_mkt)

            # Model probability uses stronger forecast displacement.
            z_model = (pred - strike) / max(cur * max(v, 0.003), 1e-9)
            model_yes = _sigmoid(1.4 * z_model)

            edge_yes = model_yes - market_yes
            if abs(edge_yes) < float(cfg.min_abs_edge):
                continue

            action = "BUY_YES" if edge_yes > 0 else "BUY_NO"
            side_prob = model_yes if action == "BUY_YES" else (1.0 - model_yes)
            side_market_prob = market_yes if action == "BUY_YES" else (1.0 - market_yes)

            side_market_prob = max(0.01, min(0.99, side_market_prob))
            cost_cents = max(1, min(99, int(round(side_market_prob * 100.0))))
            settled_yes = fut > strike
            won = (action == "BUY_YES" and settled_yes) or (action == "BUY_NO" and not settled_yes)
            pnl_cents = (100 - cost_cents) if won else -cost_cents

            ticker = (
                f"{series}-HIST-{ts_close.strftime('%Y%m%d%H')}"
                f"-{'T' if bps >= 0 else 'B'}{int(round(strike))}"
            )
            signal_id = f"hist_{normalized_symbol}_{int(ts_open.timestamp())}_{int(bps)}_{action.lower()}"
            outcomes.append(
                {
                    "signal_id": signal_id,
                    "ticker": ticker,
                    "symbol": normalized_symbol,
                    "action": action,
                    "edge": float(edge_yes),
                    "market_prob": float(market_yes),
                    "model_prob": float(model_yes),
                    "strike": float(strike),
                    "source": "historical_bootstrap_api",
                    "created_at": ts_open.to_pydatetime().replace(tzinfo=timezone.utc).isoformat(),
                    "close_time": ts_close.to_pydatetime().replace(tzinfo=timezone.utc).isoformat(),
                    "settled_yes": bool(settled_yes),
                    "won": bool(won),
                    "cost_cents": int(cost_cents),
                    "pnl_cents": float(pnl_cents),
                    "resolved_at": ts_close.to_pydatetime().replace(tzinfo=timezone.utc).isoformat(),
                }
            )

    if not outcomes:
        return []

    # Keep strongest signals first to reduce noise and disk volume.
    outcomes.sort(key=lambda r: abs(_safe_float(r.get("edge"), 0.0)), reverse=True)
    per_symbol = outcomes[: max(1, int(cfg.max_records_per_symbol))]
    return per_symbol


def bootstrap_learning_from_api(
    learner: KalshiSignalLearner,
    cfg: HistoricalBootstrapConfig,
) -> dict[str, Any]:
    """Fetch historical candles from APIs and seed learner outcomes."""
    fetcher = UnifiedDataFetcher()
    all_rows: list[dict[str, Any]] = []
    fetched_symbols: list[str] = []
    failed_symbols: list[str] = []

    for symbol in cfg.symbols:
        try:
            bars = fetcher.fetch_ohlcv(
                symbols=[symbol],
                timeframe=cfg.timeframe,
                days=int(cfg.days),
            )
            if bars.empty:
                failed_symbols.append(symbol)
                continue
            # Unified fetcher may return multi-symbol data.
            sym_norm = _normalize_symbol_for_series(symbol)
            bars_sym = bars.copy()
            if "symbol" in bars_sym.columns:
                bars_sym["symbol"] = bars_sym["symbol"].astype(str).str.upper().str.replace("/", "", regex=False)
                mask = bars_sym["symbol"].str.contains(sym_norm[:3], na=False)
                if mask.any():
                    bars_sym = bars_sym[mask]
            rows = build_historical_outcomes(bars_sym, symbol=symbol, cfg=cfg)
            if rows:
                all_rows.extend(rows)
                fetched_symbols.append(symbol)
            else:
                failed_symbols.append(symbol)
        except Exception:
            failed_symbols.append(symbol)

    if not all_rows:
        return {
            "ok": False,
            "reason": "no_bootstrap_rows",
            "fetched_symbols": fetched_symbols,
            "failed_symbols": failed_symbols,
            "appended": 0,
            "outcomes_total": learner.outcomes_count(),
        }

    all_rows.sort(key=lambda r: abs(_safe_float(r.get("edge"), 0.0)), reverse=True)
    capped = all_rows[: max(1, int(cfg.max_records))]
    appended = learner.append_resolved_outcomes(capped, dedupe=True, recompute=True)

    return {
        "ok": appended > 0,
        "fetched_symbols": fetched_symbols,
        "failed_symbols": failed_symbols,
        "generated": len(capped),
        "appended": appended,
        "outcomes_total": learner.outcomes_count(),
        "policy": learner.policy.to_dict(),
    }
