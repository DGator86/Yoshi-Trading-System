"""Live data aggregator used by the Kalshi scanner.

Provides an async-friendly wrapper around CCXT OHLCV fetches with caching.
"""

from __future__ import annotations

import asyncio
import copy
import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from gnosis.ingest.ccxt_loader import CCXTLoader

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_symbol_to_ccxt(symbol: str) -> str:
    s = str(symbol).upper()
    if "/" in s:
        return s
    for quote in ("USDT", "USDC", "USD", "BTC", "ETH"):
        if s.endswith(quote) and len(s) > len(quote):
            return f"{s[:-len(quote)]}/{quote}"
    return s


class DataSourceAggregator:
    """Collect and cache OHLCV snapshots for one or more symbols."""

    def __init__(
        self,
        exchange: str = "kraken",
        timeframe: str = "1m",
        lookback_bars: int = 720,
        min_refresh_sec: float = 20.0,
        rate_limit_ms: int = 250,
    ):
        self.exchange = exchange
        self.timeframe = timeframe
        self.lookback_bars = int(lookback_bars)
        self.min_refresh_sec = float(min_refresh_sec)
        self.loader = CCXTLoader(exchange=exchange, rate_limit_ms=rate_limit_ms)
        self._last_refresh_epoch = 0.0
        self._latest: dict[str, Any] = {
            "fetched_at": None,
            "source": f"ccxt:{exchange}",
            "symbols": {},
        }
        self._lock = asyncio.Lock()

    async def fetch_cycle(self, symbols: list[str]) -> dict[str, Any]:
        """Fetch/update cached OHLCV snapshot for requested symbols."""
        if not symbols:
            return self.get_latest() or {}

        async with self._lock:
            now = asyncio.get_running_loop().time()
            if (
                self._latest.get("fetched_at")
                and now - self._last_refresh_epoch < self.min_refresh_sec
            ):
                return copy.deepcopy(self._latest)

            tasks = [asyncio.to_thread(self._fetch_symbol_sync, sym) for sym in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            symbols_payload: dict[str, Any] = copy.deepcopy(self._latest.get("symbols", {}))
            for sym, res in zip(symbols, results):
                if isinstance(res, Exception):
                    logger.warning("OHLCV fetch failed for %s: %s", sym, res)
                    continue
                if res:
                    symbols_payload[sym] = res

            self._latest = {
                "fetched_at": _now_iso(),
                "source": f"ccxt:{self.exchange}",
                "symbols": symbols_payload,
            }
            self._last_refresh_epoch = now
            return copy.deepcopy(self._latest)

    def _fetch_symbol_sync(self, symbol: str) -> dict[str, Any]:
        ccxt_symbol = _normalize_symbol_to_ccxt(symbol)
        df = self.loader.fetch_ohlcv(
            ccxt_symbol,
            timeframe=self.timeframe,
            limit=self.lookback_bars,
        )
        if df.empty:
            raise RuntimeError(f"No OHLCV returned for {symbol} ({ccxt_symbol})")

        cols = ["timestamp", "open", "high", "low", "close", "volume"]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise RuntimeError(f"Missing OHLCV columns for {symbol}: {missing}")

        tail = df[cols].tail(self.lookback_bars).copy()
        records = []
        for _, row in tail.iterrows():
            ts = row["timestamp"]
            if isinstance(ts, pd.Timestamp):
                ts_s = int(ts.tz_convert("UTC").timestamp()) if ts.tzinfo else int(ts.timestamp())
            else:
                ts_s = int(pd.Timestamp(ts, tz="UTC").timestamp())
            records.append(
                {
                    "timestamp": ts_s,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                }
            )

        return {
            "ohlcv": records,
            "exchange": self.exchange,
            "timeframe": self.timeframe,
            "ccxt_symbol": ccxt_symbol,
            "fetched_at": _now_iso(),
        }

    def get_latest(self) -> dict[str, Any]:
        """Return a copy of latest in-memory snapshot."""
        return copy.deepcopy(self._latest)

