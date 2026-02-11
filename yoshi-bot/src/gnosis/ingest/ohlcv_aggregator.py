"""Live OHLCV aggregation with caching + staleness guards.

Used by always-on scripts (e.g. kalshi_scanner.py) to:
  - fetch recent OHLCV via CCXT
  - cache in-memory per symbol
  - expose a `get_latest()` view with per-symbol age/stale metadata

If the feed is stale, callers should skip trading rather than operate on
outdated state.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_symbol_to_ccxt(symbol: str) -> str:
    s = str(symbol).upper().strip()
    if "/" in s:
        return s
    for quote in ("USDT", "USDC", "USD", "BTC", "ETH"):
        if s.endswith(quote) and len(s) > len(quote):
            return f"{s[:-len(quote)]}/{quote}"
    return s


def _ts_to_epoch_s(ts: pd.Timestamp) -> int:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return int(t.timestamp())


@dataclass
class SymbolCache:
    updated_at_epoch_s: int
    last_bar_epoch_s: int
    ohlcv_df: pd.DataFrame
    error: str = ""


class DataSourceAggregator:
    """Fetch + cache OHLCV for a set of symbols."""

    def __init__(
        self,
        exchange: str = "kraken",
        timeframe: str = "1m",
        limit: int = 1000,
        rate_limit_ms: int = 250,
        sandbox: bool = False,
        max_concurrency: int = 2,
        stale_after_sec: int = 180,
    ):
        self.exchange = str(exchange).lower().strip()
        self.timeframe = str(timeframe)
        self.limit = int(max(limit, 10))
        self.rate_limit_ms = int(max(rate_limit_ms, 0))
        self.sandbox = bool(sandbox)
        self.max_concurrency = int(max(max_concurrency, 1))
        self.stale_after_sec = int(max(stale_after_sec, 1))

        self._lock = asyncio.Lock()
        self._cache: dict[str, SymbolCache] = {}

        # Lazy-init loader to keep ccxt optional outside live scripts.
        self._loader = None

    def _get_loader(self):
        if self._loader is None:
            from .ccxt_loader import CCXTLoader

            self._loader = CCXTLoader(
                exchange=self.exchange,
                rate_limit_ms=self.rate_limit_ms,
                sandbox=self.sandbox,
            )
        return self._loader

    def _fetch_symbol_sync(self, symbol: str) -> pd.DataFrame:
        loader = self._get_loader()
        ccxt_symbol = _normalize_symbol_to_ccxt(symbol)
        # When since=None, most exchanges return the latest `limit` candles.
        return loader.fetch_ohlcv(ccxt_symbol, timeframe=self.timeframe, limit=self.limit)

    async def fetch_cycle(self, symbols: list[str]) -> dict[str, Any]:
        """Fetch an update cycle for all symbols and update cache."""
        sem = asyncio.Semaphore(self.max_concurrency)

        async def _one(sym: str):
            async with sem:
                try:
                    df = await asyncio.to_thread(self._fetch_symbol_sync, sym)
                    if df is None or df.empty:
                        raise RuntimeError("empty_ohlcv")
                    last_bar_epoch_s = _ts_to_epoch_s(pd.Timestamp(df["timestamp"].iloc[-1]))
                    payload = SymbolCache(
                        updated_at_epoch_s=int(time.time()),
                        last_bar_epoch_s=last_bar_epoch_s,
                        ohlcv_df=df,
                        error="",
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    payload = SymbolCache(
                        updated_at_epoch_s=int(time.time()),
                        last_bar_epoch_s=0,
                        ohlcv_df=pd.DataFrame(),
                        error=str(exc),
                    )

                async with self._lock:
                    self._cache[str(sym)] = payload

        await asyncio.gather(*[_one(str(s)) for s in (symbols or [])])
        return self.get_latest()

    def get_latest(self, max_age_sec: Optional[int] = None) -> dict[str, Any]:
        """Return cached OHLCV + staleness metadata."""
        now = int(time.time())
        max_age = int(max_age_sec) if max_age_sec is not None else None

        symbols_out: dict[str, Any] = {}
        for sym, entry in self._cache.items():
            last_bar = int(entry.last_bar_epoch_s or 0)
            age = (now - last_bar) if last_bar > 0 else 10**9
            is_stale = bool(age > self.stale_after_sec)

            if max_age is not None and age > max_age:
                continue

            df = entry.ohlcv_df
            if df is None or df.empty:
                ohlcv_rows: list[dict[str, Any]] = []
            else:
                # Serialize minimal OHLCV schema for script consumption.
                _df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
                _df["timestamp"] = _df["timestamp"].apply(lambda x: _ts_to_epoch_s(pd.Timestamp(x)))
                ohlcv_rows = _df.to_dict(orient="records")

            symbols_out[sym] = {
                "updated_at": datetime.fromtimestamp(entry.updated_at_epoch_s, tz=timezone.utc).isoformat(),
                "last_bar_epoch_s": last_bar,
                "age_sec": int(age),
                "is_stale": bool(is_stale),
                "error": entry.error,
                "ohlcv": ohlcv_rows,
            }

        return {
            "source": f"ccxt:{self.exchange}",
            "timeframe": self.timeframe,
            "generated_at": _now_iso(),
            "stale_after_sec": int(self.stale_after_sec),
            "symbols": symbols_out,
        }

