from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

import pandas as pd

from .bars import Bar, BarManager
from .features import compute_feature_row
from .scheduler import due_timeframes, floor_to_second
from .timeframes import TF_LIST, to_utc


class DataProvider(Protocol):
    def get_closed_candles(self, symbol: str, timeframe: str, limit: int = 1) -> pd.DataFrame:
        """Return a dataframe of closed candles with columns timestamp, open, high, low, close, volume."""


@dataclass
class LiveMTFConfig:
    symbols: Iterable[str]
    target_tf: str = "1h"
    window_bars: int = 2000
    timeframes: Iterable[str] = tuple(TF_LIST)
    heartbeat_seconds: int = 1


class LiveMTFLoop:
    def __init__(self, provider: DataProvider, config: LiveMTFConfig) -> None:
        self.provider = provider
        self.config = config
        self.bar_manager = BarManager(window_bars=config.window_bars, timeframes=config.timeframes)

    def _ingest_bar(self, symbol: str, timeframe: str) -> None:
        df = self.provider.get_closed_candles(symbol, timeframe, limit=1)
        if df is None or df.empty:
            return
        row = df.iloc[-1]
        bar = Bar(
            timestamp=to_utc(row["timestamp"]),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row.get("volume", 0.0)),
        )
        self.bar_manager.update_bar(timeframe, bar)

    def run_once(self, now_ts: pd.Timestamp) -> dict:
        ts = floor_to_second(now_ts)
        due = due_timeframes(ts, self.config.timeframes)
        for tf in due:
            for symbol in self.config.symbols:
                self._ingest_bar(symbol, tf)

        predictions = {}
        for symbol in self.config.symbols:
            features = compute_feature_row(
                self.bar_manager,
                ts,
                primary_target_tf=self.config.target_tf,
                timeframes=self.config.timeframes,
            )
            predictions[symbol] = {
                "timestamp": ts,
                "features": features,
            }
        return predictions
