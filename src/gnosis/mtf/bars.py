from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, Iterable, List, Optional

import pandas as pd

from .timeframes import TF_LIST, is_timeframe_boundary, to_utc


@dataclass(frozen=True)
class Bar:
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float


class BarBuffer:
    def __init__(self, maxlen: int) -> None:
        self._bars: Deque[Bar] = deque(maxlen=maxlen)

    def append(self, bar: Bar) -> None:
        self._bars.append(bar)

    def __len__(self) -> int:
        return len(self._bars)

    def latest(self) -> Optional[Bar]:
        return self._bars[-1] if self._bars else None

    def all(self) -> List[Bar]:
        return list(self._bars)


class BarManager:
    """Maintain per-timeframe closed-bar buffers."""

    def __init__(self, window_bars: int = 2000, timeframes: Iterable[str] = TF_LIST) -> None:
        self.window_bars = int(window_bars)
        self.timeframes = list(timeframes)
        self._buffers: Dict[str, BarBuffer] = {
            tf: BarBuffer(maxlen=self.window_bars) for tf in self.timeframes
        }
        self._last_closed: Dict[str, Optional[pd.Timestamp]] = {tf: None for tf in self.timeframes}

    def update_bar(self, timeframe: str, bar: Bar) -> None:
        if timeframe not in self._buffers:
            raise KeyError(f"Unknown timeframe: {timeframe}")

        timestamp = to_utc(bar.timestamp)
        if not is_timeframe_boundary(timestamp, timeframe):
            raise ValueError(f"Bar timestamp {timestamp} not aligned to {timeframe} boundary")

        last_ts = self._last_closed[timeframe]
        if last_ts is not None and timestamp <= last_ts:
            return

        self._buffers[timeframe].append(
            Bar(
                timestamp=timestamp,
                open=float(bar.open),
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                volume=float(bar.volume),
            )
        )
        self._last_closed[timeframe] = timestamp

    def latest_bar(self, timeframe: str) -> Optional[Bar]:
        return self._buffers[timeframe].latest()

    def bars(self, timeframe: str) -> List[Bar]:
        return self._buffers[timeframe].all()

    def last_closed_timestamp(self, timeframe: str) -> Optional[pd.Timestamp]:
        return self._last_closed[timeframe]

    def has_bars(self, timeframe: str, count: int = 1) -> bool:
        return len(self._buffers[timeframe]) >= count
