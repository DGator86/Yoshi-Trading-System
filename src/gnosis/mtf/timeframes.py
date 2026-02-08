from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd

TF_LIST = ["1m", "5m", "15m", "30m", "1h", "4h", "12h", "1d"]

TF_SECONDS: Dict[str, int] = {
    "1m": 60,
    "5m": 5 * 60,
    "15m": 15 * 60,
    "30m": 30 * 60,
    "1h": 60 * 60,
    "4h": 4 * 60 * 60,
    "12h": 12 * 60 * 60,
    "1d": 24 * 60 * 60,
}


@dataclass(frozen=True)
class TimeframeBoundary:
    timeframe: str
    seconds: int


TIMEFRAME_BOUNDARIES = {
    tf: TimeframeBoundary(tf, seconds) for tf, seconds in TF_SECONDS.items()
}


def to_utc(timestamp: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def is_timeframe_boundary(timestamp: pd.Timestamp, timeframe: str) -> bool:
    """Return True if the timestamp is aligned to a timeframe boundary."""
    boundary = TIMEFRAME_BOUNDARIES[timeframe]
    ts = to_utc(timestamp)
    epoch_seconds = int(ts.timestamp())
    return epoch_seconds % boundary.seconds == 0


def next_timeframe_close(timestamp: pd.Timestamp, timeframe: str) -> pd.Timestamp:
    """Return the next bar close time for the given timeframe."""
    boundary = TIMEFRAME_BOUNDARIES[timeframe]
    ts = to_utc(timestamp)
    epoch_seconds = int(ts.timestamp())
    remainder = epoch_seconds % boundary.seconds
    if remainder == 0:
        next_seconds = epoch_seconds + boundary.seconds
    else:
        next_seconds = epoch_seconds + (boundary.seconds - remainder)
    return pd.Timestamp(next_seconds, unit="s", tz="UTC")


def minutes_to_close(timestamp: pd.Timestamp, timeframe: str) -> float:
    """Minutes remaining until the next close of timeframe."""
    next_close = next_timeframe_close(timestamp, timeframe)
    delta = next_close - to_utc(timestamp)
    return delta.total_seconds() / 60.0
