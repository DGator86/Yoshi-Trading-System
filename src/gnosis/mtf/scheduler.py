from __future__ import annotations

from typing import Iterable, List

import pandas as pd

from .timeframes import TF_LIST, TF_SECONDS, to_utc


def due_timeframes(
    timestamp: pd.Timestamp,
    timeframes: Iterable[str] = TF_LIST,
) -> List[str]:
    """Return timeframes whose bars close at the given timestamp."""
    ts = to_utc(timestamp)
    epoch_seconds = int(ts.timestamp())
    due: List[str] = []
    for tf in timeframes:
        seconds = TF_SECONDS[tf]
        if epoch_seconds % seconds == 0:
            due.append(tf)
    return due


def floor_to_second(timestamp: pd.Timestamp) -> pd.Timestamp:
    ts = to_utc(timestamp)
    return ts.floor("s")
