from __future__ import annotations

from datetime import timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd


def parse_iso_timestamp(value: str) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True)
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    return ts


def parse_ranges(ranges_arg: Optional[str]) -> Optional[List[Tuple[pd.Timestamp, pd.Timestamp]]]:
    if not ranges_arg:
        return None
    ranges: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    raw_ranges = [item.strip() for item in ranges_arg.split(",") if item.strip()]
    for raw_range in raw_ranges:
        if ":" not in raw_range:
            raise ValueError(f"Invalid range '{raw_range}'. Expected start:end format.")
        # ISO timestamps contain ":" (HH:MM:SS), so we can't split on the first colon.
        # Instead, try all split points until both sides parse as timestamps.
        parts = [p.strip() for p in raw_range.split(":")]
        if len(parts) < 2:
            raise ValueError(f"Invalid range '{raw_range}'. Expected start:end format.")

        start_ts = end_ts = None
        for i in range(1, len(parts)):
            start_str = ":".join(parts[:i]).strip()
            end_str = ":".join(parts[i:]).strip()
            if not start_str or not end_str:
                continue
            try:
                _start = parse_iso_timestamp(start_str)
                _end = parse_iso_timestamp(end_str)
            except Exception:
                continue
            start_ts, end_ts = _start, _end
            break

        if start_ts is None or end_ts is None:
            raise ValueError(f"Invalid range '{raw_range}'. Expected start:end format.")
        if start_ts >= end_ts:
            raise ValueError(f"Invalid range '{raw_range}'. Start must be before end.")
        ranges.append((start_ts, end_ts))

    sorted_ranges = sorted(ranges, key=lambda item: item[0])
    for prev, current in zip(sorted_ranges, sorted_ranges[1:]):
        if current[0] <= prev[1]:
            raise ValueError(
                f"Invalid range '{current[0].isoformat()}:{current[1].isoformat()}'. "
                "Ranges must not overlap."
            )

    return ranges


def filter_by_range(
    bars_by_tf: Dict[str, pd.DataFrame],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> Dict[str, pd.DataFrame]:
    filtered: Dict[str, pd.DataFrame] = {}
    for tf, df in bars_by_tf.items():
        if df.empty:
            filtered[tf] = df
            continue
        mask = (df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)
        filtered[tf] = df.loc[mask].reset_index(drop=True)
    return filtered


def filter_by_ranges_union(
    bars_by_tf: Dict[str, pd.DataFrame],
    ranges: List[Tuple[pd.Timestamp, pd.Timestamp]],
) -> Dict[str, pd.DataFrame]:
    filtered: Dict[str, pd.DataFrame] = {}
    for tf, df in bars_by_tf.items():
        if df.empty:
            filtered[tf] = df
            continue
        mask = pd.Series(False, index=df.index)
        for start_ts, end_ts in ranges:
            mask |= (df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)
        filtered[tf] = df.loc[mask].reset_index(drop=True)
    return filtered
