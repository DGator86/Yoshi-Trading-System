from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .bars import BarManager
from .timeframes import TF_LIST, minutes_to_close, to_utc


def _log_return(current: float, prior: float) -> float:
    if prior <= 0:
        return 0.0
    return float(np.log(current / prior))


def compute_feature_row(
    bar_manager: BarManager,
    now_ts: pd.Timestamp,
    primary_target_tf: str = "1h",
    timeframes: Iterable[str] = TF_LIST,
) -> Dict[str, float]:
    """Compute a cross-timeframe feature row using closed bars only."""
    now_ts = to_utc(now_ts)
    features: Dict[str, float] = {}

    for tf in timeframes:
        bars = bar_manager.bars(tf)
        if not bars:
            continue

        last_bar = bars[-1]
        if last_bar.timestamp > now_ts:
            raise ValueError(
                f"Lookahead detected: {tf} bar {last_bar.timestamp} beyond {now_ts}"
            )

        features[f"{tf}_close"] = last_bar.close

        if len(bars) >= 2:
            prev_bar = bars[-2]
            features[f"{tf}_logret_1"] = _log_return(last_bar.close, prev_bar.close)

        if len(bars) >= 5:
            returns = [
                _log_return(bars[i].close, bars[i - 1].close)
                for i in range(len(bars) - 4, len(bars))
            ]
            features[f"{tf}_vol_5"] = float(np.std(returns))

        if len(bars) >= 20:
            closes = np.array([b.close for b in bars[-20:]], dtype=float)
            features[f"{tf}_mean_20"] = float(np.mean(closes))
            features[f"{tf}_zscore_20"] = float(
                (last_bar.close - np.mean(closes)) / (np.std(closes) + 1e-9)
            )

    features["minutes_to_1h_close"] = minutes_to_close(now_ts, primary_target_tf)
    return features


def build_direction_labels(
    bars_1h: List[pd.Timestamp], closes: List[float]
) -> pd.DataFrame:
    """Build direction labels for 1H bars (y_t = 1 if C_{t+1} > C_t)."""
    if len(bars_1h) != len(closes):
        raise ValueError("bars_1h and closes must have matching lengths")

    data = {
        "timestamp": bars_1h[:-1],
        "close": closes[:-1],
        "next_close": closes[1:],
    }
    df = pd.DataFrame(data)
    df["label"] = (df["next_close"] > df["close"]).astype(int)
    return df
