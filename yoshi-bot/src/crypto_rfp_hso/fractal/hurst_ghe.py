"""Generalized Hurst exponent (GHE) estimation in print-time."""

from __future__ import annotations

import numpy as np


def hurst_ghe(
    log_prices: list[float],
    horizons: list[int],
    window: int,
    q: float = 2.0,
) -> float:
    """Return H in [0.2, 0.8]."""
    if not log_prices:
        return 0.5
    arr = np.asarray(log_prices, dtype=float)
    w = max(8, int(window))
    if arr.size > w:
        arr = arr[-w:]

    valid_h = sorted(set(int(h) for h in horizons if int(h) > 0))
    if len(valid_h) < 2:
        return 0.5

    moments = []
    used_h = []
    qq = max(float(q), 1e-9)
    for h in valid_h:
        if arr.size <= h:
            continue
        diffs = np.abs(arr[h:] - arr[:-h]) ** qq
        m = float(np.mean(diffs)) if diffs.size > 0 else 0.0
        if m > 0.0 and np.isfinite(m):
            moments.append(m)
            used_h.append(float(h))

    if len(moments) < 2:
        return 0.5

    x = np.log(np.asarray(used_h, dtype=float))
    y = np.log(np.asarray(moments, dtype=float))
    slope = float(np.polyfit(x, y, 1)[0])
    h_est = slope / qq
    if not np.isfinite(h_est):
        h_est = 0.5
    return float(np.clip(h_est, 0.2, 0.8))
