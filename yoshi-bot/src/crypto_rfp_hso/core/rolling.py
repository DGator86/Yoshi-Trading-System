"""Rolling/EWMA utilities."""

from __future__ import annotations

import numpy as np


def ewma(values: list[float], span: int) -> float:
    """EWMA terminal value."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    alpha = 2.0 / (max(int(span), 1) + 1.0)
    out = arr[0]
    for x in arr[1:]:
        out = alpha * x + (1.0 - alpha) * out
    return float(out)


def ewma_series(values: list[float], span: int) -> np.ndarray:
    """Full EWMA series."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.asarray([], dtype=float)
    alpha = 2.0 / (max(int(span), 1) + 1.0)
    out = np.zeros_like(arr, dtype=float)
    out[0] = arr[0]
    for i in range(1, arr.size):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def ewma_std(values: list[float], span: int, eps: float = 1e-12) -> float:
    """EWMA standard deviation using EWMA mean."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    mean_s = ewma_series(values, span)
    sq_dev = (arr - mean_s) ** 2
    var = ewma(sq_dev.tolist(), span)
    return float(np.sqrt(max(var, eps)))
