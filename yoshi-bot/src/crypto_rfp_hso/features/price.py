"""Price/diffusion features in print-time buckets."""

from __future__ import annotations

import math

import numpy as np

from crypto_rfp_hso.core.math import zscore_current
from crypto_rfp_hso.core.rolling import ewma, ewma_std


def _close_at(buckets: list[dict], idx: int) -> float:
    return float(buckets[idx].get("close", 0.0))


def compute_price_features(buckets: list[dict], k: int, config: dict) -> dict:
    """Compute return, drift, diffusion and vol-of-vol features for bucket k."""
    if not buckets or k < 0 or k >= len(buckets):
        return {
            "r_k": 0.0,
            "mu_k": 0.0,
            "sigma_k": 0.0,
            "vol_of_vol_k": 0.0,
            "sigma_z": 0.0,
            "vol_of_vol_z": 0.0,
        }

    mu_w = int(config.get("ewma_mu_window", 200))
    sigma_w = int(config.get("ewma_sigma_window", 200))

    closes = [_close_at(buckets, i) for i in range(k + 1)]
    log_returns: list[float] = []
    for i in range(1, len(closes)):
        c0 = max(closes[i - 1], 1e-12)
        c1 = max(closes[i], 1e-12)
        log_returns.append(math.log(c1 / c0))

    r_k = float(log_returns[-1]) if log_returns else 0.0
    mu_k = ewma(log_returns, mu_w) if log_returns else 0.0
    sigma_k = ewma_std(log_returns, sigma_w) if log_returns else 0.0

    sigma_hist = []
    for i in range(1, len(log_returns) + 1):
        sigma_hist.append(ewma_std(log_returns[:i], sigma_w))

    sigma_changes = np.diff(np.asarray(sigma_hist, dtype=float)).tolist() if len(sigma_hist) > 1 else []
    vol_of_vol_k = ewma_std(sigma_changes, sigma_w) if sigma_changes else 0.0

    sigma_z = zscore_current(sigma_hist, sigma_k)
    vol_of_vol_z = zscore_current(
        [abs(x) for x in sigma_changes] if sigma_changes else [0.0],
        abs(vol_of_vol_k),
    )

    return {
        "r_k": float(r_k),
        "mu_k": float(mu_k),
        "sigma_k": float(sigma_k),
        "vol_of_vol_k": float(vol_of_vol_k),
        "sigma_z": float(sigma_z),
        "vol_of_vol_z": float(vol_of_vol_z),
    }
