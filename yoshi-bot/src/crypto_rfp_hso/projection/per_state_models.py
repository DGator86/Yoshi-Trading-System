"""Per-state conditional return model fitting."""

from __future__ import annotations

import numpy as np


def _sample_excess_kurtosis(x: np.ndarray) -> float:
    if x.size < 4:
        return 0.0
    centered = x - np.mean(x)
    var = np.var(centered)
    if var <= 1e-12:
        return 0.0
    m4 = float(np.mean(centered ** 4))
    return m4 / (var ** 2) - 3.0


def _nu_from_kurtosis(excess_k: float) -> float:
    # For Student-t, excess kurtosis = 6/(nu-4), nu > 4.
    if excess_k <= 1e-9:
        return 30.0
    nu = 6.0 / excess_k + 4.0
    return float(np.clip(nu, 2.5, 100.0))


def fit_per_state_t_params(
    returns_by_state: dict[str, list[float]],
) -> dict[str, dict]:
    """Return {state: {mu, sigma, nu}}."""
    params: dict[str, dict] = {}
    for state, vals in returns_by_state.items():
        arr = np.asarray([float(v) for v in vals], dtype=float)
        if arr.size == 0:
            params[state] = {"mu": 0.0, "sigma": 1e-6, "nu": 8.0}
            continue
        mu = float(np.mean(arr))
        sigma = float(np.std(arr, ddof=1)) if arr.size > 1 else 1e-6
        sigma = max(sigma, 1e-6)
        excess_k = _sample_excess_kurtosis(arr)
        nu = _nu_from_kurtosis(excess_k)
        params[state] = {"mu": mu, "sigma": sigma, "nu": nu}
    return params
