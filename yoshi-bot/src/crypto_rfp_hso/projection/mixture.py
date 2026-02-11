"""Student-t mixture helpers for forward quantiles."""

from __future__ import annotations

import numpy as np
from scipy.stats import t as student_t

from crypto_rfp_hso.fractal.scaling import scale_sigma_hurst


def mixture_cdf(
    x: float,
    weights: dict[str, float],
    state_params: dict[str, dict],
    tau: int,
    h_eff: float,
) -> float:
    """Mixture CDF at x for tau-step return."""
    total = 0.0
    for state, w in weights.items():
        p = state_params.get(state, {"mu": 0.0, "sigma": 1e-6, "nu": 8.0})
        mu = float(p.get("mu", 0.0)) * max(int(tau), 1)
        sigma = scale_sigma_hurst(float(p.get("sigma", 1e-6)), tau=max(int(tau), 1), h_eff=h_eff)
        sigma = max(float(sigma), 1e-8)
        nu = max(float(p.get("nu", 8.0)), 2.1)
        total += float(w) * float(student_t.cdf(x, df=nu, loc=mu, scale=sigma))
    return float(np.clip(total, 0.0, 1.0))


def mixture_quantile(
    q: float,
    weights: dict[str, float],
    state_params: dict[str, dict],
    tau: int,
    h_eff: float,
) -> float:
    """Compute mixture quantile by deterministic bisection."""
    qq = float(np.clip(q, 1e-6, 1.0 - 1e-6))
    if not weights:
        return 0.0

    means = []
    scales = []
    for state, w in weights.items():
        if w <= 0:
            continue
        p = state_params.get(state, {"mu": 0.0, "sigma": 1e-6, "nu": 8.0})
        means.append(float(p.get("mu", 0.0)) * max(int(tau), 1))
        scales.append(scale_sigma_hurst(float(p.get("sigma", 1e-6)), tau=max(int(tau), 1), h_eff=h_eff))

    if not means:
        return 0.0
    lo = min(means) - 12.0 * max(scales)
    hi = max(means) + 12.0 * max(scales)
    lo = float(lo)
    hi = float(hi)
    if hi <= lo:
        return float(means[0])

    for _ in range(120):
        mid = 0.5 * (lo + hi)
        c = mixture_cdf(mid, weights=weights, state_params=state_params, tau=tau, h_eff=h_eff)
        if c < qq:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
