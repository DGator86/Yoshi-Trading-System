"""Hurst-based volatility scaling."""

from __future__ import annotations


def scale_sigma_hurst(sigma_1: float, tau: int, h_eff: float) -> float:
    """Scale one-step sigma to horizon tau with exponent h_eff."""
    t = max(int(tau), 1)
    h = float(h_eff)
    return float(max(sigma_1, 1e-12) * (t ** h))
