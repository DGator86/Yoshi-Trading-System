"""Price hazard surface and next-price distribution operators."""

from __future__ import annotations

import math

import numpy as np


def _gaussian(x: float, mu: float, sigma: float) -> float:
    s = max(float(sigma), 1e-12)
    z = (float(x) - float(mu)) / s
    return math.exp(-0.5 * z * z)


def liquidity_field_local(
    p: float,
    p_ref: float,
    liquidity_mid: float,
    curvature: float,
) -> float:
    """Simple local liquidity field L(p, n) around p_ref."""
    pref = max(abs(float(p_ref)), 1e-12)
    dist = abs(float(p) - float(p_ref)) / pref
    l0 = max(float(liquidity_mid), 1e-12)
    kappa = max(float(curvature), 0.0)
    # Liquidity decays away from reference with curvature control.
    return float(l0 / (1.0 + kappa * dist * 10_000.0))


def hazard_surface_distribution(
    p_map: float,
    sigma_price: float,
    liquidity_mid: float,
    curvature: float,
    lambda_b: float,
    lambda_s: float,
    grid_size: int = 121,
    band_mult: float = 3.0,
) -> dict:
    """Construct H(p,n) and normalized next-price distribution on a grid."""
    sigma = max(float(sigma_price), max(abs(float(p_map)), 1.0) * 1e-6)
    n = max(int(grid_size), 21)
    span = float(band_mult) * sigma
    lo = float(p_map) - span
    hi = float(p_map) + span
    grid = np.linspace(lo, hi, n)

    shift = 0.25 * sigma
    lb0 = max(float(lambda_b), 1e-12)
    ls0 = max(float(lambda_s), 1e-12)
    h_vals = np.zeros(n, dtype=float)
    for i, p in enumerate(grid):
        lb = lb0 * _gaussian(p, p_map - shift, sigma)
        ls = ls0 * _gaussian(p, p_map + shift, sigma)
        l = liquidity_field_local(p, p_map, liquidity_mid=liquidity_mid, curvature=curvature)
        h_vals[i] = (lb * ls) / max(l, 1e-12)

    total = float(np.sum(h_vals))
    if total <= 0.0:
        probs = np.ones(n, dtype=float) / n
    else:
        probs = h_vals / total

    idx = int(np.argmax(probs))
    return {
        "grid_prices": [float(x) for x in grid.tolist()],
        "probabilities": [float(x) for x in probs.tolist()],
        "hazard": [float(x) for x in h_vals.tolist()],
        "p_star": float(grid[idx]),
    }
