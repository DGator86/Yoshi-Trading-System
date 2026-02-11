"""Discrete event-index dynamics and MAP/sigma operators."""

from __future__ import annotations

import math

import numpy as np


def price_step_event(
    p_n: float,
    funding_force_n: float,
    arb_force_n: float,
    theta_n: float,
    sigma0: float,
    delta_tau_n: float,
    xi_n: float = 0.0,
    impulse_n1: float = 0.0,
) -> float:
    """One-step event-index update:

    p_{n+1} = p_n + (F_funding + F_arb) * delta_tau + sigma0 * sqrt(theta*delta_tau)*xi + J
    """
    dt = max(float(delta_tau_n), 1e-12)
    theta = max(float(theta_n), 0.0)
    diffusion = float(sigma0) * math.sqrt(theta * dt) * float(xi_n)
    drift = (float(funding_force_n) + float(arb_force_n)) * dt
    return float(p_n + drift + diffusion + float(impulse_n1))


def expected_impulse(
    liq_probability: float,
    impulse_given_liq: float,
    base_impulse_mean: float = 0.0,
) -> float:
    """Expected impulse E[J] = base + P(Lambda) * E[J | Lambda]."""
    p = float(np.clip(liq_probability, 0.0, 1.0))
    return float(base_impulse_mean + p * float(impulse_given_liq))


def map_step_event(
    p_map_n: float,
    funding_force_n: float,
    arb_force_n: float,
    delta_tau_n: float,
    expected_impulse_n1: float,
) -> float:
    """Most-likely path update (diffusion mean dropped)."""
    dt = max(float(delta_tau_n), 1e-12)
    drift = (float(funding_force_n) + float(arb_force_n)) * dt
    return float(p_map_n + drift + float(expected_impulse_n1))


def compute_liquidation_hazard_index(
    liq_density_local: float,
    liquidity_local: float,
    eps: float = 1e-12,
) -> float:
    """Approximate local liquidation hazard index LHI at MAP price."""
    return float(max(float(liq_density_local), 0.0) / (max(float(liquidity_local), 0.0) + eps))


def compute_theta_event(
    theta_base: float,
    funding_force: float,
    lhi: float,
    alpha_f: float,
    beta_l: float,
) -> float:
    """Theta = theta_base * (1 + alpha_f*|F|) * (1 + beta_l*LHI)."""
    t = max(float(theta_base), 1e-12)
    t *= 1.0 + float(alpha_f) * abs(float(funding_force))
    t *= 1.0 + float(beta_l) * max(float(lhi), 0.0)
    return float(max(t, 1e-12))


def map_path_and_bands(
    x0_log: float,
    delta_tau_seq: list[float],
    funding_force_seq: list[float],
    arb_force_seq: list[float],
    expected_impulse_seq: list[float],
    sigma0: float,
    theta_seq: list[float],
    tau0: float = 0.0,
) -> list[dict]:
    """Build MAP path + +-1sigma / +-2sigma bands in event-time.

    The output is price-space while state propagation happens in log-price.
    """
    n = min(
        len(delta_tau_seq),
        len(funding_force_seq),
        len(arb_force_seq),
        len(expected_impulse_seq),
        len(theta_seq),
    )
    if n <= 0:
        return []

    x_map = float(x0_log)
    var = 0.0
    tau_running = float(tau0)
    out: list[dict] = []
    s0 = max(float(sigma0), 1e-12)

    for i in range(n):
        dt = max(float(delta_tau_seq[i]), 1e-12)
        tau_running += dt
        x_map = map_step_event(
            p_map_n=x_map,
            funding_force_n=float(funding_force_seq[i]),
            arb_force_n=float(arb_force_seq[i]),
            delta_tau_n=dt,
            expected_impulse_n1=float(expected_impulse_seq[i]),
        )
        var += (s0 ** 2) * max(float(theta_seq[i]), 1e-12) * dt
        sigma = math.sqrt(max(var, 1e-18))

        out.append(
            {
                "tau": int(i + 1),
                "tau_event": float(tau_running),
                "delta_tau": float(dt),
                "p_map_log": float(x_map),
                "p_map": float(math.exp(x_map)),
                "var_log": float(var),
                "sigma_log": float(sigma),
                "sigma1_low": float(math.exp(x_map - sigma)),
                "sigma1_high": float(math.exp(x_map + sigma)),
                "sigma2_low": float(math.exp(x_map - 2.0 * sigma)),
                "sigma2_high": float(math.exp(x_map + 2.0 * sigma)),
                "funding_force": float(funding_force_seq[i]),
                "arb_force": float(arb_force_seq[i]),
                "expected_impulse": float(expected_impulse_seq[i]),
                "theta": float(theta_seq[i]),
            }
        )
    return out
