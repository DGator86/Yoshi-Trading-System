"""Forward collapse-location fan construction."""

from __future__ import annotations

import math

from crypto_rfp_hso.projection.mixture import mixture_quantile


def _q_name(q: float) -> str:
    return f"q{int(round(float(q) * 100)):02d}"


def build_forward_fan(
    x0: float,  # log-price or price
    forward_pi: dict[int, dict[str, float]],
    state_params: dict[str, dict],
    horizons: list[int],
    quantiles: list[float],
    H_eff: float,
    space: str = "log",  # "log" or "price"
) -> list[dict]:
    """Return list of {tau, q05, q50, q95}."""
    out: list[dict] = []
    for tau in sorted(int(h) for h in horizons):
        weights = forward_pi.get(tau, {})
        row: dict = {"tau": tau}
        for q in quantiles:
            qret = mixture_quantile(
                q=float(q),
                weights=weights,
                state_params=state_params,
                tau=tau,
                h_eff=float(H_eff),
            )
            if space == "log":
                value = math.exp(float(x0) + qret)
            else:
                value = max(float(x0) * (1.0 + qret), 1e-12)
            row[_q_name(q)] = float(value)
        out.append(row)
    return out
