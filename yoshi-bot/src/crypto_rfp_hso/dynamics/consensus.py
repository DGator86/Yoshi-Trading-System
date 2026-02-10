"""Multi-venue convergence operator."""

from __future__ import annotations


def venue_consensus_update(
    venue_prices_next: dict[str, float],
    venue_liquidity: dict[str, float],
    venue_uptime: dict[str, float] | None = None,
    kappa_arb: float = 0.2,
) -> dict[str, float]:
    """Apply arbitrage convergence update:

    p_v <- p_v + kappa * sum_{u!=v} w_u * (p_u - p_v)
    where w_u ~ liquidity_u * uptime_u.
    """
    if not venue_prices_next:
        return {}

    uptime = venue_uptime or {}
    raw_w = {}
    for v in venue_prices_next.keys():
        l = max(float(venue_liquidity.get(v, 0.0)), 0.0)
        u = max(float(uptime.get(v, 1.0)), 0.0)
        raw_w[v] = l * u

    w_sum = sum(raw_w.values())
    if w_sum <= 0.0:
        n = len(venue_prices_next)
        weights = {v: 1.0 / n for v in venue_prices_next.keys()}
    else:
        weights = {v: raw_w[v] / w_sum for v in venue_prices_next.keys()}

    kappa = max(float(kappa_arb), 0.0)
    out = {}
    for v, p_v in venue_prices_next.items():
        correction = 0.0
        for u, p_u in venue_prices_next.items():
            if u == v:
                continue
            correction += weights.get(u, 0.0) * (float(p_u) - float(p_v))
        out[v] = float(p_v) + kappa * correction
    return out
