"""Method gating and blended projection weights."""

from __future__ import annotations

from crypto_rfp_hso.core.enums import METHODS


G = {
    "Balanced": {
        "analytic_local": 0.55,
        "semi_markov_mixture": 0.40,
        "quantile_coverage": 0.10,
        "pde_density": 0.05,
    },
    "Discovery": {
        "analytic_local": 0.20,
        "semi_markov_mixture": 0.55,
        "quantile_coverage": 0.25,
        "pde_density": 0.05,
    },
    "Pinning": {
        "analytic_local": 0.50,
        "semi_markov_mixture": 0.45,
        "quantile_coverage": 0.10,
        "pde_density": 0.05,
    },
    "Shock": {
        "analytic_local": 0.05,
        "semi_markov_mixture": 0.45,
        "quantile_coverage": 0.60,
        "pde_density": 0.10,
    },
    "Transitional": {
        "analytic_local": 0.10,
        "semi_markov_mixture": 0.35,
        "quantile_coverage": 0.60,
        "pde_density": 0.05,
    },
}

ORDER_METHOD_MULT = {
    "Correlation-Driven": {
        "analytic_local": 0.60,
        "semi_markov_mixture": 1.15,
        "quantile_coverage": 1.20,
        "pde_density": 1.00,
    },
    "Positioning-Constraint": {
        "analytic_local": 0.90,
        "semi_markov_mixture": 1.20,
        "quantile_coverage": 1.05,
        "pde_density": 1.00,
    },
    "Information-Override": {
        "analytic_local": 0.40,
        "semi_markov_mixture": 1.10,
        "quantile_coverage": 1.40,
        "pde_density": 1.10,
    },
}


def compute_method_weights(
    w_class: dict[str, float],
    w_order: dict[str, float],
    reliability: dict[str, float] | None = None,
    gating_matrix: dict[str, dict[str, float]] | None = None,
    order_method_mult: dict[str, dict[str, float]] | None = None,
) -> dict[str, float]:
    """Compute normalized method weights alpha_m(k)."""
    gm = gating_matrix or G
    mult_map = order_method_mult or ORDER_METHOD_MULT
    rel = reliability or {m: 1.0 for m in METHODS}

    raw = {m: 0.0 for m in METHODS}
    for regime, prob in w_class.items():
        row = gm.get(regime, {})
        for m in METHODS:
            raw[m] += float(prob) * float(row.get(m, 0.0))

    if w_order:
        dom_order = max(w_order.items(), key=lambda kv: kv[1])[0]
        order_mult = mult_map.get(dom_order, {})
        for m in METHODS:
            raw[m] *= float(order_mult.get(m, 1.0))

    for m in METHODS:
        raw[m] *= float(rel.get(m, 1.0))

    total = sum(max(v, 0.0) for v in raw.values())
    if total <= 0.0:
        uniform = 1.0 / len(METHODS)
        return {m: uniform for m in METHODS}
    return {m: max(v, 0.0) / total for m, v in raw.items()}
