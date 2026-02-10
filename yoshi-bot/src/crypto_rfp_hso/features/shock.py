"""Shock and friction feature derivations."""

from __future__ import annotations

from crypto_rfp_hso.core.math import sigmoid


def compute_shock_features(
    price_features: dict,
    orderbook_features: dict,
    perp_features: dict,
    config: dict,
) -> dict:
    """Compute gap/book-vacuum/cascade flags and friction proxy."""
    sigma = float(price_features.get("sigma_k", 0.0))
    r_k = abs(float(price_features.get("r_k", 0.0)))
    gap_mult = float(config.get("shock_gap_sigma_mult", 4.0))
    gap_flag = 1.0 if sigma > 0.0 and r_k > gap_mult * sigma else 0.0

    spread_z = float(orderbook_features.get("spread_z", 0.0))
    depth_z = float(orderbook_features.get("depth_sum_z", 0.0))
    sigma_z = float(price_features.get("sigma_z", 0.0))
    liq_int_z = float(perp_features.get("liq_intensity_z", 0.0))

    book_vacuum = 1.0 if (depth_z < -1.5 and spread_z > 1.5) else 0.0
    liq_cascade_flag = 1.0 if (liq_int_z > 2.0 and spread_z > 1.0 and sigma_z > 1.0) else 0.0

    friction = sigmoid(
        spread_z
        + float(orderbook_features.get("impact_z", 0.0))
        + float(price_features.get("vol_of_vol_z", 0.0))
    )

    return {
        "gap_flag": float(gap_flag),
        "book_vacuum": float(book_vacuum),
        "liq_cascade_flag": float(liq_cascade_flag),
        "friction": float(friction),
    }
