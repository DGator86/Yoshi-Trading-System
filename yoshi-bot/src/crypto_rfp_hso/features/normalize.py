"""Feature vector extraction and normalization."""

from __future__ import annotations

from crypto_rfp_hso.core.math import l2_normalize


DEFAULT_SIGNATURE_KEYS = [
    "r_k",
    "mu_k",
    "sigma_k",
    "vol_of_vol_k",
    "spread_k",
    "depth_bid_10bps",
    "depth_ask_10bps",
    "imbalance_10bps",
    "book_slope_bid",
    "book_slope_ask",
    "impact_proxy",
    "U2_liq",
    "F_liq",
    "oi_k",
    "dOI_k",
    "funding_k",
    "dfunding_k",
    "liq_intensity_k",
    "liq_bias",
    "U2_perp",
    "F_perp",
    "beta_to_btc",
    "corr_to_btc",
    "coupling",
    "gap_flag",
    "book_vacuum",
    "liq_cascade_flag",
    "friction",
    "H_k",
]


def signature_to_vector(sig: dict, keys: list[str] | None = None) -> list[float]:
    """Convert a signature dict into a fixed-order vector."""
    feature_keys = keys or DEFAULT_SIGNATURE_KEYS
    return [float(sig.get(k, 0.0)) for k in feature_keys]


def normalized_signature_vector(sig: dict, keys: list[str] | None = None) -> list[float]:
    """L2-normalized feature vector."""
    vec = signature_to_vector(sig, keys=keys)
    return l2_normalize(vec).tolist()
