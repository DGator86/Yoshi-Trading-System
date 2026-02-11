"""Feature signature orchestration."""

from __future__ import annotations

import math

from crypto_rfp_hso.features.coupling import compute_coupling_features
from crypto_rfp_hso.features.orderbook import compute_orderbook_features
from crypto_rfp_hso.features.perp import compute_perp_features
from crypto_rfp_hso.features.price import compute_price_features
from crypto_rfp_hso.features.shock import compute_shock_features
from crypto_rfp_hso.fractal.hurst_ghe import hurst_ghe


def compute_feature_signature(
    buckets: list[dict],
    k: int,
    l2_snapshots: list[dict],
    perp_metrics: list[dict],
    coupling_inputs: dict,
    config: dict,
) -> dict:
    """Return FeatureSignature for bucket k."""
    if not buckets or k < 0 or k >= len(buckets):
        raise IndexError("bucket index k is out of range")

    price_features = compute_price_features(buckets=buckets, k=k, config=config)
    orderbook_features = compute_orderbook_features(l2_snapshots=l2_snapshots, k=k, config=config)
    perp_features = compute_perp_features(
        perp_metrics=perp_metrics,
        buckets=buckets,
        k=k,
        config=config,
    )

    symbol = str(buckets[k].get("symbol", coupling_inputs.get("symbol", "")) or "")
    coupling_features = compute_coupling_features(
        coupling_inputs=coupling_inputs,
        symbol=symbol,
        k=k,
        config=config,
    )

    shock_features = compute_shock_features(
        price_features=price_features,
        orderbook_features=orderbook_features,
        perp_features=perp_features,
        config=config,
    )

    closes = [float(b.get("close", 0.0)) for b in buckets[: k + 1]]
    log_prices = [math.log(max(c, 1e-12)) for c in closes]
    h_k = hurst_ghe(
        log_prices=log_prices,
        horizons=list(config.get("hurst_horizons", [1, 2, 4, 8, 16, 32])),
        window=int(config.get("hurst_window", 1500)),
        q=2.0,
    )

    coupling = float(coupling_features.get("coupling", 0.0))
    idiosyncratic_strength = (1.0 - coupling) * abs(float(price_features.get("mu_k", 0.0)))

    signature = {
        "k": int(k),
        "symbol": symbol,
        **price_features,
        **orderbook_features,
        **perp_features,
        **coupling_features,
        **shock_features,
        "H_k": float(h_k),
        "idiosyncratic_strength": float(idiosyncratic_strength),
    }
    return signature
