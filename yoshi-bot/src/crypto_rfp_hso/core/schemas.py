"""Typed schema helpers and default configuration."""

from typing import TypedDict


class PrintBucket(TypedDict, total=False):
    """Event-time print bucket schema."""

    k: int
    symbol: str
    timestamp_start: int | float | str
    timestamp_end: int | float | str
    open: float
    high: float
    low: float
    close: float
    vwap: float
    volume: float
    notional: float
    n_trades: int
    buy_volume: float
    sell_volume: float


class FeatureSignature(TypedDict, total=False):
    """Feature signature schema."""

    k: int
    symbol: str
    r_k: float
    mu_k: float
    sigma_k: float
    vol_of_vol_k: float
    H_k: float


DEFAULT_CONFIG = {
    "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    "bucket_mode": "NOTIONAL",
    "bucket_notional": {
        "BTCUSDT": 5_000_000,
        "ETHUSDT": 2_000_000,
        "SOLUSDT": 500_000,
    },
    "ewma_mu_window": 200,
    "ewma_sigma_window": 200,
    "hurst_window": 1500,
    "hurst_horizons": [1, 2, 4, 8, 16, 32],
    "regime_temperature": 6.0,
    "order_temperature": 5.0,
    "tau_min_duration": 5,
    "forward_horizons": [1, 2, 4, 8, 16, 32],
    "fan_quantiles": [0.05, 0.50, 0.95],
    "entropy_epsilon": 1e-9,
    "impact_notional_usd": 100_000.0,
    "shock_gap_sigma_mult": 4.0,
    "h_min": 0.01,
    "h_max": 0.99,
    "base_alpha": 0.7,
}
