"""Tests for crypto regime taxonomy (MR/TR/CP/EX/LQ)."""

import numpy as np
import pandas as pd

from gnosis.regimes.crypto_taxonomy import (
    CryptoRegimeConfig,
    build_regime_ledger_1m,
)


def _synthetic_1m(symbol: str = "BTCUSDT", n: int = 500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="min", tz="UTC")
    rets = rng.normal(0, 0.0008, size=n)
    px = 50000 * np.exp(np.cumsum(rets))
    # Make OHLC reasonable.
    high = px * (1 + rng.uniform(0.0, 0.0008, size=n))
    low = px * (1 - rng.uniform(0.0, 0.0008, size=n))
    open_ = np.roll(px, 1)
    open_[0] = px[0]
    vol = rng.lognormal(mean=4.0, sigma=0.4, size=n)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": symbol,
            "open": open_,
            "high": high,
            "low": low,
            "close": px,
            "volume": vol,
        }
    )


def test_regime_ledger_probabilities_sum_to_one():
    bars = pd.concat([_synthetic_1m("BTCUSDT"), _synthetic_1m("ETHUSDT", seed=9)], ignore_index=True)
    cfg = CryptoRegimeConfig(n_short=30, n_long=120, n_persist=20, er_short=10, er_long=30, beta_regime=3.0)
    ledger = build_regime_ledger_1m(bars, cfg=cfg)

    for col in ["p_MR", "p_TR", "p_CP", "p_EX", "p_LQ", "conf", "regime_label"]:
        assert col in ledger.columns

    ps = ledger[["p_MR", "p_TR", "p_CP", "p_EX", "p_LQ"]].sum(axis=1).to_numpy()
    assert np.allclose(ps, 1.0, atol=1e-6)

    pmax = ledger[["p_MR", "p_TR", "p_CP", "p_EX", "p_LQ"]].max(axis=1).to_numpy()
    assert np.allclose(pmax, ledger["conf"].to_numpy(), atol=1e-9)


def test_liquidity_overlay_probs_sum_to_one():
    bars = _synthetic_1m("SOLUSDT", n=400)
    cfg = CryptoRegimeConfig(n_short=30, n_long=120, n_persist=20, er_short=10, er_long=30, beta_liquidity=3.0)
    ledger = build_regime_ledger_1m(bars, cfg=cfg)

    for col in ["p_TD", "p_NR", "p_TH", "liq_overlay"]:
        assert col in ledger.columns
    ps = ledger[["p_TD", "p_NR", "p_TH"]].sum(axis=1).to_numpy()
    assert np.allclose(ps, 1.0, atol=1e-6)


def test_transition_flags_consistent():
    bars = _synthetic_1m("BTCUSDT", n=250)
    cfg = CryptoRegimeConfig(n_short=20, n_long=80, n_persist=15, er_short=10, er_long=30)
    ledger = build_regime_ledger_1m(bars, cfg=cfg).reset_index(drop=True)

    assert "transition_flag" in ledger.columns
    assert "transition_type" in ledger.columns
    # transition_type non-empty iff transition_flag True
    nonempty = ledger["transition_type"].astype(str).str.len().to_numpy() > 0
    flag = ledger["transition_flag"].fillna(False).to_numpy().astype(bool)
    assert np.array_equal(nonempty, flag)

