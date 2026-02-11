"""Smoke test for regime-first walk-forward runner."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from gnosis.regimes.crypto_taxonomy import CryptoRegimeConfig, add_multihorizon_regime_probs_from_1m, build_regime_ledger_1m
from gnosis.regime_first.walkforward import run_regime_first_walkforward


def _synthetic_bars(symbol: str, days: int = 3, seed: int = 1) -> pd.DataFrame:
    n = int(days * 24 * 60)
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-02-01", periods=n, freq="min", tz="UTC")
    rets = rng.normal(0, 0.0010, size=n)
    px = 3000 * np.exp(np.cumsum(rets))
    high = px * (1 + rng.uniform(0.0, 0.0012, size=n))
    low = px * (1 - rng.uniform(0.0, 0.0012, size=n))
    open_ = np.roll(px, 1)
    open_[0] = px[0]
    vol = rng.lognormal(mean=3.0, sigma=0.6, size=n)
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


def test_regime_first_walkforward_runs(tmp_path):
    bars = pd.concat(
        [
            _synthetic_bars("ETHUSDT", days=3, seed=3),
            _synthetic_bars("BTCUSDT", days=3, seed=5),
        ],
        ignore_index=True,
    )

    # Smaller windows for speed in test.
    rcfg = CryptoRegimeConfig(n_short=30, n_long=120, n_persist=20, er_short=10, er_long=30, beta_regime=3.0)
    ledger = build_regime_ledger_1m(bars, cfg=rcfg)
    ledger = add_multihorizon_regime_probs_from_1m(ledger)

    cfg_raw = {
        "system": {"instrument": "perp"},
        "blending": {
            "playbook_weights": {
                "MR": {"mr_fade": 1.0, "trend_follow": 0.2, "breakout_transition": 0.2, "panic_only": 0.0},
                "TR": {"mr_fade": 0.2, "trend_follow": 1.0, "breakout_transition": 0.6, "panic_only": 0.0},
                "CP": {"mr_fade": 0.2, "trend_follow": 0.2, "breakout_transition": 0.5, "panic_only": 0.0},
                "EX": {"mr_fade": 0.1, "trend_follow": 0.6, "breakout_transition": 0.4, "panic_only": 0.0},
                "LQ": {"mr_fade": 0.0, "trend_follow": 0.0, "breakout_transition": 0.0, "panic_only": 1.0},
            },
        },
        "risk": {
            "initial_capital": 10_000.0,
            "position_size_pct": 0.20,
            "entry_threshold": 0.02,  # low for smoke
            "min_permission": 0.10,
            "permission": {"MR": 1.0, "TR": 1.0, "CP": 0.4, "EX": 0.6, "LQ": 0.0},
            "thresholds_by_regime": {
                "stop_atr_mult": {"MR": 1.2, "TR": 1.8, "CP": 1.5, "EX": 2.2, "LQ": 3.0},
                "takeprofit_atr_mult": {"MR": 0.9, "TR": 2.4, "CP": 1.2, "EX": 2.8, "LQ": 0.0},
            },
        },
        "execution": {
            "fees": {"taker_bps": 4.0, "maker_bps": 2.0},
            "slippage": {
                "k_spread": 0.35,
                "k_vol": 0.15,
                "regime_multiplier": {"MR": 1.0, "TR": 1.1, "CP": 1.0, "EX": 1.7, "LQ": 2.8},
                "liquidity_overlay_multiplier": {"TD": 0.8, "NR": 1.0, "TH": 1.6},
            },
        },
        "walkforward": {"train_days": 1, "test_days": 1, "roll_days": 1, "regime_coverage": {"shift_flag_rules": {"max_js_divergence": 1.0}}},
    }

    summary = run_regime_first_walkforward(ledger, cfg_raw=cfg_raw, out_dir=tmp_path)
    assert isinstance(summary, dict)
    assert summary.get("n_blocks", 0) >= 1

    # Artifacts exist
    assert (tmp_path / "walkforward_summary.json").exists()
    assert (tmp_path / "trade_ledger.parquet").exists() or (tmp_path / "trade_ledger.csv").exists()

    # Summary JSON is parseable and contains learned weights.
    data = json.loads((tmp_path / "walkforward_summary.json").read_text(encoding="utf-8"))
    assert "blocks" in data and len(data["blocks"]) >= 1
    assert "learned" in data["blocks"][0]
    assert "playbook_weights" in data["blocks"][0]["learned"]

