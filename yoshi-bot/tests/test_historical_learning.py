"""Tests for API-based historical learning bootstrap."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.gnosis.execution.historical_learning import (  # noqa: E402
    HistoricalBootstrapConfig,
    bootstrap_learning_from_api,
    build_historical_outcomes,
)
from src.gnosis.execution.signal_learning import KalshiSignalLearner  # noqa: E402


def _synthetic_ohlcv(n: int = 240) -> pd.DataFrame:
    ts = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    prices = []
    p = 68000.0
    for i in range(n):
        # Mild trend + oscillation produces mixed outcomes.
        p = p * (1.0 + 0.0008 + 0.0025 * ((i % 17) - 8) / 8.0)
        prices.append(p)
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": prices,
            "high": [x * 1.0015 for x in prices],
            "low": [x * 0.9985 for x in prices],
            "close": prices,
            "volume": [1200.0 + ((i % 13) * 20.0) for i in range(n)],
            "symbol": ["BTCUSDT"] * n,
        }
    )
    return df


def test_build_historical_outcomes_generates_rows():
    cfg = HistoricalBootstrapConfig(
        symbols=["BTCUSDT"],
        days=30,
        timeframe="1h",
        horizon_bars=1,
        min_abs_edge=0.05,
        max_records=500,
    )
    rows = build_historical_outcomes(_synthetic_ohlcv(), symbol="BTCUSDT", cfg=cfg)
    assert rows, "expected non-empty synthetic outcomes"
    first = rows[0]
    assert first["action"] in {"BUY_YES", "BUY_NO"}
    assert "pnl_cents" in first
    assert "settled_yes" in first


def test_bootstrap_learning_from_api_appends_and_updates_policy(tmp_path, monkeypatch):
    learner = KalshiSignalLearner(
        state_path=tmp_path / "state.json",
        outcomes_path=tmp_path / "outcomes.jsonl",
        policy_path=tmp_path / "policy.json",
        min_samples=10,
        lookback=200,
        base_yes_edge=0.10,
        base_no_edge=0.13,
    )

    class _StubFetcher:
        def __init__(self, *args, **kwargs):
            pass

        def fetch_ohlcv(self, symbols, timeframe="1h", days=30, start=None, end=None, provider=None):
            _ = (symbols, timeframe, days, start, end, provider)
            return _synthetic_ohlcv()

    import src.gnosis.execution.historical_learning as hl

    monkeypatch.setattr(hl, "UnifiedDataFetcher", _StubFetcher)

    cfg = HistoricalBootstrapConfig(
        symbols=["BTCUSDT"],
        days=60,
        timeframe="1h",
        horizon_bars=1,
        min_abs_edge=0.05,
        max_records=600,
    )
    report = bootstrap_learning_from_api(learner, cfg)
    assert report["ok"] is True
    assert report["appended"] > 0
    assert learner.outcomes_count() >= report["appended"]
    assert learner.policy.n_resolved >= 10
