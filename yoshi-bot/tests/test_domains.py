"""Tests for domain aggregation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from gnosis.ingest import generate_stub_prints
from gnosis.domains import DomainAggregator, compute_features


def test_domain_aggregator():
    """Test domain aggregation."""
    prints_df = generate_stub_prints(["BTCUSDT"], n_days=1, trades_per_day=500, seed=42)

    config = {"domains": {"D0": {"n_trades": 100}}}
    agg = DomainAggregator(config)
    bars = agg.aggregate(prints_df, "D0")

    assert isinstance(bars, pd.DataFrame)
    assert len(bars) > 0
    assert "open" in bars.columns
    assert "high" in bars.columns
    assert "low" in bars.columns
    assert "close" in bars.columns
    assert "volume" in bars.columns


def test_compute_features():
    """Test feature computation."""
    prints_df = generate_stub_prints(["BTCUSDT"], n_days=2, trades_per_day=500, seed=42)

    config = {"domains": {"D0": {"n_trades": 100}}}
    agg = DomainAggregator(config)
    bars = agg.aggregate(prints_df, "D0")
    features = compute_features(bars)

    assert "returns" in features.columns
    assert "realized_vol" in features.columns
    assert "ofi" in features.columns
