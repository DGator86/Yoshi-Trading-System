"""Tests for regime classification."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gnosis.ingest import generate_stub_prints
from gnosis.domains import DomainAggregator, compute_features
from gnosis.regimes import KPCOFGSClassifier


def test_kpcofgs_classifier():
    """Test KPCOFGS classification."""
    prints_df = generate_stub_prints(["BTCUSDT"], n_days=2, trades_per_day=500, seed=42)

    config = {"domains": {"D0": {"n_trades": 100}}}
    agg = DomainAggregator(config)
    bars = agg.aggregate(prints_df, "D0")
    features = compute_features(bars)

    classifier = KPCOFGSClassifier({})
    result = classifier.classify(features)

    assert "K" in result.columns
    assert "P" in result.columns
    assert "C" in result.columns
    assert "O" in result.columns
    assert "F" in result.columns
    assert "G" in result.columns
    assert "S" in result.columns


def test_kpcofgs_probability_outputs():
    """Test KPCOFGS classification produces probability distributions (Phase C)."""
    import numpy as np

    prints_df = generate_stub_prints(["BTCUSDT"], n_days=2, trades_per_day=500, seed=42)

    config = {"domains": {"D0": {"n_trades": 100}}}
    agg = DomainAggregator(config)
    bars = agg.aggregate(prints_df, "D0")
    features = compute_features(bars)

    classifier = KPCOFGSClassifier({})
    result = classifier.classify(features)

    # Check label columns
    for level in ["K", "P", "C", "O", "F", "G", "S"]:
        assert f"{level}_label" in result.columns, f"Missing {level}_label column"
        assert f"{level}_pmax" in result.columns, f"Missing {level}_pmax column"
        assert f"{level}_entropy" in result.columns, f"Missing {level}_entropy column"

    # Check regime_entropy
    assert "regime_entropy" in result.columns, "Missing regime_entropy column"

    # Check pmax values are in [0, 1]
    for level in ["K", "P", "C", "O", "F", "G", "S"]:
        pmax = result[f"{level}_pmax"].values
        assert np.all(pmax >= 0) and np.all(pmax <= 1), f"{level}_pmax not in [0, 1]"

    # Check entropy values are non-negative
    for level in ["K", "P", "C", "O", "F", "G", "S"]:
        entropy = result[f"{level}_entropy"].values
        assert np.all(entropy >= 0), f"{level}_entropy contains negative values"

    # Check S probabilities exist and sum to 1
    s_prob_cols = [c for c in result.columns if c.startswith("S_prob_")]
    assert len(s_prob_cols) > 0, "Missing S probability columns"
    s_probs = result[s_prob_cols].values
    row_sums = s_probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), "S probabilities don't sum to 1"
