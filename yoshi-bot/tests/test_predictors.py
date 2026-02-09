"""Tests for prediction models."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from gnosis.ingest import generate_stub_prints
from gnosis.domains import DomainAggregator, compute_features
from gnosis.regimes import KPCOFGSClassifier
from gnosis.particle import ParticleState
from gnosis.predictors import QuantilePredictor, BaselinePredictor
from gnosis.harness import compute_future_returns


def test_quantile_predictor():
    """Test quantile predictor."""
    prints_df = generate_stub_prints(["BTCUSDT"], n_days=10, trades_per_day=1000, seed=42)

    config = {"domains": {"D0": {"n_trades": 100}}}
    agg = DomainAggregator(config)
    bars = agg.aggregate(prints_df, "D0")
    features = compute_features(bars)

    classifier = KPCOFGSClassifier({})
    features = classifier.classify(features)

    particle = ParticleState({})
    features = particle.compute_state(features)

    features = compute_future_returns(features, horizon_bars=5)

    # Split train/test - use first 50 bars for train, next 20 for test
    train = features.iloc[:50]
    test = features.iloc[50:70]

    predictor = QuantilePredictor({"predictor": {"quantiles": [0.05, 0.50, 0.95]}})
    predictor.fit(train, "future_return")
    preds = predictor.predict(test)

    assert "q05" in preds.columns
    assert "q50" in preds.columns
    assert "q95" in preds.columns
    assert "x_hat" in preds.columns
    assert "sigma_hat" in preds.columns


def test_baseline_predictor():
    """Test baseline predictor."""
    prints_df = generate_stub_prints(["BTCUSDT"], n_days=2, trades_per_day=500, seed=42)

    config = {"domains": {"D0": {"n_trades": 100}}}
    agg = DomainAggregator(config)
    bars = agg.aggregate(prints_df, "D0")
    features = compute_features(bars)

    baseline = BaselinePredictor()
    preds = baseline.predict(features)

    assert "q05" in preds.columns
    assert "q50" in preds.columns
    assert "q95" in preds.columns
    # Baseline point estimate should be 0 (random walk)
    assert np.allclose(preds["x_hat"], 0.0)
