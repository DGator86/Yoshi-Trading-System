"""Tests for validation harness."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from gnosis.harness import (
    WalkForwardHarness,
    pinball_loss,
    coverage,
    sharpness,
    IsotonicCalibrator,
    compute_ece,
    compute_stability_metrics,
)


def test_walkforward_harness():
    """Test walk-forward fold generation."""
    df = pd.DataFrame({"x": range(1000)})

    config = {
        "outer_folds": 3,
        "train_days": 180,
        "val_days": 30,
        "test_days": 30,
    }
    harness = WalkForwardHarness(config)
    folds = list(harness.generate_folds(df))

    assert len(folds) > 0
    for fold in folds:
        assert fold.train_end <= fold.val_start
        assert fold.val_end <= fold.test_start


def test_pinball_loss():
    """Test pinball loss computation."""
    y_true = np.array([0.1, 0.2, 0.3])
    y_pred = np.array([0.15, 0.15, 0.35])

    loss = pinball_loss(y_true, y_pred, 0.5)
    assert isinstance(loss, float)
    assert loss >= 0


def test_coverage():
    """Test coverage computation."""
    y_true = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    q_low = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    q_high = np.array([0.2, 0.3, 0.4, 0.5, 0.6])

    cov = coverage(y_true, q_low, q_high)
    assert cov == 1.0  # All within intervals


def test_sharpness():
    """Test sharpness computation."""
    q_low = np.array([0.0, 0.1])
    q_high = np.array([0.2, 0.3])

    sharp = sharpness(q_low, q_high)
    assert sharp == 0.2  # Average width


def test_isotonic_calibrator():
    """Test isotonic calibrator (Phase C)."""
    # Create synthetic data
    probs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    outcomes = np.array([0, 0, 0, 1, 0, 1, 1, 1, 1])

    calibrator = IsotonicCalibrator(n_bins=5)
    calibrator.fit(probs, outcomes)

    # Test calibration
    test_probs = np.array([0.15, 0.55, 0.85])
    calibrated = calibrator.calibrate(test_probs)

    # Calibrated probs should be in [0, 1]
    assert np.all(calibrated >= 0) and np.all(calibrated <= 1)
    # Should have same length
    assert len(calibrated) == len(test_probs)


def test_compute_ece():
    """Test ECE computation (Phase C)."""
    probs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    outcomes = np.array([0, 0, 0, 1, 1, 1])

    result = compute_ece(probs, outcomes, n_bins=5)

    assert "ece" in result
    assert "n_bins" in result
    assert "bins" in result
    assert result["ece"] >= 0
    assert result["n_bins"] == 5
    assert isinstance(result["bins"], list)


def test_compute_stability_metrics():
    """Test stability metrics (Phase C)."""
    df = pd.DataFrame({
        "K_label": ["K_A", "K_A", "K_B", "K_B", "K_A"],
        "K_entropy": [0.5, 0.6, 0.7, 0.8, 0.5],
        "P_label": ["P_A", "P_A", "P_A", "P_A", "P_A"],
        "P_entropy": [0.3, 0.3, 0.3, 0.3, 0.3],
    })

    stability = compute_stability_metrics(df, levels=["K", "P"])

    assert "K_flip_rate" in stability
    assert "K_avg_entropy" in stability
    assert "P_flip_rate" in stability
    assert "P_avg_entropy" in stability
    assert "overall_flip_rate" in stability

    # K has 2 flips in 4 transitions = 0.5
    assert stability["K_flip_rate"] == 0.5
    # P has 0 flips = 0.0
    assert stability["P_flip_rate"] == 0.0
    # K avg entropy = mean([0.5, 0.6, 0.7, 0.8, 0.5]) = 0.62
    assert abs(stability["K_avg_entropy"] - 0.62) < 0.01
