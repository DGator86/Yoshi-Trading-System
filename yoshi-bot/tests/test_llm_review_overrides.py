"""Tests for LLM adaptive override sanitization."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gnosis.review import sanitize_experiment_overrides  # noqa: E402


def test_sanitize_experiment_overrides_clamps_and_whitelists():
    raw = {
        "models": {
            "predictor": {
                "sigma_scale": 99,  # clamp to 3.0
                "backend": "quantile",
                "l2_reg": "0.00001",  # clamp to 1e-4
                "drop_table": True,  # should be dropped
            }
        },
        "regimes": {"confidence_floor_scale": 0.01},  # clamp to 0.5
        "some_other": {"key": 1},
    }
    out = sanitize_experiment_overrides(raw)
    assert out["models"]["predictor"]["sigma_scale"] == 3.0
    assert out["models"]["predictor"]["backend"] == "quantile"
    assert out["models"]["predictor"]["l2_reg"] == 1e-4
    assert "drop_table" not in out["models"]["predictor"]
    assert out["regimes"]["confidence_floor_scale"] == 0.5
    assert "some_other" not in out


def test_sanitize_experiment_overrides_accepts_dot_paths():
    raw = {
        "models.predictor.sigma_scale": 0.1,  # clamp to 0.5
        "regimes.confidence_floor_scale": 2.0,  # clamp to 1.5
        "models.predictor.backend": "invalid",  # drop
    }
    out = sanitize_experiment_overrides(raw)
    assert out["models"]["predictor"]["sigma_scale"] == 0.5
    assert out["regimes"]["confidence_floor_scale"] == 1.5
    assert "backend" not in out["models"]["predictor"]

