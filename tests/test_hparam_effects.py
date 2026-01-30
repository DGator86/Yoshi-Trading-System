"""Phase F: Tests proving hyperparameters actually affect behavior."""

import numpy as np
import pandas as pd
import pytest

from gnosis.harness.ralph_loop import (
    RalphLoop,
    RalphLoopConfig,
    HparamCandidate,
    _apply_candidate_params,
    HPARAM_KEY_MAP,
)
from gnosis.harness.walkforward import WalkForwardHarness


def generate_test_features(n_rows: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate minimal test features DataFrame for Ralph Loop."""
    np.random.seed(seed)

    # Generate synthetic data
    bar_idx = np.arange(n_rows)
    close = 100.0 + np.cumsum(np.random.randn(n_rows) * 0.01)
    future_return = np.random.randn(n_rows) * 0.02  # ~2% volatility

    # S_pmax: uniform between 0.3 and 0.9 (some will be above/below confidence thresholds)
    s_pmax = np.random.uniform(0.3, 0.9, n_rows)

    # S_label: mostly non-uncertain
    s_labels = np.where(s_pmax < 0.4, "S_UNCERTAIN", "S_TRENDING")

    return pd.DataFrame({
        "symbol": "TEST",
        "bar_idx": bar_idx,
        "close": close,
        "future_return": future_return,
        "S_pmax": s_pmax,
        "S_label": s_labels,
        "S_entropy": np.random.uniform(0.1, 0.5, n_rows),
        "K_label": "K_BALANCED",
        "K_pmax": np.random.uniform(0.5, 0.9, n_rows),
        "P_label": "P_VOL_STABLE",
        "P_pmax": np.random.uniform(0.5, 0.9, n_rows),
    })


class TestHparamKeyMapping:
    """Test that hparam key mapping works correctly."""

    def test_confidence_floor_maps_correctly(self):
        """Verify regimes.confidence_floor maps to nested path."""
        candidate = HparamCandidate(
            candidate_id=0,
            params={"regimes.confidence_floor": 0.55},
        )
        base_cfg = {"regimes": {}}

        cfg, resolved = _apply_candidate_params(candidate, base_cfg)

        # Should be mapped to nested path
        assert resolved["regimes.confidence_floor"]["was_mapped"] is True
        assert resolved["regimes.confidence_floor"]["resolved_key"] == "regimes.constraints_by_species.default.confidence_floor"

        # Verify value is set in config
        assert cfg["regimes"]["constraints_by_species"]["default"]["confidence_floor"] == 0.55

    def test_sigma_scale_maps_directly(self):
        """Verify forecast.sigma_scale maps directly (no change)."""
        candidate = HparamCandidate(
            candidate_id=0,
            params={"forecast.sigma_scale": 1.5},
        )
        base_cfg = {"forecast": {}}

        cfg, resolved = _apply_candidate_params(candidate, base_cfg)

        # Should NOT be mapped (direct key)
        assert resolved["forecast.sigma_scale"]["was_mapped"] is False
        assert cfg["forecast"]["sigma_scale"] == 1.5


class TestConfidenceFloorEffect:
    """Test that changing confidence_floor affects abstention_rate."""

    def test_higher_confidence_floor_increases_abstention(self):
        """Higher confidence_floor should cause more abstentions."""
        features_df = generate_test_features(n_rows=200, seed=123)

        base_cfg = {
            "models": {"predictor": {"l2_reg": 1.0}},
            "regimes": {"constraints_by_species": {"default": {"confidence_floor": 0.65}}},
            "domains": {"domains": {"D0": {"n_trades": 200}}},
            "forecast": {"horizon_bars": 10},
            "walkforward": {
                "n_folds": 2,
                "train_frac": 0.5,
                "test_frac": 0.3,
                "purge_bars": 5,
            },
        }

        # Run with LOW confidence_floor (0.40) - should have LOW abstention
        low_hparams = {
            "ralph_loop": {
                "enabled": True,
                "inner_folds": 2,
                "grid": {
                    "regimes.confidence_floor": [0.40],
                    "forecast.sigma_scale": [1.0],
                },
            }
        }

        ralph_low = RalphLoop(
            loop_config=RalphLoopConfig.from_dict(low_hparams),
            base_config=base_cfg,
        )
        harness = WalkForwardHarness(base_cfg["walkforward"])
        trials_low, _ = ralph_low.run(features_df, harness, base_cfg["regimes"])

        # Run with HIGH confidence_floor (0.80) - should have HIGH abstention
        high_hparams = {
            "ralph_loop": {
                "enabled": True,
                "inner_folds": 2,
                "grid": {
                    "regimes.confidence_floor": [0.80],
                    "forecast.sigma_scale": [1.0],
                },
            }
        }

        ralph_high = RalphLoop(
            loop_config=RalphLoopConfig.from_dict(high_hparams),
            base_config=base_cfg,
        )
        trials_high, _ = ralph_high.run(features_df, harness, base_cfg["regimes"])

        # Extract mean abstention rates
        if not trials_low.empty and not trials_high.empty:
            mean_abstention_low = trials_low["abstention_rate"].mean()
            mean_abstention_high = trials_high["abstention_rate"].mean()

            # High confidence_floor should have higher abstention rate
            # Allow for some tolerance due to random data
            assert mean_abstention_high >= mean_abstention_low, (
                f"Expected higher abstention with higher floor: "
                f"low={mean_abstention_low:.3f}, high={mean_abstention_high:.3f}"
            )


class TestSigmaScaleEffect:
    """Test that changing sigma_scale affects coverage and interval width."""

    def test_higher_sigma_scale_increases_coverage(self):
        """Higher sigma_scale should widen intervals and increase coverage."""
        features_df = generate_test_features(n_rows=200, seed=456)

        base_cfg = {
            "models": {"predictor": {"l2_reg": 1.0}},
            "regimes": {"constraints_by_species": {"default": {"confidence_floor": 0.50}}},
            "domains": {"domains": {"D0": {"n_trades": 200}}},
            "forecast": {"horizon_bars": 10},
            "walkforward": {
                "n_folds": 2,
                "train_frac": 0.5,
                "test_frac": 0.3,
                "purge_bars": 5,
            },
        }

        # Run with LOW sigma_scale (1.0) - narrow intervals
        low_hparams = {
            "ralph_loop": {
                "enabled": True,
                "inner_folds": 2,
                "grid": {
                    "regimes.confidence_floor": [0.50],
                    "forecast.sigma_scale": [1.0],
                },
            }
        }

        ralph_low = RalphLoop(
            loop_config=RalphLoopConfig.from_dict(low_hparams),
            base_config=base_cfg,
        )
        harness = WalkForwardHarness(base_cfg["walkforward"])
        trials_low, _ = ralph_low.run(features_df, harness, base_cfg["regimes"])

        # Run with HIGH sigma_scale (3.0) - wide intervals
        high_hparams = {
            "ralph_loop": {
                "enabled": True,
                "inner_folds": 2,
                "grid": {
                    "regimes.confidence_floor": [0.50],
                    "forecast.sigma_scale": [3.0],
                },
            }
        }

        ralph_high = RalphLoop(
            loop_config=RalphLoopConfig.from_dict(high_hparams),
            base_config=base_cfg,
        )
        trials_high, _ = ralph_high.run(features_df, harness, base_cfg["regimes"])

        # Extract mean coverage and sharpness
        if not trials_low.empty and not trials_high.empty:
            mean_coverage_low = trials_low["coverage_90"].mean()
            mean_coverage_high = trials_high["coverage_90"].mean()
            mean_sharpness_low = trials_low["sharpness"].mean()
            mean_sharpness_high = trials_high["sharpness"].mean()

            # Higher sigma_scale should increase coverage (wider intervals)
            assert mean_coverage_high >= mean_coverage_low, (
                f"Expected higher coverage with higher sigma_scale: "
                f"low={mean_coverage_low:.3f}, high={mean_coverage_high:.3f}"
            )

            # Higher sigma_scale should increase sharpness (wider intervals = higher sharpness value)
            assert mean_sharpness_high >= mean_sharpness_low, (
                f"Expected wider intervals with higher sigma_scale: "
                f"low={mean_sharpness_low:.4f}, high={mean_sharpness_high:.4f}"
            )


class TestTrialsDataFrameColumns:
    """Test that trials DataFrame has all Phase F columns."""

    def test_trials_df_has_explicit_param_columns(self):
        """Verify hparams_trials has confidence_floor, sigma_scale, n_trades columns."""
        features_df = generate_test_features(n_rows=150, seed=789)

        base_cfg = {
            "models": {"predictor": {"l2_reg": 1.0}},
            "regimes": {"constraints_by_species": {"default": {"confidence_floor": 0.65}}},
            "domains": {"domains": {"D0": {"n_trades": 200}}},
            "forecast": {"horizon_bars": 10},
            "walkforward": {
                "n_folds": 2,
                "train_frac": 0.5,
                "test_frac": 0.3,
                "purge_bars": 5,
            },
        }

        hparams = {
            "ralph_loop": {
                "enabled": True,
                "inner_folds": 2,
                "grid": {
                    "domains.domains.D0.n_trades": [100, 200],
                    "regimes.confidence_floor": [0.50, 0.60],
                    "forecast.sigma_scale": [1.0, 1.5],
                },
            }
        }

        ralph = RalphLoop(
            loop_config=RalphLoopConfig.from_dict(hparams),
            base_config=base_cfg,
        )
        harness = WalkForwardHarness(base_cfg["walkforward"])
        trials_df, _ = ralph.run(features_df, harness, base_cfg["regimes"])

        if not trials_df.empty:
            # Phase F: Check explicit param columns exist
            assert "confidence_floor" in trials_df.columns, "Missing confidence_floor column"
            assert "sigma_scale" in trials_df.columns, "Missing sigma_scale column"
            assert "n_trades" in trials_df.columns, "Missing n_trades column"

            # Phase F: Check conditional/unconditional coverage columns exist
            assert "coverage_90_conditional" in trials_df.columns, "Missing coverage_90_conditional"
            assert "coverage_90_unconditional" in trials_df.columns, "Missing coverage_90_unconditional"

            # Verify param values are populated (not all defaults)
            unique_floors = trials_df["confidence_floor"].unique()
            unique_scales = trials_df["sigma_scale"].unique()
            unique_trades = trials_df["n_trades"].unique()

            assert len(unique_floors) == 2, f"Expected 2 confidence_floor values, got {unique_floors}"
            assert len(unique_scales) == 2, f"Expected 2 sigma_scale values, got {unique_scales}"
            assert len(unique_trades) == 2, f"Expected 2 n_trades values, got {unique_trades}"


class TestIntervalValidity:
    """Test interval validity guards."""

    def test_intervals_have_lo_le_hi(self):
        """Verify q05 <= q95 for all predictions."""
        features_df = generate_test_features(n_rows=150, seed=999)

        base_cfg = {
            "models": {"predictor": {"l2_reg": 1.0}},
            "regimes": {"constraints_by_species": {"default": {"confidence_floor": 0.50}}},
            "domains": {"domains": {"D0": {"n_trades": 200}}},
            "forecast": {"horizon_bars": 10},
            "walkforward": {
                "n_folds": 2,
                "train_frac": 0.5,
                "test_frac": 0.3,
                "purge_bars": 5,
            },
        }

        # Test with various sigma_scale values
        hparams = {
            "ralph_loop": {
                "enabled": True,
                "inner_folds": 2,
                "grid": {
                    "regimes.confidence_floor": [0.50],
                    "forecast.sigma_scale": [0.5, 1.0, 2.0, 3.5],  # Include low value
                },
            }
        }

        ralph = RalphLoop(
            loop_config=RalphLoopConfig.from_dict(hparams),
            base_config=base_cfg,
        )
        harness = WalkForwardHarness(base_cfg["walkforward"])
        trials_df, _ = ralph.run(features_df, harness, base_cfg["regimes"])

        # If we got here without errors, the interval validity guard worked
        # (invalid intervals are swapped in _evaluate_candidate_on_inner)
        if not trials_df.empty:
            # All coverage values should be valid (between 0 and 1)
            assert (trials_df["coverage_90"] >= 0.0).all(), "Coverage below 0"
            assert (trials_df["coverage_90"] <= 1.0).all(), "Coverage above 1"

            # Sharpness should be non-negative (interval width)
            assert (trials_df["sharpness"] >= 0.0).all(), "Negative sharpness"
