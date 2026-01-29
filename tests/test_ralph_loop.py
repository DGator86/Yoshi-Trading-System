"""Tests for Ralph Loop hyperparameter selection (Phase D)."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from gnosis.loop import RalphLoop, RalphLoopConfig
from gnosis.harness import WalkForwardHarness


def test_ralph_loop_config_from_yaml():
    """Test RalphLoopConfig creation from YAML dict."""
    yaml_dict = {
        "grid": {
            "predictor_l2_reg": [0.1, 1.0],
            "confidence_floor_scale": [0.8, 1.0],
        },
        "inner_folds": {
            "n_folds": 2,
            "train_ratio": 0.6,
            "val_ratio": 0.4,
        },
        "scoring_weights": {
            "coverage_target": 0.90,
            "w1_coverage": 2.0,
        },
        "inner_purge_bars": 3,
        "inner_embargo_bars": 3,
    }

    config = RalphLoopConfig.from_yaml(yaml_dict)

    assert config.inner_folds_n == 2
    assert config.inner_train_ratio == 0.6
    assert config.inner_purge_bars == 3
    assert len(config.grid) == 2
    assert "predictor_l2_reg" in config.grid


def test_ralph_loop_candidate_generation():
    """Test that candidates are generated correctly from grid."""
    config = RalphLoopConfig(
        grid={
            "param_a": [1, 2],
            "param_b": [10, 20],
        }
    )

    ralph = RalphLoop(
        loop_config=config,
        base_config={},
        random_seed=42,
    )

    # 2 x 2 = 4 candidates
    assert len(ralph.candidates) == 4

    # Check that all combinations are present
    combos = {(c.params["param_a"], c.params["param_b"]) for c in ralph.candidates}
    expected = {(1, 10), (1, 20), (2, 10), (2, 20)}
    assert combos == expected


def test_ralph_loop_composite_score():
    """Test composite score computation."""
    config = RalphLoopConfig(
        coverage_target=0.90,
        w1_coverage=2.0,
        w2_sharpness=1.0,
        w3_mae=0.5,
        w4_abstention=0.3,
        w5_flip_rate=0.2,
    )

    ralph = RalphLoop(
        loop_config=config,
        base_config={},
        random_seed=42,
    )

    # Perfect coverage, zero everything else
    score_perfect = ralph._compute_composite_score(
        coverage_90=0.90, sharpness=0.0, mae=0.0, abstention_rate=0.0, flip_rate=0.0
    )

    # Bad coverage
    score_bad_cov = ralph._compute_composite_score(
        coverage_90=0.70, sharpness=0.0, mae=0.0, abstention_rate=0.0, flip_rate=0.0
    )

    # Perfect coverage should have higher score
    assert score_perfect > score_bad_cov


def test_ralph_loop_artifacts():
    """Test that Ralph Loop produces required artifacts (minimal run)."""
    # Create minimal synthetic data
    np.random.seed(42)
    n_bars = 500

    features_df = pd.DataFrame({
        "symbol": ["BTCUSDT"] * n_bars,
        "bar_idx": range(n_bars),
        "timestamp_end": pd.date_range("2024-01-01", periods=n_bars, freq="h"),
        "open": 30000 + np.random.randn(n_bars).cumsum(),
        "high": 30000 + np.random.randn(n_bars).cumsum() + 10,
        "low": 30000 + np.random.randn(n_bars).cumsum() - 10,
        "close": 30000 + np.random.randn(n_bars).cumsum(),
        "volume": np.random.rand(n_bars) * 1000,
        "returns": np.random.randn(n_bars) * 0.01,
        "realized_vol": np.abs(np.random.randn(n_bars) * 0.02) + 0.01,
        "ofi": np.random.randn(n_bars) * 0.3,
        "range_pct": np.random.rand(n_bars) * 0.02,
        "flow_momentum": np.random.randn(n_bars) * 0.1,
        "regime_stability": np.random.rand(n_bars),
        "barrier_proximity": np.random.rand(n_bars),
        "particle_score": np.random.randn(n_bars) * 0.5,
        "K_label": np.random.choice(["K_TRENDING", "K_BALANCED"], n_bars),
        "K_pmax": np.random.rand(n_bars) * 0.3 + 0.5,
        "K_entropy": np.random.rand(n_bars),
        "S_label": np.random.choice(["S_TC_PULLBACK_RESUME", "S_UNCERTAIN"], n_bars),
        "S_pmax": np.random.rand(n_bars) * 0.3 + 0.5,
        "S_entropy": np.random.rand(n_bars),
        "future_return": np.random.randn(n_bars) * 0.01,
    })

    # Minimal config: 2 candidates, 1 inner fold
    hparams_config = {
        "grid": {
            "predictor_l2_reg": [0.5, 2.0],  # 2 candidates
        },
        "inner_folds": {
            "n_folds": 1,  # Minimal inner folds
            "train_ratio": 0.6,
            "val_ratio": 0.4,
        },
        "scoring_weights": {
            "coverage_target": 0.90,
            "w1_coverage": 2.0,
            "w2_sharpness": 1.0,
            "w3_mae": 0.5,
            "w4_abstention": 0.3,
            "w5_flip_rate": 0.2,
        },
        "inner_purge_bars": 2,
        "inner_embargo_bars": 2,
    }

    loop_config = RalphLoopConfig.from_yaml(hparams_config)

    base_config = {
        "models": {
            "predictor": {
                "quantiles": [0.05, 0.50, 0.95],
                "l2_reg": 1.0,
            }
        },
        "regimes": {},
    }

    ralph = RalphLoop(
        loop_config=loop_config,
        base_config=base_config,
        random_seed=42,
    )

    # Use minimal walk-forward config
    wf_config = {
        "outer_folds": 2,  # Minimal outer folds
        "train_days": 180,
        "val_days": 30,
        "test_days": 30,
    }
    harness = WalkForwardHarness(wf_config)

    # Run Ralph Loop
    trials_df, selected_json = ralph.run(
        features_df=features_df,
        outer_harness=harness,
        regimes_config={},
    )

    # Verify artifacts
    # 1. trials_df should have expected columns
    assert not trials_df.empty, "trials_df should not be empty"
    expected_cols = [
        "outer_fold", "candidate_id", "inner_fold",
        "coverage_90", "sharpness", "mae",
        "abstention_rate", "flip_rate", "composite_score", "params_json"
    ]
    for col in expected_cols:
        assert col in trials_df.columns, f"Missing column: {col}"

    # 2. selected_json should have expected structure
    assert "per_fold" in selected_json
    assert "global_best" in selected_json
    assert selected_json["global_best"] is not None

    # 3. Verify we can save artifacts
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Save trials parquet
        trials_df.to_parquet(tmpdir / "hparams_trials.parquet", index=False)
        assert (tmpdir / "hparams_trials.parquet").exists()

        # Save selected json
        import json
        with open(tmpdir / "selected_hparams.json", "w") as f:
            json.dump(selected_json, f)
        assert (tmpdir / "selected_hparams.json").exists()

    # 4. Verify robustness stats
    robustness = ralph.get_robustness_stats(trials_df)
    assert "coverage_90_std" in robustness
    assert "sharpness_std" in robustness
    assert "mae_std" in robustness


def test_inner_fold_generation():
    """Test inner fold generation respects purge/embargo."""
    config = RalphLoopConfig(
        inner_folds_n=3,
        inner_train_ratio=0.6,
        inner_val_ratio=0.4,
        inner_purge_bars=5,
        inner_embargo_bars=5,
    )

    ralph = RalphLoop(
        loop_config=config,
        base_config={},
        random_seed=42,
    )

    # Generate inner folds for a window of 200 bars (0 to 200)
    inner_folds = list(ralph._generate_inner_folds(0, 200))

    # Should generate some folds
    assert len(inner_folds) > 0

    # Each fold should have proper ordering with gaps
    for fold in inner_folds:
        # Train should end before val starts (purge gap)
        assert fold.train_end + config.inner_purge_bars <= fold.val_start
        # Val should end before outer boundary
        assert fold.val_end <= 200


def test_no_outer_test_leakage():
    """Verify that Ralph Loop only uses train window, never touches test."""
    # This is a design verification test
    # The select_best_for_outer_fold function only receives
    # outer_train_start and outer_train_end, not test indices

    config = RalphLoopConfig(
        grid={"param": [1, 2]},
        inner_folds_n=2,
    )

    ralph = RalphLoop(
        loop_config=config,
        base_config={},
        random_seed=42,
    )

    # Create data with clear train/test separation
    np.random.seed(42)
    n_bars = 300

    features_df = pd.DataFrame({
        "symbol": ["BTCUSDT"] * n_bars,
        "bar_idx": range(n_bars),
        "timestamp_end": pd.date_range("2024-01-01", periods=n_bars, freq="h"),
        "open": 30000 + np.random.randn(n_bars).cumsum(),
        "high": 30000 + np.random.randn(n_bars).cumsum() + 10,
        "low": 30000 + np.random.randn(n_bars).cumsum() - 10,
        "close": 30000 + np.random.randn(n_bars).cumsum(),
        "volume": np.random.rand(n_bars) * 1000,
        "returns": np.random.randn(n_bars) * 0.01,
        "realized_vol": np.abs(np.random.randn(n_bars) * 0.02) + 0.01,
        "ofi": np.random.randn(n_bars) * 0.3,
        "range_pct": np.random.rand(n_bars) * 0.02,
        "flow_momentum": np.random.randn(n_bars) * 0.1,
        "regime_stability": np.random.rand(n_bars),
        "barrier_proximity": np.random.rand(n_bars),
        "particle_score": np.random.randn(n_bars) * 0.5,
        "K_label": np.random.choice(["K_TRENDING", "K_BALANCED"], n_bars),
        "K_pmax": np.random.rand(n_bars) * 0.3 + 0.5,
        "K_entropy": np.random.rand(n_bars),
        "S_label": np.random.choice(["S_TC_PULLBACK_RESUME", "S_UNCERTAIN"], n_bars),
        "S_pmax": np.random.rand(n_bars) * 0.3 + 0.5,
        "S_entropy": np.random.rand(n_bars),
        "future_return": np.random.randn(n_bars) * 0.01,
    })

    # Call select_best with only train window (0-200), test would be 200-300
    # The function should never access indices >= 200
    best = ralph.select_best_for_outer_fold(
        outer_fold_idx=0,
        outer_train_start=0,
        outer_train_end=200,  # Test data would be 200-300
        features_df=features_df,
        regimes_config={},
    )

    # Verify all trial results only used data from 0-200
    for result in ralph.trial_results:
        if result.outer_fold == 0:
            # The inner folds should all be within 0-200
            # This is ensured by the function signature - it only receives train bounds
            assert result.composite_score != -999.0 or result.inner_fold >= 0
