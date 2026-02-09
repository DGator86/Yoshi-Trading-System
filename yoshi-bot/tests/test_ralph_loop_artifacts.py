"""Tests for Ralph Loop artifact generation and determinism."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from run_experiment import load_config, run_experiment


def test_ralph_loop_artifacts_exist(tmp_path):
    """Test that Ralph Loop produces hparams_trials.parquet and selected_hparams.json."""
    out_dir = tmp_path / "ralph_artifacts"

    # Load base config
    config_path = Path(__file__).parent.parent / "configs" / "experiment.yaml"
    config = load_config(str(config_path))
    config["artifacts"] = {"out_dir": str(out_dir)}

    # Minimal hparams config with tiny grid (2 combos) and inner_folds=1 for speed
    hparams_config = {
        "ralph": {
            "enabled": True,
            "target_coverage": 0.90,
            "weights": {
                "coverage": 4.0,
                "wis": 1.0,
                "is90": 0.5,
                "mae": 1.0,
                "abstention": 0.5,
            },
            "inner_folds": 1,
            "purge_bars": 5,
            "embargo_bars": 5,
            "grid": {
                "forecast.sigma_scale": [1.0, 1.15],
            },
        }
    }

    # Run experiment with hparams
    report = run_experiment(config, config_path=str(config_path), hparams_config=hparams_config)

    # Assert artifacts exist
    assert (out_dir / "hparams_trials.parquet").exists(), "hparams_trials.parquet not created"
    assert (out_dir / "selected_hparams.json").exists(), "selected_hparams.json not created"

    # Verify hparams_trials.parquet structure
    trials_df = pd.read_parquet(out_dir / "hparams_trials.parquet")
    assert "outer_fold" in trials_df.columns
    assert "candidate_id" in trials_df.columns
    assert "inner_fold" in trials_df.columns
    assert "composite_score" in trials_df.columns
    assert "params_json" in trials_df.columns

    # Verify selected_hparams.json structure
    with open(out_dir / "selected_hparams.json") as f:
        selected = json.load(f)
    assert "per_fold" in selected
    assert "global_best" in selected


def test_ralph_loop_determinism(tmp_path):
    """Test that running Ralph Loop twice yields identical selected_hparams.json."""
    out_dir_1 = tmp_path / "run1"
    out_dir_2 = tmp_path / "run2"

    config_path = Path(__file__).parent.parent / "configs" / "experiment.yaml"
    config = load_config(str(config_path))

    # Minimal hparams config
    hparams_config = {
        "ralph": {
            "enabled": True,
            "target_coverage": 0.90,
            "weights": {
                "coverage": 4.0,
                "wis": 1.0,
                "is90": 0.5,
                "mae": 1.0,
                "abstention": 0.5,
            },
            "inner_folds": 1,
            "purge_bars": 5,
            "embargo_bars": 5,
            "grid": {
                "forecast.sigma_scale": [1.0, 1.15],
            },
        }
    }

    # Run 1
    config_1 = config.copy()
    config_1["artifacts"] = {"out_dir": str(out_dir_1)}
    run_experiment(config_1, config_path=str(config_path), hparams_config=hparams_config)

    # Run 2
    config_2 = config.copy()
    config_2["artifacts"] = {"out_dir": str(out_dir_2)}
    run_experiment(config_2, config_path=str(config_path), hparams_config=hparams_config)

    # Load both selected_hparams.json
    with open(out_dir_1 / "selected_hparams.json") as f:
        selected_1 = json.load(f)
    with open(out_dir_2 / "selected_hparams.json") as f:
        selected_2 = json.load(f)

    # Compare (normalize to JSON strings for comparison)
    assert json.dumps(selected_1, sort_keys=True) == json.dumps(selected_2, sort_keys=True), (
        f"selected_hparams.json differs between runs:\n"
        f"Run 1: {json.dumps(selected_1, sort_keys=True)}\n"
        f"Run 2: {json.dumps(selected_2, sort_keys=True)}"
    )


def test_ralph_loop_report_includes_results(tmp_path):
    """Test that report.json includes Ralph Loop section when enabled."""
    out_dir = tmp_path / "ralph_report"

    config_path = Path(__file__).parent.parent / "configs" / "experiment.yaml"
    config = load_config(str(config_path))
    config["artifacts"] = {"out_dir": str(out_dir)}

    hparams_config = {
        "ralph": {
            "enabled": True,
            "target_coverage": 0.90,
            "weights": {"coverage": 4.0, "wis": 1.0, "is90": 0.5, "mae": 1.0, "abstention": 0.5},
            "inner_folds": 1,
            "purge_bars": 5,
            "embargo_bars": 5,
            "grid": {"forecast.sigma_scale": [1.0, 1.15]},
        }
    }

    report = run_experiment(config, config_path=str(config_path), hparams_config=hparams_config)

    # Verify Ralph Loop results in report
    assert "ralph_loop" in report
    assert report["ralph_loop"]["enabled"] is True
    assert "n_candidates" in report["ralph_loop"]
    assert "selected_params" in report["ralph_loop"]
    assert "robustness" in report["ralph_loop"]


def test_ralph_loop_phase_e_structural_hparams(tmp_path):
    """Phase E: Test structural hyperparameters (D0.n_trades) with caching."""
    out_dir = tmp_path / "phase_e_structural"

    config_path = Path(__file__).parent.parent / "configs" / "experiment.yaml"
    config = load_config(str(config_path))
    config["artifacts"] = {"out_dir": str(out_dir)}

    # Phase E: Grid with two D0.n_trades values and one abstain threshold
    hparams_config = {
        "ralph": {
            "enabled": True,
            "target_coverage": 0.90,
            "weights": {"coverage": 4.0, "wis": 1.0, "is90": 0.5, "mae": 1.0, "abstention": 0.5},
            "inner_folds": 1,
            "purge_bars": 5,
            "embargo_bars": 5,
            "grid": {
                "domains.domains.D0.n_trades": [100, 200],  # Structural param
                "regimes.confidence_floor": [0.55],  # Abstain threshold (mapped key)
            },
        }
    }

    report = run_experiment(config, config_path=str(config_path), hparams_config=hparams_config)

    # Verify artifacts exist
    assert (out_dir / "hparams_trials.parquet").exists()
    assert (out_dir / "selected_hparams.json").exists()

    # Verify trials_df not empty
    trials_df = pd.read_parquet(out_dir / "hparams_trials.parquet")
    assert len(trials_df) > 0
    assert "resolved_params_json" in trials_df.columns

    # Verify selected_hparams.json structure
    with open(out_dir / "selected_hparams.json") as f:
        selected = json.load(f)

    assert "per_fold" in selected
    assert "global_best" in selected
    assert "phase_e_info" in selected

    # Verify Phase E info
    phase_e_info = selected["phase_e_info"]
    assert "feature_cache_stats" in phase_e_info
    assert "key_mappings" in phase_e_info
    assert phase_e_info["has_structural_params"] is True


def test_ralph_loop_phase_e_determinism(tmp_path):
    """Phase E: Test determinism with structural hparams across two runs."""
    out_dir_1 = tmp_path / "run1"
    out_dir_2 = tmp_path / "run2"

    config_path = Path(__file__).parent.parent / "configs" / "experiment.yaml"
    config = load_config(str(config_path))

    # Phase E: Grid with structural params
    hparams_config = {
        "ralph": {
            "enabled": True,
            "target_coverage": 0.90,
            "weights": {"coverage": 4.0, "wis": 1.0, "is90": 0.5, "mae": 1.0, "abstention": 0.5},
            "inner_folds": 1,
            "purge_bars": 5,
            "embargo_bars": 5,
            "grid": {
                "domains.domains.D0.n_trades": [100, 200],
                "regimes.confidence_floor": [0.55],
            },
        }
    }

    # Run 1
    config_1 = config.copy()
    config_1["artifacts"] = {"out_dir": str(out_dir_1)}
    run_experiment(config_1, config_path=str(config_path), hparams_config=hparams_config)

    # Run 2
    config_2 = config.copy()
    config_2["artifacts"] = {"out_dir": str(out_dir_2)}
    run_experiment(config_2, config_path=str(config_path), hparams_config=hparams_config)

    # Load both selected_hparams.json
    with open(out_dir_1 / "selected_hparams.json") as f:
        selected_1 = json.load(f)
    with open(out_dir_2 / "selected_hparams.json") as f:
        selected_2 = json.load(f)

    # Compare per_fold and global_best (exclude phase_e_info cache stats which may differ)
    assert selected_1["per_fold"] == selected_2["per_fold"], (
        f"per_fold differs:\nRun 1: {selected_1['per_fold']}\nRun 2: {selected_2['per_fold']}"
    )
    assert selected_1["global_best"] == selected_2["global_best"], (
        f"global_best differs:\nRun 1: {selected_1['global_best']}\nRun 2: {selected_2['global_best']}"
    )


def test_ralph_loop_key_mapping(tmp_path):
    """Phase E: Test that key-path mapping works for non-existent keys."""
    out_dir = tmp_path / "key_mapping"

    config_path = Path(__file__).parent.parent / "configs" / "experiment.yaml"
    config = load_config(str(config_path))
    config["artifacts"] = {"out_dir": str(out_dir)}

    # Use keys that need mapping
    hparams_config = {
        "ralph": {
            "enabled": True,
            "target_coverage": 0.90,
            "weights": {"coverage": 4.0, "wis": 1.0, "is90": 0.5, "mae": 1.0, "abstention": 0.5},
            "inner_folds": 1,
            "purge_bars": 5,
            "embargo_bars": 5,
            "grid": {
                "regimes.confidence_floor": [0.45, 0.55],  # Mapped to constraints_by_species.default.confidence_floor
            },
        }
    }

    report = run_experiment(config, config_path=str(config_path), hparams_config=hparams_config)

    # Verify trials have resolved_params_json
    trials_df = pd.read_parquet(out_dir / "hparams_trials.parquet")
    assert "resolved_params_json" in trials_df.columns
    assert len(trials_df) > 0

    # Check that mapping is recorded
    first_resolved = json.loads(trials_df.iloc[0]["resolved_params_json"])
    assert "regimes.confidence_floor" in first_resolved
    assert first_resolved["regimes.confidence_floor"]["was_mapped"] is True
    assert first_resolved["regimes.confidence_floor"]["resolved_key"] == "regimes.constraints_by_species.default.confidence_floor"
