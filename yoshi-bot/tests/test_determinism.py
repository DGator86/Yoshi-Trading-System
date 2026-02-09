"""Tests for experiment determinism and reproducibility."""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from run_experiment import load_config, run_experiment


def test_experiment_determinism(tmp_path):
    """Test that running the experiment twice with same config produces identical report_hash."""
    # Create two separate output directories
    out_dir_1 = tmp_path / "run1"
    out_dir_2 = tmp_path / "run2"

    # Load the default config
    config_path = Path(__file__).parent.parent / "configs" / "experiment.yaml"
    config = load_config(str(config_path))

    # Run experiment twice with different output directories
    config_1 = config.copy()
    config_1["artifacts"] = {"out_dir": str(out_dir_1)}
    report_1 = run_experiment(config_1, config_path=str(config_path))

    config_2 = config.copy()
    config_2["artifacts"] = {"out_dir": str(out_dir_2)}
    report_2 = run_experiment(config_2, config_path=str(config_path))

    # Assert report_hash matches
    assert "report_hash" in report_1, "report_hash missing from first run"
    assert "report_hash" in report_2, "report_hash missing from second run"
    assert report_1["report_hash"] == report_2["report_hash"], (
        f"report_hash mismatch: {report_1['report_hash']} != {report_2['report_hash']}"
    )

    # Also verify the report.json files on disk match
    with open(out_dir_1 / "report.json") as f:
        disk_report_1 = json.load(f)
    with open(out_dir_2 / "report.json") as f:
        disk_report_2 = json.load(f)

    assert disk_report_1["report_hash"] == disk_report_2["report_hash"]

    # Verify run_metadata.json has required fields
    with open(out_dir_1 / "run_metadata.json") as f:
        metadata = json.load(f)

    required_fields = [
        "git_commit",
        "config_hash",
        "data_manifest_hash",
        "report_hash",
        "python_version",
        "platform",
    ]
    for field in required_fields:
        assert field in metadata, f"run_metadata.json missing required field: {field}"
