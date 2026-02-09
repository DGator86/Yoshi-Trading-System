"""Configuration loading and hashing utilities for experiments."""
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Dict, Any

import yaml


def get_git_commit() -> str:
    """Get current git commit hash, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return "unknown"


def compute_config_hash(config_dir: Path) -> str:
    """Compute SHA256 hash of concatenated config files in sorted order.

    Args:
        config_dir: Directory containing YAML config files.

    Returns:
        SHA256 hex digest of all config files combined.
    """
    config_files = sorted(config_dir.glob("*.yaml"))
    hasher = hashlib.sha256()
    for config_file in config_files:
        hasher.update(config_file.read_bytes())
    return hasher.hexdigest()


def compute_data_manifest_hash(manifest_path: Path) -> str:
    """Compute SHA256 hash of data manifest file.

    Args:
        manifest_path: Path to the data manifest JSON file.

    Returns:
        SHA256 hex digest or 'no_manifest' if file doesn't exist.
    """
    if manifest_path.exists():
        return hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return "no_manifest"


def compute_report_hash(report: Dict[str, Any]) -> str:
    """Compute SHA256 hash of report with volatile keys removed.

    Volatile keys (run_id, started_at, completed_at) are excluded to ensure
    deterministic hashing of the same logical report.

    Args:
        report: Report dictionary to hash.

    Returns:
        SHA256 hex digest of the stable report content.
    """
    # Create copy without volatile keys
    volatile_keys = ("run_id", "started_at", "completed_at")
    stable_report = {
        k: v
        for k, v in report.items()
        if k not in volatile_keys
    }
    # Use sorted keys and consistent JSON formatting for determinism
    report_json = json.dumps(stable_report, sort_keys=True, default=str)
    return hashlib.sha256(report_json.encode()).hexdigest()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration from YAML with auxiliary configs.

    Loads the main experiment config and merges in auxiliary configs
    (domains, models, regimes, costs) from the same directory.

    Args:
        config_path: Path to the main experiment config YAML.

    Returns:
        Merged configuration dictionary.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load auxiliary configs from same directory
    config_dir = Path(config_path).parent
    auxiliary_configs = ["domains", "models", "regimes", "costs"]

    for name in auxiliary_configs:
        aux_path = config_dir / f"{name}.yaml"
        if aux_path.exists():
            with open(aux_path) as f:
                config[name] = yaml.safe_load(f)

    return config
