"""Tests for waterfall_autoresume helper behavior."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    target = repo_root / "yoshi-bot/scripts/waterfall_autoresume.py"
    spec = importlib.util.spec_from_file_location("waterfall_autoresume", str(target))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_normalize_timeframes_default_and_parse():
    mod = _load_module()
    assert mod.normalize_timeframes("") == ["1h", "30m", "15m", "5m", "1m"]
    assert mod.normalize_timeframes("1h,30m, 5m") == ["1h", "30m", "5m"]


def test_build_run_args_passes_seed_and_allow_flag():
    mod = _load_module()
    args = SimpleNamespace(
        symbol="BTCUSDT",
        n_bars=2000,
        snapshot_lookback_bars=600,
        sigma_target=1.0,
        max_attempts_per_bar=400,
        allow_unconverged_progress=True,
        base_mc_iterations=8000,
        base_mc_steps=48,
        adaptive_sigma_calibration=True,
        sigma_calibration_rate=0.85,
        sigma_scale_max=8.0,
        ralph_explore_rate=0.15,
        seed=42,
        log_every=50,
        ohlcv_providers="coinapi,binance_public",
        data_timeout_s=10,
        data_max_retries=1,
        output_dir="reports/waterfall_backtest",
    )
    run_args = mod._build_run_args(args, timeframe="30m", start_offset=306, run_seed=31415)
    assert run_args.timeframes == "30m"
    assert run_args.start_offset == 306
    assert run_args.seed == 31415
    assert run_args.allow_unconverged_progress is True


def test_load_state_initializes_skip_tracking(tmp_path):
    mod = _load_module()
    state_path = tmp_path / "state.json"
    state = mod._load_state(state_path, ["30m"])
    tf = state["timeframes"]["30m"]
    assert tf["skipped_count"] == 0
    assert tf["skipped_offsets"] == []
