"""Unit tests for waterfall backtest helper logic."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    target = repo_root / "yoshi-bot/scripts/waterfall_backtest.py"
    spec = importlib.util.spec_from_file_location("waterfall_backtest", str(target))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_timeframe_to_minutes():
    mod = _load_module()
    assert mod.timeframe_to_minutes("1m") == 1
    assert mod.timeframe_to_minutes("5m") == 5
    assert mod.timeframe_to_minutes("1h") == 60
    assert mod.timeframe_to_minutes("4h") == 240
    assert mod.timeframe_to_minutes("1d") == 1440


def test_bars_to_days_scales_with_timeframe():
    mod = _load_module()
    d_1m = mod.bars_to_days(2000, "1m")
    d_1h = mod.bars_to_days(2000, "1h")
    assert d_1h > d_1m
    assert d_1m >= 3
