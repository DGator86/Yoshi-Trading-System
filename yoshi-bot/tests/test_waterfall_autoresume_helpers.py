"""Tests for waterfall_autoresume helper behavior."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


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
