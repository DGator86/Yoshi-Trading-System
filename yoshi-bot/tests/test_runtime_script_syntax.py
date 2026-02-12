"""Compile critical runtime scripts to catch syntax regressions in CI."""

from __future__ import annotations

import py_compile
from pathlib import Path


def test_runtime_scripts_compile():
    repo_root = Path(__file__).resolve().parents[2]
    targets = [
        repo_root / "yoshi-bot/scripts/kalshi_scanner.py",
        repo_root / "yoshi-bot/scripts/bootstrap_signal_learning.py",
        repo_root / "yoshi-bot/scripts/waterfall_backtest.py",
        repo_root / "yoshi-bot/scripts/waterfall_autoresume.py",
        repo_root / "clawdbot/scripts/yoshi-bridge.py",
        repo_root / "clawdbot/scripts/kalshi-edge-scanner.py",
        repo_root / "clawdbot/scripts/forecaster/engine.py",
        repo_root / "clawdbot/scripts/forecaster/regime_gate.py",
        repo_root / "clawdbot/scripts/forecaster/diagnose.py",
        repo_root / "shared/trading_signals.py",
    ]
    for path in targets:
        py_compile.compile(str(path), doraise=True)

