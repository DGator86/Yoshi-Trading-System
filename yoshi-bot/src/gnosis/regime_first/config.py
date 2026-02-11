"""Config loading for regime-first crypto walk-forward/backtest."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class RegimeFirstConfig:
    raw: dict[str, Any]

    @property
    def system(self) -> dict[str, Any]:
        return self.raw.get("system", {}) or {}

    @property
    def regimes(self) -> dict[str, Any]:
        return self.raw.get("regimes", {}) or {}

    @property
    def playbooks(self) -> list[dict[str, Any]]:
        return self.raw.get("playbooks", []) or []

    @property
    def blending(self) -> dict[str, Any]:
        return self.raw.get("blending", {}) or {}

    @property
    def risk(self) -> dict[str, Any]:
        return self.raw.get("risk", {}) or {}

    @property
    def execution(self) -> dict[str, Any]:
        return self.raw.get("execution", {}) or {}

    @property
    def walkforward(self) -> dict[str, Any]:
        return self.raw.get("walkforward", {}) or {}


def load_regime_first_config(path: str | Path) -> RegimeFirstConfig:
    p = Path(path)
    with open(p, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return RegimeFirstConfig(raw=raw)

