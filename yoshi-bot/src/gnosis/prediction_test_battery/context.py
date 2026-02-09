from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class ForecastArtifact:
    name: str
    horizon: int
    predictions: pd.DataFrame
    features: Optional[pd.DataFrame] = None
    metadata: Dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        required = {"timestamp", "y_true", "y_pred"}
        missing = required - set(self.predictions.columns)
        if missing:
            raise ValueError(f"Predictions missing required columns: {missing}")
        if "y_prob" not in self.predictions.columns:
            self.predictions = self.predictions.assign(y_prob=np.nan)
        self.predictions = self.predictions.sort_values("timestamp").reset_index(drop=True)


@dataclass
class WalkForwardSplit:
    train_idx: np.ndarray
    test_idx: np.ndarray
    embargo: int = 0


@dataclass
class BatteryContext:
    artifact: ForecastArtifact
    candles: Optional[pd.DataFrame] = None
    splits: List[WalkForwardSplit] = field(default_factory=list)
    feature_groups: Dict[str, List[str]] = field(default_factory=dict)
    config: Dict[str, float] = field(default_factory=dict)

    def ensure(self) -> None:
        self.artifact.validate()
        if self.candles is not None:
            self.candles = self.candles.sort_values("timestamp").reset_index(drop=True)
