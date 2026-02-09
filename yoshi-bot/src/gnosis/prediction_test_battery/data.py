from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from gnosis.prediction_test_battery.context import ForecastArtifact


def load_predictions(path: Path) -> ForecastArtifact:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    horizon = int(df.get("horizon", pd.Series([1])).iloc[0])
    return ForecastArtifact(name=path.stem, horizon=horizon, predictions=df)


def load_candles(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def generate_synthetic_data(n: int = 240, seed: int = 42) -> tuple[pd.DataFrame, ForecastArtifact]:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2023-01-01", periods=n, freq="H")
    returns = rng.normal(0, 0.01, size=n)
    prices = 100 * np.exp(np.cumsum(returns))
    high = prices * (1 + rng.normal(0.001, 0.002, size=n))
    low = prices * (1 - rng.normal(0.001, 0.002, size=n))
    candles = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": prices,
            "high": high,
            "low": low,
            "close": prices,
            "volume": rng.lognormal(mean=2, sigma=0.2, size=n),
        }
    )
    y_true = (pd.Series(returns).shift(-1).fillna(0) > 0).astype(int).to_numpy()
    y_prob = np.clip(0.5 + returns * 5, 0.01, 0.99)
    predictions = pd.DataFrame(
        {
            "timestamp": timestamps,
            "y_true": y_true,
            "y_pred": returns,
            "y_prob": y_prob,
        }
    )
    features = pd.DataFrame(
        {
            "ret_1": returns,
            "vol_24": pd.Series(returns).rolling(24).std().fillna(0),
        }
    )
    artifact = ForecastArtifact(name="synthetic", horizon=1, predictions=predictions, features=features)
    return candles, artifact


def load_features(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])
    return df
