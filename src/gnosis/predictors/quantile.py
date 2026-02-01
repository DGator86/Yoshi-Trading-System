"""Quantile prediction models."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


class QuantilePredictor:
    """Simple quantile predictor using tilted loss approximation."""

    def __init__(self, models_config: dict):
        self.config = models_config.get("predictor", {})
        self.quantiles = self.config.get("quantiles", [0.05, 0.50, 0.95])
        self.l2_reg = self.config.get("l2_reg", 1.0)
        self.models = {}

    def _get_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract feature matrix."""
        feature_cols = [
            "returns", "realized_vol", "ofi", "range_pct",
            "flow_momentum", "regime_stability", "barrier_proximity",
            "particle_score"
        ]
        # Only use columns that exist
        available = [c for c in feature_cols if c in df.columns]
        X = df[available].fillna(0).values
        return X

    def fit(self, train_df: pd.DataFrame, target_col: str = "future_return") -> None:
        """Fit quantile models."""
        X = self._get_features(train_df)
        y = train_df[target_col].fillna(0).values

        for q in self.quantiles:
            # Use Ridge regression with sample weights approximating quantile loss
            # This is a simple approximation; for production use proper quantile regression
            weights = np.where(y > 0, q, 1 - q)
            model = Ridge(alpha=self.l2_reg)
            model.fit(X, y, sample_weight=weights)
            self.models[q] = model

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate quantile predictions."""
        X = self._get_features(df)
        result = df[["symbol", "bar_idx", "timestamp_end", "close"]].copy()

        for q in self.quantiles:
            if q in self.models:
                pred = self.models[q].predict(X)
                result[f"q{int(q*100):02d}"] = pred
            else:
                result[f"q{int(q*100):02d}"] = 0.0

        # Point estimate is median
        result["x_hat"] = result["q50"]

        # Uncertainty from IQR (before any scaling)
        result["sigma_hat"] = (result["q95"] - result["q05"]) / 3.29  # approx std

        # Apply sigma_scale to widen/narrow intervals (SINGLE application)
        # Priority: models.predictor.sigma_scale > models.sigma_scale > 1.0
        sigma_scale = 1.0
        if isinstance(self.config, dict):
            sigma_scale = float(self.config.get('sigma_scale', 1.0))
            _pcfg = self.config.get('predictor', {})
            if isinstance(_pcfg, dict) and 'sigma_scale' in _pcfg:
                sigma_scale = float(_pcfg['sigma_scale'])

        if sigma_scale <= 0:
            sigma_scale = 1.0

        if sigma_scale != 1.0:
            center = result['q50']
            half = (result['q95'] - result['q05']) / 2.0
            half = half * sigma_scale
            result['q05'] = center - half
            result['q95'] = center + half
            result['sigma_hat'] = (result['q95'] - result['q05']) / 3.29

        return result


class BaselinePredictor:
    """Random walk + realized vol cone baseline."""

    def __init__(self):
        pass

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate baseline predictions (random walk + vol cone)."""
        result = df[["symbol", "bar_idx", "timestamp_end", "close"]].copy()

        # Random walk: point estimate = 0 return
        result["x_hat"] = 0.0

        # Vol cone: use realized vol for uncertainty
        vol = df["realized_vol"].fillna(df["realized_vol"].median())
        result["sigma_hat"] = vol

        # Quantiles from normal distribution
        result["q05"] = -1.645 * vol
        result["q50"] = 0.0
        result["q95"] = 1.645 * vol

        return result
