"""Feature engineering and preparation for training.

Combines domain features, regime features, and particle state
into a unified feature matrix for model training.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import hashlib
import json


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""

    # Feature groups to include
    include_returns: bool = True
    include_volatility: bool = True
    include_ofi: bool = True
    include_regime: bool = True
    include_particle: bool = True
    include_technical: bool = True

    # Regime feature settings
    regime_levels: List[str] = field(default_factory=lambda: ['K', 'P', 'C', 'O', 'F', 'G', 'S'])
    include_regime_probs: bool = True
    include_regime_entropy: bool = True

    # Technical indicator settings
    lookback_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50])

    # Normalization
    normalize: bool = True
    clip_outliers: float = 5.0  # Clip at n std deviations


class FeatureEngineer:
    """Engineer features from bar data.

    Produces a feature matrix X suitable for model training.
    Features are designed to be predictive of future returns
    while avoiding any lookahead bias.
    """

    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()
        self._feature_names: List[str] = []
        self._feature_stats: Dict[str, dict] = {}

    def fit(self, df: pd.DataFrame) -> 'FeatureEngineer':
        """Fit feature statistics on training data.

        Args:
            df: Training DataFrame with bar data

        Returns:
            self for chaining
        """
        # Compute feature statistics for normalization
        features = self._extract_raw_features(df)

        for col in features.columns:
            vals = features[col].dropna()
            if len(vals) > 0:
                self._feature_stats[col] = {
                    'mean': float(vals.mean()),
                    'std': float(vals.std()) + 1e-8,
                    'min': float(vals.min()),
                    'max': float(vals.max()),
                }

        self._feature_names = list(features.columns)
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Transform bar data into feature matrix.

        Args:
            df: DataFrame with bar data

        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        features = self._extract_raw_features(df)

        if self.config.normalize and len(self._feature_stats) > 0:
            features = self._normalize_features(features)

        if self.config.clip_outliers > 0:
            features = features.clip(
                -self.config.clip_outliers,
                self.config.clip_outliers
            )

        return features.fillna(0).values, list(features.columns)

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)

    def _extract_raw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract all raw features from DataFrame."""
        features = pd.DataFrame(index=df.index)

        # Return-based features
        if self.config.include_returns:
            features = pd.concat([features, self._returns_features(df)], axis=1)

        # Volatility features
        if self.config.include_volatility:
            features = pd.concat([features, self._volatility_features(df)], axis=1)

        # Order flow features
        if self.config.include_ofi:
            features = pd.concat([features, self._ofi_features(df)], axis=1)

        # Regime features
        if self.config.include_regime:
            features = pd.concat([features, self._regime_features(df)], axis=1)

        # Particle state features
        if self.config.include_particle:
            features = pd.concat([features, self._particle_features(df)], axis=1)

        # Technical features
        if self.config.include_technical:
            features = pd.concat([features, self._technical_features(df)], axis=1)

        return features

    def _returns_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract return-based features."""
        features = pd.DataFrame(index=df.index)

        if 'returns' in df.columns:
            features['returns'] = df['returns']

            for period in self.config.lookback_periods:
                features[f'returns_ma_{period}'] = df.groupby('symbol')['returns'].transform(
                    lambda x: x.rolling(period, min_periods=1).mean()
                )
                features[f'returns_momentum_{period}'] = df['returns'] - features[f'returns_ma_{period}']

        return features

    def _volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract volatility-based features."""
        features = pd.DataFrame(index=df.index)

        if 'realized_vol' in df.columns:
            features['realized_vol'] = df['realized_vol']

            # Vol regime (high/low relative to moving average)
            for period in [20, 50]:
                vol_ma = df.groupby('symbol')['realized_vol'].transform(
                    lambda x: x.rolling(period, min_periods=5).mean()
                )
                features[f'vol_ratio_{period}'] = df['realized_vol'] / (vol_ma + 1e-8)

        if 'range_pct' in df.columns:
            features['range_pct'] = df['range_pct']
            features['range_vol_ratio'] = df['range_pct'] / (df['realized_vol'] + 1e-8)

        return features

    def _ofi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract order flow imbalance features."""
        features = pd.DataFrame(index=df.index)

        if 'ofi' in df.columns:
            features['ofi'] = df['ofi']

            # OFI momentum
            for period in [5, 10, 20]:
                features[f'ofi_ma_{period}'] = df.groupby('symbol')['ofi'].transform(
                    lambda x: x.rolling(period, min_periods=1).mean()
                )
                features[f'ofi_std_{period}'] = df.groupby('symbol')['ofi'].transform(
                    lambda x: x.rolling(period, min_periods=1).std()
                )

        return features

    def _regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract regime classification features."""
        features = pd.DataFrame(index=df.index)

        for level in self.config.regime_levels:
            # Pmax (confidence)
            pmax_col = f'{level}_pmax'
            if pmax_col in df.columns:
                features[pmax_col] = df[pmax_col]

            # Entropy
            if self.config.include_regime_entropy:
                entropy_col = f'{level}_entropy'
                if entropy_col in df.columns:
                    features[entropy_col] = df[entropy_col]

            # Label encoding (one-hot or numeric)
            label_col = f'{level}_label'
            if label_col in df.columns:
                # Use pmax as proxy for label strength
                pass

        # Total regime entropy
        if 'regime_entropy' in df.columns:
            features['regime_entropy'] = df['regime_entropy']

        return features

    def _particle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract particle state features."""
        features = pd.DataFrame(index=df.index)

        particle_cols = [
            'flow_momentum', 'regime_stability', 'barrier_proximity', 'particle_score'
        ]

        for col in particle_cols:
            if col in df.columns:
                features[col] = df[col]

        return features

    def _technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract technical indicator features."""
        features = pd.DataFrame(index=df.index)

        if 'close' in df.columns:
            close = df['close']

            # Price momentum indicators
            for period in self.config.lookback_periods:
                ma = df.groupby('symbol')['close'].transform(
                    lambda x: x.rolling(period, min_periods=1).mean()
                )
                features[f'price_ma_ratio_{period}'] = close / (ma + 1e-8) - 1

            # RSI-like
            if 'returns' in df.columns:
                gains = df['returns'].clip(lower=0)
                losses = (-df['returns']).clip(lower=0)

                for period in [14]:
                    avg_gain = df.groupby('symbol').apply(
                        lambda g: gains.loc[g.index].rolling(period, min_periods=1).mean()
                    ).reset_index(level=0, drop=True)
                    avg_loss = df.groupby('symbol').apply(
                        lambda g: losses.loc[g.index].rolling(period, min_periods=1).mean()
                    ).reset_index(level=0, drop=True)

                    rs = avg_gain / (avg_loss + 1e-8)
                    features[f'rsi_{period}'] = 100 - 100 / (1 + rs)

        return features

    def _normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Normalize features using fitted statistics."""
        result = features.copy()

        for col in features.columns:
            if col in self._feature_stats:
                stats = self._feature_stats[col]
                result[col] = (features[col] - stats['mean']) / stats['std']

        return result

    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self._feature_names

    def get_schema_hash(self) -> str:
        """Get hash of feature schema for versioning."""
        schema = {
            'feature_names': self._feature_names,
            'config': {
                'include_returns': self.config.include_returns,
                'include_volatility': self.config.include_volatility,
                'include_ofi': self.config.include_ofi,
                'include_regime': self.config.include_regime,
                'include_particle': self.config.include_particle,
                'include_technical': self.config.include_technical,
            }
        }
        schema_str = json.dumps(schema, sort_keys=True)
        return hashlib.sha256(schema_str.encode()).hexdigest()[:16]


def prepare_training_data(
    df: pd.DataFrame,
    target_col: str = 'future_return',
    feature_config: FeatureConfig = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Prepare training data with features and targets.

    Args:
        df: DataFrame with bar data and labels
        target_col: Name of target column
        feature_config: Feature configuration

    Returns:
        Tuple of (X, y, feature_names)
    """
    engineer = FeatureEngineer(feature_config)
    X, feature_names = engineer.fit_transform(df)

    y = df[target_col].fillna(0).values

    # Remove rows with NaN targets
    valid_mask = ~np.isnan(df[target_col].values)
    X = X[valid_mask]
    y = y[valid_mask]

    return X, y, feature_names
