"""Supervised learning models for price prediction.

Implements:
- Quantile regression (Ridge, GBM)
- Direction classification
- Calibrated confidence scores
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import json
import pickle

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV

from .config import SupervisedConfig, get_git_commit


@dataclass
class ModelArtifact:
    """Container for trained model artifacts."""

    model_type: str
    horizon: int
    quantiles: List[float]
    models: Dict[float, Any]  # quantile -> fitted model
    calibrator: Optional[Any] = None
    feature_names: List[str] = field(default_factory=list)
    feature_schema_hash: str = ''
    training_metrics: Dict[str, float] = field(default_factory=dict)
    git_commit: str = ''

    def save(self, path: str) -> None:
        """Save artifact to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save models
        with open(path / 'models.pkl', 'wb') as f:
            pickle.dump(self.models, f)

        # Save calibrator
        if self.calibrator is not None:
            with open(path / 'calibrator.pkl', 'wb') as f:
                pickle.dump(self.calibrator, f)

        # Save metadata
        meta = {
            'model_type': self.model_type,
            'horizon': self.horizon,
            'quantiles': self.quantiles,
            'feature_names': self.feature_names,
            'feature_schema_hash': self.feature_schema_hash,
            'training_metrics': self.training_metrics,
            'git_commit': self.git_commit,
        }
        with open(path / 'metadata.json', 'w') as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'ModelArtifact':
        """Load artifact from disk."""
        path = Path(path)

        with open(path / 'metadata.json') as f:
            meta = json.load(f)

        with open(path / 'models.pkl', 'rb') as f:
            models = pickle.load(f)

        calibrator = None
        calibrator_path = path / 'calibrator.pkl'
        if calibrator_path.exists():
            with open(calibrator_path, 'rb') as f:
                calibrator = pickle.load(f)

        return cls(
            model_type=meta['model_type'],
            horizon=meta['horizon'],
            quantiles=meta['quantiles'],
            models=models,
            calibrator=calibrator,
            feature_names=meta.get('feature_names', []),
            feature_schema_hash=meta.get('feature_schema_hash', ''),
            training_metrics=meta.get('training_metrics', {}),
            git_commit=meta.get('git_commit', ''),
        )


class QuantileRegressor:
    """Quantile regression model for return prediction.

    Predicts multiple quantiles of the return distribution,
    providing both point estimates and uncertainty.
    """

    def __init__(self, config: SupervisedConfig):
        self.config = config
        self.models: Dict[float, Ridge] = {}
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> 'QuantileRegressor':
        """Fit quantile regression models.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
            sample_weight: Optional sample weights

        Returns:
            self for chaining
        """
        for q in self.config.quantiles:
            # Use tilted loss approximation with sample weights
            if sample_weight is None:
                weights = np.where(y > 0, q, 1 - q)
            else:
                weights = sample_weight * np.where(y > 0, q, 1 - q)

            model = Ridge(alpha=self.config.l2_reg, random_state=self.config.random_seed)
            model.fit(X, y, sample_weight=weights)
            self.models[q] = model

        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate quantile predictions.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Dictionary with quantile predictions and derived metrics
        """
        if not self._fitted:
            raise ValueError("Model not fitted")

        result = {}

        # Predict each quantile
        for q in self.config.quantiles:
            key = f'q{int(q*100):02d}'
            result[key] = self.models[q].predict(X)

        # Point estimate is median
        if 0.50 in self.models:
            result['x_hat'] = self.models[0.50].predict(X)
        else:
            # Use closest to median
            closest_q = min(self.config.quantiles, key=lambda q: abs(q - 0.50))
            result['x_hat'] = self.models[closest_q].predict(X)

        # Uncertainty from IQR or 90% interval
        if 0.05 in self.models and 0.95 in self.models:
            q05 = self.models[0.05].predict(X)
            q95 = self.models[0.95].predict(X)
            result['sigma_hat'] = (q95 - q05) / 3.29  # Approximate std
            result['interval_90_lower'] = q05
            result['interval_90_upper'] = q95
        elif 0.25 in self.models and 0.75 in self.models:
            q25 = self.models[0.25].predict(X)
            q75 = self.models[0.75].predict(X)
            result['sigma_hat'] = (q75 - q25) / 1.35  # IQR to std
            result['interval_50_lower'] = q25
            result['interval_50_upper'] = q75

        return result


class DirectionClassifier:
    """Classification model for direction prediction.

    Predicts probability of UP/FLAT/DOWN outcomes.
    """

    def __init__(self, config: SupervisedConfig):
        self.config = config
        self.model = None
        self.calibrator = None
        self._classes = ['DOWN', 'FLAT', 'UP']
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,  # Numeric: -1, 0, 1
        calibrate: bool = True
    ) -> 'DirectionClassifier':
        """Fit direction classifier.

        Args:
            X: Feature matrix
            y: Target labels (-1=DOWN, 0=FLAT, 1=UP)
            calibrate: Whether to calibrate probabilities

        Returns:
            self for chaining
        """
        # Convert numeric to class indices
        y_class = y + 1  # -1,0,1 -> 0,1,2

        if calibrate and self.config.calibrate_confidence:
            base_model = LogisticRegression(
                C=1.0 / self.config.l2_reg,
                random_state=self.config.random_seed,
                max_iter=1000,
                multi_class='multinomial',
            )
            self.model = CalibratedClassifierCV(
                base_model,
                method=self.config.calibration_method,
                cv=3,
            )
        else:
            self.model = LogisticRegression(
                C=1.0 / self.config.l2_reg,
                random_state=self.config.random_seed,
                max_iter=1000,
                multi_class='multinomial',
            )

        self.model.fit(X, y_class)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate direction predictions with probabilities.

        Args:
            X: Feature matrix

        Returns:
            Dictionary with predictions and probabilities
        """
        if not self._fitted:
            raise ValueError("Model not fitted")

        probs = self.model.predict_proba(X)
        pred_class = self.model.predict(X)

        result = {
            'direction_pred': np.array([self._classes[int(c)] for c in pred_class]),
            'direction_pred_num': pred_class - 1,  # Back to -1,0,1
            'prob_down': probs[:, 0],
            'prob_flat': probs[:, 1],
            'prob_up': probs[:, 2],
            'confidence': np.max(probs, axis=1),
        }

        return result


class ConfidenceCalibrator:
    """Calibrate prediction confidence scores.

    Uses isotonic regression to map raw confidence to
    calibrated probabilities.
    """

    def __init__(self):
        self.calibrator = None
        self._fitted = False

    def fit(
        self,
        predicted_confidence: np.ndarray,
        actual_outcomes: np.ndarray  # 1 if prediction was correct, 0 otherwise
    ) -> 'ConfidenceCalibrator':
        """Fit calibrator on validation data.

        Args:
            predicted_confidence: Raw confidence scores
            actual_outcomes: Binary correctness indicators

        Returns:
            self for chaining
        """
        self.calibrator = IsotonicRegression(
            out_of_bounds='clip',
            increasing=True,
        )
        self.calibrator.fit(predicted_confidence, actual_outcomes)
        self._fitted = True
        return self

    def calibrate(self, confidence: np.ndarray) -> np.ndarray:
        """Apply calibration to confidence scores.

        Args:
            confidence: Raw confidence scores

        Returns:
            Calibrated confidence scores
        """
        if not self._fitted:
            return confidence

        return self.calibrator.predict(confidence)


class SupervisedTrainer:
    """Train supervised models for all horizons.

    Coordinates training of quantile regressors and/or
    direction classifiers across multiple forecast horizons.
    """

    def __init__(self, config: SupervisedConfig):
        self.config = config
        self.artifacts: Dict[int, ModelArtifact] = {}
        self._feature_names: List[str] = []
        self._feature_schema_hash: str = ''

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_names: List[str],
        feature_schema_hash: str = '',
    ) -> 'SupervisedTrainer':
        """Fit models for all horizons.

        Args:
            train_df: Training DataFrame with features and labels
            feature_names: List of feature column names
            feature_schema_hash: Hash of feature schema

        Returns:
            self for chaining
        """
        self._feature_names = feature_names
        self._feature_schema_hash = feature_schema_hash

        X = train_df[feature_names].fillna(0).values

        for horizon in self.config.horizons:
            target_col = f'fwd_ret_{horizon}'
            direction_col = f'direction_{horizon}_num'

            if target_col not in train_df.columns:
                continue

            y = train_df[target_col].fillna(0).values

            # Fit quantile regressor
            quantile_model = QuantileRegressor(self.config)
            quantile_model.fit(X, y)

            # Fit direction classifier if available
            direction_model = None
            if direction_col in train_df.columns:
                y_dir = train_df[direction_col].fillna(0).values.astype(int)
                direction_model = DirectionClassifier(self.config)
                direction_model.fit(X, y_dir)

            # Store artifact
            self.artifacts[horizon] = ModelArtifact(
                model_type=self.config.model_type,
                horizon=horizon,
                quantiles=self.config.quantiles,
                models={
                    'quantile': quantile_model,
                    'direction': direction_model,
                },
                feature_names=feature_names,
                feature_schema_hash=feature_schema_hash,
                git_commit=get_git_commit(),
            )

        return self

    def predict(
        self,
        df: pd.DataFrame,
        horizons: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """Generate predictions for all horizons.

        Args:
            df: DataFrame with features
            horizons: Specific horizons to predict (default: all)

        Returns:
            DataFrame with predictions
        """
        horizons = horizons or list(self.artifacts.keys())
        X = df[self._feature_names].fillna(0).values

        result = df[['symbol', 'bar_idx', 'timestamp_end', 'close']].copy()

        for horizon in horizons:
            if horizon not in self.artifacts:
                continue

            artifact = self.artifacts[horizon]
            quantile_model = artifact.models['quantile']
            direction_model = artifact.models.get('direction')

            # Quantile predictions
            q_preds = quantile_model.predict(X)
            for key, vals in q_preds.items():
                result[f'{key}_h{horizon}'] = vals

            # Direction predictions
            if direction_model is not None:
                d_preds = direction_model.predict(X)
                for key, vals in d_preds.items():
                    result[f'{key}_h{horizon}'] = vals

        return result

    def calibrate(
        self,
        val_df: pd.DataFrame,
        horizons: Optional[List[int]] = None
    ) -> None:
        """Calibrate confidence scores using validation data.

        Args:
            val_df: Validation DataFrame with predictions and actuals
            horizons: Horizons to calibrate
        """
        horizons = horizons or list(self.artifacts.keys())
        X = val_df[self._feature_names].fillna(0).values

        for horizon in horizons:
            if horizon not in self.artifacts:
                continue

            artifact = self.artifacts[horizon]
            direction_model = artifact.models.get('direction')

            if direction_model is None:
                continue

            # Get predictions
            d_preds = direction_model.predict(X)

            # Get actual outcomes
            direction_col = f'direction_{horizon}_num'
            if direction_col not in val_df.columns:
                continue

            y_true = val_df[direction_col].values
            y_pred = d_preds['direction_pred_num']

            # Compute correctness
            correct = (y_true == y_pred).astype(float)

            # Fit calibrator
            calibrator = ConfidenceCalibrator()
            calibrator.fit(d_preds['confidence'], correct)

            artifact.calibrator = calibrator

    def evaluate(
        self,
        test_df: pd.DataFrame,
        horizons: Optional[List[int]] = None
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate models on test data.

        Args:
            test_df: Test DataFrame with features and actuals
            horizons: Horizons to evaluate

        Returns:
            Dictionary of metrics per horizon
        """
        horizons = horizons or list(self.artifacts.keys())
        X = test_df[self._feature_names].fillna(0).values

        metrics = {}

        for horizon in horizons:
            if horizon not in self.artifacts:
                continue

            artifact = self.artifacts[horizon]
            quantile_model = artifact.models['quantile']
            direction_model = artifact.models.get('direction')

            target_col = f'fwd_ret_{horizon}'
            direction_col = f'direction_{horizon}_num'

            if target_col not in test_df.columns:
                continue

            y_true = test_df[target_col].fillna(0).values
            valid_mask = ~np.isnan(test_df[target_col].values)

            h_metrics = {}

            # Quantile metrics
            q_preds = quantile_model.predict(X)

            h_metrics['mae'] = float(np.mean(np.abs(y_true[valid_mask] - q_preds['x_hat'][valid_mask])))

            if 'interval_90_lower' in q_preds and 'interval_90_upper' in q_preds:
                in_interval = (
                    (y_true >= q_preds['interval_90_lower']) &
                    (y_true <= q_preds['interval_90_upper'])
                )
                h_metrics['coverage_90'] = float(np.mean(in_interval[valid_mask]))
                h_metrics['sharpness_90'] = float(np.mean(
                    q_preds['interval_90_upper'][valid_mask] -
                    q_preds['interval_90_lower'][valid_mask]
                ))

            # Direction metrics
            if direction_model is not None and direction_col in test_df.columns:
                d_preds = direction_model.predict(X)
                y_dir = test_df[direction_col].fillna(0).values.astype(int)

                h_metrics['direction_accuracy'] = float(np.mean(
                    d_preds['direction_pred_num'][valid_mask] == y_dir[valid_mask]
                ))

                # Calibrated confidence
                if artifact.calibrator is not None:
                    calibrated_conf = artifact.calibrator.calibrate(d_preds['confidence'])
                    h_metrics['avg_calibrated_confidence'] = float(np.mean(calibrated_conf[valid_mask]))

            metrics[f'horizon_{horizon}'] = h_metrics

        return metrics

    def save(self, base_path: str) -> None:
        """Save all artifacts.

        Args:
            base_path: Base directory for saving
        """
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)

        for horizon, artifact in self.artifacts.items():
            artifact.save(str(base_path / f'horizon_{horizon}'))

        # Save overall metadata
        meta = {
            'horizons': list(self.artifacts.keys()),
            'feature_names': self._feature_names,
            'feature_schema_hash': self._feature_schema_hash,
            'config': {
                'model_type': self.config.model_type,
                'quantiles': self.config.quantiles,
                'l2_reg': self.config.l2_reg,
            },
        }
        with open(base_path / 'trainer_metadata.json', 'w') as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, base_path: str, config: SupervisedConfig = None) -> 'SupervisedTrainer':
        """Load trainer from disk.

        Args:
            base_path: Base directory
            config: Config to use (loads from metadata if None)

        Returns:
            Loaded SupervisedTrainer
        """
        base_path = Path(base_path)

        with open(base_path / 'trainer_metadata.json') as f:
            meta = json.load(f)

        if config is None:
            config = SupervisedConfig(**meta.get('config', {}))

        trainer = cls(config)
        trainer._feature_names = meta.get('feature_names', [])
        trainer._feature_schema_hash = meta.get('feature_schema_hash', '')

        for horizon in meta.get('horizons', []):
            artifact = ModelArtifact.load(str(base_path / f'horizon_{horizon}'))
            trainer.artifacts[horizon] = artifact

        return trainer
