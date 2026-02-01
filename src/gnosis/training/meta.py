"""Meta-learning and parameter optimization.

Implements:
- Meta-learner for trade decisions
- Strategy parameter optimizer (Optuna/Grid)
- Risk-adjusted objective functions
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Callable
from pathlib import Path
import json
import pickle

from .config import MetaConfig


@dataclass
class StrategyParams:
    """Strategy parameters to optimize."""

    # Signal thresholds
    signal_threshold: float = 0.0
    confidence_threshold: float = 0.5

    # Position sizing
    base_size_pct: float = 0.1  # Base position as % of equity
    size_multiplier_min: float = 0.5
    size_multiplier_max: float = 2.0

    # Risk limits
    max_position_pct: float = 0.5
    max_drawdown: float = 0.1
    stop_loss_pct: float = 0.02

    # Execution
    max_turnover_daily: float = 5.0
    min_holding_bars: int = 1

    # Horizon selection
    preferred_horizons: List[int] = field(default_factory=lambda: [4, 8])

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'signal_threshold': self.signal_threshold,
            'confidence_threshold': self.confidence_threshold,
            'base_size_pct': self.base_size_pct,
            'size_multiplier_min': self.size_multiplier_min,
            'size_multiplier_max': self.size_multiplier_max,
            'max_position_pct': self.max_position_pct,
            'max_drawdown': self.max_drawdown,
            'stop_loss_pct': self.stop_loss_pct,
            'max_turnover_daily': self.max_turnover_daily,
            'min_holding_bars': self.min_holding_bars,
            'preferred_horizons': self.preferred_horizons,
        }


class MetaLearner:
    """Meta-learner for trade decisions.

    Takes supervised model outputs and context features
    to decide:
    - When to trade
    - Position sizing multiplier
    - Which horizon/model to trust
    """

    def __init__(self, config: MetaConfig):
        self.config = config
        self.trade_model = None  # When to trade
        self.sizing_model = None  # How much to trade
        self.horizon_model = None  # Which horizon to use
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        trade_labels: np.ndarray,  # 1 = profitable trade, 0 = not
        sizing_labels: np.ndarray,  # Optimal size multiplier
        horizon_labels: np.ndarray,  # Best horizon index
    ) -> 'MetaLearner':
        """Fit meta-learner models.

        Args:
            X: Feature matrix (supervised outputs + context)
            trade_labels: Binary trade/no-trade labels
            sizing_labels: Continuous sizing multipliers
            horizon_labels: Discrete horizon indices

        Returns:
            self for chaining
        """
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

        # Trade decision model
        self.trade_model = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=3,
            random_state=self.config.random_seed,
        )
        self.trade_model.fit(X, trade_labels)

        # Sizing model (regression)
        self.sizing_model = GradientBoostingRegressor(
            n_estimators=50,
            max_depth=3,
            random_state=self.config.random_seed,
        )
        self.sizing_model.fit(X, sizing_labels)

        # Horizon selection model
        self.horizon_model = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=3,
            random_state=self.config.random_seed,
        )
        self.horizon_model.fit(X, horizon_labels)

        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate meta-predictions.

        Args:
            X: Feature matrix

        Returns:
            Dictionary with trade signals, sizing, horizon selection
        """
        if not self._fitted:
            raise ValueError("MetaLearner not fitted")

        trade_proba = self.trade_model.predict_proba(X)[:, 1]
        sizing = self.sizing_model.predict(X)
        horizon_idx = self.horizon_model.predict(X)

        return {
            'trade_signal': (trade_proba > 0.5).astype(int),
            'trade_confidence': trade_proba,
            'sizing_multiplier': np.clip(sizing, 0.1, 3.0),
            'horizon_idx': horizon_idx.astype(int),
        }

    def save(self, path: str) -> None:
        """Save meta-learner to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / 'trade_model.pkl', 'wb') as f:
            pickle.dump(self.trade_model, f)
        with open(path / 'sizing_model.pkl', 'wb') as f:
            pickle.dump(self.sizing_model, f)
        with open(path / 'horizon_model.pkl', 'wb') as f:
            pickle.dump(self.horizon_model, f)

    @classmethod
    def load(cls, path: str, config: MetaConfig = None) -> 'MetaLearner':
        """Load meta-learner from disk."""
        path = Path(path)
        config = config or MetaConfig()

        learner = cls(config)

        with open(path / 'trade_model.pkl', 'rb') as f:
            learner.trade_model = pickle.load(f)
        with open(path / 'sizing_model.pkl', 'rb') as f:
            learner.sizing_model = pickle.load(f)
        with open(path / 'horizon_model.pkl', 'rb') as f:
            learner.horizon_model = pickle.load(f)

        learner._fitted = True
        return learner


class RiskAdjustedObjective:
    """Risk-adjusted objective function for optimization.

    Computes Sharpe/Sortino with penalties for drawdown
    and turnover.
    """

    def __init__(
        self,
        objective_type: str = 'sharpe',
        drawdown_penalty: float = 0.5,
        turnover_penalty: float = 0.1,
        annualization_factor: float = 252.0,
    ):
        self.objective_type = objective_type
        self.drawdown_penalty = drawdown_penalty
        self.turnover_penalty = turnover_penalty
        self.annualization_factor = annualization_factor

    def __call__(
        self,
        returns: np.ndarray,
        turnover: Optional[np.ndarray] = None
    ) -> float:
        """Compute risk-adjusted objective.

        Args:
            returns: Array of period returns
            turnover: Array of turnover rates

        Returns:
            Objective value (higher is better)
        """
        if len(returns) == 0 or np.all(np.isnan(returns)):
            return -np.inf

        returns = returns[~np.isnan(returns)]

        if self.objective_type == 'sharpe':
            base_score = self._sharpe(returns)
        elif self.objective_type == 'sortino':
            base_score = self._sortino(returns)
        elif self.objective_type == 'calmar':
            base_score = self._calmar(returns)
        else:
            base_score = self._sharpe(returns)

        # Drawdown penalty
        cumret = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumret)
        drawdown = (peak - cumret) / peak
        max_dd = np.max(drawdown) if len(drawdown) > 0 else 0

        dd_penalty = self.drawdown_penalty * max_dd

        # Turnover penalty
        turn_penalty = 0.0
        if turnover is not None and len(turnover) > 0:
            avg_turnover = np.mean(turnover)
            turn_penalty = self.turnover_penalty * avg_turnover

        return base_score - dd_penalty - turn_penalty

    def _sharpe(self, returns: np.ndarray) -> float:
        """Compute annualized Sharpe ratio."""
        if len(returns) < 2:
            return 0.0

        mean_ret = np.mean(returns) * self.annualization_factor
        std_ret = np.std(returns, ddof=1) * np.sqrt(self.annualization_factor)

        if std_ret < 1e-8:
            return 0.0

        return mean_ret / std_ret

    def _sortino(self, returns: np.ndarray) -> float:
        """Compute annualized Sortino ratio."""
        if len(returns) < 2:
            return 0.0

        mean_ret = np.mean(returns) * self.annualization_factor
        downside = returns[returns < 0]

        if len(downside) < 2:
            return mean_ret / 0.01  # Assume minimal downside

        downside_std = np.std(downside, ddof=1) * np.sqrt(self.annualization_factor)

        if downside_std < 1e-8:
            return mean_ret / 0.01

        return mean_ret / downside_std

    def _calmar(self, returns: np.ndarray) -> float:
        """Compute Calmar ratio."""
        cumret = np.cumprod(1 + returns)
        total_return = cumret[-1] - 1 if len(cumret) > 0 else 0

        peak = np.maximum.accumulate(cumret)
        drawdown = (peak - cumret) / peak
        max_dd = np.max(drawdown) if len(drawdown) > 0 else 0

        if max_dd < 1e-8:
            return total_return / 0.01

        return total_return / max_dd


class StrategyOptimizer:
    """Optimize strategy parameters using walk-forward validation.

    Supports:
    - Grid search
    - Optuna optimization
    - Bayesian optimization
    """

    def __init__(self, config: MetaConfig):
        self.config = config
        self.best_params: Optional[StrategyParams] = None
        self.trials: List[Dict] = []
        self.objective = RiskAdjustedObjective(
            objective_type=config.objective,
            drawdown_penalty=config.drawdown_penalty,
            turnover_penalty=config.turnover_penalty,
        )

    def optimize(
        self,
        evaluate_fn: Callable[[StrategyParams], Tuple[np.ndarray, np.ndarray]],
        param_grid: Optional[Dict[str, List]] = None
    ) -> StrategyParams:
        """Optimize strategy parameters.

        Args:
            evaluate_fn: Function that takes params and returns (returns, turnover)
            param_grid: Parameter search space

        Returns:
            Optimal StrategyParams
        """
        if self.config.optimizer == 'optuna':
            return self._optimize_optuna(evaluate_fn, param_grid)
        elif self.config.optimizer == 'grid':
            return self._optimize_grid(evaluate_fn, param_grid)
        else:
            return self._optimize_grid(evaluate_fn, param_grid)

    def _optimize_grid(
        self,
        evaluate_fn: Callable,
        param_grid: Optional[Dict[str, List]]
    ) -> StrategyParams:
        """Grid search optimization."""
        if param_grid is None:
            param_grid = self._default_grid()

        from itertools import product

        keys = list(param_grid.keys())
        values = list(param_grid.values())

        best_score = -np.inf
        best_params = StrategyParams()

        for combo in product(*values):
            param_dict = dict(zip(keys, combo))
            params = StrategyParams(**param_dict)

            try:
                returns, turnover = evaluate_fn(params)
                score = self.objective(returns, turnover)
            except Exception as e:
                score = -np.inf

            self.trials.append({
                'params': param_dict,
                'score': score,
            })

            if score > best_score:
                best_score = score
                best_params = params

        self.best_params = best_params
        return best_params

    def _optimize_optuna(
        self,
        evaluate_fn: Callable,
        param_grid: Optional[Dict[str, List]]
    ) -> StrategyParams:
        """Optuna optimization."""
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            # Fall back to grid search
            return self._optimize_grid(evaluate_fn, param_grid)

        def objective(trial):
            params = StrategyParams(
                signal_threshold=trial.suggest_float('signal_threshold', -0.01, 0.01),
                confidence_threshold=trial.suggest_float('confidence_threshold', 0.3, 0.8),
                base_size_pct=trial.suggest_float('base_size_pct', 0.05, 0.3),
                max_position_pct=trial.suggest_float('max_position_pct', 0.3, 1.0),
                stop_loss_pct=trial.suggest_float('stop_loss_pct', 0.01, 0.05),
            )

            try:
                returns, turnover = evaluate_fn(params)
                score = self.objective(returns, turnover)
            except Exception:
                score = -np.inf

            self.trials.append({
                'params': params.to_dict(),
                'score': score,
            })

            return score

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.config.random_seed),
        )
        study.optimize(objective, n_trials=self.config.n_trials, show_progress_bar=False)

        best_trial = study.best_trial
        self.best_params = StrategyParams(
            signal_threshold=best_trial.params['signal_threshold'],
            confidence_threshold=best_trial.params['confidence_threshold'],
            base_size_pct=best_trial.params['base_size_pct'],
            max_position_pct=best_trial.params['max_position_pct'],
            stop_loss_pct=best_trial.params['stop_loss_pct'],
        )

        return self.best_params

    def _default_grid(self) -> Dict[str, List]:
        """Default parameter grid."""
        return {
            'signal_threshold': [-0.005, 0.0, 0.005],
            'confidence_threshold': [0.4, 0.5, 0.6, 0.7],
            'base_size_pct': [0.1, 0.2],
            'max_position_pct': [0.5, 1.0],
            'stop_loss_pct': [0.02, 0.03],
        }

    def save(self, path: str) -> None:
        """Save optimization results."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.best_params is not None:
            with open(path / 'best_params.json', 'w') as f:
                json.dump(self.best_params.to_dict(), f, indent=2)

        with open(path / 'trials.json', 'w') as f:
            json.dump(self.trials, f, indent=2, default=str)


class MetaTrainer:
    """Train meta-learning components.

    Coordinates:
    - Meta-learner training
    - Strategy parameter optimization
    - Walk-forward evaluation
    """

    def __init__(self, config: MetaConfig):
        self.config = config
        self.meta_learner: Optional[MetaLearner] = None
        self.optimizer: Optional[StrategyOptimizer] = None
        self.best_params: Optional[StrategyParams] = None

    def fit(
        self,
        supervised_predictions: pd.DataFrame,
        context_features: pd.DataFrame,
        actual_returns: pd.DataFrame,
    ) -> 'MetaTrainer':
        """Fit meta-learning components.

        Args:
            supervised_predictions: Predictions from supervised models
            context_features: Regime tags and other context
            actual_returns: Actual forward returns

        Returns:
            self for chaining
        """
        # Prepare features for meta-learner
        X = self._prepare_meta_features(supervised_predictions, context_features)

        # Generate meta-labels from actual outcomes
        trade_labels, sizing_labels, horizon_labels = self._generate_meta_labels(
            supervised_predictions, actual_returns
        )

        # Fit meta-learner
        self.meta_learner = MetaLearner(self.config)
        self.meta_learner.fit(X, trade_labels, sizing_labels, horizon_labels)

        return self

    def optimize_params(
        self,
        evaluate_fn: Callable[[StrategyParams], Tuple[np.ndarray, np.ndarray]]
    ) -> StrategyParams:
        """Optimize strategy parameters.

        Args:
            evaluate_fn: Evaluation function

        Returns:
            Optimal parameters
        """
        self.optimizer = StrategyOptimizer(self.config)
        self.best_params = self.optimizer.optimize(evaluate_fn)
        return self.best_params

    def _prepare_meta_features(
        self,
        predictions: pd.DataFrame,
        context: pd.DataFrame
    ) -> np.ndarray:
        """Prepare feature matrix for meta-learner."""
        # Collect prediction features
        pred_cols = [c for c in predictions.columns
                     if c.startswith(('x_hat', 'sigma_hat', 'confidence', 'prob_'))]

        # Collect context features
        ctx_cols = [c for c in context.columns
                    if c.endswith(('_pmax', '_entropy')) or c.startswith('regime')]

        all_cols = pred_cols + [c for c in ctx_cols if c in context.columns]

        # Merge if needed
        if len(pred_cols) > 0 and len(ctx_cols) > 0:
            merged = predictions.merge(
                context[['symbol', 'bar_idx'] + [c for c in ctx_cols if c in context.columns]],
                on=['symbol', 'bar_idx'],
                how='left'
            )
            return merged[all_cols].fillna(0).values

        if len(pred_cols) > 0:
            return predictions[pred_cols].fillna(0).values

        return context[ctx_cols].fillna(0).values if ctx_cols else np.zeros((len(context), 1))

    def _generate_meta_labels(
        self,
        predictions: pd.DataFrame,
        actual_returns: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate labels for meta-learner training."""
        n = len(predictions)

        # Trade labels: 1 if actual return > 0 (simplification)
        if 'future_return' in actual_returns.columns:
            trade_labels = (actual_returns['future_return'] > 0).astype(int).values
        else:
            trade_labels = np.ones(n)

        # Sizing labels: optimal size based on realized return / volatility
        # Simplified: use return magnitude
        if 'future_return' in actual_returns.columns:
            sizing_labels = np.abs(actual_returns['future_return'].fillna(0).values)
            sizing_labels = np.clip(sizing_labels / 0.01, 0.1, 3.0)  # Normalize
        else:
            sizing_labels = np.ones(n)

        # Horizon labels: best horizon (requires multi-horizon returns)
        horizon_labels = np.zeros(n, dtype=int)  # Default to first horizon

        return trade_labels, sizing_labels, horizon_labels

    def save(self, path: str) -> None:
        """Save meta-trainer components."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.meta_learner is not None:
            self.meta_learner.save(str(path / 'meta_learner'))

        if self.optimizer is not None:
            self.optimizer.save(str(path / 'optimizer'))

        if self.best_params is not None:
            with open(path / 'best_params.json', 'w') as f:
                json.dump(self.best_params.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str, config: MetaConfig = None) -> 'MetaTrainer':
        """Load meta-trainer from disk."""
        path = Path(path)
        config = config or MetaConfig()

        trainer = cls(config)

        meta_learner_path = path / 'meta_learner'
        if meta_learner_path.exists():
            trainer.meta_learner = MetaLearner.load(str(meta_learner_path), config)

        best_params_path = path / 'best_params.json'
        if best_params_path.exists():
            with open(best_params_path) as f:
                params_dict = json.load(f)
            trainer.best_params = StrategyParams(**params_dict)

        return trainer
