"""Evaluation metrics for the training system.

Provides:
- Walk-forward equity curve computation
- Per-regime metrics
- Calibration curves
- Performance statistics
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    avg_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    n_trades: int = 0
    avg_trade_return: float = 0.0
    volatility: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'max_drawdown': self.max_drawdown,
            'avg_drawdown': self.avg_drawdown,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'n_trades': self.n_trades,
            'avg_trade_return': self.avg_trade_return,
            'volatility': self.volatility,
        }


def compute_equity_curve(
    returns: np.ndarray,
    initial_capital: float = 10000.0
) -> np.ndarray:
    """Compute equity curve from returns.

    Args:
        returns: Array of period returns
        initial_capital: Starting capital

    Returns:
        Equity curve array
    """
    cumret = np.cumprod(1 + returns)
    return initial_capital * cumret


def compute_drawdown_series(equity: np.ndarray) -> np.ndarray:
    """Compute drawdown series from equity curve.

    Args:
        equity: Equity curve

    Returns:
        Drawdown series (as positive values)
    """
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    return drawdown


def compute_performance_metrics(
    returns: np.ndarray,
    annualization_factor: float = 252.0,
    risk_free_rate: float = 0.0
) -> PerformanceMetrics:
    """Compute comprehensive performance metrics.

    Args:
        returns: Array of period returns
        annualization_factor: Factor for annualization
        risk_free_rate: Risk-free rate (annualized)

    Returns:
        PerformanceMetrics object
    """
    if len(returns) == 0 or np.all(np.isnan(returns)):
        return PerformanceMetrics()

    returns = returns[~np.isnan(returns)]
    n = len(returns)

    if n == 0:
        return PerformanceMetrics()

    # Basic statistics
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1) if n > 1 else 0.0

    # Total and annualized return
    total_return = (1 + returns).prod() - 1
    periods_per_year = annualization_factor
    years = n / periods_per_year
    annualized_return = (1 + total_return) ** (1 / max(years, 0.01)) - 1 if years > 0 else 0

    # Volatility
    volatility = std_return * np.sqrt(annualization_factor)

    # Sharpe ratio
    excess_return = mean_return - risk_free_rate / annualization_factor
    sharpe = (excess_return / std_return * np.sqrt(annualization_factor)) if std_return > 1e-8 else 0

    # Sortino ratio
    downside_returns = returns[returns < 0]
    downside_std = np.std(downside_returns, ddof=1) if len(downside_returns) > 1 else std_return
    sortino = (excess_return / downside_std * np.sqrt(annualization_factor)) if downside_std > 1e-8 else 0

    # Drawdown
    equity = compute_equity_curve(returns)
    drawdowns = compute_drawdown_series(equity)
    max_drawdown = np.max(drawdowns)
    avg_drawdown = np.mean(drawdowns)

    # Calmar ratio
    calmar = annualized_return / max_drawdown if max_drawdown > 1e-8 else 0

    # Win rate and trade statistics
    wins = returns > 0
    losses = returns < 0
    win_rate = np.mean(wins) if len(returns) > 0 else 0

    total_gains = np.sum(returns[wins]) if np.any(wins) else 0
    total_losses = -np.sum(returns[losses]) if np.any(losses) else 0
    profit_factor = total_gains / total_losses if total_losses > 1e-8 else float('inf')

    return PerformanceMetrics(
        total_return=float(total_return),
        annualized_return=float(annualized_return),
        sharpe_ratio=float(sharpe),
        sortino_ratio=float(sortino),
        calmar_ratio=float(calmar),
        max_drawdown=float(max_drawdown),
        avg_drawdown=float(avg_drawdown),
        win_rate=float(win_rate),
        profit_factor=float(min(profit_factor, 100)),  # Cap for display
        n_trades=int(n),
        avg_trade_return=float(mean_return),
        volatility=float(volatility),
    )


def compute_per_regime_metrics(
    returns: np.ndarray,
    regime_labels: np.ndarray,
    annualization_factor: float = 252.0
) -> Dict[str, PerformanceMetrics]:
    """Compute metrics for each regime.

    Args:
        returns: Array of period returns
        regime_labels: Array of regime labels
        annualization_factor: Factor for annualization

    Returns:
        Dictionary mapping regime label to metrics
    """
    unique_regimes = np.unique(regime_labels[~pd.isna(regime_labels)])
    regime_metrics = {}

    for regime in unique_regimes:
        mask = regime_labels == regime
        regime_returns = returns[mask]

        if len(regime_returns) > 0:
            metrics = compute_performance_metrics(regime_returns, annualization_factor)
            regime_metrics[str(regime)] = metrics

    return regime_metrics


def compute_calibration_curve(
    predicted_confidence: np.ndarray,
    actual_outcomes: np.ndarray,
    n_bins: int = 10
) -> Dict[str, np.ndarray]:
    """Compute calibration curve data.

    Args:
        predicted_confidence: Predicted confidence scores
        actual_outcomes: Binary outcomes (1 if correct, 0 if not)
        n_bins: Number of calibration bins

    Returns:
        Dictionary with calibration curve data
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    predicted_probs = []
    actual_freqs = []
    counts = []

    for i in range(n_bins):
        mask = (predicted_confidence >= bin_edges[i]) & (predicted_confidence < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (predicted_confidence >= bin_edges[i]) & (predicted_confidence <= bin_edges[i + 1])

        n_in_bin = np.sum(mask)
        if n_in_bin > 0:
            predicted_probs.append(np.mean(predicted_confidence[mask]))
            actual_freqs.append(np.mean(actual_outcomes[mask]))
            counts.append(n_in_bin)
        else:
            predicted_probs.append(bin_centers[i])
            actual_freqs.append(np.nan)
            counts.append(0)

    predicted_probs = np.array(predicted_probs)
    actual_freqs = np.array(actual_freqs)
    counts = np.array(counts)

    # Expected Calibration Error
    valid_mask = counts > 0
    if np.any(valid_mask):
        ece = np.sum(counts[valid_mask] * np.abs(predicted_probs[valid_mask] - actual_freqs[valid_mask])) / np.sum(counts[valid_mask])
    else:
        ece = 0.0

    return {
        'bin_centers': bin_centers,
        'predicted_probs': predicted_probs,
        'actual_freqs': actual_freqs,
        'counts': counts,
        'ece': float(ece),
    }


def compute_interval_metrics(
    y_true: np.ndarray,
    q_lower: np.ndarray,
    q_upper: np.ndarray,
    nominal_coverage: float = 0.90
) -> Dict[str, float]:
    """Compute prediction interval metrics.

    Args:
        y_true: Actual values
        q_lower: Lower quantile predictions
        q_upper: Upper quantile predictions
        nominal_coverage: Nominal coverage level

    Returns:
        Dictionary of interval metrics
    """
    valid = ~(np.isnan(y_true) | np.isnan(q_lower) | np.isnan(q_upper))
    y_true = y_true[valid]
    q_lower = q_lower[valid]
    q_upper = q_upper[valid]

    if len(y_true) == 0:
        return {
            'coverage': np.nan,
            'sharpness': np.nan,
            'wis': np.nan,
        }

    # Coverage
    in_interval = (y_true >= q_lower) & (y_true <= q_upper)
    coverage = np.mean(in_interval)

    # Sharpness (average interval width)
    sharpness = np.mean(q_upper - q_lower)

    # Weighted Interval Score (WIS)
    alpha = 1 - nominal_coverage
    wis = sharpness + (2 / alpha) * (
        (q_lower - y_true) * (y_true < q_lower) +
        (y_true - q_upper) * (y_true > q_upper)
    ).mean()

    return {
        'coverage': float(coverage),
        'sharpness': float(sharpness),
        'wis': float(wis),
        'coverage_gap': float(abs(coverage - nominal_coverage)),
    }


class WalkForwardEvaluator:
    """Evaluate strategy using walk-forward methodology."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        annualization_factor: float = 252.0
    ):
        self.initial_capital = initial_capital
        self.annualization_factor = annualization_factor

    def evaluate(
        self,
        predictions_df: pd.DataFrame,
        fold_col: str = 'fold_idx',
        return_col: str = 'strategy_return',
        regime_col: Optional[str] = None
    ) -> Dict:
        """Run walk-forward evaluation.

        Args:
            predictions_df: DataFrame with predictions and returns
            fold_col: Column containing fold indices
            return_col: Column containing strategy returns
            regime_col: Optional column for regime-based analysis

        Returns:
            Dictionary with evaluation results
        """
        results = {
            'folds': [],
            'overall': None,
            'equity_curve': None,
            'per_regime': None,
        }

        # Evaluate each fold
        all_returns = []

        for fold_idx in sorted(predictions_df[fold_col].unique()):
            fold_df = predictions_df[predictions_df[fold_col] == fold_idx]
            fold_returns = fold_df[return_col].values

            fold_metrics = compute_performance_metrics(
                fold_returns, self.annualization_factor
            )

            results['folds'].append({
                'fold_idx': int(fold_idx),
                'n_samples': len(fold_returns),
                'metrics': fold_metrics.to_dict(),
            })

            all_returns.extend(fold_returns)

        # Overall metrics
        all_returns = np.array(all_returns)
        results['overall'] = compute_performance_metrics(
            all_returns, self.annualization_factor
        ).to_dict()

        # Equity curve
        results['equity_curve'] = compute_equity_curve(
            all_returns, self.initial_capital
        ).tolist()

        # Per-regime metrics
        if regime_col and regime_col in predictions_df.columns:
            regime_labels = predictions_df[regime_col].values
            # Align with all_returns (flatten folds)
            sorted_df = predictions_df.sort_values([fold_col, 'bar_idx'])
            aligned_labels = sorted_df[regime_col].values

            if len(aligned_labels) == len(all_returns):
                per_regime = compute_per_regime_metrics(
                    all_returns, aligned_labels, self.annualization_factor
                )
                results['per_regime'] = {
                    k: v.to_dict() for k, v in per_regime.items()
                }

        return results
