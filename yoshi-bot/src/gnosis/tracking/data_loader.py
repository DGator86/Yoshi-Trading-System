"""Data loading utilities for predictions, stats, and metrics."""

import json
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np


@dataclass
class PredictionStats:
    """Statistics for predictions."""
    total_predictions: int
    coverage_90: float
    coverage_80: float
    coverage_50: float
    sharpness_mean: float
    directional_accuracy: float
    mae: float
    rmse: float
    calibration_error: float
    abstention_rate: float
    brier_score: Optional[float] = None


@dataclass
class TradeStats:
    """Trading performance statistics."""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    avg_trade_return: float


@dataclass
class RegimeSnapshot:
    """Current regime probabilities for a symbol."""
    symbol: str
    timestamp: str
    K_prob: float  # Knowledge regime
    P_prob: float  # Prediction regime
    C_prob: float  # Confirmation regime
    O_prob: float  # Overextension regime
    F_prob: float  # Fatigue regime
    G_prob: float  # Gravity regime
    S_prob: float  # Stability regime


class PredictionDataLoader:
    """Load prediction data from parquet and JSON files."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)

    def load_predictions(self) -> pd.DataFrame:
        """Load predictions.parquet file."""
        predictions_file = self.run_dir / "predictions.parquet"
        if not predictions_file.exists():
            raise FileNotFoundError(f"Predictions file not found: {predictions_file}")

        df = pd.read_parquet(predictions_file)
        df['timestamp_end'] = pd.to_datetime(df['timestamp_end'])
        return df

    def load_trades(self) -> pd.DataFrame:
        """Load trades.parquet file."""
        trades_file = self.run_dir / "trades.parquet"
        if not trades_file.exists():
            return pd.DataFrame()  # Empty if trades don't exist

        df = pd.read_parquet(trades_file)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

    def load_equity_curve(self) -> pd.DataFrame:
        """Load equity_curve.parquet file."""
        equity_file = self.run_dir / "equity_curve.parquet"
        if not equity_file.exists():
            return pd.DataFrame()

        df = pd.read_parquet(equity_file)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

    def load_stats(self) -> dict[str, Any]:
        """Load backtest stats.json."""
        stats_file = self.run_dir / "backtest" / "stats.json"
        if not stats_file.exists():
            # Try root level
            stats_file = self.run_dir / "stats.json"

        if stats_file.exists():
            with open(stats_file) as f:
                return json.load(f)
        return {}

    def load_report(self) -> dict[str, Any]:
        """Load report.json."""
        report_file = self.run_dir / "report.json"
        if not report_file.exists():
            return {}

        with open(report_file) as f:
            return json.load(f)

    def load_regime_snapshot(self) -> dict[str, Any]:
        """Load regime_snapshot.json."""
        regime_file = self.run_dir / "regime_snapshot.json"
        if not regime_file.exists():
            return {}

        with open(regime_file) as f:
            return json.load(f)

    def extract_prediction_stats(self) -> PredictionStats:
        """Extract key prediction statistics from report."""
        report = self.load_report()

        return PredictionStats(
            total_predictions=report.get('total_predictions', 0),
            coverage_90=report.get('coverage_90', 0),
            coverage_80=report.get('coverage_80', 0),
            coverage_50=report.get('coverage_50', 0),
            sharpness_mean=report.get('sharpness_mean', 0),
            directional_accuracy=report.get('directional_accuracy', 0),
            mae=report.get('mae', 0),
            rmse=report.get('rmse', 0),
            calibration_error=report.get('calibration_error', 0),
            abstention_rate=report.get('abstention_rate', 0),
            brier_score=report.get('brier_score', None),
        )

    def extract_trade_stats(self) -> TradeStats:
        """Extract key trading statistics from stats."""
        stats = self.load_stats()

        return TradeStats(
            total_return=stats.get('total_return_pct', 0),
            annualized_return=stats.get('annualized_return_pct', 0),
            sharpe_ratio=stats.get('sharpe_ratio', 0),
            sortino_ratio=stats.get('sortino_ratio', 0),
            calmar_ratio=stats.get('calmar_ratio', 0),
            max_drawdown=stats.get('max_drawdown_pct', 0),
            win_rate=stats.get('win_rate', 0),
            profit_factor=stats.get('profit_factor', 0),
            num_trades=stats.get('num_trades', 0),
            avg_trade_return=stats.get('avg_trade_return_pct', 0),
        )

    def calculate_accuracy_by_confidence(self) -> dict[str, float]:
        """Calculate directional accuracy bucketed by prediction confidence."""
        predictions = self.load_predictions()

        if 'S_pmax_calibrated' not in predictions.columns or 'future_return' not in predictions.columns:
            return {}

        # Filter out abstained predictions
        preds_valid = predictions[~predictions.get('abstain', False)]

        if len(preds_valid) == 0:
            return {}

        # Bucket by confidence level
        bins = [0, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
        preds_valid['confidence_bin'] = pd.cut(
            preds_valid['S_pmax_calibrated'],
            bins=bins,
            labels=[f"{bins[i]:.0%}-{bins[i+1]:.0%}" for i in range(len(bins)-1)]
        )

        accuracy_by_bin = preds_valid.groupby('confidence_bin', observed=True).apply(
            lambda g: (np.sign(g['future_return']) == np.sign(g['q50'] - g['close'])).mean()
        )

        return accuracy_by_bin.to_dict()

    def calculate_accuracy_decay(self) -> dict[str, float]:
        """Calculate directional accuracy degradation over time horizon."""
        predictions = self.load_predictions()

        if 'bar_idx' not in predictions.columns or 'future_return' not in predictions.columns:
            return {}

        # Group predictions by time horizon into buckets
        max_horizon = predictions['bar_idx'].max()
        horizons = [1, 5, 10, 20, 50]

        decay = {}
        for h in horizons:
            if h > max_horizon:
                continue
            subset = predictions[predictions['bar_idx'] <= max_horizon - h]
            if len(subset) > 0:
                acc = (np.sign(subset['future_return']) == np.sign(subset['q50'] - subset['close'])).mean()
                decay[f"{h}_bar"] = float(acc)

        return decay


class MetricsCalculator:
    """Calculate derived metrics from raw data."""

    @staticmethod
    def calculate_calibration_curve(predictions: pd.DataFrame) -> tuple[list[float], list[float]]:
        """Calculate observed vs predicted probability curve."""
        if 'S_pmax_calibrated' not in predictions.columns:
            return [], []

        preds_valid = predictions[~predictions.get('abstain', False)]

        if len(preds_valid) == 0:
            return [], []

        # Bin predictions
        bins = np.linspace(0, 1, 11)
        bin_means = []
        bin_accs = []

        for i in range(len(bins) - 1):
            mask = (preds_valid['S_pmax_calibrated'] >= bins[i]) & (preds_valid['S_pmax_calibrated'] < bins[i+1])
            if mask.sum() > 0:
                subset = preds_valid[mask]
                acc = (np.sign(subset['future_return']) == np.sign(subset['q50'] - subset['close'])).mean()
                bin_means.append((bins[i] + bins[i+1]) / 2)
                bin_accs.append(float(acc))

        return bin_means, bin_accs

    @staticmethod
    def calculate_equity_metrics(equity_curve: pd.DataFrame) -> dict[str, float]:
        """Calculate equity curve statistics."""
        if equity_curve.empty or 'equity' not in equity_curve.columns:
            return {}

        eq = equity_curve['equity'].values
        returns = np.diff(eq) / eq[:-1]

        return {
            'max_equity': float(eq.max()),
            'min_equity': float(eq.min()),
            'final_equity': float(eq[-1]),
            'volatility': float(np.std(returns)),
            'skewness': float(pd.Series(returns).skew()),
            'kurtosis': float(pd.Series(returns).kurtosis()),
        }
