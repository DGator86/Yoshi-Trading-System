"""Prediction accuracy evaluation for Yoshi forecasting system.

Evaluates prediction quality across multiple time horizons with metrics:
- Directional accuracy (correct up/down prediction)
- Interval coverage (actuals within predicted range)
- Point prediction error (MAE, RMSE)
- Calibration (predicted probabilities match actual frequencies)
- Accuracy decay over time horizons
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class AccuracyMetrics:
    """Prediction accuracy metrics for a single horizon."""

    horizon_bars: int
    horizon_time: str  # Human-readable (e.g., "1.2 days")

    # Directional accuracy
    directional_accuracy: float  # % correct up/down predictions
    directional_accuracy_long: float  # % correct when predicting up
    directional_accuracy_short: float  # % correct when predicting down

    # Interval coverage
    coverage_90: float  # % of actuals within 90% interval
    coverage_80: float  # % of actuals within 80% interval
    coverage_50: float  # % of actuals within 50% interval

    # Point prediction error
    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Squared Error
    mape: float  # Mean Absolute Percentage Error

    # Calibration
    calibration_error: float  # ECE (Expected Calibration Error)

    # Sample size
    n_predictions: int
    n_non_abstain: int


@dataclass
class TimeDecayAnalysis:
    """Analysis of how accuracy decays over time."""

    horizons: List[AccuracyMetrics]
    decay_rate: float  # % accuracy lost per day
    half_life_bars: float  # Bars until accuracy drops to 50%
    optimal_horizon: int  # Best horizon for accuracy/time tradeoff


class PredictionEvaluator:
    """Evaluate prediction accuracy across time horizons.

    Example output:
        Horizon    Time       Directional   Coverage   MAE
        -------    ----       -----------   --------   ---
        1 bar      0.1 days   72.3%         91.2%      0.23%
        5 bars     0.5 days   65.1%         89.5%      0.45%
        10 bars    1.0 days   58.7%         87.3%      0.78%
        20 bars    2.0 days   52.4%         85.1%      1.12%
    """

    def __init__(
        self,
        horizons: List[int] = None,
        bars_per_day: float = 10.0,  # Approximate bars per day
    ):
        """Initialize evaluator.

        Args:
            horizons: List of bar horizons to evaluate (default: [1, 5, 10, 20, 50])
            bars_per_day: Average number of bars per day for time conversion
        """
        self.horizons = horizons or [1, 3, 5, 10, 20, 50]
        self.bars_per_day = bars_per_day

    def _bars_to_time(self, bars: int) -> str:
        """Convert bars to human-readable time string."""
        days = bars / self.bars_per_day

        if days < 1/24:
            minutes = days * 24 * 60
            return f"{minutes:.0f} min"
        elif days < 1:
            hours = days * 24
            return f"{hours:.1f} hours"
        else:
            return f"{days:.1f} days"

    def evaluate_horizon(
        self,
        predictions_df: pd.DataFrame,
        actuals_df: pd.DataFrame,
        horizon_bars: int,
    ) -> AccuracyMetrics:
        """Evaluate prediction accuracy for a specific horizon.

        Args:
            predictions_df: DataFrame with q05, q50, q95, x_hat, abstain
            actuals_df: DataFrame with actual future returns
            horizon_bars: Number of bars ahead being predicted

        Returns:
            AccuracyMetrics for this horizon
        """
        # Merge predictions with actuals
        df = self._align_predictions_actuals(predictions_df, actuals_df, horizon_bars)

        if len(df) == 0:
            return self._empty_metrics(horizon_bars)

        # Filter non-abstain predictions
        non_abstain = df[~df["abstain"]] if "abstain" in df.columns else df
        n_non_abstain = len(non_abstain)

        if n_non_abstain == 0:
            return self._empty_metrics(horizon_bars)

        # === DIRECTIONAL ACCURACY ===
        pred_direction = np.sign(non_abstain["x_hat"])
        actual_direction = np.sign(non_abstain["actual_return"])

        correct = (pred_direction == actual_direction) | (actual_direction == 0)
        directional_accuracy = correct.mean()

        # Accuracy when predicting up vs down
        long_mask = pred_direction > 0
        short_mask = pred_direction < 0

        if long_mask.sum() > 0:
            directional_accuracy_long = correct[long_mask].mean()
        else:
            directional_accuracy_long = 0.0

        if short_mask.sum() > 0:
            directional_accuracy_short = correct[short_mask].mean()
        else:
            directional_accuracy_short = 0.0

        # === INTERVAL COVERAGE ===
        coverage_90 = self._compute_coverage(non_abstain, 0.90)
        coverage_80 = self._compute_coverage(non_abstain, 0.80)
        coverage_50 = self._compute_coverage(non_abstain, 0.50)

        # === POINT PREDICTION ERROR ===
        errors = non_abstain["actual_return"] - non_abstain["x_hat"]
        mae = np.abs(errors).mean()
        rmse = np.sqrt((errors ** 2).mean())

        # MAPE (avoid division by zero)
        actual_nonzero = non_abstain[non_abstain["actual_return"].abs() > 1e-8]
        if len(actual_nonzero) > 0:
            mape = (np.abs(actual_nonzero["actual_return"] - actual_nonzero["x_hat"]) /
                    np.abs(actual_nonzero["actual_return"])).mean()
        else:
            mape = 0.0

        # === CALIBRATION ERROR ===
        calibration_error = self._compute_calibration_error(non_abstain)

        return AccuracyMetrics(
            horizon_bars=horizon_bars,
            horizon_time=self._bars_to_time(horizon_bars),
            directional_accuracy=directional_accuracy,
            directional_accuracy_long=directional_accuracy_long,
            directional_accuracy_short=directional_accuracy_short,
            coverage_90=coverage_90,
            coverage_80=coverage_80,
            coverage_50=coverage_50,
            mae=mae,
            rmse=rmse,
            mape=mape,
            calibration_error=calibration_error,
            n_predictions=len(df),
            n_non_abstain=n_non_abstain,
        )

    def _align_predictions_actuals(
        self,
        predictions_df: pd.DataFrame,
        actuals_df: pd.DataFrame,
        horizon_bars: int,
    ) -> pd.DataFrame:
        """Align predictions with actual future returns."""
        df = predictions_df.copy()

        # Check if predictions already has future_return (horizon=1 case)
        if "future_return" in df.columns and horizon_bars == 1:
            # Use the existing future_return for 1-bar horizon
            df["actual_return"] = df["future_return"]
        elif "future_return" in actuals_df.columns and horizon_bars == 1:
            # Use pre-computed future return from actuals (1-bar only)
            if "future_return" not in df.columns:
                df = df.merge(
                    actuals_df[["symbol", "bar_idx", "future_return"]],
                    on=["symbol", "bar_idx"],
                    how="left"
                )
                df = df.rename(columns={"future_return": "actual_return"})
            else:
                df["actual_return"] = df["future_return"]
        else:
            # Compute from close prices for multi-bar horizons
            actuals = actuals_df.copy()
            actuals["future_close"] = actuals.groupby("symbol")["close"].shift(-horizon_bars)
            actuals["actual_return"] = (actuals["future_close"] - actuals["close"]) / actuals["close"]

            df = df.merge(
                actuals[["symbol", "bar_idx", "actual_return"]],
                on=["symbol", "bar_idx"],
                how="left",
                suffixes=("", "_actual")
            )
            # Handle case where actual_return might be duplicated
            if "actual_return_actual" in df.columns:
                df["actual_return"] = df["actual_return_actual"]
                df = df.drop(columns=["actual_return_actual"])

        # Drop rows without actual returns (at end of data)
        df = df.dropna(subset=["actual_return"])

        return df

    def _compute_coverage(self, df: pd.DataFrame, target_coverage: float) -> float:
        """Compute interval coverage at a given confidence level."""
        if "q05" not in df.columns or "q95" not in df.columns:
            return 0.0

        # For 90% coverage, use q05 and q95 directly
        # For other levels, interpolate
        if target_coverage == 0.90:
            lower = df["q05"]
            upper = df["q95"]
        elif target_coverage == 0.80:
            # Interpolate: 80% interval is narrower than 90%
            center = df["q50"]
            half_width_90 = (df["q95"] - df["q05"]) / 2
            # 80% is about 84% of 90% width (based on normal distribution)
            half_width_80 = half_width_90 * 0.84
            lower = center - half_width_80
            upper = center + half_width_80
        elif target_coverage == 0.50:
            center = df["q50"]
            half_width_90 = (df["q95"] - df["q05"]) / 2
            # 50% is about 51% of 90% width
            half_width_50 = half_width_90 * 0.51
            lower = center - half_width_50
            upper = center + half_width_50
        else:
            # Default to 90%
            lower = df["q05"]
            upper = df["q95"]

        in_interval = (df["actual_return"] >= lower) & (df["actual_return"] <= upper)
        return in_interval.mean()

    def _compute_calibration_error(self, df: pd.DataFrame) -> float:
        """Compute Expected Calibration Error (ECE)."""
        if "q05" not in df.columns or "q95" not in df.columns:
            return 0.0

        # Bin predictions by confidence (interval width)
        df = df.copy()
        df["interval_width"] = df["q95"] - df["q05"]
        df["in_interval"] = (df["actual_return"] >= df["q05"]) & (df["actual_return"] <= df["q95"])

        # Create bins
        try:
            df["confidence_bin"] = pd.qcut(df["interval_width"], q=10, duplicates="drop")
        except ValueError:
            return 0.0

        # Compute ECE
        ece = 0.0
        for bin_name, group in df.groupby("confidence_bin", observed=True):
            expected_coverage = 0.90  # We're using 90% intervals
            actual_coverage = group["in_interval"].mean()
            bin_weight = len(group) / len(df)
            ece += bin_weight * abs(actual_coverage - expected_coverage)

        return ece

    def _empty_metrics(self, horizon_bars: int) -> AccuracyMetrics:
        """Return empty metrics when no data available."""
        return AccuracyMetrics(
            horizon_bars=horizon_bars,
            horizon_time=self._bars_to_time(horizon_bars),
            directional_accuracy=0.0,
            directional_accuracy_long=0.0,
            directional_accuracy_short=0.0,
            coverage_90=0.0,
            coverage_80=0.0,
            coverage_50=0.0,
            mae=0.0,
            rmse=0.0,
            mape=0.0,
            calibration_error=0.0,
            n_predictions=0,
            n_non_abstain=0,
        )

    def evaluate_all_horizons(
        self,
        predictions_df: pd.DataFrame,
        actuals_df: pd.DataFrame,
    ) -> TimeDecayAnalysis:
        """Evaluate accuracy across all configured horizons.

        Args:
            predictions_df: DataFrame with predictions
            actuals_df: DataFrame with actual prices/returns

        Returns:
            TimeDecayAnalysis with metrics for each horizon
        """
        metrics_list = []

        for horizon in self.horizons:
            metrics = self.evaluate_horizon(predictions_df, actuals_df, horizon)
            metrics_list.append(metrics)

        # Compute decay rate (linear regression on accuracy vs horizon)
        if len(metrics_list) >= 2:
            horizons = np.array([m.horizon_bars for m in metrics_list])
            accuracies = np.array([m.directional_accuracy for m in metrics_list])

            # Filter valid accuracies
            valid = accuracies > 0
            if valid.sum() >= 2:
                # Linear fit: accuracy = a - decay_rate * horizon
                z = np.polyfit(horizons[valid], accuracies[valid], 1)
                decay_rate = -z[0] * self.bars_per_day  # Per day

                # Half-life: when does accuracy reach 50%?
                if z[0] != 0:
                    half_life_bars = (z[1] - 0.5) / (-z[0])
                else:
                    half_life_bars = float('inf')
            else:
                decay_rate = 0.0
                half_life_bars = float('inf')
        else:
            decay_rate = 0.0
            half_life_bars = float('inf')

        # Find optimal horizon (best accuracy with reasonable time)
        valid_metrics = [m for m in metrics_list if m.directional_accuracy > 0.5]
        if valid_metrics:
            optimal = max(valid_metrics, key=lambda m: m.directional_accuracy)
            optimal_horizon = optimal.horizon_bars
        else:
            optimal_horizon = self.horizons[0] if self.horizons else 1

        return TimeDecayAnalysis(
            horizons=metrics_list,
            decay_rate=decay_rate,
            half_life_bars=half_life_bars,
            optimal_horizon=optimal_horizon,
        )

    def generate_report(self, analysis: TimeDecayAnalysis) -> str:
        """Generate human-readable accuracy report.

        Args:
            analysis: TimeDecayAnalysis from evaluate_all_horizons

        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 70)
        lines.append("YOSHI PREDICTION ACCURACY REPORT")
        lines.append("=" * 70)
        lines.append("")

        # Summary
        if analysis.horizons:
            best = max(analysis.horizons, key=lambda m: m.directional_accuracy)
            lines.append(f"Best Accuracy: {best.directional_accuracy:.1%} at {best.horizon_time}")
            lines.append(f"Accuracy Decay Rate: {analysis.decay_rate:.1%} per day")
            if analysis.half_life_bars < float('inf'):
                lines.append(f"Half-Life: {self._bars_to_time(analysis.half_life_bars)}")
            lines.append("")

        # Table header
        lines.append("Horizon    Time         Direction   Coverage90  MAE      RMSE     Samples")
        lines.append("-" * 70)

        # Table rows
        for m in analysis.horizons:
            lines.append(
                f"{m.horizon_bars:>3} bars   {m.horizon_time:>10}   "
                f"{m.directional_accuracy:>6.1%}      {m.coverage_90:>6.1%}      "
                f"{m.mae:>6.2%}   {m.rmse:>6.2%}   {m.n_non_abstain:>6}"
            )

        lines.append("")
        lines.append("=" * 70)

        # Interpretation
        lines.append("")
        lines.append("INTERPRETATION:")
        if analysis.horizons:
            best = max(analysis.horizons, key=lambda m: m.directional_accuracy)
            if best.directional_accuracy >= 0.6:
                lines.append(f"  - Yoshi predicts with {best.directional_accuracy:.0%} accuracy")
                lines.append(f"    at t-minus {best.horizon_time}")
            else:
                lines.append("  - Prediction accuracy is below 60% (near random)")
                lines.append("  - Model improvements needed")

            if analysis.decay_rate > 0.1:
                lines.append(f"  - Accuracy decays {analysis.decay_rate:.0%} per day")
                lines.append("  - Short-term predictions recommended")

        return "\n".join(lines)


def evaluate_predictions(
    predictions_path: str,
    data_path: str,
    horizons: List[int] = None,
    bars_per_day: float = 10.0,
) -> Tuple[TimeDecayAnalysis, str]:
    """Convenience function to evaluate predictions from files.

    Args:
        predictions_path: Path to predictions parquet
        data_path: Path to bars/features parquet with close prices
        horizons: Horizons to evaluate
        bars_per_day: Bars per day conversion factor

    Returns:
        Tuple of (TimeDecayAnalysis, report_string)
    """
    predictions_df = pd.read_parquet(predictions_path)
    actuals_df = pd.read_parquet(data_path)

    evaluator = PredictionEvaluator(horizons=horizons, bars_per_day=bars_per_day)
    analysis = evaluator.evaluate_all_horizons(predictions_df, actuals_df)
    report = evaluator.generate_report(analysis)

    return analysis, report
