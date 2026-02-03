#!/usr/bin/env python3
"""Run the Yoshi Improvement Loop to iteratively improve prediction accuracy.

Usage:
    python scripts/run_improvement_loop.py --data data/parquet/prints.parquet
    python scripts/run_improvement_loop.py --target-accuracy 0.60 --max-iterations 100
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gnosis.harness.improvement_loop import (
    YoshiImprovementLoop,
    MetricTarget,
    MetricType,
    Variable,
    get_all_variables,
    get_predictor_variables,
    get_default_targets,
)
from gnosis.domains import DomainAggregator, compute_features
from gnosis.regimes import KPCOFGSClassifier
from gnosis.particle import ParticleState, PriceParticle
from gnosis.predictors.quantile import QuantilePredictor
from gnosis.harness.walkforward import compute_future_returns, WalkForwardHarness
from gnosis.evaluation.accuracy import PredictionEvaluator


class YoshiEvaluator:
    """Evaluator that runs a full prediction pipeline and returns metrics."""

    def __init__(
        self,
        prints_df: pd.DataFrame,
        base_config: dict,
        n_folds: int = 3,
        horizon_bars: int = 1,
    ):
        """Initialize evaluator.

        Args:
            prints_df: Raw print/trade data
            base_config: Base configuration dict
            n_folds: Number of walk-forward folds
            horizon_bars: Prediction horizon in bars
        """
        self.prints_df = prints_df
        self.base_config = base_config
        self.n_folds = n_folds
        self.horizon_bars = horizon_bars
        self._cache = {}

    def _merge_config(self, overrides: dict) -> dict:
        """Merge overrides into base config."""
        config = copy.deepcopy(self.base_config)

        def deep_merge(base, override):
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value

        deep_merge(config, overrides)
        return config

    def _build_features(self, config: dict) -> pd.DataFrame:
        """Build features from prints using config."""
        # Check cache
        n_trades = config.get("domains", {}).get("domains", {}).get("D0", {}).get("n_trades", 200)
        cache_key = f"features_{n_trades}"

        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        # Build domain config
        domain_cfg = config.get("domains", {"domains": {"D0": {"n_trades": n_trades}}})
        if "domains" not in domain_cfg:
            domain_cfg = {"domains": {"D0": {"n_trades": n_trades}}}

        # 1. Aggregate prints
        aggregator = DomainAggregator(domain_cfg)
        bars_df = aggregator.aggregate(self.prints_df, "D0")

        # 2. Compute basic features
        features_df = compute_features(bars_df, extended=True)

        # 3. Classify regimes
        regimes_cfg = config.get("regimes", {"confidence_floor": 0.65})
        classifier = KPCOFGSClassifier(regimes_cfg)
        features_df = classifier.classify(features_df)

        # 4. Compute particle state (basic)
        models_cfg = config.get("models", {})
        particle = ParticleState(models_cfg)
        features_df = particle.compute_state(features_df)

        # 5. Compute particle physics features (advanced)
        physics_config = models_cfg.get("particle_physics", {})
        price_particle = PriceParticle(physics_config)
        features_df = price_particle.compute_features(features_df)

        # 6. Compute future returns
        features_df = compute_future_returns(features_df, horizon_bars=self.horizon_bars)

        # Sort
        features_df = features_df.sort_values(["symbol", "bar_idx"]).reset_index(drop=True)

        # Cache
        self._cache[cache_key] = features_df.copy()

        return features_df

    def evaluate(self, config: dict) -> dict:
        """Evaluate config and return metrics.

        Args:
            config: Configuration dict with hyperparameters

        Returns:
            Dict with metrics:
                - directional_accuracy
                - coverage_90
                - mae
                - rmse
                - calibration_error
        """
        merged_config = self._merge_config(config)
        features_df = self._build_features(merged_config)

        # Drop NaN targets
        features_df = features_df.dropna(subset=["future_return"])

        if len(features_df) < 50:
            return {
                "directional_accuracy": 0.0,
                "coverage_90": 0.0,
                "mae": 1.0,
                "rmse": 1.0,
                "calibration_error": 1.0,
            }

        # Walk-forward validation
        n_samples = len(features_df)
        fold_size = n_samples // self.n_folds

        all_predictions = []
        all_actuals = []

        models_cfg = merged_config.get("models", {})

        for fold_idx in range(self.n_folds):
            # Train on first folds, validate on current fold
            val_start = fold_idx * fold_size
            val_end = min((fold_idx + 1) * fold_size, n_samples)

            if fold_idx == 0:
                # Can't validate on first fold without training data
                continue

            train_df = features_df.iloc[:val_start].copy()
            val_df = features_df.iloc[val_start:val_end].copy()

            if len(train_df) < 20 or len(val_df) < 10:
                continue

            # Train predictor
            predictor = QuantilePredictor(models_cfg)
            predictor.fit(train_df, "future_return")

            # Predict
            preds = predictor.predict(val_df)

            # Store predictions and actuals
            all_predictions.append(preds)
            all_actuals.append(val_df[["symbol", "bar_idx", "future_return"]])

        if not all_predictions:
            return {
                "directional_accuracy": 0.0,
                "coverage_90": 0.0,
                "mae": 1.0,
                "rmse": 1.0,
                "calibration_error": 1.0,
            }

        # Combine all predictions
        preds_df = pd.concat(all_predictions, ignore_index=True)
        actuals_df = pd.concat(all_actuals, ignore_index=True)

        # Merge predictions with actuals
        eval_df = preds_df.merge(
            actuals_df,
            on=["symbol", "bar_idx"],
            how="inner",
        )

        if len(eval_df) == 0:
            return {
                "directional_accuracy": 0.0,
                "coverage_90": 0.0,
                "mae": 1.0,
                "rmse": 1.0,
                "calibration_error": 1.0,
            }

        # Compute metrics
        # Directional accuracy
        pred_dir = np.sign(eval_df["x_hat"].fillna(0))
        actual_dir = np.sign(eval_df["future_return"])
        correct = (pred_dir == actual_dir) | (actual_dir == 0)
        directional_accuracy = correct.mean()

        # Coverage
        in_interval = (
            (eval_df["future_return"] >= eval_df["q05"]) &
            (eval_df["future_return"] <= eval_df["q95"])
        )
        coverage_90 = in_interval.mean()

        # MAE
        errors = eval_df["future_return"] - eval_df["x_hat"].fillna(0)
        mae = np.abs(errors).mean()

        # RMSE
        rmse = np.sqrt((errors ** 2).mean())

        # Calibration error (simplified)
        calibration_error = abs(coverage_90 - 0.90)

        return {
            "directional_accuracy": directional_accuracy,
            "coverage_90": coverage_90,
            "mae": mae,
            "rmse": rmse,
            "calibration_error": calibration_error,
        }


def main():
    parser = argparse.ArgumentParser(description="Run Yoshi Improvement Loop")
    parser.add_argument(
        "--data", type=str, default="data/parquet/prints.parquet",
        help="Path to prints parquet file"
    )
    parser.add_argument(
        "--target-accuracy", type=float, default=0.55,
        help="Target directional accuracy (default: 0.55)"
    )
    parser.add_argument(
        "--target-coverage", type=float, default=0.90,
        help="Target 90%% coverage (default: 0.90)"
    )
    parser.add_argument(
        "--max-iterations", type=int, default=50,
        help="Max iterations per target (default: 50)"
    )
    parser.add_argument(
        "--patience", type=int, default=10,
        help="Stop if no improvement for N iterations (default: 10)"
    )
    parser.add_argument(
        "--output", type=str, default="reports/improvement",
        help="Output directory for results"
    )
    args = parser.parse_args()

    print("="*60)
    print("YOSHI IMPROVEMENT LOOP")
    print("="*60)

    # Load data
    print(f"\nLoading data from {args.data}...")
    prints_df = pd.read_parquet(args.data)
    print(f"  {len(prints_df):,} prints loaded")

    # Base config
    base_config = {
        "domains": {
            "domains": {
                "D0": {"n_trades": 200}
            }
        },
        "regimes": {
            "confidence_floor": 0.65
        },
        "models": {
            "predictor": {
                "backend": "gradient_boost",  # Best for nonlinear physics features
                "l2_reg": 0.1,
                "quantiles": [0.05, 0.50, 0.95],
                "extended_features": True,  # Enable physics features
                "normalize": True,
            },
            "particle": {
                "flow_span": 10,
                "flow_weight": 1.0,
                "regime_weight": 1.0,
                "barrier_weight": 1.0,
            },
            "particle_physics": {
                "velocity_span": 5,
                "acceleration_span": 3,
                "potential_lookback": 50,
            }
        }
    }

    # Create evaluator
    print("\nInitializing evaluator...")
    evaluator = YoshiEvaluator(
        prints_df=prints_df,
        base_config=base_config,
        n_folds=5,
        horizon_bars=1,
    )

    # Initial evaluation
    print("\nRunning initial evaluation...")
    initial_metrics = evaluator.evaluate({})
    print(f"  Directional accuracy: {initial_metrics['directional_accuracy']:.1%}")
    print(f"  Coverage (90%%): {initial_metrics['coverage_90']:.1%}")
    print(f"  MAE: {initial_metrics['mae']:.4f}")

    # Define targets
    targets = [
        MetricTarget(
            name="directional_accuracy",
            metric_type=MetricType.DIRECTIONAL_ACCURACY,
            target_value=args.target_accuracy,
            direction="maximize",
            priority=1,
        ),
        MetricTarget(
            name="coverage",
            metric_type=MetricType.COVERAGE_90,
            target_value=args.target_coverage,
            direction="maximize",
            priority=2,
        ),
    ]

    # Define variables to tune
    variables = get_all_variables()

    # Create improvement loop
    loop = YoshiImprovementLoop(
        targets=targets,
        variables=variables,
        evaluate_fn=evaluator.evaluate,
        max_iterations_per_target=args.max_iterations,
        patience=args.patience,
        verbose=True,
    )

    # Run the loop
    print("\n" + "="*60)
    print("STARTING IMPROVEMENT LOOP")
    print("="*60)
    start_time = time.time()

    results = loop.run(base_config)

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.1f}s")

    # Generate report
    report = loop.generate_report(results)
    print("\n" + report)

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save report
    report_path = output_dir / "improvement_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")

    # Save final config
    if results:
        final_config = results[-1].final_config
        config_path = output_dir / "optimized_config.json"
        with open(config_path, "w") as f:
            json.dump(final_config, f, indent=2)
        print(f"Optimized config saved to {config_path}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for result in results:
        status = "✓" if result.target_achieved else "✗"
        print(f"{status} {result.target_name}: {result.final_value:.4f} (target: {result.target_value})")

    # Return exit code based on success
    all_achieved = all(r.target_achieved for r in results)
    return 0 if all_achieved else 1


if __name__ == "__main__":
    sys.exit(main())
