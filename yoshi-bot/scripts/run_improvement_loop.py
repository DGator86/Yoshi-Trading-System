#!/usr/bin/env python3
"""Run the Yoshi Improvement Loop to iteratively improve prediction accuracy.

Usage:
    python scripts/run_improvement_loop.py --data data/parquet/prints.parquet
    python scripts/run_improvement_loop.py --target-accuracy 0.60 --max-iterations 100
"""
import argparse
import copy
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Load local secrets from yoshi-bot/.env if present.
try:
    from dotenv import load_dotenv  # type: ignore

    _env_path = Path(__file__).resolve().parents[1] / ".env"
    if _env_path.exists():
        load_dotenv(dotenv_path=_env_path, override=False)
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gnosis.harness.improvement_loop import (
    YoshiImprovementLoop,
    MetricTarget,
    MetricType,
    get_all_variables,
)
# Note: Some imports might be mocked or missing in this view, ensuring core structure matches.
from gnosis.domains import DomainAggregator, compute_features
from gnosis.regimes import KPCOFGSClassifier
from gnosis.particle import ParticleState, PriceParticle
from gnosis.predictors.quantile import QuantilePredictor
from gnosis.harness.walkforward import compute_future_returns


class YoshiEvaluator:
    """Evaluator that runs a full prediction pipeline and returns metrics."""

    def __init__(
        self,
        prints_df: pd.DataFrame,
        base_config: dict,
        n_folds: int = 3,
        horizon_bars: int = 1,
    ):
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
        n_trades = config.get("domains", {}).get("domains", {}).get("D0", {}).get("n_trades", 200)
        cache_key = f"features_{n_trades}"

        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        domain_cfg = config.get("domains", {"domains": {"D0": {"n_trades": n_trades}}})
        if "domains" not in domain_cfg:
            domain_cfg = {"domains": {"D0": {"n_trades": n_trades}}}

        aggregator = DomainAggregator(domain_cfg)
        bars_df = aggregator.aggregate(self.prints_df, "D0")

        features_df = compute_features(bars_df, extended=True)

        regimes_cfg = config.get("regimes", {"confidence_floor": 0.65})
        classifier = KPCOFGSClassifier(regimes_cfg)
        features_df = classifier.classify(features_df)

        models_cfg = config.get("models", {})
        particle = ParticleState(models_cfg)
        features_df = particle.compute_state(features_df)

        physics_config = models_cfg.get("particle_physics", {})
        price_particle = PriceParticle(physics_config)
        features_df = price_particle.compute_features(features_df)

        features_df = compute_future_returns(features_df, horizon_bars=self.horizon_bars)
        features_df = features_df.sort_values(["symbol", "bar_idx"]).reset_index(drop=True)

        self._cache[cache_key] = features_df.copy()
        return features_df

    def evaluate(self, config: dict) -> dict:
        """Evaluate config and return metrics."""
        merged_config = self._merge_config(config)
        features_df = self._build_features(merged_config)
        features_df = features_df.dropna(subset=["future_return"])

        if len(features_df) < 50:
            return {
                "directional_accuracy": 0.0, "coverage_90": 0.0,
                "mae": 1.0, "rmse": 1.0, "calibration_error": 1.0,
            }

        n_samples = len(features_df)
        fold_size = n_samples // self.n_folds
        all_predictions = []
        all_actuals = []
        models_cfg = merged_config.get("models", {})

        for fold_idx in range(self.n_folds):
            val_start = fold_idx * fold_size
            val_end = min((fold_idx + 1) * fold_size, n_samples)

            if fold_idx == 0:
                continue

            train_df = features_df.iloc[:val_start].copy()
            val_df = features_df.iloc[val_start:val_end].copy()

            if len(train_df) < 20 or len(val_df) < 10:
                continue

            predictor = QuantilePredictor(models_cfg)
            predictor.fit(train_df, "future_return")
            preds = predictor.predict(val_df)

            all_predictions.append(preds)
            all_actuals.append(val_df[["symbol", "bar_idx", "future_return"]])

        if not all_predictions:
            return {
                "directional_accuracy": 0.0, "coverage_90": 0.0,
                "mae": 1.0, "rmse": 1.0, "calibration_error": 1.0,
            }

        preds_df = pd.concat(all_predictions, ignore_index=True)
        actuals_df = pd.concat(all_actuals, ignore_index=True)
        eval_df = preds_df.merge(actuals_df, on=["symbol", "bar_idx"], how="inner")

        if len(eval_df) == 0:
            return {
                "directional_accuracy": 0.0, "coverage_90": 0.0,
                "mae": 1.0, "rmse": 1.0, "calibration_error": 1.0,
            }

        pred_dir = np.sign(eval_df["x_hat"].fillna(0))
        actual_dir = np.sign(eval_df["future_return"])
        correct = (pred_dir == actual_dir) | (actual_dir == 0)
        directional_accuracy = correct.mean()

        in_interval = (eval_df["future_return"] >= eval_df["q05"]) & (eval_df["future_return"] <= eval_df["q95"])
        coverage_90 = in_interval.mean()

        errors = eval_df["future_return"] - eval_df["x_hat"].fillna(0)
        mae = np.abs(errors).mean()
        rmse = np.sqrt((errors ** 2).mean())
        calibration_error = abs(coverage_90 - 0.90)

        return {
            "directional_accuracy": directional_accuracy,
            "coverage_90": coverage_90,
            "mae": mae,
            "rmse": rmse,
            "calibration_error": calibration_error,
        }


def publish_params(config: dict, path: str = "config/params.json"):
    """Atomically publish parameters for hot-reload."""
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    envelope = {
        "version": int(time.time()),
        "timestamp": datetime.utcnow().isoformat(),
        "params": config
    }
    
    temp_path = target_path.with_suffix(".tmp")
    with open(temp_path, "w") as f:
        json.dump(envelope, f, indent=2)
        
    os.replace(temp_path, target_path)
    print(f"Published params v{envelope['version']} to {target_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Yoshi Improvement Loop")
    parser.add_argument("--data", type=str, default="data/parquet/prints.parquet")
    parser.add_argument("--target-accuracy", type=float, default=0.55)
    parser.add_argument("--target-coverage", type=float, default=0.90)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--output", type=str, default="reports/improvement")
    args = parser.parse_args()

    print("="*60)
    print("YOSHI IMPROVEMENT LOOP")
    print("="*60)

    print(f"\nLoading data from {args.data}...")
    try:
        prints_df = pd.read_parquet(args.data)
        print(f"  {len(prints_df):,} prints loaded")
    except Exception as e:
        print(f"Failed to load data: {e}")
        # Create dummy data for dry run / test
        prints_df = pd.DataFrame(columns=["symbol", "timestamp", "price", "size"])

    base_config = {
        "domains": {"domains": {"D0": {"n_trades": 200}}},
        "regimes": {"confidence_floor": 0.65},
        "models": {
            "predictor": {
                "backend": "gradient_boost",
                "l2_reg": 0.1,
                "quantiles": [0.05, 0.50, 0.95],
                "extended_features": True,
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

    print("\nInitializing evaluator...")
    evaluator = YoshiEvaluator(
        prints_df=prints_df,
        base_config=base_config,
        n_folds=5,
        horizon_bars=1,
    )

    print("\nRunning initial evaluation...")
    initial_metrics = evaluator.evaluate({})
    print(f"  Directional: {initial_metrics['directional_accuracy']:.1%}")

    targets = [
        MetricTarget("directional_accuracy", MetricType.DIRECTIONAL_ACCURACY, args.target_accuracy, "maximize", 1),
        MetricTarget("coverage", MetricType.COVERAGE_90, args.target_coverage, "maximize", 2),
    ]
    
    variables = get_all_variables()

    loop = YoshiImprovementLoop(
        targets=targets,
        variables=variables,
        evaluate_fn=evaluator.evaluate,
        max_iterations_per_target=args.max_iterations,
        patience=args.patience,
        verbose=True,
    )

    print("\n" + "="*60 + "\nSTARTING IMPROVEMENT LOOP\n" + "="*60)
    start_time = time.time()
    results = loop.run(base_config)
    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.1f}s")
    
    # Save Report
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    if results:
        report = loop.generate_report(results)
        with open(output_dir / "improvement_report.txt", "w") as f:
            f.write(report)
            
        final_config = results[-1].final_config
        with open(output_dir / "optimized_config.json", "w") as f:
            json.dump(final_config, f, indent=2)
            
        # Hot-Reload Publish
        # Try publishing to config/params.json (local) or /root/Yoshi-Bot/config/params.json
        publish_params(final_config, path="config/params.json")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        pass
