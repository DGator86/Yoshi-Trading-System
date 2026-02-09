#!/usr/bin/env python3
"""Supervised learning training CLI.

Usage:
    python -m gnosis.train.supervised --config configs/training.yaml
    python -m gnosis.train.supervised --config configs/training.yaml --horizons 1,4,8
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from gnosis.training.config import TrainingConfig, SupervisedConfig, RunMetadata
from gnosis.training.supervised import SupervisedTrainer
from gnosis.datasets import (
    generate_labels,
    create_purged_splits,
    FeatureEngineer,
    FeatureConfig,
)


def load_data(config: TrainingConfig) -> pd.DataFrame:
    """Load and prepare data for training."""
    data_path = Path(config.data_path)

    if not data_path.exists():
        # Try to generate synthetic data
        from gnosis.ingest import load_or_create_prints
        prints_df = load_or_create_prints(
            parquet_path=str(data_path),
            symbols=config.symbols,
            days=config.train_days + config.val_days + config.test_days,
        )
    else:
        prints_df = pd.read_parquet(data_path)

    # Filter symbols
    if config.symbols:
        prints_df = prints_df[prints_df['symbol'].isin(config.symbols)]

    return prints_df


def prepare_features(
    bars_df: pd.DataFrame,
    config: TrainingConfig
) -> pd.DataFrame:
    """Prepare features and labels."""
    from gnosis.domains import compute_features
    from gnosis.regimes import KPCOFGSClassifier
    from gnosis.particle import ParticleState

    # Compute base features
    df = compute_features(bars_df)

    # Add regime features
    classifier = KPCOFGSClassifier({})
    df = classifier.classify(df)

    # Add particle state
    particle = ParticleState({})
    df = particle.compute(df)

    # Generate labels
    df = generate_labels(
        df,
        horizons=config.supervised.horizons,
        include_triple_barrier=True,
    )

    return df


def run_supervised_training(
    config: TrainingConfig,
    horizons: Optional[List[int]] = None,
    output_dir: Optional[str] = None,
) -> dict:
    """Run supervised training pipeline.

    Args:
        config: Training configuration
        horizons: Override horizons from config
        output_dir: Override output directory

    Returns:
        Dictionary with training results
    """
    # Setup
    if horizons:
        config.supervised.horizons = horizons

    run_dir = Path(output_dir or config.runs_dir) / config.experiment_name / 'supervised'
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create run metadata
    metadata = RunMetadata.create(
        experiment_name=config.experiment_name,
        stage='supervised',
        config=config.to_dict(),
    )

    print(f"Run ID: {metadata.run_id}")
    print(f"Output: {run_dir}")

    # Load data
    print("\nLoading data...")
    prints_df = load_data(config)
    print(f"  Loaded {len(prints_df)} trades")

    # Create bars
    print("\nCreating bars...")
    from gnosis.datasets import create_bars, compute_bar_features
    bars_df = create_bars(
        prints_df,
        bar_type=config.bar_type,
        n_ticks=config.n_ticks,
    )
    bars_df = compute_bar_features(bars_df)
    print(f"  Created {len(bars_df)} bars")

    # Prepare features
    print("\nPreparing features...")
    df = prepare_features(bars_df, config)
    print(f"  Features: {df.shape[1]} columns")

    # Create splits
    print("\nCreating walk-forward splits...")
    splits = create_purged_splits(
        df,
        n_splits=config.n_outer_folds,
        train_size=config.train_days * len(df) // 30,  # Approximate
        test_size=config.test_days * len(df) // 30,
        purge_gap=config.purge_bars,
        embargo_gap=config.embargo_bars,
        horizons=config.supervised.horizons,
    )
    print(f"  Created {len(splits)} folds")

    # Extract feature columns
    feature_engineer = FeatureEngineer(FeatureConfig())
    _, feature_names = feature_engineer.fit_transform(df)
    feature_schema_hash = feature_engineer.get_schema_hash()

    print(f"  Using {len(feature_names)} features")

    # Train on each fold
    all_metrics = []
    all_predictions = []

    for split in splits:
        print(f"\n--- Fold {split.fold_idx} ---")
        print(f"  Train: {len(split.train_indices)} samples")
        print(f"  Test: {len(split.test_indices)} samples")

        train_df = df.iloc[split.train_indices].copy()
        test_df = df.iloc[split.test_indices].copy()

        # Train supervised model
        trainer = SupervisedTrainer(config.supervised)
        trainer.fit(train_df, feature_names, feature_schema_hash)

        # Calibrate on validation portion of train
        val_size = len(train_df) // 5
        if val_size > 0:
            val_df = train_df.iloc[-val_size:]
            trainer.calibrate(val_df)

        # Evaluate on test
        fold_metrics = trainer.evaluate(test_df)
        all_metrics.append({
            'fold_idx': split.fold_idx,
            **{f'{h}_{k}': v for h, hm in fold_metrics.items() for k, v in hm.items()}
        })

        # Generate predictions
        preds = trainer.predict(test_df)
        preds['fold_idx'] = split.fold_idx
        all_predictions.append(preds)

        # Print metrics
        for horizon, hm in fold_metrics.items():
            print(f"  {horizon}:")
            for k, v in hm.items():
                print(f"    {k}: {v:.4f}")

        # Save fold model
        trainer.save(str(run_dir / f'fold_{split.fold_idx}'))

    # Aggregate results
    metrics_df = pd.DataFrame(all_metrics)
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    # Compute overall metrics
    overall_metrics = {}
    for col in metrics_df.columns:
        if col != 'fold_idx':
            overall_metrics[f'{col}_mean'] = float(metrics_df[col].mean())
            overall_metrics[f'{col}_std'] = float(metrics_df[col].std())

    # Save results
    print("\nSaving results...")
    metrics_df.to_csv(run_dir / 'fold_metrics.csv', index=False)
    predictions_df.to_parquet(run_dir / 'predictions.parquet', index=False)

    with open(run_dir / 'overall_metrics.json', 'w') as f:
        json.dump(overall_metrics, f, indent=2)

    # Update metadata
    metadata.completed_at = datetime.utcnow().isoformat()
    metadata.status = 'completed'
    metadata.metrics = overall_metrics
    metadata.save(str(run_dir / 'run_metadata.json'))

    # Save config
    config.save_yaml(str(run_dir / 'config.yaml'))

    print(f"\nTraining complete!")
    print(f"Results saved to: {run_dir}")

    return {
        'run_id': metadata.run_id,
        'metrics': overall_metrics,
        'output_dir': str(run_dir),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Train supervised learning models for price prediction'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/training.yaml',
        help='Path to training configuration YAML'
    )
    parser.add_argument(
        '--horizons',
        type=str,
        default=None,
        help='Comma-separated list of horizons (e.g., 1,4,8,16)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Override output directory'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='Override experiment name'
    )

    args = parser.parse_args()

    # Load or create config
    config_path = Path(args.config)
    if config_path.exists():
        config = TrainingConfig.from_yaml(str(config_path))
    else:
        print(f"Config not found at {config_path}, using defaults")
        config = TrainingConfig()

    # Apply overrides
    if args.experiment_name:
        config.experiment_name = args.experiment_name

    horizons = None
    if args.horizons:
        horizons = [int(h) for h in args.horizons.split(',')]

    # Run training
    result = run_supervised_training(
        config,
        horizons=horizons,
        output_dir=args.output_dir,
    )

    print(f"\nRun ID: {result['run_id']}")


if __name__ == '__main__':
    main()
