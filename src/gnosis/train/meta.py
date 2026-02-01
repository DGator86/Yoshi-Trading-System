#!/usr/bin/env python3
"""Meta-learning and parameter optimization CLI.

Usage:
    python -m gnosis.train.meta --config configs/training.yaml
    python -m gnosis.train.meta --config configs/training.yaml --n-trials 50
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from gnosis.training.config import TrainingConfig, MetaConfig, RunMetadata
from gnosis.training.meta import MetaTrainer, StrategyParams, StrategyOptimizer
from gnosis.training.supervised import SupervisedTrainer
from gnosis.datasets import create_purged_splits


def load_supervised_predictions(
    supervised_dir: Path
) -> Tuple[pd.DataFrame, SupervisedTrainer]:
    """Load predictions from supervised training stage."""
    if not supervised_dir.exists():
        raise FileNotFoundError(f"Supervised results not found at {supervised_dir}")

    # Load predictions
    preds_path = supervised_dir / 'predictions.parquet'
    if preds_path.exists():
        predictions = pd.read_parquet(preds_path)
    else:
        raise FileNotFoundError(f"Predictions not found at {preds_path}")

    # Load first fold trainer for feature names
    fold_dirs = sorted(supervised_dir.glob('fold_*'))
    if fold_dirs:
        trainer = SupervisedTrainer.load(str(fold_dirs[0]))
    else:
        trainer = None

    return predictions, trainer


def simulate_strategy(
    predictions: pd.DataFrame,
    params: StrategyParams,
    initial_capital: float = 10000.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate strategy with given parameters.

    Args:
        predictions: Supervised model predictions
        params: Strategy parameters
        initial_capital: Starting capital

    Returns:
        Tuple of (returns array, turnover array)
    """
    # Sort by time
    df = predictions.sort_values(['symbol', 'timestamp_end']).reset_index(drop=True)

    returns = []
    turnovers = []
    position = 0.0
    equity = initial_capital

    for i, row in df.iterrows():
        # Get prediction columns (use horizon 4 as default)
        x_hat = row.get('x_hat_h4', row.get('x_hat', 0))
        confidence = row.get('confidence_h4', row.get('confidence', 0.5))
        actual_return = row.get('future_return', row.get('fwd_ret_4', 0))

        if pd.isna(actual_return):
            actual_return = 0

        # Signal: long if x_hat > threshold and confident
        signal = 0
        if x_hat > params.signal_threshold and confidence > params.confidence_threshold:
            signal = 1

        # Target position
        target_pos = signal * params.base_size_pct
        target_pos = min(target_pos, params.max_position_pct)

        # Trade
        trade_size = target_pos - position
        turnover = abs(trade_size)
        turnovers.append(turnover)

        # Update position
        position = target_pos

        # Compute return
        period_return = position * actual_return
        returns.append(period_return)

        # Update equity
        equity *= (1 + period_return)

        # Stop loss check
        if period_return < -params.stop_loss_pct:
            position = 0  # Exit position

    return np.array(returns), np.array(turnovers)


def run_meta_training(
    config: TrainingConfig,
    supervised_dir: Optional[str] = None,
    n_trials: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> dict:
    """Run meta-learning and parameter optimization.

    Args:
        config: Training configuration
        supervised_dir: Path to supervised training results
        n_trials: Override number of optimization trials
        output_dir: Override output directory

    Returns:
        Dictionary with results
    """
    # Setup
    if n_trials:
        config.meta.n_trials = n_trials

    base_run_dir = Path(output_dir or config.runs_dir) / config.experiment_name
    supervised_dir = Path(supervised_dir or base_run_dir / 'supervised')
    run_dir = base_run_dir / 'meta'
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create run metadata
    metadata = RunMetadata.create(
        experiment_name=config.experiment_name,
        stage='meta',
        config=config.to_dict(),
    )

    print(f"Run ID: {metadata.run_id}")
    print(f"Output: {run_dir}")

    # Load supervised predictions
    print("\nLoading supervised predictions...")
    try:
        predictions, supervisor_trainer = load_supervised_predictions(supervised_dir)
        print(f"  Loaded {len(predictions)} predictions")
    except FileNotFoundError as e:
        print(f"  Warning: {e}")
        print("  Generating mock predictions for demo...")
        # Generate mock predictions for testing
        n_samples = 1000
        predictions = pd.DataFrame({
            'symbol': ['BTCUSDT'] * n_samples,
            'bar_idx': range(n_samples),
            'timestamp_end': pd.date_range('2024-01-01', periods=n_samples, freq='h'),
            'close': 50000 + np.cumsum(np.random.randn(n_samples) * 100),
            'x_hat_h4': np.random.randn(n_samples) * 0.01,
            'confidence_h4': np.random.uniform(0.3, 0.9, n_samples),
            'future_return': np.random.randn(n_samples) * 0.02,
        })

    # Split for inner CV
    print("\nPreparing inner CV splits...")
    n_inner = config.meta.n_inner_folds
    n = len(predictions)
    fold_size = n // n_inner

    inner_folds = []
    for i in range(n_inner):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < n_inner - 1 else n
        train_indices = np.concatenate([
            np.arange(0, test_start),
            np.arange(test_end, n)
        ])
        test_indices = np.arange(test_start, test_end)
        inner_folds.append((train_indices, test_indices))

    print(f"  Created {len(inner_folds)} inner folds")

    # Create evaluation function for optimizer
    def evaluate_params(params: StrategyParams) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate parameters on inner CV."""
        all_returns = []
        all_turnovers = []

        for train_idx, test_idx in inner_folds:
            test_preds = predictions.iloc[test_idx]
            returns, turnover = simulate_strategy(test_preds, params)
            all_returns.extend(returns)
            all_turnovers.extend(turnover)

        return np.array(all_returns), np.array(all_turnovers)

    # Run parameter optimization
    print("\nOptimizing strategy parameters...")
    print(f"  Method: {config.meta.optimizer}")
    print(f"  Trials: {config.meta.n_trials}")

    optimizer = StrategyOptimizer(config.meta)
    best_params = optimizer.optimize(evaluate_params)

    print("\nBest parameters:")
    for key, val in best_params.to_dict().items():
        print(f"  {key}: {val}")

    # Evaluate best params on full data
    print("\nEvaluating best parameters...")
    returns, turnover = simulate_strategy(predictions, best_params)

    final_metrics = {
        'total_return': float((1 + returns).prod() - 1),
        'sharpe': float(np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)),
        'max_drawdown': float(_compute_max_drawdown(returns)),
        'avg_turnover': float(np.mean(turnover)),
        'n_trials': len(optimizer.trials),
    }

    print("\nFinal metrics:")
    for key, val in final_metrics.items():
        print(f"  {key}: {val:.4f}")

    # Train meta-learner on full data
    print("\nTraining meta-learner...")
    meta_trainer = MetaTrainer(config.meta)

    # Prepare context features (regime info would come from features_df)
    context_features = predictions[['symbol', 'bar_idx']].copy()
    if 'confidence_h4' in predictions.columns:
        context_features['confidence'] = predictions['confidence_h4']

    actual_returns = predictions[['symbol', 'bar_idx']].copy()
    if 'future_return' in predictions.columns:
        actual_returns['future_return'] = predictions['future_return']
    elif 'fwd_ret_4' in predictions.columns:
        actual_returns['future_return'] = predictions['fwd_ret_4']

    meta_trainer.fit(predictions, context_features, actual_returns)
    meta_trainer.best_params = best_params

    # Save results
    print("\nSaving results...")
    meta_trainer.save(str(run_dir))
    optimizer.save(str(run_dir / 'optimizer'))

    with open(run_dir / 'best_params.json', 'w') as f:
        json.dump(best_params.to_dict(), f, indent=2)

    with open(run_dir / 'final_metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=2)

    # Update metadata
    metadata.completed_at = datetime.utcnow().isoformat()
    metadata.status = 'completed'
    metadata.metrics = final_metrics
    metadata.save(str(run_dir / 'run_metadata.json'))

    # Save config
    config.save_yaml(str(run_dir / 'config.yaml'))

    print(f"\nMeta-training complete!")
    print(f"Results saved to: {run_dir}")

    return {
        'run_id': metadata.run_id,
        'best_params': best_params.to_dict(),
        'metrics': final_metrics,
        'output_dir': str(run_dir),
    }


def _compute_max_drawdown(returns: np.ndarray) -> float:
    """Compute maximum drawdown from returns array."""
    cumret = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cumret)
    drawdown = (peak - cumret) / peak
    return np.max(drawdown) if len(drawdown) > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(
        description='Train meta-learner and optimize strategy parameters'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/training.yaml',
        help='Path to training configuration YAML'
    )
    parser.add_argument(
        '--supervised-dir',
        type=str,
        default=None,
        help='Path to supervised training results'
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=None,
        help='Number of optimization trials'
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

    # Run training
    result = run_meta_training(
        config,
        supervised_dir=args.supervised_dir,
        n_trials=args.n_trials,
        output_dir=args.output_dir,
    )

    print(f"\nRun ID: {result['run_id']}")


if __name__ == '__main__':
    main()
