#!/usr/bin/env python3
"""Reinforcement learning training CLI.

Usage:
    python -m gnosis.train.rl --config configs/training.yaml
    python -m gnosis.train.rl --config configs/training.yaml --algorithm cql --n-epochs 50
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from gnosis.training.config import TrainingConfig, RLConfig, RunMetadata
from gnosis.rl import (
    TradingEnv,
    EnvConfig,
    create_env_from_predictions,
    create_agent,
    AgentConfig,
    ReplayBuffer,
)


def load_predictions(
    supervised_dir: Path,
    meta_dir: Optional[Path] = None
) -> pd.DataFrame:
    """Load predictions from previous training stages."""
    # Try supervised predictions
    preds_path = supervised_dir / 'predictions.parquet'
    if preds_path.exists():
        predictions = pd.read_parquet(preds_path)
    else:
        # Generate mock predictions for testing
        print("  Generating mock predictions for demo...")
        n_samples = 2000
        predictions = pd.DataFrame({
            'symbol': ['BTCUSDT'] * n_samples,
            'bar_idx': range(n_samples),
            'timestamp_end': pd.date_range('2024-01-01', periods=n_samples, freq='h'),
            'close': 50000 + np.cumsum(np.random.randn(n_samples) * 100),
            'returns': np.random.randn(n_samples) * 0.01,
            'realized_vol': 0.02 + np.random.rand(n_samples) * 0.01,
            'ofi': np.random.randn(n_samples) * 0.3,
            'x_hat_h4': np.random.randn(n_samples) * 0.01,
            'sigma_hat_h4': 0.02 + np.random.rand(n_samples) * 0.01,
            'confidence_h4': np.random.uniform(0.3, 0.9, n_samples),
            'future_return': np.random.randn(n_samples) * 0.02,
        })

    # Add regime features if meta results available
    if meta_dir and (meta_dir / 'meta_predictions.parquet').exists():
        meta_preds = pd.read_parquet(meta_dir / 'meta_predictions.parquet')
        predictions = predictions.merge(
            meta_preds, on=['symbol', 'bar_idx'], how='left', suffixes=('', '_meta')
        )

    return predictions


def collect_offline_data(
    env: TradingEnv,
    buffer: ReplayBuffer,
    n_episodes: int = 10,
    behavior_policy: str = 'random'
) -> Dict[str, float]:
    """Collect offline data using behavior policy.

    Args:
        env: Trading environment
        buffer: Replay buffer to fill
        n_episodes: Number of episodes to collect
        behavior_policy: Type of behavior policy

    Returns:
        Collection statistics
    """
    total_reward = 0.0
    total_steps = 0

    for episode in range(n_episodes):
        # Start at random point
        n_data = env.n_steps - env.config.lookback_bars - 100
        start_idx = np.random.randint(env.config.lookback_bars, n_data)

        state = env.reset(start_idx=start_idx)
        done = False
        episode_reward = 0.0
        episode_steps = 0

        while not done and episode_steps < 500:
            # Behavior policy
            if behavior_policy == 'random':
                action = np.random.randint(0, env.action_space_dim)
            elif behavior_policy == 'hold':
                # Always hold current position
                action = 4  # HOLD
            elif behavior_policy == 'momentum':
                # Simple momentum: long if recent returns positive
                if hasattr(env, '_current_idx'):
                    idx = env._current_idx
                    if 'returns' in env.data.columns:
                        ret = env.data.loc[idx, 'returns']
                        action = 2 if ret > 0 else 0  # LONG_FULL or FLAT
                    else:
                        action = np.random.randint(0, env.action_space_dim)
                else:
                    action = np.random.randint(0, env.action_space_dim)
            else:
                action = np.random.randint(0, env.action_space_dim)

            next_state, reward, done, info = env.step(action)

            buffer.add(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_steps += 1

        total_reward += episode_reward

    return {
        'n_transitions': len(buffer),
        'total_reward': total_reward,
        'avg_episode_reward': total_reward / n_episodes,
        'total_steps': total_steps,
    }


def train_offline(
    agent,
    buffer: ReplayBuffer,
    n_epochs: int,
    batch_size: int,
    log_interval: int = 100
) -> List[Dict[str, float]]:
    """Train agent offline on collected data.

    Args:
        agent: RL agent
        buffer: Replay buffer with offline data
        n_epochs: Number of training epochs
        batch_size: Training batch size
        log_interval: Steps between logging

    Returns:
        List of training metrics
    """
    metrics_history = []
    total_steps = 0

    n_batches_per_epoch = max(1, len(buffer) // batch_size)

    for epoch in range(n_epochs):
        epoch_metrics = {}

        for batch_idx in range(n_batches_per_epoch):
            batch = buffer.sample(batch_size)
            step_metrics = agent.train_step(batch)

            for k, v in step_metrics.items():
                if k not in epoch_metrics:
                    epoch_metrics[k] = []
                epoch_metrics[k].append(v)

            total_steps += 1

        # Average epoch metrics
        avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
        avg_metrics['epoch'] = epoch
        metrics_history.append(avg_metrics)

        if (epoch + 1) % log_interval == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{n_epochs}")
            for k, v in avg_metrics.items():
                if k != 'epoch':
                    print(f"    {k}: {v:.4f}")

    return metrics_history


def evaluate_agent(
    agent,
    env: TradingEnv,
    n_episodes: int = 5,
    deterministic: bool = True
) -> Dict[str, float]:
    """Evaluate trained agent.

    Args:
        agent: Trained RL agent
        env: Trading environment
        n_episodes: Number of evaluation episodes
        deterministic: Use deterministic action selection

    Returns:
        Evaluation metrics
    """
    total_reward = 0.0
    total_equity_gain = 0.0
    max_drawdowns = []
    episode_lengths = []

    for episode in range(n_episodes):
        # Start at beginning
        state = env.reset(start_idx=env.config.lookback_bars)
        done = False
        episode_reward = 0.0
        episode_length = 0
        start_equity = env.state.equity

        while not done and episode_length < 500:
            action = agent.select_action(state, deterministic=deterministic)
            state, reward, done, info = env.step(action)

            episode_reward += reward
            episode_length += 1

        total_reward += episode_reward
        total_equity_gain += (env.state.equity - start_equity) / start_equity
        max_drawdowns.append(info.get('drawdown', 0))
        episode_lengths.append(episode_length)

    return {
        'avg_reward': total_reward / n_episodes,
        'avg_equity_gain': total_equity_gain / n_episodes,
        'avg_max_drawdown': np.mean(max_drawdowns),
        'avg_episode_length': np.mean(episode_lengths),
    }


def run_rl_training(
    config: TrainingConfig,
    supervised_dir: Optional[str] = None,
    algorithm: Optional[str] = None,
    n_epochs: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> dict:
    """Run reinforcement learning training.

    Args:
        config: Training configuration
        supervised_dir: Path to supervised training results
        algorithm: Override RL algorithm
        n_epochs: Override number of epochs
        output_dir: Override output directory

    Returns:
        Dictionary with results
    """
    # Setup
    if algorithm:
        config.rl.algorithm = algorithm
    if n_epochs:
        config.rl.n_epochs = n_epochs

    base_run_dir = Path(output_dir or config.runs_dir) / config.experiment_name
    supervised_dir = Path(supervised_dir or base_run_dir / 'supervised')
    run_dir = base_run_dir / 'rl'
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create run metadata
    metadata = RunMetadata.create(
        experiment_name=config.experiment_name,
        stage='rl',
        config=config.to_dict(),
    )

    print(f"Run ID: {metadata.run_id}")
    print(f"Output: {run_dir}")

    # Load predictions
    print("\nLoading predictions...")
    predictions = load_predictions(supervised_dir)
    print(f"  Loaded {len(predictions)} samples")

    # Create environment
    print("\nCreating environment...")
    env_config = EnvConfig(
        include_predictions=config.rl.include_predictions,
        include_positions=config.rl.include_positions,
        include_regime=config.rl.include_regime,
        reward_scale=config.rl.reward_scale,
        drawdown_penalty=config.rl.drawdown_penalty,
        turnover_penalty=config.rl.turnover_penalty,
        max_position=config.rl.max_position,
        max_daily_loss=config.rl.max_daily_loss,
        initial_capital=10000.0,
    )
    env = create_env_from_predictions(predictions, env_config)
    print(f"  Observation dim: {env.observation_space_dim}")
    print(f"  Action dim: {env.action_space_dim}")

    # Create replay buffer
    print("\nCollecting offline data...")
    buffer = ReplayBuffer(capacity=config.rl.offline_buffer_size)

    # Collect data with multiple behavior policies
    for policy in ['random', 'momentum', 'hold']:
        stats = collect_offline_data(
            env, buffer,
            n_episodes=10,
            behavior_policy=policy
        )
        print(f"  {policy}: {stats['n_transitions']} transitions, avg reward {stats['avg_episode_reward']:.2f}")

    print(f"  Total buffer size: {len(buffer)}")

    # Create agent
    print(f"\nCreating {config.rl.algorithm.upper()} agent...")
    agent_config = AgentConfig(
        hidden_dims=config.rl.hidden_dims,
        learning_rate=config.rl.learning_rate,
        batch_size=config.rl.batch_size,
        cql_alpha=config.rl.cql_alpha,
        random_seed=config.rl.random_seed,
    )
    agent = create_agent(
        config.rl.algorithm,
        env.observation_space_dim,
        env.action_space_dim,
        agent_config
    )

    # Train offline
    print(f"\nTraining for {config.rl.n_epochs} epochs...")
    metrics_history = train_offline(
        agent, buffer,
        n_epochs=config.rl.n_epochs,
        batch_size=config.rl.batch_size,
        log_interval=max(1, config.rl.n_epochs // 10)
    )

    # Evaluate
    print("\nEvaluating agent...")
    eval_metrics = evaluate_agent(agent, env, n_episodes=5)
    print("  Results:")
    for k, v in eval_metrics.items():
        print(f"    {k}: {v:.4f}")

    # Save results
    print("\nSaving results...")
    agent.save(str(run_dir / 'agent'))
    buffer.save(str(run_dir / 'buffer.pkl'))

    # Save training history
    metrics_df = pd.DataFrame(metrics_history)
    metrics_df.to_csv(run_dir / 'training_history.csv', index=False)

    # Save evaluation metrics
    with open(run_dir / 'eval_metrics.json', 'w') as f:
        json.dump(eval_metrics, f, indent=2)

    # Update metadata
    metadata.completed_at = datetime.utcnow().isoformat()
    metadata.status = 'completed'
    metadata.metrics = eval_metrics
    metadata.save(str(run_dir / 'run_metadata.json'))

    # Save config
    config.save_yaml(str(run_dir / 'config.yaml'))

    print(f"\nRL training complete!")
    print(f"Results saved to: {run_dir}")

    return {
        'run_id': metadata.run_id,
        'eval_metrics': eval_metrics,
        'output_dir': str(run_dir),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Train reinforcement learning agent for trading'
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
        '--algorithm',
        type=str,
        choices=['bc', 'cql', 'iql'],
        default=None,
        help='RL algorithm to use'
    )
    parser.add_argument(
        '--n-epochs',
        type=int,
        default=None,
        help='Number of training epochs'
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
    result = run_rl_training(
        config,
        supervised_dir=args.supervised_dir,
        algorithm=args.algorithm,
        n_epochs=args.n_epochs,
        output_dir=args.output_dir,
    )

    print(f"\nRun ID: {result['run_id']}")


if __name__ == '__main__':
    main()
