"""Training configuration and utilities."""

import hashlib
import json
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import yaml


@dataclass
class SupervisedConfig:
    """Configuration for supervised learning stage."""

    # Model type: 'ridge_quantile', 'gbm_quantile', 'classifier'
    model_type: str = 'ridge_quantile'

    # Quantile settings
    quantiles: List[float] = field(default_factory=lambda: [0.05, 0.25, 0.50, 0.75, 0.95])

    # Regularization
    l2_reg: float = 1.0
    l1_reg: float = 0.0

    # For GBM models
    n_estimators: int = 100
    max_depth: int = 5
    learning_rate: float = 0.1

    # Classification settings
    n_classes: int = 3  # UP/FLAT/DOWN

    # Calibration
    calibrate_confidence: bool = True
    calibration_method: str = 'isotonic'  # 'isotonic' or 'platt'

    # Horizons to train on
    horizons: List[int] = field(default_factory=lambda: [1, 4, 8, 16])

    # Training settings
    random_seed: int = 1337


@dataclass
class MetaConfig:
    """Configuration for meta/parameter learning stage."""

    # Optimizer: 'optuna', 'grid', 'bayesian'
    optimizer: str = 'optuna'

    # Number of trials
    n_trials: int = 100

    # Objective
    objective: str = 'sharpe'  # 'sharpe', 'sortino', 'calmar'

    # Penalty weights
    drawdown_penalty: float = 0.5
    turnover_penalty: float = 0.1

    # Parameter search space (will be populated from config file)
    param_grid: Dict[str, Any] = field(default_factory=dict)

    # Inner CV folds for parameter selection
    n_inner_folds: int = 3

    # Walk-forward settings
    train_size: int = 500
    val_size: int = 100

    random_seed: int = 1337


@dataclass
class RLConfig:
    """Configuration for reinforcement learning stage."""

    # Algorithm: 'cql', 'iql', 'bc', 'awac'
    algorithm: str = 'cql'

    # State space
    include_predictions: bool = True
    include_positions: bool = True
    include_regime: bool = True

    # Action space
    actions: List[str] = field(default_factory=lambda: ['flat', 'long_small', 'long_full', 'reduce', 'exit'])
    continuous_sizing: bool = False

    # Reward shaping
    reward_scale: float = 1.0
    drawdown_penalty: float = 2.0
    turnover_penalty: float = 0.01
    tail_loss_penalty: float = 1.0

    # Guardrails
    max_position: float = 1.0
    max_daily_loss: float = 0.05
    max_turnover: float = 5.0  # Times per day

    # Network architecture
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    learning_rate: float = 3e-4

    # Training settings
    batch_size: int = 256
    n_epochs: int = 100
    offline_buffer_size: int = 100000

    # CQL specific
    cql_alpha: float = 1.0

    random_seed: int = 1337


@dataclass
class TrainingConfig:
    """Master configuration for the training pipeline."""

    # Stage configurations
    supervised: SupervisedConfig = field(default_factory=SupervisedConfig)
    meta: MetaConfig = field(default_factory=MetaConfig)
    rl: RLConfig = field(default_factory=RLConfig)

    # Data settings
    data_path: str = 'data/parquet/prints.parquet'
    symbols: List[str] = field(default_factory=lambda: ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'])

    # Bar settings
    bar_type: str = 'tick'
    n_ticks: int = 200

    # Split settings
    n_outer_folds: int = 5
    train_days: int = 180
    val_days: int = 30
    test_days: int = 30
    purge_bars: int = 32
    embargo_bars: int = 8

    # Output
    runs_dir: str = 'runs'
    experiment_name: str = 'default'

    # Reproducibility
    random_seed: int = 1337

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str) -> 'TrainingConfig':
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        # Handle nested configs
        supervised = SupervisedConfig(**data.get('supervised', {}))
        meta = MetaConfig(**data.get('meta', {}))
        rl = RLConfig(**data.get('rl', {}))

        # Remove nested dicts from top level
        top_level = {k: v for k, v in data.items()
                     if k not in ('supervised', 'meta', 'rl')}

        return cls(
            supervised=supervised,
            meta=meta,
            rl=rl,
            **top_level
        )

    def save_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return 'unknown'


def compute_config_hash(config: dict) -> str:
    """Compute hash of configuration for reproducibility."""
    config_str = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


@dataclass
class RunMetadata:
    """Metadata for a training run."""

    run_id: str
    experiment_name: str
    stage: str  # 'supervised', 'meta', 'rl'
    started_at: str
    completed_at: Optional[str] = None
    git_commit: str = ''
    config_hash: str = ''
    status: str = 'running'  # 'running', 'completed', 'failed'
    metrics: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        experiment_name: str,
        stage: str,
        config: dict
    ) -> 'RunMetadata':
        """Create new run metadata."""
        import uuid
        run_id = datetime.utcnow().strftime('%Y%m%d_%H%M%S_') + str(uuid.uuid4())[:8]

        return cls(
            run_id=run_id,
            experiment_name=experiment_name,
            stage=stage,
            started_at=datetime.utcnow().isoformat(),
            git_commit=get_git_commit(),
            config_hash=compute_config_hash(config),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    def save(self, path: str) -> None:
        """Save metadata to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'RunMetadata':
        """Load metadata from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)
