"""Training module for supervised, meta, and RL learning.

This module provides:
- Supervised learning for return prediction
- Meta-learning for trade decisions
- Strategy parameter optimization
- Training configuration and artifacts
"""

from .config import (
    SupervisedConfig,
    MetaConfig,
    RLConfig,
    TrainingConfig,
    RunMetadata,
    get_git_commit,
    compute_config_hash,
)

from .supervised import (
    ModelArtifact,
    QuantileRegressor,
    DirectionClassifier,
    ConfidenceCalibrator,
    SupervisedTrainer,
)

from .meta import (
    StrategyParams,
    MetaLearner,
    RiskAdjustedObjective,
    StrategyOptimizer,
    MetaTrainer,
)

__all__ = [
    # Config
    'SupervisedConfig',
    'MetaConfig',
    'RLConfig',
    'TrainingConfig',
    'RunMetadata',
    'get_git_commit',
    'compute_config_hash',
    # Supervised
    'ModelArtifact',
    'QuantileRegressor',
    'DirectionClassifier',
    'ConfidenceCalibrator',
    'SupervisedTrainer',
    # Meta
    'StrategyParams',
    'MetaLearner',
    'RiskAdjustedObjective',
    'StrategyOptimizer',
    'MetaTrainer',
]
