"""Validation harness module."""
from .walkforward import WalkForwardHarness, Fold, compute_future_returns
from .scoring import (
    pinball_loss,
    coverage,
    sharpness,
    crps_empirical,
    evaluate_predictions,
    IsotonicCalibrator,
    compute_ece,
    compute_stability_metrics,
)
from .ralph_loop import RalphLoop, RalphLoopConfig
from .bregman_optimizer import (
    ProjectFWOptimizer,
    ProjectFWConfig,
    OptimizationResult,
    BregmanFunction,
)

__all__ = [
    "WalkForwardHarness",
    "Fold",
    "compute_future_returns",
    "pinball_loss",
    "coverage",
    "sharpness",
    "crps_empirical",
    "evaluate_predictions",
    "IsotonicCalibrator",
    "compute_ece",
    "compute_stability_metrics",
    "RalphLoop",
    "RalphLoopConfig",
    "ProjectFWOptimizer",
    "ProjectFWConfig",
    "OptimizationResult",
    "BregmanFunction",
]

from .trade_walkforward import TradeWalkForwardHarness, TradeFold
