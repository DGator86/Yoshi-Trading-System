"""Yoshi Improvement Loop - iterative optimization until targets are met.

This implements a coordinate descent approach:
1. Define target metrics (e.g., directional_accuracy >= 0.60)
2. For each metric not yet satisfied:
   - Try single-variable perturbations
   - Keep changes that improve the metric
   - Stop when target met or no improvements found
3. Move to next metric
"""
import copy
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class MetricType(Enum):
    """Types of metrics to optimize."""
    DIRECTIONAL_ACCURACY = "directional_accuracy"
    COVERAGE_90 = "coverage_90"
    MAE = "mae"
    SHARPE = "sharpe"
    CALIBRATION = "calibration_error"


@dataclass
class MetricTarget:
    """A target metric to achieve."""
    name: str
    metric_type: MetricType
    target_value: float
    direction: str = "maximize"  # "maximize" or "minimize"
    priority: int = 1  # Lower = higher priority

    def is_satisfied(self, current_value: float) -> bool:
        """Check if target is met."""
        if self.direction == "maximize":
            return current_value >= self.target_value
        else:
            return current_value <= self.target_value

    def improvement(self, old_value: float, new_value: float) -> float:
        """Calculate improvement (positive = better)."""
        if self.direction == "maximize":
            return new_value - old_value
        else:
            return old_value - new_value


@dataclass
class Variable:
    """A variable that can be tuned."""
    name: str
    path: str  # Config path like "models.predictor.l2_reg"
    current_value: Any
    candidates: List[Any]  # Values to try
    variable_type: str = "continuous"  # "continuous", "discrete", "categorical"

    def get_perturbations(self) -> List[Any]:
        """Get list of values to try."""
        return [v for v in self.candidates if v != self.current_value]


@dataclass
class IterationResult:
    """Result of a single iteration."""
    iteration: int
    variable_name: str
    old_value: Any
    new_value: Any
    old_metrics: Dict[str, float]
    new_metrics: Dict[str, float]
    improvement: float
    accepted: bool
    duration_seconds: float


@dataclass
class LoopResult:
    """Result of the full improvement loop."""
    target_name: str
    target_achieved: bool
    final_value: float
    target_value: float
    iterations: int
    improvements_made: int
    total_duration_seconds: float
    iteration_history: List[IterationResult] = field(default_factory=list)
    final_config: Dict = field(default_factory=dict)


class YoshiImprovementLoop:
    """Iterative improvement loop for Yoshi predictions.

    Usage:
        loop = YoshiImprovementLoop(
            targets=[
                MetricTarget("accuracy", MetricType.DIRECTIONAL_ACCURACY, 0.60, "maximize"),
                MetricTarget("coverage", MetricType.COVERAGE_90, 0.90, "maximize"),
            ],
            variables=[
                Variable("l2_reg", "models.predictor.l2_reg", 1.0, [0.01, 0.1, 1.0, 10.0]),
                Variable("backend", "models.predictor.backend", "ridge", ["ridge", "quantile"]),
            ],
            evaluate_fn=my_evaluate_function,
            max_iterations_per_target=50,
        )

        results = loop.run(initial_config)
    """

    def __init__(
        self,
        targets: List[MetricTarget],
        variables: List[Variable],
        evaluate_fn: Callable[[Dict], Dict[str, float]],
        max_iterations_per_target: int = 50,
        patience: int = 10,  # Stop if no improvement for N iterations
        verbose: bool = True,
    ):
        """Initialize the improvement loop.

        Args:
            targets: List of MetricTarget to achieve, in priority order
            variables: List of Variable that can be tuned
            evaluate_fn: Function that takes config and returns metrics dict
            max_iterations_per_target: Max iterations before giving up on a target
            patience: Stop early if no improvement for this many iterations
            verbose: Print progress
        """
        self.targets = sorted(targets, key=lambda t: t.priority)
        self.variables = {v.name: v for v in variables}
        self.evaluate_fn = evaluate_fn
        self.max_iterations_per_target = max_iterations_per_target
        self.patience = patience
        self.verbose = verbose

        self._current_config: Dict = {}
        self._current_metrics: Dict[str, float] = {}
        self._best_config: Dict = {}
        self._best_metrics: Dict[str, float] = {}

    def _log(self, msg: str):
        """Print if verbose."""
        if self.verbose:
            print(msg)

    def _get_config_value(self, config: Dict, path: str) -> Any:
        """Get value from nested config using dot path."""
        keys = path.split(".")
        value = config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value

    def _set_config_value(self, config: Dict, path: str, value: Any) -> Dict:
        """Set value in nested config using dot path."""
        config = copy.deepcopy(config)
        keys = path.split(".")
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
        return config

    def _evaluate(self, config: Dict) -> Dict[str, float]:
        """Evaluate config and return metrics."""
        return self.evaluate_fn(config)

    def _try_perturbation(
        self,
        variable: Variable,
        new_value: Any,
        target: MetricTarget,
    ) -> Tuple[Dict[str, float], float, bool]:
        """Try a single perturbation and check if it improves the target.

        Returns:
            (new_metrics, improvement, should_accept)
        """
        # Create new config with perturbation
        new_config = self._set_config_value(
            self._current_config, variable.path, new_value
        )

        # Evaluate
        new_metrics = self._evaluate(new_config)

        # Calculate improvement
        old_value = self._current_metrics.get(target.metric_type.value, 0)
        new_metric_value = new_metrics.get(target.metric_type.value, 0)
        improvement = target.improvement(old_value, new_metric_value)

        # Accept if improved
        should_accept = improvement > 0

        return new_metrics, improvement, should_accept

    def _optimize_for_target(
        self,
        target: MetricTarget,
        config: Dict,
    ) -> LoopResult:
        """Optimize until target is met or give up.

        Args:
            target: The metric target to achieve
            config: Starting configuration

        Returns:
            LoopResult with final state
        """
        self._log(f"\n{'='*60}")
        self._log(f"OPTIMIZING: {target.name}")
        self._log(f"Target: {target.metric_type.value} {'>=' if target.direction == 'maximize' else '<='} {target.target_value}")
        self._log(f"{'='*60}")

        self._current_config = copy.deepcopy(config)
        self._current_metrics = self._evaluate(self._current_config)

        current_value = self._current_metrics.get(target.metric_type.value, 0)
        self._log(f"Starting value: {current_value:.4f}")

        if target.is_satisfied(current_value):
            self._log(f"Target already satisfied!")
            return LoopResult(
                target_name=target.name,
                target_achieved=True,
                final_value=current_value,
                target_value=target.target_value,
                iterations=0,
                improvements_made=0,
                total_duration_seconds=0,
                final_config=self._current_config,
            )

        start_time = time.time()
        iteration_history: List[IterationResult] = []
        iterations_without_improvement = 0
        improvements_made = 0

        for iteration in range(1, self.max_iterations_per_target + 1):
            iter_start = time.time()

            # Try each variable
            best_improvement = 0
            best_variable = None
            best_new_value = None
            best_new_metrics = None

            for var_name, variable in self.variables.items():
                for new_value in variable.get_perturbations():
                    new_metrics, improvement, _ = self._try_perturbation(
                        variable, new_value, target
                    )

                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_variable = variable
                        best_new_value = new_value
                        best_new_metrics = new_metrics

            # Record iteration
            iter_result = IterationResult(
                iteration=iteration,
                variable_name=best_variable.name if best_variable else "none",
                old_value=self._get_config_value(self._current_config, best_variable.path) if best_variable else None,
                new_value=best_new_value,
                old_metrics=copy.deepcopy(self._current_metrics),
                new_metrics=best_new_metrics if best_new_metrics else {},
                improvement=best_improvement,
                accepted=best_improvement > 0,
                duration_seconds=time.time() - iter_start,
            )
            iteration_history.append(iter_result)

            if best_improvement > 0:
                # Accept the change
                self._current_config = self._set_config_value(
                    self._current_config, best_variable.path, best_new_value
                )
                self._current_metrics = best_new_metrics
                best_variable.current_value = best_new_value

                current_value = self._current_metrics.get(target.metric_type.value, 0)
                improvements_made += 1
                iterations_without_improvement = 0

                self._log(f"  [{iteration}] {best_variable.name}: {iter_result.old_value} -> {best_new_value}")
                self._log(f"       {target.metric_type.value}: {current_value:.4f} (+{best_improvement:.4f})")

                # Check if target achieved
                if target.is_satisfied(current_value):
                    self._log(f"\n✓ TARGET ACHIEVED at iteration {iteration}!")
                    return LoopResult(
                        target_name=target.name,
                        target_achieved=True,
                        final_value=current_value,
                        target_value=target.target_value,
                        iterations=iteration,
                        improvements_made=improvements_made,
                        total_duration_seconds=time.time() - start_time,
                        iteration_history=iteration_history,
                        final_config=copy.deepcopy(self._current_config),
                    )
            else:
                iterations_without_improvement += 1
                self._log(f"  [{iteration}] No improvement found")

                # Check patience
                if iterations_without_improvement >= self.patience:
                    self._log(f"\n✗ No improvement for {self.patience} iterations, stopping")
                    break

        current_value = self._current_metrics.get(target.metric_type.value, 0)
        return LoopResult(
            target_name=target.name,
            target_achieved=False,
            final_value=current_value,
            target_value=target.target_value,
            iterations=len(iteration_history),
            improvements_made=improvements_made,
            total_duration_seconds=time.time() - start_time,
            iteration_history=iteration_history,
            final_config=copy.deepcopy(self._current_config),
        )

    def run(self, initial_config: Dict) -> List[LoopResult]:
        """Run the full improvement loop across all targets.

        Args:
            initial_config: Starting configuration

        Returns:
            List of LoopResult, one per target
        """
        self._log("\n" + "="*60)
        self._log("YOSHI IMPROVEMENT LOOP")
        self._log("="*60)
        self._log(f"Targets: {[t.name for t in self.targets]}")
        self._log(f"Variables: {list(self.variables.keys())}")
        self._log(f"Max iterations per target: {self.max_iterations_per_target}")

        results: List[LoopResult] = []
        current_config = copy.deepcopy(initial_config)

        for target in self.targets:
            result = self._optimize_for_target(target, current_config)
            results.append(result)

            # Use the optimized config for next target
            current_config = result.final_config

        # Summary
        self._log("\n" + "="*60)
        self._log("IMPROVEMENT LOOP COMPLETE")
        self._log("="*60)
        for result in results:
            status = "✓" if result.target_achieved else "✗"
            self._log(f"{status} {result.target_name}: {result.final_value:.4f} (target: {result.target_value})")
            self._log(f"   Iterations: {result.iterations}, Improvements: {result.improvements_made}")

        return results

    def generate_report(self, results: List[LoopResult]) -> str:
        """Generate a human-readable report of the improvement loop."""
        lines = []
        lines.append("="*70)
        lines.append("YOSHI IMPROVEMENT LOOP REPORT")
        lines.append("="*70)
        lines.append("")

        total_iterations = sum(r.iterations for r in results)
        total_improvements = sum(r.improvements_made for r in results)
        total_time = sum(r.total_duration_seconds for r in results)
        targets_achieved = sum(1 for r in results if r.target_achieved)

        lines.append(f"Total iterations: {total_iterations}")
        lines.append(f"Total improvements: {total_improvements}")
        lines.append(f"Targets achieved: {targets_achieved}/{len(results)}")
        lines.append(f"Total time: {total_time:.1f}s")
        lines.append("")

        for result in results:
            status = "ACHIEVED" if result.target_achieved else "NOT ACHIEVED"
            lines.append(f"--- {result.target_name} [{status}] ---")
            lines.append(f"  Target: {result.target_value}")
            lines.append(f"  Final:  {result.final_value:.4f}")
            lines.append(f"  Iterations: {result.iterations}")
            lines.append(f"  Improvements: {result.improvements_made}")
            lines.append("")

            if result.iteration_history:
                lines.append("  Key changes:")
                for iter_result in result.iteration_history:
                    if iter_result.accepted:
                        lines.append(
                            f"    [{iter_result.iteration}] {iter_result.variable_name}: "
                            f"{iter_result.old_value} -> {iter_result.new_value} "
                            f"(+{iter_result.improvement:.4f})"
                        )
                lines.append("")

        return "\n".join(lines)


# =============================================================================
# Pre-built variable sets for common optimization scenarios
# =============================================================================

def get_predictor_variables() -> List[Variable]:
    """Get tunable variables for the predictor."""
    return [
        Variable(
            name="l2_reg",
            path="models.predictor.l2_reg",
            current_value=1.0,
            candidates=[0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            variable_type="continuous",
        ),
        Variable(
            name="backend",
            path="models.predictor.backend",
            current_value="ridge",
            candidates=["ridge", "quantile", "gradient_boost"],
            variable_type="categorical",
        ),
        Variable(
            name="extended_features",
            path="models.predictor.extended_features",
            current_value=False,
            candidates=[True, False],
            variable_type="categorical",
        ),
        Variable(
            name="normalize",
            path="models.predictor.normalize",
            current_value=True,
            candidates=[True, False],
            variable_type="categorical",
        ),
    ]


def get_domain_variables() -> List[Variable]:
    """Get tunable variables for domain aggregation."""
    return [
        Variable(
            name="n_trades",
            path="domains.domains.D0.n_trades",
            current_value=200,
            candidates=[50, 100, 150, 200, 300, 500, 1000],
            variable_type="discrete",
        ),
    ]


def get_regime_variables() -> List[Variable]:
    """Get tunable variables for regime classification."""
    return [
        Variable(
            name="confidence_floor",
            path="regimes.confidence_floor",
            current_value=0.65,
            candidates=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
            variable_type="continuous",
        ),
    ]


def get_particle_variables() -> List[Variable]:
    """Get tunable variables for particle state."""
    return [
        Variable(
            name="flow_span",
            path="models.particle.flow_span",
            current_value=10,
            candidates=[5, 10, 15, 20, 30],
            variable_type="discrete",
        ),
        Variable(
            name="flow_weight",
            path="models.particle.flow_weight",
            current_value=1.0,
            candidates=[0.5, 1.0, 1.5, 2.0],
            variable_type="continuous",
        ),
    ]


def get_steering_field_variables() -> List[Variable]:
    """Get tunable variables for all steering field modules.

    These control the physics-inspired prediction engine:
    - Funding rate aggregation
    - Liquidation cascade dynamics
    - Gamma fields (options)
    - Cross-asset coupling (macro)
    - Time-of-day effects
    - Order book depth analysis
    """
    variables = []

    # === FUNDING RATE VARIABLES ===
    variables.extend([
        Variable(
            name="funding_strength_base",
            path="particle.funding.funding_strength_base",
            current_value=12.0,
            candidates=[6.0, 9.0, 12.0, 15.0, 18.0, 24.0],
            variable_type="continuous",
        ),
        Variable(
            name="funding_ema_alpha",
            path="particle.funding.ema_alpha",
            current_value=0.3,
            candidates=[0.1, 0.2, 0.3, 0.5, 0.7],
            variable_type="continuous",
        ),
        Variable(
            name="funding_weight_binance",
            path="particle.funding.weight_binance",
            current_value=0.50,
            candidates=[0.3, 0.4, 0.5, 0.6, 0.7],
            variable_type="continuous",
        ),
    ])

    # === LIQUIDATION VARIABLES ===
    variables.extend([
        Variable(
            name="liq_repulsion_strength",
            path="particle.liquidation.repulsion_strength",
            current_value=1e-6,
            candidates=[5e-7, 1e-6, 2e-6, 5e-6],
            variable_type="continuous",
        ),
        Variable(
            name="liq_cascade_strength",
            path="particle.liquidation.cascade_strength",
            current_value=1e-5,
            candidates=[5e-6, 1e-5, 2e-5, 5e-5],
            variable_type="continuous",
        ),
        Variable(
            name="liq_repulsion_zone_pct",
            path="particle.liquidation.repulsion_zone_pct",
            current_value=0.01,
            candidates=[0.005, 0.01, 0.015, 0.02],
            variable_type="continuous",
        ),
    ])

    # === GAMMA FIELD VARIABLES ===
    variables.extend([
        Variable(
            name="gamma_strength",
            path="particle.gamma.gamma_strength",
            current_value=0.001,
            candidates=[0.0005, 0.001, 0.002, 0.005],
            variable_type="continuous",
        ),
        Variable(
            name="gamma_max_pain_strength",
            path="particle.gamma.max_pain_strength",
            current_value=0.0005,
            candidates=[0.0002, 0.0005, 0.001, 0.002],
            variable_type="continuous",
        ),
    ])

    # === MACRO COUPLING VARIABLES ===
    variables.extend([
        Variable(
            name="macro_beta_spx",
            path="particle.macro.beta_spx",
            current_value=0.3,
            candidates=[0.1, 0.2, 0.3, 0.4, 0.5],
            variable_type="continuous",
        ),
        Variable(
            name="macro_beta_dxy",
            path="particle.macro.beta_dxy",
            current_value=-0.2,
            candidates=[-0.4, -0.3, -0.2, -0.1, 0.0],
            variable_type="continuous",
        ),
        Variable(
            name="macro_force_strength",
            path="particle.macro.macro_force_strength",
            current_value=0.5,
            candidates=[0.2, 0.3, 0.5, 0.7, 1.0],
            variable_type="continuous",
        ),
    ])

    # === TEMPORAL VARIABLES ===
    variables.extend([
        Variable(
            name="temporal_us_vol_mult",
            path="particle.temporal.us_vol_mult",
            current_value=1.3,
            candidates=[1.0, 1.2, 1.3, 1.5, 1.7],
            variable_type="continuous",
        ),
        Variable(
            name="temporal_overlap_vol_mult",
            path="particle.temporal.overlap_vol_mult",
            current_value=1.4,
            candidates=[1.2, 1.4, 1.6, 1.8],
            variable_type="continuous",
        ),
        Variable(
            name="temporal_recent_vol_weight",
            path="particle.temporal.recent_vol_weight",
            current_value=0.3,
            candidates=[0.1, 0.2, 0.3, 0.4, 0.5],
            variable_type="continuous",
        ),
    ])

    # === ORDER BOOK VARIABLES ===
    variables.extend([
        Variable(
            name="ob_imbalance_strength",
            path="particle.orderbook.imbalance_strength",
            current_value=0.08,
            candidates=[0.04, 0.06, 0.08, 0.10, 0.12],
            variable_type="continuous",
        ),
        Variable(
            name="ob_weight_level_1",
            path="particle.orderbook.weight_level_1",
            current_value=0.40,
            candidates=[0.3, 0.4, 0.5, 0.6],
            variable_type="continuous",
        ),
        Variable(
            name="ob_flow_ema_alpha",
            path="particle.orderbook.flow_ema_alpha",
            current_value=0.2,
            candidates=[0.1, 0.2, 0.3, 0.5],
            variable_type="continuous",
        ),
    ])

    return variables


def get_all_variables() -> List[Variable]:
    """Get all tunable variables."""
    return (
        get_predictor_variables() +
        get_domain_variables() +
        get_regime_variables() +
        get_particle_variables() +
        get_steering_field_variables()
    )


def get_default_targets() -> List[MetricTarget]:
    """Get default optimization targets."""
    return [
        MetricTarget(
            name="directional_accuracy",
            metric_type=MetricType.DIRECTIONAL_ACCURACY,
            target_value=0.55,  # Start modest - 55% is already useful
            direction="maximize",
            priority=1,
        ),
        MetricTarget(
            name="coverage",
            metric_type=MetricType.COVERAGE_90,
            target_value=0.90,
            direction="maximize",
            priority=2,
        ),
        MetricTarget(
            name="calibration",
            metric_type=MetricType.CALIBRATION,
            target_value=0.05,  # Want low calibration error
            direction="minimize",
            priority=3,
        ),
    ]
