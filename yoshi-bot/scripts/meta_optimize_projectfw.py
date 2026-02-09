"""Meta-optimization: Use ProjectFW to optimize its own hyperparameters.

This script demonstrates recursive optimization where ProjectFW optimizes
its own configuration parameters for best performance on a test suite.
"""
import numpy as np
import time
from typing import Dict, List
from dataclasses import asdict

from gnosis.harness.bregman_optimizer import (
    ProjectFWOptimizer,
    ProjectFWConfig,
    get_projectfw_hyperparameters,
)


def create_test_problems(n_problems: int = 5) -> List[Dict]:
    """Create a diverse suite of test optimization problems.
    
    Args:
        n_problems: Number of test problems to create
        
    Returns:
        List of problem dictionaries with objective, bounds, and optimal value
    """
    problems = []
    
    # Problem 1: Simple quadratic
    problems.append({
        "name": "Quadratic",
        "objective": lambda x: float(np.sum((x - 3.0) ** 2)),
        "bounds": [(0.0, 10.0)] * 5,
        "optimal_value": 0.0,
    })
    
    # Problem 2: Rosenbrock-like
    def rosenbrock_nd(x):
        """N-dimensional Rosenbrock function."""
        return float(sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 
                        for i in range(len(x)-1)))
    
    problems.append({
        "name": "Rosenbrock",
        "objective": rosenbrock_nd,
        "bounds": [(-2.0, 2.0)] * 4,
        "optimal_value": 0.0,
    })
    
    # Problem 3: Multi-modal
    def rastrigin(x):
        """Rastrigin function (many local minima)."""
        A = 10
        n = len(x)
        return float(A * n + sum(x_i**2 - A * np.cos(2 * np.pi * x_i) for x_i in x))
    
    problems.append({
        "name": "Rastrigin",
        "objective": rastrigin,
        "bounds": [(-5.12, 5.12)] * 6,
        "optimal_value": 0.0,
    })
    
    # Problem 4: Sphere with offset
    problems.append({
        "name": "Sphere-Offset",
        "objective": lambda x: float(np.sum((x - np.arange(len(x))) ** 2)),
        "bounds": [(-10.0, 10.0)] * 8,
        "optimal_value": 0.0,
    })
    
    # Problem 5: Ackley function
    def ackley(x):
        """Ackley function (difficult multi-modal)."""
        n = len(x)
        sum_sq = np.sum(x ** 2)
        sum_cos = np.sum(np.cos(2 * np.pi * x))
        return float(-20 * np.exp(-0.2 * np.sqrt(sum_sq / n)) 
                    - np.exp(sum_cos / n) + 20 + np.e)
    
    problems.append({
        "name": "Ackley",
        "objective": ackley,
        "bounds": [(-5.0, 5.0)] * 5,
        "optimal_value": 0.0,
    })
    
    return problems[:n_problems]


def evaluate_projectfw_config(
    config_params: np.ndarray,
    test_problems: List[Dict],
    time_budget: float = 1.0,
) -> float:
    """Evaluate ProjectFW performance with given configuration.
    
    Args:
        config_params: Array of normalized hyperparameter values [0, 1]
        test_problems: List of test problems
        time_budget: Maximum time per problem (seconds)
        
    Returns:
        Aggregate performance score (lower is better)
    """
    # Decode parameters from normalized [0, 1] to actual ranges
    param_specs = get_projectfw_hyperparameters()
    
    # Map normalized params to actual values
    max_iterations = int(config_params[0] * 100 + 10)  # 10-110
    tolerance = 10 ** (config_params[1] * (-8) - 2)  # 1e-10 to 1e-2
    alpha = 0.5 + config_params[2] * 0.5  # 0.5 to 1.0
    epsilon_init = 0.1 + config_params[3] * 9.9  # 0.1 to 10.0
    epsilon_decay = 0.8 + config_params[4] * 0.19  # 0.8 to 0.99
    epsilon_min = 10 ** (config_params[5] * (-6) - 2)  # 1e-8 to 1e-2
    patience = int(config_params[6] * 40 + 5)  # 5 to 45
    
    # Create config
    config = ProjectFWConfig(
        max_iterations=max_iterations,
        tolerance=tolerance,
        alpha=alpha,
        epsilon_init=epsilon_init,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        patience=patience,
        verbose=False,
    )
    
    # Evaluate on test problems
    scores = []
    times = []
    
    for problem in test_problems:
        optimizer = ProjectFWOptimizer(config)
        
        try:
            start = time.time()
            result = optimizer.optimize(
                objective=problem["objective"],
                bounds=problem["bounds"],
            )
            elapsed = time.time() - start
            
            # Score: objective gap + time penalty
            gap = max(0.0, result.f_best - problem["optimal_value"])
            time_penalty = max(0.0, elapsed - time_budget) * 10.0
            
            score = gap + time_penalty
            scores.append(score)
            times.append(elapsed)
            
        except Exception as e:
            # Penalize failed configurations
            scores.append(1000.0)
            times.append(time_budget * 2)
    
    # Aggregate score: mean + std (prefer consistent performance)
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    mean_time = np.mean(times)
    
    # Weighted combination
    total_score = mean_score + 0.2 * std_score + 0.1 * mean_time
    
    return float(total_score)


def meta_optimize_projectfw(
    n_test_problems: int = 5,
    meta_iterations: int = 30,
) -> Dict:
    """Meta-optimize ProjectFW hyperparameters using ProjectFW itself.
    
    This is recursive optimization: ProjectFW optimizes its own config.
    
    Args:
        n_test_problems: Number of test problems for evaluation
        meta_iterations: Number of meta-optimization iterations
        
    Returns:
        Dictionary with optimal config and results
    """
    print("=" * 70)
    print("META-OPTIMIZATION: ProjectFW Optimizing Itself")
    print("=" * 70)
    
    # Create test suite
    test_problems = create_test_problems(n_test_problems)
    print(f"\nTest Suite: {len(test_problems)} problems")
    for p in test_problems:
        print(f"  - {p['name']}: {len(p['bounds'])}D")
    
    # Define meta-objective
    def meta_objective(config_params: np.ndarray) -> float:
        """Objective for meta-optimization."""
        score = evaluate_projectfw_config(config_params, test_problems, time_budget=0.5)
        return score
    
    # Meta-optimization using ProjectFW with conservative settings
    print(f"\nRunning meta-optimization ({meta_iterations} iterations)...")
    
    meta_config = ProjectFWConfig(
        max_iterations=meta_iterations,
        tolerance=1e-4,
        alpha=0.85,
        epsilon_init=1.0,
        epsilon_decay=0.92,
        patience=8,
        verbose=True,
    )
    
    meta_optimizer = ProjectFWOptimizer(meta_config)
    
    # Bounds: 7 normalized parameters in [0, 1]
    bounds = [(0.0, 1.0) for _ in range(7)]
    
    # Initial guess: middle of range
    x0 = np.array([0.5] * 7)
    
    result = meta_optimizer.optimize(meta_objective, bounds, x0=x0)
    
    # Decode optimal configuration
    optimal_params = result.x_best
    optimal_config = ProjectFWConfig(
        max_iterations=int(optimal_params[0] * 100 + 10),
        tolerance=10 ** (optimal_params[1] * (-8) - 2),
        alpha=0.5 + optimal_params[2] * 0.5,
        epsilon_init=0.1 + optimal_params[3] * 9.9,
        epsilon_decay=0.8 + optimal_params[4] * 0.19,
        epsilon_min=10 ** (optimal_params[5] * (-6) - 2),
        patience=int(optimal_params[6] * 40 + 5),
    )
    
    # Compare against default config
    print(f"\n{'='*70}")
    print("Comparing Optimal vs Default Configuration")
    print(f"{'='*70}")
    
    default_config = ProjectFWConfig()
    
    default_score = evaluate_projectfw_config(
        np.array([0.5] * 7),  # Default corresponds to middle
        test_problems,
        time_budget=0.5,
    )
    
    optimal_score = result.f_best
    
    improvement = (default_score - optimal_score) / default_score * 100
    
    print(f"\nDefault Config Score: {default_score:.6f}")
    print(f"Optimal Config Score: {optimal_score:.6f}")
    print(f"Improvement: {improvement:.2f}%")
    
    print(f"\n{'='*70}")
    print("Optimal Configuration:")
    print(f"{'='*70}")
    for key, value in asdict(optimal_config).items():
        if key not in ['verbose', 'use_ip_solver', 'bregman_param', 'bregman_fn',
                       'ip_time_limit', 'ip_gap_tolerance', 'min_step_size', 
                       'max_step_size', 'log_frequency', 'track_best']:
            default_value = getattr(default_config, key)
            print(f"  {key:20s}: {value:12.6g}  (default: {default_value:12.6g})")
    
    return {
        "optimal_config": optimal_config,
        "optimal_score": optimal_score,
        "default_score": default_score,
        "improvement_pct": improvement,
        "meta_iterations": result.iterations,
        "test_problems": [p["name"] for p in test_problems],
    }


def main():
    """Run meta-optimization experiment."""
    np.random.seed(42)
    
    results = meta_optimize_projectfw(
        n_test_problems=5,
        meta_iterations=30,
    )
    
    print(f"\n{'='*70}")
    print("Meta-Optimization Complete!")
    print(f"{'='*70}")
    print(f"\nKey Findings:")
    print(f"  - Improvement over default: {results['improvement_pct']:.2f}%")
    print(f"  - Meta-iterations: {results['meta_iterations']}")
    print(f"  - Test suite: {', '.join(results['test_problems'])}")
    
    print(f"\nRecommendation:")
    print(f"  Use the optimal configuration above for better ProjectFW performance")
    print(f"  on diverse optimization problems.")


if __name__ == "__main__":
    main()
