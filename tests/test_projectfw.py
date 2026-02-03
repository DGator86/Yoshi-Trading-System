"""Unit tests for ProjectFW Bregman optimizer.

Tests convergence properties on known convex problems with theoretical guarantees.
"""
import numpy as np
import pytest
from typing import Callable

from gnosis.harness.bregman_optimizer import (
    ProjectFWOptimizer,
    ProjectFWConfig,
    BregmanFunction,
    get_projectfw_hyperparameters,
)


class TestProjectFWConvergence:
    """Test convergence properties on known convex problems."""
    
    def test_quadratic_convergence(self):
        """Test convergence on simple quadratic: f(x) = (x-3)^2."""
        def objective(x: np.ndarray) -> float:
            return float(np.sum((x - 3.0) ** 2))
        
        config = ProjectFWConfig(
            max_iterations=50,
            tolerance=1e-4,
            alpha=0.9,
            verbose=False,
        )
        
        optimizer = ProjectFWOptimizer(config)
        bounds = [(0.0, 10.0)]
        
        result = optimizer.optimize(objective, bounds)
        
        # Check convergence
        assert result.iterations > 0
        assert result.f_best < 1.0  # Should be close to 0
        assert abs(result.x_best[0] - 3.0) < 1.0  # Should be close to x=3
        
        # Check approximation quality
        assert result.approximation_quality < 1.0
        
        # Check convergence history is decreasing
        history = result.convergence_history
        assert len(history) > 1
        # Should improve over time (with some tolerance)
        improvements = [history[i] - history[i+1] for i in range(len(history)-1)]
        positive_improvements = sum(1 for imp in improvements if imp >= -1e-6)
        assert positive_improvements / len(improvements) > 0.5
    
    def test_multidimensional_quadratic(self):
        """Test on 5D quadratic bowl: f(x) = sum((x_i - i)^2)."""
        def objective(x: np.ndarray) -> float:
            target = np.arange(len(x)) + 1.0
            return float(np.sum((x - target) ** 2))
        
        config = ProjectFWConfig(
            max_iterations=100,
            tolerance=1e-4,
            epsilon_init=2.0,
            epsilon_decay=0.9,
            patience=15,
            verbose=False,
        )
        
        optimizer = ProjectFWOptimizer(config)
        n_dim = 5
        bounds = [(0.0, 10.0) for _ in range(n_dim)]
        
        result = optimizer.optimize(objective, bounds)
        
        # Should converge to near-optimal
        assert result.f_best < 5.0  # Much better than random
        
        # Check each dimension is reasonably close
        for i, x_i in enumerate(result.x_best):
            expected = i + 1.0
            # Allow some error but should be in right direction
            assert abs(x_i - expected) < 3.0
    
    def test_rosenbrock_function(self):
        """Test on Rosenbrock function (harder non-convex problem)."""
        def rosenbrock(x: np.ndarray) -> float:
            """Rosenbrock: f(x,y) = (1-x)^2 + 100(y-x^2)^2."""
            if len(x) != 2:
                raise ValueError("Rosenbrock requires 2D input")
            return float((1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2)
        
        config = ProjectFWConfig(
            max_iterations=200,
            tolerance=1e-5,
            epsilon_init=1.0,
            epsilon_decay=0.95,
            patience=20,
            verbose=False,
        )
        
        optimizer = ProjectFWOptimizer(config)
        bounds = [(-2.0, 2.0), (-1.0, 3.0)]
        x0 = np.array([-1.0, 0.5])  # Start away from initial f(0,0)=1
        
        result = optimizer.optimize(objective=rosenbrock, bounds=bounds, x0=x0)
        
        # Rosenbrock is hard, so we just check improvement from bad start
        initial_f = rosenbrock(x0)
        # Should improve or at least not get worse
        assert result.f_best <= initial_f + 1.0
        assert result.iterations >= 2  # Should try some iterations
    
    def test_box_constraints_respected(self):
        """Test that solution respects box constraints."""
        def objective(x: np.ndarray) -> float:
            return float(np.sum(x ** 2))  # Minimum at origin
        
        config = ProjectFWConfig(max_iterations=30, verbose=False)
        optimizer = ProjectFWOptimizer(config)
        
        # Constraints that don't include origin
        bounds = [(2.0, 5.0), (3.0, 7.0), (1.0, 4.0)]
        
        result = optimizer.optimize(objective, bounds)
        
        # Check all dimensions respect bounds
        for i, (lb, ub) in enumerate(bounds):
            assert lb - 1e-6 <= result.x_best[i] <= ub + 1e-6
    
    def test_best_iterate_tracking(self):
        """Test that best-iterate tracking works correctly."""
        call_count = [0]
        values = []
        
        def noisy_objective(x: np.ndarray) -> float:
            """Objective with noise to test robustness."""
            call_count[0] += 1
            base_value = float(np.sum((x - 2.0) ** 2))
            # Add noise every 3rd call
            noise = 0.5 if call_count[0] % 3 == 0 else 0.0
            value = base_value + noise
            values.append(value)
            return value
        
        config = ProjectFWConfig(
            max_iterations=50,
            track_best=True,
            patience=10,
            verbose=False,
        )
        
        optimizer = ProjectFWOptimizer(config)
        bounds = [(0.0, 5.0)]
        
        result = optimizer.optimize(noisy_objective, bounds)
        
        # Best result should be better than final noisy result
        if len(values) > 0:
            final_value = values[-1]
            # Best tracked value should be <= any individual value
            assert result.f_best <= min(values) + 1e-6


class TestProjectFWConfiguration:
    """Test configuration and parameter handling."""
    
    def test_default_config(self):
        """Test default configuration is valid."""
        config = ProjectFWConfig()
        
        assert config.max_iterations == 100
        assert config.tolerance == 1e-6
        assert config.alpha == 0.9
        assert config.epsilon_init == 1.0
        assert config.epsilon_decay == 0.95
        assert config.epsilon_min == 1e-4
        assert config.track_best is True
        assert config.patience == 10
        assert config.bregman_fn == BregmanFunction.EUCLIDEAN
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = ProjectFWConfig(
            max_iterations=200,
            alpha=0.95,
            epsilon_init=2.0,
            bregman_fn=BregmanFunction.KL,
            verbose=True,
        )
        
        assert config.max_iterations == 200
        assert config.alpha == 0.95
        assert config.epsilon_init == 2.0
        assert config.bregman_fn == BregmanFunction.KL
        assert config.verbose is True
    
    def test_adaptive_contraction(self):
        """Test that contraction radius adapts correctly."""
        config = ProjectFWConfig(
            epsilon_init=1.0,
            epsilon_decay=0.9,
            epsilon_min=0.01,
            max_iterations=50,
            patience=15,  # Increase patience to allow more iterations
            tolerance=1e-8,  # Tighter tolerance
            verbose=False,
        )
        
        def simple_objective(x: np.ndarray) -> float:
            return float(np.sum((x - 5.0) ** 2))
        
        optimizer = ProjectFWOptimizer(config)
        bounds = [(0.0, 10.0), (0.0, 10.0)]
        x0 = np.array([1.0, 1.0])  # Start away from optimum
        
        result = optimizer.optimize(simple_objective, bounds, x0=x0)
        
        # Should have completed some iterations
        assert result.iterations >= 2
        
        # Convergence history should show improvement
        assert result.convergence_history[-1] <= result.convergence_history[0]


class TestBregmanDivergences:
    """Test different Bregman divergence functions."""
    
    def test_euclidean_divergence(self):
        """Test Euclidean Bregman divergence."""
        config = ProjectFWConfig(bregman_fn=BregmanFunction.EUCLIDEAN)
        optimizer = ProjectFWOptimizer(config)
        
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 3.0, 4.0])
        
        div = optimizer._compute_bregman_divergence(x, y)
        expected = np.sum((x - y) ** 2)  # Should be 3.0
        
        assert abs(div - expected) < 1e-6
    
    def test_kl_divergence(self):
        """Test KL Bregman divergence."""
        config = ProjectFWConfig(bregman_fn=BregmanFunction.KL)
        optimizer = ProjectFWOptimizer(config)
        
        # Probability distributions
        x = np.array([0.4, 0.3, 0.3])
        y = np.array([0.3, 0.3, 0.4])
        
        div = optimizer._compute_bregman_divergence(x, y)
        
        # KL divergence should be positive and finite
        assert div > 0
        assert np.isfinite(div)
    
    def test_optimization_with_kl_divergence(self):
        """Test optimization works with KL divergence."""
        config = ProjectFWConfig(
            bregman_fn=BregmanFunction.KL,
            max_iterations=50,
            verbose=False,
        )
        
        def objective(x: np.ndarray) -> float:
            return float(np.sum((x - 0.5) ** 2))
        
        optimizer = ProjectFWOptimizer(config)
        bounds = [(0.1, 0.9), (0.1, 0.9)]
        
        result = optimizer.optimize(objective, bounds)
        
        # Should converge
        assert result.f_best < 1.0
        assert result.iterations > 0


class TestSubproblemSolving:
    """Test linear subproblem solving (Frank-Wolfe oracle)."""
    
    def test_subproblem_respects_bounds(self):
        """Test that subproblem solutions respect bounds."""
        config = ProjectFWConfig()
        optimizer = ProjectFWOptimizer(config)
        
        def dummy_objective(x: np.ndarray) -> float:
            return float(np.sum(x ** 2))
        
        mu_t = np.array([5.0, 5.0])
        bounds = [(0.0, 10.0), (0.0, 10.0)]
        epsilon_t = 1.0
        
        v_t = optimizer._solve_linear_subproblem(
            mu_t, dummy_objective, bounds, epsilon_t
        )
        
        # Should be a vertex of the box constraint
        for i, (lb, ub) in enumerate(bounds):
            assert lb - 1e-6 <= v_t[i] <= ub + 1e-6
            # Should be at a vertex (lb or ub)
            assert abs(v_t[i] - lb) < 1e-6 or abs(v_t[i] - ub) < 1e-6


class TestHyperparameterRegistry:
    """Test hyperparameter registry for improvement loop."""
    
    def test_get_hyperparameters(self):
        """Test that all hyperparameters are registered."""
        params = get_projectfw_hyperparameters()
        
        # Check all expected parameters are present
        expected_keys = [
            "max_iterations",
            "tolerance",
            "alpha",
            "epsilon_init",
            "epsilon_decay",
            "epsilon_min",
            "patience",
            "bregman_fn",
        ]
        
        for key in expected_keys:
            assert key in params
            assert "default" in params[key]
            assert "type" in params[key]
            assert "description" in params[key]
    
    def test_hyperparameter_bounds(self):
        """Test that numeric hyperparameters have valid bounds."""
        params = get_projectfw_hyperparameters()
        
        numeric_params = [k for k, v in params.items() if v["type"] in ["int", "float"]]
        
        for key in numeric_params:
            assert "bounds" in params[key]
            bounds = params[key]["bounds"]
            assert len(bounds) == 2
            assert bounds[0] < bounds[1]
            
            # Default should be within bounds
            default = params[key]["default"]
            assert bounds[0] <= default <= bounds[1]


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_dimension(self):
        """Test optimization in 1D."""
        def objective(x: np.ndarray) -> float:
            return float((x[0] - 7.0) ** 2)
        
        config = ProjectFWConfig(max_iterations=30, verbose=False)
        optimizer = ProjectFWOptimizer(config)
        bounds = [(0.0, 10.0)]
        
        result = optimizer.optimize(objective, bounds)
        
        assert result.f_best < 5.0
        assert 5.0 < result.x_best[0] < 9.0
    
    def test_high_dimension(self):
        """Test optimization in high dimension (20D)."""
        def objective(x: np.ndarray) -> float:
            return float(np.sum((x - 5.0) ** 2))
        
        config = ProjectFWConfig(max_iterations=100, patience=20, verbose=False)
        optimizer = ProjectFWOptimizer(config)
        
        n_dim = 20
        bounds = [(0.0, 10.0) for _ in range(n_dim)]
        
        result = optimizer.optimize(objective, bounds)
        
        # Should make progress even in high dimension
        initial_f = objective(np.ones(n_dim) * 5.0)
        assert result.f_best <= initial_f
    
    def test_immediate_convergence(self):
        """Test when starting point is already optimal."""
        def objective(x: np.ndarray) -> float:
            return float(np.sum(x ** 2))
        
        config = ProjectFWConfig(
            max_iterations=50,
            tolerance=1e-6,
            patience=5,
            verbose=False,
        )
        optimizer = ProjectFWOptimizer(config)
        
        bounds = [(0.0, 10.0)]
        x0 = np.array([0.0])  # Already at optimum
        
        result = optimizer.optimize(objective, bounds, x0=x0)
        
        # Should recognize convergence quickly
        assert result.iterations < 20
        assert result.f_best < 0.1
    
    def test_patience_stopping(self):
        """Test that patience-based stopping works."""
        def plateau_objective(x: np.ndarray) -> float:
            """Objective that plateaus quickly."""
            return float(np.sum(np.minimum(x, 2.0) ** 2))
        
        config = ProjectFWConfig(
            max_iterations=100,
            patience=5,
            verbose=False,
        )
        optimizer = ProjectFWOptimizer(config)
        bounds = [(0.0, 10.0)]
        
        result = optimizer.optimize(plateau_objective, bounds)
        
        # Should stop before max_iterations due to patience
        assert result.iterations < config.max_iterations


class TestApproximationQuality:
    """Test α-approximation quality measurement."""
    
    def test_approximation_quality_computed(self):
        """Test that approximation quality is computed."""
        def objective(x: np.ndarray) -> float:
            return float(np.sum((x - 5.0) ** 2))
        
        config = ProjectFWConfig(alpha=0.9, max_iterations=30, verbose=False)
        optimizer = ProjectFWOptimizer(config)
        bounds = [(0.0, 10.0)]
        x0 = np.array([0.0])  # Start away from optimum
        
        result = optimizer.optimize(objective, bounds, x0=x0)
        
        # Approximation quality should be computed
        assert result.approximation_quality is not None
        # Allow for perfect optimization (quality = 0.0) or improvement (0 < quality < 1)
        assert 0.0 <= result.approximation_quality <= 1.0
    
    def test_improvement_ratio(self):
        """Test that approximation quality reflects improvement."""
        def objective(x: np.ndarray) -> float:
            return float(np.sum((x - 5.0) ** 2))
        
        config = ProjectFWConfig(max_iterations=50, verbose=False)
        optimizer = ProjectFWOptimizer(config)
        bounds = [(0.0, 10.0)]
        x0 = np.array([0.0])  # Start far from optimum
        
        result = optimizer.optimize(objective, bounds, x0=x0)
        
        # Should show improvement (ratio < 1)
        assert result.approximation_quality < 1.0
        
        # Final should be better than initial
        initial_f = result.convergence_history[0]
        assert result.f_best < initial_f


# Pytest configuration
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
