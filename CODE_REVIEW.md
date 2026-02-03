# Yoshi-Bot Code Review

## Executive Summary

Yoshi-Bot is a quantitative trading prediction system that uses physics-inspired models to forecast cryptocurrency/asset prices. The codebase demonstrates sophisticated architecture with nested walk-forward validation, multiple prediction backends, and comprehensive evaluation metrics.

---

## Architecture Overview

### Core Components

| Module | Purpose | Key Files |
|--------|---------|-----------|
| **Particle Physics** | Price dynamics modeling | `particle/physics.py`, `particle/quantum.py` |
| **Predictors** | Quantile forecasting | `predictors/quantile.py`, `predictors/bregman_fw.py` |
| **Harness** | Walk-forward validation | `harness/walkforward.py`, `harness/ralph_loop.py` |
| **Backtest** | Trading simulation | `backtest/runner.py`, `backtest/portfolio.py` |
| **Evaluation** | Accuracy measurement | `evaluation/accuracy.py`, `harness/scoring.py` |
| **Improvement** | Hyperparameter tuning | `harness/improvement_loop.py` |

---

## Detailed Review

### 1. Particle Physics Framework (`src/gnosis/particle/physics.py`)

**Concept**: Models price as a particle moving through potential fields.

**Strengths**:
- Well-documented physics analogies (velocity, acceleration, jerk, mass, forces, energy)
- Comprehensive feature engineering with 35+ derived features
- Clean separation into kinematics, forces, and energy components

**Observations**:
- The physics metaphor is creative but should be treated as feature engineering, not literal physics
- Features like `momentum_alignment`, `breakout_potential`, and `reversion_potential` combine multiple signals effectively
- Mass is defined as inverse volatility - higher vol = lower mass = easier to move prices

**Key Features Generated**:
```
Kinematics: velocity, acceleration, jerk, momentum_alignment
Mass/Forces: mass, force_ofi, force_newton, force_impulse
Energy: kinetic_energy, potential_energy, energy_injection
Potential Field: distance_to_support/resistance, field_gradient
Volume: vwap, vwap_zscore, volume_momentum
```

### 2. Quantum Price Engine (`src/gnosis/particle/quantum.py`)

**Concept**: Monte Carlo simulation with regime-switching and steering fields.

**Strengths**:
- Regime detection (trending/ranging/volatile) with distinct physics parameters
- Multi-scale momentum aggregation (1m, 5m, 15m, 1h)
- Jump-diffusion model for liquidation cascades
- Comprehensive confidence intervals (50%, 68%, 80%, 90%, 95%, 99%)

**Key Components**:
- `SteeringForces`: Funding, imbalance, gravity, spring, liquidation, momentum, friction
- `GARCH-like volatility forecasting` with floor protection
- 10,000 Monte Carlo simulations by default

**Potential Improvements**:
- REGIME_PARAMS are hardcoded - could benefit from adaptive calibration
- Volatility floor of 2% may need adjustment per asset class

### 3. Walk-Forward Validation (`src/gnosis/harness/walkforward.py`)

**Strengths**:
- Proper purge/embargo gaps between train/val/test sets
- Configurable fold structure (outer_folds, train_days, val_days, test_days)
- Horizon-aware gap calculation

**Implementation**:
```python
train_end -> [PURGE_GAP] -> val_start
val_end -> [EMBARGO_GAP] -> test_start
```

### 4. Ralph Loop (`src/gnosis/harness/ralph_loop.py`)

**Purpose**: Nested hyperparameter selection without data leakage.

**Architecture**:
- Outer folds for true out-of-sample evaluation
- Inner folds within each outer-train for hyperparameter selection
- Never touches outer test for selection

**Scoring Composite**:
```python
score = 4.0 * |coverage - target| + 1.0 * WIS + 0.5 * IS90 + 1.0 * MAE + 0.5 * abstention_rate
```

**Notable Features**:
- Feature caching for structural hyperparameters (n_trades variations)
- Key-path mapping for config resolution
- Conditional vs unconditional coverage tracking

### 5. Quantile Predictor (`src/gnosis/predictors/quantile.py`)

**Backends Supported**:
1. **Ridge**: Fast baseline with sample weighting
2. **QuantileRegressor**: Proper quantile loss (sklearn)
3. **Bregman-FW**: Frank-Wolfe optimization with constraints
4. **GradientBoosting**: Nonlinear quantile regression

**Feature Sets**:
- Basic: returns, realized_vol, ofi, range_pct, flow_momentum, regime_stability
- Extended: Adds particle physics features (35+ additional)

### 6. Backtest Runner (`src/gnosis/backtest/runner.py`)

**Execution Model** (No Lookahead):
```
Bar t: Observe close[t], generate signal, create pending order
Bar t+1: Execute pending order at close[t+1] with slippage + fees
```

**Key Safeguards**:
- Assertion to prevent same-bar fills
- MIN_TRADE_SIZE filter (1e-8) to avoid dust trades
- Cash-only clamp for buys to prevent exceeding available capital

### 7. Prediction Evaluator (`src/gnosis/evaluation/accuracy.py`)

**Metrics Computed**:
- Directional accuracy (overall, long, short)
- Interval coverage (90%, 80%, 50%)
- Point errors (MAE, RMSE, MAPE)
- Expected Calibration Error (ECE)
- Time decay analysis

**Output Example**:
```
Horizon    Time         Direction   Coverage90  MAE      RMSE     Samples
-------    ----         ---------   ----------  ---      ----     -------
  1 bars      0.1 days    72.3%       91.2%     0.23%    0.35%      1000
  5 bars      0.5 days    65.1%       89.5%     0.45%    0.67%       996
 10 bars      1.0 days    58.7%       87.3%     0.78%    1.12%       990
```

### 8. Improvement Loop (`src/gnosis/harness/improvement_loop.py`)

**Algorithm**: Coordinate descent with target-based stopping.

**Process**:
1. Define targets (e.g., directional_accuracy >= 0.55)
2. For each unsatisfied target:
   - Try single-variable perturbations
   - Accept changes that improve the metric
   - Stop when target met or patience exhausted
3. Move to next target

---

## Risk Assessment

### Potential Overfitting Risks

1. **Many Features**: 35+ particle physics features could lead to spurious correlations
2. **Multiple Hyperparameters**: Grid search across confidence_floor, sigma_scale, n_trades, l2_reg, backend
3. **Nested Validation**: While the Ralph Loop prevents leakage, small datasets could still overfit

### Mitigation Strategies Present

1. Purge/embargo gaps in walk-forward
2. Inner CV for hyperparameter selection
3. L2 regularization in predictors
4. Abstention mechanism for low-confidence predictions

---

## Code Quality Assessment

### Strengths

| Aspect | Rating | Notes |
|--------|--------|-------|
| Documentation | Excellent | Comprehensive docstrings, clear module descriptions |
| Modularity | Excellent | Clean separation of concerns |
| Type Hints | Good | Dataclasses used extensively |
| Error Handling | Good | Graceful fallbacks (e.g., backend unavailable) |
| Reproducibility | Good | Random seeds used throughout |
| Testing | Present | Test files exist for major components |

### Areas for Enhancement

1. **Configuration Management**: Many magic numbers in REGIME_PARAMS could be externalized
2. **Logging**: Could benefit from structured logging beyond print statements
3. **Performance**: Some rolling operations could be vectorized further

---

## Recommendations

### Short-term
1. Add configuration file for REGIME_PARAMS tuning
2. Implement structured logging for production monitoring
3. Add unit tests for edge cases in physics calculations

### Medium-term
1. Consider adaptive regime parameters based on historical performance
2. Implement backtesting dashboard for visualization
3. Add model interpretability features (SHAP values for feature importance)

### Long-term
1. Evaluate simpler baseline models to ensure physics features add value
2. Consider ensemble approaches combining multiple backends
3. Implement online learning for regime parameter adaptation

---

## Conclusion

Yoshi-Bot demonstrates sophisticated quantitative trading methodology with proper cross-validation, multiple prediction approaches, and comprehensive evaluation. The physics-inspired feature engineering is creative, though the true predictive value should be validated against simpler baselines. The codebase is well-organized and maintainable.

**Overall Assessment**: Production-ready architecture with appropriate safeguards against common pitfalls (lookahead bias, data leakage). The physics metaphor should be treated as creative feature engineering rather than literal market dynamics.

---

*Review Date: 2026-02-03*
*Reviewer: Claude Code Review*
