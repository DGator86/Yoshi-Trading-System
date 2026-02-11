# Yoshi Trading System - Repository Review

## Overview

The **Yoshi Trading System** is a physics-inspired quantitative trading platform for
Kalshi crypto prediction markets (hourly BTC/ETH binary-option contracts). It is a
monorepo (~139K lines of Python across 266 files) composed of two main subsystems:

| Component | Role | Scale |
|-----------|------|-------|
| **yoshi-bot** | Forecasting & ML engine | 99 files, 24 subpackages |
| **clawdbot** | Telegram interface, orchestration, execution | 114 files, 11 subpackages |

### Architecture at a Glance

```
Kalshi / CoinGecko / CCXT Exchanges
         |
    kalshi_scanner.py          -- real-time OHLCV fetch
         |
    PriceParticle              -- 35+ physics-derived features
         |
    QuantumPriceEngine         -- 10,000 Monte Carlo simulations, regime switching
         |
    QuantilePredictor          -- ML forecast (Ridge / Bregman-FW / GradientBoosting)
         |
    Ralph Loop                 -- walk-forward CV + hyperparameter tuning
         |
    Value-Play Detection       -- edge = model_prob - market_prob > threshold
         |
    scanner.log --> yoshi-bridge.py --> Trading Core API (FastAPI :8000)
         |
    Circuit Breaker --> Kalshi Order --> Telegram Alert --> User
```

### Key Design Decisions

1. **Physics model first** - price modeled as a particle in a force field (velocity,
   acceleration, spring forces, friction, energy). Interpretable by design.
2. **Probabilistic forecasts** - 6 prediction intervals (50%-99%) with an abstention
   mechanism; the system can decline to trade.
3. **Regime-aware** - separate physics parameters for Trending, Ranging, and Volatile
   market states.
4. **Walk-forward validation** - nested CV with purge gaps and embargo periods
   (Ralph Loop) to prevent overfitting.
5. **Modular execution** - FastAPI decouples forecast from trade placement; Telegram
   provides the user interface.

### Infrastructure

- **Host**: DigitalOcean droplet (8 GB RAM, 2 vCPU, Ubuntu 24.04)
- **Services**: 6 systemd units (`trading-core`, `kalshi-scanner`, `yoshi-bridge`,
  `clawdbot`, `ralph-learner`, `ralph-learner.timer`)
- **Markets**: Kalshi KXBTC / KXETH hourly contracts

---

## SWOT Analysis

### Strengths

| # | Strength | Detail |
|---|----------|--------|
| S1 | **Interpretable physics framework** | Unlike black-box ML, the particle model yields intuitive features (momentum = velocity, support = potential energy, trend = force). Debugging and improving the model is tractable because the features have physical meaning. |
| S2 | **Rigorous walk-forward validation** | Ralph Loop's nested CV with purge gaps and embargo periods is best-practice for financial time series and actively prevents the most common quant strategy killer: overfitting to historical data. |
| S3 | **Probabilistic forecasts with abstention** | 6 prediction intervals (50%-99%) plus a confidence floor that allows the system to say "I don't know." Most retail strategies always take a position; Yoshi can sit out low-confidence periods. |
| S4 | **Regime-aware adaptation** | Separate physics parameters for trending, ranging, and volatile markets. Strategies that optimize for a single regime fail when the market shifts; Yoshi models all three explicitly. |
| S5 | **Production-ready architecture** | FastAPI + systemd + Telegram + circuit breakers. This is a deployable system with health checks, safety cutoffs, and user-facing controls — not a Jupyter notebook. |
| S6 | **Multi-backend ML** | 4 predictor backends (Ridge, QuantileRegressor, Bregman-FW, GradientBoosting) with automatic selection. No single-model fragility. |
| S7 | **Comprehensive prediction test battery** | 8 synthetic test suites evaluating calibration, sharpness, coverage, and directional accuracy. More rigorous than most institutional quant shops. |
| S8 | **Niche market focus** | Kalshi hourly crypto contracts are lower-competition, higher-edge environments compared to spot crypto or equities. The binary-option format is naturally suited to probabilistic forecasting. |

### Weaknesses

| # | Weakness | Severity | Detail |
|---|----------|----------|--------|
| W1 | **Hardcoded secrets in git history** | Critical | 2 Telegram bot tokens, an OpenRouter API key, CoinGecko/CoinMarketCap keys, and the VPS IP address are committed in plaintext across 10+ files. `.gitignore` prevents new leaks but these are already in history. Requires token rotation + `git filter-repo`. |
| W2 | **~40-50% code duplication** | High | `yoshi-bot/src/gnosis/` (99 files) and `clawdbot/gnosis/` (58 files) share roughly half their code but maintain diverging copies. There are 3 separate copies of `kalshi_client.py` with different implementations. Bug fixes in one copy won't reach the others. |
| W3 | **~200 magic numbers without justification** | High | Spring constants, friction coefficients, energy weights, and regime thresholds in `physics.py` and `quantum.py` are hardcoded with no sensitivity analysis or empirical calibration. |
| W4 | **Potential data leakage in feature computation** | High | `FeatureCacheManager.get_or_compute()` in `ralph_loop.py` computes features on the full dataset before CV folding. Physics features that use look-ahead indicators could inflate validation metrics. |
| W5 | **Unrealistic execution model** | High | Slippage is fixed OR vol-proportional (not both). Fees are only applied on the sell side. Orders exceeding cash are silently scaled down. No position limits in the backtest runner. Backtests likely overstate real P&L. |
| W6 | **No CI/CD pipeline** | High | 21 test files exist but are run manually. No automated checks on push, no coverage tracking, no dependency vulnerability scanning. Regressions can ship silently. |
| W7 | **Unpinned dependencies** | High | `pyproject.toml` lists `numpy`, `pandas`, `scipy` with zero version constraints. A fresh `pip install` could pull numpy 2.0 (breaking) or a compromised package. |
| W8 | **Silent error swallowing** | Medium | Bare `except Exception: pass` in quantum param reload; Kalshi client returns `None` for both "no data" and "API error"; PEM parser returns an invalid key on all parse failures. Critical paths fail silently. |
| W9 | **Fragile deployment** | Medium | `vps-deploy.sh` runs `git reset --hard && git clean -fd` with no backup, no rollback, no post-deploy health check, and credentials embedded in the script body. |
| W10 | **Low test coverage on safety-critical code** | Medium | The circuit breaker (last line of defense against runaway losses) has zero tests. No negative/edge-case tests anywhere. No `conftest.py` for shared fixtures. |

### Opportunities

| # | Opportunity | Detail |
|---|------------|--------|
| O1 | **8 identified feature gaps ready to implement** | Cross-exchange funding aggregation, time-of-day vol multipliers, liquidation heatmaps, multi-level order book depth, gamma fields, cross-asset coupling. Each independently improves signal quality. ~1,300 lines estimated (per `FEATURE_GAPS.md`). |
| O2 | **Extract `gnosis-core` shared library** | Deduplicating yoshi-bot and clawdbot into a single shared package would eliminate ~30K duplicated lines, stop divergence, and make the system significantly easier to maintain and extend. |
| O3 | **Ensemble multiple prediction backends** | The 4 backends currently compete via Ralph selection. Stacking or Bayesian model averaging could improve forecast accuracy with minimal code changes. |
| O4 | **Expand to more Kalshi markets** | Currently BTC/ETH only. SOL, economic events (CPI, Fed rate), and weather markets use the same binary-option format and could reuse the entire pipeline with new data adapters. |
| O5 | **Live performance dashboards** | No Sharpe ratio, drawdown, or win-rate tracking in production. A metrics service would enable data-driven parameter tuning and strategy iteration. |
| O6 | **CI/CD with secret scanning** | Adding `git-secrets` + `pip-audit` + `pytest` to a GitHub Actions pipeline catches credentials, vulnerable deps, and regressions automatically. Low effort, high payoff. |
| O7 | **Empirical physics parameter calibration** | Walk-forward grid search over spring constants, friction, and energy weights could replace the 200+ magic numbers with data-driven values. The infrastructure (Ralph Loop) already supports this. |
| O8 | **RL agent for position sizing** | The `rl/` module is scaffolded but incomplete. A reinforcement learning agent optimizing Kelly-criterion position sizing could improve risk-adjusted returns. |
| O9 | **Multi-timeframe stacking** | The `mtf/` module is scaffolded. Combining 5m, 15m, 1h, and 4h forecasts could improve hourly contract accuracy by capturing both micro and macro trends. |

### Threats

| # | Threat | Detail |
|---|--------|--------|
| T1 | **Credential compromise** | Telegram tokens and API keys are in git history. If this repo has ever been public or forked, those credentials are exposed. Attackers could hijack the Telegram bot, drain API credits, or target the VPS directly. |
| T2 | **Overfitting despite safeguards** | 35+ features, 8+ hyperparameters, and 200+ magic numbers create a vast search space. Even with walk-forward CV, researcher degrees of freedom (feature selection, config choices) can introduce implicit overfitting. |
| T3 | **Kalshi regulatory or API changes** | Kalshi is CFTC-regulated and has changed its API, fee structure, and contract offerings multiple times. A single breaking API change halts all trading. |
| T4 | **Unseen market regimes** | Physics parameters are calibrated to historical data. A novel regime (regulatory shock, exchange collapse, unprecedented volatility) could produce garbage forecasts with high stated confidence. |
| T5 | **Single point of failure: one VPS** | One DigitalOcean droplet runs everything. No redundancy, no failover, no geographic distribution. A droplet outage means complete trading halt with no automatic recovery. |
| T6 | **Backtest-to-live P&L gap** | The unrealistic execution model (W5) means backtested profitability likely overstates live performance. This is the most common failure mode for quant strategies — looking good on paper, losing money live. |
| T7 | **Dependency supply-chain risk** | Unpinned deps + no vulnerability scanning means a compromised or typosquatted PyPI package could inject malicious code during installation. |
| T8 | **Accelerating code divergence** | As features are added independently to yoshi-bot and clawdbot, the two gnosis copies diverge further. A critical bug fix missed in one copy will eventually cause a production incident. |

---

## Prioritized Recommendations

### Immediate (This Week)

1. **Rotate all leaked credentials** and rewrite git history with `git filter-repo`
   - Revoke: both Telegram tokens, OpenRouter key, CoinGecko/CoinMarketCap keys
   - Regenerate new tokens, store only in `.env`
   - Force-push cleaned history (coordinate with any collaborators)

2. **Pin all dependencies** in `pyproject.toml` and `requirements-ml.txt`
   - Use compatible-release constraints: `numpy~=1.26`, `pandas~=2.1`, etc.
   - Add `pip-audit` to pre-commit or CI

### Short-Term (Next 2 Weeks)

3. **Extract `gnosis-core` shared library** to eliminate duplication
   - Single source of truth for `kalshi_client`, `execution/`, `particle/`, `predictors/`
   - Both yoshi-bot and clawdbot import from it

4. **Fix feature computation data leakage** in `ralph_loop.py`
   - Compute features per fold (train window only, then transform test window)
   - Re-validate all backtest results after the fix

5. **Add CI/CD pipeline** (GitHub Actions)
   - `pytest` on push, `git-secrets` pre-commit, `pip-audit` scheduled

6. **Improve execution model realism**
   - Combined fixed + vol-proportional slippage
   - Fees on both buy and sell sides
   - Log warnings on order scaling; enforce position limits

### Medium-Term (Next Month)

7. **Calibrate physics parameters empirically** via Ralph Loop grid search
8. **Add circuit breaker tests** and edge-case coverage for all safety-critical code
9. **Replace silent error swallowing** with structured logging + alerting
10. **Harden deployment**: backup before reset, rollback on failure, post-deploy health check

### Long-Term (Next Quarter)

11. **Expand to additional Kalshi markets** (SOL, economic events)
12. **Implement ensemble forecasting** (stack the 4 ML backends)
13. **Complete RL position-sizing agent**
14. **Add live performance dashboards** (Sharpe, drawdown, win rate)
15. **Add VPS redundancy** (second droplet, health-check failover)
