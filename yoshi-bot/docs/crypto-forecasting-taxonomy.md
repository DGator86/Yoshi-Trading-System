# Crypto Forecasting Taxonomy (Practical, Modular)

This document captures a practical forecasting map for crypto and maps it to
implementation constraints suitable for production systems.

Core principle:
- do not optimize for point-price heroics
- optimize for calibrated distributions, regime probabilities, and tail risk

---

## 1) Forecasting targets (must be explicit)

Recommended target set:
- next-horizon return
- direction
- quantiles / distribution
- volatility / range
- tail event and jump probability
- regime label
- barrier probability

Why:
- point-price alone is unstable in heavy-tailed crypto return processes.

---

## 2) Twelve forecasting paradigms (module view)

1. technical_price_action
2. classical_timeseries
3. macro_factor
4. derivatives_positioning
5. orderflow_microstructure
6. onchain_priors
7. sentiment_attention
8. ml_tabular_meta
9. deep_sequence_distribution
10. regime_state_machine
11. scenario_monte_carlo
12. crowd_prediction_markets

Implementation anchor:
- `src/gnosis/forecasting/modular_ensemble.py`

The module registry in code defines:
- inputs
- output schema
- trust conditions
- failure modes

---

## 3) Conditions of application (gating policy)

Use a regime-aware gating policy, not static blending.

Inputs:
- regime probabilities
- liquidity state (spread + depth)
- leverage fragility index (LFI)
- jump probability
- event-window flag

Outputs:
- module weights `w_i`
- confidence scalar `c_t` for interval widening/narrowing

Rule examples:
- cascade/LFI-high: upweight derivatives + scenario + regime machine, downweight pure TA
- trend + healthy liquidity: upweight momentum/ML sequence modules
- range regime: upweight mean-reversion/statistical modules
- event windows: reduce confidence, emphasize distribution and tail-risk modules

---

## 4) Evaluation protocol (non-negotiable)

- walk-forward or expanding-window validation only
- embargo enabled
- no random split evaluation for time series
- report per-regime and per-volatility-bucket metrics

Target-aligned metrics:
- direction: hit rate + balanced metrics (MCC)
- quantiles/distribution: pinball loss, coverage, sharpness, calibration
- tails/jumps: Brier score + precision/recall
- regimes: log loss + whipsaw/transition quality

Serialized policy + metric defaults:
- `configs/crypto_forecasting_taxonomy.yaml`

---

## 5) Prompt artifact for external LLM integration

The reusable handoff prompt is versioned in:
- `docs/crypto-forecasting-taxonomy-prompt.txt`

