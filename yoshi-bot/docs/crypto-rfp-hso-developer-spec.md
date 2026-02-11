# Crypto RFP HSO Developer Handoff Spec (BTC/ETH/SOL)

This document captures the crypto-native handoff specification for the
Regime Field Probability Hilbert Space Overlay + Walkforward Collapse
Projection system in this repository.

Design intent:
- print-first (quantized event-time)
- prints as collapse events
- candles used only for visualization and scheduler boundaries

---

## 0) Objective

Build a crypto prediction system that:
1. Quantizes time by trade prints into notional buckets (event-clock).
2. Computes a market field state per bucket (FeatureSignature).
3. Projects that state into regime probabilities (Hilbert projection + entropy).
4. Computes a 25-node state posterior (Class x Order).
5. Uses a semi-Markov transition operator for duration-aware propagation.
6. Produces a forward collapse-location distribution (q05/q50/q95).
7. Renders:
   - historical regime shading (color = dominant regime, opacity = 1 - entropy)
   - forward regime ribbon
   - forward quantile fan

Universe: BTCUSDT, ETHUSDT, SOLUSDT (single venue first, multi-venue later).

---

## 1) Data requirements

Minimum viable per symbol:
- trade prints: timestamp, price, size, side (if available)
- L2 snapshots: best bid/ask + depth ladder (top-N, e.g. 50 levels)
- perp OI time series
- funding rate + next/last funding timestamps
- liquidation prints or liquidation proxy by side/time

Enhanced:
- multi-venue aggregation (Binance + OKX + Bybit)
- Deribit options OI/IV for BTC/ETH
- on-chain flow metrics
- news/event flags

Hard constraint:
- skipping book + OI/funding + liquidations reduces collapse prediction quality.

---

## 2) Quantized time (print buckets)

Default mode: NOTIONAL buckets.

Bucket closes when traded notional reaches threshold:
- BTCUSDT: 5,000,000 USD
- ETHUSDT: 2,000,000 USD
- SOLUSDT:   500,000 USD

Required bucket outputs:
- OHLC
- VWAP
- volume
- notional
- n_trades

Reason:
- event-time comparability across volatility/trade-frequency regimes.

---

## 3) Regime taxonomy

### 3.1 Class (5)
- Balanced
- Discovery
- Pinning
- Shock
- Transitional

### 3.2 Order (5)
- Liquidity-Contained
- Liquidity-Release
- Positioning-Constraint
- Information-Override
- Correlation-Driven

### 3.3 Node space
- 25 nodes: `{Class}|{Order}`
- posterior from outer product with validity gating.

---

## 4) FeatureSignature per bucket

### 4.1 Price/diffusion
- `r_k = ln(close_k / close_{k-1})`
- `mu_k`, `sigma_k` (EWMA)
- `vol_of_vol_k` (EWMA std of sigma changes)

### 4.2 Liquidity/microstructure (L2)
- mid, spread
- bid/ask depth at x bps (x in {5,10,25})
- imbalance at 10bps
- book slopes
- impact proxy
- local liquidity surrogates:
  - `U2_liq` (stiffness proxy)
  - `F_liq` (imbalance * stiffness)

### 4.3 Perp positioning / forced flow
- OI, dOI
- funding, dfunding
- liquidation notionals (long/short), liq intensity
- positioning surrogates:
  - `U2_perp`
  - `F_perp`

### 4.4 Coupling
- rolling beta/corr vs BTC
- coupling score from `|beta| * corr`

### 4.5 Shock detectors
- gap flag
- book vacuum
- liquidation cascade flag

### 4.6 Friction
- function of spread, impact_proxy, vol_of_vol

### 4.7 Hurst
- GHE (`q=2`) on print-time log prices
- clamp to [0.2, 0.8]

---

## 5) Hilbert regime projection

1. Build normalized feature vector `psi_k`.
2. Use class templates `t_r` (bootstrapped from heuristics).
3. Score with dot products and softmax:
   - `score_r = dot(t_r, psi_k)`
   - `w_class = softmax(temp * score_r)`
4. Entropy overlay:
   - `H_norm = H / ln(5)`
   - opacity `= 1 - H_norm`

---

## 6) Order scoring

Softmax over 5 order scores:
- Liquidity-Contained
- Liquidity-Release
- Positioning-Constraint
- Information-Override
- Correlation-Driven

Initial formulas:
- LC: `+U2_liq - sigma - |mu| + (1 - friction)`
- LR: `+(1 - U2_liq) + |mu| + sigma - friction`
- PC: `+U2_perp + |funding| + liq_intensity + |dOI|`
- IO: `+gap_flag + liq_cascade_flag + book_vacuum`
- CD: `+coupling + corr_to_btc - idiosyncratic_strength`

---

## 7) Node posterior + age

- `w_node(c,o) = w_class(c) * w_order(o)` then apply validity mask.
- `dom_node = argmax(w_node)`
- `age = prev_age + 1` if unchanged else `1`.

---

## 8) Semi-Markov transition model

From dominant node history:
- fit per-state duration tails (`alpha_s`) using power-law MLE for `d >= tau_min`
- fit exit matrix `T_exit` with diagonal forced to zero

Hazard:
- `h_s(age) = 1 - (age/(age+1))**alpha_s`, clamped to `[h_min, h_max]`

One-step transitions:
- stay: `1 - h_s(age)`
- exit: `h_s(age) * T_exit[s][j]`

Forward propagation:
- produce `pi_forward[tau][state]`
- optionally aggregate to `w_class_forward[tau]`.

---

## 9) Collapse (price) projection

Per-node return models:
- Student-t `(mu_s, sigma_s, nu_s)` (Gaussian fallback)

Hurst scaling:
- `sigma_s(tau) = sigma_s_1 * tau**H_eff`
- suggested blend:
  - `H_eff = 0.5 * H_now + 0.5 * H_class_mean`

Mixture at horizon `tau`:
- weighted by `pi_forward[tau][s]`
- extract quantiles q05/q50/q95
- convert log-price back to price if needed.

---

## 10) Compounding method aggregator

Methods:
1. analytic_local
2. semi_markov_mixture
3. quantile_coverage
4. pde_density (optional/low weight in V1)

State gating:
- `g_m = sum_r w_class[r] * G[r][m]`
- multiply by reliability `R_m`
- apply order-conditioned multipliers
- renormalize.

V1 recommendation:
- mainline methods are (1) + (2)
- quantile/pde paths optional.

---

## 11) Rendering payload

Historical per bucket:
- class posterior + dominant class + entropy
- order posterior + dominant order
- node posterior + dominant node + age

Forward at latest bucket:
- horizons
- forward class probabilities
- fan rows `{tau, q05, q50, q95}`
- fade alpha per horizon:
  - `alpha_tau = base_alpha * (1 - entropy_now) * max_state_prob(tau)`.

---

## 12) Module layout

The implementation follows:

- `crypto_rfp_hso/core`
- `crypto_rfp_hso/data`
- `crypto_rfp_hso/features`
- `crypto_rfp_hso/hilbert`
- `crypto_rfp_hso/regime`
- `crypto_rfp_hso/fractal`
- `crypto_rfp_hso/transitions`
- `crypto_rfp_hso/projection`
- `crypto_rfp_hso/overlay`
- `crypto_rfp_hso/pipelines`

with tests under `tests/test_crypto_rfp_hso_pipeline.py`.

---

## 13) Strict function signatures

Canonical signatures are implemented for:
- `bucketize_prints_notional`
- `compute_feature_signature`
- `hilbert_project`
- `score_orders`
- `compute_node_posterior`
- `hurst_ghe`
- `fit_semi_markov_params`
- `propagate_semi_markov`
- `fit_per_state_t_params`
- `build_forward_fan`
- `build_overlay_payload`

See `src/crypto_rfp_hso/__init__.py` for exported public API names.

---

## 14) Default config

Canonical defaults are stored in:
- `src/crypto_rfp_hso/core/schemas.py` (`DEFAULT_CONFIG`)
- `configs/crypto_rfp_hso.yaml` (repo-level serialized defaults)

Defaults include:
- symbols
- bucket_notional thresholds
- EWMA/hurst windows
- temperatures
- horizons/quantiles
- event-time controls
- hazard/temperature controls.

---

## 15) Implementation order (fast path)

1. Ingest prints + L2 + perp + liquidations.
2. Notional bucketization.
3. FeatureSignature.
4. Hilbert projection + templates.
5. Entropy overlay.
6. Order + node posterior + age.
7. Semi-Markov fit.
8. Per-state return fit.
9. Forward `pi` + fan.
10. Overlay payload emission.

---

## 16) Gemini handoff prompt artifact

The ready-to-copy prompt is versioned in:
- `docs/crypto-rfp-hso-gemini-prompt.txt`

---

## Repository mapping notes

In this repository, the following are already implemented and wired:
- event-time quantization (`canonical` and `information`)
- event-time MAP path and sigma bands
- next-price hazard surface distribution
- validity masking (`VALID_MASK`)
- method gating (`G`, `ORDER_METHOD_MULT`)
- continuous scanner and continuous learning supervisor integration paths.

