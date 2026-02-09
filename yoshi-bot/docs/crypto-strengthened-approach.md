# Crypto Price-as-a-Particle: Strengthened Indicator Integration

This document captures a crypto-first version of the strengthened, physics-consistent approach where **prints are truth** and **bars are secondary**. It treats log-price as a particle in event-time and maps common indicators into force fields, diffusion geometry, and jump hazards.

---

## State and Dynamics (Event-Time)

Let the particle be log-price: \(x_\tau = \log p_\tau\), evolving in event-time \(\tau\) (each trade or micro-batch of prints).

Core SDE form:

\[
  dx_\tau = \mu_\tau\, d\tau - \nabla V(x_\tau)\, d\tau + \sigma_\tau\, dW_\tau + J_\tau
\]

Where:
- \(J_\tau\): jump term from liquidation cascades, stop-runs, or news shocks
- \(V\): total potential (liquidity + VWAP anchors + “technicals” reinterpreted as fields)
- \(\sigma_\tau\): diffusion (volatility geometry)

---

## 1) VWAP / AVWAP (Crypto): Mass Anchors + Fair-Value Wells (Tier 1)

### What changes in crypto
Structural anchor points matter more than “session VWAP”:
- Anchored VWAP to major liquidation events
- Anchored VWAP to funding regime flips
- Anchored VWAP to breakout impulse start
- Anchored VWAP to daily/weekly open (if you must reference wall-time)

### Physics interpretation
Each AVWAP is a center-of-mass well:

\[
  V_{\text{AVWAP}}(x) = \sum_i \alpha_i \cdot \text{vol}_i \cdot |x - \log(\text{AVWAP}_i)|
\]

- \(\alpha_i\) increases when the anchor corresponds to a high leverage flush or major OI reset
- Multiple AVWAPs competing → orbital trapping between wells

**Use-case:** “Where is fair value migrating?” and “Which well dominates?”

---

## 2) EMA/SMA (Crypto): Drift Surfaces + Restoring Forces (Tier 1)

Crypto trends hard and mean-reverts violently. MAs are useful when treated as equilibrium surfaces.

Define:
- Distance to EMA → restoring force strength
- EMA slope → background drift direction

\[
  V_{\text{MA}}(x) = k_\ell \cdot |x - \log(\text{EMA}_\ell)|
\]

\[
  \mu_\tau \propto \frac{d}{d\tau}\log(\text{EMA}_\ell)
\]

**Key crypto twist:** MA clustering on higher timeframe + perps leverage = “loaded spring.”

---

## 3) Bollinger Bands (Crypto): Volatility Curvature + Entropy Pressure (Tier 1)

Bollingers matter in crypto because volatility regimes are abrupt.

Treat as diffusion geometry, not directional breakouts:
- Band width = local \(\sigma_\tau\)
- Squeeze = stored potential energy
- Expansion = energy release already underway

\[
  \sigma_\tau = f(\text{BB\_width}_\tau)
\]

Treat bands as soft barriers:
- Near upper band in squeeze → higher probability of boundary test
- Escape probability depends on liquidity curvature + liquidation density (see below)

---

## 4) RSI (Crypto): Momentum Saturation / Impulse Efficiency (Tier 2)

RSI is useful in crypto only as a throttle on impulse effectiveness (diminishing returns).

Interpretation:
- RSI is a proxy for velocity persistence + exhaustion
- High RSI → marginal buy impulses produce less displacement
- Low RSI → marginal sell impulses produce less displacement

Scale impulse response:

\[
  J_\tau \leftarrow J_\tau \cdot g(\text{RSI}_\tau)
\]

This prevents the classic crypto failure mode: “RSI overbought” (wrong) vs “RSI saturated so impulse efficiency drops” (right).

---

## 5) Ichimoku (Crypto): Regime Topology / Allowed Zones (Tier 2)

Ichimoku is effective as a regime classifier:
- Inside cloud → diffusive/choppy manifold (mean-reversion dominates)
- Outside cloud → ballistic manifold (trend forces dominate)
- Cloud thickness → regime stability barrier

So you don’t trade “crosses.” Instead:

\[
  \text{Regime}_\tau \in \{\text{diffusive}, \text{ballistic}, \text{transition}\}
\]

Switch which forces exist based on regime:
- Diffusive: MA wells stronger, volatility reverts
- Ballistic: MA wells weaken, drift dominates

---

## Crypto-Specific Forces (Real “Forces”)

### A) Funding rate → Continuous external force
Funding is pressure encouraging mean reversion in crowded perps positioning:
- Positive funding: longs paying → upward trend can become unstable
- Negative funding: shorts paying → downside can become unstable

Model as:

\[
  F_{\text{fund}}(\tau) = \beta \cdot \text{funding}_\tau
\]

(Sign conventions depend on coordinate choice.)

### B) Open interest (OI) + OI change → Stored leverage energy
Rising OI during a move = leverage piling in → higher jump risk.

Model as hazard multiplier:

\[
  \lambda_\tau = \lambda_0 \cdot h(\Delta OI_\tau, OI_\tau)
\]

### C) Liquidation level density → Potential cliffs (jump landscape)
Approximate liquidation bands (even heuristically) as unstable boundaries.

\[
  V_{\text{liq}}(x) = \sum_j c_j \cdot \phi(x - \ell_j)
\]

Where \(\ell_j\) are liquidation clusters.

### D) CVD (cumulative volume delta) / aggressive flow → Impulse directionality
CVD informs the sign and persistence of impulses \(J_\tau\), aligning with the “prints are truth” rule.

---

## Unified Crypto Potential (Clean Integration)

**Total potential:**
\[
  V = V_{\text{liqbook}} + V_{\text{AVWAP}} + V_{\text{MA}} + V_{\text{liq}}
\]

**Diffusion:**
\[
  \sigma_\tau = f(\text{BB width}, \text{realized vol}, \text{spread}, \text{depth})
\]

**Drift:**
\[
  \mu_\tau = \mu(\text{MA slope}, \text{regime}, \text{funding})
\]

**Jump hazard:**
\[
  \lambda_\tau = \lambda(\Delta OI, \text{liq density}, \text{squeeze state})
\]

This is how “RSI/EMA/VWAP/Bollingers/Ichimoku” become physics instead of superstition.

---

## Implementation Order (No Fluff)

1. AVWAP anchor set (impulse start + major flush + weekly open)
2. Bollinger-driven \(\sigma_\tau\) + squeeze detector
3. MA-well + slope drift
4. Funding + OI hazard multiplier
5. RSI throttle last, Ichimoku only as regime gate
