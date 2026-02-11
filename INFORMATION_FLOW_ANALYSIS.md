# Yoshi Trading System — Information Flow Analysis & Optimization Report

## 1. Complete Information Flow

The system is a **monorepo** with two primary subsystems:

- **Yoshi-Bot** (`/yoshi-bot`) — "The Brain": quantitative logic, physics models, ML, data ingestion
- **ClawdBot** (`/clawdbot`) — "The Interface": Telegram UI, orchestration, bridge to execution

They share a namespace (`gnosis`) but are **separate Python packages** with significant code duplication.

### 1.1 End-to-End Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                         EXTERNAL WORLD                               │
│  Binance/CoinGecko/CMC/YFinance    Kalshi API    Telegram User       │
└────────────┬───────────────────────────┬──────────────┬──────────────┘
             │                           │              │
             ▼                           │              ▼
┌────────────────────────┐               │   ┌──────────────────────┐
│ STAGE 1: DATA INGESTION│               │   │  TELEGRAM BOT        │
│ (yoshi-bot)            │               │   │  (clawdbot)          │
│                        │               │   │                      │
│ UnifiedDataFetcher     │               │   │ Long-poll getUpdates │
│  ├─ BinancePublicProv. │               │   │ /scan, /status,      │
│  ├─ CoinGeckoProvider  │               │   │ /ralph, /params      │
│  ├─ YFinanceProvider   │               │   └──────────┬───────────┘
│  └─ CoinMarketCapProv. │               │              │
│                        │               │              ▼
│ CCXTLoader (live OHLCV)│               │   ┌──────────────────────┐
│ CryptoSourceScanner    │               │   │  ORCHESTRATOR        │
│ loader.py (parquet/    │               │   │  (clawdbot)          │
│   stub fallback)       │               │   │                      │
└────────────┬───────────┘               │   │  UnifiedOrchestrator │
             │ pd.DataFrame              │   │  run_cycle():        │
             │ (OHLCV / prints)          │   │   1. Ralph params    │
             ▼                           │   │   2. Forecast        │
┌────────────────────────┐               │   │   3. Kalshi scan     │
│ STAGE 2: FEATURE       │               │   │   4. Ralph learn     │
│ ENGINEERING             │               │   └──────────┬───────────┘
│                        │               │              │
│ MTF Bar Manager        │               │              │
│  └─ Multi-timeframe    │               │              │
│     bar aggregation    │               │              │
│                        │               │              │
│ PriceParticle.compute_features()       │              │
│  ├─ Kinematics (v, a, jerk)           │              │
│  ├─ Mass (1/σ)                        │              │
│  ├─ Forces (OFI, F=ma)               │              │
│  ├─ Energy (KE, PE, Total)           │              │
│  ├─ Support/Resistance fields         │              │
│  ├─ VWAP / Volume Profile            │              │
│  ├─ Anchored VWAP wells              │              │
│  ├─ Bollinger geometry                │              │
│  ├─ RSI throttle                      │              │
│  ├─ Ichimoku regime topology          │              │
│  ├─ Crypto forces (funding, OI, liq)  │              │
│  └─ Composite particle_physics_score  │              │
│                                        │              │
│ ParticleState.compute_state()         │              │
│  ├─ Flow momentum                     │              │
│  ├─ Regime stability                  │              │
│  └─ Barrier proximity                 │              │
│                                        │              │
│ crypto_rfp_hso pipeline               │              │
│  ├─ Features (price/perp/orderbook)   │              │
│  ├─ Fractal analysis (Hurst/GHE)      │              │
│  ├─ Hilbert space projection          │              │
│  ├─ Regime detection (semi-Markov)    │              │
│  └─ Projection (hazard surface)       │              │
└────────────┬───────────┘               │              │
             │ feature-rich DataFrame    │              │
             ▼                           │              │
┌────────────────────────┐               │              │
│ STAGE 3: REGIME        │               │              │
│ CLASSIFICATION          │               │              │
│                        │◄──────────────┼──────────────┘
│ KPCOFGSClassifier      │   (Bridge     │   (Orchestrator calls
│  7-level hierarchy:    │    adapts     │    both forecast and
│  K: Kinetics           │    snap →     │    KPCOFGS enrichment)
│  P: Pressure (vol)     │    DataFrame  │
│  C: Current (flow)     │    for KPCOFGS│
│  O: Opening (breakout) │    use)       │
│  F: Force (momentum)   │               │
│  G: Game (meta-level)  │               │
│  S: Scenario (final)   │               │
│                        │               │
│ QuantumPriceEngine     │               │
│  .detect_market_regime │               │
│  (trending/ranging/    │               │
│   volatile)            │               │
└────────────┬───────────┘               │
             │ regime + enriched data    │
             ▼                           │
┌────────────────────────┐               │
│ STAGE 4: FORECASTING   │               │
│                        │               │
│ A) 14-Paradigm Ensemble│               │
│    (ClawdBot Forecaster│               │
│     engine.py)         │               │
│    ├─ TechnicalModule  │               │
│    ├─ ClassicalStats   │               │
│    ├─ MacroFactor      │               │
│    ├─ Derivatives      │               │
│    ├─ Microstructure   │               │
│    ├─ OnChain          │               │
│    ├─ Sentiment        │               │
│    ├─ MetaLearner      │               │
│    ├─ Sequence         │               │
│    ├─ CrowdPrior       │               │
│    ├─ RegimeDetector   │               │
│    ├─ MonteCarlo       │               │
│    ├─ ParticleCandle   │               │
│    └─ ManifoldPattern  │               │
│                        │               │
│ B) QuantumPriceEngine  │               │
│    Monte Carlo (10k    │               │
│    paths) with:        │               │
│    ├─ Funding force    │               │
│    ├─ OB imbalance     │               │
│    ├─ Gravity (S/R)    │               │
│    ├─ Spring (VWAP MR) │               │
│    ├─ Liquidation force│               │
│    ├─ Multi-scale mom. │               │
│    ├─ Gamma fields     │               │
│    ├─ Macro coupling   │               │
│    └─ Time-of-day      │               │
│                        │               │
│ C) HybridPredictor     │               │
│    (LightGBM + temp.)  │               │
│                        │               │
│ D) RegimeGate          │               │
│    Blocks bad regimes  │               │
│                        │               │
│ E) AutoFixPipeline     │               │
│    (Isotonic/Platt     │               │
│     calibration)       │               │
└────────────┬───────────┘               │
             │ PredictionResult /        │
             │ ForecastResult            │
             ▼                           │
┌────────────────────────┐               │
│ STAGE 5: SIGNAL GEN    │               │
│ & EDGE DETECTION       │               │
│                        │               │
│ KalshiScanner          │               │
│  ├─ Fetch open markets │◄──────────────┘ (Kalshi API)
│  ├─ Compute model_prob │
│  │   from forecast     │
│  ├─ Compute edge =     │
│  │   model - market    │
│  ├─ Kelly sizing       │
│  └─ Rank by EV         │
│                        │
│ KalshiAnalyzer (LLM)   │
│  ├─ Filter noise       │
│  ├─ BUY/WATCH/PASS     │
│  └─ Value score        │
│                        │
│ ArbitrageDetector      │
│  ├─ Spread arb         │
│  └─ Model-edge arb     │
└────────────┬───────────┘
             │ value_plays[]
             ▼
┌────────────────────────┐
│ STAGE 6: LLM REASONING │
│ (optional)             │
│                        │
│ ReasoningEngine        │
│  ├─ Full analysis      │
│  ├─ Regime deep dive   │
│  ├─ Trade plan         │
│  ├─ Risk assessment    │
│  ├─ Extrapolation      │
│  └─ Self-critique      │
│                        │
│ Uses: Ollama or API LLM│
└────────────┬───────────┘
             │
             ▼
┌────────────────────────┐    ┌──────────────────────┐
│ STAGE 7: BRIDGE        │    │ STAGE 7b: TELEGRAM   │
│ (clawdbot)             │    │ ALERTS               │
│                        │    │                      │
│ yoshi-bridge.py        │    │ formatter.py         │
│  ├─ Poll scanner logs  │    │ → Phone-friendly     │
│  ├─ Regex parse signals│    │   BUY/WATCH alerts   │
│  ├─ Buffer in asyncio  │    │ → Cycle reports      │
│  │   Queue             │    │ → Ralph summaries    │
│  └─ HTTP POST to Core  │    └──────────┬───────────┘
└────────────┬───────────┘               │
             │ TradeProposal             │ to User
             ▼                           ▼
┌────────────────────────┐
│ STAGE 8: EXECUTION     │
│ Trading Core (FastAPI)  │
│  :8000                 │
│                        │
│ POST /propose          │
│  ├─ Validate edge      │
│  ├─ CircuitBreaker     │
│  │   (3-fail trip)     │
│  ├─ KalshiClient       │
│  │   place_order()     │
│  └─ Track positions    │
│                        │
│ POST /kill-switch      │
│ POST /pause / /resume  │
│ GET  /status           │
└────────────┬───────────┘
             │ order_result
             ▼
         Kalshi API
```

### 1.2 Feedback Loop (Ralph Meta-Learner)

```
┌───────────────────────────────────────────────┐
│              RALPH LEARNING LOOP               │
│                                                │
│  ┌──────────┐   ┌──────────────┐              │
│  │HyperParam│──▶│  Pipeline    │              │
│  │Manager   │   │  (forecast + │              │
│  │          │   │   scan +     │              │
│  │ explore/ │   │   execute)   │              │
│  │ exploit  │   └──────┬───────┘              │
│  │ (90/10)  │          │                      │
│  └─────▲────┘          │ predictions          │
│        │               ▼                      │
│        │   ┌──────────────────┐               │
│        │   │PredictionTracker │               │
│        │   │  record()        │               │
│        │   │  auto_resolve()  │               │
│        │   │  compute_metrics │               │
│        │   │  ├─ Brier score  │               │
│        │   │  ├─ Hit rate     │               │
│        │   │  ├─ PnL          │               │
│        │   │  └─ Calibration  │               │
│        │   └──────┬───────────┘               │
│        │          │ metrics                   │
│        └──────────┘ (feed back)               │
│                                                │
│  Hot-Reload: Ralph → config/params.json       │
│  → QuantumPriceEngine reads every 10 cycles   │
└───────────────────────────────────────────────┘
```

### 1.3 Service Topology (VPS)

```
systemd services:
  trading-core.service    → FastAPI on :8000 (always-on)
  kalshi-scanner.service  → Scanner loop → logs value plays
  yoshi-bridge.service    → Polls scanner logs → POST /propose
  clawdbot.service        → Telegram bot + orchestrator loop
  continuous-learning     → ML retraining supervisor
  ralph-learner.timer     → Periodic Ralph optimization (oneshot)
```

---

## 2. Identified Issues & Optimization Opportunities

### 2.1 CRITICAL: Massive Code Duplication Between Packages

**Problem:** `clawdbot/gnosis/` and `yoshi-bot/src/gnosis/` are **parallel copies** of core modules that have **diverged**:

| Module | clawdbot | yoshi-bot | Divergence |
|--------|----------|-----------|------------|
| `particle/physics.py` | `PriceParticle` | `PriceParticle` | Identical (wasteful) |
| `particle/quantum.py` | `detect_regime()` | `detect_market_regime()` | **Method name diverged** |
| `regimes/kpcofgs.py` | `KPCOFGSClassifier` | `KPCOFGSClassifier` | Copies can drift |
| `backtest/` | Full copy | Full copy | Risk of silent bugs |
| `calibration/` | Copy | Copy | Same |
| `particle/` (all) | Full copy | Full copy | Full duplication |
| `predictors/` | Copy | Copy + `.bak` files | yoshi-bot has backup clutter |

**Impact:** Any bugfix or improvement must be applied in **two places**, or the systems silently diverge. The `detect_regime` vs `detect_market_regime` divergence is already a live inconsistency -- clawdbot and yoshi-bot may classify the same market data into different regimes.

**Fix:** Extract `gnosis` into a single shared Python package (e.g., in a top-level `/libs/gnosis/` directory) imported by both `clawdbot` and `yoshi-bot`. Use the monorepo's `PYTHONPATH` to share it.

---

### 2.2 HIGH: Log-File Polling Bridge Is Fragile

**Problem:** The `yoshi-bridge.py` service works by **regex-parsing scanner log output** to extract trade signals:

```python
EDGE_PATTERN = re.compile(r"EDGE:\s*([+-]?\d+\.?\d*)%", re.IGNORECASE)
SYMBOL_PATTERN = re.compile(r"\*?(BTC|ETH|SOL)USDT\*?", re.IGNORECASE)
# ... 5+ more regex patterns
```

This is extremely fragile:
- Any change to scanner output format silently breaks signal forwarding
- No schema validation on parsed data
- Regex patterns can mis-match or miss data
- There's a `TIKCER_PATTERN` typo (line 41 of `yoshi-bridge.py`)

**Fix:** Replace log-file parsing with a **structured message queue** or direct HTTP API call. The scanner should POST JSON directly to the Trading Core's `/propose` endpoint, or write to a shared Redis/ZeroMQ queue with typed schemas. The bridge service becomes unnecessary.

---

### 2.3 HIGH: Global `np.random.seed()` Usage

**Problem:** Found in 15+ files across the codebase:

```python
# yoshi-bot/src/gnosis/particle/quantum.py
if random_seed is not None:
    np.random.seed(random_seed)  # GLOBAL state!
```

Using `np.random.seed()` sets **global** random state, meaning:
- Parallel/concurrent Monte Carlo simulations interfere with each other
- Non-deterministic behavior in production (race conditions)
- Tests can bleed random state across test cases

**Fix:** Use `np.random.default_rng(seed)` throughout (which is already done in `loader.py` -- inconsistent). Replace all `np.random.seed()` + `np.random.normal()` calls with instance-based RNG:

```python
self.rng = np.random.default_rng(random_seed)
brownian_shocks = self.rng.normal(0, 1, (n_sims, n_steps))
```

---

### 2.4 HIGH: In-Memory-Only Trading State

**Problem:** The Trading Core uses a plain Python object for all state:

```python
class TradingState:
    def __init__(self):
        self.proposals: Dict[str, Dict] = {}
        self.positions: List[Dict] = []
        self.orders: List[Dict] = []
```

If the FastAPI process restarts (which `systemd` does with `Restart=always`), **all open positions, orders, and proposals are lost**. There is no persistence layer.

**Fix:** Add SQLite or Redis persistence for `TradingState`. At minimum, write positions/orders to a JSON file on each mutation and load on startup.

---

### 2.5 MEDIUM: `_generate_prints_from_ohlcv` Uses Row-by-Row Iteration

**Problem:** The `UnifiedDataFetcher._generate_prints_from_ohlcv` method uses `iterrows()` with nested Python `for` loops:

```python
for _, bar in ohlcv_df.iterrows():  # Slow row-by-row
    for i in range(1, n_trades - 1):  # Nested loop per bar
        ...
    for i in range(n_trades):  # Another nested loop
        records.append({...})
```

For 30 days of 1-minute data (43,200 bars x ~50 trades/bar = ~2.16M records), this is extremely slow.

**Fix:** Vectorize with NumPy. Pre-allocate arrays for all bars at once, generate prices/quantities in batch, and construct the DataFrame in one pass (similar to how `generate_stub_prints` already does it).

---

### 2.6 MEDIUM: `print()` Statements Instead of Logging

**Problem:** The `UnifiedDataFetcher` (and many other modules) use 14+ raw `print()` calls for operational messages:

```python
print(f"Fetching {symbol} OHLCV from {provider_name}...")
print(f"  Got {len(df)} candles")
print(f"  Error: {e}")
```

This means:
- No log level control (can't suppress verbose output in production)
- No structured logging for monitoring/alerting
- No way to filter by module

**Fix:** Replace with `logging.getLogger(__name__)` throughout. Some modules (like `trading_core.py`) already use proper logging -- make it consistent.

---

### 2.7 MEDIUM: Redundant Regime Detection Systems

**Problem:** There are **four separate regime detection systems** that can produce conflicting classifications:

1. **KPCOFGS** (`regimes/kpcofgs.py`): 7-level hierarchical, rule-based
2. **QuantumPriceEngine.detect_market_regime()**: Trend/ranging/volatile, threshold-based
3. **RegimeDetector** (in `forecaster/modules.py`): Used by the 14-paradigm ensemble
4. **crypto_rfp_hso regime** (`regime/genus_map.py`): Semi-Markov + node posterior

The Bridge (`bridge.py`) maps KPCOFGS S-labels to 6 ClawdBot regime states, but the QuantumPriceEngine uses its own 3-state regime independently. These are never reconciled -- the same market data can be "trending" in one system and "ranging" in another.

**Fix:** Create a single `RegimeConsensus` class that takes all regime signals and produces one authoritative classification. Weight by each system's historical accuracy via Ralph's feedback loop.

---

### 2.8 MEDIUM: Hot-Reload Path Is Hardcoded

**Problem:** The `QuantumPriceEngine` hardcodes a VPS-specific path for Ralph's parameter updates:

```python
self.param_store_path = "/root/Yoshi-Bot/config/params.json"
```

This breaks in any environment other than the specific VPS deployment. The fallback (`config/params.json`) only works if CWD is the yoshi-bot directory.

**Fix:** Use environment variable (`RALPH_PARAMS_PATH`) with a sensible default relative to the project root, or use the `data_dir` configuration that Ralph already has.

---

### 2.9 MEDIUM: No Caching of Expensive Computations

**Problem:** The `PriceParticle.compute_features()` method computes **60+ features** using 20+ rolling window operations with `groupby().transform()` calls. These are recomputed from scratch every cycle, even when the data overlaps 99%+ with the previous cycle (only 1 new bar added).

**Fix:** Implement incremental feature computation. Cache the feature DataFrame and only compute features for new bars, appending to the cache. Rolling window calculations can be maintained incrementally for most indicators.

---

### 2.10 LOW: `UnifiedDataFetcher` Re-instantiated Per Call

**Problem:** The convenience functions `fetch_crypto_data()` and `fetch_crypto_prints()` create a new `UnifiedDataFetcher` instance on every call:

```python
def fetch_crypto_data(symbols, ...):
    fetcher = UnifiedDataFetcher(...)  # New instance every time
    return fetcher.fetch_ohlcv(...)
```

Each instantiation initializes all providers (HTTP connections, API health checks).

**Fix:** Use a module-level singleton or caching pattern for the fetcher instance.

---

### 2.11 LOW: Backup Files Committed to Repository

**Problem:** Multiple `.bak.*` files are tracked in git:

```
yoshi-bot/src/gnosis/predictors/quantile.py.bak.1769719329
yoshi-bot/src/gnosis/predictors/quantile.py.bak.1769719537
yoshi-bot/src/gnosis/predictors/quantile.py.bak.1769720291
yoshi-bot/scripts/run_experiment.py.bak.1769727234
```

These add noise and clutter.

**Fix:** Add `*.bak.*` to `.gitignore` and remove from tracking.

---

### 2.12 LOW: Duplicate `detect_regime` Signature Divergence

**Problem:** As noted, `clawdbot/gnosis/particle/quantum.py` has `detect_regime()` while `yoshi-bot/src/gnosis/particle/quantum.py` has `detect_market_regime()`. The yoshi-bot version is called via `self.detect_regime()` in both `predict()` and `predict_enhanced()` -- but that method doesn't exist under that name in yoshi-bot. This means `yoshi-bot`'s QuantumPriceEngine **may raise `AttributeError` at runtime** unless there's a missing method alias.

**Fix:** Audit and unify the method names across both packages. This is a symptom of the broader duplication problem (2.1).

---

## 3. Optimization Priority Matrix

| # | Issue | Severity | Effort | Impact |
|---|-------|----------|--------|--------|
| 2.1 | Code duplication / diverged packages | CRITICAL | High | Eliminates entire class of bugs |
| 2.2 | Log-file regex bridge | HIGH | Medium | Reliability + latency |
| 2.3 | Global `np.random.seed` | HIGH | Low | Correctness of MC simulations |
| 2.4 | In-memory-only trading state | HIGH | Medium | Data safety on restart |
| 2.5 | Row-by-row OHLCV→prints conversion | MEDIUM | Medium | 10-100x speedup |
| 2.6 | `print()` vs `logging` | MEDIUM | Low | Operability |
| 2.7 | Redundant regime detection | MEDIUM | High | Signal consistency |
| 2.8 | Hardcoded hot-reload path | MEDIUM | Low | Portability |
| 2.9 | No incremental feature cache | MEDIUM | Medium | Cycle latency |
| 2.10 | Fetcher re-instantiation | LOW | Low | Minor latency |
| 2.11 | Backup files in git | LOW | Trivial | Repo hygiene |
| 2.12 | detect_regime name divergence | LOW | Low | Prevents runtime error |

---

## 4. Recommended Optimization Roadmap

### Phase 1: Quick Wins (1-2 days)
- Fix `np.random.seed` → `default_rng` (2.3)
- Fix hardcoded path (2.8)
- Add `.bak.*` to `.gitignore` (2.11)
- Replace `print()` with `logging` in key modules (2.6)
- Fix `detect_regime` name divergence (2.12)

### Phase 2: Reliability (3-5 days)
- Add SQLite/JSON persistence to TradingState (2.4)
- Replace log-parsing bridge with direct API calls (2.2)
- Vectorize `_generate_prints_from_ohlcv` (2.5)

### Phase 3: Architecture (1-2 weeks)
- Unify `gnosis` into shared package (2.1)
- Build `RegimeConsensus` (2.7)
- Implement incremental feature caching (2.9)
