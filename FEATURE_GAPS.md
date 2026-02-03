# Feature Gap Analysis: Theoretical Framework vs Implementation

This document identifies features described in the physics-inspired Bitcoin prediction framework that are **not yet implemented** in Yoshi-Bot.

---

## Summary

| Feature | Status | Priority |
|---------|--------|----------|
| Cross-Exchange Funding Rate Aggregation | ❌ Missing | High |
| Liquidation Heatmaps/Density Maps | ❌ Missing | High |
| Gamma Fields from Options | ❌ Missing | Medium |
| Cross-Asset Coupling (SPX, DXY) | ❌ Missing | Medium |
| Time-of-Day Volatility Multipliers | ❌ Missing | High |
| Multi-Level Order Book Depth | ❌ Missing | Medium |
| Empirical Calibration Framework | ⚠️ Partial | High |
| Data Collection Pipeline | ❌ Missing | High |

---

## Detailed Gap Analysis

### 1. Cross-Exchange Funding Rate Aggregation ❌

**Described in Framework:**
```python
def get_funding_rates(self):
    """Get funding rates from multiple exchanges"""
    funding_data = {}
    exchanges = ['binance', 'bybit', 'okx']

    for exch_name in exchanges:
        exch = getattr(ccxt, exch_name)({'enableRateLimit': True})
        funding = exch.fetch_funding_rate('BTC/USDT:USDT')
        funding_data[exch_name] = funding['fundingRate']

    return funding_data
```

**Current Implementation:**
- `quantum.py:214-227` accepts a single `funding_rate` float parameter
- No multi-exchange aggregation
- No weighted averaging across venues

**What's Needed:**
```python
class FundingRateAggregator:
    def aggregate_funding(self, rates: Dict[str, float]) -> float:
        """Aggregate funding rates across exchanges with volume weighting."""
        # Weight by exchange volume/liquidity
        weights = {'binance': 0.5, 'bybit': 0.3, 'okx': 0.2}
        return sum(rates[ex] * w for ex, w in weights.items() if ex in rates)
```

---

### 2. Liquidation Heatmaps/Density Maps ❌

**Described in Framework:**
```python
def calculate_liquidation_field(price, features):
    """Calculate force from nearby liquidation clusters"""
    liq_data = features['liquidation_heatmap']
    # Find liquidations within 2% of current price
    nearby_liqs = liq_data[
        (liq_data['price'] > price * 0.98) &
        (liq_data['price'] < price * 1.02)
    ]
```

**Current Implementation:**
- `quantum.py:305-333` has `calculate_liquidation_force()` but requires pre-computed `liquidation_levels` to be passed in
- No built-in liquidation heatmap data source
- No density function: `ρ_liq(p) = density of liquidations at price p`

**What's Needed:**
```python
class LiquidationHeatmapProvider:
    def fetch_liquidation_heatmap(self) -> pd.DataFrame:
        """Fetch liquidation levels from Coinglass or similar API."""
        # Returns DataFrame with columns: price, long_volume, short_volume
        pass

    def compute_liquidation_density(self, price: float) -> float:
        """Compute liquidation density function at price level."""
        pass
```

---

### 3. Gamma Fields from Options ❌

**Described in Framework:**
```
Γ_net = Σ(Γ_strike × Open_Interest_strike)

Near options expiry, "max pain" levels (where most options expire worthless)
can create weak gravitational effects.
```

**Current Implementation:**
- **Completely absent** from codebase
- No options data integration
- No gamma exposure calculations
- No max pain computation

**What's Needed:**
```python
class GammaFieldCalculator:
    def fetch_options_data(self, symbol: str) -> pd.DataFrame:
        """Fetch options chain with open interest and greeks."""
        pass

    def calculate_net_gamma(self, options_df: pd.DataFrame) -> float:
        """Calculate net gamma exposure at current price."""
        return (options_df['gamma'] * options_df['open_interest']).sum()

    def calculate_max_pain(self, options_df: pd.DataFrame) -> float:
        """Calculate max pain strike level."""
        pass

    def gamma_steering_force(self, current_price: float, net_gamma: float) -> float:
        """Calculate steering force from dealer gamma hedging."""
        # Negative gamma = dealers buy dips, sell rips (stabilizing)
        # Positive gamma = dealers chase price (destabilizing)
        pass
```

---

### 4. Cross-Asset Coupling Fields ❌

**Described in Framework:**
```
dP_BTC/dt = β_SPX × dP_SPX/dt + β_DXY × dP_DXY/dt + crypto-specific forces
```

**Current Implementation:**
- Docstring in `quantum.py:7` mentions "Cross-asset coupling fields (external forces)"
- **No actual implementation exists**
- No SPX/DXY data fetching
- No correlation/beta calculations

**What's Needed:**
```python
class CrossAssetCoupling:
    def fetch_macro_assets(self) -> Dict[str, pd.DataFrame]:
        """Fetch SPX, DXY, Gold, etc. price data."""
        pass

    def calculate_rolling_betas(
        self,
        btc_returns: pd.Series,
        macro_returns: Dict[str, pd.Series],
        window: int = 168  # 1 week hourly
    ) -> Dict[str, float]:
        """Calculate rolling beta coefficients."""
        betas = {}
        for asset, returns in macro_returns.items():
            cov = btc_returns.rolling(window).cov(returns)
            var = returns.rolling(window).var()
            betas[asset] = (cov / var).iloc[-1]
        return betas

    def macro_drift_contribution(
        self,
        betas: Dict[str, float],
        macro_returns: Dict[str, float]  # Current period returns
    ) -> float:
        """Calculate drift from macro asset movements."""
        return sum(betas[a] * macro_returns[a] for a in betas)
```

---

### 5. Time-of-Day Volatility Multipliers ❌

**Described in Framework:**
```python
# Empirically derived hourly volatility multipliers
vol_multipliers = {
    0: 0.85, 1: 0.80, 2: 0.80, 3: 1.10,  # London open
    4: 1.20, 5: 1.15, 6: 1.05, 7: 0.95,
    8: 0.95, 9: 1.25, 10: 1.35, 11: 1.30,  # US markets
    12: 1.25, 13: 1.20, 14: 1.15, 15: 1.20,
    16: 1.10, 17: 1.05, 18: 0.95, 19: 0.95,
    20: 1.00, 21: 1.05, 22: 1.00, 23: 0.90  # Asian session
}
```

**Current Implementation:**
- **Not implemented**
- Volatility is regime-dependent but not time-dependent
- No hourly seasonality patterns

**What's Needed:**
```python
class TimeOfDayEffects:
    VOL_MULTIPLIERS = {
        0: 0.85, 1: 0.80, 2: 0.80, 3: 1.10,
        # ... full 24-hour pattern
    }

    def get_volatility_multiplier(self, hour: int) -> float:
        """Get time-of-day volatility scaling factor."""
        return self.VOL_MULTIPLIERS.get(hour, 1.0)

    def get_directional_bias(self, hour: int, day_of_week: int) -> float:
        """Get time-based directional bias (if any)."""
        # Could incorporate day-of-week effects too
        pass
```

---

### 6. Multi-Level Order Book Depth Analysis ❌

**Described in Framework:**
```python
# Compute depth at multiple price levels
for pct in [0.005, 0.01, 0.02]:  # 0.5%, 1%, 2%
    bid_threshold = mid_price * (1 - pct)
    ask_threshold = mid_price * (1 + pct)

    bid_depth = np.sum(bids[bids[:, 0] >= bid_threshold][:, 1])
    ask_depth = np.sum(asks[asks[:, 0] <= ask_threshold][:, 1])

    level_pressure = (bid_depth - ask_depth) / (bid_depth + ask_depth)
    depth_pressures.append(level_pressure)

# Weight closer levels more heavily
weights = np.array([0.5, 0.3, 0.2])
avg_pressure = np.average(depth_pressures, weights=weights)
```

**Current Implementation:**
- `quantum.py:229-246` only takes total `bid_volume` and `ask_volume`
- No depth-at-levels analysis
- No weighted pressure calculation

**What's Needed:**
```python
class OrderBookAnalyzer:
    def analyze_depth_levels(
        self,
        orderbook: Dict,
        current_price: float,
        levels: List[float] = [0.005, 0.01, 0.02]
    ) -> Dict[str, float]:
        """Analyze order book depth at multiple price levels."""
        bids = np.array(orderbook['bids'])
        asks = np.array(orderbook['asks'])

        pressures = []
        for pct in levels:
            bid_thresh = current_price * (1 - pct)
            ask_thresh = current_price * (1 + pct)

            bid_depth = bids[bids[:, 0] >= bid_thresh, 1].sum()
            ask_depth = asks[asks[:, 0] <= ask_thresh, 1].sum()

            if bid_depth + ask_depth > 0:
                pressures.append((bid_depth - ask_depth) / (bid_depth + ask_depth))
            else:
                pressures.append(0)

        # Weighted average (closer levels weighted more)
        weights = [0.5, 0.3, 0.2][:len(pressures)]
        return {
            'weighted_pressure': np.average(pressures, weights=weights),
            'depth_0.5pct': pressures[0] if len(pressures) > 0 else 0,
            'depth_1pct': pressures[1] if len(pressures) > 1 else 0,
            'depth_2pct': pressures[2] if len(pressures) > 2 else 0,
        }
```

---

### 7. Empirical Calibration Framework ⚠️ Partial

**Described in Framework:**
```python
def calibrate_physics_parameters(historical_data, lookback_days=90):
    """
    Calibrate physics engine parameters using historical performance

    Optimize for:
    1. Calibration accuracy (predicted probabilities match actual outcomes)
    2. Sharpness (narrow confidence intervals when certain)
    3. Regime-specific performance
    """
    parameter_grid = {
        'funding_strength': [8.0, 12.0, 15.0, 18.0],
        'imbalance_strength': [0.04, 0.06, 0.08, 0.10],
        'momentum_decay': [0.2, 0.4, 0.6, 0.8],
        'jump_intensity': [0.01, 0.02, 0.05, 0.08]
    }
    # Grid search with time-series cross-validation
```

**Current Implementation:**
- `improvement_loop.py` handles hyperparameter optimization for predictors
- `REGIME_PARAMS` in `quantum.py:94-122` are **hardcoded**
- No calibration framework for physics parameters specifically

**What's Needed:**
```python
class PhysicsParameterCalibrator:
    def calibrate_regime_params(
        self,
        historical_df: pd.DataFrame,
        param_grid: Dict[str, List[float]],
        n_folds: int = 5
    ) -> Dict[MarketRegime, RegimeParameters]:
        """Calibrate REGIME_PARAMS using walk-forward validation."""
        pass

    def evaluate_calibration(
        self,
        predictions: List[PredictionResult],
        actuals: pd.Series
    ) -> Dict[str, float]:
        """Evaluate prediction calibration accuracy."""
        # Check if X% confidence intervals contain X% of outcomes
        pass
```

---

### 8. Data Collection Pipeline ❌

**Described in Framework:**
```python
class BitcoinDataCollector:
    def __init__(self):
        self.exchange = ccxt.binance({'enableRateLimit': True})

    def collect_market_state(self):
        """Collect all data needed for prediction"""
        return {
            'ohlcv_1m': self.get_ohlcv_data('1m', 24),
            'ohlcv_5m': self.get_ohlcv_data('5m', 72),
            'ohlcv_1h': self.get_ohlcv_data('1h', 168),
            'orderbook': self.get_orderbook_snapshot(),
            'funding_rates': self.get_funding_rates()
        }
```

**Current Implementation:**
- `ingest/ccxt_loader.py` exists but is separate from quantum engine
- `quantum.py:predict()` expects pre-processed DataFrames passed in
- No integrated real-time data collection pipeline

**What's Needed:**
- Unified data collector that feeds directly into `QuantumPriceEngine`
- Multi-timeframe OHLCV collection
- Real-time order book snapshots
- Cross-exchange funding rate aggregation

---

## Implementation Priority

### Phase 1 (High Priority)
1. **Time-of-Day Volatility** - Easy win, improves intraday predictions
2. **Cross-Exchange Funding** - Critical for funding force accuracy
3. **Data Collection Pipeline** - Required for real-time predictions

### Phase 2 (Medium Priority)
4. **Multi-Level Order Book Depth** - Improves imbalance accuracy
5. **Liquidation Heatmaps** - Requires external data source (Coinglass API)
6. **Empirical Calibration** - Tune hardcoded REGIME_PARAMS

### Phase 3 (Lower Priority)
7. **Cross-Asset Coupling** - SPX/DXY correlation (requires macro data)
8. **Gamma Fields** - Options data is harder to obtain for crypto

---

## Estimated Effort

| Feature | Lines of Code | External Dependencies |
|---------|---------------|----------------------|
| Time-of-Day Vol | ~50 | None |
| Cross-Exchange Funding | ~100 | ccxt (already present) |
| Data Pipeline | ~200 | ccxt |
| Multi-Level OB Depth | ~80 | None |
| Liquidation Heatmaps | ~150 | Coinglass API |
| Calibration Framework | ~300 | None |
| Cross-Asset Coupling | ~200 | yfinance or similar |
| Gamma Fields | ~250 | Deribit API |

**Total: ~1,300 lines of new code**

---

*Generated: 2026-02-03*
