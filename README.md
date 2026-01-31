# Gnosis Particle Bot (Forecasting)

## What it does
Given print (trade) data, it produces probabilistic forecasts:
- target wall-clock time y
- point estimate x_hat
- uncertainty sigma_hat
- quantiles q05/q50/q95
and a KPCOFGS regime taxonomy output K..S to narrow context.

## Data Sources

The bot supports two data sources:

### 1. Stub Data (Default)
Synthetic data generated for testing and development:
```bash
python scripts/fetch_coingecko_data.py --symbols BTCUSDT ETHUSDT --stub
```

### 2. CoinGecko API (Real Market Data)
Fetch real cryptocurrency market data from CoinGecko:

```bash
# Set API key in environment
export COINGECKO_API_KEY=your-api-key-here

# Or copy .env.example to .env and set your API key
cp .env.example .env

# Fetch data
python scripts/fetch_coingecko_data.py --symbols BTCUSDT ETHUSDT --days 30
```

Supported symbols: BTCUSDT, ETHUSDT, SOLUSDT

## Required Outputs (per run)
- data/manifests/data_manifest.json
- reports/latest/report.json
- reports/latest/report.md
- reports/latest/predictions.parquet
- reports/latest/trades.parquet
- reports/latest/feature_registry.json
- reports/latest/run_metadata.json

## PASS/FAIL
PASS requires:
- nested walk-forward with purge/embargo
- baseline computed
- 90% interval coverage in [0.87, 0.93] across OOS folds
- improved sharpness vs baseline at similar coverage
- deterministic outputs given seed/config
