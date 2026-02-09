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

## Crypto Price-as-a-Particle (Strengthened Approach)

For the crypto-first, physics-consistent interpretation of VWAP, EMA, Bollinger Bands,
RSI, Ichimoku, funding, OI, liquidations, and CVD, see:

- [docs/crypto-strengthened-approach.md](docs/crypto-strengthened-approach.md)

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

## Prediction Test Battery
The prediction test battery lives in `gnosis.prediction_test_battery` and can be run
against platform artifacts or synthetic data. It evaluates robustness across suites
0, A, B, C, D, E, F, and G and produces PASS/WARN/FAIL scorecards plus JSON/Markdown
reports.

### Quickstart (Synthetic)
```bash
python -m gnosis.prediction_test_battery.cli run --synthetic --suite full
```

### Quickstart (Artifact + candles)
```bash
python -m gnosis.prediction_test_battery.cli run \\
  --artifact-path path/to/predictions.csv \\
  --candles-path path/to/candles.csv \\
  --features-path path/to/features.csv \\
  --suite full
```

### Adding a new test
1. Create a new `BaseTest` in `gnosis/prediction_test_battery/suites.py`.
2. Add it to the appropriate suite in `SUITE_TESTS`.
3. Ensure it returns a `TestResult` and include recommended actions for WARN/FAIL.

## Moltbot AI + Service Setup
To connect an AI planner and your external services (Slack, webhooks, etc.), use
the Moltbot orchestration layer.

### 1) Configure Moltbot
Edit `configs/moltbot.yaml` with:
- Your AI provider settings (OpenAI-compatible endpoint + model)
- Your API key environment variable name
- Your risk constraints
- Any webhook services you want notified

Example config is already provided in `configs/moltbot.yaml`.

### 2) Set your API key
```bash
export OPENAI_API_KEY=your-api-key
```

### 3) Call Moltbot in your pipeline
```python
from gnosis.execution import MoltbotOrchestrator, load_moltbot_config

config = load_moltbot_config("configs/moltbot.yaml")
orchestrator = MoltbotOrchestrator(config)

forecast = {
    "symbol": "BTCUSDT",
    "direction": "up",
    "confidence": 0.72,
    "q05": 64200,
    "q50": 66800,
    "q95": 70100,
}

trade_plan = orchestrator.propose_trade(forecast)
orchestrator.notify(trade_plan)
```

This keeps Yoshi focused on forecasting while Moltbot handles AI reasoning and
service integration.
