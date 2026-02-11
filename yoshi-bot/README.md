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

## Secrets & Credential Hygiene

All credentials must be provided via environment variables only.

- Setup guide:
  - [docs/secrets-and-credentials.md](docs/secrets-and-credentials.md)
- Copy template:
  - `cp .env.example .env`

If any key has been exposed in chat/logs, rotate it immediately before use.

## Continuous Crypto Source Scanner

The repository now includes a persistent source scanner that continuously polls
multiple exchanges and writes normalized snapshots for downstream models.

### Run once (smoke check)
```bash
python3 yoshi-bot/scripts/run_crypto_source_scanner.py --once \
  --config yoshi-bot/configs/crypto_source_scanner.yaml
```

### Run continuously
```bash
python3 yoshi-bot/scripts/run_crypto_source_scanner.py \
  --config yoshi-bot/configs/crypto_source_scanner.yaml
```

Outputs are written to `data/live/crypto_sources/`:
- `latest.json` (full latest cycle)
- `consensus_latest.json` (cross-source consensus)
- `snapshots_YYYYMMDD.ndjson` (append-only history)

### systemd service

Service unit: `services/crypto-source-scanner.service`

```bash
sudo cp services/crypto-source-scanner.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now crypto-source-scanner
sudo systemctl status crypto-source-scanner
```

## Continuous Learning + Optimization Supervisor (Always-On)

To keep ML/backtests/hyperparameter adaptation running continuously, use the
domain-triggered supervisor:

```bash
python3 yoshi-bot/scripts/run_continuous_learning.py \
  --config yoshi-bot/configs/continuous_learning.yaml
```

It monitors timeframe close events and triggers a full cycle on the configured
relative quantized cadence (`run_every_bars`, default `1`, i.e. every close).

`n=2000` is used as the rolling backtest timeline window (`fetch_n`), not as
the trigger threshold:
- run experiment (with Ralph Loop hyperparameter selection if `hparams.yaml` is set)
- run backtest on the last `fetch_n` timeline units (per symbol)
- optional extra improvement-loop run

### One-shot scheduler tick
```bash
python3 yoshi-bot/scripts/run_continuous_learning.py --once \
  --config yoshi-bot/configs/continuous_learning.yaml
```

### systemd service

Service unit: `services/continuous-learning.service`

```bash
sudo cp services/continuous-learning.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now continuous-learning
sudo systemctl status continuous-learning
```

State is persisted at:
- `reports/continuous/supervisor_state.json`

Per-domain run summaries now also include active forecasting module weights and
confidence (from the modular taxonomy gating scaffold) in:
- `reports/continuous/<timeframe>/<run_id>/domain_run_summary.json`

#### Practical settings

- Trigger cadence:
  - `run_every_bars=1` means each timeframe runs on each bar close.
  - Increase to `2/3/...` to reduce compute load.
- Backtest window:
  - `fetch_n=2000` gives these approximate history spans:
    - `1m`: ~33h
    - `5m`: ~6.9d
    - `15m`: ~20.8d
    - `30m`: ~41.7d
    - `1h`: ~83.3d
    - `4h`: ~333d
    - `1d`: ~2000d

## Crypto Price-as-a-Particle (Strengthened Approach)

For the crypto-first, physics-consistent interpretation of VWAP, EMA, Bollinger Bands,
RSI, Ichimoku, funding, OI, liquidations, and CVD, see:

- [docs/crypto-strengthened-approach.md](docs/crypto-strengthened-approach.md)

## Crypto RFP HSO Developer Handoff Assets

Canonical handoff assets for the Regime Field Probability Hilbert Space Overlay
and walkforward collapse projection stack are now versioned in-repo:

- Full developer spec:
  - [docs/crypto-rfp-hso-developer-spec.md](docs/crypto-rfp-hso-developer-spec.md)
- Gemini prompt artifact:
  - [docs/crypto-rfp-hso-gemini-prompt.txt](docs/crypto-rfp-hso-gemini-prompt.txt)
- Canonical defaults/constants YAML:
  - [configs/crypto_rfp_hso.yaml](configs/crypto_rfp_hso.yaml)

The YAML includes:
- default runtime config values (bucketing, event-time, horizons, hazard controls)
- `VALID_MASK`
- regime-method gating matrix `G`
- `ORDER_METHOD_MULT`
- event alphabet used for explicit event-time semantics.

## Modular Forecasting Taxonomy Assets (12 Paradigms)

A practical, implementation-oriented taxonomy for combining TA, classical stats,
macro, derivatives, microstructure, on-chain, sentiment, ML, deep sequence,
regime state machine, scenario MC, and crowd priors is versioned here:

- Spec:
  - [docs/crypto-forecasting-taxonomy.md](docs/crypto-forecasting-taxonomy.md)
- Prompt artifact:
  - [docs/crypto-forecasting-taxonomy-prompt.txt](docs/crypto-forecasting-taxonomy-prompt.txt)
- Config/policy defaults:
  - [configs/crypto_forecasting_taxonomy.yaml](configs/crypto_forecasting_taxonomy.yaml)
- Code module (registry + gating + confidence):
  - `src/gnosis/forecasting/modular_ensemble.py`

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
