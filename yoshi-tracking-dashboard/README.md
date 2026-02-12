# Yoshi Prediction Tracking Dashboard

Real-time interactive dashboard for tracking ML prediction performance, model effectiveness, and price prediction validation.

## Features

- 📊 **Model Effectiveness Metrics** - Win rate, Sharpe ratio, profit factor, max drawdown
- 📈 **ML Statistical Progress** - Directional accuracy, calibration curves, confidence intervals, abstention rates
- 🔮 **Price Prediction Validation** - Accuracy by confidence level, time horizon decay, calibration analysis
- ⚡ **Real-time Updates** - Auto-refreshes every 30 seconds
- 📉 **Equity Curve Analytics** - Volatility, skewness, kurtosis analysis
- 🎯 **Latest Predictions** - View most recent predictions with confidence and outcomes

## Getting Started

### Prerequisites

- Node.js >= 18
- The Yoshi tracking server running (see below)

### Installation

```bash
cd yoshi-tracking-dashboard
npm install
```

### Start the Dashboard

```bash
npm run dev
```

The dashboard will open at `http://localhost:3000`

### Backend Setup

Make sure the tracking server is running in another terminal:

```bash
cd yoshi-bot
python scripts/tracking_server.py --run-dir ./path/to/run_results
```

The server exposes:
- `http://localhost:8001/api/dashboard` - Complete dashboard data
- `http://localhost:8001/api/prediction-stats` - ML accuracy metrics
- `http://localhost:8001/api/trade-stats` - Trading performance metrics
- `http://localhost:8001/health` - Health check

## Dashboard Sections

### 1. Model Effectiveness & Trading Performance
Shows key trading metrics:
- Total and annualized returns
- Sharpe and Sortino ratios
- Win rate and profit factor
- Maximum drawdown
- Trade statistics

### 2. ML Statistical Progress & Calibration
Displays prediction quality metrics:
- Directional accuracy trending
- Prediction interval coverage (50%, 80%, 90%)
- Calibration curve (predicted vs observed probability)
- Prediction error metrics (MAE, RMSE)
- Sharpness and abstention statistics

### 3. Price Prediction Validation & Accuracy
Analyzes time-agnostic prediction performance:
- Accuracy bucketed by confidence level
- Accuracy decay over forecast horizons
- Coverage across different timeframes
- Robustness indicators

### 4. Latest Predictions
Real-time table showing:
- Most recent 10 predictions
- Actual vs predicted prices
- 90% confidence intervals
- Prediction confidence
- Hit/miss classification
- Hit rate statistics

## Configuration

### API Base URL

By default connects to `http://localhost:8001/api`. Override with environment variable:

```bash
VITE_API_BASE=http://your-server.com/api npm run dev
```

### Refresh Interval

Modify the refresh interval in `src/App.jsx`:

```javascript
// Default: 30000ms (30 seconds)
const interval = setInterval(fetchDashboardData, 30000)
```

## Build for Production

```bash
npm run build
```

Output will be in the `dist/` directory.

## Tech Stack

- **React 18** - UI framework
- **Tailwind CSS** - Styling
- **Vite** - Build tool and dev server
- **Axios** - HTTP client
- **Plotly.js** - Advanced charts (when needed)

## Development

```bash
# Start development server with hot reload
npm run dev

# Run linter
npm run lint

# Build for production
npm run build

# Preview production build locally
npm run preview
```

## Project Structure

```
src/
├── App.jsx                    # Main app component
├── main.jsx                   # Entry point
├── index.css                  # Tailwind styles
├── components/
│   ├── Header.jsx            # Dashboard header
│   ├── MetricCards.jsx       # Trading performance cards
│   ├── PerformanceCharts.jsx # ML metrics and equity analysis
│   ├── AccuracyAnalysis.jsx  # Prediction accuracy breakdowns
│   └── PredictionTable.jsx   # Real-time predictions table
```

## About Yoshi

The Yoshi Trading System is a unified, physics-inspired prediction market trading system that combines:
- Particle physics models for price prediction
- Quantum Monte Carlo simulations
- Ralph meta-learner for hyperparameter optimization
- Real-time Kalshi market integration

This dashboard provides transparency into the ML and trading components' real-time performance.
