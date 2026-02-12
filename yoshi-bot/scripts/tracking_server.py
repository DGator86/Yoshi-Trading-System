#!/usr/bin/env python3
"""FastAPI server for Yoshi Prediction Tracking Dashboard.

Serves ML statistics, model effectiveness, and price prediction validation data.
"""

import json
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from gnosis.tracking.data_loader import (
    PredictionDataLoader,
    MetricsCalculator,
    PredictionStats,
    TradeStats,
    RegimeSnapshot,
)


# Pydantic models for API responses
class PredictionStatsResponse(BaseModel):
    total_predictions: int
    coverage_90: float
    coverage_80: float
    coverage_50: float
    sharpness_mean: float
    directional_accuracy: float
    mae: float
    rmse: float
    calibration_error: float
    abstention_rate: float
    brier_score: Optional[float] = None


class TradeStatsResponse(BaseModel):
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    avg_trade_return: float


class PredictionResponse(BaseModel):
    symbol: str
    timestamp: str
    actual_price: float
    predicted_median: float
    predicted_q05: float
    predicted_q95: float
    confidence: float
    actual_return: float
    correct: bool


class EquityMetricsResponse(BaseModel):
    max_equity: float
    min_equity: float
    final_equity: float
    volatility: float
    skewness: float
    kurtosis: float


class DashboardDataResponse(BaseModel):
    """Complete dashboard data snapshot."""
    prediction_stats: PredictionStatsResponse
    trade_stats: TradeStatsResponse
    equity_metrics: EquityMetricsResponse
    latest_predictions: list[PredictionResponse]
    accuracy_by_confidence: dict[str, float]
    accuracy_decay: dict[str, float]
    calibration_curve: dict[str, list[float]]


# FastAPI app
app = FastAPI(
    title="Yoshi Prediction Tracking API",
    description="Real-time tracking for ML prediction performance, model effectiveness, and price validation",
    version="1.0.0",
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global data loader (initialized with run directory)
data_loader: Optional[PredictionDataLoader] = None


def init_loader(run_dir: Path):
    """Initialize data loader with run directory."""
    global data_loader
    data_loader = PredictionDataLoader(run_dir)


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "yoshi-tracking"}


@app.get("/api/prediction-stats", response_model=PredictionStatsResponse)
def get_prediction_stats():
    """Get ML prediction statistics and accuracy metrics."""
    if not data_loader:
        raise HTTPException(status_code=503, detail="Data loader not initialized")

    stats = data_loader.extract_prediction_stats()
    return stats


@app.get("/api/trade-stats", response_model=TradeStatsResponse)
def get_trade_stats():
    """Get trading performance and model effectiveness metrics."""
    if not data_loader:
        raise HTTPException(status_code=503, detail="Data loader not initialized")

    stats = data_loader.extract_trade_stats()
    return stats


@app.get("/api/equity-metrics", response_model=EquityMetricsResponse)
def get_equity_metrics():
    """Get equity curve statistics."""
    if not data_loader:
        raise HTTPException(status_code=503, detail="Data loader not initialized")

    equity_curve = data_loader.load_equity_curve()
    metrics = MetricsCalculator.calculate_equity_metrics(equity_curve)

    if not metrics:
        raise HTTPException(status_code=404, detail="No equity curve data available")

    return metrics


@app.get("/api/latest-predictions", response_model=list[PredictionResponse])
def get_latest_predictions(limit: int = Query(10, ge=1, le=100)):
    """Get most recent predictions with actual outcomes."""
    if not data_loader:
        raise HTTPException(status_code=503, detail="Data loader not initialized")

    predictions = data_loader.load_predictions()

    if predictions.empty:
        return []

    # Sort by timestamp and take latest
    predictions_sorted = predictions.sort_values('timestamp_end', ascending=False).head(limit)

    results = []
    for _, row in predictions_sorted.iterrows():
        correct = (row['future_return'] > 0 and row['q50'] > row['close']) or \
                  (row['future_return'] < 0 and row['q50'] < row['close'])

        results.append(PredictionResponse(
            symbol=row.get('symbol', 'UNKNOWN'),
            timestamp=str(row['timestamp_end']),
            actual_price=float(row['close']),
            predicted_median=float(row['q50']),
            predicted_q05=float(row['q05']),
            predicted_q95=float(row['q95']),
            confidence=float(row.get('S_pmax_calibrated', 0.5)),
            actual_return=float(row.get('future_return', 0)),
            correct=bool(correct),
        ))

    return results


@app.get("/api/accuracy-by-confidence")
def get_accuracy_by_confidence():
    """Get directional accuracy bucketed by prediction confidence."""
    if not data_loader:
        raise HTTPException(status_code=503, detail="Data loader not initialized")

    return data_loader.calculate_accuracy_by_confidence()


@app.get("/api/accuracy-decay")
def get_accuracy_decay():
    """Get prediction accuracy degradation over forecast horizons."""
    if not data_loader:
        raise HTTPException(status_code=503, detail="Data loader not initialized")

    return data_loader.calculate_accuracy_decay()


@app.get("/api/calibration-curve")
def get_calibration_curve():
    """Get calibration curve (predicted vs observed probability)."""
    if not data_loader:
        raise HTTPException(status_code=503, detail="Data loader not initialized")

    predictions = data_loader.load_predictions()
    predicted_probs, observed_accs = MetricsCalculator.calculate_calibration_curve(predictions)

    return {
        "predicted_probabilities": predicted_probs,
        "observed_accuracies": observed_accs,
    }


@app.get("/api/dashboard", response_model=DashboardDataResponse)
def get_dashboard_data():
    """Get complete dashboard data in a single request."""
    if not data_loader:
        raise HTTPException(status_code=503, detail="Data loader not initialized")

    prediction_stats = data_loader.extract_prediction_stats()
    trade_stats = data_loader.extract_trade_stats()
    equity_curve = data_loader.load_equity_curve()
    equity_metrics = MetricsCalculator.calculate_equity_metrics(equity_curve)

    predictions = data_loader.load_predictions()
    predictions_sorted = predictions.sort_values('timestamp_end', ascending=False).head(10)

    latest_predictions = []
    for _, row in predictions_sorted.iterrows():
        correct = (row['future_return'] > 0 and row['q50'] > row['close']) or \
                  (row['future_return'] < 0 and row['q50'] < row['close'])

        latest_predictions.append(PredictionResponse(
            symbol=row.get('symbol', 'UNKNOWN'),
            timestamp=str(row['timestamp_end']),
            actual_price=float(row['close']),
            predicted_median=float(row['q50']),
            predicted_q05=float(row['q05']),
            predicted_q95=float(row['q95']),
            confidence=float(row.get('S_pmax_calibrated', 0.5)),
            actual_return=float(row.get('future_return', 0)),
            correct=bool(correct),
        ))

    predicted_probs, observed_accs = MetricsCalculator.calculate_calibration_curve(predictions)

    return DashboardDataResponse(
        prediction_stats=prediction_stats,
        trade_stats=trade_stats,
        equity_metrics=equity_metrics,
        latest_predictions=latest_predictions,
        accuracy_by_confidence=data_loader.calculate_accuracy_by_confidence(),
        accuracy_decay=data_loader.calculate_accuracy_decay(),
        calibration_curve={
            "predicted_probabilities": predicted_probs,
            "observed_accuracies": observed_accs,
        },
    )


def main():
    """Start the tracking server."""
    import argparse

    parser = argparse.ArgumentParser(description="Yoshi Prediction Tracking Server")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path.cwd() / "run_results",
        help="Path to experiment run directory containing predictions and stats",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8001, help="Server port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code changes")

    args = parser.parse_args()

    # Initialize data loader
    if not args.run_dir.exists():
        print(f"Warning: Run directory not found: {args.run_dir}")
        print(f"Server will start but will return 503 until valid data is provided")

    init_loader(args.run_dir)

    # Start server
    uvicorn.run(
        "tracking_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
