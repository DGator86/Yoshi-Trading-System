"""Data loading and stub generation for print-dominant data."""
import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd


def generate_stub_prints(
    symbols: list[str],
    start_date: str = "2023-01-01",
    n_days: int = 365,
    trades_per_day: int = 50000,
    seed: int = 1337,
) -> pd.DataFrame:
    """Generate stub print (trade) data for testing."""
    rng = np.random.default_rng(seed)

    records = []
    base_prices = {"BTCUSDT": 30000.0, "ETHUSDT": 2000.0, "SOLUSDT": 25.0}

    start = pd.Timestamp(start_date)

    for sym in symbols:
        price = base_prices.get(sym, 100.0)
        vol = price * 0.02  # 2% daily vol

        for day in range(n_days):
            # Back-compat alias: n_days + trades_per_day -> total n_trades
            if n_days is not None:
                n_trades = int(n_days) * int(trades_per_day)

            day_start = start + timedelta(days=day)
            # Add variation but ensure at least 10 trades
            variation = min(trades_per_day // 2, 5000)
            n_trades = max(10, trades_per_day + rng.integers(-variation, variation + 1))

            # Random walk for prices
            returns = rng.normal(0, vol / np.sqrt(n_trades), n_trades)
            prices = price * np.exp(np.cumsum(returns / price))
            price = prices[-1]  # carry forward

            # Timestamps spread across the day
            offsets_ms = np.sort(rng.integers(0, 86400000, n_trades))
            timestamps = [day_start + timedelta(milliseconds=int(o)) for o in offsets_ms]

            # Quantities and sides
            qtys = rng.exponential(0.1, n_trades)
            sides = rng.choice(["BUY", "SELL"], n_trades, p=[0.5, 0.5])

            for i in range(n_trades):
                records.append({
                    "timestamp": timestamps[i],
                    "symbol": sym,
                    "price": round(prices[i], 2),
                    "quantity": round(qtys[i], 6),
                    "side": sides[i],
                    "trade_id": f"{sym}_{day}_{i}",
                })

    df = pd.DataFrame(records)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_or_create_prints(
    parquet_dir: str | Path,
    symbols: list[str],
    seed: int = 1337,
    mode: str = "parquet",
    live_config: dict | None = None,
) -> pd.DataFrame:
    """Load prints from parquet, fetch live data, or create stub data.

    Args:
        parquet_dir: Directory for parquet files
        symbols: List of trading pairs (e.g., ['BTCUSDT', 'ETHUSDT'])
        seed: Random seed for stub data generation
        mode: Data source mode - 'parquet', 'stub', or 'live'
        live_config: Configuration dict for live data fetching:
            - exchange: Exchange name (default: 'binance')
            - days: Days of historical data (default: 30)
            - api_key: Optional API key
            - secret: Optional API secret

    Returns:
        DataFrame with print (trade) data
    """
    parquet_dir = Path(parquet_dir)
    parquet_dir.mkdir(parents=True, exist_ok=True)

    prints_path = parquet_dir / "prints.parquet"

    # Mode: parquet - load from file if exists
    if mode == "parquet" and prints_path.exists():
        return pd.read_parquet(prints_path)

    # Mode: live - fetch real data via CCXT
    if mode == "live":
        try:
            from .ccxt_loader import fetch_live_prints
        except ImportError:
            raise ImportError(
                "CCXT is required for live data mode. "
                "Install with: pip install ccxt"
            )

        live_config = live_config or {}
        exchange = live_config.get("exchange", "binance")
        days = live_config.get("days", 30)
        api_key = live_config.get("api_key")
        secret = live_config.get("secret")

        print(f"Fetching live data from {exchange} for {symbols}...")
        df = fetch_live_prints(
            symbols=symbols,
            exchange=exchange,
            days=days,
            api_key=api_key,
            secret=secret,
        )

        # Cache to parquet for future runs
        df.to_parquet(prints_path, index=False)
        print(f"Cached {len(df)} trades to {prints_path}")
        return df

    # Mode: stub or fallback - generate synthetic data
    df = generate_stub_prints(symbols, seed=seed, n_days=30, trades_per_day=5000)
    df.to_parquet(prints_path, index=False)
    return df


def create_data_manifest(
    prints_df: pd.DataFrame,
    manifest_path: str | Path,
) -> dict:
    """Create data manifest with hash of input data."""
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute hash of data
    data_bytes = prints_df.to_parquet()
    data_hash = hashlib.sha256(data_bytes).hexdigest()

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": len(prints_df),
        "symbols": sorted(prints_df["symbol"].unique().tolist()),
        "min_timestamp": str(prints_df["timestamp"].min()),
        "max_timestamp": str(prints_df["timestamp"].max()),
        "data_hash": data_hash,
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest
