"""Data loading and stub generation for print-dominant data."""
import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

# Import CoinGecko at module level for better code organization
try:
    from .coingecko import fetch_coingecko_prints
    COINGECKO_AVAILABLE = True
except ImportError:
    COINGECKO_AVAILABLE = False


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
    use_coingecko: bool = False,
    coingecko_api_key: str | None = None,
) -> pd.DataFrame:
    """
    Load prints from parquet or create data.
    
    Args:
        parquet_dir: Directory to store/load parquet files
        symbols: List of trading symbols
        seed: Random seed for stub data generation
        use_coingecko: If True, fetch real data from CoinGecko API
        coingecko_api_key: CoinGecko API key (optional, can use env var)
        
    Returns:
        DataFrame with print data
    """
    parquet_dir = Path(parquet_dir)
    parquet_dir.mkdir(parents=True, exist_ok=True)

    prints_path = parquet_dir / "prints.parquet"

    if prints_path.exists():
        return pd.read_parquet(prints_path)

    if use_coingecko:
        # Fetch real data from CoinGecko
        if not COINGECKO_AVAILABLE:
            raise ImportError("CoinGecko module not available. Check installation.")
        df = fetch_coingecko_prints(
            symbols=symbols,
            api_key=coingecko_api_key,
            days=30,
            trades_per_interval=10,
        )
    else:
        # Generate and save stub data
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
