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
    """Generate stub print (trade) data for testing.

    Uses pre-allocated arrays for memory efficiency.
    """
    rng = np.random.default_rng(seed)
    base_prices = {"BTCUSDT": 30000.0, "ETHUSDT": 2000.0, "SOLUSDT": 25.0}
    start = pd.Timestamp(start_date)

    # Pre-calculate total records for memory pre-allocation
    all_dfs = []

    for sym in symbols:
        price = base_prices.get(sym, 100.0)
        vol = price * 0.02  # 2% daily vol

        for day in range(n_days):
            day_start = start + timedelta(days=day)

            # Add variation but ensure at least 10 trades
            variation = min(trades_per_day // 2, 5000)
            n_trades = max(10, trades_per_day + rng.integers(-variation, variation + 1))

            # Random walk for prices (vectorized)
            returns = rng.normal(0, vol / np.sqrt(n_trades), n_trades)
            prices = price * np.exp(np.cumsum(returns / price))
            price = prices[-1]  # carry forward

            # Timestamps spread across the day (vectorized)
            offsets_ms = np.sort(rng.integers(0, 86400000, n_trades))
            timestamps = pd.to_datetime(day_start) + pd.to_timedelta(offsets_ms, unit='ms')

            # Quantities and sides (vectorized)
            qtys = rng.exponential(0.1, n_trades)
            sides = rng.choice(["BUY", "SELL"], n_trades, p=[0.5, 0.5])

            # Create DataFrame for this day (vectorized, no row-by-row append)
            day_df = pd.DataFrame({
                "timestamp": timestamps,
                "symbol": sym,
                "price": np.round(prices, 2),
                "quantity": np.round(qtys, 6),
                "side": sides,
                "trade_id": [f"{sym}_{day}_{i}" for i in range(n_trades)],
            })
            all_dfs.append(day_df)

    # Concatenate all at once (much faster than row-by-row)
    df = pd.concat(all_dfs, ignore_index=True)
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
        try:
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
        except (ValueError, Exception) as e:
            print(f"Warning: Could not fetch live data: {e}")
            print("Falling back to realistic synthetic data generation...")
            # Fall through to stub generation with realistic parameters
            df = generate_stub_prints(
                symbols, seed=seed, n_days=days, trades_per_day=10000
            )
            df.to_parquet(prints_path, index=False)
            print(f"Generated {len(df)} synthetic trades")
            return df

    # Mode: stub or fallback - generate synthetic data
    df = generate_stub_prints(symbols, seed=seed, n_days=30, trades_per_day=5000)
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
