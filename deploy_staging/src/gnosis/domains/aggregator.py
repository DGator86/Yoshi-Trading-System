"""Domain aggregation based on trade counts."""
import numpy as np
import pandas as pd


class DomainAggregator:
    """Aggregates print data into domains (D0-D3) based on trade counts."""

    def __init__(self, domain_config: dict):
        self.domain_config = domain_config

    def aggregate(self, prints_df: pd.DataFrame, domain: str) -> pd.DataFrame:
        """Aggregate prints into domain bars based on trade count."""
        n_trades = self.domain_config["domains"][domain]["n_trades"]

        bars = []
        for symbol in prints_df["symbol"].unique():
            sym_prints = prints_df[prints_df["symbol"] == symbol].copy()
            sym_prints = sym_prints.sort_values("timestamp")

            n_bars = len(sym_prints) // n_trades
            for i in range(n_bars):
                start_idx = i * n_trades
                end_idx = (i + 1) * n_trades
                chunk = sym_prints.iloc[start_idx:end_idx]

                bars.append({
                    "symbol": symbol,
                    "bar_idx": i,
                    "timestamp_start": chunk["timestamp"].iloc[0],
                    "timestamp_end": chunk["timestamp"].iloc[-1],
                    "open": chunk["price"].iloc[0],
                    "high": chunk["price"].max(),
                    "low": chunk["price"].min(),
                    "close": chunk["price"].iloc[-1],
                    "volume": chunk["quantity"].sum(),
                    "n_trades": len(chunk),
                    "buy_volume": chunk[chunk["side"] == "BUY"]["quantity"].sum(),
                    "sell_volume": chunk[chunk["side"] == "SELL"]["quantity"].sum(),
                })

        return pd.DataFrame(bars)


def compute_features(bars_df: pd.DataFrame, extended: bool = True) -> pd.DataFrame:
    """Compute features from domain bars.

    Args:
        bars_df: DataFrame with OHLCV bars
        extended: Whether to compute extended feature set

    Returns:
        DataFrame with computed features
    """
    df = bars_df.copy()

    # === Core Features ===

    # Returns
    df["returns"] = df.groupby("symbol")["close"].pct_change()

    # Realized volatility (trailing 20 bars)
    df["realized_vol"] = df.groupby("symbol")["returns"].transform(
        lambda x: x.rolling(20, min_periods=5).std()
    )

    # Order flow imbalance
    df["ofi"] = (df["buy_volume"] - df["sell_volume"]) / (df["buy_volume"] + df["sell_volume"] + 1e-9)

    # Price range
    df["range_pct"] = (df["high"] - df["low"]) / df["close"]

    if not extended:
        return df

    # === Extended Features ===

    # OFI momentum (rate of change of order flow imbalance)
    df["ofi_momentum"] = df.groupby("symbol")["ofi"].transform(
        lambda x: x.diff().rolling(5, min_periods=1).mean()
    )

    # Volume-weighted returns (stronger moves on higher volume)
    vol_ma = df.groupby("symbol")["volume"].transform(
        lambda x: x.rolling(20, min_periods=1).mean()
    )
    df["volume_weighted_returns"] = df["returns"] * (df["volume"] / (vol_ma + 1e-9))

    # Price position within recent range (0=low, 1=high)
    high_20 = df.groupby("symbol")["high"].transform(
        lambda x: x.rolling(20, min_periods=1).max()
    )
    low_20 = df.groupby("symbol")["low"].transform(
        lambda x: x.rolling(20, min_periods=1).min()
    )
    df["price_position"] = (df["close"] - low_20) / (high_20 - low_20 + 1e-9)

    # Intrabar volatility (high-low normalized)
    df["intrabar_vol"] = (df["high"] - df["low"]) / (df["close"] + 1e-9)

    # Return acceleration (momentum of momentum)
    df["return_accel"] = df.groupby("symbol")["returns"].transform(
        lambda x: x.rolling(5, min_periods=1).mean().diff()
    )

    # Volume momentum
    df["volume_momentum"] = df.groupby("symbol")["volume"].transform(
        lambda x: x.pct_change().rolling(5, min_periods=1).mean()
    )

    # Relative strength (rolling sum of positive vs negative returns)
    def rsi_like(returns, window=14):
        """Compute RSI-like indicator."""
        gains = returns.where(returns > 0, 0).rolling(window, min_periods=1).mean()
        losses = (-returns).where(returns < 0, 0).rolling(window, min_periods=1).mean()
        rs = gains / (losses + 1e-9)
        return rs / (1 + rs)  # Normalized to [0, 1]

    df["relative_strength"] = df.groupby("symbol")["returns"].transform(
        lambda x: rsi_like(x)
    )

    # Trend strength (absolute cumulative return over window)
    df["trend_strength"] = df.groupby("symbol")["returns"].transform(
        lambda x: x.rolling(10, min_periods=1).sum().abs()
    )

    # Volatility regime (current vol vs median vol)
    median_vol = df.groupby("symbol")["realized_vol"].transform(
        lambda x: x.rolling(100, min_periods=20).median()
    )
    df["vol_regime"] = df["realized_vol"] / (median_vol + 1e-9)

    # Fill NaN values
    feature_cols = [
        "ofi_momentum", "volume_weighted_returns", "price_position",
        "intrabar_vol", "return_accel", "volume_momentum",
        "relative_strength", "trend_strength", "vol_regime"
    ]
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df
