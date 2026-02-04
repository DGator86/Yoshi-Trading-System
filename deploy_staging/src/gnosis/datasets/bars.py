"""Multi-bar type aggregation for timeframe-agnostic evaluation.

Supports:
- Time bars (fixed time intervals)
- Tick bars (fixed number of ticks/trades)
- Volume bars (fixed volume)
- Dollar bars (fixed dollar volume)
- Event bars (triggered by specific events)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List, Callable
from enum import Enum


class BarType(Enum):
    """Types of bars for aggregation."""
    TIME = "time"
    TICK = "tick"
    VOLUME = "volume"
    DOLLAR = "dollar"
    EVENT = "event"


@dataclass
class BarConfig:
    """Configuration for bar aggregation."""

    bar_type: BarType = BarType.TICK

    # Time bar settings
    time_interval: str = "1h"  # pandas resample frequency

    # Tick/trade bar settings
    n_ticks: int = 200

    # Volume bar settings
    volume_threshold: float = 1000.0

    # Dollar bar settings
    dollar_threshold: float = 100000.0

    # Event bar settings
    event_column: str = None


class BarAggregator:
    """Aggregate raw trade data into various bar types.

    All bar types produce consistent OHLCV output with:
    - symbol, bar_idx, timestamp_start, timestamp_end
    - open, high, low, close, volume
    - buy_volume, sell_volume, n_trades
    """

    def __init__(self, config: BarConfig = None):
        self.config = config or BarConfig()

    def aggregate(
        self,
        prints_df: pd.DataFrame,
        bar_type: Optional[BarType] = None
    ) -> pd.DataFrame:
        """Aggregate prints into bars.

        Args:
            prints_df: DataFrame with columns:
                - timestamp, symbol, price, quantity, side
            bar_type: Override config bar type

        Returns:
            DataFrame with OHLCV bars
        """
        bt = bar_type or self.config.bar_type

        if bt == BarType.TIME:
            return self._aggregate_time_bars(prints_df)
        elif bt == BarType.TICK:
            return self._aggregate_tick_bars(prints_df)
        elif bt == BarType.VOLUME:
            return self._aggregate_volume_bars(prints_df)
        elif bt == BarType.DOLLAR:
            return self._aggregate_dollar_bars(prints_df)
        elif bt == BarType.EVENT:
            return self._aggregate_event_bars(prints_df)
        else:
            raise ValueError(f"Unknown bar type: {bt}")

    def _aggregate_time_bars(self, prints_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate into time-based bars."""
        bars = []

        for symbol in prints_df['symbol'].unique():
            sym_prints = prints_df[prints_df['symbol'] == symbol].copy()
            sym_prints = sym_prints.sort_values('timestamp')
            sym_prints = sym_prints.set_index('timestamp')

            # Resample to time interval
            resampled = sym_prints.resample(self.config.time_interval)

            bar_idx = 0
            for ts, group in resampled:
                if len(group) == 0:
                    continue

                bars.append(self._create_bar(symbol, bar_idx, group))
                bar_idx += 1

        return pd.DataFrame(bars)

    def _aggregate_tick_bars(self, prints_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate into tick-count bars."""
        bars = []
        n_ticks = self.config.n_ticks

        for symbol in prints_df['symbol'].unique():
            sym_prints = prints_df[prints_df['symbol'] == symbol].copy()
            sym_prints = sym_prints.sort_values('timestamp').reset_index(drop=True)

            n_bars = len(sym_prints) // n_ticks

            for i in range(n_bars):
                start_idx = i * n_ticks
                end_idx = (i + 1) * n_ticks
                chunk = sym_prints.iloc[start_idx:end_idx]

                bars.append(self._create_bar(symbol, i, chunk))

        return pd.DataFrame(bars)

    def _aggregate_volume_bars(self, prints_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate into volume-based bars."""
        bars = []
        threshold = self.config.volume_threshold

        for symbol in prints_df['symbol'].unique():
            sym_prints = prints_df[prints_df['symbol'] == symbol].copy()
            sym_prints = sym_prints.sort_values('timestamp').reset_index(drop=True)

            cumvol = 0.0
            bar_start = 0
            bar_idx = 0

            for i, row in sym_prints.iterrows():
                cumvol += row['quantity']

                if cumvol >= threshold:
                    chunk = sym_prints.iloc[bar_start:i+1]
                    bars.append(self._create_bar(symbol, bar_idx, chunk))
                    bar_idx += 1
                    bar_start = i + 1
                    cumvol = 0.0

        return pd.DataFrame(bars)

    def _aggregate_dollar_bars(self, prints_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate into dollar-volume-based bars."""
        bars = []
        threshold = self.config.dollar_threshold

        for symbol in prints_df['symbol'].unique():
            sym_prints = prints_df[prints_df['symbol'] == symbol].copy()
            sym_prints = sym_prints.sort_values('timestamp').reset_index(drop=True)

            cumdollar = 0.0
            bar_start = 0
            bar_idx = 0

            for i, row in sym_prints.iterrows():
                cumdollar += row['quantity'] * row['price']

                if cumdollar >= threshold:
                    chunk = sym_prints.iloc[bar_start:i+1]
                    bars.append(self._create_bar(symbol, bar_idx, chunk))
                    bar_idx += 1
                    bar_start = i + 1
                    cumdollar = 0.0

        return pd.DataFrame(bars)

    def _aggregate_event_bars(self, prints_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate into event-triggered bars.

        Events are detected when event_column changes from 0 to non-zero.
        """
        bars = []
        event_col = self.config.event_column

        if event_col is None or event_col not in prints_df.columns:
            raise ValueError(f"Event column '{event_col}' not found")

        for symbol in prints_df['symbol'].unique():
            sym_prints = prints_df[prints_df['symbol'] == symbol].copy()
            sym_prints = sym_prints.sort_values('timestamp').reset_index(drop=True)

            # Find event boundaries
            events = sym_prints[event_col].values
            boundaries = np.where(np.diff(events != 0, prepend=0))[0]
            boundaries = np.append(boundaries, len(sym_prints))

            bar_idx = 0
            for i in range(len(boundaries) - 1):
                start = boundaries[i]
                end = boundaries[i + 1]

                if end > start:
                    chunk = sym_prints.iloc[start:end]
                    bars.append(self._create_bar(symbol, bar_idx, chunk))
                    bar_idx += 1

        return pd.DataFrame(bars)

    def _create_bar(
        self,
        symbol: str,
        bar_idx: int,
        chunk: pd.DataFrame
    ) -> dict:
        """Create a single bar from a chunk of prints."""
        # Handle timestamp index vs column
        if 'timestamp' in chunk.columns:
            timestamps = chunk['timestamp']
        else:
            timestamps = chunk.index

        buy_mask = chunk['side'] == 'BUY'
        sell_mask = chunk['side'] == 'SELL'

        return {
            'symbol': symbol,
            'bar_idx': bar_idx,
            'timestamp_start': timestamps.iloc[0],
            'timestamp_end': timestamps.iloc[-1],
            'open': chunk['price'].iloc[0],
            'high': chunk['price'].max(),
            'low': chunk['price'].min(),
            'close': chunk['price'].iloc[-1],
            'volume': chunk['quantity'].sum(),
            'buy_volume': chunk.loc[buy_mask, 'quantity'].sum() if buy_mask.any() else 0.0,
            'sell_volume': chunk.loc[sell_mask, 'quantity'].sum() if sell_mask.any() else 0.0,
            'n_trades': len(chunk),
        }


def compute_bar_features(bars_df: pd.DataFrame) -> pd.DataFrame:
    """Compute standard features from bar data.

    Features computed:
    - returns: bar-to-bar percentage change
    - realized_vol: rolling volatility
    - ofi: order flow imbalance
    - range_pct: bar range as percentage
    - vwap: volume-weighted average price (if tick data available)
    """
    df = bars_df.copy()

    # Returns
    df['returns'] = df.groupby('symbol')['close'].pct_change()

    # Realized volatility (trailing 20 bars)
    df['realized_vol'] = df.groupby('symbol')['returns'].transform(
        lambda x: x.rolling(20, min_periods=5).std()
    )

    # Order flow imbalance
    df['ofi'] = (df['buy_volume'] - df['sell_volume']) / (
        df['buy_volume'] + df['sell_volume'] + 1e-9
    )

    # Price range
    df['range_pct'] = (df['high'] - df['low']) / df['close']

    # VWAP-like metrics (if we had tick-level data preserved)
    if 'vwap' not in df.columns:
        df['vwap'] = (df['high'] + df['low'] + df['close']) / 3

    # Tick intensity (trades per bar, normalized)
    mean_trades = df['n_trades'].mean()
    df['tick_intensity'] = df['n_trades'] / max(mean_trades, 1)

    # Volume intensity
    df['vol_intensity'] = df.groupby('symbol')['volume'].transform(
        lambda x: x / x.rolling(20, min_periods=1).mean()
    )

    return df


def create_bars(
    prints_df: pd.DataFrame,
    bar_type: str = 'tick',
    **kwargs
) -> pd.DataFrame:
    """Convenience function to create bars.

    Args:
        prints_df: Raw trade/print data
        bar_type: One of 'time', 'tick', 'volume', 'dollar', 'event'
        **kwargs: Additional BarConfig parameters

    Returns:
        DataFrame with OHLCV bars
    """
    bt = BarType(bar_type)
    config = BarConfig(bar_type=bt, **kwargs)
    aggregator = BarAggregator(config)

    return aggregator.aggregate(prints_df)
