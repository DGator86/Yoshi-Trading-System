"""Label generation for supervised learning.

Generates multiple types of labels for training:
- Forward returns over multiple horizons
- Direction (up/down/flat) classification
- Volatility/range proxies
- Triple-barrier outcomes (profit/stop/time)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict


@dataclass
class LabelConfig:
    """Configuration for label generation."""

    # Multi-horizon settings
    horizons: List[int] = None  # Default: [1, 2, 4, 8, 16, 32]

    # Direction thresholds (as fraction of volatility)
    direction_threshold_vol_mult: float = 0.5  # |ret| > vol * mult => up/down

    # Triple-barrier settings
    profit_take_mult: float = 2.0  # Take profit at vol * mult
    stop_loss_mult: float = 1.0    # Stop loss at vol * mult
    max_holding_bars: int = 32     # Max bars before time exit

    # Volatility lookback
    vol_lookback: int = 20

    def __post_init__(self):
        if self.horizons is None:
            self.horizons = [1, 2, 4, 8, 16, 32]


class LabelGenerator:
    """Generate labels for supervised learning.

    Computes:
    - Forward returns at multiple horizons
    - Direction labels (UP/DOWN/FLAT)
    - Volatility proxies
    - Triple-barrier outcomes

    All labels are computed WITHOUT lookahead - labels at time t
    only use price information from t onwards (future), which is
    the target we're predicting.
    """

    def __init__(self, config: LabelConfig = None):
        self.config = config or LabelConfig()

    def generate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all labels for a dataframe.

        Args:
            df: DataFrame with columns: symbol, bar_idx, timestamp_end,
                close, high, low, realized_vol

        Returns:
            DataFrame with original columns plus all label columns
        """
        result = df.copy()

        # Compute volatility if not present
        if 'realized_vol' not in result.columns:
            result['realized_vol'] = result.groupby('symbol')['close'].transform(
                lambda x: x.pct_change().rolling(
                    self.config.vol_lookback, min_periods=5
                ).std()
            )

        # Generate forward returns for all horizons
        result = self._add_forward_returns(result)

        # Generate direction labels for all horizons
        result = self._add_direction_labels(result)

        # Generate volatility proxies
        result = self._add_volatility_proxies(result)

        # Generate triple-barrier outcomes
        result = self._add_triple_barrier(result)

        return result

    def _add_forward_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add forward return columns for all horizons."""
        result = df.copy()

        for h in self.config.horizons:
            result[f'fwd_ret_{h}'] = result.groupby('symbol')['close'].transform(
                lambda x: x.shift(-h) / x - 1
            )

        # Legacy column name for compatibility
        if 'future_return' not in result.columns and len(self.config.horizons) > 0:
            # Use the middle horizon as default
            default_h = self.config.horizons[len(self.config.horizons) // 2]
            result['future_return'] = result[f'fwd_ret_{default_h}']

        return result

    def _add_direction_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add direction labels (UP/DOWN/FLAT) for all horizons."""
        result = df.copy()
        threshold_mult = self.config.direction_threshold_vol_mult

        for h in self.config.horizons:
            ret_col = f'fwd_ret_{h}'
            if ret_col not in result.columns:
                continue

            # Threshold is volatility scaled by horizon sqrt(h)
            vol = result['realized_vol'].fillna(0.01)
            threshold = vol * threshold_mult * np.sqrt(h)

            # Classify direction
            direction = np.where(
                result[ret_col] > threshold,
                'UP',
                np.where(
                    result[ret_col] < -threshold,
                    'DOWN',
                    'FLAT'
                )
            )
            result[f'direction_{h}'] = direction

            # Numeric encoding: 1=UP, 0=FLAT, -1=DOWN
            result[f'direction_{h}_num'] = np.where(
                result[ret_col] > threshold, 1,
                np.where(result[ret_col] < -threshold, -1, 0)
            )

        return result

    def _add_volatility_proxies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility/range proxy labels."""
        result = df.copy()

        for h in self.config.horizons:
            # Forward realized volatility
            result[f'fwd_vol_{h}'] = result.groupby('symbol')['close'].transform(
                lambda x: x.pct_change().shift(-1).rolling(h, min_periods=1).std().shift(-h+1)
            )

            # Forward range (max - min) / close
            result[f'fwd_range_{h}'] = result.groupby('symbol').apply(
                lambda g: self._compute_forward_range(g, h)
            ).reset_index(level=0, drop=True)

        return result

    def _compute_forward_range(self, group: pd.DataFrame, horizon: int) -> pd.Series:
        """Compute forward price range for a symbol group."""
        n = len(group)
        ranges = np.full(n, np.nan)

        closes = group['close'].values
        highs = group['high'].values if 'high' in group.columns else closes
        lows = group['low'].values if 'low' in group.columns else closes

        for i in range(n - horizon):
            future_highs = highs[i+1:i+1+horizon]
            future_lows = lows[i+1:i+1+horizon]
            if len(future_highs) > 0:
                max_high = np.max(future_highs)
                min_low = np.min(future_lows)
                ranges[i] = (max_high - min_low) / closes[i]

        return pd.Series(ranges, index=group.index)

    def _add_triple_barrier(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add triple-barrier outcome labels.

        Triple-barrier method from Lopez de Prado:
        - Upper barrier: take profit
        - Lower barrier: stop loss
        - Vertical barrier: time exit

        Outcome is which barrier is hit first.
        """
        result = df.copy()

        # Compute barriers
        vol = result['realized_vol'].fillna(0.01)
        take_profit = vol * self.config.profit_take_mult
        stop_loss = vol * self.config.stop_loss_mult
        max_bars = self.config.max_holding_bars

        # Apply triple-barrier by symbol
        tb_outcomes = result.groupby('symbol').apply(
            lambda g: self._compute_triple_barrier_group(
                g, take_profit.loc[g.index], stop_loss.loc[g.index], max_bars
            )
        ).reset_index(level=0, drop=True)

        result['tb_outcome'] = tb_outcomes['outcome']
        result['tb_bars_held'] = tb_outcomes['bars_held']
        result['tb_return'] = tb_outcomes['return']

        # Numeric encoding: 1=profit, 0=time, -1=stop
        result['tb_outcome_num'] = np.where(
            result['tb_outcome'] == 'PROFIT', 1,
            np.where(result['tb_outcome'] == 'STOP', -1, 0)
        )

        return result

    def _compute_triple_barrier_group(
        self,
        group: pd.DataFrame,
        take_profit: pd.Series,
        stop_loss: pd.Series,
        max_bars: int
    ) -> pd.DataFrame:
        """Compute triple-barrier outcomes for a symbol group."""
        n = len(group)
        outcomes = ['TIME'] * n
        bars_held = [max_bars] * n
        returns = [0.0] * n

        closes = group['close'].values
        tp = take_profit.values
        sl = stop_loss.values

        for i in range(n - 1):
            entry_price = closes[i]

            for j in range(1, min(max_bars + 1, n - i)):
                ret = (closes[i + j] - entry_price) / entry_price

                # Check upper barrier (take profit)
                if ret >= tp[i]:
                    outcomes[i] = 'PROFIT'
                    bars_held[i] = j
                    returns[i] = ret
                    break

                # Check lower barrier (stop loss)
                if ret <= -sl[i]:
                    outcomes[i] = 'STOP'
                    bars_held[i] = j
                    returns[i] = ret
                    break
            else:
                # Time exit (vertical barrier)
                if i + max_bars < n:
                    returns[i] = (closes[i + max_bars] - entry_price) / entry_price

        return pd.DataFrame({
            'outcome': outcomes,
            'bars_held': bars_held,
            'return': returns
        }, index=group.index)


def generate_labels(
    df: pd.DataFrame,
    horizons: List[int] = None,
    include_triple_barrier: bool = True,
    **config_kwargs
) -> pd.DataFrame:
    """Convenience function to generate all labels.

    Args:
        df: Input DataFrame with OHLC data
        horizons: List of forecast horizons (default: [1, 2, 4, 8, 16, 32])
        include_triple_barrier: Whether to compute triple-barrier labels
        **config_kwargs: Additional LabelConfig parameters

    Returns:
        DataFrame with all label columns added
    """
    config = LabelConfig(horizons=horizons, **config_kwargs)
    generator = LabelGenerator(config)

    if include_triple_barrier:
        return generator.generate_all(df)
    else:
        result = df.copy()
        if 'realized_vol' not in result.columns:
            result['realized_vol'] = result.groupby('symbol')['close'].transform(
                lambda x: x.pct_change().rolling(
                    config.vol_lookback, min_periods=5
                ).std()
            )
        result = generator._add_forward_returns(result)
        result = generator._add_direction_labels(result)
        result = generator._add_volatility_proxies(result)
        return result
