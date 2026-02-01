"""Signal generation from forecasts."""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SignalConfig:
    """Configuration for signal generation."""

    mode: str = "x_hat_threshold"  # "x_hat_threshold", "quantile_skew"
    long_threshold: float = 0.0  # Go long when x_hat > threshold
    use_abstain: bool = True  # Respect abstain flag from predictions
    min_confidence: float = 0.0  # Min S_pmax to take position


class SignalGenerator:
    """Convert forecasts into trading signals (LONG=1, FLAT=0)."""

    def __init__(self, config: SignalConfig):
        self.config = config

    def generate_single(self, row: pd.Series) -> int:
        """Generate signal for a single prediction row.

        Args:
            row: Series with x_hat, abstain, and optionally S_pmax

        Returns:
            1 for LONG, 0 for FLAT
        """
        # Respect abstain flag
        if self.config.use_abstain and row.get("abstain", False):
            return 0

        # Check confidence threshold
        if "S_pmax" in row and row["S_pmax"] < self.config.min_confidence:
            return 0

        if self.config.mode == "x_hat_threshold":
            x_hat = row.get("x_hat", 0.0)
            if pd.isna(x_hat):
                return 0
            return 1 if x_hat > self.config.long_threshold else 0

        elif self.config.mode == "quantile_skew":
            q05 = row.get("q05", 0.0)
            q50 = row.get("q50", 0.0)
            q95 = row.get("q95", 0.0)
            if pd.isna(q05) or pd.isna(q50) or pd.isna(q95):
                return 0
            upside = q95 - q50
            downside = q50 - q05
            # Go long if more upside than downside
            if downside > 0 and upside > downside:
                return 1
            return 0

        return 0

    def generate(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Add 'signal' column to predictions DataFrame (vectorized).

        Args:
            predictions_df: DataFrame with forecast columns

        Returns:
            Copy of DataFrame with 'signal' column added (1=LONG, 0=FLAT)
        """
        df = predictions_df.copy()
        n = len(df)

        # Start with all signals as 0 (FLAT)
        signals = np.zeros(n, dtype=int)

        # Build mask of rows that pass the abstain filter
        if self.config.use_abstain and "abstain" in df.columns:
            abstain_mask = df["abstain"].fillna(False).astype(bool).values
        else:
            abstain_mask = np.zeros(n, dtype=bool)

        # Build mask of rows that pass the confidence filter
        if "S_pmax" in df.columns and self.config.min_confidence > 0:
            conf_mask = df["S_pmax"].fillna(0).values < self.config.min_confidence
        else:
            conf_mask = np.zeros(n, dtype=bool)

        # Combine filters: rows that are NOT filtered out
        valid_mask = ~abstain_mask & ~conf_mask

        if self.config.mode == "x_hat_threshold":
            # Get x_hat values, replacing NaN with threshold (will result in no signal)
            x_hat = df["x_hat"].fillna(self.config.long_threshold).values
            # Signal is 1 where x_hat > threshold AND row is valid
            signals = np.where(valid_mask & (x_hat > self.config.long_threshold), 1, 0)

        elif self.config.mode == "quantile_skew":
            q05 = df.get("q05", pd.Series(0.0, index=df.index)).fillna(0).values
            q50 = df.get("q50", pd.Series(0.0, index=df.index)).fillna(0).values
            q95 = df.get("q95", pd.Series(0.0, index=df.index)).fillna(0).values

            upside = q95 - q50
            downside = q50 - q05

            # Go long if more upside than downside (and downside > 0)
            skew_condition = (downside > 0) & (upside > downside)
            signals = np.where(valid_mask & skew_condition, 1, 0)

        df["signal"] = signals
        return df
