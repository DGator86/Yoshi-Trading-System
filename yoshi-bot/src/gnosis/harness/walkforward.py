"""Walk-forward validation harness with purge/embargo."""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Iterator


@dataclass
class Fold:
    """Represents a single walk-forward fold."""
    fold_idx: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int


class WalkForwardHarness:
    """Nested walk-forward with purge and embargo."""

    def __init__(
        self,
        config: dict | None = None,
        *,
        # Back-compat: bar-based constructor used by some tests/integrations.
        n_folds: int | None = None,
        train_size: int | None = None,
        test_size: int | None = None,
        val_size: int | None = None,
        purge_bars: int | None = None,
        embargo_bars: int | None = None,
    ):
        cfg = config or {}

        self._mode = "days"

        # If caller passes bar-based params, switch to bar-mode folds.
        if n_folds is not None or train_size is not None or test_size is not None:
            self._mode = "bars"
            self.outer_folds = int(n_folds or cfg.get("outer_folds", 8))
            self.train_bars = int(train_size or cfg.get("train_bars", 300))
            self.val_bars = int(val_size or cfg.get("val_bars", 0))
            self.test_bars = int(test_size or cfg.get("test_bars", 100))
            self.purge_bars = int(purge_bars or cfg.get("purge_bars", 0))
            # Default embargo to 0 unless explicitly provided.
            self.embargo_bars = int(embargo_bars) if embargo_bars is not None else int(cfg.get("embargo_bars", 0))
            return

        # Default: day-ratio mode (existing behavior).
        self.outer_folds = cfg.get("outer_folds", 8)
        self.train_days = cfg.get("train_days", 180)
        self.val_days = cfg.get("val_days", 30)
        self.test_days = cfg.get("test_days", 30)
        self.purge_trades = cfg.get("purge_trades", "HORIZON")
        self.embargo_trades = cfg.get("embargo_trades", "HORIZON")

    def generate_folds(self, df: pd.DataFrame, horizon_trades: int = 2000) -> Iterator[Fold]:
        """Generate walk-forward folds with purge/embargo.

        Uses proportional allocation based on configured day ratios.
        """
        n_bars = len(df)
        if n_bars < 50:
            return  # Not enough data for any folds

        if self._mode == "bars":
            # Rolling window walk-forward on bar counts.
            step = max(1, int(self.test_bars))
            for i in range(int(self.outer_folds)):
                train_start = i * step
                train_end = train_start + int(self.train_bars)

                val_start = train_end + int(self.purge_bars)
                val_end = val_start + int(self.val_bars)

                test_start = val_end + int(self.embargo_bars)
                test_end = min(test_start + int(self.test_bars), n_bars)

                if train_end > n_bars or test_end <= test_start:
                    break

                yield Fold(
                    fold_idx=i,
                    train_start=train_start,
                    train_end=train_end,
                    val_start=val_start,
                    val_end=val_end,
                    test_start=test_start,
                    test_end=test_end,
                )
            return

        total_days = self.train_days + self.val_days + self.test_days

        # Calculate bar proportions based on day ratios
        train_ratio = self.train_days / total_days
        val_ratio = self.val_days / total_days
        test_ratio = self.test_days / total_days

        # Purge/embargo in bars (based on horizon)
        purge_bars = max(1, horizon_trades // 200)  # D0 has 200 trades per bar
        embargo_bars = purge_bars

        # Total bars needed for one fold (train + val + test + gaps)
        gap_bars = purge_bars + embargo_bars

        # Reserve space for gaps and compute usable bars per fold
        # We need: outer_folds * (train + val + test) + gaps between them
        bars_per_fold = (n_bars - gap_bars * self.outer_folds) // self.outer_folds
        if bars_per_fold < 30:
            # Fall back to simpler approach: use all data with proportional splits
            bars_per_fold = n_bars // max(2, self.outer_folds)

        train_bars = max(10, int(bars_per_fold * train_ratio))
        val_bars = max(5, int(bars_per_fold * val_ratio))
        test_bars = max(5, int(bars_per_fold * test_ratio))

        # Step size between folds
        step = (n_bars - train_bars - val_bars - test_bars - gap_bars) // max(1, self.outer_folds - 1)
        step = max(1, step)

        for i in range(self.outer_folds):
            train_start = i * step
            train_end = train_start + train_bars

            # Purge gap between train and val
            val_start = train_end + purge_bars
            val_end = val_start + val_bars

            # Embargo gap between val and test
            test_start = val_end + embargo_bars
            test_end = min(test_start + test_bars, n_bars)

            if test_end <= test_start or val_end > n_bars:
                break

            yield Fold(
                fold_idx=i,
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end,
            )


def compute_future_returns(df: pd.DataFrame, horizon_bars: int = 10) -> pd.DataFrame:
    """Compute future returns for prediction targets."""
    result = df.copy()
    result["future_return"] = result.groupby("symbol")["close"].transform(
        lambda x: x.shift(-horizon_bars) / x - 1
    )
    return result
