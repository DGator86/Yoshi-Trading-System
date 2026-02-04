"""Time series cross-validation splits with purging and embargo.

Implements walk-forward validation with:
- Purge gap: removes samples that would leak information from train to test
- Embargo gap: additional buffer after purge
- Multiple split strategies: expanding, rolling, combinatorial
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Iterator, List, Tuple, Optional, Dict
from enum import Enum


class SplitStrategy(Enum):
    """Walk-forward split strategy."""
    EXPANDING = "expanding"  # Train window grows over time
    ROLLING = "rolling"      # Fixed train window slides
    COMBINATORIAL = "combinatorial"  # Combinatorial purged CV


@dataclass
class SplitConfig:
    """Configuration for time series splits."""

    # Number of folds
    n_splits: int = 5

    # Window sizes (in bars)
    train_size: int = 500
    val_size: int = 100
    test_size: int = 100

    # Purge and embargo (in bars)
    purge_gap: int = 10
    embargo_gap: int = 5

    # Strategy
    strategy: SplitStrategy = SplitStrategy.ROLLING

    # For expanding window
    min_train_size: int = 200

    # Whether to use validation set
    use_validation: bool = True


@dataclass
class Split:
    """Represents a single CV split."""
    fold_idx: int
    train_indices: np.ndarray
    val_indices: Optional[np.ndarray]
    test_indices: np.ndarray

    # Metadata
    train_start: int = 0
    train_end: int = 0
    val_start: Optional[int] = None
    val_end: Optional[int] = None
    test_start: int = 0
    test_end: int = 0


class PurgedKFold:
    """K-fold cross-validation with purging and embargo.

    Ensures no information leakage from future to past by:
    1. Purge: Remove samples from train that are within `purge_gap` bars
             of any test sample
    2. Embargo: After purging, add additional `embargo_gap` buffer

    This is essential for time series with autocorrelation.
    """

    def __init__(self, config: SplitConfig = None):
        self.config = config or SplitConfig()

    def split(
        self,
        df: pd.DataFrame,
        group_col: str = 'symbol'
    ) -> Iterator[Split]:
        """Generate purged k-fold splits.

        Args:
            df: DataFrame with time series data
            group_col: Column to group by (e.g., 'symbol')

        Yields:
            Split objects with train/val/test indices
        """
        n = len(df)
        indices = np.arange(n)

        if self.config.strategy == SplitStrategy.ROLLING:
            yield from self._rolling_splits(indices, n)
        elif self.config.strategy == SplitStrategy.EXPANDING:
            yield from self._expanding_splits(indices, n)
        else:
            yield from self._combinatorial_splits(indices, n)

    def _rolling_splits(
        self,
        indices: np.ndarray,
        n: int
    ) -> Iterator[Split]:
        """Generate rolling window splits."""
        train_size = self.config.train_size
        val_size = self.config.val_size if self.config.use_validation else 0
        test_size = self.config.test_size
        purge = self.config.purge_gap
        embargo = self.config.embargo_gap

        total_window = train_size + purge + val_size + embargo + test_size
        step = (n - total_window) // max(1, self.config.n_splits - 1)
        step = max(1, step)

        for fold_idx in range(self.config.n_splits):
            start = fold_idx * step

            train_start = start
            train_end = train_start + train_size

            if train_end >= n:
                break

            if self.config.use_validation:
                val_start = train_end + purge
                val_end = val_start + val_size
                test_start = val_end + embargo
            else:
                val_start = None
                val_end = None
                test_start = train_end + purge + embargo

            test_end = min(test_start + test_size, n)

            if test_end <= test_start:
                break

            # Purge train indices that are too close to test
            train_indices = self._purge_indices(
                indices[train_start:train_end],
                test_start, test_end,
                purge, embargo
            )

            yield Split(
                fold_idx=fold_idx,
                train_indices=train_indices,
                val_indices=indices[val_start:val_end] if val_start is not None else None,
                test_indices=indices[test_start:test_end],
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end,
            )

    def _expanding_splits(
        self,
        indices: np.ndarray,
        n: int
    ) -> Iterator[Split]:
        """Generate expanding window splits."""
        min_train = self.config.min_train_size
        val_size = self.config.val_size if self.config.use_validation else 0
        test_size = self.config.test_size
        purge = self.config.purge_gap
        embargo = self.config.embargo_gap

        # Calculate test positions
        available = n - min_train - purge - val_size - embargo - test_size
        step = available // max(1, self.config.n_splits)
        step = max(1, step)

        for fold_idx in range(self.config.n_splits):
            train_end = min_train + fold_idx * step
            train_start = 0  # Expanding window

            if self.config.use_validation:
                val_start = train_end + purge
                val_end = val_start + val_size
                test_start = val_end + embargo
            else:
                val_start = None
                val_end = None
                test_start = train_end + purge + embargo

            test_end = min(test_start + test_size, n)

            if test_end <= test_start:
                break

            train_indices = self._purge_indices(
                indices[train_start:train_end],
                test_start, test_end,
                purge, embargo
            )

            yield Split(
                fold_idx=fold_idx,
                train_indices=train_indices,
                val_indices=indices[val_start:val_end] if val_start is not None else None,
                test_indices=indices[test_start:test_end],
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end,
            )

    def _combinatorial_splits(
        self,
        indices: np.ndarray,
        n: int
    ) -> Iterator[Split]:
        """Generate combinatorial purged CV splits.

        Following Lopez de Prado's CPCV method.
        """
        # Divide data into groups
        n_groups = self.config.n_splits + 2
        group_size = n // n_groups
        groups = [
            indices[i * group_size: (i + 1) * group_size]
            for i in range(n_groups)
        ]

        purge = self.config.purge_gap
        embargo = self.config.embargo_gap

        for fold_idx in range(self.config.n_splits):
            # Test is 2 consecutive groups
            test_group_start = fold_idx
            test_indices = np.concatenate([
                groups[test_group_start],
                groups[test_group_start + 1] if test_group_start + 1 < n_groups else []
            ])

            if len(test_indices) == 0:
                continue

            test_start = test_indices[0]
            test_end = test_indices[-1] + 1

            # Train is all other groups, with purging
            train_groups = [
                g for i, g in enumerate(groups)
                if i not in (test_group_start, test_group_start + 1)
            ]

            if len(train_groups) == 0:
                continue

            all_train = np.concatenate(train_groups)
            train_indices = self._purge_indices(
                all_train, test_start, test_end, purge, embargo
            )

            yield Split(
                fold_idx=fold_idx,
                train_indices=train_indices,
                val_indices=None,
                test_indices=test_indices,
                train_start=int(train_indices[0]) if len(train_indices) > 0 else 0,
                train_end=int(train_indices[-1]) + 1 if len(train_indices) > 0 else 0,
                test_start=test_start,
                test_end=test_end,
            )

    def _purge_indices(
        self,
        train_indices: np.ndarray,
        test_start: int,
        test_end: int,
        purge: int,
        embargo: int
    ) -> np.ndarray:
        """Remove train indices that are too close to test set.

        Args:
            train_indices: Original train indices
            test_start: First index of test set
            test_end: Last index + 1 of test set
            purge: Number of bars to purge before test
            embargo: Additional buffer after purge

        Returns:
            Purged train indices
        """
        # Remove indices within purge distance of test start
        purge_start = test_start - purge - embargo

        mask = train_indices < purge_start
        return train_indices[mask]


class TimeSeriesSplitter:
    """High-level interface for time series splitting.

    Supports:
    - Multiple symbols with aligned splits
    - Horizon-aware purging
    - Walk-forward with train/val/test
    """

    def __init__(self, config: SplitConfig = None):
        self.config = config or SplitConfig()
        self._kfold = PurgedKFold(self.config)

    def split(
        self,
        df: pd.DataFrame,
        horizons: List[int] = None
    ) -> Iterator[Split]:
        """Generate time series splits.

        Args:
            df: DataFrame with time series data
            horizons: List of forecast horizons (used to set purge gap)

        Yields:
            Split objects
        """
        # Adjust purge gap based on max horizon
        if horizons is not None and len(horizons) > 0:
            max_horizon = max(horizons)
            self.config.purge_gap = max(self.config.purge_gap, max_horizon)

        yield from self._kfold.split(df)

    def get_train_val_test(
        self,
        df: pd.DataFrame,
        split: Split
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
        """Get train/val/test DataFrames for a split.

        Args:
            df: Full DataFrame
            split: Split object

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        train_df = df.iloc[split.train_indices].copy()
        val_df = df.iloc[split.val_indices].copy() if split.val_indices is not None else None
        test_df = df.iloc[split.test_indices].copy()

        return train_df, val_df, test_df


def create_purged_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    train_size: int = 500,
    test_size: int = 100,
    purge_gap: int = 10,
    embargo_gap: int = 5,
    strategy: str = 'rolling',
    horizons: List[int] = None
) -> List[Split]:
    """Convenience function to create purged CV splits.

    Args:
        df: Input DataFrame
        n_splits: Number of folds
        train_size: Training window size (bars)
        test_size: Test window size (bars)
        purge_gap: Purge gap (bars)
        embargo_gap: Embargo gap (bars)
        strategy: 'rolling', 'expanding', or 'combinatorial'
        horizons: Forecast horizons (used to adjust purge)

    Returns:
        List of Split objects
    """
    strategy_enum = SplitStrategy(strategy)

    config = SplitConfig(
        n_splits=n_splits,
        train_size=train_size,
        test_size=test_size,
        purge_gap=purge_gap,
        embargo_gap=embargo_gap,
        strategy=strategy_enum,
        use_validation=False,
    )

    # Adjust purge based on horizons
    if horizons is not None and len(horizons) > 0:
        config.purge_gap = max(config.purge_gap, max(horizons))

    splitter = TimeSeriesSplitter(config)
    return list(splitter.split(df, horizons))


def check_leakage(
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    horizon: int,
    timestamps: Optional[np.ndarray] = None
) -> Dict:
    """Check for potential information leakage between train and test.

    Args:
        train_indices: Training set indices
        test_indices: Test set indices
        horizon: Forecast horizon
        timestamps: Optional timestamps for time-based checks

    Returns:
        Dictionary with leakage check results
    """
    train_max = np.max(train_indices)
    test_min = np.min(test_indices)

    gap = test_min - train_max - 1

    has_leakage = gap < horizon
    overlap = np.intersect1d(train_indices, test_indices)

    return {
        'has_leakage': has_leakage,
        'gap_bars': int(gap),
        'required_gap': horizon,
        'has_overlap': len(overlap) > 0,
        'overlap_count': len(overlap),
        'train_max_idx': int(train_max),
        'test_min_idx': int(test_min),
    }
