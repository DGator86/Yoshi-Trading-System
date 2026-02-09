from __future__ import annotations

from typing import List

import numpy as np

from gnosis.prediction_test_battery.context import WalkForwardSplit


def build_walk_forward_splits(
    n_samples: int,
    train_size: int,
    test_size: int,
    step_size: int,
    embargo: int = 0,
) -> List[WalkForwardSplit]:
    splits: List[WalkForwardSplit] = []
    start = 0
    while start + train_size + test_size <= n_samples:
        train_idx = np.arange(start, start + train_size)
        test_start = start + train_size + embargo
        test_idx = np.arange(test_start, test_start + test_size)
        splits.append(WalkForwardSplit(train_idx=train_idx, test_idx=test_idx, embargo=embargo))
        start += step_size
    return splits


def check_split_integrity(splits: List[WalkForwardSplit]) -> List[str]:
    issues: List[str] = []
    for idx, split in enumerate(splits):
        overlap = np.intersect1d(split.train_idx, split.test_idx)
        if overlap.size > 0:
            issues.append(f"Split {idx} has {overlap.size} overlapping samples")
        if split.test_idx.min(initial=0) <= split.train_idx.max(initial=-1):
            min_gap = split.test_idx.min(initial=0) - split.train_idx.max(initial=-1) - 1
            if min_gap < split.embargo:
                issues.append(f"Split {idx} embargo too small: {min_gap} < {split.embargo}")
    return issues
