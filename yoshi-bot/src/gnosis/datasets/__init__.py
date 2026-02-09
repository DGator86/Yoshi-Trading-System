"""Datasets module for training data preparation.

This module provides:
- Multi-horizon label generation
- Purged/embargoed CV splits
- Multiple bar type aggregation
- Feature engineering
"""

from .labels import (
    LabelConfig,
    LabelGenerator,
    generate_labels,
)

from .splits import (
    SplitConfig,
    SplitStrategy,
    Split,
    PurgedKFold,
    TimeSeriesSplitter,
    create_purged_splits,
    check_leakage,
)

from .bars import (
    BarType,
    BarConfig,
    BarAggregator,
    compute_bar_features,
    create_bars,
)

from .features import (
    FeatureConfig,
    FeatureEngineer,
    prepare_training_data,
)

__all__ = [
    # Labels
    'LabelConfig',
    'LabelGenerator',
    'generate_labels',
    # Splits
    'SplitConfig',
    'SplitStrategy',
    'Split',
    'PurgedKFold',
    'TimeSeriesSplitter',
    'create_purged_splits',
    'check_leakage',
    # Bars
    'BarType',
    'BarConfig',
    'BarAggregator',
    'compute_bar_features',
    'create_bars',
    # Features
    'FeatureConfig',
    'FeatureEngineer',
    'prepare_training_data',
]
