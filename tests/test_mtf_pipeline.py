import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gnosis.mtf.bars import Bar, BarManager
from gnosis.mtf.features import build_direction_labels, compute_feature_row
from gnosis.mtf.live import LiveMTFConfig, LiveMTFLoop


class StubProvider:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def get_closed_candles(self, symbol: str, timeframe: str, limit: int = 1) -> pd.DataFrame:
        self.calls.append((symbol, timeframe, limit))
        key = (symbol, timeframe)
        return self.responses.get(key, pd.DataFrame())


def test_no_lookahead_feature_guard():
    manager = BarManager(window_bars=2000, timeframes=["1h"])
    manager.update_bar(
        "1h",
        Bar(
            timestamp=pd.Timestamp("2024-01-01T10:00:00Z"),
            open=100,
            high=110,
            low=90,
            close=105,
            volume=1,
        ),
    )
    manager.update_bar(
        "1h",
        Bar(
            timestamp=pd.Timestamp("2024-01-01T11:00:00Z"),
            open=105,
            high=120,
            low=100,
            close=115,
            volume=1,
        ),
    )

    now_ts = pd.Timestamp("2024-01-01T10:30:00Z")
    try:
        compute_feature_row(manager, now_ts, primary_target_tf="1h", timeframes=["1h"])
        assert False, "Expected lookahead detection"
    except ValueError as exc:
        assert "Lookahead" in str(exc)


def test_1h_features_use_closed_bar():
    manager = BarManager(window_bars=2000, timeframes=["1h"])
    manager.update_bar(
        "1h",
        Bar(
            timestamp=pd.Timestamp("2024-01-01T10:00:00Z"),
            open=100,
            high=110,
            low=90,
            close=105,
            volume=1,
        ),
    )
    now_ts = pd.Timestamp("2024-01-01T10:30:00Z")
    features = compute_feature_row(manager, now_ts, primary_target_tf="1h", timeframes=["1h"])
    assert features["1h_close"] == 105


def test_label_alignment_next_bar_close():
    bars = [
        pd.Timestamp("2024-01-01T10:00:00Z"),
        pd.Timestamp("2024-01-01T11:00:00Z"),
        pd.Timestamp("2024-01-01T12:00:00Z"),
    ]
    closes = [100, 110, 90]
    labels = build_direction_labels(bars, closes)
    assert labels.loc[0, "label"] == 1
    assert labels.loc[1, "label"] == 0


def test_missing_bars_deterministic():
    provider = StubProvider({})
    config = LiveMTFConfig(symbols=["BTCUSDT"], timeframes=["1m", "1h"], heartbeat_seconds=1)
    loop = LiveMTFLoop(provider, config)
    now_ts = pd.Timestamp("2024-01-01T10:00:00Z")
    predictions = loop.run_once(now_ts)

    assert provider.calls
    features = predictions["BTCUSDT"]["features"]
    assert "minutes_to_1h_close" in features
    assert len(features) == 1
