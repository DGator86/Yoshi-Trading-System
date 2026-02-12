"""Regression tests for ingest provider edge cases."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.gnosis.ingest.providers.binance_public import _epoch_to_datetime_utc  # noqa: E402
from src.gnosis.ingest.providers.coingecko import CoinGeckoProvider  # noqa: E402


def test_binance_epoch_parser_handles_microseconds():
    # 2026-02-11 00:00:00 UTC in microseconds.
    micro_ts = pd.Series([1770768000000000, 1770771600000000])
    dt = _epoch_to_datetime_utc(micro_ts)
    assert str(dt.iloc[0].year) == "2026"
    assert dt.iloc[0].month == 2
    assert dt.iloc[0].day == 11


def test_coingecko_ohlc_days_coercion():
    assert CoinGeckoProvider._coerce_ohlc_days(1) == 1
    assert CoinGeckoProvider._coerce_ohlc_days(8) == 14
    assert CoinGeckoProvider._coerce_ohlc_days(120) == 180
    assert CoinGeckoProvider._coerce_ohlc_days(1000) == 365
