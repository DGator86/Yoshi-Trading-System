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
from src.gnosis.ingest.providers.unified import UnifiedConfig  # noqa: E402


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


def test_unified_config_accepts_common_key_aliases(monkeypatch):
    monkeypatch.delenv("COINGECKO_API_KEY", raising=False)
    monkeypatch.delenv("COINMARKETCAP_API_KEY", raising=False)
    monkeypatch.delenv("COINAPI_API_KEY", raising=False)
    monkeypatch.setenv("COINGECKO_KEY", "cg_demo")
    monkeypatch.setenv("CMC_API_KEY", "cmc_demo")
    monkeypatch.setenv("COINAPI_KEY", "coinapi_demo")

    cfg = UnifiedConfig.from_env()
    assert cfg.coingecko_api_key == "cg_demo"
    assert cfg.coinmarketcap_api_key == "cmc_demo"
    assert cfg.coinapi_api_key == "coinapi_demo"


def test_unified_config_prefers_coinapi_when_key_present(monkeypatch):
    monkeypatch.setenv("COINAPI_API_KEY", "coinapi_live")
    monkeypatch.delenv("OHLCV_PROVIDERS", raising=False)
    monkeypatch.delenv("OHLCV_PROVIDER_ORDER", raising=False)

    cfg = UnifiedConfig.from_env()
    assert cfg.ohlcv_providers[0] == "coinapi"


def test_unified_config_allows_timeout_retry_and_provider_env(monkeypatch):
    monkeypatch.setenv("OHLCV_PROVIDERS", "coingecko,binance_public")
    monkeypatch.setenv("DATA_TIMEOUT_S", "9")
    monkeypatch.setenv("DATA_MAX_RETRIES", "1")

    cfg = UnifiedConfig.from_env()
    assert cfg.ohlcv_providers == ["coingecko", "binance_public"]
    assert cfg.timeout_s == 9
    assert cfg.max_retries == 1
