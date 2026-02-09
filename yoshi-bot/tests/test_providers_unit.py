"""Unit tests for multi-source data providers (no network required)."""
import pytest
import pandas as pd
from datetime import datetime, timezone

from src.gnosis.ingest.providers.base import DataProvider, ProviderConfig


class TestProviderConfig:
    """Tests for ProviderConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ProviderConfig()
        assert config.api_key is None
        assert config.api_secret is None
        assert config.rate_limit_ms == 100
        assert config.timeout_s == 30
        assert config.max_retries == 3

    def test_custom_config(self):
        """Test custom configuration."""
        config = ProviderConfig(
            api_key="test-key",
            api_secret="test-secret",
            rate_limit_ms=200,
            timeout_s=60,
            max_retries=5,
            extra={"sandbox": True},
        )
        assert config.api_key == "test-key"
        assert config.api_secret == "test-secret"
        assert config.rate_limit_ms == 200
        assert config.timeout_s == 60
        assert config.max_retries == 5
        assert config.extra["sandbox"] is True


class TestDataProviderBase:
    """Tests for DataProvider base class."""

    def test_normalize_symbol_uppercase(self):
        """Test symbol normalization for uppercase input."""
        assert DataProvider.normalize_symbol("BTC") == "BTC"
        assert DataProvider.normalize_symbol("eth") == "ETH"
        assert DataProvider.normalize_symbol("Sol") == "SOL"

    def test_normalize_symbol_name_mapping(self):
        """Test symbol normalization for common names."""
        assert DataProvider.normalize_symbol("bitcoin") == "BTC"
        assert DataProvider.normalize_symbol("ethereum") == "ETH"
        assert DataProvider.normalize_symbol("SOLANA") == "SOL"
        assert DataProvider.normalize_symbol("Dogecoin") == "DOGE"

    def test_normalize_symbol_pair_format(self):
        """Test symbol normalization for pair formats."""
        assert DataProvider.normalize_symbol("BTC/USDT") == "BTC"
        assert DataProvider.normalize_symbol("ETH/USD") == "ETH"
        assert DataProvider.normalize_symbol("SOL/BTC") == "SOL"

    def test_normalize_symbol_concatenated(self):
        """Test symbol normalization for concatenated pairs."""
        assert DataProvider.normalize_symbol("BTCUSDT") == "BTC"
        assert DataProvider.normalize_symbol("ETHUSDC") == "ETH"
        assert DataProvider.normalize_symbol("SOLBUSD") == "SOL"
        assert DataProvider.normalize_symbol("ADAEUR") == "ADA"


class TestCoinGeckoProvider:
    """Tests for CoinGecko provider."""

    def test_import(self):
        """Test provider can be imported."""
        from src.gnosis.ingest.providers import CoinGeckoProvider
        assert CoinGeckoProvider is not None

    def test_init_free_tier(self):
        """Test initialization without API key."""
        from src.gnosis.ingest.providers import CoinGeckoProvider
        provider = CoinGeckoProvider()
        assert provider.name == "coingecko"
        assert provider.base_url == "https://api.coingecko.com/api/v3"

    def test_init_pro_tier(self):
        """Test initialization with pro API key."""
        from src.gnosis.ingest.providers import CoinGeckoProvider, ProviderConfig
        config = ProviderConfig(api_key="CG-test-pro-key")
        provider = CoinGeckoProvider(config)
        assert provider.base_url == "https://pro-api.coingecko.com/api/v3"
        assert provider.header_key == "x-cg-pro-api-key"

    def test_symbol_to_id_mapping(self):
        """Test symbol to CoinGecko ID mapping."""
        from src.gnosis.ingest.providers import CoinGeckoProvider
        provider = CoinGeckoProvider()
        assert provider._get_coin_id("BTC") == "bitcoin"
        assert provider._get_coin_id("ETH") == "ethereum"
        assert provider._get_coin_id("SOL") == "solana"


class TestCoinMarketCapProvider:
    """Tests for CoinMarketCap provider."""

    def test_import(self):
        """Test provider can be imported."""
        from src.gnosis.ingest.providers import CoinMarketCapProvider
        assert CoinMarketCapProvider is not None

    def test_requires_api_key(self):
        """Test that initialization requires API key."""
        from src.gnosis.ingest.providers import CoinMarketCapProvider
        with pytest.raises(ValueError, match="requires an API key"):
            CoinMarketCapProvider()

    def test_init_with_key(self):
        """Test initialization with API key."""
        from src.gnosis.ingest.providers import CoinMarketCapProvider, ProviderConfig
        config = ProviderConfig(api_key="test-key")
        provider = CoinMarketCapProvider(config)
        assert provider.name == "coinmarketcap"

    def test_symbol_to_id_mapping(self):
        """Test symbol to CMC ID mapping."""
        from src.gnosis.ingest.providers import CoinMarketCapProvider, ProviderConfig
        config = ProviderConfig(api_key="test-key")
        provider = CoinMarketCapProvider(config)
        assert provider._get_cmc_id("BTC") == 1
        assert provider._get_cmc_id("ETH") == 1027
        assert provider._get_cmc_id("SOL") == 5426


class TestYFinanceProvider:
    """Tests for YFinance provider."""

    def test_import(self):
        """Test provider can be imported."""
        from src.gnosis.ingest.providers import YFinanceProvider
        assert YFinanceProvider is not None

    def test_no_api_key_required(self):
        """Test that no API key is required."""
        from src.gnosis.ingest.providers import YFinanceProvider
        # Should not raise even without yfinance installed
        # (lazy import handles missing module)
        provider = YFinanceProvider()
        assert provider.requires_api_key is False

    def test_symbol_mapping(self):
        """Test symbol to yfinance format mapping."""
        from src.gnosis.ingest.providers import YFinanceProvider
        provider = YFinanceProvider()
        assert provider._get_yf_symbol("BTC") == "BTC-USD"
        assert provider._get_yf_symbol("ETH") == "ETH-USD"
        assert provider._get_yf_symbol("BTCUSDT") == "BTC-USD"

    def test_interval_normalization(self):
        """Test timeframe to yfinance interval conversion."""
        from src.gnosis.ingest.providers import YFinanceProvider
        provider = YFinanceProvider()
        assert provider._normalize_interval("1h") == "1h"
        assert provider._normalize_interval("daily") == "1d"
        assert provider._normalize_interval("hourly") == "1h"


class TestBinancePublicProvider:
    """Tests for Binance public data provider."""

    def test_import(self):
        """Test provider can be imported."""
        from src.gnosis.ingest.providers import BinancePublicProvider
        assert BinancePublicProvider is not None

    def test_init(self):
        """Test basic initialization."""
        from src.gnosis.ingest.providers import BinancePublicProvider
        provider = BinancePublicProvider()
        assert provider.name == "binance_public"
        assert provider.supports_ohlcv is True
        assert provider.supports_trades is True
        assert provider.requires_api_key is False

    def test_valid_intervals(self):
        """Test that all standard intervals are supported."""
        from src.gnosis.ingest.providers import BinancePublicProvider
        expected = ["1s", "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]
        assert BinancePublicProvider.KLINE_INTERVALS == expected


class TestUnifiedDataFetcher:
    """Tests for unified data fetcher."""

    def test_import(self):
        """Test fetcher can be imported."""
        from src.gnosis.ingest.providers import UnifiedDataFetcher
        assert UnifiedDataFetcher is not None

    def test_init_with_keys(self):
        """Test initialization with API keys."""
        from src.gnosis.ingest.providers import UnifiedDataFetcher
        fetcher = UnifiedDataFetcher(
            coingecko_key="test-cg-key",
            coinmarketcap_key="test-cmc-key",
        )
        # Should have multiple providers
        assert len(fetcher.list_providers()) >= 1

    def test_list_providers(self):
        """Test listing available providers."""
        from src.gnosis.ingest.providers import UnifiedDataFetcher
        fetcher = UnifiedDataFetcher(coinmarketcap_key="test-key")
        providers = fetcher.list_providers()
        assert isinstance(providers, list)
        # At minimum, should have providers that don't require network
        assert "binance_public" in providers or "coingecko" in providers

    def test_generate_prints_from_ohlcv(self):
        """Test generating prints from OHLCV data."""
        from src.gnosis.ingest.providers import UnifiedDataFetcher
        fetcher = UnifiedDataFetcher(coinmarketcap_key="test-key")

        # Create sample OHLCV data
        ohlcv_df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="1min", tz="UTC"),
            "open": [100.0] * 10,
            "high": [102.0] * 10,
            "low": [98.0] * 10,
            "close": [101.0] * 10,
            "volume": [1000.0] * 10,
            "symbol": ["BTCUSDT"] * 10,
        })

        prints_df = fetcher._generate_prints_from_ohlcv(ohlcv_df, trades_per_bar=10)

        assert not prints_df.empty
        assert "timestamp" in prints_df.columns
        assert "symbol" in prints_df.columns
        assert "price" in prints_df.columns
        assert "quantity" in prints_df.columns
        assert "side" in prints_df.columns
        assert "trade_id" in prints_df.columns
        # Should have roughly 10 bars * 10 trades = ~100 prints
        assert len(prints_df) >= 50


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_fetch_crypto_data_import(self):
        """Test convenience function can be imported."""
        from src.gnosis.ingest.providers.unified import fetch_crypto_data
        assert callable(fetch_crypto_data)

    def test_fetch_crypto_prints_import(self):
        """Test convenience function can be imported."""
        from src.gnosis.ingest.providers.unified import fetch_crypto_prints
        assert callable(fetch_crypto_prints)


class TestIngestExports:
    """Tests for ingest package exports."""

    def test_all_providers_exported(self):
        """Test that all providers are exported from ingest package."""
        from src.gnosis.ingest import (
            DataProvider,
            ProviderConfig,
            CoinGeckoProvider,
            CoinMarketCapProvider,
            YFinanceProvider,
            BinancePublicProvider,
            UnifiedDataFetcher,
        )
        assert DataProvider is not None
        assert ProviderConfig is not None
        assert CoinGeckoProvider is not None
        assert CoinMarketCapProvider is not None
        assert YFinanceProvider is not None
        assert BinancePublicProvider is not None
        assert UnifiedDataFetcher is not None

    def test_convenience_functions_exported(self):
        """Test convenience functions are exported."""
        from src.gnosis.ingest import fetch_crypto_data, fetch_crypto_prints
        assert callable(fetch_crypto_data)
        assert callable(fetch_crypto_prints)
