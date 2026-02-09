"""Unified multi-source cryptocurrency data fetcher.

Combines multiple data providers with automatic fallback and
best-source selection based on data availability and quality.
"""
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Type

import pandas as pd

from .base import DataProvider, ProviderConfig
from .coingecko import CoinGeckoProvider
from .coinmarketcap import CoinMarketCapProvider
from .yfinance import YFinanceProvider
from .binance_public import BinancePublicProvider


@dataclass
class UnifiedConfig:
    """Configuration for the unified data fetcher."""

    # API keys (can also be set via environment variables)
    coingecko_api_key: Optional[str] = None
    coinmarketcap_api_key: Optional[str] = None

    # Provider preferences (order of fallback)
    ohlcv_providers: List[str] = field(
        default_factory=lambda: ["binance_public", "yfinance", "coingecko", "coinmarketcap"]
    )
    trade_providers: List[str] = field(
        default_factory=lambda: ["binance_public"]
    )

    # General settings
    cache_dir: Optional[str] = None
    rate_limit_ms: int = 100
    timeout_s: int = 30

    @classmethod
    def from_env(cls) -> "UnifiedConfig":
        """Create config from environment variables.

        Environment variables:
        - COINGECKO_API_KEY
        - COINMARKETCAP_API_KEY
        - CRYPTO_API_KEY (generic fallback)
        - DATA_CACHE_DIR

        Returns:
            UnifiedConfig with values from environment
        """
        return cls(
            coingecko_api_key=os.getenv("COINGECKO_API_KEY"),
            coinmarketcap_api_key=os.getenv("COINMARKETCAP_API_KEY"),
            cache_dir=os.getenv("DATA_CACHE_DIR"),
        )


class UnifiedDataFetcher:
    """Unified interface to fetch cryptocurrency data from multiple sources.

    Automatically falls back between providers if one fails.
    Combines data from multiple sources when beneficial.

    Example:
        ```python
        fetcher = UnifiedDataFetcher()
        df = fetcher.fetch_ohlcv(['BTC', 'ETH'], days=30)
        ```

    Attributes:
        config: Unified configuration
        providers: Dictionary of initialized providers
    """

    # Registry of available providers
    PROVIDER_CLASSES: Dict[str, Type[DataProvider]] = {
        "coingecko": CoinGeckoProvider,
        "coinmarketcap": CoinMarketCapProvider,
        "yfinance": YFinanceProvider,
        "binance_public": BinancePublicProvider,
    }

    def __init__(
        self,
        config: Optional[UnifiedConfig] = None,
        coingecko_key: Optional[str] = None,
        coinmarketcap_key: Optional[str] = None,
    ):
        """Initialize the unified data fetcher.

        Args:
            config: Unified configuration (uses env vars if not provided)
            coingecko_key: Override CoinGecko API key
            coinmarketcap_key: Override CoinMarketCap API key
        """
        self.config = config or UnifiedConfig.from_env()

        # Override keys if provided directly
        if coingecko_key:
            self.config.coingecko_api_key = coingecko_key
        if coinmarketcap_key:
            self.config.coinmarketcap_api_key = coinmarketcap_key

        self.providers: Dict[str, DataProvider] = {}
        self._initialize_providers()

    def _initialize_providers(self) -> None:
        """Initialize all available providers."""
        provider_configs = {
            "coingecko": ProviderConfig(
                api_key=self.config.coingecko_api_key,
                rate_limit_ms=self.config.rate_limit_ms,
                timeout_s=self.config.timeout_s,
            ),
            "coinmarketcap": ProviderConfig(
                api_key=self.config.coinmarketcap_api_key,
                rate_limit_ms=self.config.rate_limit_ms,
                timeout_s=self.config.timeout_s,
            ),
            "yfinance": ProviderConfig(
                rate_limit_ms=self.config.rate_limit_ms,
                timeout_s=self.config.timeout_s,
            ),
            "binance_public": ProviderConfig(
                rate_limit_ms=self.config.rate_limit_ms,
                timeout_s=self.config.timeout_s,
            ),
        }

        for name, provider_class in self.PROVIDER_CLASSES.items():
            try:
                cfg = provider_configs.get(name, ProviderConfig())

                # Skip providers that require API keys we don't have
                if provider_class.requires_api_key and not cfg.api_key:
                    print(f"Skipping {name}: API key not configured")
                    continue

                self.providers[name] = provider_class(cfg)
                print(f"Initialized {name} provider")

            except Exception as e:
                print(f"Failed to initialize {name}: {e}")

    def fetch_ohlcv(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        days: int = 30,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        provider: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for multiple symbols.

        Tries providers in order until successful.

        Args:
            symbols: List of symbols (e.g., ['BTC', 'ETH', 'BTCUSDT'])
            timeframe: Candle interval
            days: Number of days of history
            start: Start datetime (alternative to days)
            end: End datetime
            provider: Force specific provider (skip fallback)

        Returns:
            Combined DataFrame with OHLCV data for all symbols
        """
        # Determine provider order
        if provider:
            providers_to_try = [provider] if provider in self.providers else []
        else:
            providers_to_try = [
                p for p in self.config.ohlcv_providers
                if p in self.providers
            ]

        if not providers_to_try:
            raise ValueError("No OHLCV providers available")

        all_data = []
        failed_symbols = list(symbols)

        for provider_name in providers_to_try:
            if not failed_symbols:
                break

            prov = self.providers[provider_name]
            symbols_to_fetch = failed_symbols.copy()
            failed_symbols = []

            for symbol in symbols_to_fetch:
                try:
                    print(f"Fetching {symbol} OHLCV from {provider_name}...")
                    df = prov.fetch_ohlcv(
                        symbol,
                        timeframe=timeframe,
                        days=days,
                        start=start,
                        end=end,
                    )

                    if not df.empty:
                        all_data.append(df)
                        print(f"  Got {len(df)} candles")
                    else:
                        failed_symbols.append(symbol)
                        print(f"  No data returned")

                except Exception as e:
                    failed_symbols.append(symbol)
                    print(f"  Error: {e}")

        if failed_symbols:
            print(f"Warning: Could not fetch data for: {failed_symbols}")

        if not all_data:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume", "symbol"]
            )

        return pd.concat(all_data, ignore_index=True).sort_values(
            ["symbol", "timestamp"]
        ).reset_index(drop=True)

    def fetch_trades(
        self,
        symbols: List[str],
        days: int = 1,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        provider: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch trade-level data for multiple symbols.

        Args:
            symbols: List of symbols
            days: Number of days (limited to 7 for most providers)
            start: Start datetime
            end: End datetime
            provider: Force specific provider

        Returns:
            Combined DataFrame with trade data
        """
        if provider:
            providers_to_try = [provider] if provider in self.providers else []
        else:
            providers_to_try = [
                p for p in self.config.trade_providers
                if p in self.providers
            ]

        if not providers_to_try:
            raise ValueError("No trade data providers available")

        all_data = []

        for provider_name in providers_to_try:
            prov = self.providers[provider_name]

            if not prov.supports_trades:
                continue

            for symbol in symbols:
                try:
                    print(f"Fetching {symbol} trades from {provider_name}...")
                    df = prov.fetch_trades(
                        symbol,
                        days=days,
                        start=start,
                        end=end,
                    )

                    if not df.empty:
                        all_data.append(df)
                        print(f"  Got {len(df)} trades")

                except Exception as e:
                    print(f"  Error: {e}")

            if all_data:
                break  # Got data from this provider

        if not all_data:
            return pd.DataFrame(
                columns=["timestamp", "symbol", "price", "quantity", "side", "trade_id"]
            )

        return pd.concat(all_data, ignore_index=True).sort_values(
            "timestamp"
        ).reset_index(drop=True)

    def fetch_prints(
        self,
        symbols: List[str],
        days: int = 7,
        trades_per_bar: int = 50,
    ) -> pd.DataFrame:
        """Fetch print data compatible with Gnosis loader.

        Tries to get real trade data first, falls back to
        generating prints from OHLCV data.

        Args:
            symbols: List of symbols
            days: Number of days of history
            trades_per_bar: Trades per OHLCV bar if generating

        Returns:
            DataFrame with print data for Gnosis
        """
        # Try real trade data first
        try:
            df = self.fetch_trades(symbols, days=days)
            if not df.empty and len(df) > 1000:
                print(f"Using real trade data: {len(df)} prints")
                return df
        except Exception as e:
            print(f"Could not fetch trades: {e}")

        # Fall back to generating from OHLCV
        print("Generating prints from OHLCV data...")
        ohlcv_df = self.fetch_ohlcv(symbols, timeframe="1m", days=days)

        if ohlcv_df.empty:
            raise ValueError(f"Could not fetch any data for: {symbols}")

        return self._generate_prints_from_ohlcv(ohlcv_df, trades_per_bar)

    def _generate_prints_from_ohlcv(
        self,
        ohlcv_df: pd.DataFrame,
        trades_per_bar: int = 50,
    ) -> pd.DataFrame:
        """Generate synthetic prints from OHLCV data.

        Args:
            ohlcv_df: OHLCV DataFrame
            trades_per_bar: Average trades per bar

        Returns:
            DataFrame with synthetic prints
        """
        import numpy as np

        rng = np.random.default_rng(seed=42)
        records = []

        for _, bar in ohlcv_df.iterrows():
            symbol = bar["symbol"]
            bar_start = bar["timestamp"]

            # Vary trades based on volume
            vol_factor = max(0.5, min(2.0, bar["volume"] / (ohlcv_df["volume"].mean() + 1e-10)))
            n_trades = max(10, int(trades_per_bar * vol_factor * rng.uniform(0.7, 1.3)))

            o, h, l, c = bar["open"], bar["high"], bar["low"], bar["close"]
            price_range = h - l if h > l else o * 0.001

            # Generate prices
            prices = np.zeros(n_trades)
            prices[0] = o
            prices[-1] = c
            for i in range(1, n_trades - 1):
                progress = i / (n_trades - 1)
                target = o + (c - o) * progress
                noise = rng.normal(0, price_range * 0.1)
                prices[i] = np.clip(target + noise, l, h)

            # Timestamps
            offsets_ms = np.sort(rng.integers(0, 60000, n_trades))
            timestamps = [bar_start + pd.Timedelta(milliseconds=int(ms)) for ms in offsets_ms]

            # Quantities
            total_vol = bar["volume"]
            qty_weights = rng.exponential(1.0, n_trades)
            qty_weights /= qty_weights.sum()
            quantities = qty_weights * total_vol

            for i in range(n_trades):
                side = "BUY" if (i == 0 and c > o) or (i > 0 and prices[i] > prices[i-1]) else "SELL"
                records.append({
                    "timestamp": timestamps[i],
                    "symbol": symbol,
                    "price": round(float(prices[i]), 8),
                    "quantity": round(float(quantities[i]), 8),
                    "side": side,
                    "trade_id": f"{symbol}_{bar_start.value}_{i}",
                })

        return pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)

    def get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current prices for multiple symbols.

        Args:
            symbols: List of symbols

        Returns:
            Dictionary mapping symbol to USD price
        """
        # Try providers in order
        for name in ["coingecko", "yfinance", "coinmarketcap"]:
            if name not in self.providers:
                continue

            try:
                prices = self.providers[name].fetch_current_price(symbols)
                if prices:
                    return prices
            except Exception:
                continue

        return {}

    def health_check(self) -> Dict[str, bool]:
        """Check health of all providers.

        Returns:
            Dictionary mapping provider name to health status
        """
        results = {}
        for name, prov in self.providers.items():
            results[name] = prov.health_check()
        return results

    def list_providers(self) -> List[str]:
        """Get list of available providers.

        Returns:
            List of provider names
        """
        return list(self.providers.keys())


def fetch_crypto_data(
    symbols: List[str],
    days: int = 30,
    timeframe: str = "1h",
    coingecko_key: Optional[str] = None,
    coinmarketcap_key: Optional[str] = None,
) -> pd.DataFrame:
    """Convenience function to fetch cryptocurrency OHLCV data.

    Uses the unified fetcher with automatic fallback.

    Args:
        symbols: List of symbols (e.g., ['BTC', 'ETH'])
        days: Number of days of history
        timeframe: Candle interval
        coingecko_key: Optional CoinGecko API key
        coinmarketcap_key: Optional CoinMarketCap API key

    Returns:
        DataFrame with OHLCV data
    """
    fetcher = UnifiedDataFetcher(
        coingecko_key=coingecko_key,
        coinmarketcap_key=coinmarketcap_key,
    )
    return fetcher.fetch_ohlcv(symbols, timeframe=timeframe, days=days)


def fetch_crypto_prints(
    symbols: List[str],
    days: int = 7,
    coingecko_key: Optional[str] = None,
    coinmarketcap_key: Optional[str] = None,
) -> pd.DataFrame:
    """Convenience function to fetch print data for Gnosis.

    Args:
        symbols: List of symbols
        days: Number of days
        coingecko_key: Optional CoinGecko API key
        coinmarketcap_key: Optional CoinMarketCap API key

    Returns:
        DataFrame with print data
    """
    fetcher = UnifiedDataFetcher(
        coingecko_key=coingecko_key,
        coinmarketcap_key=coinmarketcap_key,
    )
    return fetcher.fetch_prints(symbols, days=days)
