"""Base class for cryptocurrency data providers."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class ProviderConfig:
    """Configuration for a data provider."""

    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    base_url: Optional[str] = None
    rate_limit_ms: int = 100
    timeout_s: int = 30
    max_retries: int = 3
    extra: Dict = field(default_factory=dict)


class DataProvider(ABC):
    """Abstract base class for cryptocurrency data providers.

    All providers must implement methods to fetch OHLCV and/or trade data
    with a consistent interface.
    """

    name: str = "base"
    supports_ohlcv: bool = True
    supports_trades: bool = False
    requires_api_key: bool = False

    def __init__(self, config: Optional[ProviderConfig] = None):
        """Initialize provider with optional configuration.

        Args:
            config: Provider-specific configuration
        """
        self.config = config or ProviderConfig()
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration for this provider."""
        if self.requires_api_key and not self.config.api_key:
            raise ValueError(
                f"{self.name} provider requires an API key. "
                f"Set it in the config or environment variable."
            )

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        days: Optional[int] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV (candlestick) data for a symbol.

        Args:
            symbol: Trading pair (e.g., 'BTC', 'BTCUSDT', 'bitcoin')
            timeframe: Candle interval ('1m', '5m', '1h', '1d', etc.)
            days: Number of historical days to fetch
            start: Start datetime (alternative to days)
            end: End datetime (defaults to now)

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume, symbol
        """
        pass

    def fetch_trades(
        self,
        symbol: str,
        days: Optional[int] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch individual trade data for a symbol.

        Args:
            symbol: Trading pair
            days: Number of historical days
            start: Start datetime
            end: End datetime

        Returns:
            DataFrame with columns: timestamp, symbol, price, quantity, side, trade_id
        """
        raise NotImplementedError(
            f"{self.name} provider does not support trade-level data"
        )

    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols from this provider.

        Returns:
            List of symbol strings
        """
        raise NotImplementedError(
            f"{self.name} provider does not implement symbol listing"
        )

    def health_check(self) -> bool:
        """Check if the provider API is accessible.

        Returns:
            True if provider is healthy, False otherwise
        """
        try:
            # Try to fetch a small amount of data
            df = self.fetch_ohlcv("BTC", timeframe="1d", days=1)
            return len(df) > 0
        except Exception:
            return False

    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        """Normalize symbol to a standard format.

        Converts various formats to uppercase base asset:
        - 'bitcoin' -> 'BTC'
        - 'BTC/USDT' -> 'BTC'
        - 'BTCUSDT' -> 'BTC'

        Args:
            symbol: Input symbol in any format

        Returns:
            Normalized symbol string
        """
        symbol = symbol.upper()

        # Handle common name mappings
        name_map = {
            "BITCOIN": "BTC",
            "ETHEREUM": "ETH",
            "SOLANA": "SOL",
            "RIPPLE": "XRP",
            "CARDANO": "ADA",
            "DOGECOIN": "DOGE",
            "POLKADOT": "DOT",
            "POLYGON": "MATIC",
            "CHAINLINK": "LINK",
            "AVALANCHE": "AVAX",
            "LITECOIN": "LTC",
        }

        if symbol in name_map:
            return name_map[symbol]

        # Handle pair formats
        if "/" in symbol:
            return symbol.split("/")[0]

        # Handle concatenated pairs (order matters - check longer suffixes first)
        for quote in ["USDT", "BUSD", "USDC", "USD", "EUR", "BTC", "ETH"]:
            if symbol.endswith(quote) and len(symbol) > len(quote):
                return symbol[:-len(quote)]

        return symbol
