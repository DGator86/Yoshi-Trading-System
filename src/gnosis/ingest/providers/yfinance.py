"""Yahoo Finance data provider using yfinance library.

yfinance provides free access to historical market data including
cryptocurrencies traded on major exchanges.

Repository: https://github.com/ranaroussi/yfinance
"""
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd

from .base import DataProvider, ProviderConfig


class YFinanceProvider(DataProvider):
    """Yahoo Finance data provider via yfinance library.

    Provides free access to historical OHLCV data for:
    - Cryptocurrencies (BTC-USD, ETH-USD, etc.)
    - Stocks, ETFs, indices, and more

    No API key required.

    Attributes:
        name: Provider identifier
        supports_ohlcv: Whether OHLCV data is available
        requires_api_key: No API key needed
    """

    name = "yfinance"
    supports_ohlcv = True
    supports_trades = False
    requires_api_key = False

    # yfinance crypto symbols use -USD suffix
    CRYPTO_SYMBOLS = {
        "BTC": "BTC-USD",
        "ETH": "ETH-USD",
        "SOL": "SOL-USD",
        "XRP": "XRP-USD",
        "ADA": "ADA-USD",
        "DOGE": "DOGE-USD",
        "DOT": "DOT-USD",
        "MATIC": "MATIC-USD",
        "LINK": "LINK-USD",
        "AVAX": "AVAX-USD",
        "LTC": "LTC-USD",
        "BNB": "BNB-USD",
        "SHIB": "SHIB-USD",
        "TRX": "TRX-USD",
        "ATOM": "ATOM-USD",
        "UNI": "UNI-USD",
        "XLM": "XLM-USD",
        "BCH": "BCH-USD",
        "FIL": "FIL-USD",
        "APT": "APT-USD",
        "ARB": "ARB-USD",
        "OP": "OP-USD",
        "NEAR": "NEAR-USD",
        "ICP": "ICP-USD",
    }

    # Valid intervals for yfinance
    VALID_INTERVALS = [
        "1m", "2m", "5m", "15m", "30m", "60m", "90m",
        "1h", "1d", "5d", "1wk", "1mo", "3mo"
    ]

    def __init__(self, config: Optional[ProviderConfig] = None):
        """Initialize yfinance provider.

        Args:
            config: Provider configuration (API key not needed)
        """
        super().__init__(config)
        self._yf = None

    def _get_yfinance(self):
        """Lazy load yfinance to avoid import errors if not installed."""
        if self._yf is None:
            try:
                import yfinance as yf
                self._yf = yf
            except ImportError:
                raise ImportError(
                    "yfinance is required for this provider. "
                    "Install it with: pip install yfinance"
                )
        return self._yf

    def _get_yf_symbol(self, symbol: str) -> str:
        """Convert symbol to yfinance format.

        Args:
            symbol: Symbol like 'BTC', 'BTCUSDT', 'bitcoin'

        Returns:
            yfinance symbol like 'BTC-USD'
        """
        normalized = self.normalize_symbol(symbol)

        # Check our mapping
        if normalized in self.CRYPTO_SYMBOLS:
            return self.CRYPTO_SYMBOLS[normalized]

        # Default: assume crypto with -USD suffix
        return f"{normalized}-USD"

    def _normalize_interval(self, timeframe: str) -> str:
        """Convert timeframe to yfinance interval format.

        Args:
            timeframe: Interval like '1h', '1d', 'hourly'

        Returns:
            yfinance interval string
        """
        # Direct mapping
        if timeframe in self.VALID_INTERVALS:
            return timeframe

        # Common aliases
        aliases = {
            "hourly": "1h",
            "daily": "1d",
            "weekly": "1wk",
            "monthly": "1mo",
            "4h": "1h",  # yfinance doesn't support 4h, use 1h
        }

        return aliases.get(timeframe, "1d")

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        days: Optional[int] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance.

        Args:
            symbol: Symbol like 'BTC', 'ETH', 'BTCUSDT'
            timeframe: Interval ('1m', '5m', '1h', '1d', etc.)
            days: Number of days of history
            start: Start datetime
            end: End datetime

        Returns:
            DataFrame with OHLCV data

        Note:
            - 1m data: max 7 days
            - 2m-90m data: max 60 days
            - 1h data: max 730 days
            - 1d+ data: no limit
        """
        yf = self._get_yfinance()
        yf_symbol = self._get_yf_symbol(symbol)
        interval = self._normalize_interval(timeframe)
        output_symbol = self.normalize_symbol(symbol) + "USDT"

        # Calculate period
        end_dt = end or datetime.now(timezone.utc)
        if days is not None:
            start_dt = end_dt - timedelta(days=days)
        elif start is not None:
            start_dt = start
        else:
            start_dt = end_dt - timedelta(days=30)

        # yfinance has limits on intraday data
        max_days = self._get_max_days_for_interval(interval)
        actual_days = (end_dt - start_dt).days
        if actual_days > max_days:
            print(
                f"yfinance {interval} interval limited to {max_days} days, "
                f"truncating from {actual_days}"
            )
            start_dt = end_dt - timedelta(days=max_days)

        try:
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(
                start=start_dt.strftime("%Y-%m-%d"),
                end=end_dt.strftime("%Y-%m-%d"),
                interval=interval,
            )

            if df.empty:
                return pd.DataFrame(
                    columns=["timestamp", "open", "high", "low", "close", "volume", "symbol"]
                )

            # Reset index and rename columns
            df = df.reset_index()

            # Handle both 'Date' and 'Datetime' index names
            date_col = "Date" if "Date" in df.columns else "Datetime"
            if date_col in df.columns:
                df = df.rename(columns={date_col: "timestamp"})

            # Ensure timezone-aware
            if df["timestamp"].dt.tz is None:
                df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
            else:
                df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")

            # Standardize column names
            df = df.rename(columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            })

            # Select and order columns
            df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
            df["symbol"] = output_symbol

            return df.sort_values("timestamp").reset_index(drop=True)

        except Exception as e:
            print(f"yfinance error for {yf_symbol}: {e}")
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume", "symbol"]
            )

    def _get_max_days_for_interval(self, interval: str) -> int:
        """Get maximum days of data available for an interval.

        Args:
            interval: yfinance interval

        Returns:
            Maximum number of days
        """
        limits = {
            "1m": 7,
            "2m": 60,
            "5m": 60,
            "15m": 60,
            "30m": 60,
            "60m": 730,
            "90m": 60,
            "1h": 730,
            "1d": 10000,
            "5d": 10000,
            "1wk": 10000,
            "1mo": 10000,
            "3mo": 10000,
        }
        return limits.get(interval, 30)

    def get_available_symbols(self) -> List[str]:
        """Get list of known crypto symbols.

        Note: yfinance supports many more symbols including stocks.
        This returns only the crypto symbols we've mapped.

        Returns:
            List of crypto symbol strings
        """
        return list(self.CRYPTO_SYMBOLS.keys())

    def fetch_current_price(self, symbols: List[str]) -> Dict[str, float]:
        """Fetch current prices for multiple symbols.

        Args:
            symbols: List of symbols

        Returns:
            Dictionary mapping symbol to USD price
        """
        yf = self._get_yfinance()
        result = {}

        for symbol in symbols:
            try:
                yf_symbol = self._get_yf_symbol(symbol)
                ticker = yf.Ticker(yf_symbol)
                info = ticker.info

                # Try different price fields
                price = (
                    info.get("regularMarketPrice")
                    or info.get("currentPrice")
                    or info.get("previousClose")
                    or 0
                )
                result[self.normalize_symbol(symbol)] = float(price)

            except Exception:
                result[self.normalize_symbol(symbol)] = 0

        return result

    def fetch_info(self, symbol: str) -> Dict:
        """Fetch detailed info for a symbol.

        Args:
            symbol: Symbol to look up

        Returns:
            Dictionary with ticker info
        """
        yf = self._get_yfinance()
        try:
            yf_symbol = self._get_yf_symbol(symbol)
            ticker = yf.Ticker(yf_symbol)
            return ticker.info
        except Exception:
            return {}
