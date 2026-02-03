"""CoinGecko API data provider.

CoinGecko provides comprehensive cryptocurrency data including prices,
market cap, volume, and historical OHLCV data.

API Documentation: https://docs.coingecko.com/
"""
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd
import requests

from .base import DataProvider, ProviderConfig


class CoinGeckoProvider(DataProvider):
    """CoinGecko API data provider.

    Supports both free and pro API tiers:
    - Free: 30 calls/minute, no API key required
    - Demo: 30 calls/minute with API key
    - Pro: 500 calls/minute with pro API key

    Attributes:
        name: Provider identifier
        supports_ohlcv: Whether OHLCV data is available
        requires_api_key: Whether API key is required (False for free tier)
    """

    name = "coingecko"
    supports_ohlcv = True
    supports_trades = False
    requires_api_key = False  # Free tier available

    # Base URLs for different tiers
    FREE_API_URL = "https://api.coingecko.com/api/v3"
    PRO_API_URL = "https://pro-api.coingecko.com/api/v3"

    # CoinGecko uses lowercase IDs
    SYMBOL_TO_ID = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "SOL": "solana",
        "XRP": "ripple",
        "ADA": "cardano",
        "DOGE": "dogecoin",
        "DOT": "polkadot",
        "MATIC": "matic-network",
        "LINK": "chainlink",
        "AVAX": "avalanche-2",
        "LTC": "litecoin",
        "BNB": "binancecoin",
        "SHIB": "shiba-inu",
        "TRX": "tron",
        "ATOM": "cosmos",
        "UNI": "uniswap",
        "XLM": "stellar",
        "BCH": "bitcoin-cash",
        "FIL": "filecoin",
        "APT": "aptos",
        "ARB": "arbitrum",
        "OP": "optimism",
        "NEAR": "near",
        "ICP": "internet-computer",
    }

    def __init__(self, config: Optional[ProviderConfig] = None):
        """Initialize CoinGecko provider.

        Args:
            config: Provider configuration with optional API key
        """
        super().__init__(config)

        # Select API endpoint based on whether we have an API key
        if self.config.api_key:
            # Check if it's a pro key (starts with CG-)
            if self.config.api_key.startswith("CG-"):
                self.base_url = self.PRO_API_URL
                self.header_key = "x-cg-pro-api-key"
            else:
                self.base_url = self.FREE_API_URL
                self.header_key = "x-cg-demo-api-key"
        else:
            self.base_url = self.FREE_API_URL
            self.header_key = None

        self._last_request_time = 0
        self._coin_list_cache: Optional[Dict[str, str]] = None

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed_ms = (time.time() * 1000) - self._last_request_time
        if elapsed_ms < self.config.rate_limit_ms:
            time.sleep((self.config.rate_limit_ms - elapsed_ms) / 1000)
        self._last_request_time = time.time() * 1000

    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make an API request with rate limiting and retries.

        Args:
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            JSON response as dictionary

        Raises:
            requests.RequestException: If request fails after retries
        """
        url = f"{self.base_url}/{endpoint}"
        headers = {}

        if self.header_key and self.config.api_key:
            headers[self.header_key] = self.config.api_key

        last_error = None
        for attempt in range(self.config.max_retries):
            self._rate_limit()
            try:
                response = requests.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.config.timeout_s,
                )
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff

        raise last_error

    def _get_coin_id(self, symbol: str) -> str:
        """Convert symbol to CoinGecko coin ID.

        Args:
            symbol: Symbol like 'BTC', 'ETH', 'BTCUSDT'

        Returns:
            CoinGecko coin ID like 'bitcoin', 'ethereum'
        """
        normalized = self.normalize_symbol(symbol)

        # Check our mapping first
        if normalized in self.SYMBOL_TO_ID:
            return self.SYMBOL_TO_ID[normalized]

        # Lazy load full coin list if needed
        if self._coin_list_cache is None:
            self._load_coin_list()

        # Search in cache
        if normalized.lower() in self._coin_list_cache:
            return self._coin_list_cache[normalized.lower()]

        # Default to lowercase symbol
        return normalized.lower()

    def _load_coin_list(self) -> None:
        """Load and cache the full coin list from CoinGecko."""
        try:
            coins = self._make_request("coins/list")
            self._coin_list_cache = {}
            for coin in coins:
                # Map by symbol (lowercase)
                sym = coin.get("symbol", "").lower()
                if sym and sym not in self._coin_list_cache:
                    self._coin_list_cache[sym] = coin["id"]
        except Exception:
            self._coin_list_cache = {}

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        days: Optional[int] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from CoinGecko.

        CoinGecko provides OHLC data at specific granularities based on time range:
        - 1-2 days: 30-minute candles
        - 3-30 days: 4-hour candles
        - 31+ days: daily candles

        Args:
            symbol: Symbol like 'BTC', 'ETH', 'bitcoin'
            timeframe: Ignored (CoinGecko auto-selects based on range)
            days: Number of days of history (default: 30)
            start: Start datetime (converted to days)
            end: End datetime

        Returns:
            DataFrame with OHLCV data
        """
        coin_id = self._get_coin_id(symbol)
        normalized_symbol = self.normalize_symbol(symbol) + "USDT"

        # Calculate days from start/end if provided
        if days is None:
            if start is not None:
                end = end or datetime.now(timezone.utc)
                days = (end - start).days
            else:
                days = 30

        # CoinGecko OHLC endpoint
        params = {
            "vs_currency": "usd",
            "days": str(days),
        }

        try:
            data = self._make_request(f"coins/{coin_id}/ohlc", params)
        except requests.RequestException as e:
            print(f"CoinGecko API error: {e}")
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume", "symbol"]
            )

        if not data:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume", "symbol"]
            )

        # Convert to DataFrame
        # CoinGecko OHLC format: [timestamp_ms, open, high, low, close]
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["symbol"] = normalized_symbol

        # CoinGecko OHLC doesn't include volume, fetch separately
        df["volume"] = self._fetch_volume_data(coin_id, days, df)

        return df.sort_values("timestamp").reset_index(drop=True)

    def _fetch_volume_data(
        self, coin_id: str, days: int, ohlc_df: pd.DataFrame
    ) -> pd.Series:
        """Fetch volume data and align with OHLC timestamps.

        Args:
            coin_id: CoinGecko coin ID
            days: Number of days
            ohlc_df: OHLC DataFrame to align volume with

        Returns:
            Series of volume values
        """
        try:
            params = {"vs_currency": "usd", "days": str(days)}
            data = self._make_request(f"coins/{coin_id}/market_chart", params)

            if "total_volumes" in data and data["total_volumes"]:
                vol_df = pd.DataFrame(data["total_volumes"], columns=["ts", "volume"])
                vol_df["ts"] = pd.to_datetime(vol_df["ts"], unit="ms", utc=True)

                # Merge on nearest timestamp
                ohlc_df = ohlc_df.copy()
                ohlc_df["_merge_ts"] = ohlc_df["timestamp"]
                vol_df = vol_df.rename(columns={"ts": "_merge_ts"})

                merged = pd.merge_asof(
                    ohlc_df.sort_values("_merge_ts"),
                    vol_df.sort_values("_merge_ts"),
                    on="_merge_ts",
                    direction="nearest",
                )
                return merged["volume"].fillna(0)

        except Exception:
            pass

        # Default to zero volume if fetch fails
        return pd.Series([0.0] * len(ohlc_df))

    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols from CoinGecko.

        Returns:
            List of symbol strings (uppercase)
        """
        if self._coin_list_cache is None:
            self._load_coin_list()

        return [sym.upper() for sym in self._coin_list_cache.keys()]

    def fetch_current_price(self, symbols: List[str]) -> Dict[str, float]:
        """Fetch current prices for multiple symbols.

        Args:
            symbols: List of symbols

        Returns:
            Dictionary mapping symbol to USD price
        """
        coin_ids = [self._get_coin_id(s) for s in symbols]

        params = {
            "ids": ",".join(coin_ids),
            "vs_currencies": "usd",
        }

        try:
            data = self._make_request("simple/price", params)
            return {
                self.normalize_symbol(sym): data.get(self._get_coin_id(sym), {}).get(
                    "usd", 0
                )
                for sym in symbols
            }
        except Exception:
            return {}
