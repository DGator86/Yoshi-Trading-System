"""CoinMarketCap API data provider.

CoinMarketCap provides cryptocurrency market data including prices,
market cap, volume rankings, and historical data.

API Documentation: https://coinmarketcap.com/api/documentation/v1/
"""
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd
import requests

from .base import DataProvider, ProviderConfig


class CoinMarketCapProvider(DataProvider):
    """CoinMarketCap API data provider.

    Requires an API key. Plans:
    - Basic: $0/month - Latest quotes, metadata (10,000 calls/month)
    - Startup: $79/month - Historical data, OHLCV
    - Standard: $299/month - All endpoints

    Note: Historical OHLCV requires Startup plan or higher.

    Attributes:
        name: Provider identifier
        supports_ohlcv: Whether OHLCV data is available
        requires_api_key: API key is required
    """

    name = "coinmarketcap"
    supports_ohlcv = True
    supports_trades = False
    requires_api_key = True

    BASE_URL = "https://pro-api.coinmarketcap.com/v1"
    SANDBOX_URL = "https://sandbox-api.coinmarketcap.com/v1"

    # CMC uses numeric IDs, but also supports symbols
    SYMBOL_TO_ID = {
        "BTC": 1,
        "ETH": 1027,
        "SOL": 5426,
        "XRP": 52,
        "ADA": 2010,
        "DOGE": 74,
        "DOT": 6636,
        "MATIC": 3890,
        "LINK": 1975,
        "AVAX": 5805,
        "LTC": 2,
        "BNB": 1839,
        "SHIB": 5994,
        "TRX": 1958,
        "ATOM": 3794,
        "UNI": 7083,
        "XLM": 512,
        "BCH": 1831,
        "FIL": 2280,
        "APT": 21794,
        "ARB": 11841,
        "OP": 11840,
        "NEAR": 6535,
        "ICP": 8916,
    }

    def __init__(self, config: Optional[ProviderConfig] = None):
        """Initialize CoinMarketCap provider.

        Args:
            config: Provider configuration with API key
        """
        super().__init__(config)

        self.base_url = (
            self.SANDBOX_URL
            if self.config.extra.get("sandbox", False)
            else self.BASE_URL
        )
        self._last_request_time = 0
        self._id_map_cache: Optional[Dict[str, int]] = None

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
            JSON response data

        Raises:
            requests.RequestException: If request fails after retries
        """
        url = f"{self.base_url}/{endpoint}"
        headers = {
            "X-CMC_PRO_API_KEY": self.config.api_key,
            "Accept": "application/json",
        }

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
                result = response.json()

                # CMC wraps data in a "data" key
                if "status" in result and result["status"].get("error_code", 0) != 0:
                    raise requests.RequestException(
                        result["status"].get("error_message", "Unknown error")
                    )

                return result.get("data", result)

            except requests.RequestException as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)

        raise last_error

    def _get_cmc_id(self, symbol: str) -> int:
        """Convert symbol to CoinMarketCap ID.

        Args:
            symbol: Symbol like 'BTC', 'ETH'

        Returns:
            CoinMarketCap numeric ID
        """
        normalized = self.normalize_symbol(symbol)

        if normalized in self.SYMBOL_TO_ID:
            return self.SYMBOL_TO_ID[normalized]

        # Lazy load ID map if needed
        if self._id_map_cache is None:
            self._load_id_map()

        return self._id_map_cache.get(normalized, 1)  # Default to BTC

    def _load_id_map(self) -> None:
        """Load and cache the cryptocurrency ID map."""
        try:
            data = self._make_request("cryptocurrency/map", {"limit": 5000})
            self._id_map_cache = {}
            for coin in data:
                sym = coin.get("symbol", "").upper()
                if sym:
                    self._id_map_cache[sym] = coin["id"]
        except Exception:
            self._id_map_cache = {}

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        days: Optional[int] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from CoinMarketCap.

        Note: Historical OHLCV requires Startup plan ($79/month) or higher.
        This method falls back to daily quotes for Basic plan users.

        Args:
            symbol: Symbol like 'BTC', 'ETH'
            timeframe: 'daily', 'hourly', 'weekly', 'monthly' or interval shortcuts
            days: Number of days of history
            start: Start datetime
            end: End datetime

        Returns:
            DataFrame with OHLCV data
        """
        normalized = self.normalize_symbol(symbol)
        cmc_symbol = normalized + "USDT"

        # Calculate time range
        end_dt = end or datetime.now(timezone.utc)
        if days is not None:
            start_dt = end_dt - timedelta(days=days)
        elif start is not None:
            start_dt = start
        else:
            start_dt = end_dt - timedelta(days=30)

        # Map timeframe to CMC interval
        interval_map = {
            "1d": "daily",
            "daily": "daily",
            "1h": "hourly",
            "hourly": "hourly",
            "1w": "weekly",
            "weekly": "weekly",
            "1M": "monthly",
            "monthly": "monthly",
        }
        interval = interval_map.get(timeframe, "daily")

        try:
            # Try OHLCV endpoint (requires Startup plan)
            params = {
                "symbol": normalized,
                "time_start": start_dt.strftime("%Y-%m-%d"),
                "time_end": end_dt.strftime("%Y-%m-%d"),
                "interval": interval,
                "convert": "USD",
            }

            data = self._make_request("cryptocurrency/ohlcv/historical", params)

            # Parse OHLCV response
            if isinstance(data, dict) and "quotes" in data:
                quotes = data["quotes"]
            elif isinstance(data, list):
                quotes = data
            else:
                quotes = []

            if quotes:
                records = []
                for quote in quotes:
                    q = quote.get("quote", {}).get("USD", quote)
                    records.append({
                        "timestamp": pd.to_datetime(
                            quote.get("time_open", quote.get("timestamp")), utc=True
                        ),
                        "open": q.get("open", q.get("price", 0)),
                        "high": q.get("high", q.get("price", 0)),
                        "low": q.get("low", q.get("price", 0)),
                        "close": q.get("close", q.get("price", 0)),
                        "volume": q.get("volume", 0),
                        "symbol": cmc_symbol,
                    })

                return pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)

        except requests.RequestException as e:
            # OHLCV might not be available on Basic plan
            if "402" in str(e) or "upgrade" in str(e).lower():
                print(
                    f"CoinMarketCap OHLCV requires Startup plan. "
                    f"Falling back to daily quotes."
                )
            else:
                print(f"CoinMarketCap API error: {e}")

        # Fallback: Use quotes/historical for daily data
        return self._fetch_historical_quotes(normalized, start_dt, end_dt, cmc_symbol)

    def _fetch_historical_quotes(
        self,
        symbol: str,
        start_dt: datetime,
        end_dt: datetime,
        output_symbol: str,
    ) -> pd.DataFrame:
        """Fetch historical quotes as fallback for OHLCV.

        Args:
            symbol: Normalized symbol
            start_dt: Start datetime
            end_dt: End datetime
            output_symbol: Symbol for output DataFrame

        Returns:
            DataFrame with price data (close only, no OHLC)
        """
        try:
            params = {
                "symbol": symbol,
                "time_start": start_dt.strftime("%Y-%m-%d"),
                "time_end": end_dt.strftime("%Y-%m-%d"),
                "interval": "daily",
                "convert": "USD",
            }

            data = self._make_request("cryptocurrency/quotes/historical", params)

            if isinstance(data, dict) and "quotes" in data:
                quotes = data["quotes"]
            else:
                quotes = []

            if quotes:
                records = []
                for quote in quotes:
                    usd = quote.get("quote", {}).get("USD", {})
                    price = usd.get("price", 0)
                    records.append({
                        "timestamp": pd.to_datetime(quote.get("timestamp"), utc=True),
                        "open": price,
                        "high": price,
                        "low": price,
                        "close": price,
                        "volume": usd.get("volume_24h", 0),
                        "symbol": output_symbol,
                    })

                return pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)

        except Exception as e:
            print(f"CoinMarketCap historical quotes error: {e}")

        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume", "symbol"]
        )

    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols from CoinMarketCap.

        Returns:
            List of symbol strings
        """
        if self._id_map_cache is None:
            self._load_id_map()

        return list(self._id_map_cache.keys())

    def fetch_current_price(self, symbols: List[str]) -> Dict[str, float]:
        """Fetch current prices for multiple symbols.

        This endpoint is available on all plans including Basic.

        Args:
            symbols: List of symbols

        Returns:
            Dictionary mapping symbol to USD price
        """
        normalized = [self.normalize_symbol(s) for s in symbols]

        try:
            params = {
                "symbol": ",".join(normalized),
                "convert": "USD",
            }
            data = self._make_request("cryptocurrency/quotes/latest", params)

            result = {}
            for sym in normalized:
                if sym in data:
                    quote = data[sym].get("quote", {}).get("USD", {})
                    result[sym] = quote.get("price", 0)

            return result

        except Exception:
            return {}

    def fetch_global_metrics(self) -> Dict:
        """Fetch global cryptocurrency market metrics.

        Returns:
            Dictionary with global market data
        """
        try:
            data = self._make_request("global-metrics/quotes/latest")
            return data.get("quote", {}).get("USD", data)
        except Exception:
            return {}
