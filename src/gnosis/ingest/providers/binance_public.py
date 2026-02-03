"""Binance Public Data provider.

Fetches historical market data from Binance's public data repository.
No API key required - this uses the same data source as tradezap.

Data Source: https://data.binance.vision/
"""
import io
import zipfile
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import pandas as pd
import requests

from .base import DataProvider, ProviderConfig


class BinancePublicProvider(DataProvider):
    """Binance Public Data repository provider.

    Downloads historical data from Binance's public data archive.
    No API key required. Data is available in daily and monthly files.

    Supported data types:
    - OHLCV (klines) at various timeframes
    - Trade data (individual trades)
    - Aggregated trades

    Attributes:
        name: Provider identifier
        supports_ohlcv: OHLCV data is available
        supports_trades: Trade-level data is available
        requires_api_key: No API key needed
    """

    name = "binance_public"
    supports_ohlcv = True
    supports_trades = True
    requires_api_key = False

    BASE_URL = "https://data.binance.vision/data"

    # Available kline intervals
    KLINE_INTERVALS = [
        "1s", "1m", "3m", "5m", "15m", "30m",
        "1h", "2h", "4h", "6h", "8h", "12h",
        "1d", "3d", "1w", "1M"
    ]

    def __init__(self, config: Optional[ProviderConfig] = None):
        """Initialize Binance public data provider.

        Args:
            config: Provider configuration
        """
        super().__init__(config)
        self._session = requests.Session()

    def _download_zip(self, url: str) -> Optional[bytes]:
        """Download and extract a ZIP file from Binance.

        Args:
            url: Full URL to the ZIP file

        Returns:
            CSV content as bytes, or None if not found
        """
        try:
            response = self._session.get(url, timeout=self.config.timeout_s)
            if response.status_code == 404:
                return None
            response.raise_for_status()

            # Extract CSV from ZIP
            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                # Get first CSV file in archive
                csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                if csv_names:
                    return zf.read(csv_names[0])

            return None

        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return None

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        days: Optional[int] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Binance public repository.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT', 'BTC', 'BTC/USDT')
            timeframe: Kline interval (1m, 5m, 1h, 4h, 1d, etc.)
            days: Number of days of history
            start: Start datetime
            end: End datetime

        Returns:
            DataFrame with OHLCV data
        """
        # Normalize symbol to BTCUSDT format
        normalized = self.normalize_symbol(symbol)
        pair = f"{normalized}USDT"

        # Validate interval
        if timeframe not in self.KLINE_INTERVALS:
            print(f"Invalid interval {timeframe}, using 1h")
            timeframe = "1h"

        # Calculate date range
        end_dt = end or datetime.now(timezone.utc)
        if days is not None:
            start_dt = end_dt - timedelta(days=days)
        elif start is not None:
            start_dt = start
        else:
            start_dt = end_dt - timedelta(days=30)

        # Collect data for each day
        all_data = []
        current = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)

        while current <= end_dt:
            year = current.year
            month = f"{current.month:02d}"
            day = f"{current.day:02d}"

            # Try daily file first
            url = (
                f"{self.BASE_URL}/spot/daily/klines/{pair}/{timeframe}/"
                f"{pair}-{timeframe}-{year}-{month}-{day}.zip"
            )

            csv_data = self._download_zip(url)
            if csv_data:
                df = self._parse_klines_csv(csv_data, pair)
                all_data.append(df)

            current += timedelta(days=1)

        if not all_data:
            # Try monthly files as fallback
            all_data = self._fetch_monthly_klines(pair, timeframe, start_dt, end_dt)

        if not all_data:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume", "symbol"]
            )

        # Combine and filter
        df = pd.concat(all_data, ignore_index=True)
        df = df.drop_duplicates(subset=["timestamp"])
        df = df[
            (df["timestamp"] >= start_dt) &
            (df["timestamp"] <= end_dt)
        ]

        return df.sort_values("timestamp").reset_index(drop=True)

    def _fetch_monthly_klines(
        self,
        pair: str,
        interval: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> List[pd.DataFrame]:
        """Fetch monthly kline files as fallback.

        Args:
            pair: Trading pair
            interval: Kline interval
            start_dt: Start datetime
            end_dt: End datetime

        Returns:
            List of DataFrames
        """
        all_data = []
        current = start_dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        while current <= end_dt:
            year = current.year
            month = f"{current.month:02d}"

            url = (
                f"{self.BASE_URL}/spot/monthly/klines/{pair}/{interval}/"
                f"{pair}-{interval}-{year}-{month}.zip"
            )

            csv_data = self._download_zip(url)
            if csv_data:
                df = self._parse_klines_csv(csv_data, pair)
                all_data.append(df)

            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

        return all_data

    def _parse_klines_csv(self, csv_data: bytes, symbol: str) -> pd.DataFrame:
        """Parse Binance klines CSV data.

        Binance klines format:
        open_time, open, high, low, close, volume, close_time,
        quote_volume, trades, taker_buy_base, taker_buy_quote, ignore

        Args:
            csv_data: Raw CSV bytes
            symbol: Symbol to add to DataFrame

        Returns:
            Parsed DataFrame
        """
        columns = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ]

        df = pd.read_csv(
            io.BytesIO(csv_data),
            header=None,
            names=columns,
        )

        # Convert timestamp
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)

        # Select columns
        df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        df["symbol"] = symbol

        # Convert numeric columns
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def fetch_trades(
        self,
        symbol: str,
        days: Optional[int] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch trade data from Binance public repository.

        Args:
            symbol: Trading pair
            days: Number of days
            start: Start datetime
            end: End datetime

        Returns:
            DataFrame with trade data
        """
        normalized = self.normalize_symbol(symbol)
        pair = f"{normalized}USDT"

        # Calculate date range
        end_dt = end or datetime.now(timezone.utc)
        if days is not None:
            start_dt = end_dt - timedelta(days=days)
        elif start is not None:
            start_dt = start
        else:
            start_dt = end_dt - timedelta(days=1)

        # Limit to 7 days max (trade files are large)
        if (end_dt - start_dt).days > 7:
            print("Trade data limited to 7 days due to file size")
            start_dt = end_dt - timedelta(days=7)

        all_data = []
        current = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)

        while current <= end_dt:
            year = current.year
            month = f"{current.month:02d}"
            day = f"{current.day:02d}"

            url = (
                f"{self.BASE_URL}/spot/daily/trades/{pair}/"
                f"{pair}-trades-{year}-{month}-{day}.zip"
            )

            csv_data = self._download_zip(url)
            if csv_data:
                df = self._parse_trades_csv(csv_data, pair)
                all_data.append(df)

            current += timedelta(days=1)

        if not all_data:
            return pd.DataFrame(
                columns=["timestamp", "symbol", "price", "quantity", "side", "trade_id"]
            )

        df = pd.concat(all_data, ignore_index=True)
        df = df[
            (df["timestamp"] >= start_dt) &
            (df["timestamp"] <= end_dt)
        ]

        return df.sort_values("timestamp").reset_index(drop=True)

    def _parse_trades_csv(self, csv_data: bytes, symbol: str) -> pd.DataFrame:
        """Parse Binance trades CSV data.

        Binance trades format:
        id, price, qty, quoteQty, time, isBuyerMaker, isBestMatch

        Args:
            csv_data: Raw CSV bytes
            symbol: Symbol to add

        Returns:
            Parsed DataFrame
        """
        columns = [
            "trade_id", "price", "qty", "quote_qty",
            "time", "is_buyer_maker", "is_best_match"
        ]

        df = pd.read_csv(
            io.BytesIO(csv_data),
            header=None,
            names=columns,
        )

        # Convert timestamp
        df["timestamp"] = pd.to_datetime(df["time"], unit="ms", utc=True)

        # Determine side (is_buyer_maker=True means the maker was buyer, so trade was SELL)
        df["side"] = df["is_buyer_maker"].apply(lambda x: "SELL" if x else "BUY")

        # Rename columns
        df = df.rename(columns={"qty": "quantity"})

        # Select columns
        df = df[["timestamp", "price", "quantity", "side", "trade_id"]].copy()
        df["symbol"] = symbol

        # Convert numeric
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
        df["trade_id"] = df["trade_id"].astype(str)

        return df

    def get_available_symbols(self) -> List[str]:
        """Get commonly available trading pairs.

        Note: Binance has hundreds of pairs. This returns major ones.

        Returns:
            List of symbol strings
        """
        return [
            "BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "DOT",
            "MATIC", "LINK", "AVAX", "LTC", "BNB", "SHIB", "TRX",
            "ATOM", "UNI", "XLM", "BCH", "FIL", "APT", "ARB", "OP"
        ]

    def fetch_aggTrades(
        self,
        symbol: str,
        days: int = 1,
    ) -> pd.DataFrame:
        """Fetch aggregated trade data.

        Aggregated trades combine trades executed at the same time and price.
        Files are smaller than individual trades.

        Args:
            symbol: Trading pair
            days: Number of days

        Returns:
            DataFrame with aggregated trade data
        """
        normalized = self.normalize_symbol(symbol)
        pair = f"{normalized}USDT"

        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=days)

        all_data = []
        current = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)

        while current <= end_dt:
            year = current.year
            month = f"{current.month:02d}"
            day = f"{current.day:02d}"

            url = (
                f"{self.BASE_URL}/spot/daily/aggTrades/{pair}/"
                f"{pair}-aggTrades-{year}-{month}-{day}.zip"
            )

            csv_data = self._download_zip(url)
            if csv_data:
                df = self._parse_agg_trades_csv(csv_data, pair)
                all_data.append(df)

            current += timedelta(days=1)

        if not all_data:
            return pd.DataFrame(
                columns=["timestamp", "symbol", "price", "quantity", "side", "trade_id"]
            )

        return pd.concat(all_data, ignore_index=True).sort_values("timestamp").reset_index(drop=True)

    def _parse_agg_trades_csv(self, csv_data: bytes, symbol: str) -> pd.DataFrame:
        """Parse aggregated trades CSV.

        Format: agg_trade_id, price, qty, first_trade_id, last_trade_id, time, is_buyer_maker

        Args:
            csv_data: Raw CSV bytes
            symbol: Symbol to add

        Returns:
            Parsed DataFrame
        """
        columns = [
            "trade_id", "price", "qty", "first_id", "last_id",
            "time", "is_buyer_maker"
        ]

        df = pd.read_csv(
            io.BytesIO(csv_data),
            header=None,
            names=columns,
        )

        df["timestamp"] = pd.to_datetime(df["time"], unit="ms", utc=True)
        df["side"] = df["is_buyer_maker"].apply(lambda x: "SELL" if x else "BUY")
        df = df.rename(columns={"qty": "quantity"})
        df["symbol"] = symbol
        df["trade_id"] = df["trade_id"].astype(str)

        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")

        return df[["timestamp", "symbol", "price", "quantity", "side", "trade_id"]]
