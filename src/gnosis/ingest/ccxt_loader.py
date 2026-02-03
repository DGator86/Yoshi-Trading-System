"""CCXT-based cryptocurrency data loader.

This module provides functionality to fetch real cryptocurrency market data
using the CCXT library, which supports 100+ exchanges with a unified API.

Usage:
    from gnosis.ingest.ccxt_loader import CCXTLoader

    loader = CCXTLoader(exchange='binance')
    trades_df = loader.fetch_trades('BTC/USDT', days=7)
    ohlcv_df = loader.fetch_ohlcv('BTC/USDT', timeframe='1h', days=30)
"""
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

# Rate limiting constants
DEFAULT_RATE_LIMIT_MS = 100  # Minimum ms between requests
MAX_RETRIES = 3
RETRY_DELAY_S = 1.0

# Default fetch limits per exchange type
DEFAULT_OHLCV_LIMIT = 1000
DEFAULT_TRADES_LIMIT = 1000


class CCXTLoader:
    """Load cryptocurrency data from exchanges via CCXT.

    Provides methods to fetch OHLCV bars and trade-level data with
    automatic pagination, rate limiting, and error handling.

    Attributes:
        exchange_id: Name of the exchange (e.g., 'binance', 'kraken')
        exchange: CCXT exchange instance
        rate_limit_ms: Minimum milliseconds between API calls
    """

    # Supported exchanges with their specific configurations
    EXCHANGE_CONFIGS = {
        'binance': {'ohlcv_limit': 1000, 'trades_limit': 1000},
        'binanceus': {'ohlcv_limit': 1000, 'trades_limit': 1000},
        'kraken': {'ohlcv_limit': 720, 'trades_limit': 1000},
        'coinbase': {'ohlcv_limit': 300, 'trades_limit': 1000},
        'okx': {'ohlcv_limit': 300, 'trades_limit': 100},
        'bybit': {'ohlcv_limit': 200, 'trades_limit': 1000},
        'kucoin': {'ohlcv_limit': 1500, 'trades_limit': 1000},
        'gateio': {'ohlcv_limit': 1000, 'trades_limit': 1000},
    }

    def __init__(
        self,
        exchange: str = 'binance',
        api_key: Optional[str] = None,
        secret: Optional[str] = None,
        rate_limit_ms: int = DEFAULT_RATE_LIMIT_MS,
        sandbox: bool = False,
    ):
        """Initialize CCXT loader for an exchange.

        Args:
            exchange: Exchange name (e.g., 'binance', 'kraken', 'coinbase')
            api_key: Optional API key for authenticated endpoints
            secret: Optional API secret for authenticated endpoints
            rate_limit_ms: Minimum milliseconds between API requests
            sandbox: If True, use exchange sandbox/testnet mode

        Raises:
            ImportError: If ccxt is not installed
            ValueError: If exchange is not supported by ccxt
        """
        try:
            import ccxt
        except ImportError:
            raise ImportError(
                "CCXT is required for live data fetching. "
                "Install it with: pip install ccxt"
            )

        self.exchange_id = exchange.lower()
        self.rate_limit_ms = rate_limit_ms
        self._last_request_time = 0

        # Get exchange class
        if not hasattr(ccxt, self.exchange_id):
            available = [x for x in dir(ccxt) if not x.startswith('_')]
            raise ValueError(
                f"Exchange '{exchange}' not found in CCXT. "
                f"Available exchanges include: {', '.join(available[:10])}..."
            )

        # Configure exchange
        config = {
            'enableRateLimit': True,
            'rateLimit': rate_limit_ms,
        }

        if api_key and secret:
            config['apiKey'] = api_key
            config['secret'] = secret

        # Create exchange instance
        exchange_class = getattr(ccxt, self.exchange_id)
        self.exchange = exchange_class(config)

        # Enable sandbox mode if requested
        if sandbox:
            self.exchange.set_sandbox_mode(True)

        # Get exchange-specific limits
        self._config = self.EXCHANGE_CONFIGS.get(
            self.exchange_id,
            {'ohlcv_limit': DEFAULT_OHLCV_LIMIT, 'trades_limit': DEFAULT_TRADES_LIMIT}
        )

    def _rate_limit(self) -> None:
        """Enforce rate limiting between API calls."""
        elapsed = (time.time() * 1000) - self._last_request_time
        if elapsed < self.rate_limit_ms:
            time.sleep((self.rate_limit_ms - elapsed) / 1000)
        self._last_request_time = time.time() * 1000

    def _retry_request(self, func, *args, **kwargs):
        """Execute a request with retry logic.

        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result of the function call

        Raises:
            Exception: If all retries fail
        """
        import ccxt

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                self._rate_limit()
                return func(*args, **kwargs)
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY_S * (attempt + 1))
            except ccxt.RateLimitExceeded as e:
                last_error = e
                # Exponential backoff for rate limits
                time.sleep(RETRY_DELAY_S * (2 ** attempt))

        raise last_error

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = '1h',
        since: Optional[Union[datetime, str, int]] = None,
        days: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV (candlestick) data for a symbol.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT', 'ETH/USD')
            timeframe: Candle timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
            since: Start time as datetime, ISO string, or Unix timestamp (ms)
            days: Alternative to since - fetch last N days of data
            limit: Maximum number of candles (uses exchange default if not set)

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume, symbol

        Raises:
            ValueError: If symbol or timeframe is invalid
        """
        # Parse since parameter
        if days is not None:
            since_dt = datetime.now(timezone.utc) - timedelta(days=days)
            since_ms = int(since_dt.timestamp() * 1000)
        elif since is not None:
            since_ms = self._parse_timestamp(since)
        else:
            since_ms = None

        # Use exchange-specific limit
        fetch_limit = limit or self._config['ohlcv_limit']

        # Fetch data with pagination if needed
        all_ohlcv = []
        current_since = since_ms

        while True:
            ohlcv = self._retry_request(
                self.exchange.fetch_ohlcv,
                symbol,
                timeframe,
                since=current_since,
                limit=fetch_limit,
            )

            if not ohlcv:
                break

            all_ohlcv.extend(ohlcv)

            # Check if we need to continue paginating
            if len(ohlcv) < fetch_limit:
                break

            # Move to next page
            current_since = ohlcv[-1][0] + 1

            # Safety limit to prevent infinite loops
            if len(all_ohlcv) > 100000:
                break

        if not all_ohlcv:
            return pd.DataFrame(columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol'
            ])

        # Convert to DataFrame
        df = pd.DataFrame(
            all_ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

        # Add symbol column
        df['symbol'] = symbol.replace('/', '')

        # Remove duplicates and sort
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')

        return df.reset_index(drop=True)

    def fetch_trades(
        self,
        symbol: str,
        since: Optional[Union[datetime, str, int]] = None,
        days: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch trade-level data for a symbol.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT', 'ETH/USD')
            since: Start time as datetime, ISO string, or Unix timestamp (ms)
            days: Alternative to since - fetch last N days of data
            limit: Maximum number of trades per request

        Returns:
            DataFrame with columns matching the gnosis print format:
                timestamp, symbol, price, quantity, side, trade_id

        Note:
            Trade data can be very large. Use reasonable time ranges.
        """
        # Parse since parameter
        if days is not None:
            since_dt = datetime.now(timezone.utc) - timedelta(days=days)
            since_ms = int(since_dt.timestamp() * 1000)
        elif since is not None:
            since_ms = self._parse_timestamp(since)
        else:
            # Default to last 24 hours if no time specified
            since_dt = datetime.now(timezone.utc) - timedelta(days=1)
            since_ms = int(since_dt.timestamp() * 1000)

        # Use exchange-specific limit
        fetch_limit = limit or self._config['trades_limit']

        # Fetch data with pagination
        all_trades = []
        current_since = since_ms
        last_trade_id = None

        while True:
            try:
                trades = self._retry_request(
                    self.exchange.fetch_trades,
                    symbol,
                    since=current_since,
                    limit=fetch_limit,
                )
            except Exception as e:
                # Some exchanges don't support public trade history
                print(f"Warning: Could not fetch trades: {e}")
                break

            if not trades:
                break

            # Filter out duplicates based on trade ID
            new_trades = [
                t for t in trades
                if last_trade_id is None or t.get('id') != last_trade_id
            ]

            if not new_trades:
                break

            all_trades.extend(new_trades)
            last_trade_id = new_trades[-1].get('id')

            # Check if we need more data
            if len(trades) < fetch_limit:
                break

            # Move to next page using last timestamp
            current_since = trades[-1]['timestamp'] + 1

            # Safety limit
            if len(all_trades) > 500000:
                break

        if not all_trades:
            return pd.DataFrame(columns=[
                'timestamp', 'symbol', 'price', 'quantity', 'side', 'trade_id'
            ])

        # Convert to DataFrame in gnosis print format
        df = pd.DataFrame([
            {
                'timestamp': pd.Timestamp(t['timestamp'], unit='ms', tz='UTC'),
                'symbol': symbol.replace('/', ''),
                'price': float(t['price']),
                'quantity': float(t['amount']),
                'side': t['side'].upper(),  # 'BUY' or 'SELL'
                'trade_id': str(t.get('id', '')),
            }
            for t in all_trades
        ])

        # Remove duplicates and sort
        df = df.drop_duplicates(subset=['trade_id']).sort_values('timestamp')

        return df.reset_index(drop=True)

    def fetch_ticker(self, symbol: str) -> Dict:
        """Fetch current ticker data for a symbol.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')

        Returns:
            Dictionary with ticker data (bid, ask, last, volume, etc.)
        """
        return self._retry_request(self.exchange.fetch_ticker, symbol)

    def get_markets(self) -> List[str]:
        """Get list of available trading pairs on the exchange.

        Returns:
            List of symbol strings (e.g., ['BTC/USDT', 'ETH/USDT', ...])
        """
        self._retry_request(self.exchange.load_markets)
        return list(self.exchange.markets.keys())

    def _parse_timestamp(self, ts: Union[datetime, str, int]) -> int:
        """Parse various timestamp formats to Unix milliseconds.

        Args:
            ts: Timestamp as datetime, ISO string, or Unix ms

        Returns:
            Unix timestamp in milliseconds
        """
        if isinstance(ts, datetime):
            return int(ts.timestamp() * 1000)
        elif isinstance(ts, str):
            return self.exchange.parse8601(ts)
        elif isinstance(ts, (int, float)):
            return int(ts)
        else:
            raise ValueError(f"Cannot parse timestamp: {ts}")


def fetch_live_prints(
    symbols: List[str],
    exchange: str = 'binance',
    days: int = 7,
    api_key: Optional[str] = None,
    secret: Optional[str] = None,
    fallback_to_ohlcv: bool = True,
) -> pd.DataFrame:
    """Convenience function to fetch trade prints for multiple symbols.

    This function provides a drop-in replacement for generate_stub_prints()
    that fetches real market data. If trade data is unavailable, it can
    generate synthetic prints from OHLCV data.

    Args:
        symbols: List of trading pairs (e.g., ['BTCUSDT', 'ETHUSDT'])
        exchange: Exchange to fetch from (default: 'binance')
        days: Number of days of historical data to fetch
        api_key: Optional API key
        secret: Optional API secret
        fallback_to_ohlcv: If True, generate prints from OHLCV when trades unavailable

    Returns:
        DataFrame with columns: timestamp, symbol, price, quantity, side, trade_id
    """
    loader = CCXTLoader(
        exchange=exchange,
        api_key=api_key,
        secret=secret,
    )

    all_prints = []

    for symbol in symbols:
        # Convert BTCUSDT -> BTC/USDT format if needed
        if '/' not in symbol:
            # Assume USDT quote currency
            if symbol.endswith('USDT'):
                ccxt_symbol = f"{symbol[:-4]}/USDT"
            elif symbol.endswith('USD'):
                ccxt_symbol = f"{symbol[:-3]}/USD"
            else:
                ccxt_symbol = symbol
        else:
            ccxt_symbol = symbol

        print(f"Fetching trades for {ccxt_symbol} from {exchange}...")

        try:
            trades_df = loader.fetch_trades(ccxt_symbol, days=days)
            if not trades_df.empty:
                all_prints.append(trades_df)
                print(f"  Fetched {len(trades_df)} trades")
                continue
        except Exception as e:
            print(f"  Warning: Could not fetch trades for {ccxt_symbol}: {e}")

        # Fallback: Generate prints from OHLCV data
        if fallback_to_ohlcv:
            print(f"  Falling back to OHLCV-based print generation...")
            try:
                ohlcv_df = loader.fetch_ohlcv(ccxt_symbol, timeframe='1m', days=days)
                if not ohlcv_df.empty:
                    prints_from_ohlcv = _generate_prints_from_ohlcv(ohlcv_df)
                    all_prints.append(prints_from_ohlcv)
                    print(f"  Generated {len(prints_from_ohlcv)} synthetic prints from {len(ohlcv_df)} OHLCV bars")
            except Exception as e2:
                print(f"  Warning: Could not fetch OHLCV for {ccxt_symbol}: {e2}")

    if not all_prints:
        raise ValueError(f"Could not fetch any trade data for symbols: {symbols}")

    return pd.concat(all_prints, ignore_index=True).sort_values('timestamp')


def _generate_prints_from_ohlcv(ohlcv_df: pd.DataFrame, trades_per_bar: int = 50) -> pd.DataFrame:
    """Generate synthetic trade prints from OHLCV data.

    Creates realistic trade data by distributing trades within each OHLCV bar
    using the bar's price range and volume.

    Args:
        ohlcv_df: DataFrame with OHLCV data (timestamp, open, high, low, close, volume, symbol)
        trades_per_bar: Average number of trades to generate per bar

    Returns:
        DataFrame with synthetic trade prints
    """
    rng = np.random.default_rng(seed=42)  # Deterministic for reproducibility
    records = []

    for _, bar in ohlcv_df.iterrows():
        symbol = bar['symbol']
        bar_start = bar['timestamp']

        # Number of trades varies with volume (normalized)
        vol_factor = max(0.5, min(2.0, bar['volume'] / (ohlcv_df['volume'].mean() + 1e-10)))
        n_trades = max(10, int(trades_per_bar * vol_factor * rng.uniform(0.7, 1.3)))

        # Price path within the bar (OHLC-guided random walk)
        o, h, l, c = bar['open'], bar['high'], bar['low'], bar['close']
        price_range = h - l if h > l else o * 0.001

        # Generate price path that respects OHLC
        prices = np.zeros(n_trades)
        prices[0] = o
        prices[-1] = c

        # Fill intermediate prices with constrained random walk
        for i in range(1, n_trades - 1):
            progress = i / (n_trades - 1)
            target = o + (c - o) * progress
            noise = rng.normal(0, price_range * 0.1)
            prices[i] = np.clip(target + noise, l, h)

        # Timestamps spread within the bar (assuming 1-minute bars)
        offsets_ms = np.sort(rng.integers(0, 60000, n_trades))
        timestamps = [bar_start + pd.Timedelta(milliseconds=int(ms)) for ms in offsets_ms]

        # Quantities based on volume distribution
        total_vol = bar['volume']
        qty_weights = rng.exponential(1.0, n_trades)
        qty_weights /= qty_weights.sum()
        quantities = qty_weights * total_vol

        # Sides based on price movement
        for i in range(n_trades):
            if i == 0:
                side = 'BUY' if c > o else 'SELL'
            else:
                side = 'BUY' if prices[i] > prices[i-1] else 'SELL'

            records.append({
                'timestamp': timestamps[i],
                'symbol': symbol,
                'price': round(float(prices[i]), 8),
                'quantity': round(float(quantities[i]), 8),
                'side': side,
                'trade_id': f"{symbol}_{bar_start.value}_{i}",
            })

    df = pd.DataFrame(records)
    return df.sort_values('timestamp').reset_index(drop=True)


def fetch_live_ohlcv(
    symbols: List[str],
    exchange: str = 'binance',
    timeframe: str = '1h',
    days: int = 30,
    api_key: Optional[str] = None,
    secret: Optional[str] = None,
) -> pd.DataFrame:
    """Convenience function to fetch OHLCV data for multiple symbols.

    Args:
        symbols: List of trading pairs (e.g., ['BTCUSDT', 'ETHUSDT'])
        exchange: Exchange to fetch from (default: 'binance')
        timeframe: Candle timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
        days: Number of days of historical data to fetch
        api_key: Optional API key
        secret: Optional API secret

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume, symbol
    """
    loader = CCXTLoader(
        exchange=exchange,
        api_key=api_key,
        secret=secret,
    )

    all_ohlcv = []

    for symbol in symbols:
        # Convert BTCUSDT -> BTC/USDT format if needed
        if '/' not in symbol:
            if symbol.endswith('USDT'):
                ccxt_symbol = f"{symbol[:-4]}/USDT"
            elif symbol.endswith('USD'):
                ccxt_symbol = f"{symbol[:-3]}/USD"
            else:
                ccxt_symbol = symbol
        else:
            ccxt_symbol = symbol

        print(f"Fetching OHLCV for {ccxt_symbol} from {exchange}...")

        try:
            ohlcv_df = loader.fetch_ohlcv(ccxt_symbol, timeframe=timeframe, days=days)
            if not ohlcv_df.empty:
                all_ohlcv.append(ohlcv_df)
                print(f"  Fetched {len(ohlcv_df)} candles")
        except Exception as e:
            print(f"  Warning: Could not fetch {ccxt_symbol}: {e}")

    if not all_ohlcv:
        raise ValueError(f"Could not fetch any OHLCV data for symbols: {symbols}")

    return pd.concat(all_ohlcv, ignore_index=True).sort_values(['symbol', 'timestamp'])
