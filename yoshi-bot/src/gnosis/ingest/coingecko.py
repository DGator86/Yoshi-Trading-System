"""CoinGecko API client for fetching cryptocurrency market data."""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests


class CoinGeckoClient:
    """Client for interacting with CoinGecko API."""

    BASE_URL = "https://api.coingecko.com/api/v3"
    
    # Rate limiting configuration
    RATE_LIMIT_DELAY_SECONDS = 1.5
    
    # Synthetic trade generation parameters
    PRICE_VARIATION_PERCENT = 0.001  # ±0.1% price variation for synthetic trades
    SECONDS_PER_TRADE = 10  # Time offset between synthetic trades
    
    # Mapping from common trading pair symbols to CoinGecko coin IDs
    SYMBOL_TO_COIN_ID = {
        "BTCUSDT": "bitcoin",
        "ETHUSDT": "ethereum",
        "SOLUSDT": "solana",
    }

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize CoinGecko client.
        
        Args:
            api_key: CoinGecko API key. If not provided, will try to read from
                     COINGECKO_API_KEY environment variable.
        """
        self.api_key = api_key or os.getenv("COINGECKO_API_KEY")
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"x-cg-demo-api-key": self.api_key})

    def get_coin_id(self, symbol: str) -> str:
        """
        Convert trading pair symbol to CoinGecko coin ID.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            
        Returns:
            CoinGecko coin ID (e.g., "bitcoin")
            
        Raises:
            ValueError: If symbol is not supported
        """
        coin_id = self.SYMBOL_TO_COIN_ID.get(symbol)
        if not coin_id:
            raise ValueError(f"Unsupported symbol: {symbol}. Supported symbols: {list(self.SYMBOL_TO_COIN_ID.keys())}")
        return coin_id

    def fetch_market_chart(
        self,
        coin_id: str,
        vs_currency: str = "usd",
        days: int = 30,
    ) -> dict:
        """
        Fetch market chart data from CoinGecko.
        
        Args:
            coin_id: CoinGecko coin ID (e.g., "bitcoin")
            vs_currency: Currency to compare against (default: "usd")
            days: Number of days of historical data (default: 30)
            
        Returns:
            Dictionary with price, market cap, and volume data
        """
        url = f"{self.BASE_URL}/coins/{coin_id}/market_chart"
        params = {
            "vs_currency": vs_currency,
            "days": days,
        }
        
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def convert_to_prints(
        self,
        symbol: str,
        market_data: dict,
        trades_per_interval: int = 10,
    ) -> pd.DataFrame:
        """
        Convert CoinGecko market data to print (trade) format.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            market_data: Market chart data from CoinGecko
            trades_per_interval: Number of synthetic trades to generate per price point
            
        Returns:
            DataFrame with print data matching the expected schema
        """
        prices = market_data.get("prices", [])
        
        if not prices:
            raise ValueError("No price data available")
        
        records = []
        
        for idx, (timestamp_ms, price) in enumerate(prices):
            timestamp = pd.Timestamp(timestamp_ms, unit="ms")
            
            # Generate synthetic trades around each price point
            for i in range(trades_per_interval):
                # Add slight price variation
                price_variation = price * (
                    1 + (i - trades_per_interval / 2) * self.PRICE_VARIATION_PERCENT / trades_per_interval
                )
                
                # Alternate between BUY and SELL
                side = "BUY" if i % 2 == 0 else "SELL"
                
                # Small random quantity
                quantity = 0.1 + (i * 0.01)
                
                # Offset timestamp slightly within the interval
                time_offset = timedelta(seconds=i * self.SECONDS_PER_TRADE)
                
                records.append({
                    "timestamp": timestamp + time_offset,
                    "symbol": symbol,
                    "price": round(price_variation, 2),
                    "quantity": round(quantity, 6),
                    "side": side,
                    "trade_id": f"{symbol}_{idx}_{i}",
                })
        
        df = pd.DataFrame(records)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def fetch_prints_for_symbol(
        self,
        symbol: str,
        days: int = 30,
        trades_per_interval: int = 10,
    ) -> pd.DataFrame:
        """
        Fetch and convert market data to print format for a given symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            days: Number of days of historical data
            trades_per_interval: Number of synthetic trades per price point
            
        Returns:
            DataFrame with print data
        """
        coin_id = self.get_coin_id(symbol)
        market_data = self.fetch_market_chart(coin_id, days=days)
        return self.convert_to_prints(symbol, market_data, trades_per_interval)


def fetch_coingecko_prints(
    symbols: list[str],
    api_key: Optional[str] = None,
    days: int = 30,
    trades_per_interval: int = 10,
) -> pd.DataFrame:
    """
    Fetch print data from CoinGecko for multiple symbols.
    
    Args:
        symbols: List of trading pair symbols
        api_key: CoinGecko API key (optional)
        days: Number of days of historical data
        trades_per_interval: Number of synthetic trades per price point
        
    Returns:
        Combined DataFrame with print data for all symbols
    """
    client = CoinGeckoClient(api_key)
    dfs = []
    
    for symbol in symbols:
        try:
            df = client.fetch_prints_for_symbol(
                symbol=symbol,
                days=days,
                trades_per_interval=trades_per_interval,
            )
            dfs.append(df)
            # Rate limiting to respect API limits
            time.sleep(CoinGeckoClient.RATE_LIMIT_DELAY_SECONDS)
        except requests.RequestException as e:
            print(f"Warning: Request failed for {symbol} ({type(e).__name__}): {e}")
            continue
        except ValueError as e:
            print(f"Warning: Invalid data for {symbol}: {e}")
            continue
        except Exception as e:
            print(f"Warning: Unexpected error fetching data for {symbol} ({type(e).__name__}): {e}")
            continue
    
    if not dfs:
        raise ValueError("Failed to fetch data for any symbols")
    
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.sort_values("timestamp").reset_index(drop=True)
    return combined_df
