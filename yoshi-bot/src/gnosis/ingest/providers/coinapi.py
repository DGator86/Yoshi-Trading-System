"""CoinAPI.io data provider.

CoinAPI offers historical OHLCV across many exchanges with a unified schema.
Docs: https://docs.coinapi.io/
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd
import requests

from .base import DataProvider, ProviderConfig


class CoinAPIProvider(DataProvider):
    """CoinAPI.io OHLCV provider."""

    name = "coinapi"
    supports_ohlcv = True
    supports_trades = False
    requires_api_key = True

    BASE_URL = "https://rest.coinapi.io/v1"

    PERIOD_MAP = {
        "1m": "1MIN",
        "5m": "5MIN",
        "15m": "15MIN",
        "30m": "30MIN",
        "1h": "1HRS",
        "4h": "4HRS",
        "1d": "1DAY",
    }

    def __init__(self, config: Optional[ProviderConfig] = None):
        super().__init__(config)
        self.base_url = self.config.base_url or self.BASE_URL
        self._session = requests.Session()

    @staticmethod
    def _to_asset(symbol: str) -> str:
        s = DataProvider.normalize_symbol(symbol)
        return s.upper()

    def _request(self, endpoint: str, params: Optional[Dict] = None) -> Dict | List:
        headers = {
            "X-CoinAPI-Key": self.config.api_key or "",
            "Accept": "application/json",
        }
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        last_err = None
        for _ in range(max(1, self.config.max_retries)):
            try:
                resp = self._session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.config.timeout_s,
                )
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                last_err = e
        raise last_err

    def _symbol_candidates(self, asset: str) -> list[str]:
        # Try common venues/quotes.
        return [
            f"BINANCE_SPOT_{asset}_USDT",
            f"KRAKEN_SPOT_{asset}_USD",
            f"COINBASE_SPOT_{asset}_USD",
            f"BITSTAMP_SPOT_{asset}_USD",
        ]

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        days: Optional[int] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        asset = self._to_asset(symbol)
        period = self.PERIOD_MAP.get(timeframe, "1HRS")

        end_dt = end or datetime.now(timezone.utc)
        if days is not None:
            start_dt = end_dt - timedelta(days=days)
        elif start is not None:
            start_dt = start
        else:
            start_dt = end_dt - timedelta(days=30)

        params = {
            "period_id": period,
            "time_start": start_dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "time_end": end_dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "limit": 5000,
        }

        data = None
        for sym_id in self._symbol_candidates(asset):
            try:
                raw = self._request(f"ohlcv/{sym_id}/history", params=params)
                if isinstance(raw, list) and raw:
                    data = (sym_id, raw)
                    break
            except Exception:
                continue

        if data is None:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume", "symbol"]
            )

        sym_id, rows = data
        records = []
        for row in rows:
            records.append(
                {
                    "timestamp": pd.to_datetime(row.get("time_period_start"), utc=True, errors="coerce"),
                    "open": row.get("price_open", 0.0),
                    "high": row.get("price_high", 0.0),
                    "low": row.get("price_low", 0.0),
                    "close": row.get("price_close", 0.0),
                    "volume": row.get("volume_traded", 0.0),
                    "symbol": f"{asset}USDT",
                    "source_symbol_id": sym_id,
                }
            )

        df = pd.DataFrame(records)
        if df.empty:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume", "symbol"]
            )
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["timestamp", "close"]).sort_values("timestamp").reset_index(drop=True)
        return df[["timestamp", "open", "high", "low", "close", "volume", "symbol"]]

    def get_available_symbols(self) -> List[str]:
        # Lightweight static set to avoid expensive symbol catalog calls.
        return ["BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "BNB", "AVAX", "LINK"]

    def fetch_current_price(self, symbols: List[str]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for sym in symbols:
            asset = self._to_asset(sym)
            try:
                data = self._request(f"exchangerate/{asset}/USD")
                out[asset] = float(data.get("rate", 0.0))
            except Exception:
                continue
        return out
