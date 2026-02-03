"""Ingest package exports."""
from __future__ import annotations

from .loader import load_or_create_prints, create_data_manifest
from .loader import generate_stub_prints as _generate_stub_prints

# CCXT-based live data fetching (optional - requires ccxt package)
try:
    from .ccxt_loader import (
        CCXTLoader,
        fetch_live_prints,
        fetch_live_ohlcv,
    )
    _HAS_CCXT = True
except ImportError:
    _HAS_CCXT = False
    CCXTLoader = None
    fetch_live_prints = None
    fetch_live_ohlcv = None

# Multi-source providers
try:
    from .providers import (
        DataProvider,
        ProviderConfig,
        CoinGeckoProvider,
        CoinMarketCapProvider,
        YFinanceProvider,
        BinancePublicProvider,
        UnifiedDataFetcher,
    )
    from .providers.unified import fetch_crypto_data, fetch_crypto_prints
    _HAS_PROVIDERS = True
except ImportError:
    _HAS_PROVIDERS = False
    DataProvider = None
    ProviderConfig = None
    CoinGeckoProvider = None
    CoinMarketCapProvider = None
    YFinanceProvider = None
    BinancePublicProvider = None
    UnifiedDataFetcher = None
    fetch_crypto_data = None
    fetch_crypto_prints = None
# Optional CoinGecko imports - gracefully handle if dependencies missing
try:
    from .coingecko import fetch_coingecko_prints, CoinGeckoClient
    _COINGECKO_AVAILABLE = True
except ImportError:
    _COINGECKO_AVAILABLE = False
    fetch_coingecko_prints = None
    CoinGeckoClient = None

__all__ = [
    "load_or_create_prints",
    "create_data_manifest",
    "generate_stub_prints",
    # CCXT exports (available if ccxt is installed)
    "CCXTLoader",
    "fetch_live_prints",
    "fetch_live_ohlcv",
    # Multi-source providers
    "DataProvider",
    "ProviderConfig",
    "CoinGeckoProvider",
    "CoinMarketCapProvider",
    "YFinanceProvider",
    "BinancePublicProvider",
    "UnifiedDataFetcher",
    "fetch_crypto_data",
    "fetch_crypto_prints",
]

]

# Only export CoinGecko classes if available
if _COINGECKO_AVAILABLE:
    __all__.extend(["fetch_coingecko_prints", "CoinGeckoClient"])

def generate_stub_prints(
    symbols: list[str],
    n_days: int = 365,
    trades_per_day: int = 50000,
    start_date: str = "2023-01-01",
    seed: int = 1337,
):
    """
    Stable public API for tests:
      generate_stub_prints(symbols, n_days=..., trades_per_day=..., seed=...)
    """
    return _generate_stub_prints(
        symbols=symbols,
        start_date=start_date,
        n_days=n_days,
        trades_per_day=trades_per_day,
        seed=seed,
    )
