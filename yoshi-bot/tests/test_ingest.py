"""Tests for data ingestion."""
import sys
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import pytest
from gnosis.ingest import generate_stub_prints, CoinGeckoClient


def test_generate_stub_prints():
    """Test stub print generation."""
    df = generate_stub_prints(["BTCUSDT"], n_days=2, trades_per_day=100, seed=42)

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "timestamp" in df.columns
    assert "symbol" in df.columns
    assert "price" in df.columns
    assert "quantity" in df.columns
    assert "side" in df.columns
    assert df["symbol"].iloc[0] == "BTCUSDT"


def test_generate_stub_prints_deterministic():
    """Test that stub generation is deterministic."""
    df1 = generate_stub_prints(["BTCUSDT"], n_days=1, trades_per_day=50, seed=123)
    df2 = generate_stub_prints(["BTCUSDT"], n_days=1, trades_per_day=50, seed=123)

    pd.testing.assert_frame_equal(df1, df2)


def test_coingecko_client_initialization():
    """Test CoinGecko client initialization."""
    # Test with explicit API key
    client = CoinGeckoClient(api_key="test-key")
    assert client.api_key == "test-key"
    
    # Test without API key
    client = CoinGeckoClient()
    assert client.api_key is None or isinstance(client.api_key, str)


def test_coingecko_get_coin_id():
    """Test symbol to coin ID mapping."""
    client = CoinGeckoClient()
    
    assert client.get_coin_id("BTCUSDT") == "bitcoin"
    assert client.get_coin_id("ETHUSDT") == "ethereum"
    assert client.get_coin_id("SOLUSDT") == "solana"
    
    # Test unsupported symbol
    with pytest.raises(ValueError, match="Unsupported symbol"):
        client.get_coin_id("UNSUPPORTED")


def test_coingecko_convert_to_prints():
    """Test conversion of CoinGecko data to print format."""
    client = CoinGeckoClient()
    
    # Mock market data
    mock_data = {
        "prices": [
            [1609459200000, 29000.0],  # 2021-01-01 00:00:00
            [1609545600000, 29500.0],  # 2021-01-02 00:00:00
        ]
    }
    
    df = client.convert_to_prints("BTCUSDT", mock_data, trades_per_interval=5)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10  # 2 price points * 5 trades per interval
    assert "timestamp" in df.columns
    assert "symbol" in df.columns
    assert "price" in df.columns
    assert "quantity" in df.columns
    assert "side" in df.columns
    assert "trade_id" in df.columns
    assert all(df["symbol"] == "BTCUSDT")
    assert all(df["side"].isin(["BUY", "SELL"]))


@patch("gnosis.ingest.coingecko.requests.Session.get")
def test_coingecko_fetch_market_chart(mock_get):
    """Test fetching market chart from CoinGecko API."""
    # Mock the API response
    mock_response = Mock()
    mock_response.json.return_value = {
        "prices": [[1609459200000, 29000.0]],
        "market_caps": [[1609459200000, 500000000000]],
        "total_volumes": [[1609459200000, 50000000000]],
    }
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response
    
    client = CoinGeckoClient(api_key="test-key")
    data = client.fetch_market_chart("bitcoin", days=1)
    
    assert "prices" in data
    assert len(data["prices"]) > 0
    mock_get.assert_called_once()

