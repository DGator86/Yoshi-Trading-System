"""Tests for continuous crypto source scanner."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gnosis.ingest.source_scanner import (  # noqa: E402
    CryptoSourceScanner,
    CryptoSourceScannerConfig,
    SourceSpec,
)


class _FakeLoader:
    def __init__(self, exchange: str):
        self.exchange_name = exchange

    def fetch_ticker(self, symbol: str):
        base = {"BTC/USDT": 100.0, "ETH/USDT": 50.0, "SOL/USDT": 20.0}.get(symbol, 10.0)
        bump = {"binance": 0.0, "okx": 0.2, "bybit": -0.1}.get(self.exchange_name, 0.0)
        last = base + bump
        return {
            "timestamp": 1_700_000_000_000,
            "bid": last - 0.05,
            "ask": last + 0.05,
            "last": last,
            "baseVolume": 1000,
            "quoteVolume": 100000,
            "high": last + 2.0,
            "low": last - 2.0,
            "percentage": 1.2,
        }

    def fetch_order_book(self, symbol: str, limit: int):
        last = self.fetch_ticker(symbol)["last"]
        bids = [[last - 0.05 - i * 0.01, 5 + i] for i in range(min(limit, 5))]
        asks = [[last + 0.05 + i * 0.01, 4 + i] for i in range(min(limit, 5))]
        return {
            "timestamp": 1_700_000_000_001,
            "bids": bids,
            "asks": asks,
        }

    def fetch_funding_rate(self, symbol: str):
        return {"fundingRate": 0.0001, "nextFundingTimestamp": 1_700_000_100_000}

    def fetch_open_interest(self, symbol: str):
        return {"openInterest": 123456.0}


def _loader_factory(source: SourceSpec):
    return _FakeLoader(source.exchange.lower())


def test_scan_once_builds_snapshots_and_consensus(tmp_path: Path):
    config = CryptoSourceScannerConfig(
        symbols=["BTCUSDT", "ETHUSDT"],
        sources=[
            SourceSpec(
                exchange="binance",
                symbols=["BTCUSDT", "ETHUSDT"],
                ccxt_symbol_overrides={"BTCUSDT": "BTC/USDT", "ETHUSDT": "ETH/USDT"},
            ),
            SourceSpec(
                exchange="okx",
                symbols=["BTCUSDT", "ETHUSDT"],
                ccxt_symbol_overrides={"BTCUSDT": "BTC/USDT", "ETHUSDT": "ETH/USDT"},
            ),
        ],
        output_dir=str(tmp_path / "scanner"),
        poll_interval_sec=0.1,
    )
    scanner = CryptoSourceScanner(config=config, loader_factory=_loader_factory)
    result = scanner.scan_once()

    assert "snapshots" in result and len(result["snapshots"]) == 4
    assert "consensus" in result and "BTCUSDT" in result["consensus"]
    btc = result["consensus"]["BTCUSDT"]
    assert btc["n_sources"] == 2
    assert btc["min_last"] <= btc["median_last"] <= btc["max_last"]

    sample = result["snapshots"][0]
    assert "spread_bps" in sample
    assert "funding_rate" in sample
    assert "open_interest" in sample


def test_persist_scan_writes_latest_and_ndjson(tmp_path: Path):
    out_dir = tmp_path / "scanner_data"
    config = CryptoSourceScannerConfig(
        symbols=["BTCUSDT"],
        sources=[
            SourceSpec(
                exchange="binance",
                symbols=["BTCUSDT"],
                ccxt_symbol_overrides={"BTCUSDT": "BTC/USDT"},
            )
        ],
        output_dir=str(out_dir),
    )
    scanner = CryptoSourceScanner(config=config, loader_factory=_loader_factory)
    result = scanner.run_once()

    latest = out_dir / "latest.json"
    consensus = out_dir / "consensus_latest.json"
    assert latest.exists()
    assert consensus.exists()

    lines = list(out_dir.glob("snapshots_*.ndjson"))
    assert lines, "Expected at least one NDJSON snapshot file"
    with open(latest, encoding="utf-8") as handle:
        latest_payload = json.load(handle)
    assert len(latest_payload.get("snapshots", [])) == len(result.get("snapshots", []))


def test_config_from_yaml(tmp_path: Path):
    cfg_path = tmp_path / "scanner.yaml"
    cfg_path.write_text(
        """
scanner:
  symbols: [BTCUSDT]
  poll_interval_sec: 3
  output_dir: data/live/crypto_sources
  sources:
    - exchange: binance
      symbols: [BTCUSDT]
      ccxt_symbol_overrides:
        BTCUSDT: BTC/USDT
""".strip()
    )
    config = CryptoSourceScannerConfig.from_yaml(cfg_path)
    assert config.symbols == ["BTCUSDT"]
    assert config.poll_interval_sec == 3
    assert config.sources[0].exchange == "binance"
    assert config.sources[0].ccxt_symbol_overrides["BTCUSDT"] == "BTC/USDT"
