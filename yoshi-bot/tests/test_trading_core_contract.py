"""Contract tests for Trading Core /propose payload compatibility."""

from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.trading_signals import make_trade_signal  # noqa: E402


def _load_core(monkeypatch, tmp_path):
    db_path = tmp_path / "trading_core_test.sqlite3"
    monkeypatch.setenv("TRADING_CORE_DB_PATH", str(db_path))
    monkeypatch.setenv("TRADING_CORE_MIN_EDGE", "0.01")

    import src.gnosis.execution.trading_core as trading_core  # noqa: E402

    trading_core = importlib.reload(trading_core)
    class _StubKalshi:
        def place_order(self, **kwargs):
            return {
                "order_id": f"ord-{kwargs.get('client_order_id', 'x')}",
                "ticker": kwargs.get("ticker"),
            }

    trading_core.kalshi = _StubKalshi()
    return trading_core


def test_propose_accepts_canonical_payload_and_is_idempotent(monkeypatch, tmp_path):
    trading_core = _load_core(monkeypatch, tmp_path)

    payload = make_trade_signal(
        symbol="BTCUSDT",
        ticker="KXBTC-TEST-T100000",
        action="BUY_YES",
        strike=100000.0,
        market_prob=0.49,
        model_prob=0.58,
        edge=0.09,
        source="unit_test",
    ).to_payload()

    proposal = trading_core.TradeProposal(**payload)
    first_json = asyncio.run(trading_core.propose_trade(proposal))
    assert first_json["success"] is True
    assert "Executed" in first_json["message"]

    second_json = asyncio.run(trading_core.propose_trade(proposal))
    assert second_json == first_json

    orders = asyncio.run(trading_core.get_orders())
    assert len(orders) == 1

    proposals = asyncio.run(trading_core.get_proposals())
    assert len(proposals) == 1


def test_propose_rejects_legacy_noncanonical_payload(monkeypatch, tmp_path):
    trading_core = _load_core(monkeypatch, tmp_path)

    legacy_payload = {
        "exchange": "kalshi",
        "symbol": "BTCUSDT",
        "side": "buy",
        "type": "market",
        "amount": 1,
    }
    with pytest.raises(ValidationError):
        trading_core.TradeProposal(**legacy_payload)

