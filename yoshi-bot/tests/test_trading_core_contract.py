"""Contract tests for Trading Core /propose payload compatibility."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from fastapi.testclient import TestClient

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
    trading_core.kalshi.place_order = lambda **kwargs: {  # type: ignore[assignment]
        "order_id": f"ord-{kwargs.get('client_order_id', 'x')}",
        "ticker": kwargs.get("ticker"),
    }
    return trading_core


def test_propose_accepts_canonical_payload_and_is_idempotent(monkeypatch, tmp_path):
    trading_core = _load_core(monkeypatch, tmp_path)
    client = TestClient(trading_core.app)

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

    first = client.post("/propose", json=payload)
    assert first.status_code == 200
    first_json = first.json()
    assert first_json["success"] is True
    assert "Executed" in first_json["message"]

    second = client.post("/propose", json=payload)
    assert second.status_code == 200
    assert second.json() == first_json

    orders = client.get("/orders")
    assert orders.status_code == 200
    assert len(orders.json()) == 1

    proposals = client.get("/proposals")
    assert proposals.status_code == 200
    assert len(proposals.json()) == 1


def test_propose_rejects_legacy_noncanonical_payload(monkeypatch, tmp_path):
    trading_core = _load_core(monkeypatch, tmp_path)
    client = TestClient(trading_core.app)

    legacy_payload = {
        "exchange": "kalshi",
        "symbol": "BTCUSDT",
        "side": "buy",
        "type": "market",
        "amount": 1,
    }
    resp = client.post("/propose", json=legacy_payload)
    assert resp.status_code == 422

