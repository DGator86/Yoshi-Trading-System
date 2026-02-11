"""Shared scanner/bridge/core signal schema tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.trading_signals import (  # noqa: E402
    make_trade_signal,
    parse_event_line,
    wrap_signal_event,
)


def test_trade_signal_event_roundtrip():
    signal = make_trade_signal(
        symbol="ETHUSDT",
        ticker="KXETH-TEST-T4000",
        action="BUY_NO",
        strike=4000.0,
        market_prob=0.62,
        model_prob=0.52,
        edge=-0.10,
        source="unit_test",
    )
    event = wrap_signal_event(signal)
    parsed = parse_event_line(json.dumps(event))
    assert parsed is not None
    assert parsed["signal"]["idempotency_key"] == signal.idempotency_key
    assert parsed["signal"]["signal_id"] == signal.signal_id


def test_trade_signal_parser_rejects_invalid_lines():
    assert parse_event_line("") is None
    assert parse_event_line("not-json") is None
    assert parse_event_line(json.dumps({"event_type": "noop"})) is None

