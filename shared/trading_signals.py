"""Shared signal schema and helpers for scanner -> bridge -> core flow."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SIGNAL_EVENTS_PATH_DEFAULT = "data/signals/scanner_signals.jsonl"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_action(action: str) -> str:
    a = str(action or "").strip().upper().replace(" ", "_")
    if a in {"BUY_YES", "BUY_NO"}:
        return a
    raise ValueError(f"Unsupported action: {action}")


def _norm_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float(default)


def _norm_symbol(symbol: str) -> str:
    s = str(symbol or "").upper().strip()
    return s if s else "BTCUSDT"


def build_idempotency_key(
    symbol: str,
    ticker: str,
    action: str,
    strike: float,
    model_prob: float,
    market_prob: float,
) -> str:
    seed = (
        f"{_norm_symbol(symbol)}|{str(ticker).strip().upper()}|{normalize_action(action)}|"
        f"{_norm_float(strike):.4f}|{_norm_float(model_prob):.6f}|{_norm_float(market_prob):.6f}"
    )
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return f"sig_{digest[:24]}"


@dataclass
class TradeSignal:
    """Canonical proposal payload transported between services."""

    symbol: str
    ticker: str
    action: str
    strike: float
    market_prob: float
    model_prob: float
    edge: float
    idempotency_key: str
    signal_id: str
    source: str = "kalshi_scanner"
    created_at: str = ""

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["symbol"] = _norm_symbol(self.symbol)
        payload["action"] = normalize_action(self.action)
        payload["strike"] = _norm_float(self.strike)
        payload["market_prob"] = _norm_float(self.market_prob)
        payload["model_prob"] = _norm_float(self.model_prob)
        payload["edge"] = _norm_float(self.edge)
        payload["created_at"] = self.created_at or utc_now_iso()
        return payload


def make_trade_signal(
    *,
    symbol: str,
    ticker: str,
    action: str,
    strike: float,
    market_prob: float,
    model_prob: float,
    edge: float,
    source: str = "kalshi_scanner",
) -> TradeSignal:
    created_at = utc_now_iso()
    idk = build_idempotency_key(
        symbol=symbol,
        ticker=ticker,
        action=action,
        strike=strike,
        model_prob=model_prob,
        market_prob=market_prob,
    )
    signal_id = f"{idk}_{int(datetime.now(timezone.utc).timestamp())}"
    return TradeSignal(
        symbol=_norm_symbol(symbol),
        ticker=str(ticker),
        action=normalize_action(action),
        strike=_norm_float(strike),
        market_prob=_norm_float(market_prob),
        model_prob=_norm_float(model_prob),
        edge=_norm_float(edge),
        idempotency_key=idk,
        signal_id=signal_id,
        source=source,
        created_at=created_at,
    )


def wrap_signal_event(signal: TradeSignal) -> dict[str, Any]:
    return {
        "event_type": "trade_signal",
        "event_version": 1,
        "created_at": utc_now_iso(),
        "signal": signal.to_payload(),
    }


def append_event_jsonl(path: str | Path, event: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def parse_event_line(line: str) -> dict[str, Any] | None:
    if not line or not line.strip():
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    if obj.get("event_type") != "trade_signal":
        return None
    sig = obj.get("signal")
    if not isinstance(sig, dict):
        return None
    # Minimal shape guard to keep downstream handling simple.
    required = ("symbol", "ticker", "action", "edge", "idempotency_key", "signal_id")
    if any(k not in sig for k in required):
        return None
    return obj

