"""FastAPI Trading Core with persistence + idempotent proposal handling."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.gnosis.execution.circuit_breaker import CircuitBreaker, CircuitOpenError
from src.gnosis.utils.kalshi_client import KalshiClient
import src.gnosis.utils.notifications as notify

# Add repository root for shared schema import.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from shared.trading_signals import build_idempotency_key  # noqa: E402

logger = logging.getLogger(__name__)
app = FastAPI(title="Yoshi-Bot Trading Core")

DB_PATH = os.getenv("TRADING_CORE_DB_PATH", "data/trading_core.sqlite3")
MIN_EDGE = float(os.getenv("TRADING_CORE_MIN_EDGE", "0.05"))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float(default)


class TradeProposal(BaseModel):
    """Single trade proposal from scanner/bridge."""

    proposal_id: str = ""
    symbol: str
    ticker: Optional[str] = None
    action: str  # BUY_YES | BUY_NO
    strike: float
    market_prob: float
    model_prob: float
    edge: float
    idempotency_key: Optional[str] = None
    signal_id: Optional[str] = None
    source: Optional[str] = None
    created_at: Optional[str] = None
    raw_forecast: Optional[Dict[str, Any]] = None


class ActionResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


class StatusResponse(BaseModel):
    status: str
    active: bool
    is_paused: bool
    kill_switch_active: bool
    proposals_count: int
    positions_count: int
    orders_count: int
    timestamp: datetime
    version: str = "1.1.0"


class SqliteStore:
    """Small persistence layer for core state and trade artifacts."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS kv_state (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS proposals (
                        proposal_id TEXT PRIMARY KEY,
                        idempotency_key TEXT UNIQUE,
                        created_at TEXT NOT NULL,
                        status TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        response_json TEXT
                    );
                    CREATE TABLE IF NOT EXISTS orders (
                        local_id TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL,
                        payload_json TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS positions (
                        local_id TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL,
                        payload_json TEXT NOT NULL
                    );
                    """
                )
                conn.commit()
            finally:
                conn.close()

    def get_state_bool(self, key: str, default: bool) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute("SELECT value FROM kv_state WHERE key = ?", (key,)).fetchone()
                if not row:
                    return bool(default)
                return str(row["value"]).lower() == "true"
            finally:
                conn.close()

    def set_state_bool(self, key: str, value: bool):
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT INTO kv_state(key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (key, "true" if value else "false"),
                )
                conn.commit()
            finally:
                conn.close()

    def proposal_by_idempotency(self, idempotency_key: str) -> Optional[dict]:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT proposal_id, idempotency_key, status, payload_json, response_json
                    FROM proposals
                    WHERE idempotency_key = ?
                    """,
                    (idempotency_key,),
                ).fetchone()
                if not row:
                    return None
                return {
                    "proposal_id": row["proposal_id"],
                    "idempotency_key": row["idempotency_key"],
                    "status": row["status"],
                    "payload": json.loads(row["payload_json"]),
                    "response": json.loads(row["response_json"]) if row["response_json"] else None,
                }
            finally:
                conn.close()

    def upsert_proposal(
        self,
        proposal_id: str,
        idempotency_key: str,
        status: str,
        payload: dict,
        response: Optional[dict] = None,
    ):
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO proposals(
                        proposal_id, idempotency_key, created_at, status, payload_json, response_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(proposal_id) DO UPDATE SET
                        status = excluded.status,
                        payload_json = excluded.payload_json,
                        response_json = excluded.response_json
                    """,
                    (
                        proposal_id,
                        idempotency_key,
                        _utc_now_iso(),
                        status,
                        json.dumps(payload, default=str, sort_keys=True),
                        json.dumps(response, default=str, sort_keys=True) if response else None,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def list_proposals(self) -> Dict[str, Dict]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT proposal_id, status, payload_json, response_json, created_at
                    FROM proposals
                    ORDER BY created_at DESC
                    """
                ).fetchall()
                out: Dict[str, Dict] = {}
                for row in rows:
                    payload = json.loads(row["payload_json"])
                    payload["status"] = row["status"]
                    payload["created_at"] = row["created_at"]
                    if row["response_json"]:
                        payload["response"] = json.loads(row["response_json"])
                    out[row["proposal_id"]] = payload
                return out
            finally:
                conn.close()

    def _append(self, table: str, payload: dict):
        local_id = str(uuid.uuid4())
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    f"INSERT INTO {table}(local_id, created_at, payload_json) VALUES (?, ?, ?)",
                    (local_id, _utc_now_iso(), json.dumps(payload, default=str, sort_keys=True)),
                )
                conn.commit()
            finally:
                conn.close()

    def add_order(self, payload: dict):
        self._append("orders", payload)

    def add_position(self, payload: dict):
        self._append("positions", payload)

    def list_orders(self) -> List[Dict]:
        return self._list_table("orders")

    def list_positions(self) -> List[Dict]:
        return self._list_table("positions")

    def _list_table(self, table: str) -> List[Dict]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    f"SELECT local_id, created_at, payload_json FROM {table} ORDER BY created_at DESC"
                ).fetchall()
                result = []
                for row in rows:
                    payload = json.loads(row["payload_json"])
                    payload["_local_id"] = row["local_id"]
                    payload["_created_at"] = row["created_at"]
                    result.append(payload)
                return result
            finally:
                conn.close()

    def clear_positions(self):
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("DELETE FROM positions")
                conn.commit()
            finally:
                conn.close()


class TradingState:
    def __init__(self, store: SqliteStore):
        self.store = store
        self.start_time = datetime.now(timezone.utc)
        self.is_active = self.store.get_state_bool("is_active", True)
        self.kill_switch_active = self.store.get_state_bool("kill_switch_active", False)

    def set_active(self, value: bool):
        self.is_active = bool(value)
        self.store.set_state_bool("is_active", self.is_active)

    def set_kill_switch(self, value: bool):
        self.kill_switch_active = bool(value)
        self.store.set_state_bool("kill_switch_active", self.kill_switch_active)


store = SqliteStore(DB_PATH)
state = TradingState(store)
kalshi = KalshiClient()


def on_breaker_trip(failure_count: int):
    msg = f"circuit_breaker_tripped failures={failure_count}"
    logger.critical(msg)
    try:
        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, notify.send_telegram_alert_sync, msg)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("circuit_breaker_alert_failed err=%s", exc)


breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60, on_trip=on_breaker_trip)


def _response(success: bool, message: str, data: Optional[dict] = None) -> dict:
    return {"success": bool(success), "message": message, "data": data}


def _proposal_dict(proposal: TradeProposal, proposal_id: str, idempotency_key: str) -> dict:
    payload = proposal.model_dump()
    payload["proposal_id"] = proposal_id
    payload["idempotency_key"] = idempotency_key
    payload["created_at"] = payload.get("created_at") or _utc_now_iso()
    payload["edge"] = _safe_float(payload.get("edge"), 0.0)
    payload["model_prob"] = _safe_float(payload.get("model_prob"), 0.5)
    payload["market_prob"] = _safe_float(payload.get("market_prob"), 0.5)
    payload["strike"] = _safe_float(payload.get("strike"), 0.0)
    payload["action"] = str(payload.get("action", "")).upper()
    return payload


def _ensure_idempotency(proposal: TradeProposal) -> str:
    if proposal.idempotency_key:
        return proposal.idempotency_key
    return build_idempotency_key(
        symbol=proposal.symbol,
        ticker=proposal.ticker or "",
        action=proposal.action,
        strike=proposal.strike,
        model_prob=proposal.model_prob,
        market_prob=proposal.market_prob,
    )


@app.get("/status", response_model=StatusResponse)
async def get_status() -> Dict[str, Any]:
    proposals = store.list_proposals()
    positions = store.list_positions()
    orders = store.list_orders()
    return {
        "status": "running" if state.is_active else "paused",
        "active": state.is_active,
        "is_paused": not state.is_active,
        "kill_switch_active": state.kill_switch_active,
        "proposals_count": len(proposals),
        "positions_count": len(positions),
        "orders_count": len(orders),
        "timestamp": datetime.now(timezone.utc),
        "version": "1.1.0",
    }


@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "healthy"}


@app.get("/positions")
async def get_positions() -> List[Dict]:
    return store.list_positions()


@app.get("/orders")
async def get_orders() -> List[Dict]:
    return store.list_orders()


@app.get("/proposals")
async def get_proposals() -> Dict[str, Dict]:
    return store.list_proposals()


@app.post("/propose", response_model=ActionResponse)
async def propose_trade(proposal: TradeProposal) -> Dict[str, Any]:
    if not state.is_active:
        return _response(False, "Trading Core is paused.")
    if state.kill_switch_active:
        return _response(False, "Kill switch active; refusing new proposals.")

    proposal_id = proposal.proposal_id or str(uuid.uuid4())[:8]
    idempotency_key = _ensure_idempotency(proposal)

    existing = store.proposal_by_idempotency(idempotency_key)
    if existing:
        if existing.get("response"):
            logger.info(
                "duplicate_proposal proposal_id=%s idempotency_key=%s signal_id=%s",
                existing.get("proposal_id"),
                idempotency_key,
                proposal.signal_id,
            )
            return existing["response"]
        # Reuse the original row if a prior write exists without a response.
        proposal_id = existing["proposal_id"]

    payload = _proposal_dict(proposal, proposal_id, idempotency_key)
    store.upsert_proposal(
        proposal_id=proposal_id,
        idempotency_key=idempotency_key,
        status="received",
        payload=payload,
    )
    logger.info(
        "proposal_received proposal_id=%s idempotency_key=%s signal_id=%s symbol=%s action=%s edge=%.4f",
        proposal_id,
        idempotency_key,
        payload.get("signal_id"),
        payload.get("symbol"),
        payload.get("action"),
        payload.get("edge"),
    )

    if not payload.get("ticker"):
        resp = _response(False, f"Proposal {proposal_id} has no ticker. Cannot execute.")
        store.upsert_proposal(proposal_id, idempotency_key, "invalid", payload, resp)
        return resp

    if abs(payload["edge"]) < MIN_EDGE:
        resp = _response(
            True,
            f"Proposal {proposal_id} edge too small ({payload['edge']:.1%}). Skipping execution.",
        )
        store.upsert_proposal(proposal_id, idempotency_key, "skipped_small_edge", payload, resp)
        return resp

    side = "no" if "NO" in payload["action"] else "yes"

    try:
        loop = asyncio.get_running_loop()

        def call_kalshi():
            return kalshi.place_order(
                ticker=payload["ticker"],
                action="buy",
                side=side,
                count=1,
                client_order_id=proposal_id,
            )

        order_result = await breaker.call(loop.run_in_executor, None, call_kalshi)
        if not order_result:
            resp = _response(False, "Order execution failed (API returned None).")
            store.upsert_proposal(proposal_id, idempotency_key, "execution_failed", payload, resp)
            return resp

        order_record = {
            "proposal_id": proposal_id,
            "idempotency_key": idempotency_key,
            "signal_id": payload.get("signal_id"),
            "order": order_result,
        }
        store.add_order(order_record)
        store.add_position(order_record)
        resp = _response(
            True,
            f"Executed {side.upper()} on {payload['ticker']} (Order ID: {order_result.get('order_id')})",
            order_result,
        )
        store.upsert_proposal(proposal_id, idempotency_key, "executed", payload, resp)
        logger.info(
            "proposal_executed proposal_id=%s idempotency_key=%s signal_id=%s order_id=%s",
            proposal_id,
            idempotency_key,
            payload.get("signal_id"),
            order_result.get("order_id"),
        )
        return resp
    except CircuitOpenError as exc:
        resp = _response(False, f"Circuit Breaker OPEN: {exc}")
        store.upsert_proposal(proposal_id, idempotency_key, "circuit_open", payload, resp)
        return resp
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("proposal_execution_error proposal_id=%s err=%s", proposal_id, exc)
        resp = _response(False, f"Execution error: {exc}")
        store.upsert_proposal(proposal_id, idempotency_key, "execution_error", payload, resp)
        return resp


@app.post("/approve/{proposal_id}", response_model=ActionResponse)
async def approve_trade(proposal_id: str) -> Dict[str, Any]:
    proposals = store.list_proposals()
    if proposal_id not in proposals:
        raise HTTPException(status_code=404, detail="Proposal not found")
    return _response(False, "Manual approval path is deprecated; submit via /propose.")


@app.post("/kill-switch", response_model=ActionResponse)
async def kill_switch() -> Dict[str, Any]:
    state.set_active(False)
    state.set_kill_switch(True)
    logger.warning("kill_switch_activated")
    return _response(True, "All trading halted. System safety engaged.")


@app.post("/pause", response_model=ActionResponse)
async def pause_trading() -> Dict[str, Any]:
    state.set_active(False)
    return _response(True, "Trading paused.")


@app.post("/resume", response_model=ActionResponse)
async def resume_trading() -> Dict[str, Any]:
    state.set_active(True)
    state.set_kill_switch(False)
    return _response(True, "Trading resumed.")


@app.post("/flatten", response_model=ActionResponse)
async def flatten_positions() -> Dict[str, Any]:
    store.clear_positions()
    return _response(True, "All positions flattened.")

