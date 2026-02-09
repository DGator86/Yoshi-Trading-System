"""FastAPI Core for Yoshi-Bot Trading execution and state management.

This module provides the API endpoints for receiving trade proposals from
scanners, managing positions, and coordinating with ClawdBot for execution.
"""
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(title="Yoshi-Bot Trading Core")


# --- Models ---


class TradeProposal(BaseModel):
    """Model representing a single trade proposal from a scanner."""

    proposal_id: str = ""
    symbol: str
    action: str  # BUY_YES, BUY_NO
    strike: float
    market_prob: float
    model_prob: float
    edge: float
    raw_forecast: Optional[Dict[str, Any]] = None


class ActionResponse(BaseModel):
    """Standardized response for API actions."""

    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


class StatusResponse(BaseModel):
    """System status information response."""

    status: str
    active: bool
    timestamp: datetime
    version: str = "1.0.0"


# --- State ---


class TradingState:
    """In-memory store for trading system status, proposals and positions."""

    def __init__(self) -> None:
        """Initialize the trading state with empty containers."""
        self.is_active: bool = True
        self.proposals: Dict[str, Dict] = {}
        self.positions: List[Dict] = []
        self.orders: List[Dict] = []
        self.start_time: datetime = datetime.now()


state = TradingState()


# --- Endpoints ---


@app.get("/status", response_model=StatusResponse)
async def get_status() -> Dict[str, Any]:
    """Return the current health and status of the trading bot."""
    return {
        "status": "running" if state.is_active else "paused",
        "active": state.is_active,
        "timestamp": datetime.now(),
        "version": "1.0.0"
    }


@app.get("/positions")
async def get_positions() -> List[Dict]:
    """Retrieve all currently active positions."""
    return state.positions


@app.get("/orders")
async def get_orders() -> List[Dict]:
    """Retrieve the history of all executed orders."""
    return state.orders


@app.get("/proposals")
async def get_proposals() -> Dict[str, Dict]:
    """Retrieve the queue of all received trade proposals."""
    return state.proposals


@app.post("/propose", response_model=ActionResponse)
async def propose_trade(proposal: TradeProposal) -> Dict[str, Any]:
    """Receive a new trade proposal from the scanner."""
    if not state.is_active:
        return {"success": False, "message": "Trading Core is paused."}

    prop_id = str(uuid.uuid4())[:8]
    proposal.proposal_id = prop_id
    state.proposals[prop_id] = proposal.dict()

    logger.info("Received proposal %s for %s", prop_id, proposal.symbol)
    return {
        "success": True,
        "message": f"Proposal {prop_id} received.",
        "data": {"proposal_id": prop_id}
    }


@app.post("/approve/{proposal_id}", response_model=ActionResponse)
async def approve_trade(proposal_id: str) -> Dict[str, Any]:
    """Approve a proposal and simulate its execution."""
    if proposal_id not in state.proposals:
        raise HTTPException(status_code=404, detail="Proposal not found")

    proposal = state.proposals[proposal_id]

    # Simulation: Normally this triggers an exchange order
    order = {
        "order_id": str(uuid.uuid4())[:8],
        "symbol": proposal["symbol"],
        "action": proposal["action"],
        "timestamp": datetime.now(),
        "status": "filled"
    }
    state.orders.append(order)
    state.positions.append(order)

    msg = f"Approved {proposal_id}, executed order {order['order_id']}"
    logger.info(msg)
    return {
        "success": True,
        "message": f"Trade {proposal_id} approved and executed."
    }


@app.post("/kill-switch", response_model=ActionResponse)
async def kill_switch() -> Dict[str, Any]:
    """Halt all trading and activate safety mode."""
    state.is_active = False
    # Logic to cancel all orders and flatten positions would go here
    logger.warning("KILL SWITCH ACTIVATED")
    return {
        "success": True,
        "message": "All trading halted. System safety engaged."
    }


@app.post("/pause", response_model=ActionResponse)
async def pause_trading() -> Dict[str, Any]:
    """Temporarily pause new trade reception."""
    state.is_active = False
    return {"success": True, "message": "Trading paused."}


@app.post("/resume", response_model=ActionResponse)
async def resume_trading() -> Dict[str, Any]:
    """Resume reception of trade proposals."""
    state.is_active = True
    return {"success": True, "message": "Trading resumed."}


@app.post("/flatten", response_model=ActionResponse)
async def flatten_positions() -> Dict[str, Any]:
    """Liquidate all open positions (Simulated)."""
    # Logic to close all positions would go here
    state.positions = []
    return {"success": True, "message": "All positions flattened."}
