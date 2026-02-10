"""FastAPI Core for Yoshi-Bot Trading execution and state management.

This module provides the API endpoints for receiving trade proposals from
scanners, managing positions, and coordinating with ClawdBot for execution.
"""
import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from functools import partial

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.gnosis.utils.kalshi_client import KalshiClient
from src.gnosis.execution.circuit_breaker import CircuitBreaker, CircuitOpenError
import src.gnosis.utils.notifications as notify

logger = logging.getLogger(__name__)

app = FastAPI(title="Yoshi-Bot Trading Core")


# --- Models ---


class TradeProposal(BaseModel):
    """Model representing a single trade proposal from a scanner."""

    proposal_id: str = ""
    symbol: str
    ticker: Optional[str] = None  # Kalshi Ticker (e.g. KXBTC-...)
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
kalshi = KalshiClient()


def on_breaker_trip(failure_count: int):
    """Callback when circuit breaker trips."""
    msg = f"⚠️ CIRCUIT BREAKER TRIPPED after {failure_count} failures! Kalshi API halted."
    logger.critical(msg)
    try:
        loop = asyncio.get_running_loop()
        # Run sync notification in thread pool to avoid blocking
        loop.run_in_executor(None, notify.send_telegram_alert_sync, msg)
    except Exception as e:
        logger.error(f"Failed to send alert: {e}")


breaker = CircuitBreaker(
    failure_threshold=3,
    recovery_timeout=60,
    on_trip=on_breaker_trip
)


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


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Simple health check endpoint for monitoring."""
    return {"status": "healthy"}


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
    """Receive and EXECUTE a new trade proposal from the scanner."""
    if not state.is_active:
        return {"success": False, "message": "Trading Core is paused."}

    prop_id = str(uuid.uuid4())[:8]
    if not proposal.proposal_id:
        proposal.proposal_id = prop_id
    
    state.proposals[prop_id] = proposal.dict()
    logger.info("Received proposal %s for %s (%s)", prop_id, proposal.symbol, proposal.action)

    # 1. Validation
    if not proposal.ticker:
        msg = f"Proposal {prop_id} has no ticker. Cannot execute."
        logger.warning(msg)
        return {"success": False, "message": msg}

    if abs(proposal.edge) < 0.05:
        # Just log, don't execute if edge is too small (already filtered by scanner though)
        msg = f"Proposal {prop_id} edge too small ({proposal.edge:.1%}). Skipping execution."
        logger.info(msg)
        return {"success": True, "message": msg}

    # 2. Execution via Circuit Breaker
    action = proposal.action.upper()
    side = "yes"
    if "NO" in action:
        # Only support buying YES/NO contracts directly if Kalshi supports it. 
        # Usually buying NO means selling YES or buying NO contracts.
        # Assuming Kalshi v2 'side' param: 'yes' or 'no'
        side = "no"

    count = 1  # Fixed size for now

    try:
        # Wrap sync call in executor
        loop = asyncio.get_running_loop()
        
        # Define the sync call to wrap
        def call_kalshi():
            return kalshi.place_order(
                ticker=proposal.ticker,
                action="buy",
                side=side,
                count=count,
                client_order_id=prop_id
            )

        # Execute via circuit breaker
        order_result = await breaker.call(
            loop.run_in_executor,
            None,
            call_kalshi
        )

        if order_result:
            state.orders.append(order_result)
            state.positions.append(order_result) # Track position
            msg = f"Executed {side.upper()} on {proposal.ticker} (Order ID: {order_result.get('order_id')})"
            logger.info(msg)
            return {
                "success": True,
                "message": msg,
                "data": order_result
            }
        else:
            return {
                "success": False,
                "message": "Order execution failed (API returned None)."
            }

    except CircuitOpenError as e:
        logger.error(f"Circuit Breaker prevented order execution: {e}")
        return {
            "success": False,
            "message": f"Circuit Breaker OPEN: {e}"
        }
    except Exception as e:
        logger.error(f"Execution error for {prop_id}: {e}")
        return {
            "success": False,
            "message": f"Execution error: {str(e)}"
        }


@app.post("/approve/{proposal_id}", response_model=ActionResponse)
async def approve_trade(proposal_id: str) -> Dict[str, Any]:
    """Approve a proposal manually (if not auto-executed)."""
    if proposal_id not in state.proposals:
        raise HTTPException(status_code=404, detail="Proposal not found")
    
    # Manual approval logic implies re-triggering execution handled in /propose
    # For now, keep simulation or move logic to a shared function.
    # User instructions focus on /propose.
    
    return {"success": False, "message": "Manual approval not yet refactored to use CircuitBreaker."}


@app.post("/kill-switch", response_model=ActionResponse)
async def kill_switch() -> Dict[str, Any]:
    """Halt all trading and activate safety mode."""
    state.is_active = False
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
    state.positions = []
    return {"success": True, "message": "All positions flattened."}
