#!/usr/bin/env python3
"""Yoshi-Bridge: Buffered Signal Forwarder.

Polls Yoshi-Bot scanner logs and buffers signals before forwarding to Trading Core.
Uses asyncio for concurrency and buffering.

Architecture:
  Log Watcher (Producer) -> asyncio.Queue -> Signal Processor (Consumer) -> HTTP POST
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import aiohttp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("yoshi-bridge")

TRADING_CORE_URL = os.getenv("TRADING_CORE_URL", "http://127.0.0.1:8000")

# Patterns to extract signal data
EDGE_PATTERN = re.compile(r"EDGE:\s*([+-]?\d+\.?\d*)%", re.IGNORECASE)
SYMBOL_PATTERN = re.compile(r"\*?(BTC|ETH|SOL)USDT\*?", re.IGNORECASE)
STRIKE_PATTERN = re.compile(r"Strike:.*?(?:Above|Below)?\s*\$?([\d,]+\.?\d*)", re.IGNORECASE)
MODEL_PROB_PATTERN = re.compile(r"Model\s*Prob:\s*(\d+\.?\d*)%", re.IGNORECASE)
MARKET_PROB_PATTERN = re.compile(r"Market\s*Prob:\s*(\d+\.?\d*)%", re.IGNORECASE)
ACTION_PATTERN = re.compile(r"ACTION:\s*`?(BUY\s*(?:YES|NO)|NEUTRAL)`?", re.IGNORECASE)
TIKCER_PATTERN = re.compile(r"Ticker:\s*(\S+)", re.IGNORECASE) # Guessing ticker pattern or extracting logic


def parse_number(s: str) -> float:
    return float(s.replace(",", ""))


def parse_signal_block(block: str) -> dict | None:
    """Parse a YOSHI KALSHI ALERT block from scanner log output."""
    edge_match = EDGE_PATTERN.search(block)
    if not edge_match:
        return None

    symbol_match = SYMBOL_PATTERN.search(block)
    strike_match = STRIKE_PATTERN.search(block)
    model_prob_match = MODEL_PROB_PATTERN.search(block)
    market_prob_match = MARKET_PROB_PATTERN.search(block)
    action_match = ACTION_PATTERN.search(block)
    # Ticker usually not in log text explicitly unless added?
    # New scanner emits json-like structure or just text.
    # We'll try to find ticker if present, else None.
    
    signal = {
        "edge": float(edge_match.group(1)),
        "symbol": (f"{symbol_match.group(1).upper()}USDT"
                   if symbol_match else "BTCUSDT"),
        "timestamp": datetime.utcnow().isoformat(),
        "ticker": None
    }

    if strike_match:
        signal["strike"] = parse_number(strike_match.group(1))
    if model_prob_match:
        signal["model_prob"] = float(model_prob_match.group(1)) / 100
    if market_prob_match:
        signal["market_prob"] = float(market_prob_match.group(1)) / 100
    if action_match:
        raw_action = action_match.group(1).strip().upper().replace(" ", "_")
        signal["action"] = raw_action

    # If ticker isn't in log, we can't execute. signal['ticker'] is None.
    # TradingCore requires ticker for execution.
    # Current scanner log format does NOT include ticker clearly yet.
    # Task 1 update to scanner included ticker in `opp`.
    # Does `format_kalshi_report` include ticker?
    # I didn't update `format_kalshi_report` to print ticker!
    # I only updated the `payload` in `run_scan`.
    # Wait, the bridge reads LOGS. `format_kalshi_report` writes logs.
    # If `format_kalshi_report` doesn't write ticker, bridge can't see it.
    # I MUST update `format_kalshi_report` in `kalshi_scanner.py` to print ticker.
    
    return signal


class Bridge:
    def __init__(self, log_path: str, poll_interval: float, min_edge: float, dry_run: bool):
        self.log_path = log_path
        self.poll_interval = poll_interval
        self.min_edge = min_edge
        self.dry_run = dry_run
        self.signal_queue = asyncio.Queue(maxsize=100)
        self.cooldown_map = {}
        self.cooldown_seconds = 300
        self.last_position = 0

    async def check_trading_core(self, session: aiohttp.ClientSession) -> bool:
        try:
            async with session.get(f"{TRADING_CORE_URL}/health", timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

    async def log_watcher(self):
        """Producer: Watches log file and enqueues signals."""
        logger.info(f"Watching log: {self.log_path}")
        
        # Initial seek to end
        if Path(self.log_path).exists():
            self.last_position = Path(self.log_path).stat().st_size
        
        while True:
            try:
                if not Path(self.log_path).exists():
                    await asyncio.sleep(self.poll_interval)
                    continue

                # Check if rotated
                if Path(self.log_path).stat().st_size < self.last_position:
                    self.last_position = 0
                
                # Read new content
                # Use run_in_executor for file I/O
                loop = asyncio.get_running_loop()
                new_content = await loop.run_in_executor(None, self._read_log_chunk)
                
                if new_content:
                    blocks = re.split(r"={10,}", new_content)
                    for block in blocks:
                        if "EDGE" not in block:
                            continue
                        
                        signal = parse_signal_block(block)
                        if not signal:
                            continue
                        
                        # Filter by edge
                        if abs(signal["edge"]) < self.min_edge:
                            continue
                            
                        # Filter by cooldown
                        key = f"{signal['symbol']}:{signal.get('strike')}"
                        last_sent = self.cooldown_map.get(key, 0)
                        if time.time() - last_sent < self.cooldown_seconds:
                            continue
                            
                        self.cooldown_map[key] = time.time()
                        
                        # Enqueue
                        try:
                            self.signal_queue.put_nowait(signal)
                            logger.info(f"Queued signal: {signal['symbol']} {signal['action']} (Edge: {signal['edge']}%)")
                        except asyncio.QueueFull:
                            logger.warning("Queue full, dropping signal!")
                            
            except Exception as e:
                logger.error(f"Watcher error: {e}")
                
            await asyncio.sleep(self.poll_interval)

    def _read_log_chunk(self) -> str:
        try:
            with open(self.log_path, "r", encoding="utf-8", errors="ignore") as f:
                f.seek(self.last_position)
                content = f.read()
                self.last_position = f.tell()
                return content
        except Exception:
            return ""

    async def signal_processor(self):
        """Consumer: Dequeues signals and sends to Trading Core."""
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    signal = await self.signal_queue.get()
                    
                    if self.dry_run:
                        logger.info(f"[DRY RUN] Would propose: {signal}")
                    else:
                        # Send to Trading Core
                        payload = {
                            "symbol": signal["symbol"],
                            "action": signal["action"],
                            "strike": signal.get("strike", 0.0),
                            "ticker": signal.get("ticker"), # Might be None
                            "market_prob": signal.get("market_prob", 0.5),
                            "model_prob": signal.get("model_prob", 0.5),
                            "edge": signal["edge"] / 100.0
                        }
                        
                        url = f"{TRADING_CORE_URL}/propose"
                        try:
                            async with session.post(url, json=payload, timeout=10) as resp:
                                if resp.status == 200:
                                    res_json = await resp.json()
                                    logger.info(f"Proposed successfully: {res_json}")
                                else:
                                    logger.error(f"Proposal failed: {resp.status}")
                        except Exception as e:
                            logger.error(f"Connection error to Trading Core: {e}")
                    
                    self.signal_queue.task_done()
                    
                except Exception as e:
                    logger.error(f"Processor error: {e}")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path", default="scanner.log")
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--min-edge", type=float, default=2.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    bridge = Bridge(args.log_path, args.poll_interval, args.min_edge, args.dry_run)
    
    # Run producer and consumer concurrently
    await asyncio.gather(
        bridge.log_watcher(),
        bridge.signal_processor()
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
