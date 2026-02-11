#!/usr/bin/env python3
"""Yoshi bridge: structured signal queue -> Trading Core /propose.

Canonical runtime path:
    kalshi_scanner.py --bridge  -> data/signals/scanner_signals.jsonl
    yoshi-bridge.py             -> POST /propose
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import sys
import time
from pathlib import Path

import aiohttp

# Add monorepo root so shared schema module is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.trading_signals import (  # noqa: E402
    SIGNAL_EVENTS_PATH_DEFAULT,
    make_trade_signal,
    parse_event_line,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("yoshi-bridge")

TRADING_CORE_URL = os.getenv("TRADING_CORE_URL", "http://127.0.0.1:8000")


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


_LEGACY_BLOCK_PATTERN = re.compile(
    r"\*(?P<symbol>[A-Z0-9/_-]+)\*.*?"
    r"Ticker:\s*(?P<ticker>[A-Z0-9_-]+).*?"
    r"Strike:\s*\*Above\s*\$(?P<strike>[0-9,]+(?:\.[0-9]+)?)\*.*?"
    r"Market Prob:\s*(?P<market_prob>-?[0-9]+(?:\.[0-9]+)?)%.*?"
    r"Model Prob:\s*(?P<model_prob>-?[0-9]+(?:\.[0-9]+)?)%.*?"
    r"EDGE:\s*(?P<edge>[+-]?[0-9]+(?:\.[0-9]+)?)%.*?"
    r"ACTION:\s*`(?P<action>BUY YES|BUY NO|NEUTRAL)`",
    re.DOTALL,
)


class Bridge:
    """Consume structured scanner events and forward proposals to core."""

    def __init__(
        self,
        signal_path: str,
        poll_interval: float,
        min_edge_pct: float,
        dry_run: bool,
        start_from_beginning: bool = False,
    ):
        self.signal_path = signal_path
        self.poll_interval = float(poll_interval)
        self.min_edge_pct = float(min_edge_pct)
        self.dry_run = bool(dry_run)
        self.start_from_beginning = bool(start_from_beginning)
        self.legacy_mode = not str(signal_path).lower().endswith(".jsonl")

        self.signal_queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=512)
        self.last_position = 0
        self._seen_idempotency: dict[str, float] = {}
        self._seen_ttl_sec = 3600.0
        self._legacy_buffer = ""

    def _should_skip_duplicate(self, idempotency_key: str) -> bool:
        now = time.time()
        # Opportunistic GC.
        expired = [k for k, ts in self._seen_idempotency.items() if now - ts > self._seen_ttl_sec]
        for k in expired:
            self._seen_idempotency.pop(k, None)

        if idempotency_key in self._seen_idempotency:
            return True
        self._seen_idempotency[idempotency_key] = now
        return False

    def _read_signal_chunk(self) -> str:
        path = Path(self.signal_path)
        if not path.exists():
            return ""
        if path.stat().st_size < self.last_position:
            # Rotated/truncated.
            self.last_position = 0
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            handle.seek(self.last_position)
            content = handle.read()
            self.last_position = handle.tell()
            return content

    async def signal_watcher(self):
        """Producer: read JSONL signal events and enqueue valid proposals."""
        logger.info(
            "watching_path path=%s mode=%s",
            self.signal_path,
            "legacy_log" if self.legacy_mode else "jsonl",
        )

        p = Path(self.signal_path)
        if p.exists() and not self.start_from_beginning:
            self.last_position = p.stat().st_size
            logger.info("tail_mode enabled start_pos=%d", self.last_position)

        while True:
            try:
                loop = asyncio.get_running_loop()
                chunk = await loop.run_in_executor(None, self._read_signal_chunk)
                if chunk:
                    if self.legacy_mode:
                        self._legacy_buffer += chunk
                        matches = list(_LEGACY_BLOCK_PATTERN.finditer(self._legacy_buffer))
                        for match in matches:
                            gd = match.groupdict()
                            action = "BUY_YES" if gd["action"] == "BUY YES" else "BUY_NO"
                            signal = make_trade_signal(
                                symbol=gd["symbol"],
                                ticker=gd["ticker"],
                                action=action,
                                strike=float(gd["strike"].replace(",", "")),
                                market_prob=_safe_float(gd["market_prob"], 50.0) / 100.0,
                                model_prob=_safe_float(gd["model_prob"], 50.0) / 100.0,
                                edge=_safe_float(gd["edge"], 0.0) / 100.0,
                                source="yoshi_bridge_legacy_log",
                            ).to_payload()
                            await self._enqueue_signal(signal)
                        if matches:
                            self._legacy_buffer = self._legacy_buffer[matches[-1].end():]
                        elif len(self._legacy_buffer) > 12000:
                            self._legacy_buffer = self._legacy_buffer[-4000:]
                    else:
                        for raw_line in chunk.splitlines():
                            event = parse_event_line(raw_line)
                            if not event:
                                continue
                            await self._enqueue_signal(event["signal"])
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception("watcher_error err=%s", exc)

            await asyncio.sleep(self.poll_interval)

    async def _enqueue_signal(self, signal: dict):
        edge = abs(_safe_float(signal.get("edge"), 0.0))
        edge_pct = edge * 100.0
        if edge_pct < self.min_edge_pct:
            return
        idk = str(signal.get("idempotency_key", ""))
        if not idk:
            logger.warning("drop_signal missing_idempotency signal_id=%s", signal.get("signal_id"))
            return
        if self._should_skip_duplicate(idk):
            logger.info("skip_duplicate idempotency_key=%s", idk)
            return
        try:
            self.signal_queue.put_nowait(signal)
            logger.info(
                "signal_queued signal_id=%s idempotency_key=%s edge_pct=%.2f",
                signal.get("signal_id"),
                idk,
                edge_pct,
            )
        except asyncio.QueueFull:
            logger.warning("queue_full dropping signal_id=%s", signal.get("signal_id"))

    async def signal_processor(self):
        """Consumer: POST queued proposals to Trading Core."""
        async with aiohttp.ClientSession() as session:
            while True:
                signal = await self.signal_queue.get()
                try:
                    payload = {
                        "symbol": signal.get("symbol"),
                        "ticker": signal.get("ticker"),
                        "action": signal.get("action"),
                        "strike": _safe_float(signal.get("strike"), 0.0),
                        "market_prob": _safe_float(signal.get("market_prob"), 0.5),
                        "model_prob": _safe_float(signal.get("model_prob"), 0.5),
                        "edge": _safe_float(signal.get("edge"), 0.0),
                        "idempotency_key": signal.get("idempotency_key"),
                        "signal_id": signal.get("signal_id"),
                        "source": signal.get("source", "yoshi_bridge"),
                        "created_at": signal.get("created_at"),
                    }
                    if self.dry_run:
                        logger.info(
                            "dry_run_propose signal_id=%s idempotency_key=%s",
                            payload.get("signal_id"),
                            payload.get("idempotency_key"),
                        )
                    else:
                        url = f"{TRADING_CORE_URL}/propose"
                        async with session.post(url, json=payload, timeout=10) as resp:
                            body = await resp.text()
                            if resp.status == 200:
                                logger.info(
                                    "proposal_ok status=%d signal_id=%s idempotency_key=%s",
                                    resp.status,
                                    payload.get("signal_id"),
                                    payload.get("idempotency_key"),
                                )
                            else:
                                logger.error(
                                    "proposal_failed status=%d signal_id=%s body=%s",
                                    resp.status,
                                    payload.get("signal_id"),
                                    body[:500],
                                )
                except Exception as exc:  # pylint: disable=broad-except
                    logger.exception("processor_error signal_id=%s err=%s", signal.get("signal_id"), exc)
                finally:
                    self.signal_queue.task_done()


async def main():
    parser = argparse.ArgumentParser(description="Structured signal bridge")
    parser.add_argument("--signal-path", default=SIGNAL_EVENTS_PATH_DEFAULT)
    parser.add_argument("--log-path", default=None, help="Backward-compatible alias for --signal-path")
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--min-edge", type=float, default=2.0, help="Min edge in percent")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--from-start", action="store_true", help="Process existing file from byte 0")
    args = parser.parse_args()
    selected_path = args.log_path or args.signal_path

    bridge = Bridge(
        signal_path=selected_path,
        poll_interval=args.poll_interval,
        min_edge_pct=args.min_edge,
        dry_run=args.dry_run,
        start_from_beginning=args.from_start,
    )

    await asyncio.gather(bridge.signal_watcher(), bridge.signal_processor())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

