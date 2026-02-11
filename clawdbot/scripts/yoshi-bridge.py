#!/usr/bin/env python3
"""Yoshi-Bridge: durable proposal forwarder (outbox -> Trading Core).

This replaces the fragile "parse scanner.log" integration.

Flow:
  kalshi_scanner.py enqueues proposals to an outbox directory as JSON files.
  yoshi-bridge.py forwards pending outbox files to Trading Core /propose and
  archives them to sent/.

Env:
  - TRADING_CORE_URL (default: http://127.0.0.1:8000)
  - YOSHI_OUTBOX_DIR (default: data/outbox)

Legacy CLI flags (accepted for backwards compatibility):
  --log-path, --min-edge (ignored)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("yoshi-bridge")


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _recover_stuck_sending(pending_dir: Path, max_age_sec: int = 300) -> int:
    """Return number of .sending files reverted back to .json."""
    now = int(time.time())
    recovered = 0
    for p in pending_dir.glob("*.sending"):
        try:
            age = now - int(p.stat().st_mtime)
            if age < int(max_age_sec):
                continue
            p.replace(p.with_suffix(".json"))
            recovered += 1
        except Exception:
            continue
    return recovered


def _check_trading_core(trading_core_url: str, timeout_s: float = 3.0) -> bool:
    try:
        resp = requests.get(f"{trading_core_url.rstrip('/')}/health", timeout=timeout_s)
        return resp.status_code == 200
    except Exception:
        return False


def flush_outbox_once(
    trading_core_url: str,
    outbox_root: Path,
    max_send: int = 50,
    timeout_s: float = 5.0,
    dry_run: bool = False,
) -> dict[str, Any]:
    pending_dir = outbox_root / "proposals"
    sent_dir = outbox_root / "sent"
    _safe_mkdir(pending_dir)
    _safe_mkdir(sent_dir)

    _recover_stuck_sending(pending_dir)

    base = trading_core_url.rstrip("/")
    url = f"{base}/propose"

    pending = sorted(pending_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    sent = 0
    failed = 0
    errors: list[str] = []

    for p in pending[: max(int(max_send), 1)]:
        sending = p.with_suffix(".sending")
        try:
            p.replace(sending)  # claim
        except Exception:
            continue

        try:
            with open(sending, encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            failed += 1
            errors.append(f"{sending.name}: read_error: {exc}")
            try:
                sending.replace(p)
            except Exception:
                pass
            continue

        if dry_run:
            sent += 1
            try:
                sending.replace(sent_dir / sending.with_suffix(".json").name)
            except Exception:
                pass
            continue

        try:
            resp = requests.post(url, json=payload, timeout=timeout_s)
            if resp.status_code != 200:
                failed += 1
                errors.append(f"{sending.name}: http_{resp.status_code}: {resp.text[:200]}")
                sending.replace(p)
                continue

            try:
                body = resp.json()
            except Exception as exc:  # pylint: disable=broad-except
                failed += 1
                errors.append(f"{sending.name}: bad_json: {exc}")
                sending.replace(p)
                continue

            if body.get("success", True) is True:
                sent += 1
                sending.replace(sent_dir / sending.with_suffix(".json").name)
            else:
                failed += 1
                errors.append(f"{sending.name}: rejected: {body.get('message')}")
                sending.replace(p)
        except Exception as exc:  # pylint: disable=broad-except
            failed += 1
            errors.append(f"{sending.name}: post_error: {exc}")
            try:
                sending.replace(p)
            except Exception:
                pass

    return {
        "pending_total": int(len(pending)),
        "sent": int(sent),
        "failed": int(failed),
        "errors": errors[:10],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Yoshi Bridge (outbox forwarder)")
    parser.add_argument("--poll-interval", type=float, default=10.0, help="Seconds between flush cycles")
    parser.add_argument("--max-send", type=int, default=50, help="Max proposals to forward per cycle")
    parser.add_argument("--timeout", type=float, default=5.0, help="HTTP timeout seconds")
    parser.add_argument("--dry-run", action="store_true", help="Archive without posting to Trading Core")
    parser.add_argument("--once", action="store_true", help="Run one flush cycle and exit")

    # Backwards compatible (ignored): previously used for log parsing.
    parser.add_argument("--log-path", default="scanner.log")
    parser.add_argument("--min-edge", type=float, default=2.0)

    args = parser.parse_args()

    trading_core_url = os.getenv("TRADING_CORE_URL", "http://127.0.0.1:8000")
    outbox_root = Path(os.getenv("YOSHI_OUTBOX_DIR", "data/outbox"))

    if args.log_path or args.min_edge:
        # These are accepted so older deploy scripts keep working.
        logger.info("Running in outbox mode (log parsing deprecated). outbox=%s", outbox_root)

    while True:
        if not _check_trading_core(trading_core_url, timeout_s=min(3.0, float(args.timeout))):
            logger.warning("Trading Core not healthy; will retry.")
        else:
            res = flush_outbox_once(
                trading_core_url=trading_core_url,
                outbox_root=outbox_root,
                max_send=int(args.max_send),
                timeout_s=float(args.timeout),
                dry_run=bool(args.dry_run),
            )
            if res.get("sent") or res.get("failed"):
                logger.info("Outbox flush: %s", res)

        if args.once:
            return 0
        time.sleep(max(float(args.poll_interval), 0.2))


if __name__ == "__main__":
    raise SystemExit(main())
