#!/usr/bin/env python3
"""Discover TELEGRAM_CHAT_ID via getUpdates and optionally write to .env.

Usage:
  # 1) Send any message to your bot in Telegram
  # 2) Run:
  python3 scripts/set_telegram_chat_id.py

  # Write it into .env (in-place):
  python3 scripts/set_telegram_chat_id.py --write --env-path /root/Yoshi-Trading-System/.env
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Optional

import requests

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass


def _get_updates(token: str) -> dict[str, Any]:
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    return resp.json()


def _extract_chat_id(payload: dict[str, Any]) -> Optional[int]:
    if not payload.get("ok"):
        return None
    results = payload.get("result") or []
    if not results:
        return None
    # Prefer the latest update.
    results = sorted(results, key=lambda r: int(r.get("update_id", 0)))
    for res in reversed(results):
        msg = res.get("message") or res.get("edited_message") or {}
        chat = msg.get("chat") or {}
        chat_id = chat.get("id")
        if isinstance(chat_id, int):
            return chat_id
        if isinstance(chat_id, str) and chat_id.lstrip("-").isdigit():
            return int(chat_id)
    return None


def _upsert_env_var(env_path: Path, key: str, value: str) -> None:
    lines: list[str] = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    out: list[str] = []
    found = False
    for line in lines:
        if line.startswith(f"{key}="):
            out.append(f"{key}={value}")
            found = True
        else:
            out.append(line)
    if not found:
        if out and out[-1].strip() != "":
            out.append("")
        out.append(f"{key}={value}")

    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Set TELEGRAM_CHAT_ID from Telegram updates")
    parser.add_argument("--write", action="store_true", help="Write TELEGRAM_CHAT_ID into env file")
    parser.add_argument("--env-path", type=Path, default=Path(".env"), help="Env file path (default: ./.env)")
    args = parser.parse_args()

    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise SystemExit("TELEGRAM_BOT_TOKEN is not set. Put it in .env or export it.")

    payload = _get_updates(token)
    chat_id = _extract_chat_id(payload)
    if chat_id is None:
        print("No chat id found yet. Send a message to your bot in Telegram, then rerun.")
        return 2

    print(f"TELEGRAM_CHAT_ID={chat_id}")
    if args.write:
        _upsert_env_var(args.env_path, "TELEGRAM_CHAT_ID", str(chat_id))
        try:
            os.chmod(args.env_path, 0o600)
        except Exception:
            pass
        print(f"Wrote TELEGRAM_CHAT_ID to {args.env_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

