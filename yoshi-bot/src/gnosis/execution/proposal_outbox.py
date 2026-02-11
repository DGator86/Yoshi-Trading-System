"""Durable outbox for trade proposals (at-least-once delivery, idempotent IDs).

Why:
  - Scanners/bridges should not lose proposals when Trading Core is restarting.
  - We want safe retries without double-executing orders.

How:
  - Each proposal is written as a JSON file to an outbox directory.
  - A flusher attempts HTTP POST to Trading Core /propose.
  - On success, the file is archived to a `sent/` directory.

Important:
  - Trading Core must treat `proposal_id` as idempotency key.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import requests


def _now_epoch_s() -> int:
    return int(time.time())


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    tmp.replace(path)


@dataclass
class OutboxConfig:
    root_dir: str = "data/outbox"
    pending_dirname: str = "proposals"
    sent_dirname: str = "sent"
    stuck_sending_age_sec: int = 300


class ProposalOutbox:
    def __init__(self, config: Optional[OutboxConfig] = None):
        self.config = config or OutboxConfig()
        self.root = Path(self.config.root_dir)
        self.pending = self.root / self.config.pending_dirname
        self.sent = self.root / self.config.sent_dirname
        self.pending.mkdir(parents=True, exist_ok=True)
        self.sent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _ensure_proposal_id(payload: dict[str, Any]) -> str:
        pid = str(payload.get("proposal_id") or "").strip()
        if not pid:
            pid = uuid.uuid4().hex[:12]
            payload["proposal_id"] = pid
        return pid

    def enqueue(self, payload: dict[str, Any]) -> Path:
        """Write proposal to pending outbox and return path."""
        payload = dict(payload)
        pid = self._ensure_proposal_id(payload)
        payload.setdefault("enqueued_at_epoch_s", _now_epoch_s())
        path = self.pending / f"{pid}.json"
        _atomic_write_json(path, payload)
        return path

    def _recover_stuck_sending(self) -> int:
        """Return count of recovered .sending files."""
        recovered = 0
        now = _now_epoch_s()
        for p in self.pending.glob("*.sending"):
            try:
                age = now - int(p.stat().st_mtime)
                if age < int(self.config.stuck_sending_age_sec):
                    continue
                p.replace(p.with_suffix(".json"))
                recovered += 1
            except Exception:
                continue
        return recovered

    def flush(
        self,
        trading_core_url: str,
        max_send: int = 50,
        timeout_s: float = 5.0,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Attempt to deliver pending proposals to Trading Core.

        Returns:
            dict with counts and errors.
        """
        self._recover_stuck_sending()

        base = str(trading_core_url).rstrip("/")
        url = f"{base}/propose"

        sent = 0
        failed = 0
        errors: list[str] = []

        # Oldest first to preserve time ordering.
        pending = sorted(self.pending.glob("*.json"), key=lambda p: p.stat().st_mtime)
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
                # Return to pending for future inspection.
                try:
                    sending.replace(p)
                except Exception:
                    pass
                continue

            if dry_run:
                sent += 1
                archive = self.sent / sending.with_suffix(".json").name
                try:
                    sending.replace(archive)
                except Exception:
                    pass
                continue

            try:
                resp = requests.post(url, json=payload, timeout=timeout_s)
                if resp.status_code != 200:
                    failed += 1
                    errors.append(f"{sending.name}: http_{resp.status_code}: {resp.text[:200]}")
                    # Put back for retry later.
                    sending.replace(p)
                    continue

                # Prefer application-level success to avoid dropping while paused.
                try:
                    body = resp.json()
                except Exception as exc:  # pylint: disable=broad-except
                    failed += 1
                    errors.append(f"{sending.name}: bad_json: {exc}")
                    sending.replace(p)
                    continue

                ok = body.get("success", True)
                if ok is True:
                    sent += 1
                    archive = self.sent / sending.with_suffix(".json").name
                    sending.replace(archive)
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

