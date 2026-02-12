"""
Telegram Bot — Long-polling bot for the trading system.
=========================================================
Uses stdlib urllib (no extra deps) for reliability on VPS.
Runs the orchestrator in a background thread and sends
BUY alerts + cycle reports to the configured chat.

Architecture:
  Main thread:  Telegram long-polling (getUpdates)
  Worker thread: Orchestrator.run_loop() with callback

Commands:
  /scan    — Force a scan cycle now
  /status  — System status + Ralph learning state
  /ralph   — Detailed Ralph learning report
  /params  — Current hyperparameters
  /help    — Command list
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib import request, error as urlerror, parse as urlparse

from gnosis.telegram.formatter import MessageFormatter


# ── Telegram API Client (stdlib only) ─────────────────────────
class TelegramAPI:
    """Minimal Telegram Bot API client using urllib."""

    BASE = "https://api.telegram.org"

    def __init__(self, token: str):
        self.token = token
        self.base_url = f"{self.BASE}/bot{token}"

    def _call(self, method: str, data: dict = None, timeout: int = 30) -> Optional[dict]:
        """Call a Telegram Bot API method."""
        url = f"{self.base_url}/{method}"
        try:
            import requests
            if data:
                resp = requests.post(url, json=data, timeout=timeout)
            else:
                resp = requests.get(url, timeout=timeout)
            
            result = resp.json()
            if result.get("ok"):
                return result.get("result")
            
            if resp.status_code == 400:
                return {"parse_error": True, "description": result.get("description", "")}
            
            print(f"[TG] API Error {resp.status_code}: {resp.text}")
            return None
            
        except ImportError:
            # Fallback to urllib if requests is missing
            body = json.dumps(data).encode() if data else None
            headers = {"Content-Type": "application/json"} if body else {}
            req = request.Request(url, data=body, headers=headers, method="POST" if body else "GET")
            try:
                with request.urlopen(req, timeout=timeout) as resp:
                    result = json.loads(resp.read().decode())
                    if result.get("ok"):
                        return result.get("result")
                    return None
            except urlerror.HTTPError as e:
                try:
                    err_body = e.read().decode()
                    print(f"[TG] API Error {e.code}: {err_body}")
                    err_data = json.loads(err_body)
                    if err_data.get("error_code") == 400:
                        return {"parse_error": True, "description": err_data.get("description", "")}
                except Exception:
                    pass
                return None
            except Exception as e:
                print(f"[TG] Urllib Request Exception: {e}")
                return None
        except Exception as e:
            print(f"[TG] Requests Exception: {e}")
            return None

    def get_me(self) -> Optional[dict]:
        return self._call("getMe")

    def get_updates(self, offset: int = 0, timeout: int = 30) -> List[dict]:
        result = self._call("getUpdates", {
            "offset": offset,
            "timeout": timeout,
            "allowed_updates": ["message"],
        }, timeout=timeout + 5)
        return result if isinstance(result, list) else []

    def send_message(
        self,
        chat_id: str,
        text: str,
        parse_mode: str = "MarkdownV2",
        disable_preview: bool = True,
    ) -> Optional[dict]:
        """Send a message. Falls back to plain text if Markdown fails."""
        data = {
            "chat_id": chat_id,
            "text": text,
            "disable_web_page_preview": disable_preview,
        }
        if parse_mode:
            data["parse_mode"] = parse_mode

        result = self._call("sendMessage", data)

        # If markdown parse failed, retry as plain text
        if isinstance(result, dict) and result.get("parse_error"):
            # Strip markdown and retry
            plain = text
            for ch in r"*_`[]()~>#+-=|{}.!\\":
                plain = plain.replace(f"\\{ch}", ch).replace(ch, "")
            data["text"] = plain
            data.pop("parse_mode", None)
            result = self._call("sendMessage", data)

        return result


# ── Telegram Bot ──────────────────────────────────────────────
class TelegramBot:
    """
    Long-polling Telegram bot integrated with the trading system.

    Usage:
        bot = TelegramBot(token="...", chat_id="...")
        bot.run()  # blocking, handles commands + runs orchestrator

    Or programmatic:
        bot = TelegramBot(token="...", chat_id="...")
        bot.send_text("Hello!")
        bot.send_buy_alert(value_play_dict)
    """

    def __init__(
        self,
        token: str = None,
        chat_id: str = None,
        orchestrator=None,
        auto_scan_interval: float = 60.0,
        notify_on_buy: bool = True,
        notify_on_cycle: bool = False,  # only BUY alerts by default
        conversational: bool = True,
        llm_chat: bool = True,
    ):
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
        self.api = TelegramAPI(self.token) if self.token else None
        self.orchestrator = orchestrator
        self.auto_scan_interval = auto_scan_interval
        self.notify_on_buy = notify_on_buy
        self.notify_on_cycle = notify_on_cycle
        self.conversational = conversational
        self.llm_chat = llm_chat

        self._update_offset = 0
        self._running = False
        self._scan_thread: Optional[threading.Thread] = None
        self._last_cycle_result = None
        self._fmt = MessageFormatter()
        self._chat_history: Dict[str, List[Dict[str, str]]] = {}
        self._llm_client = None

    # ── Sending ───────────────────────────────────────────
    def send_text(self, text: str, parse_mode: str = "MarkdownV2") -> bool:
        """Send a text message to the configured chat."""
        if not self.api or not self.chat_id:
            print(f"[TG] Not configured (token={'set' if self.token else 'missing'}, "
                  f"chat_id={'set' if self.chat_id else 'missing'})")
            return False
        result = self.api.send_message(self.chat_id, text, parse_mode)
        return result is not None and not isinstance(result, dict)

    def send_plain(self, text: str) -> bool:
        """Send plain text (no markdown)."""
        if not self.api or not self.chat_id:
            return False
        result = self.api.send_message(self.chat_id, text, parse_mode="")
        return result is not None

    def send_cycle_report(self, result) -> bool:
        """Send a formatted cycle report."""
        try:
            msg = self._fmt.cycle_report(result)
            ok = self.send_text(msg)
            if not ok:
                # Fallback to plain text
                msg = self._fmt.cycle_report_plain(result)
                return self.send_plain(msg)
            return ok
        except Exception as e:
            return self.send_plain(f"Cycle {getattr(result, 'cycle', '?')} complete ({e})")

    def send_buy_alert(self, value_play: dict) -> bool:
        """Send a BUY signal alert."""
        try:
            msg = self._fmt.buy_alert(value_play)
            ok = self.send_text(msg)
            if not ok:
                msg = self._fmt.buy_alert_plain(value_play)
                return self.send_plain(msg)
            return ok
        except Exception as e:
            scan = value_play.get("scan", {})
            return self.send_plain(f"BUY: {scan.get('ticker','?')} ({e})")

    def send_status(self) -> bool:
        """Send system status."""
        try:
            llm_info = {}
            ralph_summary = {}
            kalshi_status = {}

            try:
                from gnosis.reasoning.client import LLMConfig
                cfg = LLMConfig.from_yaml()
                llm_info = {"environment": cfg._environment, "model": cfg.model}
            except Exception:
                pass

            if self.orchestrator:
                try:
                    ralph_summary = self.orchestrator.ralph.get_learning_summary()
                except Exception:
                    pass

            try:
                from gnosis.kalshi.scanner import KalshiAPIClient, _load_env_files
                _load_env_files()
                client = KalshiAPIClient()
                kalshi_status = client.get_exchange_status() or {}
            except Exception:
                pass

            msg = self._fmt.status_report(llm_info, ralph_summary, kalshi_status)
            ok = self.send_text(msg)
            if not ok:
                return self.send_plain(f"Status: LLM={llm_info} Ralph={ralph_summary}")
            return ok
        except Exception as e:
            return self.send_plain(f"Status error: {e}")

    def send_ralph_report(self) -> bool:
        """Send Ralph learning report."""
        try:
            if self.orchestrator:
                summary = self.orchestrator.ralph.get_learning_summary()
            else:
                from gnosis.ralph.learner import RalphLearner
                summary = RalphLearner().get_learning_summary()

            msg = self._fmt.ralph_report(summary)
            ok = self.send_text(msg)
            if not ok:
                return self.send_plain(f"Ralph: {json.dumps(summary, indent=2, default=str)[:1000]}")
            return ok
        except Exception as e:
            return self.send_plain(f"Ralph error: {e}")

    # ── Command Handling ──────────────────────────────────
    def _handle_command(self, text: str, chat_id: str):
        """Handle an incoming command."""
        cmd = text.strip().split()[0].lower().split("@")[0]  # strip @botname

        if cmd == "/start" or cmd == "/help":
            msg = self._fmt.help_message()
            self.api.send_message(chat_id, msg, "MarkdownV2")

        elif cmd in ("/chatid", "/id"):
            # Phone-friendly: show + persist so other services can send alerts.
            if not self.chat_id:
                self.chat_id = chat_id
            persisted = self._persist_chat_id(self.chat_id)
            suffix = " (saved to .env)" if persisted else ""
            self.api.send_message(chat_id, f"Chat ID: {self.chat_id}{suffix}", "")

        elif cmd == "/status":
            self.send_status()

        elif cmd == "/ralph":
            self.send_ralph_report()

        elif cmd == "/scan":
            self.api.send_message(chat_id, "🔍 Running scan\\.\\.\\.", "MarkdownV2")
            self._force_scan(chat_id)

        elif cmd == "/params":
            self._send_params(chat_id)

        elif cmd in ("/backtest", "/bt"):
            self._reply_backtest(chat_id)

        else:
            self.api.send_message(chat_id, f"Unknown command: {text.split()[0]}\nTry /help", "")

    def _persist_chat_id(self, chat_id: str) -> bool:
        """Persist TELEGRAM_CHAT_ID so other services can send alerts.

        Avoids rewriting the entire .env as key/value lines (some env files
        contain multi-line PEM values). We only upsert the TELEGRAM_CHAT_ID line.
        """
        if not chat_id:
            return False

        os.environ["TELEGRAM_CHAT_ID"] = str(chat_id)

        candidates: list[Path] = []
        explicit = os.environ.get("TELEGRAM_ENV_PATH", "").strip()
        if explicit:
            candidates.append(Path(explicit))

        candidates.extend(
            [
                Path(os.getcwd()) / ".env",
                Path("/root/Yoshi-Trading-System/.env"),
                Path("/root/Yoshi-Bot/.env"),
                Path("/root/ClawdBot-V1/.env"),
                Path("/home/root/Yoshi-Trading-System/.env"),
                Path("/home/root/Yoshi-Bot/.env"),
                Path("/home/root/ClawdBot-V1/.env"),
                Path.home() / ".env",
            ]
        )

        target = next((p for p in candidates if p.exists()), None)
        if target is None:
            # Create in current directory as a last resort.
            target = Path(os.getcwd()) / ".env"

        try:
            self._upsert_env_line(target, "TELEGRAM_CHAT_ID", str(chat_id))
            try:
                target.chmod(0o600)
            except Exception:
                pass
            return True
        except Exception:
            return False

    @staticmethod
    def _upsert_env_line(env_path: Path, key: str, value: str) -> None:
        content = ""
        if env_path.exists():
            content = env_path.read_text(encoding="utf-8", errors="ignore")
        pattern = re.compile(rf"^{re.escape(key)}=.*$", flags=re.MULTILINE)
        if pattern.search(content):
            content = pattern.sub(f"{key}={value}", content)
        else:
            if content and not content.endswith("\n"):
                content += "\n"
            content += f"\n{key}={value}\n"

        env_path.parent.mkdir(parents=True, exist_ok=True)
        env_path.write_text(content, encoding="utf-8")

    def _append_history(self, chat_id: str, role: str, text: str) -> None:
        """Keep a tiny rolling chat history for conversational context."""
        if not chat_id:
            return
        hist = self._chat_history.setdefault(chat_id, [])
        hist.append({"role": role, "text": (text or "")[:600]})
        if len(hist) > 12:
            del hist[:-12]

    def _core_get(self, path: str) -> Any:
        """Read-only helper for Trading Core HTTP endpoints."""
        base = os.environ.get("TRADING_CORE_URL", "http://127.0.0.1:8000").rstrip("/")
        url = f"{base}{path}"
        try:
            import requests
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception:
            return None

    def _reply_positions(self, chat_id: str):
        positions = self._core_get("/positions")
        if not isinstance(positions, list):
            self.api.send_message(chat_id, "Could not fetch positions right now.", "")
            return
        if not positions:
            self.api.send_message(chat_id, "No open positions right now.", "")
            return
        lines = [f"Open positions: {len(positions)}"]
        for p in positions[:5]:
            order = p.get("order", {}) if isinstance(p, dict) else {}
            ticker = order.get("ticker") or p.get("ticker", "?")
            oid = order.get("order_id") or p.get("_local_id", "?")
            lines.append(f"- {ticker} (order_id={oid})")
        self.api.send_message(chat_id, "\n".join(lines), "")

    def _reply_orders(self, chat_id: str):
        orders = self._core_get("/orders")
        if not isinstance(orders, list):
            self.api.send_message(chat_id, "Could not fetch orders right now.", "")
            return
        if not orders:
            self.api.send_message(chat_id, "No orders have executed yet.", "")
            return
        lines = [f"Recent orders: {len(orders)}"]
        for o in orders[:5]:
            order = o.get("order", {}) if isinstance(o, dict) else {}
            lines.append(
                f"- {order.get('ticker', '?')} | order_id={order.get('order_id', '?')}"
            )
        self.api.send_message(chat_id, "\n".join(lines), "")

    def _reply_signals(self, chat_id: str):
        proposals = self._core_get("/proposals")
        if not isinstance(proposals, dict):
            self.api.send_message(chat_id, "Could not fetch proposal queue right now.", "")
            return
        if not proposals:
            self.api.send_message(chat_id, "No proposals yet. Send /scan to create one.", "")
            return
        items = list(proposals.items())[:5]
        lines = [f"Recent proposals: {len(proposals)}"]
        for pid, p in items:
            lines.append(
                f"- {pid}: {p.get('action', '?')} {p.get('ticker', '?')} "
                f"edge={float(p.get('edge', 0.0)):+.1%} status={p.get('status', '?')}"
            )
        self.api.send_message(chat_id, "\n".join(lines), "")

    def _reply_last_prediction(self, chat_id: str):
        if self._last_cycle_result is None:
            self.api.send_message(
                chat_id,
                "I don't have an in-memory cycle snapshot yet. Send /scan and ask again.",
                "",
            )
            return
        d = self._last_cycle_result.to_dict() if hasattr(self._last_cycle_result, "to_dict") else {}
        fc = d.get("forecast", {}) or {}
        if not fc:
            self.api.send_message(chat_id, "Last cycle had no forecast payload.", "")
            return
        msg = (
            f"Latest forecast snapshot ({d.get('timestamp','unknown')}):\n"
            f"- Symbol: {fc.get('symbol','?')}\n"
            f"- Current: ${float(fc.get('current_price',0.0)):,.2f}\n"
            f"- Predicted: ${float(fc.get('predicted_price',0.0)):,.2f}\n"
            f"- Direction: {fc.get('direction','?')} | Conf: {float(fc.get('confidence',0.0)):.1%}\n"
            f"- Regime: {fc.get('regime','?')}"
        )
        self.api.send_message(chat_id, msg, "")

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _signal_data_paths(self, filename: str) -> list[Path]:
        roots = [
            Path(os.getcwd()),
            Path("/root/Yoshi-Trading-System"),
            Path("/root/Yoshi-Bot"),
            Path("/home/root/Yoshi-Trading-System"),
            Path("/home/root/Yoshi-Bot"),
        ]
        # Preserve order while deduplicating.
        seen: set[str] = set()
        paths: list[Path] = []
        for root in roots:
            p = (root / "data" / "signals" / filename).resolve()
            key = str(p)
            if key in seen:
                continue
            seen.add(key)
            paths.append(p)
        return paths

    def _load_recent_signal_outcomes(self, max_rows: int = 200) -> tuple[list[dict[str, Any]], Optional[Path]]:
        for path in self._signal_data_paths("signal_outcomes.jsonl"):
            if not path.exists():
                continue
            rows: list[dict[str, Any]] = []
            try:
                with path.open("r", encoding="utf-8", errors="ignore") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(row, dict):
                            rows.append(row)
                            if len(rows) > max_rows:
                                del rows[: len(rows) - max_rows]
                return rows, path
            except Exception:
                continue
        return [], None

    def _load_json_file(self, filename: str) -> tuple[Optional[dict[str, Any]], Optional[Path]]:
        for path in self._signal_data_paths(filename):
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
                if isinstance(data, dict):
                    return data, path
            except Exception:
                continue
        return None, None

    def _format_backtest_summary(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return (
                "Backtest summary: no resolved signal outcomes yet.\n"
                "Once contracts settle, I’ll report win rate, avg PnL, and adaptive thresholds."
            )

        n = len(rows)
        wins = sum(1 for r in rows if bool(r.get("won")))
        pnl_values = [self._safe_float(r.get("pnl_cents"), 0.0) for r in rows]
        avg_pnl = (sum(pnl_values) / n) if n else 0.0

        yes = [r for r in rows if str(r.get("action", "")).upper() == "BUY_YES"]
        no = [r for r in rows if str(r.get("action", "")).upper() == "BUY_NO"]
        yes_wr = (sum(1 for r in yes if bool(r.get("won"))) / len(yes)) if yes else 0.0
        no_wr = (sum(1 for r in no if bool(r.get("won"))) / len(no)) if no else 0.0

        policy, _ = self._load_json_file("learned_policy.json")
        state, _ = self._load_json_file("learning_state.json")
        pending = len((state or {}).get("pending", {})) if isinstance(state, dict) else 0

        lines = [
            f"Backtest summary (last {n} resolved):",
            f"- Win rate: {wins / n:.1%}",
            f"- Avg PnL: {avg_pnl:+.1f}c per signal",
            f"- BUY_YES: n={len(yes)}, win={yes_wr:.1%}",
            f"- BUY_NO: n={len(no)}, win={no_wr:.1%}",
        ]
        if isinstance(policy, dict):
            yes_thr = self._safe_float(policy.get("min_edge_buy_yes"), 0.10)
            no_thr = self._safe_float(policy.get("min_edge_buy_no"), 0.13)
            mode = str(policy.get("mode", "learning"))
            lines.append(
                f"- Active thresholds: BUY_YES >= {yes_thr:.1%}, BUY_NO <= -{no_thr:.1%} ({mode})"
            )
        lines.append(f"- Pending unresolved signals: {pending}")
        return "\n".join(lines)

    def _reply_backtest(self, chat_id: str):
        rows, _ = self._load_recent_signal_outcomes(max_rows=300)
        msg = self._format_backtest_summary(rows)
        self.api.send_message(chat_id, msg, "")

    def _llm_reply(self, chat_id: str, user_text: str) -> Optional[str]:
        """Try LLM-backed conversational reply with strict fallback behavior."""
        if not self.llm_chat:
            return None
        try:
            if self._llm_client is None:
                from gnosis.reasoning.client import LLMClient
                self._llm_client = LLMClient()
            core_status = self._core_get("/status")
            proposals = self._core_get("/proposals")
            if isinstance(proposals, dict):
                proposals = list(proposals.values())[:3]
            else:
                proposals = []
            context = {
                "scan_interval_seconds": self.auto_scan_interval,
                "last_cycle": (
                    self._last_cycle_result.to_dict()
                    if hasattr(self._last_cycle_result, "to_dict")
                    else None
                ),
                "core_status": core_status,
                "recent_proposals": proposals,
                "history": self._chat_history.get(chat_id, [])[-6:],
            }
            system = (
                "You are ClawdBot's conversational assistant. "
                "Be concise, practical, and accurate. "
                "Do not claim an order executed unless context explicitly shows it. "
                "For risky actions (kill switch, pause, flatten), advise explicit command usage."
            )
            user = f"User message: {user_text}\n\nContext JSON:\n{json.dumps(context, default=str)[:5000]}"
            resp = self._llm_client.chat(system_prompt=system, user_prompt=user, max_tokens=450)
            if resp.error:
                return None
            if getattr(resp, "is_stub", False):
                # Stub mode should still feel conversational, not JSON-dumpy.
                return None
            content = (resp.content or "").strip()
            if not content:
                return None
            return content[:3000]
        except Exception:
            return None

    def _handle_conversation(self, text: str, chat_id: str):
        """Natural-language message handling for back-and-forth chat."""
        msg = (text or "").strip()
        if not msg:
            return
        low = msg.lower()
        self._append_history(chat_id, "user", msg)

        # Intent shortcuts for common operations.
        if re.search(r"\b(hello|hi|hey|yo|sup|what'?s up)\b", low):
            self.api.send_message(
                chat_id,
                "Hey — I’m here. I can chat, run scans, and summarize live backtest quality. "
                "Try: 'run a scan now' or 'show backtest summary'.",
                "",
            )
            return
        if any(k in low for k in ("run scan", "scan now", "new scan", "predict now", "scan again")):
            self.api.send_message(chat_id, "Running a fresh scan now...", "")
            self._force_scan(chat_id)
            return
        if any(k in low for k in ("backtest", "back test", "win rate", "signal quality", "learning summary")):
            self._reply_backtest(chat_id)
            return
        if any(k in low for k in ("status", "health", "alive", "system check")):
            self.send_status()
            return
        if "position" in low:
            self._reply_positions(chat_id)
            return
        if "order" in low:
            self._reply_orders(chat_id)
            return
        if any(k in low for k in ("proposal", "signal", "edge alert")):
            self._reply_signals(chat_id)
            return
        if any(k in low for k in ("last prediction", "predict at", "what did", "forecast at", "predicted at")):
            self._reply_last_prediction(chat_id)
            return
        if any(k in low for k in ("every minute", "scan interval", "how often", "frequency")):
            self.api.send_message(
                chat_id,
                (
                    f"Bot conversational/orchestrator scan loop runs every {int(self.auto_scan_interval)}s. "
                    "The dedicated Yoshi scanner service is separately configured and may use a different interval."
                ),
                "",
            )
            return
        if any(k in low for k in ("pause", "resume", "kill switch", "flatten")):
            self.api.send_message(
                chat_id,
                "For safety, use explicit admin controls (/status + Trading Core API endpoints) for risk actions.",
                "",
            )
            return

        llm_text = self._llm_reply(chat_id, msg)
        if llm_text:
            self._append_history(chat_id, "assistant", llm_text)
            self.api.send_message(chat_id, llm_text, "")
            return

        fallback = (
            "I can chat here, and also handle trading ops.\n"
            "Try one of these:\n"
            "- /scan\n"
            "- /backtest\n"
            "- /status\n"
            "- 'show positions' or 'show orders'\n"
            "- 'what was the last prediction?'"
        )
        self._append_history(chat_id, "assistant", fallback)
        self.api.send_message(chat_id, fallback, "")

    def _force_scan(self, chat_id: str):
        """Force a scan cycle and report results."""
        try:
            if self.orchestrator:
                result = self.orchestrator.run_cycle()
                self._last_cycle_result = result
                self.send_cycle_report(result)

                # Send BUY alerts
                for vp in result.to_dict().get("value_plays", []):
                    if vp.get("recommendation") == "BUY":
                        self.send_buy_alert(vp)
            else:
                self.api.send_message(chat_id, "No orchestrator configured", "")
        except Exception as e:
            self.api.send_message(chat_id, f"Scan error: {e}", "")

    def _send_params(self, chat_id: str):
        """Send current hyperparameters."""
        try:
            if self.orchestrator:
                params = self.orchestrator.ralph.get_params()
                d = params.to_dict()
            else:
                from gnosis.ralph.hyperparams import HyperParams
                d = HyperParams().to_dict()

            lines = ["Current Hyperparameters:\n"]
            for k, v in sorted(d.items()):
                if isinstance(v, float):
                    lines.append(f"  {k}: {v:.4f}")
                else:
                    lines.append(f"  {k}: {v}")
            self.api.send_message(chat_id, "\n".join(lines), "")
        except Exception as e:
            self.api.send_message(chat_id, f"Params error: {e}", "")

    # ── Orchestrator Callback ─────────────────────────────
    def _on_cycle(self, result):
        """Called after each orchestrator cycle."""
        self._last_cycle_result = result
        d = result.to_dict() if hasattr(result, "to_dict") else result

        # Send BUY alerts
        if self.notify_on_buy:
            for vp in d.get("value_plays", []):
                if vp.get("recommendation") == "BUY":
                    try:
                        self.send_buy_alert(vp)
                    except Exception:
                        pass

        # Optionally send full cycle report
        if self.notify_on_cycle:
            try:
                self.send_cycle_report(result)
            except Exception:
                pass

    # ── Background Scanner ────────────────────────────────
    def _scan_loop(self):
        """Background thread: runs orchestrator continuously."""
        if not self.orchestrator:
            return

        try:
            self.orchestrator.run_loop(
                interval_s=self.auto_scan_interval,
                on_cycle=self._on_cycle,
            )
        except Exception as e:
            try:
                self.send_plain(f"Scanner stopped: {e}")
            except Exception:
                pass

    # ── Main Loop ─────────────────────────────────────────
    def run(self):
        """Start the bot (blocking). Handles commands + runs scanner."""
        if not self.api:
            print("[TG] ERROR: TELEGRAM_BOT_TOKEN not set")
            print("[TG] Set it in .env or pass token= to TelegramBot()")
            return

        # Verify token
        me = self.api.get_me()
        if not me:
            print("[TG] ERROR: Invalid bot token")
            return

        bot_name = me.get("username", "?")
        print(f"[TG] Bot started: @{bot_name}")

        if self.chat_id:
            print(f"[TG] Chat ID: {self.chat_id}")
            self.send_plain(f"ClawdBot online. Type /help for commands.")
        else:
            print("[TG] WARNING: No TELEGRAM_CHAT_ID set.")
            print("[TG] Send any message to the bot to discover your chat ID.")

        # Start background scanner
        if self.orchestrator:
            self._scan_thread = threading.Thread(
                target=self._scan_loop,
                daemon=True,
                name="orchestrator-scanner",
            )
            self._scan_thread.start()
            print(f"[TG] Scanner running (interval: {self.auto_scan_interval}s)")

        # Long-polling loop
        self._running = True
        print("[TG] Listening for commands...")

        try:
            while self._running:
                try:
                    updates = self.api.get_updates(
                        offset=self._update_offset,
                        timeout=30,
                    )
                    for update in updates:
                        self._update_offset = update.get("update_id", 0) + 1
                        msg = update.get("message", {})
                        text = msg.get("text", "")
                        from_chat = str(msg.get("chat", {}).get("id", ""))

                        # Auto-discover chat ID
                        if not self.chat_id and from_chat:
                            self.chat_id = from_chat
                            print(f"[TG] Chat ID discovered: {self.chat_id}")
                            self._persist_chat_id(self.chat_id)
                            self.api.send_message(
                                from_chat,
                                f"Chat ID set: {from_chat}\nSaved to .env for alerts.",
                                "",
                            )

                        if text.startswith("/"):
                            print(f"[TG] Command: {text} (from {from_chat})")
                            self._handle_command(text, from_chat)
                        elif self.conversational and text.strip():
                            self._handle_conversation(text, from_chat)

                except Exception as e:
                    # Network hiccup — wait and retry
                    time.sleep(5)

        except KeyboardInterrupt:
            print("\n[TG] Bot stopped.")
            self._running = False

    def stop(self):
        """Stop the bot."""
        self._running = False
