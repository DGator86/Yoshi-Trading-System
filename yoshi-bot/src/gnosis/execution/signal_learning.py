"""Adaptive Kalshi signal learning + backtesting policy loop.

This module keeps a lightweight online-learning loop for signal quality:

1) Track emitted trade signals (pending outcomes)
2) Resolve outcomes once Kalshi contracts settle
3) Backtest threshold candidates on recent resolved outcomes
4) Publish adaptive runtime thresholds used by the live scanner

The goal is to reduce noisy directional spam (especially weak BUY_NO calls)
without adding heavy ML dependencies.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _safe_iso_to_dt(value: str | None) -> Optional[datetime]:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return float(sum(vals)) / float(len(vals))


def _std(values: Iterable[float]) -> float:
    vals = list(values)
    if len(vals) < 2:
        return 0.0
    mu = _mean(vals)
    var = sum((x - mu) ** 2 for x in vals) / float(len(vals) - 1)
    return math.sqrt(var)


@dataclass(frozen=True)
class ThresholdPolicy:
    """Runtime decision thresholds learned from recent resolved outcomes."""

    min_edge_buy_yes: float
    min_edge_buy_no: float
    n_resolved: int
    updated_at: str
    mode: str = "cold_start"
    yes_hit_rate: float = 0.0
    no_hit_rate: float = 0.0
    avg_pnl_cents: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_edge_buy_yes": round(float(self.min_edge_buy_yes), 4),
            "min_edge_buy_no": round(float(self.min_edge_buy_no), 4),
            "n_resolved": int(self.n_resolved),
            "updated_at": self.updated_at,
            "mode": self.mode,
            "yes_hit_rate": round(float(self.yes_hit_rate), 4),
            "no_hit_rate": round(float(self.no_hit_rate), 4),
            "avg_pnl_cents": round(float(self.avg_pnl_cents), 4),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ThresholdPolicy":
        return cls(
            min_edge_buy_yes=float(payload.get("min_edge_buy_yes", 0.10)),
            min_edge_buy_no=float(payload.get("min_edge_buy_no", 0.13)),
            n_resolved=int(payload.get("n_resolved", 0)),
            updated_at=str(payload.get("updated_at", _now_utc().isoformat())),
            mode=str(payload.get("mode", "cold_start")),
            yes_hit_rate=float(payload.get("yes_hit_rate", 0.0)),
            no_hit_rate=float(payload.get("no_hit_rate", 0.0)),
            avg_pnl_cents=float(payload.get("avg_pnl_cents", 0.0)),
        )


class KalshiSignalLearner:
    """Online resolution + rolling backtest optimizer for scanner thresholds."""

    def __init__(
        self,
        *,
        state_path: str | Path = "data/signals/learning_state.json",
        outcomes_path: str | Path = "data/signals/signal_outcomes.jsonl",
        policy_path: str | Path = "data/signals/learned_policy.json",
        min_samples: int = 30,
        lookback: int = 300,
        base_yes_edge: float = 0.10,
        base_no_edge: float = 0.13,
        settle_grace_minutes: float = 2.0,
    ):
        self.state_path = Path(state_path)
        self.outcomes_path = Path(outcomes_path)
        self.policy_path = Path(policy_path)
        self.min_samples = max(5, int(min_samples))
        self.lookback = max(20, int(lookback))
        self.base_yes_edge = max(0.01, float(base_yes_edge))
        self.base_no_edge = max(self.base_yes_edge, float(base_no_edge))
        self.settle_grace = timedelta(minutes=max(0.0, float(settle_grace_minutes)))

        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.outcomes_path.parent.mkdir(parents=True, exist_ok=True)
        self.policy_path.parent.mkdir(parents=True, exist_ok=True)

        self._pending: dict[str, dict[str, Any]] = {}
        self._policy = ThresholdPolicy(
            min_edge_buy_yes=self.base_yes_edge,
            min_edge_buy_no=self.base_no_edge,
            n_resolved=0,
            updated_at=_now_utc().isoformat(),
            mode="cold_start",
        )
        self._load_state()
        self._load_policy()

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def policy(self) -> ThresholdPolicy:
        return self._policy

    def effective_thresholds(self, fallback_edge: float = 0.10) -> tuple[float, float]:
        """Return runtime thresholds clamped by the operator fallback."""
        floor = max(0.01, float(fallback_edge))
        yes_edge = max(floor, float(self._policy.min_edge_buy_yes))
        no_edge = max(floor, float(self._policy.min_edge_buy_no))
        return yes_edge, no_edge

    def classify_edge(self, edge: float, fallback_edge: float = 0.10) -> str:
        """Map edge -> BUY_YES / BUY_NO / NEUTRAL using learned thresholds."""
        yes_edge, no_edge = self.effective_thresholds(fallback_edge=fallback_edge)
        if edge >= yes_edge:
            return "BUY_YES"
        if edge <= -no_edge:
            return "BUY_NO"
        return "NEUTRAL"

    def record_signal(self, signal: dict[str, Any]) -> bool:
        """Track an emitted signal for later outcome resolution.

        Returns False when the signal is skipped as a near-duplicate pending bet.
        """
        signal_id = str(signal.get("signal_id", "")).strip()
        if not signal_id:
            return False

        ticker = str(signal.get("ticker", "")).strip()
        action = str(signal.get("action", "")).strip().upper()
        if action not in {"BUY_YES", "BUY_NO"}:
            return False

        # De-dupe unresolved contract/side spam in minute loops.
        for rec in self._pending.values():
            if rec.get("ticker") == ticker and rec.get("action") == action:
                return False

        created_at = str(signal.get("created_at", _now_utc().isoformat()))
        market_prob = float(signal.get("market_prob", 0.5))
        market_prob = max(0.01, min(0.99, market_prob))
        edge = float(signal.get("edge", 0.0))

        self._pending[signal_id] = {
            "signal_id": signal_id,
            "ticker": ticker,
            "symbol": str(signal.get("symbol", "")),
            "action": action,
            "edge": edge,
            "market_prob": market_prob,
            "model_prob": float(signal.get("model_prob", 0.5)),
            "strike": float(signal.get("strike", 0.0)),
            "source": str(signal.get("source", "kalshi_scanner")),
            "created_at": created_at,
            "close_time": str(signal.get("close_time", "")),
        }
        self._save_state()
        return True

    def resolve_pending(
        self,
        fetch_market: Callable[[str], Optional[dict[str, Any]]],
        *,
        max_checks: int = 25,
    ) -> int:
        """Resolve pending signals with settled Kalshi outcomes.

        Args:
            fetch_market: Callable that returns market payload for ticker.
            max_checks: Max unresolved contracts to poll in this cycle.
        """
        if not self._pending:
            return 0

        now = _now_utc()
        checks = 0
        resolved_records: list[dict[str, Any]] = []

        # Resolve oldest first to clear backlog quickly.
        pending_ids = sorted(
            self._pending.keys(),
            key=lambda sid: self._pending[sid].get("created_at", ""),
        )

        for sid in pending_ids:
            if checks >= max(1, int(max_checks)):
                break
            rec = self._pending.get(sid)
            if not rec:
                continue

            close_time = _safe_iso_to_dt(rec.get("close_time"))
            if close_time is not None and now < (close_time + self.settle_grace):
                continue

            ticker = str(rec.get("ticker", "")).strip()
            if not ticker:
                continue

            checks += 1
            market = fetch_market(ticker) or {}
            settled_yes = self._extract_settlement_yes(market)
            if settled_yes is None:
                continue

            action = rec.get("action", "BUY_YES")
            won = bool((action == "BUY_YES" and settled_yes) or (action == "BUY_NO" and not settled_yes))
            market_prob = float(rec.get("market_prob", 0.5))
            cost_cents = int(round((market_prob if action == "BUY_YES" else (1.0 - market_prob)) * 100.0))
            cost_cents = max(1, min(99, cost_cents))
            pnl_cents = (100 - cost_cents) if won else -cost_cents

            resolved = {
                **rec,
                "settled_yes": bool(settled_yes),
                "won": bool(won),
                "cost_cents": cost_cents,
                "pnl_cents": float(pnl_cents),
                "resolved_at": now.isoformat(),
            }
            resolved_records.append(resolved)
            self._pending.pop(sid, None)

        if not resolved_records:
            return 0

        with self.outcomes_path.open("a", encoding="utf-8") as f:
            for row in resolved_records:
                f.write(json.dumps(row, separators=(",", ":"), default=str) + "\n")

        self.recompute_policy()
        self._save_state()
        return len(resolved_records)

    def recompute_policy(self) -> ThresholdPolicy:
        """Backtest threshold candidates on recent outcomes and publish policy."""
        outcomes = self._load_recent_outcomes(self.lookback)
        if len(outcomes) < self.min_samples:
            self._policy = ThresholdPolicy(
                min_edge_buy_yes=self.base_yes_edge,
                min_edge_buy_no=self.base_no_edge,
                n_resolved=len(outcomes),
                updated_at=_now_utc().isoformat(),
                mode="cold_start",
            )
            self._save_policy()
            return self._policy

        yes_rows = [r for r in outcomes if str(r.get("action", "")).upper() == "BUY_YES"]
        no_rows = [r for r in outcomes if str(r.get("action", "")).upper() == "BUY_NO"]

        yes_thr = self._optimize_side_threshold(
            yes_rows,
            base_edge=self.base_yes_edge,
            candidates=(0.05, 0.07, 0.09, 0.11, 0.13, 0.16, 0.20),
        )
        no_thr = self._optimize_side_threshold(
            no_rows,
            base_edge=self.base_no_edge,
            candidates=(0.07, 0.10, 0.12, 0.15, 0.18, 0.22, 0.26),
        )

        yes_hit = _mean(1.0 if bool(r.get("won")) else 0.0 for r in yes_rows) if yes_rows else 0.0
        no_hit = _mean(1.0 if bool(r.get("won")) else 0.0 for r in no_rows) if no_rows else 0.0
        avg_pnl = _mean(float(r.get("pnl_cents", 0.0)) for r in outcomes)

        self._policy = ThresholdPolicy(
            min_edge_buy_yes=max(self.base_yes_edge, yes_thr),
            min_edge_buy_no=max(self.base_no_edge, no_thr),
            n_resolved=len(outcomes),
            updated_at=_now_utc().isoformat(),
            mode="learning",
            yes_hit_rate=yes_hit,
            no_hit_rate=no_hit,
            avg_pnl_cents=avg_pnl,
        )
        self._save_policy()
        return self._policy

    @staticmethod
    def _extract_settlement_yes(market: dict[str, Any]) -> Optional[bool]:
        """Extract settled YES/NO outcome from a Kalshi market payload."""
        if not isinstance(market, dict) or not market:
            return None

        def parse_boolish(value: Any) -> Optional[bool]:
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                if value in (0, 0.0):
                    return False
                if value in (1, 1.0):
                    return True
            if not isinstance(value, str):
                return None
            text = value.strip().lower()
            yes_tokens = {"yes", "y", "true", "won_yes", "above", "up"}
            no_tokens = {"no", "n", "false", "won_no", "below", "down"}
            if text in yes_tokens:
                return True
            if text in no_tokens:
                return False
            return None

        # Explicit outcome fields (best source).
        for key in (
            "result",
            "settlement_result",
            "outcome",
            "winner",
            "winning_side",
            "final_outcome",
        ):
            parsed = parse_boolish(market.get(key))
            if parsed is not None:
                return parsed

        # Settlement price style fields.
        yes_fields = (
            "yes_settle",
            "yes_settlement",
            "yes_settlement_price",
            "settlement_yes_price",
            "final_yes_price",
        )
        no_fields = (
            "no_settle",
            "no_settlement",
            "no_settlement_price",
            "settlement_no_price",
            "final_no_price",
        )
        yes_val = next((market.get(k) for k in yes_fields if market.get(k) is not None), None)
        no_val = next((market.get(k) for k in no_fields if market.get(k) is not None), None)
        if isinstance(yes_val, (int, float)) and isinstance(no_val, (int, float)):
            if float(yes_val) >= 99 and float(no_val) <= 1:
                return True
            if float(yes_val) <= 1 and float(no_val) >= 99:
                return False

        status = str(market.get("status", "")).strip().lower()
        settled_like = (
            "settled" in status
            or "final" in status
            or bool(market.get("settled", False))
            or bool(market.get("is_settled", False))
        )
        if not settled_like:
            return None

        # Last-resort fallback if market is settled and only one side at 100.
        yes_bid = market.get("yes_bid")
        yes_ask = market.get("yes_ask")
        no_bid = market.get("no_bid")
        no_ask = market.get("no_ask")
        prices = [p for p in (yes_bid, yes_ask, no_bid, no_ask) if isinstance(p, (int, float))]
        if prices:
            if (isinstance(yes_bid, (int, float)) and yes_bid >= 99) or (
                isinstance(yes_ask, (int, float)) and yes_ask >= 99
            ):
                return True
            if (isinstance(no_bid, (int, float)) and no_bid >= 99) or (
                isinstance(no_ask, (int, float)) and no_ask >= 99
            ):
                return False

        return None

    def _optimize_side_threshold(
        self,
        rows: list[dict[str, Any]],
        *,
        base_edge: float,
        candidates: tuple[float, ...],
    ) -> float:
        if not rows:
            return base_edge

        abs_edges = [abs(float(r.get("edge", 0.0))) for r in rows]
        if not abs_edges:
            return base_edge

        n_side = len(rows)
        # Keep requirements modest so smaller side-specific samples can still adapt.
        min_trades = max(3, min(12, int(0.20 * n_side)))
        best_thr = base_edge
        best_score = float("-inf")

        for thr in candidates:
            chosen = [
                r for r in rows
                if abs(float(r.get("edge", 0.0))) >= float(thr)
            ]
            if len(chosen) < min_trades:
                continue
            pnls = [float(r.get("pnl_cents", 0.0)) for r in chosen]
            wins = [1.0 if bool(r.get("won", False)) else 0.0 for r in chosen]
            mean_pnl = _mean(pnls)
            hit_rate = _mean(wins)
            pnl_std = _std(pnls)

            # Risk-adjusted objective:
            # reward positive expectancy + stable returns + enough observations.
            score = (
                mean_pnl
                + (hit_rate - 0.5) * 15.0
                - pnl_std * 0.05
                + math.log1p(len(chosen)) * 1.5
            )
            if score > best_score:
                best_score = score
                best_thr = float(thr)

        return max(base_edge, best_thr)

    def _load_recent_outcomes(self, limit: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if not self.outcomes_path.exists():
            return rows
        try:
            with self.outcomes_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            return []
        if limit <= 0:
            return rows
        return rows[-limit:]

    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        pending = payload.get("pending", {})
        if isinstance(pending, dict):
            self._pending = {
                str(k): v for k, v in pending.items()
                if isinstance(v, dict)
            }

    def _save_state(self) -> None:
        payload = {
            "updated_at": _now_utc().isoformat(),
            "pending": self._pending,
            "policy": self._policy.to_dict(),
        }
        self.state_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def _load_policy(self) -> None:
        if not self.policy_path.exists():
            return
        try:
            payload = json.loads(self.policy_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(payload, dict):
            self._policy = ThresholdPolicy.from_dict(payload)

    def _save_policy(self) -> None:
        self.policy_path.write_text(
            json.dumps(self._policy.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
