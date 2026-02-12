"""Tests for adaptive Kalshi signal learning policy."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.gnosis.execution.signal_learning import KalshiSignalLearner  # noqa: E402


def _new_learner(tmp_path: Path, min_samples: int = 6) -> KalshiSignalLearner:
    return KalshiSignalLearner(
        state_path=tmp_path / "state.json",
        outcomes_path=tmp_path / "outcomes.jsonl",
        policy_path=tmp_path / "policy.json",
        min_samples=min_samples,
        lookback=200,
        base_yes_edge=0.10,
        base_no_edge=0.13,
    )


def test_policy_stays_in_cold_start_without_samples(tmp_path):
    learner = _new_learner(tmp_path, min_samples=10)
    policy = learner.recompute_policy()
    assert policy.mode == "cold_start"
    assert policy.n_resolved == 0
    yes_edge, no_edge = learner.effective_thresholds(fallback_edge=0.10)
    assert yes_edge >= 0.10
    assert no_edge >= 0.13


def test_resolve_pending_marks_outcome_and_appends_jsonl(tmp_path):
    learner = _new_learner(tmp_path)
    learner.record_signal(
        {
            "signal_id": "sig_1",
            "ticker": "KXBTC-TEST-T100000",
            "symbol": "BTCUSDT",
            "action": "BUY_NO",
            "edge": -0.15,
            "market_prob": 0.72,
            "model_prob": 0.57,
            "strike": 100000.0,
            "source": "unit_test",
            "created_at": "2026-01-01T00:00:00+00:00",
        }
    )

    def _fetch_market(_ticker: str) -> dict:
        return {"status": "settled", "result": "no"}

    resolved = learner.resolve_pending(_fetch_market, max_checks=5)
    assert resolved == 1
    assert learner.pending_count == 0

    lines = (tmp_path / "outcomes.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["won"] is True
    assert row["settled_yes"] is False
    assert row["pnl_cents"] > 0


def test_backtest_raises_buy_no_threshold_when_weak_edges_fail(tmp_path):
    learner = _new_learner(tmp_path)
    outcomes_path = tmp_path / "outcomes.jsonl"

    rows = []
    # BUY_NO: weak edges lose, stronger edges win -> threshold should move up.
    for edge, won in [
        (-0.08, False),
        (-0.09, False),
        (-0.10, False),
        (-0.11, False),
        (-0.13, False),
        (-0.14, False),
        (-0.16, True),
        (-0.18, True),
        (-0.20, True),
    ]:
        market_prob = 0.65
        cost = int(round((1.0 - market_prob) * 100))
        pnl = (100 - cost) if won else -cost
        rows.append(
            {
                "signal_id": f"no_{abs(edge)}",
                "ticker": "KXBTC-TEST-T100000",
                "action": "BUY_NO",
                "edge": edge,
                "market_prob": market_prob,
                "pnl_cents": pnl,
                "won": won,
            }
        )

    # BUY_YES: neutral quality so YES threshold should stay near base.
    for edge, won in [(0.11, True), (0.12, True), (0.13, False), (0.14, True)]:
        market_prob = 0.45
        cost = int(round(market_prob * 100))
        pnl = (100 - cost) if won else -cost
        rows.append(
            {
                "signal_id": f"yes_{edge}",
                "ticker": "KXBTC-TEST-T100000",
                "action": "BUY_YES",
                "edge": edge,
                "market_prob": market_prob,
                "pnl_cents": pnl,
                "won": won,
            }
        )

    outcomes_path.write_text(
        "\n".join(json.dumps(r, separators=(",", ":")) for r in rows) + "\n",
        encoding="utf-8",
    )

    policy = learner.recompute_policy()
    assert policy.mode == "learning"
    assert policy.n_resolved == len(rows)
    assert policy.min_edge_buy_no >= 0.14
    assert policy.min_edge_buy_yes >= 0.10
