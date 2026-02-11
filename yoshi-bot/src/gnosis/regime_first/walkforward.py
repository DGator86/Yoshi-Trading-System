"""Regime-aware walk-forward runner (crypto regime-first)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from gnosis.regime_first.backtest import run_regime_first_backtest
from gnosis.regime_first.blending import compute_router_action, learn_playbook_weights
from gnosis.regime_first.playbooks import add_playbook_outputs
from gnosis.regime_first.reporting import (
    js_divergence,
    metrics_by_group,
    regime_distribution,
    regime_robustness_score,
    tail_intensity_stats,
    transition_matrix,
)


def _ts(x: Any) -> pd.Timestamp:
    return pd.to_datetime(x, utc=True)


def _window_blocks(
    start: pd.Timestamp,
    end: pd.Timestamp,
    train_days: int,
    test_days: int,
    roll_days: int,
) -> list[dict[str, pd.Timestamp]]:
    blocks: list[dict[str, pd.Timestamp]] = []
    cur = start
    td = pd.Timedelta(days=int(train_days))
    te = pd.Timedelta(days=int(test_days))
    rd = pd.Timedelta(days=int(roll_days))
    i = 0
    while True:
        train_start = cur
        train_end = train_start + td
        test_start = train_end
        test_end = test_start + te
        if test_end > end:
            break
        blocks.append(
            {
                "block_idx": i,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )
        i += 1
        cur = cur + rd
    return blocks


def run_regime_first_walkforward(
    regime_ledger: pd.DataFrame,
    cfg_raw: dict[str, Any],
    out_dir: str | Path,
) -> dict[str, Any]:
    """Run regime-aware walk-forward and emit required artifacts."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Add playbook outputs and initial router score using base W.
    df = add_playbook_outputs(regime_ledger)

    blending = cfg_raw.get("blending", {}) or {}
    base_W = blending.get("playbook_weights", {}) or {}
    risk_cfg = cfg_raw.get("risk", {}) or {}

    # Router scores will be re-computed per-block after learning W.
    # Persist the regime ledger as the spine artifact.
    # (Parquet preferred; CSV is easy to inspect.)
    try:
        df.to_parquet(out_path / "regime_ledger_1m.parquet", index=False)
    except Exception:
        df.to_csv(out_path / "regime_ledger_1m.csv", index=False)

    walk = cfg_raw.get("walkforward", {}) or {}
    train_days = int(walk.get("train_days", 180))
    test_days = int(walk.get("test_days", 30))
    roll_days = int(walk.get("roll_days", 30))

    tmin = _ts(df["timestamp"].min())
    tmax = _ts(df["timestamp"].max())
    blocks = _window_blocks(tmin, tmax, train_days=train_days, test_days=test_days, roll_days=roll_days)

    coverage = (walk.get("regime_coverage") or {}) if isinstance(walk, dict) else {}
    min_train_share = (coverage.get("min_train_share") or {}) if isinstance(coverage, dict) else {}
    shift_rules = (coverage.get("shift_flag_rules") or {}) if isinstance(coverage, dict) else {}
    max_js = float(shift_rules.get("max_js_divergence", 0.15))
    factor = float(shift_rules.get("test_vs_train_factor", 4.0))

    reporting = (walk.get("reporting") or {}) if isinstance(walk, dict) else {}
    stress_tests = reporting.get("cost_stress_tests", []) or []

    all_trades = []
    block_summaries: list[dict[str, Any]] = []

    for b in blocks:
        bdir = out_path / f"block_{int(b['block_idx']):03d}"
        bdir.mkdir(parents=True, exist_ok=True)

        train = df[(df["timestamp"] >= b["train_start"]) & (df["timestamp"] < b["train_end"])].copy()
        test = df[(df["timestamp"] >= b["test_start"]) & (df["timestamp"] < b["test_end"])].copy()

        train_dist = regime_distribution(train, use_expected=True)
        test_dist = regime_distribution(test, use_expected=True)
        train_trans = transition_matrix(train)
        test_trans = transition_matrix(test)
        js = js_divergence(train_dist, test_dist)

        # Coverage / shift flags.
        flags = {"js_divergence": float(js), "js_flag": bool(js > max_js), "regime_shift_flags": []}
        for r, min_share in (min_train_share.items() if isinstance(min_train_share, dict) else []):
            try:
                ms = float(min_share)
            except Exception:
                continue
            if ms <= 0:
                continue
            if test_dist.get(r, 0.0) > factor * ms and train_dist.get(r, 0.0) < ms:
                flags["regime_shift_flags"].append(
                    {
                        "regime": str(r),
                        "train_share": float(train_dist.get(r, 0.0)),
                        "test_share": float(test_dist.get(r, 0.0)),
                        "min_train_share": float(ms),
                        "factor": float(factor),
                    }
                )

        # Learn W[r,k] on train block.
        learned_W, learn_diag = learn_playbook_weights(train, base_W=base_W)

        # Compute router score for test block using learned weights.
        test_scored = compute_router_action(test, W=learned_W, risk_cfg=risk_cfg)

        bt = run_regime_first_backtest(test_scored, cfg_raw, W=learned_W)
        trades = bt.trade_ledger.copy()
        all_trades.append(trades)

        # Per-regime + per-playbook attribution.
        by_regime = metrics_by_group(trades, "regime_entry_label")
        by_playbook = metrics_by_group(trades, "playbook_id")
        by_transition_entry = metrics_by_group(trades, "transition_entry") if "transition_entry" in trades.columns else {}
        rrs = regime_robustness_score(trades)

        # Cost stress tests: rerun with multipliers.
        stress_results = []
        for st in stress_tests:
            if not isinstance(st, dict):
                continue
            fm = float(st.get("fee_mult", 1.0))
            sm = float(st.get("slip_mult", 1.0))
            bt_stress = run_regime_first_backtest(test_scored, cfg_raw, W=learned_W, fee_mult=fm, slip_mult=sm)
            stress_results.append({"fee_mult": fm, "slip_mult": sm, "stats": bt_stress.stats})

        summary = {
            "block_idx": int(b["block_idx"]),
            "train_start": str(b["train_start"]),
            "train_end": str(b["train_end"]),
            "test_start": str(b["test_start"]),
            "test_end": str(b["test_end"]),
            "distribution_shift": {
                "train_regime_dist": train_dist,
                "test_regime_dist": test_dist,
                "train_transition_matrix": train_trans,
                "test_transition_matrix": test_trans,
                "tail_intensity_train": tail_intensity_stats(train),
                "tail_intensity_test": tail_intensity_stats(test),
                "flags": flags,
            },
            "learned": {
                "playbook_weights": learned_W,
                "diagnostics": learn_diag,
            },
            "performance": {
                "overall": bt.stats,
                "by_regime": by_regime,
                "by_playbook": by_playbook,
                "by_transition_entry": by_transition_entry,
                "rrs": rrs,
                "cost_stress_tests": stress_results,
            },
        }

        # Write block artifacts.
        with open(bdir / "walkforward_block_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        try:
            trades.to_parquet(bdir / "trade_ledger.parquet", index=False)
        except Exception:
            trades.to_csv(bdir / "trade_ledger.csv", index=False)

        block_summaries.append(summary)

    trades_all = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    try:
        trades_all.to_parquet(out_path / "trade_ledger.parquet", index=False)
    except Exception:
        trades_all.to_csv(out_path / "trade_ledger.csv", index=False)

    wf_summary = {"n_blocks": int(len(block_summaries)), "blocks": block_summaries}
    with open(out_path / "walkforward_summary.json", "w", encoding="utf-8") as f:
        json.dump(wf_summary, f, indent=2, sort_keys=True)

    return wf_summary

