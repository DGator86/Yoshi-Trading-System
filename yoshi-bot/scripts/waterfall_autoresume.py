#!/usr/bin/env python3
"""Auto-resume orchestrator for sigma-gated waterfall backtests.

Runs waterfall_backtest timeframe-by-timeframe and, on strict sigma failures,
restarts from the failed offset until the timeframe completes or safety limits
are hit.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None


def normalize_timeframes(value: str) -> list[str]:
    items = [x.strip() for x in str(value).split(",") if x.strip()]
    return items if items else ["1h", "30m", "15m", "5m", "1m"]


def _load_waterfall_module():
    path = Path(__file__).with_name("waterfall_backtest.py")
    spec = importlib.util.spec_from_file_location("waterfall_backtest_runtime", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load waterfall_backtest.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _build_run_args(
    args: argparse.Namespace,
    timeframe: str,
    start_offset: int,
    *,
    run_seed: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        symbol=args.symbol,
        timeframes=timeframe,
        start_offset=int(max(1, start_offset)),
        n_bars=int(args.n_bars),
        snapshot_lookback_bars=int(args.snapshot_lookback_bars),
        sigma_target=float(args.sigma_target),
        max_attempts_per_bar=int(args.max_attempts_per_bar),
        allow_unconverged_progress=bool(args.allow_unconverged_progress),
        base_mc_iterations=int(args.base_mc_iterations),
        base_mc_steps=int(args.base_mc_steps),
        adaptive_sigma_calibration=bool(args.adaptive_sigma_calibration),
        sigma_calibration_rate=float(args.sigma_calibration_rate),
        sigma_scale_max=float(args.sigma_scale_max),
        ralph_explore_rate=float(args.ralph_explore_rate),
        seed=int(run_seed),
        log_every=int(args.log_every),
        ohlcv_providers=str(args.ohlcv_providers),
        data_timeout_s=int(args.data_timeout_s),
        data_max_retries=int(args.data_max_retries),
        output_dir=str(args.output_dir),
    )


def _load_state(path: Path, timeframes: list[str]) -> dict[str, Any]:
    state = {"timeframes": {}, "updated_at": ""}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                state.update(payload)
        except Exception:
            pass
    tf_state = state.setdefault("timeframes", {})
    for tf in timeframes:
        if tf not in tf_state or not isinstance(tf_state.get(tf), dict):
            tf_state[tf] = {
                "completed": False,
                "next_offset": 1,
                "runs": 0,
                "last_failed_offset": None,
                "stall_count": 0,
                "skipped_count": 0,
                "skipped_offsets": [],
            }
        else:
            tf_state[tf].setdefault("skipped_count", 0)
            tf_state[tf].setdefault("skipped_offsets", [])
    return state


def _save_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auto-resume strict waterfall backtest")
    p.add_argument("--symbol", type=str, default="BTCUSDT")
    p.add_argument("--timeframes", type=str, default="1h,30m,15m,5m,1m")
    p.add_argument("--start-offset", type=int, default=1, help="Initial offset for first timeframe")
    p.add_argument("--n-bars", type=int, default=2000)
    p.add_argument("--snapshot-lookback-bars", type=int, default=600)
    p.add_argument("--sigma-target", type=float, default=1.0)
    p.add_argument("--max-attempts-per-bar", type=int, default=180)
    p.add_argument(
        "--allow-unconverged-progress",
        action="store_true",
        help="Pass through to waterfall_backtest to continue even when a bar fails sigma target",
    )
    p.add_argument(
        "--adaptive-sigma-calibration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable adaptive sigma calibration while converging each offset",
    )
    p.add_argument("--sigma-calibration-rate", type=float, default=0.7)
    p.add_argument("--sigma-scale-max", type=float, default=4.5)
    p.add_argument("--base-mc-iterations", type=int, default=8000)
    p.add_argument("--base-mc-steps", type=int, default=48)
    p.add_argument("--ralph-explore-rate", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--seed-jitter-per-run",
        type=int,
        default=101,
        help="Adds deterministic seed jitter each rerun to diversify hyperparam exploration",
    )
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument(
        "--ohlcv-providers",
        type=str,
        default="coinapi,binance_public,coingecko,coinmarketcap,yfinance",
        help="Comma-separated provider order passed to waterfall_backtest",
    )
    p.add_argument("--data-timeout-s", type=int, default=12, help="Per-request provider timeout")
    p.add_argument("--data-max-retries", type=int, default=1, help="Per-request retries for providers")
    p.add_argument("--sleep-between-runs", type=float, default=1.0)
    p.add_argument("--max-runs-per-timeframe", type=int, default=200)
    p.add_argument("--stall-limit", type=int, default=10, help="Abort timeframe after N repeated failed offsets")
    p.add_argument(
        "--skip-stalled-offset",
        action="store_true",
        help="After hitting stall-limit on one offset, skip that offset and continue",
    )
    p.add_argument(
        "--max-skipped-per-timeframe",
        type=int,
        default=8,
        help="Safety cap on number of skipped offsets per timeframe",
    )
    p.add_argument("--output-dir", type=str, default="reports/waterfall_backtest")
    p.add_argument("--state-path", type=str, default="")
    p.add_argument("--reset-state", action="store_true")
    return p.parse_args()


def main() -> int:
    if load_dotenv is not None:
        try:
            repo_root = Path(__file__).resolve().parents[1]
            load_dotenv(repo_root / ".env")
            load_dotenv()
        except Exception:
            pass

    args = parse_args()
    timeframes = normalize_timeframes(args.timeframes)
    output_dir = Path(args.output_dir)
    state_path = Path(args.state_path) if args.state_path else output_dir / "autoresume_state.json"
    if args.reset_state and state_path.exists():
        state_path.unlink(missing_ok=True)

    wf = _load_waterfall_module()
    state = _load_state(state_path, timeframes)

    # Seed first timeframe start offset from CLI when fresh.
    first_tf = timeframes[0]
    tf_state = state["timeframes"][first_tf]
    if int(tf_state.get("runs", 0)) == 0 and int(tf_state.get("next_offset", 1)) <= 1:
        tf_state["next_offset"] = max(1, int(args.start_offset))

    print(f"[autoresume] symbol={args.symbol} timeframes={timeframes}")
    print(f"[autoresume] state_path={state_path}")

    overall_ok = True
    for tf in timeframes:
        meta = state["timeframes"][tf]
        if meta.get("completed"):
            print(f"[autoresume] skip completed timeframe={tf}")
            continue

        while True:
            if int(meta.get("runs", 0)) >= int(args.max_runs_per_timeframe):
                print(f"[autoresume] max runs reached timeframe={tf}")
                overall_ok = False
                break

            start_offset = max(1, int(meta.get("next_offset", 1)))
            run_seed = int(args.seed) + int(meta.get("runs", 0)) * int(args.seed_jitter_per_run)
            run_args = _build_run_args(
                args,
                timeframe=tf,
                start_offset=start_offset,
                run_seed=run_seed,
            )
            print(
                f"[autoresume] run timeframe={tf} run={int(meta.get('runs', 0)) + 1} "
                f"start_offset={start_offset} seed={run_seed}"
            )

            result = wf.run_waterfall(run_args)
            tf_summary = (result.get("timeframes") or [{}])[0]
            meta["runs"] = int(meta.get("runs", 0)) + 1

            status = str(tf_summary.get("status", "unknown"))
            converged = int(tf_summary.get("converged_count", 0))
            failed = int(tf_summary.get("failed_count", 0))
            print(
                f"[autoresume] result timeframe={tf} status={status} converged={converged} failed={failed} "
                f"mean_err_sigma={tf_summary.get('mean_error_sigma')}"
            )

            if status == "stopped_unconverged" and result.get("stopped_early"):
                failed_offset = int(tf_summary.get("failed_offset") or start_offset)
                prev_failed = meta.get("last_failed_offset")
                if prev_failed == failed_offset:
                    meta["stall_count"] = int(meta.get("stall_count", 0)) + 1
                else:
                    meta["stall_count"] = 0

                meta["last_failed_offset"] = failed_offset
                meta["next_offset"] = failed_offset
                if int(meta.get("stall_count", 0)) >= int(args.stall_limit):
                    skipped_count = int(meta.get("skipped_count", 0))
                    can_skip = bool(args.skip_stalled_offset) and skipped_count < int(args.max_skipped_per_timeframe)
                    if can_skip:
                        skip_list = meta.setdefault("skipped_offsets", [])
                        if failed_offset not in skip_list:
                            skip_list.append(failed_offset)
                        meta["skipped_count"] = skipped_count + 1
                        meta["next_offset"] = failed_offset + 1
                        meta["last_failed_offset"] = None
                        meta["stall_count"] = 0
                        print(
                            f"[autoresume] skip stalled offset timeframe={tf} offset={failed_offset} "
                            f"skipped_count={meta.get('skipped_count')}/{int(args.max_skipped_per_timeframe)}"
                        )
                        _save_state(state_path, state)
                        time.sleep(max(0.0, float(args.sleep_between_runs)))
                        continue
                    print(
                        f"[autoresume] stall limit reached timeframe={tf} offset={failed_offset} "
                        f"stall_count={meta.get('stall_count')}"
                    )
                    overall_ok = False
                    _save_state(state_path, state)
                    break

                _save_state(state_path, state)
                time.sleep(max(0.0, float(args.sleep_between_runs)))
                continue

            # Completed timeframe or non-recoverable data issue.
            if status in {"ok", "insufficient_history", "no_data"} and not result.get("stopped_early", False):
                meta["completed"] = True
                meta["next_offset"] = int(args.n_bars) + 1
                meta["last_failed_offset"] = None
                meta["stall_count"] = 0
                _save_state(state_path, state)
                break

            # Unknown stop status -> fail safe.
            overall_ok = False
            _save_state(state_path, state)
            break

    _save_state(state_path, state)
    complete = all(bool(state["timeframes"][tf].get("completed")) for tf in timeframes)
    print(f"[autoresume] complete={complete} overall_ok={overall_ok}")
    return 0 if (complete and overall_ok) else 2


if __name__ == "__main__":
    raise SystemExit(main())
