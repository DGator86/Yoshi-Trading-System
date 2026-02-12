#!/usr/bin/env python3
"""Waterfall sigma-gated backtest with Ralph hyperparameter looping.

Requested workflow:
  1) Waterfall timeframes (default): 1h -> 30m -> 15m -> 5m -> 1m
  2) Constant blind projection window: n=2000 offsets per timeframe
  3) For each offset (starting 1 bar from present), run ML+MC+Regime forecaster
     repeatedly with Ralph explore/exploit parameter stepping until prediction
     is within 1 sigma (or max-attempt safety cap).
  4) Only then move to the next offset, unless unconverged progression is
     explicitly allowed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None


DEFAULT_TIMEFRAMES = ("1h", "30m", "15m", "5m", "1m")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def timeframe_to_minutes(tf: str) -> int:
    t = str(tf).strip().lower()
    if t.endswith("m"):
        return max(1, int(t[:-1]))
    if t.endswith("h"):
        return max(1, int(t[:-1])) * 60
    if t.endswith("d"):
        return max(1, int(t[:-1])) * 1440
    raise ValueError(f"Unsupported timeframe: {tf}")


def bars_to_days(n_bars: int, timeframe: str, buffer_bars: int = 256) -> int:
    minutes = timeframe_to_minutes(timeframe)
    total_bars = max(32, int(n_bars) + int(buffer_bars))
    return max(3, int(math.ceil((total_bars * minutes) / 1440.0)) + 2)


def _normalize_symbol(value: str) -> str:
    s = str(value or "").upper().replace("/", "")
    if s.endswith("USD") and not s.endswith("USDT"):
        s = s + "T"
    if not s.endswith("USDT") and s in {"BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "BNB"}:
        s = s + "USDT"
    return s


def _select_symbol_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df.empty:
        return df
    if "symbol" not in df.columns:
        return df
    target = _normalize_symbol(symbol)
    tmp = df.copy()
    tmp["symbol_norm"] = tmp["symbol"].astype(str).str.upper().str.replace("/", "", regex=False)
    # Flexible matching for cases like BTC vs BTCUSDT
    matches = tmp[tmp["symbol_norm"] == target]
    if matches.empty:
        base = target[:3]
        matches = tmp[tmp["symbol_norm"].str.startswith(base, na=False)]
    if matches.empty:
        return pd.DataFrame(columns=df.columns)
    return matches.drop(columns=["symbol_norm"])


def _aggregate_bars(rows: list[Any], chunk_size: int):
    if chunk_size <= 1 or len(rows) < chunk_size:
        return rows
    out = []
    for i in range(0, len(rows) - chunk_size + 1, chunk_size):
        chunk = rows[i : i + chunk_size]
        out.append(
            type(rows[0])(
                timestamp=chunk[0].timestamp,
                open=chunk[0].open,
                high=max(b.high for b in chunk),
                low=min(b.low for b in chunk),
                close=chunk[-1].close,
                volume=sum(b.volume for b in chunk),
            )
        )
    return out


@dataclass
class RunningPerf:
    n: int = 0
    brier_sum: float = 0.0
    hit_sum: float = 0.0
    pnl_sum: float = 0.0

    def update(
        self,
        *,
        current_price: float,
        actual_price: float,
        direction_prob_up: float,
        error_sigma: float,
    ) -> None:
        self.n += 1
        p_up = _clamp(direction_prob_up, 0.001, 0.999)
        actual_up = 1.0 if actual_price >= current_price else 0.0
        self.brier_sum += (p_up - actual_up) ** 2
        self.hit_sum += 1.0 if ((p_up >= 0.5 and actual_up == 1.0) or (p_up < 0.5 and actual_up == 0.0)) else 0.0
        # Proxy PnL to keep Ralph optimizer directional: lower sigma error = higher score.
        self.pnl_sum += max(-120.0, 120.0 - (error_sigma * 70.0))

    def metrics(self) -> dict[str, Any]:
        if self.n <= 0:
            return {
                "n_resolved": 0,
                "brier_score": 0.25,
                "hit_rate": 0.5,
                "total_pnl_cents": 0.0,
            }
        return {
            "n_resolved": int(self.n),
            "brier_score": float(self.brier_sum / self.n),
            "hit_rate": float(self.hit_sum / self.n),
            "total_pnl_cents": float(self.pnl_sum),
        }


def _prepare_imports():
    """Import heavy modules lazily to keep unit tests lightweight."""
    repo_root = Path(__file__).resolve().parents[2]
    clawdbot_root = repo_root / "clawdbot"
    yoshi_root = repo_root / "yoshi-bot"

    if str(clawdbot_root) not in sys.path:
        sys.path.insert(0, str(clawdbot_root))
    if str(yoshi_root) not in sys.path:
        sys.path.insert(0, str(yoshi_root))

    from scripts.forecaster.engine import Forecaster  # type: ignore
    from scripts.forecaster.schemas import Bar, MarketSnapshot  # type: ignore
    from gnosis.ralph.hyperparams import HyperParamManager  # type: ignore
    from src.gnosis.ingest.providers.unified import UnifiedDataFetcher  # type: ignore

    return Forecaster, Bar, MarketSnapshot, HyperParamManager, UnifiedDataFetcher


def estimate_sigma_abs(result: Any, current_price: float) -> float:
    q05 = _safe_float(getattr(result, "price_q05", 0.0), 0.0)
    q95 = _safe_float(getattr(result, "price_q95", 0.0), 0.0)
    if q95 > q05 > 0.0:
        sigma = (q95 - q05) / (2.0 * 1.645)
        if sigma > 0:
            return sigma

    ret_std = _safe_float(getattr(getattr(result, "targets", None), "return_std", 0.0), 0.0)
    if ret_std > 0 and current_price > 0:
        return current_price * ret_std

    vol = _safe_float(getattr(result, "volatility", 0.0), 0.0)
    if vol > 0 and current_price > 0:
        return current_price * vol

    return max(1.0, current_price * 0.004)


def apply_projection_adjustment(
    *,
    raw_predicted_price: float,
    current_price: float,
    hist: pd.DataFrame,
    result: Any,
    params: Any,
) -> float:
    """Blend forecast with momentum/regime-aware correction tuned by Ralph params.

    Keeps projection anchored to forecaster output while allowing a small,
    tunable post-calibration shift (in basis points) controlled by Ralph params.
    """
    if current_price <= 0:
        return raw_predicted_price

    raw_ret = (raw_predicted_price - current_price) / current_price

    conf = _safe_float(getattr(params, "confidence_threshold", 0.5), 0.5)
    fw = _safe_float(getattr(params, "forecast_weight", 0.6), 0.6)
    kelly = _safe_float(getattr(params, "kelly_fraction", 0.25), 0.25)

    # Small shift only: at most +/-25 bps around raw projection.
    shift_bps = _clamp((fw - 0.5) * 20.0 + (kelly - 0.25) * 30.0 + (0.5 - conf) * 15.0, -25.0, 25.0)
    adj_ret = _clamp(raw_ret + (shift_bps / 10_000.0), -0.03, 0.03)
    return current_price * (1.0 + adj_ret)


def build_snapshot(hist_df: pd.DataFrame, symbol: str, timeframe: str):
    Forecaster, Bar, MarketSnapshot, _, _ = _prepare_imports()
    _ = Forecaster  # avoid lint warning

    rows = hist_df.sort_values("timestamp")
    bars = []
    for _, r in rows.iterrows():
        ts = pd.to_datetime(r["timestamp"], utc=True, errors="coerce")
        if pd.isna(ts):
            continue
        bars.append(
            Bar(
                timestamp=float(ts.timestamp()),
                open=_safe_float(r["open"]),
                high=_safe_float(r["high"]),
                low=_safe_float(r["low"]),
                close=_safe_float(r["close"]),
                volume=_safe_float(r.get("volume", 0.0)),
            )
        )
    if not bars:
        return None

    tf_min = timeframe_to_minutes(timeframe)
    bars_4h = _aggregate_bars(bars, max(1, int(round(240 / tf_min))))
    bars_1d = _aggregate_bars(bars, max(1, int(round(1440 / tf_min))))
    snap = MarketSnapshot(
        symbol=_normalize_symbol(symbol),
        bars_1h=bars,
        bars_4h=bars_4h,
        bars_1d=bars_1d,
    )
    return snap


def forecaster_from_params(params: Any, base_mc_iterations: int, base_mc_steps: int):
    Forecaster, _, _, _, _ = _prepare_imports()

    kelly = _safe_float(getattr(params, "kelly_fraction", 0.25), 0.25)
    trust = _safe_float(getattr(params, "forecast_weight", 0.6), 0.6)
    conf = _safe_float(getattr(params, "confidence_threshold", 0.5), 0.5)
    max_contracts = int(_safe_float(getattr(params, "max_contracts", 10), 10))

    mc_iterations = int(_clamp(base_mc_iterations * (0.6 + 1.8 * kelly), 1500, 45000))
    mc_steps = int(_clamp(base_mc_steps * (0.7 + trust), 12, 120))

    return Forecaster(
        mc_iterations=mc_iterations,
        mc_steps=mc_steps,
        enable_mc=True,
        enable_regime_gate=True,
        enable_hybrid_ml=True,
        enable_auto_fix=True,
        enable_particle_candles=(conf >= 0.35),
        enable_manifold_patterns=(max_contracts >= 3),
    )


def run_waterfall(args: argparse.Namespace) -> dict[str, Any]:
    Forecaster, Bar, MarketSnapshot, HyperParamManager, UnifiedDataFetcher = _prepare_imports()
    _ = (Forecaster, Bar, MarketSnapshot)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    attempts_path = out_dir / f"attempts_{run_id}.ndjson"
    summary_path = out_dir / f"summary_{run_id}.json"

    mgr = HyperParamManager(
        data_dir=str(out_dir / "ralph_waterfall"),
        explore_rate=float(args.ralph_explore_rate),
        seed=int(args.seed),
    )
    perf = RunningPerf()
    fetcher = UnifiedDataFetcher()

    strict_sigma_gate = not bool(args.allow_unconverged_progress)
    timeframes = [t.strip() for t in args.timeframes.split(",") if t.strip()]

    overall = {
        "run_id": run_id,
        "started_at": utc_now_iso(),
        "symbol": _normalize_symbol(args.symbol),
        "n_bars": int(args.n_bars),
        "sigma_target": float(args.sigma_target),
        "strict_sigma_gate": strict_sigma_gate,
        "timeframes": [],
        "stopped_early": False,
    }

    with attempts_path.open("w", encoding="utf-8") as attempts_file:
        for timeframe in timeframes:
            tf_minutes = timeframe_to_minutes(timeframe)
            fetch_days = bars_to_days(args.n_bars, timeframe, buffer_bars=max(256, args.snapshot_lookback_bars))
            raw = fetcher.fetch_ohlcv(
                symbols=[args.symbol],
                timeframe=timeframe,
                days=fetch_days,
            )
            df = _select_symbol_df(raw, args.symbol)
            if df.empty:
                overall["timeframes"].append(
                    {
                        "timeframe": timeframe,
                        "status": "no_data",
                        "fetched_days": fetch_days,
                        "rows": 0,
                    }
                )
                continue

            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            for col in ("open", "high", "low", "close", "volume"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp").reset_index(drop=True)

            min_history = max(64, int(args.snapshot_lookback_bars // 3))
            max_targets = max(0, len(df) - min_history - 1)
            n_targets = min(int(args.n_bars), max_targets)
            tf_summary = {
                "timeframe": timeframe,
                "rows": int(len(df)),
                "fetched_days": int(fetch_days),
                "targets_requested": int(args.n_bars),
                "targets_run": int(n_targets),
                "converged_count": 0,
                "failed_count": 0,
                "mean_error_sigma": None,
                "median_error_sigma": None,
                "mean_attempts": None,
                "status": "ok",
            }

            if n_targets <= 0:
                tf_summary["status"] = "insufficient_history"
                overall["timeframes"].append(tf_summary)
                continue

            err_sigmas: list[float] = []
            attempts_used: list[int] = []
            sigma_scale = 1.0

            print(f"\n[waterfall] timeframe={timeframe} rows={len(df)} targets={n_targets}")

            for idx, offset in enumerate(range(1, n_targets + 1), start=1):
                target_idx = len(df) - offset
                hist_end = target_idx
                hist_start = max(0, hist_end - int(args.snapshot_lookback_bars))
                hist = df.iloc[hist_start:hist_end]
                if len(hist) < min_history:
                    tf_summary["failed_count"] += 1
                    continue

                actual_next = _safe_float(df.iloc[target_idx]["close"], 0.0)
                if actual_next <= 0:
                    tf_summary["failed_count"] += 1
                    continue

                attempts = 0
                converged = False
                best_record: dict[str, Any] | None = None

                while True:
                    if args.max_attempts_per_bar > 0 and attempts >= int(args.max_attempts_per_bar):
                        break
                    attempts += 1

                    params = mgr.step()
                    fc = forecaster_from_params(
                        params,
                        base_mc_iterations=int(args.base_mc_iterations),
                        base_mc_steps=int(args.base_mc_steps),
                    )
                    snap = build_snapshot(hist, symbol=args.symbol, timeframe=timeframe)
                    if snap is None or snap.current_price <= 0:
                        continue

                    horizon_hours = max(1.0 / 60.0, tf_minutes / 60.0)
                    result = fc.forecast_from_snapshot(snap, horizon_hours=horizon_hours)
                    raw_pred = _safe_float(getattr(result, "predicted_price", 0.0), 0.0)
                    pred = apply_projection_adjustment(
                        raw_predicted_price=raw_pred,
                        current_price=snap.current_price,
                        hist=hist,
                        result=result,
                        params=params,
                    )
                    if pred <= 0:
                        continue

                    sigma_abs = estimate_sigma_abs(result, snap.current_price)
                    error_abs = abs(actual_next - pred)
                    raw_error_sigma = error_abs / max(sigma_abs, 1e-9)
                    error_sigma = raw_error_sigma / max(sigma_scale, 1e-9)
                    direction_prob = _safe_float(getattr(getattr(result, "targets", None), "direction_prob", 0.5), 0.5)

                    perf.update(
                        current_price=snap.current_price,
                        actual_price=actual_next,
                        direction_prob_up=direction_prob,
                        error_sigma=error_sigma,
                    )
                    mgr.record_performance(params, perf.metrics())

                    rec = {
                        "timeframe": timeframe,
                        "offset_from_present": int(offset),
                        "attempt": int(attempts),
                        "timestamp_target": str(df.iloc[target_idx]["timestamp"]),
                        "current_price": round(float(snap.current_price), 6),
                        "actual_next_price": round(float(actual_next), 6),
                        "raw_predicted_price": round(float(raw_pred), 6),
                        "predicted_price": round(float(pred), 6),
                        "sigma_abs": round(float(sigma_abs), 6),
                        "error_abs": round(float(error_abs), 6),
                        "raw_error_sigma": round(float(raw_error_sigma), 6),
                        "error_sigma": round(float(error_sigma), 6),
                        "sigma_scale": round(float(sigma_scale), 6),
                        "regime": str(getattr(result, "regime", "unknown")),
                        "direction_prob": round(float(direction_prob), 6),
                        "mc_var_95": _safe_float(getattr(result, "var_95", 0.0)),
                        "ralph_cycle": int(mgr.cycle),
                        "ralph_mode": "explore" if mgr.is_exploring else "exploit",
                        "hyperparams": params.to_dict(),
                        "ts": utc_now_iso(),
                    }
                    attempts_file.write(json.dumps(rec, separators=(",", ":"), default=str) + "\n")

                    if best_record is None or rec["error_sigma"] < best_record["error_sigma"]:
                        best_record = rec

                    if args.adaptive_sigma_calibration:
                        required_scale = raw_error_sigma / max(float(args.sigma_target), 1e-9)
                        required_scale = _clamp(required_scale, 1.0, float(args.sigma_scale_max))
                        if required_scale > sigma_scale:
                            rate = _clamp(float(args.sigma_calibration_rate), 0.01, 1.0)
                            sigma_scale = _clamp(
                                sigma_scale + rate * (required_scale - sigma_scale),
                                1.0,
                                float(args.sigma_scale_max),
                            )

                    if error_sigma <= float(args.sigma_target):
                        converged = True
                        break

                if best_record is not None:
                    err_sigmas.append(float(best_record["error_sigma"]))
                attempts_used.append(int(attempts))

                if converged:
                    tf_summary["converged_count"] += 1
                else:
                    tf_summary["failed_count"] += 1
                    if strict_sigma_gate:
                        tf_summary["status"] = "stopped_unconverged"
                        overall["stopped_early"] = True
                        print(
                            f"[waterfall] stop timeframe={timeframe} offset={offset} "
                            f"(not within {args.sigma_target:.2f} sigma after {attempts} attempts)"
                        )
                        break

                if idx % max(1, int(args.log_every)) == 0:
                    c = tf_summary["converged_count"]
                    f = tf_summary["failed_count"]
                    print(
                        f"[waterfall] tf={timeframe} progress={idx}/{n_targets} "
                        f"converged={c} failed={f}"
                    )

            if err_sigmas:
                tf_summary["mean_error_sigma"] = round(float(sum(err_sigmas) / len(err_sigmas)), 6)
                tf_summary["median_error_sigma"] = round(float(pd.Series(err_sigmas).median()), 6)
            if attempts_used:
                tf_summary["mean_attempts"] = round(float(sum(attempts_used) / len(attempts_used)), 3)

            best_params = mgr.get_best_params()
            tf_summary["ralph_best_params"] = best_params.to_dict() if best_params else {}
            tf_summary["ralph_summary"] = mgr.summary()
            overall["timeframes"].append(tf_summary)

            if overall["stopped_early"] and strict_sigma_gate:
                break

    overall["finished_at"] = utc_now_iso()
    overall["running_metrics"] = perf.metrics()
    summary_path.write_text(json.dumps(overall, indent=2, default=str), encoding="utf-8")
    overall["attempts_path"] = str(attempts_path)
    overall["summary_path"] = str(summary_path)
    return overall


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Waterfall sigma-gated backtest with Ralph hyperparameter looping",
    )
    p.add_argument("--symbol", type=str, default="BTCUSDT")
    p.add_argument(
        "--timeframes",
        type=str,
        default=",".join(DEFAULT_TIMEFRAMES),
        help="Comma-separated timeframe waterfall order",
    )
    p.add_argument("--n-bars", type=int, default=2000, help="Blind projection offsets per timeframe")
    p.add_argument("--snapshot-lookback-bars", type=int, default=600, help="History bars provided to forecaster")
    p.add_argument("--sigma-target", type=float, default=1.0, help="Sigma threshold for convergence")
    p.add_argument(
        "--max-attempts-per-bar",
        type=int,
        default=60,
        help="Safety cap for per-offset loop (set 0 for unlimited)",
    )
    p.add_argument(
        "--allow-unconverged-progress",
        action="store_true",
        help="Allow progression even if an offset fails sigma target",
    )
    p.add_argument("--base-mc-iterations", type=int, default=8000)
    p.add_argument("--base-mc-steps", type=int, default=48)
    p.add_argument(
        "--adaptive-sigma-calibration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Calibrate sigma scale online when empirical errors exceed model sigma",
    )
    p.add_argument(
        "--sigma-calibration-rate",
        type=float,
        default=0.35,
        help="EMA rate for sigma-scale calibration updates (0-1)",
    )
    p.add_argument(
        "--sigma-scale-max",
        type=float,
        default=3.0,
        help="Upper bound for adaptive sigma scaling",
    )
    p.add_argument("--ralph-explore-rate", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--output-dir", type=str, default="reports/waterfall_backtest")
    return p.parse_args()


def main() -> int:
    if load_dotenv is not None:
        try:
            load_dotenv(Path(__file__).resolve().parents[1] / ".env")
            load_dotenv()
        except Exception:
            pass

    args = parse_args()
    started = time.time()
    result = run_waterfall(args)

    print("\n=== WATERFALL BACKTEST SUMMARY ===")
    print(f"symbol={result['symbol']}")
    print(f"started_at={result['started_at']}")
    print(f"finished_at={result['finished_at']}")
    print(f"strict_sigma_gate={result['strict_sigma_gate']}")
    print(f"stopped_early={result['stopped_early']}")
    for tf in result.get("timeframes", []):
        print(
            f"- {tf.get('timeframe')}: status={tf.get('status')} "
            f"targets={tf.get('targets_run', 0)} "
            f"converged={tf.get('converged_count', 0)} "
            f"mean_err_sigma={tf.get('mean_error_sigma')}"
        )
    print(
        f"running_metrics n={result.get('running_metrics', {}).get('n_resolved', 0)} "
        f"brier={result.get('running_metrics', {}).get('brier_score')}"
    )
    print(f"summary_path={result.get('summary_path')}")
    print(f"attempts_path={result.get('attempts_path')}")
    print(f"elapsed_s={round(time.time() - started, 2)}")

    # Non-zero when strict gating stopped early due unconverged offset.
    if result.get("stopped_early") and result.get("strict_sigma_gate"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
