"""Regime-first backtest engine for crypto (1m regime ledger driven).

This backtester is intentionally "regime-first":
  - The regime ledger (with p_r and overlays) is the primary input.
  - The router score is a blend of playbook actions weighted by regime probs.
  - Execution costs (slippage/spread) scale with regime + liquidity overlay.
  - Outputs include a trade ledger with regime/transition attribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import json

import numpy as np
import pandas as pd

from gnosis.backtest.stats import StatsCalculator
from gnosis.regime_first.blending import PLAYBOOKS, REGIMES


def _safe_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float(default)


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, float(x))))


def _blend_by_regime(p_row: np.ndarray, per_regime: dict[str, float], default: float) -> float:
    vals = np.array([_safe_float(per_regime.get(r, default), default) for r in REGIMES], dtype=float)
    return float(np.dot(p_row, vals))


def _slip_components(
    row: pd.Series,
    exec_cfg: dict[str, Any],
) -> tuple[float, float, float]:
    """Return (spread_rate, vol_rate, total_rate) for this bar."""
    sl = (exec_cfg.get("slippage") or {}) if isinstance(exec_cfg, dict) else {}
    k_spread = _safe_float(sl.get("k_spread", 0.35), 0.35)
    k_vol = _safe_float(sl.get("k_vol", 0.15), 0.15)

    # Spread proxy: (H-L)/C (dimensionless).
    high = _safe_float(row.get("high"), 0.0)
    low = _safe_float(row.get("low"), 0.0)
    close = max(_safe_float(row.get("close"), 0.0), 1e-12)
    spread_proxy = max((high - low) / close, 0.0)

    # Vol proxy: ATR_1m / C (dimensionless).
    atr = max(_safe_float(row.get("atr_1m"), 0.0), 0.0)
    atr_frac = atr / close

    base_spread = k_spread * spread_proxy
    base_vol = k_vol * atr_frac

    # Regime multiplier blended by p_final
    mult_r = (sl.get("regime_multiplier") or {}) if isinstance(sl, dict) else {}
    p = np.array([_safe_float(row.get(f"p_{r}_final"), 0.0) for r in REGIMES], dtype=float)
    if p.sum() <= 0:
        p = np.array([1, 0, 0, 0, 0], dtype=float)
    p = p / p.sum()
    m_reg = _blend_by_regime(p, mult_r, default=1.0)

    # Liquidity overlay multiplier blended by overlay probs
    mult_o = (sl.get("liquidity_overlay_multiplier") or {}) if isinstance(sl, dict) else {}
    p_td = _safe_float(row.get("p_TD"), 0.0)
    p_nr = _safe_float(row.get("p_NR"), 1.0)
    p_th = _safe_float(row.get("p_TH"), 0.0)
    denom = max(p_td + p_nr + p_th, 1e-12)
    p_td, p_nr, p_th = p_td / denom, p_nr / denom, p_th / denom
    m_liq = (
        p_td * _safe_float(mult_o.get("TD", 1.0), 1.0)
        + p_nr * _safe_float(mult_o.get("NR", 1.0), 1.0)
        + p_th * _safe_float(mult_o.get("TH", 1.0), 1.0)
    )

    spread_rate = base_spread * m_reg * m_liq
    vol_rate = base_vol * m_reg * m_liq
    total = spread_rate + vol_rate
    # Hard cap to avoid pathological fills on bad data.
    total = float(min(max(total, 0.0), 0.05))  # 5% per fill cap
    # Keep components consistent with cap.
    if spread_rate + vol_rate > 0:
        scale = total / (spread_rate + vol_rate)
        spread_rate *= scale
        vol_rate *= scale
    return float(spread_rate), float(vol_rate), float(total)


def _apply_slip(px_model: float, side: str, slip_rate_total: float) -> float:
    if side.upper() == "BUY":
        return float(px_model * (1.0 + slip_rate_total))
    return float(px_model * (1.0 - slip_rate_total))


def _fee_rate(exec_cfg: dict[str, Any]) -> float:
    fees = (exec_cfg.get("fees") or {}) if isinstance(exec_cfg, dict) else {}
    taker_bps = _safe_float(fees.get("taker_bps", 4.0), 4.0)
    return float(taker_bps / 10000.0)


@dataclass
class BacktestOutput:
    equity_curve: pd.DataFrame
    trade_ledger: pd.DataFrame
    stats: dict[str, Any]
    diagnostics: dict[str, Any]


def run_regime_first_backtest(
    ledger: pd.DataFrame,
    raw_cfg: dict[str, Any],
    W: dict[str, dict[str, float]],
    *,
    fee_mult: float = 1.0,
    slip_mult: float = 1.0,
) -> BacktestOutput:
    """Run a simple 1m, next-open execution backtest driven by router_score.

    The ledger must already include:
      - router_score, permission_P, top_playbook
      - p_<R>_final for regimes
      - overlay probabilities p_TD/p_NR/p_TH
    """
    df = ledger.copy()
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    risk = raw_cfg.get("risk", {}) or {}
    exec_cfg = raw_cfg.get("execution", {}) or {}
    system = raw_cfg.get("system", {}) or {}
    instrument = str(system.get("instrument", "spot"))

    initial_capital_total = _safe_float(risk.get("initial_capital", 10_000.0), 10_000.0)
    pos_pct = _safe_float(risk.get("position_size_pct", 0.10), 0.10)
    entry_threshold = _safe_float(risk.get("entry_threshold", 0.15), 0.15)
    min_permission = _safe_float(risk.get("min_permission", 0.35), 0.35)

    thresholds_by_regime = (risk.get("thresholds_by_regime") or {}) if isinstance(risk, dict) else {}
    stop_mults = (thresholds_by_regime.get("stop_atr_mult") or {}) if isinstance(thresholds_by_regime, dict) else {}
    tp_mults = (thresholds_by_regime.get("takeprofit_atr_mult") or {}) if isinstance(thresholds_by_regime, dict) else {}

    # Allocate capital equally across symbols.
    symbols = sorted(df["symbol"].unique().tolist())
    cap_per_symbol = initial_capital_total / max(len(symbols), 1)

    fee_rate = _fee_rate(exec_cfg) * float(fee_mult)

    trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []

    for sym in symbols:
        s = df[df["symbol"] == sym].sort_values("timestamp").reset_index(drop=True)
        if len(s) < 5:
            continue

        cash = float(cap_per_symbol)
        pos_qty = 0.0  # signed; +long, -short
        entry_ts: Optional[pd.Timestamp] = None
        entry_px_model = 0.0
        entry_px_exec = 0.0
        entry_fee = 0.0
        entry_spread_cost = 0.0
        entry_slip_cost = 0.0
        entry_playbook = ""
        entry_router_score = 0.0
        entry_regime = ""
        entry_p_vec: dict[str, float] = {}
        entry_transition = ""
        stop_px: Optional[float] = None
        tp_px: Optional[float] = None
        mfe = 0.0
        mae = 0.0
        bars_in_trade = 0
        dwell: dict[str, int] = {r: 0 for r in REGIMES}
        transition_events: list[str] = []
        last_regime_in_trade: Optional[str] = None

        pending_target = 0  # -1 short, 0 flat, +1 long
        pending_reason = ""

        def _pos_sign(q: float) -> int:
            if q > 0:
                return 1
            if q < 0:
                return -1
            return 0

        trade_id = 0

        for i in range(len(s)):
            row = s.iloc[i]
            ts = row["timestamp"]
            o = _safe_float(row.get("open"), 0.0)
            h = _safe_float(row.get("high"), o)
            l = _safe_float(row.get("low"), o)
            c = _safe_float(row.get("close"), o)

            spread_rate, vol_rate, slip_rate = _slip_components(row, exec_cfg)
            slip_rate *= float(slip_mult)
            spread_rate *= float(slip_mult)
            vol_rate *= float(slip_mult)
            slip_rate = float(min(max(slip_rate, 0.0), 0.05))

            # ---- 1) Execute pending position change at bar open ----
            cur_sign = _pos_sign(pos_qty)
            if pending_target != cur_sign:
                # Close existing position (market at open)
                if pos_qty != 0.0:
                    side_close = "SELL" if pos_qty > 0 else "BUY"
                    exit_px_model = o
                    exit_px_exec = _apply_slip(exit_px_model, side_close, slip_rate)
                    exit_fee = abs(pos_qty) * exit_px_exec * fee_rate
                    cash += pos_qty * exit_px_exec
                    cash -= exit_fee

                    pnl_gross = pos_qty * (exit_px_model - entry_px_model)
                    pnl_exec = pos_qty * (exit_px_exec - entry_px_exec)
                    fees = entry_fee + exit_fee
                    spread_cost = entry_spread_cost + abs(pos_qty) * exit_px_model * spread_rate
                    slippage_cost = entry_slip_cost + abs(pos_qty) * exit_px_model * vol_rate
                    pnl_net = pnl_exec - fees

                    trade_id += 1
                    trades.append(
                        {
                            "trade_id": f"{sym}-{trade_id}",
                            "symbol": sym,
                            "side": "long" if pos_qty > 0 else "short",
                            "instrument": instrument,
                            "entry_ts": entry_ts,
                            "exit_ts": ts,
                            "entry_px_model": float(entry_px_model),
                            "exit_px_model": float(exit_px_model),
                            "entry_px_exec": float(entry_px_exec),
                            "exit_px_exec": float(exit_px_exec),
                            "qty": float(abs(pos_qty)),
                            "notional": float(abs(pos_qty) * entry_px_exec),
                            "fees": float(fees),
                            "slippage_cost": float(slippage_cost),
                            "spread_cost": float(spread_cost),
                            "funding_pnl": 0.0,
                            "pnl_gross": float(pnl_gross),
                            "pnl_net": float(pnl_net),
                            "ret_net": float(pnl_net / max(abs(pos_qty) * entry_px_exec, 1e-12)),
                            "mfe": float(mfe),
                            "mae": float(mae),
                            "hold_minutes": int((ts - entry_ts).total_seconds() // 60) if entry_ts is not None else 0,
                            "bars_in_trade": int(bars_in_trade),
                            "regime_entry_label": str(entry_regime),
                            "regime_exit_label": str(row.get("regime_label_final", row.get("regime_label", ""))),
                            "transition_entry": str(entry_transition),
                            "p_vec_entry": json.dumps(entry_p_vec, sort_keys=True),
                            "regime_dwell_minutes": json.dumps({k: int(v) for k, v in dwell.items()}, sort_keys=True),
                            "transition_events": json.dumps(list(transition_events)),
                            "playbook_id": str(entry_playbook),
                            "router_score": float(entry_router_score),
                            "exit_reason": "signal_flip",
                            "standdown_reason": "",
                        }
                    )

                    # Reset position state
                    pos_qty = 0.0
                    entry_ts = None
                    entry_px_model = 0.0
                    entry_px_exec = 0.0
                    entry_fee = 0.0
                    entry_spread_cost = 0.0
                    entry_slip_cost = 0.0
                    entry_playbook = ""
                    entry_router_score = 0.0
                    entry_regime = ""
                    entry_p_vec = {}
                    entry_transition = ""
                    stop_px = None
                    tp_px = None
                    mfe = 0.0
                    mae = 0.0
                    bars_in_trade = 0
                    dwell = {r: 0 for r in REGIMES}
                    transition_events = []
                    last_regime_in_trade = None

                # Open new position (market at open)
                if pending_target != 0:
                    if instrument == "spot" and pending_target < 0:
                        # Spot mode: no shorts; ignore.
                        pending_target = 0
                    else:
                        notional = max(cash * pos_pct, 0.0)
                        side_open = "BUY" if pending_target > 0 else "SELL"
                        entry_px_model = o
                        entry_px_exec = _apply_slip(entry_px_model, side_open, slip_rate)
                        qty = notional / max(entry_px_exec, 1e-12)
                        fee = notional * fee_rate
                        # Update cash/position
                        if side_open == "BUY":
                            cash -= qty * entry_px_exec
                            cash -= fee
                            pos_qty = qty
                        else:
                            cash += qty * entry_px_exec
                            cash -= fee
                            pos_qty = -qty

                        entry_ts = ts
                        entry_fee = fee
                        entry_spread_cost = qty * entry_px_model * spread_rate
                        entry_slip_cost = qty * entry_px_model * vol_rate
                        entry_playbook = str(row.get("top_playbook", ""))
                        entry_router_score = float(row.get("router_score", 0.0))
                        entry_regime = str(row.get("regime_label_final", row.get("regime_label", "")))
                        entry_p_vec = {r: float(row.get(f"p_{r}_final", 0.0)) for r in REGIMES}
                        entry_transition = str(row.get("transition_type_final", row.get("transition_type", "")))

                        # Stops/TP based on blended per-regime ATR multiples.
                        p_vec = np.array([entry_p_vec[r] for r in REGIMES], dtype=float)
                        denom = max(p_vec.sum(), 1e-12)
                        p_vec = p_vec / denom
                        stop_mult = _blend_by_regime(p_vec, stop_mults, default=1.8)
                        tp_mult = _blend_by_regime(p_vec, tp_mults, default=1.8)
                        atr = max(_safe_float(row.get("atr_1m"), 0.0), 0.0)
                        if pos_qty > 0:
                            stop_px = entry_px_exec - stop_mult * atr
                            tp_px = entry_px_exec + tp_mult * atr if tp_mult > 0 else None
                        else:
                            stop_px = entry_px_exec + stop_mult * atr
                            tp_px = entry_px_exec - tp_mult * atr if tp_mult > 0 else None

                        mfe = 0.0
                        mae = 0.0
                        bars_in_trade = 0
                        dwell = {r: 0 for r in REGIMES}
                        transition_events = []
                        last_regime_in_trade = None

                pending_target = pending_target  # explicit
                pending_reason = ""

            # ---- 2) If position open, update attribution and check stop/TP ----
            if pos_qty != 0.0 and entry_ts is not None:
                bars_in_trade += 1
                cur_reg = str(row.get("regime_label_final", row.get("regime_label", "")))
                if cur_reg in dwell:
                    dwell[cur_reg] += 1
                if last_regime_in_trade is not None and cur_reg != last_regime_in_trade:
                    transition_events.append(f"{last_regime_in_trade}->{cur_reg}")
                last_regime_in_trade = cur_reg

                # Update MFE/MAE from OHLC.
                if pos_qty > 0:
                    mfe = max(mfe, (h - entry_px_exec) / max(entry_px_exec, 1e-12))
                    mae = min(mae, (l - entry_px_exec) / max(entry_px_exec, 1e-12))
                else:
                    mfe = max(mfe, (entry_px_exec - l) / max(entry_px_exec, 1e-12))
                    mae = min(mae, (entry_px_exec - h) / max(entry_px_exec, 1e-12))

                # Stop/TP trigger logic (conservative if both hit).
                exit_reason = None
                exit_px_model = None
                if pos_qty > 0:
                    stop_hit = (stop_px is not None) and (l <= float(stop_px))
                    tp_hit = (tp_px is not None) and (h >= float(tp_px))
                    if stop_hit and tp_hit:
                        exit_reason = "stop"
                    elif stop_hit:
                        exit_reason = "stop"
                    elif tp_hit:
                        exit_reason = "takeprofit"
                    if exit_reason == "stop":
                        exit_px_model = min(float(stop_px), o)  # gap down -> worse
                    elif exit_reason == "takeprofit":
                        exit_px_model = max(float(tp_px), o)  # gap up -> better
                else:
                    stop_hit = (stop_px is not None) and (h >= float(stop_px))
                    tp_hit = (tp_px is not None) and (l <= float(tp_px))
                    if stop_hit and tp_hit:
                        exit_reason = "stop"
                    elif stop_hit:
                        exit_reason = "stop"
                    elif tp_hit:
                        exit_reason = "takeprofit"
                    if exit_reason == "stop":
                        exit_px_model = max(float(stop_px), o)  # gap up -> worse
                    elif exit_reason == "takeprofit":
                        exit_px_model = min(float(tp_px), o)  # gap down -> better

                if exit_reason is not None and exit_px_model is not None:
                    side_close = "SELL" if pos_qty > 0 else "BUY"
                    exit_px_exec = _apply_slip(float(exit_px_model), side_close, slip_rate)
                    exit_fee = abs(pos_qty) * exit_px_exec * fee_rate
                    cash += pos_qty * exit_px_exec
                    cash -= exit_fee

                    pnl_gross = pos_qty * (float(exit_px_model) - entry_px_model)
                    pnl_exec = pos_qty * (exit_px_exec - entry_px_exec)
                    fees = entry_fee + exit_fee
                    spread_cost = entry_spread_cost + abs(pos_qty) * float(exit_px_model) * spread_rate
                    slippage_cost = entry_slip_cost + abs(pos_qty) * float(exit_px_model) * vol_rate
                    pnl_net = pnl_exec - fees

                    trade_id += 1
                    trades.append(
                        {
                            "trade_id": f"{sym}-{trade_id}",
                            "symbol": sym,
                            "side": "long" if pos_qty > 0 else "short",
                            "instrument": instrument,
                            "entry_ts": entry_ts,
                            "exit_ts": ts,
                            "entry_px_model": float(entry_px_model),
                            "exit_px_model": float(exit_px_model),
                            "entry_px_exec": float(entry_px_exec),
                            "exit_px_exec": float(exit_px_exec),
                            "qty": float(abs(pos_qty)),
                            "notional": float(abs(pos_qty) * entry_px_exec),
                            "fees": float(fees),
                            "slippage_cost": float(slippage_cost),
                            "spread_cost": float(spread_cost),
                            "funding_pnl": 0.0,
                            "pnl_gross": float(pnl_gross),
                            "pnl_net": float(pnl_net),
                            "ret_net": float(pnl_net / max(abs(pos_qty) * entry_px_exec, 1e-12)),
                            "mfe": float(mfe),
                            "mae": float(mae),
                            "hold_minutes": int((ts - entry_ts).total_seconds() // 60),
                            "bars_in_trade": int(bars_in_trade),
                            "regime_entry_label": str(entry_regime),
                            "regime_exit_label": str(row.get("regime_label_final", row.get("regime_label", ""))),
                            "transition_entry": str(entry_transition),
                            "p_vec_entry": json.dumps(entry_p_vec, sort_keys=True),
                            "regime_dwell_minutes": json.dumps({k: int(v) for k, v in dwell.items()}, sort_keys=True),
                            "transition_events": json.dumps(list(transition_events)),
                            "playbook_id": str(entry_playbook),
                            "router_score": float(entry_router_score),
                            "exit_reason": str(exit_reason),
                            "standdown_reason": "",
                        }
                    )

                    # Reset
                    pos_qty = 0.0
                    entry_ts = None
                    entry_px_model = 0.0
                    entry_px_exec = 0.0
                    entry_fee = 0.0
                    entry_spread_cost = 0.0
                    entry_slip_cost = 0.0
                    entry_playbook = ""
                    entry_router_score = 0.0
                    entry_regime = ""
                    entry_p_vec = {}
                    entry_transition = ""
                    stop_px = None
                    tp_px = None
                    mfe = 0.0
                    mae = 0.0
                    bars_in_trade = 0
                    dwell = {r: 0 for r in REGIMES}
                    transition_events = []
                    last_regime_in_trade = None

            # ---- 3) Mark-to-market equity at close ----
            equity = cash + pos_qty * c
            equity_rows.append({"timestamp": ts, "symbol": sym, "equity": float(equity)})

            # ---- 4) Decide next target for next bar open ----
            A = _safe_float(row.get("router_score", 0.0), 0.0)
            P = _safe_float(row.get("permission_P", 1.0), 1.0)
            if P < min_permission:
                pending_target = 0
                pending_reason = "permission"
            elif abs(A) < entry_threshold:
                pending_target = 0
                pending_reason = "weak_score"
            else:
                desired = 1 if A > 0 else -1
                if instrument == "spot" and desired < 0:
                    desired = 0
                pending_target = desired
                pending_reason = ""

        # Force close at last close if still open.
        if pos_qty != 0.0 and entry_ts is not None:
            row = s.iloc[-1]
            ts = row["timestamp"]
            c = _safe_float(row.get("close"), 0.0)
            spread_rate, vol_rate, slip_rate = _slip_components(row, exec_cfg)
            slip_rate *= float(slip_mult)
            side_close = "SELL" if pos_qty > 0 else "BUY"
            exit_px_model = c
            exit_px_exec = _apply_slip(exit_px_model, side_close, slip_rate)
            exit_fee = abs(pos_qty) * exit_px_exec * fee_rate
            cash += pos_qty * exit_px_exec
            cash -= exit_fee

            pnl_gross = pos_qty * (exit_px_model - entry_px_model)
            pnl_exec = pos_qty * (exit_px_exec - entry_px_exec)
            fees = entry_fee + exit_fee
            spread_cost = entry_spread_cost + abs(pos_qty) * exit_px_model * spread_rate
            slippage_cost = entry_slip_cost + abs(pos_qty) * exit_px_model * vol_rate
            pnl_net = pnl_exec - fees

            trade_id += 1
            trades.append(
                {
                    "trade_id": f"{sym}-{trade_id}",
                    "symbol": sym,
                    "side": "long" if pos_qty > 0 else "short",
                    "instrument": instrument,
                    "entry_ts": entry_ts,
                    "exit_ts": ts,
                    "entry_px_model": float(entry_px_model),
                    "exit_px_model": float(exit_px_model),
                    "entry_px_exec": float(entry_px_exec),
                    "exit_px_exec": float(exit_px_exec),
                    "qty": float(abs(pos_qty)),
                    "notional": float(abs(pos_qty) * entry_px_exec),
                    "fees": float(fees),
                    "slippage_cost": float(slippage_cost),
                    "spread_cost": float(spread_cost),
                    "funding_pnl": 0.0,
                    "pnl_gross": float(pnl_gross),
                    "pnl_net": float(pnl_net),
                    "ret_net": float(pnl_net / max(abs(pos_qty) * entry_px_exec, 1e-12)),
                    "mfe": float(mfe),
                    "mae": float(mae),
                    "hold_minutes": int((ts - entry_ts).total_seconds() // 60),
                    "bars_in_trade": int(bars_in_trade),
                    "regime_entry_label": str(entry_regime),
                    "regime_exit_label": str(row.get("regime_label_final", row.get("regime_label", ""))),
                    "transition_entry": str(entry_transition),
                    "p_vec_entry": json.dumps(entry_p_vec, sort_keys=True),
                    "regime_dwell_minutes": json.dumps({k: int(v) for k, v in dwell.items()}, sort_keys=True),
                    "transition_events": json.dumps(list(transition_events)),
                    "playbook_id": str(entry_playbook),
                    "router_score": float(entry_router_score),
                    "exit_reason": "eod_close",
                    "standdown_reason": "",
                }
            )

    equity_curve = pd.DataFrame(equity_rows).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    trade_ledger = pd.DataFrame(trades)

    # Aggregate equity across symbols to compute global stats.
    if not equity_curve.empty:
        eq = equity_curve.groupby("timestamp")["equity"].sum().reset_index()
        stats_trades = trade_ledger.rename(columns={"fees": "fee", "pnl_net": "pnl"}).copy()
        stats_trades["fee"] = stats_trades.get("fee", 0.0)
        stats_trades["pnl"] = stats_trades.get("pnl", 0.0)
        stats = StatsCalculator.compute(eq, stats_trades, initial_capital=float(initial_capital_total))
    else:
        stats = StatsCalculator.compute(pd.DataFrame(columns=["timestamp", "equity"]), pd.DataFrame(), float(initial_capital_total))

    diagnostics = {
        "n_symbols": int(len(symbols)),
        "symbols": symbols,
        "fee_mult": float(fee_mult),
        "slip_mult": float(slip_mult),
        "W": W,
    }
    return BacktestOutput(equity_curve=equity_curve, trade_ledger=trade_ledger, stats=stats, diagnostics=diagnostics)

