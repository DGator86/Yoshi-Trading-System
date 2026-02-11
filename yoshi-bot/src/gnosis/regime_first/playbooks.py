"""Regime playbooks (vectorized) used by regime-first router/backtest.

Each playbook emits per-bar:
  - s in [-1, +1]  (directional intent)
  - q in [0, 1]    (quality)
  - u in [0, 1]    (urgency)
  - a = s*q*u      (comparable action score)

These playbooks are intentionally simple baselines; the regime-first framework
is designed to evaluate/replace them over time.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _clip11(x: np.ndarray) -> np.ndarray:
    return np.clip(x, -1.0, 1.0)


def _safe_col(df: pd.DataFrame, name: str, default: float = 0.0) -> np.ndarray:
    if name not in df.columns:
        return np.full(len(df), float(default), dtype=float)
    return pd.to_numeric(df[name], errors="coerce").fillna(default).to_numpy(dtype=float)


def add_playbook_outputs(df_ledger: pd.DataFrame) -> pd.DataFrame:
    """Add baseline playbook outputs to the regime ledger.

    Adds columns:
      pb_<id>_{s,q,u,a}
    """
    df = df_ledger.copy()

    close = _safe_col(df, "close", default=0.0)
    open_ = _safe_col(df, "open", default=0.0)
    atr = _safe_col(df, "atr_1m", default=0.0)
    tr_shock = _safe_col(df, "tr_shock", default=1.0)
    vol_imp = _safe_col(df, "volume_impulse", default=1.0)
    comp = _safe_col(df, "comp", default=1.0)
    clv = _safe_col(df, "clv", default=0.0)
    er60 = _safe_col(df, "er_60m", default=0.0)
    persist = _safe_col(df, "directional_persistence", default=0.0)

    # ===== MR fade =====
    # Fade deviation from EMA(30) using ATR as scale.
    if "symbol" in df.columns:
        ema30 = df.groupby("symbol")["close"].transform(lambda s: s.ewm(span=30, adjust=False, min_periods=10).mean())
        ema30 = pd.to_numeric(ema30, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        ema30 = pd.Series(close).ewm(span=30, adjust=False, min_periods=10).mean().to_numpy(dtype=float)

    dev = (close - ema30)
    dev_atr = dev / (atr + 1e-12)
    s_mr = _clip11(-np.sign(dev_atr))  # fade
    q_mr = _clip01((np.abs(dev_atr) - 0.5) / (2.0 - 0.5))  # ramp 0.5..2.0
    u_mr = _clip01(1.0 - (tr_shock - 1.2) / (3.0 - 1.2))  # less urgent in shocks
    a_mr = _clip11(s_mr * q_mr * u_mr)

    df["pb_mr_fade_s"] = s_mr
    df["pb_mr_fade_q"] = q_mr
    df["pb_mr_fade_u"] = u_mr
    df["pb_mr_fade_a"] = a_mr

    # ===== Trend follow =====
    # Follow EMA slope (1h-ish) and/or persistence + ER.
    slope = _safe_col(df, "slope_ema_1h", default=0.0)
    s_tr = _clip11(np.sign(slope))
    q_tr = _clip01(((er60 - 0.25) / (0.60 - 0.25)) * ((persist - 0.35) / (0.80 - 0.35)))
    u_tr = _clip01((np.abs(clv) - 0.20) / (0.85 - 0.20))
    a_tr = _clip11(s_tr * q_tr * u_tr)

    df["pb_trend_follow_s"] = s_tr
    df["pb_trend_follow_q"] = q_tr
    df["pb_trend_follow_u"] = u_tr
    df["pb_trend_follow_a"] = a_tr

    # ===== Breakout / transition =====
    # Designed to activate when post-compression expansion begins.
    inv_comp = 1.0 / (comp + 1e-12)
    q_bo = _clip01(((tr_shock - 1.2) / (2.5 - 1.2)) * ((vol_imp - 1.2) / (2.5 - 1.2)))
    u_bo = _clip01((inv_comp - 1.2) / (1.8 - 1.2))
    # Direction uses close-to-close return sign (momentum).
    ret = _safe_col(df, "ret_1m", default=0.0)
    s_bo = _clip11(np.sign(ret))
    a_bo = _clip11(s_bo * q_bo * u_bo)

    df["pb_breakout_transition_s"] = s_bo
    df["pb_breakout_transition_q"] = q_bo
    df["pb_breakout_transition_u"] = u_bo
    df["pb_breakout_transition_a"] = a_bo

    # ===== Panic only =====
    # Baseline behavior is stand-down (a=0). If enabled later, can implement
    # reversal/mean reversion in LQ with strict constraints.
    df["pb_panic_only_s"] = 0.0
    df["pb_panic_only_q"] = 0.0
    df["pb_panic_only_u"] = 0.0
    df["pb_panic_only_a"] = 0.0

    return df


def playbook_ids() -> list[str]:
    return ["mr_fade", "trend_follow", "breakout_transition", "panic_only"]

