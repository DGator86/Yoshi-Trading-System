"""Regime-first reporting: distributions, shift flags, attribution metrics."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from gnosis.regime_first.blending import REGIMES


def _safe_prob_vec(d: dict[str, float]) -> np.ndarray:
    p = np.array([float(d.get(r, 0.0)) for r in REGIMES], dtype=float)
    p = np.clip(p, 0.0, 1.0)
    s = float(p.sum())
    if s <= 0.0:
        p[:] = 1.0 / len(REGIMES)
    else:
        p /= s
    return p


def js_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """Jensen–Shannon divergence (base-e)."""
    P = _safe_prob_vec(p)
    Q = _safe_prob_vec(q)
    M = 0.5 * (P + Q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = (a > 0) & (b > 0)
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * _kl(P, M) + 0.5 * _kl(Q, M)


def regime_distribution(df: pd.DataFrame, *, use_expected: bool = True) -> dict[str, float]:
    """Return % minutes in each regime (expected or hard label)."""
    if df.empty:
        return {r: 0.0 for r in REGIMES}
    if use_expected and all(f"p_{r}_final" in df.columns for r in REGIMES):
        probs = df[[f"p_{r}_final" for r in REGIMES]].to_numpy(dtype=float)
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        mean = probs.mean(axis=0)
        mean = mean / max(mean.sum(), 1e-12)
        return {r: float(mean[i]) for i, r in enumerate(REGIMES)}

    lab = df.get("regime_label_final", df.get("regime_label", "")).astype(str)
    counts = lab.value_counts()
    total = float(len(lab))
    return {r: float(counts.get(r, 0) / max(total, 1.0)) for r in REGIMES}


def transition_matrix(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Empirical transition matrix on final labels (row-normalized)."""
    out = {a: {b: 0.0 for b in REGIMES} for a in REGIMES}
    if df.empty:
        return out
    lab = df.get("regime_label_final", df.get("regime_label", "")).astype(str)
    prev = lab.shift(1)
    mask = prev.notna()
    for a, b in zip(prev[mask], lab[mask]):
        if a in out and b in out[a]:
            out[a][b] += 1.0
    # Normalize rows.
    for a in REGIMES:
        s = sum(out[a].values())
        if s > 0:
            for b in REGIMES:
                out[a][b] /= s
    return out


def tail_intensity_stats(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty or "ret_1m" not in df.columns:
        return {"q95_abs_ret": None, "q99_abs_ret": None}
    x = pd.to_numeric(df["ret_1m"], errors="coerce").abs().dropna().to_numpy(dtype=float)
    if len(x) == 0:
        return {"q95_abs_ret": None, "q99_abs_ret": None}
    return {
        "q95_abs_ret": float(np.quantile(x, 0.95)),
        "q99_abs_ret": float(np.quantile(x, 0.99)),
    }


def metrics_by_group(trades: pd.DataFrame, group_col: str) -> dict[str, Any]:
    """Compute basic per-group trade metrics."""
    if trades.empty or group_col not in trades.columns:
        return {}
    out: dict[str, Any] = {}
    for key, g in trades.groupby(group_col):
        gg = g.copy()
        n = int(len(gg))
        pnl = pd.to_numeric(gg.get("pnl_net", 0.0), errors="coerce").fillna(0.0)
        ret = pd.to_numeric(gg.get("ret_net", 0.0), errors="coerce").fillna(0.0)
        hold = pd.to_numeric(gg.get("hold_minutes", 0.0), errors="coerce").fillna(0.0)
        fees = pd.to_numeric(gg.get("fees", 0.0), errors="coerce").fillna(0.0)
        slip = pd.to_numeric(gg.get("slippage_cost", 0.0), errors="coerce").fillna(0.0)
        spread = pd.to_numeric(gg.get("spread_cost", 0.0), errors="coerce").fillna(0.0)
        out[str(key)] = {
            "n_trades": n,
            "pnl_net_sum": float(pnl.sum()),
            "pnl_net_mean": float(pnl.mean() if n else 0.0),
            "ret_net_mean": float(ret.mean() if n else 0.0),
            "win_rate": float((pnl > 0).mean() if n else 0.0),
            "hold_minutes_mean": float(hold.mean() if n else 0.0),
            "fees_sum": float(fees.sum()),
            "slippage_sum": float(slip.sum()),
            "spread_sum": float(spread.sum()),
        }
    return out


def playbook_confusion_matrix(trades: pd.DataFrame) -> dict[str, Any]:
    """Confusion matrix: playbook_id vs regime_entry_label."""
    if trades.empty:
        return {"counts": {}, "row_norm": {}}
    if "playbook_id" not in trades.columns or "regime_entry_label" not in trades.columns:
        return {"counts": {}, "row_norm": {}}

    cm = pd.crosstab(
        trades["regime_entry_label"].astype(str),
        trades["playbook_id"].astype(str),
        dropna=False,
    )
    counts = {r: {c: int(cm.loc[r, c]) for c in cm.columns} for r in cm.index}
    row_norm = {}
    for r in cm.index:
        s = float(cm.loc[r].sum())
        row_norm[r] = {c: float(cm.loc[r, c] / s) if s > 0 else 0.0 for c in cm.columns}
    return {"counts": counts, "row_norm": row_norm}


def regime_robustness_score(trades: pd.DataFrame) -> dict[str, Any]:
    """A simple regime robustness score (RRS) based on per-regime trade EV.

    This is a first-pass implementation (meant to be audited/improved).
    """
    if trades.empty or "regime_entry_label" not in trades.columns:
        return {"rrs": 0.0, "by_regime": {}}

    by = metrics_by_group(trades, "regime_entry_label")
    # Compute EV per regime in "net return per trade" terms if available.
    ev = {}
    n = {}
    for r in REGIMES:
        g = trades[trades["regime_entry_label"].astype(str) == r]
        n[r] = int(len(g))
        if len(g) == 0:
            ev[r] = 0.0
        else:
            # Prefer return; fall back to pnl_net scaled by notional.
            if "ret_net" in g.columns:
                ev[r] = float(pd.to_numeric(g["ret_net"], errors="coerce").fillna(0.0).mean())
            else:
                pnl = pd.to_numeric(g.get("pnl_net", 0.0), errors="coerce").fillna(0.0)
                notional = pd.to_numeric(g.get("notional", 1.0), errors="coerce").replace(0.0, 1.0).fillna(1.0)
                ev[r] = float((pnl / notional).mean())

    # Weight regimes by sqrt(N) but cap dominance.
    w = np.array([min(np.sqrt(n[r]), 50.0) for r in REGIMES], dtype=float)
    w = w / max(w.sum(), 1e-12)

    ev_vec = np.array([ev[r] for r in REGIMES], dtype=float)
    # Reward positive EV across regimes; tanh squashes outliers.
    ev_score = float(np.dot(w, np.tanh(ev_vec / 0.002)))  # 0.2% trade return scale

    # Concentration penalty: if all PnL comes from one regime, penalize.
    pnl_by_reg = np.array([float(by.get(r, {}).get("pnl_net_sum", 0.0)) for r in REGIMES], dtype=float)
    pnl_abs = np.abs(pnl_by_reg)
    if pnl_abs.sum() <= 0:
        conc_pen = 0.0
    else:
        share = pnl_abs / pnl_abs.sum()
        hhi = float(np.sum(share * share))
        conc_pen = float(_clamp((hhi - 0.25) / (1.0 - 0.25), 0.0, 1.0))  # 0 at diversified, 1 at fully concentrated

    # Tail regime penalty: EX/LQ catastrophic loss gets amplified.
    tail_loss = min(0.0, ev.get("EX", 0.0)) + min(0.0, ev.get("LQ", 0.0))
    tail_pen = float(_clamp(abs(tail_loss) / 0.005, 0.0, 1.0))  # 0.5% loss threshold

    rrs = ev_score * (1.0 - 0.5 * conc_pen) * (1.0 - 0.7 * tail_pen)
    return {
        "rrs": float(rrs),
        "ev": {r: float(ev[r]) for r in REGIMES},
        "n": {r: int(n[r]) for r in REGIMES},
        "concentration_penalty": float(conc_pen),
        "tail_penalty": float(tail_pen),
    }


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, float(x))))

