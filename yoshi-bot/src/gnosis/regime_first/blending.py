"""Regime-first blending: learn W[r,k] and compute blended router action A."""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from gnosis.regimes.crypto_taxonomy import softmax_matrix


REGIMES = ["MR", "TR", "CP", "EX", "LQ"]
PLAYBOOKS = ["mr_fade", "trend_follow", "breakout_transition", "panic_only"]


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float(default)


def _get_playbook_a(df: pd.DataFrame) -> np.ndarray:
    cols = [
        "pb_mr_fade_a",
        "pb_trend_follow_a",
        "pb_breakout_transition_a",
        "pb_panic_only_a",
    ]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing playbook columns: {missing}")
    return df[cols].to_numpy(dtype=float)


def _get_p_final(df: pd.DataFrame) -> np.ndarray:
    cols = [f"p_{r}_final" for r in REGIMES]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing p_final columns: {missing}")
    return df[cols].to_numpy(dtype=float)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    s = np.sum(x, axis=1, keepdims=True)
    s = np.where(s <= 0.0, 1.0, s)
    return x / s


def _coerce_W(base_W: dict[str, Any]) -> np.ndarray:
    """Convert W dict to matrix shape (R,K)."""
    W = np.zeros((len(REGIMES), len(PLAYBOOKS)), dtype=float)
    for i, r in enumerate(REGIMES):
        row = base_W.get(r, {}) if isinstance(base_W, dict) else {}
        for j, k in enumerate(PLAYBOOKS):
            W[i, j] = max(_safe_float(row.get(k, 0.0), 0.0), 0.0)
        # If row sums to zero, allow a neutral default.
        if W[i].sum() <= 0.0:
            W[i, :] = 1.0 / len(PLAYBOOKS)
        else:
            W[i, :] = W[i, :] / W[i, :].sum()
    return W


def W_to_dict(W: np.ndarray) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for i, r in enumerate(REGIMES):
        out[r] = {k: float(W[i, j]) for j, k in enumerate(PLAYBOOKS)}
    return out


def learn_playbook_weights(
    train_df: pd.DataFrame,
    base_W: dict[str, Any],
    gamma: float = 500.0,
    min_samples_per_regime: int = 200,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    """Learn W[r,k] from train data via playbook EV proxy.

    EV proxy:
      ev[r,k] = mean( a_k(t) * ret_{t+1} ) on bars with regime=r

    We then softmax over playbooks per regime to get weights.
    """
    df = train_df.copy()
    if "symbol" in df.columns:
        df = df.sort_values(["symbol", "timestamp"])
        fwd = df.groupby("symbol")["ret_1m"].shift(-1)
    else:
        df = df.sort_values("timestamp")
        fwd = df["ret_1m"].shift(-1)

    df["ret_fwd_1m"] = pd.to_numeric(fwd, errors="coerce")
    a = _get_playbook_a(df)
    y = df["ret_fwd_1m"].fillna(0.0).to_numpy(dtype=float)

    base = _coerce_W(base_W)
    learned = base.copy()
    ev_mat = np.zeros_like(base)
    n_mat = np.zeros((len(REGIMES),), dtype=int)

    labels = df.get("regime_label_final", df.get("regime_label", "")).astype(str)

    for i, r in enumerate(REGIMES):
        mask = labels == r
        idx = np.where(mask.to_numpy())[0]
        n = int(len(idx))
        n_mat[i] = n
        if n < int(min_samples_per_regime):
            continue
        # ev across playbooks
        yy = y[idx]
        aa = a[idx, :]
        ev = np.mean(aa * yy.reshape(-1, 1), axis=0)
        ev = np.nan_to_num(ev, nan=0.0, posinf=0.0, neginf=0.0)
        ev_mat[i, :] = ev

        # Softmax over playbooks with temperature gamma.
        p = softmax_matrix(ev.reshape(1, -1), beta=float(gamma)).reshape(-1)

        # Blend with base weights with strength increasing with sample size.
        alpha = min(0.80, n / 10_000.0)
        learned[i, :] = (1.0 - alpha) * base[i, :] + alpha * p
        learned[i, :] = learned[i, :] / max(learned[i, :].sum(), 1e-12)

    diagnostics = {
        "gamma": float(gamma),
        "min_samples_per_regime": int(min_samples_per_regime),
        "n_samples": {REGIMES[i]: int(n_mat[i]) for i in range(len(REGIMES))},
        "ev": {REGIMES[i]: {PLAYBOOKS[j]: float(ev_mat[i, j]) for j in range(len(PLAYBOOKS))} for i in range(len(REGIMES))},
        "base_W": W_to_dict(base),
        "learned_W": W_to_dict(learned),
    }
    return W_to_dict(learned), diagnostics


def compute_router_action(
    df: pd.DataFrame,
    W: dict[str, dict[str, float]],
    risk_cfg: dict[str, Any],
) -> pd.DataFrame:
    """Compute blended router action score A and permission P (vectorized)."""
    out = df.copy()
    Wm = _coerce_W(W)
    a = _get_playbook_a(out)  # (N,K)
    p = _get_p_final(out)  # (N,R)

    # contrib[r] = sum_k W[r,k] * a_k
    contrib = a @ Wm.T  # (N,R)
    A_base = np.sum(contrib * p, axis=1)

    # Gates from ledger (if missing, treat as 1).
    if "G_stab" in out.columns:
        g_stab_s = pd.to_numeric(out["G_stab"], errors="coerce").fillna(1.0)
    else:
        g_stab_s = pd.Series(1.0, index=out.index)
    if "G_dis" in out.columns:
        g_dis_s = pd.to_numeric(out["G_dis"], errors="coerce").fillna(1.0)
    else:
        g_dis_s = pd.Series(1.0, index=out.index)
    g_stab = g_stab_s.to_numpy(dtype=float)
    g_dis = g_dis_s.to_numpy(dtype=float)
    A = A_base * g_stab * g_dis

    # Regime permission P_trade (blended).
    perm = (risk_cfg.get("permission") or {}) if isinstance(risk_cfg, dict) else {}
    P_trade = np.array([max(_safe_float(perm.get(r, 1.0), 1.0), 0.0) for r in REGIMES], dtype=float)
    P = np.sum(p * P_trade.reshape(1, -1), axis=1)

    # Determine top playbook by absolute contribution (for confusion matrix).
    # per-playbook contribution = sum_r p_r * W[r,k] * a_k
    # (N,K): a_k * sum_r p_r * W[r,k]
    weights_per_pb = p @ Wm  # (N,K)
    pb_contrib = a * weights_per_pb
    top_idx = np.argmax(np.abs(pb_contrib), axis=1)
    top_pb = np.array(PLAYBOOKS, dtype=object)[top_idx]

    out["router_score"] = A
    out["permission_P"] = P
    out["top_playbook"] = top_pb
    out["router_score_base"] = A_base
    return out

