"""Regime ledger builder (1m OHLCV backbone)."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from gnosis.regimes.crypto_taxonomy import (
    CryptoRegimeConfig,
    CryptoRegimeThresholds,
    add_multihorizon_regime_probs_from_1m,
    build_regime_ledger_1m,
)


def _get_nested(d: dict[str, Any], path: list[str], default: Any) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def build_regime_ledger(
    bars_1m: pd.DataFrame,
    cfg_raw: dict[str, Any],
) -> pd.DataFrame:
    """Build the required 1m regime ledger and p_final mixing columns."""
    regimes = cfg_raw.get("regimes", {}) or {}
    clf = regimes.get("classifier", {}) or {}

    beta = float(clf.get("beta", 3.0))
    thresholds_raw = _get_nested(clf, ["thresholds"], {})
    th = CryptoRegimeThresholds()
    # Map a subset of YAML thresholds to internal names (keep stable structure).
    # The caller may omit thresholds; defaults remain.
    # liquidation
    liq = thresholds_raw.get("liquidation", {}) if isinstance(thresholds_raw, dict) else {}
    if isinstance(liq, dict):
        th.lq_tr_shock_a = float(liq.get("tr_shock", th.lq_tr_shock_a))
        th.lq_vol_imp_a = float(liq.get("volume_impulse", th.lq_vol_imp_a))
        th.lq_ret_shock_a = float(liq.get("shock_score", th.lq_ret_shock_a))
        th.lq_wick_a = float(liq.get("wick_ratio", th.lq_wick_a))
    exp = thresholds_raw.get("expansion", {}) if isinstance(thresholds_raw, dict) else {}
    if isinstance(exp, dict):
        th.ex_tr_shock_a = float(exp.get("tr_shock", th.ex_tr_shock_a))
        th.ex_vol_imp_a = float(exp.get("volume_impulse", th.ex_vol_imp_a))

    cfg = CryptoRegimeConfig(beta_regime=beta, thresholds=th)

    ledger = build_regime_ledger_1m(bars_1m, cfg=cfg)

    blending = cfg_raw.get("blending", {}) or {}
    hm = blending.get("horizon_mix", {}) or {}
    lambdas = hm.get("lambdas", {"1m": 0.40, "15m": 0.30, "1h": 0.20, "4h": 0.10})
    dis = hm.get("disagreement_penalty", {"D0": 0.15, "D1": 0.35})
    trans = blending.get("transition", {}) or {}
    stab = trans.get("stabilizer", {"Delta0": 0.10, "Delta1": 0.30})

    ledger = add_multihorizon_regime_probs_from_1m(
        ledger,
        lambdas=dict(lambdas) if isinstance(lambdas, dict) else None,
        disagreement_penalty=dict(dis) if isinstance(dis, dict) else None,
        stabilizer=dict(stab) if isinstance(stab, dict) else None,
    )
    return ledger

