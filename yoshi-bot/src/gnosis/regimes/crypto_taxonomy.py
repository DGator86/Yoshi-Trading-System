"""Crypto regime taxonomy + rule-softmax classifier (regime-first backbone).

Implements a deterministic, debuggable regime scorer that emits:
  - distance scores d_MR, d_TR, d_CP, d_EX, d_LQ
  - probabilities p_* via softmax(beta * d_*)
  - confidence = max(p_*)
  - liquidity overlay TD/NR/TH (probabilistic)
  - funding overlay FP+/F0/FP- (optional, probabilistic)
  - transition flags and TV-distance churn Δ for blending/standdown

Design goal:
  rules-first, model-second — stable under walk-forward and easy to audit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import timezone
from typing import Any, Dict, Iterable, Literal, Optional

import numpy as np
import pandas as pd

RegimeLabel = Literal["MR", "TR", "CP", "EX", "LQ"]
LiqOverlayLabel = Literal["TD", "NR", "TH"]
FundOverlayLabel = Literal["FP+", "F0", "FP-"]


def _eps() -> float:
    return 1e-12


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, float(x))))


def ramp01(x: float | np.ndarray, a: float, b: float) -> float | np.ndarray:
    """Ramp x into [0,1] with thresholds a<b."""
    if b <= a:
        raise ValueError("ramp01 requires b>a")
    return np.clip((np.asarray(x) - a) / (b - a), 0.0, 1.0)


def softmax_dict(scores: dict[str, float], beta: float) -> dict[str, float]:
    """Softmax over a dict of scalar scores."""
    keys = list(scores.keys())
    vals = np.array([float(scores[k]) for k in keys], dtype=float)
    if not np.isfinite(vals).all():
        vals = np.nan_to_num(vals, nan=0.0, posinf=10.0, neginf=-10.0)
    m = float(np.max(vals))
    exps = np.exp(beta * (vals - m))
    z = float(np.sum(exps))
    if z <= 0:
        # Fallback to uniform distribution.
        p = np.ones_like(vals) / max(len(vals), 1)
    else:
        p = exps / z
    return {k: float(v) for k, v in zip(keys, p)}


def softmax_matrix(scores: np.ndarray, beta: float) -> np.ndarray:
    """Row-wise softmax for scores shape (N, K)."""
    x = np.asarray(scores, dtype=float)
    if x.ndim != 2:
        raise ValueError("softmax_matrix expects 2D array")
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
    x = np.clip(x, -50.0, 50.0)
    m = np.max(x, axis=1, keepdims=True)
    exps = np.exp(beta * (x - m))
    z = np.sum(exps, axis=1, keepdims=True)
    z = np.where(z <= 0.0, 1.0, z)
    return exps / z


def total_variation_distance(p: dict[str, float], q: dict[str, float]) -> float:
    """0.5 * L1 distance between two discrete distributions."""
    keys = set(p.keys()) | set(q.keys())
    return 0.5 * float(sum(abs(float(p.get(k, 0.0)) - float(q.get(k, 0.0))) for k in keys))


def _ensure_utc(ts: pd.Series) -> pd.Series:
    t = pd.to_datetime(ts, utc=True, errors="coerce")
    # Always tz-aware UTC.
    if getattr(t.dt, "tz", None) is None:
        t = t.dt.tz_localize(timezone.utc)
    return t


def _rolling_median(x: pd.Series, window: int, min_periods: int) -> pd.Series:
    return x.rolling(int(window), min_periods=int(min_periods)).median()


def rolling_mad(x: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Rolling MAD: median(|x - median(x)|) with aligned rolling median."""
    med = _rolling_median(x, window=window, min_periods=min_periods)
    dev = (x - med).abs()
    return _rolling_median(dev, window=window, min_periods=min_periods)


def robust_zscore(x: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Robust z-score using rolling median and MAD."""
    med = _rolling_median(x, window=window, min_periods=min_periods)
    mad = rolling_mad(x, window=window, min_periods=min_periods)
    denom = (1.4826 * mad) + _eps()
    return (x - med) / denom


def efficiency_ratio(close: pd.Series, horizon_bars: int) -> pd.Series:
    """ER = net / path in [0,1]. horizon_bars >= 1."""
    h = max(int(horizon_bars), 1)
    net = (close - close.shift(h)).abs()
    path = close.diff().abs().rolling(h, min_periods=h).sum()
    return (net / (path + _eps())).clip(lower=0.0, upper=1.0)


@dataclass
class CryptoRegimeThresholds:
    # Liquidation cascade (LQ)
    lq_tr_shock_a: float = 2.5
    lq_tr_shock_b: float = 5.0
    lq_ret_shock_a: float = 2.5
    lq_ret_shock_b: float = 5.0
    lq_vol_imp_a: float = 2.0
    lq_vol_imp_b: float = 4.0
    lq_wick_a: float = 0.45
    lq_wick_b: float = 0.75
    lq_gap_a: float = 0.5
    lq_gap_b: float = 2.0
    lq_act_center: float = 0.60
    lq_distance_scale: float = 3.0

    # Expansion (EX)
    ex_tr_shock_a: float = 1.8
    ex_tr_shock_b: float = 3.5
    ex_vol_imp_a: float = 1.5
    ex_vol_imp_b: float = 3.0
    ex_post_comp_a: float = 1.0
    ex_post_comp_b: float = 1.6
    ex_act_center: float = 0.45
    ex_distance_scale: float = 2.5

    # Compression (CP)
    cp_comp_a: float = 1.3
    cp_comp_b: float = 2.2
    cp_tr_shock_a: float = 1.2
    cp_tr_shock_b: float = 2.0
    cp_er60_a: float = 0.25
    cp_er60_b: float = 0.55
    cp_persist_a: float = 0.35
    cp_persist_b: float = 0.75
    cp_act_center: float = 0.55
    cp_distance_scale: float = 2.0

    # Trend (TR)
    tr_er_a: float = 0.25
    tr_er_b: float = 0.60
    tr_persist_a: float = 0.35
    tr_persist_b: float = 0.80
    tr_clv_a: float = 0.20
    tr_clv_b: float = 0.85
    tr_wick_pen_a: float = 0.45
    tr_wick_pen_b: float = 0.80
    tr_act_center: float = 0.45
    tr_distance_scale: float = 2.2

    # Mean reversion (MR)
    mr_er60_a: float = 0.20
    mr_er60_b: float = 0.55
    mr_cross_a: float = 0.10
    mr_cross_b: float = 0.45
    mr_wick_a: float = 0.30
    mr_wick_b: float = 0.70
    mr_tr_shock_pen_a: float = 1.6
    mr_tr_shock_pen_b: float = 3.0
    mr_persist_pen_a: float = 0.35
    mr_persist_pen_b: float = 0.80
    mr_act_center: float = 0.40
    mr_distance_scale: float = 2.0

    # Liquidity overlay thresholds (robust z ramps)
    liq_th_z_gap_a: float = 1.0
    liq_th_z_gap_b: float = 3.0
    liq_th_z_tr_a: float = 1.0
    liq_th_z_tr_b: float = 3.0
    liq_th_z_vol_a: float = 0.5
    liq_th_z_vol_b: float = 2.5
    liq_td_z_gap_a: float = 0.5
    liq_td_z_gap_b: float = 2.0
    liq_td_z_tr_a: float = 0.5
    liq_td_z_tr_b: float = 2.0

    # Funding overlay thresholds (robust z)
    fund_extreme_a: float = 1.5
    fund_extreme_b: float = 3.5


@dataclass
class CryptoRegimeConfig:
    """Configuration for regime ledger construction."""

    # Rolling windows are expressed in minutes (bar units when timeframe=1m).
    n_short: int = 60
    n_long: int = 360
    n_persist: int = 30
    ema_cross_span: int = 30

    # ER horizons in minutes.
    er_short: int = 15
    er_long: int = 60

    # Softmax sharpness.
    beta_regime: float = 3.0
    beta_liquidity: float = 3.0
    beta_funding: float = 3.0

    thresholds: CryptoRegimeThresholds = field(default_factory=CryptoRegimeThresholds)


def compute_features_1m(bars_1m: pd.DataFrame, cfg: CryptoRegimeConfig) -> pd.DataFrame:
    """Compute OHLCV-derived features on 1m bars (per symbol)."""
    required = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
    missing = required - set(bars_1m.columns)
    if missing:
        raise ValueError(f"bars_1m missing required columns: {sorted(missing)}")

    df = bars_1m.copy()
    df["timestamp"] = _ensure_utc(df["timestamp"])
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    out = []
    for sym, sdf in df.groupby("symbol", sort=False):
        s = sdf.copy()
        s["prev_close"] = s["close"].shift(1)
        s["ret_1m"] = np.log((s["close"] + _eps()) / (s["prev_close"] + _eps()))

        # True Range (TR)
        hl = s["high"] - s["low"]
        hc = (s["high"] - s["prev_close"]).abs()
        lc = (s["low"] - s["prev_close"]).abs()
        s["tr_1m"] = np.nanmax(np.vstack([hl.to_numpy(), hc.to_numpy(), lc.to_numpy()]), axis=0)

        # Primitives
        s["range_1m"] = hl
        s["body_1m"] = (s["close"] - s["open"]).abs()
        s["gap_1m"] = (s["open"] - s["prev_close"]).abs()
        s["hlc3"] = (s["high"] + s["low"] + s["close"]) / 3.0

        # Wicks / CLV
        uw = s["high"] - np.maximum(s["open"], s["close"])
        lw = np.minimum(s["open"], s["close"]) - s["low"]
        s["wick_ratio"] = (uw + lw) / (s["range_1m"] + _eps())
        s["clv"] = ((s["close"] - s["low"]) - (s["high"] - s["close"])) / (s["range_1m"] + _eps())

        # Rolling stats (robust where possible)
        n_short = int(cfg.n_short)
        n_long = int(cfg.n_long)
        min_short = max(10, n_short // 4)
        min_long = max(20, n_long // 6)

        tr_med = _rolling_median(s["tr_1m"], window=n_short, min_periods=min_short)
        vol_med = _rolling_median(s["volume"], window=n_short, min_periods=min_short)

        s["tr_shock"] = s["tr_1m"] / (tr_med + _eps())
        s["shock_score"] = (s["ret_1m"].abs()) / ((1.4826 * rolling_mad(s["ret_1m"], n_short, min_short)) + _eps())
        s["volume_impulse"] = s["volume"] / (vol_med + _eps())
        s["gap_norm"] = s["gap_1m"] / (tr_med + _eps())

        # ATR / RV at "1m scale" (still per 1m bar)
        s["atr_1m"] = s["tr_1m"].ewm(span=n_short, adjust=False, min_periods=min_short).mean()
        s["rv_1m"] = s["ret_1m"].rolling(n_short, min_periods=min_short).std()

        # Compression
        atr_short = s["tr_1m"].ewm(span=n_short, adjust=False, min_periods=min_short).mean()
        atr_long = s["tr_1m"].ewm(span=n_long, adjust=False, min_periods=min_long).mean()
        s["atr_long"] = atr_long
        s["comp"] = atr_short / (atr_long + _eps())

        # ER
        s["er_15m"] = efficiency_ratio(s["close"], horizon_bars=int(cfg.er_short))
        s["er_60m"] = efficiency_ratio(s["close"], horizon_bars=int(cfg.er_long))

        # Persistence
        direction = np.sign((s["close"] - s["prev_close"]).fillna(0.0)).astype(float)
        s["directional_persistence"] = direction.rolling(cfg.n_persist, min_periods=max(5, cfg.n_persist // 3)).mean().abs()

        # Chop proxy: sign changes of (C - EMA(C,30))
        ema = s["close"].ewm(span=int(cfg.ema_cross_span), adjust=False, min_periods=max(5, int(cfg.ema_cross_span) // 3)).mean()
        sgn = np.sign((s["close"] - ema).fillna(0.0))
        changed = (sgn != sgn.shift(1)) & (sgn != 0) & (sgn.shift(1) != 0)
        s["cross"] = changed.astype(float).rolling(cfg.n_persist, min_periods=max(5, cfg.n_persist // 3)).mean()

        # VWAP slope (15m rolling) + EMA slope (1h-ish)
        v = s["volume"].fillna(0.0)
        vwap_15 = (s["hlc3"] * v).rolling(15, min_periods=5).sum() / (v.rolling(15, min_periods=5).sum() + _eps())
        s["slope_vwap_15m"] = (vwap_15 - vwap_15.shift(15)) / (15.0 * (s["close"].shift(15).abs() + _eps()))

        ema_60 = s["close"].ewm(span=60, adjust=False, min_periods=20).mean()
        s["slope_ema_1h"] = (ema_60 - ema_60.shift(60)) / (60.0 * (s["close"].shift(60).abs() + _eps()))

        # Time structure
        s["dow"] = s["timestamp"].dt.dayofweek.astype(int)
        s["hour_utc"] = s["timestamp"].dt.hour.astype(int)
        s["weekend_flag"] = s["dow"].isin([5, 6])
        s["session_bucket"] = pd.cut(
            s["hour_utc"],
            bins=[-1, 6, 12, 18, 23],
            labels=["asia", "europe", "us", "us_asia_overlap"],
        ).astype(str)

        out.append(s)

    return pd.concat(out, ignore_index=True)


def classify_regimes(df_feat: pd.DataFrame, cfg: CryptoRegimeConfig) -> pd.DataFrame:
    """Attach d_*, p_* regime columns + label/confidence + transitions."""
    df = df_feat.copy()
    regimes: list[RegimeLabel] = ["MR", "TR", "CP", "EX", "LQ"]

    th = cfg.thresholds
    tr_shock = df["tr_shock"].astype(float).fillna(0.0).to_numpy()
    ret_shock = df["shock_score"].astype(float).fillna(0.0).to_numpy()
    vol_imp = df["volume_impulse"].astype(float).fillna(0.0).to_numpy()
    wick_ratio = df["wick_ratio"].astype(float).fillna(0.0).to_numpy()
    gap_norm = df["gap_norm"].astype(float).fillna(0.0).to_numpy()
    er_15 = df["er_15m"].astype(float).fillna(0.0).to_numpy()
    er_60 = df["er_60m"].astype(float).fillna(0.0).to_numpy()
    comp = df["comp"].astype(float).fillna(1.0).to_numpy()
    persist = df["directional_persistence"].astype(float).fillna(0.0).to_numpy()
    cross = df["cross"].astype(float).fillna(0.0).to_numpy()
    clv = df["clv"].astype(float).fillna(0.0).to_numpy()

    # LQ activation
    A_tr = ramp01(tr_shock, th.lq_tr_shock_a, th.lq_tr_shock_b)
    A_ret = ramp01(ret_shock, th.lq_ret_shock_a, th.lq_ret_shock_b)
    A_vol = ramp01(vol_imp, th.lq_vol_imp_a, th.lq_vol_imp_b)
    A_wick = ramp01(wick_ratio, th.lq_wick_a, th.lq_wick_b)
    A_gap = ramp01(gap_norm, th.lq_gap_a, th.lq_gap_b)
    LQ_act = 0.35 * A_tr + 0.25 * A_ret + 0.20 * A_vol + 0.10 * A_wick + 0.10 * A_gap
    pen_LQ = ramp01(LQ_act, 0.6, 0.85)

    # EX activation
    E_tr = ramp01(tr_shock, th.ex_tr_shock_a, th.ex_tr_shock_b)
    E_vol = ramp01(vol_imp, th.ex_vol_imp_a, th.ex_vol_imp_b)
    E_comp = ramp01(1.0 / (comp + _eps()), th.ex_post_comp_a, th.ex_post_comp_b)
    EX_act = 0.45 * E_tr + 0.35 * E_vol + 0.20 * E_comp - 0.50 * pen_LQ

    # CP activation
    C_comp = ramp01(1.0 / (comp + _eps()), th.cp_comp_a, th.cp_comp_b)
    C_tr = 1.0 - ramp01(tr_shock, th.cp_tr_shock_a, th.cp_tr_shock_b)
    C_er = 1.0 - ramp01(er_60, th.cp_er60_a, th.cp_er60_b)
    C_pers = 1.0 - ramp01(persist, th.cp_persist_a, th.cp_persist_b)
    CP_act = 0.45 * C_comp + 0.20 * C_tr + 0.20 * C_er + 0.15 * C_pers

    # TR activation
    T_er15 = ramp01(er_15, th.tr_er_a, th.tr_er_b)
    T_er60 = ramp01(er_60, th.tr_er_a, th.tr_er_b)
    T_pers = ramp01(persist, th.tr_persist_a, th.tr_persist_b)
    T_clv = ramp01(np.abs(clv), th.tr_clv_a, th.tr_clv_b)
    T_wpen = ramp01(wick_ratio, th.tr_wick_pen_a, th.tr_wick_pen_b)
    TR_act = 0.30 * T_er15 + 0.25 * T_er60 + 0.25 * T_pers + 0.20 * T_clv - 0.35 * T_wpen
    TR_act = TR_act - 0.60 * pen_LQ

    # MR activation
    M_er = 1.0 - ramp01(er_60, th.mr_er60_a, th.mr_er60_b)
    M_cross = ramp01(cross, th.mr_cross_a, th.mr_cross_b)
    M_wick = ramp01(wick_ratio, th.mr_wick_a, th.mr_wick_b)
    M_shpen = ramp01(tr_shock, th.mr_tr_shock_pen_a, th.mr_tr_shock_pen_b)
    M_ppen = ramp01(persist, th.mr_persist_pen_a, th.mr_persist_pen_b)
    MR_act = 0.35 * M_er + 0.30 * M_cross + 0.20 * M_wick - 0.25 * M_shpen - 0.25 * M_ppen
    MR_act = MR_act - 0.80 * pen_LQ - 0.35 * ramp01(EX_act, 0.55, 0.85)

    d_MR = th.mr_distance_scale * (MR_act - th.mr_act_center)
    d_TR = th.tr_distance_scale * (TR_act - th.tr_act_center)
    d_CP = th.cp_distance_scale * (CP_act - th.cp_act_center)
    d_EX = th.ex_distance_scale * (EX_act - th.ex_act_center)
    d_LQ = th.lq_distance_scale * (LQ_act - th.lq_act_center)

    df["d_MR"] = d_MR
    df["d_TR"] = d_TR
    df["d_CP"] = d_CP
    df["d_EX"] = d_EX
    df["d_LQ"] = d_LQ

    scores = np.vstack([d_MR, d_TR, d_CP, d_EX, d_LQ]).T
    probs = softmax_matrix(scores, beta=float(cfg.beta_regime))
    df["p_MR"] = probs[:, 0]
    df["p_TR"] = probs[:, 1]
    df["p_CP"] = probs[:, 2]
    df["p_EX"] = probs[:, 3]
    df["p_LQ"] = probs[:, 4]

    pmax = probs.max(axis=1)
    arg = probs.argmax(axis=1)
    df["conf"] = pmax
    df["regime_label"] = np.array(regimes, dtype=object)[arg]

    # Transition flags/types.
    prev = df["regime_label"].shift(1)
    df["transition_flag"] = (df["regime_label"] != prev) & prev.notna()
    df["transition_type"] = np.where(
        df["transition_flag"],
        prev.astype(str) + "->" + df["regime_label"].astype(str),
        "",
    )
    return df


def classify_liquidity_overlay(df_feat: pd.DataFrame, cfg: CryptoRegimeConfig) -> pd.DataFrame:
    """Attach liquidity overlay probabilities + label based on OHLCV proxies."""
    th = cfg.thresholds
    df = df_feat.copy()

    # Robust z-scores of stress proxies.
    n = int(cfg.n_short)
    minp = max(10, n // 4)
    z_gap = robust_zscore(df["gap_norm"].fillna(0.0), window=n, min_periods=minp)
    z_tr = robust_zscore(df["tr_shock"].fillna(1.0), window=n, min_periods=minp)
    z_vol = robust_zscore(df["volume_impulse"].fillna(1.0), window=n, min_periods=minp)

    TH_act = (
        0.45 * ramp01(z_gap, th.liq_th_z_gap_a, th.liq_th_z_gap_b)
        + 0.35 * ramp01(z_tr, th.liq_th_z_tr_a, th.liq_th_z_tr_b)
        + 0.20 * ramp01(z_vol, th.liq_th_z_vol_a, th.liq_th_z_vol_b)
    )
    TD_act = (
        0.50 * (1.0 - ramp01(z_gap, th.liq_td_z_gap_a, th.liq_td_z_gap_b))
        + 0.50 * (1.0 - ramp01(z_tr, th.liq_td_z_tr_a, th.liq_td_z_tr_b))
    )

    d_TD = 2.0 * (TD_act - 0.55)
    d_NR = np.zeros(len(df), dtype=float)
    d_TH = 2.0 * (TH_act - 0.45)

    scores = np.vstack([np.asarray(d_TD, dtype=float), d_NR, np.asarray(d_TH, dtype=float)]).T
    probs = softmax_matrix(scores, beta=float(cfg.beta_liquidity))
    df["p_TD"] = probs[:, 0]
    df["p_NR"] = probs[:, 1]
    df["p_TH"] = probs[:, 2]
    df["liq_overlay"] = np.array(["TD", "NR", "TH"], dtype=object)[probs.argmax(axis=1)]
    return df


def classify_funding_overlay(df_feat: pd.DataFrame, cfg: CryptoRegimeConfig, funding_col: str = "funding") -> pd.DataFrame:
    """Attach funding overlay probabilities + label if funding series exists."""
    df = df_feat.copy()
    if funding_col not in df.columns:
        df["p_FPplus"] = 0.0
        df["p_F0"] = 1.0
        df["p_FPminus"] = 0.0
        df["fund_overlay"] = "F0"
        return df

    th = cfg.thresholds
    n = int(cfg.n_long)
    minp = max(20, n // 6)
    fz = robust_zscore(df[funding_col].astype(float), window=n, min_periods=minp).fillna(0.0)

    FP_plus_act = ramp01(fz, th.fund_extreme_a, th.fund_extreme_b)
    FP_minus_act = ramp01(-fz, th.fund_extreme_a, th.fund_extreme_b)

    d_FPplus = 2.0 * (FP_plus_act - 0.5)
    d_FPminus = 2.0 * (FP_minus_act - 0.5)
    d_F0 = np.zeros(len(df), dtype=float)

    scores = np.vstack([np.asarray(d_FPplus, dtype=float), d_F0, np.asarray(d_FPminus, dtype=float)]).T
    probs = softmax_matrix(scores, beta=float(cfg.beta_funding))
    df["p_FPplus"] = probs[:, 0]
    df["p_F0"] = probs[:, 1]
    df["p_FPminus"] = probs[:, 2]
    df["fund_overlay"] = np.array(["FP+", "F0", "FP-"], dtype=object)[probs.argmax(axis=1)]
    return df


def build_regime_ledger_1m(
    bars_1m: pd.DataFrame,
    cfg: Optional[CryptoRegimeConfig] = None,
) -> pd.DataFrame:
    """Build a regime ledger on 1m bars.

    Output contains:
      - OHLCV + derived features
      - distances/probabilities for MR/TR/CP/EX/LQ (sum to 1)
      - conf, regime_label
      - liquidity overlay TD/NR/TH (probabilistic)
      - funding overlay (if funding col exists)
      - transition flag/type
    """
    cfg = cfg or CryptoRegimeConfig()
    feat = compute_features_1m(bars_1m, cfg)
    feat = classify_regimes(feat, cfg)
    feat = classify_liquidity_overlay(feat, cfg)
    feat = classify_funding_overlay(feat, cfg)

    # Sanity: probs sum to 1 (numerically).
    ps = feat[["p_MR", "p_TR", "p_CP", "p_EX", "p_LQ"]].sum(axis=1)
    feat["p_sum"] = ps
    return feat

