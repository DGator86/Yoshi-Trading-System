from __future__ import annotations
import numpy as np
import pandas as pd

def interval_score(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    width = hi - lo
    below = (lo - y) * (y < lo)
    above = (y - hi) * (y > hi)
    penalty = (2.0 / alpha) * (below + above)
    return width + penalty

def score_predictions(df: pd.DataFrame, y_col: str = "future_return") -> dict:
    need = {y_col, "q05", "q50", "q95"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        return {"wis": None, "is90": None, "mae": None, "missing": missing}

    d = df[df[y_col].notna()].copy()
    if len(d) == 0:
        return {"wis": None, "is90": None, "mae": None, "missing": []}

    y = d[y_col].to_numpy()
    q05 = d["q05"].to_numpy()
    q50 = d["q50"].to_numpy()
    q95 = d["q95"].to_numpy()

    is90 = interval_score(y, q05, q95, alpha=0.10)
    mae = np.abs(y - q50)

    wis = np.mean(0.5 * is90 + 0.5 * mae)

    return {
        "is90": float(np.mean(is90)),
        "wis": float(wis),
        "mae": float(np.mean(mae)),
        "n": int(len(d)),
    }
