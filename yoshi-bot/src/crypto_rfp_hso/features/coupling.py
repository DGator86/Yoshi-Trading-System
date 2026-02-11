"""Cross-asset coupling features."""

from __future__ import annotations

import numpy as np

from crypto_rfp_hso.core.math import safe_div, sigmoid, zscore_current


def _extract_asset_returns(coupling_inputs: dict, symbol: str) -> list[float]:
    asset_returns = coupling_inputs.get("asset_returns")
    if isinstance(asset_returns, list):
        return [float(x) for x in asset_returns]
    if isinstance(asset_returns, dict):
        if symbol in asset_returns and isinstance(asset_returns[symbol], list):
            return [float(x) for x in asset_returns[symbol]]
        for v in asset_returns.values():
            if isinstance(v, list):
                return [float(x) for x in v]
    return []


def _rolling_beta_corr(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if a.size < 3 or b.size < 3:
        return 0.0, 0.0
    var_b = float(np.var(b))
    cov_ab = float(np.cov(a, b)[0, 1]) if b.size > 1 else 0.0
    beta = safe_div(cov_ab, var_b)
    corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else 0.0
    if not np.isfinite(corr):
        corr = 0.0
    return beta, corr


def compute_coupling_features(
    coupling_inputs: dict,
    symbol: str,
    k: int,
    config: dict,
) -> dict:
    """Compute beta/correlation-to-BTC coupling proxy."""
    btc_returns = [float(x) for x in coupling_inputs.get("btc_returns", [])]
    asset_returns = _extract_asset_returns(coupling_inputs, symbol)

    if not asset_returns:
        return {
            "beta_to_btc": 0.0,
            "corr_to_btc": 0.0,
            "coupling": 0.0,
        }

    if not btc_returns and symbol.upper().startswith("BTC"):
        return {
            "beta_to_btc": 1.0,
            "corr_to_btc": 1.0,
            "coupling": 0.5,
        }

    n = min(len(asset_returns), len(btc_returns)) if btc_returns else 0
    if n < 5:
        return {
            "beta_to_btc": 0.0,
            "corr_to_btc": 0.0,
            "coupling": 0.0,
        }

    end = min(k + 1, n)
    window = int(config.get("coupling_window", 128))
    start = max(0, end - window)
    a = np.asarray(asset_returns[start:end], dtype=float)
    b = np.asarray(btc_returns[start:end], dtype=float)
    beta, corr = _rolling_beta_corr(a, b)

    product_now = abs(beta) * max(corr, 0.0)

    hist = []
    min_len = min(len(asset_returns), len(btc_returns))
    for idx in range(max(5, window), min_len + 1):
        aa = np.asarray(asset_returns[idx - window : idx], dtype=float)
        bb = np.asarray(btc_returns[idx - window : idx], dtype=float)
        bbeta, ccorr = _rolling_beta_corr(aa, bb)
        hist.append(abs(bbeta) * max(ccorr, 0.0))

    z = zscore_current(hist or [0.0], product_now)
    coupling = sigmoid(z)

    return {
        "beta_to_btc": float(beta),
        "corr_to_btc": float(corr),
        "coupling": float(coupling),
    }
