"""Perp positioning and forced-flow feature extraction."""

from __future__ import annotations

from crypto_rfp_hso.core.math import safe_div, sigmoid, zscore_current


def _metric_at(perp_metrics: list[dict], idx: int) -> dict:
    if not perp_metrics:
        return {}
    j = max(0, min(idx, len(perp_metrics) - 1))
    return perp_metrics[j]


def compute_perp_features(
    perp_metrics: list[dict],
    buckets: list[dict],
    k: int,
    config: dict,
) -> dict:
    """Compute OI/funding/liquidation features and U2_perp/F_perp."""
    if not perp_metrics:
        return {
            "oi_k": 0.0,
            "dOI_k": 0.0,
            "funding_k": 0.0,
            "dfunding_k": 0.0,
            "liq_long_notional_k": 0.0,
            "liq_short_notional_k": 0.0,
            "liq_intensity_k": 0.0,
            "liq_bias": 0.0,
            "U2_perp": 0.5,
            "F_perp": 0.0,
            "liq_intensity_z": 0.0,
        }

    cur = _metric_at(perp_metrics, k)
    prev = _metric_at(perp_metrics, max(0, k - 1))

    oi = float(cur.get("oi", cur.get("open_interest", 0.0)))
    oi_prev = float(prev.get("oi", prev.get("open_interest", 0.0)))
    funding = float(cur.get("funding", cur.get("funding_rate", 0.0)))
    funding_prev = float(prev.get("funding", prev.get("funding_rate", 0.0)))
    liq_long = float(cur.get("liq_long_notional", cur.get("long_liq", 0.0)))
    liq_short = float(cur.get("liq_short_notional", cur.get("short_liq", 0.0)))

    d_oi = safe_div(oi - oi_prev, oi_prev)
    d_funding = funding - funding_prev

    notional_bucket = float(buckets[k].get("notional", 0.0)) if buckets and 0 <= k < len(buckets) else 0.0
    liq_intensity = safe_div(liq_long + liq_short, notional_bucket)
    liq_bias = safe_div(liq_long - liq_short, liq_long + liq_short)

    upto = max(0, min(k, len(perp_metrics) - 1))
    funding_hist = []
    d_oi_hist = []
    liq_intensity_hist = []
    for idx in range(upto + 1):
        now = _metric_at(perp_metrics, idx)
        prv = _metric_at(perp_metrics, max(0, idx - 1))
        oi_now = float(now.get("oi", now.get("open_interest", 0.0)))
        oi_prv = float(prv.get("oi", prv.get("open_interest", 0.0)))
        f_now = float(now.get("funding", now.get("funding_rate", 0.0)))
        ll = float(now.get("liq_long_notional", now.get("long_liq", 0.0)))
        ls = float(now.get("liq_short_notional", now.get("short_liq", 0.0)))
        b_notional = float(buckets[idx].get("notional", 0.0)) if idx < len(buckets) else 0.0

        funding_hist.append(abs(f_now))
        d_oi_hist.append(abs(safe_div(oi_now - oi_prv, oi_prv)))
        liq_intensity_hist.append(abs(safe_div(ll + ls, b_notional)))

    z_funding = zscore_current(funding_hist, abs(funding))
    z_doi = zscore_current(d_oi_hist, abs(d_oi))
    z_liq_int = zscore_current(liq_intensity_hist, abs(liq_intensity))

    u2_perp = sigmoid(z_funding + z_doi + z_liq_int)
    sign_term = 1.0 if funding >= 0 else -1.0
    f_perp = sign_term * u2_perp + 0.25 * liq_bias * u2_perp

    return {
        "oi_k": float(oi),
        "dOI_k": float(d_oi),
        "funding_k": float(funding),
        "dfunding_k": float(d_funding),
        "liq_long_notional_k": float(liq_long),
        "liq_short_notional_k": float(liq_short),
        "liq_intensity_k": float(liq_intensity),
        "liq_bias": float(liq_bias),
        "U2_perp": float(u2_perp),
        "F_perp": float(f_perp),
        "liq_intensity_z": float(z_liq_int),
    }
