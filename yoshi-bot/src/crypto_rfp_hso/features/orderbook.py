"""L2 liquidity/microstructure feature extraction."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from crypto_rfp_hso.core.math import safe_div, sigmoid, zscore_current


def _parse_levels(levels: Iterable) -> list[tuple[float, float]]:
    parsed: list[tuple[float, float]] = []
    for lvl in levels or []:
        if isinstance(lvl, dict):
            p = float(lvl.get("price", lvl.get("px", 0.0)))
            s = float(lvl.get("size", lvl.get("qty", lvl.get("sz", 0.0))))
        else:
            if not isinstance(lvl, (list, tuple)) or len(lvl) < 2:
                continue
            p = float(lvl[0])
            s = float(lvl[1])
        if p > 0.0 and s > 0.0:
            parsed.append((p, s))
    return parsed


def _depth_within_bps(levels: list[tuple[float, float]], mid: float, bps: float, side: str) -> float:
    if mid <= 0.0:
        return 0.0
    limit = float(bps)
    acc = 0.0
    for price, size in levels:
        dist_bps = abs(price - mid) / mid * 10_000.0
        if dist_bps <= limit:
            if side == "bid" and price <= mid:
                acc += price * size
            elif side == "ask" and price >= mid:
                acc += price * size
    return float(acc)


def _book_slope(levels: list[tuple[float, float]], mid: float) -> float:
    if mid <= 0.0 or len(levels) < 2:
        return 0.0
    dists = []
    cum_depth = []
    running = 0.0
    for p, s in levels:
        dist = abs(p - mid) / mid * 10_000.0
        running += max(p * s, 0.0)
        if dist > 0.0 and running > 0.0:
            dists.append(dist)
            cum_depth.append(running)
    if len(dists) < 2:
        return 0.0
    x = np.asarray(dists, dtype=float)
    y = np.log(np.asarray(cum_depth, dtype=float) + 1e-12)
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)


def _impact_for_notional(levels: list[tuple[float, float]], notional_usd: float, mid: float) -> float:
    if not levels or mid <= 0.0 or notional_usd <= 0.0:
        return 0.0
    remaining = float(notional_usd)
    total_notional = 0.0
    total_qty = 0.0
    for price, size in levels:
        level_notional = price * size
        take = min(remaining, level_notional)
        total_notional += take
        total_qty += take / price
        remaining -= take
        if remaining <= 0.0:
            break
    if total_qty <= 0.0:
        return 0.0
    avg_exec = total_notional / total_qty
    return abs(avg_exec - mid) / mid


def _snapshot_at(l2_snapshots: list[dict], idx: int) -> dict:
    if not l2_snapshots:
        return {}
    j = max(0, min(idx, len(l2_snapshots) - 1))
    return l2_snapshots[j]


def compute_orderbook_features(l2_snapshots: list[dict], k: int, config: dict) -> dict:
    """Compute spread/depth/imbalance/slope/impact + U2_liq/F_liq."""
    if not l2_snapshots:
        return {
            "mid_k": 0.0,
            "spread_k": 0.0,
            "depth_bid_5bps": 0.0,
            "depth_ask_5bps": 0.0,
            "depth_bid_10bps": 0.0,
            "depth_ask_10bps": 0.0,
            "depth_bid_25bps": 0.0,
            "depth_ask_25bps": 0.0,
            "imbalance_10bps": 0.0,
            "book_slope_bid": 0.0,
            "book_slope_ask": 0.0,
            "impact_proxy": 0.0,
            "U2_liq": 0.5,
            "F_liq": 0.0,
            "spread_z": 0.0,
            "impact_z": 0.0,
            "depth_sum_z": 0.0,
        }

    snap = _snapshot_at(l2_snapshots, k)
    bids = sorted(_parse_levels(snap.get("bids", [])), key=lambda x: x[0], reverse=True)
    asks = sorted(_parse_levels(snap.get("asks", [])), key=lambda x: x[0])
    if not bids or not asks:
        return compute_orderbook_features([], k, config)

    bid1 = bids[0][0]
    ask1 = asks[0][0]
    mid = 0.5 * (bid1 + ask1)
    spread = safe_div(ask1 - bid1, mid)

    d_bid_5 = _depth_within_bps(bids, mid, 5.0, "bid")
    d_ask_5 = _depth_within_bps(asks, mid, 5.0, "ask")
    d_bid_10 = _depth_within_bps(bids, mid, 10.0, "bid")
    d_ask_10 = _depth_within_bps(asks, mid, 10.0, "ask")
    d_bid_25 = _depth_within_bps(bids, mid, 25.0, "bid")
    d_ask_25 = _depth_within_bps(asks, mid, 25.0, "ask")

    imbalance_10 = safe_div(d_bid_10 - d_ask_10, d_bid_10 + d_ask_10)
    slope_bid = _book_slope(bids, mid)
    slope_ask = _book_slope(asks, mid)

    impact_notional = float(config.get("impact_notional_usd", 100_000.0))
    impact_buy = _impact_for_notional(asks, impact_notional, mid)
    impact_sell = _impact_for_notional(list(reversed(bids)), impact_notional, mid)
    impact = 0.5 * (impact_buy + impact_sell)

    # Historical z-scores from snapshots up to k (index-aligned approximation).
    upto = max(0, min(k, len(l2_snapshots) - 1))
    spread_hist = []
    depth_hist = []
    impact_hist = []
    for idx in range(upto + 1):
        s = _snapshot_at(l2_snapshots, idx)
        b = sorted(_parse_levels(s.get("bids", [])), key=lambda x: x[0], reverse=True)
        a = sorted(_parse_levels(s.get("asks", [])), key=lambda x: x[0])
        if not b or not a:
            continue
        m = 0.5 * (b[0][0] + a[0][0])
        sp = safe_div(a[0][0] - b[0][0], m)
        db = _depth_within_bps(b, m, 10.0, "bid")
        da = _depth_within_bps(a, m, 10.0, "ask")
        im = 0.5 * (
            _impact_for_notional(a, impact_notional, m)
            + _impact_for_notional(list(reversed(b)), impact_notional, m)
        )
        spread_hist.append(sp)
        depth_hist.append(db + da)
        impact_hist.append(im)

    depth_sum = d_bid_10 + d_ask_10
    spread_z = zscore_current(spread_hist, spread)
    depth_z = zscore_current(depth_hist, depth_sum)
    impact_z = zscore_current(impact_hist, impact)

    stiffness = depth_z - spread_z
    u2_liq = sigmoid(stiffness)
    f_liq = imbalance_10 * u2_liq

    return {
        "mid_k": float(mid),
        "spread_k": float(spread),
        "depth_bid_5bps": float(d_bid_5),
        "depth_ask_5bps": float(d_ask_5),
        "depth_bid_10bps": float(d_bid_10),
        "depth_ask_10bps": float(d_ask_10),
        "depth_bid_25bps": float(d_bid_25),
        "depth_ask_25bps": float(d_ask_25),
        "imbalance_10bps": float(imbalance_10),
        "book_slope_bid": float(slope_bid),
        "book_slope_ask": float(slope_ask),
        "impact_proxy": float(impact),
        "U2_liq": float(u2_liq),
        "F_liq": float(f_liq),
        "spread_z": float(spread_z),
        "impact_z": float(impact_z),
        "depth_sum_z": float(depth_z),
    }
