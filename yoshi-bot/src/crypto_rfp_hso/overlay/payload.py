"""Chart overlay payload construction."""

from __future__ import annotations

from crypto_rfp_hso.core.enums import CLASSES


def _class_from_node(state_key: str) -> str:
    if "|" not in state_key:
        return state_key
    return state_key.split("|", 1)[0]


def build_overlay_payload(
    k0: int,
    buckets: list[dict],
    hilbert: dict[int, dict],
    order: dict[int, dict],
    node: dict[int, dict],
    forward_pi: dict[int, dict[str, float]],
    fan: list[dict],
    config: dict,
) -> dict:
    """Return JSON payload for chart overlay."""
    historical = []
    n = len(buckets)
    for k in range(n):
        b = buckets[k]
        h = hilbert.get(k, {})
        o = order.get(k, {})
        nd = node.get(k, {})
        entropy_now = float(h.get("entropy_norm", 1.0))
        historical.append(
            {
                "k": k,
                "tau": b.get("tau", k + 1),
                "delta_tau": b.get("delta_tau", 1.0),
                "timestamp_start": b.get("timestamp_start"),
                "timestamp_end": b.get("timestamp_end"),
                "close": float(b.get("close", 0.0)),
                "activity": float(b.get("activity", 0.0)),
                "curvature": float(b.get("curvature", 0.0)),
                "funding_force": float(b.get("funding_force", 0.0)),
                "lhi": float(b.get("lhi", 0.0)),
                "theta": float(b.get("theta", 1.0)),
                "w_class": h.get("w_class", {}),
                "dominant_class": h.get("dominant_class"),
                "entropy_norm": entropy_now,
                "opacity": max(0.0, 1.0 - entropy_now),
                "w_order": o.get("w_order", {}),
                "dominant_order": o.get("dominant_order"),
                "w_node": nd.get("w_node", {}),
                "dominant_node": nd.get("dom_node"),
                "age": nd.get("age"),
            }
        )

    now_h = hilbert.get(k0, {})
    entropy_now = float(now_h.get("entropy_norm", 1.0))
    top2_margin = float(now_h.get("top2_margin", 1.0))
    base_alpha = float(config.get("base_alpha", 0.7))

    horizons = sorted(forward_pi.keys())
    regime_ribbon = []
    fan_with_alpha = []

    fan_by_tau = {int(row.get("tau")): row for row in fan}
    for tau in horizons:
        pi_state = forward_pi.get(tau, {})
        w_class_forward = {cls: 0.0 for cls in CLASSES}
        for state, p in pi_state.items():
            cls = _class_from_node(state)
            if cls in w_class_forward:
                w_class_forward[cls] += float(p)

        max_state_prob = max([float(v) for v in pi_state.values()] or [0.0])
        confidence = (1.0 - entropy_now) * top2_margin * max_state_prob
        alpha_tau = max(0.0, base_alpha * confidence)

        regime_ribbon.append(
            {
                "tau": int(tau),
                "w_class_forward": w_class_forward,
                "max_state_prob": float(max_state_prob),
                "alpha_tau": float(alpha_tau),
            }
        )

        row = dict(fan_by_tau.get(tau, {"tau": int(tau)}))
        row["alpha_tau"] = float(alpha_tau)
        fan_with_alpha.append(row)

    return {
        "k0": int(k0),
        "historical": historical,
        "forward": {
            "horizons": [int(h) for h in horizons],
            "w_class_forward": regime_ribbon,
            "fan": fan_with_alpha,
        },
    }
