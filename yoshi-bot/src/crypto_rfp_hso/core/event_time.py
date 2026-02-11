"""Event-time quantization utilities.

This module makes event-time construction explicit and mechanically bounded.
"""

from __future__ import annotations

import numpy as np

from crypto_rfp_hso.core.math import clamp

# Explicit event alphabet.
EVENT_ALPHABET = (
    "trade",
    "book",
    "funding_update",
    "liq_print",
    "onchain_pulse",
    "dex_swap",
    "mev_bundle",
)


def tau_weight(
    q_k: float,
    q0: float,
    beta: float,
    delta_l_k: float,
    l0: float,
    gamma: float,
    delta_f_k: float,
    f0: float,
    delta: float,
    w_min: float,
    w_max: float,
) -> float:
    """Compute clipped information-time increment.

    w_k = clip((q_k/q0)^beta + gamma*|delta_l_k/l0| + delta*|delta_f_k/f0|, w_min, w_max)
    """
    q_ref = max(float(q0), 1e-12)
    l_ref = max(float(l0), 1e-12)
    f_ref = max(float(f0), 1e-12)
    core = (max(float(q_k), 0.0) / q_ref) ** float(beta)
    core += float(gamma) * abs(float(delta_l_k) / l_ref)
    core += float(delta) * abs(float(delta_f_k) / f_ref)
    return clamp(core, float(w_min), float(w_max))


def build_event_time_index(
    buckets: list[dict],
    signatures: dict[int, dict],
    mode: str,
    config: dict,
) -> dict:
    """Build explicit event-time coordinates (tau, delta_tau) for each event index.

    Args:
        buckets: event-clock buckets
        signatures: per-k feature signatures
        mode: "canonical" or "information"
        config: runtime config
    """
    n = len(buckets)
    if n == 0:
        return {
            "mode": mode,
            "delta_tau": [],
            "tau": [],
            "q0": 1.0,
            "l0": 1.0,
            "f0": 1.0,
        }

    tau_mode = str(mode or "canonical").lower()
    if tau_mode not in {"canonical", "information"}:
        tau_mode = "canonical"

    notional = np.asarray([float(b.get("notional", 0.0)) for b in buckets], dtype=float)
    liquidity = np.asarray(
        [
            float(signatures.get(k, {}).get("depth_bid_10bps", 0.0))
            + float(signatures.get(k, {}).get("depth_ask_10bps", 0.0))
            for k in range(n)
        ],
        dtype=float,
    )
    funding = np.asarray([float(signatures.get(k, {}).get("funding_k", 0.0)) for k in range(n)], dtype=float)

    q0 = float(config.get("tau_q0", np.median(notional[notional > 0.0]) if np.any(notional > 0.0) else 1.0))
    l0 = float(config.get("tau_l0", np.median(liquidity[liquidity > 0.0]) if np.any(liquidity > 0.0) else 1.0))
    f_nonzero = np.abs(funding[np.abs(funding) > 0.0])
    f0 = float(config.get("tau_f0", np.median(f_nonzero) if f_nonzero.size > 0 else 1.0))

    beta = float(config.get("tau_beta", 0.6))
    gamma = float(config.get("tau_gamma", 0.4))
    delta = float(config.get("tau_delta", 0.4))
    w_min = float(config.get("tau_w_min", 0.25))
    w_max = float(config.get("tau_w_max", 4.0))

    delta_tau: list[float] = []
    if tau_mode == "canonical":
        delta_tau = [1.0 for _ in range(n)]
    else:
        for k in range(n):
            dl = liquidity[k] - liquidity[k - 1] if k > 0 else 0.0
            df = funding[k] - funding[k - 1] if k > 0 else 0.0
            wk = tau_weight(
                q_k=notional[k],
                q0=q0,
                beta=beta,
                delta_l_k=dl,
                l0=l0,
                gamma=gamma,
                delta_f_k=df,
                f0=f0,
                delta=delta,
                w_min=w_min,
                w_max=w_max,
            )
            delta_tau.append(float(max(wk, 1e-12)))

    tau: list[float] = []
    running = 0.0
    for dt in delta_tau:
        running += float(dt)
        tau.append(float(running))

    return {
        "mode": tau_mode,
        "delta_tau": delta_tau,
        "tau": tau,
        "q0": float(q0),
        "l0": float(l0),
        "f0": float(f0),
    }
