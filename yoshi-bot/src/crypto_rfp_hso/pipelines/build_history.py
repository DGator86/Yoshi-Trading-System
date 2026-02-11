"""End-to-end historical build pipeline."""

from __future__ import annotations

import math

from crypto_rfp_hso.core.event_time import build_event_time_index
from crypto_rfp_hso.core.schemas import DEFAULT_CONFIG
from crypto_rfp_hso.dynamics.discrete import (
    compute_liquidation_hazard_index,
    compute_theta_event,
    expected_impulse,
    map_path_and_bands,
)
from crypto_rfp_hso.features.normalize import signature_to_vector
from crypto_rfp_hso.features.signature import compute_feature_signature
from crypto_rfp_hso.hilbert.project import hilbert_project
from crypto_rfp_hso.hilbert.templates_fit import fit_templates_from_labels, heuristic_class_label
from crypto_rfp_hso.overlay.payload import build_overlay_payload
from crypto_rfp_hso.projection.aggregator import compute_method_weights
from crypto_rfp_hso.projection.fan import build_forward_fan
from crypto_rfp_hso.projection.hazard_surface import hazard_surface_distribution
from crypto_rfp_hso.projection.per_state_models import fit_per_state_t_params
from crypto_rfp_hso.regime.node_posterior import compute_node_posterior
from crypto_rfp_hso.regime.order_scoring import score_orders
from crypto_rfp_hso.regime.validity_mask import VALID_MASK
from crypto_rfp_hso.transitions.semimarkov_fit import fit_semi_markov_params
from crypto_rfp_hso.transitions.semimarkov_propagate import propagate_semi_markov


def _merge_config(config: dict | None) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    return cfg


def _build_returns_from_buckets(buckets: list[dict]) -> list[float]:
    out = []
    for i in range(1, len(buckets)):
        c0 = max(float(buckets[i - 1].get("close", 0.0)), 1e-12)
        c1 = max(float(buckets[i].get("close", 0.0)), 1e-12)
        out.append(math.log(c1 / c0))
    return out


def _build_physics_diagnostics(signatures: dict[int, dict], config: dict) -> dict[int, dict]:
    """Build event-time diagnostics per event index."""
    if not signatures:
        return {}

    k_sorted = sorted(signatures.keys())
    sigma_vals = [abs(float(signatures[k].get("sigma_k", 0.0))) for k in k_sorted]
    sigma_base = sum(sigma_vals) / len(sigma_vals) if sigma_vals else 1e-6
    sigma_base = max(sigma_base, 1e-6)

    alpha_f = float(config.get("theta_alpha_f", 1.0))
    beta_l = float(config.get("theta_beta_l", 1.0))
    out: dict[int, dict] = {}
    for k in k_sorted:
        sig = signatures[k]
        liquidity = float(sig.get("depth_bid_10bps", 0.0)) + float(sig.get("depth_ask_10bps", 0.0))
        curvature = 1.0 / max(liquidity, 1e-12)
        liq_density = float(sig.get("liq_long_notional_k", 0.0)) + float(sig.get("liq_short_notional_k", 0.0))
        lhi = compute_liquidation_hazard_index(liq_density_local=liq_density, liquidity_local=liquidity)
        funding_force = float(sig.get("F_perp", 0.0))
        theta_base = (abs(float(sig.get("sigma_k", 0.0))) / sigma_base) ** 2
        theta = compute_theta_event(
            theta_base=theta_base,
            funding_force=funding_force,
            lhi=lhi,
            alpha_f=alpha_f,
            beta_l=beta_l,
        )
        out[k] = {
            "activity": float(sig.get("notional_k", 0.0)),
            "liquidity": float(liquidity),
            "curvature": float(curvature),
            "funding_force": float(funding_force),
            "lhi": float(lhi),
            "theta": float(theta),
        }
    return out


def build_history(
    buckets: list[dict],
    l2_snapshots: list[dict],
    perp_metrics: list[dict],
    coupling_inputs: dict | None = None,
    config: dict | None = None,
    templates: dict[str, list[float]] | None = None,
    valid_mask: dict[str, dict[str, int]] | None = None,
) -> dict:
    """Build full historical regime/posterior/projection payload."""
    cfg = _merge_config(config)
    cp_inputs = coupling_inputs or {}
    if not buckets:
        return {
            "signatures": {},
            "hilbert": {},
            "order": {},
            "node": {},
            "event_time": {"mode": cfg.get("tau_mode", "canonical"), "delta_tau": [], "tau": []},
            "forward_pi": {},
            "fan": [],
            "payload": {
                "k0": -1,
                "historical": [],
                "forward": {
                    "horizons": [],
                    "w_class_forward": [],
                    "fan": [],
                    "map_path": [],
                },
            },
        }

    signatures: dict[int, dict] = {}
    signature_list = []
    for k in range(len(buckets)):
        sig = compute_feature_signature(
            buckets=buckets,
            k=k,
            l2_snapshots=l2_snapshots,
            perp_metrics=perp_metrics,
            coupling_inputs=cp_inputs,
            config=cfg,
        )
        signatures[k] = sig
        signature_list.append(sig)

    # Ensure activity is explicit in signatures for downstream diagnostics.
    for k in range(len(buckets)):
        signatures[k]["notional_k"] = float(buckets[k].get("notional", 0.0))
        signatures[k]["n_trades_k"] = float(buckets[k].get("n_trades", 0.0))

    event_time = build_event_time_index(
        buckets=buckets,
        signatures=signatures,
        mode=str(cfg.get("tau_mode", "canonical")),
        config=cfg,
    )
    physics_diag = _build_physics_diagnostics(signatures=signatures, config=cfg)

    if templates is None:
        labels = [heuristic_class_label(sig) for sig in signature_list]
        templates = fit_templates_from_labels(signature_list, labels)

    hilbert_hist: dict[int, dict] = {}
    order_hist: dict[int, dict] = {}
    node_hist: dict[int, dict] = {}

    prev_dom_node = None
    prev_age = 0
    vmask = valid_mask or VALID_MASK
    for k, sig in signatures.items():
        vec = signature_to_vector(sig, keys=cfg.get("hilbert_feature_keys"))
        w_class, dom_class, margin, ent = hilbert_project(
            sig_vec=vec,
            templates=templates,
            temperature=float(cfg.get("regime_temperature", 6.0)),
        )
        w_order = score_orders(sig=sig, temperature=float(cfg.get("order_temperature", 5.0)))
        w_node, dom_node, age = compute_node_posterior(
            w_class=w_class,
            w_order=w_order,
            valid_mask=vmask,
            prev_dom_node=prev_dom_node,
            prev_age=prev_age,
        )
        prev_dom_node = dom_node
        prev_age = age

        hilbert_hist[k] = {
            "w_class": w_class,
            "dominant_class": dom_class,
            "top2_margin": margin,
            "entropy_norm": ent,
        }
        order_hist[k] = {
            "w_order": w_order,
            "dominant_order": max(w_order.items(), key=lambda kv: kv[1])[0],
        }
        node_hist[k] = {
            "w_node": w_node,
            "dom_node": dom_node,
            "age": age,
        }

    dom_nodes = [node_hist[k]["dom_node"] for k in range(len(buckets))]
    all_states = list(node_hist[len(buckets) - 1]["w_node"].keys())
    alpha_by_state, t_exit = fit_semi_markov_params(
        dom_nodes=dom_nodes,
        all_states=all_states,
        tau_min=int(cfg.get("tau_min_duration", 5)),
    )

    returns = _build_returns_from_buckets(buckets)
    returns_by_state: dict[str, list[float]] = {s: [] for s in all_states}
    for i in range(1, len(buckets)):
        st = node_hist[i]["dom_node"]
        returns_by_state.setdefault(st, []).append(returns[i - 1])
    state_params = fit_per_state_t_params(returns_by_state=returns_by_state)

    k0 = len(buckets) - 1
    pi0 = node_hist[k0]["w_node"]
    dom_state = node_hist[k0]["dom_node"]
    age0 = int(node_hist[k0]["age"])
    horizons = [int(h) for h in cfg.get("forward_horizons", [1, 2, 4, 8, 16, 32])]
    forward_pi = propagate_semi_markov(
        pi0=pi0,
        dom_state=dom_state,
        age0=age0,
        alpha_by_state=alpha_by_state,
        T_exit=t_exit,
        horizons=horizons,
    )

    h_now = float(signatures[k0].get("H_k", 0.5))
    dom_class_now = str(hilbert_hist[k0].get("dominant_class"))
    h_vals = [float(signatures[i].get("H_k", 0.5)) for i in range(len(buckets)) if hilbert_hist[i]["dominant_class"] == dom_class_now]
    h_class = sum(h_vals) / len(h_vals) if h_vals else h_now
    h_eff = 0.5 * h_now + 0.5 * h_class

    x0 = math.log(max(float(buckets[k0].get("close", 0.0)), 1e-12))
    fan = build_forward_fan(
        x0=x0,
        forward_pi=forward_pi,
        state_params=state_params,
        horizons=horizons,
        quantiles=list(cfg.get("fan_quantiles", [0.05, 0.50, 0.95])),
        H_eff=h_eff,
        space="log",
    )

    # Build event-time MAP path + sigma bands.
    max_h = max(horizons) if horizons else 1
    dt_ref = (
        float(event_time.get("delta_tau", [1.0])[-1])
        if event_time.get("delta_tau")
        else 1.0
    )
    dt_seq = [dt_ref for _ in range(max_h)]

    sig0 = signatures[k0]
    diag0 = physics_diag.get(k0, {})
    funding_force_now = float(diag0.get("funding_force", 0.0))
    arb_force_now = float(sig0.get("coupling", 0.0)) * float(sig0.get("beta_to_btc", 0.0)) * float(sig0.get("mu_k", 0.0))
    liq_bias = float(sig0.get("liq_bias", 0.0))
    sigma0 = max(abs(float(sig0.get("sigma_k", 0.0))), 1e-6)
    lhi_now = float(diag0.get("lhi", 0.0))
    liq_prob = float(1.0 - math.exp(-max(lhi_now, 0.0)))
    impulse_if_liq = liq_bias * 0.5 * sigma0
    exp_impulse_now = expected_impulse(liq_probability=liq_prob, impulse_given_liq=impulse_if_liq)
    theta_now = float(diag0.get("theta", 1.0))

    funding_seq = [funding_force_now for _ in range(max_h)]
    arb_seq = [arb_force_now for _ in range(max_h)]
    impulse_seq = [exp_impulse_now for _ in range(max_h)]
    theta_seq = [theta_now for _ in range(max_h)]

    x0_log = x0
    map_path_full = map_path_and_bands(
        x0_log=x0_log,
        delta_tau_seq=dt_seq,
        funding_force_seq=funding_seq,
        arb_force_seq=arb_seq,
        expected_impulse_seq=impulse_seq,
        sigma0=sigma0,
        theta_seq=theta_seq,
        tau0=float(event_time.get("tau", [0.0])[k0]) if event_time.get("tau") else 0.0,
    )
    map_by_tau = {int(r["tau"]): r for r in map_path_full}
    for row in fan:
        tau = int(row.get("tau", 0))
        if tau in map_by_tau:
            row.update(
                {
                    "p_map": float(map_by_tau[tau]["p_map"]),
                    "sigma1_low": float(map_by_tau[tau]["sigma1_low"]),
                    "sigma1_high": float(map_by_tau[tau]["sigma1_high"]),
                    "sigma2_low": float(map_by_tau[tau]["sigma2_low"]),
                    "sigma2_high": float(map_by_tau[tau]["sigma2_high"]),
                    "theta": float(map_by_tau[tau]["theta"]),
                    "lhi": float(lhi_now),
                    "tau_event": float(map_by_tau[tau]["tau_event"]),
                }
            )

    # Next-price distribution from hazard surface around first MAP step.
    if map_path_full:
        first = map_path_full[0]
        sigma_price = max(float(first["p_map"]) * max(float(first["sigma_log"]), 1e-8), 1e-8)
        hazard_next = hazard_surface_distribution(
            p_map=float(first["p_map"]),
            sigma_price=sigma_price,
            liquidity_mid=max(float(diag0.get("liquidity", 0.0)), 1e-8),
            curvature=float(diag0.get("curvature", 0.0)),
            lambda_b=1.0 + max(liq_bias, 0.0),
            lambda_s=1.0 + max(-liq_bias, 0.0),
            grid_size=int(cfg.get("hazard_grid_size", 121)),
            band_mult=float(cfg.get("hazard_band_mult", 3.0)),
        )
    else:
        hazard_next = {"grid_prices": [], "probabilities": [], "hazard": [], "p_star": float(buckets[k0].get("close", 0.0))}

    method_weights = compute_method_weights(
        w_class=hilbert_hist[k0]["w_class"],
        w_order=order_hist[k0]["w_order"],
        reliability=cfg.get("method_reliability", None),
    )

    # Annotate historical buckets with explicit event-time coordinates and diagnostics.
    buckets_aug = []
    for k, b in enumerate(buckets):
        b2 = dict(b)
        b2["tau"] = float(event_time.get("tau", [])[k]) if k < len(event_time.get("tau", [])) else float(k + 1)
        b2["delta_tau"] = float(event_time.get("delta_tau", [])[k]) if k < len(event_time.get("delta_tau", [])) else 1.0
        diag = physics_diag.get(k, {})
        b2["activity"] = float(diag.get("activity", 0.0))
        b2["curvature"] = float(diag.get("curvature", 0.0))
        b2["funding_force"] = float(diag.get("funding_force", 0.0))
        b2["lhi"] = float(diag.get("lhi", 0.0))
        b2["theta"] = float(diag.get("theta", 1.0))
        buckets_aug.append(b2)

    payload = build_overlay_payload(
        k0=k0,
        buckets=buckets_aug,
        hilbert=hilbert_hist,
        order=order_hist,
        node=node_hist,
        forward_pi=forward_pi,
        fan=fan,
        config=cfg,
    )
    payload["forward"]["map_path"] = [dict(r) for r in map_path_full if int(r.get("tau", 0)) in set(horizons)]
    payload["forward"]["next_price_distribution"] = hazard_next
    payload["forward"]["event_time_mode"] = event_time.get("mode", cfg.get("tau_mode", "canonical"))
    payload["forward"]["event_time_delta_tau_ref"] = float(dt_ref)

    return {
        "config": cfg,
        "templates": templates,
        "signatures": signatures,
        "hilbert": hilbert_hist,
        "order": order_hist,
        "node": node_hist,
        "event_time": event_time,
        "physics_diagnostics": physics_diag,
        "alpha_by_state": alpha_by_state,
        "T_exit": t_exit,
        "state_params": state_params,
        "forward_pi": forward_pi,
        "fan": fan,
        "method_weights": method_weights,
        "payload": payload,
    }
