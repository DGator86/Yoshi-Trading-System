"""End-to-end historical build pipeline."""

from __future__ import annotations

import math

from crypto_rfp_hso.core.schemas import DEFAULT_CONFIG
from crypto_rfp_hso.features.normalize import signature_to_vector
from crypto_rfp_hso.features.signature import compute_feature_signature
from crypto_rfp_hso.hilbert.project import hilbert_project
from crypto_rfp_hso.hilbert.templates_fit import fit_templates_from_labels, heuristic_class_label
from crypto_rfp_hso.overlay.payload import build_overlay_payload
from crypto_rfp_hso.projection.aggregator import compute_method_weights
from crypto_rfp_hso.projection.fan import build_forward_fan
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
            "forward_pi": {},
            "fan": [],
            "payload": {"k0": -1, "historical": [], "forward": {"horizons": [], "w_class_forward": [], "fan": []}},
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

    method_weights = compute_method_weights(
        w_class=hilbert_hist[k0]["w_class"],
        w_order=order_hist[k0]["w_order"],
        reliability=cfg.get("method_reliability", None),
    )

    payload = build_overlay_payload(
        k0=k0,
        buckets=buckets,
        hilbert=hilbert_hist,
        order=order_hist,
        node=node_hist,
        forward_pi=forward_pi,
        fan=fan,
        config=cfg,
    )

    return {
        "config": cfg,
        "templates": templates,
        "signatures": signatures,
        "hilbert": hilbert_hist,
        "order": order_hist,
        "node": node_hist,
        "alpha_by_state": alpha_by_state,
        "T_exit": t_exit,
        "state_params": state_params,
        "forward_pi": forward_pi,
        "fan": fan,
        "method_weights": method_weights,
        "payload": payload,
    }
