"""Tests for crypto_rfp_hso end-to-end behavior."""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crypto_rfp_hso import (  # noqa: E402
    bucketize_prints_notional,
    build_forward_fan,
    build_overlay_payload,
    compute_feature_signature,
    compute_node_posterior,
    fit_per_state_t_params,
    fit_semi_markov_params,
    hilbert_project,
    hurst_ghe,
    propagate_semi_markov,
    score_orders,
)
from crypto_rfp_hso.features.normalize import signature_to_vector  # noqa: E402
from crypto_rfp_hso.hilbert.templates_fit import (  # noqa: E402
    fit_templates_from_labels,
    heuristic_class_label,
)
from crypto_rfp_hso.pipelines.build_history import build_history  # noqa: E402
from crypto_rfp_hso.projection.aggregator import G, ORDER_METHOD_MULT  # noqa: E402
from crypto_rfp_hso.regime.validity_mask import VALID_MASK  # noqa: E402


def _make_prints(n: int = 800) -> list[dict]:
    rng = np.random.default_rng(42)
    prints = []
    px = 50_000.0
    for i in range(n):
        ret = rng.normal(0.0, 0.0008)
        px *= math.exp(ret)
        size = float(rng.uniform(0.01, 0.8))
        side = "BUY" if rng.uniform() > 0.5 else "SELL"
        prints.append(
            {
                "timestamp": i,
                "symbol": "BTCUSDT",
                "price": float(px),
                "size": size,
                "side": side,
            }
        )
    return prints


def _make_l2_for_buckets(buckets: list[dict]) -> list[dict]:
    out = []
    for b in buckets:
        mid = float(b["close"])
        bids = []
        asks = []
        for i in range(1, 21):
            p_bid = mid * (1.0 - i * 0.0001)
            p_ask = mid * (1.0 + i * 0.0001)
            qty = 20.0 / (i + 1.0)
            bids.append([p_bid, qty])
            asks.append([p_ask, qty * 0.95])
        out.append(
            {
                "timestamp": b["timestamp_end"],
                "symbol": b.get("symbol", "BTCUSDT"),
                "bids": bids,
                "asks": asks,
            }
        )
    return out


def _make_perp_for_buckets(buckets: list[dict]) -> list[dict]:
    out = []
    oi = 1_000_000_000.0
    funding = 0.00001
    for i, b in enumerate(buckets):
        oi *= 1.0 + (0.0005 if i % 3 else -0.0002)
        funding += 0.000001 * ((i % 5) - 2)
        out.append(
            {
                "timestamp": b["timestamp_end"],
                "symbol": b.get("symbol", "BTCUSDT"),
                "oi": oi,
                "funding": funding,
                "liq_long_notional": 25_000.0 + i * 10.0,
                "liq_short_notional": 23_000.0 + i * 8.0,
            }
        )
    return out


def _coupling_inputs_from_buckets(buckets: list[dict]) -> dict:
    returns = []
    for i in range(1, len(buckets)):
        c0 = float(buckets[i - 1]["close"])
        c1 = float(buckets[i]["close"])
        returns.append(math.log(c1 / c0))
    # Use same synthetic BTC series for self-coupling in tests.
    return {
        "btc_returns": returns,
        "asset_returns": {"BTCUSDT": returns},
        "symbol": "BTCUSDT",
    }


def test_strict_signatures_and_probabilities():
    prints = _make_prints()
    buckets = bucketize_prints_notional(prints, notional_usd=2_000_000.0)
    assert len(buckets) > 5

    l2 = _make_l2_for_buckets(buckets)
    perp = _make_perp_for_buckets(buckets)
    coupling_inputs = _coupling_inputs_from_buckets(buckets)
    k = len(buckets) - 1

    sig = compute_feature_signature(
        buckets=buckets,
        k=k,
        l2_snapshots=l2,
        perp_metrics=perp,
        coupling_inputs=coupling_inputs,
        config={},
    )
    assert "mu_k" in sig and "sigma_k" in sig and "H_k" in sig
    assert 0.2 <= sig["H_k"] <= 0.8

    # Template fit + Hilbert projection
    signatures = [
        compute_feature_signature(
            buckets=buckets,
            k=i,
            l2_snapshots=l2,
            perp_metrics=perp,
            coupling_inputs=coupling_inputs,
            config={},
        )
        for i in range(len(buckets))
    ]
    labels = [heuristic_class_label(s) for s in signatures]
    templates = fit_templates_from_labels(signatures, labels)
    vec = signature_to_vector(sig)
    w_class, dominant_class, top2_margin, entropy_norm = hilbert_project(vec, templates, temperature=6.0)
    assert dominant_class in w_class
    assert 0.0 <= top2_margin <= 1.0
    assert 0.0 <= entropy_norm <= 1.0
    assert abs(sum(w_class.values()) - 1.0) < 1e-8

    w_order = score_orders(sig, temperature=5.0)
    assert abs(sum(w_order.values()) - 1.0) < 1e-8
    assert "Liquidity-Contained" in w_order

    w_node, dom_node, age = compute_node_posterior(
        w_class=w_class,
        w_order=w_order,
        valid_mask=VALID_MASK,
        prev_dom_node=None,
        prev_age=0,
    )
    assert dom_node in w_node
    assert age == 1
    assert abs(sum(w_node.values()) - 1.0) < 1e-8
    assert w_node["Shock|Liquidity-Contained"] == 0.0


def test_semimarkov_fan_payload_and_end_to_end_pipeline():
    prints = _make_prints()
    buckets = bucketize_prints_notional(prints, notional_usd=2_000_000.0)
    l2 = _make_l2_for_buckets(buckets)
    perp = _make_perp_for_buckets(buckets)
    coupling_inputs = _coupling_inputs_from_buckets(buckets)

    built = build_history(
        buckets=buckets,
        l2_snapshots=l2,
        perp_metrics=perp,
        coupling_inputs=coupling_inputs,
        config={},
    )

    node_hist = built["node"]
    dom_nodes = [node_hist[i]["dom_node"] for i in range(len(buckets))]
    all_states = list(node_hist[len(buckets) - 1]["w_node"].keys())
    alpha_by_state, t_exit = fit_semi_markov_params(dom_nodes, all_states, tau_min=5)
    assert set(alpha_by_state.keys()) == set(all_states)
    assert set(t_exit.keys()) == set(all_states)

    pi0 = node_hist[len(buckets) - 1]["w_node"]
    dom_state = node_hist[len(buckets) - 1]["dom_node"]
    age0 = node_hist[len(buckets) - 1]["age"]
    horizons = [1, 2, 4, 8]
    forward_pi = propagate_semi_markov(pi0, dom_state, age0, alpha_by_state, t_exit, horizons=horizons)
    assert set(forward_pi.keys()) == set(horizons)
    for tau, dist in forward_pi.items():
        assert abs(sum(dist.values()) - 1.0) < 1e-8
        assert tau in horizons

    # Fit conditional returns and forward fan
    returns_by_state = {}
    for i in range(1, len(buckets)):
        state = node_hist[i]["dom_node"]
        r = math.log(float(buckets[i]["close"]) / float(buckets[i - 1]["close"]))
        returns_by_state.setdefault(state, []).append(r)
    state_params = fit_per_state_t_params(returns_by_state)
    x0 = math.log(float(buckets[-1]["close"]))
    fan = build_forward_fan(
        x0=x0,
        forward_pi=forward_pi,
        state_params=state_params,
        horizons=horizons,
        quantiles=[0.05, 0.50, 0.95],
        H_eff=0.5,
        space="log",
    )
    assert len(fan) == len(horizons)
    for row in fan:
        assert row["q05"] <= row["q50"] <= row["q95"]
        assert row["q05"] > 0.0

    payload = build_overlay_payload(
        k0=len(buckets) - 1,
        buckets=buckets,
        hilbert=built["hilbert"],
        order=built["order"],
        node=built["node"],
        forward_pi=forward_pi,
        fan=fan,
        config={"base_alpha": 0.7},
    )
    assert payload["k0"] == len(buckets) - 1
    assert len(payload["historical"]) == len(buckets)
    assert payload["forward"]["fan"]

    # Ensure requested gating constants are present.
    assert G["Shock"]["quantile_coverage"] > G["Shock"]["analytic_local"]
    assert ORDER_METHOD_MULT["Correlation-Driven"]["quantile_coverage"] > 1.0


def test_hurst_signature_stability():
    rng = np.random.default_rng(123)
    increments = rng.normal(0.0, 1.0, 5000)
    log_prices = np.cumsum(increments).tolist()
    h = hurst_ghe(
        log_prices=log_prices,
        horizons=[1, 2, 4, 8, 16, 32],
        window=1500,
        q=2.0,
    )
    assert 0.2 <= h <= 0.8
