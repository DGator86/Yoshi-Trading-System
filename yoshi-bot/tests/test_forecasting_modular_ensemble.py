"""Tests for modular forecasting taxonomy + gating policy."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gnosis.forecasting.modular_ensemble import (  # noqa: E402
    DEFAULT_MODULE_ORDER,
    GatingInputs,
    build_default_module_registry,
    compute_module_weights,
    recommended_metrics_for_target,
)


def test_registry_has_all_12_modules():
    reg = build_default_module_registry()
    assert len(reg) == 12
    assert list(DEFAULT_MODULE_ORDER) == list(reg.keys())
    assert "orderflow_microstructure" in reg
    assert "derivatives_positioning" in reg
    assert "regime_state_machine" in reg


def test_cascade_and_lfi_high_reweights_toward_tail_models():
    base = {k: 1.0 for k in DEFAULT_MODULE_ORDER}
    weights, confidence = compute_module_weights(
        GatingInputs(
            regime_probs={"cascade_risk": 0.45, "range": 0.20, "trend_up": 0.10},
            spread_bps=16.0,
            depth_norm=0.20,
            lfi=1.8,
            jump_probability=0.35,
            event_window=False,
        ),
        base_weights=base,
    )
    assert abs(sum(weights.values()) - 1.0) < 1e-12
    assert weights["derivatives_positioning"] > weights["technical_price_action"]
    assert weights["scenario_monte_carlo"] > weights["technical_price_action"]
    assert 0.05 <= confidence <= 1.0


def test_event_window_reduces_confidence():
    common = dict(
        regime_probs={"trend_up": 0.55, "range": 0.10, "cascade_risk": 0.05},
        spread_bps=4.0,
        depth_norm=0.9,
        lfi=0.3,
        jump_probability=0.05,
    )
    _, c_no_event = compute_module_weights(GatingInputs(event_window=False, **common))
    _, c_event = compute_module_weights(GatingInputs(event_window=True, **common))
    assert c_event < c_no_event


def test_recommended_metrics_mapping():
    direction = recommended_metrics_for_target("direction")
    quantiles = recommended_metrics_for_target("quantiles")
    assert "mcc" in direction
    assert "pinball_loss" in quantiles
