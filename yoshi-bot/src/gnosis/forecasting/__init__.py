"""Modular crypto forecasting taxonomy and gating utilities."""

from gnosis.forecasting.modular_ensemble import (
    DEFAULT_MODULE_ORDER,
    ForecastModuleSpec,
    GatingInputs,
    GatingPolicyConfig,
    build_default_module_registry,
    compute_module_weights,
    recommended_metrics_for_target,
)

__all__ = [
    "DEFAULT_MODULE_ORDER",
    "ForecastModuleSpec",
    "GatingInputs",
    "GatingPolicyConfig",
    "build_default_module_registry",
    "compute_module_weights",
    "recommended_metrics_for_target",
]
