"""Modular forecasting taxonomy + regime/liquidity gating policy.

This module encodes a practical 12-paradigm forecasting stack with:
- explicit module specs (inputs, outputs, trust/failure conditions),
- gating based on regime probabilities + liquidity + leverage fragility,
- confidence scaling for forecast distribution width control,
- target-to-metrics recommendations for robust evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


DEFAULT_MODULE_ORDER = (
    "technical_price_action",
    "classical_timeseries",
    "macro_factor",
    "derivatives_positioning",
    "orderflow_microstructure",
    "onchain_priors",
    "sentiment_attention",
    "ml_tabular_meta",
    "deep_sequence_distribution",
    "regime_state_machine",
    "scenario_monte_carlo",
    "crowd_prediction_markets",
)


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _normalize_positive(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(float(v), 0.0) for v in weights.values())
    if total <= 0.0:
        uniform = 1.0 / max(len(weights), 1)
        return {k: uniform for k in weights}
    return {k: max(float(v), 0.0) / total for k, v in weights.items()}


@dataclass(frozen=True)
class ForecastModuleSpec:
    """Interface-level definition for one forecasting paradigm module."""

    key: str
    target_types: list[str]
    input_domains: list[str]
    output_schema: str
    trusted_when: list[str]
    failure_modes: list[str]


@dataclass
class GatingInputs:
    """Runtime gating inputs for module blending."""

    regime_probs: dict[str, float]
    spread_bps: float
    depth_norm: float
    lfi: float
    jump_probability: float
    event_window: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class GatingPolicyConfig:
    """Rule thresholds and multipliers for module gating."""

    cascade_prob_threshold: float = 0.30
    trend_prob_threshold: float = 0.50
    range_prob_threshold: float = 0.50
    lfi_high_threshold: float = 1.0
    jump_high_threshold: float = 0.20
    spread_wide_bps: float = 12.0
    depth_thin_threshold: float = 0.35

    # Confidence controls
    min_confidence: float = 0.05
    max_confidence: float = 1.0
    event_confidence_mult: float = 0.70


def build_default_module_registry() -> dict[str, ForecastModuleSpec]:
    """Return canonical 12-paradigm module registry."""
    modules = {
        "technical_price_action": ForecastModuleSpec(
            key="technical_price_action",
            target_types=["direction", "next_return", "volatility"],
            input_domains=["ohlcv", "multi_timeframe_ohlcv"],
            output_schema="direction_prob, drift_score, vol_regime_features",
            trusted_when=[
                "trend/range regime is stable",
                "liquidity is normal and spreads are contained",
            ],
            failure_modes=[
                "jump/cascade events",
                "regime flips and event shocks",
            ],
        ),
        "classical_timeseries": ForecastModuleSpec(
            key="classical_timeseries",
            target_types=["volatility", "distribution", "regime"],
            input_domains=["returns", "realized_vol", "state_space"],
            output_schema="sigma_forecast, latent_state, regime_transition_scores",
            trusted_when=[
                "short-lived autocorrelation structure persists",
                "vol clustering dominates return signal",
            ],
            failure_modes=[
                "structural breaks and heavy-tail jumps",
                "over-reliance on point-return prediction",
            ],
        ),
        "macro_factor": ForecastModuleSpec(
            key="macro_factor",
            target_types=["regime", "drift_context", "risk_state"],
            input_domains=["cross_asset", "macro", "rates", "dxy"],
            output_schema="beta_residual, macro_regime_probabilities",
            trusted_when=[
                "crypto trades as macro beta",
                "risk-on/off phases dominate",
            ],
            failure_modes=[
                "crypto idiosyncratic shocks dominate",
                "exchange/protocol event windows",
            ],
        ),
        "derivatives_positioning": ForecastModuleSpec(
            key="derivatives_positioning",
            target_types=["tail_risk", "jump_probability", "direction"],
            input_domains=["funding", "open_interest", "basis", "liquidations", "options_iv"],
            output_schema="LFI, squeeze_risk, tail_event_probability",
            trusted_when=[
                "crowded leverage is visible",
                "funding/OI/liquidation stress is elevated",
            ],
            failure_modes=[
                "partial venue coverage",
                "persistent extreme funding without immediate unwind",
            ],
        ),
        "orderflow_microstructure": ForecastModuleSpec(
            key="orderflow_microstructure",
            target_types=["short_direction", "near_term_vol_expansion"],
            input_domains=["trades", "l2_book", "ofi", "trade_imbalance"],
            output_schema="short_horizon_direction_prob, liquidity_break_risk",
            trusted_when=[
                "high-quality flow data is available",
                "microstructure is stable and execution-aware",
            ],
            failure_modes=[
                "fragmented/off-exchange flow blind spots",
                "exogenous shocks overriding local flow",
            ],
        ),
        "onchain_priors": ForecastModuleSpec(
            key="onchain_priors",
            target_types=["cycle_regime", "slow_drift_prior", "risk_context"],
            input_domains=["exchange_flows", "holder_behavior", "mvrv", "supply_metrics"],
            output_schema="slow_regime_prior, structural_risk_bias",
            trusted_when=[
                "BTC/ETH cycle context dominates medium horizon",
                "on-chain footprint remains representative",
            ],
            failure_modes=[
                "short-horizon timing expectations",
                "off-chain migration reducing signal representativeness",
            ],
        ),
        "sentiment_attention": ForecastModuleSpec(
            key="sentiment_attention",
            target_types=["extreme_regime_filter", "tail_risk_modulator"],
            input_domains=["news", "social", "search_trends", "mindshare"],
            output_schema="euphoria_panic_score, attention_shock_flag",
            trusted_when=[
                "mania/panic extremes",
                "combined with leverage and liquidity signals",
            ],
            failure_modes=[
                "bot/manipulation contamination",
                "lagging sentiment that follows price",
            ],
        ),
        "ml_tabular_meta": ForecastModuleSpec(
            key="ml_tabular_meta",
            target_types=["direction", "quantiles", "jump_probability"],
            input_domains=["engineered_feature_matrix"],
            output_schema="calibrated_probs, quantiles, feature_attribution",
            trusted_when=[
                "strict walk-forward validation",
                "leakage controls and regime-aware features",
            ],
            failure_modes=[
                "random splits / leakage",
                "nonstationary drift with unadapted features",
            ],
        ),
        "deep_sequence_distribution": ForecastModuleSpec(
            key="deep_sequence_distribution",
            target_types=["distribution", "quantiles", "volatility"],
            input_domains=["multivariate_sequences", "regime_features"],
            output_schema="quantile_paths, uncertainty_embeddings",
            trusted_when=[
                "large clean history and stable schema",
                "distributional targets over point forecasts",
            ],
            failure_modes=[
                "rapid regime shift",
                "memorized historical patterns that fail out-of-sample",
            ],
        ),
        "regime_state_machine": ForecastModuleSpec(
            key="regime_state_machine",
            target_types=["regime", "gating_weights", "confidence"],
            input_domains=["vol", "liquidity", "lfi", "jump_features", "trend_features"],
            output_schema="regime_probabilities, module_weight_priors",
            trusted_when=[
                "regime detector is stable and calibrated",
                "used as a meta-layer over specialist forecasters",
            ],
            failure_modes=[
                "lag/whipsaw in transition zones",
                "overconfident single-regime assignment",
            ],
        ),
        "scenario_monte_carlo": ForecastModuleSpec(
            key="scenario_monte_carlo",
            target_types=["distribution", "barrier_probability", "stress_envelope"],
            input_domains=["regime_conditioned_drift_vol_jump"],
            output_schema="path_ensemble_quantiles, barrier_probs, stress_ranges",
            trusted_when=[
                "risk envelope and scenario analysis",
                "tail-aware uncertainty communication",
            ],
            failure_modes=[
                "misread as precise timing predictor",
                "scenario plausibility confused with probability calibration",
            ],
        ),
        "crowd_prediction_markets": ForecastModuleSpec(
            key="crowd_prediction_markets",
            target_types=["barrier_probability", "event_probability_prior"],
            input_domains=["market_implied_probs"],
            output_schema="external_prior_probabilities",
            trusted_when=[
                "well-specified and liquid binary outcomes",
                "used as prior/sanity check only",
            ],
            failure_modes=[
                "thin/manipulable contracts",
                "poorly specified outcome definitions",
            ],
        ),
    }
    return modules


def compute_module_weights(
    inputs: GatingInputs,
    base_weights: dict[str, float] | None = None,
    config: GatingPolicyConfig | None = None,
) -> tuple[dict[str, float], float]:
    """Compute gated module weights + confidence scalar.

    Returns:
        (weights, confidence)
    """
    cfg = config or GatingPolicyConfig()
    modules = list(DEFAULT_MODULE_ORDER)
    weights = dict(base_weights) if base_weights else {k: 1.0 for k in modules}
    for key in modules:
        weights.setdefault(key, 1.0)

    rp = {k: float(v) for k, v in (inputs.regime_probs or {}).items()}
    p_trend = max(rp.get("trend_up", 0.0), rp.get("trend_down", 0.0))
    p_range = float(rp.get("range", 0.0))
    p_cascade = max(float(rp.get("cascade_risk", 0.0)), float(rp.get("volatility_expansion", 0.0)))

    thin_liquidity = (
        float(inputs.depth_norm) <= float(cfg.depth_thin_threshold)
        or float(inputs.spread_bps) >= float(cfg.spread_wide_bps)
    )
    high_lfi = float(inputs.lfi) >= float(cfg.lfi_high_threshold)
    high_jump = float(inputs.jump_probability) >= float(cfg.jump_high_threshold)

    # Cascade / fragility regime: rely more on derivatives + tail-risk modules.
    if p_cascade >= cfg.cascade_prob_threshold or high_lfi or high_jump:
        for key, mult in (
            ("derivatives_positioning", 1.45),
            ("orderflow_microstructure", 1.25),
            ("scenario_monte_carlo", 1.35),
            ("regime_state_machine", 1.25),
        ):
            weights[key] *= mult
        for key, mult in (
            ("technical_price_action", 0.60),
            ("classical_timeseries", 0.85),
            ("onchain_priors", 0.85),
        ):
            weights[key] *= mult

    # Trend regime with healthy liquidity: momentum/sequence models gain weight.
    if p_trend >= cfg.trend_prob_threshold and not thin_liquidity:
        for key, mult in (
            ("technical_price_action", 1.30),
            ("ml_tabular_meta", 1.15),
            ("deep_sequence_distribution", 1.10),
        ):
            weights[key] *= mult

    # Range regime: mean-reversion/statistical tools become more trusted.
    if p_range >= cfg.range_prob_threshold:
        for key, mult in (
            ("technical_price_action", 1.10),
            ("classical_timeseries", 1.25),
            ("ml_tabular_meta", 1.10),
        ):
            weights[key] *= mult
        weights["deep_sequence_distribution"] *= 0.95

    # Event windows: reduce confidence and shift toward distributional/tail modules.
    if bool(inputs.event_window):
        for key, mult in (
            ("scenario_monte_carlo", 1.30),
            ("derivatives_positioning", 1.15),
            ("crowd_prediction_markets", 1.10),
        ):
            weights[key] *= mult
        for key, mult in (
            ("technical_price_action", 0.65),
            ("classical_timeseries", 0.85),
        ):
            weights[key] *= mult

    # Liquidity stress downweights fragile directional modules.
    if thin_liquidity:
        weights["technical_price_action"] *= 0.80
        weights["orderflow_microstructure"] *= 0.90
        weights["scenario_monte_carlo"] *= 1.15
        weights["regime_state_machine"] *= 1.10

    norm_w = _normalize_positive(weights)

    # Confidence decreases with jump/lfi/liquidity stress and event windows.
    regime_certainty = max(rp.values()) if rp else 0.0
    liquidity_health = 1.0 if not thin_liquidity else 0.65
    risk_penalty = 0.35 * _clamp(inputs.jump_probability, 0.0, 1.0) + 0.20 * _clamp(inputs.lfi / 3.0, 0.0, 1.0)
    confidence = _clamp(0.35 + 0.45 * regime_certainty + 0.20 * liquidity_health - risk_penalty, 0.0, 1.0)
    if bool(inputs.event_window):
        confidence *= cfg.event_confidence_mult
    confidence = _clamp(confidence, cfg.min_confidence, cfg.max_confidence)

    return norm_w, float(confidence)


def recommended_metrics_for_target(target: str) -> list[str]:
    """Return recommended evaluation metrics for a forecasting target."""
    t = str(target).lower().strip()
    mapping = {
        "direction": ["hit_rate", "balanced_accuracy", "mcc", "regime_conditional_hit_rate"],
        "next_return": ["mae", "rmse", "spearman_corr", "ic"],
        "distribution": ["pinball_loss", "crps", "calibration_curve", "interval_coverage"],
        "quantiles": ["pinball_loss", "interval_coverage", "sharpness", "coverage_gap"],
        "volatility": ["mae", "rmse", "qlike", "vol_regime_bucket_error"],
        "tail_risk": ["brier_score", "precision_recall", "roc_auc", "event_recall_at_k"],
        "jump_probability": ["brier_score", "precision_recall", "f1", "calibration_curve"],
        "regime": ["log_loss", "balanced_accuracy", "transition_f1", "whipsaw_rate"],
        "barrier_probability": ["brier_score", "log_loss", "reliability_curve"],
    }
    return mapping.get(t, ["mae", "rmse"])
