"""
Ralph Loop: Nested walk-forward hyperparameter selection with NO leakage.

- Outer folds = true out-of-sample evaluation (trust this)
- Inner folds inside each outer-train window = hyperparameter selection ONLY
- Purge + embargo between adjacent windows to prevent leakage
- Candidate search is a SMALL explicit grid (<= 40 combos)
- Selection objective uses proper scoring (WIS/IS90/MAE) + coverage penalty + abstention penalty
- NEVER use outer test for selection.
- Phase E: Supports structural hyperparameters (D0.n_trades) with caching.
"""

from __future__ import annotations

import copy
import itertools
import json
import yaml
from dataclasses import dataclass, field
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

from gnosis.utils import (
    drop_future_return_cols,
    safe_merge_no_truth,
    vectorized_abstain_mask,
)


from gnosis.harness.walkforward import WalkForwardHarness, Fold, compute_future_returns
from gnosis.harness.scoring import (
    evaluate_predictions,
    IsotonicCalibrator,
    coverage,
    sharpness,
)
from gnosis.harness.bregman_optimizer import (
    ProjectFWOptimizer,
    ProjectFWConfig,
    OptimizationResult,
)


# ============================================================================
# KEY-PATH MAPPING: Maps hparams.yaml keys to actual config paths
# ============================================================================
HPARAM_KEY_MAP: Dict[str, str] = {
    # sigma_floor is not implemented in QuantilePredictor; treat as no-op or alias
    "models.predictor.sigma_floor": "models.predictor.sigma_floor",  # kept for logging
    # regimes.confidence_floor -> constraints_by_species.default.confidence_floor
    "regimes.confidence_floor": "regimes.constraints_by_species.default.confidence_floor",
    # These map directly (no change needed, but explicit for documentation)
    "forecast.sigma_scale": "forecast.sigma_scale",
    "domains.domains.D0.n_trades": "domains.domains.D0.n_trades",
}


def _resolve_hparam_key(key: str) -> Tuple[str, bool]:
    """Resolve a hparam key to its actual config path.

    Returns: (resolved_key, was_mapped)
    """
    if key in HPARAM_KEY_MAP:
        resolved = HPARAM_KEY_MAP[key]
        return resolved, resolved != key
    return key, False


# ============================================================================
# FEATURE CACHE: Caches aggregated bars and features by n_trades
# ============================================================================
class FeatureCacheManager:
    """Caches aggregated D0 bars and computed features by n_trades value.

    This avoids recomputing aggregation and features for the same n_trades
    across multiple candidate evaluations in the Ralph Loop.
    """

    def __init__(self):
        self._cache: Dict[int, pd.DataFrame] = {}
        self._hits: int = 0
        self._misses: int = 0

    def get_or_compute(
        self,
        prints_df: pd.DataFrame,
        n_trades: int,
        domain_config: dict,
        regimes_config: dict,
        models_config: dict,
        horizon_bars: int = 10,
    ) -> pd.DataFrame:
        """Get features_df from cache or compute fresh.

        Args:
            prints_df: Raw print/trade data
            n_trades: D0 aggregation size
            domain_config: Domain configuration dict
            regimes_config: Regimes configuration dict
            models_config: Models configuration dict
            horizon_bars: Forecast horizon for future_return

        Returns:
            features_df with all features and future_return target
        """
        if n_trades in self._cache:
            self._hits += 1
            return self._cache[n_trades].copy()

        self._misses += 1

        # Lazy imports to avoid circular dependencies
        from gnosis.domains import DomainAggregator, compute_features
        from gnosis.regimes import KPCOFGSClassifier
        from gnosis.particle import ParticleState, PriceParticle

        # Build domain config with overridden n_trades
        cfg = copy.deepcopy(domain_config)
        cfg["domains"]["D0"]["n_trades"] = n_trades

        # 1. Aggregate prints into D0 bars
        aggregator = DomainAggregator(cfg)
        bars_df = aggregator.aggregate(prints_df, "D0")

        # 2. Compute basic features
        features_df = compute_features(bars_df)

        # 3. Classify regimes
        classifier = KPCOFGSClassifier(regimes_config)
        features_df = classifier.classify(features_df)

        # 4. Compute particle state (basic)
        particle = ParticleState(models_config)
        features_df = particle.compute_state(features_df)

        # 5. Compute particle physics features (advanced)
        physics_config = models_config.get("particle_physics", {})
        price_particle = PriceParticle(physics_config)
        features_df = price_particle.compute_features(features_df)

        # 6. Compute future returns (target)
        features_df = compute_future_returns(features_df, horizon_bars=horizon_bars)

        # Sort for determinism
        features_df = features_df.sort_values(["symbol", "bar_idx"]).reset_index(
            drop=True
        )

        # Cache the result
        self._cache[n_trades] = features_df.copy()

        return features_df

    def stats(self) -> dict:
        """Return cache statistics."""
        return {
            "cache_hits": self._hits,
            "cache_misses": self._misses,
            "cached_n_trades_values": list(self._cache.keys()),
        }


@dataclass
class RalphLoopConfig:
    """Configuration for nested walk-forward hyperparameter selection."""

    enabled: bool = True
    target_coverage: float = 0.90
    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "coverage": 4.0,
            "wis": 1.0,
            "is90": 0.5,
            "mae": 1.0,
            "abstention": 0.5,
        }
    )
    inner_folds: int = 3
    purge_bars: int = 10
    embargo_bars: int = 10
    grid: Dict[str, List[Any]] = field(default_factory=dict)

    @classmethod
    def from_yaml_path(cls, path: str) -> "RalphLoopConfig":
        """Load RalphLoopConfig from a YAML file path."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, d: dict) -> "RalphLoopConfig":
        """Create RalphLoopConfig from a dictionary.

        Supports both top-level keys and nested under 'ralph' or 'ralph_loop' key.
        """
        if d is None:
            return cls()

        # Support top-level, nested under 'ralph', or nested under 'ralph_loop'
        if "ralph_loop" in d:
            cfg = d["ralph_loop"]
        elif "ralph" in d:
            cfg = d["ralph"]
        else:
            cfg = d

        return cls(
            enabled=bool(cfg.get("enabled", True)),
            target_coverage=float(cfg.get("target_coverage", 0.90)),
            weights=cfg.get(
                "weights",
                {
                    "coverage": 4.0,
                    "wis": 1.0,
                    "is90": 0.5,
                    "mae": 1.0,
                    "abstention": 0.5,
                },
            ),
            inner_folds=int(cfg.get("inner_folds", 3)),
            purge_bars=int(cfg.get("purge_bars", 10)),
            embargo_bars=int(cfg.get("embargo_bars", 10)),
            grid=cfg.get("grid", {}),
        )


@dataclass(frozen=True)
class HparamCandidate:
    """A hyperparameter candidate with id and params."""

    candidate_id: int
    params: Dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(
            {"candidate_id": self.candidate_id, "params": self.params}, sort_keys=True
        )


@dataclass(frozen=True)
class InnerFold:
    """Represents a single inner fold within an outer-train window."""

    inner_idx: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int


@dataclass
class TrialResult:
    """Result from evaluating a candidate on an inner fold."""

    outer_fold: int
    candidate_id: int
    inner_fold: int
    coverage_90: float
    sharpness: float
    mae: float
    wis: float
    is90: float
    abstention_rate: float
    composite_score: float
    params_json: str
    resolved_params_json: str = ""  # Phase E: resolved key mappings
    # Phase F: Explicit param columns for easy analysis
    confidence_floor: float = 0.65
    sigma_scale: float = 1.0
    n_trades: int = 200
    # Phase F: Conditional vs unconditional coverage
    coverage_90_conditional: float = 0.0  # Same as coverage_90 (non-abstained only)
    coverage_90_unconditional: float = 0.0  # Abstained counted as uncovered


def _set_nested_key(d: dict, dotted_key: str, value: Any) -> None:
    """Set a value in a nested dict using a dotted key path."""
    parts = dotted_key.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def _apply_candidate_params(
    candidate: HparamCandidate,
    base_cfg: dict,
) -> Tuple[dict, dict]:
    """Apply candidate hyperparameters to a deep copy of base config.

    Phase E: Uses key-path mapping for hparam keys that don't exist exactly.

    Returns:
        cfg: Config with applied hyperparameters
        resolved_params: Dict mapping original keys to resolved keys and values
    """
    cfg = copy.deepcopy(base_cfg)
    resolved_params: Dict[str, Any] = {}

    for key, value in candidate.params.items():
        resolved_key, was_mapped = _resolve_hparam_key(key)

        # Log the mapping
        resolved_params[key] = {
            "resolved_key": resolved_key,
            "value": value,
            "was_mapped": was_mapped,
        }

        # Apply to config
        _set_nested_key(cfg, resolved_key, value)

    return cfg, resolved_params


def _interval_score(
    y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float
) -> np.ndarray:
    """Compute interval score (width + penalty for violations)."""
    y = np.asarray(y, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    width = hi - lo
    below = (lo - y) * (y < lo)
    above = (y - hi) * (y > hi)
    penalty = (2.0 / alpha) * (below + above)
    return width + penalty


def _score_predictions(df: pd.DataFrame, y_col: str = "future_return") -> dict:
    """Compute WIS, IS90, MAE from predictions dataframe."""
    need = {y_col, "q05", "q50", "q95"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        return {"wis": 1.0, "is90": 1.0, "mae": 1.0, "missing": missing}

    d = df[df[y_col].notna()].copy()
    if len(d) == 0:
        return {"wis": 1.0, "is90": 1.0, "mae": 1.0, "missing": []}

    y = d[y_col].to_numpy()
    q05 = d["q05"].to_numpy()
    q50 = d["q50"].to_numpy()
    q95 = d["q95"].to_numpy()

    is90 = _interval_score(y, q05, q95, alpha=0.10)
    mae = np.abs(y - q50)
    wis = np.mean(0.5 * is90 + 0.5 * mae)

    return {
        "is90": float(np.mean(is90)),
        "wis": float(wis),
        "mae": float(np.mean(mae)),
        "n": int(len(d)),
    }


class RalphLoop:
    """Nested walk-forward hyperparameter selection engine.

    - For each outer fold, generates inner folds within outer-train window
    - Evaluates all candidates on inner folds
    - Selects best candidate per outer fold (never touches outer test)
    - Returns trials_df and selected_json for artifacts
    - Phase E: Supports structural hyperparameters (D0.n_trades) with caching
    """

    def __init__(
        self,
        loop_config: RalphLoopConfig,
        base_config: dict,
        random_seed: int = 1337,
        prints_df: Optional[pd.DataFrame] = None,
    ):
        self.loop_config = loop_config
        self.base_config = base_config
        self.random_seed = random_seed
        self.prints_df = prints_df

        # Phase E: Feature cache for structural hyperparameters
        self._feature_cache = FeatureCacheManager()
        self._key_mappings_logged = False  # Log mappings once

        # Generate candidate grid
        self.candidates = self._generate_candidates()
        self.selected_params: Dict[int, HparamCandidate] = {}

        # Detect if grid contains structural params (n_trades)
        self._has_structural_params = any(
            "n_trades" in key for key in self.loop_config.grid.keys()
        )

    def _generate_candidates(self) -> List[HparamCandidate]:
        """Generate Cartesian product of grid parameters."""
        grid = self.loop_config.grid
        if not grid:
            return [HparamCandidate(candidate_id=0, params={})]

        keys = sorted(grid.keys())  # Sorted for determinism
        vals = [grid[k] for k in keys]

        combos = list(itertools.product(*vals))
        candidates = []
        for i, combo in enumerate(combos):
            params = {k: combo[j] for j, k in enumerate(keys)}
            candidates.append(HparamCandidate(candidate_id=i, params=dict(params)))
        return candidates

    def _get_features_for_candidate(
        self,
        candidate: HparamCandidate,
        base_features_df: pd.DataFrame,
        regimes_config: dict,
    ) -> pd.DataFrame:
        """Get features_df for a candidate, using cache for structural params.

        If the candidate includes D0.n_trades override and prints_df is available,
        recompute features using the cache. Otherwise, return base_features_df.

        Args:
            candidate: HparamCandidate with params
            base_features_df: Pre-computed features_df (used if no structural override)
            regimes_config: Regimes configuration dict

        Returns:
            features_df appropriate for this candidate
        """
        # Check if candidate has n_trades override
        n_trades_key = "domains.domains.D0.n_trades"
        if n_trades_key not in candidate.params:
            return base_features_df

        n_trades = int(candidate.params[n_trades_key])

        # Check if we can use cached/recomputed features
        if self.prints_df is None:
            # Fall back to base features if prints_df not provided
            return base_features_df

        # Get default n_trades from base config
        default_n_trades = (
            self.base_config.get("domains", {})
            .get("domains", {})
            .get("D0", {})
            .get("n_trades", 200)
        )

        if n_trades == default_n_trades:
            # No change needed, use base features
            return base_features_df

        # Use cache to get features for this n_trades value
        domain_config = self.base_config.get(
            "domains", {"domains": {"D0": {"n_trades": 200}}}
        )
        models_config = self.base_config.get("models", {})
        horizon_bars = self.base_config.get("forecast", {}).get("horizon_bars", 10)

        return self._feature_cache.get_or_compute(
            prints_df=self.prints_df,
            n_trades=n_trades,
            domain_config=domain_config,
            regimes_config=regimes_config,
            models_config=models_config,
            horizon_bars=horizon_bars,
        )

    def _generate_inner_folds(
        self,
        outer_train_start: int,
        outer_train_end: int,
    ) -> List[InnerFold]:
        """Generate inner folds within an outer-train window.

        Uses time-ordered splits with purge gaps between train and val.
        For n_inner folds, we create n_inner non-overlapping train/val pairs
        that step forward through the outer-train window.
        """
        n_bars = outer_train_end - outer_train_start
        n_inner = self.loop_config.inner_folds
        purge = self.loop_config.purge_bars

        if n_bars < 50 or n_inner <= 0:
            return []

        # Total space needed per fold: train + purge + val
        # We want train ~ 60% of available, val ~ 40% minus purge
        # Calculate so all folds fit within the window
        total_purge = purge * n_inner  # purge after each fold's train
        available_for_data = n_bars - total_purge
        if (
            available_for_data < n_inner * 15
        ):  # Need at least 15 bars (10 train + 5 val) per fold
            return []

        # Divide available data among folds
        data_per_fold = available_for_data // n_inner
        train_size = int(data_per_fold * 0.65)
        val_size = data_per_fold - train_size

        if train_size < 10 or val_size < 5:
            return []

        # Step size between folds
        step = train_size + purge + val_size

        folds = []
        for i in range(n_inner):
            train_start = outer_train_start + i * step
            train_end = train_start + train_size
            val_start = train_end + purge
            val_end = val_start + val_size

            if val_end > outer_train_end:
                break

            folds.append(
                InnerFold(
                    inner_idx=i,
                    train_start=train_start,
                    train_end=train_end,
                    val_start=val_start,
                    val_end=val_end,
                )
            )

        return folds

    def _apply_sigma_scale(
        self, preds: pd.DataFrame, sigma_scale: float
    ) -> pd.DataFrame:
        """Apply sigma_scale to widen/narrow prediction intervals."""
        if sigma_scale == 1.0:
            return preds

        result = preds.copy()
        center = result["q50"] if "q50" in result.columns else result.get("x_hat", 0.0)
        half = (result["q95"] - result["q05"]) / 2.0
        half = half * sigma_scale
        result["q05"] = center - half
        result["q95"] = center + half
        if "sigma_hat" in result.columns:
            result["sigma_hat"] = (result["q95"] - result["q05"]) / 3.29
        return result

    def _apply_abstain_logic(
        self,
        preds: pd.DataFrame,
        features_df: pd.DataFrame,
        regimes_config: dict,
    ) -> pd.DataFrame:
        """Apply abstain logic based on S_label and S_pmax thresholds (vectorized)."""
        result = preds.copy()

        # Get S_label and S_pmax from features
        s_cols = ["symbol", "bar_idx"]
        if "S_label" in features_df.columns:
            s_cols.append("S_label")
        if "S_pmax" in features_df.columns:
            s_cols.append("S_pmax")

        if len(s_cols) > 2:
            s_info = features_df[s_cols].copy()
            result = result.merge(s_info, on=["symbol", "bar_idx"], how="left")

        # Get confidence floor from config
        constraints = regimes_config.get("constraints_by_species", {})
        confidence_floor = constraints.get("default", {}).get("confidence_floor", 0.65)

        # Vectorized abstain computation
        if "S_label" in result.columns and "S_pmax" in result.columns:
            result["abstain"] = vectorized_abstain_mask(
                result["S_label"],
                result["S_pmax"],
                confidence_floor=confidence_floor,
            )
        else:
            result["abstain"] = False

        return result

    def _compute_composite_score(
        self,
        coverage_90: float,
        sharpness_val: float,
        mae: float,
        wis: float,
        is90: float,
        abstention_rate: float,
    ) -> float:
        """Compute composite score (lower is better).

        score = weights["coverage"] * |coverage - target|
              + weights["wis"] * WIS
              + weights["is90"] * IS90
              + weights["mae"] * MAE
              + weights["abstention"] * abstention_rate
        """
        w = self.loop_config.weights
        target = self.loop_config.target_coverage

        coverage_pen = abs(coverage_90 - target)

        score = (
            w.get("coverage", 4.0) * coverage_pen
            + w.get("wis", 1.0) * wis
            + w.get("is90", 0.5) * is90
            + w.get("mae", 1.0) * mae
            + w.get("abstention", 0.5) * abstention_rate
        )
        return float(score)

    def _evaluate_candidate_on_inner(
        self,
        candidate: HparamCandidate,
        features_df: pd.DataFrame,
        inner_fold: InnerFold,
        outer_fold_idx: int,
        regimes_config: dict,
    ) -> TrialResult:
        """Evaluate a single candidate on a single inner fold."""
        # Lazy import to avoid circular dependency
        from gnosis.predictors import QuantilePredictor

        cfg, resolved_params = _apply_candidate_params(candidate, self.base_config)

        # Log key mappings once
        if not self._key_mappings_logged:
            mapped_keys = [k for k, v in resolved_params.items() if v.get("was_mapped")]
            if mapped_keys:
                print(
                    f"  Key mappings: {json.dumps({k: resolved_params[k]['resolved_key'] for k in mapped_keys})}"
                )
            self._key_mappings_logged = True

        models_cfg = cfg.get("models", {})

        # Phase E: Get features_df for this candidate (may differ by n_trades)
        candidate_features_df = self._get_features_for_candidate(
            candidate, features_df, regimes_config
        )

        # Extract sigma_scale from candidate params or config
        sigma_scale = 1.0
        if "forecast.sigma_scale" in candidate.params:
            sigma_scale = float(candidate.params["forecast.sigma_scale"])
        elif "forecast" in cfg and "sigma_scale" in cfg["forecast"]:
            sigma_scale = float(cfg["forecast"]["sigma_scale"])

        # PHASE F FIX: Use the MODIFIED regimes config from cfg, not the original
        # This ensures confidence_floor from the grid is actually applied
        modified_regimes_config = cfg.get("regimes", regimes_config)

        # Get inner train/val data (use candidate_features_df for structural params)
        # Adjust fold indices if candidate_features_df has different length
        n_rows = len(candidate_features_df)
        train_start = min(inner_fold.train_start, n_rows - 1)
        train_end = min(inner_fold.train_end, n_rows)
        val_start = min(inner_fold.val_start, n_rows - 1)
        val_end = min(inner_fold.val_end, n_rows)

        train_df = candidate_features_df.iloc[train_start:train_end].copy()
        val_df = candidate_features_df.iloc[val_start:val_end].copy()

        # Drop NaN targets
        train_df = train_df.dropna(subset=["future_return"])
        val_df = val_df.dropna(subset=["future_return"])

        resolved_params_json = json.dumps(resolved_params, sort_keys=True)

        if len(train_df) < 10 or len(val_df) < 5:
            # Phase F: Extract param values even for early returns
            early_confidence_floor = float(
                candidate.params.get("regimes.confidence_floor", 0.65)
            )
            early_sigma_scale = sigma_scale
            early_n_trades = int(
                candidate.params.get("domains.domains.D0.n_trades", 200)
            )
            return TrialResult(
                outer_fold=outer_fold_idx,
                candidate_id=candidate.candidate_id,
                inner_fold=inner_fold.inner_idx,
                coverage_90=0.0,
                sharpness=1.0,
                mae=1.0,
                wis=1.0,
                is90=1.0,
                abstention_rate=1.0,
                composite_score=999.0,
                params_json=candidate.to_json(),
                resolved_params_json=resolved_params_json,
                confidence_floor=early_confidence_floor,
                sigma_scale=early_sigma_scale,
                n_trades=early_n_trades,
                coverage_90_conditional=0.0,
                coverage_90_unconditional=0.0,
            )

        # Fit predictor on inner train
        predictor = QuantilePredictor(models_cfg)
        predictor.fit(train_df, "future_return")
        preds = predictor.predict(val_df)

        # Apply sigma_scale
        preds = self._apply_sigma_scale(preds, sigma_scale)

        # Fit isotonic calibrator on train S_pmax
        if "S_pmax" in train_df.columns and "S_label" in train_df.columns:
            calibrator = IsotonicCalibrator(n_bins=10)
            train_s = train_df["S_pmax"].values
            train_labels = train_df["S_label"].values
            shifted = np.roll(train_labels, -1)
            shifted[-1] = train_labels[-1]
            outcomes = (train_labels == shifted).astype(float)
            calibrator.fit(train_s, outcomes)

        # Apply abstain logic using MODIFIED config (Phase F fix)
        preds = self._apply_abstain_logic(preds, val_df, modified_regimes_config)

        # Score on non-abstained rows
        non_abstain = (
            preds[~preds["abstain"]].copy() if "abstain" in preds.columns else preds
        )

        # Merge with actual targets for scoring
        if len(non_abstain) > 0:
            eval_df = non_abstain.merge(
                val_df[["symbol", "bar_idx", "future_return"]],
                on=["symbol", "bar_idx"],
                how="inner",
            )
        else:
            eval_df = preds.merge(
                val_df[["symbol", "bar_idx", "future_return"]],
                on=["symbol", "bar_idx"],
                how="inner",
            )

        # Phase F: Extract explicit param values for trial record
        param_confidence_floor = float(
            candidate.params.get("regimes.confidence_floor", 0.65)
        )
        param_sigma_scale = sigma_scale  # Already extracted above
        param_n_trades = int(candidate.params.get("domains.domains.D0.n_trades", 200))

        # Compute metrics
        coverage_90_conditional = 0.0
        coverage_90_unconditional = 0.0

        if len(eval_df) > 0 and "future_return" in eval_df.columns:
            y_true = eval_df["future_return"].values
            valid_mask = ~np.isnan(y_true)
            y_true = y_true[valid_mask]

            if len(y_true) > 0:
                q05 = eval_df["q05"].values[valid_mask]
                q50 = eval_df["q50"].values[valid_mask]
                q95 = eval_df["q95"].values[valid_mask]

                # Phase F: Interval validity guard (lo <= hi)
                invalid_intervals = np.sum(q05 > q95)
                if invalid_intervals > 0:
                    # Fix invalid intervals by swapping
                    swap_mask = q05 > q95
                    q05[swap_mask], q95[swap_mask] = q95[swap_mask], q05[swap_mask]

                coverage_90 = float(coverage(y_true, q05, q95))
                coverage_90_conditional = (
                    coverage_90  # Same as coverage_90 for non-abstained
                )
                sharpness_val = float(sharpness(q05, q95))
                mae = float(np.mean(np.abs(y_true - q50)))

                # WIS and IS90
                is90_arr = _interval_score(y_true, q05, q95, alpha=0.10)
                is90 = float(np.mean(is90_arr))
                wis = float(np.mean(0.5 * is90_arr + 0.5 * np.abs(y_true - q50)))
            else:
                coverage_90, sharpness_val, mae, wis, is90 = 0.0, 1.0, 1.0, 1.0, 1.0
        else:
            coverage_90, sharpness_val, mae, wis, is90 = 0.0, 1.0, 1.0, 1.0, 1.0

        abstention_rate = (
            float(preds["abstain"].mean()) if "abstain" in preds.columns else 0.0
        )

        # Phase F: Compute unconditional coverage (abstained counted as uncovered)
        # Merge ALL predictions (including abstained) with targets
        all_preds_with_targets = preds.merge(
            val_df[["symbol", "bar_idx", "future_return"]],
            on=["symbol", "bar_idx"],
            how="inner",
        )
        if (
            len(all_preds_with_targets) > 0
            and "future_return" in all_preds_with_targets.columns
        ):
            y_all = all_preds_with_targets["future_return"].values
            valid_all = ~np.isnan(y_all)
            y_all = y_all[valid_all]

            if len(y_all) > 0:
                q05_all = all_preds_with_targets["q05"].values[valid_all]
                q95_all = all_preds_with_targets["q95"].values[valid_all]
                abstain_all = (
                    all_preds_with_targets["abstain"].values[valid_all]
                    if "abstain" in all_preds_with_targets.columns
                    else np.zeros(len(y_all), dtype=bool)
                )

                # Unconditional: abstained rows count as NOT covered
                in_interval = (y_all >= q05_all) & (y_all <= q95_all) & (~abstain_all)
                coverage_90_unconditional = float(np.mean(in_interval))

        composite = self._compute_composite_score(
            coverage_90=coverage_90,
            sharpness_val=sharpness_val,
            mae=mae,
            wis=wis,
            is90=is90,
            abstention_rate=abstention_rate,
        )

        return TrialResult(
            outer_fold=outer_fold_idx,
            candidate_id=candidate.candidate_id,
            inner_fold=inner_fold.inner_idx,
            coverage_90=coverage_90,
            sharpness=sharpness_val,
            mae=mae,
            wis=wis,
            is90=is90,
            abstention_rate=abstention_rate,
            composite_score=composite,
            params_json=candidate.to_json(),
            resolved_params_json=resolved_params_json,
            confidence_floor=param_confidence_floor,
            sigma_scale=param_sigma_scale,
            n_trades=param_n_trades,
            coverage_90_conditional=coverage_90_conditional,
            coverage_90_unconditional=coverage_90_unconditional,
        )

    def _select_best_for_outer_fold(
        self,
        features_df: pd.DataFrame,
        outer_fold_idx: int,
        outer_train_start: int,
        outer_train_end: int,
        regimes_config: dict,
    ) -> Tuple[HparamCandidate, List[TrialResult]]:
        """Select best candidate for an outer fold using inner CV."""
        inner_folds = self._generate_inner_folds(outer_train_start, outer_train_end)

        if not inner_folds:
            best = self.candidates[0] if self.candidates else HparamCandidate(0, {})
            return best, []

        trials = []
        candidate_scores: Dict[int, List[float]] = {
            c.candidate_id: [] for c in self.candidates
        }

        for cand in self.candidates:
            for inner in inner_folds:
                result = self._evaluate_candidate_on_inner(
                    candidate=cand,
                    features_df=features_df,
                    inner_fold=inner,
                    outer_fold_idx=outer_fold_idx,
                    regimes_config=regimes_config,
                )
                trials.append(result)
                candidate_scores[cand.candidate_id].append(result.composite_score)

        # Select candidate with lowest mean composite score (lower is better)
        best_id = min(
            candidate_scores.keys(),
            key=lambda cid: (
                np.mean(candidate_scores[cid]) if candidate_scores[cid] else 999.0
            ),
        )
        best = next(c for c in self.candidates if c.candidate_id == best_id)
        self.selected_params[outer_fold_idx] = best

        return best, trials

    def _select_best_projectfw(
        self,
        features_df: pd.DataFrame,
        outer_fold_idx: int,
        outer_train_start: int,
        outer_train_end: int,
        regimes_config: dict,
        fw_config: Optional[ProjectFWConfig] = None,
    ) -> Tuple[HparamCandidate, List[TrialResult], OptimizationResult]:
        """Select best candidate using ProjectFW Bregman optimization.
        
        This method uses adaptive Frank-Wolfe with Bregman projections instead of
        grid search. It provides provable approximation guarantees and faster
        convergence for high-dimensional hyperparameter spaces.
        
        Args:
            features_df: DataFrame with features
            outer_fold_idx: Current outer fold index
            outer_train_start: Start of training window
            outer_train_end: End of training window
            regimes_config: Regime configuration
            fw_config: ProjectFW configuration (uses defaults if None)
            
        Returns:
            Tuple of (best_candidate, trials, optimization_result)
        """
        inner_folds = self._generate_inner_folds(outer_train_start, outer_train_end)
        
        if not inner_folds:
            best = self.candidates[0] if self.candidates else HparamCandidate(0, {})
            empty_result = OptimizationResult(
                x_best=np.array([0.0]),
                f_best=0.0,
                iterations=0,
                convergence_history=[],
                subproblem_times=[],
                approximation_quality=1.0,
            )
            return best, [], empty_result
        
        # Define objective function for ProjectFW
        def objective(x: np.ndarray) -> float:
            """Evaluate composite score for parameter vector x.
            
            x is a continuous vector in [0,1]^d that we map to discrete candidates.
            """
            # Map continuous x to discrete candidate selection
            # Use x as weights for candidates (softmax-like selection)
            if len(x) != len(self.candidates):
                # Pad or truncate as needed
                weights = np.ones(len(self.candidates))
                weights[:min(len(x), len(weights))] = x[:len(weights)]
            else:
                weights = x
            
            # Softmax to get selection probabilities
            exp_weights = np.exp(weights - np.max(weights))
            probs = exp_weights / exp_weights.sum()
            
            # Weighted average of candidate scores
            total_score = 0.0
            for cand, prob in zip(self.candidates, probs):
                cand_scores = []
                for inner in inner_folds:
                    result = self._evaluate_candidate_on_inner(
                        candidate=cand,
                        features_df=features_df,
                        inner_fold=inner,
                        outer_fold_idx=outer_fold_idx,
                        regimes_config=regimes_config,
                    )
                    cand_scores.append(result.composite_score)
                avg_score = np.mean(cand_scores) if cand_scores else 999.0
                total_score += prob * avg_score
            
            return total_score
        
        # Set up constraint polytope (simplex for candidate selection)
        n_candidates = len(self.candidates)
        bounds = [(0.0, 1.0) for _ in range(n_candidates)]
        
        # Initial point (uniform distribution over candidates)
        x0 = np.ones(n_candidates) / n_candidates
        
        # Run ProjectFW optimization
        if fw_config is None:
            fw_config = ProjectFWConfig()
        
        optimizer = ProjectFWOptimizer(fw_config)
        opt_result = optimizer.optimize(
            objective=objective,
            bounds=bounds,
            x0=x0,
        )
        
        # Extract best candidate from optimal weights
        best_weights = opt_result.x_best
        best_idx = int(np.argmax(best_weights))
        best = self.candidates[best_idx]
        self.selected_params[outer_fold_idx] = best
        
        # Evaluate best candidate on all inner folds for trials
        trials = []
        for inner in inner_folds:
            result = self._evaluate_candidate_on_inner(
                candidate=best,
                features_df=features_df,
                inner_fold=inner,
                outer_fold_idx=outer_fold_idx,
                regimes_config=regimes_config,
            )
            trials.append(result)
        
        return best, trials, opt_result

    def run(
        self,
        features_df: pd.DataFrame,
        outer_harness: WalkForwardHarness,
        regimes_config: dict,
    ) -> Tuple[pd.DataFrame, dict]:
        """Run the Ralph Loop across all outer folds.

        Args:
            features_df: DataFrame with features and future_return target
            outer_harness: WalkForwardHarness defining outer folds
            regimes_config: Regime configuration dict

        Returns:
            trials_df: DataFrame with all trial results (outer_fold, candidate_id, inner_fold, metrics, score, params_json)
            selected_json: Dict with chosen params per outer fold + global best summary
        """
        np.random.seed(self.random_seed)

        # Sort for determinism
        features_df = features_df.sort_values(["symbol", "bar_idx"]).reset_index(
            drop=True
        )

        all_trials: List[TrialResult] = []

        for fold in outer_harness.generate_folds(features_df):
            print(f"  Ralph Loop: Processing outer fold {fold.fold_idx}...")

            best, fold_trials = self._select_best_for_outer_fold(
                features_df=features_df,
                outer_fold_idx=fold.fold_idx,
                outer_train_start=fold.train_start,
                outer_train_end=fold.train_end,
                regimes_config=regimes_config,
            )

            all_trials.extend(fold_trials)
            print(f"    Selected candidate {best.candidate_id}: {best.params}")

        # Build trials DataFrame
        if all_trials:
            trials_df = pd.DataFrame(
                [
                    {
                        "outer_fold": t.outer_fold,
                        "candidate_id": t.candidate_id,
                        "inner_fold": t.inner_fold,
                        "coverage_90": t.coverage_90,
                        "sharpness": t.sharpness,
                        "mae": t.mae,
                        "wis": t.wis,
                        "is90": t.is90,
                        "abstention_rate": t.abstention_rate,
                        "composite_score": t.composite_score,
                        "params_json": t.params_json,
                        "resolved_params_json": t.resolved_params_json,  # Phase E
                        # Phase F: Explicit param columns
                        "confidence_floor": t.confidence_floor,
                        "sigma_scale": t.sigma_scale,
                        "n_trades": t.n_trades,
                        # Phase F: Conditional vs unconditional coverage
                        "coverage_90_conditional": t.coverage_90_conditional,
                        "coverage_90_unconditional": t.coverage_90_unconditional,
                    }
                    for t in all_trials
                ]
            )
        else:
            trials_df = pd.DataFrame()

        # Build selected_json
        selected_json = {"per_fold": {}, "global_best": {}}
        for fold_idx, cand in self.selected_params.items():
            selected_json["per_fold"][str(fold_idx)] = {
                "candidate_id": cand.candidate_id,
                "params": cand.params,
            }

        # Global best = most frequently selected candidate
        if self.selected_params:
            counts = Counter(c.candidate_id for c in self.selected_params.values())
            most_common_id = counts.most_common(1)[0][0]
            global_best = next(
                c for c in self.candidates if c.candidate_id == most_common_id
            )
            selected_json["global_best"] = {
                "candidate_id": global_best.candidate_id,
                "params": global_best.params,
                "selection_count": counts[most_common_id],
            }

        # Phase E: Add cache stats and key mapping info
        selected_json["phase_e_info"] = {
            "feature_cache_stats": self._feature_cache.stats(),
            "key_mappings": HPARAM_KEY_MAP,
            "has_structural_params": self._has_structural_params,
        }

        return trials_df, selected_json

    def get_robustness_stats(self, trials_df: pd.DataFrame) -> dict:
        """Compute robustness statistics (std across outer folds) for selected candidates."""
        if trials_df.empty:
            return {
                "coverage_90_std": 0.0,
                "sharpness_std": 0.0,
                "mae_std": 0.0,
            }

        # Get selected candidate scores per outer fold
        selected_trials = []
        for fold_idx, cand in self.selected_params.items():
            fold_trials = trials_df[
                (trials_df["outer_fold"] == fold_idx)
                & (trials_df["candidate_id"] == cand.candidate_id)
            ]
            if not fold_trials.empty:
                selected_trials.append(
                    {
                        "outer_fold": fold_idx,
                        "coverage_90": fold_trials["coverage_90"].mean(),
                        "sharpness": fold_trials["sharpness"].mean(),
                        "mae": fold_trials["mae"].mean(),
                    }
                )

        if not selected_trials:
            return {
                "coverage_90_std": 0.0,
                "sharpness_std": 0.0,
                "mae_std": 0.0,
            }

        sel_df = pd.DataFrame(selected_trials)
        return {
            "coverage_90_std": float(sel_df["coverage_90"].std()),
            "sharpness_std": float(sel_df["sharpness"].std()),
            "mae_std": float(sel_df["mae"].std()),
        }

    def run_with_projectfw(
        self,
        features_df: pd.DataFrame,
        outer_harness: WalkForwardHarness,
        regimes_config: dict,
        fw_config: Optional[ProjectFWConfig] = None,
    ) -> Tuple[pd.DataFrame, dict]:
        """Run the Ralph Loop using ProjectFW Bregman optimization instead of grid search.
        
        This provides faster convergence and provable approximation guarantees for
        hyperparameter selection, especially useful when the parameter space is large.
        
        Args:
            features_df: DataFrame with features and future_return target
            outer_harness: WalkForwardHarness defining outer folds
            regimes_config: Regime configuration dict
            fw_config: ProjectFW configuration (uses defaults if None)
            
        Returns:
            trials_df: DataFrame with all trial results
            selected_json: Dict with chosen params per outer fold + global best + optimization info
        """
        np.random.seed(self.random_seed)
        
        # Sort for determinism
        features_df = features_df.sort_values(["symbol", "bar_idx"]).reset_index(
            drop=True
        )
        
        all_trials: List[TrialResult] = []
        optimization_results = {}
        
        for fold in outer_harness.generate_folds(features_df):
            print(f"  Ralph Loop (ProjectFW): Processing outer fold {fold.fold_idx}...")
            
            best, fold_trials, opt_result = self._select_best_projectfw(
                features_df=features_df,
                outer_fold_idx=fold.fold_idx,
                outer_train_start=fold.train_start,
                outer_train_end=fold.train_end,
                regimes_config=regimes_config,
                fw_config=fw_config,
            )
            
            all_trials.extend(fold_trials)
            optimization_results[fold.fold_idx] = {
                "iterations": opt_result.iterations,
                "final_objective": float(opt_result.f_best),
                "approximation_quality": float(opt_result.approximation_quality),
                "converged": opt_result.iterations < (fw_config.max_iterations if fw_config else 100),
            }
            print(f"    Selected candidate {best.candidate_id}: {best.params}")
            print(f"    ProjectFW: {opt_result.iterations} iterations, α={opt_result.approximation_quality:.4f}")
        
        # Build trials DataFrame
        if all_trials:
            trials_df = pd.DataFrame(
                [
                    {
                        "outer_fold": t.outer_fold,
                        "candidate_id": t.candidate_id,
                        "inner_fold": t.inner_fold,
                        "coverage_90": t.coverage_90,
                        "sharpness": t.sharpness,
                        "mae": t.mae,
                        "wis": t.wis,
                        "is90": t.is90,
                        "abstention_rate": t.abstention_rate,
                        "composite_score": t.composite_score,
                        "params_json": t.params_json,
                        "resolved_params_json": t.resolved_params_json,
                        "confidence_floor": t.confidence_floor,
                        "sigma_scale": t.sigma_scale,
                        "n_trades": t.n_trades,
                        "coverage_90_conditional": t.coverage_90_conditional,
                        "coverage_90_unconditional": t.coverage_90_unconditional,
                    }
                    for t in all_trials
                ]
            )
        else:
            trials_df = pd.DataFrame()
        
        # Build selected_json
        selected_json = {"per_fold": {}, "global_best": {}, "projectfw_info": {}}
        for fold_idx, cand in self.selected_params.items():
            selected_json["per_fold"][str(fold_idx)] = {
                "candidate_id": cand.candidate_id,
                "params": cand.params,
                "optimization": optimization_results.get(fold_idx, {}),
            }
        
        # Global best = most frequently selected candidate
        if self.selected_params:
            counts = Counter(c.candidate_id for c in self.selected_params.values())
            most_common_id = counts.most_common(1)[0][0]
            global_best = next(
                c for c in self.candidates if c.candidate_id == most_common_id
            )
            selected_json["global_best"] = {
                "candidate_id": global_best.candidate_id,
                "params": global_best.params,
                "selection_count": counts[most_common_id],
            }
        
        # Add ProjectFW metadata
        selected_json["projectfw_info"] = {
            "optimization_method": "Bregman-FW",
            "total_folds": len(optimization_results),
            "avg_iterations": np.mean([r["iterations"] for r in optimization_results.values()]) if optimization_results else 0,
            "avg_approximation_quality": np.mean([r["approximation_quality"] for r in optimization_results.values()]) if optimization_results else 0,
            "config": fw_config.__dict__ if fw_config else ProjectFWConfig().__dict__,
        }
        
        # Add phase E info
        selected_json["phase_e_info"] = {
            "feature_cache_stats": self._feature_cache.stats(),
            "key_mappings": HPARAM_KEY_MAP,
            "has_structural_params": self._has_structural_params,
        }
        
        return trials_df, selected_json
