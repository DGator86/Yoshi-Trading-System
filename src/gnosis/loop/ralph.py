"""Ralph Loop: Nested walk-forward hyperparameter selection.

Implements nested cross-validation for hyperparameter selection
without data leakage - outer test folds are never used for selection.
"""
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from gnosis.harness import WalkForwardHarness, Fold, evaluate_predictions, compute_stability_metrics
from gnosis.domains import DomainAggregator, compute_features
from gnosis.regimes import KPCOFGSClassifier
from gnosis.particle import ParticleState
from gnosis.predictors import QuantilePredictor


@dataclass
class HparamCandidate:
    """A single hyperparameter candidate configuration."""
    candidate_id: int
    params: Dict[str, Any]

    def to_json(self) -> str:
        """Serialize params to JSON string."""
        return json.dumps(self.params, sort_keys=True)


@dataclass
class InnerFold:
    """Represents an inner fold within an outer training window."""
    inner_idx: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int


@dataclass
class TrialResult:
    """Result of evaluating one candidate on one inner fold."""
    outer_fold: int
    candidate_id: int
    inner_fold: int
    coverage_90: float
    sharpness: float
    mae: float
    abstention_rate: float
    flip_rate: float
    composite_score: float
    params_json: str


@dataclass
class RalphLoopConfig:
    """Configuration for Ralph Loop hyperparameter search."""
    grid: Dict[str, List[Any]] = field(default_factory=dict)
    inner_folds_n: int = 3
    inner_train_ratio: float = 0.6
    inner_val_ratio: float = 0.4
    inner_purge_bars: int = 5
    inner_embargo_bars: int = 5
    # Scoring weights
    coverage_target: float = 0.90
    w1_coverage: float = 2.0
    w2_sharpness: float = 1.0
    w3_mae: float = 0.5
    w4_abstention: float = 0.3
    w5_flip_rate: float = 0.2

    @classmethod
    def from_yaml(cls, hparams_config: dict) -> "RalphLoopConfig":
        """Create config from parsed YAML dict."""
        grid = hparams_config.get("grid", {})
        inner_cfg = hparams_config.get("inner_folds", {})
        scoring = hparams_config.get("scoring_weights", {})

        return cls(
            grid=grid,
            inner_folds_n=inner_cfg.get("n_folds", 3),
            inner_train_ratio=inner_cfg.get("train_ratio", 0.6),
            inner_val_ratio=inner_cfg.get("val_ratio", 0.4),
            inner_purge_bars=hparams_config.get("inner_purge_bars", 5),
            inner_embargo_bars=hparams_config.get("inner_embargo_bars", 5),
            coverage_target=scoring.get("coverage_target", 0.90),
            w1_coverage=scoring.get("w1_coverage", 2.0),
            w2_sharpness=scoring.get("w2_sharpness", 1.0),
            w3_mae=scoring.get("w3_mae", 0.5),
            w4_abstention=scoring.get("w4_abstention", 0.3),
            w5_flip_rate=scoring.get("w5_flip_rate", 0.2),
        )


class RalphLoop:
    """Nested walk-forward hyperparameter selection.

    For each outer fold:
      1. Build inner folds from outer-train window
      2. For each candidate param set:
         - Run pipeline on inner train, evaluate on inner val
         - Compute composite score
      3. Select best candidate by average inner score
      4. Retrain with selected params on full outer-train
      5. Evaluate on outer-test (no leakage - outer-test never seen during selection)
    """

    def __init__(
        self,
        loop_config: RalphLoopConfig,
        base_config: dict,
        random_seed: int = 1337,
    ):
        self.loop_config = loop_config
        self.base_config = base_config
        self.random_seed = random_seed
        self.candidates = self._generate_candidates()
        self.trial_results: List[TrialResult] = []
        self.selected_params: Dict[int, HparamCandidate] = {}

    def _generate_candidates(self) -> List[HparamCandidate]:
        """Generate all candidate parameter combinations from grid."""
        grid = self.loop_config.grid
        if not grid:
            # No grid defined - use single default
            return [HparamCandidate(candidate_id=0, params={})]

        # Get all parameter names and their values
        param_names = sorted(grid.keys())
        param_values = [grid[name] for name in param_names]

        # Generate cartesian product
        candidates = []
        for i, combo in enumerate(itertools.product(*param_values)):
            params = dict(zip(param_names, combo))
            candidates.append(HparamCandidate(candidate_id=i, params=params))

        return candidates

    def _generate_inner_folds(
        self,
        outer_train_start: int,
        outer_train_end: int,
    ) -> Iterator[InnerFold]:
        """Generate inner folds from an outer training window."""
        n_bars = outer_train_end - outer_train_start
        n_inner = self.loop_config.inner_folds_n

        if n_bars < 20:
            # Not enough data for inner folds
            return

        purge = self.loop_config.inner_purge_bars

        # Calculate train and val sizes to fit within available bars
        # Reserve space for purge gap between train and val
        usable_bars = n_bars - purge
        train_ratio = self.loop_config.inner_train_ratio
        val_ratio = self.loop_config.inner_val_ratio
        total_ratio = train_ratio + val_ratio

        # Normalize ratios
        train_size = int(usable_bars * train_ratio / total_ratio)
        val_size = int(usable_bars * val_ratio / total_ratio)

        # Ensure minimum sizes
        train_size = max(train_size, 10)
        val_size = max(val_size, 5)

        # Total window needed per inner fold (no embargo for inner - only train/val)
        window_size = train_size + purge + val_size

        if window_size > n_bars:
            # Still too large - reduce sizes
            train_size = max(10, int(n_bars * 0.5))
            val_size = max(5, int(n_bars * 0.3))
            window_size = train_size + purge + val_size
            if window_size > n_bars:
                # Give up if still too large
                return

        # Step between inner folds
        if n_inner <= 1:
            step = 0
        else:
            remaining = n_bars - window_size
            step = max(1, remaining // (n_inner - 1)) if remaining > 0 else 0

        for i in range(n_inner):
            inner_offset = i * step

            train_start = outer_train_start + inner_offset
            train_end = train_start + train_size

            val_start = train_end + purge
            val_end = val_start + val_size

            # Check bounds
            if val_end > outer_train_end:
                break

            yield InnerFold(
                inner_idx=i,
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
            )

    def _apply_candidate_params(
        self,
        candidate: HparamCandidate,
        base_config: dict,
    ) -> dict:
        """Apply candidate parameters to base config."""
        config = _deep_copy_dict(base_config)

        for key, value in candidate.params.items():
            if key == "domains_D0_n_trades":
                if "domains" not in config:
                    config["domains"] = {"domains": {}}
                if "domains" not in config["domains"]:
                    config["domains"]["domains"] = {}
                if "D0" not in config["domains"]["domains"]:
                    config["domains"]["domains"]["D0"] = {}
                config["domains"]["domains"]["D0"]["n_trades"] = value

            elif key == "predictor_l2_reg":
                if "models" not in config:
                    config["models"] = {}
                if "predictor" not in config["models"]:
                    config["models"]["predictor"] = {}
                config["models"]["predictor"]["l2_reg"] = value

            elif key == "confidence_floor_scale":
                if "regimes" not in config:
                    config["regimes"] = {}
                config["regimes"]["confidence_floor_scale"] = value

            elif key == "particle_flow_span":
                if "models" not in config:
                    config["models"] = {}
                if "particle" not in config["models"]:
                    config["models"]["particle"] = {}
                config["models"]["particle"]["flow_span"] = value

        return config

    def _compute_composite_score(
        self,
        coverage_90: float,
        sharpness: float,
        mae: float,
        abstention_rate: float,
        flip_rate: float,
    ) -> float:
        """Compute composite score for a trial. Higher is better."""
        cfg = self.loop_config

        # Coverage term: want coverage close to target
        coverage_error = abs(coverage_90 - cfg.coverage_target)
        coverage_score = 1.0 - coverage_error

        # Composite: coverage contributes positively, others negatively
        score = (
            cfg.w1_coverage * coverage_score
            - cfg.w2_sharpness * sharpness
            - cfg.w3_mae * mae
            - cfg.w4_abstention * abstention_rate
            - cfg.w5_flip_rate * flip_rate
        )

        return score

    def _evaluate_candidate_on_inner(
        self,
        candidate: HparamCandidate,
        features_df: pd.DataFrame,
        inner_fold: InnerFold,
        outer_fold_idx: int,
        regimes_config: dict,
    ) -> TrialResult:
        """Evaluate a candidate on one inner fold."""
        # Get train/val data
        train_df = features_df.iloc[inner_fold.train_start:inner_fold.train_end].copy()
        val_df = features_df.iloc[inner_fold.val_start:inner_fold.val_end].copy()

        if len(train_df) < 5 or len(val_df) < 3:
            # Not enough data - return worst-case result
            return TrialResult(
                outer_fold=outer_fold_idx,
                candidate_id=candidate.candidate_id,
                inner_fold=inner_fold.inner_idx,
                coverage_90=0.0,
                sharpness=1.0,
                mae=1.0,
                abstention_rate=1.0,
                flip_rate=1.0,
                composite_score=-999.0,
                params_json=candidate.to_json(),
            )

        # Apply candidate params
        config = self._apply_candidate_params(candidate, self.base_config)
        models_config = config.get("models", {})

        # Fit predictor
        predictor = QuantilePredictor(models_config)
        predictor.fit(train_df, "future_return")

        # Predict on validation
        preds = predictor.predict(val_df)

        # Merge S_label and S_pmax for abstain logic
        s_cols = ["symbol", "bar_idx", "S_label", "S_pmax"]
        available_s_cols = [c for c in s_cols if c in val_df.columns]
        if len(available_s_cols) >= 3:  # Need at least symbol, bar_idx, and one S column
            preds = preds.merge(
                val_df[available_s_cols],
                on=["symbol", "bar_idx"],
                how="left",
            )

        # Apply abstain logic
        confidence_floor_scale = candidate.params.get("confidence_floor_scale", 1.0)
        base_floor = 0.65  # default
        adjusted_floor = base_floor * confidence_floor_scale

        abstain_mask = np.zeros(len(preds), dtype=bool)
        if "S_label" in preds.columns and "S_pmax" in preds.columns:
            for idx in range(len(preds)):
                s_label = preds.iloc[idx].get("S_label", "S_UNCERTAIN")
                s_pmax = preds.iloc[idx].get("S_pmax", 0.0)
                if pd.isna(s_label):
                    s_label = "S_UNCERTAIN"
                if pd.isna(s_pmax):
                    s_pmax = 0.0
                if s_label == "S_UNCERTAIN" or s_pmax < adjusted_floor:
                    abstain_mask[idx] = True

        preds["abstain"] = abstain_mask
        abstention_rate = float(preds["abstain"].mean())

        # Evaluate non-abstained predictions
        non_abstain = preds[~preds["abstain"]].copy()
        if len(non_abstain) > 0:
            metrics = evaluate_predictions(non_abstain, val_df, "future_return")
        else:
            metrics = evaluate_predictions(preds, val_df, "future_return")

        coverage_90 = metrics.get("coverage_90", 0.0)
        sharpness = metrics.get("sharpness", 1.0)
        mae = metrics.get("mae", 1.0)

        if np.isnan(coverage_90):
            coverage_90 = 0.0
        if np.isnan(sharpness):
            sharpness = 1.0
        if np.isnan(mae):
            mae = 1.0

        # Compute stability (flip rate) on validation window
        stability = compute_stability_metrics(val_df)
        flip_rate = stability.get("overall_flip_rate", 0.0)

        # Compute composite score
        score = self._compute_composite_score(
            coverage_90, sharpness, mae, abstention_rate, flip_rate
        )

        return TrialResult(
            outer_fold=outer_fold_idx,
            candidate_id=candidate.candidate_id,
            inner_fold=inner_fold.inner_idx,
            coverage_90=coverage_90,
            sharpness=sharpness,
            mae=mae,
            abstention_rate=abstention_rate,
            flip_rate=flip_rate,
            composite_score=score,
            params_json=candidate.to_json(),
        )

    def select_best_for_outer_fold(
        self,
        outer_fold_idx: int,
        outer_train_start: int,
        outer_train_end: int,
        features_df: pd.DataFrame,
        regimes_config: dict,
    ) -> HparamCandidate:
        """Select best hyperparameters for one outer fold using inner validation.

        Returns the candidate with highest average composite score across inner folds.
        """
        inner_folds = list(self._generate_inner_folds(outer_train_start, outer_train_end))

        if not inner_folds:
            # Fallback: use first candidate if no inner folds possible
            return self.candidates[0] if self.candidates else HparamCandidate(0, {})

        candidate_scores: Dict[int, List[float]] = {c.candidate_id: [] for c in self.candidates}

        for candidate in self.candidates:
            for inner_fold in inner_folds:
                result = self._evaluate_candidate_on_inner(
                    candidate, features_df, inner_fold, outer_fold_idx, regimes_config
                )
                self.trial_results.append(result)
                candidate_scores[candidate.candidate_id].append(result.composite_score)

        # Select best by average score
        best_candidate_id = max(
            candidate_scores.keys(),
            key=lambda cid: np.mean(candidate_scores[cid]) if candidate_scores[cid] else -999
        )

        best_candidate = next(c for c in self.candidates if c.candidate_id == best_candidate_id)
        self.selected_params[outer_fold_idx] = best_candidate

        return best_candidate

    def run(
        self,
        features_df: pd.DataFrame,
        outer_harness: WalkForwardHarness,
        regimes_config: dict,
    ) -> Tuple[pd.DataFrame, dict]:
        """Run the full Ralph Loop.

        Returns:
            - trials_df: DataFrame of all trial results (hparams_trials.parquet)
            - selected_json: dict of selected params per fold (selected_hparams.json)
        """
        np.random.seed(self.random_seed)

        # Generate outer folds
        outer_folds = list(outer_harness.generate_folds(features_df))

        for fold in outer_folds:
            print(f"  Ralph Loop: Processing outer fold {fold.fold_idx}...")

            # Select best params using ONLY the outer-train window
            best = self.select_best_for_outer_fold(
                outer_fold_idx=fold.fold_idx,
                outer_train_start=fold.train_start,
                outer_train_end=fold.train_end,
                features_df=features_df,
                regimes_config=regimes_config,
            )
            print(f"    Selected candidate {best.candidate_id}: {best.params}")

        # Build trials DataFrame
        trials_records = []
        for r in self.trial_results:
            trials_records.append({
                "outer_fold": r.outer_fold,
                "candidate_id": r.candidate_id,
                "inner_fold": r.inner_fold,
                "coverage_90": r.coverage_90,
                "sharpness": r.sharpness,
                "mae": r.mae,
                "abstention_rate": r.abstention_rate,
                "flip_rate": r.flip_rate,
                "composite_score": r.composite_score,
                "params_json": r.params_json,
            })
        trials_df = pd.DataFrame(trials_records)

        # Build selected params JSON
        selected_json = {
            "per_fold": {},
            "global_best": None,
        }

        for fold_idx, candidate in self.selected_params.items():
            selected_json["per_fold"][str(fold_idx)] = {
                "candidate_id": candidate.candidate_id,
                "params": candidate.params,
            }

        # Determine global best (most frequently selected, or highest avg score)
        if self.selected_params:
            # Count how often each candidate was selected
            from collections import Counter
            selection_counts = Counter(c.candidate_id for c in self.selected_params.values())
            most_common_id = selection_counts.most_common(1)[0][0]
            global_best = next(c for c in self.candidates if c.candidate_id == most_common_id)
            selected_json["global_best"] = {
                "candidate_id": global_best.candidate_id,
                "params": global_best.params,
                "selection_count": selection_counts[most_common_id],
            }

        return trials_df, selected_json

    def get_robustness_stats(self, trials_df: pd.DataFrame) -> dict:
        """Compute robustness statistics (stddev across outer folds)."""
        if trials_df.empty:
            return {
                "coverage_90_std": 0.0,
                "sharpness_std": 0.0,
                "mae_std": 0.0,
            }

        # Get results for selected candidates only
        selected_results = []
        for fold_idx, candidate in self.selected_params.items():
            fold_trials = trials_df[
                (trials_df["outer_fold"] == fold_idx) &
                (trials_df["candidate_id"] == candidate.candidate_id)
            ]
            if not fold_trials.empty:
                # Average across inner folds
                selected_results.append({
                    "coverage_90": fold_trials["coverage_90"].mean(),
                    "sharpness": fold_trials["sharpness"].mean(),
                    "mae": fold_trials["mae"].mean(),
                })

        if not selected_results:
            return {
                "coverage_90_std": 0.0,
                "sharpness_std": 0.0,
                "mae_std": 0.0,
            }

        results_df = pd.DataFrame(selected_results)
        return {
            "coverage_90_std": float(results_df["coverage_90"].std()),
            "sharpness_std": float(results_df["sharpness"].std()),
            "mae_std": float(results_df["mae"].std()),
        }


def _deep_copy_dict(d: dict) -> dict:
    """Deep copy a nested dict without importing copy module."""
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _deep_copy_dict(v)
        elif isinstance(v, list):
            result[k] = v.copy()
        else:
            result[k] = v
    return result
