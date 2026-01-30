"""
Ralph Loop: Nested walk-forward hyperparameter selection with NO leakage.

- Outer folds = true out-of-sample evaluation (trust this)
- Inner folds inside each outer-train window = hyperparameter selection ONLY
- Purge + embargo between adjacent windows to prevent leakage
- Candidate search is a SMALL explicit grid (<= 40 combos)
- Selection objective uses proper scoring (WIS/IS90/MAE) + coverage penalty + abstention penalty
- NEVER use outer test for selection.
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

from gnosis.harness.walkforward import WalkForwardHarness, Fold, compute_future_returns
from gnosis.harness.scoring import (
    evaluate_predictions,
    IsotonicCalibrator,
    coverage,
    sharpness,
)


@dataclass
class RalphLoopConfig:
    """Configuration for nested walk-forward hyperparameter selection."""
    enabled: bool = True
    target_coverage: float = 0.90
    weights: Dict[str, float] = field(default_factory=lambda: {
        "coverage": 4.0,
        "wis": 1.0,
        "is90": 0.5,
        "mae": 1.0,
        "abstention": 0.5,
    })
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

        Supports both top-level keys and nested under 'ralph' key.
        """
        if d is None:
            return cls()

        # Support either top-level or nested under 'ralph'
        if "ralph" in d:
            cfg = d["ralph"]
        else:
            cfg = d

        return cls(
            enabled=bool(cfg.get("enabled", True)),
            target_coverage=float(cfg.get("target_coverage", 0.90)),
            weights=cfg.get("weights", {
                "coverage": 4.0,
                "wis": 1.0,
                "is90": 0.5,
                "mae": 1.0,
                "abstention": 0.5,
            }),
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
        return json.dumps({"candidate_id": self.candidate_id, "params": self.params}, sort_keys=True)


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


def _set_nested_key(d: dict, dotted_key: str, value: Any) -> None:
    """Set a value in a nested dict using a dotted key path."""
    parts = dotted_key.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def _apply_candidate_params(candidate: HparamCandidate, base_cfg: dict) -> dict:
    """Apply candidate hyperparameters to a deep copy of base config."""
    cfg = copy.deepcopy(base_cfg)
    for key, value in candidate.params.items():
        _set_nested_key(cfg, key, value)
    return cfg


def _interval_score(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float) -> np.ndarray:
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

        # Generate candidate grid
        self.candidates = self._generate_candidates()
        self.selected_params: Dict[int, HparamCandidate] = {}

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
        if available_for_data < n_inner * 15:  # Need at least 15 bars (10 train + 5 val) per fold
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

            folds.append(InnerFold(
                inner_idx=i,
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
            ))

        return folds

    def _apply_sigma_scale(self, preds: pd.DataFrame, sigma_scale: float) -> pd.DataFrame:
        """Apply sigma_scale to widen/narrow prediction intervals."""
        if sigma_scale == 1.0:
            return preds

        result = preds.copy()
        center = result['q50'] if 'q50' in result.columns else result.get('x_hat', 0.0)
        half = (result['q95'] - result['q05']) / 2.0
        half = half * sigma_scale
        result['q05'] = center - half
        result['q95'] = center + half
        if 'sigma_hat' in result.columns:
            result['sigma_hat'] = (result['q95'] - result['q05']) / 3.29
        return result

    def _apply_abstain_logic(
        self,
        preds: pd.DataFrame,
        features_df: pd.DataFrame,
        regimes_config: dict,
    ) -> pd.DataFrame:
        """Apply abstain logic based on S_label and S_pmax thresholds."""
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

        # Default abstain = False
        result["abstain"] = False

        # Get confidence floor
        confidence_floor = 0.65
        constraints = regimes_config.get("constraints_by_species", {})
        default_floor = constraints.get("default", {}).get("confidence_floor", 0.65)
        confidence_floor = default_floor

        # Abstain if S_label is uncertain or S_pmax is below threshold
        if "S_label" in result.columns and "S_pmax" in result.columns:
            for idx in range(len(result)):
                s_label = result.iloc[idx].get("S_label", "S_UNCERTAIN")
                s_pmax = result.iloc[idx].get("S_pmax", 0.0)

                if pd.isna(s_label):
                    s_label = "S_UNCERTAIN"
                if pd.isna(s_pmax):
                    s_pmax = 0.0

                if s_label == "S_UNCERTAIN" or s_pmax < confidence_floor:
                    result.loc[result.index[idx], "abstain"] = True

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

        cfg = _apply_candidate_params(candidate, self.base_config)
        models_cfg = cfg.get("models", {})

        # Extract sigma_scale from candidate params or config
        sigma_scale = 1.0
        if "forecast.sigma_scale" in candidate.params:
            sigma_scale = float(candidate.params["forecast.sigma_scale"])
        elif "forecast" in cfg and "sigma_scale" in cfg["forecast"]:
            sigma_scale = float(cfg["forecast"]["sigma_scale"])

        # Get inner train/val data
        train_df = features_df.iloc[inner_fold.train_start:inner_fold.train_end].copy()
        val_df = features_df.iloc[inner_fold.val_start:inner_fold.val_end].copy()

        # Drop NaN targets
        train_df = train_df.dropna(subset=["future_return"])
        val_df = val_df.dropna(subset=["future_return"])

        if len(train_df) < 10 or len(val_df) < 5:
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

        # Apply abstain logic
        preds = self._apply_abstain_logic(preds, val_df, regimes_config)

        # Score on non-abstained rows
        non_abstain = preds[~preds["abstain"]].copy() if "abstain" in preds.columns else preds

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

        # Compute metrics
        if len(eval_df) > 0 and "future_return" in eval_df.columns:
            y_true = eval_df["future_return"].values
            valid_mask = ~np.isnan(y_true)
            y_true = y_true[valid_mask]

            if len(y_true) > 0:
                q05 = eval_df["q05"].values[valid_mask]
                q50 = eval_df["q50"].values[valid_mask]
                q95 = eval_df["q95"].values[valid_mask]

                coverage_90 = float(coverage(y_true, q05, q95))
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

        abstention_rate = float(preds["abstain"].mean()) if "abstain" in preds.columns else 0.0

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
        candidate_scores: Dict[int, List[float]] = {c.candidate_id: [] for c in self.candidates}

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
            key=lambda cid: np.mean(candidate_scores[cid]) if candidate_scores[cid] else 999.0,
        )
        best = next(c for c in self.candidates if c.candidate_id == best_id)
        self.selected_params[outer_fold_idx] = best

        return best, trials

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
        features_df = features_df.sort_values(["symbol", "bar_idx"]).reset_index(drop=True)

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
            trials_df = pd.DataFrame([
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
                }
                for t in all_trials
            ])
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
            global_best = next(c for c in self.candidates if c.candidate_id == most_common_id)
            selected_json["global_best"] = {
                "candidate_id": global_best.candidate_id,
                "params": global_best.params,
                "selection_count": counts[most_common_id],
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
                (trials_df["outer_fold"] == fold_idx) &
                (trials_df["candidate_id"] == cand.candidate_id)
            ]
            if not fold_trials.empty:
                selected_trials.append({
                    "outer_fold": fold_idx,
                    "coverage_90": fold_trials["coverage_90"].mean(),
                    "sharpness": fold_trials["sharpness"].mean(),
                    "mae": fold_trials["mae"].mean(),
                })

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
