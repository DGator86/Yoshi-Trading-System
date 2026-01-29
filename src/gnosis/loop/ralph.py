"""
Ralph Loop: Nested walk-forward hyperparameter selection (NO leakage)
REAL FIX VERSION:

- Outer/inner folds are defined in TRADE index (prints rows), invariant to domains.D0.n_trades.
- Candidate evaluation *rebuilds the pipeline* inside each inner fold:
  prints -> bars (with candidate domains.D0.n_trades) -> features -> KPCOFGS -> particle state -> targets
- Candidate selection score uses ONLY inner validation bars (never outer test).

This makes upstream hparams (domains_D0_n_trades, particle_flow_span) *actually* honored.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from collections import Counter
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from gnosis.harness.trade_walkforward import TradeWalkForwardHarness, TradeFold
from gnosis.harness import WalkForwardHarness  # kept for compatibility elsewhere
from gnosis.domains import compute_features
from gnosis.domains import DomainAggregator
from gnosis.regimes.kpcofgs import KPCOFGSClassifier
from gnosis.particle import ParticleState
from gnosis.harness import compute_future_returns
from gnosis.harness.scoring import (
    evaluate_predictions,
    IsotonicCalibrator,
    compute_ece,
    compute_stability_metrics,
)
from gnosis.predictors import QuantilePredictor
from gnosis.baseline import BaselinePredictor
from gnosis.regimes.constraints import apply_abstain_logic


@dataclass(frozen=True)
class HparamCandidate:
    candidate_id: int
    params: dict

    def to_json(self) -> str:
        return json.dumps({"candidate_id": self.candidate_id, "params": self.params}, sort_keys=True)


@dataclass(frozen=True)
class InnerFold:
    inner_idx: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    outer_fold: int


@dataclass(frozen=True)
class TrialResult:
    outer_fold: int
    candidate_id: int
    inner_fold: int
    coverage_90: float
    sharpness: float
    mae: float
    abstention_rate: float
    composite_score: float
    params_json: str


@dataclass
class RalphLoopConfig:
    # How many inner folds inside each outer-train window
    inner_folds_n: int = 3
    # Inner train/val ratios are expressed in trades
    inner_train_ratio: float = 0.6
    inner_val_ratio: float = 0.4

    # Purge/embargo in trades (defaults computed from horizon_trades in runner)
    inner_purge_trades: int = 0
    inner_embargo_trades: int = 0

    # Composite score weights
    w1_coverage: float = 0.6
    w2_sharpness: float = 0.2
    w3_mae: float = 0.2

    # Target coverage for 90% interval
    target_coverage: float = 0.90


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


class RalphLoop:
    def __init__(self, base_config: dict, hparams_config: dict):
        self.base_config = base_config
        self.hparams_config = hparams_config or {}

        # loop config
        inner_cfg = self.hparams_config.get("inner_folds", {})
        self.loop_config = RalphLoopConfig(
            inner_folds_n=int(inner_cfg.get("n_folds", 3)),
            inner_train_ratio=float(inner_cfg.get("train_ratio", 0.6)),
            inner_val_ratio=float(inner_cfg.get("val_ratio", 0.4)),
            inner_purge_trades=int(self.hparams_config.get("inner_purge_trades", 0)),
            inner_embargo_trades=int(self.hparams_config.get("inner_embargo_trades", 0)),
            w1_coverage=float(self.hparams_config.get("w1_coverage", 0.6)),
            w2_sharpness=float(self.hparams_config.get("w2_sharpness", 0.2)),
            w3_mae=float(self.hparams_config.get("w3_mae", 0.2)),
            target_coverage=float(self.hparams_config.get("target_coverage", 0.90)),
        )

        self.candidates = self._generate_candidates()
        self.selected_params: Dict[int, HparamCandidate] = {}

    def _generate_candidates(self) -> List[HparamCandidate]:
        grid = self.hparams_config.get("grid", {})
        if not grid:
            return [HparamCandidate(candidate_id=0, params={})]

        keys = list(grid.keys())
        vals = [grid[k] for k in keys]

        combos = list(itertools.product(*vals))
        candidates: List[HparamCandidate] = []
        for i, combo in enumerate(combos):
            params = {k: combo[j] for j, k in enumerate(keys)}
            candidates.append(HparamCandidate(candidate_id=i, params=params))
        return candidates

    def _apply_candidate_params(self, candidate: HparamCandidate, base_cfg: dict) -> dict:
        cfg = json.loads(json.dumps(base_cfg))  # deep copy
        for key, value in candidate.params.items():
            # keys in hparams.yaml are flat; we map some known ones
            if key == "domains_D0_n_trades":
                cfg.setdefault("domains", {}).setdefault("D0", {})["n_trades"] = int(value)
            elif key == "particle_flow_span":
                cfg.setdefault("particle", {}).setdefault("flow", {})["span"] = int(value)
            elif key == "predictor_l2_reg":
                cfg.setdefault("models", {}).setdefault("predictor", {})["l2_reg"] = float(value)
            elif key == "confidence_floor_scale":
                cfg.setdefault("regimes", {})["confidence_floor_scale"] = float(value)
            else:
                # generic dotted path support, e.g. "models.predictor.l2_reg"
                if "." in key:
                    parts = key.split(".")
                    d = cfg
                    for p in parts[:-1]:
                        d = d.setdefault(p, {})
                    d[parts[-1]] = value
                else:
                    cfg[key] = value
        return cfg

    def _build_features_from_prints(self, prints_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
        # 1) aggregate prints -> bars
        agg = DomainAggregator(cfg)
        bars = agg.aggregate(prints_df, "D0")

        # 2) features
        feats = compute_features(bars)

        # 3) regimes
        regimes_cfg = cfg.get("regimes", {})
        classifier = KPCOFGSClassifier(regimes_cfg)
        feats = classifier.classify(feats)

        # 4) particle state (may use particle.flow.span)
        feats = ParticleState(feats, cfg.get("particle", {}))

        # 5) targets (future_return)
        horizon_bars = int(cfg.get("targets", {}).get("horizon_bars", 10))
        feats = compute_future_returns(feats, horizon_bars=horizon_bars)

        return feats

    def _compute_composite_score(self, coverage_90: float, sharpness: float, mae: float) -> float:
        # higher better
        coverage_error = abs(coverage_90 - self.loop_config.target_coverage)
        coverage_score = 1.0 - coverage_error
        score = (
            self.loop_config.w1_coverage * coverage_score
            + self.loop_config.w2_sharpness * (1.0 - sharpness)  # narrower is better
            + self.loop_config.w3_mae * (1.0 - mae)              # lower is better
        )
        return float(score)

    def _generate_inner_folds(self, outer_train_start: int, outer_train_end: int) -> List[InnerFold]:
        n_trades = outer_train_end - outer_train_start
        n_inner = int(self.loop_config.inner_folds_n)
        if n_trades < 100 or n_inner <= 0:
            return []

        # inner sizes in trades
        train_size = int(n_trades * self.loop_config.inner_train_ratio)
        val_size = int(n_trades * self.loop_config.inner_val_ratio)

        purge = int(self.loop_config.inner_purge_trades)

        # ensure usable
        window = train_size + purge + val_size
        if window >= n_trades:
            # fallback: shrink val
            val_size = max(1, n_trades - train_size - purge - 1)
            window = train_size + purge + val_size
            if window >= n_trades:
                return []

        remaining = n_trades - window
        step = 0 if n_inner <= 1 else max(1, remaining // (n_inner - 1))

        folds = []
        for i in range(n_inner):
            offset = i * step
            train_start = outer_train_start + offset
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
                    outer_fold=-1,
                )
            )
        return folds

    def _evaluate_candidate_on_inner(
        self,
        candidate: HparamCandidate,
        prints_df: pd.DataFrame,
        inner_fold: InnerFold,
        outer_fold_idx: int,
    ) -> TrialResult:
        cfg = self._apply_candidate_params(candidate, self.base_config)
        regimes_cfg = cfg.get("regimes", {})
        models_cfg = cfg.get("models", {})

        # Build a local window from train_start to val_end (inclusive)
        window_prints = prints_df.iloc[inner_fold.train_start:inner_fold.val_end].copy()
        if len(window_prints) < 1000:
            return TrialResult(
                outer_fold=outer_fold_idx,
                candidate_id=candidate.candidate_id,
                inner_fold=inner_fold.inner_idx,
                coverage_90=0.0,
                sharpness=1.0,
                mae=1.0,
                abstention_rate=1.0,
                composite_score=-999.0,
                params_json=candidate.to_json(),
            )

        feats = self._build_features_from_prints(window_prints, cfg)

        # Split into train/val BAR segments based on trades_per_bar in this candidate
        tpb = int(cfg.get("domains", {}).get("D0", {}).get("n_trades", 200))
        tpb = max(1, tpb)

        train_trades = inner_fold.train_end - inner_fold.train_start
        purge_trades = inner_fold.val_start - inner_fold.train_end
        val_trades = inner_fold.val_end - inner_fold.val_start

        train_bars = train_trades // tpb
        purge_bars = _ceil_div(purge_trades, tpb)
        val_bars = val_trades // tpb

        if train_bars < 10 or val_bars < 5:
            return TrialResult(
                outer_fold=outer_fold_idx,
                candidate_id=candidate.candidate_id,
                inner_fold=inner_fold.inner_idx,
                coverage_90=0.0,
                sharpness=1.0,
                mae=1.0,
                abstention_rate=1.0,
                composite_score=-999.0,
                params_json=candidate.to_json(),
            )

        train_df = feats.iloc[:train_bars].copy()
        val_df = feats.iloc[train_bars + purge_bars: train_bars + purge_bars + val_bars].copy()

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
                abstention_rate=1.0,
                composite_score=-999.0,
                params_json=candidate.to_json(),
            )

        predictor = QuantilePredictor(models_cfg)
        predictor.fit(train_df, "future_return")
        preds = predictor.predict(val_df)

        # calibration (same approach as Phase C)
        calibrator = IsotonicCalibrator(n_bins=10)
        train_s = train_df["S_pmax"].values
        train_labels = train_df["S_label"].values
        shifted = np.roll(train_labels, -1)
        shifted[-1] = train_labels[-1]
        outcomes = (train_labels == shifted).astype(float)
        calibrator.fit(train_s, outcomes)

        val_s_raw = val_df["S_pmax"].values.copy()
        val_s_cal = calibrator.calibrate(val_s_raw)
        val_df = val_df.copy()
        val_df["S_pmax_calibrated"] = val_s_cal

        preds = apply_abstain_logic(preds, val_df, regimes_cfg)

        # evaluate (exclude abstain)
        non_abstain = preds[~preds["abstain"]].copy() if "abstain" in preds.columns else preds
        metrics = evaluate_predictions(non_abstain if len(non_abstain) > 0 else preds, val_df, "future_return")

        abstention_rate = float(preds["abstain"].mean()) if "abstain" in preds.columns else 0.0
        score = self._compute_composite_score(
            coverage_90=float(metrics.get("coverage_90", 0.0)),
            sharpness=float(metrics.get("sharpness", 1.0)),
            mae=float(metrics.get("mae", 1.0)),
        )

        return TrialResult(
            outer_fold=outer_fold_idx,
            candidate_id=candidate.candidate_id,
            inner_fold=inner_fold.inner_idx,
            coverage_90=float(metrics.get("coverage_90", 0.0)),
            sharpness=float(metrics.get("sharpness", 1.0)),
            mae=float(metrics.get("mae", 1.0)),
            abstention_rate=abstention_rate,
            composite_score=score,
            params_json=candidate.to_json(),
        )

    def select_best_for_outer_fold(self, prints_df: pd.DataFrame, outer_fold_idx: int, outer_train_start: int, outer_train_end: int) -> HparamCandidate:
        inner_folds = self._generate_inner_folds(outer_train_start, outer_train_end)
        if not inner_folds:
            return self.candidates[0] if self.candidates else HparamCandidate(0, {})

        candidate_scores: Dict[int, List[float]] = {c.candidate_id: [] for c in self.candidates}
        for cand in self.candidates:
            for inner in inner_folds:
                r = self._evaluate_candidate_on_inner(cand, prints_df, inner, outer_fold_idx)
                candidate_scores[cand.candidate_id].append(r.composite_score)

        best_id = max(candidate_scores.keys(), key=lambda cid: np.mean(candidate_scores[cid]) if candidate_scores[cid] else -999)
        best = next(c for c in self.candidates if c.candidate_id == best_id)
        self.selected_params[outer_fold_idx] = best
        return best

    def run(self, prints_df: pd.DataFrame, outer_harness: TradeWalkForwardHarness) -> Tuple[pd.DataFrame, dict]:
        outer_folds = list(outer_harness.generate_folds(len(prints_df)))

        trials: List[TrialResult] = []

        for fold in outer_folds:
            print(f"  Ralph Loop: Processing outer fold {fold.fold_idx}...")
            best = self.select_best_for_outer_fold(
                prints_df=prints_df,
                outer_fold_idx=fold.fold_idx,
                outer_train_start=fold.train_start,
                outer_train_end=fold.train_end,
            )
            print(f"    Selected candidate {best.candidate_id}: {best.params}")

            # For audit: also store the inner trials for the selected fold/candidates if desired later
            # (We keep this simple: we only store per-candidate inner scores already computed above by re-running quickly)
            inner_folds = self._generate_inner_folds(fold.train_start, fold.train_end)
            for cand in self.candidates:
                for inner in inner_folds:
                    r = self._evaluate_candidate_on_inner(cand, prints_df, inner, fold.fold_idx)
                    trials.append(r)

        trials_df = pd.DataFrame([t.__dict__ for t in trials])

        selected_json = {"per_fold": {}, "global_best": {}}
        for fold_idx, cand in self.selected_params.items():
            selected_json["per_fold"][str(fold_idx)] = {"candidate_id": cand.candidate_id, "params": cand.params}

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
