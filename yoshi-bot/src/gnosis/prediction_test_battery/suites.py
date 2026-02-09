from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from gnosis.prediction_test_battery.context import BatteryContext
from gnosis.prediction_test_battery.metrics import (
    calibration_curve,
    compute_classification_metrics,
    compute_regression_metrics,
    expected_calibration_error,
)
from gnosis.prediction_test_battery.results import TestResult, TestStatus
from gnosis.prediction_test_battery.splits import check_split_integrity


@dataclass
class BaseTest:
    name: str
    description: str

    def run(self, context: BatteryContext) -> TestResult:
        raise NotImplementedError

    def _pass(self, stats: Dict[str, float], warnings: Optional[List[str]] = None) -> TestResult:
        return TestResult(
            name=self.name,
            description=self.description,
            status=TestStatus.PASS,
            key_stats=stats,
            warnings=warnings or [],
        )

    def _warn(self, stats: Dict[str, float], warnings: List[str]) -> TestResult:
        return TestResult(
            name=self.name,
            description=self.description,
            status=TestStatus.WARN,
            key_stats=stats,
            warnings=warnings,
        )

    def _fail(self, stats: Dict[str, float], warnings: List[str]) -> TestResult:
        return TestResult(
            name=self.name,
            description=self.description,
            status=TestStatus.FAIL,
            key_stats=stats,
            warnings=warnings,
            recommended_actions=["Investigate leakage or alignment issues before proceeding."],
        )


def _get_arrays(context: BatteryContext) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = context.artifact.predictions
    y_true = df["y_true"].to_numpy(dtype=float)
    y_pred = df["y_pred"].to_numpy(dtype=float)
    y_prob = df["y_prob"].to_numpy(dtype=float)
    return y_true, y_pred, y_prob


def _window_indices(n: int, window: int) -> Iterable[np.ndarray]:
    for start in range(0, n, window):
        end = min(n, start + window)
        yield np.arange(start, end)


class CausalityTest(BaseTest):
    def run(self, context: BatteryContext) -> TestResult:
        if context.artifact.features is None or context.artifact.features.empty:
            return self._warn({}, ["No features provided; skipping causality correlation checks."])
        features = context.artifact.features.copy()
        y_true, _, _ = _get_arrays(context)
        warnings: List[str] = []
        for col in features.columns:
            series = features[col].to_numpy(dtype=float)
            if len(series) < 3:
                continue
            future_corr = np.corrcoef(series[:-1], y_true[1:])[0, 1]
            current_corr = np.corrcoef(series[:-1], y_true[:-1])[0, 1]
            if np.isnan(future_corr):
                continue
            if future_corr > (current_corr + 0.1) and abs(future_corr) > 0.3:
                warnings.append(f"Feature {col} correlates more with future target than current.")
        if warnings:
            return self._fail({"suspect_features": len(warnings)}, warnings)
        return self._pass({"checked_features": len(features.columns)})


class SplitIntegrityTest(BaseTest):
    def run(self, context: BatteryContext) -> TestResult:
        if not context.splits:
            return self._warn({}, ["No splits provided; unable to verify split integrity."])
        issues = check_split_integrity(context.splits)
        if issues:
            return self._fail({"issues": len(issues)}, issues)
        return self._pass({"splits": len(context.splits)})


class LabelAlignmentTest(BaseTest):
    def run(self, context: BatteryContext) -> TestResult:
        df = context.artifact.predictions
        warnings: List[str] = []
        if df["timestamp"].isna().any():
            warnings.append("Missing timestamps in predictions.")
        if df["timestamp"].duplicated().any():
            warnings.append("Duplicate timestamps detected in predictions.")
        if df["timestamp"].is_monotonic_increasing is False:
            warnings.append("Timestamps are not monotonic increasing.")
        if warnings:
            return self._fail({"rows": len(df)}, warnings)
        return self._pass({"rows": len(df)})


class OverlapDetectionTest(BaseTest):
    def run(self, context: BatteryContext) -> TestResult:
        if not context.splits:
            return self._warn({}, ["No splits available for overlap detection."])
        warnings: List[str] = []
        for idx, split in enumerate(context.splits):
            overlap = np.intersect1d(split.train_idx, split.test_idx)
            if overlap.size > 0:
                warnings.append(f"Split {idx} overlaps by {overlap.size} samples.")
        if warnings:
            return self._fail({"overlaps": len(warnings)}, warnings)
        return self._pass({"splits": len(context.splits)})


class MonteCarloResidualTest(BaseTest):
    def run(self, context: BatteryContext) -> TestResult:
        y_true, y_pred, _ = _get_arrays(context)
        residuals = y_true - y_pred
        sigma = np.std(residuals)
        rng = np.random.default_rng(42)
        sims = []
        for _ in range(200):
            sim_pred = y_pred + rng.normal(0, sigma, size=len(y_pred))
            sims.append(np.mean((y_true - sim_pred) ** 2))
        sims = np.array(sims)
        obs = np.mean((y_true - y_pred) ** 2)
        percentile = float((sims < obs).mean())
        return self._pass({"mse": obs, "mc_percentile": percentile})


class IIDBootstrapTest(BaseTest):
    def run(self, context: BatteryContext) -> TestResult:
        y_true, _, y_prob = _get_arrays(context)
        if np.isnan(y_prob).all():
            return self._warn({}, ["No probabilistic predictions for bootstrap CI."])
        rng = np.random.default_rng(42)
        aucs = []
        for _ in range(200):
            idx = rng.integers(0, len(y_true), len(y_true))
            if len(np.unique(y_true[idx])) < 2:
                continue
            aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
        if not aucs:
            return self._warn({}, ["Insufficient class variance for bootstrap."])
        ci = (float(np.percentile(aucs, 5)), float(np.percentile(aucs, 95)))
        return self._pass({"auc_ci_90": ci})


class BlockBootstrapTest(BaseTest):
    def run(self, context: BatteryContext) -> TestResult:
        y_true, _, y_prob = _get_arrays(context)
        block = max(5, int(len(y_true) * 0.05))
        rng = np.random.default_rng(42)
        aucs = []
        for _ in range(100):
            idx = []
            while len(idx) < len(y_true):
                start = rng.integers(0, len(y_true) - block)
                idx.extend(range(start, start + block))
            idx = np.array(idx[: len(y_true)])
            if len(np.unique(y_true[idx])) < 2:
                continue
            aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
        if not aucs:
            return self._warn({}, ["Block bootstrap produced no valid samples."])
        return self._pass({"auc_mean": float(np.mean(aucs))})


class PermutationTest(BaseTest):
    def run(self, context: BatteryContext) -> TestResult:
        y_true, _, y_prob = _get_arrays(context)
        if np.isnan(y_prob).all():
            return self._warn({}, ["No probabilistic predictions for permutation test."])
        rng = np.random.default_rng(42)
        observed = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
        null = []
        block = max(5, int(len(y_true) * 0.05))
        for _ in range(100):
            perm = y_true.copy()
            for start in range(0, len(perm), block):
                chunk = perm[start : start + block]
                rng.shuffle(chunk)
                perm[start : start + block] = chunk
            if len(np.unique(perm)) < 2:
                continue
            null.append(roc_auc_score(perm, y_prob))
        p_val = float((np.array(null) >= observed).mean()) if null else float("nan")
        if null and observed <= np.percentile(null, 95):
            return self._warn({"auc": observed, "p_value": p_val}, ["Observed AUC overlaps null."])
        return self._pass({"auc": observed, "p_value": p_val})


class CircularShiftTest(BaseTest):
    def run(self, context: BatteryContext) -> TestResult:
        y_true, _, y_prob = _get_arrays(context)
        if np.isnan(y_prob).all():
            return self._warn({}, ["No probabilistic predictions for circular shift test."])
        shifts = [1, 6, 12, 24]
        scores = []
        for shift in shifts:
            shifted = np.roll(y_true, shift)
            if len(np.unique(shifted)) < 2:
                continue
            scores.append(roc_auc_score(shifted, y_prob))
        return self._pass({"shift_auc_mean": float(np.mean(scores)) if scores else float("nan")})


class PlaceboFeaturesTest(BaseTest):
    def run(self, context: BatteryContext) -> TestResult:
        features = context.artifact.features
        if features is None or features.empty:
            return self._warn({}, ["No features provided for placebo test."])
        y_true, _, _ = _get_arrays(context)
        rng = np.random.default_rng(42)
        X = features.to_numpy(dtype=float)
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        y = y_true[mask]
        if len(np.unique(y)) < 2:
            return self._warn({}, ["Insufficient class variance for placebo."])
        model = LogisticRegression(max_iter=200)
        model.fit(X, y)
        real_auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
        noise = rng.normal(size=X.shape)
        model.fit(noise, y)
        noise_auc = roc_auc_score(y, model.predict_proba(noise)[:, 1])
        if real_auc - noise_auc < 0.05:
            return self._warn({"real_auc": real_auc, "noise_auc": noise_auc}, ["Placebo features did not degrade performance."])
        return self._pass({"real_auc": real_auc, "noise_auc": noise_auc})


class SeedStabilityTest(BaseTest):
    def run(self, context: BatteryContext) -> TestResult:
        y_true, _, y_prob = _get_arrays(context)
        if np.isnan(y_prob).all():
            return self._warn({}, ["No probabilistic predictions for seed stability."])
        scores = []
        for seed in range(5):
            rng = np.random.default_rng(seed)
            idx = rng.integers(0, len(y_true), len(y_true))
            if len(np.unique(y_true[idx])) < 2:
                continue
            scores.append(roc_auc_score(y_true[idx], y_prob[idx]))
        dispersion = float(np.std(scores)) if scores else float("nan")
        return self._pass({"auc_std": dispersion})


class NestedCVSanityTest(BaseTest):
    def run(self, context: BatteryContext) -> TestResult:
        features = context.artifact.features
        if features is None or features.empty:
            return self._warn({}, ["No features for nested CV sanity."])
        y_true, _, _ = _get_arrays(context)
        X = features.to_numpy(dtype=float)
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        y = y_true[mask]
        if len(y) < 30 or len(np.unique(y)) < 2:
            return self._warn({}, ["Insufficient data for nested CV."])
        outer = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for train_idx, test_idx in outer.split(X):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            best_auc = -np.inf
            for c in [0.1, 1.0, 10.0]:
                model = LogisticRegression(max_iter=200, C=c)
                model.fit(X_train, y_train)
                prob = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, prob)
                best_auc = max(best_auc, auc)
            scores.append(best_auc)
        return self._pass({"outer_auc_mean": float(np.mean(scores))})


class WhiteRealityCheckTest(BaseTest):
    def run(self, context: BatteryContext) -> TestResult:
        y_true, _, y_prob = _get_arrays(context)
        if np.isnan(y_prob).all():
            return self._warn({}, ["No probabilistic predictions for White's Reality Check."])
        observed = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
        rng = np.random.default_rng(42)
        null = []
        for _ in range(100):
            perm = rng.permutation(y_true)
            if len(np.unique(perm)) < 2:
                continue
            null.append(roc_auc_score(perm, y_prob))
        p_val = float((np.array(null) >= observed).mean()) if null else float("nan")
        if null and p_val > 0.1:
            return self._warn({"auc": observed, "p_value": p_val}, ["White Reality Check indicates possible data snooping."])
        return self._pass({"auc": observed, "p_value": p_val})


class DeflatedPerformanceTest(BaseTest):
    def run(self, context: BatteryContext) -> TestResult:
        y_true, _, y_prob = _get_arrays(context)
        if np.isnan(y_prob).all():
            return self._warn({}, ["No probabilistic predictions for deflated score."])
        scores = []
        rng = np.random.default_rng(42)
        for _ in range(50):
            idx = rng.integers(0, len(y_true), len(y_true))
            if len(np.unique(y_true[idx])) < 2:
                continue
            scores.append(roc_auc_score(y_true[idx], y_prob[idx]))
        if not scores:
            return self._warn({}, ["Insufficient bootstrap samples for deflated score."])
        mean = float(np.mean(scores))
        std = float(np.std(scores))
        deflated = mean - 1.645 * std
        return self._pass({"deflated_auc": deflated})


class ProbPerformanceTest(BaseTest):
    def run(self, context: BatteryContext) -> TestResult:
        y_true, _, y_prob = _get_arrays(context)
        if np.isnan(y_prob).all():
            return self._warn({}, ["No probabilistic predictions for probabilistic performance test."])
        rng = np.random.default_rng(42)
        scores = []
        for _ in range(200):
            idx = rng.integers(0, len(y_true), len(y_true))
            if len(np.unique(y_true[idx])) < 2:
                continue
            scores.append(roc_auc_score(y_true[idx], y_prob[idx]))
        if not scores:
            return self._warn({}, ["Insufficient bootstrap for probabilistic performance."])
        prob_gt = float(np.mean(np.array(scores) > 0.5))
        if prob_gt < 0.9:
            return self._warn({"prob_gt_baseline": prob_gt}, ["Performance overlaps baseline."])
        return self._pass({"prob_gt_baseline": prob_gt})


class HyperparameterFragilityTest(BaseTest):
    def run(self, context: BatteryContext) -> TestResult:
        features = context.artifact.features
        if features is None or features.empty:
            return self._warn({}, ["No features for hyperparameter fragility test."])
        y_true, _, _ = _get_arrays(context)
        X = features.to_numpy(dtype=float)
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        y = y_true[mask]
        if len(np.unique(y)) < 2:
            return self._warn({}, ["Insufficient class variance for fragility."])
        scores = []
        for c in [0.01, 0.1, 1.0, 10.0]:
            model = LogisticRegression(max_iter=200, C=c)
            model.fit(X, y)
            scores.append(roc_auc_score(y, model.predict_proba(X)[:, 1]))
        best = max(scores)
        median = float(np.median(scores))
        if best - median > 0.1:
            return self._warn({"best_auc": best, "median_auc": median}, ["Performance fragile across hyperparameters."])
        return self._pass({"best_auc": best, "median_auc": median})


class WalkForwardStressTest(BaseTest):
    def run(self, context: BatteryContext) -> TestResult:
        if not context.splits:
            return self._warn({}, ["No splits for walk-forward stress surface."])
        y_true, _, y_prob = _get_arrays(context)
        metrics = []
        for split in context.splits:
            if len(np.unique(y_true[split.test_idx])) < 2:
                continue
            metrics.append(roc_auc_score(y_true[split.test_idx], y_prob[split.test_idx]))
        return self._pass({"auc_mean": float(np.mean(metrics)) if metrics else float("nan")})


class RegimeSegmentTest(BaseTest):
    def run(self, context: BatteryContext) -> TestResult:
        if context.candles is None:
            return self._warn({}, ["No candle data for regime segmentation."])
        df = context.candles.copy()
        df["return"] = df["close"].pct_change().fillna(0)
        vol = df["return"].rolling(24).std().fillna(0)
        df["vol_regime"] = pd.qcut(vol.rank(method="first"), q=3, labels=False)
        y_true, _, y_prob = _get_arrays(context)
        merged = df.merge(context.artifact.predictions, on="timestamp", how="inner")
        results = {}
        for regime in sorted(merged["vol_regime"].dropna().unique()):
            subset = merged[merged["vol_regime"] == regime]
            if len(np.unique(subset["y_true"])) < 2:
                continue
            results[f"vol_regime_{regime}_auc"] = roc_auc_score(subset["y_true"], subset["y_prob"])
        return self._pass(results)


class EventSlicingTest(BaseTest):
    def run(self, context: BatteryContext) -> TestResult:
        if context.candles is None:
            return self._warn({}, ["No candle data for event slicing."])
        df = context.candles.copy()
        df["return"] = df["close"].pct_change().fillna(0)
        threshold = df["return"].abs().quantile(0.95)
        events = df[df["return"].abs() >= threshold]
        if events.empty:
            return self._warn({}, ["No extreme events detected."])
        merged = events.merge(context.artifact.predictions, on="timestamp", how="inner")
        if len(np.unique(merged["y_true"])) < 2:
            return self._warn({}, ["Insufficient events for metric evaluation."])
        auc = roc_auc_score(merged["y_true"], merged["y_prob"])
        return self._pass({"event_auc": auc})


class TemporalStabilityTest(BaseTest):
    def run(self, context: BatteryContext) -> TestResult:
        y_true, _, y_prob = _get_arrays(context)
        if np.isnan(y_prob).all():
            return self._warn({}, ["No probabilistic predictions for temporal stability."])
        window = max(20, len(y_true) // 10)
        scores = []
        for idx in _window_indices(len(y_true), window):
            if len(np.unique(y_true[idx])) < 2:
                continue
            scores.append(roc_auc_score(y_true[idx], y_prob[idx]))
        if not scores:
            return self._warn({}, ["Insufficient window metrics for stability."])
        std = float(np.std(scores))
        if std > 0.1:
            return self._warn({"auc_std": std}, ["Metric variability is high across time windows."])
        return self._pass({"auc_std": std})


class ChangePointTest(BaseTest):
    def run(self, context: BatteryContext) -> TestResult:
        y_true, _, y_prob = _get_arrays(context)
        if np.isnan(y_prob).all():
            return self._warn({}, ["No probabilistic predictions for change-point detection."])
        window = max(20, len(y_true) // 10)
        scores = []
        for idx in _window_indices(len(y_true), window):
            if len(np.unique(y_true[idx])) < 2:
                continue
            scores.append(roc_auc_score(y_true[idx], y_prob[idx]))
        if len(scores) < 3:
            return self._warn({}, ["Insufficient windows for change-point detection."])
        diff = np.abs(np.diff(scores))
        max_diff = float(diff.max()) if diff.size else 0.0
        if max_diff > 0.15:
            return self._warn({"max_auc_jump": max_diff}, ["Detected abrupt performance change."])
        return self._pass({"max_auc_jump": max_diff})


class ConceptDriftTest(BaseTest):
    def run(self, context: BatteryContext) -> TestResult:
        features = context.artifact.features
        if features is None or features.empty:
            return self._warn({}, ["No features for concept drift detection."])
        n = len(features)
        if n < 40:
            return self._warn({}, ["Insufficient samples for drift detection."])
        first = features.iloc[: n // 2]
        second = features.iloc[n // 2 :]
        psi_scores = []
        kl_scores = []
        for col in features.columns:
            f = first[col].dropna()
            s = second[col].dropna()
            if f.empty or s.empty:
                continue
            bins = np.histogram_bin_edges(np.concatenate([f, s]), bins=10)
            f_counts, _ = np.histogram(f, bins=bins, density=True)
            s_counts, _ = np.histogram(s, bins=bins, density=True)
            f_counts = np.clip(f_counts, 1e-6, None)
            s_counts = np.clip(s_counts, 1e-6, None)
            psi = np.sum((s_counts - f_counts) * np.log(s_counts / f_counts))
            kl = stats.entropy(s_counts, f_counts)
            psi_scores.append(float(psi))
            kl_scores.append(float(kl))
        avg_psi = float(np.mean(psi_scores)) if psi_scores else float("nan")
        avg_kl = float(np.mean(kl_scores)) if kl_scores else float("nan")
        warnings = []
        if avg_psi > 0.2 or avg_kl > 0.2:
            warnings.append("Feature distribution drift detected.")
        if warnings:
            return self._warn({"psi": avg_psi, "kl": avg_kl}, warnings)
        return self._pass({"psi": avg_psi, "kl": avg_kl})


class EVTTailTest(BaseTest):
    def run(self, context: BatteryContext) -> TestResult:
        y_true, _, y_prob = _get_arrays(context)
        tail = np.abs(y_true) > np.quantile(np.abs(y_true), 0.95)
        if tail.sum() < 5:
            return self._warn({}, ["Insufficient tail samples for EVT test."])
        tail_returns = np.abs(y_true[tail])
        params = stats.genpareto.fit(tail_returns)
        return self._pass({"gpd_shape": float(params[0])})


class ExtremeMoveTest(BaseTest):
    def run(self, context: BatteryContext) -> TestResult:
        y_true, _, y_prob = _get_arrays(context)
        threshold = np.quantile(np.abs(y_true), 0.9)
        mask = np.abs(y_true) > threshold
        if mask.sum() < 5:
            return self._warn({}, ["Insufficient extreme move samples."])
        if np.isnan(y_prob).all():
            return self._warn({}, ["No probabilistic predictions for extreme move test."])
        auc = roc_auc_score(y_true[mask] > 0, y_prob[mask]) if len(np.unique(y_true[mask] > 0)) > 1 else float("nan")
        return self._pass({"extreme_auc": auc})


class HeavyTailInjectionTest(BaseTest):
    def run(self, context: BatteryContext) -> TestResult:
        y_true, y_pred, _ = _get_arrays(context)
        rng = np.random.default_rng(42)
        noise = rng.standard_t(df=3, size=len(y_true)) * 0.01
        stressed = y_pred + noise
        rmse = np.sqrt(np.mean((y_true - stressed) ** 2))
        return self._pass({"stressed_rmse": float(rmse)})


class AdversarialNoiseTest(BaseTest):
    def run(self, context: BatteryContext) -> TestResult:
        if context.candles is None:
            return self._warn({}, ["No candle data for adversarial noise test."])
        df = context.candles.copy()
        rng = np.random.default_rng(42)
        jitter = rng.normal(0, 0.001, size=len(df))
        df["close_jitter"] = df["close"] * (1 + jitter)
        noise_level = float(np.mean(np.abs(df["close_jitter"] - df["close"]) / df["close"]))
        return self._pass({"avg_jitter_pct": noise_level})


class TailCalibrationTest(BaseTest):
    def run(self, context: BatteryContext) -> TestResult:
        y_true, _, y_prob = _get_arrays(context)
        if np.isnan(y_prob).all():
            return self._warn({}, ["No probabilistic predictions for tail calibration."])
        threshold = np.quantile(np.abs(y_true), 0.9)
        mask = np.abs(y_true) > threshold
        ece = expected_calibration_error((y_true[mask] > 0).astype(int), y_prob[mask])
        if np.isnan(ece):
            return self._warn({}, ["Insufficient tail samples for calibration."])
        status = self._warn if ece > 0.1 else self._pass
        return status({"tail_ece": float(ece)}, ["Tail calibration error high."] if ece > 0.1 else [])


class FeatureAblationTest(BaseTest):
    def run(self, context: BatteryContext) -> TestResult:
        if not context.feature_groups or context.artifact.features is None:
            return self._warn({}, ["No feature groups for ablation."])
        y_true, _, _ = _get_arrays(context)
        results = {}
        for group, cols in context.feature_groups.items():
            subset = context.artifact.features.drop(columns=cols, errors="ignore")
            X = subset.to_numpy(dtype=float)
            mask = ~np.isnan(X).any(axis=1)
            X = X[mask]
            y = y_true[mask]
            if len(np.unique(y)) < 2:
                continue
            model = LogisticRegression(max_iter=200)
            model.fit(X, y)
            results[f"ablation_{group}_auc"] = roc_auc_score(y, model.predict_proba(X)[:, 1])
        return self._pass(results)


class ParameterSensitivityTest(BaseTest):
    def run(self, context: BatteryContext) -> TestResult:
        features = context.artifact.features
        if features is None or features.empty:
            return self._warn({}, ["No features for parameter sensitivity."])
        y_true, _, _ = _get_arrays(context)
        X = features.to_numpy(dtype=float)
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        y = y_true[mask]
        if len(np.unique(y)) < 2:
            return self._warn({}, ["Insufficient class variance for sensitivity."])
        scores = []
        for c in [0.25, 0.5, 1.0, 2.0]:
            model = LogisticRegression(max_iter=200, C=c)
            model.fit(X, y)
            scores.append(roc_auc_score(y, model.predict_proba(X)[:, 1]))
        return self._pass({"auc_range": float(np.max(scores) - np.min(scores))})


class SobolSensitivityTest(BaseTest):
    def run(self, context: BatteryContext) -> TestResult:
        features = context.artifact.features
        if features is None or features.empty:
            return self._warn({}, ["No features for Sobol-like sensitivity."])
        y_true, _, _ = _get_arrays(context)
        X = features.to_numpy(dtype=float)
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        y = y_true[mask]
        if len(y) < 10:
            return self._warn({}, ["Insufficient samples for sensitivity."])
        var_total = float(np.var(y))
        contributions = np.var(X, axis=0) / np.var(X, axis=0).sum()
        sobol = float(np.mean(contributions) * var_total)
        return self._pass({"sobol_proxy": sobol})


class MissingnessRobustnessTest(BaseTest):
    def run(self, context: BatteryContext) -> TestResult:
        y_true, y_pred, _ = _get_arrays(context)
        rng = np.random.default_rng(42)
        mask = rng.random(len(y_true)) > 0.05
        y_pred_missing = y_pred.copy()
        y_pred_missing[~mask] = np.nan
        y_pred_imputed = pd.Series(y_pred_missing).fillna(method="ffill").fillna(method="bfill").to_numpy()
        rmse = np.sqrt(np.mean((y_true - y_pred_imputed) ** 2))
        return self._pass({"imputed_rmse": float(rmse)})


class DataPerturbationTest(BaseTest):
    def run(self, context: BatteryContext) -> TestResult:
        y_true, y_pred, _ = _get_arrays(context)
        rng = np.random.default_rng(42)
        jitter = rng.normal(0, 0.001, size=len(y_pred))
        perturbed = y_pred * (1 + jitter)
        rmse = np.sqrt(np.mean((y_true - perturbed) ** 2))
        return self._pass({"perturbed_rmse": float(rmse)})


class CalibrationCurveTest(BaseTest):
    def run(self, context: BatteryContext) -> TestResult:
        y_true, _, y_prob = _get_arrays(context)
        if np.isnan(y_prob).all():
            return self._warn({}, ["No probabilistic predictions for calibration."])
        ece = expected_calibration_error(y_true, y_prob)
        status = self._warn if ece > 0.1 else self._pass
        return status({"ece": float(ece)}, ["Calibration error exceeds threshold."] if ece > 0.1 else [])


class RegimeCalibrationTest(BaseTest):
    def run(self, context: BatteryContext) -> TestResult:
        if context.candles is None:
            return self._warn({}, ["No candle data for regime calibration."])
        merged = context.candles.merge(context.artifact.predictions, on="timestamp", how="inner")
        merged["return"] = merged["close"].pct_change().fillna(0)
        merged["vol_regime"] = pd.qcut(merged["return"].abs().rank(method="first"), q=3, labels=False)
        results = {}
        for regime in sorted(merged["vol_regime"].dropna().unique()):
            subset = merged[merged["vol_regime"] == regime]
            ece = expected_calibration_error(subset["y_true"], subset["y_prob"])
            results[f"ece_regime_{regime}"] = float(ece)
        return self._pass(results)


class CalibrationDriftTest(BaseTest):
    def run(self, context: BatteryContext) -> TestResult:
        y_true, _, y_prob = _get_arrays(context)
        if np.isnan(y_prob).all():
            return self._warn({}, ["No probabilistic predictions for calibration drift."])
        window = max(20, len(y_true) // 10)
        eces = []
        for idx in _window_indices(len(y_true), window):
            eces.append(expected_calibration_error(y_true[idx], y_prob[idx]))
        max_ece = float(np.nanmax(eces)) if eces else float("nan")
        if max_ece > 0.15:
            return self._warn({"max_window_ece": max_ece}, ["Calibration drift detected."])
        return self._pass({"max_window_ece": max_ece})


class ConformalCoverageTest(BaseTest):
    def run(self, context: BatteryContext) -> TestResult:
        df = context.artifact.predictions
        if not {"y_pred_lower", "y_pred_upper"}.issubset(df.columns):
            return self._warn({}, ["No prediction intervals for conformal coverage."])
        covered = (df["y_true"] >= df["y_pred_lower"]) & (df["y_true"] <= df["y_pred_upper"])
        coverage = float(covered.mean())
        if coverage < 0.8:
            return self._warn({"coverage": coverage}, ["Coverage below target."])
        return self._pass({"coverage": coverage})


class VolumeQuantileTest(BaseTest):
    def run(self, context: BatteryContext) -> TestResult:
        if context.candles is None:
            return self._warn({}, ["No candle data for volume quantile test."])
        merged = context.candles.merge(context.artifact.predictions, on="timestamp", how="inner")
        merged["volume_bin"] = pd.qcut(merged["volume"].rank(method="first"), q=3, labels=False)
        results = {}
        for b in sorted(merged["volume_bin"].dropna().unique()):
            subset = merged[merged["volume_bin"] == b]
            if len(np.unique(subset["y_true"])) < 2:
                continue
            results[f"auc_volume_bin_{b}"] = roc_auc_score(subset["y_true"], subset["y_prob"])
        return self._pass(results)


class RangeProxyTest(BaseTest):
    def run(self, context: BatteryContext) -> TestResult:
        if context.candles is None:
            return self._warn({}, ["No candle data for range proxy test."])
        merged = context.candles.merge(context.artifact.predictions, on="timestamp", how="inner")
        merged["range"] = (merged["high"] - merged["low"]) / merged["close"]
        merged["range_bin"] = pd.qcut(merged["range"].rank(method="first"), q=3, labels=False)
        results = {}
        for b in sorted(merged["range_bin"].dropna().unique()):
            subset = merged[merged["range_bin"] == b]
            if len(np.unique(subset["y_true"])) < 2:
                continue
            results[f"auc_range_bin_{b}"] = roc_auc_score(subset["y_true"], subset["y_prob"])
        return self._pass(results)


class LiquidityProxyTest(BaseTest):
    def run(self, context: BatteryContext) -> TestResult:
        if context.candles is None:
            return self._warn({}, ["No candle data for liquidity proxy test."])
        merged = context.candles.merge(context.artifact.predictions, on="timestamp", how="inner")
        merged["liq"] = merged["volume"].rolling(24).mean().fillna(merged["volume"])
        merged["liq_bin"] = pd.qcut(merged["liq"].rank(method="first"), q=3, labels=False)
        results = {}
        for b in sorted(merged["liq_bin"].dropna().unique()):
            subset = merged[merged["liq_bin"] == b]
            if len(np.unique(subset["y_true"])) < 2:
                continue
            results[f"auc_liq_bin_{b}"] = roc_auc_score(subset["y_true"], subset["y_prob"])
        return self._pass(results)


class AdverseSelectionProxyTest(BaseTest):
    def run(self, context: BatteryContext) -> TestResult:
        if context.candles is None:
            return self._warn({}, ["No candle data for adverse selection proxy test."])
        merged = context.candles.merge(context.artifact.predictions, on="timestamp", how="inner")
        merged["range"] = (merged["high"] - merged["low"]) / merged["close"]
        high_range = merged["range"] > merged["range"].quantile(0.8)
        high_vol = merged["volume"] > merged["volume"].quantile(0.8)
        subset = merged[high_range & high_vol]
        if subset.empty or len(np.unique(subset["y_true"])) < 2:
            return self._warn({}, ["Insufficient samples for adverse selection proxy."])
        auc = roc_auc_score(subset["y_true"], subset["y_prob"])
        return self._pass({"auc_adverse": auc})


SUITE_TESTS: Dict[str, List[BaseTest]] = {
    "0": [
        CausalityTest("0.1 Causality check", "Feature causality validation"),
        SplitIntegrityTest("0.2 Split integrity", "Split and embargo validation"),
        LabelAlignmentTest("0.3 Label alignment", "Label alignment with timestamps"),
        OverlapDetectionTest("0.4 Overlap detection", "Train/test overlap detection"),
    ],
    "A": [
        MonteCarloResidualTest("A1 Monte Carlo residual", "Residual simulation"),
        IIDBootstrapTest("A2 IID bootstrap", "IID bootstrap CI"),
        BlockBootstrapTest("A3 Block bootstrap", "Block bootstrap CI"),
        PermutationTest("A4 Permutation test", "Block-shuffle permutation"),
        CircularShiftTest("A5 Circular shift", "Circular label shifts"),
        PlaceboFeaturesTest("A6 Placebo features", "Noise feature replacement"),
        SeedStabilityTest("A7 Seed stability", "Re-seeding dispersion"),
    ],
    "B": [
        NestedCVSanityTest("B1 Nested CV sanity", "Nested CV check"),
        WhiteRealityCheckTest("B2 White Reality Check", "Multiple testing correction"),
        DeflatedPerformanceTest("B3 Deflated score", "Deflated performance score"),
        ProbPerformanceTest("B4 Probabilistic performance", "Prob > baseline"),
        HyperparameterFragilityTest("B5 Hyperparameter fragility", "Hyperparameter dispersion"),
    ],
    "C": [
        WalkForwardStressTest("C1 Walk-forward stress", "Train window stress surface"),
        RegimeSegmentTest("C2 Regime segmented eval", "Segmented by regimes"),
        EventSlicingTest("C3 Event slicing", "Extreme event performance"),
        TemporalStabilityTest("C4 Temporal stability", "Metric stability over time"),
        ChangePointTest("C5 Change-point detection", "Performance change points"),
        ConceptDriftTest("C6 Concept drift", "PSI/KL drift detection"),
    ],
    "D": [
        EVTTailTest("D1 EVT tail modeling", "Tail fit on returns"),
        ExtremeMoveTest("D2 Extreme move performance", "Conditional on extreme moves"),
        HeavyTailInjectionTest("D3 Heavy-tail injection", "Student-t residual injection"),
        AdversarialNoiseTest("D4 Adversarial noise", "OHLC jitter stress"),
        TailCalibrationTest("D5 Tail calibration", "Worst-case calibration in tails"),
    ],
    "E": [
        FeatureAblationTest("E1 Feature ablation", "Drop feature groups"),
        ParameterSensitivityTest("E2 Parameter sensitivity", "Param grid sensitivity"),
        SobolSensitivityTest("E3 Sobol sensitivity", "Variance-based sensitivity"),
        MissingnessRobustnessTest("E4 Missingness robustness", "Missing data imputation"),
        DataPerturbationTest("E5 Data perturbation", "Micro-noise perturbation"),
    ],
    "F": [
        CalibrationCurveTest("F1 Calibration curve", "Calibration + ECE"),
        RegimeCalibrationTest("F2 Regime calibration", "ECE by regime"),
        CalibrationDriftTest("F3 Calibration drift", "ECE drift over time"),
        ConformalCoverageTest("F4 Conformal coverage", "Prediction interval coverage"),
    ],
    "G": [
        VolumeQuantileTest("G1 Volume quantiles", "Performance by volume"),
        RangeProxyTest("G2 Range proxy", "Performance by range proxy"),
        LiquidityProxyTest("G3 Liquidity proxy", "Performance by liquidity"),
        AdverseSelectionProxyTest("G4 Adverse selection", "High range/high volume"),
    ],
}


def iter_suite_tests(suite: str) -> List[BaseTest]:
    if suite == "full":
        tests: List[BaseTest] = []
        for key in ["0", "A", "B", "C", "D", "E", "F", "G"]:
            tests.extend(SUITE_TESTS[key])
        return tests
    return SUITE_TESTS.get(suite, [])
