import pandas as pd

from gnosis.prediction_test_battery.battery import run_all_tests
from gnosis.prediction_test_battery.context import BatteryContext
from gnosis.prediction_test_battery.data import generate_synthetic_data
from gnosis.prediction_test_battery.results import TestResult
from gnosis.prediction_test_battery.suites import SUITE_TESTS
from gnosis.prediction_test_battery.splits import build_walk_forward_splits


def _build_context() -> BatteryContext:
    candles, artifact = generate_synthetic_data(n=120)
    splits = build_walk_forward_splits(n_samples=len(artifact.predictions), train_size=60, test_size=20, step_size=20)
    return BatteryContext(artifact=artifact, candles=candles, splits=splits)


def test_suite_results_schema():
    context = _build_context()
    for suite in SUITE_TESTS.keys():
        report = run_all_tests(context, suite=suite)
        assert all(isinstance(result, TestResult) for result in report.results)


def test_suite_zero_fail_fast():
    context = _build_context()
    context.artifact.predictions.loc[0, "timestamp"] = pd.NaT
    report = run_all_tests(context, suite="0")
    assert report.results[0].status.value in {"FAIL", "WARN"}


def test_end_to_end_integration():
    context = _build_context()
    report = run_all_tests(context, suite="full")
    assert report.results
