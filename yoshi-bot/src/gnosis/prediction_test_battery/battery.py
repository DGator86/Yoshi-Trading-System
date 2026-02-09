from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

from gnosis.prediction_test_battery.context import BatteryContext
from gnosis.prediction_test_battery.reporting import TestBatteryReport
from gnosis.prediction_test_battery.results import TestResult, TestStatus
from gnosis.prediction_test_battery.suites import SUITE_TESTS, iter_suite_tests


def run_suite(context: BatteryContext, suite: str) -> List[TestResult]:
    results: List[TestResult] = []
    for test in iter_suite_tests(suite):
        result = test.run(context)
        results.append(result)
        if suite in {"0", "full"} and result.status == TestStatus.FAIL:
            break
    return results


def run_all_tests(
    context: BatteryContext,
    suite: str = "full",
    report_dir: Optional[Path] = None,
) -> TestBatteryReport:
    context.ensure()
    if suite == "full":
        suites = ["0", "A", "B", "C", "D", "E", "F", "G"]
    else:
        suites = [suite]

    results: List[TestResult] = []
    warnings: List[str] = []
    for s in suites:
        suite_results = run_suite(context, s)
        results.extend(suite_results)
        if s == "0" and any(result.status == TestStatus.FAIL for result in suite_results):
            warnings.append("Suite 0 failure: halting remaining suites.")
            break

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report = TestBatteryReport(run_id=run_id, results=results, warnings=warnings)

    if report_dir is not None:
        report_path = report_dir / f"reports_{run_id}"
        report.write(report_path)

    return report


def available_suites() -> Iterable[str]:
    return SUITE_TESTS.keys()
