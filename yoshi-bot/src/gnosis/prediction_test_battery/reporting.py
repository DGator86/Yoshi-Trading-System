from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from gnosis.prediction_test_battery.results import TestResult, TestStatus


@dataclass
class TestBatteryReport:
    run_id: str
    results: List[TestResult]
    warnings: List[str]

    def scorecard(self) -> List[Dict[str, str]]:
        return [
            {"name": result.name, "status": result.status.value, "description": result.description}
            for result in self.results
        ]

    def summary_markdown(self) -> str:
        lines = [f"# Test Battery Report ({self.run_id})", "", "## Scorecard"]
        lines.append("| Test | Status | Description |")
        lines.append("| --- | --- | --- |")
        for row in self.scorecard():
            lines.append(f"| {row['name']} | {row['status']} | {row['description']} |")
        if self.warnings:
            lines.append("\n## Warnings")
            lines.extend([f"- {warning}" for warning in self.warnings])
        return "\n".join(lines)

    def to_json(self) -> Dict[str, object]:
        return {
            "run_id": self.run_id,
            "results": [result.to_dict() for result in self.results],
            "warnings": self.warnings,
        }

    def write(self, base_path: Path) -> Path:
        base_path.mkdir(parents=True, exist_ok=True)
        summary_path = base_path / "summary.md"
        results_path = base_path / "results.json"
        scorecard_path = base_path / "scorecard.csv"

        summary_path.write_text(self.summary_markdown(), encoding="utf-8")
        results_path.write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")
        scorecard_path.write_text(
            "name,status,description\n" + "\n".join(
                [f"{row['name']},{row['status']},{row['description']}" for row in self.scorecard()]
            ),
            encoding="utf-8",
        )
        return base_path


def combine_status(results: List[TestResult]) -> TestStatus:
    if any(result.status == TestStatus.FAIL for result in results):
        return TestStatus.FAIL
    if any(result.status == TestStatus.WARN for result in results):
        return TestStatus.WARN
    return TestStatus.PASS
