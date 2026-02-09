from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TestStatus(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass
class TestResult:
    name: str
    status: TestStatus
    description: str
    key_stats: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    plots: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "description": self.description,
            "key_stats": self.key_stats,
            "warnings": self.warnings,
            "recommended_actions": self.recommended_actions,
            "plots": self.plots or [],
        }
