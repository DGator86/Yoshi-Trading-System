"""Loop automation module for iterative improvement."""
from gnosis.loop.continuous_learning import (
    DomainSpec,
    ContinuousLearningConfig,
    ContinuousLearningSupervisor,
)

# Ralph loop has broader optional dependencies; keep import lazy/fault-tolerant.
try:
    from gnosis.loop.ralph import RalphLoop, RalphLoopConfig, HparamCandidate
except Exception:  # pragma: no cover - optional dependency surface
    RalphLoop = None
    RalphLoopConfig = None
    HparamCandidate = None

__all__ = [
    "RalphLoop",
    "RalphLoopConfig",
    "HparamCandidate",
    "DomainSpec",
    "ContinuousLearningConfig",
    "ContinuousLearningSupervisor",
]
