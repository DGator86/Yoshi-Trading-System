"""Loop automation module for iterative improvement."""
from gnosis.loop.ralph import RalphLoop, RalphLoopConfig, HparamCandidate
from gnosis.loop.continuous_learning import (
    DomainSpec,
    ContinuousLearningConfig,
    ContinuousLearningSupervisor,
)

__all__ = [
    "RalphLoop",
    "RalphLoopConfig",
    "HparamCandidate",
    "DomainSpec",
    "ContinuousLearningConfig",
    "ContinuousLearningSupervisor",
]
