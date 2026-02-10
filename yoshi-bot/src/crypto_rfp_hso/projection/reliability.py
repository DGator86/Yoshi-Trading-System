"""Online reliability tracking for projection methods."""

from __future__ import annotations

from crypto_rfp_hso.core.enums import METHODS


class ReliabilityTracker:
    """Track method reliability from walkforward losses."""

    def __init__(self, decay: float = 0.98):
        self.decay = float(decay)
        self._score = {m: 1.0 for m in METHODS}

    def update(self, losses: dict[str, float]) -> None:
        """Update reliability with per-method loss (lower is better)."""
        for m in METHODS:
            loss = float(losses.get(m, 0.0))
            quality = 1.0 / (1.0 + max(loss, 0.0))
            self._score[m] = self.decay * self._score[m] + (1.0 - self.decay) * quality

    def weights(self) -> dict[str, float]:
        """Normalized reliability weights."""
        total = sum(max(v, 0.0) for v in self._score.values())
        if total <= 0.0:
            u = 1.0 / len(METHODS)
            return {m: u for m in METHODS}
        return {m: max(v, 0.0) / total for m, v in self._score.items()}
