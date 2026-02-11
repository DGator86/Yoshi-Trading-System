"""Walkforward reliability update helper."""

from __future__ import annotations

from crypto_rfp_hso.projection.reliability import ReliabilityTracker


def walkforward_update(
    tracker: ReliabilityTracker,
    realized: dict[str, float],
    predicted: dict[str, float],
) -> dict[str, float]:
    """Update method reliability using absolute error losses."""
    losses = {}
    for method, pred in predicted.items():
        truth = float(realized.get(method, pred))
        losses[method] = abs(float(pred) - truth)
    tracker.update(losses=losses)
    return tracker.weights()
