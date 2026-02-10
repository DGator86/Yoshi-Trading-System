"""Entropy helpers for regime uncertainty."""

from __future__ import annotations

from crypto_rfp_hso.core.math import entropy_norm


def normalized_entropy(probabilities: dict[str, float], eps: float = 1e-9) -> float:
    """Normalized entropy of a probability dict."""
    return entropy_norm(probabilities.values(), eps=eps)
