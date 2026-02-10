"""Template fitting/bootstrap for class projection."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from crypto_rfp_hso.core.enums import CLASSES
from crypto_rfp_hso.core.math import l2_normalize
from crypto_rfp_hso.features.normalize import DEFAULT_SIGNATURE_KEYS, signature_to_vector


def heuristic_class_label(sig: dict) -> str:
    """Bootstrap label from rule-based heuristics."""
    shock_score = float(sig.get("gap_flag", 0.0) + sig.get("liq_cascade_flag", 0.0) + sig.get("book_vacuum", 0.0))
    sigma = abs(float(sig.get("sigma_k", 0.0)))
    mu = abs(float(sig.get("mu_k", 0.0)))
    u2_liq = float(sig.get("U2_liq", 0.5))
    friction = float(sig.get("friction", 0.5))
    entropy_proxy = 1.0 - abs(float(sig.get("F_liq", 0.0))) * 0.5

    if shock_score >= 1.0:
        return "Shock"
    if mu > sigma * 0.75 and friction < 0.65:
        return "Discovery"
    if u2_liq > 0.65 and mu < sigma * 0.5:
        return "Balanced"
    if u2_liq > 0.55 and friction > 0.55:
        return "Pinning"
    if entropy_proxy > 0.7:
        return "Transitional"
    return "Balanced"


def fit_templates_from_labels(
    signatures: list[dict],
    labels: list[str],
    feature_keys: list[str] | None = None,
) -> dict[str, list[float]]:
    """Fit class templates as normalized class-wise means."""
    if len(signatures) != len(labels):
        raise ValueError("signatures and labels must have equal length")

    keys = feature_keys or DEFAULT_SIGNATURE_KEYS
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for sig, label in zip(signatures, labels):
        grouped[label].append(np.asarray(signature_to_vector(sig, keys=keys), dtype=float))

    dim = len(keys)
    templates: dict[str, list[float]] = {}
    for cls in CLASSES:
        vecs = grouped.get(cls, [])
        if vecs:
            mean_vec = np.mean(np.stack(vecs, axis=0), axis=0)
        else:
            mean_vec = np.zeros(dim, dtype=float)
        templates[cls] = l2_normalize(mean_vec).tolist()
    return templates
