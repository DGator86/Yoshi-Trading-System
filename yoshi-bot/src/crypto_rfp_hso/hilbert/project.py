"""Finite-dimensional Hilbert regime projection."""

from __future__ import annotations

import numpy as np

from crypto_rfp_hso.core.math import entropy_norm, l2_normalize, softmax


def hilbert_project(
    sig_vec: list[float],
    templates: dict[str, list[float]],
    temperature: float,
) -> tuple[dict[str, float], str, float, float]:
    """(w_class, dominant_class, top2_margin, entropy_norm)."""
    if not templates:
        raise ValueError("templates must be non-empty")

    keys = list(templates.keys())
    psi = l2_normalize(sig_vec)
    scores = []
    for key in keys:
        t = l2_normalize(templates[key])
        if psi.size != t.size:
            # Deterministic shape alignment by truncating to common prefix.
            m = min(psi.size, t.size)
            s = float(np.dot(psi[:m], t[:m])) if m > 0 else 0.0
        else:
            s = float(np.dot(psi, t))
        scores.append(s)

    probs = softmax(scores, temperature=temperature)
    w_class = {k: float(p) for k, p in zip(keys, probs)}

    ranked = sorted(w_class.items(), key=lambda kv: kv[1], reverse=True)
    dominant = ranked[0][0]
    top1 = ranked[0][1]
    top2 = ranked[1][1] if len(ranked) > 1 else 0.0
    top2_margin = float(top1 - top2)
    h_norm = entropy_norm(w_class.values())

    return w_class, dominant, top2_margin, float(h_norm)
