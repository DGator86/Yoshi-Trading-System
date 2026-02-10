"""Math utilities for deterministic probabilistic modeling."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def safe_div(numerator: float, denominator: float, eps: float = 1e-12) -> float:
    """Safely divide with epsilon."""
    return float(numerator) / float(denominator + eps)


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    x = float(np.clip(x, -60.0, 60.0))
    return 1.0 / (1.0 + math.exp(-x))


def softmax(values: Iterable[float], temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax with temperature."""
    v = np.asarray(list(values), dtype=float)
    if v.size == 0:
        return v
    temp = max(float(temperature), 1e-9)
    z = v * temp
    z = z - np.max(z)
    e = np.exp(np.clip(z, -60.0, 60.0))
    s = np.sum(e)
    if s <= 0.0 or not np.isfinite(s):
        return np.ones_like(v) / float(v.size)
    return e / s


def entropy_norm(probs: Iterable[float], eps: float = 1e-9) -> float:
    """Normalized Shannon entropy in [0, 1]."""
    p = np.asarray(list(probs), dtype=float)
    if p.size <= 1:
        return 0.0
    p = np.clip(p, eps, 1.0)
    p = p / np.sum(p)
    ent = -float(np.sum(p * np.log(p)))
    return ent / math.log(float(p.size))


def l2_normalize(vec: Iterable[float], eps: float = 1e-12) -> np.ndarray:
    """Normalize a vector to unit norm."""
    x = np.asarray(list(vec), dtype=float)
    norm = float(np.linalg.norm(x))
    if norm <= eps:
        if x.size == 0:
            return x
        return np.zeros_like(x)
    return x / norm


def zscore_current(values: Iterable[float], x: float, eps: float = 1e-12) -> float:
    """Z-score of x against values."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size < 2:
        return 0.0
    mu = float(np.mean(arr))
    sd = float(np.std(arr))
    if sd <= eps or not np.isfinite(sd):
        return 0.0
    return (float(x) - mu) / sd


def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp x to [lo, hi]."""
    return float(np.clip(float(x), float(lo), float(hi)))
