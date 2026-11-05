"""Convex two-model blending with explicit calibration observations."""

from __future__ import annotations

import numpy as np


def convex_weight(actual, first, second, sample_weight=None):
    """Least-squares weight on first, constrained to [0, 1], no intercept.

    Call with earlier out-of-time predictions only, never final validation
    labels. The function has no dependency on any outer experiment artifact.
    """
    actual, first, second = [np.asarray(v, dtype=float) for v in (actual, first, second)]
    if actual.ndim != 1 or first.shape != actual.shape or second.shape != actual.shape or not len(actual):
        raise ValueError("Calibration arrays must be matching nonempty vectors")
    weights = np.ones(len(actual)) if sample_weight is None else np.asarray(sample_weight, float)
    if weights.shape != actual.shape or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Invalid calibration weights")
    if not all(np.isfinite(v).all() for v in (actual, first, second)):
        raise ValueError("Calibration values must be finite")
    difference = first - second
    denominator = float(np.sum(weights * difference ** 2))
    if denominator <= np.finfo(float).eps:
        return 0.5
    return float(np.clip(np.sum(weights * difference * (actual - second)) / denominator, 0.0, 1.0))


def blend(first, second, weight):
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("Blend weight must be in [0, 1]")
    first, second = np.asarray(first, float), np.asarray(second, float)
    if first.shape != second.shape:
        raise ValueError("Prediction shapes differ")
    return weight * first + (1 - weight) * second
