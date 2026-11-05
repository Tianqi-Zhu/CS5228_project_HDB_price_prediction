"""Focused causal representations of the existing, origin-aligned features.

No targets or cross-row statistics are read. The supplied residual baseline
must be the original forecast for each row, never its observed rent.
"""
from __future__ import annotations

from itertools import combinations
import numpy as np
import pandas as pd

from analysis.experiments.calendar_encoding import encoded_features
from analysis.experiments.feature_diversified_ensemble import validate_predictions


VARIANTS = ("cyclical_only", "relative_history", "horizon_dynamics")
LEVEL_COLUMNS = (
    [f"HIST_{g}_{s}" for g in ("SIGNATURE", "BLOCK_FLAT", "AREA_FLAT")
     for s in ("MEAN", "LAST_MEAN")]
    + [f"RECENT_{g}_{w}M_MEAN" for g in ("BLOCK_FLAT", "AREA_FLAT") for w in (3, 6, 12)]
    + [f"RECENT_BLOCK_FLAT_{w}M_SHRUNK_MEAN" for w in (3, 6, 12)]
    + ["TEMP_FLAT_RECENT3_MEAN", "TEMP_FLAT_RECENT12_MEAN",
       "TEMP_FLAT_DAMPED_FORECAST", "TEMP_LOCAL_FORECAST"]
)


def temporal_view(frame, months, baseline, variant):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown temporal view: {variant}")
    baseline = np.asarray(baseline, dtype=float)
    if baseline.shape != (len(frame),) or not np.isfinite(baseline).all():
        raise ValueError("Aligned finite original residual baselines required")
    if variant == "cyclical_only":
        return encoded_features(frame, months, "cyclical")
    result = frame.copy()
    if variant == "relative_history":
        # Remove absolute market level from the input coordinates as well as
        # the unchanged response. Preserve every original NaN and support field.
        for column in LEVEL_COLUMNS:
            result[f"{column}_REL_BASE"] = frame[column].to_numpy(float) - baseline
        return result.drop(columns=LEVEL_COLUMNS)
    horizon = frame.TEMP_FORECAST_HORIZON.to_numpy(float)
    if np.any(np.isfinite(horizon) & (horizon < 1)):
        raise ValueError("Known forecast horizons must be positive")
    logh, decay = np.log1p(horizon), np.sqrt(horizon)
    result["VIEW_HORIZON_LOG1P"] = logh
    result["VIEW_TYPE_MOMENTUM_X_LOGH"] = (
        frame.TEMP_FLAT_RECENT3_MEAN - frame.TEMP_FLAT_RECENT12_MEAN) * logh
    result["VIEW_LOCAL_PREMIUM_DECAY"] = (
        frame.TEMP_LOCAL_FORECAST - frame.TEMP_FLAT_DAMPED_FORECAST) / decay
    result["VIEW_BLOCK_PREMIUM_DECAY"] = (
        frame.RECENT_BLOCK_FLAT_3M_SHRUNK_MEAN - frame.RECENT_AREA_FLAT_3M_MEAN) / decay
    result["VIEW_AREA_MOMENTUM_DECAY"] = frame.AREA_RECENT_MINUS_LONG / decay
    result["VIEW_BLOCK_SUPPORT_DECAY"] = frame.RECENT_BLOCK_FLAT_3M_COUNT / horizon
    return result


def simplex_weights(actual, predictions, sample_weight=None):
    """Exact active-face convex MSE solution for up to five predictors."""
    p = validate_predictions(predictions)
    y = np.asarray(actual, dtype=float)
    v = np.ones(len(p)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if not 1 <= p.shape[1] <= 5 or y.shape != (len(p),) or v.shape != y.shape:
        raise ValueError("Aligned targets/weights and one to five predictors required")
    if not np.isfinite(y).all() or not np.isfinite(v).all() or (v <= 0).any():
        raise ValueError("Finite targets and strictly positive weights required")
    v = v / v.sum()
    error = p - y[:, None]
    scale = max(float(np.sqrt(np.sum(v[:, None] * error**2) / p.shape[1])), 1.)
    q = (error / scale).T @ (v[:, None] * (error / scale))
    candidates = []
    for size in range(1, p.shape[1] + 1):
        for subset in combinations(range(p.shape[1]), size):
            face = q[np.ix_(subset, subset)]
            system = np.block([[face, np.ones((size, 1))], [np.ones((1, size)), np.zeros((1, 1))]])
            z = np.linalg.lstsq(system, np.r_[np.zeros(size), 1.], rcond=None)[0][:size]
            if (z < -1e-9).any() or not np.isclose(z.sum(), 1., atol=1e-8):
                continue
            w = np.zeros(p.shape[1]); w[list(subset)] = np.maximum(z, 0); w /= w.sum()
            candidates.append((float(np.sum(v * (error @ w)**2)), w))
    best = min(loss for loss, _ in candidates)
    tied = [(loss, w) for loss, w in candidates if loss <= best + max(1e-9, best * 1e-13)]
    loss, w = min(tied, key=lambda item: (-item[1][0], np.count_nonzero(item[1]), tuple(-item[1])))
    return w, {"weighted_mse_sgd2": loss, "active_members": int(np.count_nonzero(w)),
               "solver": "Exact active-face convex MSE; nonnegative sum-to-one; no intercept"}
