"""Label-free calendar encodings with complete coverage from the first month.

Intake windows are recurring monthly proxies, not exact school-term dates.
No observed future rents, economic values, or future calendar announcements
are required. Existing rent histories and residual baselines remain untouched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


MONTH_COLUMN = "APPROVAL_MONTH_OF_YEAR"
TIME_COLUMN = "APPROVAL_MONTH_INDEX"
ENCODINGS = ("integer", "categorical", "cyclical")
CALENDAR_COLUMNS = [
    "CAL_PRIMARY_START_PROXY", "CAL_PRIMARY_PRESTART_1M", "CAL_PRIMARY_PRESTART_2M",
    "CAL_UNIVERSITY_START_PROXY", "CAL_UNIVERSITY_PRESTART_1M", "CAL_UNIVERSITY_PRESTART_2M",
]
INTERACTION_COLUMNS = [
    "CAL_PRIMARY_START_X_SCHOOL_ACCESS", "CAL_PRIMARY_PRESTART_X_SCHOOL_ACCESS",
    "CAL_PRIMARY_START_X_SCHOOL_COUNT", "CAL_PRIMARY_PRESTART_X_SCHOOL_COUNT",
    "CAL_SCHOOL_LOCATION_MISSING",
]


def encoded_features(frame, months, encoding, *, calendar=False, interactions=False):
    if encoding not in ENCODINGS:
        raise ValueError(f"Unknown month encoding: {encoding}")
    if interactions and not calendar:
        raise ValueError("Interactions require calendar main effects")
    periods = pd.PeriodIndex(months, freq="M")
    if len(periods) != len(frame) or periods.isna().any():
        raise ValueError("Calendar months must be complete and aligned with rows")
    month = np.asarray(periods.month)
    np.testing.assert_array_equal(frame[MONTH_COLUMN].to_numpy(), month)
    np.testing.assert_array_equal(frame[TIME_COLUMN].to_numpy(), periods.asi8 - pd.Period("2021-01", "M").ordinal)
    result = frame.copy()
    if encoding == "categorical":
        result[MONTH_COLUMN] = pd.Categorical(month, categories=range(1, 13))
    elif encoding == "cyclical":
        result = result.drop(columns=MONTH_COLUMN)
        angle = 2 * np.pi * (month - 1) / 12
        result["CAL_MONTH_SIN"] = np.sin(angle)
        result["CAL_MONTH_COS"] = np.cos(angle)
    if calendar:
        # Primary-school annual opening proxy: January. University semester
        # opening proxies: January/August. Lead windows allow approval month
        # to precede school opening, without claiming actual move-in dates.
        for name, values in zip(CALENDAR_COLUMNS, [(1,), (12,), (11,), (1, 8), (12, 7), (11, 6)], strict=True):
            result[name] = np.isin(month, values).astype(float)
    if interactions:
        distance = frame.NEAREST_SCHOOL_KM.to_numpy(float)
        count = frame["SCHOOL_COUNT_1.0KM"].to_numpy(float)
        missing = ~np.isfinite(distance) | ~np.isfinite(count)
        access = np.where(np.isfinite(distance), 1 / (1 + np.maximum(distance, 0)), 0.0)
        count = np.where(np.isfinite(count), np.maximum(count, 0), 0.0)
        start = result.CAL_PRIMARY_START_PROXY.to_numpy()
        pre = result.CAL_PRIMARY_PRESTART_1M.to_numpy() + result.CAL_PRIMARY_PRESTART_2M.to_numpy()
        for name, value in zip(INTERACTION_COLUMNS, [start * access, pre * access, start * count, pre * count, missing.astype(float)], strict=True):
            result[name] = value
    changed = set(result.columns) - set(frame.columns)
    if changed and (result[list(changed)].isna().any().any() or not np.isfinite(result[list(changed)].to_numpy(float)).all()):
        raise AssertionError("New calendar features must be finite for every row")
    untouched = [c for c in frame if c != MONTH_COLUMN]
    pd.testing.assert_frame_equal(result[untouched], frame[untouched])
    return result


def stratified_queries(months, row_ids, per_month=200, seed=42):
    """Fixed, label-independent monthly query sample; full context is retained."""
    periods = pd.PeriodIndex(months, freq="M")
    rng = np.random.default_rng(seed)
    selected, weights = [], []
    for month in periods.unique().sort_values():
        positions = np.flatnonzero(periods == month)
        chosen = np.sort(rng.choice(positions, size=min(per_month, len(positions)), replace=False))
        selected.extend(chosen)
        weights.extend([len(positions) / len(chosen)] * len(chosen))
    order = np.argsort(selected)
    selected, weights = np.asarray(selected)[order], np.asarray(weights)[order]
    if len(np.unique(np.asarray(row_ids)[selected])) != len(selected):
        raise ValueError("Repeated query row IDs")
    return selected, weights
