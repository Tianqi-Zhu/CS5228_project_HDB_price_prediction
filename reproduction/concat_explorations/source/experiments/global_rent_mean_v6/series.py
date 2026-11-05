"""Full-market monthly nominal rent mean, as a causal per-origin feature bank.

`M_t` is the mean `MONTHLY_RENT` over **all** observed transactions in calendar month `t`,
in SGD per rented unit. It is deliberately not rent per square metre and not a cumulative
average: the point of this bank is the global nominal level, which the frozen history block
does not carry (its `hist_rent__global` is cumulative, and the v5 recent features are
log per-sqm).

A month with no transactions is missing, never zero.

## What this is not

Raw nominal rent moves with flat-type, floor-area and location mix, so `M_t` is a **raw
market proxy**, not a composition-adjusted index. A correlation computed over the whole
timeline is a property of that proxy, not evidence that an individual dwelling's rent
becomes predictable. Nothing here extrapolates a slope forward.

## Windowing choice

The multi-month statistics are the **equal-weight mean of the monthly means**, not a pool of
all transactions across those months. The two differ whenever monthly volumes differ, and
the pooled version would silently weight busy months more. This module always uses the
monthly-mean-then-average form, and the feature names say so.
"""

from __future__ import annotations

import numpy as np

WINDOWS = (3, 6, 12)


class GlobalRentError(ValueError):
    pass


def build_series(frame, *, rent_col: str, month_col: str, fit_end: int):
    """Monthly mean nominal rent and transaction count, months 0..fit_end.

    `frame` must already exclude months after `fit_end`; a later month raises rather than
    being dropped silently, because a silently truncated panel is a leak waiting to happen.
    """
    months = frame[month_col].to_numpy(dtype=np.int64)
    if months.size and int(months.max()) > fit_end:
        raise GlobalRentError(
            f"frame contains month {int(months.max())} > fit_end {fit_end}")
    rent = frame[rent_col].to_numpy(dtype=np.float64)

    n_months = int(fit_end) + 1
    counts = np.zeros(n_months)
    sums = np.zeros(n_months)
    np.add.at(counts, months, 1.0)
    np.add.at(sums, months, rent)

    means = np.full(n_months, np.nan)
    observed = counts > 0
    means[observed] = sums[observed] / counts[observed]
    return means, counts


def feature_names() -> list[str]:
    out = ["grm_latest_log", "grm_latest_age", "grm_dlog_1"]
    for w in WINDOWS:
        out.append(f"grm_mean{w}_log")
        out.append(f"grm_valid{w}")
    out.append("grm_log_last3_over_prior3")
    out.append("grm_count_latest")
    out.append("grm_count_mean12")
    return out


def features_at(means: np.ndarray, counts: np.ndarray, origin: int) -> np.ndarray:
    """Feature vector using only months <= origin. No backward fill, no future month."""
    if origin >= len(means):
        raise GlobalRentError(f"origin {origin} beyond the built series")

    def month_mean(month: int) -> float:
        if month < 0 or month >= len(means):
            return float("nan")
        return float(means[month])

    def window_mean(window: int, end: int):
        """Equal-weight mean of the monthly means over `window` months ending at `end`."""
        low = max(0, end - window + 1)
        if low > end:
            return float("nan"), 0
        block = means[low:end + 1]
        valid = int(np.isfinite(block).sum())
        if valid == 0:
            return float("nan"), 0
        return float(np.nanmean(block)), valid

    values: list[float] = []

    # "latest" means the newest level OBSERVED at or before this origin, not the origin
    # month itself. Those coincide whenever the origin month has transactions, which is the
    # case throughout the present data — but if the origin month were empty, reading it
    # directly would emit NaN while the name promises an available level. Walking back
    # keeps the name and the implementation consistent; `age` reports how far back it went.
    latest = float("nan")
    age = float("nan")
    for back in range(0, origin + 1):
        candidate = month_mean(origin - back)
        if np.isfinite(candidate):
            latest = candidate
            age = float(back)
            break
    values.append(float(np.log(latest)) if np.isfinite(latest) and latest > 0
                  else float("nan"))
    values.append(age)

    # The previous *observed* level before the latest one. Skipping only one calendar month
    # would still be empty after a multi-month gap and turn the change into NaN, so walk
    # back from the latest observation until a month with transactions is found.
    prev = float("nan")
    if np.isfinite(age):
        first = int(origin - age)
        for back in range(1, first + 1):
            candidate = month_mean(first - back)
            if np.isfinite(candidate):
                prev = candidate
                break
    if np.isfinite(latest) and np.isfinite(prev) and latest > 0 and prev > 0:
        values.append(float(np.log(latest) - np.log(prev)))
    else:
        values.append(float("nan"))

    for window in WINDOWS:
        mean, valid = window_mean(window, origin)
        values.append(float(np.log(mean)) if np.isfinite(mean) and mean > 0
                      else float("nan"))
        values.append(float(valid))

    last3, valid3 = window_mean(3, origin)
    prior3, validprior = window_mean(3, origin - 3)
    if (np.isfinite(last3) and np.isfinite(prior3) and last3 > 0 and prior3 > 0
            and valid3 > 0 and validprior > 0):
        values.append(float(np.log(last3) - np.log(prior3)))
    else:
        values.append(float("nan"))

    values.append(float(counts[origin]) if origin < len(counts) else float("nan"))
    low12 = max(0, origin - 11)
    values.append(float(counts[low12:origin + 1].mean()) if origin >= 0 else float("nan"))
    return np.asarray(values, dtype=np.float64)
