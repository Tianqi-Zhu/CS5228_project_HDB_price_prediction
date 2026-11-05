"""Map each supervised row to a macro observation window.

Why windows are keyed by cutoff month
-------------------------------------
1,403 daily rows do not imply 1,403 independent stock-to-rent samples. In a given
stage there are only a few dozen distinct observation cutoffs, and every supervised
row sharing a cutoff shares one window. The export therefore stores each distinct
window **once** plus an index from rows to windows, instead of duplicating a
63-day tensor per property transaction.

Policy handling
---------------
``origin_asof`` (default): the cutoff is the row's own ``origin_month_index``. For
a validation row the origin already equals the split cutoff, so nothing after the
cutoff can be read. For a stage-A holdout row the origin equals the inner cutoff,
which is the panel's ``fit_end``.

``target_previous_month``: the cutoff is ``target_month_index - 1``. This is a
separate conditional experiment: it necessarily reads observations dated after the
rent model's cutoff for validation rows, so it must never be described as strict
fixed-cutoff forecasting. The two policies never share a metric table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.fusion_contract_v3 import contract


def macro_cutoff_months(samples: pd.DataFrame, policy: str) -> np.ndarray:
    """Per-row observation cutoff month for the given policy."""
    if policy not in contract.POLICIES:
        raise ValueError(f"unknown policy {policy!r}; expected one of {contract.POLICIES}")
    origin = samples["origin_month_index"].to_numpy(dtype=np.int64)
    target = samples["target_month_index"].to_numpy(dtype=np.int64)
    if policy == contract.POLICY_ORIGIN_ASOF:
        return origin
    if policy == contract.POLICY_TARGET_PREVIOUS_MONTH:
        cutoff = target - 1
        # Under this policy a cutoff may legitimately exceed the rent cutoff, but it
        # must never reach the target month itself.
        if (cutoff >= target).any():  # pragma: no cover - defensive
            raise AssertionError("target_previous_month reached the target month")
        return cutoff
    raise ValueError(policy)


def window_index(cutoff_months: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map per-row cutoffs to a dense window index.

    Returns ``(row_window_index, unique_cutoffs, counts)`` where ``counts`` is the
    number of rows mapping to each unique cutoff. ``counts.sum()`` equals the number
    of rows, which the verifier checks.
    """
    cutoffs = np.asarray(cutoff_months, dtype=np.int64)
    unique, inverse, counts = np.unique(cutoffs, return_inverse=True, return_counts=True)
    return inverse.astype(np.int32), unique.astype(np.int32), counts.astype(np.int32)


def build_macro_windows(
    samples: pd.DataFrame,
    policy: str,
    dbs_frame=None,
    coe_monthly=None,
) -> tuple[dict[str, np.ndarray], dict]:
    """Assemble every window array for one stage's sample table.

    ``dbs_frame`` and ``coe_monthly`` may be ``None`` to skip a branch; the returned
    dict then simply omits those keys. Both branches are keyed by the same unique
    cutoff list so a single ``row_window_index`` serves both.
    """
    cutoffs = macro_cutoff_months(samples, policy)
    row_window, unique, counts = window_index(cutoffs)

    arrays: dict[str, np.ndarray] = {
        "window_cutoff_month": unique,
        "window_count": counts,
        "row_window_index": np.asarray(row_window, dtype=np.int32),
    }
    summary: dict = {
        "policy": policy,
        "n_windows": int(len(unique)),
        "n_rows": int(len(samples)),
        "cutoff_range": [int(unique.min()), int(unique.max())] if len(unique) else [],
        "cutoff_range_months": (
            [contract.month_label(int(unique.min())), contract.month_label(int(unique.max()))]
            if len(unique) else []
        ),
        "window_count_sum": int(counts.sum()),
    }

    if dbs_frame is not None:
        n = len(unique)
        length = contract.DBS_LOOKBACK_ROWS
        seq = np.zeros((n, length, len(contract.DBS_CHANNELS)), dtype=np.float32)
        mask = np.zeros((n, length), dtype=np.float32)
        dates = np.zeros((n, length), dtype=np.int32)
        asof = np.zeros(n, dtype=np.int32)
        scalars = np.full((n, len(contract.DBS_SCALARS)), np.nan, dtype=np.float32)
        scalar_mask = np.zeros((n, len(contract.DBS_SCALARS)), dtype=np.float32)
        for index, month in enumerate(unique):
            position = _dbs.asof_position(dbs_frame, int(month))
            block, block_mask, block_dates = _dbs.window_arrays(dbs_frame, position, length)
            seq[index] = block
            mask[index] = block_mask
            dates[index] = block_dates
            values, values_mask = _dbs.scalar_arrays(dbs_frame, position, int(month))
            scalars[index] = values
            scalar_mask[index] = values_mask
            if position >= 0:
                asof[index] = int(dbs_frame.dates[position].strftime("%Y%m%d"))
        arrays.update({
            "dbs_seq": seq, "dbs_mask": mask, "dbs_dates": dates,
            "dbs_scalars": scalars, "dbs_scalar_mask": scalar_mask, "dbs_asof_date": asof,
        })
        summary["dbs"] = {
            "symbol": contract.DBS_SYMBOL,
            "lookback_rows": length,
            "windows_with_any_observation": int((mask.sum(axis=1) > 0).sum()),
            "windows_with_full_lookback": int((mask.sum(axis=1) >= length).sum()),
            "asof_range": [int(asof[asof > 0].min()), int(asof.max())] if (asof > 0).any() else [],
        }

    if coe_monthly is not None:
        n = len(unique)
        months = contract.COE_LOOKBACK_MONTHS
        n_cat = len(contract.COE_CATEGORIES)
        seq = np.zeros((n, n_cat, months, len(contract.COE_CHANNELS)), dtype=np.float32)
        mask = np.zeros((n, n_cat, months), dtype=np.float32)
        month_ids = np.zeros((n, months), dtype=np.int32)
        incomplete = np.zeros((n, n_cat, months), dtype=np.float32)
        scalars = np.full((n, n_cat, len(contract.COE_SCALARS)), np.nan, dtype=np.float32)
        scalar_mask = np.zeros((n, n_cat, len(contract.COE_SCALARS)), dtype=np.float32)
        for index, month in enumerate(unique):
            block, block_mask, block_months, block_incomplete = _coe.window_arrays(
                coe_monthly, int(month), months)
            seq[index] = block
            mask[index] = block_mask
            month_ids[index] = block_months
            incomplete[index] = block_incomplete
            values, values_mask = _coe.scalar_arrays(coe_monthly, int(month))
            scalars[index] = values
            scalar_mask[index] = values_mask
        arrays.update({
            "coe_seq": seq, "coe_mask": mask, "coe_months": month_ids,
            "coe_scalars": scalars, "coe_scalar_mask": scalar_mask,
            "coe_incomplete": incomplete,
        })
        summary["coe"] = {
            "categories": list(contract.COE_CATEGORIES),
            "lookback_months": months,
            "windows_with_any_observation": int((mask.sum(axis=(1, 2)) > 0).sum()),
            "windows_with_full_history": int((mask.sum(axis=2) >= months).all(axis=1).sum()),
            "incomplete_window_cells": int(incomplete.sum()),
        }

    return arrays, summary


# Imported lazily at module bottom to keep the public surface tiny while still
# allowing the helpers above to reference the modules without a cycle.
from experiments.fusion_contract_v3 import coe as _coe  # noqa: E402
from experiments.fusion_contract_v3 import dbs as _dbs  # noqa: E402
