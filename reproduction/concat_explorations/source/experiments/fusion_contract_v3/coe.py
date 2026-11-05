"""COE monthly branch: parse, aggregate by category-month, build causal windows.

Parsing and aggregation rules
-----------------------------
* Currency and thousands separators are parsed explicitly; the columns arrive as
  strings and a naive ``astype(float)`` would fail rather than silently produce a
  wrong number.
* ``MONTH`` is a three-letter name and is mapped through an explicit table, never
  lexically.
* Premium is the **mean** of ``QUOTA_PREMIUM`` across the month's auctions.
  ``QUOTA`` and ``BIDS_RECEIVED`` are **summed** before the bids/quota ratio is
  taken, so a month with two rounds is not averaged as two separate ratios.
* ``PREVAILING_QUOTA_PREMIUM`` is **not used**. In this source it is ``0`` for all
  134 category-E rows; that is a structural artefact of the column, not a
  missing-value code and not a real premium. Using ``QUOTA_PREMIUM`` avoids
  inventing a distinction between "zero" and "missing" that the file does not
  support. A genuine zero premium would be preserved as ``0.0`` with
  ``observed = 1``.
* A category-month with fewer than two rounds, or the last month present in the
  source, is flagged ``incomplete``. The final 2026-08 has one round for A/B/C and
  no row at all for E; both facts are represented rather than smoothed away.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from experiments.fusion_contract_v3 import contract


def _parse_money(value) -> float:
    """'$40,609' -> 40609.0; blank/NaN -> NaN. Parsed, never cast."""
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip()
    if text in ("", "nan", "NaN", "None", "-"):
        return float("nan")
    text = text.replace("$", "").replace(",", "").strip()
    if text in ("", "-"):
        return float("nan")
    try:
        return float(text)
    except ValueError as error:
        raise ValueError(f"unparseable numeric field {value!r}") from error


@dataclass
class COEMonthly:
    """Per (category, month) aggregates over the whole source."""

    categories: tuple[str, ...]
    months: np.ndarray              # sorted unique month indices
    premium: np.ndarray             # (n_cat, n_months) mean premium
    quota: np.ndarray               # summed quota
    bids: np.ndarray                # summed bids
    ratio: np.ndarray               # sum(bids)/sum(quota), NaN when quota == 0
    auctions: np.ndarray            # rounds present
    observed: np.ndarray            # 1 where at least one auction
    incomplete: np.ndarray          # 1 where partial or final month
    last_source_month: int

    def month_position(self, month_index: int) -> int:
        """Position of ``month_index`` in :attr:`months`, or -1."""
        hits = np.flatnonzero(self.months == month_index)
        return int(hits[-1]) if hits.size else -1


def parse_coe_frame(path=None) -> pd.DataFrame:
    path = path or contract.COE_SOURCE
    frame = pd.read_csv(path)
    required = ["YEAR", "CATEGORY", "MONTH", "ROUND", "QUOTA_PREMIUM", "QUOTA", "BIDS_RECEIVED"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"COE source missing columns: {missing}")

    out = pd.DataFrame({
        "year": pd.to_numeric(frame["YEAR"], errors="raise").astype(int),
        "category": frame["CATEGORY"].astype(str).str.strip().str.lower(),
        "month_name": frame["MONTH"].astype(str).str.strip().str.lower(),
        "round": pd.to_numeric(frame["ROUND"], errors="raise").astype(int),
        "premium": frame["QUOTA_PREMIUM"].map(_parse_money),
        "quota": frame["QUOTA"].map(_parse_money),
        "bids": frame["BIDS_RECEIVED"].map(_parse_money),
    })
    unmapped = sorted(set(out["month_name"]) - set(contract.MONTH_ABBREVIATIONS))
    if unmapped:
        raise ValueError(f"unmapped month names: {unmapped}")
    out["month"] = out["month_name"].map(contract.MONTH_ABBREVIATIONS).astype(int)
    out["month_index"] = (out["year"] - 2021) * 12 + out["month"] - 1

    unexpected = sorted(set(out["category"]) - set(contract.COE_CATEGORIES))
    if unexpected:
        raise ValueError(f"unexpected categories: {unexpected}")
    return out


def build_coe_monthly(frame: pd.DataFrame) -> COEMonthly:
    """Aggregate parsed rows to (category, month)."""
    if frame.empty:
        raise ValueError("empty COE frame")
    months = np.array(sorted(frame["month_index"].unique()), dtype=np.int64)
    categories = tuple(contract.COE_CATEGORIES)
    n_cat, n_months = len(categories), len(months)
    position = {int(month): index for index, month in enumerate(months)}

    premium = np.full((n_cat, n_months), np.nan)
    quota = np.full((n_cat, n_months), np.nan)
    bids = np.full((n_cat, n_months), np.nan)
    ratio = np.full((n_cat, n_months), np.nan)
    auctions = np.zeros((n_cat, n_months))
    observed = np.zeros((n_cat, n_months))
    incomplete = np.zeros((n_cat, n_months))

    last_source_month = int(months.max())
    for cat_index, category in enumerate(categories):
        subset = frame[frame["category"] == category]
        for month_value, group in subset.groupby("month_index", observed=True):
            column = position[int(month_value)]
            auctions[cat_index, column] = float(len(group))
            if len(group) == 0:
                continue
            observed[cat_index, column] = 1.0
            premium[cat_index, column] = float(np.nanmean(group["premium"].to_numpy(dtype=float)))
            quota_total = float(np.nansum(group["quota"].to_numpy(dtype=float)))
            bids_total = float(np.nansum(group["bids"].to_numpy(dtype=float)))
            quota[cat_index, column] = quota_total
            bids[cat_index, column] = bids_total
            if quota_total > 0:
                ratio[cat_index, column] = bids_total / quota_total
            # Zero quota is masked for the ratio only; premium/quota stay real.
            if len(group) < contract.COE_MIN_ROUNDS_COMPLETE:
                incomplete[cat_index, column] = 1.0
    # The final month present in the source is never a complete month.
    incomplete[:, position[last_source_month]] = 1.0

    return COEMonthly(
        categories=categories, months=months, premium=premium, quota=quota, bids=bids,
        ratio=ratio, auctions=auctions, observed=observed, incomplete=incomplete,
        last_source_month=last_source_month,
    )


def window_arrays(monthly: COEMonthly, month_index: int,
                  length: int = contract.COE_LOOKBACK_MONTHS):
    """12-month window ending at ``month_index`` (inclusive), oldest-first.

    Returns ``(sequence, mask, months, incomplete)`` with shapes
    ``(4, length, 6)``, ``(4, length)``, ``(length,)``, ``(4, length)``.
    Months before the first observed month, or beyond the source, are padding.
    """
    n_cat = len(monthly.categories)
    channels = len(contract.COE_CHANNELS)
    sequence = np.zeros((n_cat, length, channels), dtype=np.float32)
    mask = np.zeros((n_cat, length), dtype=np.float32)
    month_ids = np.zeros(length, dtype=np.int32)
    incomplete = np.zeros((n_cat, length), dtype=np.float32)
    if month_index < 0:
        return sequence, mask, month_ids, incomplete

    # absolute month indices covered by the window
    start = month_index - length + 1
    for offset in range(length):
        absolute = start + offset
        if absolute < 0:
            continue
        # YYYYMM so that 0 is unambiguously "padding" (month index 0 is a real month).
        month_ids[offset] = (2021 + absolute // 12) * 100 + (absolute % 12 + 1)
        position = monthly.month_position(absolute)
        if position < 0:
            continue
        for cat_index in range(n_cat):
            if monthly.observed[cat_index, position] <= 0:
                continue
            sequence[cat_index, offset] = np.array([
                np.nan_to_num(monthly.premium[cat_index, position], nan=0.0),
                np.nan_to_num(monthly.quota[cat_index, position], nan=0.0),
                np.nan_to_num(monthly.bids[cat_index, position], nan=0.0),
                np.nan_to_num(monthly.ratio[cat_index, position], nan=0.0),
                monthly.auctions[cat_index, position],
                1.0,
            ], dtype=np.float32)
            mask[cat_index, offset] = 1.0
            incomplete[cat_index, offset] = monthly.incomplete[cat_index, position]
    return sequence, mask, month_ids, incomplete


def scalar_arrays(monthly: COEMonthly, month_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-category scalars at ``month_index``; shapes ``(4, 9)`` each."""
    n_cat = len(monthly.categories)
    values = np.full((n_cat, len(contract.COE_SCALARS)), np.nan)
    mask = np.zeros((n_cat, len(contract.COE_SCALARS)), dtype=np.float32)
    if month_index < 0:
        return values.astype(np.float32), mask

    index = {name: position for position, name in enumerate(contract.COE_SCALARS)}
    # months visible at this cutoff, newest first
    visible = [month for month in monthly.months if month <= month_index]

    for cat_index in range(n_cat):
        observed_months = [
            month for month in visible
            if monthly.observed[cat_index, monthly.month_position(month)] > 0
        ]
        if not observed_months:
            continue
        latest = observed_months[-1]
        latest_pos = monthly.month_position(latest)

        def put(name: str, value: float, ok: bool = True) -> None:
            if ok and np.isfinite(value):
                values[cat_index, index[name]] = value
                mask[cat_index, index[name]] = 1.0

        premium_latest = monthly.premium[cat_index, latest_pos]
        put("coe_premium_latest", float(premium_latest), np.isfinite(premium_latest))

        def premium_at(month: int) -> float:
            if month not in observed_months:
                return float("nan")
            return float(monthly.premium[cat_index, monthly.month_position(month)])

        previous = premium_at(latest - 1)
        put("coe_premium_chg_1m", float(premium_latest - previous), np.isfinite(previous))

        three_back = premium_at(latest - 3)
        put("coe_premium_chg_3m", float(premium_latest - three_back), np.isfinite(three_back))

        ratio_latest = monthly.ratio[cat_index, latest_pos]
        put("coe_ratio_latest", float(ratio_latest), np.isfinite(ratio_latest))
        ratio_previous = float("nan")
        if latest - 1 in observed_months:
            ratio_previous = float(monthly.ratio[cat_index, monthly.month_position(latest - 1)])
        put("coe_ratio_chg_1m", float(ratio_latest - ratio_previous), np.isfinite(ratio_previous))

        trailing = [month for month in observed_months if month >= latest - 11]
        put("coe_auctions_12m",
            float(sum(monthly.auctions[cat_index, monthly.month_position(month)]
                      for month in trailing)))
        put("coe_n_months_12", float(len(trailing)))
        put("coe_incomplete_latest", float(monthly.incomplete[cat_index, latest_pos]))

        trailing_premia = [float(monthly.premium[cat_index, monthly.month_position(month)])
                           for month in trailing]
        trailing_premia = [value for value in trailing_premia if np.isfinite(value)]
        if trailing_premia:
            reference = float(np.mean(trailing_premia))
            if reference != 0:
                put("coe_premium_rel_12m", float(premium_latest) / reference - 1.0)

    return values.astype(np.float32), mask
