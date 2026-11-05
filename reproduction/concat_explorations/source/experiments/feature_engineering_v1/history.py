"""Time-safe historical rent summaries.

Contract implemented here
-------------------------
* Aggregates are built **only** from rows whose approval month is <= the split
  cutoff.  Rows after the cutoff never enter any aggregate, encoder or index.
* A training row dated ``t`` sees aggregates over months ``< t`` (its own month
  and every later month are excluded).
* A validation row dated ``t > c`` sees the frozen snapshot at ``c``: all
  months ``<= c`` and nothing after, regardless of forecast horizon ``h``.
* Early months are never backfilled with later observations.
* Every produced mean is accompanied by a support count / observed-month count
  or an explicit missing flag.
* ``*_recency`` is measured from the row's *information boundary*, not from its
  own date: ``boundary - last_observed_month``.  For a training row dated ``t``
  the boundary is ``t-1``; for a validation row it is the cutoff ``c``.  Using
  one identical definition on both sides avoids a systematic one-month offset
  between fitting and inference.

Hierarchy and smoothing
-----------------------
Smoothed means are an empirical-Bayes recursion over a fixed chain::

    geo chain     : global -> town -> subzone -> street -> block
    geo_ft chain  : global -> flat_type -> town x flat_type -> block x flat_type

with ``value_level = (sum_level + m_level * value_parent) / (count_level + m_level)``
and ``m_level`` in ``config.SMOOTHING`` (rows).  A sparse level therefore
inherits most of its value from its parent instead of collapsing to a global
constant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config

LEVEL_COLUMNS: dict[str, tuple[str, ...]] = {
    "global": (),
    "town": ("TOWN_CLEAN",),
    "subzone": ("SUBZONE",),
    "street": ("TOWN_CLEAN", "STREET_CLEAN"),
    "block": ("BLOCK_ID",),
    "flattype": ("FLAT_TYPE_CLEAN",),
    "town_ft": ("TOWN_CLEAN", "FLAT_TYPE_CLEAN"),
    "block_ft": ("BLOCK_ID", "FLAT_TYPE_CLEAN"),
}

CHAINS: dict[str, tuple[str, ...]] = {
    "geo": ("global", "town", "subzone", "street", "block"),
    "geo_ft": ("global", "flattype", "town_ft", "block_ft"),
}


@dataclass(frozen=True)
class FeatureRequest:
    """One emitted feature.

    ``level`` is the level the statistic is read from; ``chain`` is the blending
    hierarchy used when ``statistic == 'mean'``.
    """

    name: str
    chain: str
    value: str          # "rent" | "psm"
    statistic: str      # "mean" | "count" | "months" | "recency"
    level: str = "block"


C1_REQUESTS = (
    FeatureRequest("hist_rent_block", "geo", "rent", "mean", "block"),
    FeatureRequest("hist_rent_street", "geo", "rent", "mean", "street"),
    FeatureRequest("hist_rent_subzone", "geo", "rent", "mean", "subzone"),
    FeatureRequest("hist_rent_town", "geo", "rent", "mean", "town"),
    FeatureRequest("hist_rent_block_n", "geo", "rent", "count", "block"),
    FeatureRequest("hist_rent_street_n", "geo", "rent", "count", "street"),
    FeatureRequest("hist_rent_town_n", "geo", "rent", "count", "town"),
    FeatureRequest("hist_rent_block_months", "geo", "rent", "months", "block"),
    FeatureRequest("hist_rent_block_recency", "geo", "rent", "recency", "block"),
    FeatureRequest("hist_rent_street_recency", "geo", "rent", "recency", "street"),
)
C2_REQUESTS = tuple(
    FeatureRequest(request.name.replace("hist_rent", "hist_psm"), request.chain,
                   "psm", request.statistic, request.level)
    for request in C1_REQUESTS
)
C3_REQUESTS = (
    FeatureRequest("hist_rentft_block_ft", "geo_ft", "rent", "mean", "block_ft"),
    FeatureRequest("hist_rentft_town_ft", "geo_ft", "rent", "mean", "town_ft"),
    FeatureRequest("hist_rentft_flattype", "geo_ft", "rent", "mean", "flattype"),
    FeatureRequest("hist_rentft_block_ft_n", "geo_ft", "rent", "count", "block_ft"),
    FeatureRequest("hist_rentft_block_ft_months", "geo_ft", "rent", "months", "block_ft"),
    FeatureRequest("hist_rentft_block_ft_recency", "geo_ft", "rent", "recency", "block_ft"),
    FeatureRequest("hist_psmft_block_ft", "geo_ft", "psm", "mean", "block_ft"),
    FeatureRequest("hist_psmft_town_ft", "geo_ft", "psm", "mean", "town_ft"),
)
# Public group name -> request tuple, used by the CLI and by config.GROUP_LABELS.
HISTORY_REQUESTS = {"hist_rent": C1_REQUESTS, "hist_psm": C2_REQUESTS, "hist_ft": C3_REQUESTS}


# ------------------------------------------------ helpers
def level_codes(frame: pd.DataFrame) -> dict[str, tuple[np.ndarray, list]]:
    """Integer code per level plus its category list (no target involved)."""
    codes: dict[str, tuple[np.ndarray, list]] = {}
    for level, columns in LEVEL_COLUMNS.items():
        if not columns:
            codes[level] = (np.zeros(len(frame), dtype=np.int64), ["__global__"])
            continue
        keys = frame[columns[0]].fillna("unknown").astype(str)
        for column in columns[1:]:
            keys = keys.str.cat(frame[column].fillna("unknown").astype(str), sep="|")
        values, uniques = pd.factorize(keys, sort=False)
        codes[level] = (np.asarray(values, dtype=np.int64), [str(item) for item in uniques])
    return codes


def _quantity(frame: pd.DataFrame, value: str) -> np.ndarray:
    rent = frame[config.TARGET].to_numpy(dtype=float)
    if value == "rent":
        return rent
    if value == "psm":
        area = frame["FLOOR_AREA_SQM"].to_numpy(dtype=float)
        return np.divide(rent, area, out=np.full_like(rent, np.nan), where=area > 0)
    raise ValueError(f"unknown value column {value!r}")


def build_history_features(
    frame: pd.DataFrame,
    split: dict,
    groups: tuple[str, ...],
    smoothing: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """History features for every row of ``frame`` plus a manifest block."""
    smoothing = dict(smoothing or config.SMOOTHING)
    cutoff = int(split["cutoff_index"])
    month = frame[config.MONTH_INDEX_COL].to_numpy()
    fitting = month <= cutoff
    if not fitting.any():
        raise ValueError("no fitting rows for history aggregates")

    requests: list[FeatureRequest] = []
    for group in groups:
        requests.extend(HISTORY_REQUESTS[group])
    if not requests:
        return pd.DataFrame(index=frame.index), {"enabled": False}

    codes = level_codes(frame)
    row_codes = {level: values[0] for level, values in codes.items()}
    snapshot = np.where(fitting, month - 1, cutoff)
    if (snapshot > cutoff).any():  # pragma: no cover - defensive
        raise AssertionError("snapshot crosses the cutoff")

    values = sorted({request.value for request in requests})
    quantities = {value: _quantity(frame, value) for value in values}

    # Available months per level come from the fitting window only.
    state: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for value in values:
        for level, (_code, uniques) in codes.items():
            size = len(uniques)
            state[(value, level)] = {
                "sum": np.zeros(size), "n": np.zeros(size),
                "months": np.zeros(size), "last": np.full(size, -1, dtype=np.int64),
            }

    columns = [request.name for request in requests]
    result = pd.DataFrame(np.nan, index=frame.index, columns=columns)

    def blended(value: str, chain: tuple[str, ...], rows: np.ndarray) -> np.ndarray:
        base = state[(value, "global")]
        if base["n"][0] <= 0:
            return np.full(len(rows), np.nan)
        current = np.full(len(rows), base["sum"][0] / base["n"][0])
        for level in chain[1:]:
            code = row_codes[level][rows]
            entry = state[(value, level)]
            prior = smoothing.get(level, 10.0)
            current = (entry["sum"][code] + prior * current) / (entry["n"][code] + prior)
        return current

    pending: dict[int, np.ndarray] = {}
    for month_value in np.unique(snapshot):
        pending[int(month_value)] = np.flatnonzero(snapshot == month_value)

    # Order matters and is load-bearing: a row whose snapshot is month ``s`` must
    # see months ``0..s`` *inclusive*.  Month ``s``'s labels are therefore
    # absorbed **before** the rows that snapshot at ``s`` are served.  Serving
    # first would leave the aggregate one month short (rows dated ``t`` would
    # only see ``0..t-2``, and validation would freeze at ``c-1``).
    # No row ever sees its own month, because a training row dated ``t``
    # snapshots at ``t-1`` and validation rows snapshot at ``c`` (< their own month).
    for month_value in range(cutoff + 1):
        # 1. absorb month_value's labels
        for value in values:
            quantity = quantities[value]
            for level, (codes_at_level, uniques) in codes.items():
                mask = fitting & (month == month_value) & np.isfinite(quantity)
                if not mask.any():
                    continue
                local = codes_at_level[mask]
                entry = state[(value, level)]
                entry["sum"] += np.bincount(local, weights=quantity[mask], minlength=len(uniques))
                entry["n"] += np.bincount(local, minlength=len(uniques))
                entry["months"][np.unique(local)] += 1.0
                entry["last"][np.unique(local)] = month_value

        # 2. serve every row whose snapshot is exactly this month
        rows = pending.get(month_value)
        if rows is None or not rows.size:
            continue
        block = {}
        for request in requests:
            if request.statistic == "mean":
                block[request.name] = blended(request.value, CHAINS[request.chain], rows)
                continue
            entry = state[(request.value, request.level)]
            code = row_codes[request.level][rows]
            if request.statistic == "count":
                block[request.name] = entry["n"][code]
            elif request.statistic == "months":
                block[request.name] = entry["months"][code]
            elif request.statistic == "recency":
                last = entry["last"][code]
                block[request.name] = np.where(last >= 0, month_value - last, np.nan)
            else:  # pragma: no cover
                raise ValueError(request.statistic)
        result.iloc[rows, :] = np.column_stack([block[name] for name in columns])

    # Rows in the very first month have no usable history at all: support
    # counts are explicitly zero and means/recency stay missing.
    no_history = frame.index[snapshot < 0]
    if len(no_history):
        mean_columns = [column for column in columns
                        if not (column.endswith("_n") or column.endswith("_months") or column.endswith("_recency"))]
        support_columns = [column for column in columns
                           if column.endswith("_n") or column.endswith("_months")]
        result.loc[no_history, mean_columns] = np.nan
        result.loc[no_history, support_columns] = 0.0

    # Support flags: an explicit 0/1 column instead of relying on NaN handling.
    if "hist_rent" in groups:
        result["hist_rent_block_seen"] = (result["hist_rent_block_n"] > 0).astype(float)
    if "hist_ft" in groups:
        result["hist_rentft_block_ft_seen"] = (result["hist_rentft_block_ft_n"] > 0).astype(float)

    manifest = {
        "enabled": True,
        "groups": list(groups),
        "cutoff_month": split["cutoff_month"],
        "cutoff_index": cutoff,
        "fitting_rows": int(fitting.sum()),
        "smoothing_rows": smoothing,
        "chains": {name: list(chain) for name, chain in CHAINS.items()},
        "level_sizes": {level: len(uniques) for level, (_code, uniques) in codes.items()},
        "features": columns + [c for c in result.columns if c.endswith("_seen")],
        "snapshot_rule": "training row t -> months < t; validation row -> months <= cutoff",
        "value_columns": values,
        "missing_counts": {column: int(result[column].isna().sum()) for column in result.columns},
    }
    return result, manifest


def build_window_features(
    frame: pd.DataFrame,
    split: dict,
    groups: tuple[str, ...],
) -> tuple[pd.DataFrame, dict]:
    """Trailing 3/6/12-month town x flat-type, town and flat-type summaries."""
    cutoff = int(split["cutoff_index"])
    month = frame[config.MONTH_INDEX_COL].to_numpy()
    fitting = month <= cutoff
    snapshot = np.where(fitting, month - 1, cutoff)

    codes = level_codes(frame)
    wanted = [value for value, group in (("rent", "win_rent"), ("psm", "win_psm")) if group in groups]

    levels = ("town_ft", "town", "flattype")
    width = cutoff + 2
    prefixes: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for value in wanted:
        quantity = _quantity(frame, value)
        for level in levels:
            codes_at_level, uniques = codes[level]
            sums = np.zeros((len(uniques), width))
            counts = np.zeros((len(uniques), width))
            for month_value in range(cutoff + 1):
                mask = fitting & (month == month_value) & np.isfinite(quantity)
                if not mask.any():
                    continue
                sums[:, month_value + 1] = np.bincount(codes_at_level[mask], weights=quantity[mask], minlength=len(uniques))
                counts[:, month_value + 1] = np.bincount(codes_at_level[mask], minlength=len(uniques))
            prefixes[(value, level)] = (np.cumsum(sums, axis=1), np.cumsum(counts, axis=1))

    data: dict[str, np.ndarray] = {}
    for value in wanted:
        for window in config.MARKET_WINDOWS:
            end = snapshot + 1
            start = np.maximum(end - window, 0)
            for level in levels:
                sum_prefix, count_prefix = prefixes[(value, level)]
                code = codes[level][0]
                total = sum_prefix[code, end] - sum_prefix[code, start]
                count = count_prefix[code, end] - count_prefix[code, start]
                data[f"win{window}_{value}_{level}_mean"] = np.divide(
                    total, count, out=np.full(len(frame), np.nan), where=count > 0)
            sum_prefix, count_prefix = prefixes[(value, "town_ft")]
            code = codes["town_ft"][0]
            data[f"win{window}_{value}_town_ft_n"] = count_prefix[code, end] - count_prefix[code, start]

    result = pd.DataFrame(data, index=frame.index)
    # Rows in the first training month have no history at all: means are
    # missing, and support counts are explicitly zero (not NaN).
    no_history = frame.index[snapshot < 0]
    mean_columns = [column for column in result.columns if column.endswith("_mean")]
    count_columns = [column for column in result.columns if column.endswith("_n")]
    result.loc[no_history, mean_columns] = np.nan
    result.loc[no_history, count_columns] = 0.0

    manifest = {
        "enabled": bool(wanted),
        "groups": list(groups),
        "windows": list(config.MARKET_WINDOWS),
        "levels": list(levels),
        "value_columns": wanted,
        "cutoff_index": cutoff,
        "snapshot_rule": "window ends at month-1 for training rows, cutoff for validation rows",
        "features": list(result.columns),
        "rows_without_history": int((snapshot < 0).sum()),
    }
    return result, manifest
