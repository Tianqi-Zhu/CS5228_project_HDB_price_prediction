"""Point-in-time same-block, same-flat-type anniversary rent features.

The reference pool contains fitting rows only. For a fitting row, every lookup
is at least eleven months earlier; for a query row, the pool is frozen at the
forecast origin. Source-month flat-type means normalize historical rents only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


LAGS = (12, 24, 36)
SUFFIXES = ("N", "RENT", "RATIO", "SD", "LOW")
FEATURES = [f"ANN_BLOCK_{lag}_R1_{suffix}" for lag in LAGS for suffix in SUFFIXES]


def _month(values: pd.Series) -> np.ndarray:
    dates = pd.to_datetime(values, format="%Y-%m", errors="raise")
    return (dates.dt.year * 12 + dates.dt.month).to_numpy(np.int64)


def _key(rows: pd.DataFrame) -> pd.Series:
    return (
        rows["TOWN"].astype("string") + "|" + rows["STREET"].astype("string")
        + "|" + rows["BLOCK"].astype("string") + "|"
        + rows["FLAT_TYPE_CLEAN"].astype("string")
    )


class AnniversaryHistory:
    """Aggregate source rents by block, flat type and source month."""

    def fit(self, source: pd.DataFrame) -> "AnniversaryHistory":
        if source.empty or source["MONTHLY_RENT"].isna().any():
            raise ValueError("A nonempty, fully labelled fitting pool is required")
        month = _month(source["RENT_APPROVAL_DATE"])
        rent = source["MONTHLY_RENT"].to_numpy(float)
        reference = source.groupby(
            ["RENT_APPROVAL_DATE", "FLAT_TYPE_CLEAN"], observed=True
        )["MONTHLY_RENT"].transform("mean").to_numpy(float)
        if not np.isfinite(rent).all() or not np.isfinite(reference).all() or (reference <= 0).any():
            raise ValueError("Invalid rent or source-month flat-type reference")
        ratio = rent / reference
        table = pd.DataFrame({
            "key": _key(source).to_numpy(), "month": month,
            "n": np.ones(len(source)), "rent": rent, "ratio": ratio,
            "ratio2": ratio**2, "low": (ratio < 0.65).astype(float),
        })
        self.table_ = table.groupby(["key", "month"], sort=False).sum()
        self.last_source_month_ = int(month.max())
        return self

    def transform(self, rows: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "table_"):
            raise ValueError("Fit the anniversary history before transforming")
        month = _month(rows["RENT_APPROVAL_DATE"])
        key = _key(rows).to_numpy()
        result: dict[str, np.ndarray] = {}
        for lag in LAGS:
            totals = np.zeros((len(rows), 5), dtype=float)
            for delta in (-1, 0, 1):
                source_month = month - lag + delta
                if np.any(source_month >= month):
                    raise AssertionError("Anniversary lookup reached the current month")
                index = pd.MultiIndex.from_arrays([key, source_month], names=["key", "month"])
                totals += self.table_.reindex(index).fillna(0).to_numpy(float)
            n, rent, ratio, ratio2, low = totals.T
            denominator = np.where(n > 0, n, np.nan)
            prefix = f"ANN_BLOCK_{lag}_R1_"
            result[prefix + "N"] = n
            result[prefix + "RENT"] = rent / denominator
            result[prefix + "RATIO"] = ratio / denominator
            result[prefix + "SD"] = np.sqrt(np.maximum(0, ratio2 / denominator - (ratio / denominator)**2))
            result[prefix + "LOW"] = low / denominator
        return pd.DataFrame(result, index=rows.index)[FEATURES]
