"""Read-only source checks and feature creation shared by experiments."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.config import VALIDATION_START


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_main(raw: pd.DataFrame) -> pd.DataFrame:
    """Add features to a copy while preserving all source rows and CSV files."""
    frame = raw.copy(deep=True)
    frame["FLAT_TYPE_CLEAN"] = (
        frame["FLAT_TYPE"].fillna("unknown").astype(str).str.lower().str.strip()
        .str.replace(r"^(\d+)\s*[- ]\s*room$", r"\1-room", regex=True)
    )
    frame["FLAT_MODEL_CLEAN"] = frame["FLAT_MODEL"].fillna("unknown").astype(str).str.lower().str.strip()
    frame["TOWN_CLEAN"] = frame["TOWN"].fillna("unknown").astype(str).str.lower().str.strip()
    date = pd.to_datetime(frame["RENT_APPROVAL_DATE"], format="%Y-%m", errors="coerce")
    if date.isna().any():
        raise ValueError(f"Invalid RENT_APPROVAL_DATE in {int(date.isna().sum())} rows")
    frame["APPROVAL_MONTH_INDEX"] = (date.dt.year - 2021) * 12 + date.dt.month - 1
    frame["APPROVAL_MONTH_OF_YEAR"] = date.dt.month
    frame["FLAT_AGE_YEARS"] = date.dt.year - frame["LEASE_COMMENCE_DATE"]
    frame.loc[frame["FLAT_AGE_YEARS"] < 0, "FLAT_AGE_YEARS"] = np.nan
    return frame


def temporal_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by approval month; keep every row and prevent month overlap."""
    fitting = frame[frame["RENT_APPROVAL_DATE"] < VALIDATION_START].copy()
    validation = frame[frame["RENT_APPROVAL_DATE"] >= VALIDATION_START].copy()
    if fitting.empty or validation.empty:
        raise ValueError("Temporal split has an empty fitting or validation partition")
    if fitting["RENT_APPROVAL_DATE"].max() >= validation["RENT_APPROVAL_DATE"].min():
        raise AssertionError("Fitting and validation months overlap")
    return fitting, validation


def normalized_address(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized join keys for the HDB block table."""
    result = frame.copy()
    for column in ("TOWN", "BLOCK", "STREET"):
        result[column] = (
            result[column].astype("string").str.casefold().str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )
    return result


def add_core_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add report-ready core property features without changing source columns."""
    result = prepare_main(frame)
    if "MONTHLY_RENT" in result:
        result["LOG_MONTHLY_RENT"] = np.log1p(result["MONTHLY_RENT"].clip(lower=0))
        result["RENT_PER_SQM"] = result["MONTHLY_RENT"] / result["FLOOR_AREA_SQM"].replace(0, np.nan)
    result["LEASE_COMMENCE_YEAR"] = result["LEASE_COMMENCE_DATE"]
    result["APPROVAL_YEAR"] = pd.to_datetime(result["RENT_APPROVAL_DATE"]).dt.year
    result["BUILDING_AGE_AT_APPROVAL"] = result["FLAT_AGE_YEARS"]
    return result


def core_quality_report(frame: pd.DataFrame) -> dict:
    """Return compact quality, duplicate, outlier, and correlation diagnostics."""
    numeric = frame.select_dtypes(include=np.number)
    outliers = {}
    for column in numeric:
        q1, q3 = numeric[column].quantile([.25, .75])
        iqr = q3 - q1
        outliers[column] = int(((numeric[column] < q1 - 1.5 * iqr) | (numeric[column] > q3 + 1.5 * iqr)).sum())
    return {
        "rows": int(len(frame)),
        "missing_by_column": frame.isna().sum().astype(int).to_dict(),
        "exact_duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_property_month_rows": int(frame.duplicated(["RENT_APPROVAL_DATE", "TOWN", "BLOCK", "STREET", "FLAT_TYPE", "FLOOR_AREA_SQM"]).sum()),
        "iqr_outlier_counts": outliers,
        "numeric_correlations": numeric.corr().round(4).to_dict(),
    }


def add_economic_features(frame: pd.DataFrame, coe: pd.DataFrame | None = None,
                          stocks: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach calendar-month economic lags for a monthly-updated forecast.

    Lag 1 uses the preceding calendar month; lag 3/6 use three/six months
    before approval. Callers must evaluate this as a monthly-updated scenario,
    since future-month observations are unknown at a fixed training cutoff.
    """
    result = frame.copy()
    periods = pd.to_datetime(result["RENT_APPROVAL_DATE"], format="%Y-%m").dt.to_period("M")
    series = []
    if coe is not None:
        c = coe.copy()
        dates = pd.to_datetime(c["YEAR"].astype(str) + "-" + c["MONTH"].astype(str).str[:3] + "-01", format="%Y-%b-%d", errors="coerce")
        c["_period"] = dates.dt.to_period("M")
        c["QUOTA_PREMIUM"] = pd.to_numeric(c["QUOTA_PREMIUM"].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
        c["CATEGORY"] = c["CATEGORY"].astype(str).str.upper()
        wide = c.groupby(["_period", "CATEGORY"])["QUOTA_PREMIUM"].median().unstack("CATEGORY")
        wide.columns = [f"COE_{str(x).upper()}_MEDIAN" for x in wide.columns]
        series.append(wide)
    if stocks is not None:
        s = stocks.copy()
        s["DATE"] = pd.to_datetime(s["DATE"], errors="coerce")
        s["ADJUSTED_CLOSE"] = pd.to_numeric(s["ADJUSTED_CLOSE"], errors="coerce")
        s = s.dropna(subset=["DATE", "ADJUSTED_CLOSE"]).sort_values(["SYMBOL", "DATE"])
        s["_period"] = s["DATE"].dt.to_period("M")
        closes = s.groupby(["SYMBOL", "_period"])["ADJUSTED_CLOSE"].last().unstack("_period").T
        returns = closes.pct_change(fill_method=None)
        stock_month = pd.DataFrame({
            "STOCK_MEDIAN_RETURN": returns.median(axis=1),
            "STOCK_RETURN_VOLATILITY": returns.std(axis=1),
        })
        series.append(stock_month)
    if not series:
        return result
    monthly = pd.concat(series, axis=1).sort_index()
    full_months = pd.period_range(min(monthly.index.min(), periods.min()), max(monthly.index.max(), periods.max()), freq="M")
    monthly = monthly.reindex(full_months)
    lagged = pd.concat({f"{column}_LAG{lag}": monthly[column].shift(lag)
                        for column in monthly for lag in (1, 3, 6)}, axis=1)
    lagged.index.name = "_period"
    result["_period"] = periods
    result = result.join(lagged, on="_period").drop(columns="_period")
    return result


def add_spatial_features(frame: pd.DataFrame, blocks: pd.DataFrame, mrt: pd.DataFrame,
                         malls: pd.DataFrame, schools: pd.DataFrame,
                         radii_km=(0.5, 1.0, 2.0)) -> pd.DataFrame:
    """Join block attributes and generate reusable nearest/count amenity features."""
    from sklearn.neighbors import BallTree
    if tuple(radii_km) != (0.5, 1.0, 2.0):
        raise ValueError("The current density feature requires radii (0.5, 1.0, 2.0) km")
    keys = ["TOWN", "BLOCK", "STREET"]
    block = normalized_address(blocks)
    if block.duplicated(keys).any():
        raise ValueError("HDB block table has duplicate normalized address keys")
    block = block.copy()
    block["POSTAL_CODE"] = block["POSTAL_CODE"].astype("string").replace("NIL", pd.NA)
    for column in ("MAX_FLOOR", "YEAR_COMPLETED"):
        block[column] = pd.to_numeric(block[column], errors="coerce").replace(-1, np.nan)
    origin = np.radians(block[["LATITUDE", "LONGITUDE"]].to_numpy(float))
    for prefix, places in (("MRT", mrt[mrt["STATUS"].astype(str).str.casefold().eq("open")]), ("MALL", malls), ("SCHOOL", schools[schools["MAINLEVEL_CODE"].astype(str).str.casefold().str.contains("primary|p1")])):
        target = places[["LATITUDE", "LONGITUDE"]].drop_duplicates().dropna()
        tree = BallTree(np.radians(target.to_numpy(float)), metric="haversine")
        dist, _ = tree.query(origin, k=1)
        block[f"NEAREST_{prefix}_KM"] = dist[:, 0] * 6371.0088
        for radius in radii_km:
            counts = tree.query_radius(origin, r=radius / 6371.0088, count_only=True)
            block[f"{prefix}_COUNT_{radius:.1f}KM"] = counts
    cbd = np.radians(np.asarray([[1.290270, 103.851959]]))
    block["DISTANCE_TO_CBD_KM"] = BallTree(cbd, metric="haversine").query(origin, k=1)[0][:, 0] * 6371.0088
    block["MRT_ACCESS_SCORE"] = 1.0 / (1.0 + block["NEAREST_MRT_KM"])
    block["AMENITY_DENSITY_1KM"] = block["MRT_COUNT_1.0KM"] + block["MALL_COUNT_1.0KM"] + block["SCHOOL_COUNT_1.0KM"]
    result = normalized_address(frame).merge(block, on=keys, how="left", validate="many_to_one")
    result["BUILDING_AGE_FROM_COMPLETION"] = pd.to_datetime(result["RENT_APPROVAL_DATE"], format="%Y-%m").dt.year - result["YEAR_COMPLETED"]
    return result


def candidate_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Create interpretable interaction candidates for tree/linear ablations."""
    result = frame.copy()
    if "FLOOR_AREA_SQM" in result:
        age = result["FLAT_AGE_YEARS"].fillna(0) if "FLAT_AGE_YEARS" in result else 0
        result["AREA_X_AGE"] = result["FLOOR_AREA_SQM"] * age
    if "NEAREST_MRT_KM" in result:
        result["AREA_X_MRT_ACCESS"] = result["FLOOR_AREA_SQM"] / (1 + result["NEAREST_MRT_KM"])
    if "FLAT_TYPE_CLEAN" in result and "TOWN_CLEAN" in result:
        result["TOWN_X_FLAT_TYPE"] = result["TOWN_CLEAN"].astype(str) + "__" + result["FLAT_TYPE_CLEAN"].astype(str)
    return result
