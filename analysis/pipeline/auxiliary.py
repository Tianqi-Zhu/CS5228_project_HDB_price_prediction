"""Leakage-conscious auxiliary features for modeling experiments.

Static geographic features are always available.  Supplied contemporaneous
economic series are opt-in because they assume that the month's market data is
available when predicting that same month (the competition files cover the
entire test horizon).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from analysis.pipeline.config import AUX
from analysis.pipeline.data import norm_block, norm_text, prepare_main
from analysis.pipeline.spatial import EARTH_RADIUS_KM, primary_schools


BLOCK_KEYS = ["TOWN", "BLOCK", "STREET"]
MACRO_NUMERIC = [
    "COE_PREMIUM_A", "COE_PREMIUM_B", "COE_PREMIUM_C", "COE_PREMIUM_E",
    "COE_PREMIUM_MEAN", "COE_DEMAND_RATIO_MEAN",
    "STOCK_MARKET_LEVEL", "STOCK_BANK_LEVEL", "STOCK_REIT_LEVEL",
    "STOCK_PROPERTY_LEVEL", "STOCK_MARKET_MOM_3M", "STOCK_REIT_MOM_3M",
]

STOCK_GROUPS = {
    "MARKET": None,
    "BANK": ["DBS Group", "OCBC Bank", "UOB"],
    "REIT": [
        "Ascendas Reit", "CapitaLand Mall Trust", "Keppel REIT",
        "Mapletree Commercial Trust", "Mapletree Industrial Trust",
        "Mapletree Logistics Trust", "Suntec REIT",
    ],
    "PROPERTY": ["City Developments", "UOL Group", "Singapore Land"],
}


def _normalized_keys(frame: pd.DataFrame) -> pd.DataFrame:
    keys = pd.DataFrame(index=frame.index)
    keys["TOWN_KEY"] = norm_text(frame["TOWN"])
    keys["BLOCK_KEY"] = norm_block(frame["BLOCK"])
    keys["STREET_KEY"] = norm_text(frame["STREET"])
    return keys


def _facility_features(points: pd.DataFrame, places: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Nearest distance and neighbourhood counts for valid coordinates."""
    result = pd.DataFrame(index=points.index)
    origin = points[["LATITUDE", "LONGITUDE"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    target = (places[["LATITUDE", "LONGITUDE"]].apply(pd.to_numeric, errors="coerce")
              .dropna().drop_duplicates().to_numpy(float))
    valid = np.isfinite(origin).all(axis=1)
    result[f"{prefix}_DIST_KM"] = np.nan
    for radius in (0.5, 1.0, 2.0):
        result[f"{prefix}_COUNT_{str(radius).replace('.', '_')}KM"] = 0.0
    if not len(target) or not valid.any():
        return result
    tree = BallTree(np.radians(target), metric="haversine")
    query = np.radians(origin[valid])
    result.loc[valid, f"{prefix}_DIST_KM"] = tree.query(query, k=1)[0][:, 0] * EARTH_RADIUS_KM
    for radius in (0.5, 1.0, 2.0):
        counts = tree.query_radius(query, r=radius / EARTH_RADIUS_KM, count_only=True)
        result.loc[valid, f"{prefix}_COUNT_{str(radius).replace('.', '_')}KM"] = counts
    return result


def _money_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")


def _monthly_macro_features() -> pd.DataFrame:
    """Build one row per month from the supplied COE and stock files."""
    coe = pd.read_csv(AUX / "sg-coe-prices.csv")
    month_number = pd.to_datetime(coe["MONTH"], format="%b").dt.month
    coe["RENT_APPROVAL_DATE"] = (
        coe["YEAR"].astype(int).astype(str) + "-" + month_number.astype(str).str.zfill(2)
    )
    coe["PREMIUM"] = _money_number(coe["QUOTA_PREMIUM"])
    coe["QUOTA_NUMBER"] = _money_number(coe["QUOTA"])
    coe["BIDS_NUMBER"] = _money_number(coe["BIDS_RECEIVED"])
    coe["DEMAND_RATIO"] = coe["BIDS_NUMBER"] / coe["QUOTA_NUMBER"].replace(0, np.nan)
    premiums = coe.pivot_table(
        index="RENT_APPROVAL_DATE", columns="CATEGORY", values="PREMIUM", aggfunc="mean",
    ).rename(columns=lambda value: f"COE_PREMIUM_{str(value).upper()}")
    coe_summary = coe.groupby("RENT_APPROVAL_DATE").agg(
        COE_PREMIUM_MEAN=("PREMIUM", "mean"),
        COE_DEMAND_RATIO_MEAN=("DEMAND_RATIO", "mean"),
    )
    monthly = coe_summary.join(premiums, how="outer")

    stocks = pd.read_csv(AUX / "sg-stock-prices.csv")
    stocks["RENT_APPROVAL_DATE"] = pd.to_datetime(stocks["DATE"]).dt.strftime("%Y-%m")
    stocks["PRICE"] = pd.to_numeric(stocks["ADJUSTED_CLOSE"], errors="coerce")
    prices = stocks.pivot_table(
        index="RENT_APPROVAL_DATE", columns="NAME", values="PRICE", aggfunc="mean",
    ).sort_index()
    first = prices.apply(lambda values: values.dropna().iloc[0] if values.notna().any() else np.nan)
    levels = prices.divide(first.replace(0, np.nan), axis=1)
    for label, names in STOCK_GROUPS.items():
        selected = levels if names is None else levels.reindex(columns=names)
        monthly[f"STOCK_{label}_LEVEL"] = selected.median(axis=1, skipna=True)
    monthly["STOCK_MARKET_MOM_3M"] = monthly["STOCK_MARKET_LEVEL"].pct_change(3, fill_method=None)
    monthly["STOCK_REIT_MOM_3M"] = monthly["STOCK_REIT_LEVEL"].pct_change(3, fill_method=None)
    return monthly.reset_index()


def build_auxiliary_features(raw: pd.DataFrame, *, include_macro: bool = False) -> pd.DataFrame:
    """Return row-preserving property, time, block and facility features."""
    frame = prepare_main(raw)
    keys = _normalized_keys(frame)
    blocks = pd.read_csv(AUX / "sg-hdb-block.csv", dtype={"BLOCK": "string", "POSTAL_CODE": "string"})
    block_keys = _normalized_keys(blocks)
    blocks = pd.concat([block_keys, blocks.drop(columns=BLOCK_KEYS)], axis=1)
    if blocks.duplicated(["TOWN_KEY", "BLOCK_KEY", "STREET_KEY"]).any():
        raise ValueError("Normalized auxiliary block keys are not unique")

    row_order = np.arange(len(frame))
    joined = pd.concat([frame.reset_index(drop=True), keys.reset_index(drop=True)], axis=1)
    joined["_ROW_ORDER"] = row_order
    joined = joined.merge(
        blocks, how="left", on=["TOWN_KEY", "BLOCK_KEY", "STREET_KEY"],
        validate="many_to_one", sort=False,
    ).sort_values("_ROW_ORDER", kind="stable").reset_index(drop=True)
    if len(joined) != len(frame) or not np.array_equal(joined.pop("_ROW_ORDER"), row_order):
        raise AssertionError("Auxiliary join changed row count or order")

    joined["ADDRESS_KEY"] = (
        joined["TOWN_KEY"].fillna("unknown") + "|" +
        joined["BLOCK_KEY"].fillna("unknown") + "|" +
        joined["STREET_KEY"].fillna("unknown")
    )
    joined["STREET_CLEAN"] = joined["STREET_KEY"].fillna("unknown")
    joined["BLOCK_CLEAN"] = joined["BLOCK_KEY"].fillna("unknown")
    joined["POSTAL_CODE_CLEAN"] = joined["POSTAL_CODE"].replace({"NIL": pd.NA}).fillna("unknown").astype(str)
    for column in ("SUBZONE", "PLANNING_AREA", "REGION"):
        joined[f"{column}_CLEAN"] = norm_text(joined[column]).fillna("unknown")

    for column in ("LATITUDE", "LONGITUDE", "MAX_FLOOR", "YEAR_COMPLETED"):
        joined[column] = pd.to_numeric(joined[column], errors="coerce")
    joined[["MAX_FLOOR", "YEAR_COMPLETED"]] = joined[["MAX_FLOOR", "YEAR_COMPLETED"]].replace(-1, np.nan)

    approval = pd.to_datetime(joined["RENT_APPROVAL_DATE"], format="%Y-%m", errors="raise")
    joined["APPROVAL_YEAR"] = approval.dt.year
    joined["APPROVAL_MONTH_SIN"] = np.sin(2 * np.pi * approval.dt.month / 12)
    joined["APPROVAL_MONTH_COS"] = np.cos(2 * np.pi * approval.dt.month / 12)
    joined["BUILDING_AGE_YEARS"] = approval.dt.year - joined["YEAR_COMPLETED"]
    joined.loc[joined["BUILDING_AGE_YEARS"] < 0, "BUILDING_AGE_YEARS"] = np.nan
    joined["COMPLETION_LEASE_GAP"] = joined["LEASE_COMMENCE_DATE"] - joined["YEAR_COMPLETED"]
    joined["ROOM_COUNT"] = pd.to_numeric(
        joined["FLAT_TYPE_CLEAN"].astype(str).str.extract(r"(\d+)", expand=False), errors="coerce"
    )
    joined["AREA_PER_ROOM"] = joined["FLOOR_AREA_SQM"] / joined["ROOM_COUNT"].replace(0, np.nan)

    mrt = pd.read_csv(AUX / "sg-mrt-stations.csv")
    mrt = mrt.loc[mrt["STATUS"].astype(str).str.casefold().eq("open")]
    malls = pd.read_csv(AUX / "sg-shopping-malls.csv")
    schools = primary_schools(pd.read_csv(AUX / "sg-schools.csv"), include_mixed=False)
    point_columns = joined[["LATITUDE", "LONGITUDE"]]
    for places, prefix in ((mrt, "MRT"), (malls, "MALL"), (schools, "PRIMARY_SCHOOL")):
        joined = pd.concat([joined, _facility_features(point_columns, places, prefix)], axis=1)

    cbd = pd.DataFrame({"LATITUDE": [1.2840], "LONGITUDE": [103.8510]})
    joined["CBD_DIST_KM"] = _facility_features(point_columns, cbd, "CBD")["CBD_DIST_KM"]
    if include_macro:
        before = len(joined)
        joined = joined.merge(
            _monthly_macro_features(), on="RENT_APPROVAL_DATE", how="left",
            validate="many_to_one", sort=False,
        )
        if len(joined) != before:
            raise AssertionError("Monthly macro join changed row count")
    return joined
