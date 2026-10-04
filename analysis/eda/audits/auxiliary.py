"""Audit supplied auxiliary CSVs and their fit to the HDB rental dataset.

Run from the repository root with the dependencies in requirements-model.txt:
    python -m analysis.eda.run --scope auxiliary

Only reports/eda/supplementary/auxiliary is written. Source CSV hashes are
checked before and after the analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from analysis.pipeline.data import check_output_path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis.pipeline.spatial import EARTH_RADIUS_KM, nearest_distances, primary_schools

from analysis.pipeline.config import AUX, DATA, EDA_REPORTS, ROOT
from analysis.pipeline.data import normalized_address, sha256

DEFAULT_OUT = EDA_REPORTS / "supplementary" / "auxiliary"
OUT = DEFAULT_OUT
AUX_NAMES = (
    "sg-hdb-block.csv",
    "sg-mrt-stations.csv",
    "sg-shopping-malls.csv",
    "sg-schools.csv",
    "sg-coe-prices.csv",
    "sg-stock-prices.csv",
)
SOURCES = [DATA / "train.csv", DATA / "test.csv", *(AUX / name for name in AUX_NAMES)]
ADDRESS = ["TOWN", "BLOCK", "STREET"]


def quality(frame: pd.DataFrame) -> dict:
    return {
        "rows": len(frame),
        "columns": frame.columns.tolist(),
        "missing_by_column": frame.isna().sum().astype(int).to_dict(),
        "missing_cells": int(frame.isna().sum().sum()),
        "exact_duplicate_rows": int(frame.duplicated().sum()),
    }


def coordinate_quality(frame: pd.DataFrame) -> dict:
    latitude = pd.to_numeric(frame["LATITUDE"], errors="coerce")
    longitude = pd.to_numeric(frame["LONGITUDE"], errors="coerce")
    valid = latitude.between(-90, 90) & longitude.between(-180, 180)
    return {
        "invalid_or_missing_coordinate_rows": int((~valid).sum()),
        "latitude_range": [float(latitude.min()), float(latitude.max())],
        "longitude_range": [float(longitude.min()), float(longitude.max())],
        "duplicate_coordinate_rows": int(frame.duplicated(["LATITUDE", "LONGITUDE"]).sum()),
        "unique_coordinate_pairs": int(frame[["LATITUDE", "LONGITUDE"]].drop_duplicates().shape[0]),
    }


def month_coverage(dates: pd.Series) -> dict:
    months = dates.dt.to_period("M")
    observed = pd.PeriodIndex(months.dropna().unique()).sort_values()
    expected = pd.period_range(observed.min(), observed.max(), freq="M")
    return {
        "first_month": str(observed.min()),
        "last_month": str(observed.max()),
        "observed_months": len(observed),
        "missing_months_within_range": [str(month) for month in expected.difference(observed)],
        "invalid_date_rows": int(dates.isna().sum()),
    }


def distance_summary(values: np.ndarray, destinations: pd.DataFrame) -> dict:
    return {
        "hdb_blocks": len(values),
        "destination_rows": len(destinations),
        "unique_destination_locations": int(
            destinations[["LATITUDE", "LONGITUDE"]].drop_duplicates().shape[0]
        ),
        "min_km": round(float(values.min()), 3),
        "median_km": round(float(np.median(values)), 3),
        "p90_km": round(float(np.quantile(values, .90)), 3),
        "p99_km": round(float(np.quantile(values, .99)), 3),
        "max_km": round(float(values.max()), 3),
        "blocks_within_1km_pct": round(float(np.mean(values <= 1) * 100), 2),
    }


def plot_locations(blocks: pd.DataFrame, places: dict[str, pd.DataFrame]) -> None:
    colors = {"Open MRT": "#D14B43", "Malls": "#D68B29", "Primary schools": "#3979AD"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True, layout="constrained")
    for ax, (label, frame) in zip(axes, places.items()):
        ax.scatter(blocks["LONGITUDE"], blocks["LATITUDE"], s=2, alpha=.22, color="#667786", label="HDB blocks")
        ax.scatter(frame["LONGITUDE"], frame["LATITUDE"], s=25, alpha=.8, color=colors[label], label=label)
        ax.set_title(label)
        ax.set_xlabel("Longitude")
        ax.grid(alpha=.15)
        ax.legend(loc="lower right", markerscale=2)
    axes[0].set_ylabel("Latitude")
    fig.suptitle("Supplied locations (current snapshot)", fontsize=16)
    fig.savefig(OUT / "01_auxiliary_locations.png", dpi=160, facecolor="white")
    plt.close(fig)


def plot_distances(distances: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5), layout="constrained")
    colors = {"Open MRT": "#D14B43", "Malls": "#D68B29", "Primary schools": "#3979AD"}
    for label, values in distances.items():
        ordered = np.sort(values)
        ax.plot(ordered, np.arange(1, len(ordered) + 1) / len(ordered), label=label, linewidth=2.5, color=colors[label])
    ax.axvline(1, color="#59636C", linestyle="--", linewidth=1, label="1 km")
    ax.set(xlabel="Straight-line distance to nearest location (km)", ylabel="Share of HDB blocks", ylim=(0, 1.01))
    ax.set_title("Nearest amenity distances from 9,464 HDB blocks")
    ax.grid(alpha=.2)
    ax.legend()
    fig.savefig(OUT / "02_nearest_distances.png", dpi=160, facecolor="white")
    plt.close(fig)


def plot_time_coverage(coe_dates: pd.Series, stock_dates: pd.Series, stocks: pd.DataFrame) -> None:
    coe_monthly = coe_dates.dt.to_period("M").value_counts().sort_index()
    stock_monthly = stocks.loc[stocks["ADJUSTED_CLOSE"].notna()].groupby(
        stock_dates.loc[stocks["ADJUSTED_CLOSE"].notna()].dt.to_period("M")
    )["SYMBOL"].nunique()
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, layout="constrained")
    for ax, series, label, color in (
        (axes[0], coe_monthly, "COE category-round records", "#D68B29"),
        (axes[1], stock_monthly, "Stock symbols with a valid quote", "#3979AD"),
    ):
        ax.plot(series.index.to_timestamp(), series.to_numpy(), color=color, linewidth=2)
        ax.axvline(pd.Timestamp("2025-03-01"), color="#59636C", linestyle="--", linewidth=1)
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[1].set_xlabel("Month (dashed line: last training month)")
    fig.suptitle("Coverage of economic auxiliary tables", fontsize=16)
    fig.savefig(OUT / "03_economic_coverage.png", dpi=160, facecolor="white")
    plt.close(fig)


def main(output_dir=None) -> None:
    global OUT
    OUT = check_output_path(Path(output_dir) if output_dir is not None else DEFAULT_OUT)
    before = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in SOURCES}
    frames = {
        name: pd.read_csv(AUX / name, dtype={"POSTAL_CODE": "string"} if name == "sg-hdb-block.csv" else None)
        for name in AUX_NAMES
    }
    blocks = frames["sg-hdb-block.csv"]
    mrt = frames["sg-mrt-stations.csv"]
    malls = frames["sg-shopping-malls.csv"]
    schools = frames["sg-schools.csv"]
    coe = frames["sg-coe-prices.csv"]
    stocks = frames["sg-stock-prices.csv"]
    summary = {"sources_sha256": before, "files": {name: quality(frame) for name, frame in frames.items()}}

    normalized_blocks = normalized_address(blocks)
    block_keys = normalized_blocks[ADDRESS]
    key_duplicates = int(block_keys.duplicated().sum())
    if key_duplicates:
        raise ValueError(f"HDB block table has {key_duplicates} duplicate normalized address keys")
    block_details = normalized_blocks[ADDRESS + ["POSTAL_CODE", "LATITUDE", "LONGITUDE", "MAX_FLOOR", "YEAR_COMPLETED"]]
    joins = {}
    for name in ("train", "test"):
        raw = pd.read_csv(DATA / f"{name}.csv")
        joined = normalized_address(raw).merge(block_details, on=ADDRESS, how="left", validate="m:1", indicator=True)
        matches = joined["_merge"].eq("both")
        approval_year = pd.to_datetime(joined["RENT_APPROVAL_DATE"], format="%Y-%m").dt.year
        valid_completion = joined["YEAR_COMPLETED"] > 0
        joins[name] = {
            "rows": len(joined),
            "matched_rows": int(matches.sum()),
            "matched_pct": round(float(matches.mean() * 100), 2),
            "unmatched_rows": int((~matches).sum()),
            "rows_with_nil_postal_code": int(joined["POSTAL_CODE"].eq("NIL").sum()),
            "rows_with_unknown_max_floor": int((joined["MAX_FLOOR"] <= 0).sum()),
            "rows_with_unknown_completion_year": int((joined["YEAR_COMPLETED"] <= 0).sum()),
            "rows_completion_after_approval_year": int((valid_completion & (joined["YEAR_COMPLETED"] > approval_year)).sum()),
            "rows_completion_lease_gap_over_5_years": int((valid_completion & ((joined["YEAR_COMPLETED"] - joined["LEASE_COMMENCE_DATE"]).abs() > 5)).sum()),
        }
    summary["block_join"] = {
        "key": ADDRESS,
        "normalization": "Unicode casefold, trim edges, collapse whitespace in each key column",
        "normalized_key_duplicates": key_duplicates,
        "train": joins["train"],
        "test": joins["test"],
    }
    summary["blocks"] = {
        "towns": int(blocks["TOWN"].nunique()),
        "postal_code_nil_rows": int(blocks["POSTAL_CODE"].eq("NIL").sum()),
        "postal_code_other_malformed_rows": int((~blocks["POSTAL_CODE"].eq("NIL") & ~blocks["POSTAL_CODE"].str.fullmatch(r"\d{6}")).sum()),
        "unknown_max_floor_rows": int((blocks["MAX_FLOOR"] <= 0).sum()),
        "unknown_completion_year_rows": int((blocks["YEAR_COMPLETED"] <= 0).sum()),
        "completion_year_valid_range": [int(blocks.loc[blocks.YEAR_COMPLETED > 0, "YEAR_COMPLETED"].min()), int(blocks.YEAR_COMPLETED.max())],
        "max_floor_valid_range": [int(blocks.loc[blocks.MAX_FLOOR > 0, "MAX_FLOOR"].min()), int(blocks.MAX_FLOOR.max())],
        "coordinates": coordinate_quality(blocks),
    }

    open_mrt = mrt[mrt["STATUS"].eq("open")]
    planned_mrt = mrt[mrt["STATUS"].eq("planned")]
    open_sites = set(zip(open_mrt.LATITUDE, open_mrt.LONGITUDE))
    primary = primary_schools(schools, include_mixed=True)
    summary["places"] = {
        "mrt": {
            "status_counts": {str(key): int(value) for key, value in mrt.STATUS.value_counts().items()},
            "unique_codes": int(mrt.CODE.nunique()),
            "open_unique_locations": int(open_mrt[["LATITUDE", "LONGITUDE"]].drop_duplicates().shape[0]),
            "planned_rows_at_an_open_location": int(sum((lat, lon) in open_sites for lat, lon in zip(planned_mrt.LATITUDE, planned_mrt.LONGITUDE))),
            "coordinates": coordinate_quality(mrt),
        },
        "malls": {"unique_names": int(malls.NAME.nunique()), "coordinates": coordinate_quality(malls)},
        "schools": {
            "unique_names": int(schools.NAME.nunique()),
            "primary_or_p1_rows": len(primary),
            "main_levels": {str(key): int(value) for key, value in schools.MAINLEVEL_CODE.value_counts().items()},
            "coordinates": coordinate_quality(schools),
        },
    }
    destinations = {"Open MRT": open_mrt, "Malls": malls, "Primary schools": primary}
    distances = {name: nearest_distances(blocks, frame) for name, frame in destinations.items()}
    summary["nearest_distance_current_snapshot"] = {
        name: distance_summary(distances[name], frame) for name, frame in destinations.items()
    }

    coe_dates = pd.to_datetime(coe.YEAR.astype(str) + "-" + coe.MONTH, format="%Y-%b", errors="coerce")
    if coe_dates.isna().any():
        raise ValueError("Invalid COE YEAR/MONTH values")
    coe_numeric = {}
    for column in ("QUOTA_PREMIUM", "PREVAILING_QUOTA_PREMIUM", "QUOTA", "BIDS_RECEIVED"):
        coe_numeric[column] = pd.to_numeric(coe[column].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
    summary["coe"] = {
        "date_coverage": month_coverage(coe_dates),
        "categories": {str(key): int(value) for key, value in coe.CATEGORY.value_counts().items()},
        "rounds": {str(key): int(value) for key, value in coe.ROUND.value_counts().items()},
        "duplicate_year_month_category_round": int(coe.duplicated(["YEAR", "MONTH", "CATEGORY", "ROUND"]).sum()),
        "unparseable_numeric_by_column": {column: int(value.isna().sum()) for column, value in coe_numeric.items()},
        "nonpositive_numeric_by_column": {column: int(value.le(0).sum()) for column, value in coe_numeric.items()},
        "quota_premium_range_sgd": [float(coe_numeric["QUOTA_PREMIUM"].min()), float(coe_numeric["QUOTA_PREMIUM"].max())],
        "quota_range": [int(coe_numeric["QUOTA"].min()), int(coe_numeric["QUOTA"].max())],
        "bids_received_range": [int(coe_numeric["BIDS_RECEIVED"].min()), int(coe_numeric["BIDS_RECEIVED"].max())],
        "monthly_record_count_range": [int(coe_dates.value_counts().min()), int(coe_dates.value_counts().max())],
        "last_month_record_count": int((coe_dates == coe_dates.max()).sum()),
    }
    stock_dates = pd.to_datetime(stocks.DATE, errors="coerce")
    quotes = ["OPEN", "HIGH", "LOW", "CLOSE", "ADJUSTED_CLOSE", "VOLUME"]
    valid_quotes = stocks.ADJUSTED_CLOSE.notna()
    complete_ohlc = stocks.dropna(subset=["OPEN", "HIGH", "LOW", "CLOSE"])
    ohlc_anomaly = (
        complete_ohlc.OPEN.lt(complete_ohlc.LOW) | complete_ohlc.OPEN.gt(complete_ohlc.HIGH)
        | complete_ohlc.CLOSE.lt(complete_ohlc.LOW) | complete_ohlc.CLOSE.gt(complete_ohlc.HIGH)
        | complete_ohlc.HIGH.lt(complete_ohlc.LOW)
    )
    symbol_suffix = (
        stocks[["SYMBOL"]].drop_duplicates().SYMBOL.str.extract(r"\.([A-Z]+)$", expand=False)
        .fillna("no suffix").value_counts()
    )
    per_symbol = stocks.groupby("SYMBOL").size()
    monthly_symbols = stocks.loc[valid_quotes].groupby(stock_dates.loc[valid_quotes].dt.to_period("M"))["SYMBOL"].nunique()
    summary["stocks"] = {
        "date_coverage": month_coverage(stock_dates),
        "symbols": int(stocks.SYMBOL.nunique()),
        "symbol_suffix_counts": {str(key): int(value) for key, value in symbol_suffix.items()},
        "symbol_row_count_range": [int(per_symbol.min()), int(per_symbol.max())],
        "duplicate_symbol_date": int(stocks.duplicated(["SYMBOL", "DATE"]).sum()),
        "all_quote_columns_missing_rows": int(stocks[quotes].isna().all(axis=1).sum()),
        "partially_missing_quote_rows": int((stocks[quotes].isna().any(axis=1) & ~stocks[quotes].isna().all(axis=1)).sum()),
        "zero_volume_rows": int(stocks.VOLUME.eq(0).sum()),
        "nonpositive_adjusted_close_rows": int(stocks.ADJUSTED_CLOSE.le(0).sum()),
        "adjusted_close_range_across_symbols": [float(stocks.ADJUSTED_CLOSE.min()), float(stocks.ADJUSTED_CLOSE.max())],
        "ohlc_range_anomaly_rows": int(ohlc_anomaly.sum()),
        "ohlc_range_anomaly_symbols": {str(key): int(value) for key, value in complete_ohlc.loc[ohlc_anomaly, "SYMBOL"].value_counts().items()},
        "monthly_valid_symbol_count_range": [int(monthly_symbols.min()), int(monthly_symbols.max())],
        "first_month_valid_symbols": int(monthly_symbols.iloc[0]),
        "last_month_valid_symbols": int(monthly_symbols.iloc[-1]),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    plot_locations(blocks, destinations)
    plot_distances(distances)
    plot_time_coverage(coe_dates, stock_dates, stocks)
    after = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in SOURCES}
    if before != after:
        raise AssertionError("A source CSV changed during the auxiliary EDA")
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote auxiliary audit and three figures to {OUT}")
    print(f"HDB joins: train {joins['train']['matched_pct']}%, test {joins['test']['matched_pct']}%")
    print(f"Stock rows with all quote fields missing: {summary['stocks']['all_quote_columns_missing_rows']}")


if __name__ == "__main__":
    main()
