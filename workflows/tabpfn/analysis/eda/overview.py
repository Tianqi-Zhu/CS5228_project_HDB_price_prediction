#!/usr/bin/env python3
"""Reproducible exploratory audit for the CS5228 HDB rental competition.

Run from the repository root: python -m analysis.eda.run --scope overview
Writes only experiment/eda/{figures,tables,summary.json}; source CSV files are read-only.
"""
from __future__ import annotations

import json
import hashlib
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.pipeline.data import check_output_path, norm_text, norm_flat_type, norm_block
from analysis.pipeline.spatial import nearest_distances as haversine_nearest, primary_schools

from analysis.pipeline.config import ROOT, DATA, DATE, TARGET
DEFAULT_OUT = ROOT / "experiment" / "eda"
OUT = DEFAULT_OUT
FIG = OUT / "figures"
TAB = OUT / "tables"
IDENTITY = ["TOWN", "BLOCK", "STREET"]
COLORS = {"navy": "#17324D", "teal": "#258C8C", "gold": "#D89B33", "coral": "#D96852", "blue": "#5A83B7"}


def clean_key(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["_town"] = norm_text(x["TOWN"])
    x["_block"] = norm_block(x["BLOCK"])
    x["_street"] = norm_text(x["STREET"])
    return x


def num(x):
    if pd.isna(x):
        return None
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(x) if math.isfinite(float(x)) else None
    return x


def save_table(df: pd.DataFrame, name: str) -> None:
    df.to_csv(TAB / name, index=False)


def chart_style():
    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 180, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.labelcolor": "#34495e",
                         "xtick.color": "#52616b", "ytick.color": "#52616b"})


def savefig(name: str):
    plt.tight_layout()
    plt.savefig(FIG / name, bbox_inches="tight", facecolor="white")
    plt.close()


def ks_stat(a: pd.Series, b: pd.Series) -> float | None:
    """Two-sample empirical KS distance (D only; no p-value assumptions)."""
    x = pd.to_numeric(a, errors="coerce").dropna().to_numpy(float)
    y = pd.to_numeric(b, errors="coerce").dropna().to_numpy(float)
    if not len(x) or not len(y):
        return None
    vals = np.sort(np.concatenate([x, y]))
    fx = np.searchsorted(np.sort(x), vals, side="right") / len(x)
    fy = np.searchsorted(np.sort(y), vals, side="right") / len(y)
    return float(np.max(np.abs(fx - fy)))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p.astype(float) + 1e-12
    q = q.astype(float) + 1e-12
    p /= p.sum(); q /= q.sum()
    m = (p + q) / 2
    return float((np.sum(p * np.log2(p / m)) + np.sum(q * np.log2(q / m))) / 2)


def median_baselines(train: pd.DataFrame) -> pd.DataFrame:
    d = train.copy()
    d["_date"] = pd.to_datetime(d[DATE], errors="coerce")
    d["_month"] = d["_date"].dt.to_period("M").astype("string")
    d["_flat"] = norm_flat_type(d["FLAT_TYPE"])
    d["_town"] = norm_text(d["TOWN"])
    splits = [("temporal_2024plus", "2024-01", "2025-03"), ("recent_2025Q1", "2025-01", "2025-03")]
    rows = []
    for split, lo, hi in splits:
        valid_mask = d["_month"].between(lo, hi)
        fit = d.loc[d["_month"] < lo]
        val = d.loc[valid_mask]
        if fit.empty or val.empty:
            continue
        global_med = float(fit[TARGET].median())
        group_med = fit.groupby(["_town", "_flat"], dropna=False)[TARGET].median()
        idx = pd.MultiIndex.from_arrays([val["_town"], val["_flat"]])
        pred_group = pd.Series(idx.map(group_med), index=val.index, dtype=float).fillna(global_med)
        for label, pred in [("training_median", pd.Series(global_med, index=val.index)),
                            ("train_only_town_flat_median", pred_group)]:
            err = pred.to_numpy() - val[TARGET].to_numpy()
            rows.append({"split": split, "fit_period_end": str(fit["_month"].max()),
                         "validation_period": f"{lo} to {hi}", "validation_rows": len(val),
                         "baseline": label, "mae": float(np.mean(np.abs(err))),
                         "rmse": float(np.sqrt(np.mean(err ** 2))),
                         "median_absolute_error": float(np.median(np.abs(err)))})
        recent_fit = fit[fit["_date"] >= pd.Period(lo, freq="M").start_time - pd.DateOffset(months=12)]
        recent_global = float(recent_fit[TARGET].median())
        recent_group = recent_fit.groupby(["_town", "_flat"], dropna=False)[TARGET].median()
        recent_idx = pd.MultiIndex.from_arrays([val["_town"], val["_flat"]])
        recent_pred = pd.Series(recent_idx.map(recent_group), index=val.index, dtype=float).fillna(recent_global)
        err = recent_pred.to_numpy() - val[TARGET].to_numpy()
        rows.append({"split": split, "fit_period_end": str(fit["_month"].max()),
                     "validation_period": f"{lo} to {hi}", "validation_rows": len(val),
                     "baseline": "previous_12m_town_flat_median", "mae": float(np.mean(np.abs(err)),),
                     "rmse": float(np.sqrt(np.mean(err ** 2))),
                     "median_absolute_error": float(np.median(np.abs(err)))})
    # Random holdout uses a fixed seed, and is explicitly a different estimand.
    rng = np.random.default_rng(5228)
    order = rng.permutation(len(d))
    cut = int(len(d) * .8)
    fit, val = d.iloc[order[:cut]], d.iloc[order[cut:]]
    global_med = float(fit[TARGET].median())
    gm = fit.groupby(["_town", "_flat"], dropna=False)[TARGET].median()
    gi = pd.MultiIndex.from_arrays([val["_town"], val["_flat"]])
    gp = pd.Series(gi.map(gm), index=val.index, dtype=float).fillna(global_med)
    for label, pred in [("training_median", pd.Series(global_med, index=val.index)), ("train_only_town_flat_median", gp)]:
        err = pred.to_numpy() - val[TARGET].to_numpy()
        rows.append({"split": "random_80_20_seed5228", "fit_period_end": "mixed dates",
                     "validation_period": "random 20% of labeled rows", "validation_rows": len(val),
                     "baseline": label, "mae": float(np.mean(np.abs(err))),
                     "rmse": float(np.sqrt(np.mean(err ** 2))),
                     "median_absolute_error": float(np.median(np.abs(err)))})
    return pd.DataFrame(rows)


def main(output_dir=None) -> None:
    global OUT, FIG, TAB
    OUT = check_output_path(Path(output_dir) if output_dir is not None else DEFAULT_OUT)
    FIG, TAB = OUT / "figures", OUT / "tables"
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
    # Remove a figure name from an earlier draft whose label implied building age.
    stale = FIG / "06_rent_by_building_age.png"
    if stale.exists():
        stale.unlink()
    chart_style()
    train_raw = pd.read_csv(DATA / "train.csv", low_memory=False)
    test_raw = pd.read_csv(DATA / "test.csv", low_memory=False)
    source_paths = [DATA / "train.csv", DATA / "test.csv", *sorted((DATA / "auxiliary").glob("*.csv"))]
    source_hashes_before = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    train, test = train_raw.copy(), test_raw.copy()
    train["_split"] = "train"; test["_split"] = "test"
    allx = pd.concat([train.drop(columns=[TARGET]), test], ignore_index=True)
    train["_date"] = pd.to_datetime(train[DATE], errors="coerce")
    test["_date"] = pd.to_datetime(test[DATE], errors="coerce")
    train["_month"] = train["_date"].dt.to_period("M").astype("string")
    test["_month"] = test["_date"].dt.to_period("M").astype("string")

    # Source integrity, data quality and target profile.
    quality = []
    for name, df in [("train", train_raw), ("test", test_raw)]:
        for c in df.columns:
            quality.append({"dataset": name, "check": "column", "field": c, "value": str(df[c].dtype)})
            quality.append({"dataset": name, "check": "missing_count", "field": c, "value": int(df[c].isna().sum())})
        quality += [{"dataset": name, "check": "rows", "field": "__all__", "value": len(df)},
                    {"dataset": name, "check": "exact_duplicate_rows", "field": "__all__", "value": int(df.duplicated().sum())}]
    for dfname, df in [("train", train_raw), ("test", test_raw)]:
        for c in df.columns:
            if df[c].nunique(dropna=False) <= 1:
                quality.append({"dataset": dfname, "check": "constant_column", "field": c, "value": df[c].iloc[0] if len(df) else None})
    save_table(pd.DataFrame(quality), "data_quality.csv")
    y = pd.to_numeric(train[TARGET], errors="coerce")
    qs = y.quantile([0, .01, .05, .25, .5, .75, .95, .99, 1])
    target_summary = {"count": int(y.notna().sum()), "mean": float(y.mean()), "std": float(y.std()),
                      "skew": float(y.skew()), "quantiles": {str(k): float(v) for k, v in qs.items()},
                      "negative_or_zero_count": int((y <= 0).sum())}
    plt.figure(figsize=(7, 4)); plt.hist(y, bins=45, color=COLORS["teal"], edgecolor="white")
    plt.axvline(y.median(), color=COLORS["coral"], lw=2, label=f"Median ${y.median():,.0f}")
    plt.xlabel("Monthly rent (SGD)"); plt.ylabel("Records"); plt.title("Training target distribution"); plt.legend(frameon=False)
    savefig("01_target_distribution.png")

    # Monthly target and composition.
    month = train.groupby("_month", observed=True)[TARGET].agg(rows="size", mean="mean", median="median", q25=lambda z:z.quantile(.25), q75=lambda z:z.quantile(.75)).reset_index()
    month.rename(columns={"_month": "month"}, inplace=True); save_table(month, "monthly_rent_trend.csv")
    plt.figure(figsize=(10, 4.5)); plt.plot(month.month, month["median"], color=COLORS["navy"], lw=2, label="Median")
    plt.plot(month.month, month["mean"], color=COLORS["gold"], lw=1.5, label="Mean")
    plt.fill_between(np.arange(len(month)), month.q25, month.q75, color=COLORS["teal"], alpha=.15, label="IQR")
    plt.xticks(np.arange(0, len(month), max(1, len(month)//12)), month.month.iloc[::max(1, len(month)//12)], rotation=45, ha="right")
    plt.ylabel("Monthly rent (SGD)"); plt.xlabel("Approval month"); plt.title("Rent rises over the observed period"); plt.legend(frameon=False, ncol=3)
    savefig("02_monthly_rent_trend.png")
    yearly = train.assign(year=train["_date"].dt.year).groupby("year")[TARGET].agg(rows="size", mean="mean", median="median", q25=lambda z:z.quantile(.25), q75=lambda z:z.quantile(.75)).reset_index()
    save_table(yearly, "annual_rent_trend.csv")

    # Categoricals: normalized flat-type audit, town / flat type summaries.
    catrows = []
    for c in ["TOWN", "FLAT_TYPE", "FLAT_MODEL", "STREET", "BLOCK"]:
        s = norm_flat_type(train_raw[c]) if c == "FLAT_TYPE" else norm_text(train_raw[c])
        catrows.append({"field": c, "raw_unique": int(train_raw[c].nunique(dropna=False)),
                        "normalized_unique": int(s.nunique(dropna=False)), "missing": int(s.isna().sum()),
                        "top_normalized_values": " | ".join(s.value_counts().head(8).index.astype(str))})
    save_table(pd.DataFrame(catrows), "categorical_normalization.csv")
    for c in ["TOWN", "FLAT_TYPE", "FLAT_MODEL", "STREET", "BLOCK"]:
        normfn = norm_flat_type if c == "FLAT_TYPE" else (norm_block if c == "BLOCK" else norm_text)
        rawn = train_raw[c].astype("string").str.strip()
        normalized = normfn(train_raw[c])
        idx = pd.DataFrame({"raw": rawn, "normalized": normalized}).dropna().drop_duplicates()
        collapsed = idx.groupby("normalized").raw.nunique()
        catrows[[x["field"] for x in catrows].index(c)]["normalized_label_collapses"] = int((collapsed > 1).sum())
    save_table(pd.DataFrame(catrows), "categorical_normalization.csv")
    cat = train.assign(_town=norm_text(train.TOWN), _flat=norm_flat_type(train.FLAT_TYPE))
    for c, label in [("_town", "town"), ("_flat", "flat_type")]:
        q = cat.groupby(c, dropna=False)[TARGET].agg(rows="size", mean="mean", median="median", q25=lambda z:z.quantile(.25), q75=lambda z:z.quantile(.75)).reset_index().rename(columns={c: label}).sort_values("median", ascending=False)
        save_table(q, f"rent_by_{label}.csv")
    popular_flats = cat["_flat"].value_counts().head(5).index
    flat_month = cat[cat["_flat"].isin(popular_flats)].groupby(["_month", "_flat"], observed=True)[TARGET].agg(rows="size", median="median").reset_index()
    flat_month.rename(columns={"_month": "month", "_flat": "flat_type"}, inplace=True)
    save_table(flat_month, "monthly_rent_by_flat_type.csv")
    plt.figure(figsize=(9, 4.5))
    for ft, part in flat_month.groupby("flat_type"):
        plt.plot(part.month, part["median"], lw=1.7, label=ft)
    plt.xticks(np.arange(0, len(month), max(1, len(month)//12)), month.month.iloc[::max(1, len(month)//12)], rotation=45, ha="right")
    plt.ylabel("Median monthly rent (SGD)"); plt.xlabel("Approval month"); plt.title("Rent trends differ across common flat types")
    plt.legend(frameon=False, ncol=3); savefig("09_monthly_rent_by_flat_type.png")
    townq = cat.groupby("_town")[TARGET].median().sort_values()
    plt.figure(figsize=(8, 6)); plt.barh(townq.index.astype(str), townq.values, color=COLORS["blue"])
    plt.xlabel("Median monthly rent (SGD)"); plt.title("Median rent varies by town"); savefig("03_median_rent_by_town.png")
    flatq = cat.groupby("_flat")[TARGET].median().sort_values()
    plt.figure(figsize=(8, 4.5)); plt.barh(flatq.index.astype(str), flatq.values, color=COLORS["gold"])
    plt.xlabel("Median monthly rent (SGD)"); plt.title("Median rent varies by flat type"); savefig("04_median_rent_by_flat_type.png")

    # Numeric property patterns and invalid-age audit.
    approval_year = train["_date"].dt.year
    age = approval_year - pd.to_numeric(train["LEASE_COMMENCE_DATE"], errors="coerce")
    area = pd.to_numeric(train["FLOOR_AREA_SQM"], errors="coerce")
    binned = pd.DataFrame({"area_sqm": area, "rent": y}).dropna()
    binned["area_bin"] = pd.cut(binned.area_sqm, bins=np.arange(20, max(161, int(binned.area_sqm.max()+10)), 10), right=False)
    area_summary = binned.groupby("area_bin", observed=True).rent.agg(rows="size", median="median", mean="mean").reset_index()
    area_summary["area_bin"] = area_summary["area_bin"].astype(str); save_table(area_summary, "rent_by_floor_area_bin.csv")
    plt.figure(figsize=(7, 4.5)); plt.scatter(area, y, s=5, alpha=.12, color=COLORS["blue"], rasterized=True)
    medarea = area_summary.copy(); medarea["mid"] = medarea.area_bin.str.extract(r"\[(\d+),")[0].astype(float) + 5
    plt.plot(medarea.mid, medarea["median"], color=COLORS["coral"], marker="o", lw=2, label="10 sqm bin median")
    plt.xlabel("Floor area (sqm)"); plt.ylabel("Monthly rent (SGD)"); plt.title("Rent generally increases with floor area"); plt.legend(frameon=False)
    savefig("05_rent_vs_floor_area.png")
    age_df = pd.DataFrame({"years_since_lease_commencement": age, "rent": y}).dropna()
    age_df["age_bin"] = pd.cut(age_df.years_since_lease_commencement, bins=[-100, 0, 5, 10, 20, 30, 40, 50, 200], right=False)
    age_summary = age_df.groupby("age_bin", observed=True).rent.agg(rows="size", median="median", mean="mean").reset_index()
    age_summary["age_bin"] = age_summary["age_bin"].astype(str); save_table(age_summary, "rent_by_building_age.csv")
    plt.figure(figsize=(7, 4)); plt.plot(age_summary.age_bin, age_summary["median"], marker="o", color=COLORS["teal"])
    plt.xticks(rotation=30, ha="right"); plt.xlabel("Building age at approval (years)"); plt.ylabel("Median monthly rent (SGD)")
    plt.title("Rent by years since lease commencement"); savefig("06_rent_by_lease_commencement_age.png")
    prop_summary = {"area_sqm": {"min": num(area.min()), "median": num(area.median()), "max": num(area.max()), "spearman_approx": num(area.rank().corr(y.rank()))},
                    "years_since_lease_commencement": {"min": num(age.min()), "median": num(age.median()), "max": num(age.max()),
                                     "negative_count": int((age < 0).sum()), "missing_count": int(age.isna().sum())}}

    # Repeated observations are preserved; expose conflicts rather than deduplicating.
    features = [c for c in train_raw.columns if c != TARGET]
    rent_conflicts = train_raw.groupby(features, dropna=False)[TARGET].agg(rows="size", unique_rents="nunique", rent_min="min", rent_max="max", rent_median="median").reset_index()
    rent_conflicts = rent_conflicts[rent_conflicts.rows > 1].sort_values(["unique_rents", "rows"], ascending=False)
    save_table(rent_conflicts, "repeated_feature_rows.csv")
    repeated_groups = train_raw.groupby(features, dropna=False)[TARGET].size()
    # Exact duplicated rows are a subset of same-feature repetitions; retain as collected.

    # Train/test drift: normalized category labels, numeric KS distance, date and per-field coverage.
    drift = []
    cat_cols = ["TOWN", "BLOCK", "STREET", "FLAT_TYPE", "FLAT_MODEL", "FURNISHED"]
    num_cols = ["FLOOR_AREA_SQM", "LEASE_COMMENCE_DATE", "FEE"]
    for c in cat_cols:
        normfn = norm_flat_type if c == "FLAT_TYPE" else (norm_block if c == "BLOCK" else norm_text)
        a, b = normfn(train_raw[c]).fillna("<missing>"), normfn(test_raw[c]).fillna("<missing>")
        cats = sorted(set(a.unique()) | set(b.unique()))
        ac, bc = a.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy(), b.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
        unseen = sorted(set(b.unique()) - set(a.unique()))
        drift.append({"feature": c, "kind": "categorical", "train_nunique": a.nunique(), "test_nunique": b.nunique(),
                      "unseen_test_categories": len(unseen), "unseen_test_rows": int(b.isin(unseen).sum()),
                      "unseen_examples": " | ".join(unseen[:20]),
                      "js_divergence_bits": js_divergence(ac, bc), "ks_statistic": None})
    for c in num_cols:
        drift.append({"feature": c, "kind": "numeric", "train_nunique": train_raw[c].nunique(), "test_nunique": test_raw[c].nunique(),
                      "unseen_test_categories": None, "unseen_examples": "", "js_divergence_bits": None,
                      "ks_statistic": ks_stat(train_raw[c], test_raw[c])})
    unseen_months = sorted(set(test["_month"].dropna()) - set(train["_month"].dropna()))
    unseen_month_rows = int(test["_month"].isin(unseen_months).sum())
    drift.append({"feature": DATE, "kind": "date_month", "train_nunique": train["_month"].nunique(), "test_nunique": test["_month"].nunique(),
                  "unseen_test_categories": len(unseen_months), "unseen_test_rows": unseen_month_rows,
                  "unseen_examples": " | ".join(sorted(set(test["_month"].dropna()) - set(train["_month"].dropna()))[:20]),
                  "js_divergence_bits": js_divergence(train["_month"].value_counts(normalize=True).reindex(sorted(set(train["_month"]) | set(test["_month"])), fill_value=0).to_numpy(),
                                                       test["_month"].value_counts(normalize=True).reindex(sorted(set(train["_month"]) | set(test["_month"])), fill_value=0).to_numpy()), "ks_statistic": None})
    drift_df = pd.DataFrame(drift); save_table(drift_df, "train_test_drift.csv")
    plt.figure(figsize=(8, 4.2)); nums = drift_df[drift_df.kind == "numeric"].sort_values("ks_statistic")
    plt.barh(nums.feature, nums.ks_statistic, color=COLORS["coral"]); plt.xlim(0, 1); plt.xlabel("Two-sample KS distance (0–1)")
    plt.title("Numeric train/test distribution shift"); savefig("07_numeric_train_test_drift.png")

    # HDB block join diagnostics and age discrepancy. Guard many-to-one after validating key conflicts.
    block = pd.read_csv(DATA / "auxiliary" / "sg-hdb-block.csv", low_memory=False)
    blockk = clean_key(block)
    key = ["_town", "_block", "_street"]
    dupkey = blockk.duplicated(key, keep=False)
    coord_conflict = blockk.loc[dupkey].groupby(key, dropna=False)[["LATITUDE", "LONGITUDE"]].nunique().max(axis=1)
    if (coord_conflict > 1).any():
        raise ValueError("Ambiguous HDB auxiliary coordinates for duplicated join keys; refusing arbitrary match.")
    blockk = blockk.drop_duplicates(key, keep="first")
    combined = clean_key(allx)
    combined["_source_index"] = np.arange(len(combined))
    merged = combined.merge(blockk[key + ["LATITUDE", "LONGITUDE", "MAX_FLOOR", "YEAR_COMPLETED", "SUBZONE", "PLANNING_AREA", "REGION"]], on=key, how="left", validate="many_to_one", indicator=True)
    assert len(merged) == len(allx) and np.array_equal(merged["_source_index"].to_numpy(), np.arange(len(allx)))
    block_coverage = {"source_block_rows": len(block), "unique_join_keys": len(blockk), "duplicate_key_rows": int(dupkey.sum()),
                      "duplicate_keys_with_coordinate_conflict": int((coord_conflict > 1).sum()),
                      "train_matched_rows": int((merged.iloc[:len(train_raw)]["_merge"] == "both").sum()),
                      "train_rows": len(train_raw), "test_matched_rows": int((merged.iloc[len(train_raw):]["_merge"] == "both").sum()),
                      "test_rows": len(test_raw)}
    # Distances on merged block centroids; duplicate same coordinates in auxiliary lists are collapsed.
    mrt = pd.read_csv(DATA / "auxiliary" / "sg-mrt-stations.csv", low_memory=False)
    statuses = sorted(mrt.STATUS.astype(str).str.strip().str.lower().unique())
    mrt["_status"] = mrt.STATUS.astype(str).str.strip().str.lower()
    mrt_open = mrt[mrt._status == "open"].drop_duplicates(["LATITUDE", "LONGITUDE"])
    mrt_future = mrt[mrt._status != "open"].drop_duplicates(["LATITUDE", "LONGITUDE"])
    malls = pd.read_csv(DATA / "auxiliary" / "sg-shopping-malls.csv", low_memory=False).drop_duplicates(["LATITUDE", "LONGITUDE"])
    schools = pd.read_csv(DATA / "auxiliary" / "sg-schools.csv", low_memory=False)
    prim = primary_schools(schools, include_mixed=False).drop_duplicates(["LATITUDE", "LONGITUDE"])
    other_schools = schools.drop_duplicates(["LATITUDE", "LONGITUDE"])
    loc = merged[["LATITUDE", "LONGITUDE"]].copy()
    loc["distance_open_mrt_km"] = haversine_nearest(loc, mrt_open)
    loc["distance_nonopen_mrt_km"] = haversine_nearest(loc, mrt_future)
    loc["distance_mall_km"] = haversine_nearest(loc, malls)
    loc["distance_primary_school_km"] = haversine_nearest(loc, prim)
    loc["distance_any_school_km"] = haversine_nearest(loc, other_schools)
    cbd = pd.DataFrame({"LATITUDE": [1.2840], "LONGITUDE": [103.8510]})  # Raffles Place MRT reference point
    loc["distance_raffles_place_km"] = haversine_nearest(loc, cbd)
    # A declared, reproducible approximate CBD point; labels never imply official boundary distance.
    merged = pd.concat([merged.reset_index(drop=True), loc.drop(columns=["LATITUDE", "LONGITUDE"]).reset_index(drop=True)], axis=1)
    merged["_rent"] = pd.concat([train_raw[TARGET], pd.Series([np.nan] * len(test_raw))], ignore_index=True)
    loc_rows = []
    for nm in ["distance_open_mrt_km", "distance_nonopen_mrt_km", "distance_mall_km", "distance_primary_school_km", "distance_any_school_km", "distance_raffles_place_km"]:
        for split, sl in [("train", merged.iloc[:len(train_raw)]), ("test", merged.iloc[len(train_raw):])]:
            v = pd.to_numeric(sl[nm], errors="coerce")
            rent_corr = sl[nm].rank().corr(sl["_rent"].rank()) if split == "train" and sl[nm].notna().sum() > 1 else np.nan
            loc_rows.append({"feature": nm, "split": split, "rows": len(sl), "available_rows": int(v.notna().sum()),
                             "median_km": num(v.median()), "p90_km": num(v.quantile(.9)), "spearman_with_rent": num(rent_corr)})
    save_table(pd.DataFrame(loc_rows), "location_feature_summary.csv")
    for nm in ["distance_open_mrt_km", "distance_mall_km", "distance_primary_school_km"]:
        z = merged.iloc[:len(train_raw)][[nm, "_rent"]].dropna().copy()
        if len(z) > 0:
            z["distance_band_km"] = pd.cut(z[nm], bins=[0, .3, .6, 1, 2, 5, 50], include_lowest=True)
            q = z.groupby("distance_band_km", observed=True)["_rent"].agg(rows="size", median="median", mean="mean").reset_index()
            q["distance_band_km"] = q.distance_band_km.astype(str)
            save_table(q, nm.replace("distance_", "rent_by_" ).replace("_km", "_distance.csv"))
            if nm == "distance_open_mrt_km":
                plt.figure(figsize=(7, 4))
                plt.bar(q.distance_band_km, q["median"], color=COLORS["teal"])
                plt.ylabel("Median monthly rent (SGD)"); plt.xlabel("Distance to nearest open MRT (km)")
                plt.xticks(rotation=20, ha="right"); plt.title("Rent by distance to nearest open MRT")
                savefig("10_rent_by_open_mrt_distance.png")
    geo = merged.iloc[:len(train_raw)].groupby(["LATITUDE", "LONGITUDE"], dropna=True).agg(median_rent=("_rent", "median"), rows=("_rent", "size")).reset_index()
    save_table(geo, "block_coordinate_rent_summary.csv")
    plt.figure(figsize=(7, 6))
    pts = plt.scatter(geo.LONGITUDE, geo.LATITUDE, c=geo.median_rent, s=np.clip(geo.rows, 4, 90),
                      cmap="viridis", alpha=.75, linewidths=0)
    plt.colorbar(pts, label="Median monthly rent (SGD)")
    plt.xlabel("Longitude"); plt.ylabel("Latitude"); plt.title("Block centroids colored by median rent")
    savefig("11_block_rent_map.png")
    spatial_profile = {"mrt_statuses": statuses, "open_station_locations": len(mrt_open), "nonopen_station_locations": len(mrt_future),
                       "mall_locations": len(malls), "school_locations_all": len(other_schools), "school_locations_primary": len(prim),
                       "cbd_reference": {"label": "Raffles Place MRT approximate point", "latitude": 1.2840, "longitude": 103.8510},
                       "block_join": block_coverage,
                       "interpretation_caution": "Unadjusted distance-rent associations mix town, flat type, area and time; they are descriptive, not causal effects."}
    plt.figure(figsize=(8, 4.5))
    locsummary = pd.DataFrame(loc_rows)
    pl = locsummary[(locsummary.split == "train") & (locsummary.feature.isin(["distance_open_mrt_km", "distance_mall_km", "distance_primary_school_km"]))]
    plt.bar(["Open MRT", "Mall", "Primary school"], [locsummary.loc[(locsummary.split=="train") & (locsummary.feature==n),"spearman_with_rent"].iloc[0] for n in pl.feature], color=[COLORS["teal"],COLORS["gold"],COLORS["blue"]])
    plt.axhline(0, color="#333333", lw=.7); plt.ylabel("Spearman correlation with rent"); plt.title("Raw distance associations are descriptive only")
    savefig("08_raw_distance_rent_association.png")

    # Auxiliary temporal/macro schema and coverage only. Do not merge future macro observations.
    aux_audit = []
    for p in sorted((DATA / "auxiliary").glob("*.csv")):
        a = pd.read_csv(p, low_memory=False)
        aux_audit.append({"file": p.name, "rows": len(a), "columns": " | ".join(a.columns),
                          "missing_cells": int(a.isna().sum().sum()), "duplicate_rows": int(a.duplicated().sum()),
                          "date_min": str(pd.to_datetime(a["DATE"], errors="coerce").min().date()) if "DATE" in a else None,
                          "date_max": str(pd.to_datetime(a["DATE"], errors="coerce").max().date()) if "DATE" in a else None,
                          "year_min": num(pd.to_numeric(a["YEAR"], errors="coerce").min()) if "YEAR" in a else None,
                          "year_max": num(pd.to_numeric(a["YEAR"], errors="coerce").max()) if "YEAR" in a else None})
    save_table(pd.DataFrame(aux_audit), "auxiliary_data_audit.csv")
    coe = pd.read_csv(DATA / "auxiliary" / "sg-coe-prices.csv", low_memory=False)
    stock = pd.read_csv(DATA / "auxiliary" / "sg-stock-prices.csv", low_memory=False)
    def parse_currency(col):
        return pd.to_numeric(col.astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
    macro = {"coe": {"rows": len(coe), "date_range_year": [int(coe.YEAR.min()), int(coe.YEAR.max())],
                      "premium_parse_rate": float(parse_currency(coe.QUOTA_PREMIUM).notna().mean()),
                      "categories": int(coe.CATEGORY.nunique()), "rounds_per_month_values": sorted(coe.ROUND.dropna().unique().astype(int).tolist())},
             "stocks": {"rows": len(stock), "date_range": [str(stock.DATE.min()), str(stock.DATE.max())],
                        "symbols": int(stock.SYMBOL.nunique()), "companies": int(stock.NAME.nunique())},
             "leakage_note": "Economic series need a date-aware aggregation using only information available by approval date; same-month observations may be published later. No macro features are merged in this EDA."}
    # Split input sanity and output-preservation evidence.
    assert len(train_raw) == len(train) and len(test_raw) == len(test)
    assert test_raw.index.equals(test.index)
    baselines = median_baselines(train_raw); save_table(baselines, "baseline_validation.csv")
    summary = {"project": "CS5228 HDB monthly rent prediction", "script": "analysis/eda/overview.py",
               "inputs": {"train_rows": len(train_raw), "test_rows": len(test_raw), "train_columns": train_raw.columns.tolist(), "test_columns": test_raw.columns.tolist(),
                          "train_date_min": str(train["_date"].min().date()), "train_date_max": str(train["_date"].max().date()),
                          "test_date_min": str(test["_date"].min().date()), "test_date_max": str(test["_date"].max().date()),
                          "test_row_count_and_order_preserved": True},
               "target": target_summary, "property": prop_summary, "spatial": spatial_profile, "macro_auxiliary": macro,
               "repeated_feature_combinations": int((train_raw.groupby(features, dropna=False)[TARGET].size() > 1).sum()),
               "repeated_feature_combinations_with_multiple_rents": int((train_raw.groupby(features, dropna=False)[TARGET].nunique() > 1).sum()),
               "baseline_validation": baselines.to_dict(orient="records"),
               "interpretation_notes": ["Random validation mixes historical regimes and should not stand in for future-period performance.",
                                        "Distance correlations are unadjusted associations.",
                                        "Years since lease commencement is approval calendar year minus lease commencement year; one row is negative.",
                                        "KS and Jensen-Shannon results describe distribution distance; no significance tests are reported."]}
    source_hashes_after = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    summary["source_sha256_before"] = source_hashes_before
    summary["source_sha256_after"] = source_hashes_after
    summary["source_files_unchanged"] = source_hashes_before == source_hashes_after
    summary["duplicate_audit"] = {"exact_duplicate_rows": int(train_raw.duplicated().sum()),
                                  "feature_groups_repeated": int((repeated_groups > 1).sum()),
                                  "rows_in_repeated_feature_groups": int(repeated_groups[repeated_groups > 1].sum()),
                                  "repeated_groups_with_multiple_rents": int((train_raw.groupby(features, dropna=False)[TARGET].nunique() > 1).sum())}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=num) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "tables": len(list(TAB.glob("*.csv"))), "figures": len(list(FIG.glob("*.png"))),
                      "train_rows": len(train_raw), "test_rows": len(test_raw), "train_dates": [summary["inputs"]["train_date_min"], summary["inputs"]["train_date_max"]],
                      "test_dates": [summary["inputs"]["test_date_min"], summary["inputs"]["test_date_max"]], "block_coverage": block_coverage}, indent=2))


if __name__ == "__main__":
    main()
