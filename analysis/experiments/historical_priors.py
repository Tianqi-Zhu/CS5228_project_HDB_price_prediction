"""Hierarchical recent-rent priors for repeated properties and locations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from analysis.pipeline.auxiliary import build_auxiliary_features
from analysis.pipeline.config import DATA, DATE, EXPERIMENT_REPORTS, TARGET
from analysis.pipeline.data import check_output_path, load_main, temporal_split


LEVELS = [
    (["TOWN_CLEAN", "FLAT_TYPE_CLEAN"], 80.0),
    (["STREET_CLEAN", "FLAT_TYPE_CLEAN"], 35.0),
    (["ADDRESS_KEY", "FLAT_TYPE_CLEAN"], 18.0),
    (["ADDRESS_KEY", "FLAT_TYPE_CLEAN", "FLAT_MODEL_CLEAN"], 10.0),
    (["ADDRESS_KEY", "FLAT_TYPE_CLEAN", "FLAT_MODEL_CLEAN", "AREA_BIN_5"], 6.0),
    (["ADDRESS_KEY", "FLAT_TYPE_CLEAN", "FLAT_MODEL_CLEAN", "AREA_BIN_5", "LEASE_COMMENCE_DATE"], 3.0),
]


def _prepare(frame):
    frame = frame.copy()
    frame["AREA_BIN_5"] = (pd.to_numeric(frame["FLOOR_AREA_SQM"], errors="coerce") / 5).round() * 5
    return frame


def _monthly_baseline(fitting, validation, method, months):
    monthly = (fitting.groupby(DATE, as_index=False)[TARGET].mean()
               .assign(month=lambda x: pd.to_datetime(x[DATE], format="%Y-%m"))
               .sort_values("month"))
    recent = monthly.tail(months)
    validation_month = pd.to_datetime(validation[DATE], format="%Y-%m")
    if method == "constant":
        return np.full(len(validation), recent[TARGET].mean(), dtype=float)
    x = ((recent["month"].dt.year - 2021) * 12 + recent["month"].dt.month - 1).to_numpy(float)
    y = recent[TARGET].to_numpy(float)
    slope, intercept = np.polyfit(x, y, 1)
    slope = float(np.clip(slope, -5.0, 15.0))
    xv = ((validation_month.dt.year - 2021) * 12 + validation_month.dt.month - 1).to_numpy(float)
    return intercept + slope * xv


def hierarchical_prediction(
    fitting, validation, *, base_method="constant", base_months=3,
    premium_window=24, alpha_scale=1.0, decay_half_life=None,
):
    fitting, validation = _prepare(fitting), _prepare(validation)
    month_mean = fitting.groupby(DATE)[TARGET].transform("mean")
    fitting["_PREMIUM"] = fitting[TARGET] - month_mean
    cutoff = pd.to_datetime(validation[DATE].min(), format="%Y-%m")
    dates = pd.to_datetime(fitting[DATE], format="%Y-%m")
    premium_fit = fitting.loc[dates >= cutoff - pd.DateOffset(months=premium_window)].copy()
    if decay_half_life:
        premium_dates = pd.to_datetime(premium_fit[DATE], format="%Y-%m")
        age_months = ((cutoff.year - premium_dates.dt.year) * 12 +
                      cutoff.month - premium_dates.dt.month).to_numpy(float)
        weights = np.exp(-np.log(2) * age_months / decay_half_life)
        premium_fit["_WEIGHT"] = weights / weights.mean()
    else:
        premium_fit["_WEIGHT"] = 1.0
    premium_fit["_WEIGHTED_PREMIUM"] = premium_fit["_PREMIUM"] * premium_fit["_WEIGHT"]
    prediction = _monthly_baseline(fitting, validation, base_method, base_months)
    premium = np.zeros(len(validation), dtype=float)
    coverage = {}
    for columns, base_alpha in LEVELS:
        stats = premium_fit.groupby(columns, dropna=False).agg(
            weighted_sum=("_WEIGHTED_PREMIUM", "sum"),
            weight_sum=("_WEIGHT", "sum"),
            count=("_PREMIUM", "size"),
        ).reset_index()
        matched = validation[columns].merge(stats, how="left", on=columns, sort=False)
        count = matched["count"].fillna(0).to_numpy(float)
        total = matched["weighted_sum"].fillna(0).to_numpy(float)
        weight = matched["weight_sum"].fillna(0).to_numpy(float)
        alpha = base_alpha * alpha_scale
        premium = np.where(count > 0, (total + alpha * premium) / (weight + alpha), premium)
        coverage["+".join(columns)] = float(np.mean(count > 0))
    return prediction + premium, coverage


def score(actual, prediction):
    return {
        "rmse": float(mean_squared_error(actual, prediction) ** 0.5),
        "mae": float(mean_absolute_error(actual, prediction)),
        "bias": float(np.mean(prediction - actual)),
    }


def search(frame, validation_start="2024-04"):
    fitting, validation = temporal_split(frame, validation_start)
    actual = validation[TARGET].to_numpy(float)
    rows = []
    # Search the useful region found by the initial broad sweep, including
    # recency-weighted histories for changing neighbourhood rent premiums.
    for method in ("linear",):
        for base_months in (6, 12):
            for premium_window in (24, 36, 48):
                for alpha_scale in (0.6, 1.0, 1.7, 2.5):
                    for decay_half_life in (None, 6, 12, 18, 24):
                        prediction, coverage = hierarchical_prediction(
                            fitting, validation, base_method=method, base_months=base_months,
                            premium_window=premium_window, alpha_scale=alpha_scale,
                            decay_half_life=decay_half_life,
                        )
                        rows.append({
                            "base_method": method, "base_months": base_months,
                            "premium_window": premium_window, "alpha_scale": alpha_scale,
                            "decay_half_life": decay_half_life,
                            **score(actual, prediction), "prediction": prediction, "coverage": coverage,
                        })
    return fitting, validation, sorted(rows, key=lambda row: row["rmse"])


def run(output_dir=EXPERIMENT_REPORTS / "deep_learning" / "historical_priors"):
    output_dir = check_output_path(Path(output_dir)); output_dir.mkdir(parents=True, exist_ok=True)
    frame = build_auxiliary_features(load_main(DATA / "train.csv", require_target=True))
    fitting, validation, rows = search(frame)
    serializable = [{key: value for key, value in row.items() if key != "prediction"} for row in rows]
    pd.DataFrame([{key: value for key, value in row.items() if key not in ("prediction", "coverage")}
                  for row in rows]).to_csv(output_dir / "search.csv", index=False)
    best = rows[0]
    pd.DataFrame({"index": validation.index, DATE: validation[DATE].to_numpy(),
                  "actual": validation[TARGET].to_numpy(), "prediction": best["prediction"]}).to_csv(
                      output_dir / "predictions.csv", index=False)
    artifact = {
        "model": "hierarchical_historical_prior", "metrics": score(validation[TARGET], best["prediction"]),
        "parameters": {key: best[key] for key in (
            "base_method", "base_months", "premium_window", "alpha_scale", "decay_half_life",
        )},
        "coverage": best["coverage"], "top_results": serializable[:20],
    }
    (output_dir / "results.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(artifact, indent=2))
    return artifact


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    return run(args.output_dir or EXPERIMENT_REPORTS / "deep_learning" / "historical_priors")


if __name__ == "__main__":
    main()
