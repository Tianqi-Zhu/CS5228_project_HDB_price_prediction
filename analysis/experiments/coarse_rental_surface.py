"""One coarse, fixed-origin test of changing local rental premiums."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.config import AUX, DATA
from analysis.evaluation import metrics
from analysis.experiments.dynamic_market_index import (
    OUT, REFERENCE_CATEGORICAL, REFERENCE_NUMERIC, reference_model,
)
from analysis.experiments.spatiotemporal_hybrid import FOLDS, MONTH, TARGET, spatial_frame


GRID_SIZE = 16
CELL = "_RENTAL_CELL"
TYPE = "FLAT_TYPE_CLEAN"
COMMON_TYPES = ("3-room", "4-room", "5-room", "executive")
LATE_FOLD = ("2023-12", "2025-04")


def assign_cells(rows: pd.DataFrame, limits: tuple[float, float, float, float],
                 size: int = GRID_SIZE) -> pd.DataFrame:
    """Map block coordinates to a fixed HDB-bounding-box grid."""
    lat_min, lat_max, lon_min, lon_max = limits
    if not (lat_max > lat_min and lon_max > lon_min):
        raise ValueError("Invalid grid bounds")
    result = rows.copy()
    y = np.clip(np.floor((result["LATITUDE"].to_numpy(float) - lat_min) /
                         (lat_max - lat_min) * size), 0, size - 1).astype(int)
    x = np.clip(np.floor((result["LONGITUDE"].to_numpy(float) - lon_min) /
                         (lon_max - lon_min) * size), 0, size - 1).astype(int)
    result[CELL] = y * size + x
    return result


def forecast_local_change(fitting: pd.DataFrame, future: pd.DataFrame,
                          residuals: np.ndarray) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Forecast a shrunken cell × flat-type premium change from two past windows."""
    origin = fitting[MONTH].max()
    if not (future[MONTH] > origin).all():
        raise AssertionError("Future surface request includes an observed month")
    data = fitting[[MONTH, CELL, TYPE]].copy()
    data["residual"] = np.asarray(residuals, float)
    flat_level = data.groupby([MONTH, TYPE], observed=True)["residual"].transform("mean")
    data["local_premium"] = data["residual"] - flat_level
    age = origin.ordinal - data[MONTH].astype("int64")
    recent = data[(age >= 0) & (age <= 2)].groupby([CELL, TYPE], observed=True)["local_premium"].agg(
        recent_mean="mean", recent_count="size",
    )
    previous = data[(age >= 3) & (age <= 5)].groupby([CELL, TYPE], observed=True)["local_premium"].agg(
        previous_mean="mean", previous_count="size",
    )
    slopes = recent.join(previous, how="outer")
    strength = np.minimum(slopes["recent_count"], slopes["previous_count"])
    slopes["slope"] = ((slopes["recent_mean"] - slopes["previous_mean"]) / 3.0 *
                       strength / (strength + 20.0)).fillna(0.0)
    key = pd.MultiIndex.from_frame(future[[CELL, TYPE]])
    slope = slopes["slope"].reindex(key).fillna(0.0).to_numpy(float)
    available = slopes["recent_count"].notna() & slopes["previous_count"].notna()
    coverage = available.reindex(key).fillna(False).to_numpy(bool)
    horizon = future[MONTH].astype("int64").to_numpy() - origin.ordinal
    delta = slope * (1.0 - 0.7 ** horizon) / 0.3
    return delta, coverage, slopes.reset_index()


def occupancy(fitting: pd.DataFrame, supported_cells: int) -> dict:
    common = fitting[fitting[TYPE].isin(COMMON_TYPES)]
    counts = common.groupby([MONTH, CELL, TYPE], observed=True).size()
    months = fitting[MONTH].nunique()
    possible = months * supported_cells * len(COMMON_TYPES)
    return {
        "n_months": months, "supported_cells": supported_cells,
        "common_flat_types": len(COMMON_TYPES),
        "observed_cell_type_months": len(counts),
        "empty_fraction": 1.0 - len(counts) / possible,
        "median_transactions_per_observed_cell_type_month": float(counts.median()),
    }


def evaluate() -> None:
    prior = pd.read_csv(OUT / "validation_predictions.csv")
    blocks = pd.read_csv(AUX / "sg-hdb-block.csv")
    bounds = (float(blocks["LATITUDE"].min()), float(blocks["LATITUDE"].max()),
              float(blocks["LONGITUDE"].min()), float(blocks["LONGITUDE"].max()))
    support = assign_cells(blocks, bounds)[CELL].nunique()
    frame = assign_cells(spatial_frame(pd.read_csv(DATA / "train.csv")), bounds)
    fold_scores, occupancy_rows, predictions = [], [], []
    for start_text, end_text in [*FOLDS, LATE_FOLD]:
        start, end = pd.Period(start_text, freq="M"), pd.Period(end_text, freq="M")
        fitting = frame[frame[MONTH] < start].copy()
        validation = frame[(frame[MONTH] >= start) & (frame[MONTH] < end)].copy()
        reference = reference_model(fitting)
        columns = REFERENCE_CATEGORICAL + REFERENCE_NUMERIC
        residual = fitting[TARGET].to_numpy(float) - reference.predict(fitting[columns])
        local_delta, coverage, _ = forecast_local_change(fitting, validation, residual)
        base = prior[prior["validation_start"].eq(start_text)].set_index("row_id").loc[validation.index]
        if not np.allclose(base[TARGET], validation[TARGET]):
            raise AssertionError("Saved row predictions are misaligned")
        cb = base["catboost"].to_numpy(float)
        flat = cb + base["delta_flat_type_damped"].to_numpy(float)
        variants = {
            "catboost": cb,
            "catboost+flat_type_damped": flat,
            "catboost+local_16x16": cb + local_delta,
            "catboost+flat_type_damped+local_16x16": flat + local_delta,
        }
        for label, values in variants.items():
            fold_scores.append({"validation_start": start_text, "model": label,
                                "n": len(validation), **metrics(validation[TARGET], values)})
        occupancy_rows.append({"validation_start": start_text, **occupancy(fitting, support),
                               "validation_rows_with_two_window_local_history_pct":
                               100.0 * float(coverage.mean())})
        predictions.append(pd.DataFrame({
            "validation_start": start_text, "row_id": validation.index,
            "actual": validation[TARGET].to_numpy(float),
            "catboost": cb, "flat_type_damped": flat,
            "local_delta": local_delta, "covered": coverage,
        }))
        print(f"{start_text}: base {fold_scores[-4]['rmse_sgd']:.2f}; local {fold_scores[-2]['rmse_sgd']:.2f}; "
              f"coverage {coverage.mean():.1%}", flush=True)
    scores = pd.concat(predictions, ignore_index=True)
    disjoint = scores[scores["validation_start"].isin([start for start, _ in FOLDS])]
    pooled = pd.DataFrame([
        {"model": label, "n": len(disjoint), **metrics(disjoint["actual"], values)}
        for label, values in {
            "catboost": disjoint["catboost"],
            "catboost+flat_type_damped": disjoint["flat_type_damped"],
            "catboost+local_16x16": disjoint["catboost"] + disjoint["local_delta"],
            "catboost+flat_type_damped+local_16x16": disjoint["flat_type_damped"] + disjoint["local_delta"],
        }.items()
    ]).sort_values("rmse_sgd")
    pd.DataFrame(fold_scores).to_csv(OUT / "surface_fold_scores.csv", index=False, float_format="%.4f")
    pd.DataFrame(occupancy_rows).to_csv(OUT / "surface_occupancy.csv", index=False, float_format="%.4f")
    pooled.to_csv(OUT / "surface_pooled_scores.csv", index=False, float_format="%.4f")
    scores.to_csv(OUT / "surface_predictions.csv", index=False, float_format="%.4f")
    print(pooled.to_string(index=False), flush=True)


if __name__ == "__main__":
    evaluate()
