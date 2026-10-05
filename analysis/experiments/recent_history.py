"""Strictly-prior recent rental statistics and CatBoost ablation."""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from analysis.config import DATA
from analysis.data import sha256
from analysis.evaluation import metrics
from analysis.experiments.dynamic_market_index import (
    OUT, FEATURES, TREE_SETTINGS, fit_tree,
)
from analysis.experiments.spatiotemporal_hybrid import (
    FOLDS, HISTORY_GRAINS, MONTH, TARGET, historical_features, model_inputs,
    spatial_frame,
)


WINDOWS = (3, 6, 12)
GRAINS = {name: HISTORY_GRAINS[name] for name in ("BLOCK_FLAT", "AREA_FLAT")}
LATE_FOLD = ("2023-12", "2025-04")
RECENT_FEATURES = [
    f"RECENT_{grain}_{window}M_{stat}"
    for grain in GRAINS for window in WINDOWS for stat in ("MEAN", "COUNT")
] + [f"RECENT_BLOCK_FLAT_{window}M_SHRUNK_MEAN" for window in WINDOWS]


def _prefix_lookup(monthly: pd.DataFrame, group_ids: np.ndarray,
                   upper_months: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = pd.DataFrame({"group": group_ids, "lookup_month": upper_months,
                         "_row": np.arange(len(group_ids))})
    right = monthly[["group", "month_ordinal", "cum_sum", "cum_count"]]
    joined = pd.merge_asof(
        left.sort_values("lookup_month"), right.sort_values("month_ordinal"),
        by="group", left_on="lookup_month", right_on="month_ordinal",
        direction="backward",
    ).sort_values("_row")
    return joined["cum_sum"].fillna(0).to_numpy(float), joined["cum_count"].fillna(0).to_numpy(float)


def recent_history_features(fitting: pd.DataFrame, prediction: pd.DataFrame
                            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create calendar-window means/counts with no same-month target leakage."""
    if TARGET not in fitting or fitting.empty or prediction.empty:
        raise ValueError("Nonempty fitting labels and prediction rows required")
    origin = fitting[MONTH].max()
    if not (prediction[MONTH] > origin).all():
        raise AssertionError("Prediction month is not later than all fitting labels")
    outputs = []
    for queries, frozen in ((fitting, False), (prediction, True)):
        feature_frame = pd.DataFrame(index=queries.index)
        for grain, keys in GRAINS.items():
            monthly = fitting.groupby(keys + [MONTH], dropna=False, observed=True)[TARGET].agg(
                monthly_sum="sum", monthly_count="size",
            ).reset_index()
            monthly["group"] = pd.util.hash_pandas_object(monthly[keys], index=False).to_numpy()
            monthly["month_ordinal"] = monthly[MONTH].astype("int64")
            monthly = monthly.sort_values(["group", "month_ordinal"])
            monthly["cum_sum"] = monthly.groupby("group")["monthly_sum"].cumsum()
            monthly["cum_count"] = monthly.groupby("group")["monthly_count"].cumsum()
            query_groups = pd.util.hash_pandas_object(queries[keys], index=False).to_numpy()
            upper = (
                np.full(len(queries), origin.ordinal, dtype=np.int64) if frozen
                else queries[MONTH].astype("int64").to_numpy() - 1
            )
            upper_sum, upper_count = _prefix_lookup(monthly, query_groups, upper)
            for window in WINDOWS:
                lower_sum, lower_count = _prefix_lookup(monthly, query_groups, upper - window)
                total = upper_sum - lower_sum
                count = upper_count - lower_count
                prefix = f"RECENT_{grain}_{window}M"
                feature_frame[f"{prefix}_COUNT"] = count
                feature_frame[f"{prefix}_MEAN"] = np.divide(
                    total, count, out=np.full(len(count), np.nan), where=count > 0,
                )
                feature_frame[f"_{prefix}_SUM"] = total
        for window in WINDOWS:
            block = f"RECENT_BLOCK_FLAT_{window}M"
            area = f"RECENT_AREA_FLAT_{window}M"
            count = feature_frame[f"{block}_COUNT"].to_numpy(float)
            total = feature_frame[f"_{block}_SUM"].to_numpy(float)
            parent = feature_frame[f"{area}_MEAN"].to_numpy(float)
            feature_frame[f"{block}_SHRUNK_MEAN"] = (total + 5.0 * parent) / (count + 5.0)
            feature_frame = feature_frame.drop(columns=f"_{block}_SUM")
            feature_frame = feature_frame.drop(columns=f"_{area}_SUM")
        outputs.append(feature_frame[RECENT_FEATURES])
    return outputs[0], outputs[1]


def evaluate() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prior = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    train_path = DATA / "train.csv"
    if prior["source_sha256"]["train.csv"] != sha256(train_path):
        raise ValueError("Training CSV differs from the earlier model comparison")
    baseline = pd.read_csv(OUT / "validation_predictions.csv")
    frame = spatial_frame(pd.read_csv(train_path))
    folds = []
    monthly = []
    rows = []
    elapsed = []
    for start_text, end_text in [*FOLDS, LATE_FOLD]:
        start, end = pd.Period(start_text, freq="M"), pd.Period(end_text, freq="M")
        fitting = frame[frame[MONTH] < start].copy()
        validation = frame[(frame[MONTH] >= start) & (frame[MONTH] < end)].copy()
        base = baseline[baseline["validation_start"].eq(start_text)].set_index("row_id").loc[validation.index]
        if not np.allclose(base[TARGET], validation[TARGET]):
            raise AssertionError("Earlier predictions are not aligned with validation rows")
        fit_history, val_history = historical_features(fitting, validation)
        fit_recent, val_recent = recent_history_features(fitting, validation)
        x_fit, x_val = model_inputs(
            fitting.join(fit_history).join(fit_recent),
            validation.join(val_history).join(val_recent),
            FEATURES + RECENT_FEATURES,
        )
        predicted, seconds = fit_tree("catboost", fitting, validation, x_fit, x_val)
        elapsed.append({"validation_start": start_text, "fit_predict_seconds": seconds})
        for label, values in (("catboost", base["catboost"].to_numpy(float)),
                              ("catboost_recent", predicted)):
            folds.append({"validation_start": start_text, "model": label, "n": len(validation),
                          **metrics(validation[TARGET], values)})
            for month in validation[MONTH].unique():
                mask = validation[MONTH].eq(month).to_numpy()
                monthly.append({"validation_start": start_text, "month": str(month),
                                "model": label, "n": int(mask.sum()),
                                **metrics(validation.loc[mask, TARGET], values[mask])})
        rows.append(pd.DataFrame({"validation_start": start_text,
                                  "row_id": validation.index,
                                  "month": validation[MONTH].astype(str).to_numpy(),
                                  "actual": validation[TARGET].to_numpy(float),
                                  "catboost": base["catboost"].to_numpy(float),
                                  "catboost_recent": predicted}))
        print(f"{start_text}: base {folds[-2]['rmse_sgd']:.2f}; recent {folds[-1]['rmse_sgd']:.2f}; {seconds:.1f}s", flush=True)
    scores = pd.concat(rows, ignore_index=True)
    disjoint = scores[scores["validation_start"].isin([start for start, _ in FOLDS])]
    pooled = pd.DataFrame([
        {"model": name, "n": len(disjoint), **metrics(disjoint["actual"], disjoint[name])}
        for name in ("catboost", "catboost_recent")
    ])
    pd.DataFrame(folds).to_csv(OUT / "recent_history_fold_scores.csv", index=False, float_format="%.4f")
    pd.DataFrame(monthly).to_csv(OUT / "recent_history_monthly_scores.csv", index=False, float_format="%.4f")
    pooled.to_csv(OUT / "recent_history_pooled_scores.csv", index=False, float_format="%.4f")
    pd.DataFrame(elapsed).to_csv(OUT / "recent_history_fit_durations.csv", index=False, float_format="%.2f")
    scores.to_csv(OUT / "recent_history_predictions.csv", index=False, float_format="%.4f")
    print(pooled.to_string(index=False), flush=True)


if __name__ == "__main__":
    evaluate()
