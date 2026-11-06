"""Full-history tree comparison and fixed-origin rental-index forecast.

The 2021 hedonic reference is frozen before every forecasted market month. At
each validation origin, all target-derived index values end before the origin.
The experiment deliberately tests a coarse market representation before a
gridded rental surface or PCA is considered.
"""

from __future__ import annotations

import json
import time

import catboost
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from analysis.config import DATA, REPORTS
from analysis.data import sha256
from analysis.evaluation import metrics
from analysis.experiments.spatiotemporal_hybrid import (
    BASE_CATEGORICAL, BASE_FEATURES, FOLDS, HISTORY_FEATURES, MONTH,
    ROW_MODEL_SETTINGS, TARGET, historical_features, model_inputs, spatial_frame,
)


OUT = REPORTS / "dynamic_market_index"
AREA = "PLANNING_AREA"
TYPE = "FLAT_TYPE_CLEAN"
GROUP = [AREA, TYPE]
LATE_FOLD = ("2023-12", "2025-04")  # Overlaps the three disjoint folds.
REFERENCE_END = pd.Period("2022-01", freq="M")
SHRINK_COUNT = 20.0
FEATURES = BASE_FEATURES + HISTORY_FEATURES
REFERENCE_CATEGORICAL = [TYPE, "FLAT_MODEL_CLEAN", "TOWN_CLEAN"]
REFERENCE_NUMERIC = ["FLOOR_AREA_SQM", "FLAT_AGE_YEARS"]
TREE_SETTINGS = {
    "hist_gradient_boosting": ROW_MODEL_SETTINGS,
    "lightgbm": {
        "n_estimators": 500, "learning_rate": 0.05, "num_leaves": 31,
        "min_child_samples": 30, "reg_lambda": 3.0, "verbosity": -1,
        "n_jobs": 4, "random_state": 42,
    },
    "catboost": {
        "iterations": 600, "depth": 7, "learning_rate": 0.05,
        "l2_leaf_reg": 5.0, "loss_function": "RMSE", "verbose": False,
        "thread_count": 4, "random_seed": 42, "allow_writing_files": False,
    },
}
INDEX_METHODS = ("persistence", "recent3", "damped")
INDEX_LEVELS = ("flat_type", "area_flat_type")


def reference_model(fitting: pd.DataFrame) -> Pipeline:
    """Fix a composition reference using 2021 labels only."""
    early = fitting[fitting[MONTH] < REFERENCE_END]
    if early.empty or early[MONTH].max() >= REFERENCE_END:
        raise ValueError("A 2021 reference period is required")
    features = REFERENCE_CATEGORICAL + REFERENCE_NUMERIC
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), REFERENCE_CATEGORICAL),
        ("num", SimpleImputer(strategy="median"), REFERENCE_NUMERIC),
    ])
    model = Pipeline([("prep", prep), ("ridge", Ridge(alpha=100.0))])
    model.fit(early[features], early[TARGET].to_numpy(float))
    return model


def observed_indices(rows: pd.DataFrame, residuals: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return flat-type and shrunken area × flat-type residual indices."""
    data = rows[[MONTH, AREA, TYPE]].copy()
    data["residual"] = np.asarray(residuals, float)
    flat = data.groupby([MONTH, TYPE], observed=True)["residual"].agg(
        flat_sum="sum", flat_count="size",
    ).reset_index()
    flat["flat_level"] = flat["flat_sum"] / flat["flat_count"]
    group = data.groupby([MONTH, AREA, TYPE], observed=True)["residual"].agg(
        group_sum="sum", group_count="size",
    ).reset_index().merge(flat[[MONTH, TYPE, "flat_level"]], on=[MONTH, TYPE], validate="many_to_one")
    group["group_level"] = (
        group["group_sum"] + SHRINK_COUNT * group["flat_level"]
    ) / (group["group_count"] + SHRINK_COUNT)
    return flat, group


def forecast_level(values: np.ndarray, horizon: int, method: str) -> float:
    """Fixed, inexpensive forecasts of one monthly index."""
    arr = np.asarray(values, float)
    if len(arr) == 0 or horizon < 1:
        raise ValueError("Forecast requires history and a positive horizon")
    if method == "persistence":
        return float(arr[-1])
    recent = float(arr[-min(3, len(arr)):].mean())
    if method == "recent3":
        return recent
    if method != "damped":
        raise ValueError(f"Unknown index forecast: {method}")
    if len(arr) < 6:
        return recent
    previous = float(arr[-6:-3].mean())
    slope = float(np.clip((recent - previous) / 3.0, -100.0, 100.0))
    return recent + slope * (1.0 - 0.7 ** horizon) / 0.3


def forecast_indices(fitting: pd.DataFrame, future: pd.DataFrame,
                     flat: pd.DataFrame, group: pd.DataFrame,
                     method: str, level: str) -> pd.DataFrame:
    """Forecast every requested row-group/month from the fitting snapshot."""
    origin = fitting[MONTH].max()
    if not (future[MONTH] > origin).all():
        raise AssertionError("Market forecast includes an observed or earlier month")
    requested = future[[MONTH, AREA, TYPE]].drop_duplicates().copy()
    flat_history = {
        kind: part.sort_values(MONTH)["flat_level"].to_numpy(float)
        for kind, part in flat.groupby(TYPE, observed=True)
    }
    group_history = {
        key: part.sort_values(MONTH)[[MONTH, "group_level", "flat_level"]]
        for key, part in group.groupby(GROUP, observed=True)
    }
    rows = []
    for month, area, kind in requested.itertuples(index=False, name=None):
        history = flat_history.get(kind)
        if history is None:
            rows.append((month, area, kind, 0.0, 0.0, 0.0))
            continue
        horizon = month.ordinal - origin.ordinal
        base_flat = float(history[-1])
        predicted_flat = forecast_level(history, horizon, method)
        premium = 0.0
        if level == "area_flat_type":
            local = group_history.get((area, kind))
            if local is not None and not local.empty:
                last = local.iloc[-1]
                age = origin.ordinal - last[MONTH].ordinal
                premium = float(last["group_level"] - last["flat_level"]) * 0.9 ** age
        elif level != "flat_type":
            raise ValueError(f"Unknown index level: {level}")
        current = base_flat + premium
        predicted = predicted_flat + (premium if method == "persistence" else premium * 0.9 ** horizon)
        rows.append((month, area, kind, current, predicted, predicted - current))
    return pd.DataFrame(rows, columns=[MONTH, AREA, TYPE, "origin_level", "forecast_level", "delta"])


def attach_forecast(rows: pd.DataFrame, forecast: pd.DataFrame, value: str) -> np.ndarray:
    key = pd.MultiIndex.from_frame(rows[[MONTH, AREA, TYPE]])
    lookup = forecast.set_index([MONTH, AREA, TYPE])[value]
    result = lookup.reindex(key).to_numpy(float)
    if not np.isfinite(result).all():
        raise AssertionError("A requested market forecast was not generated")
    return result


def fit_tree(label: str, fitting: pd.DataFrame, validation: pd.DataFrame,
             x_fit: pd.DataFrame, x_validation: pd.DataFrame) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    actual = fitting[TARGET].to_numpy(float)
    if label == "hist_gradient_boosting":
        model = HistGradientBoostingRegressor(**TREE_SETTINGS[label])
        model.fit(x_fit, actual)
        predicted = model.predict(x_validation)
    elif label == "lightgbm":
        model = lgb.LGBMRegressor(**TREE_SETTINGS[label])
        model.fit(x_fit, actual, categorical_feature=[c for c in x_fit if c in BASE_CATEGORICAL])
        predicted = model.predict(x_validation)
    elif label == "catboost":
        train = x_fit.copy()
        future = x_validation.copy()
        categories = [c for c in train if c in BASE_CATEGORICAL]
        for col in categories:
            train[col] = train[col].astype("string").fillna("__missing__").astype(str)
            future[col] = future[col].astype("string").fillna("__missing__").astype(str)
        model = catboost.CatBoostRegressor(**TREE_SETTINGS[label])
        model.fit(train, actual, cat_features=categories)
        predicted = model.predict(future)
    else:
        raise ValueError(label)
    return np.asarray(predicted, float), time.perf_counter() - start


def evaluation() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train_path = DATA / "train.csv"
    frame = spatial_frame(pd.read_csv(train_path))
    predictions, group_scores, durations = [], [], []
    for start_text, end_text in [*FOLDS, LATE_FOLD]:
        start, end = pd.Period(start_text, freq="M"), pd.Period(end_text, freq="M")
        fitting = frame[frame[MONTH] < start].copy()
        validation = frame[(frame[MONTH] >= start) & (frame[MONTH] < end)].copy()
        if fitting.empty or validation.empty or fitting[MONTH].max() >= validation[MONTH].min():
            raise AssertionError("Invalid fixed-origin fold")
        print(f"Fold {start_text}: {len(fitting)} fit, {len(validation)} validation", flush=True)
        fit_history, val_history = historical_features(fitting, validation)
        x_fit, x_val = model_inputs(
            fitting.join(fit_history), validation.join(val_history), FEATURES,
        )
        reference = reference_model(fitting)
        reference_columns = REFERENCE_CATEGORICAL + REFERENCE_NUMERIC
        fit_residual = fitting[TARGET].to_numpy(float) - reference.predict(fitting[reference_columns])
        val_residual = validation[TARGET].to_numpy(float) - reference.predict(validation[reference_columns])
        fit_flat, fit_group = observed_indices(fitting, fit_residual)
        val_flat, val_group = observed_indices(validation, val_residual)
        if fit_group[MONTH].max() >= validation[MONTH].min():
            raise AssertionError("Market index consumed validation labels")
        fold = validation[[MONTH, AREA, TYPE, TARGET]].copy()
        fold["row_id"] = validation.index.to_numpy()
        fold["validation_start"] = start_text
        for label in TREE_SETTINGS:
            predicted, seconds = fit_tree(label, fitting, validation, x_fit, x_val)
            fold[label] = predicted
            durations.append({"validation_start": start_text, "model": label, "fit_predict_seconds": seconds})
            score = metrics(fold[TARGET], predicted)
            print(f"  {label}: RMSE {score['rmse_sgd']:.2f}; {seconds:.1f}s", flush=True)
        for method in INDEX_METHODS:
            for level in INDEX_LEVELS:
                name = f"{level}_{method}"
                forecast = forecast_indices(fitting, validation, fit_flat, fit_group, method, level)
                fold[f"delta_{name}"] = attach_forecast(validation, forecast, "delta")
                observed = val_group[[MONTH, AREA, TYPE, "group_level", "group_count"]].merge(
                    forecast, on=[MONTH, AREA, TYPE], how="inner", validate="one_to_one",
                )
                observed = observed[observed["group_count"] >= 10]
                if not observed.empty:
                    error = observed["forecast_level"] - observed["group_level"]
                    group_scores.append({
                        "validation_start": start_text, "index": name,
                        "n_group_months": len(observed),
                        "rmse_group_sgd": float(np.sqrt(np.mean(error ** 2))),
                        "weighted_rmse_sgd": float(np.sqrt(np.average(error ** 2, weights=observed["group_count"]))),
                    })
        predictions.append(fold)
    all_rows = pd.concat(predictions, ignore_index=True)
    disjoint = all_rows[all_rows["validation_start"].isin([start for start, _ in FOLDS])]
    row_models = pd.DataFrame([
        {"model": label, "n": len(disjoint), **metrics(disjoint[TARGET], disjoint[label])}
        for label in TREE_SETTINGS
    ]).sort_values("rmse_sgd")
    best = str(row_models.iloc[0]["model"])
    print(f"Selected row model from disjoint folds: {best}", flush=True)
    candidates = {label: (lambda rows, label=label: rows[label]) for label in TREE_SETTINGS}
    for method in INDEX_METHODS:
        if method == "persistence":
            continue
        for level in INDEX_LEVELS:
            name = f"{level}_{method}"
            candidates[f"{best}+{name}"] = lambda rows, name=name: rows[best] + rows[f"delta_{name}"]
    fold_scores = []
    monthly_scores = []
    for start_text, part in all_rows.groupby("validation_start", sort=False):
        for candidate, predict in candidates.items():
            values = predict(part)
            fold_scores.append({"validation_start": start_text, "model": candidate,
                                "n": len(part), **metrics(part[TARGET], values)})
            for month, month_rows in part.groupby(MONTH, sort=True):
                monthly_scores.append({"validation_start": start_text, "month": str(month),
                                       "model": candidate, "n": len(month_rows),
                                       **metrics(month_rows[TARGET], predict(month_rows))})
    pooled = pd.DataFrame([
        {"model": candidate, "n": len(disjoint), **metrics(disjoint[TARGET], predict(disjoint))}
        for candidate, predict in candidates.items()
    ]).sort_values("rmse_sgd")
    baseline_rmse = float(pooled.loc[pooled["model"].eq(best), "rmse_sgd"].iloc[0])
    pooled["delta_rmse_vs_row_model_sgd"] = pooled["rmse_sgd"] - baseline_rmse
    row_models.to_csv(OUT / "row_model_scores.csv", index=False, float_format="%.4f")
    pd.DataFrame(fold_scores).to_csv(OUT / "fold_scores.csv", index=False, float_format="%.4f")
    pd.DataFrame(monthly_scores).to_csv(OUT / "monthly_scores.csv", index=False, float_format="%.4f")
    pooled.to_csv(OUT / "pooled_scores.csv", index=False, float_format="%.4f")
    pd.DataFrame(group_scores).to_csv(OUT / "index_forecast_scores.csv", index=False, float_format="%.4f")
    pd.DataFrame(durations).to_csv(OUT / "fit_durations.csv", index=False, float_format="%.2f")
    keep = ["validation_start", "row_id", MONTH, AREA, TYPE, TARGET, *TREE_SETTINGS,
            *(f"delta_{level}_{method}" for method in INDEX_METHODS for level in INDEX_LEVELS)]
    export = all_rows[keep].copy()
    export[MONTH] = export[MONTH].astype(str)
    export.to_csv(OUT / "validation_predictions.csv", index=False, float_format="%.4f")
    disjoint_folds = [start for start, _ in FOLDS]
    fold_table = pd.DataFrame(fold_scores)
    group_table = pd.DataFrame(group_scores)
    gate_candidates = []
    for method in ("recent3", "damped"):
        for level in INDEX_LEVELS:
            candidate = f"{best}+{level}_{method}"
            row_improves = (
                pooled.loc[pooled["model"].eq(candidate), "rmse_sgd"].iloc[0]
                <= 0.99 * baseline_rmse
            ) and all(
                fold_table.loc[(fold_table["validation_start"] == start) & (fold_table["model"] == candidate), "rmse_sgd"].iloc[0]
                < fold_table.loc[(fold_table["validation_start"] == start) & (fold_table["model"] == best), "rmse_sgd"].iloc[0]
                for start in disjoint_folds
            )
            forecast_improves = all(
                group_table.loc[(group_table["validation_start"] == start) & (group_table["index"] == f"{level}_{method}"), "weighted_rmse_sgd"].iloc[0]
                < group_table.loc[(group_table["validation_start"] == start) & (group_table["index"] == f"{level}_persistence"), "weighted_rmse_sgd"].iloc[0]
                for start in disjoint_folds
            )
            if row_improves and forecast_improves:
                gate_candidates.append(candidate)
    result = {
        "question": "Do a strong historical tree or a coarse forecast rental index improve fixed-origin rent RMSE?",
        "source_sha256": {"train.csv": sha256(train_path)},
        "folds": [*FOLDS, LATE_FOLD], "overlap_fold": LATE_FOLD,
        "selected_row_model": best, "row_model_settings": TREE_SETTINGS,
        "reference": "2021-only Ridge hedonic fit; flat-type and area × flat-type monthly mean residual indices",
        "index": {"shrink_count": SHRINK_COUNT, "methods": INDEX_METHODS,
                  "levels": INDEX_LEVELS, "damped_phi": 0.7, "group_premium_phi": 0.9},
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
                     "lightgbm": lgb.__version__, "catboost": catboost.__version__},
        "pooled_scores": pooled.to_dict(orient="records"),
        "grid_gate_passed": bool(gate_candidates),
        "grid_gate_candidates": gate_candidates,
        "note": "16-month overlapping fold is reported separately and never pooled; no validation labels feed market forecasts.",
    }
    (OUT / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(pooled.to_string(index=False), flush=True)
    print(f"Saved experiment in {OUT}", flush=True)


if __name__ == "__main__":
    evaluation()
