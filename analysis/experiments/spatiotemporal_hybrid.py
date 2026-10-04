"""Fixed-origin evaluation of a hierarchical spatiotemporal rent model.

The row model predicts an individual rent from property and static spatial
features. Two optional additions are evaluated:

* leakage-safe historical rent summaries computed only from earlier months;
* a recursive planning-area x flat-type forecast of row-model residuals.

Unlike the exploratory one-step-ahead panel analysis, validation labels are
never fed back during a fold. The same rule is used when producing test-set
predictions from all labeled training data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from analysis.pipeline.auxiliary import build_auxiliary_features
from analysis.pipeline.config import AUX, DATA, FEATURE_SETS, REPORTS
from analysis.pipeline.data import sha256
from analysis.pipeline.evaluation import metrics
from analysis.experiments.auxiliary_ablation import AMENITIES, BLOCK


OUT = REPORTS / "spatiotemporal_hybrid"
TARGET = "MONTHLY_RENT"
MONTH = "_MONTH"
GROUP_KEYS = ["PLANNING_AREA", "FLAT_TYPE_CLEAN"]
BASE_FEATURES = FEATURE_SETS["C_property_time_town"] + BLOCK + AMENITIES
BASE_CATEGORICAL = {
    "FLAT_TYPE_CLEAN", "FLAT_MODEL_CLEAN", "TOWN_CLEAN",
    "SUBZONE", "PLANNING_AREA", "REGION",
}
HISTORY_GRAINS = {
    "SIGNATURE": [
        "TOWN", "BLOCK", "STREET", "FLAT_TYPE_CLEAN", "FLAT_MODEL_CLEAN",
        "FLOOR_AREA_SQM", "LEASE_COMMENCE_DATE",
    ],
    "BLOCK_FLAT": ["TOWN", "BLOCK", "STREET", "FLAT_TYPE_CLEAN"],
    "AREA_FLAT": GROUP_KEYS,
}
HISTORY_SUFFIXES = ["MEAN", "COUNT", "LAST_MEAN", "RECENCY_MONTHS"]
HISTORY_FEATURES = [f"HIST_{grain}_{suffix}" for grain in HISTORY_GRAINS for suffix in HISTORY_SUFFIXES]
VARIANTS = [
    "spatial_baseline",
    "spatial_spatiotemporal",
    "historical",
    "historical_spatiotemporal",
]
FOLDS = [
    ("2023-01", "2023-07"),
    ("2023-07", "2024-04"),
    ("2024-04", "2025-04"),
]
ROW_MODEL_SETTINGS = {
    "max_iter": 200,
    "learning_rate": 0.05,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 30,
    "l2_regularization": 3.0,
    "categorical_features": "from_dtype",
    "early_stopping": False,
    "random_state": 42,
}
TEMPORAL_ALPHA = 25.0
TEMPORAL_FEATURES = [
    "lag1", "lag3_mean", "neighbor_lag1", "flat_type_lag1", "global_lag1",
    "month_index", "season_sin", "season_cos",
]
SOURCES = [
    DATA / "train.csv", DATA / "test.csv", AUX / "sg-hdb-block.csv",
    AUX / "sg-mrt-stations.csv", AUX / "sg-shopping-malls.csv", AUX / "sg-schools.csv",
]


@dataclass
class TemporalForecast:
    values: pd.DataFrame
    training_rows: int
    groups: int
    clipped_predictions: int


def add_month(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach a monthly Period without modifying the caller's frame."""
    result = frame.copy()
    result[MONTH] = pd.PeriodIndex(result["RENT_APPROVAL_DATE"], freq="M")
    return result


def spatial_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Build property, block and amenity features for labeled or test rows."""
    frame = build_auxiliary_features(raw)
    if len(frame) != len(raw):
        raise AssertionError("Spatial feature join changed the row count")
    if frame[["LATITUDE", "LONGITUDE", "PLANNING_AREA"]].isna().any().any():
        raise AssertionError("Some rental addresses did not match the HDB block table")
    return add_month(frame)


def _month_distance(later: pd.Series, earlier: pd.Series) -> pd.Series:
    distance = later.astype("int64") - earlier.astype("int64")
    return distance.where(earlier.notna())


def historical_features(fitting: pd.DataFrame, prediction: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create strictly-prior fitting histories and forecast-origin histories."""
    if TARGET not in fitting:
        raise ValueError("Fitting data must contain the target")
    fit_result = pd.DataFrame(index=fitting.index)
    pred_result = pd.DataFrame(index=prediction.index)
    for name, keys in HISTORY_GRAINS.items():
        prefix = f"HIST_{name}"
        monthly = (
            fitting.groupby(keys + [MONTH], dropna=False)[TARGET]
            .agg(_sum="sum", _count="size", _mean="mean")
            .reset_index().sort_values(keys + [MONTH])
        )
        grouped = monthly.groupby(keys, dropna=False, sort=False)
        monthly[f"{prefix}_COUNT"] = grouped["_count"].cumsum() - monthly["_count"]
        prior_sum = grouped["_sum"].cumsum() - monthly["_sum"]
        monthly[f"{prefix}_MEAN"] = prior_sum / monthly[f"{prefix}_COUNT"].replace(0, np.nan)
        monthly[f"{prefix}_LAST_MEAN"] = grouped["_mean"].shift(1)
        monthly[f"{prefix}_RECENCY_MONTHS"] = _month_distance(monthly[MONTH], grouped[MONTH].shift(1))
        feature_columns = [f"{prefix}_{suffix}" for suffix in HISTORY_SUFFIXES]
        fit_join = (
            fitting[keys + [MONTH]].reset_index(names="_ROW_INDEX")
            .merge(monthly[keys + [MONTH] + feature_columns], on=keys + [MONTH], how="left", validate="many_to_one")
            .set_index("_ROW_INDEX").reindex(fitting.index)
        )
        fit_result[feature_columns] = fit_join[feature_columns]

        totals = monthly.groupby(keys, dropna=False).agg(_total_sum=("_sum", "sum"), **{
            f"{prefix}_COUNT": ("_count", "sum")
        }).reset_index()
        totals[f"{prefix}_MEAN"] = totals["_total_sum"] / totals[f"{prefix}_COUNT"].replace(0, np.nan)
        latest = (
            monthly.sort_values(MONTH).groupby(keys, dropna=False, as_index=False).tail(1)
            [keys + [MONTH, "_mean"]]
            .rename(columns={MONTH: "_LAST_MONTH", "_mean": f"{prefix}_LAST_MEAN"})
        )
        pred_join = (
            prediction[keys + [MONTH]].reset_index(names="_ROW_INDEX")
            .merge(totals.drop(columns="_total_sum"), on=keys, how="left", validate="many_to_one")
            .merge(latest, on=keys, how="left", validate="many_to_one")
            .set_index("_ROW_INDEX").reindex(prediction.index)
        )
        pred_join[f"{prefix}_RECENCY_MONTHS"] = _month_distance(pred_join[MONTH], pred_join["_LAST_MONTH"])
        pred_result[feature_columns] = pred_join[feature_columns]
    return fit_result, pred_result


def model_inputs(fitting: pd.DataFrame, prediction: pd.DataFrame,
                 features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Learn categorical levels from fitting rows only."""
    x_fit = fitting[features].copy()
    x_pred = prediction[features].copy()
    for column in features:
        if column not in BASE_CATEGORICAL:
            continue
        known = pd.Index(x_fit[column].dropna().astype(str).unique())
        dtype = pd.CategoricalDtype(categories=known)
        x_fit[column] = x_fit[column].astype("string").astype(dtype)
        x_pred[column] = x_pred[column].astype("string").astype(dtype)
    return x_fit, x_pred


def fit_row_model(fitting: pd.DataFrame, prediction: pd.DataFrame,
                  with_history: bool) -> tuple[np.ndarray, np.ndarray]:
    """Fit one row-level model and return fitting and future predictions."""
    fit = fitting.copy()
    pred = prediction.copy()
    features = list(BASE_FEATURES)
    if with_history:
        fit_hist, pred_hist = historical_features(fit, pred)
        fit = fit.join(fit_hist)
        pred = pred.join(pred_hist)
        features += HISTORY_FEATURES
    x_fit, x_pred = model_inputs(fit, pred, features)
    model = HistGradientBoostingRegressor(**ROW_MODEL_SETTINGS)
    model.fit(x_fit, fit[TARGET].to_numpy(float))
    return np.asarray(model.predict(x_fit), float), np.asarray(model.predict(x_pred), float)


def _area_centroids(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby("PLANNING_AREA")[["LATITUDE", "LONGITUDE"]].median().sort_index()


def _neighbor_weights(groups: pd.DataFrame, centroids: pd.DataFrame, k: int = 4) -> np.ndarray:
    """Return inverse-distance weights among groups of the same flat type."""
    coordinates = groups[["PLANNING_AREA"]].join(centroids, on="PLANNING_AREA")[["LATITUDE", "LONGITUDE"]].to_numpy(float)
    radians = np.radians(coordinates)
    lat, lon = radians[:, 0][:, None], radians[:, 1][:, None]
    hav = np.sin((lat - lat.T) / 2) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin((lon - lon.T) / 2) ** 2
    distance = 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(hav, 0, 1)))
    weights = np.zeros_like(distance)
    flat_types = groups["FLAT_TYPE_CLEAN"].astype(str).to_numpy()
    for row in range(len(groups)):
        eligible = np.flatnonzero((flat_types == flat_types[row]) & (np.arange(len(groups)) != row))
        if len(eligible) == 0:
            continue
        nearest = eligible[np.argsort(distance[row, eligible])[:k]]
        raw = 1.0 / np.maximum(distance[row, nearest], 0.25)
        weights[row, nearest] = raw / raw.sum()
    return weights


def _temporal_pipeline() -> Pipeline:
    preprocessing = ColumnTransformer([
        ("group", OneHotEncoder(handle_unknown="ignore"), ["group"]),
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), TEMPORAL_FEATURES),
    ])
    return Pipeline([("preprocess", preprocessing), ("ridge", Ridge(alpha=TEMPORAL_ALPHA))])


def forecast_residuals(fitting: pd.DataFrame, residuals: np.ndarray,
                       forecast_months: list[pd.Period]) -> TemporalForecast:
    """Recursively forecast area/type residuals from temporal and spatial lags."""
    data = fitting[GROUP_KEYS + [MONTH]].copy()
    data["residual"] = np.asarray(residuals, dtype=float)
    observed = data.groupby(GROUP_KEYS + [MONTH], dropna=False)["residual"].agg(value="mean", n="size").reset_index()
    groups = observed[GROUP_KEYS].drop_duplicates().sort_values(GROUP_KEYS).reset_index(drop=True)
    groups["group"] = groups["PLANNING_AREA"].astype(str) + " / " + groups["FLAT_TYPE_CLEAN"].astype(str)
    months = pd.period_range(data[MONTH].min(), data[MONTH].max(), freq="M")
    group_index = pd.MultiIndex.from_frame(groups[GROUP_KEYS])
    wide = observed.pivot(index=MONTH, columns=GROUP_KEYS, values="value").reindex(index=months, columns=group_index)
    counts = observed.pivot(index=MONTH, columns=GROUP_KEYS, values="n").reindex(index=months, columns=group_index)
    flat_month = data.groupby(["FLAT_TYPE_CLEAN", MONTH])["residual"].mean().unstack("FLAT_TYPE_CLEAN").reindex(months)
    global_month = data.groupby(MONTH)["residual"].mean().reindex(months)
    filled = wide.copy()
    for column in filled:
        fallback = flat_month[column[1]] if column[1] in flat_month else global_month
        filled[column] = filled[column].fillna(fallback).fillna(global_month).fillna(0.0)
    weights = _neighbor_weights(groups, _area_centroids(fitting))
    origin = months[0].ordinal
    training_rows = []
    for position in range(3, len(months)):
        previous = filled.iloc[position - 1].to_numpy(float)
        prior_three = filled.iloc[position - 3:position].mean(axis=0).to_numpy(float)
        neighbor = weights @ previous
        flat_lag = groups["FLAT_TYPE_CLEAN"].map(flat_month.iloc[position - 1]).to_numpy(float)
        global_lag = float(global_month.iloc[position - 1])
        current = wide.iloc[position].to_numpy(float)
        current_counts = counts.iloc[position].to_numpy(float)
        valid = np.isfinite(current) & np.isfinite(current_counts)
        month = months[position]
        for row in np.flatnonzero(valid):
            training_rows.append({
                "group": groups.loc[row, "group"], "lag1": previous[row],
                "lag3_mean": prior_three[row], "neighbor_lag1": neighbor[row],
                "flat_type_lag1": flat_lag[row], "global_lag1": global_lag,
                "month_index": month.ordinal - origin,
                "season_sin": np.sin(2 * np.pi * month.month / 12),
                "season_cos": np.cos(2 * np.pi * month.month / 12),
                "target": current[row], "sample_weight": np.sqrt(current_counts[row]),
            })
    training = pd.DataFrame(training_rows)
    if training.empty:
        raise ValueError("Insufficient monthly residual history")
    model = _temporal_pipeline()
    model.fit(training[["group"] + TEMPORAL_FEATURES], training["target"],
              ridge__sample_weight=training["sample_weight"])
    lower, upper = np.nanquantile(observed["value"], [.01, .99])
    state = [row.to_numpy(float) for _, row in filled.iterrows()]
    output, clipped = [], 0
    requested = sorted(set(forecast_months))
    for month in [value for value in requested if value <= months[-1]]:
        month_rows = groups[GROUP_KEYS].copy()
        month_rows[MONTH] = month
        month_rows["correction"] = filled.loc[month].to_numpy(float)
        output.append(month_rows)
    for month in [value for value in requested if value > months[-1]]:
        previous = state[-1]
        prior_three = np.mean(state[-3:], axis=0)
        neighbor = weights @ previous
        flat_lag = pd.Series(previous).groupby(groups["FLAT_TYPE_CLEAN"]).transform("mean").to_numpy()
        features = pd.DataFrame({
            "group": groups["group"], "lag1": previous, "lag3_mean": prior_three,
            "neighbor_lag1": neighbor, "flat_type_lag1": flat_lag,
            "global_lag1": float(np.mean(previous)), "month_index": month.ordinal - origin,
            "season_sin": np.sin(2 * np.pi * month.month / 12),
            "season_cos": np.cos(2 * np.pi * month.month / 12),
        })
        raw = np.asarray(model.predict(features[["group"] + TEMPORAL_FEATURES]), float)
        predicted = np.clip(raw, lower, upper)
        clipped += int(np.sum(raw != predicted))
        state.append(predicted)
        month_rows = groups[GROUP_KEYS].copy()
        month_rows[MONTH] = month
        month_rows["correction"] = predicted
        output.append(month_rows)
    values = pd.concat(output, ignore_index=True).sort_values([MONTH] + GROUP_KEYS).reset_index(drop=True)
    return TemporalForecast(values, len(training), len(groups), clipped)


def attach_correction(frame: pd.DataFrame, forecast: pd.DataFrame) -> np.ndarray:
    """Map group forecasts to rows, using month/type then month fallbacks."""
    joined = frame[GROUP_KEYS + [MONTH]].merge(forecast, on=GROUP_KEYS + [MONTH], how="left", validate="many_to_one")
    type_fallback = forecast.groupby(["FLAT_TYPE_CLEAN", MONTH])["correction"].mean()
    global_fallback = forecast.groupby(MONTH)["correction"].mean()
    missing = joined["correction"].isna()
    if missing.any():
        keys = pd.MultiIndex.from_frame(joined.loc[missing, ["FLAT_TYPE_CLEAN", MONTH]])
        joined.loc[missing, "correction"] = type_fallback.reindex(keys).to_numpy()
    missing = joined["correction"].isna()
    if missing.any():
        joined.loc[missing, "correction"] = joined.loc[missing, MONTH].map(global_fallback)
    return joined["correction"].fillna(0.0).to_numpy(float)


def evaluate(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    """Run expanding fixed-origin folds without validation-label feedback."""
    score_rows, monthly_rows, prediction_rows, diagnostics = [], [], [], []
    for start_text, end_text in FOLDS:
        start, end = pd.Period(start_text, freq="M"), pd.Period(end_text, freq="M")
        fitting = frame[frame[MONTH] < start].copy()
        validation = frame[(frame[MONTH] >= start) & (frame[MONTH] < end)].copy()
        if fitting.empty or validation.empty:
            continue
        print(f"Fold {start_text}: {len(fitting)} fit rows, {len(validation)} validation rows", flush=True)
        actual = validation[TARGET].to_numpy(float)
        predictions: dict[str, np.ndarray] = {}
        for label, with_history in (("spatial_baseline", False), ("historical", True)):
            fit_pred, future_pred = fit_row_model(fitting, validation, with_history)
            predictions[label] = future_pred
            temporal = forecast_residuals(fitting, fitting[TARGET].to_numpy(float) - fit_pred,
                                          validation[MONTH].drop_duplicates().sort_values().tolist())
            combined = "historical_spatiotemporal" if with_history else "spatial_spatiotemporal"
            predictions[combined] = future_pred + attach_correction(validation, temporal.values)
            diagnostics.append({
                "validation_start": start_text, "row_model": label,
                "temporal_training_group_months": temporal.training_rows,
                "temporal_groups": temporal.groups,
                "clipped_group_forecasts": temporal.clipped_predictions,
            })
        for variant in VARIANTS:
            predicted = predictions[variant]
            score_rows.append({
                "validation_start": start_text, "validation_end_exclusive": end_text,
                "variant": variant, "n_fit": len(fitting), "n_validation": len(validation),
                "horizon_months": validation[MONTH].nunique(), **metrics(actual, predicted),
            })
            fold_predictions = pd.DataFrame({
                "validation_start": start_text,
                "approval_month": validation[MONTH].astype(str).to_numpy(),
                "variant": variant, "actual": actual, "predicted": predicted,
            })
            prediction_rows.append(fold_predictions)
            for month, part in fold_predictions.groupby("approval_month", sort=True):
                monthly_rows.append({
                    "validation_start": start_text, "approval_month": month,
                    "horizon": pd.Period(month, freq="M").ordinal - start.ordinal + 1,
                    "variant": variant, "n": len(part),
                    **metrics(part["actual"], part["predicted"]),
                })
            score = score_rows[-1]
            print(f"  {variant}: RMSE={score['rmse_sgd']:.2f}; bias={score['bias_sgd']:.2f}", flush=True)
    return pd.DataFrame(score_rows), pd.DataFrame(monthly_rows), pd.concat(prediction_rows, ignore_index=True), diagnostics


def pooled_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, part in predictions.groupby("variant", sort=False):
        rows.append({"variant": variant, "n": len(part), **metrics(part["actual"], part["predicted"])})
    result = pd.DataFrame(rows)
    baseline = result.loc[result["variant"].eq("spatial_baseline"), "rmse_sgd"].iloc[0]
    result["delta_rmse_vs_spatial_baseline_sgd"] = result["rmse_sgd"] - baseline
    return result.sort_values("rmse_sgd").reset_index(drop=True)


def plot_results(scores: pd.DataFrame, monthly: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), layout="constrained")
    for variant in VARIANTS:
        part = scores[scores["variant"].eq(variant)]
        axes[0].plot(part["validation_start"], part["rmse_sgd"], marker="o", label=variant)
        horizon = monthly[monthly["variant"].eq(variant)].groupby("horizon", as_index=False).agg(rmse_sgd=("rmse_sgd", "mean"))
        axes[1].plot(horizon["horizon"], horizon["rmse_sgd"], marker="o", label=variant)
    axes[0].set(title="Fixed-origin fold RMSE", xlabel="Validation start", ylabel="RMSE (SGD)")
    axes[1].set(title="Mean monthly RMSE by forecast horizon", xlabel="Months after forecast origin", ylabel="RMSE (SGD)")
    for ax in axes:
        ax.grid(axis="y", alpha=.25)
    axes[1].legend(fontsize=8)
    fig.suptitle("Hierarchical spatiotemporal rental forecasting")
    fig.savefig(OUT / "evaluation_rmse.png", dpi=160, facecolor="white")
    plt.close(fig)


def train_final(train: pd.DataFrame, test: pd.DataFrame, best_variant: str) -> tuple[pd.DataFrame, dict]:
    """Train on all labels and generate predictions for every supplied test row."""
    with_history = best_variant.startswith("historical")
    with_temporal = best_variant.endswith("spatiotemporal")
    fit_pred, test_pred = fit_row_model(train, test, with_history)
    diagnostics = {"variant": best_variant, "with_history": with_history, "with_spatiotemporal": with_temporal}
    if with_temporal:
        temporal = forecast_residuals(train, train[TARGET].to_numpy(float) - fit_pred,
                                      test[MONTH].drop_duplicates().sort_values().tolist())
        test_pred = test_pred + attach_correction(test, temporal.values)
        diagnostics.update({
            "temporal_training_group_months": temporal.training_rows,
            "temporal_groups": temporal.groups,
            "clipped_group_forecasts": temporal.clipped_predictions,
        })
        group_forecasts = temporal.values.copy()
        group_forecasts[MONTH] = group_forecasts[MONTH].astype(str)
        group_forecasts.to_csv(OUT / "final_group_forecasts.csv", index=False, float_format="%.4f")
    detailed = pd.DataFrame({
        "Id": np.arange(len(test), dtype=int),
        "RENT_APPROVAL_DATE": test[MONTH].astype(str).to_numpy(),
        "Predicted": test_pred,
    })
    return detailed, diagnostics


def write_report(scores: pd.DataFrame, pooled: pd.DataFrame, best_variant: str) -> None:
    baseline = pooled[pooled["variant"].eq("spatial_baseline")].iloc[0]
    best = pooled[pooled["variant"].eq(best_variant)].iloc[0]
    lines = [
        "# Hierarchical spatiotemporal evaluation", "",
        "The experiment uses expanding fixed-origin validation. No validation target is used as a lag or target encoding, so each fold simulates a batch multi-month forecast.", "",
        "## Pooled result", "",
        f"The selected variant is **{best_variant}**, with pooled RMSE **S${best['rmse_sgd']:.2f}** versus **S${baseline['rmse_sgd']:.2f}** for the static spatial baseline (change {best['rmse_sgd'] - baseline['rmse_sgd']:+.2f}).", "",
        "```text", pooled.to_string(index=False, float_format=lambda value: f"{value:.2f}"), "```", "",
        "## Fold results", "", "```text",
        scores.to_string(index=False, float_format=lambda value: f"{value:.2f}"), "```", "",
        "Historical features are prior-month-only expanding summaries during fitting and are frozen at each forecast origin. The spatiotemporal correction recursively forecasts planning-area × flat-type residuals from group lags, flat-type and global lags, seasonality, linear time, and inverse-distance neighboring-area lags.",
    ]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    before = {str(path.relative_to(DATA)): sha256(path) for path in SOURCES}
    train = spatial_frame(pd.read_csv(DATA / "train.csv"))
    test = spatial_frame(pd.read_csv(DATA / "test.csv"))
    OUT.mkdir(parents=True, exist_ok=True)
    scores, monthly, predictions, diagnostics = evaluate(train)
    pooled = pooled_scores(predictions)
    best_variant = str(pooled.iloc[0]["variant"])
    scores.to_csv(OUT / "fold_metrics.csv", index=False, float_format="%.4f")
    pooled.to_csv(OUT / "pooled_metrics.csv", index=False, float_format="%.4f")
    monthly.to_csv(OUT / "monthly_errors.csv", index=False, float_format="%.4f")
    predictions.to_csv(OUT / "validation_predictions.csv", index=False, float_format="%.4f")
    plot_results(scores, monthly)
    final_predictions, final_diagnostics = train_final(train, test, best_variant)
    final_predictions.to_csv(OUT / "test_predictions.csv", index=False, float_format="%.4f")
    final_predictions[["Id", "Predicted"]].to_csv(OUT / "submission.csv", index=False, float_format="%.4f")
    write_report(scores, pooled, best_variant)
    after = {str(path.relative_to(DATA)): sha256(path) for path in SOURCES}
    if after != before:
        raise AssertionError("An input CSV changed during the experiment")
    result = {
        "evaluation_scenario": "Expanding fixed-origin batch forecasts; validation targets are never fed back",
        "folds": FOLDS, "variants": VARIANTS,
        "row_model": "scikit-learn HistGradientBoostingRegressor",
        "row_model_settings": ROW_MODEL_SETTINGS,
        "base_features": BASE_FEATURES, "historical_features": HISTORY_FEATURES,
        "temporal_group": GROUP_KEYS, "temporal_features": TEMPORAL_FEATURES,
        "temporal_ridge_alpha": TEMPORAL_ALPHA,
        "fold_scores": scores.to_dict(orient="records"),
        "pooled_scores": pooled.to_dict(orient="records"),
        "selected_final_variant": best_variant,
        "fold_diagnostics": diagnostics, "final_training": final_diagnostics,
        "source_sha256": after,
        "notes": [
            "Historical target features use strictly earlier months for fitting rows.",
            "Validation/test historical features are frozen from the fitting partition.",
            "Spatiotemporal residual forecasts are recursive and consume no future actual rent.",
            "The supplied amenities and HDB block attributes are current snapshots.",
        ],
    }
    (OUT / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Selected {best_variant}; saved evaluation and final predictions to {OUT}", flush=True)


if __name__ == "__main__":
    main()
