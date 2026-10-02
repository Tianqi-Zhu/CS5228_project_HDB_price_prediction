"""Measure the incremental value of supplied auxiliary features.

Run from the repository root:
    python analysis/compare_auxiliary_features.py

Every stage uses the same rows, validation months, LightGBM settings, and
target. Economic lags describe a monthly-updated prediction scenario.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from analysis.config import AUX, DATA, FAMILY_SETTINGS, FEATURE_SETS, REPORTS, VALIDATION_START
from analysis.data import (
    add_economic_features, add_spatial_features, candidate_interactions,
    prepare_main, sha256, temporal_split,
)
from analysis.evaluation import grouped_errors, metrics


OUT = REPORTS / "auxiliary_ablation"
BASE = FEATURE_SETS["C_property_time_town"]
BLOCK = [
    "LATITUDE", "LONGITUDE", "MAX_FLOOR", "YEAR_COMPLETED",
    "BUILDING_AGE_FROM_COMPLETION", "SUBZONE", "PLANNING_AREA", "REGION",
]
AMENITIES = [
    "NEAREST_MRT_KM", "NEAREST_MALL_KM", "NEAREST_SCHOOL_KM",
    "DISTANCE_TO_CBD_KM", "AMENITY_DENSITY_1KM",
    "MRT_COUNT_0.5KM", "MRT_COUNT_1.0KM", "MRT_COUNT_2.0KM",
    "MALL_COUNT_0.5KM", "MALL_COUNT_1.0KM", "MALL_COUNT_2.0KM",
    "SCHOOL_COUNT_0.5KM", "SCHOOL_COUNT_1.0KM", "SCHOOL_COUNT_2.0KM",
]
COE = [f"COE_{category}_MEDIAN_LAG{lag}" for category in ("A", "B", "C", "E") for lag in (1, 3, 6)]
STOCK = [f"{measure}_LAG{lag}" for measure in ("STOCK_MEDIAN_RETURN", "STOCK_RETURN_VOLATILITY") for lag in (1, 3, 6)]
INTERACTIONS = ["AREA_X_AGE", "AREA_X_MRT_ACCESS", "TOWN_X_FLAT_TYPE"]
FEATURE_STAGES = {
    "A_property": FEATURE_SETS["A_property"],
    "B_property_time": FEATURE_SETS["B_property_time"],
    "C_property_time_town": BASE,
    "D_plus_block": BASE + BLOCK,
    "E_plus_amenities": BASE + BLOCK + AMENITIES,
    "F_plus_coe": BASE + BLOCK + AMENITIES + COE,
    "G_plus_stocks": BASE + BLOCK + AMENITIES + COE + STOCK,
    "H_plus_interactions": BASE + BLOCK + AMENITIES + COE + STOCK + INTERACTIONS,
}
CATEGORICAL = {"FLAT_TYPE_CLEAN", "FLAT_MODEL_CLEAN", "TOWN_CLEAN", "SUBZONE", "PLANNING_AREA", "REGION", "TOWN_X_FLAT_TYPE"}
SOURCE_FILES = [
    DATA / "train.csv", DATA / "test.csv", AUX / "sg-hdb-block.csv",
    AUX / "sg-mrt-stations.csv", AUX / "sg-shopping-malls.csv",
    AUX / "sg-schools.csv", AUX / "sg-coe-prices.csv", AUX / "sg-stock-prices.csv",
]


def feature_frame() -> pd.DataFrame:
    """Build the shared feature table once so stages differ only by columns."""
    raw = pd.read_csv(DATA / "train.csv")
    frame = prepare_main(raw)
    frame = add_spatial_features(
        frame,
        pd.read_csv(AUX / "sg-hdb-block.csv", dtype={"POSTAL_CODE": "string"}),
        pd.read_csv(AUX / "sg-mrt-stations.csv"),
        pd.read_csv(AUX / "sg-shopping-malls.csv"),
        pd.read_csv(AUX / "sg-schools.csv"),
    )
    if len(frame) != len(raw) or frame["MONTHLY_RENT"].isna().any():
        raise AssertionError("Spatial join changed the training rows or target")
    if frame["LATITUDE"].isna().any():
        raise AssertionError("Some training addresses did not match the HDB block table")
    frame = add_economic_features(
        frame,
        pd.read_csv(AUX / "sg-coe-prices.csv"),
        pd.read_csv(AUX / "sg-stock-prices.csv"),
    )
    frame = candidate_interactions(frame)
    expected = set().union(*map(set, FEATURE_STAGES.values()))
    absent = expected.difference(frame.columns)
    if absent:
        raise AssertionError(f"Feature generation missed columns: {sorted(absent)}")
    if "MONTHLY_RENT" in expected or "RENT_PER_SQM" in expected or "LOG_MONTHLY_RENT" in expected:
        raise AssertionError("A target-derived column was selected as a predictor")
    return frame


def model_inputs(fitting: pd.DataFrame, validation: pd.DataFrame,
                 features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Learn categorical levels from fitting rows only."""
    x_fit = fitting[features].copy()
    x_val = validation[features].copy()
    categories = [column for column in features if column in CATEGORICAL]
    for column in categories:
        known = pd.Index(x_fit[column].dropna().astype(str).unique())
        dtype = pd.CategoricalDtype(categories=known)
        x_fit[column] = x_fit[column].astype("string").astype(dtype)
        x_val[column] = x_val[column].astype("string").astype(dtype)
    return x_fit, x_val, categories


def plot_scores(scores: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    values = scores["rmse_sgd"].to_numpy()
    for ax, subset, title in ((axes[0], slice(None), "All feature stages"),
                              (axes[1], slice(2, None), "Auxiliary stages (zoomed)")):
        y = values[subset]
        x = np.arange(len(y))
        labels = scores["stage"].str[0].to_numpy()[subset]
        ax.plot(x, y, marker="o", linewidth=2.5, color="#147D91")
        ax.set_xticks(x, labels)
        ax.set(title=title, xlabel="Feature stage", ylabel="Validation RMSE (SGD)")
        ax.grid(axis="y", alpha=.25)
        for xi, yi in zip(x, y):
            ax.annotate(f"{yi:.1f}", (xi, yi), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9)
    axes[0].set_ylim(values.min() - 15, values.max() + 25)
    axes[1].set_ylim(values[2:].min() - 3, values[2:].max() + 4)
    fig.suptitle("Incremental auxiliary-feature ablation | lower is better")
    fig.savefig(OUT / "ablation_rmse.png", dpi=160, facecolor="white")
    plt.close(fig)


def main() -> None:
    hashes_before = {str(path.relative_to(DATA)): sha256(path) for path in SOURCE_FILES}
    frame = feature_frame()
    fitting, validation = temporal_split(frame)
    y_fit = fitting["MONTHLY_RENT"].to_numpy(float)
    y_val = validation["MONTHLY_RENT"].to_numpy(float)
    OUT.mkdir(parents=True, exist_ok=True)
    scores, monthly_errors, coverage, importance = [], [], [], []
    best = None
    if lgb is not None:
        model_name = "LightGBM"
        settings = {**FAMILY_SETTINGS["lightgbm"], "n_jobs": 4}
        importance_note = "Training gain for every stage"
    else:
        model_name = "scikit-learn HistGradientBoosting"
        settings = {"max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 31,
                    "min_samples_leaf": 30, "l2_regularization": 3.0,
                    "categorical_features": "from_dtype", "early_stopping": False,
                    "random_state": 42}
        importance_note = "Validation permutation importance for best stage on 2,000 sampled rows; units are increase in MSE"
    for stage, features in FEATURE_STAGES.items():
        print(f"Training {stage}: {len(features)} features", flush=True)
        x_fit, x_val, categorical = model_inputs(fitting, validation, features)
        if lgb is not None:
            model = lgb.LGBMRegressor(**settings)
            model.fit(x_fit, y_fit, categorical_feature=categorical)
        else:
            model = HistGradientBoostingRegressor(**settings)
            model.fit(x_fit, y_fit)
        predicted = np.asarray(model.predict(x_val), dtype=float)
        scores.append({"stage": stage, "n_fit": len(fitting), "n_validation": len(validation),
                       "n_features": len(features), **metrics(y_val, predicted)})
        errors = pd.DataFrame({"approval_month": validation["RENT_APPROVAL_DATE"].to_numpy(),
                               "actual": y_val, "predicted": predicted})
        monthly_errors.append(grouped_errors(errors, "approval_month", stage).rename(columns={"model": "stage"}))
        for column in features:
            coverage.append({"stage": stage, "feature": column,
                             "fit_nonmissing_pct": round(float(x_fit[column].notna().mean() * 100), 2),
                             "validation_nonmissing_pct": round(float(x_val[column].notna().mean() * 100), 2)})
        if lgb is not None:
            importance.extend({"stage": stage, "feature": column, "importance": float(value), "method": "training_gain"}
                              for column, value in zip(features, model.booster_.feature_importance(importance_type="gain")))
        elif best is None or scores[-1]["rmse_sgd"] < best[0]:
            best = (scores[-1]["rmse_sgd"], stage, model, x_val, features)
        print(f"  RMSE={scores[-1]['rmse_sgd']:.2f}; MAE={scores[-1]['mae_sgd']:.2f}; bias={scores[-1]['bias_sgd']:.2f}", flush=True)
    if lgb is None:
        _, best_stage, best_model, best_x_val, best_features = best
        sample = np.linspace(0, len(best_x_val) - 1, min(2000, len(best_x_val)), dtype=int)
        perm = permutation_importance(best_model, best_x_val.iloc[sample], y_val[sample],
                                      scoring="neg_mean_squared_error", n_repeats=1,
                                      random_state=42, n_jobs=1)
        importance.extend({"stage": best_stage, "feature": column, "importance": float(value), "method": "validation_permutation_mse"}
                          for column, value in zip(best_features, perm.importances_mean))
    score_frame = pd.DataFrame(scores)
    score_frame["delta_rmse_vs_C_sgd"] = score_frame["rmse_sgd"] - score_frame.loc[score_frame["stage"] == "C_property_time_town", "rmse_sgd"].iloc[0]
    score_frame["delta_rmse_vs_previous_sgd"] = score_frame["rmse_sgd"].diff()
    score_frame.to_csv(OUT / "ablation_metrics.csv", index=False, float_format="%.4f")
    pd.concat(monthly_errors, ignore_index=True).to_csv(OUT / "monthly_errors.csv", index=False, float_format="%.4f")
    pd.DataFrame(coverage).to_csv(OUT / "feature_coverage.csv", index=False)
    pd.DataFrame(importance).to_csv(OUT / "feature_importance.csv", index=False, float_format="%.4f")
    plot_scores(score_frame)
    hashes_after = {str(path.relative_to(DATA)): sha256(path) for path in SOURCE_FILES}
    if hashes_after != hashes_before:
        raise AssertionError("An input CSV changed during the experiment")
    report = {
        "validation_rule": f"fit month < {VALIDATION_START}; validate month >= {VALIDATION_START}",
        "fit_months": [fitting["RENT_APPROVAL_DATE"].min(), fitting["RENT_APPROVAL_DATE"].max()],
        "validation_months": [validation["RENT_APPROVAL_DATE"].min(), validation["RENT_APPROVAL_DATE"].max()],
        "model": model_name, "model_settings": settings,
        "feature_importance_method": importance_note,
        "feature_stages": FEATURE_STAGES, "scores": scores,
        "source_sha256": hashes_after,
        "assumptions": [
            "All stages use the same temporal split and the same fitting and validation rows.",
            "Economic lags use prior calendar months and assume monthly-updated forecasts; this is not a fixed-origin forecast from the training cutoff.",
            "MRT, mall, and school tables are current snapshots; no historical opening dates are available.",
            "HDB address and amenity features are computed without using rental targets.",
            "Feature importance does not establish causality.",
        ],
    }
    (OUT / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved auxiliary ablation to {OUT}", flush=True)


if __name__ == "__main__":
    main()
