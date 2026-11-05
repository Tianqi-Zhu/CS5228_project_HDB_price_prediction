"""Compare traditional ML model families on the same temporal holdout and features.

Run train_baseline.py first to produce the CatBoost C result. This script trains
Ridge, Random Forest, and LightGBM in the existing conda cs5228 environment or any Python
environment with requirements-model.txt installed. Source CSVs stay unchanged.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn

from analysis.config import (
    CATEGORICAL, DATA, FAMILY_SETTINGS, FEATURE_SETS, REPORTS, SOURCES,
    VALIDATION_START,
)
from analysis.data import prepare_main, sha256, temporal_split
from analysis.evaluation import grouped_errors, metrics
from analysis.modeling import lightgbm_inputs, make_pipeline


CATBOOST_OUT = REPORTS / "model_baseline"
OUT = REPORTS / "model_families"
FEATURES = FEATURE_SETS["C_property_time_town"]
MODEL_SETTINGS = FAMILY_SETTINGS


def plot_comparison(monthly: pd.DataFrame) -> None:
    labels = ["ridge", "random_forest", "lightgbm", "C_property_time_town"]
    display = {"ridge": "Ridge", "random_forest": "Random forest", "lightgbm": "LightGBM", "C_property_time_town": "CatBoost C"}
    colors = {"ridge": "#D78036", "random_forest": "#5B7894", "lightgbm": "#8F5BB5", "C_property_time_town": "#147D91"}
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax_rmse, ax_bias) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, layout="constrained")
    fig.suptitle("Model comparison on the future holdout", fontsize=20, weight="bold")
    months = sorted(monthly["group"].unique())
    x = np.arange(len(months))
    for label in labels:
        part = monthly[monthly["model"] == label].set_index("group").loc[months]
        ax_rmse.plot(x, part["rmse_sgd"], marker="o", linewidth=2.2, color=colors[label], label=display[label])
        ax_bias.plot(x, part["bias_sgd"], marker="o", linewidth=2.2, color=colors[label])
    ax_rmse.set_ylabel("RMSE (SGD)")
    ax_rmse.legend(loc="upper left")
    ax_rmse.grid(axis="y", alpha=0.25)
    ax_bias.axhline(0, color="#193043", linewidth=1)
    ax_bias.set_ylabel("Mean prediction - actual (SGD)")
    ax_bias.set_xlabel("Rent approval month")
    ax_bias.set_xticks(x, months, rotation=40, ha="right")
    ax_bias.grid(axis="y", alpha=0.25)
    fig.savefig(OUT / "model_comparison_by_month.png", dpi=160, facecolor="white")
    plt.close(fig)


def main() -> None:
    baseline_path = CATBOOST_OUT / "results.json"
    if not baseline_path.exists():
        raise FileNotFoundError("Run analysis/train_baseline.py before this comparison")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    source_hashes_before = {path.name: sha256(path) for path in SOURCES}
    if source_hashes_before != baseline["source_sha256"]:
        raise ValueError("Source CSV hashes differ from the prior CatBoost run")
    if baseline["validation_rule"] != f"fit month < {VALIDATION_START}; validate month >= {VALIDATION_START}":
        raise ValueError("CatBoost result uses a different validation window")
    if baseline["feature_sets"]["C_property_time_town"] != FEATURES:
        raise ValueError("CatBoost C uses a different feature set")

    frame = prepare_main(pd.read_csv(DATA / "train.csv"))
    fitting, validation = temporal_split(frame)
    if len(fitting) != baseline["fitting_rows"] or len(validation) != baseline["validation_rows"]:
        raise ValueError("Fitting or validation row counts differ from CatBoost")
    x_fit, x_val = fitting[FEATURES], validation[FEATURES]
    y_fit = fitting["MONTHLY_RENT"].to_numpy(dtype=float)
    y_val = validation["MONTHLY_RENT"].to_numpy(dtype=float)
    OUT.mkdir(parents=True, exist_ok=True)
    scores = []
    monthly_tables = []

    for family in ("ridge", "random_forest", "lightgbm"):
        print(f"Training {family}", flush=True)
        if family == "lightgbm":
            lgb_fit, lgb_val = lightgbm_inputs(x_fit, x_val)
            model = lgb.LGBMRegressor(**MODEL_SETTINGS["lightgbm"])
            model.fit(lgb_fit, y_fit, categorical_feature=CATEGORICAL)
            predicted = np.asarray(model.predict(lgb_val), dtype=float)
        else:
            model = make_pipeline(family)
            model.fit(x_fit, y_fit)
            predicted = np.asarray(model.predict(x_val), dtype=float)
        score = {"model": family, "n_validation": len(y_val), **metrics(y_val, predicted)}
        scores.append(score)
        print(f"{family}: RMSE={score['rmse_sgd']:.2f} SGD; bias={score['bias_sgd']:.2f} SGD", flush=True)
        error_frame = pd.DataFrame({
            "approval_month": validation["RENT_APPROVAL_DATE"].to_numpy(),
            "actual": y_val,
            "predicted": predicted,
        })
        monthly_tables.append(grouped_errors(error_frame, "approval_month", family))

    catboost_score = next(row for row in baseline["scores"] if row["model"] == "C_property_time_town")
    scores.append(catboost_score)
    prior_monthly = pd.read_csv(CATBOOST_OUT / "monthly_errors.csv")
    monthly_tables.append(prior_monthly[prior_monthly["model"] == "C_property_time_town"].copy())
    combined_monthly = pd.concat(monthly_tables, ignore_index=True)
    pd.DataFrame(scores).to_csv(OUT / "family_metrics.csv", index=False, float_format="%.4f")
    combined_monthly.to_csv(OUT / "family_monthly_errors.csv", index=False, float_format="%.4f")
    plot_comparison(combined_monthly)

    source_hashes_after = {path.name: sha256(path) for path in SOURCES}
    if source_hashes_after != source_hashes_before:
        raise AssertionError("A source CSV changed during the model comparison")
    report = {
        "purpose": "Compare linear, bagged-tree, and two boosted-tree regressors with identical C features and temporal holdout",
        "validation_rule": baseline["validation_rule"],
        "fitting_rows": len(fitting),
        "validation_rows": len(validation),
        "features": FEATURES,
        "categorical_preprocessing": "Ridge and Random Forest use fitting-only OneHotEncoder(handle_unknown='ignore'); LightGBM uses pandas categorical columns with levels fitted on fitting rows only; CatBoost handles categories internally",
        "numeric_preprocessing": "Ridge and Random Forest use fitting-only median imputation; Ridge additionally uses StandardScaler; LightGBM and CatBoost handle numeric missing values internally",
        "settings": MODEL_SETTINGS,
        "versions": {"sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__, "lightgbm": lgb.__version__},
        "source_sha256": source_hashes_after,
        "scores": scores,
        "catboost_source": "reports/model_baseline/results.json",
        "note": "Single temporal holdout. Model parameters were chosen as simple fixed baselines, not tuned on this holdout.",
    }
    (OUT / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved model-family comparison in {OUT}", flush=True)


if __name__ == "__main__":
    main()
