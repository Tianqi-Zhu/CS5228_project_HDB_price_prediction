"""Compare a mean predictor and three CatBoost feature sets on a future holdout.

Run from the repository root after installing requirements-model.txt:
    python analysis/train_baseline.py

All cleaning happens in DataFrame copies. The three source CSV files are never
written; their SHA-256 hashes are checked before and after this script runs.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.config import (
    CATEGORICAL, CATBOOST_SETTINGS, DATA, FEATURE_SETS, REPORTS, SOURCES,
    VALIDATION_START,
)
from analysis.data import prepare_main, sha256, temporal_split
from analysis.evaluation import grouped_errors, metrics

OUT = REPORTS / "model_baseline"
MODEL_SETTINGS = CATBOOST_SETTINGS


def make_monthly_error_figure(monthly: pd.DataFrame, model: str) -> None:
    part = monthly[monthly["model"] == model].sort_values("group")
    months = part["group"].tolist()
    x = np.arange(len(months))
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax_rmse, ax_bias) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, layout="constrained")
    display_name = {
        "mean_constant": "mean predictor",
        "A_property": "A: property",
        "B_property_time": "B: property + time",
        "C_property_time_town": "C: property + time + town",
    }.get(model, model)
    fig.suptitle(f"Validation error by month: {display_name}", fontsize=20, weight="bold")
    ax_rmse.plot(x, part["rmse_sgd"], marker="o", linewidth=2.5, color="#147D91")
    ax_rmse.set_ylabel("RMSE (SGD)")
    ax_rmse.grid(axis="y", alpha=0.25)
    ax_bias.bar(x, part["bias_sgd"], color="#D78036", width=0.7)
    ax_bias.axhline(0, color="#193043", linewidth=1)
    ax_bias.set_ylabel("Mean prediction - actual (SGD)")
    ax_bias.set_xlabel("Rent approval month")
    ax_bias.set_xticks(x, months, rotation=40, ha="right")
    ax_bias.grid(axis="y", alpha=0.25)
    fig.savefig(OUT / "monthly_validation_error.png", dpi=160, facecolor="white")
    plt.close(fig)


def main() -> None:
    from catboost import CatBoostRegressor, __version__ as catboost_version

    source_hashes_before = {path.name: sha256(path) for path in SOURCES}
    raw = pd.read_csv(DATA / "train.csv")
    frame = prepare_main(raw)
    fitting, validation = temporal_split(frame)
    y_fit = fitting["MONTHLY_RENT"].to_numpy(dtype=float)
    y_val = validation["MONTHLY_RENT"].to_numpy(dtype=float)
    OUT.mkdir(parents=True, exist_ok=True)

    predictions: dict[str, np.ndarray] = {
        "mean_constant": np.full(len(validation), y_fit.mean(), dtype=float)
    }
    for label, columns in FEATURE_SETS.items():
        print(f"Training {label} with {len(columns)} features", flush=True)
        model = CatBoostRegressor(**MODEL_SETTINGS)
        model.fit(
            fitting[columns], y_fit,
            cat_features=[c for c in columns if c in CATEGORICAL],
        )
        predictions[label] = np.asarray(model.predict(validation[columns]), dtype=float)

    scores = []
    monthly_tables = []
    type_tables = []
    for label, predicted in predictions.items():
        score = {"model": label, "n_validation": len(y_val), **metrics(y_val, predicted)}
        scores.append(score)
        print(f"{label}: RMSE={score['rmse_sgd']:.2f} SGD; bias={score['bias_sgd']:.2f} SGD", flush=True)
        error_frame = pd.DataFrame({
            "approval_month": validation["RENT_APPROVAL_DATE"].to_numpy(),
            "flat_type": validation["FLAT_TYPE_CLEAN"].to_numpy(),
            "actual": y_val,
            "predicted": predicted,
        })
        monthly_tables.append(grouped_errors(error_frame, "approval_month", label))
        type_tables.append(grouped_errors(error_frame, "flat_type", label))

    score_frame = pd.DataFrame(scores)
    monthly = pd.concat(monthly_tables, ignore_index=True)
    flat_type = pd.concat(type_tables, ignore_index=True)
    score_frame.to_csv(OUT / "model_metrics.csv", index=False, float_format="%.4f")
    monthly.to_csv(OUT / "monthly_errors.csv", index=False, float_format="%.4f")
    flat_type.to_csv(OUT / "flat_type_errors.csv", index=False, float_format="%.4f")
    lowest_holdout = score_frame.loc[score_frame["rmse_sgd"].idxmin(), "model"]
    make_monthly_error_figure(monthly, lowest_holdout)

    source_hashes_after = {path.name: sha256(path) for path in SOURCES}
    if source_hashes_after != source_hashes_before:
        raise AssertionError("A source CSV changed during the run")
    report = {
        "validation_rule": f"fit month < {VALIDATION_START}; validate month >= {VALIDATION_START}",
        "fitting_rows": len(fitting),
        "validation_rows": len(validation),
        "fitting_months": [fitting["RENT_APPROVAL_DATE"].min(), fitting["RENT_APPROVAL_DATE"].max()],
        "validation_months": [validation["RENT_APPROVAL_DATE"].min(), validation["RENT_APPROVAL_DATE"].max()],
        "source_sha256": source_hashes_after,
        "versions": {"pandas": pd.__version__, "numpy": np.__version__, "catboost": catboost_version},
        "model_settings": MODEL_SETTINGS,
        "feature_sets": FEATURE_SETS,
        "categorical_features": sorted(CATEGORICAL),
        "age_anomalies_set_to_missing": int((frame["LEASE_COMMENCE_DATE"] > pd.to_datetime(frame["RENT_APPROVAL_DATE"], format="%Y-%m").dt.year).sum()),
        "scores": scores,
        "lowest_rmse_on_this_holdout": lowest_holdout,
        "notes": [
            "Mean baseline uses only the fitting-period target mean.",
            "CatBoost settings and validation rows are identical across A/B/C.",
            "No preprocessing or model uses Kaggle test target values.",
            "Scores are from one temporal holdout and should not be treated as final test performance.",
        ],
    }
    (OUT / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved results in {OUT}", flush=True)


if __name__ == "__main__":
    main()
