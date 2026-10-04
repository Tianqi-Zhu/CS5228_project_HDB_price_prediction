"""Compare traditional ML model families on the same temporal holdout and features.

Run the baseline command first to produce the CatBoost C result. This script trains
Ridge, Random Forest, and LightGBM in the existing conda cs5228 environment or any Python
environment with requirements-model.txt installed. Source CSVs stay unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn

from analysis.pipeline.config import (
    CATEGORICAL, DATA, EXPERIMENT_REPORTS, FAMILY_SETTINGS, FEATURE_SETS, SOURCES,
    VALIDATION_START,
)
from analysis.pipeline.data import check_output_path, prepare_main, sha256, temporal_split
from analysis.pipeline.evaluation import grouped_errors, metrics, plot_model_comparison
from analysis.pipeline.training import HDBTrainer, TrainerConfig


CATBOOST_OUT = EXPERIMENT_REPORTS / "model_baseline"
OUT = EXPERIMENT_REPORTS / "model_families"
FEATURES = FEATURE_SETS["C_property_time_town"]
MODEL_SETTINGS = FAMILY_SETTINGS


def main(output_dir=None, baseline_dir=None) -> None:
    out = check_output_path(Path(output_dir) if output_dir is not None else OUT)
    catboost_out = Path(baseline_dir) if baseline_dir is not None else CATBOOST_OUT
    baseline_path = catboost_out / "results.json"
    if not baseline_path.exists():
        raise FileNotFoundError("Run `python -m analysis.main baseline` before this comparison")
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
    out.mkdir(parents=True, exist_ok=True)
    scores = []
    monthly_tables = []

    for family in ("ridge", "random_forest", "lightgbm"):
        print(f"Training {family}", flush=True)
        trainer = HDBTrainer(TrainerConfig(model_family=family))
        model = trainer.fit_frame(fitting)
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
    prior_monthly = pd.read_csv(catboost_out / "monthly_errors.csv")
    monthly_tables.append(prior_monthly[prior_monthly["model"] == "C_property_time_town"].copy())
    combined_monthly = pd.concat(monthly_tables, ignore_index=True)
    pd.DataFrame(scores).to_csv(out / "family_metrics.csv", index=False, float_format="%.4f")
    combined_monthly.to_csv(out / "family_monthly_errors.csv", index=False, float_format="%.4f")
    plot_model_comparison(combined_monthly, out)

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
        "catboost_source": str(baseline_path),
        "note": "Single temporal holdout. Model parameters were chosen as simple fixed baselines, not tuned on this holdout.",
    }
    (out / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved model-family comparison in {out}", flush=True)


if __name__ == "__main__":
    main()
