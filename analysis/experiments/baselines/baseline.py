"""Compare a mean predictor and three CatBoost feature sets on a future holdout.

Run from the repository root after installing environment.yml:
    python -m analysis.main baseline

All cleaning happens in DataFrame copies. The three source CSV files are never
written; their SHA-256 hashes are checked before and after this script runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

from analysis.pipeline.config import (
    CATEGORICAL, CATBOOST_SETTINGS, DATA, EXPERIMENT_REPORTS, FEATURE_SETS, SOURCES,
    VALIDATION_START,
)
from analysis.pipeline.data import check_output_path, prepare_main, sha256, temporal_split
from analysis.pipeline.evaluation import grouped_errors, make_monthly_error_figure, metrics
from analysis.pipeline.training import HDBTrainer, TrainerConfig

OUT = EXPERIMENT_REPORTS / "baseline_mean_catboost"
MODEL_SETTINGS = CATBOOST_SETTINGS


def main(output_dir=None) -> None:
    out = check_output_path(Path(output_dir) if output_dir is not None else OUT)
    from catboost import __version__ as catboost_version

    source_hashes_before = {path.name: sha256(path) for path in SOURCES}
    raw = pd.read_csv(DATA / "train.csv")
    frame = prepare_main(raw)
    fitting, validation = temporal_split(frame)
    y_fit = fitting["MONTHLY_RENT"].to_numpy(dtype=float)
    y_val = validation["MONTHLY_RENT"].to_numpy(dtype=float)
    out.mkdir(parents=True, exist_ok=True)

    predictions: dict[str, np.ndarray] = {
        "mean_constant": np.full(len(validation), y_fit.mean(), dtype=float)
    }
    for label, columns in FEATURE_SETS.items():
        print(f"Training {label} with {len(columns)} features", flush=True)
        trainer = HDBTrainer(TrainerConfig(model_family="catboost", feature_set=label))
        model = trainer.fit_frame(fitting)
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
    score_frame.to_csv(out / "model_metrics.csv", index=False, float_format="%.4f")
    monthly.to_csv(out / "monthly_errors.csv", index=False, float_format="%.4f")
    flat_type.to_csv(out / "flat_type_errors.csv", index=False, float_format="%.4f")
    lowest_holdout = score_frame.loc[score_frame["rmse_sgd"].idxmin(), "model"]
    make_monthly_error_figure(monthly, lowest_holdout, out)

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
    (out / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved results in {out}", flush=True)


if __name__ == "__main__":
    main()
