"""Reproducible final refit/export of the documented compact baseline.

The 27-feature, fixed-400-tree model is a simple reproducible reference, not a
claim that it wins the search. Its separately measured latest RMSE is 468.49.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import joblib
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from analysis.experiments.feature_selection import SOURCES
from analysis.pipeline.config import DATA, EXPERIMENT_REPORTS, FAMILY_SETTINGS, ROOT, TARGET
from analysis.pipeline.data import check_output_path, load_main, protect_sources, sha256
from analysis.pipeline.features import FEATURE_CANDIDATES, RentalFeatureEngineer


def validate_submission(template, prediction):
    prediction = np.asarray(prediction, float)
    if list(template.columns) != ["Id", "Predicted"]:
        raise ValueError("Expected submission columns Id,Predicted")
    if prediction.shape != (len(template),) or not np.isfinite(prediction).all():
        raise ValueError("Missing, infinite or misaligned predictions")
    if not np.array_equal(template.Id.to_numpy(), np.arange(len(template))):
        raise ValueError("Submission IDs must follow original test row order")
    result = template.copy()
    result["Predicted"] = prediction
    return result


def run(output_dir=ROOT / "output/submissions/baselines/baseline_27_features",
        report_dir=EXPERIMENT_REPORTS / "baseline_27_feature_submission"):
    output = check_output_path(Path(output_dir))
    report = check_output_path(Path(report_dir))
    output.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    sources = SOURCES + [DATA / "example-submission.csv"]
    with protect_sources(sources) as hashes:
        train = load_main(DATA / "train.csv", require_target=True)
        test = load_main(DATA / "test.csv")
        template = pd.read_csv(DATA / "example-submission.csv")
        pipeline = Pipeline([("features", RentalFeatureEngineer("compact_history_27")),
            ("model", LGBMRegressor(**FAMILY_SETTINGS["lightgbm"]))])
        pipeline.fit(train.drop(columns=TARGET), train[TARGET])
        prediction = pipeline.predict(test)
        saved_model = output / "model.joblib"
        joblib.dump(pipeline, saved_model)
        restored = joblib.load(saved_model)
        restored_prediction = restored.predict(test)
        if not np.allclose(prediction, restored_prediction, rtol=0, atol=1e-9):
            raise ValueError("Saved model predictions differ after reload")
        submission = validate_submission(template, restored_prediction)
        path = output / "submission.csv"
        submission.to_csv(path, index=False)
        reread = pd.read_csv(path)
        if list(reread.columns) != list(submission.columns) or not np.array_equal(reread.Id, submission.Id):
            raise ValueError("Saved CSV schema/IDs differ")
        if not np.allclose(reread.Predicted, submission.Predicted, rtol=0, atol=1e-9):
            raise ValueError("Saved CSV prediction values differ")
        summary = {"variant": "compact_history_27", "model_settings": FAMILY_SETTINGS["lightgbm"],
            "feature_spec": asdict(FEATURE_CANDIDATES["compact_history_27"]),
            "fitting_rows": len(train), "test_rows": len(test),
            "fitting_months": [train.RENT_APPROVAL_DATE.min(), train.RENT_APPROVAL_DATE.max()],
            "test_months": [test.RENT_APPROVAL_DATE.min(), test.RENT_APPROVAL_DATE.max()],
            "sources": hashes, "submission_sha256": sha256(path),
            "prediction_min": float(prediction.min()), "prediction_max": float(prediction.max()),
            "model_reload_max_difference": float(np.max(np.abs(prediction-restored_prediction))),
            "model_path": str(saved_model.relative_to(ROOT)) if saved_model.is_relative_to(ROOT) else saved_model.name,
            "submission_path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name,
            "note": "Fixed compact baseline for reproducibility, not the tuned best model. No public-source rents or test labels used; no upload. Test-month overlap excludes same-month labels in historical features, but base model fitting uses all known training labels.",
            "actual_kaggle_score": None}
        (report / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {len(submission):,} canonical predictions: {path}", flush=True)
    return summary
