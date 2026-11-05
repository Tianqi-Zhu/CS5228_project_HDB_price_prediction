"""Validation, full-data refitting and Kaggle-format prediction outputs."""

from __future__ import annotations

from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.pipeline.config import (
    DATA, DATE, EXPERIMENT_REPORTS, MODELS, SOURCES, SUBMISSIONS, TARGET,
)
from analysis.pipeline.data import HDBPreprocessor, PreprocessorConfig, check_output_path, load_main, protect_sources
from analysis.pipeline.evaluation import grouped_errors, make_monthly_error_figure, metrics
from analysis.pipeline.training import HDBTrainer, TrainerConfig


def versions():
    return {name: version(name) for name in ("pandas", "numpy", "scikit-learn", "catboost", "lightgbm")}


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def run_validation(config: TrainerConfig, output_dir=None):
    if config.split_strategy != "time":
        raise ValueError("The validation workflow uses time splits; random splits are a separate diagnostic")
    window = "all_history" if config.fit_window_months is None else f"last_{config.fit_window_months}m"
    label = f"{config.model_family}_{config.feature_set}_{config.validation_start}_{window}"
    out = check_output_path(
        Path(output_dir) if output_dir is not None
        else EXPERIMENT_REPORTS / "validation" / label
    )
    with protect_sources(SOURCES) as hashes:
        processor = HDBPreprocessor(PreprocessorConfig(feature_set=config.feature_set))
        frame = processor.transform(load_main(DATA / "train.csv", require_target=True))
        trainer = HDBTrainer(config)
        fitting, validation = trainer.split_frame(frame)
        model = trainer.fit_frame(fitting)
        predicted = np.asarray(model.predict(validation[trainer.features]), dtype=float)
        scores = metrics(validation[TARGET].to_numpy(), predicted)
        errors = pd.DataFrame({
            "source_row": validation.index.to_numpy(),
            "approval_month": validation[DATE].to_numpy(),
            "flat_type": validation["FLAT_TYPE_CLEAN"].to_numpy(),
            "town": validation["TOWN_CLEAN"].to_numpy(),
            "actual": validation[TARGET].to_numpy(), "predicted": predicted,
        })
        errors["residual"] = errors["actual"] - errors["predicted"]
        out.mkdir(parents=True, exist_ok=True)
        errors.to_csv(out / "predictions.csv", index=False)
        monthly = grouped_errors(errors, "approval_month", config.model_family)
        monthly.to_csv(out / "monthly_errors.csv", index=False)
        grouped_errors(errors, "flat_type", config.model_family).to_csv(out / "flat_type_errors.csv", index=False)
        grouped_errors(errors, "town", config.model_family).to_csv(out / "town_errors.csv", index=False)
        make_monthly_error_figure(monthly, config.model_family, out)
        result = {
            "config": asdict(config), "features": trainer.features,
            "fitting_rows": len(fitting), "validation_rows": len(validation),
            "fitting_months": [fitting[DATE].min(), fitting[DATE].max()],
            "validation_months": [validation[DATE].min(), validation[DATE].max()],
            "scores": scores, "source_sha256": hashes, "versions": versions(),
            "note": "Local temporal validation; no Kaggle test labels or tuned hyperparameters.",
        }
        write_json(out / "results.json", result)
    print(f"Validation: RMSE={scores['rmse_sgd']:.2f}; MAE={scores['mae_sgd']:.2f}; output={out}")
    return result


def submission_frame(template: pd.DataFrame, predicted, test_rows: int):
    """Use template IDs and positional predictions; never deduplicate test rows."""
    if list(template.columns) != ["Id", "Predicted"]:
        raise ValueError("Submission template must contain exactly Id,Predicted")
    if len(template) != test_rows:
        raise ValueError("Submission template and test row counts differ")
    if template["Id"].isna().any() or template["Id"].duplicated().any():
        raise ValueError("Submission IDs must be present and unique")
    predicted = np.asarray(predicted, dtype=float)
    if predicted.ndim != 1 or len(predicted) != test_rows or not np.isfinite(predicted).all():
        raise ValueError("Predictions must be a finite vector with one value per test row")
    output = template.copy(deep=True)
    output["Predicted"] = predicted
    return output


def run_submission(config: TrainerConfig, output_path=None, model_path=None):
    """Refit the chosen configuration on all labels (or an explicit recent window).

    Does not upload anything to Kaggle. The model artifact includes its own
    preprocessor configuration, feature names and training provenance.
    """
    label = f"{config.model_family}_{config.feature_set}"
    output = check_output_path(Path(output_path) if output_path is not None else SUBMISSIONS / "baselines" / f"{label}.csv")
    artifact = check_output_path(Path(model_path) if model_path is not None else MODELS / "baseline_submissions" / f"{label}.joblib")
    if output.suffix.lower() != ".csv" or artifact.suffix.lower() != ".joblib":
        raise ValueError("Submission path must end in .csv and model path in .joblib")
    with protect_sources(SOURCES) as hashes:
        prep_config = PreprocessorConfig(feature_set=config.feature_set)
        processor = HDBPreprocessor(prep_config)
        raw_train = load_main(DATA / "train.csv", require_target=True)
        raw_test = load_main(DATA / "test.csv")
        fitting = processor.transform(raw_train)
        test = processor.transform(raw_test)
        if config.fit_window_months is not None:
            if config.fit_window_months < 1:
                raise ValueError("fit_window_months must be positive")
            dates = pd.to_datetime(fitting[DATE], format="%Y-%m")
            end = dates.max() + pd.DateOffset(months=1)
            fitting = fitting.loc[dates >= end - pd.DateOffset(months=config.fit_window_months)].copy()
        trainer = HDBTrainer(config)
        model = trainer.fit_frame(fitting)
        predicted = model.predict(test[trainer.features])
        template = pd.read_csv(DATA / "example-submission.csv")
        result = submission_frame(template, predicted, len(raw_test))
        if not test.index.equals(raw_test.index):
            raise AssertionError("Test row order changed")
        metadata = {
            "config": asdict(config), "preprocessor_config": asdict(prep_config),
            "features": trainer.features, "fitting_rows": len(fitting),
            "fitting_months": [fitting[DATE].min(), fitting[DATE].max()],
            "test_rows": len(result), "source_sha256": hashes, "versions": versions(),
            "model_path": str(artifact),
            "note": "Refitted on all available training labels unless a recent window was requested; not uploaded to Kaggle.",
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_model({"model": model, **metadata}, artifact)
        result.to_csv(output, index=False)
        write_json(output.with_suffix(".json"), metadata)
    print(f"Saved {len(result):,} test predictions to {output}; model={artifact}")
    return metadata


def predict_saved(model_path, output_path=None):
    bundle = HDBTrainer.load_model(model_path)
    output = check_output_path(Path(output_path) if output_path is not None else SUBMISSIONS / "baselines" / "saved_model.csv")
    if output.suffix.lower() != ".csv":
        raise ValueError("Submission path must end in .csv")
    if not isinstance(bundle, dict) or "preprocessor_config" not in bundle:
        raise ValueError("Use an artifact generated by the submit command")
    with protect_sources(SOURCES):
        raw = load_main(DATA / "test.csv")
        prepared = HDBPreprocessor(PreprocessorConfig(**bundle["preprocessor_config"])).transform(raw)
        predicted = bundle["model"].predict(prepared[bundle["features"]])
        result = submission_frame(pd.read_csv(DATA / "example-submission.csv"), predicted, len(raw))
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False)
    print(f"Saved predictions from the persisted model to {output}")
    return result
