"""Strong auxiliary-feature LightGBM benchmark for the deep models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from analysis.experiments.deep_learning import (
    AUX_CATEGORICAL, AUX_NUMERIC, add_target_encodings,
)
from analysis.pipeline.auxiliary import build_auxiliary_features
from analysis.pipeline.config import DATA, DATE, EXPERIMENT_REPORTS, TARGET, VALIDATION_START
from analysis.pipeline.data import check_output_path, load_main, protect_sources, temporal_split
from analysis.pipeline.paths import model_artifact_dir


def _categorical_frames(fitting, validation, columns):
    fitting = fitting.copy(); validation = validation.copy()
    for column in columns:
        levels = pd.Index(fitting[column].fillna("unknown").astype(str).unique())
        dtype = pd.CategoricalDtype(categories=levels)
        fitting[column] = fitting[column].fillna("unknown").astype(str).astype(dtype)
        values = validation[column].fillna("unknown").astype(str)
        validation[column] = values.where(values.isin(levels)).astype(dtype)
    return fitting, validation


def run(output_dir: Path, *, num_leaves=31, seed=42, target_encoding=True):
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    sources = [DATA / "train.csv", DATA / "auxiliary" / "sg-hdb-block.csv",
               DATA / "auxiliary" / "sg-mrt-stations.csv",
               DATA / "auxiliary" / "sg-shopping-malls.csv",
               DATA / "auxiliary" / "sg-schools.csv"]
    started = time.time()
    with protect_sources(sources) as hashes:
        frame = build_auxiliary_features(load_main(DATA / "train.csv", require_target=True))
        fitting, validation = temporal_split(frame, VALIDATION_START)
        numeric = list(AUX_NUMERIC)
        if target_encoding:
            fitting, validation, encoded = add_target_encodings(fitting, validation, window_months=18)
            numeric += encoded
        features = list(AUX_CATEGORICAL) + numeric
        fitting, validation = _categorical_frames(fitting, validation, AUX_CATEGORICAL)
        model = LGBMRegressor(
            objective="regression", n_estimators=3000, learning_rate=0.02,
            num_leaves=num_leaves, min_child_samples=35, max_bin=255,
            feature_fraction=0.90, bagging_fraction=0.90, bagging_freq=1,
            reg_lambda=5.0, reg_alpha=0.2, random_state=seed, n_jobs=12,
            verbosity=-1,
        )
        model.fit(
            fitting[features], fitting[TARGET],
            eval_set=[(validation[features], validation[TARGET])], eval_metric="rmse",
            categorical_feature=AUX_CATEGORICAL,
            callbacks=[early_stopping(120, verbose=False), log_evaluation(100)],
        )
        predictions = model.predict(validation[features], num_iteration=model.best_iteration_)
        actual = validation[TARGET].to_numpy(float)
        metrics = {
            "rmse": float(mean_squared_error(actual, predictions) ** 0.5),
            "mae": float(mean_absolute_error(actual, predictions)),
            "bias": float(np.mean(predictions - actual)),
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"index": validation.index, DATE: validation[DATE].to_numpy(),
                  "actual": actual, "prediction": predictions}).to_csv(output_dir / "predictions.csv", index=False)
    artifact = {
        "model": "lightgbm_auxiliary", "metrics": metrics, "num_leaves": num_leaves,
        "seed": seed, "target_encoding": target_encoding, "best_iteration": model.best_iteration_,
        "features": features, "source_sha256": hashes, "elapsed_seconds": time.time() - started,
    }
    (output_dir / "results.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    artifacts = model_artifact_dir(output_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifacts / "model.joblib")
    print(json.dumps(artifact, indent=2))
    return artifact


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--without-target-encoding", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    output = check_output_path(
        args.output_dir or EXPERIMENT_REPORTS / "baseline_deep_models" / "aux_lightgbm"
    )
    return run(output, num_leaves=args.num_leaves, seed=args.seed,
               target_encoding=not args.without_target_encoding)


if __name__ == "__main__":
    main()
