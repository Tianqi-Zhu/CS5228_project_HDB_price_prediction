"""CatBoost benchmark on leakage-conscious auxiliary and historical features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from analysis.experiments.deep_learning import AUX_CATEGORICAL, AUX_NUMERIC, add_target_encodings
from analysis.pipeline.auxiliary import MACRO_NUMERIC, build_auxiliary_features
from analysis.pipeline.config import DATA, DATE, EXPERIMENT_REPORTS, TARGET, VALIDATION_START
from analysis.pipeline.data import check_output_path, load_main, protect_sources, temporal_split
from analysis.pipeline.paths import model_artifact_dir


def _prepare_categories(fitting: pd.DataFrame, validation: pd.DataFrame, columns: list[str]):
    fitting = fitting.copy()
    validation = validation.copy()
    for column in columns:
        fitting[column] = fitting[column].fillna("unknown").astype(str)
        validation[column] = validation[column].fillna("unknown").astype(str)
    return fitting, validation


def run(
    output_dir: Path,
    *,
    depth: int = 8,
    seed: int = 42,
    target_encoding: bool = True,
    task_type: str = "GPU",
    macro_auxiliary: bool = False,
):
    from catboost import CatBoostRegressor

    sources = [
        DATA / "train.csv",
        DATA / "auxiliary" / "sg-hdb-block.csv",
        DATA / "auxiliary" / "sg-mrt-stations.csv",
        DATA / "auxiliary" / "sg-shopping-malls.csv",
        DATA / "auxiliary" / "sg-schools.csv",
    ]
    if macro_auxiliary:
        sources += [
            DATA / "auxiliary" / "sg-coe-prices.csv",
            DATA / "auxiliary" / "sg-stock-prices.csv",
        ]
    started = time.time()
    with protect_sources(sources) as hashes:
        frame = build_auxiliary_features(
            load_main(DATA / "train.csv", require_target=True), include_macro=macro_auxiliary,
        )
        fitting, validation = temporal_split(frame, VALIDATION_START)
        numeric = list(AUX_NUMERIC)
        if macro_auxiliary:
            numeric += MACRO_NUMERIC
        if target_encoding:
            fitting, validation, encoded = add_target_encodings(
                fitting, validation, window_months=18,
            )
            numeric += encoded
        categorical = list(AUX_CATEGORICAL)
        fitting, validation = _prepare_categories(fitting, validation, categorical)
        features = categorical + numeric
        model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            iterations=4000,
            learning_rate=0.035,
            depth=depth,
            l2_leaf_reg=8.0,
            random_strength=0.35,
            random_seed=seed,
            task_type=task_type.upper(),
            devices="0" if task_type.upper() == "GPU" else None,
            od_type="Iter",
            od_wait=180,
            allow_writing_files=False,
            verbose=100,
        )
        model.fit(
            fitting[features], fitting[TARGET],
            cat_features=categorical,
            eval_set=(validation[features], validation[TARGET]),
            use_best_model=True,
        )
        predictions = model.predict(validation[features])
        actual = validation[TARGET].to_numpy(float)
        metrics = {
            "rmse": float(mean_squared_error(actual, predictions) ** 0.5),
            "mae": float(mean_absolute_error(actual, predictions)),
            "bias": float(np.mean(predictions - actual)),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "index": validation.index,
        DATE: validation[DATE].to_numpy(),
        "actual": actual,
        "prediction": predictions,
    }).to_csv(output_dir / "predictions.csv", index=False)
    artifact = {
        "model": "catboost_auxiliary",
        "metrics": metrics,
        "depth": depth,
        "seed": seed,
        "task_type": task_type.upper(),
        "target_encoding": target_encoding,
        "macro_auxiliary": macro_auxiliary,
        "best_iteration": model.get_best_iteration(),
        "features": features,
        "source_sha256": hashes,
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "results.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    artifacts = model_artifact_dir(output_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    model.save_model(artifacts / "model.cbm")
    print(json.dumps(artifact, indent=2))
    return artifact


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task-type", choices=("GPU", "CPU"), default="GPU")
    parser.add_argument("--without-target-encoding", action="store_true")
    parser.add_argument("--macro-auxiliary", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    output = check_output_path(
        args.output_dir or EXPERIMENT_REPORTS / "baseline_deep_models" / f"aux_catboost_{args.task_type.lower()}"
    )
    return run(
        output,
        depth=args.depth,
        seed=args.seed,
        target_encoding=not args.without_target_encoding,
        task_type=args.task_type,
        macro_auxiliary=args.macro_auxiliary,
    )


if __name__ == "__main__":
    main()
