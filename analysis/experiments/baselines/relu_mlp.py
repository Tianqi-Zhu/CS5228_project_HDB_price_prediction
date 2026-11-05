"""CPU ReLU MLP with nested chronological epoch selection.

This integrates the useful experiment from ``relu-rental-prediction`` without
restoring its duplicate root-level data pipeline or committed submission files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from analysis.pipeline.config import DATA, DATE, EXPERIMENT_REPORTS, TARGET
from analysis.pipeline.data import check_output_path, load_main, norm_text, protect_sources
from analysis.pipeline.workflow import submission_frame


CATEGORICAL = ["TOWN_CLEAN", "STREET_CLEAN", "FLAT_TYPE_CLEAN", "FLAT_MODEL_CLEAN"]
NUMERIC = [
    "FLOOR_AREA_SQM", "LEASE_COMMENCE_DATE", "FLAT_AGE_YEARS",
    "LEASE_MISSING", "APPROVAL_MONTH_INDEX", "APPROVAL_MONTH_SIN",
    "APPROVAL_MONTH_COS",
]


def feature_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic features; fitted statistics stay in the preprocessor."""
    from analysis.pipeline.data import prepare_main

    frame = prepare_main(raw)
    dates = pd.to_datetime(frame[DATE], format="%Y-%m", errors="raise")
    lease = pd.to_numeric(frame["LEASE_COMMENCE_DATE"], errors="coerce")
    invalid_lease = lease > dates.dt.year
    frame.loc[invalid_lease, "LEASE_COMMENCE_DATE"] = np.nan
    frame["LEASE_MISSING"] = frame["LEASE_COMMENCE_DATE"].isna().astype(float)
    frame["STREET_CLEAN"] = norm_text(frame["STREET"]).fillna("unknown")
    frame["APPROVAL_MONTH_SIN"] = np.sin(2 * np.pi * dates.dt.month / 12)
    frame["APPROVAL_MONTH_COS"] = np.cos(2 * np.pi * dates.dt.month / 12)
    return frame


def iqr_filter(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Learn area/target IQR fences from a fitting partition only."""
    keep = np.ones(len(frame), dtype=bool)
    bounds = {}
    for column in ("FLOOR_AREA_SQM", TARGET):
        values = pd.to_numeric(frame[column], errors="coerce")
        q1, q3 = values.quantile([0.25, 0.75])
        lower, upper = float(q1 - 1.5 * (q3 - q1)), float(q3 + 1.5 * (q3 - q1))
        bounds[column] = [lower, upper]
        keep &= values.between(lower, upper).fillna(False).to_numpy()
    retained = frame.loc[keep].copy()
    return retained, {
        "input_rows": len(frame), "retained_rows": len(retained),
        "iqr_removed": int(len(frame) - len(retained)), "iqr_bounds": bounds,
    }


def prepare_estimator(frame: pd.DataFrame, seed: int):
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), NUMERIC),
        ("category", OneHotEncoder(
            handle_unknown="ignore", sparse_output=True, dtype=np.float32,
        ), CATEGORICAL),
    ], sparse_threshold=1.0)
    x = preprocessing.fit_transform(frame[CATEGORICAL + NUMERIC]).astype(np.float32)
    y = frame[TARGET].to_numpy(np.float32)
    mean, std = float(y.mean()), float(y.std())
    if std == 0:
        raise ValueError("Constant target cannot be standardized")
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
        learning_rate_init=0.001, alpha=0.01, batch_size=512,
        shuffle=True, random_state=seed, max_iter=1, early_stopping=False,
    )
    return {
        "preprocessing": preprocessing, "model": model,
        "target_mean": mean, "target_std": std,
    }, x, (y - mean) / std


def predict(bundle, frame: pd.DataFrame) -> np.ndarray:
    x = bundle["preprocessing"].transform(frame[CATEGORICAL + NUMERIC]).astype(np.float32)
    scaled = bundle["model"].predict(x).astype(float)
    return scaled * bundle["target_std"] + bundle["target_mean"]


def metrics(actual, predicted) -> dict:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
        "bias": float(np.mean(np.asarray(predicted) - np.asarray(actual))),
    }


def choose_epochs(
    fitting: pd.DataFrame, tuning: pd.DataFrame, *, max_epochs: int, patience: int, seed: int,
):
    retained, cleanup = iqr_filter(fitting)
    bundle, x, y = prepare_estimator(retained, seed)
    history, best_rmse, best_epoch, stale = [], float("inf"), 1, 0
    for epoch in range(1, max_epochs + 1):
        bundle["model"].partial_fit(x, y)
        predicted = predict(bundle, tuning)
        score = metrics(tuning[TARGET], predicted)
        history.append({"epoch": epoch, "training_loss_scaled": float(bundle["model"].loss_), **score})
        if score["rmse"] < best_rmse - 0.1:
            best_rmse, best_epoch, stale = score["rmse"], epoch, 0
        else:
            stale += 1
            if stale >= patience:
                break
    return best_epoch, history, cleanup


def fit_epochs(frame: pd.DataFrame, *, epochs: int, seed: int):
    retained, cleanup = iqr_filter(frame)
    bundle, x, y = prepare_estimator(retained, seed)
    for _ in range(epochs):
        bundle["model"].partial_fit(x, y)
    return bundle, cleanup


def run(
    output_dir: Path = EXPERIMENT_REPORTS / "baseline_relu_mlp",
    *, max_epochs: int = 80,
    patience: int = 12,
    seed: int = 42,
    write_submission: bool = False,
):
    output_dir = check_output_path(Path(output_dir)); output_dir.mkdir(parents=True, exist_ok=True)
    sources = [DATA / "train.csv", DATA / "test.csv", DATA / "example-submission.csv"]
    started = time.time()
    # Two numerical threads preserve the source branch's deterministic MLP run.
    with protect_sources(sources) as hashes, threadpool_limits(limits=2):
        train = feature_frame(load_main(DATA / "train.csv", require_target=True))
        selection_fit = train.loc[train[DATE] < "2024-10"].copy()
        tuning = train.loc[(train[DATE] >= "2024-10") & (train[DATE] < "2025-01")].copy()
        holdout_fit = train.loc[train[DATE] < "2025-01"].copy()
        holdout = train.loc[(train[DATE] >= "2025-01") & (train[DATE] < "2025-04")].copy()
        if min(map(len, (selection_fit, tuning, holdout_fit, holdout))) == 0:
            raise ValueError("Nested temporal split contains an empty partition")
        epochs, history, selection_cleanup = choose_epochs(
            selection_fit, tuning, max_epochs=max_epochs, patience=patience, seed=seed,
        )
        bundle, holdout_cleanup = fit_epochs(holdout_fit, epochs=epochs, seed=seed)
        prediction = predict(bundle, holdout)
        score = metrics(holdout[TARGET], prediction)

        holdout_predictions = pd.DataFrame({
            "index": holdout.index, DATE: holdout[DATE].to_numpy(),
            "actual": holdout[TARGET].to_numpy(float), "prediction": prediction,
        })
        holdout_predictions.to_csv(output_dir / "holdout_predictions.csv", index=False)
        pd.DataFrame(history).to_csv(output_dir / "epoch_selection.csv", index=False)
        monthly = holdout_predictions.groupby(DATE).apply(
            lambda part: pd.Series(metrics(part["actual"], part["prediction"])),
            include_groups=False,
        ).reset_index()
        monthly.to_csv(output_dir / "monthly_errors.csv", index=False)

        final_cleanup = None
        if write_submission:
            test = feature_frame(load_main(DATA / "test.csv"))
            final_bundle, final_cleanup = fit_epochs(train, epochs=epochs, seed=seed)
            test_prediction = predict(final_bundle, test)
            template = pd.read_csv(DATA / "example-submission.csv")
            submission_frame(template, test_prediction, len(test)).to_csv(
                output_dir / "submission.csv", index=False,
            )
            joblib.dump(final_bundle, output_dir / "model.joblib")

    result = {
        "model": "sklearn_relu_mlp_nested_temporal",
        "architecture": [128, 64, 1], "seed": seed, "selected_epochs": epochs,
        "selection_range": ["2024-10", "2024-12"],
        "holdout_range": ["2025-01", "2025-03"],
        "holdout_rows": len(holdout), "metrics": score,
        "selection_cleanup": selection_cleanup, "holdout_cleanup": holdout_cleanup,
        "final_cleanup": final_cleanup, "source_sha256": hashes,
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write-submission", action="store_true")
    args = parser.parse_args(argv)
    return run(
        args.output_dir or EXPERIMENT_REPORTS / "baseline_relu_mlp", max_epochs=args.max_epochs,
        patience=args.patience, seed=args.seed, write_submission=args.write_submission,
    )


if __name__ == "__main__":
    main()
