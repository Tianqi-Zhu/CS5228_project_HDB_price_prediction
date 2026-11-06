"""Resumable model search with inner time selection and untouched outer refits."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
from importlib.metadata import version
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from analysis.experiments.feature_selection import SOURCES
from analysis.pipeline.config import DATA, DATE, EXPERIMENT_REPORTS, TARGET
from analysis.pipeline.data import check_output_path, load_main, protect_sources, sha256
from analysis.pipeline.evaluation import grouped_errors, metrics
from analysis.pipeline.paths import feature_cache_path, model_artifact_dir
from analysis.pipeline.features import (
    COMPACT, COMPACT_CATEGORIES, FEATURE_CANDIDATES, FeatureSpec, build_feature_frame, model_inputs,
)
from analysis.pipeline.neural_models import enable_search_dependencies, fit_neural, predict_neural


OUT = EXPERIMENT_REPORTS / "comparison_model_search"
OUTER_FOLDS = {"2024-04": "2025-04", "2023-04": "2024-04", "2022-04": "2023-04", "2023-12": "2025-04"}
ENTITY_COLUMNS = ("STREET_CLEAN", "ADDRESS_KEY", "SUBZONE_CLEAN")
PROFILES = {
    "compact": FEATURE_CANDIDATES["compact_history_27"],
    "premium": FEATURE_CANDIDATES["compact_premium_research"],
    "entities": FeatureSpec(COMPACT + ENTITY_COLUMNS, COMPACT_CATEGORIES + ENTITY_COLUMNS, True,
        history_grains=("BLOCK_FLAT", "AREA_FLAT")),
    "entities_residual": FeatureSpec(COMPACT + ENTITY_COLUMNS, COMPACT_CATEGORIES + ENTITY_COLUMNS, True,
        history_grains=("BLOCK_FLAT", "AREA_FLAT"), market_context=True),
    "compact_residual": FeatureSpec(COMPACT, COMPACT_CATEGORIES, True,
        history_grains=("BLOCK_FLAT", "AREA_FLAT"), market_context=True),
}
TRIALS = {
    "lightgbm": [
        ("compact", {"num_leaves": 31, "min_child_samples": 40, "reg_lambda": 10.}),
        ("premium", {"num_leaves": 31, "min_child_samples": 40, "reg_lambda": 10.}),
        ("compact", {"num_leaves": 63, "min_child_samples": 60, "reg_lambda": 10.}),
        ("entities_residual", {"num_leaves": 31, "min_child_samples": 40, "reg_lambda": 10.}),
    ],
    "catboost": [
        ("compact", {"depth": 6, "l2_leaf_reg": 10.}),
        ("entities", {"depth": 8, "l2_leaf_reg": 12.}),
        ("entities_residual", {"depth": 6, "l2_leaf_reg": 10.}),
    ],
    "xgboost": [
        ("compact", {"max_depth": 4, "reg_lambda": 10.}),
        ("premium", {"max_depth": 6, "reg_lambda": 15.}),
        ("entities_residual", {"max_depth": 4, "reg_lambda": 10.}),
    ],
    "hist_gradient_boosting": [("compact", {"max_iter": 400}), ("compact_residual", {"max_iter": 400})],
    "extra_trees": [("compact", {"min_samples_leaf": 5}), ("compact", {"min_samples_leaf": 10})],
    "embedding_mlp": [
        ("compact", {"width": 256, "lr": .001, "quantile": True}),
        ("entities", {"width": 256, "lr": .001, "quantile": True}),
        ("entities_residual", {"width": 256, "lr": .001, "quantile": True}),
    ],
    "ft_transformer": [
        ("compact", {"width": 96, "lr": .0003, "quantile": True}),
        ("entities", {"width": 96, "lr": .0003, "quantile": True}),
        ("entities_residual", {"width": 96, "lr": .0003, "quantile": True}),
    ],
    "tabnet": [
        ("compact", {"width": 24, "layers": 4, "lr": .01, "quantile": False}),
        ("entities", {"width": 32, "layers": 4, "lr": .01, "quantile": True}),
        ("entities_residual", {"width": 24, "layers": 4, "lr": .01, "quantile": False}),
    ],
    "embedding_resnet": [
        ("compact", {"width": 256, "lr": .001, "quantile": True}),
        ("entities", {"width": 256, "lr": .001, "quantile": True, "min_category_count": 10}),
        ("entities_residual", {"width": 256, "lr": .001, "quantile": True, "min_category_count": 10}),
        ("compact", {"width": 128, "layers": 2, "lr": .0003, "quantile": True}),
        ("entities_residual", {"width": 256, "layers": 2, "lr": .0003,
            "quantile": True, "min_category_count": 10}),
    ],
    "ft_transformer_default": [
        ("compact", {"width": 192, "lr": .0001, "weight_decay": .00001, "quantile": True, "batch_size": 128}),
        ("entities_residual", {"width": 192, "lr": .0001, "weight_decay": .00001,
            "quantile": True, "batch_size": 128, "min_category_count": 10}),
    ],
}
NEURAL = {"embedding_mlp", "ft_transformer", "tabnet", "embedding_resnet", "ft_transformer_default"}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def diagnose_data(out=OUT):
    frame = load_main(DATA / "train.csv", require_target=True)
    keys = [column for column in frame if column not in (TARGET, "FEE", "FURNISHED")]
    rows = []
    for name, part in (("all_train", frame), ("latest_outer", frame.loc[frame[DATE] >= "2024-04"])):
        grouped = part.groupby(keys, dropna=False)[TARGET]
        mean = grouped.transform("mean")
        n = grouped.transform("size")
        squared_error = (part[TARGET] - mean) ** 2
        groups = grouped.ngroups
        rows.append({"partition": name, "rows": len(part), "identical_input_groups": groups,
            "rows_with_repeated_inputs": int((n > 1).sum()),
            "in_sample_identical_input_oracle_rmse": float(np.sqrt(squared_error.mean())),
            "repeat_group_pooled_within_sd": float(np.sqrt(squared_error.sum() / (len(part) - groups))),
            "note": "Oracle uses partition labels solely to diagnose identical-input ambiguity, never as a trained model or reported validation score. The pooled within-group SD is a descriptive repeat-sample estimate, not a population-wide lower bound."})
    write_json(Path(out) / "data_diagnostics.json", rows)
    print(json.dumps(rows, indent=2), flush=True)
    return rows


def prepare_fold(start, out=OUT):
    out = check_output_path(Path(out))
    inner_start = str(pd.Period(start, freq="M") - 6)
    with protect_sources(SOURCES) as hashes:
        frame = build_feature_frame(load_main(DATA / "train.csv", require_target=True))
        fitting = frame.loc[frame[DATE] < start].copy()
        validation = frame.loc[(frame[DATE] >= start) & (frame[DATE] < OUTER_FOLDS[start])].copy()
        inner_fit = fitting.loc[fitting[DATE] < inner_start].copy()
        inner_val = fitting.loc[fitting[DATE] >= inner_start].copy()
        for name, candidate in PROFILES.items():
            destination = feature_cache_path(out, start, name)
            if destination.exists():
                existing = joblib.load(destination)
                if existing["source_sha256"] == hashes and existing["spec"] == asdict(candidate):
                    print(f"Using prepared {start} {name}", flush=True)
                    continue
            stages = {}
            for stage, fit, val in (("inner", inner_fit, inner_val), ("outer", fitting, validation)):
                x_fit, x_val = model_inputs(fit, val, candidate)
                fit_offset, val_offset = np.zeros(len(fit)), np.zeros(len(val))
                eligible = np.ones(len(fit), dtype=bool)
                if name.endswith("residual"):
                    fit_offset = x_fit["MARKET_TYPE_LAST"].to_numpy(float)
                    val_offset = x_val["MARKET_TYPE_LAST"].to_numpy(float)
                    warmup = str(pd.Period(fit[DATE].min(), freq="M") + 6)
                    eligible = np.isfinite(fit_offset) & (fit[DATE].to_numpy() >= warmup)
                    if not np.isfinite(val_offset).all():
                        raise ValueError("No past market baseline for a validation row")
                stages[stage] = {"x_fit": x_fit.loc[eligible].copy(), "x_val": x_val,
                    "y_fit": fit[TARGET].to_numpy(float)[eligible] - fit_offset[eligible],
                    "y_val": val[TARGET].to_numpy(float) - val_offset,
                    "actual": val[TARGET].to_numpy(float), "offset": val_offset,
                    "source_row": val.index.to_numpy(), "approval_month": val[DATE].to_numpy(),
                    "n_fit": int(eligible.sum()), "fitting_months": [fit[DATE].min(), fit[DATE].max()]}
            destination.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"source_sha256": hashes, "spec": asdict(candidate), "categorical": candidate.categorical,
                "profile": name, "inner_start": inner_start, **stages}, destination)
            print(f"Prepared {start} {name}: inner {len(inner_fit)}/{len(inner_val)}, outer {len(fitting)}/{len(validation)}", flush=True)


def fit_tree(family, stage, categorical, settings, fixed_iterations=None):
    x_fit, x_val = stage["x_fit"].copy(), stage["x_val"].copy()
    y_fit, y_val = stage["y_fit"], stage["y_val"]
    if family == "lightgbm":
        import lightgbm as lgb
        model = lgb.LGBMRegressor(n_estimators=fixed_iterations or 2500, learning_rate=.03,
            random_state=42, n_jobs=4, verbosity=-1, **settings)
        fit_args = {} if fixed_iterations else {"eval_set": [(x_val, y_val)],
            "callbacks": [lgb.early_stopping(120, verbose=False)]}
        model.fit(x_fit, y_fit, categorical_feature=list(categorical), **fit_args)
        iterations = fixed_iterations or int(model.best_iteration_)
    elif family == "catboost":
        from catboost import CatBoostRegressor
        for column in categorical:
            x_fit[column] = x_fit[column].astype("string").fillna("__UNKNOWN__").astype(str)
            x_val[column] = x_val[column].astype("string").fillna("__UNKNOWN__").astype(str)
        model = CatBoostRegressor(iterations=fixed_iterations or 2500, learning_rate=.04,
            loss_function="RMSE", random_seed=42, thread_count=4, task_type="CPU",
            allow_writing_files=False, verbose=False, **settings)
        fit_args = {} if fixed_iterations else {"eval_set": (x_val, y_val), "early_stopping_rounds": 120,
            "use_best_model": True}
        model.fit(x_fit, y_fit, cat_features=list(categorical), **fit_args)
        iterations = fixed_iterations or int(model.get_best_iteration()) + 1
    elif family == "xgboost":
        enable_search_dependencies()
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=fixed_iterations or 2500, learning_rate=.03,
            tree_method="hist", device="cpu", enable_categorical=True, n_jobs=4,
            objective="reg:squarederror", random_state=42, subsample=.9, colsample_bytree=.9,
            early_stopping_rounds=None if fixed_iterations else 120, **settings)
        model.fit(x_fit, y_fit, **({} if fixed_iterations else {"eval_set": [(x_val, y_val)], "verbose": False}))
        iterations = fixed_iterations or int(model.best_iteration) + 1
    elif family == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(learning_rate=.05, max_leaf_nodes=31,
            min_samples_leaf=30, l2_regularization=10., categorical_features="from_dtype",
            early_stopping=False, random_state=42, **settings)
        model.fit(x_fit, y_fit)
        iterations = int(settings["max_iter"])
    elif family == "extra_trees":
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
        numeric = [column for column in x_fit if column not in categorical]
        preprocess = ColumnTransformer([
            ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric),
            ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(categorical)),
        ])
        model = Pipeline([("preprocess", preprocess), ("model", ExtraTreesRegressor(
            n_estimators=300, max_depth=28, max_features=.85, random_state=42, n_jobs=4, **settings))])
        model.fit(x_fit, y_fit)
        iterations = 300
    else:
        raise ValueError(family)
    prediction = np.asarray(model.predict(x_val), float)
    return model, prediction, iterations


def prediction_frame(stage, prediction):
    return pd.DataFrame({"source_row": stage["source_row"], "approval_month": stage["approval_month"],
        "actual": stage["actual"], "prediction": prediction})


def run_family(family, start="2024-04", out=OUT, max_epochs=40, device="cuda"):
    out = check_output_path(Path(out))
    destination = out / start / family
    if (destination / "results.json").exists():
        completed = list(destination.glob("trial_*/results.json"))
        if len(completed) >= len(TRIALS[family]):
            print(f"Completed job already exists: {start} {family}", flush=True)
            return json.loads((destination / "results.json").read_text(encoding="utf-8"))
    trials = []
    for index, (profile, settings) in enumerate(TRIALS[family]):
        directory = destination / f"trial_{index}"
        if (directory / "results.json").exists():
            trials.append(json.loads((directory / "results.json").read_text(encoding="utf-8")))
            continue
        cache = feature_cache_path(out, start, profile)
        if not cache.exists():
            prepare_fold(start, out)
        prepared = joblib.load(cache)
        stage = prepared["inner"]
        started = time.perf_counter()
        print(f"Selecting {start} {family} trial {index}: {profile}", flush=True)
        if family in NEURAL:
            settings = {"max_epochs": max_epochs, "patience": 7, "seed": 42, **settings}
            fitted = fit_neural(family, stage["x_fit"], stage["y_fit"], prepared["categorical"], settings,
                stage["x_val"], stage["y_val"], device_name=device)
            prediction = predict_neural(fitted, stage["x_val"]) + stage["offset"]
            iterations = fitted["epochs"]
            directory.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(fitted["history"]).to_csv(directory / "history.csv", index=False)
            del fitted
            import torch
            torch.cuda.empty_cache()
        else:
            fitted, prediction, iterations = fit_tree(family, stage, prepared["categorical"], settings)
            prediction += stage["offset"]
            del fitted
        gc.collect()
        directory.mkdir(parents=True, exist_ok=True)
        record = {"family": family, "profile": profile, "settings": settings,
            "selected_iterations": int(iterations), "inner_scores": metrics(stage["actual"], prediction),
            "n_fit": stage["n_fit"], "n_inner_validation": len(prediction),
            "elapsed_seconds": time.perf_counter() - started, "trial_index": index}
        prediction_frame(stage, prediction).to_csv(directory / "predictions.csv", index=False)
        write_json(directory / "results.json", record)
        trials.append(record)
        print(f"  Selected-stage RMSE={record['inner_scores']['rmse_sgd']:.2f}, iterations={iterations}", flush=True)
    selected = min(trials, key=lambda trial: trial["inner_scores"]["rmse_sgd"])
    cache = feature_cache_path(out, start, selected["profile"])
    if not cache.exists():
        prepare_fold(start, out)
    prepared = joblib.load(cache)
    stage = prepared["outer"]
    started = time.perf_counter()
    print(f"Refitting {start} {family}: {selected['profile']}, {selected['selected_iterations']} iterations/epochs", flush=True)
    artifacts = model_artifact_dir(destination)
    artifacts.mkdir(parents=True, exist_ok=True)
    if family in NEURAL:
        fitted = fit_neural(family, stage["x_fit"], stage["y_fit"], prepared["categorical"],
            selected["settings"], fixed_epochs=selected["selected_iterations"], device_name=device)
        prediction = predict_neural(fitted, stage["x_val"]) + stage["offset"]
        import torch
        if family == "tabnet":
            fitted["model"].save_model(str(artifacts / "model"))
            joblib.dump({"encoder": fitted["encoder"], "mean": fitted["mean"], "std": fitted["std"],
                "selected": selected}, artifacts / "preprocessor.joblib")
        else:
            torch.save({"state_dict": fitted["model"].state_dict(), "encoder": fitted["encoder"],
                "mean": fitted["mean"], "std": fitted["std"], "selected": selected}, artifacts / "checkpoint.pt")
        del fitted
        torch.cuda.empty_cache()
    else:
        fitted, prediction, _ = fit_tree(family, stage, prepared["categorical"],
            selected["settings"], selected["selected_iterations"])
        prediction += stage["offset"]
        joblib.dump(fitted, artifacts / "model.joblib")
        del fitted
    gc.collect()
    scores = metrics(stage["actual"], prediction)
    predictions = prediction_frame(stage, prediction)
    predictions.to_csv(destination / "predictions.csv", index=False)
    grouped_errors(predictions.rename(columns={"prediction": "predicted"}), "approval_month", family).to_csv(destination / "monthly_errors.csv", index=False)
    write_json(destination / "results.json", {"family": family, "validation_start": start,
        "validation_end_exclusive": OUTER_FOLDS[start], "selection": selected, "scores": scores,
        "fitting_rows": stage["n_fit"], "validation_rows": len(prediction),
        "source_sha256": prepared["source_sha256"], "feature_spec": prepared["spec"],
        "refit_seconds": time.perf_counter() - started,
        "note": "Epochs/iterations and profile selected only on the inner six-month time holdout; outer targets never used for early stopping."})
    print(f"OUTER {start} {family}: RMSE={scores['rmse_sgd']:.2f}, MAE={scores['mae_sgd']:.2f}", flush=True)
    return scores


def summarize(out=OUT):
    out = check_output_path(Path(out))
    rows = []
    for path in out.glob("20??-??/*/results.json"):
        result = json.loads(path.read_text(encoding="utf-8"))
        if "scores" not in result:
            continue
        rows.append({"validation_start": result["validation_start"], "family": result["family"],
            "validation_end_exclusive": result["validation_end_exclusive"],
            "horizon_months": pd.Period(result["validation_end_exclusive"], freq="M").ordinal
                - pd.Period(result["validation_start"], freq="M").ordinal,
            "profile": result["selection"]["profile"], "inner_rmse_sgd": result["selection"]["inner_scores"]["rmse_sgd"],
            "iterations": result["selection"]["selected_iterations"], "n_validation": result["validation_rows"], **result["scores"]})
    if not rows:
        return pd.DataFrame()
    scores = pd.DataFrame(rows).sort_values(["validation_start", "rmse_sgd"])
    scores.to_csv(out / "model_metrics.csv", index=False, float_format="%.6f")
    print(scores.to_string(index=False), flush=True)
    return scores


def blend(start="2024-04", out=OUT):
    """Select three members and convex weights on inner predictions only."""
    out = check_output_path(Path(out))
    jobs = []
    for path in (out / start).glob("*/results.json"):
        result = json.loads(path.read_text(encoding="utf-8"))
        if "scores" in result and result["family"] != "inner_blend":
            jobs.append((path.parent, result))
    if len(jobs) < 2:
        raise ValueError("At least two completed families are required")
    jobs.sort(key=lambda item: item[1]["selection"]["inner_scores"]["rmse_sgd"])
    jobs = jobs[:3]
    inner_frames, outer_frames = [], []
    for directory, result in jobs:
        trial = result["selection"]["trial_index"]
        inner_frames.append(pd.read_csv(directory / f"trial_{trial}" / "predictions.csv"))
        outer_frames.append(pd.read_csv(directory / "predictions.csv"))
    for frames in (inner_frames, outer_frames):
        for frame in frames[1:]:
            if not frame.source_row.equals(frames[0].source_row) or not np.allclose(frame.actual, frames[0].actual):
                raise ValueError("Ensemble prediction rows do not align")
    matrix = np.column_stack([frame.prediction for frame in inner_frames])
    actual = inner_frames[0].actual.to_numpy(float)
    if len(jobs) == 2:
        choices = [np.array([i / 10, 1 - i / 10]) for i in range(11)]
    else:
        choices = [np.array([i / 10, j / 10, (10 - i - j) / 10])
            for i in range(11) for j in range(11 - i)] + [np.ones(3) / 3]
    weights = min(choices, key=lambda weight: np.mean((matrix @ weight - actual) ** 2))
    inner_scores = metrics(actual, matrix @ weights)
    predicted = np.column_stack([frame.prediction for frame in outer_frames]) @ weights
    scores = metrics(outer_frames[0].actual.to_numpy(float), predicted)
    destination = out / start / "inner_blend"
    destination.mkdir(parents=True, exist_ok=True)
    frame = outer_frames[0].copy()
    frame["prediction"] = predicted
    frame.to_csv(destination / "predictions.csv", index=False)
    grouped_errors(frame.rename(columns={"prediction": "predicted"}), "approval_month", "inner_blend").to_csv(
        destination / "monthly_errors.csv", index=False)
    members = {result["family"]: float(weight) for (_, result), weight in zip(jobs, weights)}
    result = {"family": "inner_blend", "validation_start": start,
        "validation_end_exclusive": OUTER_FOLDS[start], "weights": members, "scores": scores,
        "selection": {"profile": "inner_selected_convex_blend", "inner_scores": inner_scores, "selected_iterations": 0},
        "fitting_rows": None, "validation_rows": len(frame),
        "note": "Members and convex weights selected on inner time holdout only; outer labels used solely for scoring."}
    write_json(destination / "results.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def report(out=OUT):
    out = check_output_path(Path(out))
    scores = summarize(out)
    if scores.empty:
        return {}
    latest = scores.loc[scores.validation_start.eq("2024-04")].sort_values("rmse_sgd")
    best = latest.iloc[0]
    predictions = pd.read_csv(out / "2024-04" / best.family / "predictions.csv")
    raw = load_main(DATA / "train.csv", require_target=True).loc[predictions.source_row].copy()
    raw["actual"] = predictions.actual.to_numpy(float)
    raw["predicted"] = predictions.prediction.to_numpy(float)
    raw["absolute_error"] = np.abs(raw.actual - raw.predicted)
    raw["rent_band"] = pd.cut(raw.actual, [0, 2000, 3000, 4000, np.inf], include_lowest=True).astype(str)
    for column in ("TOWN", "FLAT_TYPE", "rent_band"):
        grouped_errors(raw, column, str(best.family)).to_csv(out / f"best_errors_by_{column.lower()}.csv", index=False)
    raw.nlargest(30, "absolute_error")[[DATE, "TOWN", "BLOCK", "STREET", "FLAT_TYPE", "FLOOR_AREA_SQM",
        "actual", "predicted", "absolute_error"]].to_csv(out / "largest_errors.csv", index=False)
    enable_search_dependencies()
    versions = {}
    for package in ("numpy", "pandas", "scikit-learn", "lightgbm", "catboost", "torch",
                    "xgboost", "pytorch-tabnet", "rtdl-revisiting-models"):
        try:
            versions[package] = version(package)
        except Exception:
            versions[package] = "unavailable"
    completed = set(latest.family)
    unfinished_grids = [family for family, trials in TRIALS.items()
        if len(list((out / "2024-04" / family).glob("trial_*/results.json"))) < len(trials)]
    result = {"target_rmse_sgd": 200., "best_observed_outer_model": str(best.family),
        "best_observed_outer_rmse_sgd": float(best.rmse_sgd), "target_achieved": bool(best.rmse_sgd < 200),
        "missing_latest_families": sorted(set(TRIALS).difference(completed)),
        "unfinished_candidate_grids": unfinished_grids, "versions": versions,
        "validation_rule": "Inner preceding six months for profile/epoch/iteration selection; fixed-iteration outer refit.",
        "latest_validation_months": ["2024-04", "2025-03"], "latest_validation_rows": len(predictions),
        "source_sha256": {str(path.relative_to(DATA.parent)): sha256(path) for path in SOURCES},
        "error_concentration": {f"top_{percent}_percent_squared_error_share": float(
            raw.absolute_error.nlargest(max(1, int(np.ceil(len(raw) * percent / 100)))).pow(2).sum()
            / raw.absolute_error.pow(2).sum()) for percent in (1, 5, 10)},
        "implementation_sources": {"ft_transformer": "https://github.com/yandex-research/rtdl-revisiting-models",
            "tabnet": "https://github.com/dreamquark-ai/tabnet"},
        "code_sha256": {str(path.relative_to(DATA.parent)): sha256(path) for path in (
            Path(__file__), DATA.parent / "analysis/pipeline/neural_models.py", DATA.parent / "analysis/pipeline/features.py")},
        "limitations": ["Outer periods have been inspected in earlier project experiments; model-selection evidence, not independent test scores.",
            "Data ambiguity diagnostics do not establish a population-wide irreducible error bound.",
            "Model-specific preprocessing, training warmup and target decomposition are recorded per trial.",
            "Amenity tables are static snapshots without historical opening dates."]}
    write_json(out / "summary.json", result)
    table = ["| Model | Selected features | Epochs / trees | RMSE (SGD) | MAE (SGD) |",
             "| --- | --- | ---: | ---: | ---: |"]
    table += [f"| {row.family} | {row.profile} | {row.iterations} | {row.rmse_sgd:.2f} | {row.mae_sgd:.2f} |"
              for row in latest.itertuples(index=False)]
    status = "All current candidate grids completed." if not unfinished_grids else "Candidate grids still running: " + ", ".join(unfinished_grids)
    markdown = ["# Temporal model search", "", status, "",
        "Outer validation: April 2024 through March 2025, all 35,158 rows. Profile, hyperparameters and training length are chosen using only the preceding inner holdout, then refitted at fixed length.", "",
        *table, "", f"Best observed outer RMSE: **{best.rmse_sgd:.2f} SGD**. Requested target below 200 SGD achieved: **{result['target_achieved']}**.", "",
        "This period was inspected in earlier project experiments; the table is model-selection evidence, not an independent Kaggle test score. Model-specific history warmup and preprocessing are recorded per trial.", "",
        "See [method and reproduction guide](../../../docs/methods/model_comparison.md), model_metrics.csv for other annual windows, data_diagnostics.json for identical-input ambiguity, and summary.json for versions and source hashes.", ""]
    (out / "README.md").write_text("\n".join(markdown), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = {"inner_blend": "Inner-selected blend", "lightgbm": "LightGBM", "xgboost": "XGBoost",
        "catboost": "CatBoost", "hist_gradient_boosting": "HistGradientBoosting", "extra_trees": "ExtraTrees",
        "embedding_mlp": "Embedding MLP", "embedding_resnet": "Embedding ResNet", "tabnet": "TabNet",
        "ft_transformer": "FT-Transformer (96)", "ft_transformer_default": "FT-Transformer (192)"}
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.barh([labels.get(name, name) for name in latest.family], latest.rmse_sgd, color="#247d91")
    axis.invert_yaxis()
    axis.axvline(200, color="#b34136", linestyle="--", label="Requested target: 200 SGD")
    axis.set_xlabel("Outer temporal validation RMSE (SGD)")
    axis.set_title("2024-04 to 2025-03 | tuning uses earlier months")
    axis.set_xlim(0, float(latest.rmse_sgd.max()) + 55)
    for position, value in enumerate(latest.rmse_sgd):
        axis.text(value + 4, position, f"{value:.2f}", va="center", fontsize=9)
    axis.legend(loc="upper center", bbox_to_anchor=(.5, -.13), frameon=False)
    figure.tight_layout()
    figure.savefig(out / "model_comparison.png", dpi=150)
    plt.close(figure)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "summarize", "diagnose", "blend", "report"))
    parser.add_argument("--fold", choices=tuple(OUTER_FOLDS), default="2024-04")
    parser.add_argument("--families", nargs="+", choices=tuple(TRIALS), default=list(TRIALS))
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    if args.action == "prepare":
        return prepare_fold(args.fold, args.output_dir)
    if args.action == "diagnose":
        return diagnose_data(args.output_dir)
    if args.action == "blend":
        result = blend(args.fold, args.output_dir)
        summarize(args.output_dir)
        return result
    if args.action == "report":
        return report(args.output_dir)
    if args.action == "run":
        for family in args.families:
            run_family(family, args.fold, args.output_dir, args.max_epochs, args.device)
        return summarize(args.output_dir)
    return summarize(args.output_dir)


if __name__ == "__main__":
    main()
