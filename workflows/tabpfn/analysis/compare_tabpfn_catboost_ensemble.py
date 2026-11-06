"""Earlier-calibrated TabPFN/CatBoost ensembles on the full 80/20 holdout."""

from __future__ import annotations

from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catboost import CatBoostRegressor
import joblib
import numpy as np
import pandas as pd

from analysis.compare_tabpfn_calendar_80_20 import frozen_settings, weighted_metrics
from analysis.compare_tabpfn_v3_80_20 import markdown_table, write_json
from analysis.config import REPORTS, ROOT
from analysis.data import sha256
from analysis.evaluation import metrics
from analysis.experiments.dynamic_market_index import TREE_SETTINGS
from analysis.experiments.prediction_blending import blend, convex_weight


OUT = REPORTS / "tabpfn_catboost_ensemble_80_20"
ALIGNED = REPORTS / "tabpfn_v3_5_fast_aligned_temporal_80_20"
ORIGINAL = REPORTS / "tabpfn_v3_80_20"
CALENDAR = REPORTS / "tabpfn_v3_5_fast_calendar_80_20"
ORIGINS = ("2022-09", "2023-07")
CATBOOSTS = ("cb68", "cb77raw", "cb77residual")
TABPFNS = ("integer", "categorical")


def checked_bundle(directory, manifest_name="feature_manifest.json"):
    manifest = json.loads((directory / manifest_name).read_text(encoding="utf-8"))
    path = ROOT / manifest["artifact"]
    if sha256(path) != manifest["sha256"]:
        raise ValueError(f"Feature artifact changed: {path}")
    return joblib.load(path), manifest


def read_predictions(path, bundle):
    saved = pd.read_csv(path)
    np.testing.assert_array_equal(saved.row_id.to_numpy(), bundle["X_query"].index.to_numpy())
    np.testing.assert_array_equal(saved.actual.to_numpy(), bundle["y_query"])
    prediction = saved.predicted.to_numpy(float)
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite source predictions")
    return prediction


def fit_catboost(bundle, original, name, stage):
    path, timing_path = OUT / f"predictions_{stage}_{name}.csv", OUT / f"timing_{stage}_{name}.json"
    if path.exists() and timing_path.exists():
        return read_predictions(path, bundle)
    columns = list(original["X_train"]) if name == "cb68" else list(bundle["X_train"])
    x = (original["X_train"].loc[bundle["X_train"].index, columns].copy() if name == "cb68"
         else bundle["X_train"].copy())
    q = bundle["X_query"][columns].copy()
    categories = [c for c in x if isinstance(x[c].dtype, pd.CategoricalDtype)]
    for c in categories:
        x[c] = x[c].astype("string").fillna("__missing__").astype(str)
        q[c] = q[c].astype("string").fillna("__missing__").astype(str)
    y = np.asarray(bundle["y_train"], float)
    if name == "cb77residual":
        y = y - bundle["train_baseline"]
        np.testing.assert_allclose(y + bundle["train_baseline"], bundle["y_train"], atol=1e-10)
    if pd.PeriodIndex(bundle["train_month"], freq="M").max() >= pd.PeriodIndex(bundle["query_month"], freq="M").min():
        raise AssertionError("CatBoost context overlaps its query months")
    write_json(OUT / "status.json", {"status": "fitting_catboost", "stage": stage, "model": name, "context_rows": len(x)})
    print(f"Fitting {stage} {name}: {len(x):,} rows, {len(columns)} features", flush=True)
    started = time.perf_counter()
    model = CatBoostRegressor(**TREE_SETTINGS["catboost"])
    # No eval_set or early stopping: final query labels are never supplied.
    model.fit(x, y, cat_features=categories)
    fitted = time.perf_counter()
    prediction = np.asarray(model.predict(q), float)
    if name == "cb77residual":
        prediction = prediction + bundle["query_baseline"]
    elapsed = time.perf_counter() - fitted
    if not np.isfinite(prediction).all():
        raise AssertionError("CatBoost predictions are not finite")
    cache = ROOT / ".cache/tabpfn_ensemble"
    cache.mkdir(parents=True, exist_ok=True)
    model_path = cache / f"{stage}_{name}.cbm"
    model.save_model(str(model_path))
    pd.DataFrame({"row_id": q.index, "actual": bundle["y_query"], "predicted": prediction}).to_csv(
        path, index=False, float_format="%.8f")
    write_json(timing_path, {"fit_seconds": fitted - started, "predict_seconds": elapsed,
                             "context_rows": len(x), "features": len(columns), "target": "residual" if name == "cb77residual" else "raw rent",
                             "model_artifact": model_path.relative_to(ROOT).as_posix(), "sha256": sha256(model_path)})
    print(f"Completed {stage} {name} in {fitted - started:.1f}s", flush=True)
    # Calibrate on the exact persisted precision in fresh and resumed runs.
    return read_predictions(path, bundle)


def losses(actual, prediction, weights=None):
    result = metrics(actual, prediction) if weights is None else weighted_metrics(actual, prediction, weights)
    return {**result, "mse_sgd2": result["rmse_sgd"] ** 2}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    aligned, aligned_manifest = checked_bundle(ALIGNED)
    original, original_manifest = checked_bundle(ORIGINAL)
    for split in ("train", "query"):
        np.testing.assert_array_equal(aligned[f"X_{split}"].index.to_numpy(), original[f"X_{split}"].index.to_numpy())
        np.testing.assert_array_equal(aligned[f"y_{split}"], original[f"y_{split}"])
    # Query histories at the actual outer origin are identical across the old
    # 68-feature and aligned bundles. Context history policies intentionally differ.
    pd.testing.assert_frame_equal(aligned["X_query"][original["X_query"].columns], original["X_query"],
                                  check_exact=False, rtol=1e-10, atol=1e-8)
    old_settings = json.loads((ORIGINAL / "settings.json").read_text(encoding="utf-8"))
    if old_settings["catboost_settings"] != TREE_SETTINGS["catboost"] or old_settings["versions"]["catboost"] != version("catboost"):
        raise ValueError("Existing CatBoost control uses different settings or version")
    for name, expected in old_settings["source_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Underlying source changed: {name}")
    outer_sources = {"integer": ALIGNED / "predictions_tabpfn_aligned_residual.csv",
                     "categorical": CALENDAR / "predictions_outer_categorical.csv",
                     "cb68": ORIGINAL / "predictions_catboost_matched.csv"}
    inputs = dict(outer_sources)
    inner = {}
    for origin in ORIGINS:
        inner[origin], _ = checked_bundle(CALENDAR, f"inner_features_{origin}.json")
        for name in TABPFNS:
            inputs[f"inner_{origin}_{name}"] = CALENDAR / f"predictions_inner_{origin}_{name}.csv"
    code = [Path(__file__), ROOT / "analysis/experiments/prediction_blending.py", ROOT / "analysis/experiments/dynamic_market_index.py",
            ROOT / "analysis/compare_tabpfn_calendar_80_20.py", ROOT / "analysis/evaluation.py"]
    settings = {"catboost_settings": TREE_SETTINGS["catboost"], "catboost_variants": CATBOOSTS,
                "tabpfn_variants": TABPFNS, "origins": ORIGINS, "calibration_query_rows_per_month": 200,
                "calibration_weights": "month population / sampled count", "forecast_horizons": [1, 10],
                "outer_context_rows": len(aligned["X_train"]), "outer_validation_rows": len(aligned["X_query"]),
                "aligned_artifact_sha256": aligned_manifest["sha256"], "original_artifact_sha256": original_manifest["sha256"],
                "inner_artifacts": {o: json.loads((CALENDAR / f"inner_features_{o}.json").read_text())["sha256"] for o in ORIGINS},
                "input_predictions_sha256": {name: sha256(path) for name, path in inputs.items()},
                "code_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in code},
                "versions": {name: version(name) for name in ("catboost", "numpy", "pandas", "tabpfn")},
                "blend": "w * TabPFN raw-rent prediction + (1-w) * CatBoost raw-rent prediction; no intercept; 0<=w<=1",
                "weight_fitting": "earlier out-of-time queries only; exact weighted least squares",
                "selection_rule": "at least 0.25% earlier pooled RMSE gain over integer TabPFN; neither earlier fold >0.5% worse; simpler within 0.1% of best",
                "preference_order": "equal before fitted; integer before categorical; cb68 before cb77raw before cb77residual",
                "fallback_ensemble_for_reporting": "integer_cb68_equal", "outer_labels_used_for_weight_fitting": False}
    frozen_settings(OUT / "settings.json", json.loads(json.dumps(settings)))
    earlier = {}
    for origin, bundle in inner.items():
        predictions = {name: read_predictions(CALENDAR / f"predictions_inner_{origin}_{name}.csv", bundle) for name in TABPFNS}
        for name in CATBOOSTS:
            predictions[name] = fit_catboost(bundle, original, name, f"inner_{origin}")
        earlier[origin] = predictions
    actual = np.concatenate([b["y_query"] for b in inner.values()])
    weights = np.concatenate([b["query_weights"] for b in inner.values()])
    pooled = {name: np.concatenate([earlier[origin][name] for origin in ORIGINS]) for name in (*TABPFNS, *CATBOOSTS)}
    baseline = losses(actual, pooled["integer"], weights)["rmse_sgd"]
    configurations, score_rows, fold_rows = [], [], []
    for name, predicted in pooled.items():
        score_rows.append({"model": name, "kind": "standalone", **losses(actual, predicted, weights)})
    # Preference order is recorded before new outer evaluation.
    for mode in ("equal", "fitted"):
        for tfn in TABPFNS:
            for cb in CATBOOSTS:
                weight = .5 if mode == "equal" else convex_weight(actual, pooled[tfn], pooled[cb], weights)
                name = f"{tfn}_{cb}_{mode}"
                config = {"model": name, "tabpfn": tfn, "catboost": cb, "weight_tabpfn": weight, "weight_catboost": 1 - weight, "weight_mode": mode}
                configurations.append(config)
                score = losses(actual, blend(pooled[tfn], pooled[cb], weight), weights)
                consistent = True
                for origin, bundle in inner.items():
                    local = losses(bundle["y_query"], blend(earlier[origin][tfn], earlier[origin][cb], weight), bundle["query_weights"])
                    control = losses(bundle["y_query"], earlier[origin]["integer"], bundle["query_weights"])
                    consistent &= local["rmse_sgd"] <= control["rmse_sgd"] * 1.005
                    fold_rows.append({"model": name, "origin": origin, **local})
                score_rows.append({"model": name, "kind": "ensemble", "consistent": consistent,
                                   "eligible": bool(consistent and score["rmse_sgd"] < baseline * .9975), **score})
    scores = pd.DataFrame(score_rows)
    scores.to_csv(OUT / "inner_scores.csv", index=False, float_format="%.8f")
    pd.DataFrame(fold_rows).to_csv(OUT / "inner_fold_scores.csv", index=False, float_format="%.8f")
    pd.DataFrame(configurations).to_csv(OUT / "weights.csv", index=False, float_format="%.12f")
    eligible = scores.loc[scores.eligible.eq(True)]
    selected = None
    if len(eligible):
        threshold = eligible.rmse_sgd.min() * 1.001
        selected = next(c["model"] for c in configurations if c["model"] in set(eligible.loc[eligible.rmse_sgd <= threshold, "model"]))
    primary = selected or "integer_cb68_equal"
    # Frozen before new outer CatBoost inference and any ensemble outer scoring.
    frozen_settings(OUT / "weight_selection.json", {"selected": selected, "primary_ensemble": primary,
                    "weights": configurations, "outer_labels_used": False, "selection_origins": list(ORIGINS)})
    print(f"Earlier-selected ensemble: {selected}; reported primary: {primary}", flush=True)
    outer = {name: read_predictions(path, aligned) for name, path in outer_sources.items()}
    for name in CATBOOSTS[1:]:
        outer[name] = fit_catboost(aligned, original, name, "outer")
    result = pd.DataFrame({"row_id": aligned["X_query"].index, "month": aligned["query_month"], "actual": aligned["y_query"], **outer})
    for config in configurations:
        result[config["model"]] = blend(outer[config["tabpfn"]], outer[config["catboost"]], config["weight_tabpfn"])
    names = list(outer) + [c["model"] for c in configurations]
    if not result.row_id.is_unique or not np.isfinite(result[names].to_numpy()).all():
        raise AssertionError("Incomplete or nonfinite ensemble predictions")
    outer_scores = pd.DataFrame([{"model": name, "n": len(result), **losses(result.actual, result[name])} for name in names])
    monthly = pd.DataFrame([{"model": name, "month": month, "n": len(part), **losses(part.actual, part[name])}
                            for month, part in result.groupby("month", sort=True) for name in names])
    outer_scores.to_csv(OUT / "scores.csv", index=False, float_format="%.8f")
    monthly.to_csv(OUT / "monthly_scores.csv", index=False, float_format="%.8f")
    result.to_csv(OUT / "validation_predictions.csv", index=False, float_format="%.8f")
    diagnostics = []
    for tfn in TABPFNS:
        for cb in CATBOOSTS:
            diagnostics.append({"tabpfn": tfn, "catboost": cb,
                                "error_correlation": float(np.corrcoef(result[tfn] - result.actual, result[cb] - result.actual)[0, 1]),
                                "prediction_disagreement_rmse": float(np.sqrt(np.mean((result[tfn] - result[cb]) ** 2)))})
    pd.DataFrame(diagnostics).to_csv(OUT / "error_diagnostics.csv", index=False, float_format="%.8f")
    table = outer_scores.set_index("model")
    confirmed = selected is not None and table.loc[selected, "rmse_sgd"] < table.loc["integer", "rmse_sgd"]
    write_json(OUT / "selection.json", {"earlier_selected": selected, "primary_reported_ensemble": primary,
                                        "outer_confirmation_passed": bool(confirmed), "recommended": selected if confirmed else "integer"})
    primary_config = next(c for c in configurations if c["model"] == primary)
    focused = list(dict.fromkeys([primary_config["tabpfn"], primary_config["catboost"], primary,
                                 f"{primary_config['tabpfn']}_{primary_config['catboost']}_equal"]))
    report = ["# TabPFN and CatBoost validation ensembles", "",
              "The chronological 80/20 split retains 121,005 context rows (January 2021-May 2024) and "
              "28,995 validation rows (June 2024-March 2025). TabPFN predictions are hash-checked and reused "
              "from the best retained aligned residual setup and its effectively tied categorical-month variant.", "",
              "CatBoost uses 600 iterations, depth 7, learning rate 0.05, L2 regularization 5, seed 42, and four "
              "CPU threads. cb68 is the existing raw-rent control with 68 features and preceding-month context "
              "histories. cb77raw uses the 77 aligned inputs to predict raw rent. cb77residual uses those same "
              "inputs to predict the same residual target as TabPFN and adds the baseline back exactly once. "
              "No query labels enter fitting, an eval_set, or early stopping.", "",
              "Blend weights use earlier ten-month forecasts (September 2022-June 2023 and July 2023-April "
              "2024). Each uses every earlier context row and the same fixed 200-query-per-month samples as "
              "the calendar experiment, weighted by month population/sample count. Six model pairs each have "
              "a fixed 50/50 blend and a constrained weighted-least-squares blend. The latter has no intercept "
              "and a TabPFN weight in [0,1]. Outer labels never fit a weight or select a pair.", "",
              f"Earlier-selected ensemble: `{selected}`. Primary reported ensemble: `{primary}`. "
              f"Its prediction is **{primary_config['weight_tabpfn']:.6f} x TabPFN + "
              f"{primary_config['weight_catboost']:.6f} x CatBoost**.", "",
              "## Primary validation loss", "", markdown_table(outer_scores.set_index("model").loc[focused].reset_index()), "",
              f"Outer confirmation passed: `{bool(confirmed)}`. RMSE is in SGD; MSE is in SGD squared. "
              "The primary ensemble and weights were frozen before new outer scoring. If it fails confirmation, "
              "retain the original TabPFN rather than select another pair from outer scores.", "",
              "## All validation comparisons", "", markdown_table(outer_scores), "",
              "## Earlier calibration scores", "", markdown_table(scores), "",
              "## Error diversity", "", markdown_table(pd.DataFrame(diagnostics)), "",
              "Earlier query evaluation is sampled; final validation and all context sets are complete. "
              "The outer period has been examined before and is a development holdout, not an untouched "
              "test. Its ten-month horizon does not establish seventeen-month test performance.", "",
              "Weights are in weights.csv and weight_selection.json; row predictions, monthly losses, "
              "CatBoost timings and model artifacts are also saved. Binary models remain under ignored .cache/.", "",
              "Reproduce/resume: `.venv/Scripts/python.exe analysis/compare_tabpfn_catboost_ensemble.py`."]
    (OUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(OUT / "status.json", {"status": "complete", "primary_ensemble": primary})
    print(outer_scores.set_index("model").loc[focused].to_string(), flush=True)


if __name__ == "__main__":
    main()
