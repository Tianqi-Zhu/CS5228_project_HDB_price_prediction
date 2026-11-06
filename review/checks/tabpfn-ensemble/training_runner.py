"""Fresh fixed-recipe tree/TabPFN refits and Kaggle submission exports on main.

Run stages in separate processes because the archived TabPFN analysis package
and main's tree analysis package have different module layouts.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
WEIGHTS = {
    "normalized_market/compact_ratio15_market_last": .13094,
    "anniversary/broad_lgb15": .09659,
    "anniversary/broad_lgb63": .10352,
    "anniversary/exact_lgb15": .17311,
    "anniversary/exact_lgb63": .00658,
    "anniversary/window_cat6": .41380,
    "anniversary/window_lgb63": .07546,
}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def grouping():
    path = ROOT / "workflows/tabpfn/analysis/flat_model_groups.py"
    spec = importlib.util.spec_from_file_location("ensemble_flat_model_groups", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def identity():
    return {"runner_sha256": sha(__file__), "grouping_sha256": sha(ROOT / "workflows/tabpfn/analysis/flat_model_groups.py"),
            "recipe_sha256": sha(ROOT / "reproduction/f1/training_recipes.json"),
            "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted((ROOT / "data").rglob("*.csv"))
                              if "processed" not in p.relative_to(ROOT / "data").parts},
            "packages": {name: version(name) for name in ("numpy", "pandas", "lightgbm", "catboost", "tabpfn", "torch", "scikit-learn")}}


def verified_existing(path, record):
    if not path.exists():
        return False
    if not record.exists():
        raise ValueError(f"Output has no run provenance: {path}")
    saved = json.loads(record.read_text(encoding="utf-8"))
    if saved["identity"] != identity() or saved["prediction_sha256"] != sha(path):
        raise ValueError(f"Output/environment changed; choose a new --output: {path}")
    print(f"Reusing this run's verified output: {path.name}", flush=True)
    return True


def trees(args):
    import joblib
    import numpy as np
    import pandas as pd
    from analysis.experiments.frontier_search import frame_data
    from analysis.experiments.second_wave_anniversary import data
    from analysis.experiments.second_wave_cohort_extensions import fit as fit_anniversary
    from analysis.experiments.second_wave_normalized import fit as fit_normalized
    from analysis.experiments.best_ensemble_submission import prepare, predict_saved

    stage = args.stage.split("-")[1]
    if stage == "validate":
        base = data("2024-06", "2025-04")
        source, hashes = frame_data()
        raw_fit = source.loc[source.RENT_APPROVAL_DATE.lt("2024-06")]
        raw_query = source.loc[source.RENT_APPROVAL_DATE.ge("2024-06")]
        actual = np.asarray(base["y_val"]).copy()
        base = {**base, "raw_fit": raw_fit, "raw_val": raw_query,
                "reference": {method: raw_fit.groupby(["RENT_APPROVAL_DATE", "FLAT_TYPE_CLEAN"]).MONTHLY_RENT.transform(method).to_numpy(float)
                              for method in ("mean", "median")}, "y_val": np.zeros(len(actual))}
    else:
        base, hashes = prepare()
        raw_fit, raw_query = base["raw_fit"], base["raw_val"]
        actual = None
    expected = (121005, 28995) if stage == "validate" else (150000, 50000)
    assert (len(base["y_fit"]), len(base["y_val"])) == expected
    raw_train = pd.read_csv(ROOT / "data/train.csv")
    raw_test = pd.read_csv(ROOT / "data/test.csv") if stage == "predict" else raw_train
    recipes = json.loads((ROOT / "reproduction/f1/training_recipes.json").read_text(encoding="utf-8"))["history_members"]
    recipes = {item["name"].removeprefix("second_wave/"): item for item in recipes if item["name"].startswith("second_wave/")}
    for variant in args.variants:
        output = args.output / f"trees_{stage}_{variant}.csv"
        record = output.with_suffix(".json")
        if verified_existing(output, record):
            continue
        bundle = dict(base)
        if variant == "grouped":
            bundle["x_fit"], bundle["x_val"] = grouping().apply_model_group(
                base["x_fit"], base["x_val"], "grouped",
                raw_train.loc[raw_fit.index, "FLAT_MODEL"], raw_test.loc[raw_query.index, "FLAT_MODEL"])
            for col in base["x_fit"]:
                if col != "FLAT_MODEL_CLEAN":
                    pd.testing.assert_series_equal(bundle["x_fit"][col], base["x_fit"][col])
                    pd.testing.assert_series_equal(bundle["x_val"][col], base["x_val"][col])
        combined = np.zeros(len(raw_query))
        members = {}
        for name, weight in WEIGHTS.items():
            recipe = recipes[name]
            rounds, spec = recipe["iterations"], recipe["spec"]
            directory = args.output / "models" / stage / variant / name
            directory.mkdir(parents=True, exist_ok=True)
            prediction_path = directory / "prediction.csv"
            member_record = directory / "provenance.json"
            started = time.perf_counter()
            if verified_existing(prediction_path, member_record):
                frame = pd.read_csv(prediction_path)
                np.testing.assert_array_equal(frame.row_id, raw_query.index)
                predicted = frame.predicted.to_numpy(float)
                replay_difference = json.loads(member_record.read_text())["reload_max_difference"]
            else:
                print(f"TREES {stage}/{variant}: {name}, {rounds} fixed rounds", flush=True)
                if recipe["kind"] == "normalized_market":
                    model, predicted, count = fit_normalized(spec, bundle, rounds)
                else:
                    model, predicted, count, _ = fit_anniversary(spec, bundle, rounds)
                assert count == rounds and np.isfinite(predicted).all()
                checkpoint = directory / "model.joblib"
                joblib.dump({"model": model, "spec": spec}, checkpoint, compress=1)
                restored = joblib.load(checkpoint)
                replay = predict_saved(restored, bundle, recipe["kind"])
                np.testing.assert_allclose(replay, predicted, atol=1e-8, rtol=0)
                replay_difference = float(np.max(np.abs(replay - predicted)))
                pd.DataFrame({"row_id": raw_query.index, "predicted": predicted}).to_csv(prediction_path, index=False, float_format="%.17g")
                write_json(member_record, {"identity": identity(), "prediction_sha256": sha(prediction_path),
                           "model_sha256": sha(checkpoint), "reload_max_difference": replay_difference,
                           "iterations": count, "spec": spec, "seconds": time.perf_counter() - started})
                del model, restored
                gc.collect()
            combined += weight * np.asarray(predicted, float)
            members[name] = {"weight": weight, "iterations": rounds, "prediction_sha256": sha(prediction_path),
                             "reload_max_difference": replay_difference}
        frame = pd.DataFrame({"row_id": raw_query.index, "predicted": combined})
        if actual is not None:
            frame["actual"] = actual
            frame["month"] = base["date_val"]
        frame.to_csv(output, index=False, float_format="%.17g")
        write_json(record, {"identity": identity(), "prediction_sha256": sha(output), "members": members,
                           "fit_rows": len(raw_fit), "query_rows": len(raw_query), "query_labels_blinded": True,
                           "raw_source_hashes": hashes, "category_variant": variant})
        print(f"TREES complete {stage}/{variant}", flush=True)


def tabpfn(args):
    # Insert before any analysis import, so main's tree modules cannot shadow it.
    sys.path.insert(0, str(args.runtime))
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    os.environ.pop("TABPFN_MAX_BATCHED_ESTIMATOR_ROWS", None)
    import numpy as np
    import pandas as pd
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    from analysis.compare_tabpfn_anniversary_history_80_20 import source_bundle, matrices, CHECKPOINT
    stage = args.stage.split("-")[1]
    if stage == "validate":
        base = source_bundle()
        x, q = matrices("anniversary87", base)
        y, train_baseline, query_baseline = (np.asarray(base[k], float) for k in ("y_train", "train_baseline", "query_baseline"))
    else:
        from analysis.compare_zch_anniversary_tabpfn_ensemble import full_features
        base = full_features()
        x, q, y, train_baseline, query_baseline = (base[k] for k in ("x", "q", "y", "train_baseline", "query_baseline"))
    original_x, original_q = x, q
    raw_train = pd.read_csv(ROOT / "data/train.csv")
    raw_query = pd.read_csv(ROOT / "data/test.csv") if stage == "predict" else raw_train
    for variant in args.variants:
        output = args.output / f"tabpfn_{stage}_{variant}.csv"
        record = output.with_suffix(".json")
        if verified_existing(output, record):
            continue
        x, q = original_x, original_q
        if variant == "grouped":
            x, q = grouping().apply_model_group(x, q, "grouped", raw_train.loc[x.index, "FLAT_MODEL"], raw_query.loc[q.index, "FLAT_MODEL"])
        torch.set_num_threads(1)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        if sha(CHECKPOINT) != "14e0b81285bca7a6d1ef121169f27f651f76dff49cad6c0b8779d5c836018550":
            raise ValueError("Unexpected TabPFN checkpoint")
        model = TabPFNRegressor.create_default_for_version(ModelVersion.V3_5_FAST, model_path=str(CHECKPOINT),
            device="cuda", n_estimators=2, random_state=42, fit_mode="fit_with_cache", keep_cache_on_device=True,
            categorical_features_indices=[i for i,c in enumerate(x) if isinstance(x[c].dtype, pd.CategoricalDtype)],
            inference_config={"SUBSAMPLE_SAMPLES": None}, n_preprocessing_jobs=1)
        started = time.perf_counter()
        try:
            model.fit(x, np.asarray(y) - np.asarray(train_baseline))
            assert model.n_train_samples_ == len(x) and model.n_estimators_ == 2 and model.inference_config_.SUBSAMPLE_SAMPLES is None
            fitted = time.perf_counter()
            values = np.full(len(q), np.nan)
            print(f"TABPFN fitted {stage}/{variant}: {len(x):,} rows, {len(x.columns)} features", flush=True)
            for start in range(0, len(q), 200):
                stop = min(start + 200, len(q))
                torch.cuda.empty_cache()
                values[start:stop] = np.asarray(model.predict(q.iloc[start:stop], output_type="mean"), float) + np.asarray(query_baseline)[start:stop]
                if start % 4000 == 0 or stop == len(q):
                    print(f"TABPFN {stage}/{variant}: {stop:,}/{len(q):,}", flush=True)
            assert np.isfinite(values).all()
            frame = pd.DataFrame({"row_id": q.index, "predicted": values})
            if stage == "validate":
                frame["actual"] = base["y_query"]
                frame["month"] = base["query_month"]
            frame.to_csv(output, index=False, float_format="%.17g")
            write_json(record, {"identity": identity(), "prediction_sha256": sha(output), "checkpoint_sha256": sha(CHECKPOINT),
                       "features": len(x.columns), "context_rows": len(x), "query_rows": len(q), "category_variant": variant,
                       "fit_seconds": fitted - started, "predict_seconds": time.perf_counter() - fitted,
                       "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3, "subsampling": False})
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()


def export(args):
    import numpy as np
    import pandas as pd
    from analysis.experiments.canonical_final import validate_submission
    from analysis.experiments.frontier_search import score
    records = []
    sample = pd.read_csv(ROOT / "data/example-submission.csv")
    for variant in args.variants:
        for stage in ("validate", "predict"):
            parts = [pd.read_csv(args.output / f"{kind}_{stage}_{variant}.csv") for kind in ("trees", "tabpfn")]
            a,b = parts
            np.testing.assert_array_equal(a.row_id, b.row_id)
            values = .5 * a.predicted.to_numpy(float) + .5 * b.predicted.to_numpy(float)
            if stage == "validate":
                np.testing.assert_array_equal(a.actual, b.actual)
                np.testing.assert_array_equal(a.month, b.month)
                for name, prediction in (("trees", a.predicted), ("tabpfn", b.predicted), ("ensemble", values)):
                    records.append({"variant": variant, "model": name, "rows": len(values), **score(a.actual, prediction)})
                pd.DataFrame({"row_id": a.row_id, "actual": a.actual, "month": a.month, "predicted": values}).to_csv(
                    args.output / f"validation_ensemble_{variant}.csv", index=False, float_format="%.17g")
            else:
                np.testing.assert_array_equal(a.row_id, sample.Id)
                submission = validate_submission(sample, values)
                assert (submission.Predicted > 0).all() and len(submission) == 50000
                path = args.output / f"submission_ensemble_{variant}_150k.csv"
                submission.to_csv(path, index=False, float_format="%.17g")
                replay = pd.read_csv(path)
                np.testing.assert_allclose(replay.Predicted, values, atol=1e-8, rtol=0)
                write_json(path.with_suffix(".json"), {"identity": identity(), "sha256": sha(path), "rows": len(values),
                           "training_rows": 150000, "tree_weight": .5, "tabpfn_weight": .5,
                           "category_variant": variant, "actual_kaggle_score": None, "uploaded": False})
    scores = pd.DataFrame(records)
    scores.to_csv(args.output / "validation_scores.csv", index=False, float_format="%.10f")
    print(scores.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("trees-validate", "trees-predict", "tabpfn-validate", "tabpfn-predict", "export", "all"))
    parser.add_argument("--output", type=Path, default=ROOT / "output/tabpfn-ensemble")
    parser.add_argument("--runtime", type=Path, default=ROOT / "output/tabpfn-best")
    parser.add_argument("--variants", nargs="+", choices=("original", "grouped"), default=("original", "grouped"))
    args = parser.parse_args()
    args.output, args.runtime = args.output.resolve(), args.runtime.resolve()
    if not args.output.is_relative_to(ROOT / "output") or args.output == ROOT / "output":
        parser.error("Output must be a child of output/")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == "all":
        for stage in ("trees-validate", "trees-predict", "tabpfn-validate", "tabpfn-predict", "export"):
            subprocess.run([sys.executable, __file__, stage, "--output", str(args.output), "--runtime", str(args.runtime),
                            "--variants", *args.variants], check=True)
    elif args.stage.startswith("trees-"):
        trees(args)
    elif args.stage.startswith("tabpfn-"):
        tabpfn(args)
    else:
        export(args)


if __name__ == "__main__":
    main()
