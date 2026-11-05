"""Compare local TabPFN with CatBoost without future rental/economic inputs.

Each fold is a fixed-origin forecast. History construction uses the complete
pre-origin fitting frame, even when TabPFN subsamples its attention context.
Blending uses only earlier folds, with zero TabPFN weight on the first fold.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from analysis.config import DATA, REPORTS, ROOT
from analysis.data import sha256
from analysis.evaluation import metrics
from analysis.experiments.dynamic_market_index import fit_tree, TREE_SETTINGS
from analysis.experiments.spatiotemporal_hybrid import (
    BASE_CATEGORICAL, BASE_FEATURES, FOLDS, HISTORY_FEATURES, MONTH,
    SOURCES, TARGET, historical_features, model_inputs, spatial_frame,
)


OUT = REPORTS / "tabpfn"
VARIANTS = {"tabpfn_spatial": BASE_FEATURES,
            "tabpfn_history": BASE_FEATURES + HISTORY_FEATURES}
BLEND = "catboost_tabpfn_forward_blend"


def fixed_origin_split(frame: pd.DataFrame, start: str, end: str):
    origin, stop = pd.Period(start, "M"), pd.Period(end, "M")
    fitting = frame.loc[frame[MONTH] < origin].copy()
    future = frame.loc[(frame[MONTH] >= origin) & (frame[MONTH] < stop)].copy()
    if fitting.empty or future.empty or fitting[MONTH].max() >= future[MONTH].min():
        raise ValueError("A fold needs nonempty, strictly separated monthly partitions")
    return fitting, future


def feature_frames(fitting: pd.DataFrame, future: pd.DataFrame):
    # Strip future labels before calling any target-derived feature function.
    past, forecast = historical_features(fitting, future.drop(columns=TARGET, errors="ignore"))
    return fitting.join(past), future.join(forecast)


def tabpfn_inputs(fitting: pd.DataFrame, future: pd.DataFrame, columns: list[str]):
    """Keep raw numeric values and explicit, train-defined categorical types."""
    if TARGET in columns or any(c in columns for c in ("LOG_MONTHLY_RENT", "RENT_PER_SQM")):
        raise ValueError("Current-row target transformations are forbidden inputs")
    x_fit, x_future = fitting[columns].copy(), future[columns].copy()
    for column in set(columns) & BASE_CATEGORICAL:
        known = pd.Index(x_fit[column].dropna().astype(str).unique())
        for frame in (x_fit, x_future):
            values = frame[column].astype('string')
            frame[column] = pd.Categorical(values.where(values.isin(known)), categories=known)
    return x_fit, x_future


def blend_weight(previous: pd.DataFrame) -> float:
    """Constrained least-squares TabPFN weight, trained on past folds only."""
    if previous.empty:
        return 0.0
    delta = previous["tabpfn_history"].to_numpy(float) - previous["catboost"].to_numpy(float)
    residual = previous[TARGET].to_numpy(float) - previous["catboost"].to_numpy(float)
    denominator = float(delta @ delta)
    return float(np.clip((delta @ residual) / denominator, 0, 1)) if denominator > 0 else 0.0


def add_forward_blend(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = predictions.copy()
    weights = []
    for start in sorted(result["validation_start"].unique()):
        earlier = result[result["validation_start"] < start]
        weight = blend_weight(earlier)
        mask = result["validation_start"].eq(start)
        result.loc[mask, BLEND] = (
            (1 - weight) * result.loc[mask, "catboost"]
            + weight * result.loc[mask, "tabpfn_history"]
        )
        weights.append({"validation_start": start, "tabpfn_weight": weight,
                        "calibration_rows": len(earlier)})
    return result, pd.DataFrame(weights)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def local_model(args):
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    checkpoint = {"v3.5": "V3_5", "v3": "V3", "v2.6": "V2_6", "v2": "V2"}[args.model_version]
    if not hasattr(ModelVersion, checkpoint):
        raise RuntimeError(f"Installed tabpfn {version('tabpfn')} has no {args.model_version} checkpoint")
    config = {"SUBSAMPLE_SAMPLES": args.context_rows} if args.context_rows else {}
    return TabPFNRegressor.create_default_for_version(
        getattr(ModelVersion, checkpoint), device="cuda", n_estimators=args.n_estimators,
        random_state=args.seed, fit_mode=args.fit_mode, inference_config=config,
    )


def fit_tabpfn(args, x_fit, y_fit, x_future):
    import torch

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = local_model(args)
    started = time.perf_counter()
    try:
        model.fit(x_fit, np.asarray(y_fit, float))
        torch.cuda.synchronize()
        fitted = time.perf_counter()
        batches = []
        for offset in range(0, len(x_future), args.batch_size):
            batch = x_future.iloc[offset:offset + args.batch_size]
            batches.append(np.asarray(model.predict(batch, output_type="mean"), float))
            print(f"    predicted {min(offset + args.batch_size, len(x_future))}/{len(x_future)}", flush=True)
        torch.cuda.synchronize()
        predicted = np.concatenate(batches)
        if len(predicted) != len(x_future) or not np.isfinite(predicted).all():
            raise ValueError("TabPFN returned incomplete or nonfinite predictions")
        timing = {"fit_seconds": fitted - started,
                  "predict_seconds": time.perf_counter() - fitted,
                  "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024 ** 3}
        # Include the actual checkpoint paths, rather than relying on a moving default.
        timing["model_path"] = str(getattr(model, "model_path", "unknown"))
        checkpoint = Path(timing['model_path'])
        if checkpoint.is_file():
            timing['checkpoint_sha256'] = sha256(checkpoint)
        return predicted, timing
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def run_settings(args) -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled PyTorch installation and GPU are required")
    return {
        "model_version": args.model_version, "context_rows": args.context_rows,
        "n_estimators": args.n_estimators, "seed": args.seed,
        "batch_size": args.batch_size, "fit_mode": args.fit_mode,
        "folds": FOLDS, "features": VARIANTS, "catboost_settings": TREE_SETTINGS["catboost"],
        "selection_rule": "At least 1% pooled gain and lower RMSE on every fold; forward blend may tie its uncalibrated first fold but must improve both later folds",
        "sources_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in SOURCES},
        "implementation_sha256": sha256(Path(__file__)),
        "shared_code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in (
            ROOT / 'analysis/data.py', ROOT / 'analysis/experiments/spatiotemporal_hybrid.py',
            ROOT / 'analysis/experiments/dynamic_market_index.py')},
        "versions": {name: version(name) for name in ("tabpfn", "torch", "numpy", "pandas", "scikit-learn", "catboost")},
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
    }


def summarize(predictions: pd.DataFrame, out: Path):
    predictions, weights = add_forward_blend(predictions)
    models = ["catboost", *VARIANTS, BLEND]
    folds, months = [], []
    for start, part in predictions.groupby("validation_start", sort=True):
        for model in models:
            folds.append({"validation_start": start, "model": model, "n": len(part),
                          **metrics(part[TARGET], part[model])})
            for month, monthly in part.groupby(MONTH, sort=True):
                months.append({"validation_start": start, "month": str(month), "model": model,
                               "n": len(monthly), **metrics(monthly[TARGET], monthly[model])})
    pooled = pd.DataFrame([{"model": model, "n": len(predictions),
                            **metrics(predictions[TARGET], predictions[model])} for model in models])
    baseline = float(pooled.loc[pooled.model.eq("catboost"), "rmse_sgd"].iloc[0])
    pooled["delta_rmse_vs_catboost_sgd"] = pooled.rmse_sgd - baseline
    pooled = pooled.sort_values("rmse_sgd")
    for name, table in {"pooled_scores": pooled, "fold_scores": pd.DataFrame(folds),
                        "monthly_scores": pd.DataFrame(months), "blend_weights": weights,
                        "validation_predictions": predictions}.items():
        table.to_csv(out / f"{name}.csv", index=False, float_format="%.8f")
    return predictions, pooled, pd.DataFrame(folds), weights


def select_candidate(pooled: pd.DataFrame, folds: pd.DataFrame) -> str:
    """Apply the prespecified practical-gain and temporal-consistency rules."""
    baseline = folds[folds.model.eq('catboost')].set_index('validation_start').rmse_sgd
    cat_score = float(pooled.loc[pooled.model.eq('catboost'), 'rmse_sgd'].iloc[0])
    qualified = []
    for model in [*VARIANTS, BLEND]:
        scores = folds[folds.model.eq(model)].set_index('validation_start').rmse_sgd.reindex(baseline.index)
        consistent = bool((scores < baseline).all())
        if model == BLEND:
            first = min(baseline.index)
            later = baseline.index != first
            consistent = (len(baseline[later]) >= 2 and scores.loc[first] <= baseline.loc[first]
                          and bool((scores[later] < baseline[later]).all()))
        score = float(pooled.loc[pooled.model.eq(model), 'rmse_sgd'].iloc[0])
        if consistent and score <= .99 * cat_score:
            qualified.append(model)
    ordered = pooled.sort_values('rmse_sgd').model
    return next((model for model in ordered if model in qualified), 'catboost')


def write_report(out, settings, pooled, folds, weights, timings):
    def table(frame):
        # Avoid an extra optional tabulate dependency.
        rows = [[str(c) for c in frame.columns], ["---"] * len(frame.columns)]
        rows += [[f"{v:.3f}" if isinstance(v, float) else str(v) for v in row]
                 for row in frame.itertuples(index=False, name=None)]
        return "\n".join("| " + " | ".join(row) + " |" for row in rows)
    selected = select_candidate(pooled, folds)
    report = f"""# TabPFN fixed-origin rental comparison

Selected candidate: **{selected}**. Promotion requires at least 1% lower pooled RMSE and improvement on every disjoint fold. The forward blend may tie its uncalibrated first fold, but must improve both later folds. This exception was set before TabPFN scoring.

## Configuration

Local {settings['model_version']} regressor, {settings['n_estimators']} estimators, {settings['context_rows'] or 'full'} rows per estimator, seed {settings['seed']}, `{settings['fit_mode']}`. GPU: {settings['gpu']}. Target: raw monthly rent in SGD; predictive mean. Each variant has the same complete fitting population; context subsampling is internal to TabPFN. No COE or stock inputs are used.

Spatial features reuse the existing property, HDB-block, planning and amenity inputs. The history variant adds strictly earlier signature, block × flat-type and area × flat-type means/counts/last means/recency. Training histories exclude the entire current month. Validation histories freeze at the origin, with recency advancing by forecast month. Constant furnished/fee fields and current-row target transformations are excluded.

## Pooled transaction scores

{table(pooled)}

## Fold RMSE (SGD)

{table(folds.pivot(index='model', columns='validation_start', values='rmse_sgd').reset_index())}

## Forward-only blend

The TabPFN-history weight minimizes squared error on all earlier folds, constrained to [0, 1]. The first fold defaults to CatBoost (zero calibration rows). Later validation labels never choose their own weight. Consequently the pooled blend includes a CatBoost-only first fold; it is not a globally tuned blend score.

{table(weights)}

## Timing and limits

{table(pd.DataFrame(timings).drop(columns=['model_path', 'checkpoint_sha256'], errors='ignore'))}

The three validation windows are disjoint. Their months have been examined previously in this project, so these are development results rather than an untouched final test. Supplied amenity tables are current snapshots without historical opening dates. Model/package versions, source hashes, settings and checkpoint paths are recorded in `results.json`. All predictions retain original source row IDs.

Run: `.venv/Scripts/python.exe analysis/compare_tabpfn.py --output {out.relative_to(ROOT).as_posix()} --model-version {settings['model_version']} --context-rows {settings['context_rows']} --n-estimators {settings['n_estimators']} --batch-size {settings['batch_size']} --fit-mode {settings['fit_mode']}`. Use `--resume` to reuse matching completed folds. Model downloads may require authentication; data stays local.
"""
    (out / "report.md").write_text(report, encoding="utf-8")
    return selected


def evaluate(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    settings = run_settings(args)
    fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()
    settings_path = out / "settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text(encoding="utf-8")) != json.loads(json.dumps(settings, default=str)):
        raise ValueError("Output settings differ. Use a separate --output directory.")
    write_json(settings_path, settings)
    write_json(out / 'execution_status.json', {
        'status': 'running', 'completed_folds': [], 'settings': settings,
    })
    frame = spatial_frame(pd.read_csv(DATA / "train.csv"))
    completed, timings = [], []
    for start, end in FOLDS:
        path = out / f"predictions_{start}.csv"
        manifest = out / f"completed_{start}.json"
        if args.resume and path.exists() and manifest.exists():
            saved = json.loads(manifest.read_text(encoding="utf-8"))
            if saved['fingerprint'] != fingerprint or saved['prediction_sha256'] != sha256(path):
                raise ValueError("Saved fold does not match this experiment")
            completed.append(pd.read_csv(path))
            timings.extend(saved['timings'])
            print(f"Resumed completed fold {start}", flush=True)
            continue
        fitting, future = fixed_origin_split(frame, start, end)
        print(f"Fold {start}: {len(fitting)} fitting, {len(future)} validation", flush=True)
        fit, val = feature_frames(fitting, future)
        result = future[[MONTH, TARGET, "FLAT_TYPE_CLEAN"]].copy()
        result[MONTH] = result[MONTH].astype(str)
        result["row_id"] = future.index
        result["validation_start"] = start
        fold_timing = []
        x_fit, x_val = model_inputs(fit, val, VARIANTS['tabpfn_history'])
        result['catboost'], seconds = fit_tree('catboost', fitting, future, x_fit, x_val)
        fold_timing.append({"validation_start": start, "model": "catboost", "fit_predict_seconds": seconds})
        print(f"  CatBoost RMSE {metrics(result[TARGET], result.catboost)['rmse_sgd']:.2f}", flush=True)
        # Save the baseline independently, including when model downloads are blocked.
        result.to_csv(out / f"baseline_{start}.csv", index=False, float_format="%.8f")
        for name, columns in VARIANTS.items():
            x_fit, x_val = tabpfn_inputs(fit, val, columns)
            result[name], timing = fit_tabpfn(args, x_fit, fitting[TARGET], x_val)
            fold_timing.append({"validation_start": start, "model": name, **timing})
            print(f"  {name} RMSE {metrics(result[TARGET], result[name])['rmse_sgd']:.2f}", flush=True)
        result.to_csv(path, index=False, float_format="%.8f")
        write_json(manifest, {"fingerprint": fingerprint, "prediction_sha256": sha256(path), "timings": fold_timing})
        completed.append(result)
        timings.extend(fold_timing)
        # Persist interim metrics after each complete fold.
        predictions, pooled, folds, weights = summarize(pd.concat(completed, ignore_index=True), out)
        write_json(out / "progress.json", {"completed_folds": [s for s, _ in FOLDS][:len(completed)], "timings": timings})
        write_json(out / 'execution_status.json', {
            'status': 'running', 'completed_folds': [s for s, _ in FOLDS][:len(completed)],
            'settings': settings,
        })
    predictions, pooled, folds, weights = summarize(pd.concat(completed, ignore_index=True), out)
    selected = write_report(out, settings, pooled, folds, weights, timings)
    pd.DataFrame(timings).to_csv(out / "timings.csv", index=False)
    write_json(out / "results.json", {"settings": settings, "timings": timings,
                                    "selected_candidate": selected, "pooled_scores": pooled.to_dict('records'),
                                    "final_blend_weight": blend_weight(predictions)})
    (out / 'failure.json').unlink(missing_ok=True)
    write_json(out / 'execution_status.json', {
        'status': 'complete', 'completed_folds': [s for s, _ in FOLDS],
        'selected_candidate': selected, 'n_validation_rows': len(predictions),
        'settings': settings,
    })
    print(pooled.to_string(index=False), flush=True)
    print(f"Selected candidate: {selected}; saved {out}", flush=True)


def baseline_only(args):
    """Run all matched tree controls even when TabPFN access is unavailable."""
    args.output.mkdir(parents=True, exist_ok=True)
    frame = spatial_frame(pd.read_csv(DATA / 'train.csv'))
    predictions, scores, durations = [], [], []
    for start, end in FOLDS:
        fitting, future = fixed_origin_split(frame, start, end)
        fit, val = feature_frames(fitting, future)
        x_fit, x_val = model_inputs(fit, val, VARIANTS['tabpfn_history'])
        predicted, seconds = fit_tree('catboost', fitting, future, x_fit, x_val)
        result = future[[MONTH, TARGET, 'FLAT_TYPE_CLEAN']].copy()
        result[MONTH] = result[MONTH].astype(str)
        result['row_id'], result['validation_start'], result['catboost'] = future.index, start, predicted
        result.to_csv(args.output / f'baseline_{start}.csv', index=False, float_format='%.8f')
        score = {'validation_start': start, 'n': len(future), **metrics(future[TARGET], predicted)}
        scores.append(score)
        durations.append({'validation_start': start, 'model': 'catboost', 'fit_predict_seconds': seconds})
        predictions.append(result)
        print(f"CatBoost {start}: RMSE {score['rmse_sgd']:.3f}; {seconds:.1f}s", flush=True)
    pooled = pd.concat(predictions, ignore_index=True)
    pd.DataFrame(scores).to_csv(args.output / 'baseline_fold_scores.csv', index=False)
    write_json(args.output / 'baseline_results.json', {
        'model': 'catboost', 'n': len(pooled), **metrics(pooled[TARGET], pooled.catboost),
        'timings': durations, 'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in SOURCES},
        'settings': TREE_SETTINGS['catboost'], 'version': version('catboost'),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--model-version', choices=['v3.5', 'v3', 'v2.6', 'v2'], default='v3.5')
    parser.add_argument('--context-rows', type=int, default=20_000, help='Rows per estimator; 0 means full history')
    parser.add_argument('--n-estimators', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fit-mode', choices=['fit_preprocessors', 'fit_with_cache'], default='fit_with_cache')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--baseline-only', action='store_true', help='Run matched CatBoost controls without TabPFN')
    parser.add_argument('--check-access', action='store_true', help='Fit a tiny synthetic table to check CUDA and model access')
    args = parser.parse_args()
    if args.context_rows < 0 or min(args.n_estimators, args.batch_size) < 1:
        parser.error('Context must be nonnegative; estimator and batch sizes must be positive')
    # Authentication must be arranged externally; do not pop a browser or log tokens.
    os.environ.setdefault('TABPFN_NO_BROWSER', '1')
    os.environ.setdefault('TABPFN_MODEL_CACHE_DIR', str(ROOT / '.cache' / 'tabpfn'))
    # Relocate downloaded artifacts while retaining any existing user HF login.
    os.environ.setdefault('HF_HUB_CACHE', str(ROOT / '.cache' / 'huggingface' / 'hub'))
    os.environ.setdefault('TABPFN_DISABLE_TELEMETRY', '1')
    try:
        if args.baseline_only:
            baseline_only(args)
        elif args.check_access:
            # No project rows are needed to download and verify the checkpoint.
            x = pd.DataFrame({'a': np.arange(20, dtype=float), 'b': np.sin(np.arange(20))})
            predicted, timing = fit_tabpfn(args, x, 1000 + x.a * 20, x.iloc[:2])
            print(json.dumps({'predictions': predicted.tolist(), 'timing': timing}), flush=True)
        else:
            evaluate(args)
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / 'failure.json', {'type': type(exc).__name__, 'message': str(exc)})
        raise
