"""Train frozen Fast query-embedding heads on earlier origins; score the 80/20 holdout."""

from __future__ import annotations

import gc
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from analysis.compare_tabpfn_calendar_80_20 import frozen_settings
from analysis.compare_tabpfn_v3_80_20 import write_json
from analysis.compare_tabpfn_v3_5_fast_80_20 import CHECKPOINT
from analysis.config import DATA, REPORTS, ROOT
from analysis.data import sha256
from analysis.evaluation import metrics
from analysis.experiments.aligned_temporal_features import FEATURES, HISTORY_COLUMNS, LocalTemporalModel, origin_history
from analysis.experiments.calendar_encoding import MONTH_COLUMN, encoded_features, stratified_queries
from analysis.experiments.spatiotemporal_hybrid import TARGET, spatial_frame
from analysis.experiments.tabpfn_embedding_head import (HeadFeatures, mlp_correction, numeric_features,
                                                        population_weights, ridge_fit, shrink_choice,
                                                        train_mlp, weighted_rmse)


OUT = REPORTS / "tabpfn_fast_embedding_head_80_20"
CACHE = ROOT / ".cache/tabpfn_fast_embedding_head"
SOURCE = ROOT / ".cache/tabpfn_v3_features/holdout_80_20_2024-06_aligned_temporal.joblib"
OUTER = ROOT / ".cache/tabpfn_v3_features/calendar_80_20_categorical.joblib"
REFERENCE = REPORTS / "tabpfn_v3_5_fast_calendar_80_20/predictions_outer_categorical.csv"
ORIGINS = ("2022-05", "2022-09", "2023-01", "2023-05", "2023-08", "2024-02")
OUTER_ORIGIN = pd.Period("2024-06", "M")
CAL_CUTOFF = pd.Period("2023-12", "M")
BATCH = 200


def historical_bundle(base, frame, origin_str):
    path = CACHE / f"features_{origin_str}.joblib"
    manifest = OUT / f"features_{origin_str}.json"
    if path.exists() and manifest.exists():
        if sha256(path) != json.loads(manifest.read_text(encoding="utf-8"))["sha256"]:
            raise ValueError(f"Historical feature artifact changed: {path}")
        return joblib.load(path)
    origin = pd.Period(origin_str, "M")
    months = pd.PeriodIndex(base["train_month"], freq="M")
    ids = np.asarray(base["X_train"].index)
    context = np.flatnonzero(months < origin)
    possible = np.flatnonzero((months >= origin) & (months < min(origin + 10, OUTER_ORIGIN)))
    sampled, weight = stratified_queries(months[possible], ids[possible], per_month=BATCH, seed=42)
    selected = possible[sampled]
    x = base["X_train"].iloc[context].copy()
    q = base["X_train"].iloc[selected].copy()
    past = frame.loc[x.index]
    blind = frame.loc[q.index].drop(columns=TARGET)
    q.loc[:, HISTORY_COLUMNS] = origin_history(past, blind).reindex(q.index).to_numpy()
    q.loc[:, FEATURES] = LocalTemporalModel(past).features(blind, origin - 1).reindex(q.index).to_numpy()
    for column in x.select_dtypes(include="category"):
        if column == MONTH_COLUMN:
            continue
        known = pd.Index(x[column].dropna().astype(str).unique())
        for matrix in (x, q):
            values = matrix[column].astype("string")
            matrix[column] = pd.Categorical(values.where(values.isin(known)), categories=known)
    x = encoded_features(x, np.asarray(base["train_month"])[context], "categorical")
    q = encoded_features(q, np.asarray(base["train_month"])[selected], "categorical")
    baseline = q.TEMP_LOCAL_FORECAST.to_numpy(float)
    if not np.isfinite(baseline).all():
        raise AssertionError("Historical query baseline missing")
    bundle = {"X_train": x, "X_query": q,
              "y_train": np.asarray(base["y_train"], float)[context],
              "y_query": np.asarray(base["y_train"], float)[selected],
              "train_baseline": np.asarray(base["train_baseline"], float)[context],
              "query_baseline": baseline,
              "query_month": np.asarray(base["train_month"])[selected],
              "query_weights": weight}
    if months[context].max() >= origin or months[selected].min() < origin or months[selected].max() >= OUTER_ORIGIN:
        raise AssertionError("Historical episode crosses the origin or outer boundary")
    if len(np.intersect1d(x.index, q.index)):
        raise AssertionError("Query label entered TabPFN context")
    joblib.dump(bundle, path, compress=3)
    write_json(manifest, {"artifact": path.relative_to(ROOT).as_posix(), "sha256": sha256(path),
                          "origin": origin_str, "last_context_month": str(months[context].max()),
                          "context_rows": len(x), "query_rows": len(q),
                          "first_query_month": str(months[selected].min()),
                          "last_query_month": str(months[selected].max()),
                          "context_query_disjoint": True,
                          "query_sampling": "200 per target month, seed 42, label independent",
                          "feature_generation": "strict origin-1 rent histories and temporal forecast"})
    return bundle


def _fit_backbone(bundle):
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    x = bundle["X_train"]
    y = bundle["y_train"] - bundle["train_baseline"]
    torch.cuda.empty_cache()
    model = TabPFNRegressor.create_default_for_version(
        ModelVersion.V3_5_FAST, model_path=str(CHECKPOINT), device="cuda", n_estimators=1,
        random_state=42, fit_mode="fit_with_cache", keep_cache_on_device=True,
        inference_config={"SUBSAMPLE_SAMPLES": None})
    model.fit(x, y)
    if model.n_train_samples_ != len(x) or model.inference_config_.SUBSAMPLE_SAMPLES is not None:
        raise AssertionError("Backbone did not retain the full context")
    indices = model.ensemble_preprocessor_.subsample_row_indices
    if indices is not None and any(len(i) != len(x) or len(np.unique(i)) != len(x) for i in indices):
        raise AssertionError("Backbone sampled the context")
    for backbone in model.models_:
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
    return model


def embeddings_for(bundle, name, reference=None):
    """Complete query tokens, cached with row IDs and source prediction replay."""
    import torch

    path = CACHE / f"embeddings_{name}.npz"
    manifest = OUT / f"embeddings_{name}.json"
    row_ids = bundle["X_query"].index.to_numpy()
    if path.exists() and manifest.exists():
        if sha256(path) != json.loads(manifest.read_text(encoding="utf-8"))["sha256"]:
            raise ValueError(f"Embedding artifact changed: {path}")
        with np.load(path) as loaded:
            np.testing.assert_array_equal(loaded["row_id"], row_ids)
            return loaded["embedding"].copy(), loaded["prediction"].copy()
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model = _fit_backbone(bundle)
    fit_seconds = time.perf_counter() - started
    embeddings, predictions = [], []
    try:
        for offset in range(0, len(row_ids), BATCH):
            q = bundle["X_query"].iloc[offset:offset + BATCH]
            if reference is None:
                pred = model.predict(q, output_type="mean") + bundle["query_baseline"][offset:offset + len(q)]
            else:
                pred = reference[offset:offset + len(q)]
                if offset == 0:
                    replay = model.predict(q, output_type="mean") + bundle["query_baseline"][:len(q)]
                    if np.max(np.abs(replay - pred)) > 0.02:
                        raise AssertionError("Frozen backbone does not replay the saved 80/20 predictions")
            latent = np.asarray(model.get_embeddings(q, data_source="test"), np.float32)
            if latent.ndim != 3 or latent.shape[0] != 1 or latent.shape[1] != len(q) or latent.shape[2] != 1024:
                raise AssertionError(f"Unexpected query embedding shape: {latent.shape}")
            embeddings.append(latent[0])
            predictions.append(np.asarray(pred, float))
            if offset % 2000 == 0 or offset + BATCH >= len(row_ids):
                print(f"{name}: embedded {min(offset + BATCH, len(row_ids)):,}/{len(row_ids):,}", flush=True)
        latent = np.concatenate(embeddings)
        prediction = np.concatenate(predictions)
        if not np.isfinite(latent).all() or not np.isfinite(prediction).all():
            raise AssertionError("Nonfinite embedding or prediction")
        temp = path.with_suffix(".part.npz")
        np.savez(temp, row_id=row_ids, embedding=latent, prediction=prediction)
        temp.replace(path)
        write_json(manifest, {"artifact": path.relative_to(ROOT).as_posix(), "sha256": sha256(path),
                              "rows": len(row_ids), "width": latent.shape[1], "context_rows": len(bundle["X_train"]),
                              "fit_seconds": fit_seconds, "total_seconds": time.perf_counter() - started,
                              "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
                              "n_estimators": 1, "subsampled": False, "backbone_frozen": True,
                              "saved_prediction_replay": reference is not None})
        return latent, prediction
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def train_heads(numeric, latent, actual, baseline, weight, months):
    import torch

    period = pd.PeriodIndex(months, freq="M")
    train = np.asarray(period < CAL_CUTOFF)
    cal = np.asarray(period >= CAL_CUTOFF)
    if not train.any() or not cal.any() or (period >= OUTER_ORIGIN).any():
        raise AssertionError("Head training/calibration periods invalid")
    candidates = {}
    correction = actual - baseline
    selection_rows = []
    for name, kind in (("mlp_numeric", "numeric"), ("mlp_raw", "raw"),
                       ("mlp_pca32", "pca"), ("mlp_pca32_only", "pca_only"),
                       ("ridge_pca32", "pca")):
        print(f"Training {name} on {train.sum():,} earlier queries; calibrating on {cal.sum():,}", flush=True)
        prep = HeadFeatures(kind).fit(numeric[train], latent[train])
        x = prep.transform(numeric[train], latent[train])
        v = prep.transform(numeric[cal], latent[cal])
        if name.startswith("mlp"):
            model, epoch, _ = train_mlp(x, correction[train], weight[train],
                                        (v, actual[cal], baseline[cal], weight[cal]))
            delta = mlp_correction(model, v)
            parameters = {"epochs": epoch}
        else:
            ranked = []
            for alpha in (10.0, 100.0, 1000.0):
                trial = ridge_fit(x, correction[train], weight[train] / np.mean(weight[train]), alpha)
                trial_delta = trial.predict(v)
                _, trial_scores = shrink_choice(actual[cal], baseline[cal], trial_delta, weight[cal])
                ranked.append((min(trial_scores.values()), alpha, trial, trial_delta))
            _, alpha, model, delta = min(ranked, key=lambda item: (item[0], item[1]))
            parameters = {"alpha": alpha}
        gamma, curve = shrink_choice(actual[cal], baseline[cal], delta, weight[cal])
        score = curve[gamma]
        selection_rows.append({"model": name, "calibration_rmse_sgd": score,
                               "calibration_unshrunk_rmse_sgd": curve[1.0], "shrinkage": gamma, **parameters})
        candidates[name] = {"prep": prep, "model": model, "shrinkage": gamma, "parameters": parameters,
                            "calibration_curve": curve}
        print(f"  calibration RMSE {score:.3f}, unshrunk {curve[1.0]:.3f}, gamma {gamma}", flush=True)
    baseline_score = weighted_rmse(actual[cal], baseline[cal], weight[cal])
    ordered = sorted(selection_rows, key=lambda row: row["calibration_rmse_sgd"])
    eligible = [row for row in ordered if row["model"].startswith("mlp_") and
                row["model"] != "mlp_numeric" and row["shrinkage"] > 0 and
                row["calibration_rmse_sgd"] < baseline_score * 0.99]
    selected = eligible[0]["model"] if eligible else "baseline"
    torch.cuda.empty_cache()
    return candidates, pd.DataFrame(selection_rows), baseline_score, selected


def main():
    import torch

    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TABPFN_MAX_BATCHED_ESTIMATOR_ROWS", "0")
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    code = [Path(__file__), ROOT / "analysis/experiments/tabpfn_embedding_head.py",
            ROOT / "analysis/experiments/aligned_temporal_features.py",
            ROOT / "analysis/experiments/calendar_encoding.py"]
    settings = {"checkpoint_sha256": sha256(CHECKPOINT), "aligned_bundle_sha256": sha256(SOURCE),
                "outer_bundle_sha256": sha256(OUTER), "reference_sha256": sha256(REFERENCE),
                "code_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in code},
                "origins": ORIGINS, "outer_origin": str(OUTER_ORIGIN), "calibration_start": str(CAL_CUTOFF),
                "horizon_months": 10, "queries_per_month": BATCH, "batch_size": BATCH,
                "month_encoding": "categorical", "head_train_uses": "strict earlier target months",
                "head_selection_uses": "Dec 2023-May 2024 historical queries only",
                "shrinkage_grid": [0, .25, .5, 1], "embedding_pca_components": 32,
                "mlp": {"hidden": 32, "dropout": .2, "lr": .0003, "weight_decay": .01,
                        "max_epochs": 80, "patience": 10, "seed": 42},
                "head_selection_gate": "embedding MLP >=1% weighted calibration RMSE gain over TabPFN",
                "versions": {p: version(p) for p in ("tabpfn", "torch", "numpy", "pandas", "scikit-learn")},
                "gpu": torch.cuda.get_device_name()}
    frozen_settings(OUT / "settings.json", json.loads(json.dumps(settings)))
    base = joblib.load(SOURCE)
    frame = spatial_frame(pd.read_csv(DATA / "train.csv"))
    pieces = []
    for origin in ORIGINS:
        episode = historical_bundle(base, frame, origin)
        print(f"Historical origin {origin}: {len(episode['X_train']):,} context, {len(episode['X_query']):,} sampled queries", flush=True)
        embedding, prediction = embeddings_for(episode, origin)
        n = len(prediction)
        pieces.append({"embedding": embedding,
                       "numeric": numeric_features(episode["X_query"], episode["query_month"], [origin] * n, prediction),
                       "actual": episode["y_query"], "baseline": prediction,
                       "month": episode["query_month"], "origin": np.repeat(origin, n),
                       "weight": episode["query_weights"], "row_id": episode["X_query"].index.to_numpy()})
        del episode, embedding
        gc.collect()
    stacked = {key: np.concatenate([piece[key] for piece in pieces]) for key in pieces[0]}
    weight = population_weights(stacked["month"], stacked["origin"], stacked["weight"])
    train_month = pd.PeriodIndex(stacked["month"], freq="M")
    tr, ca = np.asarray(train_month < CAL_CUTOFF), np.asarray(train_month >= CAL_CUTOFF)
    if set(stacked["row_id"][tr]) & set(stacked["row_id"][ca]):
        raise AssertionError("Head train/calibration row ID overlap")
    write_json(OUT / "historical_audit.json", {"query_cases": len(train_month), "unique_row_ids": int(len(np.unique(stacked["row_id"]))),
                                                "head_train_cases": int(tr.sum()), "calibration_cases": int(ca.sum()),
                                                "first_target_month": str(train_month.min()),
                                                "last_target_month": str(train_month.max()),
                                                "head_train_last_month": str(train_month[tr].max()),
                                                "calibration_first_month": str(train_month[ca].min()),
                                                "outer_labels_used": False, "head_train_calibration_row_overlap": False,
                                                "calibration_tabpfn_rmse_sgd": weighted_rmse(stacked["actual"][ca], stacked["baseline"][ca], weight[ca])})
    candidates, selection, cal_base, selected = train_heads(
        stacked["numeric"], stacked["embedding"], stacked["actual"], stacked["baseline"], weight, stacked["month"])
    selection.to_csv(OUT / "calibration_scores.csv", index=False, float_format="%.8f")
    write_json(OUT / "selection.json", {"selected_before_outer_scoring": selected,
                                       "baseline_calibration_rmse_sgd": cal_base,
                                       "head_fit_only_before_2023_12": True,
                                       "backbone_frozen": True,
                                       "full_head_refit_after_calibration": False})
    joblib.dump(candidates, CACHE / "selected_heads.joblib", compress=3)
    print(f"Selected before outer scoring: {selected}; baseline calibration RMSE {cal_base:.3f}", flush=True)
    outer = joblib.load(OUTER)
    saved = pd.read_csv(REFERENCE)
    np.testing.assert_array_equal(saved.row_id.to_numpy(), outer["X_query"].index.to_numpy())
    np.testing.assert_array_equal(saved.actual.to_numpy(), outer["y_query"])
    if len(outer["X_train"]) != 121005 or len(outer["X_query"]) != 28995:
        raise AssertionError("Outer split changed")
    emb, prediction = embeddings_for(outer, "outer", saved.predicted.to_numpy(float))
    np.testing.assert_allclose(prediction, saved.predicted, atol=1e-8, rtol=0)
    numeric = numeric_features(outer["X_query"], outer["query_month"], [str(OUTER_ORIGIN)] * len(prediction), prediction)
    result = pd.DataFrame({"row_id": saved.row_id, "month": saved.month, "actual": saved.actual, "tabpfn": prediction})
    for name, item in candidates.items():
        x = item["prep"].transform(numeric, emb)
        delta = mlp_correction(item["model"], x) if name.startswith("mlp") else item["model"].predict(x)
        result[f"{name}_unshrunk"] = prediction + delta
        result[name] = prediction + item["shrinkage"] * delta
    result["selected"] = result[selected] if selected != "baseline" else result["tabpfn"]
    scores = pd.DataFrame([{"model": name, "rows": len(result), **metrics(result.actual, result[name]),
                            "mse_sgd2": float(np.mean((result[name] - result.actual) ** 2))}
                           for name in ["tabpfn", *candidates, *(f"{name}_unshrunk" for name in candidates), "selected"]])
    result.to_csv(OUT / "predictions.csv", index=False, float_format="%.8f")
    scores.to_csv(OUT / "scores.csv", index=False, float_format="%.8f")
    pd.DataFrame([{"model": name, "month": month, "n": len(part), **metrics(part.actual, part[name])}
                  for month, part in result.groupby("month", sort=True)
                  for name in ["tabpfn", "selected", *candidates]]).to_csv(OUT / "monthly_scores.csv", index=False, float_format="%.8f")
    base_outer = float(scores.set_index("model").loc["tabpfn", "rmse_sgd"])
    selected_outer = float(scores.set_index("model").loc["selected", "rmse_sgd"])
    lines = ["# Frozen TabPFN-3.5-Fast embedding MLP on the categorical-month 80/20 split", "",
             "The backbone uses the same one-estimator checkpoint, 77 features, all 121,005 historical context rows, "
             "and the categorical month encoding as the saved S$455.81 RMSE experiment. It is frozen. "
             "Every head training embedding is a query from an earlier forecast origin; its rent label is absent from "
             "that origin's context. The 28,995 June 2024-March 2025 holdout rows are scored only after the head "
             "settings are selected from December 2023-May 2024 historical queries.", "",
             f"Historical query cases: {len(train_month):,} ({tr.sum():,} head training, {ca.sum():,} calibration). "
             "The head preprocessors fit on head training cases only; selected heads are retained without refitting.", "",
             "| Model | Validation RMSE (S$) | Validation MSE (S$²) | MAE (S$) |", "|---|---:|---:|---:|" ]
    for row in scores.itertuples():
        lines.append(f"| {row.model} | {row.rmse_sgd:.3f} | {row.mse_sgd2:.1f} | {row.mae_sgd:.3f} |")
    lines += ["", f"Preselected result: **{selected}**, outer RMSE **S${selected_outer:.3f}**, "
              f"change **{selected_outer - base_outer:+.3f}** from frozen TabPFN.", "",
              "The outer holdout has been inspected in prior project experiments, so this is a development comparison, "
              "not an untouched estimate of generalization. Historical calibration and outer validation have different "
              "origin dates and query populations. PCA32, numeric-only, Ridge, and unshrunk heads are controls. "
              "The selected head uses a nonzero correction only if an embedding MLP cleared the prespecified "
              "1% historical calibration gain gate.", ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(OUT / "verification.json", {"rows": len(result), "row_ids_match_saved_outer": True,
                                           "reference_first_batch_replayed": True,
                                           "all_outputs_finite": bool(np.isfinite(result.select_dtypes(include="number")).all().all()),
                                           "backbone_frozen": True, "selected_before_outer_scoring": selected,
                                           "outer_rmse_sgd": selected_outer, "baseline_rmse_sgd": base_outer})
    print(scores.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
