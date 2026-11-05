"""Controlled, complete-availability calendar ablations around fixed residuals.

All context rows are retained. Earlier ten-month folds select encodings using
fixed monthly query samples, before any new outer predictions are scored.
The outer ten-month 80/20 validation is evaluated in full for every variant.
"""

from __future__ import annotations

import argparse
import gc
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from analysis.compare_tabpfn_v3_80_20 import markdown_table, write_json
from analysis.compare_tabpfn_v3_5_fast_80_20 import CHECKPOINT, predict_all_context
from analysis.config import DATA, REPORTS, ROOT
from analysis.data import sha256
from analysis.evaluation import metrics
from analysis.experiments.aligned_temporal_features import FEATURES, HISTORY_COLUMNS, LocalTemporalModel, origin_history
from analysis.experiments.calendar_encoding import MONTH_COLUMN, TIME_COLUMN, encoded_features, stratified_queries
from analysis.experiments.spatiotemporal_hybrid import MONTH, TARGET, spatial_frame


OUT = REPORTS / "tabpfn_v3_5_fast_calendar_80_20"
SOURCE = REPORTS / "tabpfn_v3_5_fast_aligned_temporal_80_20"
CACHE = ROOT / ".cache/tabpfn_v3_features"
INNER_ORIGINS = ("2022-09", "2023-07")
ENCODINGS = ("integer", "categorical", "cyclical")
ALL_VARIANTS = (*ENCODINGS, "calendar", "school_interactions")


def weighted_metrics(actual, predicted, weight):
    difference = np.asarray(predicted, float) - np.asarray(actual, float)
    return {"rmse_sgd": float(np.sqrt(np.average(difference ** 2, weights=weight))),
            "mae_sgd": float(np.average(np.abs(difference), weights=weight)),
            "bias_sgd": float(np.average(difference, weights=weight))}


def frozen_settings(path, settings):
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != settings:
        raise ValueError(f"Existing {path} uses different settings; preserve it and use a new output directory")
    write_json(path, settings)


def variant_matrices(bundle, name, calendar_encoding):
    encoding = name if name in ENCODINGS else calendar_encoding
    options = {"calendar": name in ("calendar", "school_interactions"), "interactions": name == "school_interactions"}
    x, q = [encoded_features(bundle[key], bundle[f"{split}_month"], encoding, **options)
            for split, key in (("train", "X_train"), ("query", "X_query"))]
    if list(x) != list(q):
        raise AssertionError("Context/query feature order differs")
    new = [c for c in x if c not in bundle["X_train"]]
    if new and (x[new].isna().any().any() or q[new].isna().any().any()):
        raise AssertionError("New features have a context/query availability mismatch")
    if name == "categorical" and (x[MONTH_COLUMN].dtype != q[MONTH_COLUMN].dtype or x[MONTH_COLUMN].isna().any() or q[MONTH_COLUMN].isna().any()):
        raise AssertionError("Month categories are incomplete or inconsistent")
    return x, q, {"features": len(x.columns), "encoding": encoding, "new_columns": new,
                  "new_context_missing": x[new].isna().sum().to_dict(), "new_query_missing": q[new].isna().sum().to_dict(),
                  "elapsed_month_index_preserved": True, "existing_history_missingness_preserved": True,
                  "residual_baseline_and_labels_unchanged": True}


def inner_bundle(base, frame, origin):
    path, manifest_path = CACHE / f"calendar_inner_{origin}.joblib", OUT / f"inner_features_{origin}.json"
    if path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if sha256(path) != manifest["sha256"]:
            raise ValueError("Earlier-origin feature artifact changed")
        return joblib.load(path)
    origin = pd.Period(origin, "M")
    months = pd.PeriodIndex(base["train_month"], freq="M")
    train_positions = np.flatnonzero(months < origin)
    query_positions = np.flatnonzero((months >= origin) & (months < origin + 10))
    sampled, weight = stratified_queries(months[query_positions], base["X_train"].index.to_numpy()[query_positions])
    query_positions = query_positions[sampled]
    x, q = base["X_train"].iloc[train_positions].copy(), base["X_train"].iloc[query_positions].copy()
    past = frame.loc[x.index]
    blind = frame.loc[q.index].drop(columns=TARGET)
    # The query's original pseudo-origin features are discarded and rebuilt at
    # this fixed forecast origin. No query labels enter a history or baseline.
    q.loc[:, HISTORY_COLUMNS] = origin_history(past, blind).reindex(q.index).to_numpy()
    temporal = LocalTemporalModel(past)
    q.loc[:, FEATURES] = temporal.features(blind, origin - 1).reindex(q.index).to_numpy()
    for column in x.select_dtypes(include="category"):
        known = pd.Index(x[column].dropna().astype(str).unique())
        for matrix in (x, q):
            values = matrix[column].astype("string")
            matrix[column] = pd.Categorical(values.where(values.isin(known)), categories=known)
    baseline = q.TEMP_LOCAL_FORECAST.to_numpy(float)
    if not np.isfinite(baseline).all():
        raise AssertionError("Earlier-origin query baseline is missing")
    bundle = {"X_train": x, "X_query": q, "y_train": np.asarray(base["y_train"])[train_positions],
              "y_query": np.asarray(base["y_train"])[query_positions],
              "train_month": np.asarray(base["train_month"])[train_positions],
              "query_month": np.asarray(base["train_month"])[query_positions],
              "train_baseline": np.asarray(base["train_baseline"])[train_positions], "query_baseline": baseline,
              "query_weights": weight}
    if pd.PeriodIndex(bundle["train_month"], freq="M").max() >= origin:
        raise AssertionError("Earlier validation labels entered context")
    joblib.dump(bundle, path, compress=3)
    write_json(manifest_path, {"artifact": path.relative_to(ROOT).as_posix(), "sha256": sha256(path),
                              "origin": str(origin), "context_rows": len(x), "sampled_query_rows": len(q),
                              "represented_query_rows": float(weight.sum()), "horizons": [1, 10],
                              "last_history_month": str(origin - 1), "query_sampling": "200 per month; no labels; seed 42",
                              "sampling_weights": "month population divided by month sampled count"})
    return bundle


def predict_job(bundle, name, stage, calendar_encoding):
    path, timing_path = OUT / f"predictions_{stage}_{name}.csv", OUT / f"timing_{stage}_{name}.json"
    x, q, audit = variant_matrices(bundle, name, calendar_encoding)
    write_json(OUT / f"availability_{stage}_{name}.json", audit)
    if stage == "outer":
        artifact = CACHE / f"calendar_80_20_{name}.joblib"
        artifact_manifest = OUT / f"feature_manifest_{name}.json"
        if not artifact_manifest.exists():
            joblib.dump({**bundle, "X_train": x, "X_query": q}, artifact, compress=3)
            write_json(artifact_manifest, {**audit, "artifact": artifact.relative_to(ROOT).as_posix(), "sha256": sha256(artifact),
                                           "residual_target": "y_train - train_baseline",
                                           "reconstruction": "predicted_residual + query_baseline"})
        elif sha256(artifact) != json.loads(artifact_manifest.read_text(encoding="utf-8"))["sha256"]:
            raise ValueError("Saved outer feature bundle changed")
    if path.exists() and timing_path.exists():
        saved = pd.read_csv(path)
        np.testing.assert_array_equal(saved.row_id.to_numpy(), q.index.to_numpy())
        np.testing.assert_array_equal(saved.actual.to_numpy(), bundle["y_query"])
        return saved.predicted.to_numpy(float)
    target = np.asarray(bundle["y_train"], float) - np.asarray(bundle["train_baseline"], float)
    np.testing.assert_allclose(target + bundle["train_baseline"], bundle["y_train"], atol=1e-10)
    write_json(OUT / "status.json", {"stage": stage, "variant": name, "status": "inference", "context_rows": len(x), "query_rows": len(q)})
    print(f"\n{stage}: {name}, {len(x):,} full context rows, {len(q):,} queries, {len(x.columns)} features", flush=True)
    try:
        residual, timing = predict_all_context(x, target, q, keep_cache_on_device=True)
    except Exception as exc:
        if "out of memory" not in str(exc).lower():
            raise
        write_json(OUT / f"gpu_retry_{stage}_{name}.json", {"reason": str(exc)})
        gc.collect()
        residual, timing = predict_all_context(x, target, q, keep_cache_on_device=False)
    prediction = residual + bundle["query_baseline"]
    pd.DataFrame({"row_id": q.index, "month": bundle["query_month"], "actual": bundle["y_query"],
                  "predicted": prediction}).to_csv(path, index=False, float_format="%.8f")
    write_json(timing_path, timing)
    return prediction


def summarize_inner(records):
    folds = pd.DataFrame([{k: v for k, v in record.items() if k not in ("actual", "predicted", "weight")}
                          | weighted_metrics(record["actual"], record["predicted"], record["weight"]) for record in records])
    scores = []
    for name in folds.model.unique():
        pieces = [r for r in records if r["model"] == name]
        scores.append({"model": name, **weighted_metrics(np.concatenate([r["actual"] for r in pieces]),
                       np.concatenate([r["predicted"] for r in pieces]), np.concatenate([r["weight"] for r in pieces]))})
    scores = pd.DataFrame(scores)
    scores.to_csv(OUT / "inner_scores.csv", index=False, float_format="%.8f")
    folds.to_csv(OUT / "inner_fold_scores.csv", index=False, float_format="%.8f")
    return scores, folds


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inner-only", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TABPFN_MAX_BATCHED_ESTIMATOR_ROWS", "0")
    OUT.mkdir(parents=True, exist_ok=True)
    source_audit = json.loads((SOURCE / "feature_manifest.json").read_text(encoding="utf-8"))
    source_settings = json.loads((SOURCE / "settings.json").read_text(encoding="utf-8"))
    for name, expected in source_settings["source_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Source data changed: {name}")
    artifact = ROOT / source_audit["artifact"]
    if sha256(artifact) != source_audit["sha256"]:
        raise ValueError("Aligned baseline bundle changed")
    control_path = SOURCE / "predictions_tabpfn_aligned_residual.csv"
    code = [Path(__file__), ROOT / "analysis/experiments/calendar_encoding.py",
            ROOT / "analysis/experiments/aligned_temporal_features.py", ROOT / "analysis/compare_tabpfn_v3_5_fast_80_20.py"]
    settings = {"model": "v3.5-fast", "n_estimators": 1, "seed": 42, "batch_size": 200, "subsampling": False,
                "source_artifact_sha256": source_audit["sha256"], "source_sha256": source_settings["source_sha256"],
                "checkpoint_sha256": sha256(CHECKPOINT), "control_predictions_sha256": sha256(control_path),
                "code_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in code},
                "versions": {p: version(p) for p in ("tabpfn", "torch", "numpy", "pandas", "scikit-learn")},
                "gpu": torch.cuda.get_device_name(), "inner_origins": INNER_ORIGINS, "inner_horizons": [1, 10],
                "inner_queries_per_month": 200, "inner_query_sampling_seed": 42,
                "outer_rows": 28995, "outer_context_rows": 121005,
                "calendar_encoding_rule": "lowest earlier pooled RMSE; prefer integer/categorical/cyclical in that order within 0.1% of best",
                "candidate_selection_rule": "at least 0.25% earlier pooled RMSE gain and no inner fold more than 0.5% worse; simpler within 0.1% of best",
                "calendar_windows": {"primary_start": [1], "primary_lead1": [12], "primary_lead2": [11],
                                     "university_start": [1, 8], "university_lead1": [12, 7], "university_lead2": [11, 6]},
                "calendar_sources": ["https://nus.edu.sg/registrar/calendar", "https://www.sengkangpri.moe.edu.sg/files/Parent%20Resources/2024/2024_term_1_notification.pdf"],
                "calendar_caveat": "recurring monthly proxies, not exact historical term or holiday dates; no university proximity data",
                "unchanged": ["continuous time index", "all existing rent histories", "residual baseline", "rent labels", "context row IDs"]}
    # JSON roundtrip canonicalizes tuples before comparing resume settings.
    settings = json.loads(json.dumps(settings))
    frozen_settings(OUT / "settings.json", settings)
    base = joblib.load(artifact)
    frame = spatial_frame(pd.read_csv(DATA / "train.csv"))
    inner = {origin: inner_bundle(base, frame, origin) for origin in INNER_ORIGINS}
    records = []

    def earlier(name, encoding):
        for origin, bundle in inner.items():
            predicted = predict_job(bundle, name, f"inner_{origin}", encoding)
            records.append({"model": name, "origin": origin, "query_rows": len(predicted),
                            "context_rows": len(bundle["X_train"]), "actual": bundle["y_query"],
                            "predicted": predicted, "weight": bundle["query_weights"]})

    for name in ENCODINGS:
        earlier(name, name)
        summarize_inner(records)
    encoding_scores, _ = summarize_inner(records)
    threshold = encoding_scores.rmse_sgd.min() * 1.001
    encoding = next(name for name in ENCODINGS if float(encoding_scores.set_index("model").loc[name, "rmse_sgd"]) <= threshold)
    frozen_settings(OUT / "encoding_selection.json", {"encoding": encoding, "selection_uses": "earlier sampled-query ten-month folds only"})
    for name in ("calendar", "school_interactions"):
        earlier(name, encoding)
        summarize_inner(records)
    scores, folds = summarize_inner(records)
    baseline = float(scores.set_index("model").loc["integer", "rmse_sgd"])
    eligible = []
    for name in ALL_VARIANTS[1:]:
        rows = folds[folds.model.eq(name)].set_index("origin").rmse_sgd
        control = folds[folds.model.eq("integer")].set_index("origin").rmse_sgd
        score = float(scores.set_index("model").loc[name, "rmse_sgd"])
        if score < baseline * 0.9975 and (rows <= control * 1.005).all():
            eligible.append(name)
    selected = "integer"
    if eligible:
        threshold = min(float(scores.set_index("model").loc[name, "rmse_sgd"]) for name in eligible) * 1.001
        selected = next(name for name in eligible if float(scores.set_index("model").loc[name, "rmse_sgd"]) <= threshold)
    frozen_settings(OUT / "inner_selection.json", {"model": selected, "calendar_encoding": encoding,
                                                  "eligible_models": eligible, "selected_before_new_outer_scoring": True})
    print(f"Earlier selection: encoding={encoding}, candidate={selected}", flush=True)
    if args.inner_only:
        write_json(OUT / "status.json", {"status": "inner_complete"})
        return
    saved = pd.read_csv(control_path)
    np.testing.assert_array_equal(saved.row_id.to_numpy(), base["X_query"].index.to_numpy())
    np.testing.assert_array_equal(saved.actual.to_numpy(), base["y_query"])
    replay_path = OUT / "control_replay.json"
    if not replay_path.exists():
        subset = {**base, "X_query": base["X_query"].iloc[:400], "y_query": base["y_query"][:400],
                  "query_month": base["query_month"][:400], "query_baseline": base["query_baseline"][:400]}
        replay = predict_job(subset, "integer", "control_replay", "integer")
        np.testing.assert_allclose(replay, saved.predicted.to_numpy()[:400], atol=0.02, rtol=0)
        write_json(replay_path, {"rows": 400, "all_context_rows": len(base["X_train"]),
                                 "max_difference_sgd": float(np.max(np.abs(replay - saved.predicted.to_numpy()[:400])))})
    result = pd.DataFrame({"row_id": saved.row_id, "month": base["query_month"], "actual": saved.actual,
                           "integer": saved.predicted})
    for name in ALL_VARIANTS[1:]:
        result[name] = predict_job(base, name, "outer", encoding)
    result.to_csv(OUT / "validation_predictions.csv", index=False, float_format="%.8f")
    outer_scores = pd.DataFrame([{"model": name, "n": len(result), **metrics(result.actual, result[name])} for name in ALL_VARIANTS])
    monthly = pd.DataFrame([{"model": name, "month": month, "n": len(part), **metrics(part.actual, part[name])}
                            for month, part in result.groupby("month", sort=True) for name in ALL_VARIANTS])
    outer_scores.to_csv(OUT / "scores.csv", index=False, float_format="%.8f")
    monthly.to_csv(OUT / "monthly_scores.csv", index=False, float_format="%.8f")
    winner = outer_scores.set_index("model")
    confirmed = selected != "integer" and winner.loc[selected, "rmse_sgd"] < winner.loc["integer", "rmse_sgd"]
    recommendation = selected if confirmed else "integer"
    write_json(OUT / "selection.json", {"inner_selected": selected, "outer_confirmation_passed": bool(confirmed),
                                        "recommended": recommendation, "outer_used_to_choose_another_variant": False})
    report = ["# Controlled month encoding and school-calendar proxy experiments", "",
              "The aligned residual baseline, its 37 rental-history inputs, elapsed-month index, rent labels, "
              "and all 121,005 context rows are held fixed. The complete 28,995-row June 2024-March 2025 "
              "development holdout is predicted for each candidate with the same Fast checkpoint, one estimator, "
              "seed 42, and 200-row batches. No new fitted property reference or economic model is introduced.", "",
              "Integer month is the saved S$455.86 control; its first 400 predictions are replayed with the full "
              "context before reuse. Categorical month changes only its dtype. Cyclical month replaces only "
              "month-of-year with sine/cosine and retains the continuous elapsed-month index.", "",
              "All new features are label-free and complete from the first context month, including warmup rows. "
              "Existing history warmup missingness is deliberately unchanged across candidates, rather than "
              "imputed with future data. Availability audits record zero missing entries for every added column.", "",
              "Two earlier ten-month folds (September 2022-June 2023 and July 2023-April 2024) retain every "
              "pre-origin context row. Query evaluation samples 200 rows per month independently of labels, "
              "with month population/sample weights. Fold query histories and baselines are rebuilt at the "
              "fold's fixed origin. Sampling is for earlier query evaluation only; neither context nor final "
              "80/20 validation is sampled.", "",
              f"Calendar variants use `{encoding}`, selected only on earlier folds before new outer scoring. "
              "The calendar block has six recurring monthly proxies: January annual primary-school opening, "
              "January/August university intake, and one-/two-month lead windows. These are hypotheses and "
              "feature representations, not exact historical term/holiday calendars. University campus distance "
              "is not available. The interaction block adds primary-window x primary-school access/count "
              "features plus a source-location-missing flag; flat type remains an existing categorical input.", "",
              "## Earlier selection", "", markdown_table(scores), "",
              f"Earlier selected candidate: `{selected}`. Selection requires at least 0.25% pooled improvement "
              "with neither earlier fold more than 0.5% worse. Simpler configurations win within 0.1% of best.", "",
              "## Full 80/20 results", "", markdown_table(outer_scores), "",
              f"Recommendation: `{recommendation}`. A candidate chosen on earlier folds is retained only if "
              "its full outer RMSE improves; a different candidate is not selected by inspecting outer scores. "
              "The outer holdout has been examined previously, so this remains a development result. "
              "Ten-month performance does not establish accuracy across the supplied seventeen-month test.", "",
              "Each model-ready bundle is saved under `.cache/tabpfn_v3_features/calendar_80_20_<variant>.joblib`. "
              "Use raw `y_train - train_baseline` to fit and add `query_baseline` exactly once to predictions. "
              "See feature manifests, availability audits, timings, frozen settings and selections.", "",
              "Calendar references: [NUS](https://nus.edu.sg/registrar/calendar), "
              "[MOE primary-school opening example](https://www.sengkangpri.moe.edu.sg/files/Parent%20Resources/2024/2024_term_1_notification.pdf).", "",
              "Reproduce/resume: `.venv/Scripts/python.exe analysis/compare_tabpfn_calendar_80_20.py`."]
    (OUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(OUT / "status.json", {"status": "complete", "recommended": recommendation})
    print(outer_scores.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
