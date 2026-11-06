"""Use the latest half of the training time horizon as frozen TabPFN context.

Evaluate pooled and flat-type-specific recent contexts against the unchanged
77-feature full-context baselines on all existing chronological holdout rows.
"""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd

from analysis.config import ROOT
from analysis.data import sha256
from analysis.evaluation import metrics
from analysis.compare_local_economy_landscape import immutable, write_json, frozen_predict, hashlib_array
from analysis.compare_tabpfn_v3_5_fast_80_20 import CHECKPOINT
from analysis.experiments.flat_type_inference import flat_type_partitions, assemble_predictions
from analysis.experiments.recent_time_context import recent_half_positions

NAME = "tabpfn_recent_half_77_80_20"
OUT, CACHE = ROOT / "reports" / NAME, ROOT / ".cache" / NAME


def load():
    cfg = json.loads((ROOT / "analysis/configs/local_economy_landscape.json").read_text())
    bundle_path = ROOT / cfg["baseline"]
    if sha256(bundle_path) != cfg["baseline_sha256"]:
        raise ValueError("Original 77-feature bundle changed")
    b = joblib.load(bundle_path)
    assert b["X_train"].shape == (121005, 77) and b["X_query"].shape == (28995, 77)
    reference_path = ROOT / cfg["baseline_predictions"]
    expected = json.loads((ROOT / "reports" / cfg["experiment"] / "settings.json").read_text())["baseline_predictions_sha256"]
    if sha256(reference_path) != expected:
        raise ValueError("Full-context baseline predictions changed")
    reference = pd.read_csv(reference_path).set_index("row_id").loc[b["X_query"].index]
    np.testing.assert_array_equal(reference.actual, b["y_query"])
    positions, policy = recent_half_positions(b["train_month"])
    x = b["X_train"].iloc[positions]
    partitions = flat_type_partitions(x, b["X_query"])
    if pd.PeriodIndex(b["train_month"], freq="M").max() >= pd.PeriodIndex(b["query_month"], freq="M").min():
        raise ValueError("Context must precede validation")
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    immutable(OUT / "settings.json", {"source_bundle": cfg["baseline"], "source_bundle_sha256": cfg["baseline_sha256"],
              "reference_predictions_sha256": expected, "checkpoint_sha256": sha256(CHECKPOINT),
              "selection": policy, "selected_context_ids_sha256": hashlib_array(x.index),
              "query_ids_sha256": hashlib_array(b["X_query"].index),
              "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in (Path(__file__), ROOT / "analysis/experiments/recent_time_context.py", ROOT / "analysis/experiments/flat_type_inference.py", ROOT / "analysis/compare_local_economy_landscape.py")},
              "packages": {p: version(p) for p in ("tabpfn", "torch", "numpy", "pandas")},
              "feature_policy": "All original 77 columns and dtypes unchanged. Earlier data may contribute to inherited rental histories and temporal baselines; only TabPFN context rows are truncated.",
              "inference": {"checkpoint": "v3.5-fast", "n_estimators": 1, "seed": 42, "batch_size": 200, "subsample_samples": None, "parameters_optimized": False,
                            "preprocessing": "Native model preprocessing fitted only on each retained context"},
              "unlabelled_test_used": False})
    pd.DataFrame([{"flat_type": k, "recent_context_rows": len(a), "query_rows": len(q)} for k, (a, q) in partitions.items()]).to_csv(OUT / "category_counts.csv", index=False)
    return b, reference, positions, partitions, policy


def run_one(mode, kind=None):
    b, reference, positions, partitions, _ = load()
    if mode == "pooled":
        context, queries = positions, np.arange(len(b["X_query"]))
        tag = "recent_pooled"
    else:
        a, queries = partitions[kind]
        context = positions[a]
        tag = f"recent_type_{kind.replace('-', '_')}"
    x, q = b["X_train"].iloc[context], b["X_query"].iloc[queries]
    immutable(OUT / f"job_{tag}.json", {"mode": mode, "flat_type": kind, "context_rows": len(x), "query_rows": len(q),
              "context_ids_sha256": hashlib_array(x.index), "query_ids_sha256": hashlib_array(q.index),
              "settings_sha256": sha256(OUT / "settings.json"), "input_columns": list(x.columns)})
    path, timing_path = OUT / f"predictions_{tag}.csv", OUT / f"timing_{tag}.json"
    if path.exists() and timing_path.exists():
        saved = pd.read_csv(path)
        np.testing.assert_array_equal(saved.row_id, q.index)
        np.testing.assert_array_equal(saved.actual, np.asarray(b["y_query"])[queries])
        assert np.isfinite(saved.predicted).all()
        print(f"Reused {tag}", flush=True)
        return
    print(f"{tag}: {len(x):,} context rows, {len(q):,} validation rows, 77 features", flush=True)
    residual, timing = frozen_predict(x, np.asarray(b["y_train"], float)[context] - np.asarray(b["train_baseline"])[context], q, CACHE / f"partial_{tag}.npy")
    predicted = residual + np.asarray(b["query_baseline"])[queries]
    frame = pd.DataFrame({"row_id": q.index, "month": np.asarray(b["query_month"])[queries], "flat_type": q.FLAT_TYPE_CLEAN.astype(str).to_numpy(),
                          "actual": np.asarray(b["y_query"])[queries], "predicted": predicted,
                          "full_context_predicted": reference.loc[q.index].predicted.to_numpy()})
    frame.to_csv(path, index=False, float_format="%.10f")
    timing.update(**metrics(frame.actual, frame.predicted))
    write_json(timing_path, timing)
    print(json.dumps(timing), flush=True)


def summarize():
    b, reference, positions, partitions, policy = load()
    query = b["X_query"]
    frames = [pd.read_csv(OUT / f"predictions_recent_type_{kind.replace('-', '_')}.csv") for kind in partitions]
    separate = assemble_predictions(query, frames).set_index("row_id")
    pooled = pd.read_csv(OUT / "predictions_recent_pooled.csv").set_index("row_id")
    prior_out = ROOT / "reports/tabpfn_by_flat_type_77_80_20"
    prior_settings = json.loads((prior_out / "settings.json").read_text())
    if prior_settings["baseline_sha256"] != json.loads((OUT / "settings.json").read_text())["source_bundle_sha256"] or prior_settings["checkpoint_sha256"] != sha256(CHECKPOINT):
        raise ValueError("Full-context per-type baseline uses different inputs")
    prior = pd.read_csv(prior_out / "predictions_combined.csv").set_index("row_id")
    frame = pd.DataFrame({"row_id": query.index, "flat_type": query.FLAT_TYPE_CLEAN.astype(str).to_numpy(),
                          "month": np.asarray(b["query_month"]).astype(str), "actual": b["y_query"]}).set_index("row_id")
    for name, predictions in (("full_pooled", reference), ("full_by_flat_type", prior), ("recent_pooled", pooled), ("recent_by_flat_type", separate)):
        if not predictions.index.is_unique or len(predictions) != len(query) or set(predictions.index) != set(query.index):
            raise ValueError(f"Incomplete or duplicated query IDs: {name}")
        predictions = predictions.loc[query.index]
        np.testing.assert_array_equal(predictions.actual, b["y_query"])
        np.testing.assert_array_equal(predictions.month.astype(str), np.asarray(b["query_month"]).astype(str))
        if not np.isfinite(predictions.predicted).all():
            raise ValueError(f"Non-finite predictions: {name}")
        frame[name] = predictions.predicted.to_numpy()
    frame.reset_index().to_csv(OUT / "predictions_combined_comparison.csv", index=False, float_format="%.10f")
    models = ["full_pooled", "full_by_flat_type", "recent_pooled", "recent_by_flat_type"]
    scores = pd.DataFrame([{"model": name, "query_rows": len(frame), **metrics(frame.actual, frame[name])} for name in models])
    scores["delta_rmse_vs_full_pooled_sgd"] = scores.rmse_sgd - scores.rmse_sgd.iloc[0]
    scores.to_csv(OUT / "scores.csv", index=False)
    grouped, monthly = [], []
    for kind, part in frame.groupby("flat_type"):
        for name in models:
            grouped.append({"flat_type": kind, "model": name, "query_rows": len(part), **metrics(part.actual, part[name])})
    for month, part in frame.groupby("month"):
        for name in models:
            monthly.append({"month": month, "model": name, "query_rows": len(part), **metrics(part.actual, part[name])})
    grouped = pd.DataFrame(grouped)
    grouped.to_csv(OUT / "scores_by_flat_type.csv", index=False)
    pd.DataFrame(monthly).to_csv(OUT / "monthly_scores.csv", index=False)
    timings = {"recent_pooled": json.loads((OUT / "timing_recent_pooled.json").read_text())}
    type_timings = [json.loads((OUT / f"timing_recent_type_{k.replace('-', '_')}.json").read_text()) for k in partitions]
    timings["recent_by_flat_type"] = {"fit_seconds": sum(t["fit_seconds"] for t in type_timings), "predict_seconds": sum(t["predict_seconds"] for t in type_timings),
                                      "peak_vram_gib": max(t["peak_vram_gib"] for t in type_timings)}
    write_json(OUT / "timing_summary.json", timings)
    write_json(OUT / "verification.json", {"all_28995_queries_covered_once": True, "all_row_labels_months_match": True,
               "recent_context_rows": len(positions), "full_context_rows": len(b["X_train"]), "query_rows": len(query),
               "recent_context_min_month": str(pd.PeriodIndex(b["train_month"], freq="M")[positions].min()),
               "recent_context_max_month": str(pd.PeriodIndex(b["train_month"], freq="M")[positions].max()),
               "all_contexts_unsubsampled": not timings["recent_pooled"]["subsampling"] and all(not t["subsampling"] for t in type_timings),
               "all_parameter_versions_unchanged": timings["recent_pooled"]["parameter_versions_unchanged"] and all(t["parameter_versions_unchanged"] for t in type_timings),
               "combined_predictions_sha256": sha256(OUT / "predictions_combined_comparison.csv"),
               "full_type_reference_sha256": sha256(prior_out / "predictions_combined.csv")})
    text = ["# Recent-half TabPFN context experiment", "",
            f"Use only the latest {policy['retained_calendar_months']} of {policy['total_calendar_months']} training calendar months: **September 2022–May 2024**, {len(positions):,} context rows. The original 80% training set spans January 2021–May 2024 ({len(b['X_train']):,} rows). For the odd 41-month horizon, retain the middle month in full; selection is by time, not half the row count.", "",
            "Validation is unchanged: all 28,995 June 2024–March 2025 rows. Models use the original 77 features, categorical dtypes, pretrained TabPFN-3.5 Fast checkpoint, one estimator, seed 42, 200-query batches, native training-context preprocessing and no context subsampling. No parameter optimization or validation labels enter inference.", "",
            "This is a **context-only truncation experiment**. The inherited 37 history features and temporal residual baselines are preserved; observations before September 2022 can still contribute to those precomputed inputs. This does not test rebuilding every feature and baseline using recent data exclusively. Predictions fit original training residuals and restore the original query baseline exactly once.", "",
            "Pooled recent context includes all flat types. The companion per-type run routes each query to same-type recent context, continuing the previous flat-type experiment. All holdout rows are evaluated in both modes. The unlabelled competition test and existing submission artifacts remain unchanged.", "",
            "## Overall result", "", "| Model | RMSE (SGD) | MAE (SGD) | Bias (SGD) | Delta RMSE vs full pooled |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in scores.itertuples():
        text.append(f"| {row.model} | {row.rmse_sgd:.3f} | {row.mae_sgd:.3f} | {row.bias_sgd:.3f} | {row.delta_rmse_vs_full_pooled_sgd:+.3f} |")
    text += ["", "## Per-flat-type RMSE", "", "| Flat type | Recent context | Validation | Full pooled | Recent pooled | Full separate | Recent separate |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for kind, (a, q) in partitions.items():
        values = grouped.loc[grouped.flat_type.eq(kind)].set_index("model")
        text.append(f"| {kind} | {len(a)} | {len(q)} | {values.loc['full_pooled','rmse_sgd']:.3f} | {values.loc['recent_pooled','rmse_sgd']:.3f} | {values.loc['full_by_flat_type','rmse_sgd']:.3f} | {values.loc['recent_by_flat_type','rmse_sgd']:.3f} |")
    text += ["", "## Runtime", "", "| Recent-context mode | Fit seconds | Prediction seconds | Peak allocated VRAM (GiB) |", "| --- | ---: | ---: | ---: |"]
    for name, timing in timings.items():
        text.append(f"| {name} | {timing['fit_seconds']:.2f} | {timing['predict_seconds']:.2f} | {timing['peak_vram_gib']:.3f} |")
    text += ["", "Per-type times sum the six runs; device memory reports the largest per-process PyTorch allocation. Process startup and data loading are excluded. These are development-holdout results without uncertainty intervals or significance claims. One-room recent context contains only 17 observations and its validation has only 10 observations.", "",
             "Predictions for every validation row and all four protocols are saved to `predictions_combined_comparison.csv`. Individual recent-context files are `predictions_recent_pooled.csv` and `predictions_recent_type_<flat_type>.csv`.", "",
             "Audit: [settings.json](settings.json), [scores.csv](scores.csv), [scores_by_flat_type.csv](scores_by_flat_type.csv), [monthly_scores.csv](monthly_scores.csv), [category_counts.csv](category_counts.csv), [verification.json](verification.json).", "",
             "Reproduce/resume: `.venv/Scripts/python.exe analysis/compare_tabpfn_recent_half.py --stage all`. Regenerate summary only with `--stage summarize`."]
    (OUT / "report.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(scores.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["inventory", "one", "all", "summarize"], default="all")
    parser.add_argument("--mode", choices=["pooled", "by-flat-type"], default="pooled")
    parser.add_argument("--flat-type")
    args = parser.parse_args()
    if args.stage == "one":
        if args.mode == "by-flat-type" and not args.flat_type:
            parser.error("--flat-type is required for by-flat-type mode")
        run_one(args.mode, args.flat_type)
    elif args.stage == "summarize":
        summarize()
    else:
        _, _, _, partitions, policy = load()
        if args.stage == "inventory":
            print(json.dumps(policy, indent=2))
            print((OUT / "category_counts.csv").read_text())
            return
        subprocess.run([sys.executable, str(Path(__file__)), "--stage", "one", "--mode", "pooled"], cwd=ROOT, check=True)
        for kind in sorted(partitions, key=lambda k: len(partitions[k][0])):
            subprocess.run([sys.executable, str(Path(__file__)), "--stage", "one", "--mode", "by-flat-type", "--flat-type", kind], cwd=ROOT, check=True)
        summarize()


if __name__ == "__main__":
    main()
