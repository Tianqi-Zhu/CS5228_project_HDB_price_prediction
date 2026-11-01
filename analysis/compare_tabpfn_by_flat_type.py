"""Independent frozen TabPFN contexts per flat type on the 77-feature holdout."""
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
from analysis.compare_local_economy_landscape import frozen_predict, hashlib_array, immutable, write_json
from analysis.compare_tabpfn_v3_5_fast_80_20 import CHECKPOINT
from analysis.experiments.flat_type_inference import flat_type_partitions, assemble_predictions

NAME = "tabpfn_by_flat_type_77_80_20"
OUT, CACHE = ROOT / "reports" / NAME, ROOT / ".cache" / NAME
CONFIG = ROOT / "analysis/configs/local_economy_landscape.json"


def load():
    cfg = json.loads(CONFIG.read_text())
    path = ROOT / cfg["baseline"]
    if sha256(path) != cfg["baseline_sha256"]:
        raise ValueError("77-feature source bundle changed")
    b = joblib.load(path)
    assert b["X_train"].shape == (121005, 77) and b["X_query"].shape == (28995, 77)
    predictions = ROOT / cfg["baseline_predictions"]
    parent = ROOT / "reports" / cfg["experiment"]
    expected = json.loads((parent / "settings.json").read_text())["baseline_predictions_sha256"]
    if sha256(predictions) != expected:
        raise ValueError("Frozen pooled baseline predictions changed")
    reference = pd.read_csv(predictions).set_index("row_id")
    if not reference.index.is_unique or len(reference) != len(b["X_query"]):
        raise ValueError("Baseline query IDs differ")
    reference = reference.loc[b["X_query"].index]
    np.testing.assert_array_equal(reference.actual, b["y_query"])
    partitions = flat_type_partitions(b["X_train"], b["X_query"])
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    immutable(OUT / "settings.json", {
        "baseline_bundle": str(path.relative_to(ROOT)), "baseline_sha256": cfg["baseline_sha256"],
        "baseline_predictions_sha256": expected, "checkpoint_sha256": sha256(CHECKPOINT),
        "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in (Path(__file__), ROOT / "analysis/experiments/flat_type_inference.py", ROOT / "analysis/compare_local_economy_landscape.py")},
        "packages": {p: version(p) for p in ("tabpfn", "torch", "numpy", "pandas")},
        "grouping": "FLAT_TYPE_CLEAN; route every query to same-type context; no cross-type TabPFN context",
        "feature_policy": "Original 77 columns, categorical dtypes, history values and temporal residual baselines unchanged; no tree features",
        "preprocessing": "Native checkpoint preprocessing fitted independently within each type; no external scaler or imputer",
        "inference": {"n_estimators": 1, "random_state": 42, "batch_size": 200, "subsample_samples": None, "parameter_optimization": False},
        "unlabelled_test_used": False})
    counts = [{"flat_type": kind, "context_rows": len(a), "query_rows": len(q)} for kind, (a, q) in partitions.items()]
    pd.DataFrame(counts).to_csv(OUT / "category_counts.csv", index=False)
    return b, reference, partitions


def run_one(kind):
    b, reference, partitions = load()
    context, queries = partitions[kind]
    x, q = b["X_train"].iloc[context], b["X_query"].iloc[queries]
    assert x.FLAT_TYPE_CLEAN.astype(str).eq(kind).all() and q.FLAT_TYPE_CLEAN.astype(str).eq(kind).all()
    slug = kind.replace("-", "_")
    path, timing_path = OUT / f"predictions_{slug}.csv", OUT / f"timing_{slug}.json"
    immutable(OUT / f"job_{slug}.json", {"flat_type": kind, "context_rows": len(x), "query_rows": len(q),
              "context_ids_sha256": hashlib_array(x.index), "query_ids_sha256": hashlib_array(q.index),
              "input_columns": list(x.columns), "settings_sha256": sha256(OUT / "settings.json")})
    if path.exists() and timing_path.exists():
        saved = pd.read_csv(path)
        np.testing.assert_array_equal(saved.row_id, q.index)
        np.testing.assert_array_equal(saved.actual, np.asarray(b["y_query"])[queries])
        assert np.isfinite(saved.predicted).all()
        print(f"Reused {kind}", flush=True)
        return
    print(f"{kind}: {len(x):,} context, {len(q):,} queries, {x.shape[1]} input features", flush=True)
    y = np.asarray(b["y_train"], float)[context] - np.asarray(b["train_baseline"])[context]
    residual, timing = frozen_predict(x, y, q, CACHE / f"partial_{slug}.npy")
    prediction = residual + np.asarray(b["query_baseline"])[queries]
    frame = pd.DataFrame({"row_id": q.index, "flat_type": kind, "month": np.asarray(b["query_month"])[queries],
                          "actual": np.asarray(b["y_query"])[queries], "predicted": prediction,
                          "pooled_predicted": reference.loc[q.index].predicted.to_numpy()})
    frame.to_csv(path, index=False, float_format="%.10f")
    timing.update(flat_type=kind, **metrics(frame.actual, frame.predicted))
    write_json(timing_path, timing)
    print(json.dumps(timing), flush=True)


def summarize():
    b, reference, partitions = load()
    frames, timings, rows = [], [], []
    for kind, (context, query) in partitions.items():
        slug = kind.replace("-", "_")
        frame = pd.read_csv(OUT / f"predictions_{slug}.csv")
        np.testing.assert_array_equal(frame.row_id, b["X_query"].index[query])
        np.testing.assert_array_equal(frame.actual, np.asarray(b["y_query"])[query])
        np.testing.assert_array_equal(frame.month.astype(str), np.asarray(b["query_month"])[query].astype(str))
        np.testing.assert_allclose(frame.pooled_predicted, reference.loc[frame.row_id].predicted, rtol=0, atol=1e-8)
        timings.append(json.loads((OUT / f"timing_{slug}.json").read_text()))
        frames.append(frame)
        for model, column in (("pooled_77", "pooled_predicted"), ("separate_flat_type_77", "predicted")):
            rows.append({"flat_type": kind, "model": model, "context_rows": len(context) if column == "predicted" else len(b["X_train"]),
                         "query_rows": len(frame), **metrics(frame.actual, frame[column])})
    combined = assemble_predictions(b["X_query"], frames)
    combined.to_csv(OUT / "predictions_combined.csv", index=False, float_format="%.10f")
    scores = []
    for model, column in (("pooled_77", "pooled_predicted"), ("separate_flat_type_77", "predicted")):
        scores.append({"model": model, "query_rows": len(combined), **metrics(combined.actual, combined[column])})
    score = pd.DataFrame(scores)
    score["delta_rmse_sgd"] = score.rmse_sgd - score.rmse_sgd.iloc[0]
    score.to_csv(OUT / "scores.csv", index=False)
    grouped = pd.DataFrame(rows)
    grouped.to_csv(OUT / "scores_by_flat_type.csv", index=False)
    monthly = []
    for month, part in combined.groupby("month"):
        for model, column in (("pooled_77", "pooled_predicted"), ("separate_flat_type_77", "predicted")):
            monthly.append({"month": month, "model": model, "query_rows": len(part), **metrics(part.actual, part[column])})
    pd.DataFrame(monthly).to_csv(OUT / "monthly_scores.csv", index=False)
    write_json(OUT / "verification.json", {"query_rows": len(combined), "unique_query_ids": int(combined.row_id.nunique()),
               "context_rows_partitioned": sum(len(a) for a, _ in partitions.values()), "all_predictions_finite": True,
               "query_order_preserved": True, "all_labels_and_months_match": True,
               "all_contexts_same_flat_type": True, "all_contexts_unsubsampled": all(not t["subsampling"] for t in timings),
               "all_parameter_versions_unchanged": all(t["parameter_versions_unchanged"] for t in timings),
               "prediction_files_sha256": {kind: sha256(OUT / f"predictions_{kind.replace('-', '_')}.csv") for kind in partitions}})
    totals = {"fit_seconds": sum(t["fit_seconds"] for t in timings), "predict_seconds": sum(t["predict_seconds"] for t in timings),
              "max_peak_vram_gib": max(t["peak_vram_gib"] for t in timings)}
    write_json(OUT / "timing_summary.json", totals)
    report = ["# Separate TabPFN inference by flat type", "", "All six flat types are inferred independently using only their same-type context rows. The complete chronological 80/20 holdout is retained.", "",
              "Original 77 inputs, their categorical dtypes, pre-origin histories and temporal residual baselines are unchanged. This isolates context partitioning; feature histories and the temporal baseline were not rebuilt separately. No tree-cover additions, external scaling, parameter optimization, query-label use or category selection is introduced. Each fresh process uses the same pretrained TabPFN-3.5 Fast checkpoint, one estimator, seed 42, native within-context preprocessing, 200-query batches and every available context row of its flat type.", "",
              "Context: 121,005 January 2021–May 2024 rows across the six categories. Query: every one of 28,995 June 2024–March 2025 rows. This is the previously examined development holdout, not an untouched test. The unlabelled competition test is not inferred.", "",
              "## Combined result", "", "| Model | Queries | RMSE (SGD) | MAE (SGD) | Bias (SGD) | Delta RMSE (SGD) |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in score.itertuples():
        report.append(f"| {row.model} | {row.query_rows} | {row.rmse_sgd:.3f} | {row.mae_sgd:.3f} | {row.bias_sgd:.3f} | {row.delta_rmse_sgd:+.3f} |")
    report += ["", "## Per-flat-type result", "", "| Flat type | Context | Queries | Pooled RMSE | Separate RMSE | Delta RMSE | Pooled MAE | Separate MAE |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for kind, (context, query) in partitions.items():
        subset = grouped.loc[grouped.flat_type.eq(kind)].set_index("model")
        p, s = subset.loc["pooled_77"], subset.loc["separate_flat_type_77"]
        report.append(f"| {kind} | {len(context)} | {len(query)} | {p.rmse_sgd:.3f} | {s.rmse_sgd:.3f} | {s.rmse_sgd-p.rmse_sgd:+.3f} | {p.mae_sgd:.3f} | {s.mae_sgd:.3f} |")
    improved = score.rmse_sgd.iloc[1] < score.rmse_sgd.iloc[0]
    report += ["", "The separate-context model " + ("improves" if improved else "does not improve") + " pooled holdout RMSE. No significance claim or uncertainty interval is attached to this comparison. One-room results have only 32 context rows and 10 query rows and should not be treated as a stable estimate.", "",
               f"Total fit time: {totals['fit_seconds']:.2f} s; total prediction time: {totals['predict_seconds']:.2f} s; maximum category peak PyTorch-allocated VRAM: {totals['max_peak_vram_gib']:.3f} GiB. Fresh-process startup and input-loading time are excluded.", "",
               "Inference results: `predictions_combined.csv` and one `predictions_<flat_type>.csv` per category in this report directory. These contain row IDs, months, actual rents, separate predictions and pooled-model reference predictions, all in SGD. Row-level CSVs are reproducible local artifacts and excluded from Git.", "",
               "Audit: [settings.json](settings.json), [category_counts.csv](category_counts.csv), [scores.csv](scores.csv), [scores_by_flat_type.csv](scores_by_flat_type.csv), [monthly_scores.csv](monthly_scores.csv), [verification.json](verification.json).", "",
               "Reproduce/resume: `.venv/Scripts/python.exe analysis/compare_tabpfn_by_flat_type.py --stage all`. Summarize existing completed runs with `--stage summarize`."]
    (OUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(score.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["inventory", "one", "all", "summarize"], default="all")
    parser.add_argument("--flat-type")
    args = parser.parse_args()
    if args.stage == "one":
        if not args.flat_type:
            parser.error("--flat-type is required for --stage one")
        run_one(args.flat_type)
    elif args.stage == "summarize":
        summarize()
    else:
        _, _, partitions = load()
        if args.stage == "inventory":
            print((OUT / "category_counts.csv").read_text())
            return
        # Finish small categories first, with one isolated GPU job at a time.
        for kind in sorted(partitions, key=lambda k: len(partitions[k][0])):
            subprocess.run([sys.executable, str(Path(__file__)), "--stage", "one", "--flat-type", kind], cwd=ROOT, check=True)
        summarize()


if __name__ == "__main__":
    main()
