"""Fit independent frozen TabPFN contexts per town on the 77-feature 80/20 split."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from analysis.config import ROOT
from analysis.data import sha256
from analysis.evaluation import metrics
from analysis.compare_local_economy_landscape import frozen_predict, hashlib_array, immutable, write_json
from analysis.compare_tabpfn_v3_5_fast_80_20 import CHECKPOINT


NAME = "tabpfn_by_town_77_80_20"
OUT, CACHE = ROOT / "reports" / NAME, ROOT / ".cache" / NAME
CONFIG = ROOT / "analysis/configs/local_economy_landscape.json"
MIN_FREE_MIB = 8_000


def load():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    source = ROOT / cfg["baseline"]
    if sha256(source) != cfg["baseline_sha256"]:
        raise ValueError("77-feature source bundle changed")
    bundle = joblib.load(source)
    train, query = bundle["X_train"], bundle["X_query"]
    assert train.shape == (121005, 77) and query.shape == (28995, 77)
    if not train.columns.equals(query.columns) or not train.index.is_unique or not query.index.is_unique:
        raise ValueError("Invalid train/query feature schemas or row IDs")
    if len(train.index.intersection(query.index)):
        raise ValueError("Context/query row IDs overlap")
    if pd.PeriodIndex(bundle["train_month"], freq="M").max() >= pd.PeriodIndex(bundle["query_month"], freq="M").min():
        raise ValueError("Training and validation months overlap")
    train_town, query_town = train.TOWN_CLEAN.astype("string"), query.TOWN_CLEAN.astype("string")
    if train_town.isna().any() or query_town.isna().any():
        raise ValueError("Missing town prevents complete partitioning")
    partitions = {}
    for town in sorted(query_town.unique()):
        context = np.flatnonzero(train_town.eq(town).to_numpy())
        queries = np.flatnonzero(query_town.eq(town).to_numpy())
        if not len(context):
            raise ValueError(f"No context for {town}")
        partitions[str(town)] = (context, queries)
    np.testing.assert_array_equal(np.sort(np.concatenate([x for x, _ in partitions.values()])), np.arange(len(train)))
    np.testing.assert_array_equal(np.sort(np.concatenate([q for _, q in partitions.values()])), np.arange(len(query)))

    pooled_path = ROOT / cfg["baseline_predictions"]
    parent = ROOT / "reports" / cfg["experiment"]
    pooled_hash = json.loads((parent / "settings.json").read_text(encoding="utf-8"))["baseline_predictions_sha256"]
    if sha256(pooled_path) != pooled_hash:
        raise ValueError("Pooled baseline predictions changed")
    pooled = pd.read_csv(pooled_path).set_index("row_id")
    if not pooled.index.is_unique or set(pooled.index) != set(query.index):
        raise ValueError("Pooled prediction row IDs differ")
    pooled = pooled.loc[query.index]
    np.testing.assert_array_equal(pooled.actual, np.asarray(bundle["y_query"]))

    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    immutable(OUT / "settings.json", {
        "source": str(source.relative_to(ROOT)), "source_sha256": cfg["baseline_sha256"],
        "pooled_predictions_sha256": pooled_hash, "checkpoint_sha256": sha256(CHECKPOINT),
        "code_sha256": sha256(Path(__file__)),
        "packages": {p: version(p) for p in ("tabpfn", "torch", "numpy", "pandas")},
        "grouping": "TOWN_CLEAN; each query uses only context rows from the same town",
        "feature_policy": "Unchanged 77 columns, causal histories and residual baseline from the pooled bundle",
        "inference": {"checkpoint": "TabPFN-3.5 Fast", "n_estimators": 1, "random_state": 42,
                      "batch_size": 200, "subsample_samples": None, "parameter_optimization": False},
        "split": "Chronological January 2021-May 2024 context; June 2024-March 2025 validation",
        "unlabelled_test_used": False,
    })
    pd.DataFrame([{"town": t, "context_rows": len(c), "query_rows": len(q)}
                  for t, (c, q) in partitions.items()]).to_csv(OUT / "town_counts.csv", index=False)
    return bundle, pooled, partitions


def slug(town):
    return town.replace("/", "_").replace(" ", "_")


def run_one(town):
    bundle, pooled, partitions = load()
    context, queries = partitions[town]
    train = bundle["X_train"].iloc[context]
    query = bundle["X_query"].iloc[queries]
    name = slug(town)
    path, timing_path = OUT / f"predictions_{name}.csv", OUT / f"timing_{name}.json"
    immutable(OUT / f"job_{name}.json", {
        "town": town, "context_rows": len(context), "query_rows": len(queries),
        "context_ids_sha256": hashlib_array(train.index), "query_ids_sha256": hashlib_array(query.index),
        "settings_sha256": sha256(OUT / "settings.json"),
    })
    if path.exists() and timing_path.exists():
        saved = pd.read_csv(path)
        np.testing.assert_array_equal(saved.row_id, query.index)
        np.testing.assert_array_equal(saved.actual, np.asarray(bundle["y_query"])[queries])
        assert np.isfinite(saved.predicted).all()
        print(f"Reused {town}", flush=True)
        return
    print(f"{town}: {len(context):,} context, {len(queries):,} validation rows", flush=True)
    target = np.asarray(bundle["y_train"], float)[context] - np.asarray(bundle["train_baseline"])[context]
    residual, timing = frozen_predict(train, target, query, CACHE / f"partial_{name}.npy")
    predicted = residual + np.asarray(bundle["query_baseline"])[queries]
    frame = pd.DataFrame({"row_id": query.index, "town": town,
                          "month": np.asarray(bundle["query_month"])[queries],
                          "actual": np.asarray(bundle["y_query"])[queries], "predicted": predicted,
                          "pooled_predicted": pooled.loc[query.index].predicted.to_numpy()})
    frame.to_csv(path, index=False, float_format="%.10f")
    timing.update(town=town, **metrics(frame.actual, frame.predicted))
    write_json(timing_path, timing)
    print(json.dumps(timing), flush=True)


def summarize():
    bundle, pooled, partitions = load()
    frames, rows, timings = [], [], []
    for town, (context, queries) in partitions.items():
        name = slug(town)
        frame = pd.read_csv(OUT / f"predictions_{name}.csv")
        np.testing.assert_array_equal(frame.row_id, bundle["X_query"].index[queries])
        np.testing.assert_array_equal(frame.actual, np.asarray(bundle["y_query"])[queries])
        np.testing.assert_allclose(frame.pooled_predicted, pooled.loc[frame.row_id].predicted, rtol=0, atol=1e-8)
        if not frame.town.eq(town).all() or not np.isfinite(frame.predicted).all():
            raise ValueError(f"Invalid predictions for {town}")
        frames.append(frame)
        timings.append(json.loads((OUT / f"timing_{name}.json").read_text(encoding="utf-8")))
        for model, column in (("pooled_77", "pooled_predicted"), ("town_context_77", "predicted")):
            rows.append({"town": town, "model": model, "context_rows": len(context) if column == "predicted" else len(bundle["X_train"]),
                         "query_rows": len(frame), **metrics(frame.actual, frame[column])})
    combined = pd.concat(frames, ignore_index=True).set_index("row_id")
    if not combined.index.is_unique or set(combined.index) != set(bundle["X_query"].index):
        raise ValueError("Town predictions do not cover every query exactly once")
    combined = combined.loc[bundle["X_query"].index].reset_index(names="row_id")
    combined.to_csv(OUT / "predictions_combined.csv", index=False, float_format="%.10f")
    scores = pd.DataFrame([{"model": model, "query_rows": len(combined), **metrics(combined.actual, combined[column])}
                           for model, column in (("pooled_77", "pooled_predicted"), ("town_context_77", "predicted"))])
    scores["delta_rmse_sgd"] = scores.rmse_sgd - scores.rmse_sgd.iloc[0]
    scores.to_csv(OUT / "scores.csv", index=False)
    grouped = pd.DataFrame(rows)
    grouped.to_csv(OUT / "scores_by_town.csv", index=False)
    monthly = pd.DataFrame([{"town": town, "month": month, "model": model, "query_rows": len(part),
                             **metrics(part.actual, part[column])}
                            for (town, month), part in combined.groupby(["town", "month"])
                            for model, column in (("pooled_77", "pooled_predicted"), ("town_context_77", "predicted"))])
    monthly.to_csv(OUT / "monthly_scores_by_town.csv", index=False)
    write_json(OUT / "verification.json", {
        "query_rows": len(combined), "unique_query_ids": int(combined.row_id.nunique()),
        "context_rows_partitioned": sum(len(c) for c, _ in partitions.values()),
        "all_contexts_unsubsampled": all(not t["subsampling"] for t in timings),
        "all_parameter_versions_unchanged": all(t["parameter_versions_unchanged"] for t in timings),
        "prediction_files_sha256": {town: sha256(OUT / f"predictions_{slug(town)}.csv") for town in partitions},
    })
    total_fit = sum(t["fit_seconds"] for t in timings)
    total_predict = sum(t["predict_seconds"] for t in timings)
    peak = max(t["peak_vram_gib"] for t in timings)
    six = ("bukit panjang", "choa chu kang", "hougang", "punggol", "sembawang", "sengkang")
    lines = ["# Town-specific TabPFN on the chronological 80/20 holdout", "",
             "Each of 26 towns is fitted separately using its January 2021-May 2024 labeled rentals as context; all June 2024-March 2025 rentals are inferred. The pretrained TabPFN-3.5 Fast weights are frozen: `fit` supplies town-specific labeled context, not gradient training. One estimator, seed 42, no context subsampling, 77 unchanged features, causal rental histories and original temporal residual baseline. The pooled control is the existing same-feature one-estimator run. This is a previously examined development split.", "",
             "## Overall performance", "", "| Model | Rows | RMSE (S$) | MAE (S$) | Bias (S$) | Δ RMSE (S$) |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in scores.itertuples():
        lines.append(f"| {row.model} | {row.query_rows:,} | {row.rmse_sgd:.3f} | {row.mae_sgd:.3f} | {row.bias_sgd:+.3f} | {row.delta_rmse_sgd:+.3f} |")
    lines += ["", "## Per-town performance", "",
              "Positive bias means overprediction. Δ RMSE is town-context minus pooled.", "",
              "| Town | Context | Validation | Pooled RMSE | Town RMSE | Δ RMSE | Pooled bias | Town bias |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for town, (context, queries) in partitions.items():
        pair = grouped.loc[grouped.town.eq(town)].set_index("model")
        p, t = pair.loc["pooled_77"], pair.loc["town_context_77"]
        lines.append(f"| {town} | {len(context):,} | {len(queries):,} | {p.rmse_sgd:.2f} | {t.rmse_sgd:.2f} | {t.rmse_sgd-p.rmse_sgd:+.2f} | {p.bias_sgd:+.2f} | {t.bias_sgd:+.2f} |")
    six_part = combined.loc[combined.town.isin(six)]
    lines += ["", "## Six persistent Plus overprediction towns", "",
              "These towns were identified using the earlier one-estimator Plus analysis, so this subgroup is descriptive rather than an independent selection test.", "",
              "| Model | Rows | RMSE (S$) | MAE (S$) | Bias (S$) |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for model, column in (("pooled_77", "pooled_predicted"), ("town_context_77", "predicted")):
        s = metrics(six_part.actual, six_part[column])
        lines.append(f"| {model} | {len(six_part):,} | {s['rmse_sgd']:.3f} | {s['mae_sgd']:.3f} | {s['bias_sgd']:+.3f} |")
    lines += ["", f"Runtime: {total_fit:.1f} s fit plus {total_predict:.1f} s inference; maximum PyTorch peak allocation {peak:.2f} GiB. Process startup and feature loading are excluded.", "",
              "The town context changes which rentals TabPFN sees and refits its native preprocessing. The 77 features and historical rental baseline were constructed globally and remain unchanged. No town-specific weight fine-tuning or hyperparameter selection was performed.", "",
              "Artifacts: [scores.csv](scores.csv), [scores_by_town.csv](scores_by_town.csv), [monthly_scores_by_town.csv](monthly_scores_by_town.csv), [predictions_combined.csv](predictions_combined.csv), [verification.json](verification.json).", "",
              "Reproduce: `.venv/Scripts/python.exe analysis/compare_tabpfn_by_town.py --stage all`. Resume is supported per town."]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(scores.to_string(index=False), flush=True)


def wait_for_gpu():
    while True:
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True, check=True)
            free = int(result.stdout.strip().splitlines()[0])
        except (OSError, subprocess.CalledProcessError, ValueError) as exc:
            raise RuntimeError("Cannot verify GPU capacity") from exc
        if free >= MIN_FREE_MIB:
            return
        print(f"GPU free memory {free:,} MiB below {MIN_FREE_MIB:,} MiB; waiting 30 s", flush=True)
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("inventory", "one", "all", "summarize"), default="all")
    parser.add_argument("--town")
    args = parser.parse_args()
    if args.stage == "one":
        if not args.town:
            parser.error("--town is required for --stage one")
        wait_for_gpu()
        run_one(args.town)
    elif args.stage == "summarize":
        summarize()
    else:
        _, _, partitions = load()
        if args.stage == "inventory":
            print((OUT / "town_counts.csv").read_text(encoding="utf-8"))
            return
        for town in sorted(partitions, key=lambda t: len(partitions[t][0])):
            subprocess.run([sys.executable, str(Path(__file__)), "--stage", "one", "--town", town], cwd=ROOT, check=True)
        summarize()


if __name__ == "__main__":
    main()
