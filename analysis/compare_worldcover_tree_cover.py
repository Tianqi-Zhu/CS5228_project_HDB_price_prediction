"""Append published WorldCover tree fractions to the frozen 77-feature TabPFN.

Default prepares inputs only. Smoke verifies full-context inference on 200
queries. Screen and outer explicitly run the three predeclared ablations.
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
from analysis.compare_local_economy_landscape import write_json, immutable, frozen_predict, hashlib_array
from analysis.compare_tabpfn_v3_5_fast_80_20 import CHECKPOINT
from analysis.prepare_worldcover_tree_cover import CONFIG, locations
from analysis.experiments.worldcover_tree_cover import origin_features

VARIANTS = {
    "baseline": [],
    "tree_100m": ["TREE_CLASS_FRACTION_100M", "TREE_COVER_UNAVAILABLE"],
    "tree_300m": ["TREE_CLASS_FRACTION_300M", "TREE_COVER_UNAVAILABLE"],
    "tree_100m_300m": ["TREE_CLASS_FRACTION_100M", "TREE_CLASS_FRACTION_300M", "TREE_COVER_UNAVAILABLE"],
}


def parent_paths(cfg):
    return ROOT / "reports" / cfg["parent_experiment"], ROOT / ".cache" / cfg["parent_experiment"] / "prepared.joblib"


def prepare():
    cfg, out, cache = locations()
    parent_out, parent_path = parent_paths(cfg)
    expected = json.loads((parent_out / "prepared_manifest.json").read_text())["sha256"]
    if sha256(parent_path) != expected:
        raise ValueError("Parent feature bundle changed")
    extraction = json.loads((out / "extraction_manifest.json").read_text())
    products = {}
    for year, item in extraction["outputs"].items():
        path = ROOT / item["path"]
        if sha256(path) != item["sha256"]:
            raise ValueError("Extracted tree features changed")
        products[int(year)] = pd.read_csv(path)
    links_path = parent_out / "transaction_building_links.csv"
    signatures = {"config_sha256": sha256(CONFIG), "parent_prepared_sha256": expected,
                  "extraction_manifest_sha256": sha256(out / "extraction_manifest.json"),
                  "links_sha256": sha256(links_path),
                  "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in (Path(__file__), ROOT / "analysis/experiments/worldcover_tree_cover.py")},
                  "variants": VARIANTS, "policy": "Latest published product at original pseudo-origin; conservative next-month availability; class 10 only",
                  "packages": {p: version(p) for p in ("rasterio", "pyproj", "numpy", "pandas", "tabpfn")}}
    immutable(out / "settings.json", signatures)
    path = cache / "features.joblib"
    if path.exists():
        if sha256(path) != json.loads((out / "feature_manifest.json").read_text())["sha256"]:
            raise ValueError("Prepared tree features changed")
        print("Reused verified origin-aligned features", flush=True)
        return
    links = pd.read_csv(links_path).set_index("row_id")
    if not links.index.is_unique:
        raise ValueError("Ambiguous transaction-building links")
    data = joblib.load(parent_path)
    base = data["outer"]["bundle"]
    additions, coverage, vintage = {}, [], []
    for stage, entry in data.items():
        additions[stage] = {}
        for split in ("train", "query"):
            x = entry["bundle"][f"X_{split}"]
            ids = links.reindex(x.index).building_id
            if ids.isna().any():
                raise ValueError("Unmatched building in experiment")
            if split == "train":
                origins = base["train_temporal_origins"].reindex(x.index).forecast_origin
            else:
                origins = pd.Series(pd.PeriodIndex(entry["bundle"]["query_month"], freq="M").min() - 1, index=x.index)
            fields, years = origin_features(ids, origins, products, cfg)
            if set(fields.columns) & set(x.columns):
                raise ValueError("Attempt to replace baseline columns")
            additions[stage][split] = fields
            for name in fields:
                coverage.append({"stage": stage, "split": split, "feature": name, "rows": len(fields),
                                 "missing": int(fields[name].isna().sum()), "missing_fraction": float(fields[name].isna().mean())})
            for year, count in years.astype("string").fillna("unavailable").value_counts().items():
                vintage.append({"stage": stage, "split": split, "source_year": str(year), "rows": int(count)})
            fields.to_csv(cache / f"features_{stage}_{split}.csv")
    joblib.dump(additions, path, compress=3)
    pd.DataFrame(coverage).to_csv(out / "feature_coverage.csv", index=False)
    pd.DataFrame(vintage).to_csv(out / "source_year_coverage.csv", index=False)
    write_json(out / "feature_manifest.json", {"sha256": sha256(path), "columns": list(additions["outer"]["train"]), "stages": list(additions),
               "parent_prepared_sha256": expected, "selection": "Three fixed ablations; no selection on outer labels", "test_predictions_generated": False})
    print("Prepared 100 m, 300 m and combined tree-cover ablations", flush=True)


def reference(stage, parent_out, bundle):
    if stage == "outer":
        cfg = json.loads((ROOT / "analysis/configs/local_economy_landscape.json").read_text())
        path = ROOT / cfg["baseline_predictions"]
        expected = json.loads((parent_out / "settings.json").read_text())["baseline_predictions_sha256"]
        if sha256(path) != expected:
            raise ValueError("Frozen baseline prediction file changed")
    else:
        path = parent_out / f"predictions_{stage}_baseline.csv"
    frame = pd.read_csv(path).set_index("row_id").loc[bundle["X_query"].index]
    np.testing.assert_array_equal(frame.actual, bundle["y_query"])
    return frame


def run(stage, variant, smoke=False):
    prepare()
    cfg, out, cache = locations()
    parent_out, parent_path = parent_paths(cfg)
    b = joblib.load(parent_path)[stage]["bundle"]
    additions = joblib.load(cache / "features.joblib")[stage]
    columns = VARIANTS[variant]
    x, q = [b[f"X_{split}"].join(additions[split][columns]) for split in ("train", "query")]
    for matrix, split in ((x, "train"), (q, "query")):
        pd.testing.assert_frame_equal(matrix[b[f"X_{split}"].columns], b[f"X_{split}"])
        assert len(b[f"X_{split}"].columns) == 77
    if smoke:
        q = q.iloc[:200]
    tag = f"{stage}_{variant}" + ("_smoke" if smoke else "")
    settings = {"stage": stage, "variant": variant, "smoke": smoke, "added_columns": columns,
                "context_ids_sha256": hashlib_array(x.index), "query_ids_sha256": hashlib_array(q.index),
                "features_sha256": sha256(cache / "features.joblib"), "checkpoint_sha256": sha256(CHECKPOINT),
                "settings_sha256": sha256(out / "settings.json"), "frozen_predict_code_sha256": sha256(ROOT / "analysis/compare_local_economy_landscape.py")}
    immutable(out / f"job_{tag}.json", settings)
    path, timing_path = out / f"predictions_{tag}.csv", out / f"timing_{tag}.json"
    if path.exists() and timing_path.exists():
        saved = pd.read_csv(path)
        np.testing.assert_array_equal(saved.row_id, q.index)
        np.testing.assert_array_equal(saved.actual, np.asarray(b["y_query"])[:len(q)])
        print(f"Reused {tag}", flush=True)
        return
    print(f"{tag}: {len(x):,} full context; {len(q):,} queries; +{len(columns)} features", flush=True)
    residual, timing = frozen_predict(x, np.asarray(b["y_train"], float) - b["train_baseline"], q, cache / f"partial_{tag}.npy")
    prediction = residual + np.asarray(b["query_baseline"])[:len(q)]
    if variant == "baseline":
        original = reference(stage, parent_out, b).loc[q.index]
        np.testing.assert_allclose(prediction, original.predicted, atol=1e-5, rtol=0)
        write_json(out / f"baseline_replay_{stage}.json", {"rows": len(q), "verified": True, "max_abs_difference_sgd": float(np.max(np.abs(prediction-original.predicted)))})
    frame = pd.DataFrame({"row_id": q.index, "month": np.asarray(b["query_month"])[:len(q)], "actual": np.asarray(b["y_query"])[:len(q)],
                          "predicted": prediction, "weight": np.asarray(b.get("query_weights", np.ones(len(b["X_query"]))))[:len(q)]})
    frame.to_csv(path, index=False, float_format="%.10f")
    weights, error = frame.weight.to_numpy(), frame.predicted.to_numpy() - frame.actual.to_numpy()
    timing.update(added_dimensions=len(columns), **metrics(frame.actual, frame.predicted),
                  weighted_rmse_sgd=float(np.sqrt(np.average(error**2, weights=weights))),
                  weighted_mae_sgd=float(np.average(np.abs(error), weights=weights)), smoke_only=smoke)
    write_json(timing_path, timing)
    print(json.dumps(timing), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "smoke", "screen", "outer", "all", "one"], default="prepare")
    parser.add_argument("--fold", choices=["outer", "inner_2022-09", "inner_2023-07"], default="outer")
    parser.add_argument("--variant", choices=list(VARIANTS), default="tree_100m_300m")
    args = parser.parse_args()
    if args.stage == "one":
        run(args.fold, args.variant)
        return
    prepare()
    jobs = []
    if args.stage == "smoke":
        run("outer", "baseline", smoke=True)
        run("outer", "tree_100m_300m", smoke=True)
    if args.stage in ("screen", "all"):
        jobs.extend((fold, variant) for fold in ("inner_2022-09", "inner_2023-07") for variant in VARIANTS if variant != "baseline")
    if args.stage in ("outer", "all"):
        jobs.extend(("outer", variant) for variant in VARIANTS if variant != "baseline")
    for fold, variant in jobs:
        subprocess.run([sys.executable, str(Path(__file__)), "--stage", "one", "--fold", fold, "--variant", variant], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
