"""Predeclared census regional ablation with archived planning-area income bands.

Independent extension preserves the already locked economic/park factorial.
It never modifies that feature cache, selection, or existing prediction files.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from analysis.config import ROOT
from analysis.data import sha256
from analysis.evaluation import metrics
from analysis.compare_local_economy_landscape import paths, DEFAULT_CONFIG, write_json, immutable, frozen_predict
from analysis.experiments.regional_income_features import income_features

NAME = "tabpfn_local_economy_landscape_80_20_income"
OUT, CACHE = ROOT / "reports" / NAME, ROOT / ".cache" / NAME


def prepare():
    cfg, parent_out, parent_cache = paths(DEFAULT_CONFIG)
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((OUT / "source_manifest.json").read_text())
    source = ROOT / manifest["income2022"]["path"]
    assert sha256(source) == manifest["income2022"]["sha256"]
    assert sha256(parent_cache / "prepared.joblib") == json.loads((parent_out / "prepared_manifest.json").read_text())["sha256"]
    immutable(OUT / "settings.json", {"source": manifest, "parent_prepared_sha256": sha256(parent_cache / "prepared.joblib"),
              "available_at": "2022-08-01", "feature_block": "Published 2020 population/age + three planning-area household-income-band proportions + availability flags",
              "selection": "predeclared regional ablation, one fixed representation; does not replace or retune the locked factorial",
              "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in (Path(__file__), ROOT / "analysis/experiments/regional_income_features.py")}})
    path = CACHE / "features.joblib"
    if path.exists():
        assert sha256(path) == json.loads((OUT / "feature_manifest.json").read_text())["sha256"]
        return
    table = pd.read_csv(source)
    data = joblib.load(parent_cache / "prepared.joblib")
    base = data["outer"]["bundle"]
    additions, coverage = {}, []
    for stage, entry in data.items():
        additions[stage] = {}
        for split in ("train", "query"):
            x = entry["bundle"][f"X_{split}"]
            if split == "train":
                origin = base["train_temporal_origins"].reindex(x.index).forecast_origin
            else:
                origin = pd.Series(pd.PeriodIndex(entry["bundle"]["query_month"], freq="M").min() - 1, index=x.index)
            income = income_features(x.PLANNING_AREA.astype(str), origin, table)
            regional = entry["added"][split][["CENSUS2020_POPULATION", "CENSUS2020_AGE65_SHARE", "CENSUS2020_UNAVAILABLE"]]
            additions[stage][split] = regional.join(income)
            income.to_csv(CACHE / f"income_{stage}_{split}.csv")
            for column in income:
                coverage.append({"stage": stage, "split": split, "feature": column, "rows": len(income), "missing_fraction": float(income[column].isna().mean())})
    joblib.dump(additions, path, compress=3)
    write_json(OUT / "feature_manifest.json", {"sha256": sha256(path), "columns": list(additions["outer"]["train"]),
               "geographic_support": "planning area, assigned using existing polygon-audited labels; no interpolation",
               "income_source": manifest["income2022"]["source_url"], "availability": "2022-08-01, conservative next-month cutoff after the source's July 28 2022 metadata date"})
    pd.DataFrame(coverage).to_csv(OUT / "feature_coverage.csv", index=False)
    print("Prepared the separate 7-column regional income ablation", flush=True)


def run(stage):
    cfg, parent_out, parent_cache = paths(DEFAULT_CONFIG)
    prepare()
    parent = joblib.load(parent_cache / "prepared.joblib")[stage]
    extra = joblib.load(CACHE / "features.joblib")[stage]
    b = parent["bundle"]
    x, q = [b[f"X_{split}"].join(extra[split]) for split in ("train", "query")]
    for matrix, split in ((x, "train"), (q, "query")):
        pd.testing.assert_frame_equal(matrix[b[f"X_{split}"].columns], b[f"X_{split}"])
    path = OUT / f"predictions_{stage}.csv"
    timing_path = OUT / f"timing_{stage}.json"
    if not path.exists():
        print(f"Income {stage}: {len(x):,} full context; {len(q):,} queries", flush=True)
        residual, timing = frozen_predict(x, np.asarray(b["y_train"]) - b["train_baseline"], q, CACHE / f"partial_{stage}.npy")
        prediction = residual + b["query_baseline"]
        result = pd.DataFrame({"row_id": q.index, "month": b["query_month"], "actual": b["y_query"], "predicted": prediction,
                               "weight": b.get("query_weights", np.ones(len(q)))})
        result["error"] = result.predicted - result.actual
        result["absolute_error"] = result.error.abs()
        result.to_csv(path, index=False, float_format="%.10f")
        write_json(timing_path, {**timing, "added_dimensions": 7, **metrics(result.actual, result.predicted)})
    f = pd.read_csv(path)
    np.testing.assert_array_equal(f.row_id, q.index)
    np.testing.assert_array_equal(f.actual, b["y_query"])
    if stage == "outer":
        for original, destination in ((path, parent_out / "predictions_outer_regional_income.csv"), (timing_path, parent_out / "timing_outer_regional_income.json")):
            if destination.exists():
                assert sha256(destination) == sha256(original)
            else:
                shutil.copy2(original, destination)
        write_json(parent_out / "income_extension.json", {"directory": str(OUT.relative_to(ROOT)), "features_sha256": sha256(CACHE / "features.joblib"),
                   "source_sha256": json.loads((OUT / "source_manifest.json").read_text())["income2022"]["sha256"], "separate_predeclared_regional_ablation": True})
    print(stage, metrics(f.actual, f.predicted), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "one", "all"], default="all")
    parser.add_argument("--fold", default="outer")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "one":
        run(args.fold)
    else:
        for stage in ("inner_2022-09", "inner_2023-07", "outer"):
            subprocess.run([sys.executable, str(Path(__file__)), "--stage", "one", "--fold", stage], check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
