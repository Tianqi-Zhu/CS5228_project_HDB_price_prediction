"""Materialize point-in-time, model-ready HDB features for TabPFN-3.

The saved pandas categoricals are inferred from fitting rows only. TabPFN
handles categorical encoding and numeric transforms internally, so this script
does not one-hot encode or normalize inputs. Load only artifacts generated in
this trusted workspace: joblib uses pickle internally.
"""

from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from analysis.config import DATA, REPORTS, ROOT
from analysis.data import sha256
from analysis.experiments.spatiotemporal_hybrid import FOLDS, MONTH, SOURCES, TARGET, spatial_frame
from analysis.experiments.tabpfn_comparison import fixed_origin_split
from analysis.experiments.tabpfn_adaptation_features import (
    CATEGORICAL, ENHANCED, backbone_inputs, enhanced_feature_pair,
)


OUT = ROOT / ".cache" / "tabpfn_v3_features"
REPORT = REPORTS / "tabpfn_v3_features"
FINAL_ORIGIN = pd.Period("2025-03", freq="M")
FORBIDDEN = {TARGET, "LOG_MONTHLY_RENT", "RENT_PER_SQM"}


def model_ready_pair(fitting: pd.DataFrame, query: pd.DataFrame):
    """Return category-typed, target-free inputs with preserved source row IDs."""
    if fitting.empty or query.empty or not (fitting[MONTH].max() < query[MONTH].min()):
        raise ValueError("All fitting labels must precede all query months")
    fit_features, query_features = enhanced_feature_pair(fitting, query)
    x_fit, x_query = backbone_inputs(fit_features, query_features, ENHANCED)
    categorical = sorted(CATEGORICAL.intersection(ENHANCED))
    if list(x_fit.columns) != ENHANCED or list(x_query.columns) != ENHANCED:
        raise AssertionError("Feature order changed")
    if FORBIDDEN.intersection(x_fit.columns) or FORBIDDEN.intersection(x_query.columns):
        raise AssertionError("Current-row rent entered a model input")
    if x_fit.index.has_duplicates or x_query.index.has_duplicates:
        raise AssertionError("Source row IDs are not unique")
    for column in ENHANCED:
        if column in categorical:
            if not isinstance(x_fit[column].dtype, pd.CategoricalDtype):
                raise AssertionError(f"{column} is not categorical")
            if x_fit[column].dtype != x_query[column].dtype:
                raise AssertionError(f"Different category mapping for {column}")
        elif not pd.api.types.is_numeric_dtype(x_fit[column].dtype) or not pd.api.types.is_numeric_dtype(x_query[column].dtype):
            raise AssertionError(f"{column} is not numeric")
    return x_fit, x_query, categorical


def make_bundle(fitting: pd.DataFrame, query: pd.DataFrame, origin: str, *, has_query_labels: bool):
    x_fit, x_query, categorical = model_ready_pair(fitting, query)
    bundle = {
        "X_train": x_fit,
        "y_train": fitting[TARGET].to_numpy(dtype=float),
        "X_query": x_query,
        "y_query": query[TARGET].to_numpy(dtype=float) if has_query_labels else None,
        "train_month": fitting[MONTH].astype(str).to_numpy(),
        "query_month": query[MONTH].astype(str).to_numpy(),
    }
    if not np.isfinite(bundle["y_train"]).all():
        raise AssertionError("Nonfinite fitting rents")
    if has_query_labels and not np.isfinite(bundle["y_query"]).all():
        raise AssertionError("Nonfinite validation rents")
    if len(x_fit.index.intersection(x_query.index)) and has_query_labels:
        raise AssertionError("Fitting and validation source rows overlap")
    audit = {
        "origin": origin,
        "train_rows": len(x_fit),
        "query_rows": len(x_query),
        "feature_count": len(ENHANCED),
        "categorical_columns": categorical,
        "numeric_columns": [column for column in ENHANCED if column not in categorical],
        "last_label_month": str(fitting[MONTH].max()),
        "first_query_month": str(query[MONTH].min()),
        "last_query_month": str(query[MONTH].max()),
        "query_categorical_missing_after_mapping": {
            column: int(x_query[column].isna().sum()) for column in categorical
        },
    }
    return bundle, audit


def write_bundle(name: str, bundle: dict, audit: dict):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.joblib"
    temporary = OUT / f"{name}.part.joblib"
    joblib.dump(bundle, temporary, compress=3)
    temporary.replace(path)
    loaded = joblib.load(path)
    for feature_name in ("X_train", "X_query"):
        pd.testing.assert_frame_equal(loaded[feature_name], bundle[feature_name])
    np.testing.assert_array_equal(loaded["y_train"], bundle["y_train"])
    if bundle["y_query"] is not None:
        np.testing.assert_array_equal(loaded["y_query"], bundle["y_query"])
    audit["artifact"] = str(path.relative_to(ROOT)).replace("\\", "/")
    audit["sha256"] = sha256(path)
    audit["bytes"] = path.stat().st_size
    print(f"{name}: {audit['train_rows']:,} train, {audit['query_rows']:,} query, "
          f"{audit['feature_count']} features, {audit['bytes'] / 1024**2:.1f} MiB", flush=True)
    return audit


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    if len(ENHANCED) != len(set(ENHANCED)):
        raise AssertionError("Duplicate feature names")
    train = spatial_frame(pd.read_csv(DATA / "train.csv"))
    tests = spatial_frame(pd.read_csv(DATA / "test.csv"))
    source_hashes = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
                     for path in SOURCES}
    code_paths = [Path(__file__), ROOT / "analysis/data.py",
                  ROOT / "analysis/experiments/spatiotemporal_hybrid.py",
                  ROOT / "analysis/experiments/recent_history.py",
                  ROOT / "analysis/experiments/tabpfn_adaptation_features.py"]
    code_hashes = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
                   for path in code_paths}
    audits = []
    for start, end in FOLDS:
        fitting, query = fixed_origin_split(train, start, end)
        print(f"Preparing fold {start}", flush=True)
        bundle, audit = make_bundle(fitting, query, start, has_query_labels=True)
        audits.append(write_bundle(f"fold_{start}", bundle, audit))
        (REPORT / "manifest.json").write_text(json.dumps({"status": "running", "splits": audits}, indent=2), encoding="utf-8")
    fitting = train.loc[train[MONTH] < FINAL_ORIGIN]
    if not (tests[MONTH] >= FINAL_ORIGIN).all():
        raise AssertionError("Test dates precede the strict March 2025 forecast origin")
    print("Preparing strict future test split", flush=True)
    bundle, audit = make_bundle(fitting, tests, str(FINAL_ORIGIN), has_query_labels=False)
    audits.append(write_bundle("final_strict_2025-03", bundle, audit))
    manifest = {
        "status": "complete",
        "purpose": "TabPFN-3 native categorical/numeric inputs; no one-hot encoding or scaling",
        "format": "trusted local joblib bundle; pandas categorical dtypes preserved",
        "feature_order": ENHANCED,
        "source_sha256": source_hashes,
        "code_sha256": code_hashes,
        "versions": {key: version(key) for key in ("pandas", "numpy", "joblib", "tabpfn")},
        "splits": audits,
    }
    (REPORT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lines = ["# TabPFN-3 model-ready rental features", "",
             "Numeric columns retain their values and missingness; categorical columns use pandas `category` with levels learned from fitting rows. TabPFN applies its own numeric and categorical transforms. Each rental-history input uses only preceding months; query histories freeze at the validation origin.", "",
             "Bundles are compressed trusted local joblib files under `.cache/tabpfn_v3_features/`. Load with `joblib.load`, then use `X_train`, `y_train`, and `X_query` directly in `TabPFNRegressor`. `X_query` retains source row IDs as its index. Outer validation bundles also include `y_query` for evaluation. The final strict bundle has no query labels.", "",
             "| Split | Train rows | Query rows | Last label month | First query month | File |",
             "| --- | ---: | ---: | --- | --- | --- |"]
    lines.extend(f"| {a['origin']} | {a['train_rows']:,} | {a['query_rows']:,} | {a['last_label_month']} | {a['first_query_month']} | `{a['artifact']}` |" for a in audits)
    lines.extend(["", "See `manifest.json` for ordered features, category names, hashes, and library versions."])
    (REPORT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
