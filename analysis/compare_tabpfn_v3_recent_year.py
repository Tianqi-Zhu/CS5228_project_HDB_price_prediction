"""Compare TabPFN-3 using every labelled transaction in the preceding year."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from analysis.config import REPORTS, ROOT
from analysis.data import sha256
from analysis.evaluation import metrics
from analysis.experiments.tabpfn_comparison import fit_tabpfn


PREPARED = REPORTS / "tabpfn_v3_features" / "manifest.json"
PRIOR = REPORTS / "tabpfn_v3_prepared" / "v3_1x10k"
OUTPUT = REPORTS / "tabpfn_v3_prepared" / "v3_recent_12mo_full"
FOLDS = ("2023-01", "2023-07", "2024-04")


def main() -> None:
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", str(ROOT / ".cache" / "tabpfn"))
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TABPFN_MAX_BATCHED_ESTIMATOR_ROWS", "0")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(PREPARED.read_text(encoding="utf-8"))
    settings = {
        "model_version": "v3", "training_window_months": 12,
        "context_rows": 50_000, "n_estimators": 1, "batch_size": 500,
        "seed": 42, "fit_mode": "fit_with_cache",
        "prepared_manifest_sha256": sha256(PREPARED),
        "runner_sha256": sha256(Path(__file__)),
        "fit_helper_sha256": sha256(ROOT / "analysis/experiments/tabpfn_comparison.py"),
    }
    settings_path = OUTPUT / "settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text(encoding="utf-8")) != settings:
        raise ValueError("Existing output has different settings")
    settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    args = SimpleNamespace(**settings)
    predictions = []
    for origin in FOLDS:
        audit = next(split for split in manifest["splits"] if split["origin"] == origin)
        bundle_path = ROOT / audit["artifact"]
        if sha256(bundle_path) != audit["sha256"]:
            raise ValueError(f"Prepared bundle changed: {origin}")
        bundle = joblib.load(bundle_path)  # Trusted project-generated artifact.
        months = pd.PeriodIndex(bundle["train_month"], freq="M")
        start = pd.Period(origin, freq="M") - 12
        keep = (months >= start) & (months < pd.Period(origin, freq="M"))
        selected_months = months[keep]
        if selected_months.nunique() != 12 or selected_months.min() != start:
            raise AssertionError(f"Training context does not span 12 complete months: {origin}")
        x_train = bundle["X_train"].iloc[np.flatnonzero(keep)]
        y_train = np.asarray(bundle["y_train"], dtype=float)[keep]
        if len(x_train) != len(y_train) or len(x_train) > args.context_rows:
            raise AssertionError("Training context was truncated or misaligned")
        prior = pd.read_csv(PRIOR / f"predictions_{origin}.csv").set_index("row_id")
        query = bundle["X_query"]
        prior = prior.loc[query.index]
        np.testing.assert_array_equal(np.asarray(bundle["y_query"], float), prior.actual.to_numpy(float))
        print(f"{origin}: fitting {len(x_train):,} rows from {selected_months.min()} to {selected_months.max()}", flush=True)
        predicted, timing = fit_tabpfn(args, x_train, y_train, query)
        result = pd.DataFrame({
            "row_id": query.index.to_numpy(), "validation_start": origin,
            "actual": prior.actual.to_numpy(float), "tabpfn_recent_year": predicted,
            "tabpfn_random_10k": prior.tabpfn.to_numpy(float),
            "catboost": prior.catboost.to_numpy(float),
        })
        result.to_csv(OUTPUT / f"predictions_{origin}.csv", index=False, float_format="%.8f")
        timing.update({"training_rows": len(x_train), "first_train_month": str(selected_months.min()),
                       "last_train_month": str(selected_months.max())})
        (OUTPUT / f"timing_{origin}.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
        predictions.append(result)

    pooled = pd.concat(predictions, ignore_index=True)
    names = ("tabpfn_recent_year", "tabpfn_random_10k", "catboost")
    folds = pd.DataFrame([
        {"validation_start": origin, "model": name, "n": len(part), **metrics(part.actual, part[name])}
        for origin, part in pooled.groupby("validation_start", sort=True) for name in names
    ])
    summary = pd.DataFrame([
        {"model": name, "n": len(pooled), **metrics(pooled.actual, pooled[name])}
        for name in names
    ])
    folds.to_csv(OUTPUT / "fold_scores.csv", index=False, float_format="%.8f")
    summary.to_csv(OUTPUT / "pooled_scores.csv", index=False, float_format="%.8f")
    print(folds.to_string(index=False), flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
