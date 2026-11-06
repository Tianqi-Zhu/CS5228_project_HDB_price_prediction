"""Compare feature policies using a fixed model and fixed-origin time splits."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from analysis.pipeline.config import AUX, DATA, DATE, EXPERIMENT_REPORTS, FAMILY_SETTINGS, TARGET
from analysis.pipeline.data import check_output_path, load_main, protect_sources
from analysis.pipeline.evaluation import grouped_errors, metrics
from analysis.pipeline.features import FEATURE_CANDIDATES, RentalFeatureEngineer, build_feature_frame, model_inputs


# Three equal-length, non-overlapping forecast windows. The overlapping stress
# window is reported separately and never included in the pooled ranking.
FOLDS = (
    ("2022-04", "2023-04", "selection"),
    ("2023-04", "2024-04", "selection"),
    ("2024-04", "2025-04", "selection"),
    ("2023-12", "2025-04", "stress_16m"),
)
SOURCES = [DATA / "train.csv", DATA / "test.csv"] + [
    AUX / name for name in (
        "sg-hdb-block.csv", "sg-mrt-stations.csv", "sg-shopping-malls.csv", "sg-schools.csv",
    )
]


def run(output_dir=EXPERIMENT_REPORTS / "features_selection"):
    out = check_output_path(Path(output_dir))
    out.mkdir(parents=True, exist_ok=True)
    settings = dict(FAMILY_SETTINGS["lightgbm"])
    with protect_sources(SOURCES) as hashes:
        frame = build_feature_frame(load_main(DATA / "train.csv", require_target=True))
        raw_test = load_main(DATA / "test.csv")
        scores, monthly = [], []
        for start, end, role in FOLDS:
            fitting = frame.loc[frame[DATE] < start].copy()
            validation = frame.loc[(frame[DATE] >= start) & (frame[DATE] < end)].copy()
            if fitting.empty or validation.empty:
                raise ValueError(f"Empty fold {start} to {end}")
            for name, spec in FEATURE_CANDIDATES.items():
                started = time.perf_counter()
                x_fit, x_val = model_inputs(fitting, validation, spec)
                model = lgb.LGBMRegressor(**settings)
                model.fit(x_fit, fitting[TARGET], categorical_feature=list(spec.categorical))
                predicted = np.asarray(model.predict(x_val), dtype=float)
                score = metrics(validation[TARGET].to_numpy(float), predicted)
                scores.append({
                    "validation_start": start, "validation_end_exclusive": end,
                    "role": role, "variant": name, "n_features": len(spec.all_columns),
                    "n_fit": len(fitting), "n_validation": len(validation), **score,
                    "elapsed_seconds": round(time.perf_counter() - started, 2),
                })
                errors = pd.DataFrame({
                    "approval_month": validation[DATE].to_numpy(),
                    "actual": validation[TARGET].to_numpy(float), "predicted": predicted,
                })
                by_month = grouped_errors(errors, "approval_month", name).rename(columns={"model": "variant"})
                by_month["validation_start"] = start
                by_month["role"] = role
                monthly.append(by_month)
                print(f"{start} {role} {name}: {len(spec.all_columns)} features; RMSE={score['rmse_sgd']:.2f}", flush=True)
                pd.DataFrame(scores).to_csv(out / "fold_metrics.csv", index=False, float_format="%.6f")
        scores = pd.DataFrame(scores)
        baseline = scores.loc[scores["variant"].eq("spatial_29")].set_index("validation_start")["rmse_sgd"]
        scores["delta_rmse_vs_spatial29"] = scores["rmse_sgd"] - scores["validation_start"].map(baseline)
        scores.to_csv(out / "fold_metrics.csv", index=False, float_format="%.6f")
        pd.concat(monthly, ignore_index=True).to_csv(out / "monthly_errors.csv", index=False, float_format="%.6f")
        pooled = []
        for name, part in scores.loc[scores["role"].eq("selection")].groupby("variant"):
            n = part["n_validation"].to_numpy()
            pooled.append({
                "variant": name, "n_features": int(part["n_features"].iloc[0]),
                "n_validation": int(n.sum()),
                "pooled_rmse_sgd": float(np.sqrt(np.average(part["rmse_sgd"] ** 2, weights=n))),
                "mean_fold_rmse_sgd": float(part["rmse_sgd"].mean()),
                "mae_sgd": float(np.average(part["mae_sgd"], weights=n)),
                "bias_sgd": float(np.average(part["bias_sgd"], weights=n)),
                "improved_folds_vs_spatial29": int((part["delta_rmse_vs_spatial29"] < 0).sum()),
            })
        pooled = pd.DataFrame(pooled).sort_values("pooled_rmse_sgd")
        pooled.to_csv(out / "pooled_metrics.csv", index=False, float_format="%.6f")
        result = {
            "model": "LightGBM", "model_settings": settings,
            "recommended_variant": RentalFeatureEngineer().variant,
            "best_pooled_variant": str(pooled.iloc[0]["variant"]),
            "selection_rule": "Minimize pooled RMSE on three non-overlapping 12-month folds; report the overlapping 16-month stress fold separately.",
            "versions": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "lightgbm")},
            "folds": FOLDS, "feature_candidates": {name: asdict(spec) for name, spec in FEATURE_CANDIDATES.items()},
            "source_sha256": hashes,
            "test_months": [raw_test[DATE].min(), raw_test[DATE].max()],
            "overlapping_test_month_rows": int((raw_test[DATE] <= frame[DATE].max()).sum()),
            "pooled_metrics": pooled.to_dict("records"),
            "limitations": [
                "All labeled months have been examined by earlier project experiments; these are model-selection estimates.",
                "The 16-month stress window overlaps selection folds and is excluded from pooled metrics.",
                "Facilities are static snapshots; historical opening dates are unavailable.",
                "Feature selection is conditional on one fixed LightGBM configuration, not a universal optimum.",
            ],
        }
        (out / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(pooled.to_string(index=False), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_REPORTS / "features_selection")
    args = parser.parse_args(argv)
    return run(args.output_dir)


if __name__ == "__main__":
    main()
