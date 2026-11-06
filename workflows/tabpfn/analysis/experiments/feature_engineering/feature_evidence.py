"""Retrained feature deletion effects, with paired month-block sensitivity.

Effects are conditional predictive contributions, not causal effects. Derived
columns that are not explicitly removed remain available to the model.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
import time

from lightgbm import LGBMRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.experiments.feature_selection import FOLDS, SOURCES
from analysis.pipeline.config import DATA, DATE, EXPERIMENT_REPORTS, FAMILY_SETTINGS, TARGET
from analysis.pipeline.data import check_output_path, load_main, protect_sources
from analysis.pipeline.evaluation import metrics
from analysis.pipeline.features import FEATURE_CANDIDATES, build_feature_frame, model_inputs


OUT = EXPERIMENT_REPORTS / "features_ablation_evidence"
SPEC = FEATURE_CANDIDATES["compact_history_27"]
GROUPS = {
    "property": ("FLAT_TYPE_CLEAN", "FLAT_MODEL_CLEAN", "FLOOR_AREA_SQM", "FLAT_AGE_YEARS", "MAX_FLOOR"),
    "time": ("APPROVAL_MONTH_INDEX", "APPROVAL_MONTH_SIN", "APPROVAL_MONTH_COS"),
    "location": ("TOWN_CLEAN", "PLANNING_AREA_CLEAN", "LATITUDE", "LONGITUDE", "CBD_DIST_KM"),
    "mrt": ("MRT_DIST_KM", "MRT_COUNT_1_0KM"),
    "mall": ("MALL_DIST_KM", "MALL_COUNT_1_0KM"),
    "primary_school": ("PRIMARY_SCHOOL_DIST_KM", "PRIMARY_SCHOOL_COUNT_1_0KM"),
    "block_history": tuple(c for c in SPEC.history_columns if "BLOCK_FLAT" in c),
    "area_history": tuple(c for c in SPEC.history_columns if "AREA_FLAT" in c),
}
JOINT_GROUPS = {
    "all_amenities": GROUPS["mrt"] + GROUPS["mall"] + GROUPS["primary_school"],
    "all_history": GROUPS["block_history"] + GROUPS["area_history"],
    "time_and_history": GROUPS["time"] + GROUPS["block_history"] + GROUPS["area_history"],
}
GROUP_LABELS = {
    "property": "Property (5)", "time": "Time (3)", "location": "Location (5)",
    "mrt": "MRT (2)", "mall": "Malls (2)", "primary_school": "Primary schools (2)",
    "block_history": "Block/type history (4)", "area_history": "Area/type history (4)",
    "all_amenities": "All amenities (6)", "all_history": "All history (8)",
    "time_and_history": "Time + all history (11)",
}


def ablations():
    result = {"baseline": {"kind": "baseline", "removed": ()}}
    result.update({f"drop__{c}": {"kind": "single", "removed": (c,)} for c in SPEC.all_columns})
    result.update({f"group__{name}": {"kind": "group", "removed": cols} for name, cols in GROUPS.items()})
    result.update({f"joint__{name}": {"kind": "joint", "removed": cols} for name, cols in JOINT_GROUPS.items()})
    return result


def retained_columns(columns, removed):
    unknown = set(removed) - set(columns)
    if unknown:
        raise ValueError(f"Unknown columns to remove: {sorted(unknown)}")
    result = [c for c in columns if c not in removed]
    if not result:
        raise ValueError("Ablation must retain at least one model input")
    return result


def monthly_squared_errors(months, actual, baseline, candidate):
    return pd.DataFrame({"month": np.asarray(months),
        "baseline_sse": (actual - baseline) ** 2,
        "candidate_sse": (actual - candidate) ** 2}).groupby("month", sort=True).agg(
            n=("baseline_sse", "size"), baseline_sse=("baseline_sse", "sum"),
            candidate_sse=("candidate_sse", "sum")).reset_index()


def paired_month_interval(monthly, repeats=2000, block_months=3, seed=42):
    """Circular contiguous month blocks, row-weighted RMSE in each resample.

    Exploratory sensitivity interval conditional on fixed trained predictions;
    not a confidence interval for future-market or training randomness.
    """
    if len(monthly) < 2 or repeats < 1 or block_months < 1:
        raise ValueError("Need >=2 months and positive repeat/block counts")
    n = monthly["n"].to_numpy(float)
    a = monthly["baseline_sse"].to_numpy(float)
    b = monthly["candidate_sse"].to_numpy(float)
    if (n <= 0).any() or not np.isfinite(np.column_stack([n, a, b])).all():
        raise ValueError("Invalid month counts or squared errors")
    generator = np.random.default_rng(seed)
    starts = generator.integers(0, len(n), size=(repeats, int(np.ceil(len(n) / block_months))))
    indices = ((starts[..., None] + np.arange(block_months)) % len(n)).reshape(repeats, -1)[:, :len(n)]
    denominator = n[indices].sum(axis=1)
    differences = np.sqrt(b[indices].sum(axis=1) / denominator) - np.sqrt(a[indices].sum(axis=1) / denominator)
    return {"month_block_p025": float(np.quantile(differences, .025)),
            "month_block_p975": float(np.quantile(differences, .975))}


def aggregate_effects(scores):
    result = []
    for variant, part in scores.loc[scores.role.eq("selection")].groupby("variant", sort=False):
        n = part.n_validation.to_numpy(float)
        rmse = float(np.sqrt(np.average(part.rmse_sgd ** 2, weights=n)))
        base = float(np.sqrt(np.average(part.baseline_rmse_sgd ** 2, weights=n)))
        latest = scores.loc[scores.variant.eq(variant) & scores.validation_start.eq("2024-04")].iloc[0]
        stress = scores.loc[scores.variant.eq(variant) & scores.role.eq("stress_16m")].iloc[0]
        result.append({"variant": variant, "kind": part.kind.iloc[0],
            "removed_columns": part.removed_columns.iloc[0],
            "n_remaining": int(part.n_remaining.iloc[0]), "pooled_rmse_sgd": rmse,
            "delta_pooled_rmse_sgd": rmse - base,
            "annual_folds_worse_without": int((part.delta_rmse_sgd > 0).sum()),
            "annual_folds_better_without": int((part.delta_rmse_sgd < 0).sum()),
            "latest_rmse_sgd": float(latest.rmse_sgd), "latest_delta_rmse_sgd": float(latest.delta_rmse_sgd),
            "latest_month_block_p025": float(latest.month_block_p025),
            "latest_month_block_p975": float(latest.month_block_p975),
            "stress_16m_delta_rmse_sgd": float(stress.delta_rmse_sgd)})
    return pd.DataFrame(result).sort_values("delta_pooled_rmse_sgd", ascending=False)


def plot_effects(summary, out):
    selected = summary.loc[summary.kind.isin(["group", "joint"]) & ~summary.variant.eq("joint__time_and_history")].sort_values("delta_pooled_rmse_sgd")
    names = [GROUP_LABELS[v.split("__", 1)[1]] for v in selected.variant]
    fig, ax = plt.subplots(figsize=(10, 5.8), layout="constrained")
    values = selected.delta_pooled_rmse_sgd.to_numpy()
    ax.barh(names, values, color=np.where(values >= 0, "#377eb8", "#e69543"))
    ax.axvline(0, color="#333333", lw=1)
    ax.set_xlabel("RMSE after retraining without inputs minus full-model RMSE (SGD)")
    joint = summary.loc[summary.variant.eq("joint__time_and_history"), "delta_pooled_rmse_sgd"].iloc[0]
    ax.set_title(f"Feature-group deletion: three annual windows\nTime + all history joint deletion: {joint:+.2f} SGD (omitted from this axis)")
    ax.grid(axis="x", alpha=.15)
    fig.savefig(out / "group_deletion.png", dpi=160)
    plt.close(fig)


def run(output_dir=OUT):
    out = check_output_path(Path(output_dir))
    out.mkdir(parents=True, exist_ok=True)
    settings = dict(FAMILY_SETTINGS["lightgbm"])
    definitions = ablations()
    rows, monthly, diagnostics = [], [], []
    with protect_sources(SOURCES) as hashes:
        frame = build_feature_frame(load_main(DATA / "train.csv", require_target=True))
        for start, end, role in FOLDS:
            fitting = frame.loc[frame[DATE] < start].copy()
            validation = frame.loc[(frame[DATE] >= start) & (frame[DATE] < end)].copy()
            x_fit, x_val = model_inputs(fitting, validation, SPEC)
            actual = validation[TARGET].to_numpy(float)
            baseline = None
            for variant, description in definitions.items():
                began = time.perf_counter()
                cols = retained_columns(SPEC.all_columns, description["removed"])
                categorical = [c for c in SPEC.categorical if c in cols]
                model = LGBMRegressor(**settings)
                model.fit(x_fit[cols], fitting[TARGET], categorical_feature=categorical)
                prediction = np.asarray(model.predict(x_val[cols]), float)
                score = metrics(actual, prediction)
                if variant == "baseline":
                    baseline = prediction.copy()
                    baseline_rmse = score["rmse_sgd"]
                    detail = validation[[DATE, "TOWN_CLEAN", "FLAT_TYPE_CLEAN"]].copy()
                    detail["actual"], detail["prediction"] = actual, prediction
                    detail["forecast_horizon_months"] = (
                        pd.to_datetime(detail[DATE]).dt.year * 12 + pd.to_datetime(detail[DATE]).dt.month
                        - (pd.Period(start, freq="M").year * 12 + pd.Period(start, freq="M").month - 1))
                    detail["history_count"] = x_val["HIST_BLOCK_FLAT_COUNT"].to_numpy()
                    detail["history_recency"] = x_val["HIST_BLOCK_FLAT_RECENCY_MONTHS"].to_numpy()
                    diagnostics.append(detail.assign(validation_start=start, role=role))
                month = monthly_squared_errors(validation[DATE], actual, baseline, prediction)
                interval = paired_month_interval(month)
                rows.append({"validation_start": start, "validation_end_exclusive": end,
                    "role": role, "variant": variant, "kind": description["kind"],
                    "removed_columns": "|".join(description["removed"]), "n_remaining": len(cols),
                    "n_fit": len(fitting), "n_validation": len(validation), **score,
                    "baseline_rmse_sgd": baseline_rmse, "delta_rmse_sgd": score["rmse_sgd"] - baseline_rmse,
                    **interval, "elapsed_seconds": time.perf_counter() - began})
                monthly.append(month.assign(validation_start=start, role=role, variant=variant))
                pd.DataFrame(rows).to_csv(out / "fold_effects.csv", index=False)
                print(f"{start} {variant}: RMSE {score['rmse_sgd']:.2f}; deletion delta {score['rmse_sgd']-baseline_rmse:+.2f}", flush=True)
        scores = pd.DataFrame(rows)
        summary = aggregate_effects(scores)
        summary.to_csv(out / "feature_effects.csv", index=False)
        pd.concat(monthly, ignore_index=True).to_csv(out / "monthly_paired_sse.csv", index=False)
        summarize_diagnostics(pd.concat(diagnostics, ignore_index=True), out)
        plot_effects(summary, out)
        result = {"feature_spec": asdict(SPEC), "groups": GROUPS, "joint_groups": JOINT_GROUPS,
            "model": "LightGBM", "model_settings": settings, "folds": FOLDS,
            "fits_completed": len(rows), "source_sha256": hashes,
            "deletion_delta_definition": "RMSE(retrained without specified columns) minus RMSE(full 27 columns). Positive means removal hurts; negative means removal helped this model.",
            "interval": {"method": "paired circular 3-month-block resampling", "repeats": 2000, "seed": 42,
                "meaning": "Exploratory 2.5/97.5 percentiles conditional on fixed predictions, not future-market or training uncertainty; no multiple-comparison adjustment."},
            "limitations": ["Previously examined labeled months; exploratory model-selection evidence, not independent test.",
                "Single fixed LightGBM and seed; reoptimizing a reduced model could change its ranking.",
                "Deleting inputs leaves unremoved correlated and derived inputs available; this is not removal of all raw-source information.",
                "Deletion contributions are conditional, nonadditive, and noncausal.",
                "The overlapping stress fold is not included in pooled RMSE.",
                "Facilities are supplied static snapshots without complete historical opening dates."],
            "versions": {p: version(p) for p in ("numpy", "pandas", "scikit-learn", "lightgbm")}}
        (out / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def summarize_diagnostics(frame, out):
    frame = frame.copy()
    frame["history_support"] = pd.cut(frame.history_count, [-1, 0, 3, 10, np.inf],
        labels=["unseen", "1-3", "4-10", "11+"]).astype(str)
    frame["history_age"] = pd.cut(frame.history_recency.fillna(np.inf), [-1, 6, 12, 24, np.inf],
        labels=["<=6 months", "7-12 months", "13-24 months", "25+ or unseen"]).astype(str)
    frame["actual_rent_band"] = pd.cut(frame.actual, [0, 2000, 3000, 4000, np.inf],
        labels=["<=2000", "2000-3000", "3000-4000", ">4000"]).astype(str)
    dimensions = [DATE, "forecast_horizon_months", "TOWN_CLEAN", "FLAT_TYPE_CLEAN",
                  "history_support", "history_age", "actual_rent_band"]
    rows = []
    for (start, role), part in frame.groupby(["validation_start", "role"]):
        total = float(np.sum((part.actual - part.prediction) ** 2))
        for dimension in dimensions:
            for group, subset in part.groupby(dimension, observed=True):
                scores = metrics(subset.actual.to_numpy(float), subset.prediction.to_numpy(float))
                rows.append({"validation_start": start, "role": role, "dimension": dimension,
                    "group": str(group), "n": len(subset), **scores,
                    "share_of_fold_squared_error": float(np.sum((subset.actual-subset.prediction)**2)/total)})
    pd.DataFrame(rows).to_csv(out / "error_diagnostics.csv", index=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args(argv)
    return run(args.output_dir)


if __name__ == "__main__":
    main()
