"""Audit environment, report completeness and cross-report metric consistency."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sys

import pandas as pd

from analysis.pipeline.config import DATA, EXPERIMENT_REPORTS, ROOT, STUDY_REPORTS
from analysis.pipeline.data import check_output_path, sha256


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _score(rows, key, value):
    return next(row for row in rows if row[key] == value)


def audit_results() -> dict:
    checks = []

    def check(name, condition, detail):
        checks.append({"name": name, "ok": bool(condition), "detail": detail})

    required = [
        ROOT / "reports" / "README.md",
        EXPERIMENT_REPORTS / "model_baseline" / "results.json",
        EXPERIMENT_REPORTS / "model_families" / "results.json",
        EXPERIMENT_REPORTS / "auxiliary_ablation" / "results.json",
        EXPERIMENT_REPORTS / "deep_learning" / "summary" / "results.json",
        EXPERIMENT_REPORTS / "spatiotemporal_hybrid" / "fold_metrics.csv",
        EXPERIMENT_REPORTS / "neural_tabm" / "fold_metrics.csv",
        EXPERIMENT_REPORTS / "relu_mlp" / "historical_results.json",
        EXPERIMENT_REPORTS / "spatial_benchmark" / "metrics.csv",
        STUDY_REPORTS / "spatial_features" / "EDA_summary_zh.md",
        STUDY_REPORTS / "economic_signals" / "results.json",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    check("required_report_artifacts", not missing, {"missing": missing})

    baseline = _json(EXPERIMENT_REPORTS / "model_baseline" / "results.json")
    families = _json(EXPERIMENT_REPORTS / "model_families" / "results.json")
    auxiliary = _json(EXPERIMENT_REPORTS / "auxiliary_ablation" / "results.json")
    deep = _json(EXPERIMENT_REPORTS / "deep_learning" / "summary" / "results.json")
    baseline_c = _score(baseline["scores"], "model", "C_property_time_town")
    family_c = _score(families["scores"], "model", "C_property_time_town")
    family_lgb = _score(families["scores"], "model", "lightgbm")
    auxiliary_c = _score(auxiliary["scores"], "stage", "C_property_time_town")
    check("catboost_C_reused_exactly", baseline_c["rmse_sgd"] == family_c["rmse_sgd"], {
        "baseline": baseline_c["rmse_sgd"], "family_comparison": family_c["rmse_sgd"],
    })
    check(
        "lightgbm_C_reused_exactly",
        family_lgb["rmse_sgd"] == auxiliary_c["rmse_sgd"] == deep["reference_lightgbm_rmse"],
        {"family_comparison": family_lgb["rmse_sgd"],
         "auxiliary_ablation": auxiliary_c["rmse_sgd"],
         "deep_reference": deep["reference_lightgbm_rmse"]},
    )

    hashes = {path.name: sha256(path) for path in (
        DATA / "train.csv", DATA / "test.csv", DATA / "example-submission.csv",
    )}
    hash_sets = {
        "model_baseline": baseline["source_sha256"],
        "model_families": families["source_sha256"],
        "auxiliary_ablation": auxiliary["source_sha256"],
    }
    matching = all(
        all(recorded.get(name) == hashes[name] for name in ("train.csv", "test.csv"))
        and all(recorded.get(name, digest) == digest for name, digest in hashes.items())
        for recorded in hash_sets.values()
    )
    check("source_hashes_match_current_checkout", matching, {
        "current": hashes, "recorded": hash_sets,
        "note": "Experiments may record different source subsets; shared files must match.",
    })

    spatiotemporal = pd.read_csv(EXPERIMENT_REPORTS / "spatiotemporal_hybrid" / "fold_metrics.csv")
    tabm = pd.read_csv(EXPERIMENT_REPORTS / "neural_tabm" / "fold_metrics.csv")
    left = spatiotemporal.loc[spatiotemporal["variant"].eq("historical"), [
        "validation_start", "n_validation", "rmse_sgd", "mae_sgd", "bias_sgd",
    ]].reset_index(drop=True)
    right = tabm.loc[tabm["variant"].eq("historical_tree"), [
        "validation_start", "n", "rmse_sgd", "mae_sgd", "bias_sgd",
    ]].rename(columns={"n": "n_validation"}).reset_index(drop=True)
    check("tabm_reuses_historical_tree_folds", left.equals(right), {
        "folds": left.to_dict(orient="records"),
    })

    packages = {}
    for package in ("numpy", "pandas", "scikit-learn", "lightgbm", "catboost", "torch"):
        try:
            packages[package] = version(package)
        except Exception:
            packages[package] = None
    check("running_in_cs5228_environment", Path(sys.prefix).name.casefold() == "cs5228", {
        "sys_prefix": sys.prefix, "python": sys.version.split()[0], "packages": packages,
    })

    spatial_old = pd.read_csv(EXPERIMENT_REPORTS / "spatial_benchmark" / "metrics.csv")
    relu = _json(EXPERIMENT_REPORTS / "relu_mlp" / "historical_results.json")
    auxiliary_e = _score(auxiliary["scores"], "stage", "E_plus_amenities")
    latest_tabm = tabm.loc[
        tabm["validation_start"].eq("2024-04") & tabm["variant"].eq("blend_selected_oof")
    ].iloc[0]
    differences = [
        {
            "values": [auxiliary_e["rmse_sgd"], float(spatial_old.loc[
                spatial_old["model"].eq("lightgbm_spatial"), "rmse_sgd"
            ].iloc[0])],
            "status": "not_a_conflict",
            "reason": "Same holdout but different base/time encodings and spatial feature definitions.",
        },
        {
            "values": [deep["ensembles"]["historical_residual_blend"]["rmse"],
                       relu["relu"]["rmse_sgd"]],
            "status": "not_comparable",
            "reason": "Deep blend uses a 12-month holdout; ReLU uses a nested three-month holdout.",
        },
        {
            "values": [float(latest_tabm["rmse_sgd"]), 542.7843742813743],
            "status": "not_a_conflict",
            "reason": "First value is the latest 12-month fold; second is pooled over three origins.",
        },
    ]
    return {"ok": all(item["ok"] for item in checks), "checks": checks,
            "metric_differences": differences}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args(argv)
    result = audit_results()
    if args.output is not None:
        output = check_output_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["ok"]:
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    main()
