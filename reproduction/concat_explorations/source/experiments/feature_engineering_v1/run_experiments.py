"""CLI for the CS5228 feature-engineering experiments (Task A).

Stages
------
``baseline``   reproduce the saved LightGBM spatial benchmark (477.20 SGD).
``ablation``   one feature group at a time, fixed parameters, original split.
``splits``     promising groups on the early-12 and long-16 splits.
``smoothing``  prior-strength sensitivity for the historical summaries.
``catboost``   confirmation of the best group with CatBoost.
``all``        baseline -> ablation -> smoothing -> splits -> catboost.

Every stage writes metrics, monthly/band breakdowns, aligned prediction CSVs,
preprocessing state and a manifest under ``reports/feature_engineering_v1``.
Use ``--dry-run`` to list planned runs without fitting.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import data
import features
import history
import model as modeling

REPORT = config.REPORT_DIR
PREDICTIONS = REPORT / "predictions"


# ------------------------------------------------------------------ registry
def ablation_plan() -> list[tuple[str, tuple[str, ...], str]]:
    """(run name, feature groups, note) in the order the plan specifies."""
    return [
        ("lgb_spatial_baseline", (), "reproduce saved benchmark: 26 spatial features"),
        ("A1_micro_region", ("micro_region",), "subzone + planning area + region"),
        ("A2_micro_street", ("micro_region", "micro_street"), "adds town|street"),
        ("A3_micro_block", ("micro_region", "micro_street", "micro_block"), "adds composite block id"),
        ("B1_area_relative", ("area",), "area ratio/z/percentile within flat type"),
        ("B2_area_flags", ("area", "flags"), "adds missing-metadata flags"),
        ("C1_hist_rent_geo", ("hist_rent",), "smoothed block->street->subzone->town rent history"),
        ("C2_hist_psm_geo", ("hist_psm",), "same hierarchy, rent per square metre"),
        ("C3_hist_ft", ("hist_ft",), "flat-type hierarchy (block x ft, town x ft, ft)"),
        ("D1_win_rent", ("win_rent",), "3/6/12-month town x flat-type rent windows"),
        ("D2_win_psm", ("win_psm",), "3/6/12-month town x flat-type rent-per-sqm windows"),
    ]


def best_combo(groups_results: dict[str, float]) -> tuple[str, ...]:
    """Greedy forward selection over groups using the *validation* RMSE.

    This is model selection on the outer validation split and is therefore
    reported as exploratory only; the honest comparison is the per-group
    ablation and the split re-check, not this ordering.
    """
    order = ["hist_rent", "hist_ft", "win_rent", "micro_block", "micro_street",
             "micro_region", "area", "flags", "hist_psm", "win_psm"]
    chosen: list[str] = []
    current = groups_results.get("baseline", float("inf"))
    improved = True
    while improved:
        improved = False
        best_name, best_value = None, current
        for group in order:
            if group in chosen:
                continue
            candidate = tuple(chosen + [group])
            key = "+".join(candidate)
            if key in groups_results and groups_results[key] < best_value - 1e-9:
                best_name, best_value = group, groups_results[key]
        if best_name is not None:
            chosen.append(best_name)
            current = best_value
            improved = True
    return tuple(chosen)


# ------------------------------------------------------------------ helpers
def run_name_for(groups: tuple[str, ...], model: str = "lgb") -> str:
    return f"{model}_" + ("_".join(groups) if groups else "spatial_baseline")


def persist(result: modeling.RunResult, root: Path = REPORT) -> None:
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    name = result.model
    suffix = f"{name}__{result.split_id}__seed{result.seed}"
    result.predictions.to_csv(PREDICTIONS / f"{suffix}.csv", index=False)
    modeling.write_json(PREDICTIONS / f"{suffix}.metrics.json", {
        "metrics": result.metrics, "monthly": result.monthly, "bands": result.bands,
        "feature_columns": result.columns, "preprocessing": result.preprocessing,
        "groups": result.groups, "seconds": result.seconds,
    })


def append_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    if path.exists():
        existing = pd.read_csv(path)
        frame = pd.concat([existing, frame], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def build(
    frame: pd.DataFrame,
    split: dict,
    groups: tuple[str, ...],
    cache: dict,
) -> pd.DataFrame:
    key = (split["id"], tuple(groups))
    if key not in cache:
        extra, manifest = modeling.build_feature_frame(frame, split, groups)
        cache[key] = (extra, manifest)
    return cache[key][0]


def fit(
    frame: pd.DataFrame,
    split: dict,
    groups: tuple[str, ...],
    name: str,
    engine: str,
    seed: int,
    cache: dict,
    params: dict | None = None,
) -> modeling.RunResult:
    extra = build(frame, split, groups, cache)
    started = time.perf_counter()
    runner = modeling.run_lightgbm if engine == "lgb" else modeling.run_catboost
    result = runner(frame, split, groups, name, seed=seed, params=params, feature_frame=extra)
    print(f"  {name} [{split['id']}] RMSE {result.metrics['rmse_sgd']:.4f} "
          f"MAE {result.metrics['mae_sgd']:.4f} bias {result.metrics['bias_sgd']:.2f} "
          f"n_feat {result.metrics['n_features']} ({result.seconds:.1f}s)",
          flush=True)
    result.extra["wall_seconds"] = round(time.perf_counter() - started, 2)
    persist(result)
    return result


def recursion_manifest(
    frame: pd.DataFrame,
    splits: list[str],
    groups_by_split: dict[str, tuple[str, ...]],
) -> dict:
    payload = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "splits": {}}
    for split_id in splits:
        split = config.split_spec(split_id)
        payload["splits"][split_id] = {
            "cutoff_month": split["cutoff_month"],
            "cutoff_index": split["cutoff_index"],
            "description": split["description"],
            "feature_groups": list(groups_by_split.get(split_id, ())),
        }
    return payload


# ------------------------------------------------------------------ stages
def stage_baseline(frame: pd.DataFrame, cache: dict, seeds: tuple[int, ...]) -> dict:
    print("\n[baseline] reproducing the saved LightGBM spatial benchmark", flush=True)
    split = config.split_spec("main")
    rows = []
    reference_rmse = None
    for seed in seeds:
        result = fit(frame, split, (), "lgb_spatial_baseline", "lgb", seed, cache)
        entry = dict(result.metrics)
        entry["wall_seconds"] = result.extra.get("wall_seconds")
        rows.append(entry)
        if seed == config.SEED:
            reference_rmse = result.metrics["rmse_sgd"]
    append_csv(REPORT / "metrics_ablation.csv", rows)

    reproduced = reference_rmse is not None and abs(reference_rmse - config.BENCHMARK_RMSE) < 1e-6
    discrepancy = None
    if not reproduced:
        discrepancy = {
            "saved_rmse": config.BENCHMARK_RMSE,
            "reproduced_rmse": reference_rmse,
            "absolute_difference": None if reference_rmse is None else reference_rmse - config.BENCHMARK_RMSE,
            "cause": "to be diagnosed; candidate causes are library/thread nondeterminism "
                     "or a changed source file (hashes are recorded in the manifest)",
        }
    modeling.write_json(REPORT / "baseline_reproduction.json", {
        "saved_reference_rmse": config.BENCHMARK_RMSE,
        "saved_reference_n_validation": config.BENCHMARK_N_VALIDATION,
        "reproduced_rmse": reference_rmse,
        "reproduced_exactly": reproduced,
        "tolerance": 1e-6,
        "runs": rows,
        "discrepancy": discrepancy,
        "source_sha256": data.sha256(config.SOURCE),
        "environment": modeling.environment_manifest(),
    })
    print(f"[baseline] saved {config.BENCHMARK_RMSE:.10f} / reproduced {reference_rmse:.10f} "
          f"-> {'MATCH' if reproduced else 'DISCREPANCY'}", flush=True)
    return {"reference": reference_rmse, "reproduced": reproduced, "discrepancy": discrepancy}


def stage_ablation(frame: pd.DataFrame, cache: dict, seeds: tuple[int, ...]) -> dict:
    print("\n[ablation] one feature group at a time on the main split", flush=True)
    split = config.split_spec("main")
    rows, per_group = [], {}
    for name, groups, note in ablation_plan():
        for seed in seeds:
            result = fit(frame, split, groups, name, "lgb", seed, cache)
            entry = dict(result.metrics)
            entry["note"] = note
            entry["wall_seconds"] = result.extra.get("wall_seconds")
            rows.append(entry)
            if seed == config.SEED:
                per_group[name] = {
                    "rmse": result.metrics["rmse_sgd"],
                    "mae": result.metrics["mae_sgd"],
                    "bias": result.metrics["bias_sgd"],
                    "n_features": result.metrics["n_features"],
                    "groups": list(groups),
                    "note": note,
                }
    append_csv(REPORT / "metrics_ablation.csv", rows)
    modeling.write_json(REPORT / "metrics_ablation_summary.json", per_group)

    # Incremental stacking of the promising groups (exploratory).
    combos = [
        ("S1_spatial_plus_micro_block", ("micro_region", "micro_street", "micro_block")),
        ("S2_spatial_plus_hist_rent", ("hist_rent",)),
        ("S3_micro_plus_hist_rent", ("micro_region", "micro_street", "micro_block", "hist_rent")),
        ("S4_micro_hist_win", ("micro_region", "micro_street", "micro_block", "hist_rent", "win_rent")),
        ("S5_micro_hist_ft_win", ("micro_region", "micro_street", "micro_block",
                                  "hist_rent", "hist_ft", "win_rent")),
        ("S6_all_static_and_history", ("micro_region", "micro_street", "micro_block",
                                       "area", "flags", "hist_rent", "hist_ft", "win_rent")),
    ]
    stack_rows = []
    for name, groups in combos:
        for seed in seeds:
            result = fit(frame, split, groups, name, "lgb", seed, cache)
            entry = dict(result.metrics)
            entry["note"] = "stacked combination (exploratory)"
            entry["wall_seconds"] = result.extra.get("wall_seconds")
            stack_rows.append(entry)
    append_csv(REPORT / "metrics_ablation.csv", stack_rows)
    return {"single_group": per_group, "stacked": stack_rows}


def stage_smoothing(frame: pd.DataFrame, cache: dict) -> dict:
    print("\n[smoothing] prior-strength sensitivity of the historical summaries", flush=True)
    split = config.split_spec("main")
    rows = []
    for prior in config.SMOOTHING_SENSITIVITY:
        smoothing = {level: prior for level in config.SMOOTHING}
        for groups, name in ((("hist_rent",), "C1"), (("hist_ft",), "C3")):
            payload = {
                "numeric_training_medians": {}, "categorical_training_levels": {},
                "categorical_columns": [],
            }
            extra, _ = modeling.build_feature_frame(frame, split, groups, smoothing)
            result = modeling.run_lightgbm(frame, split, groups, f"{name}_prior{prior:g}",
                                           seed=config.SEED, feature_frame=extra)
            persist(result)
            entry = dict(result.metrics)
            entry["prior_strength_rows"] = prior
            entry["note"] = "prior-strength sensitivity; not a new feature group"
            rows.append(entry)
            print(f"  prior={prior:g} {name} RMSE {result.metrics['rmse_sgd']:.2f}", flush=True)
            del payload
    append_csv(REPORT / "metrics_smoothing.csv", rows)
    return {"rows": rows}


def stage_splits(frame: pd.DataFrame, cache: dict, combos: list[tuple[str, tuple[str, ...]]],
                 seeds: tuple[int, ...]) -> dict:
    print("\n[splits] re-check on the early-12 and long-16 chronological splits", flush=True)
    rows = []
    for split_id in ("early12", "long16"):
        split = config.split_spec(split_id)
        for name, groups in combos:
            for seed in seeds:
                result = fit(frame, split, groups, name, "lgb", seed, cache)
                entry = dict(result.metrics)
                entry["description"] = split["description"]
                entry["wall_seconds"] = result.extra.get("wall_seconds")
                rows.append(entry)
    append_csv(REPORT / "metrics_splits.csv", rows)
    return {"rows": rows}


def stage_catboost(frame: pd.DataFrame, cache: dict, combos: list[tuple[str, tuple[str, ...]]]) -> dict:
    print("\n[catboost] confirmation of the shortlisted feature sets", flush=True)
    split = config.split_spec("main")
    rows = []
    for name, groups in combos:
        result = fit(frame, split, groups, name.replace("lgb_", "cat_"), "cat", config.SEED, cache)
        entry = dict(result.metrics)
        entry["note"] = "CatBoost confirmation, identical rows and features"
        entry["wall_seconds"] = result.extra.get("wall_seconds")
        rows.append(entry)
    append_csv(REPORT / "metrics_catboost.csv", rows)
    return {"rows": rows}


def stage_focused(frame: pd.DataFrame, cache: dict, seeds: tuple[int, ...]) -> dict:
    """Second-stage combinations around the strongest single group (hist_ft).

    Every candidate is evaluated on all three chronological splits so that a
    gain which only exists on one split is visible as such.
    """
    print("\n[focused] hist_ft-centred combinations across all three splits", flush=True)
    candidates = [
        ("lgb_spatial_baseline", ()),
        ("lgb_hist_ft", ("hist_ft",)),
        ("F1_hist_ft_hist_rent", ("hist_ft", "hist_rent")),
        ("F2_hist_ft_win_rent", ("hist_ft", "win_rent")),
        ("F3_hist_ft_area", ("hist_ft", "area", "flags")),
        ("F4_hist_ft_win_hist_rent", ("hist_ft", "win_rent", "hist_rent")),
        ("F5_hist_ft_win_area", ("hist_ft", "win_rent", "area", "flags")),
    ]
    rows = []
    for split_id in config.SPLITS:
        split = config.split_spec(split_id)
        for name, groups in candidates:
            for seed in seeds:
                result = fit(frame, split, groups, name, "lgb", seed, cache)
                entry = dict(result.metrics)
                entry["description"] = split["description"]
                entry["wall_seconds"] = result.extra.get("wall_seconds")
                rows.append(entry)
    append_csv(REPORT / "metrics_focused.csv", rows)
    return {"rows": rows}


def stage_seed_stability(frame: pd.DataFrame, cache: dict, combos: list[tuple[str, tuple[str, ...]]],
                         seeds: tuple[int, ...]) -> dict:
    print("\n[seeds] seed stability of the shortlisted feature sets", flush=True)
    split = config.split_spec("main")
    rows = []
    for name, groups in combos:
        for seed in seeds:
            result = fit(frame, split, groups, name, "lgb", seed, cache)
            entry = dict(result.metrics)
            entry["note"] = "seed stability"
            rows.append(entry)
    append_csv(REPORT / "metrics_seeds.csv", rows)
    summary = pd.DataFrame(rows).groupby("feature_groups")["rmse_sgd"].agg(["mean", "std", "min", "max", "count"])
    summary.reset_index().to_csv(REPORT / "metrics_seeds_summary.csv", index=False)
    return {"rows": rows}


# ------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", default="all",
                        choices=["baseline", "ablation", "smoothing", "splits", "focused",
                                 "catboost", "seeds", "all"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)

    config.LIGHTGBM_PARAMS["n_jobs"] = args.threads
    config.CATBOOST_PARAMS["thread_count"] = args.threads

    REPORT.mkdir(parents=True, exist_ok=True)
    frame = data.load_frame()
    source_hash = data.sha256(config.SOURCE)
    print(f"source rows={len(frame)} sha256={source_hash[:16]}… threads={args.threads}", flush=True)

    planned = {
        "baseline": ["lgb_spatial_baseline seed42/43/44"],
        "ablation": [f"{name} seed42/43/44" for name, _g, _n in ablation_plan()] +
                    ["S1..S6 stacked seed42/43/44"],
        "smoothing": ["C1/C3 prior 3 and 30"],
        "splits": ["baseline + micro + hist + win on early12/long16"],
        "catboost": ["baseline + hist_rent on main"],
        "seeds": ["shortlist seeds 42/43/44"],
    }
    if args.dry_run:
        print(json.dumps(planned, indent=2))
        return 0
    if modeling.LGBMRegressor is None:
        print(f"BLOCKED: LightGBM unavailable: {modeling.LIGHTGBM_IMPORT_ERROR}", file=sys.stderr)
        return 3

    seeds = config.FIXED_SEEDS
    cache: dict = {}
    outcomes: dict = {}
    started = time.perf_counter()

    splits_used: list[str] = []
    groups_by_split: dict[str, tuple[str, ...]] = {}

    if args.stage in ("baseline", "all"):
        outcomes["baseline"] = stage_baseline(frame, cache, seeds)
        splits_used.append("main")

    if args.stage in ("ablation", "all"):
        outcomes["ablation"] = stage_ablation(frame, cache, seeds)
        splits_used.append("main")

    if args.stage in ("smoothing", "all"):
        outcomes["smoothing"] = stage_smoothing(frame, cache)

    shortlist = [
        ("lgb_spatial_baseline", ()),
        ("lgb_micro_block", ("micro_region", "micro_street", "micro_block")),
        ("lgb_hist_rent", ("hist_rent",)),
        ("lgb_hist_ft", ("hist_ft",)),
        ("lgb_win_rent", ("win_rent",)),
        ("lgb_micro_hist_win", ("micro_region", "micro_street", "micro_block", "hist_rent", "win_rent")),
    ]

    if args.stage in ("splits", "all"):
        outcomes["splits"] = stage_splits(frame, cache, shortlist, seeds)
        splits_used += ["early12", "long16"]

    if args.stage in ("focused", "all"):
        outcomes["focused"] = stage_focused(frame, cache, seeds)
        splits_used += list(config.SPLITS)

    if args.stage in ("catboost", "all"):
        if modeling.CatBoostRegressor is None:
            outcomes["catboost"] = {"status": "skipped", "reason": modeling.CATBOOST_IMPORT_ERROR}
            print(f"[catboost] skipped: {modeling.CATBOOST_IMPORT_ERROR}", flush=True)
        else:
            outcomes["catboost"] = stage_catboost(
                frame, cache, [("cat_spatial_baseline", ()), ("cat_hist_rent", ("hist_rent",))])

    if args.stage in ("seeds", "all"):
        outcomes["seeds"] = stage_seed_stability(
            frame, cache,
            [("lgb_spatial_baseline", ()), ("lgb_hist_rent", ("hist_rent",)),
             ("lgb_micro_hist_win", ("micro_region", "micro_street", "micro_block", "hist_rent", "win_rent"))],
            seeds)

    elapsed = time.perf_counter() - started
    for split_id in splits_used:
        groups_by_split[split_id] = ()
    modeling.write_json(REPORT / "run_manifest.json", {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "stage": args.stage,
        "threads_per_fit": args.threads,
        "wall_seconds": round(elapsed, 1),
        "source": str(config.SOURCE.relative_to(config.ROOT)),
        "source_sha256": source_hash,
        "n_source": len(frame),
        "n_source_sha_unchanged": data.sha256(config.SOURCE) == source_hash,
        "splits": {split_id: config.split_spec(split_id) for split_id in config.SPLITS},
        "environment": modeling.environment_manifest(),
        "fixed_lightgbm_params": config.LIGHTGBM_PARAMS,
        "fixed_catboost_params": config.CATBOOST_PARAMS,
        "smoothing_defaults": config.SMOOTHING,
        "ablation_plan": [{"name": name, "groups": list(groups), "note": note}
                          for name, groups, note in ablation_plan()],
        "shortlist": [{"name": name, "groups": list(groups)} for name, groups in shortlist],
        "outcomes": outcomes,
    })
    print(f"\n[done] {elapsed:.1f}s -> {REPORT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
