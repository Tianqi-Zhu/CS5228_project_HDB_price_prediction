"""Targeted feature ablations with fixed LightGBM and frozen forecast origins."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from analysis.experiments.feature_selection import FOLDS, SOURCES
from analysis.pipeline.config import DATA, DATE, EXPERIMENT_REPORTS, FAMILY_SETTINGS, TARGET
from analysis.pipeline.data import check_output_path, load_main, protect_sources
from analysis.pipeline.evaluation import grouped_errors, metrics
from analysis.pipeline.features import (
    COMPACT, COMPACT_CATEGORIES, HISTORY_POLICIES, MARKET_COLUMNS,
    EnhancedHistoryEncoder, FeatureSpec, HistoricalRentEncoder, MarketContextEncoder,
    build_feature_frame, model_inputs,
)


def spec(*, policies=(), grains=("SIGNATURE", "BLOCK_FLAT", "AREA_FLAT"), extra=(), categories=(), market=False):
    return FeatureSpec(COMPACT + extra, COMPACT_CATEGORIES + categories, True, policies, grains, market)


CANDIDATES = {
    "baseline_31": spec(),
    "area_history_only": spec(grains=("AREA_FLAT",)),
    "area_block_history": spec(grains=("BLOCK_FLAT", "AREA_FLAT")),
    "history_smooth": spec(policies=("smooth",)),
    "history_decay6": spec(policies=("decay6",)),
    "history_decay12": spec(policies=("decay12",)),
    "history_recent12": spec(policies=("recent12",)),
    "history_smooth_decay12": spec(policies=("smooth_decay12",)),
    "history_premium_only": spec(policies=("premium",)),
    "plus_premium": spec(policies=("standard", "premium")),
    "plus_premium_decay12": spec(policies=("standard", "premium_decay12")),
    "plus_ppsm": spec(policies=("standard", "ppsm")),
    "history_coarse_smooth": spec(policies=("coarse",)),
    "plus_property": spec(extra=("AREA_PER_ROOM", "ROOM_COUNT", "COMPLETION_LEASE_GAP")),
    "plus_street": spec(extra=("STREET_CLEAN",), categories=("STREET_CLEAN",)),
    "plus_address": spec(extra=("ADDRESS_KEY",), categories=("ADDRESS_KEY",)),
    "plus_multiscale": spec(extra=tuple(f"{kind}_COUNT_{radius}KM" for kind in (
        "MRT", "MALL", "PRIMARY_SCHOOL") for radius in ("0_5", "2_0"))),
    "plus_market_context": spec(market=True),
}
REFINED_CANDIDATES = {
    "baseline_31": spec(),
    "deduplicated_27": spec(grains=("BLOCK_FLAT", "AREA_FLAT")),
    "premium_deduplicated_31": spec(policies=("standard", "premium_values"),
        grains=("BLOCK_FLAT", "AREA_FLAT")),
}


def audit_history(output_dir=EXPERIMENT_REPORTS / "features_history_optimization"):
    out = check_output_path(Path(output_dir))
    out.mkdir(parents=True, exist_ok=True)
    with protect_sources(SOURCES) as hashes:
        frame = build_feature_frame(load_main(DATA / "train.csv", require_target=True))
        block_keys = ["TOWN_KEY", "BLOCK_KEY", "STREET_KEY", "FLAT_TYPE_CLEAN"]
        varying = frame.groupby(block_keys, dropna=False)[
            ["FLAT_MODEL_CLEAN", "FLOOR_AREA_SQM", "LEASE_COMMENCE_DATE"]
        ].nunique(dropna=False)
        full_history = HistoricalRentEncoder().fit_transform(frame)
        duplicates = {}
        for suffix in ("MEAN", "COUNT", "LAST_MEAN", "RECENCY_MONTHS"):
            left = full_history[f"HIST_SIGNATURE_{suffix}"].to_numpy(float)
            right = full_history[f"HIST_BLOCK_FLAT_{suffix}"].to_numpy(float)
            duplicates[suffix] = bool(np.allclose(left, right, equal_nan=True, rtol=0, atol=0))
        coverage = []
        for start, end, role in FOLDS:
            fitting = frame.loc[frame[DATE] < start]
            validation = frame.loc[(frame[DATE] >= start) & (frame[DATE] < end)]
            histories = HistoricalRentEncoder().fit(fitting).transform(validation)
            for grain in ("SIGNATURE", "BLOCK_FLAT", "AREA_FLAT"):
                n = histories[f"HIST_{grain}_COUNT"]
                coverage.append({"validation_start": start, "role": role, "grain": grain,
                    "known_fraction": float((n > 0).mean()), "at_most_3_fraction": float((n <= 3).mean()),
                    "median_prior_count": float(n.median()),
                    "median_recency_months": float(histories[f"HIST_{grain}_RECENCY_MONTHS"].median())})
        pd.DataFrame(coverage).to_csv(out / "history_support.csv", index=False, float_format="%.6f")
        result = {"training_rows": len(frame), "block_flat_groups": len(varying),
            "groups_with_varying_signature_attributes": int((varying > 1).any(axis=1).sum()),
            "signature_history_exactly_duplicates_block_history": duplicates,
            "source_sha256": hashes}
        (out / "redundancy_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def run(output_dir=EXPERIMENT_REPORTS / "features_history_optimization", names=None, candidate_specs=None):
    out = check_output_path(Path(output_dir))
    out.mkdir(parents=True, exist_ok=True)
    source = CANDIDATES if candidate_specs is None else candidate_specs
    candidates = source if not names else {name: source[name] for name in names}
    if "baseline_31" not in candidates:
        candidates = {"baseline_31": CANDIDATES["baseline_31"], **candidates}
    settings = dict(FAMILY_SETTINGS["lightgbm"])
    with protect_sources(SOURCES) as hashes:
        frame = build_feature_frame(load_main(DATA / "train.csv", require_target=True))
        scores, monthly = [], []
        for start, end, role in FOLDS:
            fitting = frame.loc[frame[DATE] < start].copy()
            validation = frame.loc[(frame[DATE] >= start) & (frame[DATE] < end)].copy()
            cache = {}
            market_cache = None
            for name, candidate in candidates.items():
                started = time.perf_counter()
                static = FeatureSpec(candidate.columns, candidate.categorical)
                x_fit, x_val = model_inputs(fitting, validation, static)
                for policy in candidate.history_policies or ("standard",):
                    if policy not in cache:
                        encoder = (HistoricalRentEncoder() if policy == "standard"
                                   else EnhancedHistoryEncoder(HISTORY_POLICIES[policy])).fit(fitting)
                        cache[policy] = (encoder.transform(fitting), encoder.transform(validation))
                    for x, history in zip((x_fit, x_val), cache[policy]):
                        for column in history:
                            if column in candidate.history_columns:
                                x[column] = history[column].to_numpy()
                if candidate.market_context:
                    if market_cache is None:
                        market = MarketContextEncoder().fit(fitting)
                        market_cache = (market.transform(fitting), market.transform(validation))
                    for x, context in zip((x_fit, x_val), market_cache):
                        for column in MARKET_COLUMNS:
                            x[column] = context[column].to_numpy()
                assert list(x_fit) == list(candidate.all_columns)
                model = LGBMRegressor(**settings)
                model.fit(x_fit, fitting[TARGET], categorical_feature=list(candidate.categorical))
                predicted = np.asarray(model.predict(x_val), dtype=float)
                score = metrics(validation[TARGET].to_numpy(float), predicted)
                scores.append({"validation_start": start, "validation_end_exclusive": end,
                    "role": role, "variant": name, "n_features": len(candidate.all_columns),
                    "n_fit": len(fitting), "n_validation": len(validation), **score,
                    "elapsed_seconds": round(time.perf_counter() - started, 2)})
                errors = pd.DataFrame({"approval_month": validation[DATE].to_numpy(),
                    "actual": validation[TARGET].to_numpy(float), "predicted": predicted})
                by_month = grouped_errors(errors, "approval_month", name).rename(columns={"model": "variant"})
                by_month["validation_start"], by_month["role"] = start, role
                monthly.append(by_month)
                print(f"{start} {role} {name}: {len(candidate.all_columns)} features RMSE={score['rmse_sgd']:.2f}", flush=True)
                pd.DataFrame(scores).to_csv(out / "fold_metrics.csv", index=False, float_format="%.6f")
        scores = pd.DataFrame(scores)
        baseline = scores.loc[scores.variant.eq("baseline_31")].set_index("validation_start").rmse_sgd
        scores["delta_rmse_vs_baseline"] = scores.rmse_sgd - scores.validation_start.map(baseline)
        scores.to_csv(out / "fold_metrics.csv", index=False, float_format="%.6f")
        pd.concat(monthly, ignore_index=True).to_csv(out / "monthly_errors.csv", index=False, float_format="%.6f")
        pooled = []
        for name, part in scores.loc[scores.role.eq("selection")].groupby("variant"):
            n = part.n_validation.to_numpy()
            stress = scores.loc[scores.variant.eq(name) & scores.role.eq("stress_16m")].iloc[0]
            pooled.append({"variant": name, "n_features": int(part.n_features.iloc[0]),
                "pooled_rmse_sgd": float(np.sqrt(np.average(part.rmse_sgd ** 2, weights=n))),
                "mae_sgd": float(np.average(part.mae_sgd, weights=n)),
                "improved_folds": int((part.delta_rmse_vs_baseline < 0).sum()),
                "latest_rmse_sgd": float(part.loc[part.validation_start.eq("2024-04"), "rmse_sgd"].iloc[0]),
                "stress_16m_rmse_sgd": float(stress.rmse_sgd),
                "stress_delta_sgd": float(stress.delta_rmse_vs_baseline)})
        pooled = pd.DataFrame(pooled).sort_values("pooled_rmse_sgd")
        pooled.to_csv(out / "pooled_metrics.csv", index=False, float_format="%.6f")
        result = {"model": "LightGBM", "settings": settings, "folds": FOLDS,
            "candidates": {name: asdict(candidate) for name, candidate in candidates.items()},
            "history_policies": {name: asdict(policy) for name, policy in HISTORY_POLICIES.items()},
            "source_sha256": hashes, "pooled_metrics": pooled.to_dict("records"),
            "versions": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "lightgbm")},
            "limitations": ["Previously inspected labels; model-selection estimates, not a new independent holdout.",
                "16-month stress window overlaps yearly folds and is excluded from pooled RMSE.",
                "No feature group or decay parameter is tuned separately for each fold."]}
        (out / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(pooled.to_string(index=False), flush=True)
    return result


def run_market_decomposition(output_dir=EXPERIMENT_REPORTS / "features_history_optimization" / "market_decomposition"):
    """Separate feature additions from a causal change of target representation.

    Every arm drops the same first six fitting months. Histories may still use
    those observed months. Residual arms add their past-only market forecast
    back at inference; no validation rent enters that forecast.
    """
    out = check_output_path(Path(output_dir))
    out.mkdir(parents=True, exist_ok=True)
    settings = dict(FAMILY_SETTINGS["lightgbm"])
    with protect_sources(SOURCES) as hashes:
        frame = build_feature_frame(load_main(DATA / "train.csv", require_target=True))
        rows = []
        for start, end, role in FOLDS:
            fitting = frame.loc[frame[DATE] < start].copy()
            validation = frame.loc[(frame[DATE] >= start) & (frame[DATE] < end)].copy()
            candidate = spec(market=True)
            x_fit, x_val = model_inputs(fitting, validation, candidate)
            first_month = pd.Period(fitting[DATE].min(), freq="M")
            warmup_end = str(first_month + 6)
            eligible = (fitting[DATE] >= warmup_end).to_numpy() & np.isfinite(
                x_fit["MARKET_TYPE_PROJECTED"].to_numpy(float))
            for name, offset_column in (
                ("matched_31", None), ("matched_37_market_features", None),
                ("residual_last_market", "MARKET_TYPE_LAST"),
                ("residual_damped_market", "MARKET_TYPE_PROJECTED"),
            ):
                columns = list(CANDIDATES["baseline_31"].all_columns) if name == "matched_31" else list(candidate.all_columns)
                fit_offset = np.zeros(len(fitting)) if offset_column is None else x_fit[offset_column].to_numpy(float)
                val_offset = np.zeros(len(validation)) if offset_column is None else x_val[offset_column].to_numpy(float)
                model = LGBMRegressor(**settings)
                model.fit(x_fit.loc[eligible, columns], fitting[TARGET].to_numpy(float)[eligible] - fit_offset[eligible],
                    categorical_feature=list(candidate.categorical))
                prediction = np.asarray(model.predict(x_val[columns]), float) + val_offset
                score = metrics(validation[TARGET].to_numpy(float), prediction)
                rows.append({"validation_start": start, "validation_end_exclusive": end,
                    "role": role, "variant": name, "n_features": len(columns),
                    "n_fit": int(eligible.sum()), "n_validation": len(validation), **score})
                print(f"market {start} {name}: RMSE={score['rmse_sgd']:.2f}", flush=True)
                pd.DataFrame(rows).to_csv(out / "fold_metrics.csv", index=False, float_format="%.6f")
        scores = pd.DataFrame(rows)
        pooled = []
        for name, part in scores.loc[scores.role.eq("selection")].groupby("variant"):
            pooled.append({"variant": name,
                "pooled_rmse_sgd": float(np.sqrt(np.average(part.rmse_sgd ** 2, weights=part.n_validation))),
                "latest_rmse_sgd": float(part.loc[part.validation_start.eq("2024-04"), "rmse_sgd"].iloc[0]),
                "stress_16m_rmse_sgd": float(scores.loc[scores.variant.eq(name) & scores.role.eq("stress_16m"), "rmse_sgd"].iloc[0])})
        pooled = pd.DataFrame(pooled).sort_values("pooled_rmse_sgd")
        pooled.to_csv(out / "pooled_metrics.csv", index=False, float_format="%.6f")
        result = {"model": "LightGBM", "settings": settings, "warmup_months": 6,
            "source_sha256": hashes, "pooled_metrics": pooled.to_dict("records"),
            "note": "Separate target-representation experiment; every arm has identical eligible training rows. Earlier labels remain available for causal history construction."}
        (out / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(pooled.to_string(index=False), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_REPORTS / "features_history_optimization")
    parser.add_argument("--variants", nargs="+", choices=tuple(CANDIDATES))
    parser.add_argument("--stage", choices=("features", "market", "refined", "audit", "all"), default="all")
    args = parser.parse_args(argv)
    result = {}
    if args.stage in ("features", "all"):
        result["features"] = run(args.output_dir, args.variants)
    if args.stage in ("market", "all"):
        result["market"] = run_market_decomposition(args.output_dir / "market_decomposition")
    if args.stage in ("refined", "all"):
        result["refined"] = run(args.output_dir / "refined", candidate_specs=REFINED_CANDIDATES)
    if args.stage in ("audit", "all"):
        result["audit"] = audit_history(args.output_dir)
    return result


if __name__ == "__main__":
    main()
