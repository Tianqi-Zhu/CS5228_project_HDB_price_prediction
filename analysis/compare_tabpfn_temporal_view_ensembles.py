"""Focused temporal and origin-frozen macro ensembles with earlier-only weights."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.config import ROOT, AUX, REPORTS

# Older saved folds contain pandas3 StringDtype pickles. Use the already
# cached matching wheel locally without changing the shared environment or
# dependencies used by an active submission process.
_expected_pandas = json.loads((REPORTS / "tabpfn_feature_diversified_ensemble_80_20/settings.json").read_text())["packages"]["pandas"]
if version("pandas") != _expected_pandas:
    _cached = sorted((ROOT / ".uv-cache/archive-v0").glob(f"*/pandas-{_expected_pandas}.dist-info"))
    if not _cached:
        raise RuntimeError(f"Saved fold caches require pandas{_expected_pandas}; matching cached wheel unavailable")
    sys.path.insert(0, str(_cached[0].parent))
import joblib
import numpy as np
import pandas as pd

from analysis.data import sha256
from analysis.compare_local_economy_landscape import immutable, write_json, hashlib_array
from analysis.compare_tabpfn_hierarchical_latent import load_bundle, BASE, feature_source, raw_frame
from analysis.compare_tabpfn_v3_5_fast_80_20 import CHECKPOINT
from analysis import compare_tabpfn_feature_ensemble as prior
from analysis.eda.economic_signals import monthly_market
from analysis.experiments.temporal_feature_views import temporal_view, simplex_weights
from analysis.experiments.origin_macro_views import macro_view
from analysis.experiments.feature_diversified_ensemble import combine_predictions
from analysis.experiments.hierarchical_latent_residual import weighted_metrics


OUT = REPORTS / "tabpfn_temporal_view_ensembles_80_20"
CACHE = ROOT / ".cache/tabpfn_temporal_view_ensembles_80_20"
REFERENCE = prior.OUT
ORIGINS = ["2022-09", "2023-07", "2024-06"]
NEW = ["relative_history", "horizon_dynamics", "coe_origin", "stocks_origin"]
MEMBERS = ["baseline", *NEW]
SCHEMES = {
    "dynamics_optimized": ["baseline", "relative_history", "horizon_dynamics"],
    "macro_optimized": ["baseline", "coe_origin", "stocks_origin"],
    "all_optimized": MEMBERS,
    **{f"blend_{m}": ["baseline", m] for m in NEW},
}
DESCRIPTIONS = {
    "relative_history": "Replace19 absolute historical/forecast rent levels by level minus original per-row residual baseline;77 features",
    "horizon_dynamics": "Retain77 plus log horizon, type momentum x log horizon, horizon-decayed local/block premiums, area momentum and block support;83 features",
    "coe_origin": "Retain77 plus category A/C log premiums,3-month log changes,bid pressures,two damped-horizon momentum interactions,missing flag;86 features",
    "stocks_origin": "Retain77 plus .SI basket1/3-month returns,3-month volatility,coverage,damped-horizon momentum,missing flag;83 features",
}


def verify_sources():
    settings = json.loads((OUT / "settings.json").read_text())
    for group in ("source_sha256", "code_sha256"):
        for path, digest in settings[group].items():
            assert sha256(ROOT / path) == digest, path
    return settings


def load_prepared(origin, member):
    path = CACHE / f"features_{origin}_{member}.joblib"
    manifest = json.loads((OUT / f"features_{origin}_{member}.json").read_text())
    assert sha256(path) == manifest["sha256"]
    return joblib.load(path)


def dates_and_origins(bundle, split):
    x = bundle[f"X_{split}"]
    indices = x.APPROVAL_MONTH_INDEX.to_numpy(int)
    months = pd.period_range("2021-01", periods=int(indices.max()) + 1, freq="M").take(indices)
    horizon = x.TEMP_FORECAST_HORIZON.to_numpy(float)
    origins = (months - np.nan_to_num(horizon, nan=0).astype(int)).where(np.isfinite(horizon))
    if f"{split}_month" in bundle:
        np.testing.assert_array_equal(months, pd.PeriodIndex(bundle[f"{split}_month"], freq="M"))
    if f"{split}_temporal_origins" in bundle:
        saved = bundle[f"{split}_temporal_origins"].reindex(x.index).forecast_origin
        np.testing.assert_array_equal(origins.asi8, pd.PeriodIndex(saved, freq="M").asi8)
    return months, pd.Series(origins, index=x.index)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    prior.check_sources()
    sources = [CHECKPOINT, BASE, AUX / "sg-coe-prices.csv", AUX / "sg-stock-prices.csv",
               REFERENCE / "settings.json", REFERENCE / "selection.json"]
    for origin in ORIGINS[:-1]:
        sources.extend(feature_source(origin))
    for origin in ORIGINS:
        for member in ("baseline", "temporal", "spatial", "standard6"):
            sources.extend([REFERENCE / f"predictions_{origin}_{member}.csv", REFERENCE / f"timing_{origin}_{member}.json"])
    submission = REPORTS / "tabpfn_diagscale_mse_final_submission/submission_tabpfn_diagscale_mse_150k.csv"
    sources.append(submission)
    code = [Path(__file__), ROOT / "analysis/experiments/temporal_feature_views.py",
            ROOT / "analysis/experiments/origin_macro_views.py", ROOT / "analysis/eda/economic_signals.py",
            ROOT / "analysis/compare_tabpfn_feature_ensemble.py",
            ROOT / "analysis/compare_tabpfn_hierarchical_latent.py",
            ROOT / "analysis/experiments/feature_diversified_ensemble.py"]
    immutable(OUT / "settings.json", {
        "objective": "Test focused temporal representations and isolated origin-frozen COE/stock views without changing residual target",
        "origins": ORIGINS, "new_members": NEW, "definitions": DESCRIPTIONS,
        "feature_counts": dict(relative_history=77, horizon_dynamics=83, coe_origin=86, stocks_origin=83),
        "prespecified_ensembles": SCHEMES,
        "primary_comparisons": ["dynamics_optimized", "macro_optimized"],
        "secondary_comparison": "all_optimized has up to10 internal views, exceeds six-view control budget",
        "inference": "Frozen V3_5_FAST,seed42,two estimators,full unchanged context,native default estimator batching,batch200,original residual target/baseline",
        "selection": "Nonnegative sum-to-one MSE weights from earlier fixed-origin predictions only; no outer selection or calibration",
        "nested_protocol": "September2022 baseline fallback;July2023 uses only September2022;June2024 uses both earlier disjoint windows",
        "macro_availability": "Use market observations <= original per-row forecast_origin minus1month; no target-month refresh,backfill or future realized market values",
        "macro_warmup": "Native numeric NaN plus explicit history-missing flag; no future-filled startup history",
        "macro_limitations": "Supplied snapshot,not archived vintages; .SI basket is not the STI; adjusted closes/basket membership may be revised; extra month buffers unknown release timing",
        "evaluation": "Repeatedly examined development backtests,not untouched test;earlier200 queries/month with population weights;outer28995 queries;all contexts retained",
        "decision": "Report both primary ensembles without outer-based winner selection;candidate only if >=0.1% main RMSE gain,July nested transfer improves,month/building bootstrap upper95<0,and does not lose to six-view control;no deployment",
        "source_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in sources},
        "code_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in code},
        "packages": {p: version(p) for p in ("numpy", "pandas", "torch", "tabpfn", "scipy")},
    })
    verify_sources()
    if (OUT / "prepared_complete.json").exists():
        for origin in ORIGINS:
            for member in NEW: load_prepared(origin, member)
        print("PREPARED caches verified", flush=True); return
    market = monthly_market(pd.read_csv(AUX / "sg-coe-prices.csv"), pd.read_csv(AUX / "sg-stock-prices.csv"))
    coverage, audits = [], []
    for origin in ORIGINS:
        bundle = load_bundle(origin)
        assert bundle["X_train"].shape[1] == 77
        assert dates_and_origins(bundle, "train")[0].max() < pd.Period(origin, "M") <= dates_and_origins(bundle, "query")[0].min()
        assert set(bundle["X_train"].index).isdisjoint(bundle["X_query"].index)
        for member in NEW:
            views = {}
            for split in ("train", "query"):
                x = bundle[f"X_{split}"]
                months, origins = dates_and_origins(bundle, split)
                if member in ("coe_origin", "stocks_origin"):
                    z = macro_view(x, months, origins, market, member)
                    # Replay three distinct row origins, poisoning every source
                    # month unavailable under that row's cutoff, not just future queries.
                    unique = sorted(origins.dropna().unique())
                    chosen = unique[::max(1, len(unique) // 3)][:3]
                    for cut in chosen:
                        positions = np.flatnonzero(origins.eq(cut).to_numpy())[:20]
                        corrupted = market.copy(); corrupted.loc[corrupted.index >= cut] = 1e12
                        replay = macro_view(x.iloc[positions], np.asarray(months)[positions], origins.iloc[positions], corrupted, member)
                        pd.testing.assert_frame_equal(z.iloc[positions], replay)
                        audits.append(dict(origin=origin, split=split, member=member, feature_origin=str(cut),
                                           latest_market_month=str(cut - 1), rows=len(positions), unavailable_market_poisoning_no_effect=True))
                else:
                    z = temporal_view(x, months, bundle[f"{split}_baseline"], member)
                assert z.shape[1] == {"relative_history": 77, "horizon_dynamics": 83, "coe_origin": 86, "stocks_origin": 83}[member]
                assert z.index.equals(x.index)
                assert not np.isinf(z.select_dtypes("number").to_numpy(float)).any()
                unchanged = [c for c in x if c in z]
                pd.testing.assert_frame_equal(z[unchanged], x[unchanged])
                views[f"X_{split}"] = z
                for column in z:
                    if column not in x:
                        coverage.append(dict(origin=origin, member=member, split=split, feature=column,
                                             missing=int(z[column].isna().sum()), rows=len(z)))
            path = CACHE / f"features_{origin}_{member}.joblib"
            joblib.dump({**bundle, **views}, path, compress=3)
            x, z = bundle["X_train"], views["X_train"]
            write_json(OUT / f"features_{origin}_{member}.json", dict(
                sha256=sha256(path), origin=origin, member=member, feature_names=list(z),
                added=[c for c in z if c not in x], removed=[c for c in x if c not in z],
                categorical_positions=[i for i, c in enumerate(z) if isinstance(z[c].dtype, pd.CategoricalDtype)],
                context_rows=len(z), query_rows=len(views["X_query"]),
                context_ids_sha256=hashlib_array(z.index), query_ids_sha256=hashlib_array(views["X_query"].index),
                original_residual_baselines_and_raw_targets_unchanged=True))
            print("PREPARED", origin, member, z.shape, flush=True)
    pd.DataFrame(coverage).to_csv(OUT / "feature_coverage.csv", index=False)
    write_json(OUT / "macro_causality_audit.json", dict(verified=True, checks=audits, raw_query_targets_never_accepted=True))
    write_json(OUT / "prepared_complete.json", dict(verified=True, variants=NEW, origins=ORIGINS))


def read_prediction(origin, member):
    directory = OUT if member in NEW else REFERENCE
    path = directory / f"predictions_{origin}_{member}.csv"
    timing = json.loads((directory / f"timing_{origin}_{member}.json").read_text())
    assert sha256(path) == timing["prediction_sha256"]
    assert timing["parameters_unchanged"] and timing["no_adapters"] and not timing["subsampling"]
    if member != "standard6": assert timing["n_estimators"] == 2
    return pd.read_csv(path, float_precision="round_trip"), timing


def aligned(origin):
    first, _ = read_prediction(origin, "baseline")
    result = first.drop(columns="predicted").copy()
    for member in MEMBERS:
        other, _ = read_prediction(origin, member)
        for c in ("row_id", "origin", "month", "actual", "weight"):
            np.testing.assert_array_equal(result[c], other[c])
        result[member] = other.predicted.to_numpy()
    assert result.row_id.is_unique and np.isfinite(result[MEMBERS]).all().all()
    return result


def fit_weights(frame):
    models = {}
    for label, members in SCHEMES.items():
        w, info = simplex_weights(frame.actual, frame[members], frame.weight)
        models[label] = dict(members=members, weights=w.tolist(), **info)
    return models


def select():
    verify_sources()
    earlier = [aligned(o) for o in ORIGINS[:-1]]
    pool = pd.concat(earlier, ignore_index=True)
    assert not pool.row_id.duplicated().any() and pool.month.max() < ORIGINS[-1]
    selection = dict(evaluation_labels_used=False, maximum_selection_label_month=pool.month.max(),
                     selection_cases=len(pool), selection_ids_sha256=hashlib_array(pool.row_id), models=fit_weights(pool),
                     prediction_sha256={f"{o}_{m}": sha256((OUT if m in NEW else REFERENCE) / f"predictions_{o}_{m}.csv")
                                        for o in ORIGINS[:-1] for m in MEMBERS})
    immutable(OUT / "selection.json", selection)
    nested = {
        ORIGINS[0]: dict(cold_start=True, models={n: dict(members=m, weights=[1.] + [0.] * (len(m) - 1)) for n, m in SCHEMES.items()}),
        ORIGINS[1]: dict(cold_start=False, maximum_selection_label_month=earlier[0].month.max(), models=fit_weights(earlier[0])),
    }
    assert nested[ORIGINS[1]]["maximum_selection_label_month"] < ORIGINS[1]
    immutable(OUT / "nested_selection.json", nested)
    print("LOCKED EARLIER-ONLY WEIGHTS", json.dumps(selection["models"]), flush=True)


def wait_for_gpu():
    import psutil
    # Existing training/submission processes are left running. Wait for the
    # whole pipeline, including the small gaps between its GPU subprocesses.
    while True:
        busy = []
        for process in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmd = " ".join(process.info["cmdline"] or [])
                if process.info["name"] and process.info["name"].lower() == "python.exe" and (
                    "train_tabpfn_" in cmd or "compare_tabpfn_" in cmd) and Path(__file__).name not in cmd:
                    busy.append(process.info["pid"])
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
        if not busy: return
        write_json(OUT / "status.json", dict(stage="waiting_for_existing_gpu_jobs", pids=busy))
        print("WAITING for existing GPU pipelines", busy, flush=True)
        time.sleep(30)


def infer(origin, member):
    verify_sources(); wait_for_gpu()
    # Reuse exactly the audited frozen inference implementation in this
    # isolated child process; its previous artifacts and sources stay intact.
    prior.OUT, prior.CACHE = OUT, CACHE
    prior.check_sources, prior.load_prepared = verify_sources, load_prepared
    prior.infer(origin, member)


def child(origin, member):
    log = OUT / f"inference_{origin}_{member}.log"
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    for key in ("TABPFN_MAX_BATCHED_ESTIMATOR_ROWS", "TABPFN_MAX_BATCHED_ESTIMATOR_CELLS"):
        env.pop(key, None)
    with log.open("a", encoding="utf-8") as stream:
        result = subprocess.run([sys.executable, "-u", str(Path(__file__)), "--stage", "one", "--origin", origin,
                                 "--member", member], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError(f"Inference failed; inspect {log}")
    print("COMPLETED", origin, member, flush=True)


def materialize(origin, selection, nested):
    f = aligned(origin)
    schemes = selection["models"] if origin == ORIGINS[-1] else nested[origin]["models"]
    for name, spec in schemes.items(): f[name] = combine_predictions(f[spec["members"]], spec["weights"])
    for name, members in (("dynamics_equal", SCHEMES["dynamics_optimized"]), ("macro_equal", SCHEMES["macro_optimized"]), ("all_equal", MEMBERS)):
        f[name] = combine_predictions(f[members], np.full(len(members), 1 / len(members)))
    for member in ("temporal", "spatial", "standard6"):
        other, _ = read_prediction(origin, member)
        for c in ("row_id", "origin", "month", "actual", "weight"): np.testing.assert_array_equal(f[c], other[c])
        f[f"prior_{member}"] = other.predicted.to_numpy()
    previous = json.loads((REFERENCE / ("selection.json" if origin == ORIGINS[-1] else "nested_selection.json")).read_text())
    spec = previous["models"]["optimized"] if origin == ORIGINS[-1] else previous[origin]["models"]["optimized"]
    f["prior_ensemble"] = combine_predictions(f[["baseline", "prior_spatial", "prior_temporal"]], spec["weights"])
    return f, schemes


def uncertainty(frame, names):
    rows = []; rng = np.random.default_rng(42)
    models = ["baseline", "prior_standard6", *names]
    for dimension in ("month", "building"):
        groups = list(frame.groupby(dimension, sort=True).indices.values())
        den = np.array([frame.weight.iloc[idx].sum() for idx in groups])
        sums = np.array([[np.sum(frame.weight.iloc[idx] * (frame[name].iloc[idx] - frame.actual.iloc[idx])**2)
                          for name in models] for idx in groups])
        count = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=3000)
        error = np.sqrt((count @ sums) / (count @ den)[:, None])
        for name in names:
            for ref in ("baseline", "prior_standard6"):
                difference = error[:, models.index(name)] - error[:, models.index(ref)]
                rows.append(dict(cluster=dimension, model=name, reference=ref, replicates=3000,
                                 lower95_sgd=float(np.quantile(difference, .025)), upper95_sgd=float(np.quantile(difference, .975))))
    return pd.DataFrame(rows)


def table(frame):
    rows = ["| " + " | ".join(map(str, frame.columns)) + " |",
            "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = [f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v) for v in row]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def summarize():
    verify_sources()
    selection = json.loads((OUT / "selection.json").read_text()); nested = json.loads((OUT / "nested_selection.json").read_text())
    raw = raw_frame(); scores, costs, weights, subgroups, frames = [], [], [], [], []
    model_names = MEMBERS + list(SCHEMES) + ["dynamics_equal", "macro_equal", "all_equal", "prior_temporal", "prior_standard6", "prior_ensemble"]
    for origin in ORIGINS:
        f, schemes = materialize(origin, selection, nested)
        b = load_bundle(origin)
        np.testing.assert_array_equal(f.row_id, b["X_query"].index)
        np.testing.assert_array_equal(f.actual, b["y_query"])
        f["building"] = raw.loc[f.row_id, "building"].to_numpy()
        f["flat_type"] = raw.loc[f.row_id, "FLAT_TYPE_CLEAN"].to_numpy()
        f["town"] = raw.loc[f.row_id, "TOWN_CLEAN"].to_numpy()
        f["horizon"] = pd.PeriodIndex(f.month, freq="M").asi8 - pd.Period(origin, "M").ordinal + 1
        base = weighted_metrics(f.actual, f.baseline, f.weight)["rmse_sgd"]
        timings = {member: read_prediction(origin, member)[1] for member in MEMBERS + ["standard6", "temporal", "spatial"]}
        cost_members = {m: [m] for m in MEMBERS}
        for name, spec in schemes.items():
            cost_members[name] = [m for m, w in zip(spec["members"], spec["weights"]) if w > 0]
            weights.append(dict(origin=origin, model=name, fit_label_max=selection["maximum_selection_label_month"] if origin == ORIGINS[-1] else nested[origin].get("maximum_selection_label_month", "cold_start"),
                                **{m: dict(zip(spec["members"], spec["weights"])).get(m, 0.) for m in MEMBERS}))
        cost_members.update(dynamics_equal=SCHEMES["dynamics_optimized"], macro_equal=SCHEMES["macro_optimized"], all_equal=MEMBERS,
                            prior_temporal=["temporal"], prior_standard6=["standard6"])
        for name in model_names:
            result = weighted_metrics(f.actual, f[name], f.weight)
            scores.append(dict(origin=origin, model=name, n=len(f), **result, rmse_difference_sgd=result["rmse_sgd"] - base,
                               rmse_improvement_pct=100 * (base - result["rmse_sgd"]) / base))
            if name in cost_members:
                active = [timings[m] for m in cost_members[name]]
                costs.append(dict(origin=origin, model=name, fit_seconds=sum(t["fit_seconds"] for t in active),
                                  predict_seconds=sum(t["predict_seconds"] for t in active), peak_vram_gib=max(t["peak_vram_gib"] for t in active),
                                  internal_estimators=sum(t["n_estimators"] for t in active)))
        for dimension in ("month", "horizon", "flat_type", "town"):
            for value, group in f.groupby(dimension, observed=True):
                for name in ["baseline", "dynamics_optimized", "macro_optimized", "all_optimized", "prior_standard6"]:
                    subgroups.append(dict(origin=origin, dimension=dimension, group=str(value), model=name, n=len(group),
                                          **weighted_metrics(group.actual, group[name], group.weight)))
        errors = f[MEMBERS].to_numpy() - f.actual.to_numpy()[:, None]
        v = f.weight.to_numpy(); v = v / v.sum(); centered = errors - np.sum(v[:, None] * errors, axis=0)
        cov = centered.T @ (v[:, None] * centered)
        corr = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
        pd.DataFrame(corr, index=MEMBERS, columns=MEMBERS).to_csv(OUT / f"error_correlations_{origin}.csv")
        frames.append(f)
    scores, costs, weights = pd.DataFrame(scores), pd.DataFrame(costs), pd.DataFrame(weights)
    scores.to_csv(OUT / "comparison.csv", index=False); costs.to_csv(OUT / "costs.csv", index=False)
    weights.to_csv(OUT / "learned_weights.csv", index=False); pd.DataFrame(subgroups).to_csv(OUT / "subgroup_scores.csv", index=False)
    pd.concat(frames, ignore_index=True).to_csv(OUT / "predictions_all_origins.csv", index=False, float_format="%.17g")
    primary = ["dynamics_optimized", "macro_optimized", "all_optimized"]
    intervals = uncertainty(frames[-1], primary); intervals.to_csv(OUT / "paired_cluster_bootstrap.csv", index=False)
    main = scores[scores.origin.eq(ORIGINS[-1])].set_index("model")
    transfer = scores[scores.origin.eq(ORIGINS[1])].set_index("model")
    decisions = {}
    for name in primary:
        ci = intervals[intervals.model.eq(name) & intervals.reference.eq("baseline")]
        decisions[name] = dict(main_rmse_sgd=float(main.loc[name, "rmse_sgd"]),
            earlier_nested_transfer_improves=bool(transfer.loc[name, "rmse_sgd"] < transfer.loc["baseline", "rmse_sgd"]),
            candidate_for_independent_confirmation=bool(main.loc[name, "rmse_improvement_pct"] >= .1
                and transfer.loc[name, "rmse_sgd"] < transfer.loc["baseline", "rmse_sgd"]
                and (ci.upper95_sgd < 0).all() and main.loc[name, "rmse_sgd"] <= main.loc["prior_standard6", "rmse_sgd"]))
    write_json(OUT / "decision.json", dict(models=decisions, production_submission_unchanged=True, untouched_test_available=False,
                                          deployment="No automatic deployment; these are exploratory development results"))
    report = ["# Focused temporal and macro feature ensembles", "",
        "Four new frozen TabPFN views were defined before scoring. All retain the original raw targets, residual baselines, contexts, chronology and native-default two-estimator inference. No fine-tuning occurs.", "",
        "## Feature definitions", "", *[f"- **{k}**: {v}." for k, v in DESCRIPTIONS.items()], "",
        "COE/stock inputs freeze at each original row's forecast origin with one extra month of publication buffer. A June2024 query therefore reads market data through April2024 even for March2025 targets. No realized target-month macro data enters. Stocks are the supplied .SI median-return basket, not the STI. Startup NaN and explicit missing flags are retained; snapshot membership/adjusted-close vintages remain limitations.", "",
        "Relative histories use the original per-row forecast baseline, never observed rent. Horizon interactions are deterministic transforms of existing causal features; they add representation, not new observations.", "",
        "## Main development results", "", table(main.reset_index()[["model", "rmse_sgd", "mae_sgd", "bias_sgd", "rmse_difference_sgd", "rmse_improvement_pct"]]), "",
        "## Earlier-only ensemble weights", "", table(weights[weights.origin.eq(ORIGINS[-1])]), "",
        "Weights minimize original-SGD MSE with nonnegative sum-to-one constraints. Main weights use the two disjoint earlier2000-query windows, through April2024. September2022 uses baseline-only cold start; July2023 uses only the September2022 window. No outer labels select weights or a winning specification.", "",
        "## Chronological transfer", "", table(scores[scores.model.isin(["baseline", *primary, "prior_standard6"])][["origin", "model", "rmse_sgd", "mae_sgd", "rmse_difference_sgd"]]), "",
        "## Paired descriptive uncertainty", "", table(intervals), "",
        "3000 paired bootstrap replicates, seed42, separately by month/building. Ten adjacent month clusters, repeated development inspection and unaccounted weight-estimation uncertainty limit interpretation.", "",
        "## Inference cost", "", table(costs[costs.origin.eq(ORIGINS[-1])]), "",
        "Dynamics and macro triples each have at most six views, comparable to the reused same-feature seeds42/43/44 six-view control. The combined five-member ensemble uses up to ten views and has no exact ten-view compute control. Timings exclude feature preparation; reused controls were measured in a different session. Costs sum sequential active-member runs, and VRAM is their maximum.", "",
        "## Decision", "", "```json", json.dumps(decisions, indent=2), "```", "",
        "All results are development backtests on previously examined labels, not an untouched test. Inherited static geography/MRT timing limitations remain. No submission is generated or replaced by this experiment.", "",
        "Reproduce/resume: `.venv/Scripts/python.exe analysis/compare_tabpfn_temporal_view_ensembles.py --stage all`."]
    (OUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    verify_sources()
    write_json(OUT / "verification.json", dict(verified=True, prediction_alignment=True, finite_predictions=True,
        earlier_only_weights=True, all_context_rows_retained=True, original_targets_and_baselines=True,
        source_hashes_unchanged=True, no_submission_writes=True))
    write_json(OUT / "status.json", dict(stage="complete"))
    print(main[["rmse_sgd", "rmse_difference_sgd"]].to_string(), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["prepare", "earlier", "select", "outer", "one", "summarize", "all"], default="all")
    p.add_argument("--origin", choices=ORIGINS); p.add_argument("--member", choices=NEW)
    args = p.parse_args()
    if args.stage in ("prepare", "all"): prepare()
    if args.stage in ("earlier", "all"):
        for origin in ORIGINS[:-1]:
            for member in NEW: child(origin, member)
    if args.stage in ("select", "all"): select()
    if args.stage in ("outer", "all"):
        assert (OUT / "selection.json").exists()
        for member in NEW: child(ORIGINS[-1], member)
    if args.stage == "one": infer(args.origin, args.member)
    if args.stage in ("summarize", "all"): summarize()


if __name__ == "__main__": main()
