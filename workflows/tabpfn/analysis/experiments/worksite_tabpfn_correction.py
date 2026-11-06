"""Strictly local residual correction of saved out-of-time TabPFN backtests.

Zero-valued localized inputs imply exactly zero correction: predictions outside
the 3 km kernel cannot change, unlike adding features to a tree backbone.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.config import ROOT, DATA, AUX, REPORTS
from analysis.data import sha256
from analysis.experiments.company_worksite_stocks import rental_coordinates, stock_returns
from analysis.experiments.company_worksite_followup import (
    read_controls, stock_features, weighted_score, paired_month_interval, table,
)


ALPHA = 1000.0


def local_ridge(x, y, weights, query, alpha=ALPHA):
    """Weighted RMS scaling without centering/intercept preserves locality."""
    x, y, weights, query = map(np.asarray, (x, y, weights, query))
    w = weights / weights.mean()
    scale = np.sqrt(np.average(x ** 2, weights=w, axis=0))
    scale[scale < 1e-12] = 1.
    z, q = x / scale, query / scale
    coefficient = np.linalg.solve(z.T @ (w[:, None] * z) + alpha * np.eye(z.shape[1]), z.T @ (w * y))
    return q @ coefficient, coefficient / scale


def run_scenario(scenario):
    config_path = ROOT / "analysis/configs/company_worksite_followup.json"
    config = json.loads(config_path.read_text())
    source = REPORTS / "tabpfn_feature_diversified_ensemble_80_20/predictions_all_origins.csv"
    sites_path = REPORTS / "company_worksite_stocks/worksite_locations.csv"
    out = REPORTS / "company_worksite_followup/tabpfn_local_correction" / scenario
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(DATA / "train.csv")
    frame = rental_coordinates(raw, pd.read_csv(AUX / "sg-hdb-block.csv"))
    sites = pd.read_csv(sites_path)
    pred = pd.read_csv(source)
    if pred.duplicated(["origin", "row_id"]).any():
        raise ValueError("Saved backtests have duplicate query rows")
    np.testing.assert_array_equal(pred.actual, frame.loc[pred.row_id, "MONTHLY_RENT"])
    controls = read_controls()
    returns = stock_returns(pd.read_csv(AUX / "sg-stock-prices.csv"), sites.symbol.unique().tolist(), "2025-03")
    features, audits = [], []
    for origin, part in pred.groupby("origin", sort=False):
        rows = frame.loc[part.row_id].copy()
        cutoff = pd.Period(origin, "M") - 1 if scenario == "fixed_origin" else None
        extra, _, audit = stock_features(rows, sites, returns, controls, [config["primary_hypothesis"]], cutoff)
        extra.index = part.index
        features.append(extra)
        audits.extend({"origin": origin, **r} for r in audit)
    x = pd.concat(features).reindex(pred.index)
    nuisance = ["WS_KLIC_EXPOSURE", "WS_KLIC_LOCAL_SP500", "WS_KLIC_LOCAL_NASDAQSOX", "WS_KLIC_LOCAL_DEXSIUS"]
    x["WS_KLIC_LOCAL_UNAVAILABLE"] = x.WS_KLIC_EXPOSURE * x.WS_KLIC_UNAVAILABLE
    nuisance += ["WS_KLIC_LOCAL_UNAVAILABLE"]
    options = {"sector_local": nuisance, "company_local": nuisance + ["WS_KLIC_LOCAL_COMPANY"]}
    pred["distance_klic_m"] = x.WS_KLIC_DIST_KM * 1000
    residual = pred.actual - pred.optimized
    outputs, coefficients = [], []
    for evaluation, train_origins, query_origin in (("earlier_calibration", ["2022-09"], "2023-07"),
                                                    ("later_reused", ["2022-09", "2023-07"], "2024-06")):
        fitting, query = pred.origin.isin(train_origins), pred.origin == query_origin
        if pred.loc[fitting, "month"].max() >= query_origin:
            raise ValueError("A correction target is not mature before the query origin")
        for name, columns in options.items():
            correction, coefficient = local_ridge(x.loc[fitting, columns].to_numpy(float), residual[fitting].to_numpy(float),
                                                   pred.loc[fitting, "weight"].to_numpy(float), x.loc[query, columns].to_numpy(float))
            if not np.allclose(correction[x.loc[query, "WS_KLIC_EXPOSURE"].to_numpy() == 0], 0, atol=0, rtol=0):
                raise AssertionError("A local correction changed a row outside its spatial support")
            outputs.append(pred.loc[query, ["row_id", "origin", "month", "actual", "weight", "distance_klic_m"]].assign(
                phase=evaluation, variant=name, prediction=pred.loc[query, "optimized"].to_numpy() + correction,
                baseline=pred.loc[query, "optimized"].to_numpy(), correction=correction))
            coefficients.extend({"phase": evaluation, "variant": name, "feature": c, "coefficient": float(v)} for c, v in zip(columns, coefficient))
        outputs.append(pred.loc[query, ["row_id", "origin", "month", "actual", "weight", "distance_klic_m"]].assign(
            phase=evaluation, variant="uncorrected", prediction=pred.loc[query, "optimized"].to_numpy(),
            baseline=pred.loc[query, "optimized"].to_numpy(), correction=0.))
    outputs = pd.concat(outputs, ignore_index=True)
    scores = []
    for (phase, variant), part in outputs.groupby(["phase", "variant"], sort=False):
        for subset, mask in (("all_rentals", np.ones(len(part), bool)), ("klic_within_1km", part.distance_klic_m <= 1000),
                             ("outside_3km", part.distance_klic_m > 3000)):
            p = part[mask]
            scores.append({"phase": phase, "variant": variant, "subset": subset,
                           **weighted_score(p.actual, p.prediction, p.weight)})
    scores = pd.DataFrame(scores)
    calibration = scores[(scores.phase == "earlier_calibration") & (scores.subset == "klic_within_1km")].set_index("variant")
    # One fixed alpha; a 0.1% tie favors the simpler uncorrected/sector model.
    threshold = calibration.rmse_sgd.min() * 1.001
    chosen = next(name for name in ("uncorrected", "sector_local", "company_local") if calibration.loc[name, "rmse_sgd"] <= threshold)
    comparisons = []
    for subset in ("all_rentals", "klic_within_1km", "outside_3km"):
        for candidate, baseline in (("company_local", "sector_local"), ("company_local", "uncorrected"), (chosen, "uncorrected")):
            p = outputs[(outputs.phase == "later_reused") & outputs.variant.isin([candidate, baseline])]
            if subset == "klic_within_1km":
                p = p[p.distance_klic_m <= 1000]
            elif subset == "outside_3km":
                p = p[p.distance_klic_m > 3000]
            s = scores[(scores.phase == "later_reused") & (scores.subset == subset)].set_index("variant")
            comparisons.append({"subset": subset, "candidate": candidate, "baseline": baseline,
                                "earlier_selected": candidate == chosen,
                                "control_rmse": s.loc[baseline, "rmse_sgd"], "candidate_rmse": s.loc[candidate, "rmse_sgd"],
                                "delta_rmse": s.loc[candidate, "rmse_sgd"] - s.loc[baseline, "rmse_sgd"],
                                **paired_month_interval(p, baseline, candidate)})
    comparison = pd.DataFrame(comparisons)
    scores.to_csv(out / "scores.csv", index=False)
    comparison.to_csv(out / "comparisons.csv", index=False)
    pd.DataFrame(coefficients).to_csv(out / "coefficients.csv", index=False)
    pd.DataFrame(audits).to_csv(out / "feature_availability.csv", index=False)
    cache = ROOT / f".cache/company_worksite_followup/tabpfn_correction_predictions_{scenario}.csv.gz"
    outputs.to_csv(cache, index=False, compression="gzip")
    source_paths = [source, config_path, sites_path, DATA / "train.csv", AUX / "sg-hdb-block.csv", AUX / "sg-stock-prices.csv",
                    REPORTS / "company_worksite_followup/market_monthly_log_returns.csv"]
    metadata = {"scenario": scenario, "selected_on_earlier_calibration": chosen, "alpha": ALPHA, "fit_intercept": False,
                "scale": "weighted RMS without centering", "baseline_column": "optimized",
                "saved_baseline_note": "Earlier project's saved feature-ensemble backtests; no TabPFN backbone retrained and not the all-150k submission model.",
                "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in source_paths},
                "code_sha256": {str(Path(__file__).relative_to(ROOT)): sha256(Path(__file__))},
                "cutoffs_verified": True, "outside_support_unchanged": True}
    (out / "results.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report = ["# Strictly local correction of saved TabPFN ensemble backtests", "",
              f"Earlier calibration selects **{chosen}**, with a fixed Ridge alpha of {ALPHA:g}. This compares an unchanged saved TabPFN ensemble, a localized sector/FX/bias correction, and the same correction with one local KLIC return term. These are postprocessors, not a newly fitted TabPFN feature ablation.", "",
              "Fit on the 2022-09-origin out-of-time predictions and choose on the 2023-07-origin queries. Refit using both earlier prediction sets before evaluating the already examined 2024-06-origin period. All calibration rent labels mature before the next origin. Population query weights are retained. " +
              ("Signals follow the fixed-origin calendar cutoff, with zero plus an unavailable indicator for unknown future returns." if scenario == "fixed_origin" else
               "Signals use each query month's historical t-6 observations. The stock inputs update monthly; rental histories in the saved TabPFN predictions stay frozen at their original fold cutoffs."), "",
              "All correction inputs are multiplied by the fixed 1 km Gaussian exposure, truncated at 3 km. Ridge uses weighted RMS scaling, no centering and no intercept. Consequently, corrections are exactly zero beyond 3 km; changing the fitted backbone or affecting unrelated locations cannot explain a local gain here. This is a stricter locality check than the CatBoost feature ablation, where tree topology can change globally.", "",
              table(scores, ["phase", "variant", "subset", "rows", "rmse_sgd"]), "",
              table(comparison, ["subset", "candidate", "baseline", "earlier_selected", "control_rmse", "candidate_rmse", "delta_rmse", "ci_low", "ci_high"]), "",
              "Conditional intervals resample paired three-month blocks. Calibration is sparse near K&S and the later period was already inspected, so this cannot establish causality or fresh confirmation. It uses the saved feature-ensemble backtests, not the all-150k final submission model. No submission file or trained TabPFN backbone was modified.", ""]
    (out / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Saved TabPFN local correction ({scenario}): earlier selection = {chosen}")
    print(comparison[comparison.subset == "klic_within_1km"].to_string(index=False))
    return comparison.assign(scenario=scenario), metadata


def main():
    summaries, metadata = [], {}
    for scenario in ("fixed_origin", "monthly_updated"):
        comparison, result = run_scenario(scenario)
        summaries.append(comparison)
        metadata[scenario] = result
    out = REPORTS / "company_worksite_followup/tabpfn_local_correction"
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(out / "scenario_comparisons.csv", index=False)
    (out / "results.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    text = ["# Strictly local stock correction of saved TabPFN ensemble backtests", "",
            "This deterministic residual correction can change predictions only within 3 km of the frozen K&S site. It compares a local sector/FX/bias adjustment with the same model plus the company's six-month-lag return. All field scaling uses training-only weighted RMS, without centering or intercept, so zero spatial exposure guarantees exactly zero correction. No TabPFN backbone is retrained.", "",
            "The first historical origin fits the correction, the second chooses among no correction, sector-only and company-local correction, and both earlier origins refit before the already examined later period. Fixed-origin stocks stop at the original fold cutoff; monthly-updated stocks use t-6 observations for each approval month. Both preserve the original frozen rental histories. The later period is reused development evidence, not new confirmation.", "",
            table(summary[summary.subset == "klic_within_1km"],
                  ["scenario", "candidate", "baseline", "earlier_selected", "control_rmse", "candidate_rmse", "delta_rmse", "ci_low", "ci_high"]), "",
            "See [fixed-origin details](fixed_origin/report.md) and [monthly-updated details](monthly_updated/report.md) for calibration scores, full-sample errors and unchanged predictions beyond 3 km. Conditional bootstrap intervals omit model/selection uncertainty. This uses an earlier project's saved optimized feature-ensemble backtests, not the all-150k final submission model. Existing submissions are unchanged.", ""]
    (out / "report.md").write_text("\n".join(text), encoding="utf-8")


if __name__ == "__main__":
    main()
