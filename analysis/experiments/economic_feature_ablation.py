"""Compare the E amenity baseline with targeted, prior-month market signals.

Run python analysis/compare_economic_features.py from the repository root.
Every variant uses the same HistGradientBoosting settings, rental rows, and
expanding-window folds. Approval month t uses market data no later than t-1.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from analysis.pipeline.config import AUX, REPORTS
from analysis.pipeline.data import sha256
from analysis.eda.economic_signals import (
    FOLDS, SETTINGS, SOURCES, attach_prior_month, monthly_market,
)
from analysis.pipeline.evaluation import grouped_errors, metrics
from analysis.experiments.auxiliary_ablation import (
    AMENITIES, BASE, BLOCK, feature_frame, model_inputs,
)


OUT = REPORTS / "economic_feature_ablation"
E_FEATURES = BASE + BLOCK + AMENITIES
COE_ENGINEERED = [
    "COE_A_PREMIUM_LAG1", "COE_C_PREMIUM_LAG1",
    "COE_A_GROWTH_LAG1", "COE_C_GROWTH_LAG1",
    "COE_A_BID_PRESSURE_LAG1", "COE_C_BID_PRESSURE_LAG1",
]
SG_STOCK_ENGINEERED = [
    "SG_STOCK_RETURN_LAG1", "SG_STOCK_RETURN_3M_LAG1",
    "SG_STOCK_VOL_3M_LAG1",
]
VARIANTS = {
    "E_amenities": [],
    "E_plus_coe": COE_ENGINEERED,
    "E_plus_sg_stocks": SG_STOCK_ENGINEERED,
    "E_plus_both": COE_ENGINEERED + SG_STOCK_ENGINEERED,
}


def score_deltas(scores: pd.DataFrame) -> pd.DataFrame:
    """Pair each variant with E in the same forward window."""
    result = scores.copy()
    baseline = result[result["variant"].eq("E_amenities")].set_index(
        "validation_start"
    )["rmse_sgd"]
    if len(baseline) != len(FOLDS):
        raise AssertionError("One E baseline score is required per forward fold")
    result["delta_rmse_vs_E_sgd"] = (
        result["rmse_sgd"] - result["validation_start"].map(baseline)
    )
    return result


def prior_e_check(scores: pd.DataFrame) -> dict:
    """Check that E matches the earlier forward experiment."""
    path = REPORTS / "economic_signals" / "forward_scores.csv"
    if not path.exists():
        return {"available": False}
    old = pd.read_csv(path)
    old = old[old["group"].eq("baseline")].set_index("validation_start")["rmse_sgd"]
    current = scores[scores["variant"].eq("E_amenities")].set_index(
        "validation_start"
    )["rmse_sgd"]
    common = old.index.intersection(current.index)
    differences = (old.loc[common] - current.loc[common]).abs()
    return {
        "available": True,
        "matching_folds": int(len(common)),
        "max_abs_rmse_difference_sgd": float(differences.max()),
    }


def write_report(scores: pd.DataFrame, monthly: pd.DataFrame, check: dict) -> None:
    rows = "\n".join(
        f"| {r.validation_start} | {r.variant} | {r.rmse_sgd:.2f} | "
        f"{r.delta_rmse_vs_E_sgd:+.2f} |"
        for r in scores.itertuples(index=False)
    )
    better = {
        variant: int((part["delta_rmse_vs_E_sgd"] < 0).sum())
        for variant, part in scores.groupby("variant")
        if variant != "E_amenities"
    }
    month_rmse = monthly.pivot(
        index=["validation_start", "approval_month"],
        columns="variant", values="rmse_sgd",
    )
    both_month_delta = month_rmse["E_plus_both"] - month_rmse["E_amenities"]
    better_months = int((both_month_delta < 0).sum())
    reproduction = (
        f"The E scores reproduce the earlier forward E baseline within "
        f"{check['max_abs_rmse_difference_sgd']:.5f} SGD RMSE over "
        f"{check['matching_folds']} matching folds."
        if check.get("available") and check.get("matching_folds")
        else "The earlier forward E baseline was not available for a reproduction check."
    )
    lines = [
        "# E baseline versus engineered economic features",
        "",
        "Outcome: individual monthly HDB rent (SGD). Lower RMSE is better; a negative delta improves on E in the same fold.",
        "",
        "| Validation start | Variant | RMSE (SGD) | Delta vs E (SGD) |",
        "| --- | --- | ---: | ---: |",
        rows,
        "",
        "## Design",
        "",
        "E uses the original 29 property, time, block and amenity features. COE adds prior-month A/C premiums, their month-on-month growth, and bids divided by quota. Stocks use only .SI symbols: the median one-month adjusted-close return, compounded three-month return, and rolling three-month volatility of that basket return. E + both uses all nine signals.",
        "",
        "The same HistGradientBoosting model, feature preparation, rental rows and four expanding-window splits are used for every variant. The market observations for approval month t end at t-1. This is a monthly-updated scenario, not a fixed-origin forecast for the entire future test period.",
        "",
        "## Interpretation",
        "",
        "Improved folds (out of four): "
        + ", ".join(f"{name} {count}/4" for name, count in better.items()) + ".",
        (
            f"E + both has lower monthly RMSE in {better_months} of "
            f"{len(both_month_delta)} validation months; it is worse in "
            f"all four complete forward windows."
        ),
        reproduction,
        "This is a fixed-model comparison, not proof that economic signals lack predictive value. There are few independent market months, the last holdout has been examined previously, and the supplied stock-basket membership is a snapshot. Reported row-level errors do not establish causal effects.",
        "",
        "Run python analysis/compare_economic_features.py in the 5228_hw environment. See forward_scores.csv for RMSE, MAE and bias, monthly_errors.csv for temporal drift, feature_coverage.csv for missingness, and results.json for exact feature sets and source hashes.",
        "",
    ]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    source_hashes = {str(path.relative_to(AUX.parent)): sha256(path) for path in SOURCES}
    frame = feature_frame()
    baseline_rows = len(frame)
    market = monthly_market(
        pd.read_csv(AUX / "sg-coe-prices.csv"),
        pd.read_csv(AUX / "sg-stock-prices.csv"),
    )
    frame = attach_prior_month(frame, market)
    if len(frame) != baseline_rows:
        raise AssertionError("Economic join changed the training row count")
    if not set(COE_ENGINEERED + SG_STOCK_ENGINEERED).issubset(frame.columns):
        raise AssertionError("An engineered economic feature is missing")

    scores, monthly_errors, coverage = [], [], []
    for start, end in FOLDS:
        fitting = frame[frame["RENT_APPROVAL_DATE"] < start]
        validation = frame[
            (frame["RENT_APPROVAL_DATE"] >= start)
            & (frame["RENT_APPROVAL_DATE"] < end)
        ]
        if fitting.empty or validation.empty:
            raise ValueError(f"Empty fold: {start} to {end}")
        y_fit = fitting["MONTHLY_RENT"].to_numpy(float)
        y_val = validation["MONTHLY_RENT"].to_numpy(float)
        for variant, extra in VARIANTS.items():
            features = E_FEATURES + extra
            x_fit, x_val, _ = model_inputs(fitting, validation, features)
            model = HistGradientBoostingRegressor(**SETTINGS)
            print(
                f"{start} {variant}: fit {len(fitting)}, "
                f"validate {len(validation)}, features {len(features)}",
                flush=True,
            )
            model.fit(x_fit, y_fit)
            predicted = np.asarray(model.predict(x_val), dtype=float)
            scores.append({
                "validation_start": start, "validation_end_exclusive": end,
                "variant": variant, "n_features": len(features),
                "n_fit": len(fitting), "n_validation": len(validation),
                **metrics(y_val, predicted),
            })
            errors = pd.DataFrame({
                "approval_month": validation["RENT_APPROVAL_DATE"].to_numpy(),
                "actual": y_val, "predicted": predicted,
            })
            monthly = grouped_errors(errors, "approval_month", variant).rename(
                columns={"model": "variant", "group": "approval_month"}
            )
            monthly.insert(0, "validation_start", start)
            monthly_errors.append(monthly)
            for feature in extra:
                coverage.append({
                    "validation_start": start, "variant": variant,
                    "feature": feature,
                    "fit_nonmissing_pct": round(float(x_fit[feature].notna().mean() * 100), 2),
                    "validation_nonmissing_pct": round(float(x_val[feature].notna().mean() * 100), 2),
                })

    OUT.mkdir(parents=True, exist_ok=True)
    scored = score_deltas(pd.DataFrame(scores))
    scored.to_csv(OUT / "forward_scores.csv", index=False, float_format="%.4f")
    monthly_frame = pd.concat(monthly_errors, ignore_index=True)
    monthly_frame.to_csv(OUT / "monthly_errors.csv", index=False, float_format="%.4f")
    pd.DataFrame(coverage).to_csv(OUT / "feature_coverage.csv", index=False)
    check = prior_e_check(scored)
    write_report(scored, monthly_frame, check)
    if source_hashes != {
        str(path.relative_to(AUX.parent)): sha256(path) for path in SOURCES
    }:
        raise AssertionError("An input CSV changed during the experiment")
    (OUT / "results.json").write_text(json.dumps({
        "target": "MONTHLY_RENT (SGD)",
        "prediction_scenario": "monthly-updated; all market features available by end of prior calendar month",
        "baseline_features": E_FEATURES,
        "extra_features_by_variant": VARIANTS,
        "folds": FOLDS,
        "model": "sklearn HistGradientBoostingRegressor",
        "model_settings": SETTINGS,
        "prior_E_reproduction": check,
        "source_sha256": source_hashes,
    }, indent=2), encoding="utf-8")
    print(f"Saved results to {OUT}", flush=True)


if __name__ == "__main__":
    main()
