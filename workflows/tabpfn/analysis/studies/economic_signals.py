"""Compare lagged COE and Singapore-stock signals with rent and forward RMSE.

Run ``python -m analysis.main economic-signals`` from the repository root.
Market features describe a monthly-updated forecast: approval month t uses
market observations through t-1, never market observations from t.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from analysis.pipeline.config import AUX, DATA, STUDY_REPORTS, VALIDATION_START
from analysis.pipeline.data import sha256
from analysis.studies.temporal_correlations import mix_adjusted_monthly
from analysis.pipeline.evaluation import metrics
from analysis.experiments.baselines.auxiliary_ablation import (
    AMENITIES, BASE, BLOCK, feature_frame, model_inputs,
)


OUT = STUDY_REPORTS / "economic_signals"
SOURCES = [
    DATA / "train.csv", AUX / "sg-coe-prices.csv", AUX / "sg-stock-prices.csv",
    AUX / "sg-hdb-block.csv", AUX / "sg-mrt-stations.csv",
    AUX / "sg-shopping-malls.csv", AUX / "sg-schools.csv",
]
BASELINE = BASE + BLOCK + AMENITIES
GROUPS = {
    "baseline": [],
    "coe_levels": ["COE_A_PREMIUM_LAG1", "COE_C_PREMIUM_LAG1"],
    "coe_changes_pressure": [
        "COE_A_GROWTH_LAG1", "COE_C_GROWTH_LAG1",
        "COE_A_BID_PRESSURE_LAG1", "COE_C_BID_PRESSURE_LAG1",
    ],
    "sg_stocks": ["SG_STOCK_RETURN_LAG1", "SG_STOCK_RETURN_3M_LAG1", "SG_STOCK_VOL_3M_LAG1"],
    "coe_changes_plus_stocks": [
        "COE_A_GROWTH_LAG1", "COE_C_GROWTH_LAG1",
        "COE_A_BID_PRESSURE_LAG1", "COE_C_BID_PRESSURE_LAG1",
        "SG_STOCK_RETURN_LAG1", "SG_STOCK_RETURN_3M_LAG1", "SG_STOCK_VOL_3M_LAG1",
    ],
}
FOLDS = [
    ("2022-07", "2023-01"),
    ("2023-01", "2023-07"),
    ("2023-07", VALIDATION_START),
    (VALIDATION_START, "2025-04"),
]
SETTINGS = dict(max_iter=200, learning_rate=.05, max_leaf_nodes=31,
                min_samples_leaf=30, l2_regularization=3.0,
                categorical_features="from_dtype", early_stopping=False,
                random_state=42)


def _number(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values.astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")


def monthly_market(coe: pd.DataFrame, stocks: pd.DataFrame) -> pd.DataFrame:
    """Aggregate source observations by calendar month, preserving missing months."""
    c = coe.copy()
    date = pd.to_datetime(c["YEAR"].astype(str) + "-" + c["MONTH"].astype(str).str[:3] + "-01",
                          format="%Y-%b-%d", errors="coerce")
    c["month"] = date.dt.to_period("M")
    c["CATEGORY"] = c["CATEGORY"].astype(str).str.upper()
    for col in ("QUOTA_PREMIUM", "QUOTA", "BIDS_RECEIVED"):
        c[col] = _number(c[col])
    c = c.dropna(subset=["month", "CATEGORY"])
    rows = []
    for (month, category), part in c.groupby(["month", "CATEGORY"]):
        quota, bids = part["QUOTA"].sum(min_count=1), part["BIDS_RECEIVED"].sum(min_count=1)
        rows.append({"month": month, "category": category,
                     "premium": part["QUOTA_PREMIUM"].median(),
                     "bid_pressure": bids / quota if pd.notna(quota) and quota > 0 else np.nan})
    grouped = pd.DataFrame(rows)
    coe_wide = grouped.pivot(index="month", columns="category", values=["premium", "bid_pressure"])
    coe_wide.columns = [f"COE_{cat}_{'PREMIUM' if kind == 'premium' else 'BID_PRESSURE'}"
                        for kind, cat in coe_wide.columns]

    s = stocks.copy()
    s["DATE"] = pd.to_datetime(s["DATE"], errors="coerce")
    s["ADJUSTED_CLOSE"] = pd.to_numeric(s["ADJUSTED_CLOSE"], errors="coerce")
    s = s[s["SYMBOL"].astype(str).str.endswith(".SI")].dropna(subset=["DATE", "ADJUSTED_CLOSE"])
    s = s.sort_values(["SYMBOL", "DATE"]).copy()
    s["month"] = s["DATE"].dt.to_period("M")
    closes = s.groupby(["SYMBOL", "month"])["ADJUSTED_CLOSE"].last().unstack("month").T
    full = pd.period_range(min(coe_wide.index.min(), closes.index.min()),
                           max(coe_wide.index.max(), closes.index.max()), freq="M")
    closes = closes.reindex(full)
    returns = closes.pct_change(fill_method=None)
    market = coe_wide.reindex(full)
    for category in ("A", "B", "C", "E"):
        column = f"COE_{category}_PREMIUM"
        if column in market:
            market[f"COE_{category}_GROWTH"] = market[column].pct_change(fill_method=None)
    market["SG_STOCK_RETURN"] = returns.median(axis=1)
    market["SG_STOCK_COVERAGE"] = returns.notna().sum(axis=1)
    market["SG_STOCK_RETURN_3M"] = (1 + market["SG_STOCK_RETURN"]).rolling(3, min_periods=3).apply(np.prod, raw=True) - 1
    market["SG_STOCK_VOL_3M"] = market["SG_STOCK_RETURN"].rolling(3, min_periods=3).std()
    market.index.name = "month"
    return market


def attach_prior_month(frame: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Join signals from the preceding calendar month, not preceding row."""
    month = pd.to_datetime(frame["RENT_APPROVAL_DATE"], format="%Y-%m").dt.to_period("M")
    full = pd.period_range(min(market.index.min(), month.min()),
                           max(market.index.max(), month.max()), freq="M")
    shifted = market.reindex(full).shift(1).add_suffix("_LAG1")
    result = frame.copy()
    result["_month"] = month
    return result.join(shifted, on="_month").drop(columns="_month")


def correlations(monthly: pd.DataFrame) -> pd.DataFrame:
    """Descriptive rent-growth correlations before the existing holdout only."""
    fitting = monthly[monthly.index.to_timestamp() < pd.Timestamp(VALIDATION_START)]
    chosen = sorted(set(sum(GROUPS.values(), [])))
    rows = []
    for column in chosen:
        for segment, part in (("all_prevalidation", fitting),
                              ("early", fitting.iloc[:len(fitting) // 2]),
                              ("late", fitting.iloc[len(fitting) // 2:])):
            pair = part[["rent_adjusted_growth", column]].replace([np.inf, -np.inf], np.nan).dropna()
            rows.append({"feature": column, "segment": segment, "months": len(pair),
                         "pearson_r": pair.corr().iloc[0, 1] if len(pair) >= 4 else np.nan,
                         "spearman_rho": pair.corr(method="spearman").iloc[0, 1] if len(pair) >= 4 else np.nan})
    return pd.DataFrame(rows)


def forward_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Expanding-window folds; same baseline rows and model for every candidate."""
    rows = []
    for start, end in FOLDS:
        fitting = frame[frame["RENT_APPROVAL_DATE"] < start]
        validation = frame[(frame["RENT_APPROVAL_DATE"] >= start) & (frame["RENT_APPROVAL_DATE"] < end)]
        if fitting.empty or validation.empty:
            raise ValueError(f"Empty forward fold: {start}..{end}")
        for group, extra in GROUPS.items():
            features = BASELINE + extra
            x_fit, x_val, _ = model_inputs(fitting, validation, features)
            model = HistGradientBoostingRegressor(**SETTINGS)
            print(f"{start} {group}: fitting {len(fitting)} / validating {len(validation)}", flush=True)
            model.fit(x_fit, fitting["MONTHLY_RENT"])
            score = metrics(validation["MONTHLY_RENT"].to_numpy(float), model.predict(x_val))
            rows.append({"validation_start": start, "validation_end_exclusive": end,
                         "group": group, "n_fit": len(fitting), "n_validation": len(validation),
                         **score})
    result = pd.DataFrame(rows)
    baseline = result[result.group.eq("baseline")].set_index("validation_start")["rmse_sgd"]
    result["delta_rmse_vs_baseline_sgd"] = result["rmse_sgd"] - result["validation_start"].map(baseline)
    return result


def plot_signals(monthly: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, layout="constrained")
    dates = monthly.index.to_timestamp()
    spec = [
        ("COE_A_PREMIUM_LAG1", "COE A", "#d17a31", 0),
        ("COE_C_PREMIUM_LAG1", "COE C", "#8957a1", 0),
        ("COE_A_BID_PRESSURE_LAG1", "A bid/available quota", "#d17a31", 1),
        ("COE_C_BID_PRESSURE_LAG1", "C bid/available quota", "#8957a1", 1),
        ("SG_STOCK_RETURN_LAG1", ".SI median return", "#228c73", 2),
        ("SG_STOCK_RETURN_3M_LAG1", ".SI three-month return", "#2474a6", 2),
    ]
    for col, label, color, panel in spec:
        y = monthly[col] * (100 if panel == 2 else 1)
        axes[panel].plot(dates, y, label=label, color=color, linewidth=1.7)
    for ax, title, unit in zip(axes, ("COE premium (SGD)", "COE bidding pressure", "Singapore-stock returns"),
                               ("SGD", "Bids / quota", "%")):
        ax.set(title=title, ylabel=unit)
        ax.axvline(pd.Timestamp(VALIDATION_START), color="#666", linestyle="--", linewidth=1)
        ax.grid(alpha=.2)
        ax.legend(loc="upper left", ncol=2)
    axes[-1].set_xlabel("Rental approval month; all signals observed by end of prior month")
    fig.savefig(OUT / "01_monthly_market_signals.png", dpi=160, facecolor="white")
    plt.close(fig)


def plot_scores(scores: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    for group, part in scores.groupby("group", sort=False):
        if group == "baseline":
            continue
        ax.plot(part["validation_start"], part["delta_rmse_vs_baseline_sgd"],
                marker="o", linewidth=2, label=group.replace("_", " "))
    ax.axhline(0, color="#333", linewidth=1)
    ax.set(title="Forward validation: market features vs property/spatial baseline",
           xlabel="Validation start month", ylabel="Change in RMSE (SGD; lower is better)")
    ax.grid(axis="y", alpha=.25)
    ax.legend(ncol=2)
    fig.savefig(OUT / "02_forward_rmse_delta.png", dpi=160, facecolor="white")
    plt.close(fig)


def main() -> None:
    before = {str(p.relative_to(DATA)): sha256(p) for p in SOURCES}
    coe = pd.read_csv(AUX / "sg-coe-prices.csv")
    stocks = pd.read_csv(AUX / "sg-stock-prices.csv")
    market = monthly_market(coe, stocks)
    rent = mix_adjusted_monthly(pd.read_csv(DATA / "train.csv"))
    monthly = attach_prior_month(rent, market).set_index(pd.PeriodIndex(rent["RENT_APPROVAL_DATE"], freq="M"))
    monthly["rent_adjusted_growth"] = monthly["rent_adjusted"].pct_change(fill_method=None)
    corr = correlations(monthly)
    frame = attach_prior_month(feature_frame(), market)
    scores = forward_scores(frame)
    OUT.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(OUT / "monthly_series.csv", index_label="approval_month", float_format="%.6f")
    corr.to_csv(OUT / "rent_growth_correlations.csv", index=False, float_format="%.4f")
    scores.to_csv(OUT / "forward_scores.csv", index=False, float_format="%.4f")
    plot_signals(monthly)
    plot_scores(scores)
    after = {str(p.relative_to(DATA)): sha256(p) for p in SOURCES}
    if before != after:
        raise AssertionError("An input CSV changed during analysis")
    summary = {
        "target": "MONTHLY_RENT (SGD)", "validation_cutoff": VALIDATION_START,
        "prediction_scenario": "monthly-updated; lag 1 means preceding calendar month, not fixed-origin",
        "folds": FOLDS, "feature_groups": GROUPS, "model": "sklearn HistGradientBoostingRegressor",
        "model_settings": SETTINGS, "source_sha256": after,
        "limitations": [
            "Market series contain few independent months; correlations are exploratory and not causal.",
            "Mix-adjusted rent is a descriptive index fitted before the existing holdout, not a prospective fold-specific feature.",
            "The final 2024-04 fold has been used by prior experiments and is not an untouched test set.",
            "Availability of prior-month COE and stock data must be checked for the intended forecast publication date.",
            "Stock basket uses symbols ending .SI and a median of individual adjusted-close returns; basket membership is from the supplied snapshot.",
            "No hyperparameter tuning or nested feature selection was performed; compare consistency across folds.",
        ],
    }
    (OUT / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved economic-signal EDA to {OUT}", flush=True)


if __name__ == "__main__":
    main()
