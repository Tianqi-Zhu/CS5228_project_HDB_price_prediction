"""Monthly location/flat-type rent trends against lagged market signals.

Each group-month is one observation. Correlations use rent *changes* rather
than rent levels; forward Ridge models test whether market signals improve a
one-month-ahead autoregressive baseline. No causal interpretation is implied.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis.pipeline.config import AUX, DATA, REPORTS, VALIDATION_START
from analysis.pipeline.data import normalized_address, prepare_main, sha256
from analysis.eda.economic_signals import attach_prior_month, monthly_market
from analysis.pipeline.evaluation import metrics


OUT = REPORTS / "group_temporal_panel"
MIN_MONTH_RECORDS = 20
MIN_CORRELATION_MONTHS = 24
SOURCES = [DATA / "train.csv", AUX / "sg-hdb-block.csv",
           AUX / "sg-coe-prices.csv", AUX / "sg-stock-prices.csv"]
GRAINS = {
    "planning_area": ["PLANNING_AREA"],
    "flat_type": ["FLAT_TYPE_CLEAN"],
    "planning_area_x_flat_type": ["PLANNING_AREA", "FLAT_TYPE_CLEAN"],
}
SIGNALS = [
    "COE_A_GROWTH_LAG1", "COE_C_GROWTH_LAG1", "COE_C_BID_PRESSURE_LAG1",
    "SG_STOCK_RETURN_LAG1", "SG_STOCK_RETURN_3M_LAG1",
]
BASE = ["rent_lag1", "month_index", "season_sin", "season_cos"]
VARIANTS = {
    "rent_history": [],
    "plus_coe": SIGNALS[:3],
    "plus_stocks": SIGNALS[3:],
    "plus_both": SIGNALS,
}
FOLDS = [("2023-01", "2023-07"), ("2023-07", VALIDATION_START),
         (VALIDATION_START, "2025-04")]


def add_planning_area(train: pd.DataFrame, blocks: pd.DataFrame) -> pd.DataFrame:
    """Match each rental address to its planning area without modifying sources."""
    keys = ["TOWN", "BLOCK", "STREET"]
    block = normalized_address(blocks)[keys + ["PLANNING_AREA"]]
    if block.duplicated(keys).any():
        raise ValueError("Duplicate normalized addresses in HDB block data")
    frame = normalized_address(prepare_main(train)).merge(
        block, on=keys, how="left", validate="many_to_one"
    )
    if len(frame) != len(train) or frame["PLANNING_AREA"].isna().any():
        raise ValueError("Incomplete rental-to-planning-area match")
    return frame


def monthly_panel(frame: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Aggregate rent, add actual prior-month rent and prior-month markets."""
    months = pd.period_range(frame["RENT_APPROVAL_DATE"].min(),
                             frame["RENT_APPROVAL_DATE"].max(), freq="M")
    parts = []
    for grain, keys in GRAINS.items():
        aggregate = frame.groupby(keys + ["RENT_APPROVAL_DATE"], dropna=False)["MONTHLY_RENT"].agg(
            n="size", mean_rent="mean", median_rent="median"
        ).reset_index()
        for group_key, group in aggregate.groupby(keys, sort=True):
            values = group_key if isinstance(group_key, tuple) else (group_key,)
            label = " / ".join(str(value) for value in values)
            part = group.set_index(pd.PeriodIndex(group["RENT_APPROVAL_DATE"], freq="M"))
            part = part[["n", "mean_rent", "median_rent"]].reindex(months)
            part["n"] = part["n"].fillna(0).astype(int)
            part["grain"], part["group"] = grain, label
            part["month"] = months
            for target in ("mean", "median"):
                part[f"{target}_lag1"] = part[f"{target}_rent"].shift(1)
                part[f"{target}_growth"] = part[f"{target}_rent"] / part[f"{target}_lag1"] - 1
            part["n_lag1"] = part["n"].shift(1)
            parts.append(part.reset_index(drop=True))
    panel = pd.concat(parts, ignore_index=True)
    panel["RENT_APPROVAL_DATE"] = panel["month"].astype(str)
    panel = attach_prior_month(panel, market)
    panel["month_index"] = panel["month"].astype("int64") - months[0].ordinal
    panel["season_sin"] = np.sin(2 * np.pi * panel["month"].dt.month / 12)
    panel["season_cos"] = np.cos(2 * np.pi * panel["month"].dt.month / 12)
    return panel


def eligible(panel: pd.DataFrame, target: str) -> pd.DataFrame:
    return panel[(panel["n"] >= MIN_MONTH_RECORDS) &
                 (panel["n_lag1"] >= MIN_MONTH_RECORDS) &
                 panel[f"{target}_rent"].notna() &
                 panel[f"{target}_lag1"].notna()].copy()


def group_correlations(panel: pd.DataFrame) -> pd.DataFrame:
    """Prevalidation within-group correlations; each month counts once."""
    rows = []
    split = pd.Period("2022-08", freq="M")
    for target in ("mean", "median"):
        fitting = eligible(panel, target)
        fitting = fitting[fitting["RENT_APPROVAL_DATE"] < VALIDATION_START]
        for (grain, group), part in fitting.groupby(["grain", "group"], sort=True):
            if len(part) < MIN_CORRELATION_MONTHS:
                continue
            for signal in SIGNALS:
                pair = part[["month", f"{target}_growth", signal]].replace(
                    [np.inf, -np.inf], np.nan
                ).dropna()
                if len(pair) < MIN_CORRELATION_MONTHS:
                    continue
                def corr(piece: pd.DataFrame, method: str = "pearson") -> float:
                    if len(piece) < 8 or piece[f"{target}_growth"].nunique() < 2 or piece[signal].nunique() < 2:
                        return np.nan
                    return float(piece[[f"{target}_growth", signal]].corr(method=method).iloc[0, 1])
                rows.append({
                    "grain": grain, "group": group, "rent_stat": target,
                    "signal": signal, "months": len(pair),
                    "rentals_in_paired_months": int(part.loc[part["month"].isin(pair["month"]), "n"].sum()),
                    "pearson_r": corr(pair), "spearman_rho": corr(pair, "spearman"),
                    "early_r": corr(pair[pair["month"] < split]),
                    "late_r": corr(pair[pair["month"] >= split]),
                })
    return pd.DataFrame(rows)


def forward_models(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-group Ridge vs rent-history baseline on expanding time windows."""
    scores, predictions = [], []
    for target in ("mean", "median"):
        usable = eligible(panel, target)
        for (grain, group), part in usable.groupby(["grain", "group"], sort=True):
            part = part.sort_values("month")
            for start, end in FOLDS:
                fit = part[part["RENT_APPROVAL_DATE"] < start]
                val = part[(part["RENT_APPROVAL_DATE"] >= start) &
                           (part["RENT_APPROVAL_DATE"] < end)]
                if len(fit) < 18 or len(val) < 3:
                    continue
                for variant, extra in VARIANTS.items():
                    features = [f"{target}_lag1" if column == "rent_lag1" else column
                                for column in BASE] + extra
                    model = make_pipeline(SimpleImputer(strategy="median"),
                                          StandardScaler(), Ridge(alpha=10.0))
                    model.fit(fit[features], fit[f"{target}_rent"])
                    predicted = model.predict(val[features])
                    actual = val[f"{target}_rent"].to_numpy(float)
                    scores.append({"grain": grain, "group": group, "rent_stat": target,
                                   "validation_start": start, "validation_end_exclusive": end,
                                   "variant": variant, "n_fit_months": len(fit),
                                   "n_validation_months": len(val), **metrics(actual, predicted)})
                    predictions.extend({"grain": grain, "group": group, "rent_stat": target,
                                        "validation_start": start, "month": str(month),
                                        "variant": variant, "n_rentals": int(n),
                                        "actual": float(y), "predicted": float(p)}
                                       for month, n, y, p in zip(val["month"], val["n"], actual, predicted))
    score = pd.DataFrame(scores)
    keys = ["grain", "group", "rent_stat", "validation_start"]
    base = score[score["variant"].eq("rent_history")].set_index(keys)["rmse_sgd"]
    score["delta_rmse_vs_history_sgd"] = score["rmse_sgd"] - score.set_index(keys).index.map(base)
    return score, pd.DataFrame(predictions)


def pooled_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["grain", "rent_stat", "validation_start", "variant"]
    for labels, part in predictions.groupby(keys, sort=True):
        row = dict(zip(keys, labels))
        row.update({"groups": part["group"].nunique(), "group_months": len(part),
                    **metrics(part["actual"].to_numpy(), part["predicted"].to_numpy())})
        rows.append(row)
    pooled = pd.DataFrame(rows)
    base = pooled[pooled["variant"].eq("rent_history")].set_index(keys[:-1])["rmse_sgd"]
    pooled["delta_rmse_vs_history_sgd"] = pooled["rmse_sgd"] - pooled.set_index(keys[:-1]).index.map(base)
    return pooled


def plot_correlations(correlations: pd.DataFrame, rent_stat: str) -> None:
    if rent_stat not in ("mean", "median"):
        raise ValueError(f"Unsupported rent statistic: {rent_stat}")
    part = correlations[(correlations["grain"] == "planning_area") &
                        (correlations["rent_stat"] == rent_stat)]
    if part.empty:
        return
    wide = part.pivot(index="group", columns="signal", values="pearson_r").reindex(columns=SIGNALS)
    wide = wide.sort_index()
    fig, ax = plt.subplots(figsize=(11, max(6, .34 * len(wide) + 2)), layout="constrained")
    im = ax.imshow(wide.to_numpy(float), cmap="RdBu_r", vmin=-.6, vmax=.6, aspect="auto")
    ax.set_yticks(range(len(wide)), wide.index, fontsize=8)
    ax.set_xticks(range(len(wide.columns)), ["COE A growth", "COE C growth", "COE C bid pressure",
                                                  "SG return", "SG 3m return"], rotation=30, ha="right")
    ax.set(title=f"Planning-area {rent_stat}-rent growth vs prior-month market signals",
           xlabel="Pre-2024-04 Pearson r; each shown pair has at least 24 months")
    fig.colorbar(im, ax=ax, shrink=.7, label="Pearson r")
    name = "01_planning_area_correlations.png" if rent_stat == "mean" else "04_planning_area_median_correlations.png"
    fig.savefig(OUT / name, dpi=160, facecolor="white")
    plt.close(fig)


def plot_flat_type_correlations(correlations: pd.DataFrame) -> None:
    part = correlations[correlations["grain"] == "flat_type"]
    if part.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for ax, target in zip(axes, ("mean", "median")):
        wide = part[part["rent_stat"] == target].pivot(
            index="group", columns="signal", values="pearson_r"
        ).reindex(columns=SIGNALS).sort_index()
        im = ax.imshow(wide.to_numpy(float), cmap="RdBu_r", vmin=-.6, vmax=.6, aspect="auto")
        ax.set_yticks(range(len(wide)), wide.index)
        ax.set_xticks(range(len(SIGNALS)), ["COE A growth", "COE C growth", "COE C pressure",
                                            "SG return", "SG 3m return"], rotation=35, ha="right")
        ax.set_title(f"{target.capitalize()} rent growth")
    fig.colorbar(im, ax=axes, shrink=.8, label="Pre-2024-04 Pearson r")
    fig.suptitle("Flat-type rent growth vs prior-month market signals")
    fig.savefig(OUT / "03_flat_type_correlations.png", dpi=160, facecolor="white")
    plt.close(fig)


def plot_validation(pooled: pd.DataFrame) -> None:
    grains = list(GRAINS)
    fig, axes = plt.subplots(2, len(grains), figsize=(15, 8), sharex=True, layout="constrained")
    for i, target in enumerate(("mean", "median")):
        for j, grain in enumerate(grains):
            ax = axes[i, j]
            part = pooled[(pooled["grain"] == grain) & (pooled["rent_stat"] == target)]
            for variant in ("plus_coe", "plus_stocks", "plus_both"):
                series = part[part["variant"] == variant]
                ax.plot(series["validation_start"], series["delta_rmse_vs_history_sgd"],
                        marker="o", label=variant.replace("plus_", "+ ").replace("_", " "))
            ax.axhline(0, color="#333", linewidth=1)
            ax.set(title=f"{grain.replace('_', ' ')}: {target}", ylabel="RMSE change (SGD)")
            ax.grid(axis="y", alpha=.25)
            ax.tick_params(axis="x", rotation=30)
            if i == 0 and j == len(grains) - 1:
                ax.legend(fontsize=8)
    fig.suptitle("One-month-ahead Ridge: market signals vs rent-history baseline")
    fig.savefig(OUT / "02_forward_rmse_delta.png", dpi=160, facecolor="white")
    plt.close(fig)


def main() -> None:
    before = {str(p.relative_to(DATA)): sha256(p) for p in SOURCES}
    train = pd.read_csv(DATA / "train.csv")
    blocks = pd.read_csv(AUX / "sg-hdb-block.csv", dtype={"POSTAL_CODE": "string"})
    market = monthly_market(pd.read_csv(AUX / "sg-coe-prices.csv"),
                            pd.read_csv(AUX / "sg-stock-prices.csv"))
    panel = monthly_panel(add_planning_area(train, blocks), market)
    correlations = group_correlations(panel)
    scores, predictions = forward_models(panel)
    pooled = pooled_scores(predictions)
    OUT.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUT / "monthly_panel.csv", index=False, float_format="%.6f")
    correlations.to_csv(OUT / "group_correlations.csv", index=False, float_format="%.4f")
    scores.to_csv(OUT / "group_forward_scores.csv", index=False, float_format="%.4f")
    pooled.to_csv(OUT / "pooled_forward_scores.csv", index=False, float_format="%.4f")
    plot_correlations(correlations, "mean")
    plot_correlations(correlations, "median")
    plot_flat_type_correlations(correlations)
    plot_validation(pooled)
    after = {str(p.relative_to(DATA)): sha256(p) for p in SOURCES}
    if before != after:
        raise AssertionError("Source CSV changed during analysis")
    summary = {
        "target": "Monthly mean and median MONTHLY_RENT (SGD) by group",
        "geography": "PLANNING_AREA from the HDB block table is a district proxy, not an official postal district",
        "monthly_minimum_rentals": MIN_MONTH_RECORDS,
        "correlation_minimum_paired_months": MIN_CORRELATION_MONTHS,
        "correlation_period": f"Before {VALIDATION_START}",
        "correlation_target": "Within-group month-on-month mean/median rent change",
        "planning_area_chart_targets": ["Month-on-month mean rent change", "Month-on-month median rent change"],
        "prediction_scenario": "Monthly-updated one-step-ahead; prior-month group rent and market observations known",
        "model": "Per-group Ridge(alpha=10) with training-only median imputation and standardization",
        "baseline_predictors": BASE, "feature_variants": VARIANTS,
        "folds": FOLDS, "source_sha256": after,
        "limitations": [
            "The market signal has one observation per calendar month, not one independent observation per group.",
            "Correlations are exploratory across many groups/signals; no multiple-testing significance claim is made.",
            "Monthly group means and medians can change with property mix; no within-group hedonic adjustment is made.",
            "Validation uses actual previous-month group rent, so it is one-step-ahead, not a fixed-origin multi-month forecast.",
            "The 2024-04 holdout has already been examined in earlier experiments.",
        ],
    }
    (OUT / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved group temporal panel analysis to {OUT}", flush=True)


if __name__ == "__main__":
    main()
