"""Monthly rent/economic time-series diagnostics for the progress report.

Correlations and lag scans use only months before the existing validation cutoff.
All labeled months appear in the descriptive time-series plots. No plot or
correlation is a causal estimate or a replacement for forward validation.
"""

from __future__ import annotations

import json
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from analysis.pipeline.auxiliary import add_economic_features
from analysis.pipeline.config import AUX, DATA, REPORTS, VALIDATION_START
from analysis.pipeline.data import prepare_main, sha256


OUT = REPORTS / "temporal_correlations"
SOURCE_FILES = [DATA / "train.csv", AUX / "sg-coe-prices.csv", AUX / "sg-stock-prices.csv"]
COE_CATEGORIES = ("A", "B", "C", "E")
LEVEL_COLUMNS = {
    "rent_mean": "Raw rent",
    "rent_adjusted": "Mix-adjusted rent",
    **{f"COE_{c}_MEDIAN_LAG1": f"COE {c}" for c in COE_CATEGORIES},
}
CHANGE_COLUMNS = {
    "rent_adjusted_growth": "Rent growth",
    **{f"COE_{c}_GROWTH_LAG1": f"COE {c} growth" for c in COE_CATEGORIES},
    "STOCK_MEDIAN_RETURN_LAG1": "All-stock return",
    "SG_STOCK_RETURN_LAG1": "SG-stock return",
}
LAG_INDICATORS = {
    "COE_A_GROWTH_LAG1": "COE A growth",
    "COE_C_GROWTH_LAG1": "COE C growth",
    "STOCK_MEDIAN_RETURN_LAG1": "All-stock return",
    "SG_STOCK_RETURN_LAG1": "SG-stock return",
}


def mix_adjusted_monthly(train: pd.DataFrame) -> pd.DataFrame:
    """Remove a fixed property-mix effect fitted on pre-validation rows.

    The model excludes approval date. This index is descriptive, not a rent
    prediction model or an estimate of a causal time effect.
    """
    frame = prepare_main(train)
    fitting = frame["RENT_APPROVAL_DATE"] < VALIDATION_START
    categorical = ["FLAT_TYPE_CLEAN", "FLAT_MODEL_CLEAN", "TOWN_CLEAN"]
    numeric = ["FLOOR_AREA_SQM", "LEASE_COMMENCE_DATE"]
    pipeline = Pipeline([
        ("preprocess", ColumnTransformer([
            ("categories", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numbers", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), numeric),
        ])),
        ("model", Ridge(alpha=10.0)),
    ])
    features = categorical + numeric
    pipeline.fit(frame.loc[fitting, features], frame.loc[fitting, "MONTHLY_RENT"])
    anchor = float(frame.loc[fitting, "MONTHLY_RENT"].mean())
    frame["rent_adjusted"] = frame["MONTHLY_RENT"] - pipeline.predict(frame[features]) + anchor
    monthly = frame.groupby("RENT_APPROVAL_DATE").agg(
        rent_mean=("MONTHLY_RENT", "mean"),
        rent_median=("MONTHLY_RENT", "median"),
        rent_adjusted=("rent_adjusted", "mean"),
        approvals=("MONTHLY_RENT", "size"),
    ).reset_index()
    return monthly


def sg_stock_return(stocks: pd.DataFrame) -> pd.Series:
    """Median month-end adjusted-close return across .SI symbols."""
    singapore = stocks[stocks["SYMBOL"].astype(str).str.endswith(".SI")].copy()
    singapore["DATE"] = pd.to_datetime(singapore["DATE"], errors="coerce")
    singapore["ADJUSTED_CLOSE"] = pd.to_numeric(singapore["ADJUSTED_CLOSE"], errors="coerce")
    singapore = singapore.dropna(subset=["DATE", "ADJUSTED_CLOSE"])
    singapore["period"] = singapore["DATE"].dt.to_period("M")
    closes = (singapore.sort_values(["SYMBOL", "DATE"])
              .groupby(["SYMBOL", "period"])["ADJUSTED_CLOSE"].last().unstack("period").T)
    return closes.pct_change(fill_method=None).median(axis=1).rename("SG_STOCK_RETURN")


def monthly_series(train: pd.DataFrame, coe: pd.DataFrame, stocks: pd.DataFrame) -> pd.DataFrame:
    monthly = add_economic_features(mix_adjusted_monthly(train), coe, stocks)
    monthly = monthly.sort_values("RENT_APPROVAL_DATE").set_index(
        pd.PeriodIndex(monthly["RENT_APPROVAL_DATE"], freq="M")
    )
    sg = sg_stock_return(stocks)
    monthly["SG_STOCK_RETURN_LAG1"] = sg.shift(1).reindex(monthly.index)
    monthly["rent_mean_growth"] = monthly["rent_mean"].pct_change(fill_method=None)
    monthly["rent_adjusted_growth"] = monthly["rent_adjusted"].pct_change(fill_method=None)
    for category in COE_CATEGORIES:
        price = f"COE_{category}_MEDIAN_LAG1"
        monthly[f"COE_{category}_GROWTH_LAG1"] = monthly[price].pct_change(fill_method=None)
    return monthly


def pair_correlations(frame: pd.DataFrame, columns: list[str], group: str) -> pd.DataFrame:
    rows = []
    for left, right in combinations(columns, 2):
        pair = frame[[left, right]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(pair) < 4 or pair[left].nunique() < 2 or pair[right].nunique() < 2:
            continue
        rows.append({
            "group": group, "left": left, "right": right, "months": len(pair),
            "pearson_r": float(pearsonr(pair[left], pair[right]).statistic),
            "spearman_rho": float(spearmanr(pair[left], pair[right]).statistic),
        })
    return pd.DataFrame(rows)


def block_bootstrap_interval(x: np.ndarray, y: np.ndarray, *, repeats: int = 500,
                             block_months: int = 3, seed: int = 42) -> tuple[float, float]:
    """Exploratory 95% interval from resampled contiguous month blocks."""
    n = len(x)
    if n < 2 * block_months or repeats < 1:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    correlations = []
    n_blocks = int(np.ceil(n / block_months))
    for _ in range(repeats):
        starts = rng.integers(0, n - block_months + 1, size=n_blocks)
        positions = np.concatenate([np.arange(start, start + block_months) for start in starts])[:n]
        sampled_x, sampled_y = x[positions], y[positions]
        if np.std(sampled_x) and np.std(sampled_y):
            correlations.append(np.corrcoef(sampled_x, sampled_y)[0, 1])
    if len(correlations) < repeats // 2:
        return np.nan, np.nan
    low, high = np.quantile(correlations, [.025, .975])
    return float(low), float(high)


def lag_correlations(frame: pd.DataFrame, *, target: str = "rent_adjusted_growth",
                     indicators: dict[str, str] | None = None,
                     max_lag: int = 6, bootstrap_repeats: int = 500) -> pd.DataFrame:
    """Correlate rent growth at t with indicator observations at t-lag.

    Input indicator columns already refer to the previous month (LAG1), so
    lag k requires an additional shift of k-1 months.
    """
    indicators = LAG_INDICATORS if indicators is None else indicators
    rows = []
    for indicator, label in indicators.items():
        for lag in range(1, max_lag + 1):
            pair = pd.concat([frame[target], frame[indicator].shift(lag - 1)], axis=1)
            pair.columns = ["rent", "indicator"]
            pair = pair.replace([np.inf, -np.inf], np.nan).dropna()
            if len(pair) < 8 or pair.rent.nunique() < 2 or pair.indicator.nunique() < 2:
                continue
            x = pair.indicator.to_numpy(float)
            y = pair.rent.to_numpy(float)
            low, high = block_bootstrap_interval(x, y, repeats=bootstrap_repeats,
                                                  seed=42 + lag)
            rows.append({"indicator": indicator, "label": label, "lag_months": lag,
                         "months": len(pair), "pearson_r": float(np.corrcoef(x, y)[0, 1]),
                         "bootstrap_low": low, "bootstrap_high": high})
    return pd.DataFrame(rows)


def plot_levels(monthly: pd.DataFrame, fitting: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    selected = {
        "rent_mean": ("Raw rent", "#526675", "--"),
        "rent_adjusted": ("Mix-adjusted rent", "#147D91", "-"),
        "COE_A_MEDIAN_LAG1": ("COE A, lag 1", "#D78036", "-"),
        "COE_C_MEDIAN_LAG1": ("COE C, lag 1", "#8754A1", "-"),
    }
    dates = monthly.index.to_timestamp()
    for column, (label, color, style) in selected.items():
        mean, std = fitting[column].mean(), fitting[column].std()
        ax.plot(dates, (monthly[column] - mean) / std, label=label,
                color=color, linestyle=style, linewidth=2)
    ax.axvspan(pd.Timestamp(VALIDATION_START), dates.max(), color="#FBEBD9", alpha=.6,
               label="Existing validation window")
    ax.set(title="Rent and COE levels share a strong trend",
           ylabel="Standardized level (fit-period mean = 0)", xlabel="Approval month")
    ax.grid(alpha=.2)
    ax.legend(ncol=2)
    fig.savefig(OUT / "01_standardized_levels.png", dpi=160, facecolor="white")
    plt.close(fig)


def plot_changes(monthly: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, layout="constrained")
    dates = monthly.index.to_timestamp()
    groups = [
        (axes[0], [("rent_mean_growth", "Raw", "#526675"),
                   ("rent_adjusted_growth", "Mix-adjusted", "#147D91")], "Monthly rent growth"),
        (axes[1], [("COE_A_GROWTH_LAG1", "COE A", "#D78036"),
                   ("COE_C_GROWTH_LAG1", "COE C", "#8754A1")], "Prior-month COE growth"),
        (axes[2], [("STOCK_MEDIAN_RETURN_LAG1", "All symbols", "#3979AD"),
                   ("SG_STOCK_RETURN_LAG1", ".SI symbols", "#39966D")], "Prior-month stock return"),
    ]
    for ax, series, title in groups:
        for column, label, color in series:
            ax.plot(dates, monthly[column] * 100, label=label, color=color, linewidth=1.8)
        ax.axhline(0, color="#526675", linewidth=.8)
        ax.axvspan(pd.Timestamp(VALIDATION_START), dates.max(), color="#FBEBD9", alpha=.6)
        ax.set(title=title, ylabel="Percent (%)")
        ax.grid(alpha=.2)
        ax.legend(loc="upper left", ncol=2)
    axes[-1].set_xlabel("Approval month")
    fig.suptitle("Monthly changes reveal a weaker and noisier relationship")
    fig.savefig(OUT / "02_monthly_changes.png", dpi=160, facecolor="white")
    plt.close(fig)


def plot_heatmaps(fitting: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), layout="constrained")
    for ax, labels, title in ((axes[0], LEVEL_COLUMNS, "Levels (shared trends)"),
                              (axes[1], CHANGE_COLUMNS, "Changes and returns")):
        columns = list(labels)
        correlation = fitting[columns].corr().to_numpy()
        im = ax.imshow(correlation, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(columns)), labels.values(), rotation=45, ha="right")
        ax.set_yticks(range(len(columns)), labels.values())
        ax.set_title(title)
        for i in range(len(columns)):
            for j in range(len(columns)):
                value = correlation[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if abs(value) > .6 else "#193043")
    fig.colorbar(im, ax=axes, shrink=.8, label="Pearson r")
    fig.suptitle("Pre-validation monthly correlations")
    fig.savefig(OUT / "03_correlation_heatmaps.png", dpi=160, facecolor="white")
    plt.close(fig)


def plot_lags(lags: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5), layout="constrained")
    colors = ["#D78036", "#8754A1", "#3979AD", "#39966D"]
    for (indicator, part), color in zip(lags.groupby("indicator", sort=False), colors):
        part = part.sort_values("lag_months")
        x = part["lag_months"].to_numpy()
        ax.plot(x, part["pearson_r"], marker="o", label=part["label"].iloc[0], color=color)
        ax.fill_between(x, part["bootstrap_low"], part["bootstrap_high"], color=color, alpha=.12)
    ax.axhline(0, color="#526675", linewidth=1)
    ax.set(xticks=range(1, 7), xlabel="Indicator lag before rent approval (months)",
           ylabel="Pearson correlation with mix-adjusted rent growth",
           title="Lag scan on pre-validation months (shading: 95% block-bootstrap interval)")
    ax.grid(axis="y", alpha=.2)
    ax.legend(ncol=2)
    fig.savefig(OUT / "04_lag_correlations.png", dpi=160, facecolor="white")
    plt.close(fig)


def main() -> None:
    source_hashes = {str(path.relative_to(DATA)): sha256(path) for path in SOURCE_FILES}
    monthly = monthly_series(
        pd.read_csv(DATA / "train.csv"),
        pd.read_csv(AUX / "sg-coe-prices.csv"),
        pd.read_csv(AUX / "sg-stock-prices.csv"),
    )
    fitting = monthly[monthly["RENT_APPROVAL_DATE"] < VALIDATION_START]
    if len(fitting) < 24:
        raise ValueError("Too few fitting months for temporal correlation analysis")
    levels = pair_correlations(fitting, list(LEVEL_COLUMNS), "levels")
    changes = pair_correlations(fitting, list(CHANGE_COLUMNS), "changes")
    lags = lag_correlations(fitting)
    OUT.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(OUT / "monthly_series.csv", index=False, float_format="%.6f")
    pd.concat([levels, changes], ignore_index=True).to_csv(OUT / "pair_correlations.csv", index=False, float_format="%.4f")
    lags.to_csv(OUT / "lag_correlations.csv", index=False, float_format="%.4f")
    plot_levels(monthly, fitting)
    plot_changes(monthly)
    plot_heatmaps(fitting)
    plot_lags(lags)
    if {str(path.relative_to(DATA)): sha256(path) for path in SOURCE_FILES} != source_hashes:
        raise AssertionError("An input CSV changed during the analysis")
    summary = {
        "fit_months": [fitting["RENT_APPROVAL_DATE"].min(), fitting["RENT_APPROVAL_DATE"].max()],
        "descriptive_months": [monthly["RENT_APPROVAL_DATE"].min(), monthly["RENT_APPROVAL_DATE"].max()],
        "validation_start": VALIDATION_START,
        "method": "Pearson and Spearman correlations of monthly levels and growth; 1-6 month lag scan with 3-month moving-block bootstrap intervals",
        "mix_adjustment": "Ridge on flat type, flat model, town, floor area and lease commence year; fitted before validation cutoff; approval date excluded",
        "source_sha256": source_hashes,
        "limitations": [
            "Correlations use months, not rental rows, as observations.",
            "Reported intervals are exploratory and do not adjust for checking multiple indicators or lags.",
            "Short time series and serial dependence limit precision; property-mix adjustment is approximate.",
            "Economic indicators are lagged at least one calendar month; prospective availability still depends on forecast timing.",
            "Existing validation months appear in descriptive plots but are excluded from correlation estimation.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote monthly correlation diagnostics to {OUT}")


if __name__ == "__main__":
    main()
