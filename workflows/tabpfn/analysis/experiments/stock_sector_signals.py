"""Industry-stock experiments with calendar cutoffs and mature forecast labels.

Run analysis/compare_stock_sectors.py. Source CSVs and submissions are read-only.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from analysis.config import AUX, DATA, REPORTS, ROOT
from analysis.data import prepare_main, sha256


OUTER_START = pd.Period("2024-06", "M")
DEVELOPMENT_START = pd.Period("2022-07", "M")
RIDGE_ALPHA = 10.0
SECTORS = {
    "banks": ["D05.SI", "O39.SI", "U11.SI"],
    "reits_industrial": ["A17U.SI", "M44U.SI", "ME8U.SI"],
    "reits_commercial": ["C38U.SI", "K71U.SI", "N2IU.SI", "T82U.SI"],
    "reits_all": ["A17U.SI", "M44U.SI", "ME8U.SI", "C38U.SI", "K71U.SI",
                  "N2IU.SI", "T82U.SI", "BUOU.SI", "HMN.SI"],
    "developers": ["C09.SI", "U06.SI", "U14.SI", "Z25.SI"],
    "travel_hospitality": ["C6L.SI", "S58.SI", "S59.SI", "G13.SI", "H15.SI", "HMN.SI"],
    "engineering": ["S63.SI", "V03.SI", "5E2.SI"],
    "telecom_infrastructure": ["Z74.SI", "CJLU.SI"],
}
MEASURES = ("return_1m", "momentum_3m", "volatility_3m")
REFERENCE_SOURCES = {
    "bank_membership": "https://www.abs.org.sg/about-us/memberbanks",
    "reit_sectors": "https://www.reitas.sg/singapore-reits/s-reit-sectors/",
    "historical_reit_list": "https://www.reitas.sg/wp-content/uploads/2024/09/the-pdf-doc_26-Aug-2024_1.pdf",
    "netlink_business": "https://www.imda.gov.sg/regulations-and-licensing-listing/nationwide-broadband-network",
}


def stock_panel(stocks: pd.DataFrame, end: pd.Period) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Monthly rebalanced equal-weight returns, never backfill or bridge gaps.

    Broad basket also has a median-return control matching earlier experiments.
    Coverage must reach 80% of the specified basket in each month.
    """
    s = stocks.copy()
    s["DATE"] = pd.to_datetime(s["DATE"], errors="raise")
    s = s[s["SYMBOL"].str.endswith(".SI") & (s["DATE"].dt.to_period("M") <= end)].copy()
    if s.duplicated(["SYMBOL", "DATE"]).any():
        raise ValueError("Duplicate stock/date observations")
    s["ADJUSTED_CLOSE"] = pd.to_numeric(s["ADJUSTED_CLOSE"], errors="raise")
    if (s["ADJUSTED_CLOSE"] <= 0).any():
        raise ValueError("Nonpositive adjusted close")
    s["month"] = s["DATE"].dt.to_period("M")
    p = s.sort_values("DATE").groupby(["month", "SYMBOL"])["ADJUSTED_CLOSE"].last().unstack()
    p = p.reindex(pd.period_range(p.index.min(), end, freq="M"))
    returns = p.pct_change(fill_method=None)
    baskets = {"broad_sg": sorted(p.columns), **SECTORS}
    panel, coverage = pd.DataFrame(index=p.index), []
    for name, members in baskets.items():
        missing = set(members) - set(p.columns)
        if missing:
            raise ValueError(f"Missing stock basket members: {name}: {missing}")
        count = returns[members].notna().sum(axis=1)
        r = returns[members].mean(axis=1).where(count >= np.ceil(.8 * len(members)))
        panel[f"{name}__return_1m"] = r
        for month in p.index:
            coverage.append({"month": str(month), "basket": name,
                             "expected_stocks": len(members), "observed_returns": int(count.loc[month])})
    panel["broad_median__return_1m"] = returns.median(axis=1)
    for name in [*baskets, "broad_median"]:
        r = panel[f"{name}__return_1m"]
        panel[f"{name}__momentum_3m"] = (1 + r).rolling(3, min_periods=3).apply(np.prod, raw=True) - 1
        panel[f"{name}__volatility_3m"] = r.rolling(3, min_periods=3).std()
    panel.index.name = "month"
    return panel, pd.DataFrame(coverage)


def candidate_features(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Origin month m observes stocks through m; next-month rent uses lag 1.

    Delays 0/2/5 therefore correspond to lag 1/3/6 for a one-month forecast.
    """
    derived = {}
    variants = {"rent_ar": []}
    for basket in ["broad_sg", "broad_median", *SECTORS]:
        for delay in (0, 2, 5):
            columns = []
            for measure in MEASURES:
                column = f"{basket}__{measure}__delay{delay}"
                derived[column] = panel[f"{basket}__{measure}"].shift(delay)
                columns.append(column)
            variants[f"{basket}_lag{delay + 1}"] = columns
    for basket in SECTORS:
        excess = []
        for measure in MEASURES:
            column = f"{basket}__excess_{measure}"
            derived[column] = panel[f"{basket}__{measure}"] - panel[f"broad_sg__{measure}"]
            excess.append(column)
        variants[f"market_plus_{basket}"] = variants["broad_sg_lag1"] + excess
    # Overlapping REIT subgroups are omitted from the joint feature bundle.
    disjoint = [x for x in SECTORS if x not in ("reits_industrial", "reits_commercial")]
    variants["all_sectors"] = sum((variants[f"{x}_lag1"] for x in disjoint), [])
    return pd.concat([panel, pd.DataFrame(derived, index=panel.index)], axis=1), variants


def adjusted_rent_index(raw: pd.DataFrame) -> pd.DataFrame:
    """Fix property adjustment using 2021 only, before all scored forecasts.

    Additive SGD residual index anchored to the 2021 mean; no dates in model.
    Unlike an index fitted on all pre-holdout data, early forecast labels here
    cannot influence their property adjustment via later rental observations.
    """
    frame = prepare_main(raw)
    fit = frame["RENT_APPROVAL_DATE"].between("2021-01", "2021-12")
    if not fit.any():
        raise ValueError("2021 reference period absent")
    categorical = ["FLAT_TYPE_CLEAN", "FLAT_MODEL_CLEAN", "TOWN_CLEAN"]
    numeric = ["FLOOR_AREA_SQM", "LEASE_COMMENCE_DATE"]
    columns = categorical + numeric
    model = Pipeline([
        ("preprocess", ColumnTransformer([
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")),
                                   ("scale", StandardScaler())]), numeric),
        ])),
        ("regression", Ridge(alpha=10.0)),
    ])
    model.fit(frame.loc[fit, columns], frame.loc[fit, "MONTHLY_RENT"])
    frame["adjusted"] = frame["MONTHLY_RENT"] - model.predict(frame[columns]) + frame.loc[fit, "MONTHLY_RENT"].mean()
    monthly = frame.groupby("RENT_APPROVAL_DATE").agg(
        adjusted_rent=("adjusted", "mean"), raw_mean=("MONTHLY_RENT", "mean"),
        raw_median=("MONTHLY_RENT", "median"), approvals=("MONTHLY_RENT", "size"))
    monthly.index = pd.PeriodIndex(monthly.index, freq="M", name="month")
    return monthly.reindex(pd.period_range(monthly.index.min(), monthly.index.max(), freq="M"))


def ridge_forecast(x: np.ndarray, y: np.ndarray, query: np.ndarray,
                   alpha: float = RIDGE_ALPHA) -> tuple[float, np.ndarray]:
    """Median/scale/coefficients use training rows only; all-missing => zero."""
    medians = np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0.0 for c in x.T])
    filled = np.where(np.isfinite(x), x, medians)
    mean, scale = filled.mean(axis=0), filled.std(axis=0)
    scale[scale < 1e-12] = 1.0
    z = (filled - mean) / scale
    q = (np.where(np.isfinite(query), query, medians) - mean) / scale
    coefficient = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (y - y.mean()))
    return float(y.mean() + q @ coefficient), coefficient


def monthly_forecasts(monthly: pd.DataFrame, features: pd.DataFrame,
                      variants: dict[str, list[str]]) -> pd.DataFrame:
    """Direct horizon regression: training targets must mature by origin m.

    Forecasts m+h use rent/stock predictors through m. Forecast origins before
    Dec 2021 are excluded so that the property adjustment is already fitted.
    """
    idx = monthly.index
    y = monthly["adjusted_rent"]
    diff = y.diff()
    base = pd.DataFrame({"rent_change_1m": diff, "rent_change_3m": diff.rolling(3).mean(),
                         "rent_change_6m": diff.rolling(6).mean()}, index=idx)
    first_origin = max(idx.get_loc(pd.Period("2021-12", "M")), 6)
    rows = []
    for h in range(1, 11):
        f = base.join(features)
        angle = 2 * np.pi * ((idx.month - 1 + h) % 12) / 12
        f["target_month_sin"], f["target_month_cos"] = np.sin(angle), np.cos(angle)
        base_columns = list(base) + ["target_month_sin", "target_month_cos"]
        changes = y.shift(-h) - y
        for oi in range(first_origin + h + 7, len(idx) - h):
            target = idx[oi + h]
            if target < DEVELOPMENT_START:
                continue
            # At origin oi, origin j's h-step label is known iff j+h <= oi.
            train_pos = np.arange(first_origin, oi - h + 1)
            common = {"origin": str(idx[oi]), "target_month": str(target), "horizon": h,
                      "actual": float(y.iloc[oi + h]), "n_train_months": len(train_pos),
                      "max_train_target": str(idx[train_pos[-1] + h]),
                      "phase": "development" if target < OUTER_START else "outer"}
            rows.append({**common, "variant": "persistence", "prediction": float(y.iloc[oi])})
            rows.append({**common, "variant": "momentum_3m", "prediction": float(y.iloc[oi] + h * base.iloc[oi, 1])})
            for name, columns in variants.items():
                a = f[base_columns + columns].to_numpy(float)
                predicted_change, _ = ridge_forecast(a[train_pos], changes.iloc[train_pos].to_numpy(float), a[oi])
                rows.append({**common, "variant": name, "prediction": float(y.iloc[oi] + predicted_change)})
        print(f"Monthly direct forecasts: horizon {h}/10 complete", flush=True)
    return pd.DataFrame(rows)


def score_predictions(pred: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    rows = []
    for keys, part in pred.groupby(group, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        e = part["prediction"].to_numpy(float) - part["actual"].to_numpy(float)
        rows.append({**dict(zip(group, keys)), "n": len(part), "rmse_sgd": float(np.sqrt(np.mean(e ** 2))),
                     "mae_sgd": float(np.mean(np.abs(e))), "bias_sgd": float(np.mean(e))})
    return pd.DataFrame(rows)


def select_monthly(scores: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """One candidate across h=1,3,6, selected entirely before June 2024.

    Normalize MSE by persistence at each horizon so long horizons do not
    dominate. Paired origin sets are essential because long-horizon training
    has fewer mature labels; compare only dates common to every candidate.
    """
    dev = scores[(scores.phase == "development") & scores.horizon.isin([1, 3, 6])].copy()
    control = dev[dev.variant == "persistence"].set_index("horizon").rmse_sgd.pow(2)
    dev["normalized_mse"] = dev.rmse_sgd.pow(2) / dev.horizon.map(control)
    rank = dev.groupby("variant").agg(normalized_mse=("normalized_mse", "mean"),
                                      horizons=("horizon", "nunique")).sort_values("normalized_mse")
    rank = rank[rank.horizons == 3]
    baselines = ["persistence", "momentum_3m", "rent_ar"]
    stock_names = [x for x in rank.index if x not in baselines]
    industry_names = [x for x in stock_names if not x.startswith(("broad_sg", "broad_median"))]
    chosen = {"baseline": rank.loc[rank.index.intersection(baselines, sort=False)].normalized_mse.idxmin(),
              "stock": rank.loc[stock_names].normalized_mse.idxmin(),
              "industry": rank.loc[industry_names].normalized_mse.idxmin()}
    return chosen, rank.reset_index()


def block_difference(actual: np.ndarray, control: np.ndarray, candidate: np.ndarray,
                     repeats: int = 3000, block: int = 3) -> dict:
    """Paired moving-block bootstrap over months, never independent rows."""
    n = len(actual)
    if n < block:
        raise ValueError("Insufficient months for block bootstrap")
    rng = np.random.default_rng(42)
    difference = []
    for _ in range(repeats):
        starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
        positions = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        a, c, p = actual[positions], control[positions], candidate[positions]
        difference.append(np.sqrt(np.mean((p - a) ** 2)) - np.sqrt(np.mean((c - a) ** 2)))
    lower, upper = np.quantile(difference, [.025, .975])
    return {"months": n, "block_months": block, "repeats": repeats,
            "delta_rmse_ci_low": float(lower), "delta_rmse_ci_high": float(upper)}


def monthly_comparison(pred: pd.DataFrame, chosen: dict) -> pd.DataFrame:
    rows = []
    for scenario, part in (("monthly_updated_h1", pred[(pred.phase == "outer") & (pred.horizon == 1)]),
                           ("rolling_origin_h3", pred[(pred.phase == "outer") & (pred.horizon == 3)]),
                           ("rolling_origin_h6", pred[(pred.phase == "outer") & (pred.horizon == 6)]),
                           ("fixed_origin_may2024", pred[pred.origin == "2024-05"])):
        pivot = part.pivot(index="target_month", columns="variant", values="prediction").sort_index()
        actual = part.drop_duplicates("target_month").set_index("target_month").actual.reindex(pivot.index).to_numpy()
        baseline = pivot[chosen["baseline"]].to_numpy()
        names = dict.fromkeys([chosen["baseline"], chosen["stock"], chosen["industry"], "broad_sg_lag1", "broad_median_lag1"])
        for name in names:
            p = pivot[name].to_numpy()
            rmse = float(np.sqrt(np.mean((p - actual) ** 2)))
            delta = rmse - float(np.sqrt(np.mean((baseline - actual) ** 2)))
            rows.append({"scenario": scenario, "variant": name, "rmse_sgd": rmse,
                         "delta_vs_selected_rent_baseline": delta,
                         "delta_vs_matched_rent_ar": rmse - float(np.sqrt(np.mean((pivot["rent_ar"].to_numpy() - actual) ** 2))),
                         "improved_months": int(((p - actual) ** 2 < (baseline - actual) ** 2).sum()),
                         **block_difference(actual, baseline, p)})
    return pd.DataFrame(rows)


def descriptive_correlations(monthly: pd.DataFrame, features: pd.DataFrame,
                             variants: dict[str, list[str]]) -> pd.DataFrame:
    """Growth associations in development only; lag scans are exploratory."""
    # Target-month index means a delay0 stock feature is prior-month information.
    growth = monthly.adjusted_rent.diff()
    joined = features.shift(1).join(growth.rename("rent_change"))
    joined = joined[(joined.index >= pd.Period("2022-01", "M")) & (joined.index < OUTER_START)]
    rows = []
    columns = sorted(set(sum(variants.values(), [])))
    for column in columns:
        for segment, part in (("all_development", joined), ("early_2022", joined.loc[:"2022-12"]),
                              ("late_2023_2024", joined.loc["2023-01":])):
            pair = part[["rent_change", column]].dropna()
            rows.append({"feature": column, "segment": segment, "months": len(pair),
                         "pearson": pair.corr().iloc[0, 1] if len(pair) >= 5 else np.nan,
                         "spearman": pair.corr(method="spearman").iloc[0, 1] if len(pair) >= 5 else np.nan})
    return pd.DataFrame(rows)


def attach_stock_features(frame: pd.DataFrame, features: pd.DataFrame, columns: list[str],
                          cutoff: pd.Period | None = None) -> pd.DataFrame:
    """Prior calendar month, capped at the origin for fixed-origin forecasts."""
    result = frame.copy()
    months = pd.PeriodIndex(result.RENT_APPROVAL_DATE, freq="M") - 1
    if cutoff is not None:
        months = pd.PeriodIndex([min(m, cutoff) for m in months], freq="M")
    values = features.reindex(months)[columns]
    for column in columns:
        result[column] = values[column].to_numpy()
    return result


def row_experiment(features: pd.DataFrame, variants: dict[str, list[str]], out: Path) -> dict:
    """Matched 29-feature tree control; select sector on earlier row folds.

    Uses all rental rows. Outer period is existing development holdout, not
    an untouched test. No TabPFN fitting, submissions, or GPU work is needed.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from analysis.eda.economic_signals import SETTINGS
    from analysis.experiments.auxiliary_ablation import AMENITIES, BASE, BLOCK, feature_frame, model_inputs

    frame = feature_frame()
    baseline = BASE + BLOCK + AMENITIES
    options = {"no_stocks": [], "broad_sg": variants["broad_sg_lag1"],
               **{s: variants[f"{s}_lag1"] for s in SECTORS}, "all_sectors": variants["all_sectors"]}
    columns = sorted(set(sum(options.values(), [])))
    enriched = attach_stock_features(frame, features, columns)
    folds = [("2022-09", "2023-03"), ("2023-07", "2024-01"), ("2024-06", "2025-04")]
    scores, months = [], []
    for start, end in folds:
        fitting = enriched[enriched.RENT_APPROVAL_DATE < start]
        validation = enriched[(enriched.RENT_APPROVAL_DATE >= start) & (enriched.RENT_APPROVAL_DATE < end)]
        frozen = attach_stock_features(validation, features, columns, pd.Period(start, "M") - 1)
        actual = validation.MONTHLY_RENT.to_numpy(float)
        for name, extra in options.items():
            print(f"Row ablation {start}: {name} ({len(baseline) + len(extra)} features)", flush=True)
            x, q, _ = model_inputs(fitting, validation, baseline + extra)
            model = HistGradientBoostingRegressor(**SETTINGS).fit(x, fitting.MONTHLY_RENT)
            for scenario, query in (("monthly_updated", q),
                                    ("fixed_origin", model_inputs(fitting, frozen, baseline + extra)[1])):
                p = model.predict(query)
                error_frame = pd.DataFrame({"target_month": validation.RENT_APPROVAL_DATE.to_numpy(),
                                            "actual": actual, "prediction": p})
                scored = score_predictions(error_frame.assign(variant=name), ["variant"]).iloc[0].to_dict()
                scores.append({"start": start, "end_exclusive": end, "scenario": scenario,
                               "phase": "outer" if start == "2024-06" else "development",
                               "n_fit": len(fitting), "n_features": len(baseline + extra), **scored})
                for month, group in error_frame.groupby("target_month"):
                    errors = group.prediction - group.actual
                    months.append({"start": start, "scenario": scenario, "variant": name,
                                   "month": month, "n": len(group), "mse": float(np.mean(errors ** 2)),
                                   "bias": float(errors.mean())})
        pd.DataFrame(scores).to_csv(out / "row_fold_scores.csv", index=False)
        pd.DataFrame(months).to_csv(out / "row_monthly_errors.csv", index=False)
    scores, months = pd.DataFrame(scores), pd.DataFrame(months)
    selections, comparisons = {}, []
    for scenario in ("monthly_updated", "fixed_origin"):
        dev = scores[(scores.phase == "development") & (scores.scenario == scenario)].copy()
        dev["sse"] = dev.rmse_sgd.pow(2) * dev.n
        pooled = dev.groupby("variant")[["sse", "n"]].sum()
        pooled["rmse"] = np.sqrt(pooled.sse / pooled.n)
        industry = [x for x in options if x not in ("no_stocks", "broad_sg")]
        selected = pooled.loc[industry].rmse.idxmin()
        selections[scenario] = selected
        outer = scores[(scores.phase == "outer") & (scores.scenario == scenario)].set_index("variant")
        m = months[(months.start == "2024-06") & (months.scenario == scenario)]
        pivot = m.pivot(index="month", columns="variant", values="mse").sort_index()
        # Bootstrap monthly mean losses with original row counts for SGD row RMSE.
        weights = m.drop_duplicates("month").set_index("month").n.reindex(pivot.index).to_numpy()
        for name in ["no_stocks", "broad_sg", selected]:
            rng = np.random.default_rng(42)
            boot = []
            for _ in range(3000):
                starts = rng.integers(0, len(pivot) - 2, size=int(np.ceil(len(pivot) / 3)))
                positions = np.concatenate([np.arange(s, s + 3) for s in starts])[:len(pivot)]
                boot.append(np.sqrt(np.average(pivot[name].to_numpy()[positions], weights=weights[positions]))
                            - np.sqrt(np.average(pivot.no_stocks.to_numpy()[positions], weights=weights[positions])))
            low, high = np.quantile(boot, [.025, .975])
            comparisons.append({"scenario": scenario, "variant": name,
                                "selected_industry": name == selected,
                                "development_rmse": float(pooled.loc[name, "rmse"]),
                                "outer_rmse": float(outer.loc[name, "rmse_sgd"]),
                                "delta_rmse": float(outer.loc[name, "rmse_sgd"] - outer.loc["no_stocks", "rmse_sgd"]),
                                "improved_months": int((pivot[name] < pivot.no_stocks).sum()),
                                "ci_low": float(low), "ci_high": float(high)})
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(out / "row_selected_comparison.csv", index=False)
    return {"selected_industries": selections, "features": options, "baseline_features": baseline,
            "model": "HistGradientBoostingRegressor", "settings": SETTINGS, "folds": folds,
            "comparison": comparisons}


def make_plots(monthly: pd.DataFrame, panel: pd.DataFrame, comparison: pd.DataFrame,
               forecasts: pd.DataFrame, chosen: dict, out: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), layout="constrained")
    monthly[["raw_mean", "adjusted_rent"]].rename(columns={"raw_mean": "Raw mean", "adjusted_rent": "Property-adjusted mean"}).plot(ax=axes[0])
    axes[0].axvline(OUTER_START.to_timestamp(), color="#555", linestyle="--")
    axes[0].set(title="HDB rental market | adjustment fitted on 2021 only", ylabel="SGD/month", xlabel="")
    for basket in ("broad_sg", "banks", "reits_all", "developers", "travel_hospitality"):
        r = panel[f"{basket}__return_1m"].reindex(monthly.index).fillna(0)
        axes[1].plot(r.index.to_timestamp(), 100 * (1 + r).cumprod(), label=basket.replace("_", " "))
    axes[1].set(title="Stock basket paths (descriptive; forecasting uses returns)", ylabel="Index, start = 100")
    axes[1].legend(ncol=3, fontsize=8)
    fig.savefig(out / "01_rent_and_stock_paths.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    c = comparison[comparison.variant == chosen["industry"]]
    if c.empty:  # Industry can equal baseline only in an invalid configuration.
        raise AssertionError("Selected industry missing")
    x = np.arange(len(c))
    delta = c.delta_vs_selected_rent_baseline.to_numpy()
    axes[0].barh(x, delta, color=["#12857b" if v < 0 else "#c56249" for v in delta])
    axes[0].hlines(x, c.delta_rmse_ci_low, c.delta_rmse_ci_high, color="#333", linewidth=2)
    axes[0].axvline(0, color="#555", linewidth=1)
    axes[0].set_yticks(x, c.scenario.str.replace("_", " "))
    axes[0].set(title=f"Earlier-selected industry: {chosen['industry']}", xlabel="Change in index RMSE (SGD); negative improves")
    f = forecasts[forecasts.origin == "2024-05"]
    actual = f.drop_duplicates("target_month").sort_values("target_month")
    dates = pd.to_datetime(actual.target_month)
    axes[1].plot(dates, actual.actual, color="#222", linewidth=2, label="Actual adjusted rent")
    for name in dict.fromkeys([chosen["baseline"], chosen["industry"], "broad_sg_lag1"]):
        p = f[f.variant == name].sort_values("target_month")
        axes[1].plot(pd.to_datetime(p.target_month), p.prediction, marker=".", label=name.replace("_", " "))
    axes[1].set(title="All predictions made at end of May 2024", ylabel="Adjusted monthly rent (SGD)")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].legend(fontsize=8)
    fig.savefig(out / "02_selected_forecasts.png", dpi=160)
    plt.close(fig)


def write_report(out: Path, metadata: dict, rank: pd.DataFrame, comparisons: pd.DataFrame) -> None:
    chosen = metadata["monthly_selection"]
    def table(frame: pd.DataFrame, columns: list[str]) -> str:
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        for values in frame[columns].itertuples(index=False, name=None):
            lines.append("| " + " | ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in values) + " |")
        return "\n".join(lines)
    sector_table = pd.DataFrame([{"basket": k, "stocks": ", ".join(v)} for k, v in SECTORS.items()])
    text = ["# Singapore HDB rents and industry stock signals", "",
            "This is a predictive experiment, not an estimate of a causal stock-market effect. Source data and submission files were preserved.", "",
            f"Data: {metadata['rental_rows']:,} rental rows but only {metadata['rental_months']} independent calendar months (January 2021-March 2025); 43 supplied Singapore-listed stocks. All stock observations after March 2025 are excluded.", "",
            "## Decision and selected candidates", "",
            f"Earlier data selected rent-only **{chosen['baseline']}**, best stock candidate **{chosen['stock']}**, and best industry candidate **{chosen['industry']}**. These choices were frozen before evaluating June 2024-March 2025. Selection averages persistence-normalized development MSE across horizons 1, 3 and 6 months; Ridge alpha is fixed at 10.", "",
            table(comparisons, ["scenario", "variant", "rmse_sgd", "delta_vs_selected_rent_baseline", "delta_vs_matched_rent_ar", "improved_months", "delta_rmse_ci_low", "delta_rmse_ci_high"]), "",
            "Negative deltas improve rent-only forecasts. These are RMSEs of the aggregate property-adjusted monthly index, not individual property RMSEs. Intervals use paired three-month moving-block bootstrap; ten outer months and one fixed-origin episode cannot establish a robust effect.", "",
            "## Industry hypotheses", "", table(sector_table, ["basket", "stocks"]), "",
            "These are manually specified business-exposure baskets, not an exhaustive official sector taxonomy. Banks proxy credit and business conditions; REITs proxy property/financing conditions; developers proxy development sentiment; travel/hospitality proxy reopening and visitor activity; engineering proxies industrial demand; telecom infrastructure is a comparison. These mechanisms are hypotheses, not findings. Baskets overlap, notably REITs and hospitality. Industrial/commercial REIT subsets are excluded from the joint bundle to reduce duplication. Singapore listing does not imply Singapore-only revenue or HDB residential exposure. The broad basket is the supplied snapshot, not STI.", "",
            "Banks and REIT groupings were cross-checked against primary association sources. Current classifications do not reconstruct historical constituent changes. See sources below and sector_membership.csv for names supplied by the dataset.", "",
            "## Forecast design", "",
            "1. Fit a property adjustment on 2021 rentals only (town, flat type/model, area, lease commencement year); aggregate rent minus fitted property effect plus the 2021 mean. The model has no approval-date input. All scored forecast origins are after the reference period. This is a fixed-mix approximation, not an official HDB rental index.",
            "2. Baselines are persistence, extrapolated three-month momentum, and regularized rent autoregression. Rent regression inputs are latest change, trailing three/six-month mean changes, and known target-month sine/cosine. Models directly predict cumulative rent change at each horizon.",
            "3. Stock features are equal-weight monthly adjusted-close returns, compounded three-month momentum, and three-month return volatility. One basket's signals are tested at delays 0, 2, and 5 months: lag 1, 3, and 6 relative to next-month rent. Broad median-return controls match the earlier basket approach. Relative candidates add broad signals and sector-minus-broad signals; subtracting volatility is a contrast, not excess-return volatility.",
            "4. Basket returns require 80% member coverage. Prices never backfill or carry across missing months. Training-only median imputation, standardization and Ridge coefficients are refitted at each origin. Minimum training size is eight mature monthly targets; early long-horizon models are therefore fragile, and development windows differ across horizons.",
            "5. A training example at origin j and horizon h is eligible only if j+h is no later than the current origin. Market and rent information stop at that origin. Development scoring starts July 2022 and ends May 2024. The outer target months are June 2024-March 2025. Rolling h=3/6 forecasts do not know intervening rent or stock observations. The fixed-origin path uses May 2024 information for all ten horizons.", "",
            "## Earlier ranking (exploratory)", "", table(rank.head(12), ["variant", "normalized_mse", "horizons"]), "",
            "The rank explores many baskets/lags with very few months. It is not a significance test. growth_correlations.csv reports development-only raw Pearson/Spearman associations and early/late sign stability; it does not select stocks by full-sample correlation. Every candidate's forward scores are saved, including unsuccessful ones.", ""]
    if metadata.get("row_experiment"):
        row = metadata["row_experiment"]
        comp = pd.DataFrame(row["comparison"])
        selected = comp[comp.selected_industry]
        success = bool((selected.delta_rmse < 0).all() and (selected.ci_high < 0).all())
        text.extend(["## Individual-rental ablation", "",
                     "A controlled HistGradientBoosting model uses the existing 29 property/time/block/amenity features, all training rows, and the existing fixed 200-iteration settings. It adds each sector's three immediate stock features, the broad basket, or the joint bundle. Sector selection is independent of monthly-index selection: two disjoint earlier windows (September 2022-February 2023 and July-December 2023) choose the industry using row-count-weighted MSE; June 2024-March 2025 then confirms it. The monthly-updated and fixed-origin scenarios select independently on their own earlier scores.", "",
                     "Monthly-updated inputs use stocks through the prior month. Fixed-origin validation carries the last known basket features from before the fold across the forecast period; it never substitutes realized future stocks. No stock forecasting is attempted. The rental training model is frozen for both scenarios.", "",
                     table(comp, ["scenario", "variant", "development_rmse", "outer_rmse", "delta_rmse", "improved_months", "ci_low", "ci_high"]), "",
                     "Paired row-model intervals resample contiguous month blocks, retain each month's actual rental count, and recompute pooled RMSE. They are conditional on the fitted models and selected candidate, and omit selection/model-fitting uncertainty.", "",
                     ("The selected industry passes the conditional row-model improvement check in both availability scenarios. It is a candidate for a new matched TabPFN experiment, not automatic inclusion in the final submission." if success else
                      "Recommendation: retain the current submission features for now. The selected industry does not show a clear row-model improvement in both availability scenarios. A favorable aggregate correlation or retrospectively best outer sector is insufficient reason to add it."), "",
                     "This tree control does not measure incremental value for the optimized TabPFN ensemble. Before promoting any candidate, use identical TabPFN contexts/seeds and rent-history features, select on earlier origins only, and confirm on newly available rental months. The current project's outer period has already been examined by other experiments.", ""])
    text.extend(["## Limits and next evidence", "",
                 "Only 51 months cover the pandemic/reopening and a changing rent trend. Stock values were supplied as a later snapshot: survival, corporate actions, stale/delisted prices and revised adjusted prices cannot be independently reconstructed here. The broad basket includes a listing that starts September 2021; its weights change with available coverage. A frozen 2021 property model can miss changing property premiums. Rent approvals are assumed available at month end; real publication delays would require an extra lag. The ten-month holdout has been used elsewhere in this repository and is not an untouched final test. No test-set labels are available; test CSV dates extend to July 2026, much longer than the validated horizon.", "",
                 "For stronger evidence, collect earlier stock/rent history, point-in-time constituents, and economic series with actual release dates (employment, wages, migration, interest rates and housing supply). Test a small prespecified sector bundle against these controls. Predictive improvement does not identify a causal channel; this observational design cannot rule out common drivers or reopening confounding.", "",
                 "## Reproduction and outputs", "",
                 "Run `.venv/Scripts/python.exe analysis/compare_stock_sectors.py`. Use `--monthly-only` for the aggregate experiment alone, or `--output <directory>` to keep another run. `results.json` saves settings, selections and source/code hashes. Monthly forecast records include origin, horizon, target month and latest training-label date. The program checks that all input hashes remain unchanged.", "",
                 *[f"- [{key.replace('_', ' ')}]({url})" for key, url in REFERENCE_SOURCES.items()], ""])
    (out / "report.md").write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPORTS / "stock_sector_signals")
    parser.add_argument("--monthly-only", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sources = [DATA / "train.csv", AUX / "sg-stock-prices.csv"]
    if not args.monthly_only:
        sources += [AUX / f"sg-{name}.csv" for name in ("hdb-block", "mrt-stations", "shopping-malls", "schools", "coe-prices")]
    before = {str(p.relative_to(ROOT)): sha256(p) for p in sources}
    raw, stocks = pd.read_csv(DATA / "train.csv"), pd.read_csv(AUX / "sg-stock-prices.csv")
    # One thread avoids Windows sandbox named-pipe creation by joblib's
    # histogram-binning thread pool; estimator settings otherwise stay fixed.
    with threadpool_limits(limits=1):
        monthly = adjusted_rent_index(raw)
        panel, coverage = stock_panel(stocks, monthly.index.max())
        features, variants = candidate_features(panel)
        predictions = monthly_forecasts(monthly, features, variants)
        scores = score_predictions(predictions, ["phase", "horizon", "variant"])
        chosen, ranking = select_monthly(scores)
        comparison = monthly_comparison(predictions, chosen)
        monthly.to_csv(out / "monthly_rent_index.csv", index_label="month")
        panel.to_csv(out / "monthly_stock_signals.csv", index_label="month")
        coverage.to_csv(out / "stock_coverage.csv", index=False)
        predictions.to_csv(out / "monthly_forecasts.csv", index=False)
        scores.to_csv(out / "monthly_forward_scores.csv", index=False)
        ranking.to_csv(out / "development_ranking.csv", index=False)
        comparison.to_csv(out / "selected_monthly_comparison.csv", index=False)
        descriptive_correlations(monthly, features, variants).to_csv(out / "growth_correlations.csv", index=False)
        names = stocks.drop_duplicates("SYMBOL").set_index("SYMBOL").NAME
        pd.DataFrame([{"basket": b, "symbol": s, "supplied_name": names[s]}
                      for b, members in SECTORS.items() for s in members]).to_csv(out / "sector_membership.csv", index=False)
        metadata = {"rental_rows": len(raw), "rental_months": len(monthly), "outer_start": str(OUTER_START),
                    "development_start": str(DEVELOPMENT_START), "ridge_alpha": RIDGE_ALPHA,
                    "monthly_selection": chosen, "sectors": SECTORS, "features": variants,
                    "reference_sources": REFERENCE_SOURCES, "source_sha256": before,
                    "python": platform.python_version(), "sklearn": sklearn.__version__,
                    "numpy": np.__version__, "pandas": pd.__version__,
                    "thread_limit": 1,
                    "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in
                                    [Path(__file__), ROOT / "analysis/compare_stock_sectors.py", ROOT / "analysis/data.py"]}}
        if not args.monthly_only:
            metadata["row_experiment"] = row_experiment(features, variants, out)
        make_plots(monthly, panel, comparison, predictions, chosen, out)
    after = {str(p.relative_to(ROOT)): sha256(p) for p in sources}
    if before != after:
        raise AssertionError("A source CSV changed during analysis")
    metadata["input_hashes_unchanged"] = True
    (out / "results.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(out, metadata, ranking, comparison)
    print(f"Selected: {chosen}\nSaved report to {out / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
