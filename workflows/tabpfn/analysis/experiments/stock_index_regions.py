"""Historical-cap-seeded stock index and national/regional HDB mean rents."""

from __future__ import annotations

import argparse
from importlib.metadata import version
from itertools import combinations
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from analysis.config import AUX, DATA, REPORTS, ROOT
from analysis.data import sha256
from analysis.experiments.company_worksite_stocks import (
    dependent_fdr, hac_association, rental_coordinates,
)
from analysis.experiments.stock_sector_signals import block_difference, ridge_forecast


INDEX_NAMES = ("cap_seeded_22", "equal_weight_inception", "equal_weight_matched")
TARGETS = ("mean_rent", "adjusted_mean_rent")
OUTER_START = "2024-06"
REGIONS = ("central region", "east region", "north region", "north-east region", "west region")
BOOTSTRAPS = 2000


def construct_indices(stocks: pd.DataFrame, weights: pd.DataFrame, end: str):
    """Fixed initial notional units imply lagged value weights that drift.

    I_t = 100 sum_i w_i0 P_adj_it/P_adj_i0. The seed weights come from
    December 2020 ETF holdings, NOT present-day capitalisations. Equal-weight
    controls rebalance monthly, on the same 22 names and the inception set.
    """
    s = stocks[stocks.SYMBOL.str.endswith(".SI")].copy()
    s["DATE"] = pd.to_datetime(s.DATE, errors="raise")
    s = s[s.DATE.dt.to_period("M") <= pd.Period(end, "M")].copy()
    if s.duplicated(["SYMBOL", "DATE"]).any() or (s.ADJUSTED_CLOSE <= 0).any():
        raise ValueError("Invalid stock/date or adjusted prices")
    s["month"] = s.DATE.dt.to_period("M")
    prices = s.sort_values("DATE").groupby(["month", "SYMBOL"]).ADJUSTED_CLOSE.last().unstack()
    prices = prices.reindex(pd.period_range("2021-01", end, freq="M"))
    inception = sorted(prices.columns[prices.iloc[0].notna()])
    matched = weights[weights.symbol.isin(inception)].copy()
    if len(matched) < 10 or weights.symbol.duplicated().any():
        raise ValueError("Insufficient/ambiguous historical index constituents")
    members = matched.symbol.tolist()
    if prices[inception].isna().any().any():
        raise ValueError("Missing constituent prices: do not silently reweight using future availability")
    matched["seed_weight"] = matched.fair_value_sgd / matched.fair_value_sgd.sum()
    q = matched.set_index("symbol").seed_weight / prices.loc["2021-01", members]
    value = prices[members].mul(q, axis=1)
    levels = pd.DataFrame({"cap_seeded_22": 100 * value.sum(axis=1)})
    returns = prices.pct_change(fill_method=None)
    for name, basket in (("equal_weight_inception", inception), ("equal_weight_matched", members)):
        r = returns[basket].mean(axis=1)
        r.iloc[0] = 0.0
        levels[name] = 100 * (1 + r).cumprod()
    weight_audit = value.div(value.sum(axis=1), axis=0).rename_axis("month").stack().rename("end_month_weight").reset_index()
    weight_audit["return_weight_next_month"] = weight_audit.end_month_weight
    coverage = weights.copy()
    coverage["included"] = coverage.symbol.isin(members)
    coverage["seed_weight"] = coverage.symbol.map(matched.set_index("symbol").seed_weight)
    diagnostics = {"matched_stocks": len(members), "inception_stocks": len(inception),
                   "excluded_late_listings": sorted(set(prices.columns) - set(inception)),
                   "matched_source_value_share": float(matched.fair_value_sgd.sum() / weights.fair_value_sgd.sum()),
                   "seed_bank_weight": float(matched[matched.symbol.isin(["D05.SI", "O39.SI", "U11.SI"])].seed_weight.sum()),
                   "source_fair_value_sum": int(weights.fair_value_sgd.sum()),
                   "stale_monthly_series": [c for c in inception if (returns[c].eq(0).rolling(6).sum() == 6).any()]}
    if not np.allclose(levels.iloc[0], 100) or not np.allclose(weight_audit.groupby("month").end_month_weight.sum(), 1):
        raise AssertionError("Index base or weights failed")
    return levels, coverage, weight_audit, diagnostics


def regional_rent_panel(raw: pd.DataFrame, blocks: pd.DataFrame) -> pd.DataFrame:
    """Arithmetic means, with an optional fixed 2021 property-mix adjustment.

    Adjustment preserves each region's 2021 reference-level mean rather than
    artificially forcing expensive and inexpensive regions to the same level.
    """
    frame = rental_coordinates(raw, blocks)
    from analysis.experiments.local_spatial_features import address_keys

    b = blocks.copy()
    b["address_key"] = address_keys(b)
    frame["region"] = frame.address_key.map(b.set_index("address_key").REGION)
    if frame.region.isna().any() or set(frame.region) != set(REGIONS):
        raise ValueError("Incomplete or unexpected geographic regions")
    fit = frame.RENT_APPROVAL_DATE <= "2021-12"
    categories, numeric = ["address_key", "FLAT_TYPE_CLEAN", "FLAT_MODEL_CLEAN"], ["FLOOR_AREA_SQM", "LEASE_COMMENCE_DATE"]
    columns = categories + numeric
    pipeline = Pipeline([
        ("preprocess", ColumnTransformer([
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categories),
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")),
                                   ("scale", StandardScaler())]), numeric),
        ])), ("ridge", Ridge(alpha=10.0)),
    ])
    pipeline.fit(frame.loc[fit, columns], frame.loc[fit, "MONTHLY_RENT"])
    frame["property_prediction"] = pipeline.predict(frame[columns])
    frame["residual"] = frame.MONTHLY_RENT - frame.property_prediction
    anchors = frame[fit].groupby("region").property_prediction.mean()
    months = pd.period_range(frame._month.min(), frame._month.max(), freq="M")
    parts = []
    for region in ("Singapore", *REGIONS):
        f = frame if region == "Singapore" else frame[frame.region == region]
        anchor = frame.loc[fit, "property_prediction"].mean() if region == "Singapore" else anchors[region]
        p = f.groupby("_month").agg(n=("MONTHLY_RENT", "size"), mean_rent=("MONTHLY_RENT", "mean"),
                                    median_rent=("MONTHLY_RENT", "median"), residual=("residual", "mean")).reindex(months)
        p["adjusted_mean_rent"] = p.residual + anchor
        if (p.n < 20).any() or p[list(TARGETS)].isna().any().any() or (p[list(TARGETS)] <= 0).any().any():
            raise ValueError("Sparse/invalid national or regional mean")
        p["region"], p["month"] = region, months.astype(str)
        for target in TARGETS:
            for width in (1, 3, 12):
                p[f"{target}_growth_{width}m"] = 100 * np.log(p[target] / p[target].shift(width))
        parts.append(p.drop(columns="residual").reset_index(drop=True))
    return pd.concat(parts, ignore_index=True)


def block_positions(n: int, repeats=BOOTSTRAPS, block=3, seed=42):
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, size=(repeats, int(np.ceil(n / block))))
    return (starts[:, :, None] + np.arange(block)).reshape(repeats, -1)[:, :n]


def correlation_ci(x, y, repeats=BOOTSTRAPS):
    if len(x) < 12:
        return np.nan, np.nan
    pos = block_positions(len(x), repeats)
    a, b = x[pos], y[pos]
    a, b = a - a.mean(axis=1)[:, None], b - b.mean(axis=1)[:, None]
    denominator = np.sqrt((a ** 2).sum(axis=1) * (b ** 2).sum(axis=1))
    values = np.divide((a * b).sum(axis=1), denominator, out=np.full(repeats, np.nan), where=denominator > 0)
    return tuple(np.nanquantile(values, [.025, .975]))


def correlation_analysis(panel: pd.DataFrame, levels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, level_rows = [], []
    for region, local in panel.groupby("region", sort=False):
        local = local.set_index(pd.PeriodIndex(local.month, freq="M"))
        for index in INDEX_NAMES:
            for target in TARGETS:
                level_rows.append({"region": region, "target": target, "index": index,
                                   "pearson_levels": float(np.corrcoef(local[target], levels[index].reindex(local.index))[0, 1]),
                                   "warning": "Trending levels; not evidence of a return/growth relationship"})
                for width in (1, 3, 12):
                    changes = 100 * np.log(levels[index] / levels[index].shift(width))
                    for lag in (0, 1, 3, 6):
                        pair = pd.DataFrame({"rent_growth": local[f"{target}_growth_{width}m"],
                                             "stock_return": changes.shift(lag)}).dropna()
                        for phase, part in (("all_descriptive", pair),
                                            ("development", pair.loc["2022-01":"2024-05"]),
                                            ("outer_descriptive", pair.loc[OUTER_START:])):
                            if len(part) < 8:
                                continue
                            x, y = part.stock_return.to_numpy(), part.rent_growth.to_numpy()
                            low, high = correlation_ci(x, y) if width == 1 and lag == 1 else (np.nan, np.nan)
                            rows.append({"region": region, "target": target, "index": index, "growth_window_months": width,
                                         "stock_lag_months": lag, "phase": phase, "months": len(part),
                                         "pearson_r": float(np.corrcoef(x, y)[0, 1]),
                                         "spearman_rho": float(spearmanr(x, y).statistic),
                                         "correlation_ci_low": low, "correlation_ci_high": high})
    return pd.DataFrame(rows), pd.DataFrame(level_rows)


def regression_frame(local: pd.DataFrame, levels: pd.DataFrame, index: str, target: str):
    p = local.set_index(pd.PeriodIndex(local.month, freq="M"))
    g = p[f"{target}_growth_1m"]
    d = pd.DataFrame({"response": g, "rent_growth_lag1": g.shift(1), "rent_growth_mean3": g.shift(1).rolling(3).mean()}, index=p.index)
    d["month_trend"] = np.arange(len(d))
    d["season_sin"], d["season_cos"] = np.sin(2 * np.pi * p.index.month / 12), np.cos(2 * np.pi * p.index.month / 12)
    changes = 100 * np.log(levels[index] / levels[index].shift(1))
    d["index_return"] = changes.shift(1).reindex(d.index)
    d["phase"] = np.where(d.index.astype(str) < OUTER_START, "development", "outer")
    return d.loc["2022-01":].dropna()


def heterogeneous_slopes(designs: dict[str, pd.DataFrame], repeats=BOOTSTRAPS):
    """Synchronized month-block bootstrap; never treat regions as independent.

    Equality test uses centered bootstrap contrasts and a max-|t| statistic.
    Pairwise intervals are simultaneous across all ten region-pair contrasts.
    """
    common = sorted(set.intersection(*(set(d.index) for d in designs.values())))
    if len(common) < 20:
        raise ValueError("Too few months for regional equality inference")
    positions = block_positions(len(common), repeats)
    slopes, boots = [], []
    for region in REGIONS:
        d = designs[region].loc[common]
        columns = [c for c in d if c not in ("response", "phase")]
        x = np.column_stack([np.ones(len(d)), d[columns].to_numpy(float)])
        y = d.response.to_numpy(float)
        slopes.append(np.linalg.lstsq(x, y, rcond=None)[0][-1])
        # Batched pseudo-inverses handle the occasional near-singular resample.
        sampled_x, sampled_y = x[positions], y[positions]
        coefficient = np.einsum("bij,bj->bi", np.linalg.pinv(sampled_x), sampled_y)
        boots.append(coefficient[:, -1])
    slopes, boots = np.array(slopes), np.column_stack(boots)
    pairs = list(combinations(range(len(REGIONS)), 2))
    delta = np.array([slopes[i] - slopes[j] for i, j in pairs])
    draw = np.column_stack([boots[:, i] - boots[:, j] for i, j in pairs])
    scale = draw.std(axis=0, ddof=1)
    scale = np.maximum(scale, 1e-12)
    null_max = np.max(np.abs((draw - delta) / scale), axis=1)
    statistic = float(np.max(np.abs(delta / scale)))
    critical = float(np.quantile(null_max, .95))
    p = float((1 + (null_max >= statistic).sum()) / (repeats + 1))
    move = 100 * np.log1p(.1)
    pair_rows = [{"region_a": REGIONS[i], "region_b": REGIONS[j],
                  "effect_difference_per_10pct_stock": float(move * delta[k]),
                  "simultaneous_ci_low": float(move * (delta[k] - critical * scale[k])),
                  "simultaneous_ci_high": float(move * (delta[k] + critical * scale[k]))}
                 for k, (i, j) in enumerate(pairs)]
    return {"months": len(common), "equality_p_value": p, "max_t_statistic": statistic,
            "min_region_slope": float(slopes.min()), "max_region_slope": float(slopes.max()),
            "bootstraps": repeats, "block_months": 3}, pair_rows


def association_analysis(panel: pd.DataFrame, levels: pd.DataFrame):
    rows, equality, pairs = [], [], []
    for index in INDEX_NAMES:
        for target in TARGETS:
            for phase in ("development", "all_after_reference"):
                designs = {}
                for region, local in panel.groupby("region", sort=False):
                    d = regression_frame(local, levels, index, target)
                    if phase == "development":
                        d = d.loc[:"2024-05"]
                    rows.append({"region": region, "target": target, "index": index, "phase": phase,
                                 **hac_association(d)})
                    if region != "Singapore":
                        designs[region] = d
                test, contrasts = heterogeneous_slopes(designs)
                equality.append({"target": target, "index": index, "phase": phase, **test})
                pairs.extend({"target": target, "index": index, "phase": phase, **r} for r in contrasts)
    result, equality = pd.DataFrame(rows), pd.DataFrame(equality)
    result["q_all_associations"] = dependent_fdr(result.p_value)
    primary = (result["index"] == "cap_seeded_22") & (result.phase == "development")
    result.loc[primary, "q_primary_associations"] = dependent_fdr(result.loc[primary, "p_value"])
    equality["q_all_equality_tests"] = dependent_fdr(equality.equality_p_value)
    return result, equality, pd.DataFrame(pairs)


def forward_experiment(panel: pd.DataFrame, levels: pd.DataFrame):
    records, summary, selections = [], [], []
    for region, local in panel.groupby("region", sort=False):
        p = local.set_index(pd.PeriodIndex(local.month, freq="M"))
        for target in TARGETS:
            base = regression_frame(local, levels, INDEX_NAMES[0], target)
            history_cols = [c for c in base if c not in ("response", "phase", "index_return")]
            d = base[["response", "phase", *history_cols]].copy()
            for index in INDEX_NAMES:
                r = 100 * np.log(levels[index] / levels[index].shift(1))
                d[f"{index}_return1"] = r.shift(1).reindex(d.index)
                d[f"{index}_momentum3"] = (100 * np.log(levels[index] / levels[index].shift(3))).shift(1).reindex(d.index)
                d[f"{index}_volatility3"] = r.rolling(3).std().shift(1).reindex(d.index)
            d = d.dropna()
            group_records = []
            for i in range(12, len(d)):
                month = d.index[i]
                common = {"region": region, "target": target, "month": str(month), "phase": d.phase.iloc[i],
                          "actual_growth": float(d.response.iloc[i]), "n_training_months": i,
                          "last_training_target": str(d.index[i - 1]), "latest_market_month": str(month - 1),
                          "actual_mean_rent": float(p.loc[month, target])}
                predictions = {"persistence": 0.0, "momentum3": float(d.rent_growth_mean3.iloc[i])}
                for model in ("rent_history", *INDEX_NAMES):
                    extra = [] if model == "rent_history" else [f"{model}_{x}" for x in ("return1", "momentum3", "volatility3")]
                    x = d[history_cols + extra].to_numpy(float)
                    predictions[model] = ridge_forecast(x[:i], d.response.iloc[:i].to_numpy(float), x[i], 10.0)[0]
                for model, forecast in predictions.items():
                    group_records.append({**common, "model": model, "predicted_growth": forecast,
                                          "predicted_mean_rent": float(p.loc[month - 1, target] * np.exp(forecast / 100))})
            group = pd.DataFrame(group_records)
            records.append(group)
            scored = []
            for (phase, model), part in group.groupby(["phase", "model"], sort=False):
                error = part.predicted_growth - part.actual_growth
                scored.append({"region": region, "target": target, "phase": phase, "model": model,
                               "months": len(part), "growth_rmse": float(np.sqrt(np.mean(error ** 2))),
                               "mean_rent_rmse_sgd": float(np.sqrt(np.mean((part.predicted_mean_rent - part.actual_mean_rent) ** 2)))})
            scored = pd.DataFrame(scored)
            summary.append(scored)
            dev = scored[scored.phase == "development"].set_index("model")
            selected_control = dev.loc[["persistence", "momentum3", "rent_history"]].growth_rmse.idxmin()
            selected_stock = dev.loc[list(INDEX_NAMES)].growth_rmse.idxmin()
            outer = group[group.phase == "outer"]
            pivot = outer.pivot(index="month", columns="model", values="predicted_growth")
            actual = outer.drop_duplicates("month").set_index("month").actual_growth.reindex(pivot.index).to_numpy()
            b, s = pivot[selected_control].to_numpy(), pivot[selected_stock].to_numpy()
            control_rmse, stock_rmse = float(np.sqrt(np.mean((b - actual) ** 2))), float(np.sqrt(np.mean((s - actual) ** 2)))
            ci = block_difference(actual, b, s, repeats=2000)
            selections.append({"region": region, "target": target, "selected_control": selected_control,
                               "selected_stock": selected_stock,
                               "development_control_rmse": float(dev.loc[selected_control, "growth_rmse"]),
                               "development_stock_rmse": float(dev.loc[selected_stock, "growth_rmse"]),
                               "outer_control_rmse": control_rmse, "outer_stock_rmse": stock_rmse,
                               "delta_rmse": stock_rmse - control_rmse,
                               "delta_ci_low": ci["delta_rmse_ci_low"], "delta_ci_high": ci["delta_rmse_ci_high"]})
    return pd.concat(records, ignore_index=True), pd.concat(summary, ignore_index=True), pd.DataFrame(selections)


def plots(levels, panel, correlations, associations, out):
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), layout="constrained")
    for index in INDEX_NAMES:
        axes[0].plot(levels.index.to_timestamp(), levels[index], label=index.replace("_", " "))
    axes[0].set(title="Singapore stock-index constructions | January 2021 = 100", ylabel="Adjusted-return index")
    axes[0].legend()
    for region, part in panel.groupby("region", sort=False):
        axes[1].plot(pd.to_datetime(part.month), 100 * part.mean_rent / part.mean_rent.iloc[0],
                     linewidth=2.5 if region == "Singapore" else 1.3, label=region)
    axes[1].set(title="Arithmetic mean HDB rents by region | January 2021 = 100", ylabel="Rent index")
    axes[1].legend(ncol=3, fontsize=8)
    fig.savefig(out / "01_stock_and_regional_rent_indices.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for ax, target in zip(axes, TARGETS):
        c = correlations[(correlations["index"] == "cap_seeded_22") & (correlations.target == target)
                         & (correlations.phase == "all_descriptive") & (correlations.growth_window_months == 1)]
        pivot = c.pivot(index="region", columns="stock_lag_months", values="pearson_r").reindex(["Singapore", *REGIONS])
        im = ax.imshow(pivot, vmin=-.6, vmax=.6, cmap="RdBu_r", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)), [f"lag {lag}" for lag in pivot.columns])
        ax.set_yticks(range(len(pivot)), pivot.index)
        for i in range(len(pivot)):
            for j in range(len(pivot.columns)):
                ax.text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center", fontsize=10)
        ax.set_title("Raw mean-rent growth" if target == "mean_rent" else "Property-adjusted mean-rent growth")
    fig.colorbar(im, ax=axes, shrink=.8, label="Pearson r | monthly growth and stock returns")
    fig.savefig(out / "02_regional_growth_correlations.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, target in zip(axes, TARGETS):
        a = associations[(associations["index"] == "cap_seeded_22") & (associations.target == target)
                         & (associations.phase == "development") & (associations.region != "Singapore")].set_index("region").reindex(REGIONS)
        positions = np.arange(len(a))
        ax.hlines(positions, a.effect_ci_low, a.effect_ci_high, color="#444", linewidth=2)
        ax.plot(a.effect_per_10pct_stock, positions, "o", color="#277c80")
        ax.axvline(0, color="#888", linewidth=1)
        ax.set_yticks(positions, a.index)
        ax.set(title=target.replace("_", " "), xlabel="Next-month rent-growth association for +10% stock return\n(log points; development-only conditional 95% intervals)")
    fig.savefig(out / "03_regional_conditional_slopes.png", dpi=160)
    plt.close(fig)


def table(frame, columns):
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for values in frame[columns].itertuples(index=False, name=None):
        rows.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v) for v in values) + " |")
    return "\n".join(rows)


def write_report(out, methodology, diagnostics, correlations, associations, equality, selected):
    c = correlations[(correlations["index"] == "cap_seeded_22") & (correlations.growth_window_months == 1)
                     & (correlations.stock_lag_months == 1) & (correlations.phase == "all_descriptive")]
    primary = associations[(associations["index"] == "cap_seeded_22") & (associations.phase == "development")]
    tests = equality[equality["index"] == "cap_seeded_22"]
    national = c[(c.region == "Singapore") & (c.target == "mean_rent")].iloc[0]
    contemporaneous = correlations[(correlations["index"] == "cap_seeded_22") & (correlations.region == "Singapore")
                                    & (correlations.target == "mean_rent") & (correlations.growth_window_months == 1)
                                    & (correlations.stock_lag_months == 0) & (correlations.phase == "all_descriptive")].iloc[0]
    changes = pd.read_csv(out / "regional_change_summary.csv")
    earlier_stock_wins = int((selected.development_stock_rmse < selected.development_control_rmse).sum())
    text = ["# Overall stock index, mean HDB rents and regional differences", "",
            "This analysis constructs a historical-cap-seeded Singapore stock-return index, checks national mean-rent growth, and tests whether regional stock/rent associations differ. It extends the sector and worksite experiments; existing submissions are unchanged.", "",
            f"Result: same-month stock-return/mean-rent-growth correlation is {contemporaneous.pearson_r:.3f}. Prior-month correlation is {national.pearson_r:.3f} (paired month-block 95% interval {national.correlation_ci_low:.3f} to {national.correlation_ci_high:.3f}). The interval includes zero. Regional slope-equality tests do not establish differing stock responses; neither do they prove equal effects. Earlier stock forecasts beat the best rent-only control in {earlier_stock_wins} of {len(selected)} region/target cases. Retain the current feature set on this evidence.", "",
            "## Index construction and S&P comparison", "",
            "An S&P-style float-cap index uses I(t) = sum_i[price_i(t) x shares_i(t) x investable_float_i(t)] / divisor(t), with adjustments for corporate actions and constituent changes. [S&P's methodology](" + methodology["sp_methodology"] + ") describes the calculation. The supplied OHLC/adjusted-price/volume file has neither issuer shares nor free-float history; trading volume is not a replacement for shares outstanding.", "",
            f"The custom **cap_seeded_22** index uses December 2020 [STI-tracking ETF holdings]({methodology['source_document']}) as a proxy for historical capitalisation weights. The source total fair value sums exactly to S${diagnostics['source_fair_value_sum']:,}. The 22 matching stocks account for **{100 * diagnostics['matched_source_value_share']:.2f}%** of that portfolio; the eight unmatched stocks are disclosed in index_constituents.csv, rather than replaced with current winners. Initial bank weight is **{100 * diagnostics['seed_bank_weight']:.2f}%**.", "",
            "Let w(i,0) be each matched holding's fair value divided by the matched total. With P_adj(i,0) the January 2021 month-end adjusted close, calculate fixed notional units q(i)=w(i,0)/P_adj(i,0), then I(t)=100 sum_i q(i)P_adj(i,t). Portfolio value weights drift naturally; returns equal prior-month value weights times constituent returns. Dividends/splits are represented only as the adjusted-close source handles them. This is a fixed-universe, historical-cap-seeded return proxy, **not an exact free-float index, official STI, or S&P 500 replica**. It omits later float/share issuance and index reconstitution/divisor changes.", "",
            f"Controls are monthly-rebalanced **equal_weight_matched** (the same 22 names) and **equal_weight_inception** ({diagnostics['inception_stocks']} names priced in January 2021). The later 9CI listing is excluded from the inception basket; membership is not chosen by full-period returns. All indices start at 100 in January 2021. Source holdings were published **25 February 2021**, so the first two months are a retrospective descriptive backcast; all regression/forecast fitting starts in 2022, after publication. No current market caps enter any historical prediction.", "",
            "## Do stock returns correlate with mean-rent changes?", "",
            "The main comparison is monthly stock log return through t-1 versus 100*log(mean_rent(t)/mean_rent(t-1)). This uses actual arithmetic mean rents, not mean log rent. Same-month correlations are descriptive only; they are unavailable when forecasting that month's rent. Lag 3/6, three-month and twelve-month cumulative growth are sensitivity scans, never selected by full-period correlation. Overlapping long-window observations have fewer effective independent cases.", "",
            table(c, ["region", "target", "months", "pearson_r", "spearman_rho", "correlation_ci_low", "correlation_ci_high"]), "",
            "This table uses all available labeled months and is descriptive; block intervals resample three adjacent months together. Growth_correlations.csv includes separate 2022-May 2024 development and June 2024-March 2025 descriptive confirmation segments for every index. Level_correlations.csv records the correlations of trending levels solely as a warning: a high level correlation does not establish that stock returns explain rent growth.", "",
            "## Are regional relationships uniform?", "",
            "Rental addresses are joined to the supplied HDB block REGION field, giving Central, East, North, North-East and West. Raw means are paired with an adjusted arithmetic mean: a fixed 2021-only model removes building/flat-type/model/area/lease-year composition effects and restores each region's own 2021 reference property level. Both versions are reported because changing rental mix can create apparent regional differences.", "",
            "Actual mean-rent growth is broadly similar in direction across regions, while levels and monthly fluctuations differ. This descriptive endpoint comparison is distinct from testing whether stock-return coefficients differ:", "",
            table(changes[changes.target == "mean_rent"], ["region", "start_month", "end_month", "start_mean_sgd", "end_mean_sgd", "cumulative_change_pct"]), "",
            "Each region's regression relates its monthly rent growth to prior stock return, controlling its own lagged rent growth, trailing-three-month mean growth, a linear time trend and calendar sine/cosine. Coefficients are associations for a +10% stock return, not effects of an intervention. Development regressions use January 2022-May 2024 (29 monthly observations); conditional intervals use Newey-West covariance with three lags. Benjamini-Yekutieli q-values correct dependent national/regional tests.", "",
            table(primary, ["region", "target", "effect_per_10pct_stock", "effect_ci_low", "effect_ci_high", "p_value", "q_primary_associations"]), "",
            "The formal regional equality test estimates all five slopes, then resamples the SAME contiguous month blocks for every region, preserving common shocks and cross-region dependence. It tests the largest standardized pairwise slope contrast against the centered bootstrap null. Pairwise_region_contrasts.csv gives simultaneous intervals across the ten region pairs. We do not count five regional rows as five independent stock observations.", "",
            table(tests, ["target", "phase", "months", "equality_p_value", "q_all_equality_tests"]), "",
            "A large equality-test p-value means insufficient evidence of different slopes; it does not prove identical effects. The all_after_reference analysis includes the confirmation period and is descriptive sensitivity evidence. Regional_equality_tests.csv also includes the equal-weight alternatives and corrects across all twelve equality checks.", "",
            "## Forward prediction check", "",
            "Expanding one-month-ahead Ridge forecasts fit only already observed rent-growth labels, with training-only normalization. Baselines are zero rent growth (carry forward the last mean), trailing-three-month momentum, and rent-history/trend/seasonality regression. Each stock index adds prior return, prior three-month momentum and prior three-month volatility. Fitting starts in January 2022 with at least twelve mature months; earlier January 2023-May 2024 forecasts choose the control and stock index, then June 2024-March 2025 confirms them. The stock candidate remains forced for reporting even when earlier errors favor no stocks.", "",
            table(selected, ["region", "target", "selected_control", "selected_stock", "development_control_rmse", "development_stock_rmse", "outer_control_rmse", "outer_stock_rmse", "delta_rmse", "delta_ci_low", "delta_ci_high"]), "",
            "RMSE above is rent-growth log points, not individual-rental SGD RMSE. Forward_scores.csv additionally reports errors in predicted monthly mean SGD rent. Conditional paired block-bootstrap intervals omit selection/fitting uncertainty and are not adjusted for all model comparisons. This monthly-updated exercise assumes preceding-month rents are published; it is not a fixed-origin forecast of the entire Kaggle test period.", "",
            "## Practical limits and feature decision", "",
            "Use a stock-index feature only if it improves earlier and later forecasts against rent-history controls, with a stable interpretation across index definitions and regions. A correlation of levels, one favorable lag, or one favorable region is insufficient. These observational relationships cannot establish stock-market causation; employment, migration, interest rates, housing supply and reopening can drive both markets. New rent data and a matched TabPFN ablation are needed before changing the final feature ensemble.", "",
            "There are only 51 labeled months with serial dependence; the confirmation period has been examined elsewhere in this repository. Historical-cap coverage is incomplete, initial weights can become stale, companies can merge, and the price file is a later adjusted-history snapshot. A fixed 2021 mix model misses changing property premiums. Neither commercial employment exposure nor regional causal treatment is measured. Exact reproduction of a float-cap index requires point-in-time issuer/free-float shares, constituent changes and corporate-action data beyond this file.", "",
            "## Reproduction", "",
            "Run `.venv/Scripts/python.exe analysis/compare_stock_index_regions.py`. Inputs are read-only. The seed table is analysis/configs/sti_etf_weights_2020_12.csv; its source, publication date and limitations are in stock_index_methodology.json. Outputs include index levels/returns, monthly constituent weights, all regional rent panels, correlations, slope/equality tests, forward predictions, earlier selections and charts. results.json stores source/code hashes, versions, settings and cutoff checks.", ""]
    (out / "report.md").write_text("\n".join(text), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPORTS / "stock_index_regions")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    weights_path = ROOT / "analysis/configs/sti_etf_weights_2020_12.csv"
    method_path = ROOT / "analysis/configs/stock_index_methodology.json"
    sources = [DATA / "train.csv", AUX / "sg-hdb-block.csv", AUX / "sg-stock-prices.csv", weights_path, method_path]
    before = {str(p.relative_to(ROOT)): sha256(p) for p in sources}
    methodology = json.loads(method_path.read_text())
    raw = pd.read_csv(DATA / "train.csv")
    weights = pd.read_csv(weights_path)
    if weights.fair_value_sgd.sum() != methodology["reported_portfolio_value_sgd"]:
        raise AssertionError("Transcribed holdings do not reconcile with the source portfolio")
    with threadpool_limits(limits=1):
        levels, constituents, dynamic_weights, diagnostics = construct_indices(pd.read_csv(AUX / "sg-stock-prices.csv"), weights, raw.RENT_APPROVAL_DATE.max())
        panel = regional_rent_panel(raw, pd.read_csv(AUX / "sg-hdb-block.csv"))
        print("Constructed indices and regional raw/adjusted mean rents", flush=True)
        correlations, level_corr = correlation_analysis(panel, levels)
        print("Completed national and regional growth correlations", flush=True)
        associations, equality, pairs = association_analysis(panel, levels)
        print("Completed synchronized regional slope-equality tests", flush=True)
        predictions, scores, selected = forward_experiment(panel, levels)
        print("Completed expanding-window forecasts", flush=True)
        levels.to_csv(out / "monthly_index_levels.csv", index_label="month")
        (100 * np.log(levels / levels.shift(1))).to_csv(out / "monthly_index_log_returns.csv", index_label="month")
        constituents.to_csv(out / "index_constituents.csv", index=False)
        dynamic_weights.to_csv(out / "monthly_constituent_weights.csv", index=False)
        panel.to_csv(out / "monthly_regional_rents.csv", index=False)
        changes = []
        for region, part in panel.groupby("region", sort=False):
            for target in TARGETS:
                start, end = float(part[target].iloc[0]), float(part[target].iloc[-1])
                changes.append({"region": region, "target": target, "start_month": part.month.iloc[0],
                                "end_month": part.month.iloc[-1], "start_mean_sgd": start, "end_mean_sgd": end,
                                "cumulative_change_pct": 100 * (end / start - 1),
                                "monthly_growth_std_logpoints": float(part[f"{target}_growth_1m"].std())})
        pd.DataFrame(changes).to_csv(out / "regional_change_summary.csv", index=False)
        correlations.to_csv(out / "growth_correlations.csv", index=False)
        level_corr.to_csv(out / "level_correlations.csv", index=False)
        associations.to_csv(out / "conditional_associations.csv", index=False)
        equality.to_csv(out / "regional_equality_tests.csv", index=False)
        pairs.to_csv(out / "pairwise_region_contrasts.csv", index=False)
        predictions.to_csv(out / "forward_predictions.csv", index=False)
        scores.to_csv(out / "forward_scores.csv", index=False)
        selected.to_csv(out / "earlier_selected_forecast_comparison.csv", index=False)
        plots(levels, panel, correlations, associations, out)
    if before != {str(p.relative_to(ROOT)): sha256(p) for p in sources}:
        raise AssertionError("An input changed")
    if not ((predictions.last_training_target < predictions.month) & (predictions.latest_market_month < predictions.month)).all():
        raise AssertionError("Forecast information cutoff violated")
    metadata = {"methodology": methodology, "index_diagnostics": diagnostics, "rental_rows": len(raw),
                "rental_months": raw.RENT_APPROVAL_DATE.nunique(), "source_sha256": before,
                "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__),
                                ROOT / "analysis/compare_stock_index_regions.py", ROOT / "analysis/experiments/company_worksite_stocks.py",
                                ROOT / "analysis/experiments/stock_sector_signals.py"]},
                "versions": {p: version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn")},
                "settings": {"reference_year": 2021, "development_start": "2022-01", "outer_start": OUTER_START,
                             "bootstrap_repeats": BOOTSTRAPS, "block_months": 3, "ridge_alpha": 10, "thread_limit": 1},
                "source_hashes_unchanged": True, "forecast_cutoffs_verified": True}
    (out / "results.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(out, methodology, diagnostics, correlations, associations, equality, selected)
    print(f"Saved {out / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
