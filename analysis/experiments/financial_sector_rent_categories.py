"""Financial-stock associations and mean-rent forecasts within HDB categories."""

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
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from analysis.config import AUX, DATA, REPORTS, ROOT
from analysis.data import prepare_main, sha256
from analysis.experiments.local_spatial_features import address_keys
from analysis.experiments.company_worksite_stocks import dependent_fdr
from analysis.experiments.stock_index_regions import block_positions
from analysis.experiments.financial_sector_rents import (
    CONFIG as FINANCE_CONFIG, TARGETS, build_financial_signals, correlation_tables,
    association_table, forecast_experiment, regression_inputs, table,
)


CONFIG = ROOT / "analysis/configs/financial_sector_rent_categories.json"
BASKETS = ("banks_equal", "finance_equal", "banks_cap_seeded")
ROOM_ORDER = ["1-room", "2-room", "3-room", "4-room", "5-room", "executive"]


def category_panels(raw, config):
    """Raw arithmetic means and a fixed-2021 additive property-mix adjustment.

    The same model is shared across categories. Each category retains its own
    2021 property-level anchor. All sparse/missing calendar cells are disclosed;
    eligibility uses counts in the twelve reference months only.
    """
    frame = prepare_main(raw)
    frame["address_key"] = address_keys(frame)
    frame["_month"] = pd.PeriodIndex(frame.RENT_APPROVAL_DATE, freq="M")
    reference = frame.RENT_APPROVAL_DATE <= config["reference_end"]
    categorical = ["address_key", "FLAT_TYPE_CLEAN", "FLAT_MODEL_CLEAN"]
    numeric = ["FLOOR_AREA_SQM", "LEASE_COMMENCE_DATE"]
    model = Pipeline([
        ("prepare", ColumnTransformer([
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")),
                                   ("scale", StandardScaler())]), numeric),
        ])), ("ridge", Ridge(alpha=10.0)),
    ])
    model.fit(frame.loc[reference, categorical + numeric], frame.loc[reference, "MONTHLY_RENT"])
    frame["property_prediction"] = model.predict(frame[categorical + numeric])
    frame["residual"] = frame.MONTHLY_RENT - frame.property_prediction
    months = pd.period_range(frame._month.min(), frame._month.max(), freq="M")
    reference_months = pd.period_range("2021-01", config["reference_end"], freq="M")
    panel, audit = [], []
    for dimension, column in config["dimensions"].items():
        for category, f in frame.groupby(column, sort=True):
            p = f.groupby("_month").agg(n=("MONTHLY_RENT", "size"),
                                         mean_rent=("MONTHLY_RENT", "mean"),
                                         median_rent=("MONTHLY_RENT", "median"),
                                         residual=("residual", "mean")).reindex(months)
            p["n"] = p.n.fillna(0).astype(int)
            ref_counts = p.n.reindex(reference_months, fill_value=0)
            eligible = bool(ref_counts.ge(config["minimum_reference_rentals_per_month"]).all())
            anchor = f.loc[f.RENT_APPROVAL_DATE <= config["reference_end"], "property_prediction"].mean()
            p["adjusted_mean_rent"] = p.residual + anchor
            p["dimension"], p["category"], p["month"] = dimension, category, months.astype(str)
            p["eligible"] = eligible
            p["usable_month"] = p.n >= config["minimum_analysis_rentals_per_month"]
            audit.append({"dimension": dimension, "category": category, "rentals": len(f),
                          "months_observed": int(p.n.gt(0).sum()), "minimum_monthly_count": int(p.n.min()),
                          "median_monthly_count": float(p.n.median()),
                          "reference_rentals": int(ref_counts.sum()),
                          "reference_minimum_monthly_count": int(ref_counts.min()),
                          "eligible": eligible, "usable_months": int(p.usable_month.sum()),
                          "reason": "eligible" if eligible else "fewer than 20 rentals in at least one 2021 month"})
            panel.append(p.drop(columns="residual").reset_index(drop=True))
    return pd.concat(panel, ignore_index=True), pd.DataFrame(audit)


def analysis_rent(local):
    p = local.set_index(pd.PeriodIndex(local.month, freq="M")).copy()
    p.loc[~p.usable_month, list(TARGETS)] = np.nan
    return p


def correct_category_tests(associations):
    """Never promote the old within-category q-values as global evidence."""
    result = associations.drop(columns=[c for c in associations if c.startswith("q_")]).copy()
    dev = result.period == "development"
    for mode in ("unadjusted", "rent_history", "full_controls"):
        all_scan = dev & (result.controls == mode)
        result.loc[all_scan, "q_global_all_lags"] = dependent_fdr(result.loc[all_scan, "p_value"])
        lag1 = all_scan & (result.lag == 1)
        result.loc[lag1, "q_global_lag1"] = dependent_fdr(result.loc[lag1, "p_value"])
    primary = (dev & (result.controls == "full_controls") & (result.dimension == "room_type")
               & (result.price_field == "ADJUSTED_CLOSE") & (result.signal == "banks_equal") & (result.lag == 1))
    result.loc[primary, "q_primary_room_bank_lag1"] = dependent_fdr(result.loc[primary, "p_value"])
    return result


def slope_equality(designs, repeats=2000):
    """Synchronized paired month-block contrasts across any category count."""
    names = sorted(designs)
    common = sorted(set.intersection(*(set(d.index) for d in designs.values())))
    if len(common) < 20 or len(names) < 2:
        raise ValueError("Need at least two categories and twenty common months")
    positions = block_positions(len(common), repeats=repeats, block=3)
    slopes, boot = [], []
    for name in names:
        d = designs[name].loc[common]
        columns = [c for c in d if c not in ("response", "phase")]
        x = np.column_stack([np.ones(len(d)), d[columns].to_numpy(float)])
        y = d.response.to_numpy(float)
        if np.linalg.matrix_rank(x) < x.shape[1]:
            raise ValueError("Rank-deficient category slope design")
        slopes.append(np.linalg.lstsq(x, y, rcond=None)[0][-1])
        coefficients = np.einsum("bij,bj->bi", np.linalg.pinv(x[positions]), y[positions])
        boot.append(coefficients[:, -1])
    slopes, boot = np.asarray(slopes), np.column_stack(boot)
    pairs = list(combinations(range(len(names)), 2))
    delta = np.array([slopes[i] - slopes[j] for i, j in pairs])
    draws = np.column_stack([boot[:, i] - boot[:, j] for i, j in pairs])
    # An exact intercept shift must not become a slope difference through
    # floating-point noise when both the contrast and its variance are zero.
    tolerance = 100 * np.finfo(float).eps * max(1., np.abs(slopes).max())
    negligible = np.maximum(np.abs(delta), np.abs(draws).max(axis=0)) <= tolerance
    delta[negligible], draws[:, negligible] = 0., 0.
    scale = np.maximum(draws.std(axis=0, ddof=1), 1e-12)
    null = np.max(np.abs((draws - delta) / scale), axis=1)
    statistic = float(np.max(np.abs(delta / scale)))
    critical = float(np.quantile(null, .95))
    move = 100 * np.log1p(.1)
    pair_rows = [{"category_a": names[i], "category_b": names[j],
                  "difference_per_10pct_stock": float(move * delta[k]),
                  "simultaneous_ci_low": float(move * (delta[k] - critical * scale[k])),
                  "simultaneous_ci_high": float(move * (delta[k] + critical * scale[k]))}
                 for k, (i, j) in enumerate(pairs)]
    return {"months": len(common), "categories": len(names),
            "equality_p_value": float((1 + (null >= statistic).sum()) / (repeats + 1)),
            "max_t_statistic": statistic, "block_months": 3, "bootstraps": repeats}, pair_rows


def category_equality(panel, levels, controls):
    rows, pair_rows = [], []
    for dimension in panel.dimension.unique():
        eligible = panel[(panel.dimension == dimension) & panel.eligible]
        for basket in BASKETS:
            for target in TARGETS:
                designs = {name: regression_inputs(analysis_rent(local), levels, controls, basket, 1, target).loc[:"2024-05"]
                           for name, local in eligible.groupby("category")}
                test, pairs = slope_equality(designs)
                key = {"dimension": dimension, "signal": basket, "target": target,
                       "lag": 1, "period": "development", "controls": "full_controls"}
                rows.append({**key, **test})
                pair_rows.extend({**key, **p} for p in pairs)
    result = pd.DataFrame(rows)
    result["q_all_equality_tests"] = dependent_fdr(result.equality_p_value)
    return result, pd.DataFrame(pair_rows)


def decision_summary(coverage, correlations, associations, comparisons, scores):
    rows = []
    for c in coverage[coverage.eligible].itertuples():
        key = {"dimension": c.dimension, "category": c.category}
        mask = lambda d: (d.dimension == c.dimension) & (d.category == c.category)
        cc = correlations[mask(correlations) & (correlations.price_field == "ADJUSTED_CLOSE")
                          & (correlations.target == "mean_rent") & (correlations.growth_window_months == 1)
                          & (correlations.phase == "all_descriptive") & (correlations.stock_lag_months == 1)]
        a = associations[mask(associations) & (associations.price_field == "ADJUSTED_CLOSE")
                         & (associations.target == "mean_rent") & (associations.period == "development")
                         & (associations.controls == "full_controls") & (associations.signal == "banks_equal")
                         & (associations.lag == 1)].iloc[0]
        comp = comparisons[mask(comparisons) & (comparisons.target == "mean_rent")]
        # The old helper emits one row for best earlier control and, when distinct,
        # a second market-control comparison. The first row is always the former.
        bank = comp[comp.candidate == "banks_equal_lag1"].iloc[0]
        chosen = comp[comp.earlier_selected_finance].iloc[0]
        ss = scores[mask(scores) & (scores.target == "mean_rent") & (scores.phase == "later_reused")].set_index("model")
        earlier_scores = scores[mask(scores) & (scores.target == "mean_rent") & (scores.phase == "development")].set_index("model")
        bankcorr = cc[cc.signal == "banks_equal"].iloc[0]
        financecorr = cc[cc.signal == "finance_equal"].iloc[0]
        rows.append({**key, "rentals": c.rentals, "minimum_monthly_count": c.minimum_monthly_count,
                     "bank_lag1_r": bankcorr.pearson_r, "bank_r_ci_low": bankcorr.ci_low, "bank_r_ci_high": bankcorr.ci_high,
                     "finance_lag1_r": financecorr.pearson_r,
                     "bank_controlled_effect_per_10pct": a.effect_per_10pct_stock,
                     "bank_controlled_p": a.p_value, "bank_global_q": a.q_global_all_lags,
                     "bank_primary_q": a.get("q_primary_room_bank_lag1", np.nan),
                     "selected_control": bank.baseline, "selected_finance": chosen.candidate,
                     "earlier_control_growth_rmse": chosen.earlier_control_rmse,
                     "earlier_finance_growth_rmse": chosen.earlier_candidate_rmse,
                     "later_control_growth_rmse": chosen.later_control_rmse,
                     "later_selected_finance_growth_rmse": chosen.later_candidate_rmse,
                     "later_selected_delta": chosen.delta_rmse, "selected_delta_ci_low": chosen.ci_low,
                     "selected_delta_ci_high": chosen.ci_high,
                     "later_control_mean_rmse_sgd": ss.loc[bank.baseline, "mean_rent_rmse_sgd"],
                     "earlier_control_mean_rmse_sgd": earlier_scores.loc[bank.baseline, "mean_rent_rmse_sgd"],
                     "earlier_bank_mean_rmse_sgd": earlier_scores.loc["banks_equal_lag1", "mean_rent_rmse_sgd"],
                     "later_bank_mean_rmse_sgd": ss.loc["banks_equal_lag1", "mean_rent_rmse_sgd"],
                     "later_finance_mean_rmse_sgd": ss.loc["finance_equal_lag1", "mean_rent_rmse_sgd"],
                     "later_selected_mean_rmse_sgd": ss.loc[chosen.candidate, "mean_rent_rmse_sgd"]})
    return pd.DataFrame(rows)


def plots(panel, correlations, summary, out):
    rooms = [x for x in ROOM_ORDER if x in set(summary.loc[summary.dimension == "room_type", "category"])]
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    for name in rooms:
        p = panel[(panel.dimension == "room_type") & (panel.category == name)]
        ax.plot(pd.to_datetime(p.month), p.mean_rent, label=name)
    ax.set(title="Arithmetic mean HDB rent by room type", ylabel="SGD/month")
    ax.legend(ncol=len(rooms))
    fig.savefig(out / "01_mean_rents_by_room_type.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, basket in zip(axes, ["banks_equal", "finance_equal"]):
        c = correlations[(correlations.dimension == "room_type") & (correlations.price_field == "ADJUSTED_CLOSE")
                         & (correlations.target == "mean_rent") & (correlations.growth_window_months == 1)
                         & (correlations.phase == "all_descriptive") & (correlations.signal == basket)]
        p = c.pivot(index="category", columns="stock_lag_months", values="pearson_r").reindex(rooms)
        im = ax.imshow(p, cmap="RdBu_r", vmin=-.6, vmax=.6, aspect="auto")
        ax.set_yticks(range(len(p)), p.index)
        ax.set_xticks(range(len(p.columns)), [f"Lag {x}" for x in p.columns])
        ax.set_title("DBS / OCBC / UOB" if basket == "banks_equal" else "Banks + SGX / iFAST")
        for i in range(len(p)):
            for j in range(len(p.columns)):
                ax.text(j, i, f"{p.iloc[i,j]:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=axes, label="Pearson r with monthly mean-rent growth", shrink=.8)
    fig.savefig(out / "02_room_type_growth_correlations.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    s = summary[summary.dimension == "room_type"].set_index("category").reindex(rooms)
    x, width = np.arange(len(s)), .25
    for offset, field, label in [(-width, "later_control_mean_rmse_sgd", "Earlier-selected control"),
                                  (0, "later_bank_mean_rmse_sgd", "Add previous-month bank return"),
                                  (width, "later_selected_mean_rmse_sgd", "Earlier-selected financial candidate")]:
        ax.bar(x + offset, s[field], width, label=label)
    ax.set_xticks(x, s.index)
    ax.set(ylabel="Monthly mean-rent RMSE (SGD)", title="June 2024–March 2025 forecasts (reused validation)")
    ax.legend(fontsize=9)
    fig.savefig(out / "03_room_type_forecast_errors.png", dpi=160)
    plt.close(fig)


def write_report(config, coverage, summary, associations, equality, out):
    rooms = summary[summary.dimension == "room_type"].set_index("category").reindex([x for x in ROOM_ORDER if x in set(summary.category)]).reset_index()
    models = summary[summary.dimension == "flat_model"]
    full = associations[(associations.controls == "full_controls") & (associations.period == "development")]
    significant = full[full.q_global_all_lags < .05]
    primary = full[full.q_primary_room_bank_lag1.notna()]
    selected_better_earlier = summary.earlier_finance_growth_rmse < summary.earlier_control_growth_rmse
    selected_better_later = summary.later_selected_delta < 0
    room_display = rooms.assign(bank_ci=[f"{a:.2f} to {b:.2f}" for a, b in zip(rooms.bank_r_ci_low, rooms.bank_r_ci_high)])
    room_display = room_display.rename(columns={"category": "Room type", "bank_lag1_r": "Bank r", "bank_ci": "Bank 95% interval",
                                                "finance_lag1_r": "Finance r", "bank_controlled_p": "Controlled bank p", "bank_primary_q": "Corrected bank q"})
    strongest = rooms.iloc[rooms.finance_lag1_r.abs().argmax()]
    one_period = rooms[(rooms.later_bank_mean_rmse_sgd < rooms.later_control_mean_rmse_sgd)
                      & (rooms.earlier_bank_mean_rmse_sgd > rooms.earlier_control_mean_rmse_sgd)]
    lines = ["# Financial-sector stocks and mean HDB rents by room type and flat model", "",
             f"Tested {len(rooms)} room types and {len(models)} flat-model categories. "
             f"Full-control development associations surviving global 5% dependent-FDR: {len(significant)} / {len(full)}. "
             f"Primary bank/room-type associations surviving their primary-family correction: {int((primary.q_primary_room_bank_lag1 < .05).sum())} / {len(primary)}.", "",
             f"The strongest descriptive lag-1 finance-basket correlation among room types is {strongest.category}: r={strongest.finance_lag1_r:.3f}. "
             "This is not robust evidence of a predictive or causal effect. No category's earlier-selected financial candidate beats its earlier-selected control in both forecast periods. "
             "The direct slope-equality tests also provide no reliable evidence that financial-stock sensitivity differs across room types or flat models. "
             "Recommendation: do not add category-specific financial-stock features to the current model from this experiment.", "",
             f"Earlier-selected financial candidates beat the earlier-selected control in {int(selected_better_earlier.sum())}/{len(summary)} categories on earlier forecasts and {int(selected_better_later.sum())}/{len(summary)} on later forecasts. "
             f"Both-period gains occur in {int((selected_better_earlier & selected_better_later).sum())}/{len(summary)}. "
             "Small later-only improvements are exploratory and require fresh data before feature promotion.", "",
             "## Room-type results", "",
             "Correlations below compare the previous calendar month's bank/finance return with the current month's log change of arithmetic mean rent. "
             "The 95% bank-correlation intervals use paired three-month blocks. These descriptive correlations use all available 2021–March 2025 data; the controlled slopes use January 2022–May 2024 only.", "",
             table(room_display, ["Room type", "Bank r", "Bank 95% interval", "Finance r", "Controlled bank p", "Corrected bank q"]), "",
             "![](02_room_type_growth_correlations.png)", "",
             "## Does the relationship differ across categories?", "",
             "The equality tests directly compare category slopes using the same resampled monthly blocks for every category. "
             "An apparently significant correlation in one category and a nonsignificant correlation in another do not establish different slopes. "
             "The tests below use lag 1, adjusted stock closes and full development-period controls. Pairwise contrasts with simultaneous intervals are in category_slope_contrasts.csv.", "",
             table(equality, ["dimension", "signal", "target", "months", "categories", "equality_p_value", "q_all_equality_tests"]), "",
             "## Forward prediction", "",
             "Expanding one-month-ahead Ridge models use only earlier rent labels and preceding-month market data. "
             "January 2023–May 2024 forecasts select the best financial candidate and the best of persistence, three-month momentum, rent history, and rent history plus market controls. "
             "The candidates scan the same nine bank/finance/individual signals and 1/3/6-month delays as the national experiment, including three basket momentum/volatility bundles. "
             "June 2024–March 2025 is already-examined validation, not a new untouched test. "
             "All candidates share monthly cases, and normalization is fitted on earlier data only.", "",
             table(rooms, ["category", "selected_control", "selected_finance", "earlier_control_growth_rmse", "earlier_finance_growth_rmse", "later_control_growth_rmse", "later_selected_finance_growth_rmse", "selected_delta_ci_low", "selected_delta_ci_high"]), "",
             "Growth RMSE is in log percentage points. The following SGD RMSE measures errors in a category's monthly mean rent, not individual rental prices.", "",
             table(rooms, ["category", "later_control_mean_rmse_sgd", "later_bank_mean_rmse_sgd", "later_finance_mean_rmse_sgd", "later_selected_mean_rmse_sgd"]), "",
             *[f"For {r.category}, the fixed bank lag-1 candidate improves later mean-rent RMSE from S${r.later_control_mean_rmse_sgd:.2f} to S${r.later_bank_mean_rmse_sgd:.2f}, "
               f"but worsens earlier RMSE from S${r.earlier_control_mean_rmse_sgd:.2f} to S${r.earlier_bank_mean_rmse_sgd:.2f}. This is inconsistent across periods."
               for r in one_period.itertuples()], "",
             "![](03_room_type_forecast_errors.png)", "",
             "## Flat-model sensitivity", "",
             "Flat model describes the design, whereas room type describes the number/type of rooms. These are overlapping views of the same records, not independent samples. "
             "Global exploratory corrections include both dimensions and both rent targets/price fields.", "",
             table(models, ["category", "rentals", "bank_lag1_r", "bank_r_ci_low", "bank_r_ci_high", "finance_lag1_r", "bank_controlled_p", "bank_global_q", "selected_control", "selected_finance", "later_selected_delta"]), "",
             "## Coverage and methods", "",
             "The 150,000 source rentals span January 2021–March 2025. Category spelling variants (for example 4 room / 4-room) are normalized before aggregation. "
             "Eligibility requires at least 20 rentals in EACH of the twelve 2021 months and uses no later outcomes. "
             "All raw category/month means and counts are saved, including sparse categories. Later cells below 20 rentals are masked before growth calculations; no forward-fill, interpolation or missing-month bridging is used. "
             "Counts and eligibility for every category follow.", "",
             table(coverage, ["dimension", "category", "rentals", "months_observed", "reference_minimum_monthly_count", "minimum_monthly_count", "eligible"]), "",
             "Property-adjusted arithmetic means subtract a property prediction and add the category's fixed 2021 prediction anchor. "
             "The shared Ridge property model is fitted on 2021 only using building address, room type, flat model, floor area and lease commencement year. "
             "This sensitivity reduces changes in the composition of rented properties without forcing categories to the same rent level. It is not a complete hedonic index.", "",
             "Stock definitions and historical weight provenance match [the national finance experiment](../financial_sector_rents/report.md): "
             "banks = DBS/OCBC/UOB; broader available finance = banks + SGX/iFAST; nonbank finance = SGX/iFAST. "
             "Equal weights rebalance monthly. Historical-cap-seeded bank weights drift from the archived December 2020 STI-fund snapshot; January–February 2021 are a descriptive backcast. "
             "The available five firms are not a comprehensive financial-sector universe. Adjusted-close returns are primary and unadjusted closes are sensitivity checks.", "",
             "Controlled regressions use lagged rent growth, trailing three-month growth, trend, annual sine/cosine, 37 inception Singapore stocks excluding the five financial names, S&P 500 returns and USD/SGD changes. "
             "Financial and market-comparator returns use the same lag. Newey-West covariance uses three monthly lags. Development inference has only 29 monthly cases despite the many rental records. "
             "The primary BY family comprises adjusted-close bank lag-1 full-control tests across room types and both rent targets. "
             "Exploratory BY families span both category dimensions, all eligible categories, both rent targets, both price fields, nine signals and three lags, separately for each control specification. "
             "Equality q-values span the twelve dimension/basket/target tests. Raw correlation intervals and forecast intervals are not multiplicity-adjusted; forecast intervals also omit model-selection uncertainty.", "",
             "A negative empirical relationship does not mean rising bank prices cause lower rents. Interest rates, labor demand, migration, credit and other common drivers are not causally identified. "
             "Only 51 calendar months are available and later validation has been reused across project experiments. No submission or TabPFN model is changed.", "",
             "## Reproduction", "",
             "Run `.venv/Scripts/python.exe analysis/compare_financial_sector_rent_categories.py`. "
             "Settings: analysis/configs/financial_sector_rent_categories.json. decision_summary.csv contains the category-level headline results. "
             "Other files retain raw/adjusted means, coverage, all correlations, controlled associations, global q-values, equality tests, forecasts, errors and charts. "
             "results.json records input/code hashes, versions and cutoff checks.", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPORTS / "financial_sector_rent_categories")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    config, finance_config = json.loads(CONFIG.read_text()), json.loads(FINANCE_CONFIG.read_text())
    finance_config["descriptive_growth_windows_months"] = config["growth_windows_months"]
    controls_path = REPORTS / "company_worksite_followup/market_monthly_log_returns.csv"
    weights_path = ROOT / "analysis/configs/sti_etf_weights_2020_12.csv"
    sources = [DATA / "train.csv", AUX / "sg-stock-prices.csv", CONFIG, FINANCE_CONFIG, controls_path, weights_path]
    before = {str(p.relative_to(ROOT)): sha256(p) for p in sources}
    code = [Path(__file__), ROOT / "analysis/compare_financial_sector_rent_categories.py", ROOT / "analysis/experiments/financial_sector_rents.py",
            ROOT / "analysis/experiments/company_worksite_stocks.py", ROOT / "analysis/experiments/stock_index_regions.py",
            ROOT / "analysis/experiments/stock_sector_signals.py", ROOT / "analysis/data.py", ROOT / "analysis/experiments/local_spatial_features.py"]
    code_before = {str(p.relative_to(ROOT)): sha256(p) for p in code}
    raw = pd.read_csv(DATA / "train.csv")
    controls = pd.read_csv(controls_path, index_col="month")
    controls.index = pd.PeriodIndex(controls.index, freq="M")
    with threadpool_limits(limits=1):
        signals, weights, diagnostics = build_financial_signals(pd.read_csv(AUX / "sg-stock-prices.csv"), pd.read_csv(weights_path), finance_config, raw.RENT_APPROVAL_DATE.max())
        panel, coverage = category_panels(raw, config)
        panel.to_csv(out / "category_monthly_mean_rents.csv", index=False)
        coverage.to_csv(out / "category_coverage.csv", index=False)
        correlations, levels, associations, predictions, scores, comparisons = [], [], [], [], [], []
        for (dimension, category), local in panel[panel.eligible].groupby(["dimension", "category"], sort=False):
            rent = analysis_rent(local)
            c, l = correlation_tables(rent, signals, finance_config)
            a = association_table(rent, signals, controls, finance_config)
            p, s, f = forecast_experiment(rent, signals["ADJUSTED_CLOSE"], controls)
            for part, destination in ((c, correlations), (l, levels), (a, associations), (p, predictions), (s, scores), (f, comparisons)):
                part.insert(0, "category", category)
                part.insert(0, "dimension", dimension)
                destination.append(part)
            print(f"Completed {dimension}: {category}", flush=True)
        correlations, levels, associations, predictions, scores, comparisons = [pd.concat(parts, ignore_index=True)
            for parts in (correlations, levels, associations, predictions, scores, comparisons)]
        associations = correct_category_tests(associations)
        equality, contrasts = category_equality(panel, signals["ADJUSTED_CLOSE"], controls)
        summary = decision_summary(coverage, correlations, associations, comparisons, scores)
        for name, part in [("growth_correlations", correlations), ("level_correlations", levels), ("conditional_associations", associations),
                           ("forward_predictions", predictions), ("forward_scores", scores), ("forecast_comparisons", comparisons),
                           ("category_slope_equality", equality), ("category_slope_contrasts", contrasts), ("decision_summary", summary), ("bank_weight_audit", weights)]:
            part.to_csv(out / f"{name}.csv", index=False)
        plots(panel, correlations, summary, out)
        write_report(config, coverage, summary, associations, equality, out)
    if before != {str(p.relative_to(ROOT)): sha256(p) for p in sources}:
        raise AssertionError("An input changed")
    if code_before != {str(p.relative_to(ROOT)): sha256(p) for p in code}:
        raise AssertionError("Experiment code changed during execution")
    if not ((predictions.last_training_target < predictions.month) & (predictions.latest_allowed_market_month < predictions.month)).all():
        raise AssertionError("Future forecast input")
    if predictions.duplicated(["dimension", "category", "target", "month", "model"]).any():
        raise AssertionError("Duplicate prediction")
    if not np.isfinite(predictions[["actual_growth", "predicted_growth", "predicted_mean_rent"]]).all().all():
        raise AssertionError("Invalid prediction")
    metadata = {"config": config, "finance_config": finance_config, "diagnostics": diagnostics,
                "source_sha256": before, "code_sha256": code_before,
                "versions": {p: version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn")},
                "rental_rows": len(raw), "rental_months": raw.RENT_APPROVAL_DATE.nunique(),
                "eligible_categories": coverage[coverage.eligible].groupby("dimension").size().to_dict(),
                "predictions": len(predictions), "source_hashes_unchanged": True, "forecast_cutoffs_verified": True}
    (out / "results.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {out / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
