"""Dedicated finance-stock versus national HDB arithmetic mean-rent analysis."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

from analysis.config import AUX, DATA, REPORTS, ROOT
from analysis.data import sha256
from analysis.experiments.stock_index_regions import regional_rent_panel, block_positions
from analysis.experiments.company_worksite_stocks import hac_association, dependent_fdr
from analysis.experiments.stock_sector_signals import ridge_forecast, block_difference


CONFIG = ROOT / "analysis/configs/financial_sector_rents.json"
TARGETS = ("mean_rent", "adjusted_mean_rent")
FORECAST_BASKETS = ("banks_equal", "banks_cap_seeded", "finance_equal")


def build_financial_signals(stocks, seed_weights, config, end="2025-03"):
    """Returns and indices; cap seed uses documented 2020 bank weights only."""
    s = stocks[stocks.SYMBOL.str.endswith(".SI")].copy()
    s["DATE"] = pd.to_datetime(s.DATE, errors="raise")
    s = s[s.DATE.dt.to_period("M") <= pd.Period(end, "M")].copy()
    s["month"] = s.DATE.dt.to_period("M")
    if s.duplicated(["SYMBOL", "DATE"]).any():
        raise ValueError("Duplicate stock/date")
    bank_weights = seed_weights.set_index("symbol").loc[config["banks"], "fair_value_sgd"]
    bank_weights = bank_weights / bank_weights.sum()
    signals, weight_rows, diagnostics = {}, [], {}
    for price in config["price_fields"]:
        closes = s.sort_values("DATE").groupby(["month", "SYMBOL"])[price].last().unstack()
        closes = closes.reindex(pd.period_range("2021-01", end, freq="M"))
        inception = sorted(closes.columns[closes.iloc[0].notna()])
        finance = config["financial_services"]
        if set(finance) - set(inception):
            raise ValueError("A financial basket member lacks inception prices")
        nonfinance = [c for c in inception if c not in finance]
        if closes[inception].isna().any().any() or (closes[inception] <= 0).any().any():
            raise ValueError("Missing/nonpositive constituent prices; do not backfill/reweight")
        r = closes.pct_change(fill_method=None)
        levels = pd.DataFrame(index=closes.index)
        for name, members in (("banks_equal", config["banks"]), ("finance_equal", finance),
                              ("nonbank_finance_equal", config["nonbank_finance"]),
                              ("nonfinancial_market", nonfinance)):
            monthly = r[members].mean(axis=1)
            monthly.iloc[0] = 0.0
            levels[name] = 100 * (1 + monthly).cumprod()
        value = closes[config["banks"]].div(closes.loc["2021-01", config["banks"]], axis=1).mul(bank_weights, axis=1)
        levels["banks_cap_seeded"] = 100 * value.sum(axis=1)
        for symbol in finance:
            levels[symbol] = 100 * closes[symbol] / closes.loc["2021-01", symbol]
        signals[price] = levels
        drift = value.div(value.sum(axis=1), axis=0)
        for month in drift.index:
            weight_rows.extend({"price_field": price, "month": str(month), "symbol": c,
                                "end_month_weight": float(drift.loc[month, c]),
                                "seed_weight": float(bank_weights[c])} for c in config["banks"])
        diagnostics[price] = {"nonfinancial_market_stocks": len(nonfinance), "nonfinancial_members": nonfinance,
                              "late_listings_excluded": sorted(set(closes) - set(inception))}
    return signals, pd.DataFrame(weight_rows), diagnostics


def block_correlation_interval(x, y, window=1, repeats=2000):
    block = max(3, window)
    if len(x) < 2 * block:
        return np.nan, np.nan
    pos = block_positions(len(x), repeats=repeats, block=block)
    a, b = x[pos], y[pos]
    a, b = a - a.mean(axis=1)[:, None], b - b.mean(axis=1)[:, None]
    denominator = np.sqrt((a ** 2).sum(axis=1) * (b ** 2).sum(axis=1))
    values = np.divide((a * b).sum(axis=1), denominator, out=np.full(repeats, np.nan), where=denominator > 0)
    return tuple(np.nanquantile(values, [.025, .975]))


def correlation_tables(rent, signals, config):
    rows, level_rows = [], []
    for price, levels in signals.items():
        selected = [c for c in levels if c != "nonfinancial_market"]
        for target in TARGETS:
            for name in selected:
                level_rows.append({"price_field": price, "target": target, "signal": name,
                                   "pearson_levels": float(np.corrcoef(rent[target], levels[name].reindex(rent.index))[0, 1]),
                                   "note": "Descriptive trending levels; not a growth association or forecast input"})
                for window in config["descriptive_growth_windows_months"]:
                    growth = 100 * np.log(rent[target] / rent[target].shift(window))
                    returns = 100 * np.log(levels[name] / levels[name].shift(window))
                    for lag in config["lags_months"]:
                        pair = pd.DataFrame({"rent_growth": growth, "financial_return": returns.shift(lag)}).dropna()
                        periods = [("all_descriptive", pair), ("development", pair.loc["2022-01":"2024-05"]),
                                   ("later_reused", pair.loc["2024-06":])]
                        for phase, p in periods:
                            if len(p) < 8:
                                continue
                            x, y = p.financial_return.to_numpy(), p.rent_growth.to_numpy()
                            low, high = block_correlation_interval(x, y, window)
                            rows.append({"price_field": price, "target": target, "signal": name,
                                         "growth_window_months": window, "stock_lag_months": lag,
                                         "phase": phase, "months": len(p), "pearson_r": float(np.corrcoef(x, y)[0, 1]),
                                         "spearman_rho": float(spearmanr(x, y).statistic),
                                         "ci_low": low, "ci_high": high, "bootstrap_block_months": max(3, window)})
    return pd.DataFrame(rows), pd.DataFrame(level_rows)


def regression_inputs(rent, levels, controls, signal, lag, target, mode="full_controls"):
    g = 100 * np.log(rent[target] / rent[target].shift(1))
    r = 100 * np.log(levels / levels.shift(1))
    d = pd.DataFrame(index=rent.index)
    d["response"] = g
    if mode != "unadjusted":
        d["rent_growth_lag1"] = g.shift(1)
        d["rent_growth_mean3"] = g.shift(1).rolling(3).mean()
        d["trend"] = np.arange(len(d))
        d["season_sin"] = np.sin(2 * np.pi * d.index.month / 12)
        d["season_cos"] = np.cos(2 * np.pi * d.index.month / 12)
    if mode == "full_controls":
        d["nonfinancial_market"] = r.nonfinancial_market.shift(lag).reindex(d.index)
        d["US_market"] = controls.SP500.shift(lag).reindex(d.index)
        d["USD_SGD"] = controls.DEXSIUS.shift(lag).reindex(d.index)
    # Keep the signal last for the shared HAC partial-slope estimator.
    d["financial_return"] = r[signal].shift(lag).reindex(d.index)
    d["phase"] = np.where(d.index.astype(str) <= "2024-05", "development", "later_reused")
    return d.loc["2022-01":].dropna()


def association_table(rent, signals, controls, config):
    rows = []
    for price, levels in signals.items():
        for target in TARGETS:
            for signal in [c for c in levels if c != "nonfinancial_market"]:
                for lag in (1, 3, 6):
                    for mode in ("unadjusted", "rent_history", "full_controls"):
                        d = regression_inputs(rent, levels, controls, signal, lag, target, mode)
                        for period, part in (("development", d.loc[:"2024-05"]), ("all_after_reference", d)):
                            rows.append({"price_field": price, "target": target, "signal": signal, "lag": lag,
                                         "controls": mode, "period": period, **hac_association(part)})
    result = pd.DataFrame(rows)
    dev = (result.price_field == "ADJUSTED_CLOSE") & (result.period == "development")
    for mode in ("unadjusted", "rent_history", "full_controls"):
        scan = dev & (result.controls == mode)
        result.loc[scan, "q_all_lag_company_tests"] = dependent_fdr(result.loc[scan, "p_value"])
        primary = scan & (result.lag == 1)
        result.loc[primary, "q_lag1_finance_tests"] = dependent_fdr(result.loc[primary, "p_value"])
    return result


def forecast_experiment(rent, levels, controls):
    rows, scores, comparisons = [], [], []
    returns = 100 * np.log(levels / levels.shift(1))
    candidates = {}
    for signal in [c for c in levels if c != "nonfinancial_market"]:
        for lag in (1, 3, 6):
            candidates[f"{signal}_lag{lag}"] = pd.DataFrame({f"{signal}_return_lag{lag}": returns[signal].shift(lag)})
    for signal in FORECAST_BASKETS:
        for lag in (1, 3, 6):
            candidates[f"{signal}_bundle_lag{lag}"] = pd.DataFrame({
                f"{signal}_bundle_return": returns[signal].shift(lag),
                f"{signal}_bundle_momentum": (100 * np.log(levels[signal] / levels[signal].shift(3))).shift(lag),
                f"{signal}_bundle_volatility": returns[signal].rolling(3).std().shift(lag)})
    for target in TARGETS:
        d = regression_inputs(rent, levels, controls, "banks_equal", 1, target, "full_controls")
        rent_columns = ["rent_growth_lag1", "rent_growth_mean3", "trend", "season_sin", "season_cos"]
        market_columns = rent_columns + ["nonfinancial_market", "US_market", "USD_SGD"]
        feature_sets = {"rent_history": (d[rent_columns], rent_columns), "rent_market_control": (d[market_columns], market_columns)}
        for name, extra in candidates.items():
            f = d[market_columns].join(extra)
            feature_sets[name] = (f, list(f))
        # The same complete monthly cases for every candidate and baseline.
        common = d.index
        for f, columns in feature_sets.values():
            common = common.intersection(f.dropna().index)
        d = d.loc[common]
        group_rows = []
        for i in range(12, len(d)):
            month = d.index[i]
            actual = float(d.response.iloc[i])
            result = {"persistence": 0., "momentum3": float(d.rent_growth_mean3.iloc[i])}
            for name, (features, columns) in feature_sets.items():
                f = features.loc[common, columns].to_numpy(float)
                result[name] = ridge_forecast(f[:i], d.response.iloc[:i].to_numpy(), f[i], 10.)[0]
            for name, predicted in result.items():
                group_rows.append({"target": target, "month": str(month), "phase": d.phase.iloc[i], "model": name,
                                   "actual_growth": actual, "predicted_growth": predicted,
                                   "actual_mean_rent": float(rent.loc[month, target]),
                                   "predicted_mean_rent": float(rent.loc[month - 1, target] * np.exp(predicted / 100)),
                                   "last_training_target": str(d.index[i - 1]), "latest_allowed_market_month": str(month - 1),
                                   "training_months": i})
        pred = pd.DataFrame(group_rows)
        rows.append(pred)
        target_scores = []
        for (phase, model), part in pred.groupby(["phase", "model"], sort=False):
            error = part.predicted_growth - part.actual_growth
            target_scores.append({"target": target, "phase": phase, "model": model, "months": len(part),
                                  "growth_rmse": float(np.sqrt(np.mean(error ** 2))),
                                  "mean_rent_rmse_sgd": float(np.sqrt(np.mean((part.predicted_mean_rent - part.actual_mean_rent) ** 2)))})
        target_scores = pd.DataFrame(target_scores)
        scores.append(target_scores)
        dev = target_scores[target_scores.phase == "development"].set_index("model")
        controls_names = ["persistence", "momentum3", "rent_history", "rent_market_control"]
        selected_control = dev.loc[controls_names].growth_rmse.idxmin()
        selected_finance = dev.loc[list(candidates)].growth_rmse.idxmin()
        primary_bank = "banks_equal_lag1"
        later = pred[pred.phase == "later_reused"]
        p = later.pivot(index="month", columns="model", values="predicted_growth")
        actual = later.drop_duplicates("month").set_index("month").actual_growth.reindex(p.index).to_numpy()
        for candidate in dict.fromkeys([primary_bank, selected_finance, "finance_equal_lag1", "banks_cap_seeded_lag1"]):
            for baseline in dict.fromkeys([selected_control, "rent_market_control"]):
                b, c = p[baseline].to_numpy(), p[candidate].to_numpy()
                base_rmse = float(np.sqrt(np.mean((b - actual) ** 2)))
                candidate_rmse = float(np.sqrt(np.mean((c - actual) ** 2)))
                ci = block_difference(actual, b, c, repeats=2000)
                comparisons.append({"target": target, "candidate": candidate, "baseline": baseline,
                                    "earlier_selected_finance": candidate == selected_finance,
                                    "earlier_control_rmse": float(dev.loc[baseline, "growth_rmse"]),
                                    "earlier_candidate_rmse": float(dev.loc[candidate, "growth_rmse"]),
                                    "later_control_rmse": base_rmse, "later_candidate_rmse": candidate_rmse,
                                    "delta_rmse": candidate_rmse - base_rmse,
                                    "ci_low": ci["delta_rmse_ci_low"], "ci_high": ci["delta_rmse_ci_high"]})
    return pd.concat(rows, ignore_index=True), pd.concat(scores, ignore_index=True), pd.DataFrame(comparisons)


def table(frame, columns):
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for values in frame[columns].itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v) for v in values) + " |")
    return "\n".join(lines)


def plots(rent, signals, correlations, comparisons, out):
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), layout="constrained")
    levels = signals["ADJUSTED_CLOSE"]
    for name in ("banks_equal", "banks_cap_seeded", "finance_equal", "nonfinancial_market"):
        axes[0].plot(levels.index.to_timestamp(), levels[name], label=name.replace("_", " "))
    axes[0].set(title="Financial-stock baskets | January 2021 = 100", ylabel="Adjusted-return index")
    axes[0].legend(ncol=2)
    for target in TARGETS:
        axes[1].plot(rent.index.to_timestamp(), rent[target], label=target.replace("_", " "))
    axes[1].set(title="Singapore arithmetic mean monthly HDB rent", ylabel="SGD/month")
    axes[1].legend()
    fig.savefig(out / "01_finance_and_mean_rents.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
    for ax, target in zip(axes, TARGETS):
        c = correlations[(correlations.price_field == "ADJUSTED_CLOSE") & (correlations.target == target)
                         & (correlations.growth_window_months == 1) & (correlations.phase == "all_descriptive")]
        p = c.pivot(index="signal", columns="stock_lag_months", values="pearson_r")
        image = ax.imshow(p, cmap="RdBu_r", vmin=-.6, vmax=.6, aspect="auto")
        ax.set_yticks(range(len(p)), p.index)
        ax.set_xticks(range(len(p.columns)), [f"lag {x}" for x in p.columns])
        for i in range(len(p)):
            for j in range(len(p.columns)):
                ax.text(j, i, f"{p.iloc[i,j]:.2f}", ha="center", va="center", fontsize=9)
        ax.set_title(target.replace("_", " "))
    fig.colorbar(image, ax=axes, label="Pearson r | monthly financial returns vs mean-rent growth", shrink=.8)
    fig.savefig(out / "02_financial_growth_correlations.png", dpi=160)
    plt.close(fig)


def write_report(config, diagnostics, correlations, associations, comparisons, level_corr, out):
    main = correlations[(correlations.price_field == "ADJUSTED_CLOSE") & (correlations.target == "mean_rent")
                        & (correlations.growth_window_months == 1) & (correlations.phase == "all_descriptive")
                        & correlations.signal.isin(["banks_equal", "banks_cap_seeded", "finance_equal", "nonbank_finance_equal"])]
    primary = main[(main.signal == "banks_equal") & (main.stock_lag_months == 1)].iloc[0]
    controlled = associations[(associations.price_field == "ADJUSTED_CLOSE") & (associations.period == "development")
                              & (associations.target == "mean_rent") & (associations.lag == 1)
                              & (associations.controls == "full_controls")]
    dev_scan = associations[(associations.price_field == "ADJUSTED_CLOSE") & (associations.period == "development")
                            & (associations.controls == "full_controls")]
    chosen = comparisons[(comparisons.target == "mean_rent") & comparisons.earlier_selected_finance & (comparisons.baseline == "persistence")]
    if chosen.empty:
        chosen = comparisons[(comparisons.target == "mean_rent") & comparisons.earlier_selected_finance].head(1)
    candidate = chosen.iloc[0]
    text = ["# Financial-sector stocks and Singapore mean HDB rents", "",
            f"Primary result: the equal-weight DBS/OCBC/UOB basket's prior-month return correlates {primary.pearson_r:.3f} with next-month national arithmetic mean-rent growth (n={int(primary.months)}, paired month-block 95% interval {primary.ci_low:.3f} to {primary.ci_high:.3f}). This is an association, not a causal effect. See the forecasting checks below before using financial-stock features.", "",
            f"Decision: retain the current feature set. No full-control development association survives the dependent-FDR screen. Earlier forecasts selected {candidate.candidate}; later growth RMSE is {candidate.later_candidate_rmse:.3f}, versus {candidate.later_control_rmse:.3f} for {candidate.baseline}. The original bank lag-1 hypothesis also fails to beat the best rent-only control. These later months have already been examined in related experiments.", "",
            "## Baskets and weighting", "",
            "Banks: DBS (D05.SI), OCBC (O39.SI), UOB (U11.SI). Broader available financial-services basket: those three banks plus Singapore Exchange (S68.SI) and iFAST (AIY.SI). Nonbank sensitivity basket: SGX/iFAST. Five individual stocks are also reported, without selecting companies on full-period rent correlation. This is the available-file universe, not a comprehensive financial-sector index; insurers and other missing firms are not imputed.", "",
            "Equal-weight baskets rebalance monthly. The bank cap-seeded sensitivity uses only December 2020 historical STI-fund fair-value weights, initialized at January 2021 month-end prices and allowed to drift; it is not a point-in-time free-float index. That snapshot was published 25 February 2021, before all 2022-onward regression and forecast fits. January-February 2021 index paths are a descriptive backcast. Main returns use adjusted closes; CLOSE-based calculations are a sensitivity check for dividend/adjustment dependence, not an independently reconstructed corporate-action series.", "",
            f"The nonfinancial market control contains {diagnostics['ADJUSTED_CLOSE']['nonfinancial_market_stocks']} January-2021 Singapore-listed inception stocks, explicitly excluding all five financial names. This prevents a bank-heavy broad-market index from being the only sector comparator. Later listings are not added retrospectively.", "",
            "## Correlations of changes, not just levels", "",
            "Rent growth is 100*log(arithmetic_mean_rent(t)/arithmetic_mean_rent(t-1)); stock returns use log changes in basket levels. Lag 1 means the preceding calendar month's return; lag 6 means the single-month return six months earlier, not a six-month cumulative return. Lag 0 is descriptive and cannot predict a month's rent before that month's stocks are observed. We also scan three- and twelve-month cumulative changes as descriptive sensitivity checks; overlapping windows have few independent cases.", "",
            table(main, ["signal", "stock_lag_months", "months", "pearson_r", "spearman_rho", "ci_low", "ci_high"]), "",
            "Intervals resample paired contiguous monthly blocks (length max(3, growth window)), not individual rentals. A long-window interval is omitted if fewer than two blocks are available. Full-sample correlations are descriptive; growth_correlations.csv separately reports January 2022-May 2024 development and the already-examined June 2024-March 2025 period. Trending index/mean-rent level correlations are saved separately as a spurious-correlation diagnostic, never used for model selection.", "",
            "Raw mean rents are paired with an adjusted arithmetic mean using a property model fitted on 2021 only (building, flat type/model, floor area and lease year), preserving the national reference level. This tests whether changing rental composition drives a correlation. Both rent targets and both price fields are fully disclosed.", "",
            "## Does finance add an association beyond rent history and other markets?", "",
            "Development-only monthly regressions use Newey-West covariance with three monthly lags. Unadjusted slopes, rent-history/trend/seasonality controls, and full controls are reported. Full controls add nonfinancial Singapore returns, US S&P 500 returns and USD/SGD changes at the same lagged signal month. No same/future-month stock observation enters a predictive regression. No interest-rate, wage, employment or migration causal identification is claimed.", "",
            table(controlled, ["signal", "months", "effect_per_10pct_stock", "effect_ci_low", "effect_ci_high", "p_value", "q_lag1_finance_tests"]), "",
            "Effects above are rent-growth log points associated with a +10% stock/basket move, not SGD rent increases. Benjamini-Yekutieli q-values account for dependent tests: q_lag1_finance_tests covers the nine lag-1 signals times two rent targets; q_all_lag_company_tests covers all 1/3/6-month lags for each control specification. These are small-sample approximations; the 29 development months provide limited information for a many-control regression.", "",
            f"Full-control development scans surviving a 5% dependent-FDR threshold: {int((dev_scan.q_all_lag_company_tests < .05).sum())} out of {len(dev_scan)}.", "",
            "## Incremental forecast experiment", "",
            "One-month-ahead expanding Ridge (alpha 10) starts after twelve mature labels from January 2022. Baselines are zero growth/carry-forward, trailing-three-month momentum, rent-history regression and rent-history plus nonfinancial/US/FX markets. Each financial signal is tested at lag 1/3/6; the three main baskets also have return/momentum/volatility bundles. Earlier January 2023-May 2024 forecasts select a financial candidate and the best control. June 2024-March 2025 is reused confirmation, not an untouched final test. All candidates use the same monthly cases and training-only normalization.", "",
            table(comparisons, ["target", "candidate", "baseline", "earlier_selected_finance", "earlier_control_rmse", "earlier_candidate_rmse", "later_control_rmse", "later_candidate_rmse", "delta_rmse", "ci_low", "ci_high"]), "",
            "Forecast errors above are mean-rent growth log points, not individual-rental SGD RMSE. forward_scores.csv additionally reports SGD errors of reconstructed monthly means. Block intervals are conditional on fitted models and omit model/lag selection uncertainty. All candidate errors, including unfavorable ones, are retained.", "",
            "## Interpretation", "",
            "A stable finance-return association could proxy expectations about credit, economic conditions, employment or bonuses; a negative or positive empirical sign alone does not identify any of those mechanisms. Bank prices also reflect overseas business, interest margins and discount rates. Stock and rent levels can trend together while monthly changes are unrelated. Adding financial-sector features requires earlier/later forecast gains beyond rent history and nonfinancial-market controls, followed by a matched row-level model experiment and new rental data. No submission or TabPFN model is modified here.", "",
            "The rental data cover only 51 months spanning reopening and a changing rental trend. Historical company constituents, free floats and data-release vintages are incomplete. The later ten months were already examined in related project work. Statistical intervals and q-values do not create fresh validation after repeated exploration.", "",
            "## Sources and reproduction", "",
            *[f"- [{name.replace('_',' ')}]({url})" for name, url in config["sources"].items()], "",
            "Existing FRED S&P 500 and SGD/USD controls retain their worksite-follow-up source manifest. Historical bank weights retain the December 2020 STI-fund source used by the overall-index experiment.", "",
            "Run `.venv/Scripts/python.exe analysis/compare_financial_sector_rents.py`. Settings are in analysis/configs/financial_sector_rents.json. Outputs include financial indices/returns, bank weight audits, national raw/adjusted means, correlations, controlled regressions, all forward predictions/scores and charts. results.json records source/code hashes, versions and strict forecast cutoff checks.", ""]
    (out / "report.md").write_text("\n".join(text), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPORTS / "financial_sector_rents")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG.read_text())
    controls_path = REPORTS / "company_worksite_followup/market_monthly_log_returns.csv"
    weights_path = ROOT / "analysis/configs/sti_etf_weights_2020_12.csv"
    sources = [DATA / "train.csv", AUX / "sg-stock-prices.csv", AUX / "sg-hdb-block.csv", CONFIG, controls_path, weights_path]
    before = {str(p.relative_to(ROOT)): sha256(p) for p in sources}
    controls = pd.read_csv(controls_path, index_col="month")
    controls.index = pd.PeriodIndex(controls.index, freq="M")
    raw = pd.read_csv(DATA / "train.csv")
    with threadpool_limits(limits=1):
        signals, weights, diagnostics = build_financial_signals(pd.read_csv(AUX / "sg-stock-prices.csv"), pd.read_csv(weights_path), config, raw.RENT_APPROVAL_DATE.max())
        regional = regional_rent_panel(raw, pd.read_csv(AUX / "sg-hdb-block.csv"))
        rent = regional[regional.region == "Singapore"].set_index(pd.PeriodIndex(regional[regional.region == "Singapore"].month, freq="M"))
        print("Constructed bank/finance/nonfinancial baskets and national mean rents", flush=True)
        correlations, level_corr = correlation_tables(rent, signals, config)
        print("Completed financial-stock growth correlations", flush=True)
        associations = association_table(rent, signals, controls, config)
        predictions, scores, comparisons = forecast_experiment(rent, signals["ADJUSTED_CLOSE"], controls)
        print("Completed controlled associations and expanding forecasts", flush=True)
        for price, levels in signals.items():
            levels.to_csv(out / f"index_levels_{price.lower()}.csv", index_label="month")
            (100 * np.log(levels / levels.shift(1))).to_csv(out / f"index_returns_{price.lower()}.csv", index_label="month")
        weights.to_csv(out / "bank_weight_audit.csv", index=False)
        rent.to_csv(out / "national_mean_rents.csv", index_label="period")
        correlations.to_csv(out / "growth_correlations.csv", index=False)
        level_corr.to_csv(out / "level_correlations.csv", index=False)
        associations.to_csv(out / "conditional_associations.csv", index=False)
        predictions.to_csv(out / "forward_predictions.csv", index=False)
        scores.to_csv(out / "forward_scores.csv", index=False)
        comparisons.to_csv(out / "forecast_comparisons.csv", index=False)
        plots(rent, signals, correlations, comparisons, out)
    if before != {str(p.relative_to(ROOT)): sha256(p) for p in sources}:
        raise AssertionError("An input changed")
    if not ((predictions.last_training_target < predictions.month) & (predictions.latest_allowed_market_month < predictions.month)).all():
        raise AssertionError("Future forecast input")
    metadata = {"config": config, "diagnostics": diagnostics, "source_sha256": before,
                "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__), ROOT / "analysis/compare_financial_sector_rents.py",
                                ROOT / "analysis/experiments/stock_index_regions.py", ROOT / "analysis/experiments/company_worksite_stocks.py",
                                ROOT / "analysis/experiments/stock_sector_signals.py"]},
                "versions": {p: version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn")},
                "rental_rows": len(raw), "rental_months": raw.RENT_APPROVAL_DATE.nunique(),
                "source_hashes_unchanged": True, "forecast_cutoffs_verified": True}
    (out / "results.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(config, diagnostics, correlations, associations, comparisons, level_corr, out)
    print(f"Saved {out / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
