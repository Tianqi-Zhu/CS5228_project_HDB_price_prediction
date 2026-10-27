"""Direct monthly flat-type rent forecasts from pre-origin rent and COE history.

This is a small monthly panel model, not a transaction-level TabPFN fit. Each
training example maps a historical origin, a flat type, and a forecast horizon
to a later monthly rent mean. Future economic observations are never inputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis.config import AUX, DATA, REPORTS, ROOT
from analysis.data import prepare_main, sha256
from analysis.evaluation import metrics


OUT = REPORTS / "flat_type_coe_forecast"
MIN_HISTORY = 12
MAX_HORIZON = 17
ALPHAS = (100.0, 1000.0, 10_000.0, 100_000.0, 1_000_000.0)
FOLDS = (("2024-06", "2025-04"), ("2023-11", "2025-04"))
MODELS = ("last_month", "recent3", "damped_trend", "ridge_history", "ridge_history_coe")


def rental_panel(raw: pd.DataFrame) -> pd.DataFrame:
    frame = prepare_main(raw)
    frame["month"] = pd.PeriodIndex(frame.RENT_APPROVAL_DATE, freq="M")
    return frame.groupby(["month", "FLAT_TYPE_CLEAN"], observed=True).MONTHLY_RENT.agg(
        rent_sum="sum", count="size", mean="mean",
    ).reset_index().rename(columns={"FLAT_TYPE_CLEAN": "flat_type"})


def coe_monthly(raw: pd.DataFrame) -> pd.Series:
    rows = raw.loc[raw.CATEGORY.astype(str).str.strip().str.lower().eq("a")].copy()
    date = pd.to_datetime(rows.YEAR.astype(str) + "-" + rows.MONTH.astype(str).str[:3] + "-01",
                          format="%Y-%b-%d", errors="raise")
    premium = pd.to_numeric(rows.QUOTA_PREMIUM.astype(str).str.replace(r"[$,]", "", regex=True), errors="raise")
    if (premium <= 0).any():
        raise ValueError("COE category A premiums must be positive")
    return pd.Series(premium.to_numpy(float), index=date.dt.to_period("M")).groupby(level=0).median().sort_index().rename("coe_a_median")


def history_features(panel: pd.DataFrame, coe: pd.Series, kind: str, origin: pd.Period,
                     horizon: int, *, economic: bool) -> tuple[dict[str, float], float]:
    if not 1 <= horizon <= MAX_HORIZON:
        raise ValueError("Unsupported forecast horizon")
    past = panel.loc[(panel.month <= origin) & panel.flat_type.eq(kind)].sort_values("month")
    if past.empty:
        raise ValueError(f"No rental history for {kind} before {origin}")

    def average(start, end):
        rows = past.loc[(past.month >= start) & (past.month <= end)]
        return float(rows.rent_sum.sum() / rows["count"].sum()) if len(rows) else np.nan

    last = float(past.iloc[-1]["mean"])
    short = average(origin - 2, origin)
    long = average(origin - 11, origin)
    previous = average(origin - 5, origin - 3)
    short = short if np.isfinite(short) else last
    long = long if np.isfinite(long) else last
    previous = previous if np.isfinite(previous) else short
    slope = float(np.clip((short - previous) / 3, -100, 100))
    damped_h = float((1 - 0.8 ** horizon) / 0.2)
    target = origin + horizon
    features = {
        "horizon": float(horizon), "damped_horizon": damped_h,
        "target_sin": float(np.sin(2 * np.pi * target.month / 12)),
        "target_cos": float(np.cos(2 * np.pi * target.month / 12)),
        "last_minus_short": last - short, "short_minus_long": short - long,
        "recent_slope": slope, "slope_x_horizon": slope * damped_h,
        "short_long_x_horizon": (short - long) * damped_h,
        "last_observation_age": float(origin.ordinal - past.iloc[-1].month.ordinal),
    }
    if economic:
        observed = coe.loc[coe.index <= origin].reindex(pd.period_range(origin - 11, origin, freq="M"))
        if observed.isna().any():
            raise ValueError(f"COE history is incomplete before {origin}")
        logged = np.log(observed.to_numpy(float))
        features.update({
            "coe_log_long_level": float(logged.mean()),
            "coe_short_minus_long": float(logged[-3:].mean() - logged.mean()),
            "coe_change_3m": float(logged[-1] - logged[-4]),
            "coe_trend_x_horizon": float((logged[-3:].mean() - logged.mean()) * damped_h),
        })
    return features, short


def design_row(panel, coe, kind, origin, horizon, *, economic, kinds):
    values, anchor = history_features(panel, coe, kind, origin, horizon, economic=economic)
    # Shared coefficients plus regularized deviations by flat type.
    interactions = ("damped_horizon", "short_minus_long", "slope_x_horizon")
    if economic:
        interactions += ("coe_short_minus_long", "coe_change_3m", "coe_trend_x_horizon")
    for category in kinds:
        indicator = float(category == kind)
        values[f"type_{category}"] = indicator
        for feature in interactions:
            values[f"type_{category}_x_{feature}"] = indicator * values[feature]
    return values, anchor


def training_examples(panel, coe, cutoff, *, economic):
    fitting = panel.loc[panel.month <= cutoff]
    kinds = sorted(fitting.flat_type.unique())
    first = fitting.month.min()
    lookup = fitting.set_index(["month", "flat_type"])
    x, y, weight, audits = [], [], [], []
    for origin in pd.period_range(first + MIN_HISTORY - 1, cutoff - 1, freq="M"):
        for horizon in range(1, min(MAX_HORIZON, cutoff.ordinal - origin.ordinal) + 1):
            target = origin + horizon
            for kind in kinds:
                key = (target, kind)
                if key not in lookup.index:
                    continue
                features, anchor = design_row(fitting, coe, kind, origin, horizon, economic=economic, kinds=kinds)
                outcome = lookup.loc[key]
                x.append(features)
                y.append(float(outcome["mean"]) - anchor)
                # Weight the monthly mean loss by the number of rental transactions.
                weight.append(float(outcome["count"]))
                audits.append((origin, target, kind))
    if not x:
        raise ValueError("Not enough monthly history for direct forecast training")
    weights = np.asarray(weight, float)
    weights /= weights.mean()
    return pd.DataFrame(x), np.asarray(y), weights, kinds, audits


def forecast(panel, coe, cutoff, horizons, *, economic, alpha, examples=None):
    x, y, weights, kinds, audits = (training_examples(panel, coe, cutoff, economic=economic)
                                   if examples is None else examples)
    # Shrink the entire adjustment toward zero, including the type intercepts.
    # Leaving an unpenalized intercept would retain historical rent growth even
    # when regularization removes every predictive feature.
    model = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=alpha, fit_intercept=False))
    model.fit(x, y, ridge__sample_weight=weights)
    records, future_x, anchors = [], [], []
    for horizon in horizons:
        for kind in kinds:
            features, anchor = design_row(panel, coe, kind, cutoff, horizon, economic=economic, kinds=kinds)
            future_x.append(features)
            anchors.append(anchor)
            records.append({"month": cutoff + horizon, "flat_type": kind, "horizon": horizon})
    values = np.maximum(1.0, np.asarray(anchors) + model.predict(pd.DataFrame(future_x, columns=x.columns)))
    result = pd.DataFrame(records)
    result["predicted"] = values
    return result, {"alpha": alpha, "training_examples": len(x), "training_months": cutoff.ordinal - panel.month.min().ordinal + 1,
                    "last_target_month": str(max(target for _, target, _ in audits)),
                    "last_feature_origin": str(max(origin for origin, _, _ in audits))}


def weighted_mean_rmse(observed, predicted):
    return float(np.sqrt(np.average((observed["mean"].to_numpy() - predicted) ** 2, weights=observed["count"])))


def select_alpha(panel, coe, cutoff, *, economic):
    # Two six-month blocks ending before the outer forecast origin.
    origins = [cutoff - 12, cutoff - 6]
    origins = [origin for origin in origins if origin.ordinal - panel.month.min().ordinal + 1 >= 18]
    if not origins:
        return 1000.0, []
    losses = {alpha: [[], []] for alpha in ALPHAS}
    for origin in origins:
        examples = training_examples(panel, coe, origin, economic=economic)
        for alpha in ALPHAS:
            predicted, _ = forecast(panel, coe, origin, range(1, 7), economic=economic, alpha=alpha, examples=examples)
            observed = predicted.merge(panel, on=["month", "flat_type"], how="inner", validate="one_to_one")
            losses[alpha][0].extend((observed.predicted - observed["mean"]) ** 2)
            losses[alpha][1].extend(observed["count"])
    scores = [{"alpha": alpha, "inner_weighted_mean_rmse": float(np.sqrt(np.average(errors, weights=weights)))}
              for alpha, (errors, weights) in losses.items()]
    best = min(scores, key=lambda row: row["inner_weighted_mean_rmse"])["alpha"]
    return best, scores


def evaluate_fold(panel, coe, start, end):
    origin, stop = pd.Period(start, "M"), pd.Period(end, "M")
    cutoff = origin - 1
    horizons = range(1, stop.ordinal - origin.ordinal + 1)
    observed = panel.loc[(panel.month >= origin) & (panel.month < stop)].copy()
    records = observed[["month", "flat_type", "mean", "count"]].rename(columns={"mean": "actual_mean"})
    diagnostics = {}
    for name in MODELS:
        if name.startswith("ridge"):
            economic = name.endswith("_coe")
            alpha, tuning = select_alpha(panel.loc[panel.month <= cutoff], coe, cutoff, economic=economic)
            prediction, diagnostic = forecast(panel.loc[panel.month <= cutoff], coe, cutoff, horizons,
                                               economic=economic, alpha=alpha)
            diagnostics[name] = {**diagnostic, "tuning": tuning}
        else:
            values = []
            for row in records.itertuples(index=False):
                h = row.month.ordinal - cutoff.ordinal
                features, anchor = history_features(panel, coe, row.flat_type, cutoff, h, economic=False)
                value = anchor
                if name == "last_month":
                    value += features["last_minus_short"]
                elif name == "damped_trend":
                    value += features["slope_x_horizon"]
                values.append(value)
            prediction = records[["month", "flat_type"]].copy()
            prediction["predicted"] = values
        records = records.merge(prediction[["month", "flat_type", "predicted"]].rename(columns={"predicted": name}),
                                on=["month", "flat_type"], validate="one_to_one")
    score_rows = []
    for name in MODELS:
        error = records[name] - records.actual_mean
        score_rows.append({"validation_start": start, "model": name, "group_months": len(records),
                           "weighted_mean_rmse_sgd": float(np.sqrt(np.average(error ** 2, weights=records["count"]))),
                           "mean_rmse_sgd": float(np.sqrt(np.mean(error ** 2))),
                           "weighted_bias_sgd": float(np.average(error, weights=records["count"]))})
    records.insert(0, "validation_start", start)
    return records, score_rows, diagnostics


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(DATA / "train.csv")
    panel = rental_panel(raw)
    coe = coe_monthly(pd.read_csv(AUX / "sg-coe-prices.csv"))
    panel.to_csv(OUT / "monthly_flat_type_means.csv", index=False)
    coe.to_csv(OUT / "monthly_coe_a.csv")
    predictions, scores, diagnostics = [], [], {}
    for start, end in FOLDS:
        print(f"Forecasting {start} to {pd.Period(end, 'M') - 1}", flush=True)
        rows, score, diagnostic = evaluate_fold(panel, coe, start, end)
        predictions.append(rows)
        scores.extend(score)
        diagnostics[start] = diagnostic
    pd.concat(predictions, ignore_index=True).to_csv(OUT / "forecast_means.csv", index=False, float_format="%.8f")
    score_frame = pd.DataFrame(scores)
    score_frame.to_csv(OUT / "scores.csv", index=False, float_format="%.8f")
    sources = (DATA / "train.csv", AUX / "sg-coe-prices.csv")
    settings = {"economic_variable": "COE category A monthly median quota premium", "minimum_history_months": MIN_HISTORY,
                "maximum_forecast_horizon": MAX_HORIZON, "alpha_candidates": ALPHAS,
                "target": "observed arithmetic monthly mean rent by cleaned flat type; not composition adjusted",
                "validation": "fixed origin; all economic features end at that origin; two earlier six-month tuning blocks",
                "source_sha256": {path.relative_to(ROOT).as_posix(): sha256(path) for path in sources},
                "runner_sha256": sha256(Path(__file__)), "folds": FOLDS, "diagnostics": diagnostics}
    (OUT / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    from analysis.compare_tabpfn_v3_80_20 import markdown_table
    report = ["# Monthly flat-type rent forecasts with COE category A", "",
              "A pooled Ridge model maps the rent and COE histories available at an origin to a future monthly "
              "mean rent for each flat type. It predicts a change from the origin's recent three-month mean. "
              "Short history uses three months; long history uses twelve. Horizon and calendar seasonality are explicit. "
              "Regularized type interactions allow different responses while sharing statistical strength. "
              "The intercept and all adjustment coefficients are penalized, so strong regularization approaches "
              "the recent-mean baseline instead of retaining unconditional historical growth.", "",
              "COE features are the twelve-month mean log premium, three-month minus twelve-month mean log premium, "
              "three-month log change, and a damped horizon interaction. No future COE observations or "
              "validation rent labels are inputs. Category-A auction premiums are aggregated by monthly median.", "",
              markdown_table(score_frame), "",
              "Weighted mean RMSE weights monthly flat-type errors by transaction counts. These scores measure "
              "group-mean forecasting, not individual transaction accuracy, and cannot be compared directly with "
              "TabPFN's transaction RMSE. The two outer holdouts overlap, so they are reported separately.", "",
              "The target is the raw monthly arithmetic mean; changing property composition can move it. "
              "Rare flat types receive little weight and have uncertain estimates. There are only 41 training "
              "months in the 80/20 split and 34 in the 17-month split. Economic associations do not establish causality. "
              "This prototype has not yet been added to TabPFN context features.", "",
              "These are development backtests. Initial results with an unpenalized intercept are retained in "
              "`initial_unpenalized_intercept/`; they motivated penalizing the entire adjustment toward a "
              "recent-mean baseline. Adding COE does not improve the revised model in either holdout, "
              "and neither learned specification beats a recent-mean baseline on the 17-month holdout. "
              "The COE forecast feature is not promoted to the TabPFN experiment.", "",
              "Reproduce: `.venv/Scripts/python.exe analysis/compare_flat_type_economic_forecast.py`."]
    (OUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(score_frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
