"""Company returns versus property-adjusted HDB rent changes near worksites.

The response is the monthly change in the near-minus-comparison-ring log-rent
premium. National changes and fixed property differences are thus removed.
Association estimates are exploratory; chronological predictions test utility.
"""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from analysis.config import AUX, DATA, REPORTS, ROOT
from analysis.data import prepare_main, sha256
from analysis.experiments.local_spatial_features import address_keys, metric_xy
from analysis.experiments.stock_sector_signals import block_difference, ridge_forecast


OUTER_START = "2024-06"
REFERENCE_END = "2021-12"
MIN_CELL_RENTALS = 20
RADII_M = (1000, 2000, 3000)
LAGS = (1, 3, 6)
CONTROL_RING_M = (3000, 6000)
MIN_DEV_MONTHS = 24
MIN_OUTER_MONTHS = 8
ALPHA = 10.0


def rental_coordinates(raw: pd.DataFrame, blocks: pd.DataFrame) -> pd.DataFrame:
    """Unique address join, preserving every rental's source position."""
    frame = prepare_main(raw)
    frame["address_key"] = address_keys(frame)
    b = blocks.copy()
    b["address_key"] = address_keys(b)
    if b.address_key.duplicated().any():
        raise ValueError("Ambiguous HDB address coordinates")
    frame = frame.merge(b[["address_key", "LATITUDE", "LONGITUDE"]], on="address_key", how="left", validate="many_to_one", sort=False)
    if len(frame) != len(raw) or frame.LATITUDE.isna().any():
        raise ValueError("Coordinate join changed/lost rental rows")
    if not np.array_equal(frame.MONTHLY_RENT, raw.MONTHLY_RENT):
        raise ValueError("Coordinate join reordered targets")
    frame["_month"] = pd.PeriodIndex(frame.RENT_APPROVAL_DATE, freq="M")
    return frame


def property_residuals(frame: pd.DataFrame) -> np.ndarray:
    """2021-only log-rent adjustment includes fixed HDB building effects."""
    fitting = frame.RENT_APPROVAL_DATE <= REFERENCE_END
    categorical = ["address_key", "FLAT_TYPE_CLEAN", "FLAT_MODEL_CLEAN"]
    numeric = ["FLOOR_AREA_SQM", "LEASE_COMMENCE_DATE"]
    model = Pipeline([
        ("preprocess", ColumnTransformer([
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")),
                                   ("scale", StandardScaler())]), numeric),
        ])),
        ("ridge", Ridge(alpha=10.0)),
    ])
    if (frame.MONTHLY_RENT <= 0).any():
        raise ValueError("Log-rent adjustment requires positive rents")
    y = np.log(frame.MONTHLY_RENT.to_numpy(float))
    model.fit(frame.loc[fitting, categorical + numeric], y[fitting])
    return y - model.predict(frame[categorical + numeric])


def local_panel(frame: pd.DataFrame, sites: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Near minus disjoint 3-6 km ring; full calendar grid and count guards."""
    f = frame.copy()
    f["residual"] = property_residuals(f)
    months = pd.period_range(f._month.min(), f._month.max(), freq="M")
    national = f.groupby("_month").residual.mean().reindex(months) * 100
    rows, coverage, footprints = [], [], {}
    xy = metric_xy(f.LONGITUDE, f.LATITUDE)
    for site in sites.itertuples(index=False):
        center = metric_xy([site.longitude], [site.latitude])[0]
        distance = np.linalg.norm(xy - center, axis=1)
        comparison = f[(distance > CONTROL_RING_M[0]) & (distance <= CONTROL_RING_M[1])]
        controls = comparison.groupby("_month").residual.agg(["mean", "size"]).reindex(months)
        for radius in RADII_M:
            near = f[distance <= radius]
            footprints[(site.site_id, radius)] = set(near.address_key)
            n = near.groupby("_month").residual.agg(["mean", "size"]).reindex(months)
            n["size"], controls["size"] = n["size"].fillna(0), controls["size"].fillna(0)
            usable = (n["size"] >= MIN_CELL_RENTALS) & (controls["size"] >= MIN_CELL_RENTALS)
            premium = (100 * (n["mean"] - controls["mean"])).where(usable)
            part = pd.DataFrame({"month": months.astype(str), "site_id": site.site_id, "company": site.company,
                                 "symbol": site.symbol, "radius_m": radius,
                                 "near_n": n["size"].to_numpy(int), "comparison_n": controls["size"].to_numpy(int),
                                 "near_log_residual": n["mean"].to_numpy(),
                                 "comparison_log_residual": controls["mean"].to_numpy(),
                                 "premium_logpoints": premium.to_numpy(),
                                 "premium_change": premium.diff().to_numpy(),
                                 "national_change": national.diff().to_numpy()})
            dev = part[part.month.between("2022-01", "2024-05")]
            outer = part[part.month >= OUTER_START]
            coverage.append({"site_id": site.site_id, "company": site.company, "symbol": site.symbol,
                             "radius_m": radius, "rental_buildings": near.address_key.nunique(),
                             "nearest_rental_km": float(distance.min() / 1000),
                             "development_near_median_n": float(dev.near_n.median()),
                             "development_valid_months": int(dev.premium_change.notna().sum()),
                             "outer_valid_months": int(outer.premium_change.notna().sum()),
                             # Eligibility uses earlier coverage only, not outer rents/counts.
                             "eligible": bool(dev.premium_change.notna().sum() >= MIN_DEV_MONTHS)})
            rows.append(part)
    overlap = []
    for i, left in enumerate(sites.site_id):
        for right in sites.site_id.iloc[i + 1:]:
            a, b = footprints[(left, 2000)], footprints[(right, 2000)]
            overlap.append({"site_a": left, "site_b": right, "intersection_buildings": len(a & b),
                            "jaccard_2km": len(a & b) / len(a | b) if a | b else 0.0})
    return pd.concat(rows, ignore_index=True), pd.DataFrame(coverage), pd.DataFrame(overlap)


def stock_returns(stocks: pd.DataFrame, symbols: list[str], end: str) -> pd.DataFrame:
    s = stocks.copy()
    s["DATE"] = pd.to_datetime(s.DATE)
    s = s[s.DATE.dt.to_period("M") <= pd.Period(end, "M")].copy()
    s["month"] = s.DATE.dt.to_period("M")
    if s.duplicated(["SYMBOL", "DATE"]).any() or (s.ADJUSTED_CLOSE <= 0).any():
        raise ValueError("Invalid stock observations")
    prices = s.sort_values("DATE").groupby(["month", "SYMBOL"]).ADJUSTED_CLOSE.last().unstack()
    prices = prices.reindex(pd.period_range(prices.index.min(), pd.Period(end, "M"), freq="M"))
    returns = 100 * np.log(prices / prices.shift(1))
    if set(symbols) - set(returns):
        raise ValueError("Worksite company absent from stock file")
    return returns


def regression_design(local: pd.DataFrame, returns: pd.DataFrame, focal: str,
                      peer_symbols: list[str], lag: int) -> pd.DataFrame:
    """Target month t predictors stop at t-1 (company t-lag).

    All broad/peer baskets exclude the focal company. Percent log-returns
    avoid mixing absolute USD and SGD price levels for KLIC and FLEX.
    """
    d = local.copy().set_index(pd.PeriodIndex(local.month, freq="M"))
    result = pd.DataFrame(index=d.index)
    result["response"] = d.premium_change
    result["local_change_lag1"] = d.premium_change.shift(1)
    result["local_change_mean3"] = d.premium_change.shift(1).rolling(3).mean()
    result["national_change_lag1"] = d.national_change.shift(1)
    broad = [c for c in returns if c.endswith(".SI") and c != focal]
    peers = [c for c in peer_symbols if c != focal]
    result["broad_return_lag1"] = returns[broad].mean(axis=1).shift(1).reindex(result.index)
    result["peer_return_lag1"] = returns[peers].mean(axis=1).shift(1).reindex(result.index)
    result["company_return"] = returns[focal].shift(lag).reindex(result.index)
    result["phase"] = np.where(result.index.astype(str) < OUTER_START, "development", "outer")
    return result


def hac_association(design: pd.DataFrame) -> dict:
    """OLS partial slope with a small-sample-adjusted Newey-West covariance.

    Three monthly autocovariance lags; t reference is an approximation with
    so few months. The regression does not establish causal identification.
    """
    columns = [c for c in design if c not in ("response", "phase")]
    valid = design[["response", *columns]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 12:
        return {"months": len(valid), "coefficient": np.nan, "effect_per_10pct_stock": np.nan,
                "effect_ci_low": np.nan, "effect_ci_high": np.nan, "p_value": np.nan}
    x = np.column_stack([np.ones(len(valid)), valid[columns].to_numpy(float)])
    y = valid.response.to_numpy(float)
    bread = np.linalg.pinv(x.T @ x)
    beta = bread @ x.T @ y
    residual = y - x @ beta
    scores = x * residual[:, None]
    meat = scores.T @ scores
    for lag in range(1, 4):
        cross = scores[lag:].T @ scores[:-lag]
        meat += (1 - lag / 4) * (cross + cross.T)
    df = len(y) - np.linalg.matrix_rank(x)
    covariance = bread @ meat @ bread * len(y) / df
    se = np.sqrt(max(float(covariance[-1, -1]), 0))
    p = float(2 * student_t.sf(abs(beta[-1] / se), df)) if se > 0 else np.nan
    ci = float(student_t.ppf(.975, df) * se)
    stock_move = 100 * np.log1p(.1)  # Exactly a +10% price change in log units.
    return {"months": len(y), "coefficient": float(beta[-1]),
            "effect_per_10pct_stock": float(stock_move * beta[-1]),
            "effect_ci_low": float(stock_move * (beta[-1] - ci)),
            "effect_ci_high": float(stock_move * (beta[-1] + ci)), "p_value": p}


def dependent_fdr(p: pd.Series) -> pd.Series:
    """Benjamini-Yekutieli correction allows dependence across site tests."""
    out = pd.Series(np.nan, index=p.index)
    valid = p.dropna().sort_values()
    m = len(valid)
    if not m:
        return out
    adjusted = valid.to_numpy() * m * np.sum(1 / np.arange(1, m + 1)) / np.arange(1, m + 1)
    out.loc[valid.index] = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0, 1)
    return out


def forward_predictions(design: pd.DataFrame) -> pd.DataFrame:
    """Expanding monthly predictions; no future target or normalization."""
    columns = [c for c in design if c not in ("response", "phase")]
    d = design.replace([np.inf, -np.inf], np.nan).dropna(subset=["response", *columns])
    d = d[d.index >= pd.Period("2022-01", "M")]
    x = d[columns].to_numpy(float)
    y = d.response.to_numpy(float)
    rows = []
    for i in range(8, len(d)):
        base, _ = ridge_forecast(x[:i, :-1], y[:i], x[i, :-1], ALPHA)
        own, _ = ridge_forecast(x[:i], y[:i], x[i], ALPHA)
        rows.append({"month": str(d.index[i]), "phase": d.phase.iloc[i], "actual": y[i],
                     "persistence": 0.0, "rent_market_control": base, "company_stock": own,
                     "n_train_months": i, "last_training_month": str(d.index[i - 1]),
                     "last_company_stock_month": str(d.index[i] - 1)})
    return pd.DataFrame(rows)


def prediction_summary(prediction: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for phase, part in prediction.groupby("phase"):
        for model in ("persistence", "rent_market_control", "company_stock"):
            e = part[model] - part.actual
            rows.append({"phase": phase, "model": model, "months": len(part),
                         "rmse_logpoints": float(np.sqrt(np.mean(e ** 2))),
                         "mae_logpoints": float(np.mean(np.abs(e))), "bias_logpoints": float(e.mean())})
    return pd.DataFrame(rows)


def compare_outer(prediction: pd.DataFrame, baseline: str) -> dict:
    d = prediction[prediction.phase == "outer"]
    if len(d) < MIN_OUTER_MONTHS:
        return {"outer_months": len(d), "outer_confirmable": False}
    a, b, p = d.actual.to_numpy(), d[baseline].to_numpy(), d.company_stock.to_numpy()
    rmse = lambda x: float(np.sqrt(np.mean((x - a) ** 2)))
    ci = block_difference(a, b, p, repeats=2000, block=3)
    return {"outer_months": len(d), "outer_confirmable": True,
            "baseline_model": baseline, "baseline_rmse": rmse(b), "stock_rmse": rmse(p),
            "delta_rmse": rmse(p) - rmse(b),
            "delta_vs_matched_control": rmse(p) - rmse(d.rent_market_control.to_numpy()),
            "improved_months": int(((p - a) ** 2 < (b - a) ** 2).sum()),
            "delta_ci_low": ci["delta_rmse_ci_low"], "delta_ci_high": ci["delta_rmse_ci_high"]}


def run_experiments(panel: pd.DataFrame, coverage: pd.DataFrame, returns: pd.DataFrame,
                    sites: pd.DataFrame, out: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    peers = sorted(sites.symbol.unique())
    associations, forecasts, scores, primary, designs = [], [], [], [], {}
    for (site_id, radius), local in panel.groupby(["site_id", "radius_m"], sort=False):
        cov = coverage[(coverage.site_id == site_id) & (coverage.radius_m == radius)].iloc[0]
        if not cov.eligible:
            continue
        symbol = local.symbol.iloc[0]
        company = local.company.iloc[0]
        for lag in LAGS:
            key = (site_id, radius, lag)
            d = regression_design(local, returns, symbol, peers, lag)
            designs[key] = d
            dev = d[(d.index >= pd.Period("2022-01", "M")) & (d.index < pd.Period(OUTER_START, "M"))]
            association = hac_association(dev)
            early = hac_association(dev.loc[:"2022-12"])
            late = hac_association(dev.loc["2023-01":])
            info = {"site_id": site_id, "company": company, "symbol": symbol, "radius_m": radius, "lag": lag}
            associations.append({**info, **association, "early_slope": early["coefficient"], "late_slope": late["coefficient"]})
            pred = forward_predictions(d)
            if pred.empty:
                continue
            # Company lag may exceed one; record the exact observation date.
            pred["last_company_stock_month"] = (pd.PeriodIndex(pred.month, freq="M") - lag).astype(str)
            pred = pred.assign(**info)
            forecasts.append(pred)
            score = prediction_summary(pred).assign(**info)
            scores.append(score)
            development = score[score.phase == "development"].set_index("model")
            baseline = development.loc[["persistence", "rent_market_control"]].rmse_logpoints.idxmin()
            if radius == 2000 and lag == 1:
                primary.append({**info, "development_months": int(development.loc[baseline, "months"]),
                                "development_baseline": baseline,
                                "development_baseline_rmse": float(development.loc[baseline, "rmse_logpoints"]),
                                "development_stock_rmse": float(development.loc["company_stock", "rmse_logpoints"]),
                                **compare_outer(pred, baseline)})
        print(f"Completed {site_id}, radius {radius}m", flush=True)
    assoc = pd.DataFrame(associations)
    assoc["q_all_scans"] = dependent_fdr(assoc.p_value)
    mask = (assoc.radius_m == 2000) & (assoc.lag == 1)
    assoc.loc[mask, "q_primary_tests"] = dependent_fdr(assoc.loc[mask, "p_value"])
    assoc.to_csv(out / "development_associations.csv", index=False)
    forecast = pd.concat(forecasts, ignore_index=True)
    scored = pd.concat(scores, ignore_index=True)
    forecast.to_csv(out / "monthly_predictions.csv", index=False)
    scored.to_csv(out / "forward_scores.csv", index=False)
    primary = pd.DataFrame(primary).merge(assoc.loc[mask, ["site_id", "effect_per_10pct_stock", "p_value", "q_primary_tests", "early_slope", "late_slope"]], on="site_id", validate="one_to_one")
    primary.to_csv(out / "primary_2km_lag1_results.csv", index=False)
    # Company-specific selection uses earlier forecasts only; baseline competes.
    dev = scored[scored.phase == "development"].pivot(index=["company", "symbol", "site_id", "radius_m", "lag"], columns="model", values="rmse_logpoints")
    dev["best_control"] = dev[["persistence", "rent_market_control"]].idxmin(axis=1)
    dev["best_control_rmse"] = dev[["persistence", "rent_market_control"]].min(axis=1)
    dev["stock_control_ratio"] = dev.company_stock / dev.best_control_rmse
    selected = []
    for company, part in dev.reset_index().groupby("company", sort=False):
        pick = part.sort_values(["stock_control_ratio", "radius_m", "lag", "site_id"]).iloc[0]
        pred = forecast[(forecast.site_id == pick.site_id) & (forecast.radius_m == pick.radius_m) & (forecast.lag == pick.lag)]
        selected.append({"company": company, "symbol": pick.symbol, "site_id": pick.site_id,
                         "radius_m": int(pick.radius_m), "lag": int(pick.lag),
                         "development_stock_control_ratio": float(pick.stock_control_ratio),
                         "development_supports_stock": bool(pick.stock_control_ratio < 1),
                         **compare_outer(pred, pick.best_control)})
    selected = pd.DataFrame(selected)
    selected.to_csv(out / "earlier_selected_company_results.csv", index=False)
    # Negative controls: the same focal stock at remote, unrelated company sites.
    # Require site centers >=6km apart and different parent companies.
    placebo = []
    coords = metric_xy(sites.longitude, sites.latitude)
    site_positions = {s: i for i, s in enumerate(sites.site_id)}
    focal_cases = pd.concat([primary.assign(specification="primary"),
                             selected.assign(specification="earlier_selected")], ignore_index=True)
    for real in focal_cases.itertuples(index=False):
        origin_index = site_positions[real.site_id]
        for receiver in sites.itertuples(index=False):
            if receiver.symbol == real.symbol:
                continue
            distance = np.linalg.norm(coords[origin_index] - coords[site_positions[receiver.site_id]])
            if distance < 6000 or (receiver.site_id, real.radius_m, real.lag) not in designs:
                continue
            local = panel[(panel.site_id == receiver.site_id) & (panel.radius_m == real.radius_m)]
            d = regression_design(local, returns, real.symbol, peers, int(real.lag))
            pred = forward_predictions(d)
            score = prediction_summary(pred)
            development = score[score.phase == "development"].set_index("model")
            baseline = development.loc[["persistence", "rent_market_control"]].rmse_logpoints.idxmin()
            comp = compare_outer(pred, baseline)
            placebo.append({"specification": real.specification, "focal_site": real.site_id,
                            "focal_symbol": real.symbol, "radius_m": int(real.radius_m), "lag": int(real.lag),
                            "receiving_site": receiver.site_id, "distance_between_sites_km": float(distance / 1000),
                            "real_site_delta_pct": 100 * real.delta_rmse / real.baseline_rmse,
                            "remote_delta_pct": 100 * comp["delta_rmse"] / comp["baseline_rmse"] if comp["outer_confirmable"] else np.nan,
                            **comp})
    placebo = pd.DataFrame(placebo)
    placebo.to_csv(out / "remote_site_negative_controls.csv", index=False)
    spatial_summary = []
    for (spec, focal), part in placebo.groupby(["specification", "focal_site"]):
        valid = part[part.outer_confirmable & part.remote_delta_pct.notna()]
        spatial_summary.append({"specification": spec, "site_id": focal, "remote_sites": len(valid),
                                "attempted_remote_sites": len(part),
                                "real_delta_pct": float(part.real_site_delta_pct.iloc[0]),
                                "remote_median_delta_pct": float(valid.remote_delta_pct.median()),
                                "remote_sites_improving_more": int((valid.remote_delta_pct < part.real_site_delta_pct.iloc[0]).sum())})
    pd.DataFrame(spatial_summary).to_csv(out / "spatial_specificity_summary.csv", index=False)
    return primary, selected, assoc, forecast


def plots(panel: pd.DataFrame, coverage: pd.DataFrame, primary: pd.DataFrame,
          predictions: pd.DataFrame, sites: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    c = coverage[coverage.radius_m == 2000].set_index("site_id").reindex(sites.site_id)
    ax.barh(np.arange(len(c)), c.development_near_median_n, color=["#277c80" if e else "#b8b8b8" for e in c.eligible])
    ax.axvline(MIN_CELL_RENTALS, color="#555", linestyle="--", label="Minimum per monthly cell")
    ax.set_yticks(np.arange(len(c)), c.index.str.replace("_", " "))
    ax.set(xlabel="Median monthly HDB rental observations within 2 km (2022-May 2024)", title="Local data coverage around engineering/manufacturing sites")
    ax.legend()
    fig.savefig(out / "01_worksite_coverage.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    p = primary.sort_values("site_id")
    y = np.arange(len(p))
    axes[0].barh(y, p.delta_rmse, color=["#277c80" if d < 0 else "#ba6046" for d in p.delta_rmse])
    axes[0].hlines(y, p.delta_ci_low, p.delta_ci_high, color="#333", linewidth=2)
    axes[0].axvline(0, color="#555", linewidth=1)
    axes[0].set_yticks(y, p.site_id.str.replace("_", " "))
    axes[0].set(title="Primary test: 2 km, preceding-month company return", xlabel="Change in local-premium growth RMSE (log points)")
    for site in ("venture_amk", "kulicke_soffa_serangoon", "st_engineering_boon_lay", "seatrium_admiralty"):
        part = panel[(panel.site_id == site) & (panel.radius_m == 2000)]
        axes[1].plot(pd.to_datetime(part.month), part.premium_logpoints, label=site.replace("_", " "))
    axes[1].axvline(pd.Timestamp(OUTER_START), color="#555", linestyle="--")
    axes[1].set(title="Rent near site relative to the 3-6 km comparison ring", ylabel="Property-adjusted log-rent premium x 100")
    axes[1].legend(fontsize=8)
    fig.savefig(out / "02_local_stock_results.png", dpi=160)
    plt.close(fig)
    chosen = pd.read_csv(out / "earlier_selected_company_results.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    positions = np.arange(len(chosen))
    pct = 100 * chosen.delta_rmse / chosen.baseline_rmse
    axes[0].barh(positions, pct, color=["#277c80" if v < 0 else "#ba6046" for v in pct])
    axes[0].hlines(positions, 100 * chosen.delta_ci_low / chosen.baseline_rmse,
                  100 * chosen.delta_ci_high / chosen.baseline_rmse, color="#333", linewidth=2)
    axes[0].set_yticks(positions, chosen.company)
    axes[0].axvline(0, color="#555", linewidth=1)
    axes[0].set(title="All five earlier-selected company specifications", xlabel="Validation RMSE change (%); conditional intervals")
    # KLIC is shown as an exploratory result, not a newly chosen deployment model.
    k = chosen[chosen.symbol == "KLIC"].iloc[0]
    p = predictions[(predictions.site_id == k.site_id) & (predictions.radius_m == k.radius_m)
                    & (predictions.lag == k.lag) & (predictions.phase == "outer")]
    for column, label in (("actual", "Actual local relative growth"),
                          ("rent_market_control", "Rent/market control"), ("company_stock", "Control + KLIC return")):
        axes[1].plot(pd.to_datetime(p.month), p[column], marker=".", label=label)
    axes[1].set(title=f"Exploratory K&S result: {int(k.radius_m)} m, lag {int(k.lag)}", ylabel="Monthly local-premium change (log points)")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].legend(fontsize=8)
    fig.savefig(out / "03_selected_company_candidates.png", dpi=160)
    plt.close(fig)


def md_table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame[columns].itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v) for v in row) + " |")
    return "\n".join(lines)


def report(primary: pd.DataFrame, selected: pd.DataFrame, associations: pd.DataFrame,
           coverage: pd.DataFrame, sites: pd.DataFrame, metadata: dict, out: Path) -> None:
    primary_signal = primary[(primary.q_primary_tests < .05) & (primary.delta_rmse < 0) & (primary.delta_ci_high < 0)]
    promoted = selected[(selected.development_supports_stock) & (selected.delta_rmse < 0) & (selected.delta_ci_high < 0)]
    outcome = ("No site passes both the primary association screen and later forecast-improvement check. Keep company-specific location/return features out of the current submission on this evidence." if primary_signal.empty else
               "Some sites pass the primary association and later forecast-improvement screen. Treat them as candidates for new data confirmation; this is not a causal result or a TabPFN ablation.")
    rows = ["# Company stock returns and HDB rents near engineering worksites", "", outcome, "",
            f"Scope: {len(sites)} documented sites, {sites.symbol.nunique()} parent-company stocks; {len(primary)} sites have enough earlier data for the primary 2 km test. ComfortDelGro, banks and dispersed consumer-service firms are excluded following the requested manufacturing/engineering focus.", "",
            "## What is tested", "",
            "For each site, compare rentals within 2 km with rentals in a disjoint 3-6 km ring. A model fitted on 2021 rentals only removes fixed building, flat type/model, area and lease-year effects from log rent. The response is the monthly CHANGE in this adjusted near-minus-ring premium, multiplied by 100. This tests local relative rent growth rather than national rent growth or expensive neighborhoods having high rent levels.", "",
            "The primary company signal is the preceding calendar month's adjusted-close log return. Rent/stock predictors always precede the target approval month. The control model uses recent local relative changes, prior national adjusted rent growth, prior broad Singapore stock returns and an engineering/manufacturing peer basket excluding the focal company. Comparisons include a zero-change local-premium forecast. All models use the same eligible monthly cases; the better rent/market control is chosen using earlier forecasts only.", "",
            "## Primary results: 2 km, lag 1", "",
            md_table(primary, ["site_id", "development_baseline", "baseline_rmse", "stock_rmse", "delta_rmse", "improved_months", "delta_ci_low", "delta_ci_high", "effect_per_10pct_stock", "q_primary_tests"]), "",
            "RMSE concerns monthly local relative growth, in log points (approximately percentage points), not individual-rental SGD RMSE. Negative deltas favor company stock. Intervals are paired three-month moving-block bootstrap over the ten June 2024-March 2025 validation target months, conditional on the fitted models. They omit model-fitting/selection uncertainty and are not adjusted for the sensitivity search or the five company-level outer comparisons. The development-only slope column is the estimated local relative rent-growth change for a +10% stock move; it is an association, not an intervention effect.", "",
            "Association regressions use January 2022-May 2024 data with OLS and Newey-West covariance (three monthly lags, small-sample factor, approximate t reference). q_primary_tests applies Benjamini-Yekutieli multiple-testing correction across eligible 2 km/lag-1 sites; q_all_scans corrects across all eligible site/radius/lag scans. Development_associations.csv also shows early/late slopes. Small samples make these estimates fragile.", "",
            "## Earlier-selected sensitivity candidates", "",
            "Radii 1/2/3 km and company-return lags 1/3/6 are sensitivity scans, with 3-6 km comparison rings. The site/radius/lag for each company is chosen on earlier expanding-window forecasts only, using stock RMSE divided by the better control's RMSE. A forced best stock candidate can still fail to beat the control; development_supports_stock records this. Baseline competes with stock for adoption. This selection explores multiple localities and should not be interpreted as a discovered causal radius.", "",
            md_table(selected, ["company", "site_id", "radius_m", "lag", "development_stock_control_ratio", "development_supports_stock", "baseline_rmse", "stock_rmse", "delta_rmse", "delta_ci_low", "delta_ci_high"]), "",
            f"{len(promoted)} earlier-supported sensitivity candidate(s) have a negative conditional outer RMSE interval. Report these alongside the primary specification; selecting a new winner after viewing outer errors would reuse the validation data.", "",
            f"None of the {len(associations)} exploratory association scans survives a 5% dependence-adjusted false-discovery threshold." if not (associations.q_all_scans < .05).any() else
            "Some exploratory associations survive the dependence-adjusted false-discovery screen; inspect q_all_scans together with chronological prediction errors.", "",
            "The K&S 1 km/six-month-lag specification is a useful follow-up hypothesis if it improves validation. Its 1 km ring is relatively noisy and the company is US-listed; neither a conditional bootstrap interval nor an uncorrected association p-value establishes a local employment effect.", "",
            md_table(pd.read_csv(out / "spatial_specificity_summary.csv").query("specification == 'earlier_selected'"),
                     ["site_id", "remote_sites", "real_delta_pct", "remote_median_delta_pct", "remote_sites_improving_more"]), "",
            "## Coverage and geography", "",
            md_table(coverage[coverage.radius_m == 2000], ["site_id", "rental_buildings", "nearest_rental_km", "development_near_median_n", "development_valid_months", "outer_valid_months", "eligible"]), "",
            "A near or comparison monthly cell needs at least 20 rentals; no target is filled across a missing month. Sites/radii require at least 24 valid local growth months in January 2022-May 2024, without using outer coverage to decide eligibility. Prediction fits begin after eight mature complete-case monthly labels. All valid cases are saved; sparse series are excluded explicitly. At least eight outer months are needed for confirmation.", "",
            "The Tuas yard is over 12 km from the nearest HDB rental block in the supplied data, so no within-3 km HDB effect can be measured. SIA Engineering's airport site has extremely sparse nearby HDB support. These exclusions are lack of data, not estimates of zero effect. Their workers may commute farther or use other housing types.", "",
            "Site_coverage.csv lists all radii, and neighborhood_overlap.csv measures overlapping 2 km rental-building sets. Venture's Ang Mo Kio/K&S neighborhoods and its Woodlands/Seatrium Admiralty neighborhoods overlap substantially; these are not independent firm-level experiments. Remote_site_negative_controls.csv applies each focal stock to unrelated-company sites at least 6 km away, with the same radius, lag and forecast procedure, for both the primary and earlier-selected specifications. Spatial_specificity_summary.csv compares percentage RMSE changes against these remote controls. These checks assess whether any apparent gain also occurs away from that company's operations; they are exploratory, not a randomized spatial design.", "",
            "## Documented sites and sources", ""]
    for site in sites.itertuples(index=False):
        rows.append(f"- **{site.site_id} ({site.symbol})**: {site.address}, {str(site.postal).zfill(6)}. {site.role}. [Historical evidence ({site.evidence_year})]({site.source_url}); [OneMap geocode]({site.geocode_source}).")
    rows.extend(["", "Historical source documents around 2020-2022 establish addresses; the fixed-site analysis assumes continued operation through March 2025. The locations are not employment counts or an exhaustive factory inventory. A group headquarters can contain fewer operational workers than a plant. Exact facility boundaries, staff residence/commuting patterns and yearly occupancy have not been reconstructed. Current OneMap address pins locate these documented addresses and may differ from historical entrances or property centroids.", "",
                 "## Interpretation and feature decision", "",
                 "This design tests the proposed local predictive relationship more directly than a nationwide stock basket. It cannot establish that a company's stock price CAUSES nearby rents to change. Earnings, hiring, bonuses, global manufacturing demand, housing supply, COVID/reopening and transport access may drive both; stock prices may anticipate those drivers. Headquarters proximity alone is an imperfect exposure measure, especially for globally diversified companies. Seatrium's 2023 merger changes the scope of its historical stock exposure.", "",
                 "FLEX and KLIC are US-listed stocks in the supplied file. We use percentage/log returns, not mixed-currency price levels; separate US-market, currency and sector factors are not available here, so their firm-specific interpretation is weaker. Stock adjusted histories are later snapshots, not point-in-time archives. The rental data have 51 distinct months with serial dependence, not 150,000 independent stock observations. Building-specific rent premiums may evolve after the fixed 2021 adjustment. Publication timing is assumed month-end; real rental-release delays would need additional lags. This outer period has been examined in other project experiments and is not an untouched final test.", "",
                 "If a candidate is pursued, add distance-weighted company-return interactions together with standalone distance, company-return and local-history controls, rather than assigning raw stock prices to nearby homes. Require a new paired TabPFN evaluation and new rental periods before promotion. No final submission or TabPFN model was changed by this experiment.", "",
                 "## Reproduction", "",
                 "1. `.venv/Scripts/python.exe analysis/acquire_company_worksites.py` (public OneMap address geocodes, cached).",
                 "2. `.venv/Scripts/python.exe analysis/compare_company_worksite_stocks.py` (read-only inputs, single-thread CPU analysis).", "",
                 "Outputs include the adjusted monthly panel, full forward predictions, coverage/exclusions, all association scans, primary tests, earlier-selected sensitivities, spatial negative controls and charts. results.json records source/code hashes, selections, parameters and unchanged-input checks.", ""])
    (out / "report.md").write_text("\n".join(rows), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPORTS / "company_worksite_stocks")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    locations = REPORTS / "company_worksite_stocks/worksite_locations.csv"
    source_paths = [DATA / "train.csv", AUX / "sg-hdb-block.csv", AUX / "sg-stock-prices.csv",
                    ROOT / "analysis/configs/company_worksites.json", locations,
                    REPORTS / "company_worksite_stocks/geocode_manifest.json"]
    before = {str(p.relative_to(ROOT)): sha256(p) for p in source_paths}
    sites = pd.read_csv(locations, dtype={"postal": str})
    with threadpool_limits(limits=1):
        frame = rental_coordinates(pd.read_csv(DATA / "train.csv"), pd.read_csv(AUX / "sg-hdb-block.csv"))
        panel, coverage, overlap = local_panel(frame, sites)
        returns = stock_returns(pd.read_csv(AUX / "sg-stock-prices.csv"), sorted(sites.symbol.unique()), str(frame._month.max()))
        panel.to_csv(out / "monthly_local_rent_panel.csv", index=False)
        coverage.to_csv(out / "site_coverage.csv", index=False)
        overlap.to_csv(out / "neighborhood_overlap.csv", index=False)
        primary, selected, assoc, forecast = run_experiments(panel, coverage, returns, sites, out)
        plots(panel, coverage, primary, forecast, sites, out)
    if before != {str(p.relative_to(ROOT)): sha256(p) for p in source_paths}:
        raise AssertionError("Source data changed")
    if not (forecast.last_training_month < forecast.month).all():
        raise AssertionError("A target entered its forecast's training set")
    if not (forecast.last_company_stock_month < forecast.month).all():
        raise AssertionError("Same/future-month company stock signal")
    metadata = {"sites": len(sites), "companies": sites.symbol.nunique(), "rental_rows": len(frame),
                "rental_months": frame._month.nunique(), "source_sha256": before,
                "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in
                                [Path(__file__), ROOT / "analysis/compare_company_worksite_stocks.py",
                                 ROOT / "analysis/acquire_company_worksites.py", ROOT / "analysis/experiments/stock_sector_signals.py",
                                 ROOT / "analysis/experiments/local_spatial_features.py", ROOT / "analysis/data.py"]},
                "settings": {"reference_end": REFERENCE_END, "outer_start": OUTER_START, "radii_m": RADII_M,
                             "lags": LAGS, "comparison_ring_m": CONTROL_RING_M, "ridge_alpha": ALPHA,
                             "min_cell_rentals": MIN_CELL_RENTALS, "min_dev_months": MIN_DEV_MONTHS,
                             "min_outer_months": MIN_OUTER_MONTHS, "thread_limit": 1},
                "primary_tests": len(primary), "scanned_tests": len(assoc),
                "versions": {p: version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn", "pyproj")},
                "earlier_selected": selected.replace({np.nan: None}).to_dict(orient="records"),
                "input_hashes_unchanged": True, "forecast_cutoff_checks_passed": True}
    (out / "results.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report(primary, selected, assoc, coverage, sites, metadata, out)
    print(primary[["site_id", "baseline_rmse", "stock_rmse", "delta_rmse", "q_primary_tests"]].to_string(index=False), flush=True)
    print(f"Report saved to {out / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
