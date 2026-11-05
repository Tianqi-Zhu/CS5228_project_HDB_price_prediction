"""Frozen company/location hypotheses, matched placebos and rent-history ablation."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

from analysis.config import AUX, DATA, REPORTS, ROOT, CATBOOST_SETTINGS
from analysis.data import sha256
from analysis.experiments.company_worksite_stocks import (
    rental_coordinates, property_residuals, stock_returns, regression_design,
    forward_predictions, prediction_summary, compare_outer, hac_association,
)
from analysis.experiments.local_spatial_features import metric_xy


CONFIG = ROOT / "analysis/configs/company_worksite_followup.json"
CONTROL_COLUMNS = ["SP500", "NASDAQSOX", "DEXSIUS"]
BASE_VARIANTS = ["history77", "klic_nuisance", "klic_sector_local", "klic_local",
                 "all_firm_nuisance", "all_firm_sector_local", "all_firm_local"]


def read_controls():
    out = REPORTS / "company_worksite_followup"
    manifest = json.loads((out / "market_source_manifest.json").read_text())
    path = out / "market_monthly_log_returns.csv"
    if sha256(path) != manifest["monthly_returns_sha256"]:
        raise ValueError("External market controls changed")
    return pd.read_csv(path, index_col="month").set_axis(pd.PeriodIndex(pd.read_csv(path).month, freq="M"))


def match_placebos(frame, sites, config):
    """Label-independent for later rents: reference-year covariates only."""
    reference = frame[frame.RENT_APPROVAL_DATE <= "2021-12"].copy()
    buildings = frame.drop_duplicates("address_key").set_index("address_key")
    xy = metric_xy(buildings.LONGITUDE, buildings.LATITUDE)
    tree = cKDTree(xy)
    focal = sites.set_index("site_id").loc[config["primary_hypothesis"]["site_id"]]
    center = metric_xy([focal.longitude], [focal.latitude])[0]
    radius = config["primary_hypothesis"]["radius_m"]
    ref = reference.groupby("address_key").agg(n=("MONTHLY_RENT", "size"), rent=("MONTHLY_RENT", "sum"),
                area=("FLOOR_AREA_SQM", "sum"), lease=("LEASE_COMMENCE_DATE", "sum"))
    for kind in ("3-room", "4-room", "5-room"):
        ref[kind] = reference[reference.FLAT_TYPE_CLEAN == kind].groupby("address_key").size()
    ref = ref.reindex(buildings.index).fillna(0)
    counts = ref.n.to_numpy()
    numerators = ref[["rent", "area", "lease", "3-room", "4-room", "5-room"]].to_numpy(float)
    def covariates(neighbors):
        n = counts[neighbors].sum()
        return np.r_[np.log1p(n / 12), numerators[neighbors].sum(axis=0) / n] if n else np.full(7, np.nan)
    target_neighbors = tree.query_ball_point(center, radius)
    target = covariates(target_neighbors)
    candidates = np.flatnonzero((counts > 0) & (np.linalg.norm(xy - center, axis=1) >= config["placebo_min_distance_to_focal_m"]))
    values = np.array([covariates(tree.query_ball_point(xy[i], radius)) for i in candidates])
    target_density = np.expm1(target[0])
    density = np.expm1(values[:, 0])
    good = np.isfinite(values).all(axis=1) & (density >= .5 * target_density) & (density <= 2 * target_density)
    candidates, values = candidates[good], values[good]
    if len(candidates) < config["placebo_count"]:
        raise ValueError("Not enough reference-year density-matched placebo locations")
    # Population variation gives a common reference scale; no outcomes after 2021.
    scale = np.maximum(np.std(values, axis=0), 1e-6)
    score = np.mean(((values - target) / scale) ** 2, axis=1)
    selected = []
    for j in np.argsort(score, kind="stable"):
        i = candidates[j]
        if selected and np.min(np.linalg.norm(xy[[r["position"] for r in selected]] - xy[i], axis=1)) < config["placebo_min_separation_m"]:
            continue
        selected.append({"position": int(i), "match_score": float(score[j]), "values": values[j]})
        if len(selected) == config["placebo_count"]:
            break
    if len(selected) < config["placebo_count"]:
        raise ValueError("Not enough spatially spread placebo matches; do not silently change the design")
    rows = []
    for j, row in enumerate(selected, 1):
        b = buildings.iloc[row["position"]]
        rows.append({"site_id": f"matched_placebo_{j:02}", "company": "Matched reference-year HDB neighborhood",
                     "symbol": "KLIC", "latitude": b.LATITUDE, "longitude": b.LONGITUDE,
                     "address_key": buildings.index[row["position"]], "match_score": row["match_score"],
                     "reference_monthly_rentals": float(np.expm1(row["values"][0])),
                     "reference_mean_rent": float(row["values"][1]),
                     "reference_area": float(row["values"][2]),
                     "distance_from_focal_km": float(np.linalg.norm(xy[row["position"]] - center) / 1000)})
    return pd.DataFrame(rows), {"reference_monthly_rentals": float(target_density),
                                "reference_mean_rent": float(target[1]), "reference_area": float(target[2]),
                                "candidate_centers": int(len(candidates)), "selected": len(rows)}


def build_panels(frame, centers, config):
    """Calendar-complete band/ring outcomes using one fixed property model."""
    f = frame.copy()
    f["residual"] = property_residuals(f)
    months = pd.period_range(f._month.min(), f._month.max(), freq="M")
    buildings = f.drop_duplicates("address_key").set_index("address_key")
    index = buildings.index
    xy = metric_xy(buildings.LONGITUDE, buildings.LATITUDE)
    group = f.groupby(["address_key", "_month"]).residual.agg(["sum", "size"])
    totals = group["sum"].unstack("_month").reindex(index=index, columns=months).fillna(0).to_numpy()
    counts = group["size"].unstack("_month").reindex(index=index, columns=months).fillna(0).to_numpy()
    national = 100 * f.groupby("_month").residual.mean().reindex(months).diff()
    rows = []
    frozen = {r["site_id"]: r for r in [config["primary_hypothesis"], *config["secondary_hypotheses"]]}
    for center in centers.itertuples(index=False):
        point = metric_xy([center.longitude], [center.latitude])[0]
        distance = np.linalg.norm(xy - point, axis=1)
        control = (distance > 3000) & (distance <= 6000)
        cn = counts[control].sum(axis=0)
        cm = np.divide(totals[control].sum(axis=0), cn, out=np.full(len(months), np.nan), where=cn >= config["min_cell_rentals"])
        if center.site_id.startswith("matched_placebo"):
            bands = [("frozen", 0, 1000)]
        elif center.site_id == config["primary_hypothesis"]["site_id"]:
            bands = [("frozen", 0, 1000), ("band_0_1km", 0, 1000), ("band_1_2km", 1000, 2000), ("band_2_3km", 2000, 3000)]
        else:
            bands = [("frozen", 0, frozen[center.site_id]["radius_m"])]
        for band, lower, upper in bands:
            near = (distance > lower) & (distance <= upper) if lower else distance <= upper
            n = counts[near].sum(axis=0)
            mean = np.divide(totals[near].sum(axis=0), n, out=np.full(len(months), np.nan), where=n >= config["min_cell_rentals"])
            premium = pd.Series(100 * (mean - cm), index=months)
            rows.append(pd.DataFrame({"site_id": center.site_id, "company": center.company, "symbol": center.symbol,
                                      "band": band, "month": months.astype(str), "near_n": n.astype(int), "control_n": cn.astype(int),
                                      "premium_logpoints": premium.to_numpy(), "premium_change": premium.diff().to_numpy(),
                                      "national_change": national.to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def expanded_design(local, returns, controls, focal, peers, lag, strengthened):
    original = regression_design(local, returns, focal, peers, lag)
    if not strengthened:
        return original
    d = original.drop(columns=["company_return", "phase"]).copy()
    for col in CONTROL_COLUMNS:
        d[f"{col}_at_signal_month"] = controls[col].shift(lag).reindex(d.index)
    own = returns[focal].copy()
    if not focal.endswith(".SI"):
        own = own + controls.DEXSIUS.reindex(own.index)
    d["company_return"] = own.shift(lag).reindex(d.index)
    d["phase"] = original.phase
    return d


def panel_tests(panel, returns, controls, config, out):
    hypotheses = {r["site_id"]: r for r in [config["primary_hypothesis"], *config["secondary_hypotheses"]]}
    peers = sorted(set(r["symbol"] for r in hypotheses.values()) | {"S59.SI"})
    scores, associations, records = [], [], []
    for (site, band), local in panel.groupby(["site_id", "band"], sort=False):
        focal = local.symbol.iloc[0]
        lag = hypotheses.get(site, config["primary_hypothesis"])["lag"]
        for strengthened in (False, True):
            design = expanded_design(local, returns, controls, focal, peers, lag, strengthened)
            pred = forward_predictions(design)
            if pred.empty or not (pred.phase == "development").any():
                continue
            model = "sector_fx_controls" if strengthened else "original_controls"
            dev = prediction_summary(pred).query("phase == 'development'").set_index("model")
            baseline = dev.loc[["persistence", "rent_market_control"]].rmse_logpoints.idxmin()
            result = compare_outer(pred, baseline)
            scores.append({"site_id": site, "band": band, "symbol": focal, "lag": lag, "model": model,
                           "development_control_rmse": float(dev.loc[baseline, "rmse_logpoints"]),
                           "development_stock_rmse": float(dev.loc["company_stock", "rmse_logpoints"]),
                           "development_ratio": float(dev.loc["company_stock", "rmse_logpoints"] / dev.loc[baseline, "rmse_logpoints"]),
                           **result})
            if not site.startswith("matched_placebo"):
                for period, part in (("development", design.loc["2022-01":"2024-05"]),
                                     ("all_after_reference", design.loc["2022-01":])):
                    associations.append({"site_id": site, "band": band, "symbol": focal, "model": model,
                                         "period": period, **hac_association(part)})
            records.append(pred.assign(site_id=site, band=band, model=model))
    scored = pd.DataFrame(scores)
    scored["delta_rmse_pct"] = 100 * scored.delta_rmse / scored.baseline_rmse
    scored.to_csv(out / "frozen_panel_scores.csv", index=False)
    pd.DataFrame(associations).to_csv(out / "distance_band_associations.csv", index=False)
    pd.concat(records, ignore_index=True).to_csv(out / "panel_predictions.csv", index=False)
    summary = []
    for model in ("original_controls", "sector_fx_controls"):
        true = scored[(scored.site_id == config["primary_hypothesis"]["site_id"]) & (scored.band == "frozen") & (scored.model == model)].iloc[0]
        matched = scored[scored.site_id.str.startswith("matched_placebo") & (scored.model == model) & scored.outer_confirmable]
        summary.append({"model": model, "valid_placebos": len(matched), "true_delta_pct": true.delta_rmse_pct,
                        "placebo_median_delta_pct": matched.delta_rmse_pct.median(),
                        "placebos_improving_more": int((matched.delta_rmse_pct <= true.delta_rmse_pct).sum()),
                        "descriptive_rank_fraction": (1 + (matched.delta_rmse_pct <= true.delta_rmse_pct).sum()) / (1 + len(matched))})
    pd.DataFrame(summary).to_csv(out / "matched_placebo_summary.csv", index=False)
    # Same place, same lag, different company's return; common controls fixed.
    local = panel[(panel.site_id == config["primary_hypothesis"]["site_id"]) & (panel.band == "frozen")]
    design = expanded_design(local, returns, controls, "KLIC", peers, 6, True)
    assignments = []
    for candidate in ["KLIC", "V03.SI", "S63.SI", "FLEX", "5E2.SI", "S59.SI"]:
        d = design.copy()
        r = returns[candidate].copy()
        if not candidate.endswith(".SI"):
            r = r + controls.DEXSIUS.reindex(r.index)
        d["company_return"] = r.shift(6).reindex(d.index)
        p = forward_predictions(d)
        dev = prediction_summary(p).query("phase == 'development'").set_index("model")
        baseline = dev.loc[["persistence", "rent_market_control"]].rmse_logpoints.idxmin()
        assignments.append({"assigned_company_stock": candidate, "site_id": config["primary_hypothesis"]["site_id"], **compare_outer(p, baseline)})
    pd.DataFrame(assignments).to_csv(out / "wrong_company_controls.csv", index=False)
    return scored


def stock_features(rows, sites, returns, controls, hypotheses, cutoff=None):
    """Static distances, global signals, sector-local and company-local terms.

    Future/missing signals are zero with an explicit unavailable flag. A known
    t-L observation keeps its exact date even at multi-month fixed origins.
    """
    coords = metric_xy(rows.LONGITUDE, rows.LATITUDE)
    months = pd.PeriodIndex(rows.RENT_APPROVAL_DATE, freq="M")
    result, groups, audits = {}, {"nuisance": [], "sector": [], "local": []}, []
    lookup = sites.set_index("site_id")
    for hypothesis in hypotheses:
        site, symbol, lag, radius = (hypothesis[k] for k in ("site_id", "symbol", "lag", "radius_m"))
        point = metric_xy([lookup.loc[site, "longitude"]], [lookup.loc[site, "latitude"]])[0]
        distance = np.linalg.norm(coords - point, axis=1)
        exposure = np.where(distance <= 3 * radius, np.exp(-.5 * (distance / radius) ** 2), 0.0)
        prefix = f"WS_{symbol.replace('.', '_')}"
        signal_months = months - lag
        available = np.ones(len(rows), dtype=bool) if cutoff is None else signal_months <= cutoff
        own = returns[symbol].reindex(signal_months).to_numpy(float)
        external = controls[CONTROL_COLUMNS].reindex(signal_months).to_numpy(float)
        if not symbol.endswith(".SI"):
            own = own + external[:, CONTROL_COLUMNS.index("DEXSIUS")]
        missing = (~available) | ~np.isfinite(own) | ~np.isfinite(external).all(axis=1)
        own = np.where(missing, 0., own)
        external = np.where(missing[:, None], 0., external)
        nuisance = {f"{prefix}_DIST_KM": np.minimum(distance / 1000, 30), f"{prefix}_EXPOSURE": exposure,
                    f"{prefix}_RETURN": own, f"{prefix}_UNAVAILABLE": missing.astype(float)}
        nuisance.update({f"{prefix}_{col}": external[:, j] for j, col in enumerate(CONTROL_COLUMNS)})
        sector = {f"{prefix}_LOCAL_{col}": exposure * external[:, j] for j, col in enumerate(CONTROL_COLUMNS)}
        local = {f"{prefix}_LOCAL_COMPANY": exposure * own}
        result.update(nuisance); result.update(sector); result.update(local)
        groups["nuisance"] += list(nuisance)
        groups["sector"] += list(sector)
        groups["local"] += list(local)
        known = signal_months[~missing]
        audits.append({"site_id": site, "rows": len(rows), "available_rows": int((~missing).sum()),
                       "future_signal_rows": int((~available).sum()),
                       "latest_used_stock_month": str(known.max()) if len(known) else None,
                       "cutoff": str(cutoff) if cutoff is not None else "each approval month minus fixed lag"})
    return pd.DataFrame(result, index=rows.index), groups, audits


def weighted_score(actual, predicted, weights):
    error = np.asarray(predicted) - np.asarray(actual)
    return {"rmse_sgd": float(np.sqrt(np.average(error ** 2, weights=weights))),
            "mae_sgd": float(np.average(np.abs(error), weights=weights)),
            "bias_sgd": float(np.average(error, weights=weights)), "weight": float(np.sum(weights)), "rows": len(error)}


def paired_month_interval(part, baseline, candidate, repeats=2000):
    """Weighted RMSE differences with months as paired resampling units."""
    p = part.pivot(index="row_id", columns="variant", values="prediction")
    info = part.drop_duplicates("row_id").set_index("row_id").reindex(p.index)
    baseline_error, candidate_error = (p[baseline] - info.actual) ** 2, (p[candidate] - info.actual) ** 2
    totals = pd.DataFrame({"month": info.month, "weight": info.weight,
                           "base_sse": baseline_error * info.weight, "candidate_sse": candidate_error * info.weight}).groupby("month").sum().sort_index()
    n = len(totals)
    if n < 3:
        return {"months": n, "ci_low": np.nan, "ci_high": np.nan}
    rng = np.random.default_rng(42)
    values = []
    for _ in range(repeats):
        starts = rng.integers(0, n - 2, int(np.ceil(n / 3)))
        positions = np.concatenate([np.arange(s, s + 3) for s in starts])[:n]
        sample = totals.iloc[positions].sum()
        values.append(np.sqrt(sample.candidate_sse / sample.weight) - np.sqrt(sample.base_sse / sample.weight))
    low, high = np.quantile(values, [.025, .975])
    return {"months": n, "ci_low": float(low), "ci_high": float(high)}


def row_ablation(frame, sites, returns, controls, config, out):
    from catboost import CatBoostRegressor

    bundles = [("2022-09", "calendar_inner_2022-09.joblib"), ("2023-07", "calendar_inner_2023-07.joblib"),
               ("2024-06", "calendar_80_20_categorical.joblib")]
    hypotheses = [config["primary_hypothesis"], *config["secondary_hypotheses"]]
    cache = ROOT / ".cache/company_worksite_followup"
    cache.mkdir(parents=True, exist_ok=True)
    scores, records, audits, bundle_hashes, timings = [], [], [], {}, []
    import time

    focal = sites.set_index("site_id").loc[config["primary_hypothesis"]["site_id"]]
    focal_point = metric_xy([focal.longitude], [focal.latitude])[0]
    for origin, filename in bundles:
        path = ROOT / ".cache/tabpfn_v3_features" / filename
        bundle_hashes[str(path.relative_to(ROOT))] = sha256(path)
        b = joblib.load(path)  # Trusted project-generated feature artifact.
        train_ids, query_ids = b["X_train"].index, b["X_query"].index
        fitting, query = frame.loc[train_ids], frame.loc[query_ids]
        np.testing.assert_array_equal(b["y_train"], fitting.MONTHLY_RENT)
        np.testing.assert_array_equal(b["y_query"], query.MONTHLY_RENT)
        if not (fitting.RENT_APPROVAL_DATE < origin).all() or not (query.RENT_APPROVAL_DATE >= origin).all():
            raise ValueError("Cached row fold is not chronological")
        train_extra, groups, audit = stock_features(fitting, sites, returns, controls, hypotheses)
        audits.extend({"origin": origin, "split": "context", "scenario": "historical", **r} for r in audit)
        query_features = {}
        for scenario, cutoff in (("monthly_updated", None), ("fixed_origin", pd.Period(origin, "M") - 1)):
            extra, _, audit = stock_features(query, sites, returns, controls, hypotheses, cutoff)
            query_features[scenario] = extra
            audits.extend({"origin": origin, "split": "query", "scenario": scenario, **r} for r in audit)
        prefix = "WS_KLIC"
        klic = {kind: [c for c in cols if c.startswith(prefix + "_")] for kind, cols in groups.items()}
        options = {"history77": [], "klic_nuisance": klic["nuisance"],
                   "klic_sector_local": klic["nuisance"] + klic["sector"],
                   "klic_local": klic["nuisance"] + klic["sector"] + klic["local"],
                   "all_firm_nuisance": groups["nuisance"],
                   "all_firm_sector_local": groups["nuisance"] + groups["sector"],
                   "all_firm_local": groups["nuisance"] + groups["sector"] + groups["local"]}
        x_base = b["X_train"].copy()
        q_base = b["X_query"].copy()
        categories = x_base.select_dtypes("category").columns.tolist()
        for col in categories:
            x_base[col] = x_base[col].astype("string").fillna("__MISSING__").astype(str)
            q_base[col] = q_base[col].astype("string").fillna("__MISSING__").astype(str)
        residual = np.asarray(b["y_train"], float) - np.asarray(b["train_baseline"], float)
        weights = np.asarray(b.get("query_weights", np.ones(len(query))), float)
        distances = np.linalg.norm(metric_xy(query.LONGITUDE, query.LATITUDE) - focal_point, axis=1)
        subsets = {"all_rentals": np.ones(len(query), dtype=bool), "klic_within_1km": distances <= 1000,
                   "klic_1_2km": (distances > 1000) & (distances <= 2000),
                   "klic_2_3km": (distances > 2000) & (distances <= 3000),
                   "klic_3_6km": (distances > 3000) & (distances <= 6000)}
        for variant, extra in options.items():
            print(f"Row history ablation {origin}: {variant}, {77 + len(extra)} features", flush=True)
            x = pd.concat([x_base, train_extra[extra]], axis=1)
            start = time.perf_counter()
            model = CatBoostRegressor(**CATBOOST_SETTINGS).fit(x, residual, cat_features=categories)
            fit_seconds = time.perf_counter() - start
            timings.append({"origin": origin, "variant": variant, "fit_seconds": fit_seconds, "features": len(x.columns)})
            for scenario, q_extra in query_features.items():
                prediction = model.predict(pd.concat([q_base, q_extra[extra]], axis=1)) + np.asarray(b["query_baseline"], float)
                for subset, mask in subsets.items():
                    if mask.any():
                        scores.append({"origin": origin, "phase": "later_reused" if origin == "2024-06" else "earlier",
                                       "scenario": scenario, "variant": variant, "subset": subset,
                                       **weighted_score(b["y_query"][mask], prediction[mask], weights[mask])})
                records.append(pd.DataFrame({"origin": origin, "scenario": scenario, "variant": variant,
                                             "row_id": query_ids, "month": query.RENT_APPROVAL_DATE.to_numpy(),
                                             "actual": b["y_query"], "prediction": prediction,
                                             "weight": weights, "distance_klic_m": distances}))
            pd.DataFrame(scores).to_csv(out / "row_fold_scores.csv", index=False)
            pd.DataFrame(timings).to_csv(out / "row_timings.csv", index=False)
        del model, b
    pred = pd.concat(records, ignore_index=True)
    pred.to_csv(cache / "row_predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(audits).to_csv(out / "row_feature_availability.csv", index=False)
    scored = pd.DataFrame(scores)
    comparisons = []
    for scenario in ("monthly_updated", "fixed_origin"):
        for subset, mask in (("all_rentals", np.ones(len(pred), bool)), ("klic_within_1km", pred.distance_klic_m <= 1000)):
            for candidate, baseline in (("klic_local", "klic_sector_local"), ("klic_local", "klic_nuisance"),
                                        ("klic_local", "history77"), ("all_firm_local", "all_firm_sector_local"),
                                        ("all_firm_local", "history77")):
                group = scored[(scored.scenario == scenario) & (scored.subset == subset)]
                dev = group[group.phase == "earlier"].copy()
                pooled = dev.assign(sse=dev.rmse_sgd ** 2 * dev.weight).groupby("variant")[["sse", "weight"]].sum()
                earlier_b = np.sqrt(pooled.loc[baseline, "sse"] / pooled.loc[baseline, "weight"])
                earlier_c = np.sqrt(pooled.loc[candidate, "sse"] / pooled.loc[candidate, "weight"])
                outer = group[group.phase == "later_reused"].set_index("variant")
                paired = pred[mask & (pred.scenario == scenario) & (pred.origin == "2024-06") & pred.variant.isin([baseline, candidate])]
                comparisons.append({"scenario": scenario, "subset": subset, "candidate": candidate, "baseline": baseline,
                                    "earlier_control_rmse": earlier_b, "earlier_candidate_rmse": earlier_c,
                                    "earlier_delta_rmse": earlier_c - earlier_b,
                                    "later_control_rmse": outer.loc[baseline, "rmse_sgd"],
                                    "later_candidate_rmse": outer.loc[candidate, "rmse_sgd"],
                                    "later_delta_rmse": outer.loc[candidate, "rmse_sgd"] - outer.loc[baseline, "rmse_sgd"],
                                    **paired_month_interval(paired, baseline, candidate)})
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(out / "row_incremental_comparisons.csv", index=False)
    return comparison, bundle_hashes


def make_plots(scores, panel, out, row_comparison):
    focal = scores[(scores.site_id == "kulicke_soffa_serangoon") & (scores.band != "frozen")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for model, part in focal.groupby("model", sort=False):
        part = part.sort_values("band")
        axes[0].plot(part.band, part.delta_rmse_pct, marker="o", label=model.replace("_", " "))
    axes[0].axhline(0, color="#555", linewidth=1)
    axes[0].set(title="K&S distance bands | lag fixed at six months", ylabel="Later rent-growth RMSE change (%)")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].legend(fontsize=8)
    controls = scores[(scores.band == "frozen") & (scores.model == "sector_fx_controls") & scores.outer_confirmable]
    values = controls[controls.site_id.str.startswith("matched_placebo")].delta_rmse_pct
    true = controls[controls.site_id == "kulicke_soffa_serangoon"].delta_rmse_pct.iloc[0]
    axes[1].hist(values, bins=12, color="#68a5a0", edgecolor="white")
    axes[1].axvline(true, color="#ba6046", linewidth=2, label=f"Actual K&S site: {true:+.1f}%")
    axes[1].set(title="Reference-year-matched distant neighborhoods", xlabel="Later RMSE change from adding KLIC return (%)", ylabel="Number of eligible placebo centers")
    axes[1].legend(fontsize=8)
    fig.savefig(out / "01_distance_and_matched_placebos.png", dpi=160)
    plt.close(fig)
    if row_comparison is not None:
        p = row_comparison[(row_comparison.candidate == "klic_local") & (row_comparison.baseline == "klic_sector_local")]
        fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
        labels = p.scenario + " / " + p.subset
        y = np.arange(len(p))
        ax.barh(y, p.later_delta_rmse, color=["#277c80" if x < 0 else "#ba6046" for x in p.later_delta_rmse])
        ax.hlines(y, p.ci_low, p.ci_high, color="#333", linewidth=2)
        ax.axvline(0, color="#555", linewidth=1)
        ax.set_yticks(y, labels)
        ax.set(title="Incremental local KLIC interaction over local sector/FX controls", xlabel="Change in individual-rental RMSE (SGD); later period already examined")
        fig.savefig(out / "02_individual_rental_increment.png", dpi=160)
        plt.close(fig)


def table(frame, columns):
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for values in frame[columns].itertuples(index=False, name=None):
        rows.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v) for v in values) + " |")
    return "\n".join(rows)


def write_report(config, scores, row_comparison, matching, out):
    focal = scores[(scores.site_id == "kulicke_soffa_serangoon") & (scores.band == "frozen")]
    ranks = pd.read_csv(out / "matched_placebo_summary.csv")
    distance = pd.read_csv(out / "distance_band_associations.csv")
    distance = distance[(distance.site_id == "kulicke_soffa_serangoon") & (distance.band != "frozen") & (distance.period == "development")]
    wrong = pd.read_csv(out / "wrong_company_controls.csv")
    text = ["# Company stock returns near engineering worksites: follow-up", "",
            "This follows the exploratory K&S 1 km/six-month-lag finding. Company/site/radius/lag hypotheses were frozen from the earlier report. The June 2024-March 2025 rental period has already been examined: these are robustness and development checks, not fresh confirmation. No submission file is modified.", "",
            "## Is the K&S signal independent of global semiconductor and currency moves?", "",
            "The original near-minus-3-6 km comparison is repeated with US S&P 500, semiconductor-sector and USD/SGD returns measured at the same six-month-lag signal date. USD-listed company returns are also converted to SGD log returns. Existing recent local rent changes, national adjusted rent growth and broad/peer Singapore stock returns remain controls. All available observations precede approval; no current/future rental label enters its prediction fit.", "",
            table(focal, ["model", "development_control_rmse", "development_stock_rmse", "development_ratio", "baseline_rmse", "stock_rmse", "delta_rmse_pct", "delta_ci_low", "delta_ci_high"]), "",
            "Errors here concern monthly property-adjusted local relative growth (log points), not individual rents. Paired three-month block intervals are conditional on fitted models and omit the earlier radius/lag search. Exact t-6 returns are used; a six-month lag is not a six-month cumulative return.", "",
            "## Distance decay and spatial specificity", "",
            "Disjoint 0-1, 1-2 and 2-3 km bands use the same 3-6 km reference ring. A stronger association close to the site would be more consistent with a local channel, but small counts, shared homes and a site's particular geography prevent a causal interpretation. Nominal partial-slope intervals below are not a new multiple-testing discovery claim.", "",
            table(distance, ["band", "model", "months", "effect_per_10pct_stock", "effect_ci_low", "effect_ci_high", "p_value"]), "",
            f"Matched spatial placebos use {matching['selected']} HDB-building centers at least 6 km from the K&S site and at least 1.2 km from each other. Match covariates use **2021 only**: rental density, mean rent, floor area, lease year and flat-type mix. The true site's reference density is {matching['reference_monthly_rentals']:.1f} rentals/month, versus {matching['reference_mean_rent']:.0f} SGD mean rent. There is no selection on later rent growth or model errors. Some placebos can fail monthly coverage and are reported as unavailable rather than counted as successful controls.", "",
            table(ranks, ["model", "valid_placebos", "true_delta_pct", "placebo_median_delta_pct", "placebos_improving_more", "descriptive_rank_fraction"]), "",
            "The rank fraction describes this matched set; it is not a randomized-treatment p-value. Placebo neighborhoods remain correlated and may contain unrecorded employers. Sites are points, not employment-weighted factory boundaries. Matching equalizes observable reference-year housing characteristics only.", "",
            "## Same location, other companies' stocks", "",
            "Keep the K&S geography, six-month lag and sector/FX controls fixed, but replace only the company-return signal. If another company's stock works as well, the effect is less clearly company-specific. These are diagnostic controls, not new candidate selection.", "",
            table(wrong, ["assigned_company_stock", "baseline_rmse", "stock_rmse", "delta_rmse", "delta_ci_low", "delta_ci_high"]), ""]
    if row_comparison is not None:
        main = row_comparison[(row_comparison.candidate == "klic_local") & (row_comparison.baseline == "klic_sector_local")]
        text.extend(["## Does it add value for individual rentals with rent-history features?", "",
                     "A matched CatBoost residual model uses the existing 77 aligned TabPFN input features, including high-cardinality block/street IDs, strictly prior rent histories and the identical temporal baseline offsets. The backbone here is CatBoost, not a newly trained TabPFN. Every ablation retains the same context/query rows and fits y minus the existing temporal baseline, then adds that baseline once. The 2022-09 and 2023-07 origins have 2,000 monthly-stratified query samples each, pooled with their population weights; 2024-06 uses all 28,995 query rentals. Earlier/later scores are not interchangeable with the monthly-index RMSE above.", "",
                     "Nuisance inputs include site distance/exposure, standalone lagged company and US-market/sector/FX signals and availability flags. Sector-local models add distance-weighted SP500/SOX/FX terms. Only the company-local model adds exposure times the company's return, so a geometry improvement or a nearby semiconductor-cycle effect cannot alone count as company-specific value. The primary K&S kernel is Gaussian with 1 km bandwidth, truncated at 3 km. The frozen peer-site hypotheses use their earlier radii/lags. All-firm versions test the five frozen hypotheses jointly.", "",
                     "Both scenarios freeze rental histories at the fold origin. Monthly-updated stocks use each row's historical t-L observations. Fixed-origin stocks use t-L only if it is already observable at the fold cutoff; otherwise zero plus an unavailable indicator is used. Unknown future stocks are never filled with their realized values. This missingness policy is a diagnostic deployment assumption, not a forecast of future stock prices.", "",
                     table(main, ["scenario", "subset", "earlier_control_rmse", "earlier_candidate_rmse", "earlier_delta_rmse", "later_control_rmse", "later_candidate_rmse", "later_delta_rmse", "ci_low", "ci_high"]), "",
                     "See row_incremental_comparisons.csv for nuisance-only, 77-feature rent-only and all-firm controls, and row_fold_scores.csv for separate folds and distance bands. Intervals resample paired month blocks and retain rental/sample weights; they are conditional and do not account for model selection. Full reproducible row predictions are cached under .cache/company_worksite_followup/row_predictions.csv.gz.", ""])
    text.extend(["## Interpretation and promotion", "",
                 "Treat any gain as a hypothesis about predictive information, not proof that stock prices cause nearby rents. A surviving company-local signal would require better earlier pooled performance against rent-history and local-sector controls, later near-site improvement with limited whole-sample harm, robust geographic specificity, and ultimately new rental months plus a matched TabPFN evaluation. No radius or lag is reselected on the reused later period. These tests do not measure employee residence, hiring, bonuses or workplace headcounts; common drivers and commuting remain plausible explanations.", "",
                 "The original 63-scan search found no association surviving dependent multiple-testing correction. This follow-up does not reset that search history. There are only 51 rental months. Market controls and adjusted company prices are later-vintage histories; complete point-in-time release archives are unavailable. Current address pins locate historically documented sites, but exact historical plant occupancy and boundaries are unverified. Currency conversions use monthly endpoints; global market closes and official release timing can require additional availability lags in live use.", "",
                 "## Sources and reproduction", "",
                 "Sources: [S&P 500](https://fred.stlouisfed.org/series/SP500), [Nasdaq Composite](https://fred.stlouisfed.org/series/NASDAQCOM), [semiconductor sector](https://fred.stlouisfed.org/series/NASDAQSOX), [SGD per USD](https://fred.stlouisfed.org/series/DEXSIUS). Corporate addresses retain the independently documented sources from the original worksite report. The design is recorded in analysis/configs/company_worksite_followup.json.", "",
                 "Run `.venv/Scripts/python.exe analysis/acquire_worksite_market_controls.py`, then `.venv/Scripts/python.exe analysis/compare_company_worksite_followup.py`. `--panel-only` omits the individual-rental ablation. Source/code hashes and cached 77-feature bundle hashes are recorded in results.json; input data and existing experiment outputs are preserved.", ""])
    (out / "report.md").write_text("\n".join(text), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    out = REPORTS / "company_worksite_followup"
    out.mkdir(parents=True, exist_ok=True)
    sites_path = REPORTS / "company_worksite_stocks/worksite_locations.csv"
    source_paths = [DATA / "train.csv", AUX / "sg-hdb-block.csv", AUX / "sg-stock-prices.csv", CONFIG, sites_path,
                    out / "market_monthly_log_returns.csv", out / "market_source_manifest.json"]
    before = {str(p.relative_to(ROOT)): sha256(p) for p in source_paths}
    sites = pd.read_csv(sites_path)
    with threadpool_limits(limits=1):
        frame = rental_coordinates(pd.read_csv(DATA / "train.csv"), pd.read_csv(AUX / "sg-hdb-block.csv"))
        controls = read_controls()
        returns = stock_returns(pd.read_csv(AUX / "sg-stock-prices.csv"), sites.symbol.unique().tolist(), str(frame._month.max()))
        matches, matching = match_placebos(frame, sites, config)
        matches.to_csv(out / "matched_placebo_centers.csv", index=False)
        focal_ids = [config["primary_hypothesis"]["site_id"], *[r["site_id"] for r in config["secondary_hypotheses"]]]
        centers = pd.concat([sites[sites.site_id.isin(focal_ids)], matches], ignore_index=True)
        panel = build_panels(frame, centers, config)
        panel.to_csv(out / "local_monthly_panel.csv", index=False)
        scores = panel_tests(panel, returns, controls, config, out)
        row_comparison, bundle_hashes = (None, {}) if args.panel_only else row_ablation(frame, sites, returns, controls, config, out)
        make_plots(scores, panel, out, row_comparison)
    if before != {str(p.relative_to(ROOT)): sha256(p) for p in source_paths}:
        raise AssertionError("An input changed")
    metadata = {"config": config, "matching": matching, "source_sha256": before,
                "feature_bundle_sha256": bundle_hashes,
                "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__),
                                ROOT / "analysis/compare_company_worksite_followup.py", ROOT / "analysis/acquire_worksite_market_controls.py",
                                ROOT / "analysis/experiments/company_worksite_stocks.py", ROOT / "analysis/experiments/stock_sector_signals.py"]},
                "row_model_settings": CATBOOST_SETTINGS, "versions": {p: version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn", "catboost")},
                "input_hashes_unchanged": True, "row_ablation_completed": row_comparison is not None}
    (out / "results.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(config, scores, row_comparison, matching, out)
    print(f"Follow-up saved to {out / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
