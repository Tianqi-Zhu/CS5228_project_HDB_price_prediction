"""Audit saved TabPFN Plus validation errors and unlabeled submission coverage.

The eight-estimator full-data submission is never assigned a validation score:
the labeled Plus run has one estimator and a smaller training context.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

from analysis.config import ROOT, REPORTS
from analysis.data import sha256
from analysis.interpret_tabpfn_abcde import spatial_weights, moran_i, weighted_rmse


API = REPORTS / "tabpfn_api_77"
PLUS = API / "validation_v3.5_default_standard_e1/predictions.csv"
FULL = API / "submission_v3.5_plus_77_150k/predictions.csv"
ENSEMBLE = REPORTS / "tabpfn_abcde_ensemble/predictions_all_origins.csv"
ENSEMBLE_SUBMISSION = REPORTS / "tabpfn_top2_final_submission/submission_tabpfn_ABCDE_equal_150k.csv"
VALID_X = ROOT / ".cache/tabpfn_api_77_inputs/validation/query.parquet"
VALID_ARRAYS = ROOT / ".cache/tabpfn_api_77_inputs/validation/arrays.npz"
TEST_X = ROOT / ".cache/tabpfn_api_77_inputs/submission/query.parquet"
TEST_ARRAYS = ROOT / ".cache/tabpfn_api_77_inputs/submission/arrays.npz"
OUT = API / "remaining_gaps"
MODELS = ["Plus_e1", "Fast_B_e2", "ABCDE_equal"]
COLORS = {"Plus_e1": "#a21caf", "Fast_B_e2": "#2563eb", "ABCDE_equal": "#111827"}


def load():
    plus = pd.read_csv(PLUS, float_precision="round_trip")
    ensemble = pd.read_csv(ENSEMBLE, float_precision="round_trip")
    ensemble = ensemble[ensemble.origin.eq("2024-06")].reset_index(drop=True)
    x = pd.read_parquet(VALID_X)
    with np.load(VALID_ARRAYS) as a:
        y = a["y_query"].copy()
    assert len(plus) == len(ensemble) == len(x) == len(y) == 28995
    np.testing.assert_array_equal(plus.row_id, ensemble.row_id)
    np.testing.assert_array_equal(plus.row_id, x.index)
    np.testing.assert_array_equal(plus.month, ensemble.month)
    np.testing.assert_allclose(plus.actual, ensemble.actual, atol=1e-8)
    np.testing.assert_allclose(plus.actual, y, atol=1e-8)
    f = plus[["row_id", "month", "actual"]].copy()
    f["Plus_e1"] = plus.predicted.to_numpy(float)
    f["Fast_B_e2"] = ensemble.B.to_numpy(float)
    f["ABCDE_equal"] = ensemble.ABCDE_equal.to_numpy(float)
    for view in ["A", "C", "D", "E"]:
        f[view] = ensemble[view].to_numpy(float)
    f["town"] = x.TOWN_CLEAN.astype(str).to_numpy()
    f["flat_type"] = x.FLAT_TYPE_CLEAN.astype(str).to_numpy()
    f["block"] = x.BLOCK_ID.astype(str).to_numpy()
    for name in ["LATITUDE", "LONGITUDE", "FLOOR_AREA_SQM", "FLAT_AGE_YEARS",
                 "NEAREST_MRT_KM", "HIST_BLOCK_FLAT_COUNT", "TEMP_FORECAST_HORIZON"]:
        f[name] = pd.to_numeric(x[name], errors="coerce").to_numpy(float)
    test_x = pd.read_parquet(TEST_X)
    full = pd.read_csv(FULL, float_precision="round_trip")
    ensemble_full = pd.read_csv(ENSEMBLE_SUBMISSION, float_precision="round_trip")
    with np.load(TEST_ARRAYS) as a:
        test_baseline = a["query_baseline"].copy()
    assert len(test_x) == len(full) == len(test_baseline) == 50000
    assert len(ensemble_full) == len(full)
    np.testing.assert_array_equal(full.row_id, test_x.index)
    np.testing.assert_array_equal(ensemble_full.Id, full.row_id)
    assert np.isfinite(full.predicted).all()
    full["ABCDE_equal"] = ensemble_full.Predicted.to_numpy(float)
    full["difference"] = full.predicted - full.ABCDE_equal
    return f, test_x, full, test_baseline


def grouped(f, columns):
    rows = []
    for column in columns:
        for group, g in f.groupby(column, observed=True, sort=True):
            if len(g) < 25:
                continue
            for model in MODELS:
                error = g[model] - g.actual
                rows.append({"dimension": column, "group": str(group), "model": model,
                    "rows": len(g), "rmse_sgd": weighted_rmse(g.actual, g[model], np.ones(len(g))),
                    "bias_sgd": float(error.mean()), "mae_sgd": float(error.abs().mean())})
    return pd.DataFrame(rows)


def town_month(f):
    rows = []
    for (town, month), g in f.groupby(["town", "month"], sort=True):
        for model in ["Plus_e1", "ABCDE_equal"]:
            rows.append({"town": town, "month": month, "model": model, "rows": len(g),
                         "bias_sgd": float((g[model] - g.actual).mean())})
    return pd.DataFrame(rows)


def moran(f):
    rng = np.random.default_rng(5228)
    rows = []
    for month, g in f.groupby("month", sort=True):
        cell = g.groupby("block").agg(rows=("actual", "size"),
            lat=("LATITUDE", "first"), lon=("LONGITUDE", "first"),
            actual=("actual", "mean"), Plus_e1=("Plus_e1", "mean"),
            ABCDE_equal=("ABCDE_equal", "mean"))
        cell = cell[cell.rows >= 2].dropna(subset=["lat", "lon"])
        xy = np.column_stack([(cell.lon - 103.85) * 111.2 * np.cos(np.deg2rad(1.35)),
                              (cell.lat - 1.35) * 111.2])
        weights = spatial_weights(xy)
        strata = np.minimum(cell.rows.to_numpy(), 5)
        perms = []
        for _ in range(199):
            permutation = np.arange(len(cell))
            for stratum in np.unique(strata):
                ids = np.flatnonzero(strata == stratum)
                permutation[ids] = rng.permutation(ids)
            perms.append(permutation)
        for model in ["Plus_e1", "ABCDE_equal"]:
            z = (cell[model] - cell.actual).to_numpy(float)
            reference = np.asarray([moran_i(z[p], weights) for p in perms])
            rows.append({"month": month, "model": model, "blocks": len(cell),
                         "moran_i": moran_i(z, weights),
                         "reference_high": float(np.nanquantile(reference, .975))})
    return pd.DataFrame(rows)


def curves(f):
    rows = []
    for feature in ["FLOOR_AREA_SQM", "FLAT_AGE_YEARS", "NEAREST_MRT_KM", "HIST_BLOCK_FLAT_COUNT"]:
        x = f[feature]
        bins = pd.qcut(x, 8, duplicates="drop")
        for interval, g in f.groupby(bins, observed=True):
            if len(g) < 50:
                continue
            for model in MODELS:
                rows.append({"feature": feature, "bin": str(interval), "feature_mean": float(g[feature].mean()),
                             "model": model, "rows": len(g),
                             "bias_sgd": float((g[model] - g.actual).mean()),
                             "rmse_sgd": weighted_rmse(g.actual, g[model], np.ones(len(g)))})
    return pd.DataFrame(rows)


def figures(month, towns, curves, test_horizon):
    monthly = make_subplots(rows=2, cols=1, shared_xaxes=True,
        subplot_titles=["RMSE by month", "Prediction bias by month"])
    for model in MODELS:
        z = month[month.model.eq(model)]
        for row, metric in [(1, "rmse_sgd"), (2, "bias_sgd")]:
            monthly.add_trace(go.Scatter(x=z.group, y=z[metric], mode="lines+markers",
                name=model, legendgroup=model, showlegend=row == 1,
                line=dict(color=COLORS[model])), row=row, col=1)
    monthly.update_layout(height=620, title="Labeled June 2024–March 2025 validation; Plus has one estimator")
    monthly.update_yaxes(title_text="S$", row=1, col=1)
    monthly.update_yaxes(title_text="S$", row=2, col=1)

    names = sorted(towns.town.unique())
    months = sorted(towns.month.unique())
    heat = make_subplots(rows=1, cols=2, subplot_titles=["Plus e1", "ABCDE equal"])
    for col, model in [(1, "Plus_e1"), (2, "ABCDE_equal")]:
        part = towns[towns.model.eq(model)]
        table = part.assign(value=part.bias_sgd.where(part.rows >= 30)).pivot(
            index="town", columns="month", values="value").reindex(index=names, columns=months)
        heat.add_trace(go.Heatmap(z=table.to_numpy(), x=months, y=names,
            zmin=-220, zmax=220, colorscale="RdBu", reversescale=True,
            showscale=col == 2, colorbar=dict(title="S$") if col == 2 else None), row=1, col=col)
    heat.update_layout(height=800, title="Town × month bias; cells with fewer than 30 rentals hidden")
    heat.update_yaxes(autorange="reversed")

    feature_plot = make_subplots(rows=2, cols=2,
        subplot_titles=["Floor area", "Flat age", "MRT distance", "Earlier block-flat support"])
    for j, feature in enumerate(["FLOOR_AREA_SQM", "FLAT_AGE_YEARS", "NEAREST_MRT_KM", "HIST_BLOCK_FLAT_COUNT"]):
        for model in MODELS:
            part = curves[(curves.feature.eq(feature)) & (curves.model.eq(model))]
            feature_plot.add_trace(go.Scatter(x=part.feature_mean, y=part.bias_sgd,
                mode="lines+markers", name=model, legendgroup=model, showlegend=j == 0,
                line=dict(color=COLORS[model])), row=j // 2 + 1, col=j % 2 + 1)
    feature_plot.update_layout(height=700,
        title="Error across marginal feature bins; other property attributes vary")
    feature_plot.update_yaxes(title_text="Prediction − rent (S$)")

    coverage = make_subplots(rows=2, cols=1, shared_xaxes=True,
        subplot_titles=["Unlabeled test rows", "Mean absolute disagreement: eight-estimator Plus vs ABCDE equal"])
    coverage.add_trace(go.Bar(x=test_horizon.horizon, y=test_horizon.rows, name="Rows",
        marker_color=["#2563eb" if h <= 10 else "#dc2626" for h in test_horizon.horizon]), row=1, col=1)
    coverage.add_trace(go.Scatter(x=test_horizon.horizon, y=test_horizon.mean_abs_disagreement,
        mode="lines+markers", name="Mean absolute difference", line=dict(color="#a21caf")), row=2, col=1)
    coverage.add_vline(x=10.5, line_dash="dash", line_color="black")
    coverage.update_layout(height=570, title="Full-data submission coverage and model disagreement",
        xaxis2_title="Months after source cutoff")
    coverage.update_yaxes(title_text="Rows", row=1, col=1)
    coverage.update_yaxes(title_text="S$", row=2, col=1)
    return [monthly, heat, feature_plot, coverage]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    f, test_x, full, test_baseline = load()
    f["support_bin"] = pd.cut(f.HIST_BLOCK_FLAT_COUNT.fillna(-1),
        bins=[-2, 0, 3, 10, 30, np.inf], labels=["missing", "1–3", "4–10", "11–30", ">30"])
    dimensions = grouped(f, ["month", "town", "flat_type", "support_bin", "TEMP_FORECAST_HORIZON"])
    tm = town_month(f)
    eligible = tm[(tm.model.eq("Plus_e1")) & (tm.rows >= 30)]
    persistent = eligible.groupby("town").filter(
        lambda g: len(g) == 10 and bool((g.bias_sgd > 0).all())).town.unique()
    six_rows = []
    for town, part in f[f.town.isin(persistent)].groupby("town", sort=True):
        for view, column in [("Plus_e1", "Plus_e1"), ("A", "A"), ("B", "Fast_B_e2"),
                             ("C", "C"), ("D", "D"), ("E", "E"),
                             ("ABCDE_equal", "ABCDE_equal")]:
            six_rows.append({"town": town, "view": view, "rows": len(part),
                             "mean_error_sgd": float((part[column] - part.actual).mean())})
    pd.DataFrame(six_rows).to_csv(OUT / "persistent_town_view_bias.csv", index=False)
    spatial = moran(f)
    feature = curves(f)
    test_horizon = full.assign(horizon=pd.to_numeric(test_x.TEMP_FORECAST_HORIZON.to_numpy(),
        errors="coerce").astype(int)).groupby("horizon").agg(rows=("difference", "size"),
        mean_signed_disagreement=("difference", "mean"),
        mean_abs_disagreement=("difference", lambda a: float(np.mean(np.abs(a)))),
        p95_abs_disagreement=("difference", lambda a: float(np.quantile(np.abs(a), .95)))).reset_index()
    test_horizon["fraction"] = test_horizon.rows / len(test_x)
    full_groups = full.assign(horizon=pd.to_numeric(test_x.TEMP_FORECAST_HORIZON.to_numpy(),
        errors="coerce").astype(int), town=test_x.TOWN_CLEAN.astype(str).to_numpy())
    test_town = full_groups.groupby("town").agg(rows=("difference", "size"),
        mean_signed_disagreement=("difference", "mean"),
        mean_abs_disagreement=("difference", lambda a: float(np.mean(np.abs(a))))).reset_index()
    test_town.to_csv(OUT / "test_town_disagreement.csv", index=False)
    for name, data in [("grouped_errors", dimensions), ("town_month_bias", tm),
                       ("spatial_autocorrelation", spatial), ("feature_error_bins", feature),
                       ("test_horizon_coverage", test_horizon)]:
        data.to_csv(OUT / f"{name}.csv", index=False)
    charts = figures(dimensions[dimensions.dimension.eq("month")], tm, feature, test_horizon)
    notes = ["<h1>Remaining gaps in TabPFN predictions</h1>",
        "<p>The labeled Plus forecast shown here has one estimator and 121,005 context rows. The active submission has eight estimators and 150,000 context rows; its 50,000 test labels are unavailable. Errors shown below are not measured errors of that submission.</p>",
        "<p>Feature-bin errors are descriptive and may reflect correlated features. Red bars mark horizons beyond the ten months in labeled local validation.</p>"]
    for i, chart in enumerate(charts):
        notes.append(pio.to_html(chart, full_html=False, include_plotlyjs="inline" if i == 0 else False))
    (OUT / "dashboard.html").write_text("<!doctype html><html><head><meta charset='utf-8'><title>TabPFN remaining gaps</title><style>body{font-family:system-ui,sans-serif;margin:2rem auto;max-width:1400px;color:#111827}p{max-width:950px;line-height:1.5}</style></head><body>" + "\n".join(notes) + "</body></html>", encoding="utf-8")
    error = f.ABCDE_equal - f.actual
    difference = f.Plus_e1 - f.ABCDE_equal
    directional = float(np.mean(error * difference))
    squared = float(np.mean(difference ** 2))
    result = {"validation_rows": len(f), "test_rows": len(full),
        "validation_rmse": {name: weighted_rmse(f.actual, f[name], np.ones(len(f))) for name in MODELS},
        "plus_ensemble_error_correlation": float(np.corrcoef(f.Plus_e1-f.actual, error)[0, 1]),
        "directional_blend_crossmoment": directional,
        "development_only_optimal_plus_fraction": float(np.clip(-directional / squared, 0, 1)),
        "test_rows_horizon_above_10": int(test_horizon.loc[test_horizon.horizon.gt(10), "rows"].sum()),
        "test_rows_horizon_above_10_fraction": float(test_horizon.loc[test_horizon.horizon.gt(10), "fraction"].sum()),
        "full_submission_mean_signed_difference_plus_minus_abcde": float(full.difference.mean()),
        "full_submission_mean_abs_difference_plus_vs_abcde": float(full.difference.abs().mean()),
        "full_submission_mean_abs_difference_horizon_1_10": float(full_groups.loc[
            full_groups.horizon.le(10), "difference"].abs().mean()),
        "full_submission_mean_abs_difference_horizon_11_16": float(full_groups.loc[
            full_groups.horizon.gt(10), "difference"].abs().mean()),
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in
             [PLUS, FULL, ENSEMBLE, ENSEMBLE_SUBMISSION, VALID_X, VALID_ARRAYS, TEST_X, TEST_ARRAYS]},
        "alignment_checked": True, "full_submission_labels_available": False,
        "full_submission_validation_equivalent": False}
    (OUT / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "source_sha256"}, indent=2))


if __name__ == "__main__":
    main()
