"""Saved-forecast spatial, temporal and descriptive feature diagnostics for A-E.

This script does not refit TabPFN or call its GPU. Binned curves show observed
and predicted associations; they are not feature interventions or SHAP values.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

from analysis.config import ROOT, REPORTS
from analysis.data import sha256


SOURCE = REPORTS / "tabpfn_abcde_ensemble/predictions_all_origins.csv"
BUNDLE = ROOT / ".cache/tabpfn_feature_diversified_ensemble_80_20/features_2024-06_baseline_rebenchmark.joblib"
SPATIAL_BUNDLE = ROOT / ".cache/tabpfn_feature_diversified_ensemble_80_20/features_2024-06_spatial.joblib"
TEMPORAL_BUNDLE = ROOT / ".cache/tabpfn_feature_diversified_ensemble_80_20/features_2024-06_temporal.joblib"
OUT = REPORTS / "tabpfn_abcde_ensemble/interpretation"
MODELS = ["A", "B", "C", "D", "E", "ABCDE_equal"]
COLORS = {"A": "#6d28d9", "B": "#2563eb", "C": "#0891b2", "D": "#f59e0b",
          "E": "#059669", "ABCDE_equal": "#111827"}
FEATURES = ["FLOOR_AREA_SQM", "FLAT_AGE_YEARS", "NEAREST_MRT_KM",
            "DISTANCE_TO_CBD_KM", "HIST_BLOCK_FLAT_COUNT", "LOCAL_KERNEL_MASS"]


def weighted_rmse(y, p, w):
    return float(np.sqrt(np.average((np.asarray(p) - np.asarray(y)) ** 2, weights=w)))


def load():
    predictions = pd.read_csv(SOURCE, float_precision="round_trip")
    main = predictions[predictions.origin.eq("2024-06")].copy()
    if len(main) != 28995 or not main.row_id.is_unique:
        raise ValueError("Unexpected main forecast rows")
    manifests = [
        REPORTS / "tabpfn_feature_diversified_ensemble_80_20/features_2024-06_baseline_rebenchmark.json",
        REPORTS / "tabpfn_feature_diversified_ensemble_80_20/features_2024-06_spatial.json",
        REPORTS / "tabpfn_feature_diversified_ensemble_80_20/features_2024-06_temporal.json",
    ]
    bundles = []
    for path, manifest in zip([BUNDLE, SPATIAL_BUNDLE, TEMPORAL_BUNDLE], manifests):
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        if sha256(path) != metadata["sha256"]:
            raise ValueError(f"Feature manifest changed: {path}")
        bundles.append(joblib.load(path))
    base, spatial, temporal = bundles
    for other in bundles:
        np.testing.assert_array_equal(other["X_query"].index, main.row_id)
        np.testing.assert_allclose(other["y_query"], main.actual, atol=1e-8)
        np.testing.assert_array_equal(other["query_month"], main.month)
    np.testing.assert_allclose(main.ABCDE_equal, main[["A", "B", "C", "D", "E"]].mean(axis=1), atol=1e-8)
    x = base["X_query"]
    e = spatial["X_query"]
    main["town"] = x.TOWN_CLEAN.astype(str).to_numpy()
    main["flat_type"] = x.FLAT_TYPE_CLEAN.astype(str).to_numpy()
    main["block"] = x.BLOCK_ID.astype(str).to_numpy()
    main["latitude"] = pd.to_numeric(x.LATITUDE, errors="coerce").to_numpy()
    main["longitude"] = pd.to_numeric(x.LONGITUDE, errors="coerce").to_numpy()
    main["baseline"] = base["query_baseline"]
    main["horizon"] = pd.to_numeric(x.TEMP_FORECAST_HORIZON, errors="coerce").to_numpy()
    for feature in FEATURES:
        main[feature] = pd.to_numeric((e if feature in e else x)[feature], errors="coerce").to_numpy()
    return predictions, main, bundles, manifests


def scores(main):
    records = []
    for model in MODELS:
        for month, f in main.groupby("month", sort=True):
            w = f.weight.to_numpy(float)
            error = f[model].to_numpy(float) - f.actual.to_numpy(float)
            records.append({"model": model, "month": month, "rows": len(f),
                            "rmse_sgd": weighted_rmse(f.actual, f[model], w),
                            "bias_sgd": float(np.average(error, weights=w))})
    return pd.DataFrame(records)


def town_month(main):
    rows = []
    for (town, month), f in main.groupby(["town", "month"], sort=True):
        row = {"town": town, "month": month, "rows": len(f)}
        for model in ["B", "E", "ABCDE_equal"]:
            row[f"bias_{model}"] = float(np.average(f[model] - f.actual, weights=f.weight))
        rows.append(row)
    return pd.DataFrame(rows)


def town_month_interaction(main):
    """Residual town-month cell means after weighted additive town/month fit."""
    t = pd.get_dummies(main.town, dtype=float)
    m = pd.get_dummies(main.month, dtype=float)
    design = np.column_stack([np.ones(len(main)), t.iloc[:, 1:].to_numpy(),
                              m.iloc[:, 1:].to_numpy()])
    scale = np.sqrt(main.weight.to_numpy(float))
    weighted_design = design * scale[:, None]
    rows = []
    for model in ["B", "E", "ABCDE_equal"]:
        residual = (main[model] - main.actual).to_numpy(float)
        beta = np.linalg.lstsq(weighted_design, residual * scale, rcond=None)[0]
        interaction = residual - design @ beta
        frame = main[["town", "month", "weight"]].copy()
        frame["interaction"] = interaction
        for (town, month), cell in frame.groupby(["town", "month"], sort=True):
            rows.append({"model": model, "town": town, "month": month,
                         "rows": len(cell), "interaction_bias_sgd": float(np.average(
                             cell.interaction, weights=cell.weight))})
    return pd.DataFrame(rows)


def spatial_weights(coords):
    n = len(coords)
    if n < 3:
        return csr_matrix((n, n))
    distance, neighbor = cKDTree(coords).query(coords, k=min(9, n))
    rows, cols, values = [], [], []
    for i in range(n):
        good = (neighbor[i] != i) & (distance[i] <= 2.)
        ids, d = neighbor[i][good], distance[i][good]
        if len(ids):
            weight = np.exp(-.5 * d * d)
            weight /= weight.sum()
            rows.extend([i] * len(ids))
            cols.extend(ids)
            values.extend(weight)
    return csr_matrix((values, (rows, cols)), shape=(n, n))


def moran_i(values, weights):
    z = np.asarray(values, float) - np.mean(values)
    denom = float(z @ z)
    if denom == 0 or weights.sum() == 0:
        return np.nan
    return float(len(z) / weights.sum() * (z @ (weights @ z)) / denom)


def spatial_diagnostics(main):
    rng = np.random.default_rng(5228)
    rows = []
    for month, part in main.groupby("month", sort=True):
        g = part.groupby("block").agg(rows=("actual", "size"), latitude=("latitude", "first"),
            longitude=("longitude", "first"), actual=("actual", "mean"),
            B=("B", "mean"), E=("E", "mean"), ABCDE_equal=("ABCDE_equal", "mean"))
        g = g[g.rows >= 2].dropna(subset=["latitude", "longitude"])
        xy = np.column_stack([(g.longitude - 103.85) * 111.2 * np.cos(np.deg2rad(1.35)),
                              (g.latitude - 1.35) * 111.2])
        matrix = spatial_weights(xy)
        strata = np.minimum(g.rows.to_numpy(), 5)
        permutations = []
        for _ in range(199):
            permutation = np.arange(len(g))
            for s in np.unique(strata):
                ids = np.flatnonzero(strata == s)
                permutation[ids] = rng.permutation(ids)
            permutations.append(permutation)
        for model in ["B", "E", "ABCDE_equal"]:
            residual = (g[model] - g.actual).to_numpy(float)
            reference = np.array([moran_i(residual[p], matrix) for p in permutations])
            rows.append({"model": model, "month": month, "blocks": len(g),
                "moran_i": moran_i(residual, matrix),
                "reference_low": float(np.nanquantile(reference, .025)),
                "reference_high": float(np.nanquantile(reference, .975))})
    return pd.DataFrame(rows)


def feature_curves(main):
    data = main.copy()
    group = ["town", "flat_type", "month"]
    for column in ["actual", "baseline", "B", "ABCDE_equal"]:
        data[f"centered_{column}"] = data[column] - data.groupby(group)[column].transform("mean")
    records = []
    for feature in FEATURES:
        values = data[feature].replace([np.inf, -np.inf], np.nan)
        valid = values.notna()
        if feature == "LOCAL_KERNEL_MASS":
            valid &= values > 0
        edges = np.unique(np.quantile(values[valid], np.linspace(0, 1, 9)))
        if len(edges) < 4:
            continue
        bins = pd.cut(values, edges, include_lowest=True, duplicates="drop")
        for interval, f in data[valid].groupby(bins[valid], observed=True):
            if len(f) < 50:
                continue
            row = {"feature": feature, "bin": str(interval), "rows": len(f),
                   "feature_mean": float(f[feature].mean())}
            for model in ["actual", "baseline", "B", "ABCDE_equal"]:
                row[model] = float(f[f"centered_{model}"].mean())
            records.append(row)
    return pd.DataFrame(records)


def variant_contrasts(main):
    rows = []
    for model in ["A", "C", "D", "E", "ABCDE_equal"]:
        for horizon, f in main.groupby("horizon", sort=True):
            rows.append({"model": model, "horizon": int(horizon), "rows": len(f),
                "mean_prediction_minus_B": float(np.mean(f[model] - f.B)),
                "rmse_minus_B": weighted_rmse(f.actual, f[model], f.weight) -
                                weighted_rmse(f.actual, f.B, f.weight)})
    return pd.DataFrame(rows)


def figures(monthly, town, interaction, moran, curves, contrasts):
    fig_month = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.1,
                              subplot_titles=["Monthly RMSE", "Monthly prediction bias"])
    for model in MODELS:
        f = monthly[monthly.model.eq(model)]
        fig_month.add_trace(go.Scatter(x=f.month, y=f.rmse_sgd, name=model, mode="lines+markers",
            line=dict(color=COLORS[model], width=3 if model == "ABCDE_equal" else 1.6), legendgroup=model), row=1, col=1)
        fig_month.add_trace(go.Scatter(x=f.month, y=f.bias_sgd, name=model, mode="lines+markers",
            line=dict(color=COLORS[model], width=3 if model == "ABCDE_equal" else 1.6),
            legendgroup=model, showlegend=False), row=2, col=1)
    fig_month.update_layout(height=650, title="A–E and equal ensemble by forecast month", hovermode="x unified")
    fig_month.update_yaxes(title_text="RMSE (S$)", row=1, col=1)
    fig_month.update_yaxes(title_text="Prediction − rent (S$)", row=2, col=1)

    towns = sorted(town.town.unique())
    months = sorted(town.month.unique())
    fig_town = make_subplots(rows=1, cols=2, subplot_titles=["Frozen B", "A–E equal ensemble"])
    for col, model in [(1, "B"), (2, "ABCDE_equal")]:
        table = town.assign(value=town[f"bias_{model}"].where(town.rows >= 30)).pivot(
            index="town", columns="month", values="value").reindex(index=towns, columns=months)
        fig_town.add_trace(go.Heatmap(z=table.to_numpy(), x=months, y=towns, zmin=-220, zmax=220,
            colorscale="RdBu", reversescale=True, colorbar=dict(title="S$", x=1.02) if col == 2 else None,
            showscale=col == 2, hovertemplate="%{y}, %{x}<br>Bias S$%{z:.1f}<extra></extra>"), row=1, col=col)
    fig_town.update_layout(height=820, title="Town × month mean error (cells with fewer than 30 rentals hidden)")
    fig_town.update_yaxes(autorange="reversed")

    fig_inter = make_subplots(rows=1, cols=3,
        subplot_titles=["Frozen B", "Nearby view E", "A–E equal ensemble"])
    for col, model in enumerate(["B", "E", "ABCDE_equal"], 1):
        part = interaction[interaction.model.eq(model)]
        table = part.assign(value=part.interaction_bias_sgd.where(part.rows >= 30)).pivot(
            index="town", columns="month", values="value").reindex(index=towns, columns=months)
        fig_inter.add_trace(go.Heatmap(z=table.to_numpy(), x=months, y=towns,
            zmin=-150, zmax=150, colorscale="RdBu", reversescale=True,
            colorbar=dict(title="S$", x=1.02) if col == 3 else None,
            showscale=col == 3,
            hovertemplate="%{y}, %{x}<br>Interaction S$%{z:.1f}<extra></extra>"), row=1, col=col)
    fig_inter.update_layout(height=820,
        title="Town × month error after removing additive town and month bias")
    fig_inter.update_yaxes(autorange="reversed")

    fig_moran = go.Figure()
    for model in ["B", "E", "ABCDE_equal"]:
        f = moran[moran.model.eq(model)]
        fig_moran.add_trace(go.Scatter(x=f.month, y=f.moran_i, name=model, mode="lines+markers",
            line=dict(color=COLORS[model])))
    ref = moran[moran.model.eq("ABCDE_equal")]
    fig_moran.add_trace(go.Scatter(x=list(ref.month) + list(ref.month.iloc[::-1]),
        y=list(ref.reference_high) + list(ref.reference_low.iloc[::-1]),
        fill="toself", fillcolor="rgba(100,116,139,.18)", line=dict(color="rgba(0,0,0,0)"),
        name="Permutation reference (ensemble)"))
    fig_moran.update_layout(height=380, title="Nearby block residual autocorrelation",
        yaxis_title="Moran’s I", hovermode="x unified")

    fig_contrast = make_subplots(rows=1, cols=2, subplot_titles=["Average forecast difference from B",
        "RMSE difference from B within each horizon"])
    for model in ["A", "C", "D", "E", "ABCDE_equal"]:
        f = contrasts[contrasts.model.eq(model)]
        for col, y in [(1, "mean_prediction_minus_B"), (2, "rmse_minus_B")]:
            fig_contrast.add_trace(go.Scatter(x=f.horizon, y=f[y], mode="lines+markers",
                name=model, legendgroup=model, showlegend=col == 1,
                line=dict(color=COLORS[model])), row=1, col=col)
    fig_contrast.update_layout(height=420, title="How the views change forecasts; differences are not isolated feature effects")
    fig_contrast.update_xaxes(title_text="Months after origin")
    fig_contrast.update_yaxes(title_text="S$", row=1, col=1)
    fig_contrast.update_yaxes(title_text="S$", row=1, col=2)

    fig_curves = make_subplots(rows=2, cols=3, subplot_titles=FEATURES, vertical_spacing=.16)
    for idx, feature in enumerate(FEATURES):
        f = curves[curves.feature.eq(feature)].sort_values("feature_mean")
        row, col = divmod(idx, 3)
        for model, color in [("actual", "#111827"), ("baseline", "#94a3b8"),
                             ("B", COLORS["B"]), ("ABCDE_equal", COLORS["ABCDE_equal"])]:
            fig_curves.add_trace(go.Scatter(x=f.feature_mean, y=f[model], mode="lines+markers",
                name=model, legendgroup=model, showlegend=idx == 0,
                customdata=np.stack([f.rows, f["bin"]], axis=1),
                hovertemplate="Feature %{x:.2f}<br>Centered rent S$%{y:.1f}<br>Rows %{customdata[0]}<br>Bin %{customdata[1]}<extra></extra>",
                line=dict(color=color, width=2.4 if model == "actual" else 1.6)), row=row + 1, col=col + 1)
    fig_curves.update_layout(height=760, title="Within town × flat type × month associations; other attributes still vary")
    fig_curves.update_yaxes(title_text="Centered rent (S$)")
    return [fig_month, fig_town, fig_inter, fig_moran, fig_contrast, fig_curves]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    predictions, main, bundles, manifests = load()
    monthly = scores(main)
    town = town_month(main)
    interaction = town_month_interaction(main)
    moran = spatial_diagnostics(main)
    curves = feature_curves(main)
    contrasts = variant_contrasts(main)
    for name, frame in [("monthly_scores", monthly), ("town_month_bias", town),
                        ("town_month_interaction", interaction),
                        ("spatial_autocorrelation", moran), ("feature_associations", curves),
                        ("variant_contrasts", contrasts)]:
        frame.to_csv(OUT / f"{name}.csv", index=False)
    components = figures(monthly, town, interaction, moran, curves, contrasts)
    paragraphs = ["<h1>TabPFN A–E: spatial, temporal and feature diagnostics</h1>",
        "<p>These figures use saved June 2024–March 2025 development forecasts. Feature curves are conditional associations, not causal effects or direct TabPFN feature attributions. Monthly horizon and target month coincide at this origin.</p>"]
    for i, fig in enumerate(components):
        paragraphs.append(pio.to_html(fig, full_html=False, include_plotlyjs="inline" if i == 0 else False))
    html = "<!doctype html><html><head><meta charset='utf-8'><title>TabPFN A–E diagnostics</title><style>body{font-family:system-ui,sans-serif;margin:2rem auto;max-width:1400px;color:#111827}p{max-width:950px;line-height:1.5}.plotly-graph-div{margin:2rem 0}</style></head><body>" + "\n".join(paragraphs) + "</body></html>"
    (OUT / "dashboard.html").write_text(html, encoding="utf-8")
    measured = {model: weighted_rmse(main.actual, main[model], main.weight) for model in MODELS}
    write = {"source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in [SOURCE, BUNDLE,
        SPATIAL_BUNDLE, TEMPORAL_BUNDLE, *manifests]},
        "rows": len(main), "origins_in_source": sorted(predictions.origin.unique()),
        "model_rmse_sgd": measured, "member_alignment_checked": True,
        "ensemble_mean_checked": True, "no_model_inference": True,
        "moran_method": "block mean residual, >=2 rentals; 8 neighbors within 2 km; 199 support-stratified permutations",
        "feature_method": "8 marginal quantile bins, centered within town/flat-type/month; descriptive",
        "town_month_interaction_method": "weighted least squares residual after additive town plus month fit"}
    (OUT / "verification.json").write_text(json.dumps(write, indent=2), encoding="utf-8")
    print(json.dumps({"rmse": measured, "moran_above_reference": moran.assign(
        above=moran.moran_i > moran.reference_high).groupby("model").above.sum().to_dict(),
        "dashboard": str(OUT / "dashboard.html")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
