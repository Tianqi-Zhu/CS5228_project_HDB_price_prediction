"""Supported continuous surfaces of existing transformer flat-rent predictions.

Predict standardized flats at real HDB block anchors, rebuild every historical
input at May 2024, then interpolate those predictions. The continuous layer is
a display smoother, not direct model evaluation on arbitrary parcels of land.
"""
from __future__ import annotations

import argparse
import base64
import gc
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import rasterio
from rasterio.transform import rowcol
from rasterio.windows import Window, from_bounds
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits
import torch

from analysis.config import DATA, ROOT, REPORTS
from analysis.data import sha256
from analysis.experiments.gpu_capacity import wait_for_gpu
from analysis.experiments.spatial_temporal_transformer import (
    SEQUENCE_CHANNELS, forecast_context, rebuild_histories,
)
from analysis.experiments.spatiotemporal_hybrid import spatial_frame
from analysis.experiments.tabpfn_adaptation_features import identifiers
from analysis.interpret_spatial_temporal_transformer import SavedPredictor, markdown_table
from analysis.run_spatial_temporal_transformer import write_json

SOURCE = REPORTS / "spatial_temporal_transformer"
CACHE = ROOT / ".cache/spatial_temporal_transformer"
OUT = SOURCE / "rent_surface"
LAND = ROOT / ".cache/tabpfn_worldcover_tree_cover_80_20/sources/ESA_WorldCover_10m_2021_v200_N00E102_Map.tif"
SCALE_X = 111.2*np.cos(np.deg2rad(1.35))


def metric_xy(longitude, latitude):
    return np.column_stack([(np.asarray(longitude)-103.85)*SCALE_X,
                            (np.asarray(latitude)-1.35)*111.2])


def interpolation_matrix(nodes, grid, bandwidth=.30, radius=1., max_nearest=.50):
    """Compact smooth normalized kernel, with explicit local support masks.

    All anchors inside radius enter; there is no fixed-k neighbor switch.
    Grid points farther than max_nearest from anchors or with effective support
    below two are not displayed. Nodes/coordinates are in kilometres.
    """
    if bandwidth <= 0 or radius <= bandwidth or max_nearest > radius:
        raise ValueError("Invalid interpolation geometry")
    tree = cKDTree(nodes)
    distance = cKDTree(grid).sparse_distance_matrix(tree, radius, output_type="coo_matrix")
    weight = np.exp(-.5*(distance.data/bandwidth)**2)*(1-(distance.data/radius)**2)**2
    matrix = coo_matrix((weight, (distance.row, distance.col)), shape=(len(grid), len(nodes))).tocsr()
    total = np.asarray(matrix.sum(axis=1)).ravel()
    total2 = np.asarray(matrix.power(2).sum(axis=1)).ravel()
    effective = np.divide(total**2, total2, out=np.zeros_like(total), where=total2 > 0)
    nearest = tree.query(grid, k=1)[0]
    supported = (nearest <= max_nearest) & (effective >= 2.)
    normalized = matrix.multiply(np.divide(1., total, out=np.zeros_like(total), where=total > 0)[:, None]).tocsr()
    return normalized, supported, nearest, effective


def land_classes(longitudes, latitudes):
    """Read a small existing WorldCover window; classes 0 and 80 are hidden."""
    with rasterio.open(LAND) as src:
        if src.crs.to_epsg() != 4326:
            raise ValueError("Unexpected land-cover CRS")
        window = from_bounds(float(longitudes.min())-.001, float(latitudes.min())-.001,
                             float(longitudes.max())+.001, float(latitudes.max())+.001, transform=src.transform)
        window = Window(int(np.floor(window.col_off)), int(np.floor(window.row_off)),
                        int(np.ceil(window.width))+2, int(np.ceil(window.height))+2)
        values = src.read(1, window=window)
        rows, cols = rowcol(src.window_transform(window), longitudes, latitudes)
        return values[np.asarray(rows), np.asarray(cols)]


def prepare_nodes(base, args):
    fitting = identifiers(spatial_frame(pd.read_csv(DATA / "train.csv"))).loc[base["X_train"].index]
    np.testing.assert_array_equal(fitting.BLOCK_ID.astype(str), base["X_train"].BLOCK_ID.astype(str))
    origin, target = pd.Period("2024-05", "M"), pd.Period(args.month, "M")
    if not 1 <= target.ordinal-origin.ordinal <= 16:
        raise ValueError("Choose a forecast 1–16 months after the May 2024 origin")
    matched = fitting[fitting.FLAT_TYPE_CLEAN.eq(args.flat_type)]
    counts = matched.groupby("BLOCK_ID", observed=True).size()
    eligible = counts[counts >= args.min_history].index
    first = matched[matched.BLOCK_ID.isin(eligible)].drop_duplicates("BLOCK_ID").sort_values("BLOCK_ID")
    if len(first) < 100:
        raise ValueError("Insufficient supported anchors for this flat type")
    node = pd.DataFrame({"block_id": first.BLOCK_ID.to_numpy(), "town": first.TOWN_CLEAN.to_numpy(),
                         "latitude": first.LATITUDE.to_numpy(), "longitude": first.LONGITUDE.to_numpy(),
                         "fitting_type_transactions": first.BLOCK_ID.map(counts).to_numpy(),
                         "actual_year_completed": first.YEAR_COMPLETED.to_numpy(),
                         "actual_lease_commencement": first.LEASE_COMMENCE_DATE.to_numpy()})
    bundles, audits = {}, {}
    for scenario in ("reference_flat", "existing_building"):
        q = first.copy()
        x = base["X_train"].loc[first.index].copy()
        q.index = x.index = pd.RangeIndex(-len(q), 0)
        q["_MONTH"], q["RENT_APPROVAL_DATE"] = target, str(target)
        q["MONTHLY_RENT"] = np.nan
        q["FLAT_TYPE"] = q["FLAT_TYPE_CLEAN"] = args.flat_type
        q["FLAT_MODEL"] = q["FLAT_MODEL_CLEAN"] = args.flat_model
        q["FLOOR_AREA_SQM"] = args.area
        if scenario == "reference_flat":
            q["LEASE_COMMENCE_DATE"] = target.year-args.age
            q["YEAR_COMPLETED"] = target.year-args.age
            q["MAX_FLOOR"] = args.max_floor
        q["FLAT_AGE_YEARS"] = target.year-q.LEASE_COMMENCE_DATE
        q["BUILDING_AGE_FROM_COMPLETION"] = target.year-q.YEAR_COMPLETED
        q["APPROVAL_MONTH_INDEX"] = (target.year-2021)*12+target.month-1
        q["APPROVAL_MONTH_OF_YEAR"] = target.month
        updated = ["FLAT_TYPE_CLEAN", "FLAT_MODEL_CLEAN", "FLOOR_AREA_SQM", "YEAR_COMPLETED", "MAX_FLOOR",
                   "FLAT_AGE_YEARS", "BUILDING_AGE_FROM_COMPLETION", "APPROVAL_MONTH_INDEX", "APPROVAL_MONTH_OF_YEAR"]
        for column in updated:
            if isinstance(x[column].dtype, pd.CategoricalDtype):
                x[column] = pd.Categorical(q[column], categories=x[column].cat.categories)
            else:
                x[column] = q[column].to_numpy()
        combined = pd.concat([fitting, q])
        origins = pd.PeriodIndex([origin]*len(q), freq="M")
        print(f"REBUILD {scenario}: {len(q)} supported real-block anchors", flush=True)
        x, baseline = rebuild_histories(x, combined, origins)
        assert np.isfinite(baseline).all()
        np.testing.assert_array_equal(x.HIST_BLOCK_FLAT_COUNT.to_numpy(), node.fitting_type_transactions.to_numpy())
        assert combined.loc[combined._MONTH <= origin, "MONTHLY_RENT"].notna().all()
        bundles[scenario] = {"x": x, "baseline": baseline,
                              "context": forecast_context([target]*len(q), origins),
                              "sequence": np.full((len(q), 12, len(SEQUENCE_CHANNELS)), np.nan, np.float32)}
        audits[scenario] = {"rows": len(q), "rental_cutoff": str(origin), "target": str(target),
                            "max_fitting_label": str(fitting._MONTH.max()),
                            "query_labels_present": False,
                            "exact_signature_history_fraction": float(x.HIST_SIGNATURE_MEAN.notna().mean()),
                            "block_type_history_fraction": float(x.HIST_BLOCK_FLAT_MEAN.notna().mean())}
    return node, bundles, audits


def plot_maps(node, surfaces, lon, lat, classes, support, args):
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    # Use rent layers only for a common comparison scale.
    rents = np.r_[surfaces["reference_mean"][support], surfaces["existing_mean"][support]]
    lo, hi = np.floor(rents.min()/100)*100, np.ceil(rents.max()/100)*100
    towns = node.groupby("town").agg(longitude=("longitude", "median"), latitude=("latitude", "median"))
    labels = ["woodlands", "yishun", "punggol", "tampines", "bedok", "queenstown", "bishan",
              "jurong west", "choa chu kang", "ang mo kio", "central"]
    for scenario, title in [("reference", "Standardized flat and building attributes"),
                             ("existing", "Same flat size/model; actual building attributes retained")]:
        fig, ax = plt.subplots(figsize=(13, 7.6), constrained_layout=True)
        background = np.where(classes == 0, 2, np.where(classes == 80, 0, 1))
        ax.imshow(background, origin="lower", extent=extent,
                   cmap=ListedColormap(["#dceaf2", "#e8e8e4", "#fafafa"]), vmin=0, vmax=2)
        image = ax.imshow(np.ma.masked_invalid(surfaces[f"{scenario}_mean"]), origin="lower", extent=extent,
                           cmap="YlOrRd", vmin=lo, vmax=hi, interpolation="nearest")
        levels = np.arange(np.ceil(lo/250)*250, hi, 250)
        contours = ax.contour(lon, lat, np.ma.masked_invalid(surfaces[f"{scenario}_mean"]),
                               levels=levels, colors="#4b3030", linewidths=.45, alpha=.6)
        ax.clabel(contours, fmt="%.0f", fontsize=7)
        for town in labels:
            if town in towns.index:
                point = towns.loc[town]
                ax.text(point.longitude, point.latitude, town.title(), ha="center", fontsize=8,
                         bbox={"facecolor": "white", "alpha": .7, "edgecolor": "none", "pad": 1.5})
        fig.colorbar(image, ax=ax, label="Predicted flat rent (SGD/month)", shrink=.85)
        ax.set(xlabel="Longitude", ylabel="Latitude", aspect="equal",
               title=f"{args.flat_type}, {args.area:g} sqm • {args.month} • existing T0 transformer\n{title}")
        fig.text(.5, -.015, "Smoothed real-block predictions; grey land = insufficient HDB support; blue = 2021 water mask. Not land valuation.",
                  ha="center", fontsize=10)
        fig.savefig(OUT / f"{scenario}_rent_heatmap.png", dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    for ax, value, title, cmap in zip(axes, [surfaces["reference_spread"], surfaces["nearest_km"]],
                                     ["Seed range: diagnostic disagreement (not a confidence interval)", "Distance to nearest supported block"],
                                     ["magma", "viridis"]):
        im = ax.imshow(np.ma.masked_invalid(value), origin="lower", extent=extent, cmap=cmap)
        ax.set(title=title, xlabel="Longitude", ylabel="Latitude", aspect="equal")
        fig.colorbar(im, ax=ax, label="SGD" if "Seed" in title else "km", shrink=.75)
    fig.savefig(OUT / "surface_support_and_disagreement.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return float(lo), float(hi)


def dashboard(node, surfaces, lon, lat, classes, args, lo, hi):
    fig = go.Figure()
    # The background mask is context only, never a model input.
    background = np.where(classes == 0, 2, np.where(classes == 80, 0, 1))
    fig.add_trace(go.Heatmap(z=background, x=lon, y=lat, zmin=0, zmax=2,
                             colorscale=[[0, "#dceaf2"], [.5, "#e8e8e4"], [1, "#fafafa"]], showscale=False, hoverinfo="skip"))
    layers = [("Reference flat: rent", "reference_mean", "SGD/month", "YlOrRd", lo, hi),
              ("Existing building: rent", "existing_mean", "SGD/month", "YlOrRd", lo, hi),
              ("Reference flat: seed disagreement", "reference_spread", "Seed range SGD", "Magma", 0, None)]
    for i, (title, key, units, colorscale, zmin, zmax) in enumerate(layers):
        fig.add_trace(go.Heatmap(z=surfaces[key], x=lon, y=lat, visible=(i == 0), colorscale=colorscale,
                                 zmin=zmin, zmax=zmax, colorbar={"title": units}, connectgaps=False,
                                 hovertemplate="Longitude %{x:.4f}<br>Latitude %{y:.4f}<br>"+units+" %{z:.0f}<extra>Smoothed supported surface</extra>"))
    custom = np.column_stack([node.town, node.block_id, node.fitting_type_transactions,
                              node.reference_mean, node.existing_mean, node.reference_spread])
    fig.add_trace(go.Scattergl(x=node.longitude, y=node.latitude, mode="markers",
                               marker={"size": 3, "color": "#273444", "opacity": .25}, name="Actual block anchors",
                               customdata=custom,
                               hovertemplate="%{customdata[0]}<br>%{customdata[1]}<br>Fitting rentals: %{customdata[2]}<br>Reference prediction: %{customdata[3]:.0f} SGD<br>Existing-building prediction: %{customdata[4]:.0f} SGD<br>Reference seed range: %{customdata[5]:.0f} SGD<extra></extra>"))
    buttons = []
    for i, (title, *_rest) in enumerate(layers):
        visible = [True]+[j == i for j in range(len(layers))]+[True]
        buttons.append({"label": title, "method": "update", "args": [{"visible": visible}]})
    fig.update_layout(template="plotly_white", height=720, margin={"l": 60, "r": 50, "t": 100, "b": 60},
                       title=f"{args.flat_type}, {args.area:g} sqm • {args.month} • smoothed T0 transformer predictions",
                       xaxis_title="Longitude", yaxis_title="Latitude", yaxis={"scaleanchor": "x", "scaleratio": 1},
                       updatemenus=[{"buttons": buttons, "x": 0, "y": 1.08, "xanchor": "left"}],
                       legend={"orientation": "h", "y": -.12})
    png = base64.b64encode((OUT / "reference_rent_heatmap.png").read_bytes()).decode("ascii")
    text = ["<!doctype html><html><head><meta charset='utf-8'><title>HDB transformer rental surface</title>",
            "<style>body{font:16px system-ui;color:#192b3e;background:#f5f7fa}main{max-width:1250px;margin:auto;padding:28px}.note{padding:18px;background:#fff0ce;border-radius:10px}p{line-height:1.6}img{max-width:100%}a{color:#225fa0}</style></head><body><main>",
            "<h1>Where does the transformer predict higher HDB rent?</h1>",
            f"<p class='note'><strong>Scenario:</strong> {args.flat_type}, {args.area:g} sqm, {args.flat_model}, approval {args.month}. The reference view fixes flat age and building age at {args.age} years and maximum floor at {args.max_floor}. The alternative retains actual building attributes. Local identities and historical rental markets vary in both views.</p>",
            "<p>Existing T0 models trained through May 2024; average of declared seeds 42/43/44. Histories stop May 2024 for every anchor. The grid interpolates predictions at supported real HDB blocks; it is not direct neural evaluation on every land parcel or a land-price estimate.</p>",
            "<p>Zoom and hover; use the dropdown to compare building assumptions or seed disagreement. Grey means insufficient comparable HDB support. Blue uses the existing 2021 WorldCover water mask. This background is not a cadastral or administrative boundary.</p>",
            pio.to_html(fig, full_html=False, include_plotlyjs=True),
            "<p><a href='findings.md'>Methods and limits</a> · <a href='block_predictions.csv'>Block predictions</a> · <a href='supported_grid.csv.gz'>Supported grid values</a></p>",
            f"<img alt='Static reference rental surface' src='data:image/png;base64,{png}'>",
            "</main></body></html>"]
    (OUT / "rent_heatmap.html").write_text("\n".join(text), encoding="utf-8")


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flat-type", default="4-room")
    parser.add_argument("--flat-model", default="model a")
    parser.add_argument("--area", type=float, default=95.)
    parser.add_argument("--age", type=int, default=25)
    parser.add_argument("--max-floor", type=int, default=12)
    parser.add_argument("--month", default="2025-03")
    parser.add_argument("--min-history", type=int, default=5)
    parser.add_argument("--grid-km", type=float, default=.1)
    parser.add_argument("--min-free-gib", type=float, default=8.)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.area <= 0 or args.age < 0 or args.min_history < 1 or args.grid_km < .05:
        raise ValueError("Invalid reference scenario or grid resolution")
    if (OUT / "verification.json").exists():
        old = json.loads((OUT / "verification.json").read_text())["settings"]
        new = {k: v for k, v in vars(args).items() if k != "output"}
        if old != new:
            raise ValueError("Scenario changed: use a different --output directory")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    original = json.loads((SOURCE / "settings.json").read_text())["hashes"]
    for path, digest in original.items():
        assert sha256(ROOT / path) == digest, path
    manifest = json.loads((SOURCE / "prepared.json").read_text())
    assert sha256(CACHE / "prepared.joblib") == manifest["sha256"]
    with threadpool_limits(limits=4):
        base = joblib.load(CACHE / "prepared.joblib")["folds"]["outer_d0"]
        node, bundles, audits = prepare_nodes(base, args)
        for seed in (42, 43, 44):
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
            wait_for_gpu(min_free_gib=args.min_free_gib, stable_samples=2, poll_seconds=15,
                         status_path=OUT / "gpu_status.json")
            predictor = SavedPredictor("T0", seed, base, "cuda")
            for scenario, q in bundles.items():
                values = predictor.predict(q["x"], q["sequence"], q["context"], q["baseline"])
                if not np.isfinite(values).all() or (values <= 0).any():
                    raise ValueError("Invalid node predictions")
                node[f"{scenario}_seed{seed}"] = values
            del predictor
            gc.collect()
        for scenario, short in [("reference_flat", "reference"), ("existing_building", "existing")]:
            values = node[[f"{scenario}_seed{s}" for s in (42, 43, 44)]].to_numpy()
            node[f"{short}_mean"] = values.mean(axis=1)
            node[f"{short}_spread"] = values.max(axis=1)-values.min(axis=1)
        node.to_csv(OUT / "block_predictions.csv", index=False)
        longitude = np.arange(103.60, 104.10+args.grid_km/SCALE_X, args.grid_km/SCALE_X)
        latitude = np.arange(1.20, 1.48+args.grid_km/111.2, args.grid_km/111.2)
        lons, lats = np.meshgrid(longitude, latitude)
        grid = metric_xy(lons.ravel(), lats.ravel())
        matrix, supported, nearest, effective = interpolation_matrix(metric_xy(node.longitude, node.latitude), grid)
        classes = land_classes(lons.ravel(), lats.ravel()).reshape(lons.shape)
        supported &= ~np.isin(classes.ravel(), [0, 80])
        mask = supported.reshape(lons.shape)
        surfaces = {}
        for scenario, short in [("reference_flat", "reference"), ("existing_building", "existing")]:
            values = node[[f"{scenario}_seed{s}" for s in (42, 43, 44)]].to_numpy()
            predictions = matrix@values
            surfaces[f"{short}_mean"] = np.where(supported, predictions.mean(axis=1), np.nan).reshape(lons.shape)
            surfaces[f"{short}_spread"] = np.where(supported, np.ptp(predictions, axis=1), np.nan).reshape(lons.shape)
        surfaces["nearest_km"] = np.where(supported, nearest, np.nan).reshape(lons.shape)
        np.savez_compressed(OUT / "surface_grid.npz", longitude=longitude, latitude=latitude,
                            landcover=classes, supported=mask, **surfaces)
        pd.DataFrame({"longitude": lons.ravel()[supported], "latitude": lats.ravel()[supported],
                      "reference_rent_sgd": surfaces["reference_mean"].ravel()[supported],
                      "existing_building_rent_sgd": surfaces["existing_mean"].ravel()[supported],
                      "reference_seed_range_sgd": surfaces["reference_spread"].ravel()[supported],
                      "nearest_anchor_km": nearest[supported], "effective_anchors": effective[supported]}).to_csv(
                          OUT / "supported_grid.csv.gz", index=False, compression="gzip")
        lo, hi = plot_maps(node, surfaces, longitude, latitude, classes, mask, args)
        dashboard(node, surfaces, longitude, latitude, classes, args, lo, hi)
        town = node.groupby("town").agg(blocks=("reference_mean", "size"),
                                        reference_mean_sgd=("reference_mean", "mean"),
                                        existing_building_mean_sgd=("existing_mean", "mean"),
                                        seed_range_mean_sgd=("reference_spread", "mean")).reset_index()
        town.to_csv(OUT / "town_summary.csv", index=False)
        ranked = town[town.blocks >= 20].sort_values("reference_mean_sgd", ascending=False)
        lines = ["# Continuous HDB rental-expensiveness surface", "",
                 "[Open the interactive heatmap](rent_heatmap.html).", "",
                 f"Reference: {args.flat_type}, {args.area:g} sqm, {args.flat_model}, {args.age}-year flat/building age, maximum floor {args.max_floor}, approval {args.month}. Alternative: same type/size/model, actual lease/completion year and maximum floor retained.",
                 f"Existing T0 transformers, declared seeds 42/43/44; average predictions at {len(node):,} real block anchors. Each has at least {args.min_history} same-type labeled fitting transactions. All labels and histories end May 2024. No 2024-06 or later rent labels enter this map. No refitting.", "",
                 "## Meaning of the continuous surface", "",
                 "These are predicted monthly rents for hypothetical flats at supported existing HDB locations. They are not parcel/land prices, private-property rents or a causal location premium. Building identities and local rental histories vary even in the standardized view, so some unmeasured building characteristics can remain.",
                 "Every one of the 46 target-derived features and the external baseline is recomputed for the reference property and common cutoff. Exact-property signature histories may be missing; the trained preprocessor handles them. Available block/type histories and local forecast baselines remain. No arbitrary coordinate edits with stale geographic attributes are used.",
                 "The grid is a normalized compact Gaussian smoother over real-block model predictions, bandwidth 300 m, radius 1 km, with a smooth taper to zero at the boundary. Display requires an anchor within 500 m and effective support at least two. All anchors inside the kernel radius contribute. Values between blocks are interpolation, not direct transformer predictions; smoothing can hide block-level discontinuities. The 100 m default grid is rendering resolution, not prediction precision.",
                 "The existing 2021 ESA WorldCover raster masks nodata and permanent water (class 80), only for display. It is not a current coastline, administrative border or construction-eligibility layer. Grey land includes unsupported locations, forests and industrial/private-property areas; no rent is inferred there. [WorldCover legend](https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/docs/WorldCover_PUM_V2.0.pdf).", "",
                 "## Supported-town averages of the standardized scenario", "", markdown_table(ranked), "",
                 "Town summaries average supported block predictions, not population-weighted observed rents. They are conditional model scenarios; do not interpret the ordering as a causal town premium or a price for every spot in that town.", "",
                 "## Limits and verification", "",
                 "Seed range describes optimization disagreement only; it is not a calibrated forecast interval. The selected T0 architecture has the best primary mean validation score among these transformer variants, but its development period has been repeatedly inspected. The underlying forecast baseline plateaus after horizon ten. Static geography/amenity snapshots and land cover retain their original vintage limitations.",
                 "Original model weights, inputs and Kaggle submissions remain unchanged. Source and node artifacts, scenario cutoffs and interpolation settings are recorded in verification.json. Reproduce using `.venv/Scripts/python.exe analysis/plot_transformer_rent_surface.py`; use a new --output directory for a different scenario.", ""]
        (OUT / "findings.md").write_text("\n".join(lines), encoding="utf-8")
        for path, digest in original.items():
            assert sha256(ROOT / path) == digest, path
        settings = {k: v for k, v in vars(args).items() if k != "output"}
        write_json(OUT / "verification.json", {"settings": settings, "script_sha256": sha256(Path(__file__)),
                   "prepared_sha256": manifest["sha256"], "landcover_sha256": sha256(LAND),
                   "landcover_display_only": True, "query_audits": audits, "anchors": len(node),
                   "grid_shape": list(lons.shape), "supported_land_cells": int(supported.sum()),
                   "nearest_anchor_limit_km": .5, "minimum_effective_anchors": 2., "bandwidth_km": .3,
                   "kernel_radius_km": 1., "original_hashes_unchanged": True,
                   "node_predictions_sha256": sha256(OUT / "block_predictions.csv"),
                   "no_refit": True, "no_post_origin_labels_used": True})
        print("SURFACE COMPLETE " + str(OUT / "rent_heatmap.html"), flush=True)


if __name__ == "__main__":
    main()
