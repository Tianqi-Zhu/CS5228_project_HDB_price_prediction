"""Reproducible spatial/temporal diagnostics of saved models; no refitting.

Inference passes the original GPU capacity gate. Original experiments and
submissions are read-only. Figures describe prediction associations/reliance,
not causal effects of interventions on housing or economic conditions.
"""
from __future__ import annotations

import argparse
import base64
import gc
import html
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits
import torch

from analysis.config import ROOT, REPORTS
from analysis.data import sha256
from analysis.experiments.gpu_capacity import wait_for_gpu
from analysis.experiments.spatial_temporal_transformer import (
    EXPERIMENTS, RENT_CHANNELS, SEQUENCE_CHANNELS, RentalTransformer,
)
from analysis.run_spatial_temporal_transformer import arrays, weighted_rmse, write_json

SOURCE = REPORTS / "spatial_temporal_transformer"
CACHE = ROOT / ".cache/spatial_temporal_transformer"
OUT = SOURCE / "interpretation"
COLORS = {"T0": "#315c9b", "T1": "#18a999", "T2": "#d97732", "T3": "#8054a6"}
LABELS = {"T0": "Property/history features", "T1": "+ Rent sequence",
          "T2": "+ Rent and COE sequence", "T3": "+ Rent, COE and stocks"}


def sample_months(months, per_month=200, seed=2026):
    """Label-independent equal-month sample, with inverse-inclusion weights."""
    rng = np.random.default_rng(seed)
    positions, weights = [], []
    months = np.asarray(months)
    for month in sorted(set(months)):
        ids = np.flatnonzero(months == month)
        selected = np.sort(rng.choice(ids, min(len(ids), per_month), replace=False))
        positions.extend(selected)
        weights.extend([len(ids)/len(selected)]*len(selected))
    return np.asarray(positions, int), np.asarray(weights, float)


def donor_permutation(months, types, seed):
    """Shuffle whole groups within target month and flat type, retaining cohorts."""
    rng = np.random.default_rng(seed)
    keys = pd.DataFrame({"month": months, "type": np.asarray(types).astype(str)})
    permutation = np.arange(len(keys))
    for ids in keys.groupby(["month", "type"], sort=True).indices.values():
        permutation[ids] = rng.permutation(ids)
    return permutation


def ale_1d(frame, column, predict, bins=6, min_bin=20):
    """Quantile-bin first-order ALE, centered by observed bin population.

    Only one raw input is changed between each bin's adjacent endpoints. Its
    historical summaries and external baseline are intentionally held fixed.
    """
    values = frame[column].to_numpy(float)
    edges = np.unique(np.quantile(values[np.isfinite(values)], np.linspace(0, 1, bins+1)))
    if len(edges) < 3:
        return pd.DataFrame()
    # Tied/discrete areas can leave a nominal quantile bin nearly empty. Merge
    # adjacent bins before evaluating effects, never bridge an unsupported gap.
    while True:
        membership = np.clip(np.searchsorted(edges, values, side="right")-1, 0, len(edges)-2)
        counts = np.bincount(membership[np.isfinite(values)], minlength=len(edges)-1)
        sparse = np.flatnonzero(counts < min_bin)
        if not len(sparse):
            break
        if len(edges) <= 3:
            return pd.DataFrame()
        index = int(sparse[0])
        edges = np.delete(edges, index if index == len(edges)-2 else index+1)
    records = []
    for index in range(len(edges)-1):
        ids = np.flatnonzero((membership == index) & np.isfinite(values))
        if len(ids) < min_bin:
            continue
        lower, upper = frame.iloc[ids].copy(), frame.iloc[ids].copy()
        lower[column], upper[column] = edges[index], edges[index+1]
        difference = np.asarray(predict(upper))-np.asarray(predict(lower))
        records.append({"bin": index, "low": edges[index], "high": edges[index+1],
                        "midpoint": (edges[index]+edges[index+1])/2, "n": len(ids),
                        "local_difference_sgd": float(difference.mean())})
    result = pd.DataFrame(records)
    if result.empty:
        return result
    # Skipping a low-support interior bin makes accumulation across it invalid.
    if len(result) != len(edges)-1:
        return pd.DataFrame()
    uncentered = result.local_difference_sgd.cumsum()-result.local_difference_sgd/2
    result["ale_sgd"] = uncentered-np.average(uncentered, weights=result.n)
    return result


def spatial_weights(xy, neighbors=8, radius_km=2.):
    n = len(xy)
    if n < 3:
        return csr_matrix((n, n))
    distance, neighbor = cKDTree(xy).query(xy, k=min(neighbors+1, n))
    rows, cols, data = [], [], []
    for i in range(n):
        keep = (neighbor[i] != i) & (distance[i] <= radius_km)
        ids, d = neighbor[i][keep], distance[i][keep]
        if len(ids):
            weight = np.exp(-.5*d*d)
            weight /= weight.sum()
            rows.extend([i]*len(ids)); cols.extend(ids); data.extend(weight)
    return csr_matrix((data, (rows, cols)), shape=(n, n))


def moran_i(values, weights):
    z = np.asarray(values, float)-np.mean(values)
    denominator = float(z@z)
    if denominator == 0 or weights.sum() == 0:
        return float("nan")
    return float(len(z)/weights.sum()*(z@(weights@z))/denominator)


def save_plot(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def markdown_table(frame, digits=3):
    if frame.empty:
        return "No completed rows."
    rows = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"]*len(frame.columns)) + " |"]
    rows.extend("| " + " | ".join(f"{v:.{digits}f}" if isinstance(v, float) else str(v) for v in row) + " |"
                for row in frame.itertuples(index=False, name=None))
    return "\n".join(rows)


def load_predictions(bundle):
    frame = bundle["X_query"]
    result = pd.DataFrame({"row_id": frame.index, "month": bundle["query_month"],
                           "town": frame.TOWN_CLEAN.astype(str).to_numpy(),
                           "block": frame.BLOCK_ID.astype(str).to_numpy(),
                           "flat_type": frame.FLAT_TYPE_CLEAN.astype(str).to_numpy(),
                           "floor_area": frame.FLOOR_AREA_SQM.to_numpy(),
                           "latitude": frame.LATITUDE.to_numpy(), "longitude": frame.LONGITUDE.to_numpy(),
                           "actual": bundle["y_query"]})
    for variant in COLORS:
        for seed in (42, 43, 44):
            p = pd.read_csv(SOURCE / "outer" / variant / f"seed{seed}" / "predictions.csv")
            np.testing.assert_array_equal(result.row_id, p.row_id)
            np.testing.assert_array_equal(result.actual, p.actual)
            result[f"{variant}_{seed}"] = p.predicted.to_numpy()
        result[f"residual_{variant}"] = result[f"{variant}_42"]-result.actual
    return result


def descriptive_figures(frame):
    scores = pd.read_csv(SOURCE / "scores.csv")
    primary = scores[scores.variant.isin(COLORS)]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for x, variant in enumerate(COLORS):
        values = primary[primary.variant.eq(variant)].rmse_sgd.to_numpy()
        ax.scatter(x+np.array([-.08, 0, .08]), values, color=COLORS[variant], s=55)
        ax.plot([x-.22, x+.22], [values.mean()]*2, color=COLORS[variant], linewidth=3)
    ax.set(xticks=range(4), xticklabels=["T0: feature model", "T1: + rent", "T2: + COE", "T3: + stocks"],
           ylabel="Individual-rental RMSE (SGD)", title="Sequence inputs did not improve consistently across three seeds")
    ax.grid(axis="y", alpha=.2)
    save_plot(fig, "01_seed_stability")

    monthly = frame.groupby("month").mean(numeric_only=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(monthly.index, monthly.actual, color="#222222", linewidth=2.5, label="Observed mean")
    for variant in COLORS:
        prediction = monthly[[f"{variant}_{seed}" for seed in (42, 43, 44)]]
        axes[0].plot(monthly.index, prediction.mean(axis=1), color=COLORS[variant], label=variant)
        bias = prediction.subtract(monthly.actual, axis=0)
        axes[1].plot(monthly.index, bias.mean(axis=1), color=COLORS[variant], label=variant)
        axes[1].fill_between(np.arange(len(bias)), bias.min(axis=1), bias.max(axis=1), color=COLORS[variant], alpha=.13)
    axes[0].set(ylabel="Monthly mean rent (SGD)", title="Observed and predicted means: changing property mix remains")
    axes[1].axhline(0, color="#333333", linewidth=1)
    axes[1].set(ylabel="Prediction minus observed (SGD)", title="Bias; shaded area is seed range, not a confidence interval")
    axes[0].legend(ncol=5); axes[1].tick_params(axis="x", rotation=35)
    fig.tight_layout()
    save_plot(fig, "02_temporal_bias")

    blocks = frame.groupby("block").agg(n=("actual", "size"), actual=("actual", "mean"),
                                           latitude=("latitude", "first"), longitude=("longitude", "first"),
                                           town=("town", "first"), T0=("residual_T0", "mean"),
                                           T2=("residual_T2", "mean")).reset_index()
    blocks = blocks[blocks.n >= 5]
    blocks.to_csv(OUT / "block_diagnostics.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for ax, column, title in zip(axes, ["actual", "T0", "T2"],
                                 ["Observed mean rent", "T0 mean residual", "T2 mean residual"]):
        options = {"cmap": "viridis"} if column == "actual" else {
            "cmap": "RdBu_r", "norm": TwoSlopeNorm(vmin=-400, vcenter=0, vmax=400)}
        scatter = ax.scatter(blocks.longitude, blocks.latitude, c=blocks[column],
                             s=7+3*np.log1p(blocks.n), alpha=.8, **options)
        ax.set(title=title, xlabel="Longitude", ylabel="Latitude", aspect="equal")
        ax.set_xticks([103.7, 103.8, 103.9, 104.0])
        ax.tick_params(axis="both", labelsize=9)
        fig.colorbar(scatter, ax=ax, label="SGD", shrink=.65)
    fig.suptitle("HDB blocks with at least five holdout rentals; seed 42; residual = prediction - observed")
    save_plot(fig, "03_spatial_residuals")

    town_month = frame.groupby(["town", "month"]).agg(n=("actual", "size"), T0=("residual_T0", "mean"),
                                                       T2=("residual_T2", "mean")).reset_index()
    town_month.to_csv(OUT / "town_month_diagnostics.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 10), sharey=True, constrained_layout=True)
    for ax, variant in zip(axes, ["T0", "T2"]):
        table = town_month.assign(value=town_month[variant].where(town_month.n >= 30)).pivot(
            index="town", columns="month", values="value")
        im = ax.imshow(table.to_numpy(), cmap="RdBu_r", vmin=-220, vmax=220, aspect="auto")
        ax.set(xticks=range(len(table.columns)), xticklabels=table.columns,
               yticks=range(len(table)), yticklabels=table.index, title=f"{variant}: town × month mean residual")
        ax.tick_params(axis="x", rotation=55)
    fig.colorbar(im, ax=axes, label="Prediction - observed (SGD)", shrink=.7)
    fig.suptitle("Cells with fewer than 30 rentals masked; seed 42")
    save_plot(fig, "04_town_month_errors")

    records = []
    rng = np.random.default_rng(2026)
    for variant in ["T0", "T2"]:
        for month, part in frame.groupby("month"):
            cell = part.groupby("block").agg(n=("actual", "size"), latitude=("latitude", "first"),
                                             longitude=("longitude", "first"),
                                             residual=(f"residual_{variant}", "mean"))
            cell = cell[cell.n >= 2]
            xy = np.column_stack([(cell.longitude-103.85)*111.2*np.cos(np.deg2rad(1.35)),
                                  (cell.latitude-1.35)*111.2])
            weights = spatial_weights(xy)
            actual = moran_i(cell.residual, weights)
            values = cell.residual.to_numpy()
            strata = np.minimum(cell.n.to_numpy(), 5)
            reference = []
            for _ in range(199):
                perm = np.arange(len(cell))
                for group in np.unique(strata):
                    ids = np.flatnonzero(strata == group)
                    perm[ids] = rng.permutation(ids)
                reference.append(moran_i(values[perm], weights))
            lo, hi = np.nanquantile(reference, [.025, .975])
            records.append({"variant": variant, "month": month, "blocks": len(cell),
                            "moran_i": actual, "permutation_low": lo, "permutation_high": hi,
                            "permutation_mean": float(np.nanmean(reference))})
    moran = pd.DataFrame(records)
    moran.to_csv(OUT / "spatial_autocorrelation.csv", index=False)
    fig, ax = plt.subplots(figsize=(11, 4.7))
    for variant in ["T0", "T2"]:
        part = moran[moran.variant.eq(variant)]
        x = np.arange(len(part))
        ax.plot(x, part.moran_i, marker="o", color=COLORS[variant], label=variant)
        ax.fill_between(x, part.permutation_low, part.permutation_high, color=COLORS[variant], alpha=.15)
    ax.axhline(0, color="black", linewidth=.8)
    ax.set(xticks=range(10), xticklabels=sorted(frame.month.unique()), ylabel="Moran's I of block-mean residual",
           title="Spatially clustered errors; shading is a support-stratified permutation benchmark")
    ax.tick_params(axis="x", rotation=35); ax.legend(); ax.grid(alpha=.15)
    save_plot(fig, "05_spatial_autocorrelation")
    return blocks, town_month, moran


class SavedPredictor:
    def __init__(self, variant, seed, bundle, device):
        self.variant, self.seed, self.bundle = variant, seed, bundle
        self.device = torch.device(device)
        cache = CACHE / "outer" / variant / f"seed{seed}"
        report = SOURCE / "outer" / variant / f"seed{seed}"
        result = json.loads((report / "result.json").read_text())
        assert sha256(cache / "model.pt") == result["checkpoint_sha256"]
        assert sha256(cache / "preprocessors.joblib") == result["preprocessors_sha256"]
        self.pre, self.seq_pre = joblib.load(cache / "preprocessors.joblib")
        self.spec = EXPERIMENTS[variant]
        payload = torch.load(cache / "model.pt", map_location="cpu", weights_only=True)
        self.mean, self.sd = payload["mean"], payload["sd"]
        self.model = RentalTransformer(len(self.pre.numerical), self.pre.cardinalities, self.spec).to(self.device)
        self.model.load_state_dict(payload["model"])
        self.model.eval()

    @torch.inference_mode()
    def predict(self, frame, sequence, context, baseline, *, components=False):
        numeric = self.pre.transform(frame)
        seq = self.seq_pre.transform(sequence[:, -self.spec.window:], self.spec.information)
        inputs = (*numeric, seq, context)
        output, row_output = [], []
        captured = []
        hook = self.model.property_head.register_forward_hook(
            lambda _m, _i, value: captured.append(value.detach().float().cpu().numpy().reshape(-1))) if components else None
        try:
            for start in range(0, len(frame), 512):
                x = [torch.as_tensor(v[start:start+512], device=self.device) for v in inputs]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                    pred = self.model(*x)
                output.append(pred.float().cpu().numpy().astype(float))
                if components:
                    row_output.append(captured.pop())
        finally:
            if hook is not None:
                hook.remove()
        standardized = np.concatenate(output)
        prediction = np.asarray(baseline)+self.mean+self.sd*standardized
        if components:
            row = self.sd*np.concatenate(row_output).astype(float)
            # Difference includes BF16 addition rounding and exactly reconstructs output.
            temporal = self.sd*standardized-row
            return prediction, row, temporal
        return prediction


def feature_groups(columns):
    geographic = [c for c in columns if c in {"TOWN_CLEAN", "LATITUDE", "LONGITUDE", "SUBZONE", "PLANNING_AREA",
                                             "REGION", "BLOCK_ID", "STREET_ID", "DISTANCE_TO_CBD_KM", "AMENITY_DENSITY_1KM"}
                  or c.startswith(("NEAREST_", "MRT_COUNT_", "MALL_COUNT_", "SCHOOL_COUNT_"))]
    history = [c for c in columns if c.startswith(("HIST_", "RECENT_", "AREA_RECENT_", "BLOCK_RECENT_",
                                                  "BLOCK_MINUS_", "AREA_LAST_", "TEMP_"))
               and c != "TEMP_FORECAST_HORIZON"]
    prop = [c for c in columns if c in {"FLAT_TYPE_CLEAN", "FLAT_MODEL_CLEAN", "FLOOR_AREA_SQM", "FLAT_AGE_YEARS",
                                       "MAX_FLOOR", "YEAR_COMPLETED", "BUILDING_AGE_FROM_COMPLETION"}]
    return {"Property attributes": prop, "Geography + amenities": geographic,
            "Rental-history bundle": history,
            "Geography + rental history": list(dict.fromkeys(geographic+history))}


def area_effects(predictor, bundle, positions):
    frame = bundle["X_query"].iloc[positions]
    records = []
    for kind in ("3-room", "4-room", "5-room"):
        local = frame[frame.FLAT_TYPE_CLEAN.astype(str).eq(kind)]
        if len(local) < 150:
            continue
        def call(x):
            ids = bundle["X_query"].index.get_indexer(x.index)
            return predictor.predict(x, bundle["query_sequence"][ids], bundle["query_context"][ids],
                                     bundle["query_baseline"][ids])
        table = ale_1d(local, "FLOOR_AREA_SQM", call)
        if not table.empty:
            table["variant"], table["seed"], table["flat_type"] = predictor.variant, predictor.seed, kind
            records.extend(table.to_dict("records"))
    return records


def interpret_model(predictor, bundle, positions, sample_weights):
    variant, seed = predictor.variant, predictor.seed
    frame = bundle["X_query"].iloc[positions].copy()
    sequence = bundle["query_sequence"][positions].copy()
    context = bundle["query_context"][positions]
    baseline = bundle["query_baseline"][positions]
    actual = bundle["y_query"][positions]
    months = bundle["query_month"][positions]
    predict = lambda x, seq=sequence, base=baseline: predictor.predict(x, seq, context, base)
    original, row, temporal = predictor.predict(frame, sequence, context, baseline, components=True)
    original_rmse = weighted_rmse(actual, original, sample_weights)
    decomposition = pd.DataFrame({"row_id": frame.index, "variant": variant, "seed": seed,
                                   "month": months, "baseline_sgd": baseline,
                                   "residual_intercept_sgd": predictor.mean, "property_component_sgd": row,
                                   "sequence_component_sgd": temporal, "predicted": original, "actual": actual,
                                   "sample_weight": sample_weights})
    np.testing.assert_allclose(decomposition.baseline_sgd+decomposition.residual_intercept_sgd+
                               decomposition.property_component_sgd+decomposition.sequence_component_sgd,
                               decomposition.predicted, atol=1e-8, rtol=0)
    reliance, ale, occlusion, lags = [], [], [], []
    if variant in ("T0", "T2"):
        for group, columns in feature_groups(frame.columns).items():
            for repeat in range(3):
                donors = donor_permutation(months, frame.FLAT_TYPE_CLEAN, 100+repeat)
                changed = frame.copy()
                changed.loc[:, columns] = frame.iloc[donors][columns].to_numpy()
                changed_base = baseline[donors] if "TEMP_LOCAL_FORECAST" in columns else baseline
                changed_sequence = sequence[donors] if "Rental" in group or "rental history" in group else sequence
                prediction = predictor.predict(changed, changed_sequence, context, changed_base)
                reliance.append({"variant": variant, "seed": seed, "group": group, "repeat": repeat,
                                 "original_sample_rmse_sgd": original_rmse,
                                 "delta_rmse_sgd": weighted_rmse(actual, prediction, sample_weights)-original_rmse,
                                 "mean_absolute_prediction_change_sgd": float(np.average(abs(prediction-original), weights=sample_weights))})
        ale.extend(area_effects(predictor, bundle, positions))
    if variant != "T0":
        masks = {"Disable entire sequence branch": None,
                 "Remove rent channels": list(range(len(RENT_CHANNELS))),
                 "Remove COE channels": list(range(len(RENT_CHANNELS), len(RENT_CHANNELS)+4))}
        if variant == "T3":
            masks["Remove stock channel"] = [len(SEQUENCE_CHANNELS)-1]
        for label, channels in masks.items():
            if channels is None:
                changed = original-temporal
            else:
                modified = sequence.copy()
                modified[:, :, channels] = np.nan
                changed = predict(frame, seq=modified)
            occlusion.append({"variant": variant, "seed": seed, "perturbation": label,
                              "delta_rmse_sgd": weighted_rmse(actual, changed, sample_weights)-original_rmse,
                              "mean_prediction_change_sgd": float(np.average(changed-original, weights=sample_weights)),
                              "mean_absolute_prediction_change_sgd": float(np.average(abs(changed-original), weights=sample_weights))})
        for lag in range(12):
            changed_sequence = sequence.copy()
            changed_sequence[:, -(lag+1), :] = np.nan
            changed = predict(frame, seq=changed_sequence)
            lags.append({"variant": variant, "seed": seed, "lag": lag,
                         "delta_rmse_sgd": weighted_rmse(actual, changed, sample_weights)-original_rmse,
                         "mean_absolute_prediction_change_sgd": float(np.average(abs(changed-original), weights=sample_weights))})
    # All query macro paths coincide at a fixed origin; row permutation cannot
    # identify importance for them. Save their observed variation explicitly.
    macro = sequence[:, :, len(RENT_CHANNELS):]
    maximum_macro_variation = float(np.nanmax(np.nanstd(macro.astype(float), axis=0)))
    summary = {"variant": variant, "seed": seed, "sample_rmse_sgd": original_rmse,
               "sequence_gate": float(predictor.model.gate.sigmoid().detach()) if variant != "T0" else 0.,
               "mean_sequence_component_sgd": float(np.average(temporal, weights=sample_weights)),
               "mean_abs_sequence_component_sgd": float(np.average(abs(temporal), weights=sample_weights)),
               "max_macro_across_row_sd": maximum_macro_variation}
    return decomposition, reliance, ale, occlusion, lags, summary


def observed_associations(bundle, frame):
    """Descriptive feature relationships, with no hypothetical input edits."""
    four = frame.flat_type.eq("4-room").to_numpy()
    data = bundle["X_query"].iloc[np.flatnonzero(four)]
    observed = frame.loc[four].reset_index(drop=True)
    features = {"FLOOR_AREA_SQM": "Floor area (sqm)", "FLAT_AGE_YEARS": "Flat age (years)",
                "NEAREST_MRT_KM": "Nearest MRT distance (km)", "DISTANCE_TO_CBD_KM": "CBD distance (km)"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    records = []
    for ax, (column, label) in zip(axes.flat, features.items()):
        value = data[column].to_numpy(float)
        edges = np.unique(np.nanquantile(value, np.linspace(0, 1, 9)))
        membership = np.clip(np.searchsorted(edges, value, side="right")-1, 0, len(edges)-2)
        subset = []
        for index in range(len(edges)-1):
            ids = np.flatnonzero((membership == index) & np.isfinite(value))
            if len(ids) < 40:
                continue
            part = observed.iloc[ids]
            subset.append({"feature": column, "bin": index, "x": float(value[ids].mean()), "n": len(ids),
                           "observed_mean": float(part.actual.mean()), "T0": float(part.T0_42.mean()),
                           "T2": float(part.T2_42.mean())})
        table = pd.DataFrame(subset)
        records.extend(subset)
        ax.plot(table.x, table.observed_mean, color="#222222", marker="o", label="Observed")
        for variant in ("T0", "T2"):
            ax.plot(table.x, table[variant], color=COLORS[variant], marker="o", label=variant)
        ax.set(xlabel=label, ylabel="Mean rent (SGD)")
        ax.grid(alpha=.15); ax.legend()
    fig.suptitle("Observed feature associations among 4-room rentals; geography, time and other attributes still vary")
    save_plot(fig, "10_observed_feature_associations")
    pd.DataFrame(records).to_csv(OUT / "observed_feature_associations.csv", index=False)


def model_figures(reliance, ale, occlusion, lags, decomposition):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, sharex=True, constrained_layout=True)
    for ax, variant in zip(axes, ["T0", "T2"]):
        table = reliance[reliance.variant.eq(variant)].groupby(["group", "seed"]).delta_rmse_sgd.mean().unstack("seed")
        ax.barh(table.index, table.mean(axis=1), color=COLORS[variant], alpha=.65)
        for seed in table:
            ax.scatter(table[seed], table.index, color="#222222", s=24)
        ax.axvline(0, color="black", linewidth=.8)
        ax.set(title=variant, xlabel="Increase in sampled RMSE after grouped shuffle (SGD)")
    fig.suptitle("Prediction-pipeline reliance: cohort shuffles; overlapping groups cannot be added")
    save_plot(fig, "06_grouped_feature_reliance")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, kind in zip(axes, ["3-room", "4-room", "5-room"]):
        for variant in ["T0", "T2"]:
            part = ale[ale.flat_type.eq(kind) & ale.variant.eq(variant)]
            if part.empty:
                continue
            table = part.groupby("bin").agg(x=("midpoint", "mean"), effect=("ale_sgd", "mean"),
                                              low=("ale_sgd", "min"), high=("ale_sgd", "max"))
            ax.plot(table.x, table.effect, color=COLORS[variant], marker="o", label=variant)
            ax.fill_between(table.x, table.low, table.high, color=COLORS[variant], alpha=.15)
        ax.axhline(0, color="black", linewidth=.8)
        ax.set(title=kind, xlabel="Floor area (sqm)", ylabel="Centered model response (SGD)")
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
        else:
            ax.text(.5, .5, "Insufficient supported bins", ha="center", transform=ax.transAxes)
    fig.suptitle("Floor-area ALE: direct neural response; baseline/history held fixed; shading is seed range")
    save_plot(fig, "07_floor_area_response")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for variant in ["T1", "T2", "T3"]:
        part = occlusion[occlusion.variant.eq(variant)]
        table = part.groupby("perturbation").delta_rmse_sgd.agg(["mean", "min", "max"])
        labels = list(table.index)
        offset = {"T1": -.2, "T2": 0, "T3": .2}[variant]
        all_labels = ["Disable entire sequence branch", "Remove rent channels", "Remove COE channels", "Remove stock channel"]
        y = np.asarray([all_labels.index(v) for v in labels])+offset
        axes[0].errorbar(table["mean"], y, xerr=[table["mean"]-table["min"], table["max"]-table["mean"]],
                         fmt="o", color=COLORS[variant], label=variant)
        lag = lags[lags.variant.eq(variant)].groupby("lag").mean_absolute_prediction_change_sgd.agg(["mean", "min", "max"])
        axes[1].plot(lag.index, lag["mean"], marker="o", color=COLORS[variant], label=variant)
        axes[1].fill_between(lag.index, lag["min"], lag["max"], color=COLORS[variant], alpha=.15)
    axes[0].set(yticks=range(4), yticklabels=all_labels, xlabel="Occluded minus original sampled RMSE (SGD)",
                title="Frozen-model removal sensitivity; range across seeds")
    axes[0].axvline(0, color="black", linewidth=.8)
    axes[1].set(xlabel="Lag slot before rental cutoff (0 = latest)", ylabel="Mean absolute prediction change (SGD)",
                title="Mask one complete historical token; no refitting")
    axes[0].legend(); axes[1].legend()
    save_plot(fig, "08_sequence_and_lag_reliance")

    chosen = decomposition[decomposition.variant.eq("T2") & decomposition.seed.eq(42)].copy()
    chosen = chosen.merge(pd.read_csv(OUT / "sample.csv")[["row_id", "floor_area", "flat_type"]], on="row_id")
    candidates = []
    for month in ["2024-06", "2024-10", "2025-03"]:
        part = chosen[chosen.month.eq(month) & chosen.flat_type.eq("4-room")].copy()
        part["distance"] = abs(part.floor_area-part.floor_area.median())
        candidates.append(part.sort_values(["distance", "row_id"]).iloc[0])
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
    components = ["baseline_sgd", "residual_intercept_sgd", "property_component_sgd", "sequence_component_sgd"]
    for ax, case in zip(axes, candidates):
        values = case[components].to_numpy(float)
        bottom = np.r_[0, np.cumsum(values)[:-1]]
        for i, (value, base) in enumerate(zip(values, bottom)):
            ax.bar(i, value, bottom=base, color="#4677a9" if value >= 0 else "#ca6c4b")
            ax.text(i, base+value/2, f"{value:+.0f}", ha="center", va="center", fontsize=9)
        ax.axhline(case.actual, color="black", linestyle="--", label=f"Observed {case.actual:.0f}")
        ax.set_ylim(0, 1.12*max(float(np.cumsum(values).max()), float(case.actual)))
        ax.set(xticks=range(4), xticklabels=["Baseline", "Residual\nintercept", "Property\nnetwork", "Sequence\nnetwork"],
               ylabel="SGD", title=f"{case.month}, row {int(case.row_id)}\nPredicted {case.predicted:.0f}")
        ax.legend(fontsize=8)
    fig.suptitle("Exact T2 seed-42 prediction decomposition; examples selected by month/size, not error")
    save_plot(fig, "09_prediction_components")


def write_dashboard(frame, blocks, town_month, reliance, ale, summaries, moran):
    figures = []
    fig = go.Figure()
    for variant in COLORS:
        for seed in (42, 43, 44):
            part = pd.read_csv(SOURCE / "outer" / variant / f"seed{seed}" / "predictions.csv")
            residual = part.assign(residual=part.predicted-part.actual).groupby("month").residual.mean()
            fig.add_trace(go.Scatter(x=residual.index, y=residual, name=f"{variant}, seed {seed}",
                                    line={"color": COLORS[variant]}, legendgroup=variant))
    fig.update_layout(title="Monthly bias: hover to inspect seeds", yaxis_title="Prediction - observed (SGD)", height=430)
    fig.add_hline(y=0, line_color="black", line_width=1)
    figures.append(("Temporal errors", fig, "One fixed origin: target month and horizon move together. This cannot identify separate calendar and horizon effects."))
    for variant in ["T0", "T2"]:
        fig = go.Figure(go.Scattergl(x=blocks.longitude, y=blocks.latitude, mode="markers",
                                     marker={"color": blocks[variant], "colorscale": "RdBu_r", "cmin": -400,
                                             "cmax": 400, "size": 7, "colorbar": {"title": "Residual SGD"}},
                                     customdata=np.column_stack([blocks.town, blocks.block, blocks.n, blocks.actual]),
                                     hovertemplate="%{customdata[0]}<br>%{customdata[1]}<br>n=%{customdata[2]}<br>Observed mean=%{customdata[3]:.0f}<br>Residual=%{marker.color:.1f}<extra></extra>"))
        fig.update_layout(title=f"{variant}, seed 42: block-mean residuals (n ≥ 5)", height=500,
                           xaxis_title="Longitude", yaxis_title="Latitude", yaxis={"scaleanchor": "x", "scaleratio": 1})
        figures.append((f"Spatial errors: {variant}", fig, "Positive residual means overprediction. Block means reflect property composition and support; these are model errors, not geographic causal effects."))
    for variant in ["T0", "T2"]:
        table = town_month.assign(value=town_month[variant].where(town_month.n >= 30)).pivot(index="town", columns="month", values="value")
        fig = go.Figure(go.Heatmap(z=table.to_numpy(), x=list(table.columns), y=list(table.index),
                                   colorscale="RdBu_r", zmin=-220, zmax=220, colorbar={"title": "SGD"}))
        fig.update_layout(title=f"{variant}: town × month residual", height=700)
        figures.append((f"Space and time together: {variant}", fig, "Cells with fewer than 30 rentals are omitted. Persistent rows or columns reveal location or time bias; moving patches suggest remaining joint structure."))
    fig = go.Figure()
    for variant in ["T0", "T2"]:
        part = reliance[reliance.variant.eq(variant)].groupby("group").delta_rmse_sgd.mean()
        fig.add_trace(go.Bar(x=part.values, y=part.index, name=variant, orientation="h", marker_color=COLORS[variant]))
    fig.update_layout(title="Grouped feature reliance: within-month/type donor shuffles", barmode="group",
                       xaxis_title="Increase in sampled RMSE (SGD)", height=450)
    figures.append(("Feature reliance", fig, "Average across three seeds and three shuffles. History groups move the external forecast baseline too, so this is whole-pipeline reliance. Flat type is held fixed. Correlated inputs can substitute for one another, group membership overlaps, and some cross-group combinations are unrealistic. This is not an additive or causal attribution."))
    fig = go.Figure()
    for variant in ["T0", "T2"]:
        for kind in ["3-room", "4-room", "5-room"]:
            part = ale[ale.variant.eq(variant) & ale.flat_type.eq(kind)].groupby("bin").agg(x=("midpoint", "mean"), y=("ale_sgd", "mean"))
            fig.add_trace(go.Scatter(x=part.x, y=part.y, mode="lines+markers", name=f"{variant}: {kind}"))
    fig.update_layout(title="Floor-area accumulated local response", xaxis_title="Floor area (sqm)",
                       yaxis_title="Centered neural response (SGD)", height=450)
    figures.append(("Feature response shape", fig, "Within-type quantile-bin ALE; histories and external forecast baseline remain fixed. This isolates the fitted network's direct response to area, not total rent appreciation from enlarging a flat."))
    chunks = ["<!doctype html><html><head><meta charset='utf-8'><title>Transformer spatial and temporal diagnostics</title>",
              "<style>body{font:16px system-ui,sans-serif;background:#f3f5f8;color:#172535;margin:0}main{max-width:1200px;margin:auto;padding:32px}section{background:white;border-radius:12px;padding:22px;margin:22px 0}p{line-height:1.6}.note{background:#fff3d6;padding:18px;border-radius:10px}a{color:#205fa2}img{max-width:100%}</style></head><body><main>",
              "<h1>What the fitted rental transformers learned</h1>",
              "<p class='note'><strong>Predictive diagnostics, not causal effects.</strong> Inputs pass time-cutoff checks. The architecture combines property and sequence branches additively; it does not implement transaction-to-transaction spatial attention. Reused development data and 51 distinct months limit validation.</p>",
              "<p><a href='findings.md'>Findings and methods</a> · <a href='model_summaries.csv'>Branch contributions</a> · <a href='grouped_reliance.csv'>Feature reliance data</a> · <a href='floor_area_ale.csv'>Area response data</a></p>"]
    for i, (title, fig, caption) in enumerate(figures):
        fig.update_layout(template="plotly_white", margin={"l": 80, "r": 30, "t": 70, "b": 70})
        chunks.extend(["<section><h2>"+html.escape(title)+"</h2><p>"+html.escape(caption)+"</p>",
                       pio.to_html(fig, full_html=False, include_plotlyjs=True if i == 0 else False), "</section>"])
    for name, title in [("05_spatial_autocorrelation", "Remaining spatial autocorrelation"),
                         ("08_sequence_and_lag_reliance", "Temporal channel and lag sensitivity"),
                         ("09_prediction_components", "Individual prediction components"),
                         ("10_observed_feature_associations", "Observed property and accessibility associations (not causal)")]:
        if not (OUT / f"{name}.png").exists():
            continue
        encoded = base64.b64encode((OUT / f"{name}.png").read_bytes()).decode("ascii")
        chunks.append(f"<section><h2>{title}</h2><img alt='{title}' src='data:image/png;base64,{encoded}'></section>")
    chunks.append("</main></body></html>")
    (OUT / "dashboard.html").write_text("\n".join(chunks), encoding="utf-8")


def findings(summaries, reliance, occlusion, moran, replay):
    means = pd.read_csv(SOURCE / "scores.csv").query("variant in ['T0','T1','T2','T3']").groupby("variant").rmse_sgd.agg(["mean", "std"]).reset_index()
    grouped = reliance.groupby(["variant", "group"]).delta_rmse_sgd.agg(["mean", "min", "max"]).reset_index()
    branches = summaries.groupby("variant").agg(sequence_gate=("sequence_gate", "mean"),
                                                  mean_sequence_sgd=("mean_sequence_component_sgd", "mean"),
                                                  mean_abs_sequence_sgd=("mean_abs_sequence_component_sgd", "mean")).reset_index()
    above = moran.assign(above_reference=moran.moran_i > moran.permutation_high).groupby("variant").above_reference.sum()
    disabled = occlusion[occlusion.perturbation.eq("Disable entire sequence branch")].groupby("variant").delta_rmse_sgd.mean()
    disable_text = ", ".join(f"{variant} {disabled[variant]:+.2f} SGD" for variant in ("T1", "T2", "T3"))
    lines = ["# Spatial, temporal and feature-effect diagnostics", "",
             "The inputs have sensible geography and strict historical cutoffs, but these diagnostics do not establish that the model adequately learns the spatial–temporal relationship. The additional sequence branch did not improve consistently over the feature transformer.", "",
             "[Open the interactive dashboard](dashboard.html). All figures and CSVs are standalone local artifacts; no external map or online service is required.", "",
             "## Completed model comparison", "", markdown_table(means), "",
             "Means/std are across the three declared seeds, in individual-rental SGD RMSE. All use the same repeatedly examined June 2024–March 2025 development rows. T0 already has spatial, calendar and engineered rent-history inputs.", "",
             "## What is encoded and what is missing", "",
             "The property encoder attends across feature tokens, including a coordinate/Fourier token, location IDs, amenities, calendar and history summaries. The sequence encoder attends across historical monthly rent/COE/stock tokens and a known forecast-context token. Its local rent panel is selected using planning area and flat type.",
             "The property hidden state does not condition the sequence encoder; outputs combine through a single scalar gate. Direct interactions between the two encoded input blocks are absent by construction. This does not mean all spatial–temporal interactions are absent: the property branch mixes location with time/history, and the sequence already contains local rent histories. Comparable-transaction attention and distance/time attention biases were not tested.", "",
             "## Spatial and temporal errors", "",
             f"Block residual Moran's I exceeds the support-stratified permutation reference upper band in T0 {int(above.get('T0',0))}/10 months and T2 {int(above.get('T2',0))}/10 months. This is a descriptive warning about remaining spatial structure, not a causal test or a multiplicity-adjusted significance claim.",
             "The graph uses at most eight other blocks within 2 km, Gaussian distance weights with 1 km scale, row normalization, and monthly blocks with at least two rentals. The 199 permutations shuffle residual means within count strata (2, 3, 4, 5+) to partly preserve support. They do not account for fitted-model or search uncertainty.",
             "Town/month heatmaps and monthly bias plots show where prediction errors persist. Raw observed-rent maps remain confounded by property composition; they should not be read as pure location premiums.", "",
             "## Which input bundles the prediction pipeline relies on", "", markdown_table(grouped), "",
             "Grouped permutation uses a label-independent 200-query-per-month sample, population weights, and donors within month and flat type; three shuffles for each of three seeds. Flat type itself is held fixed, so its importance cannot be ranked by these shuffles. Geographic columns move together, and the Fourier token is recomputed. History shuffles move the external baseline and sequence together. A joint geography/history shuffle reduces, but cannot eliminate, incompatible feature combinations. The four groups overlap and their importances must not be added. Correlated predictors can substitute for one another, so a small marginal importance does not establish irrelevance.", "",
             "The large history-bundle effect includes the external forecast baseline, and cannot be attributed solely to the new transformer or sequence encoder. Property shuffles retain the same flat type and rent-history proxies, so their smaller effect is not evidence that property characteristics are unimportant. These group tests do not isolate the benefit of Fourier coordinates versus building IDs or other geographic encodings.", "",
             "## Direct floor-area response", "",
             "ALE is computed within 3/4/5-room types, starting with six sampled-feature quantile bins and merging adjacent sparse bins until each has at least 20 rows. It averages adjacent-endpoint changes and accumulates/centers them. The historical summaries and external baseline stay fixed, so this measures direct neural sensitivity, not the complete pipeline effect of changing a property. Shading is seed range, not a confidence interval. [ALE methodology](https://arxiv.org/abs/1612.08468).", "",
             "The additional observed-feature plot compares actual and predicted means against floor area, flat age, nearest MRT distance and CBD distance among 4-room rentals. Those are descriptive associations: other attributes, location and time still vary. They are deliberately distinguished from the model's direct area-response curve.", "",
             "## Sequence contributions and lags", "", markdown_table(branches), "",
             "The saved branch decomposition exactly reconstructs each sampled prediction. The temporal term includes BF16 addition rounding. A small gate is not an importance score because learned output scale can compensate. Compare actual SGD branch contributions and occlusion responses instead.",
             "Channel/lag masking removes values and their presence flags in the frozen model. These are sensitivity probes that can be outside the training distribution; they do not replace matched refitting comparisons T2–T1 or T3–T2. On the weighted sample, disabling the sequence branch changes RMSE on average: " + disable_text + "; negative means improvement. This is a post-hoc diagnostic, not a newly validated deployment choice. Removing COE from a model trained with COE can hurt even though adding COE through matched retraining did not reliably help: reliance and incremental predictive value are different questions. Attention weights are not presented as causal feature contributions. [Attention interpretation caution](https://aclanthology.org/N19-1357/).", "",
             "## Economic effects are not identified by this holdout", "",
             "At the fixed May 2024 rental origin, every main-holdout row shares the same national COE and stock sequence (through April 2024). Row permutation cannot identify their effects: there is no across-row macro variation. Learned responses come from the historical training origins, not 28,995 independent economic paths. A separate COE response curve from this single origin would be poorly supported. Masking and matched refit comparisons are reported instead; no causal COE/stock effect or company-local effect is claimed.", "",
             "Target month and forecast horizon also move together in this fixed-origin evaluation. Their separate effects cannot be inferred from monthly error curves. More independent origins/new rental months and an explicitly tested joint spatial–temporal architecture are needed before claiming the relationship is adequately captured.", "",
             "## Verification and scope", "",
             f"Full saved GPU prediction replay: maximum absolute discrepancy {max(r['max_prediction_difference_sgd'] for r in replay):.3g} SGD. All input, original checkpoint/preprocessor, source-code and submission hashes are checked. No model is retrained and no original experiment artifact is overwritten.",
             "Plots use primary seed 42 for geographic errors and all three declared seeds for reliance/response stability. No best-seed selection. Reproduce with `.venv/Scripts/python.exe analysis/interpret_spatial_temporal_transformer.py`; CUDA inference uses the existing capacity gate. See verification.json for exact inputs, settings, devices and replay tolerances.", ""]
    (OUT / "findings.md").write_text("\n".join(lines), encoding="utf-8")


def render_saved(bundle):
    frame = load_predictions(bundle)
    blocks, town_month, moran = descriptive_figures(frame)
    observed_associations(bundle, frame)
    tables = {name: pd.read_csv(OUT / f"{name}.csv") for name in
              ("grouped_reliance", "floor_area_ale", "sequence_occlusion", "lag_occlusion", "prediction_components", "model_summaries")}
    model_figures(tables["grouped_reliance"], tables["floor_area_ale"], tables["sequence_occlusion"],
                  tables["lag_occlusion"], tables["prediction_components"])
    verification = json.loads((OUT / "verification.json").read_text())
    write_dashboard(frame, blocks, town_month, tables["grouped_reliance"], tables["floor_area_ale"], tables["model_summaries"], moran)
    findings(tables["model_summaries"], tables["grouped_reliance"], tables["sequence_occlusion"], moran,
             verification["full_holdout_prediction_replay"])
    verification["render_code_sha256"] = sha256(Path(__file__))
    verification["figures"] = sorted(p.name for p in OUT.glob("*.png"))
    write_json(OUT / "verification.json", verification)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--min-free-gib", type=float, default=8.)
    parser.add_argument("--per-month", type=int, default=200)
    parser.add_argument("--refresh-ale", action="store_true", help="Recompute only area effects, then render saved diagnostics")
    parser.add_argument("--render-only", action="store_true", help="CPU-only plots from saved diagnostic tables")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    started = time.perf_counter()
    settings = json.loads((SOURCE / "settings.json").read_text())
    for path, digest in settings["hashes"].items():
        if sha256(ROOT / path) != digest:
            raise ValueError(f"Original source changed: {path}")
    prepared_manifest = json.loads((SOURCE / "prepared.json").read_text())
    if sha256(CACHE / "prepared.joblib") != prepared_manifest["sha256"]:
        raise ValueError("Prepared bundle changed")
    with threadpool_limits(limits=4):
        bundle = joblib.load(CACHE / "prepared.joblib")["folds"]["outer_d0"]
        if args.refresh_ale:
            sample = pd.read_csv(OUT / "sample.csv")
            positions = bundle["X_query"].index.get_indexer(sample.row_id)
            if (positions < 0).any():
                raise ValueError("Saved interpretation sample changed")
            records = []
            for variant in ("T0", "T2"):
                for seed in (42, 43, 44):
                    if torch.cuda.is_initialized():
                        torch.cuda.empty_cache()
                    wait_for_gpu(min_free_gib=args.min_free_gib, max_utilization=35, stable_samples=2,
                                 poll_seconds=15, status_path=OUT / "gpu_status.json")
                    print(f"REFRESH AREA RESPONSE {variant}/{seed}", flush=True)
                    predictor = SavedPredictor(variant, seed, bundle, args.device)
                    records.extend(area_effects(predictor, bundle, positions))
                    del predictor
                    gc.collect()
            pd.DataFrame(records).to_csv(OUT / "floor_area_ale.csv", index=False)
            verification = json.loads((OUT / "verification.json").read_text())
            verification["ale_refresh_code_sha256"] = sha256(Path(__file__))
            verification["ale_bins"] = "six initial quantile bins; adjacent sparse bins merged to minimum 20 rows"
            write_json(OUT / "verification.json", verification)
            render_saved(bundle)
            return
        if args.render_only:
            render_saved(bundle)
            return
        frame = load_predictions(bundle)
        blocks, town_month, moran = descriptive_figures(frame)
        observed_associations(bundle, frame)
        positions, sample_weights = sample_months(bundle["query_month"], args.per_month)
        frame.iloc[positions].assign(sample_weight=sample_weights).to_csv(OUT / "sample.csv", index=False)
        decompositions, reliances, ales, occlusions, lag_records, summaries, replay = [], [], [], [], [], [], []
        for variant in COLORS:
            for seed in (42, 43, 44):
                gc.collect()
                if torch.cuda.is_initialized():
                    torch.cuda.empty_cache()
                wait_for_gpu(min_free_gib=args.min_free_gib, max_utilization=35, stable_samples=2,
                             poll_seconds=15, status_path=OUT / "gpu_status.json")
                print(f"INTERPRET {variant} seed {seed}", flush=True)
                predictor = SavedPredictor(variant, seed, bundle, args.device)
                actual = pd.read_csv(SOURCE / "outer" / variant / f"seed{seed}" / "predictions.csv")
                expected = predictor.predict(bundle["X_query"], bundle["query_sequence"], bundle["query_context"], bundle["query_baseline"])
                difference = float(np.max(abs(expected-actual.predicted.to_numpy())))
                if difference > 1e-7:
                    raise ValueError(f"GPU replay mismatch {variant}/{seed}: {difference}")
                replay.append({"variant": variant, "seed": seed, "max_prediction_difference_sgd": difference})
                d, r, a, o, l, s = interpret_model(predictor, bundle, positions, sample_weights)
                decompositions.append(d); reliances.extend(r); ales.extend(a); occlusions.extend(o); lag_records.extend(l); summaries.append(s)
                print(f"COMPLETE {variant}/{seed}: mean sequence contribution={s['mean_sequence_component_sgd']:.3f} SGD", flush=True)
                del predictor
        decomposition, reliance, ale = pd.concat(decompositions), pd.DataFrame(reliances), pd.DataFrame(ales)
        occlusion, lags, summary = pd.DataFrame(occlusions), pd.DataFrame(lag_records), pd.DataFrame(summaries)
        for name, table in [("prediction_components", decomposition), ("grouped_reliance", reliance),
                            ("floor_area_ale", ale), ("sequence_occlusion", occlusion),
                            ("lag_occlusion", lags), ("model_summaries", summary)]:
            table.to_csv(OUT / f"{name}.csv", index=False)
        model_figures(reliance, ale, occlusion, lags, decomposition)
        write_dashboard(frame, blocks, town_month, reliance, ale, summary, moran)
        findings(summary, reliance, occlusion, moran, replay)
        for path, digest in settings["hashes"].items():
            assert sha256(ROOT / path) == digest, path
        verification = {"code_sha256": sha256(Path(__file__)), "prepared_sha256": prepared_manifest["sha256"],
                        "sample_rows": len(positions), "sample_seed": 2026, "population_weights": True,
                        "device": args.device, "full_holdout_prediction_replay": replay,
                        "original_sources_and_submissions_unchanged": True, "no_refits": True,
                        "seconds": time.perf_counter()-started, "figures": sorted(p.name for p in OUT.glob("*.png")),
                        "macro_row_variation": "zero in fixed-origin holdout; macro importance not identified by row permutation"}
        write_json(OUT / "verification.json", verification)
        print("INTERPRETATION COMPLETE " + str(OUT / "dashboard.html"), flush=True)


if __name__ == "__main__":
    main()
