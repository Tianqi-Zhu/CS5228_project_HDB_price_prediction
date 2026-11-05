"""Dated satellite exports and train-only PCA. No remote calls or credentials."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

AE_BANDS = [f"A{i:02d}" for i in range(64)]
DW_BANDS = ["water", "trees", "grass", "flooded_vegetation", "crops", "shrub_and_scrub", "built", "bare", "snow_and_ice"]


def validate_export(table, source):
    required = {"building_id", "observation_year", "radius_m", "observation_end", "available_at", "availability_source_url", "valid_pixel_fraction", "valid_observation_count"}
    bands = AE_BANDS if source == "alphaearth" else DW_BANDS
    if not required.union(bands) <= set(table):
        raise ValueError(f"Incomplete export: {required.union(bands) - set(table)}")
    if table.duplicated(["building_id", "observation_year", "radius_m"]).any():
        raise ValueError("Duplicate satellite building/year/scale")
    observation = pd.to_datetime(table.observation_end)
    available = pd.to_datetime(table.available_at)
    if (available < observation).any() or table.availability_source_url.isna().any():
        raise ValueError("Publication dates require documented support after observation ends")
    if not table.valid_pixel_fraction.between(0, 1).all():
        raise ValueError("Invalid valid-pixel fraction")
    if source == "alphaearth":
        year_end = pd.to_datetime(table.observation_year.astype(str) + "-12-31")
        if (observation.dt.normalize() < year_end).any():
            raise ValueError("Annual embedding cannot be available before year completion")
        if (available < pd.Timestamp("2025-07-30")).any():
            raise ValueError("AlphaEarth public release was July 30 2025")
    return bands


def join_satellite(table, queries, source, radius, *, retrospective=False):
    bands = validate_export(table, source)
    table = table.loc[table.radius_m == radius].copy()
    table["observation_end"] = pd.to_datetime(table.observation_end)
    table["available_at"] = pd.to_datetime(table.available_at)
    rows = []
    lookup = dict(tuple(table.groupby("building_id")))
    for building, origin in queries[["building_id", "forecast_origin"]].itertuples(index=False, name=None):
        candidates = lookup.get(building)
        selected = None
        if candidates is not None and pd.notna(origin):
            cutoff = pd.Period(origin, freq="M").end_time
            mask = (candidates.observation_end < cutoff) & (candidates.observation_year < cutoff.year)
            if not retrospective:
                mask &= candidates.available_at <= cutoff
            candidates = candidates.loc[mask & (candidates.valid_observation_count > 0) & (candidates.valid_pixel_fraction > 0)]
            if len(candidates):
                selected = candidates.sort_values("observation_end").iloc[-1]
        rows.append([np.nan] * len(bands) if selected is None else selected[bands].to_numpy(float))
    result = pd.DataFrame(rows, index=queries.index, columns=[f"{source}_{radius}m_{b}" for b in bands])
    result[f"{source}_{radius}m_unavailable"] = result.isna().any(axis=1).astype(float)
    return result


class TrainingPCA:
    """Means and PCA are fitted once on training locations only, no labels."""
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, training):
        x = np.asarray(training, float)
        if x.ndim != 2 or np.isnan(x).all(axis=0).any():
            raise ValueError("Cannot fit PCA when a complete input band is unavailable")
        self.mean_ = np.nanmean(x, axis=0)
        self.model_ = PCA(n_components=self.n_components, svd_solver="full", random_state=42)
        self.model_.fit(np.where(np.isfinite(x), x, self.mean_))
        return self

    def transform(self, query):
        x = np.asarray(query, float)
        missing = ~np.isfinite(x).all(axis=1)
        values = self.model_.transform(np.where(np.isfinite(x), x, self.mean_))
        # Entirely unavailable exports remain missing, rather than becoming
        # fabricated mean embeddings. Partial-band missingness has an indicator.
        values[~np.isfinite(x).any(axis=1)] = np.nan
        return values, missing.astype(float)
