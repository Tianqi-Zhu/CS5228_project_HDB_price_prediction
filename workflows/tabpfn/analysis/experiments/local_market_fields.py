"""Origin-aligned, publication-lagged local rental fields; no in-sample labels."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix


def neighborhood_matrix(xy, radius, sigma=None):
    tree = cKDTree(xy)
    neighborhoods = tree.query_ball_tree(tree, radius)
    rows = np.repeat(np.arange(len(xy)), [len(v) for v in neighborhoods])
    cols = np.concatenate(neighborhoods)
    weights = np.ones(len(rows))
    if sigma is not None:
        weights = np.exp(-.5 * (np.linalg.norm(xy[rows] - xy[cols], axis=1) / sigma)**2)
    return csr_matrix((weights, (rows, cols)), shape=(len(xy), len(xy)))


def safe_mean(total, count):
    return np.divide(total, count, out=np.full(np.shape(total), np.nan, dtype=float), where=count > 0)


def historical_market_features(past, queries, origins, buildings, config):
    """Only `past` labels are accepted. Query targets are deliberately unread.

    Observation month <= forecast_origin - publication_lag. A query must be
    strictly after its origin. Zero prior observations produce missing means.
    """
    origins = pd.Series(pd.PeriodIndex(origins, freq="M"), index=queries.index)
    if len(origins) != len(queries):
        raise ValueError("Origins and queries misaligned")
    valid = origins.notna()
    if np.any(queries.loc[valid, "_MONTH"].array <= origins.loc[valid].array):
        raise ValueError("Queries must follow their origin")
    if config["history_publication_lag_months"] < 1:
        raise ValueError("A conservative publication lag is required")
    ids = pd.Index(buildings.index)
    positions = ids.get_indexer(past.building_id)
    qpositions = ids.get_indexer(queries.building_id)
    if (positions < 0).any() or (qpositions < 0).any():
        raise ValueError("Missing building coordinates")
    xy = buildings[["x_m", "y_m"]].to_numpy()
    matrices = {r: neighborhood_matrix(xy, r) for r in config["history_radii_m"]}
    sigma = config["history_continuous_sigma_m"]
    continuous = neighborhood_matrix(xy, 4 * sigma, sigma)
    output = pd.DataFrame(index=queries.index)
    kinds = past.FLAT_TYPE_CLEAN.astype(str).to_numpy()
    qkinds = queries.FLAT_TYPE_CLEAN.astype(str).to_numpy()
    months = pd.PeriodIndex(past._MONTH, freq="M").asi8
    rents = past.MONTHLY_RENT.to_numpy(float)
    area = past.FLOOR_AREA_SQM.to_numpy(float)
    lag = config["history_publication_lag_months"]
    shrink = config["history_shrinkage_observations"]
    for origin in sorted(origins.dropna().unique()):
        latest = origin.ordinal - lag
        oi = origins.eq(origin).to_numpy()
        for kind in np.unique(qkinds[oi]):
            qi = np.flatnonzero(oi & (qkinds == kind))
            qb = qpositions[qi]
            available = (kinds == kind) & (months <= latest)
            if not available.any():
                continue
            for width in config["history_windows_months"]:
                mask = available & (months > latest - width)
                sums = np.bincount(positions[mask], weights=rents[mask], minlength=len(ids))
                counts = np.bincount(positions[mask], minlength=len(ids)).astype(float)
                national = rents[mask].mean() if mask.any() else np.nan
                for radius, matrix in matrices.items():
                    count, total = (matrix @ counts)[qb], (matrix @ sums)[qb]
                    prefix = f"LOCAL_{radius}M_{width}M"
                    raw = safe_mean(total, count)
                    smooth = (total + shrink * national) / (count + shrink)
                    output.loc[queries.index[qi], f"{prefix}_MEAN"] = raw
                    output.loc[queries.index[qi], f"{prefix}_COUNT"] = count
                    output.loc[queries.index[qi], f"{prefix}_SHRUNK"] = smooth
                    output.loc[queries.index[qi], f"{prefix}_PREMIUM"] = smooth - national
            mask = available & (months > latest - 12)
            ages = latest - months[mask]
            weights = 2. ** (-ages / config["history_temporal_half_life_months"])
            counts = np.bincount(positions[mask], weights=weights, minlength=len(ids))
            sums = np.bincount(positions[mask], weights=weights * rents[mask], minlength=len(ids))
            ppsm = np.bincount(positions[mask], weights=weights * rents[mask] / area[mask], minlength=len(ids))
            mass, total = (continuous @ counts)[qb], (continuous @ sums)[qb]
            national = np.average(rents[mask], weights=weights) if mask.any() else np.nan
            output.loc[queries.index[qi], "LOCAL_KERNEL_MEAN"] = safe_mean(total, mass)
            output.loc[queries.index[qi], "LOCAL_KERNEL_SHRUNK"] = (total + shrink * national) / (mass + shrink)
            output.loc[queries.index[qi], "LOCAL_KERNEL_MASS"] = mass
            output.loc[queries.index[qi], "LOCAL_KERNEL_PPSM"] = safe_mean((continuous @ ppsm)[qb], mass)
    for radius in config["history_radii_m"]:
        for width in config["history_windows_months"]:
            for suffix in ("MEAN", "COUNT", "SHRUNK", "PREMIUM"):
                name = f"LOCAL_{radius}M_{width}M_{suffix}"
                if name not in output:
                    output[name] = np.nan
                if suffix == "COUNT":
                    output[name] = output[name].fillna(0.)
    for name in ("LOCAL_KERNEL_MEAN", "LOCAL_KERNEL_SHRUNK", "LOCAL_KERNEL_MASS", "LOCAL_KERNEL_PPSM"):
        if name not in output:
            output[name] = np.nan
    output["LOCAL_KERNEL_MASS"] = output.LOCAL_KERNEL_MASS.fillna(0.)
    output["LOCAL_HISTORY_UNAVAILABLE"] = output.LOCAL_KERNEL_MEAN.isna().astype(float)
    return output.sort_index(axis=1)
