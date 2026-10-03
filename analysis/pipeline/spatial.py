"""Geographic helpers shared by the EDA studies.

Facilities are current snapshots. These distances are not historically
verified travel-time features and are not included in A/B/C models yet.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

EARTH_RADIUS_KM = 6371.0088


def nearest_distances(points: pd.DataFrame, places: pd.DataFrame, chunk: int = 2500):
    if chunk < 1:
        raise ValueError("chunk must be positive")
    columns = ["LATITUDE", "LONGITUDE"]
    origin = points[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    target = places[columns].apply(pd.to_numeric, errors="coerce").drop_duplicates().to_numpy(float)

    def valid(coords):
        return (np.isfinite(coords).all(axis=1) & (np.abs(coords[:, 0]) <= 90)
                & (np.abs(coords[:, 1]) <= 180))

    target = target[valid(target)]
    distances = np.full(len(points), np.nan)
    if not len(target):
        return distances
    tree = BallTree(np.radians(target), metric="haversine")
    positions = np.flatnonzero(valid(origin))
    for start in range(0, len(positions), chunk):
        index = positions[start:start + chunk]
        distances[index] = tree.query(np.radians(origin[index]), k=1)[0][:, 0] * EARTH_RADIUS_KM
    return distances


def primary_schools(schools: pd.DataFrame, *, include_mixed=True):
    """Keep the two original EDA populations explicitly distinguishable."""
    levels = schools["MAINLEVEL_CODE"].astype("string").str.casefold()
    mask = levels.str.contains(r"primary|p1", regex=True, na=False) if include_mixed else levels.str.startswith("primary", na=False)
    return schools.loc[mask].copy()
