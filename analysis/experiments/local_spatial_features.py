"""Target-free local economy/park measurements in Singapore SVY21 metres."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
import shapely
from shapely.geometry import shape
from shapely.ops import transform

PROJECT = Transformer.from_crs("EPSG:4326", "EPSG:3414", always_xy=True)


def metric_xy(longitude, latitude):
    lon, lat = np.asarray(longitude, float), np.asarray(latitude, float)
    if not np.isfinite(lon).all() or not np.isfinite(lat).all():
        raise ValueError("Coordinates must be finite; missing is not zero")
    if np.any((lon < 103) | (lon > 105) | (lat < 1) | (lat > 2)):
        raise ValueError("Coordinates outside Singapore bounding box")
    return np.column_stack(PROJECT.transform(lon, lat))


def address_keys(frame):
    return frame[["TOWN", "BLOCK", "STREET"]].astype("string").apply(
        lambda col: col.str.lower().str.strip().str.replace(r"\s+", " ", regex=True)).agg("|".join, axis=1)


def canonical_buildings(blocks, transactions):
    blocks = blocks.copy()
    blocks["address_key"] = address_keys(blocks)
    if blocks.address_key.duplicated().any():
        raise ValueError("Ambiguous building address; resolve before joining")
    xy = metric_xy(blocks.LONGITUDE, blocks.LATITUDE)
    blocks[["x_m", "y_m"]] = xy
    blocks["building_id"] = [hashlib.sha256(f"{key}|{lon:.7f}|{lat:.7f}".encode()).hexdigest()[:20]
                             for key, lon, lat in zip(blocks.address_key, blocks.LONGITUDE, blocks.LATITUDE)]
    if blocks.building_id.duplicated().any():
        raise ValueError("Canonical identifier collision")
    blocks["geocoding_quality"] = "unique exact normalized town/block/street match; supplied coordinates"
    blocks["coordinate_source"] = "data/auxiliary/sg-hdb-block.csv; original geocoder/snapshot date unknown"
    links = pd.DataFrame({"row_id": transactions.index, "address_key": address_keys(transactions)})
    links = links.merge(blocks[["address_key", "building_id"]], on="address_key", how="left", validate="many_to_one")
    if len(links) != len(transactions) or links.building_id.isna().any():
        raise ValueError("Canonical building join lost or duplicated transactions")
    return blocks.set_index("building_id", drop=False), links.set_index("row_id")


def load_geometry(path):
    doc = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    geometries, properties = [], []
    for feature in doc["features"]:
        if feature.get("geometry"):
            geometry = shapely.make_valid(transform(PROJECT.transform, shape(feature["geometry"])))
            if not geometry.is_empty:
                geometries.append(geometry)
                properties.append(feature.get("properties", {}))
    return geometries, properties


def commercial_fields(building_xy, points_xy, radii, bandwidths, prefix):
    """Counts, establishments/km2, normalized Gaussian intensity/km2.

    KDE integrates to the number of establishments (not a unit-mass PDF).
    All supplied POIs contribute; no kernel truncation or boundary correction.
    """
    b, p = np.asarray(building_xy, float), np.asarray(points_xy, float)
    if p.ndim != 2 or len(p) == 0 or not np.isfinite(p).all():
        raise ValueError("POI source unavailable; do not substitute zero establishments")
    p = np.unique(p, axis=0)
    distance = np.linalg.norm(b[:, None, :] - p[None, :, :], axis=2)
    result = {f"{prefix}_NEAREST_M": distance.min(axis=1)}
    for radius in radii:
        count = (distance <= radius).sum(axis=1).astype(float)
        result[f"{prefix}_COUNT_{radius}M"] = count
        result[f"{prefix}_DENSITY_{radius}M_KM2"] = count / (np.pi * (radius / 1000)**2)
    for sigma in bandwidths:
        result[f"{prefix}_KDE_{sigma}M_KM2"] = np.exp(-0.5 * (distance / sigma)**2).sum(axis=1) / (2 * np.pi * (sigma / 1000)**2)
    return pd.DataFrame(result)


def park_fields(building_xy, geometries, radii):
    """Managed-park polygon support, never relabelled as satellite vegetation."""
    if not geometries:
        raise ValueError("Park source unavailable")
    merged = shapely.union_all(geometries)
    patches = list(shapely.get_parts(merged))
    tree = shapely.STRtree(patches)
    points = shapely.points(np.asarray(building_xy))
    nearest = tree.nearest(points)
    data = {"PARK_NEAREST_M": shapely.distance(points, np.asarray(patches, dtype=object)[nearest])}
    for radius in radii:
        fractions, largest = [], []
        for point in points:
            buffer = point.buffer(radius, quad_segs=24)
            candidates = tree.query(buffer, predicate="intersects")
            areas = shapely.area(shapely.intersection(np.asarray(patches, dtype=object)[candidates], buffer))
            fractions.append(min(1.0, float(areas.sum() / buffer.area)))
            largest.append(float(areas.max() / buffer.area) if len(areas) else 0.)
        data[f"PARK_FRACTION_{radius}M"] = np.asarray(fractions)
        data[f"PARK_LARGEST_PATCH_FRACTION_{radius}M"] = np.asarray(largest)
    p = np.clip(data[f"PARK_FRACTION_{max(radii)}M"], 1e-12, 1 - 1e-12)
    data["PARK_BINARY_ENTROPY"] = -p * np.log(p) - (1 - p) * np.log(1 - p)
    # Piecewise constant Gaussian radial quadrature over the declared annuli.
    previous, weighted, mass, last = np.zeros(len(points)), np.zeros(len(points)), 0., 0
    sigma = 200.
    for radius in sorted(radii):
        cumulative = data[f"PARK_FRACTION_{radius}M"] * radius**2
        weight = np.exp(-0.5 * (((last + radius) / 2) / sigma)**2)
        weighted += weight * (cumulative - previous)
        mass += weight * (radius**2 - last**2)
        previous, last = cumulative, radius
    data["PARK_GAUSSIAN_200M_FRACTION"] = np.clip(weighted / mass, 0, 1)
    return pd.DataFrame(data)


def polygon_assignment(buildings, geometries, properties):
    """Actual point-in-polygon join; boundary points use covers, not nearest area."""
    tree = shapely.STRtree(geometries)
    points = shapely.points(buildings[["x_m", "y_m"]].to_numpy())
    rows = []
    for point in points:
        matches = tree.query(point, predicate="covered_by")
        if len(matches) == 1:
            prop = properties[int(matches[0])]
            rows.append([str(prop.get("PLN_AREA_N", "")).lower(), str(prop.get("SUBZONE_N", "")).lower(), 1])
        else:
            rows.append([None, None, len(matches)])
    return pd.DataFrame(rows, index=buildings.index, columns=["polygon_planning_area", "polygon_subzone", "polygon_match_count"])


def census_table(path):
    """Published Census 2020 table: preserve regional support and suppression."""
    raw = pd.read_csv(path, dtype=str)
    number = next(c for c in raw if c.lower() == "number")
    total = next(c for c in raw if re.sub(r"[^a-z]", "", c.lower()) == "totaltotal")
    elderly = [c for c in raw if re.match(r"Total_(65|70|75|80|85|90)", c)]
    if len(elderly) != 6:
        raise ValueError("Census age schema changed")
    def numeric(s):
        return pd.to_numeric(s.str.replace(",", "", regex=False), errors="coerce")
    labels = raw[number].str.strip().str.lower()
    values = pd.DataFrame({"region_label": labels.str.replace(r"\s+-\s+total$", "", regex=True),
                           "is_planning_area": labels.str.endswith(" - total"),
                           "CENSUS2020_POPULATION": numeric(raw[total])})
    values["CENSUS2020_AGE65_SHARE"] = pd.concat([numeric(raw[c]) for c in elderly], axis=1).sum(axis=1, min_count=len(elderly)) / values.CENSUS2020_POPULATION.replace(0, np.nan)
    return values


def regional_features(frame, origins, census, available_at="2021-07-01"):
    # Existing address metadata uses these same MP2019 planning-area labels.
    # Duplicate subzone names must never be resolved by arbitrary first-match.
    lookup = census.loc[census.is_planning_area].dropna(subset=["region_label"]).copy()
    duplicate = lookup.region_label.duplicated(keep=False)
    lookup = lookup.loc[~duplicate].set_index("region_label")
    result = pd.DataFrame(index=frame.index)
    for name in ("CENSUS2020_POPULATION", "CENSUS2020_AGE65_SHARE"):
        result[name] = frame.PLANNING_AREA.astype(str).str.lower().map(lookup[name])
    cutoff = pd.PeriodIndex(origins, freq="M").to_timestamp(how="end")
    valid = (~cutoff.isna()) & (cutoff >= pd.Timestamp(available_at))
    result.loc[~valid, :] = np.nan
    result["CENSUS2020_UNAVAILABLE"] = result.CENSUS2020_POPULATION.isna().astype(float)
    return result
