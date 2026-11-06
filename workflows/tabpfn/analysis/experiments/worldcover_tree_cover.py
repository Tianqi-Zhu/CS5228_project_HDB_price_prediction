"""Tree-class fractions on a metric grid, with explicit publication cutoffs."""
from __future__ import annotations

import numpy as np
import pandas as pd
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.windows import from_bounds, Window
import rasterio


def buffer_statistics(values, transform, xy, radii, tree_classes=(10,), minimum_valid=.95):
    """Equal-area pixel-centre approximation on a north-up, square metric grid.

    Zero is WorldCover nodata, not absence of trees. Class 95 (mangroves) is
    deliberately separate from the primary class-10 definition.
    """
    if transform.b != 0 or transform.d != 0 or transform.a <= 0 or transform.e >= 0:
        raise ValueError("Expected a north-up metric raster")
    if not np.isclose(transform.a, -transform.e):
        raise ValueError("Expected square metric pixels")
    rows, cols = values.shape
    scale = transform.a
    output = []
    for x, y in np.asarray(xy, float):
        record = {}
        for radius in radii:
            left = int(np.floor((x - radius - transform.c) / scale))
            right = int(np.ceil((x + radius - transform.c) / scale))
            top = int(np.floor((transform.f - y - radius) / scale))
            bottom = int(np.ceil((transform.f - y + radius) / scale))
            xs = transform.c + (np.arange(left, right) + .5) * scale
            ys = transform.f - (np.arange(top, bottom) + .5) * scale
            circle = (xs[None, :] - x)**2 + (ys[:, None] - y)**2 <= radius**2
            # Outside the raster is explicitly nodata, including in coverage.
            window = np.zeros(circle.shape, dtype=values.dtype)
            l, r, t, b = max(left, 0), min(right, cols), max(top, 0), min(bottom, rows)
            if l < r and t < b:
                window[t-top:b-top, l-left:r-left] = values[t:b, l:r]
            valid = circle & (window != 0)
            support = int(circle.sum())
            count = int(valid.sum())
            coverage = count / support if support else 0.
            fraction = float((valid & np.isin(window, tree_classes)).sum() / count) if count and coverage >= minimum_valid else np.nan
            record[f"TREE_CLASS_FRACTION_{radius}M"] = fraction
            record[f"WORLDCOVER_VALID_FRACTION_{radius}M"] = coverage
        output.append(record)
    return pd.DataFrame(output)


def extract_product(path, buildings, config):
    """Read only the Singapore window; nearest-neighbour reproject at 10 m."""
    xy = buildings[["x_m", "y_m"]].to_numpy(float)
    if not np.isfinite(xy).all() or buildings.building_id.duplicated().any():
        raise ValueError("Invalid canonical locations")
    size = config["pixel_size_m"]
    padding = max(config["radii_m"]) + 2 * size
    left, bottom = np.floor((xy.min(axis=0) - padding) / size) * size
    right, top = np.ceil((xy.max(axis=0) + padding) / size) * size
    dst_transform = from_origin(left, top, size, size)
    destination = np.zeros((int(round((top-bottom)/size)), int(round((right-left)/size))), dtype="uint8")
    with rasterio.open(path) as src:
        if src.count != 1 or src.crs != rasterio.crs.CRS.from_epsg(4326):
            raise ValueError("Expected one-band geographic WorldCover Map")
        bounds = transform_bounds(config["projected_crs"], src.crs, left, bottom, right, top, densify_pts=21)
        if bounds[0] < src.bounds.left or bounds[1] < src.bounds.bottom or bounds[2] > src.bounds.right or bounds[3] > src.bounds.top:
            raise ValueError("Canonical buffers extend beyond the configured source tile")
        window = from_bounds(*bounds, transform=src.transform)
        # Round outward to include every source pixel intersecting the window.
        col, row = int(np.floor(window.col_off)), int(np.floor(window.row_off))
        window = Window(col, row, int(np.ceil(window.col_off+window.width))-col, int(np.ceil(window.row_off+window.height))-row)
        source = src.read(1, window=window)
        if not set(np.unique(source)) <= {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100}:
            raise ValueError("Unexpected WorldCover classification codes")
        reproject(source, destination, src_transform=src.window_transform(window), src_crs=src.crs,
                  src_nodata=0, dst_transform=dst_transform, dst_crs=config["projected_crs"],
                  dst_nodata=0, resampling=Resampling.nearest)
    result = buffer_statistics(destination, dst_transform, xy, config["radii_m"],
                               config["tree_classes"], config["minimum_valid_fraction"])
    result.insert(0, "building_id", buildings.building_id.to_numpy())
    return result


def origin_features(building_ids, origins, products, config):
    """Latest *published* product, never inferred from its observation year."""
    if len(building_ids) != len(origins):
        raise ValueError("Origins and buildings must align")
    columns = [f"TREE_CLASS_FRACTION_{r}M" for r in config["radii_m"]]
    result = pd.DataFrame(np.nan, index=building_ids.index, columns=columns)
    years = pd.Series(pd.NA, index=building_ids.index, dtype="Int64", name="worldcover_year")
    cutoffs = pd.PeriodIndex(origins, freq="M").to_timestamp(how="end")
    for spec in sorted(config["products"], key=lambda p: p["available_at"]):
        if pd.Timestamp(spec["available_at"]) < pd.Timestamp(spec["published_at"]) or pd.Timestamp(spec["published_at"]) <= pd.Timestamp(f"{spec['year']}-12-31"):
            raise ValueError("Invalid publication policy")
        table = products[spec["year"]]
        if table.building_id.duplicated().any():
            raise ValueError("Duplicate building features")
        eligible = cutoffs >= pd.Timestamp(spec["available_at"])
        lookup = table.set_index("building_id")
        matched = lookup.reindex(building_ids.to_numpy())[columns].set_axis(building_ids.index)
        result.loc[eligible, columns] = matched.loc[eligible, columns]
        years.loc[eligible] = spec["year"]
    result["TREE_COVER_UNAVAILABLE"] = result.isna().any(axis=1).astype(float)
    return result, years
