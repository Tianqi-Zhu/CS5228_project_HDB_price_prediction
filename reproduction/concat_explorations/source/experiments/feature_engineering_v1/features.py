"""Static (target-independent) feature groups: micro-location, relative area, flags."""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
import history


def micro_features(frame: pd.DataFrame, groups: tuple[str, ...]) -> tuple[pd.DataFrame, dict]:
    """Categorical micro-location keys.

    All of these are target-independent identifiers, so they carry no leakage
    risk by themselves; they are model-native categoricals for LightGBM.  The
    composite block key is ``TOWN_CLEAN + STREET_CLEAN + BLOCK_CLEAN`` as
    required by the plan.
    """
    data: dict[str, pd.Series] = {}
    if "micro_region" in groups:
        data["SUBZONE_ID"] = frame["SUBZONE"].fillna("unknown").astype(str)
        data["PLANNING_AREA_ID"] = frame["PLANNING_AREA"].fillna("unknown").astype(str)
        data["REGION_ID"] = frame["REGION"].fillna("unknown").astype(str)
    if "micro_street" in groups:
        data["STREET_ID"] = frame["TOWN_CLEAN"].fillna("unknown").astype(str).str.cat(
            frame["STREET_CLEAN"].fillna("unknown").astype(str), sep="|")
    if "micro_block" in groups:
        data["BLOCK_ID"] = frame["BLOCK_ID"].astype(str)

    result = pd.DataFrame(data, index=frame.index)
    manifest = {
        "enabled": bool(data),
        "groups": list(groups),
        "columns": list(result.columns),
        "cardinality": {column: int(result[column].nunique()) for column in result.columns},
        "note": "target-independent identifiers; no label used to build them",
    }
    return result, manifest


def area_features(
    frame: pd.DataFrame,
    split: dict,
    groups: tuple[str, ...],
) -> tuple[pd.DataFrame, dict]:
    """Relative area within flat type, plus metadata missingness flags.

    Reference statistics (flat-type median / IQR) are computed on the fitting
    rows only; validation rows are transformed with those frozen values.
    """
    cutoff = int(split["cutoff_index"])
    fitting = frame[config.MONTH_INDEX_COL].to_numpy() <= cutoff
    area = pd.to_numeric(frame["FLOOR_AREA_SQM"], errors="coerce")
    flat_type = frame["FLAT_TYPE_CLEAN"].fillna("unknown").astype(str)

    data: dict[str, pd.Series] = {}
    manifest: dict = {"enabled": False, "fitting_rows": int(fitting.sum())}

    if "area" in groups or "flags" in groups:
        ref = pd.DataFrame({"area": area[fitting].to_numpy(), "ft": flat_type[fitting].to_numpy()})
        group = ref.groupby("ft")["area"]
        stats = group.agg(["median", lambda values: values.quantile(0.25),
                           lambda values: values.quantile(0.75)])
        stats.columns = ["median", "q25", "q75"]
        global_median = float(ref["area"].median())
        global_iqr = float(ref["area"].quantile(0.75) - ref["area"].quantile(0.25))
        stats["iqr"] = (stats["q75"] - stats["q25"]).where(lambda values: values > 0, global_iqr)
        stats["median"] = stats["median"].fillna(global_median)

        if "area" in groups:
            median_map = flat_type.map(stats["median"]).astype(float)
            iqr_map = flat_type.map(stats["iqr"]).astype(float).fillna(global_iqr)
            data["AREA_RATIO_FLAT_TYPE"] = area / median_map
            data["AREA_Z_FLAT_TYPE"] = (area - median_map) / iqr_map
            # Explicit train-only percentile rank (interpolated) inside flat type.
            ranks = {}
            for name, values in ref.groupby("ft")["area"]:
                ranks[name] = np.sort(values.dropna().to_numpy())
            pct = np.full(len(frame), np.nan)
            for name, sorted_values in ranks.items():
                mask = (flat_type == name).to_numpy() & area.notna().to_numpy()
                if not mask.any() or sorted_values.size == 0:
                    continue
                pct[mask] = np.searchsorted(sorted_values, area.to_numpy()[mask], side="right") / sorted_values.size
            data["AREA_PCTL_FLAT_TYPE"] = pd.Series(pct, index=frame.index)
            manifest.update({
                "flat_type_median_area": {str(k): float(v) for k, v in stats["median"].items()},
                "flat_type_iqr_area": {str(k): float(v) for k, v in stats["iqr"].items()},
                "global_median_area": global_median,
                "global_iqr_area": global_iqr,
                "columns": [c for c in data],
                "reference_rows": "fitting rows only",
            })

    if "flags" in groups:
        data["FLAG_MAX_FLOOR_MISSING"] = frame["MAX_FLOOR"].isna().astype(float)
        data["FLAG_BUILDING_AGE_MISSING"] = frame["BUILDING_AGE_YEARS"].isna().astype(float)
        data["FLAG_LEASE_AGE_MISSING"] = frame["YEARS_SINCE_LEASE_COMMENCEMENT"].isna().astype(float)
        data["FLAG_LEASE_NEGATIVE_AGE"] = (frame.get("LEASE_AGE_NEGATIVE_FLAG", pd.Series(0, index=frame.index)).fillna(0) > 0).astype(float)
        data["FLAG_BUILDING_NEGATIVE_AGE"] = (frame.get("BUILDING_AGE_NEGATIVE_FLAG", pd.Series(0, index=frame.index)).fillna(0) > 0).astype(float)
        data["FLAG_YEAR_COMPLETED_MISSING"] = frame["YEAR_COMPLETED"].isna().astype(float)
        manifest["flag_columns"] = [c for c in data]

    if not data:
        return pd.DataFrame(index=frame.index), manifest
    result = pd.DataFrame(data, index=frame.index)
    manifest["enabled"] = True
    manifest["missing_counts"] = {column: int(result[column].isna().sum()) for column in result.columns}
    return result, manifest


def build_static_features(
    frame: pd.DataFrame,
    split: dict,
    groups: tuple[str, ...],
) -> tuple[pd.DataFrame, dict]:
    """Concatenate the static groups requested for one ablation."""
    pieces = []
    manifest: dict = {}
    micro_groups = tuple(group for group in groups
                         if group in {"micro_region", "micro_street", "micro_block"})
    if micro_groups:
        block, block_manifest = micro_features(frame, micro_groups)
        pieces.append(block)
        manifest["micro"] = block_manifest
    if "area" in groups or "flags" in groups:
        block, block_manifest = area_features(frame, split, groups)
        pieces.append(block)
        manifest["area_flags"] = block_manifest
    if not pieces:
        return pd.DataFrame(index=frame.index), manifest
    return pd.concat(pieces, axis=1), manifest


def group_feature_names(groups: tuple[str, ...]) -> list[str]:
    """Column names a group tuple will produce, without touching the data."""
    names: list[str] = []
    if "micro_region" in groups:
        names += ["SUBZONE_ID", "PLANNING_AREA_ID", "REGION_ID"]
    if "micro_street" in groups:
        names += ["STREET_ID"]
    if "micro_block" in groups:
        names += ["BLOCK_ID"]
    if "area" in groups:
        names += ["AREA_RATIO_FLAT_TYPE", "AREA_Z_FLAT_TYPE", "AREA_PCTL_FLAT_TYPE"]
    if "flags" in groups:
        names += ["FLAG_MAX_FLOOR_MISSING", "FLAG_BUILDING_AGE_MISSING",
                  "FLAG_LEASE_AGE_MISSING", "FLAG_LEASE_NEGATIVE_AGE",
                  "FLAG_BUILDING_NEGATIVE_AGE", "FLAG_YEAR_COMPLETED_MISSING"]
    if "hist_rent" in groups:
        names += [request.name for request in history.C1_REQUESTS] + ["hist_rent_block_seen"]
    if "hist_psm" in groups:
        names += [request.name for request in history.C2_REQUESTS]
    if "hist_ft" in groups:
        names += [request.name for request in history.C3_REQUESTS] + ["hist_rentft_block_ft_seen"]
    if "win_rent" in groups:
        names += [f"win{window}_rent_{level}_mean" for window in config.MARKET_WINDOWS
                  for level in ("town_ft", "town", "flattype")] + \
                 [f"win{window}_rent_town_ft_n" for window in config.MARKET_WINDOWS]
    if "win_psm" in groups:
        names += [f"win{window}_psm_{level}_mean" for window in config.MARKET_WINDOWS
                  for level in ("town_ft", "town", "flattype")] + \
                 [f"win{window}_psm_town_ft_n" for window in config.MARKET_WINDOWS]
    return names
