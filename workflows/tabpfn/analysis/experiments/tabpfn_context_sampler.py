"""Leakage-checked retrieval of labeled TabPFN context rows.

Retrieval metadata is kept separate from the model feature matrix. No target
values are read by this module. A selection is shared by the query batch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


EARTH_KM = 6371.0088
STRATEGIES = ("S0", "S1", "S2", "S3", "S4")
POLICIES = ("historical_only", "available_labeled_pool")


@dataclass(frozen=True)
class SamplerConfig:
    context_size: int = 10_000
    tau_months: float = 6.0
    local_fraction: float = 0.40
    recent_fraction: float = 0.35
    spatial_scale_km: float = 2.0
    age_scale_months: float = 12.0
    area_scale_log: float = 0.2
    building_age_scale_years: float = 10.0
    spatial_weight: float = 1.0
    time_weight: float = 0.3
    area_weight: float = 0.5
    building_age_weight: float = 0.2
    flat_weight: float = 1.0
    town_weight: float = 0.5
    nearest_candidates_per_query: int = 256

    def __post_init__(self):
        if self.context_size < 1 or self.tau_months <= 0:
            raise ValueError("Context size and decay must be positive")
        if not 0 <= self.local_fraction <= 1 or not 0 <= self.recent_fraction <= 1:
            raise ValueError("Mixture fractions must be within [0, 1]")
        if self.local_fraction + self.recent_fraction > 1:
            raise ValueError("Mixture fractions exceed one")
        if min(self.spatial_scale_km, self.age_scale_months,
               self.area_scale_log, self.building_age_scale_years) <= 0:
            raise ValueError("Distance scales must be positive")
        if self.nearest_candidates_per_query < 1:
            raise ValueError("Nearest candidate count must be positive")


@dataclass(frozen=True)
class ContextMetadata:
    row_ids: np.ndarray
    month_ordinal: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    floor_area: np.ndarray
    building_age: np.ndarray
    flat_type: np.ndarray
    town: np.ndarray
    duplicate_group: np.ndarray
    building_age_imputation: float
    _row_index: pd.Index = field(repr=False)

    def positions(self, row_ids: np.ndarray) -> np.ndarray:
        positions = self._row_index.get_indexer(np.asarray(row_ids))
        if (positions < 0).any():
            raise ValueError("Unknown retrieval row ID")
        return positions


@dataclass(frozen=True)
class ContextSelection:
    row_ids: np.ndarray
    audit: dict


def metadata_from_features(
    features: pd.DataFrame,
    months: np.ndarray,
    *,
    duplicate_groups: np.ndarray | None = None,
    building_age_imputation: float | None = None,
) -> ContextMetadata:
    """Extract existing property metadata without mutating model features."""
    required = ("LATITUDE", "LONGITUDE", "FLOOR_AREA_SQM",
                "BUILDING_AGE_FROM_COMPLETION", "FLAT_TYPE_CLEAN", "TOWN_CLEAN")
    if any(column not in features for column in required):
        raise ValueError("Required retrieval metadata is absent from saved features")
    if not features.index.is_unique or len(features) != len(months):
        raise ValueError("Retrieval rows must have unique IDs and aligned months")
    ids = features.index.to_numpy().copy()
    dates = pd.PeriodIndex(months, freq="M").asi8.astype(np.int64)
    age = features.BUILDING_AGE_FROM_COMPLETION.to_numpy(dtype=float, na_value=np.nan)
    if building_age_imputation is None:
        building_age_imputation = float(np.nanmedian(age))
    if not np.isfinite(building_age_imputation):
        raise ValueError("No finite training building ages")
    age = np.where(np.isfinite(age), age, building_age_imputation)
    area = features.FLOOR_AREA_SQM.to_numpy(dtype=float, na_value=np.nan)
    lat = features.LATITUDE.to_numpy(dtype=float, na_value=np.nan)
    lon = features.LONGITUDE.to_numpy(dtype=float, na_value=np.nan)
    if not (np.isfinite(lat).all() and np.isfinite(lon).all()
            and np.isfinite(area).all() and (area > 0).all()):
        raise ValueError("Coordinates and positive floor area are required")
    if duplicate_groups is None:
        duplicate_groups = ids.copy()
    duplicate_groups = np.asarray(duplicate_groups)
    if len(duplicate_groups) != len(ids):
        raise ValueError("Duplicate-group IDs are not aligned")
    return ContextMetadata(
        row_ids=ids, month_ordinal=dates, latitude=lat, longitude=lon,
        floor_area=area, building_age=age,
        flat_type=features.FLAT_TYPE_CLEAN.astype("string").fillna("<missing>").to_numpy(str),
        town=features.TOWN_CLEAN.astype("string").fillna("<missing>").to_numpy(str),
        duplicate_group=duplicate_groups.copy(),
        building_age_imputation=float(building_age_imputation), _row_index=pd.Index(ids),
    )


def _uniform(available: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    if count == 0 or len(available) == 0:
        return available[:0]
    return rng.choice(available, size=min(count, len(available)), replace=False)


def _recent(
    eligible: np.ndarray, count: int, metadata: ContextMetadata,
    query_month: int, tau: float, rng: np.random.Generator,
) -> np.ndarray:
    if count == 0 or len(eligible) == 0:
        return eligible[:0]
    age = np.abs(query_month - metadata.month_ordinal[eligible]).astype(float)
    log_weights = -age / tau
    weights = np.exp(log_weights - np.max(log_weights))
    weights /= weights.sum()
    return rng.choice(eligible, size=min(count, len(eligible)), replace=False, p=weights)


def _local(
    eligible: np.ndarray, queries: np.ndarray, count: int,
    metadata: ContextMetadata, config: SamplerConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    if count == 0 or len(eligible) == 0:
        return eligible[:0]
    coords = np.deg2rad(np.column_stack((metadata.latitude[eligible], metadata.longitude[eligible])))
    qcoords = np.deg2rad(np.column_stack((metadata.latitude[queries], metadata.longitude[queries])))
    tree = BallTree(coords, metric="haversine")
    k = min(config.nearest_candidates_per_query, len(eligible))
    distance, local_indices = tree.query(qcoords, k=k)
    candidates = eligible[local_indices]
    score = (
        config.spatial_weight * (distance * EARTH_KM / config.spatial_scale_km)
        + config.time_weight *
        (np.abs(metadata.month_ordinal[candidates] - metadata.month_ordinal[queries, None])
         / config.age_scale_months)
        + config.area_weight *
        (np.abs(np.log(metadata.floor_area[candidates] / metadata.floor_area[queries, None]))
         / config.area_scale_log)
        + config.building_age_weight *
        (np.abs(metadata.building_age[candidates] - metadata.building_age[queries, None])
         / config.building_age_scale_years)
        + config.flat_weight *
        (metadata.flat_type[candidates] != metadata.flat_type[queries, None])
        + config.town_weight *
        (metadata.town[candidates] != metadata.town[queries, None])
    )
    ranked = np.take_along_axis(candidates, np.argsort(score, axis=1, kind="stable"), axis=1)
    order = rng.permutation(len(queries))
    pointers = np.zeros(len(queries), dtype=int)
    selected: list[int] = []
    used: set[int] = set()
    while len(selected) < count and np.any(pointers < k):
        progressed = False
        for anchor in order:
            while pointers[anchor] < k:
                candidate = int(ranked[anchor, pointers[anchor]])
                pointers[anchor] += 1
                if candidate not in used:
                    selected.append(candidate)
                    used.add(candidate)
                    progressed = True
                    break
            if len(selected) >= count:
                break
        if not progressed:
            break
    return np.asarray(selected, dtype=int)


def _diverse(
    eligible: np.ndarray, count: int, metadata: ContextMetadata,
    rng: np.random.Generator,
) -> np.ndarray:
    if count == 0 or len(eligible) == 0:
        return eligible[:0]
    # Use training-pool-only town/flat/year strata. Shuffle inside each stratum
    # and round-robin across strata for broad geographic, property and time coverage.
    strata = np.char.add(np.char.add(metadata.town[eligible], "|"), metadata.flat_type[eligible])
    strata = np.char.add(np.char.add(strata, "|"),
                         (metadata.month_ordinal[eligible] // 12).astype(str))
    groups: dict[str, list[int]] = {}
    for position, key in zip(eligible, strata, strict=True):
        groups.setdefault(str(key), []).append(int(position))
    keys = list(groups)
    rng.shuffle(keys)
    for key in keys:
        rng.shuffle(groups[key])
    chosen: list[int] = []
    round_index = 0
    while len(chosen) < count and keys:
        next_keys = []
        for key in keys:
            if round_index < len(groups[key]):
                chosen.append(groups[key][round_index])
                if len(chosen) == count:
                    break
                if round_index + 1 < len(groups[key]):
                    next_keys.append(key)
        keys = next_keys
        round_index += 1
    return np.asarray(chosen, dtype=int)


def select_context(
    query_row_ids: np.ndarray,
    eligible_labeled_row_ids: np.ndarray,
    metadata: ContextMetadata,
    *,
    strategy: str,
    information_policy: str,
    config: SamplerConfig,
    seed: int,
) -> ContextSelection:
    """Select one labeled context for an entire query batch, without labels."""
    if strategy not in STRATEGIES or information_policy not in POLICIES:
        raise ValueError("Unknown sampler strategy or information policy")
    started = perf_counter()
    query_ids = np.asarray(query_row_ids)
    eligible_ids = np.asarray(eligible_labeled_row_ids)
    if len(query_ids) == 0 or len(np.unique(query_ids)) != len(query_ids):
        raise ValueError("Query IDs must be nonempty and unique")
    if len(np.unique(eligible_ids)) != len(eligible_ids):
        raise ValueError("Eligible IDs must be unique")
    queries = metadata.positions(query_ids)
    eligible = metadata.positions(eligible_ids)
    cutoff = int(metadata.month_ordinal[queries].min())
    if information_policy == "historical_only":
        eligible = eligible[metadata.month_ordinal[eligible] < cutoff]
    excluded = np.isin(metadata.row_ids[eligible], query_ids)
    excluded |= np.isin(metadata.duplicate_group[eligible], metadata.duplicate_group[queries])
    eligible = eligible[~excluded]
    if len(eligible) == 0:
        raise ValueError("No eligible labeled context before the query cutoff")
    rng = np.random.default_rng(seed)
    budget = min(config.context_size, len(eligible))
    counts: dict[str, int] = {}
    if strategy == "S0":
        chosen = eligible.copy()
        counts["full_baseline"] = len(chosen)
    elif strategy == "S1":
        chosen = _uniform(eligible, budget, rng)
        counts["uniform"] = len(chosen)
    elif strategy == "S2":
        chosen = _recent(eligible, budget, metadata, cutoff, config.tau_months, rng)
        counts["recent"] = len(chosen)
    elif strategy == "S3":
        chosen = _local(eligible, queries, budget, metadata, config, rng)
        counts["local"] = len(chosen)
    else:
        local_count = round(budget * config.local_fraction)
        recent_count = round(budget * config.recent_fraction)
        diverse_count = budget - local_count - recent_count
        local = _local(eligible, queries, local_count, metadata, config, rng)
        remaining = eligible[~np.isin(eligible, local)]
        recent = _recent(remaining, recent_count, metadata, cutoff, config.tau_months, rng)
        remaining = remaining[~np.isin(remaining, recent)]
        diverse = _diverse(remaining, diverse_count, metadata, rng)
        chosen = np.concatenate((local, recent, diverse))
        counts.update(local=len(local), recent=len(recent), diverse=len(diverse))
    if strategy != "S0" and len(chosen) < budget:
        remaining = eligible[~np.isin(eligible, chosen)]
        refill = _uniform(remaining, budget - len(chosen), rng)
        chosen = np.concatenate((chosen, refill))
        counts["fallback_uniform"] = len(refill)
    if len(np.unique(chosen)) != len(chosen) or len(chosen) > (len(eligible) if strategy == "S0" else budget):
        raise AssertionError("Invalid context size or duplicate row ID")
    if not np.isin(chosen, eligible).all() or np.isin(chosen, queries).any():
        raise AssertionError("Context contains an ineligible or query row")
    if np.isin(metadata.duplicate_group[chosen], metadata.duplicate_group[queries]).any():
        raise AssertionError("Query duplicate group entered context")
    if information_policy == "historical_only" and not (metadata.month_ordinal[chosen] < cutoff).all():
        raise AssertionError("Future labeled row entered historical context")
    ids = metadata.row_ids[chosen].copy()
    audit = {
        "strategy": strategy, "information_policy": information_policy,
        "seed": seed, "query_rows": len(query_ids), "eligible_rows_after_policy": len(eligible),
        "requested_budget": config.context_size, "effective_size": len(ids),
        "cutoff_month": str(pd.Period(ordinal=cutoff, freq="M")),
        "component_counts": counts,
        "median_context_age_months": float(np.median(np.abs(cutoff - metadata.month_ordinal[chosen]))),
        "retrieval_seconds": perf_counter() - started,
    }
    return ContextSelection(row_ids=ids, audit=audit)
