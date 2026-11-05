"""Frozen Gate-1 comparison contract: splits, feature sets, array schema.

This module is the single source of truth that both the tree controls (here) and
the neural controls (``experiments/neural_fusion_*``) consume.  It contains no
data and no model code, only declarative constants plus small pure helpers, so it
can be imported by either pane without side effects.

Package-qualified import rule
-----------------------------
The repository already contains ``experiments/neural_fusion_v1/config.py`` and
``experiments/feature_engineering_v1/config.py``.  Importing bare ``config`` /
``data`` / ``models`` is therefore ambiguous and order-dependent.  Every module in
this package is imported as ``experiments.fusion_contract_v2.<module>`` and the
package's own modules are named to be unambiguous (``contract``, ``panel``,
``export``, ``tree_control``).  Run modules with ``python -m`` from the project
root so only the project root is on ``sys.path``.
"""

from __future__ import annotations

from pathlib import Path

CONTRACT_VERSION = "fusion-contract-v2.1.0"

PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = PACKAGE_DIR.parents[1]
SOURCE = ROOT / "data" / "processed" / "train_spatial_features.csv"
REPORT_ROOT = ROOT / "reports" / "fusion_contract_v2"

TARGET = "MONTHLY_RENT"
ID_COL = "SOURCE_ROW_ID"
DATE_COL = "RENT_APPROVAL_DATE"
MONTH_COL = "APPROVAL_MONTH_INDEX"

# ---------------------------------------------------------------- splits
# ``cutoff`` is the LAST fitting month.  ``val_start``/``val_end`` bound the
# validation window explicitly; rows after ``val_end`` belong to neither side.
SPLITS: dict[str, dict] = {
    "orig": {
        "cutoff": "2024-03", "val_start": "2024-04", "val_end": "2025-03",
        "description": "train <=2024-03, validate 2024-04..2025-03 (12 months)",
    },
    "early": {
        "cutoff": "2023-03", "val_start": "2023-04", "val_end": "2024-03",
        "description": "train <=2023-03, validate 2023-04..2024-03 (12 months)",
    },
    "long": {
        "cutoff": "2023-11", "val_start": "2023-12", "val_end": "2025-03",
        "description": "train <=2023-11, validate 2023-12..2025-03 (16 months)",
    },
}

# Stage-A (inner early-stopping) holdout length in months, carved out of the
# fitting period.  Stage A fits on months <= cutoff - INNER_HOLDOUT_MONTHS and
# evaluates on the months in between; stage B refits on all months <= cutoff.
INNER_HOLDOUT_MONTHS = 12

# ---------------------------------------------------------------- sampling
# Sample generation must be reproducible independently of any model seed: the
# horizon for a training row is drawn once here, frozen into the exported table,
# and reused by every model.
SAMPLE_SEED = 20261024
MAX_HORIZON = 16
HISTORY_LEN = 12

# ---------------------------------------------------------------- history levels
# The four levels the plan requires, in fallback order (sparse -> broad).
HISTORY_LEVELS = ("global", "flat_type", "town_ft", "block_ft")

# Empirical-Bayes prior strength in rows, one value per level.
PRIOR_ROWS = {"flat_type": 10.0, "town_ft": 10.0, "block_ft": 10.0}

# Statistics exported per level.  ``mean`` columns are the smoothed values; the
# rest are support/coverage so a consumer can tell a real observation from a
# fallback.
HISTORY_STATS = ("rent", "psm", "n", "months", "recency", "exact")

PANEL_CHANNELS = (
    "log_mean_rent", "log_mean_psm", "log_count",
    "month_sin", "month_cos", "observed",
)

# ---------------------------------------------------------------- static block
# Target-independent inputs, identical for every model in the comparison.
CATEGORICAL_COLUMNS = (
    "TOWN_CLEAN", "FLAT_TYPE_CLEAN", "FLAT_MODEL_CLEAN", "FURNISHED_CLEAN",
)
STATIC_NUMERIC_COLUMNS = (
    "FLOOR_AREA_SQM", "YEARS_SINCE_LEASE_COMMENCEMENT",
    "LATITUDE", "LONGITUDE", "MAX_FLOOR", "BUILDING_AGE_YEARS",
    "NEAREST_OPEN_MRT_KM", "OPEN_MRT_COUNT_500M", "OPEN_MRT_COUNT_1000M",
    "OPEN_MRT_COUNT_2000M", "NEAREST_MALL_KM", "MALL_COUNT_500M",
    "MALL_COUNT_1000M", "MALL_COUNT_2000M", "NEAREST_PRIMARY_SCHOOL_KM",
    "PRIMARY_SCHOOL_COUNT_500M", "PRIMARY_SCHOOL_COUNT_1000M",
    "PRIMARY_SCHOOL_COUNT_2000M", "DISTANCE_TO_CBD_KM",
)
# Missing-indicator columns that belong to the static block (not to history).
STATIC_FLAG_COLUMNS = ("MAX_FLOOR__missing", "BUILDING_AGE_YEARS__missing")

# ---------------------------------------------------------------- context block
# Horizon-aware, causally derived from the origin.  ``APPROVAL_MONTH_INDEX`` is
# NOT a static column here: the calendar enters as target/origin month so that a
# h=12 sample cannot masquerade as a h=1 sample.
CONTEXT_COLUMNS = (
    "horizon_months",
    "target_month_index",
    "origin_month_index",
    "target_month_sin", "target_month_cos",
    "origin_month_sin", "origin_month_cos",
)
ANCHOR_COLUMNS = (
    "anchor_rent_sgd", "anchor_psm_sgd", "anchor_n", "anchor_level_code",
    "anchor_missing",
)

# ---------------------------------------------------------------- feature roles
# One place that says which columns each arm may read.  Consumers must not build
# their own subset: parity is checked against these tuples.
ROLE_STATIC = "static"
ROLE_CONTEXT = "context"
ROLE_HISTORY = "history"
ROLE_SEQUENCE = "sequence"
ROLES = (ROLE_STATIC, ROLE_CONTEXT, ROLE_HISTORY, ROLE_SEQUENCE)

ARM_INPUTS: dict[str, tuple[str, ...]] = {
    # Static + context only: no historical rent information at all.
    "aligned_static": (ROLE_STATIC, ROLE_CONTEXT),
    # Adds the four-level smoothed history, same origin, same rows.
    "aligned_history": (ROLE_STATIC, ROLE_CONTEXT, ROLE_HISTORY),
    # Byte-identical scalar inputs to aligned_history plus the chronological
    # tensor.  Only sequence models may read ROLE_SEQUENCE.
    "aligned_sequence": (ROLE_STATIC, ROLE_CONTEXT, ROLE_HISTORY, ROLE_SEQUENCE),
}

STAGES = ("A", "B")
STAGE_DESCRIPTION = {
    "A": "inner fit (months <= cutoff - INNER_HOLDOUT_MONTHS) and inner holdout",
    "B": "outer fit (months <= cutoff) and outer validation",
}


# ---------------------------------------------------------------- helpers
def month_index(value: str) -> int:
    """'YYYY-MM' -> months since 2021-01 (index 0)."""
    year, month = (int(part) for part in value.split("-"))
    return (year - 2021) * 12 + month - 1


def month_label(index: int) -> str:
    if index < 0:
        return f"pre-2021-01({index})"
    return f"{2021 + index // 12:04d}-{index % 12 + 1:02d}"


def split_spec(split_id: str) -> dict:
    spec: dict = dict(SPLITS[split_id])
    spec["id"] = split_id
    spec["cutoff_index"] = month_index(spec["cutoff"])
    spec["val_start_index"] = month_index(spec["val_start"])
    spec["val_end_index"] = month_index(spec["val_end"])
    spec["inner_cutoff_index"] = spec["cutoff_index"] - INNER_HOLDOUT_MONTHS
    if spec["val_start_index"] != spec["cutoff_index"] + 1:
        raise ValueError(f"{split_id}: val_start must follow the cutoff")
    if spec["val_end_index"] < spec["val_start_index"]:
        raise ValueError(f"{split_id}: val_end precedes val_start")
    if spec["inner_cutoff_index"] < 0:
        raise ValueError(f"{split_id}: fitting period shorter than the inner holdout")
    spec["n_validation_months"] = spec["val_end_index"] - spec["cutoff_index"]
    return spec


def history_columns() -> tuple[str, ...]:
    """Every history scalar, in a fixed order: level-major, stat-minor."""
    return tuple(f"hist_{stat}__{level}" for level in HISTORY_LEVELS for stat in HISTORY_STATS)


def history_column(level: str, stat: str) -> str:
    if level not in HISTORY_LEVELS:
        raise KeyError(level)
    if stat not in HISTORY_STATS:
        raise KeyError(stat)
    return f"hist_{stat}__{level}"


def static_columns() -> tuple[str, ...]:
    return tuple(CATEGORICAL_COLUMNS) + tuple(STATIC_NUMERIC_COLUMNS) + tuple(STATIC_FLAG_COLUMNS)


def context_columns() -> tuple[str, ...]:
    return tuple(CONTEXT_COLUMNS) + tuple(ANCHOR_COLUMNS)


def columns_for_roles(roles) -> list[str]:
    """Column names for a set of roles, deduplicated, order-stable."""
    out: list[str] = []
    for role in ROLES:
        if role not in roles:
            continue
        if role == ROLE_STATIC:
            out.extend(static_columns())
        elif role == ROLE_CONTEXT:
            out.extend(context_columns())
        elif role == ROLE_HISTORY:
            out.extend(history_columns())
        elif role == ROLE_SEQUENCE:
            continue  # arrays, not scalar columns
    seen, ordered = set(), []
    for name in out:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def arm_scalar_columns(arm: str) -> list[str]:
    if arm not in ARM_INPUTS:
        raise KeyError(f"unknown arm {arm!r}")
    return columns_for_roles(ARM_INPUTS[arm])


def arm_uses_sequence(arm: str) -> bool:
    return ROLE_SEQUENCE in ARM_INPUTS[arm]


# Sample-table columns that are bookkeeping, never model inputs.
SAMPLE_KEY_COLUMNS = (
    "SOURCE_ROW_ID", "split_id", "stage", "population",
    "target_month_index", "origin_month_index", "horizon_months", "actual_rent_sgd",
)
POPULATIONS = ("fit", "holdout", "validation")


def contract_identity() -> dict:
    """Declarative fingerprint of the contract itself (no data involved)."""
    return {
        "contract_version": CONTRACT_VERSION,
        "splits": {key: split_spec(key) for key in SPLITS},
        "inner_holdout_months": INNER_HOLDOUT_MONTHS,
        "sample_seed": SAMPLE_SEED,
        "max_horizon": MAX_HORIZON,
        "history_len": HISTORY_LEN,
        "history_levels": list(HISTORY_LEVELS),
        "history_stats": list(HISTORY_STATS),
        "prior_rows": dict(PRIOR_ROWS),
        "panel_channels": list(PANEL_CHANNELS),
        "arms": {arm: list(roles) for arm, roles in ARM_INPUTS.items()},
        "history_columns": list(history_columns()),
        "static_columns": list(static_columns()),
        "context_columns": list(context_columns()),
    }
