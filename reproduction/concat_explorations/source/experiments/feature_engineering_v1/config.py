"""Configuration for the CS5228 feature-engineering experiments (Task A).

Everything here is *inside* the owned directory.  The module deliberately keeps
the benchmark's model parameters unchanged so that every ablation differs from
the reproduced baseline only by the feature block under test.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- paths
CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parents[1]                      # repository root
SOURCE = ROOT / "data/processed/train_spatial_features.csv"
REPORT_DIR = ROOT / "reports/feature_engineering_v1"

TARGET = "MONTHLY_RENT"
ID_COL = "SOURCE_ROW_ID"
DATE_COL = "RENT_APPROVAL_DATE"
MONTH_INDEX_COL = "APPROVAL_MONTH_INDEX"

SEED = 42
FIXED_SEEDS = (42, 43, 44)

# ---------------------------------------------------------------- features
# Exactly the 26 predictors used by scripts/benchmark_rent_models.py.
CAT_FEATURES = ["TOWN_CLEAN", "FLAT_TYPE_CLEAN", "FLAT_MODEL_CLEAN", "FURNISHED_CLEAN"]
BASE_NUMERIC = [
    "FLOOR_AREA_SQM", "APPROVAL_MONTH_INDEX", "APPROVAL_MONTH_SIN",
    "APPROVAL_MONTH_COS", "YEARS_SINCE_LEASE_COMMENCEMENT",
]
BASE_FEATURES = CAT_FEATURES + BASE_NUMERIC
SPATIAL_FEATURES = [
    "LATITUDE", "LONGITUDE", "MAX_FLOOR", "BUILDING_AGE_YEARS",
    "NEAREST_OPEN_MRT_KM", "OPEN_MRT_COUNT_500M", "OPEN_MRT_COUNT_1000M",
    "OPEN_MRT_COUNT_2000M", "NEAREST_MALL_KM", "MALL_COUNT_500M",
    "MALL_COUNT_1000M", "MALL_COUNT_2000M", "NEAREST_PRIMARY_SCHOOL_KM",
    "PRIMARY_SCHOOL_COUNT_500M", "PRIMARY_SCHOOL_COUNT_1000M",
    "PRIMARY_SCHOOL_COUNT_2000M", "DISTANCE_TO_CBD_KM",
]
SPATIAL_BASELINE = BASE_FEATURES + SPATIAL_FEATURES

# Columns that are categorical for the model; the first four come from the
# benchmark, the rest are added by feature groups below.
EXTRA_CATEGORICAL = {
    "micro_region": ["SUBZONE", "PLANNING_AREA", "REGION"],
    "micro_street": ["STREET_CLEAN"],
    "micro_block": ["BLOCK_ID"],
}

# Feature groups.  Order matters: later ablations stack on the earlier ones.
GROUP_LABELS = {
    "A1": ("micro_region",),
    "A2": ("micro_region", "micro_street"),
    "A3": ("micro_region", "micro_street", "micro_block"),
    "B1": ("area",),
    "B2": ("area", "flags"),
    "C1": ("hist_rent",),
    "C2": ("hist_psm",),
    "C3": ("hist_ft",),
    "D1": ("win_rent",),
    "D2": ("win_psm",),
}

# Empirical-Bayes prior strength (in rows) used when shrinking a child level
# towards its parent.  One value per hierarchy level.
SMOOTHING = {"town": 10.0, "subzone": 10.0, "street": 10.0, "block": 10.0,
             "flat_type": 10.0, "town_ft": 10.0, "block_ft": 10.0}
SMOOTHING_SENSITIVITY = (3.0, 30.0)

MARKET_WINDOWS = (3, 6, 12)

# ---------------------------------------------------------------- model params
LIGHTGBM_PARAMS = {
    "n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31,
    "min_child_samples": 30, "reg_lambda": 3.0, "random_state": SEED,
    "n_jobs": 2, "verbosity": -1,
}
CATBOOST_PARAMS = {
    "iterations": 400, "depth": 6, "learning_rate": 0.05,
    "l2_leaf_reg": 3.0, "loss_function": "RMSE", "random_seed": SEED,
    "thread_count": 2, "verbose": False, "allow_writing_files": False,
}

# ---------------------------------------------------------------- splits
# approval months are 0-based from 2021-01.
FIRST_MONTH = (2021, 1)
# Every split carries an explicit validation window.  ``valid_end`` is
# mandatory: without it a split would silently validate through the end of the
# data file and the stated horizon length would be wrong.
SPLITS = {
    "main": {
        "cutoff": "2024-03", "val_start": "2024-04", "val_end": "2025-03",
        "description": "train <=2024-03, validate 2024-04..2025-03 (12 months)",
    },
    "early12": {
        "cutoff": "2023-03", "val_start": "2023-04", "val_end": "2024-03",
        "description": "train <=2023-03, validate 2023-04..2024-03 (12 months)",
    },
    "long16": {
        "cutoff": "2023-11", "val_start": "2023-12", "val_end": "2025-03",
        "description": "train <=2023-11, validate 2023-12..2025-03 (16 months)",
    },
}
BANDS = ((1, 3), (4, 8), (9, 12), (13, 16))

# Saved benchmark reference value on the main split.
BENCHMARK_RMSE = 477.2035963820846
BENCHMARK_N_VALIDATION = 35158
BENCHMARK_N_TRAIN = 114842


def month_to_index(month: str) -> int:
    """'2024-03' -> 38 (0-based from 2021-01)."""
    year, mon = (int(part) for part in month.split("-"))
    return (year - FIRST_MONTH[0]) * 12 + (mon - FIRST_MONTH[1])


def index_to_month(index: int) -> str:
    year = FIRST_MONTH[0] + index // 12
    mon = FIRST_MONTH[1] + index % 12
    return f"{year:04d}-{mon:02d}"


def split_spec(split_id: str) -> dict:
    spec: dict = dict(SPLITS[split_id])
    spec["id"] = split_id
    spec["cutoff_index"] = month_to_index(spec["cutoff"])
    spec["cutoff_month"] = spec["cutoff"]
    spec["val_start_index"] = month_to_index(spec["val_start"])
    spec["val_end_index"] = month_to_index(spec["val_end"])
    if spec["val_start_index"] != spec["cutoff_index"] + 1:
        raise ValueError(f"split {split_id}: val_start must be the month after the cutoff")
    if spec["val_end_index"] < spec["val_start_index"]:
        raise ValueError(f"split {split_id}: val_end precedes val_start")
    spec["n_validation_months_expected"] = spec["val_end_index"] - spec["cutoff_index"]
    return spec


def band_label(low: int, high: int) -> str:
    return f"{low}-{high}"
