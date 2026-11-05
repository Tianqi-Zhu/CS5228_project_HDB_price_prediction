"""Shared constants for the neural fusion experiment.

Owned by the neural pane. Reads only existing project artefacts; writes only to
``reports/neural_fusion_v1``.
"""
from __future__ import annotations

from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parent
ROOT = EXPERIMENT.parents[1]
TRAIN_SPATIAL = ROOT / "data" / "processed" / "train_spatial_features.csv"
REPORT = ROOT / "reports" / "neural_fusion_v1"

# --- static feature set: same columns as scripts/benchmark_rent_models.py ---
CAT_FEATURES = ["TOWN_CLEAN", "FLAT_TYPE_CLEAN", "FLAT_MODEL_CLEAN", "FURNISHED_CLEAN"]
BASE_NUMERIC = [
    "FLOOR_AREA_SQM", "APPROVAL_MONTH_INDEX", "APPROVAL_MONTH_SIN",
    "APPROVAL_MONTH_COS", "YEARS_SINCE_LEASE_COMMENCEMENT",
]
SPATIAL_FEATURES = [
    "LATITUDE", "LONGITUDE", "MAX_FLOOR", "BUILDING_AGE_YEARS",
    "NEAREST_OPEN_MRT_KM", "OPEN_MRT_COUNT_500M", "OPEN_MRT_COUNT_1000M",
    "OPEN_MRT_COUNT_2000M", "NEAREST_MALL_KM", "MALL_COUNT_500M",
    "MALL_COUNT_1000M", "MALL_COUNT_2000M", "NEAREST_PRIMARY_SCHOOL_KM",
    "PRIMARY_SCHOOL_COUNT_500M", "PRIMARY_SCHOOL_COUNT_1000M",
    "PRIMARY_SCHOOL_COUNT_2000M", "DISTANCE_TO_CBD_KM",
]
NUMERIC_FEATURES = BASE_NUMERIC + SPATIAL_FEATURES
INDICATOR_FEATURES = ["MAX_FLOOR", "BUILDING_AGE_YEARS"]

# Numeric features form semantically coherent tokens instead of one token per column.
# 24 loose scalars would need 24 tokens (measured at 39.5 s/epoch on this CPU, 2
# threads) and each token carries a single scalar -- a poor use of attention.  These
# groups keep the same information, are reviewable, and cost 5 tokens.
NUMERIC_GROUPS = {
    "property": ["FLOOR_AREA_SQM", "YEARS_SINCE_LEASE_COMMENCEMENT"],
    "calendar": ["APPROVAL_MONTH_INDEX", "APPROVAL_MONTH_SIN", "APPROVAL_MONTH_COS"],
    "building": ["MAX_FLOOR", "BUILDING_AGE_YEARS",
                 "MAX_FLOOR__missing", "BUILDING_AGE_YEARS__missing"],
    "access": ["NEAREST_OPEN_MRT_KM", "OPEN_MRT_COUNT_500M", "OPEN_MRT_COUNT_1000M",
               "OPEN_MRT_COUNT_2000M", "NEAREST_MALL_KM", "MALL_COUNT_500M",
               "MALL_COUNT_1000M", "MALL_COUNT_2000M", "NEAREST_PRIMARY_SCHOOL_KM",
               "PRIMARY_SCHOOL_COUNT_500M", "PRIMARY_SCHOOL_COUNT_1000M",
               "PRIMARY_SCHOOL_COUNT_2000M", "DISTANCE_TO_CBD_KM"],
    "location": ["LATITUDE", "LONGITUDE"],
}

KEY_COLS = ["TOWN_CLEAN", "FLAT_TYPE_CLEAN"]
DATE_COL = "RENT_APPROVAL_DATE"
TARGET = "MONTHLY_RENT"
ID_COL = "SOURCE_ROW_ID"

# --- splits: cutoff is the LAST fitting month; val_end the LAST validation month ---
SPLITS = {
    "orig": {"cutoff": "2024-03", "val_end": "2025-03", "val_months": 12, "val_rows": 35158},
    "early": {"cutoff": "2023-03", "val_end": "2024-03", "val_months": 12, "val_rows": 35882},
    "long": {"cutoff": "2023-11", "val_end": "2025-03", "val_months": 16, "val_rows": None},
}

HISTORY_LEN = 12
MAX_HORIZON = 16
HORIZON_BANDS = [(1, 3), (4, 8), (9, 12), (13, 16)]

# --- panel construction ---
FALLBACK_MIN_COUNT = 5
PANEL_CHANNELS = [
    "log_median_rent", "log_median_psm", "log_count",
    "fallback_level", "month_sin", "month_cos", "recency", "exact_group",
]
STANDARDISED_CHANNELS = ["log_median_rent", "log_median_psm", "log_count"]

TARGET_SCALE = 1000.0           # model optimises kSGD; report SGD
CONTEXT_DIM = 11                # 6 horizon/calendar + 5 anchor


def month_index(value: str) -> int:
    """Calendar month string 'YYYY-MM' to index with 2021-01 == 0."""
    year, month = (int(part) for part in value.split("-"))
    return (year - 2021) * 12 + month - 1


def month_label(index: int) -> str:
    return f"{2021 + index // 12:04d}-{index % 12 + 1:02d}"
