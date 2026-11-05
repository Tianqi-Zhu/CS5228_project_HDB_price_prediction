"""Declarative constants for the v3 macro contract: paths, policy, channel names.

No data and no model code.  Importable by the consumer without side effects.

Package-qualified import rule
-----------------------------
The repository already contains ``experiments/neural_fusion_v1/config.py`` and
``experiments/feature_engineering_v1/config.py``, so a bare ``config`` /
``data`` / ``models`` import is ambiguous and order-dependent.  Every module here
is imported as ``experiments.fusion_contract_v3.<module>`` and the module names are
chosen to be unambiguous (``contract``, ``dbs``, ``coe``, ``windows``,
``exporter``, ``tree_control``, ``verify``).
"""

from __future__ import annotations

from pathlib import Path

CONTRACT_VERSION = "fusion-contract-v3.0.0"

PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = PACKAGE_DIR.parents[1]

# ---------------------------------------------------------------- sources
RENT_SOURCE = ROOT / "data" / "processed" / "train_spatial_features.csv"
STOCK_SOURCE = ROOT / "data" / "auxiliary" / "sg-stock-prices.csv"
COE_SOURCE = ROOT / "data" / "auxiliary" / "sg-coe-prices.csv"

# The immutable v2.1 rent contract this one references by path and hash.
V2_CONTRACT_DIR = ROOT / "reports" / "fusion_contract_v2" / "fusion-contract-v2.1.0__43867d0ba111"
V2_MANIFEST = V2_CONTRACT_DIR / "manifest.json"

REPORT_ROOT = ROOT / "reports" / "fusion_contract_v3"

# ---------------------------------------------------------------- splits
# Copied deliberately, not imported: v3 must not silently follow a future change
# to the v2 split definitions.  The verifier asserts these agree with v2.1.
SPLITS: dict[str, dict] = {
    "orig": {"cutoff": "2024-03", "val_start": "2024-04", "val_end": "2025-03"},
    "early": {"cutoff": "2023-03", "val_start": "2023-04", "val_end": "2024-03"},
    "long": {"cutoff": "2023-11", "val_start": "2023-12", "val_end": "2025-03"},
}

STAGES = ("A", "B")
POPULATIONS = ("fit", "holdout", "validation")

# ---------------------------------------------------------------- policy
POLICY_ORIGIN_ASOF = "origin_asof"
POLICY_TARGET_PREVIOUS_MONTH = "target_previous_month"
POLICIES = (POLICY_ORIGIN_ASOF, POLICY_TARGET_PREVIOUS_MONTH)
DEFAULT_POLICY = POLICY_ORIGIN_ASOF

POLICY_DESCRIPTION = {
    POLICY_ORIGIN_ASOF: (
        "macro observations stop at the end of the sample's origin month; "
        "validation uses the split cutoff, stage-A holdout the inner cutoff"
    ),
    POLICY_TARGET_PREVIOUS_MONTH: (
        "macro observations stop at the end of the month before the target month; "
        "reads later than the rent cutoff and is never strict fixed-cutoff forecasting"
    ),
}

# ---------------------------------------------------------------- DBS
DBS_SYMBOL = "D05.SI"
DBS_LOOKBACK_ROWS = 63          # trading rows in the exported window
DBS_HISTORY_ROWS = 126          # longest scalar lookback (6 months)

DBS_CHANNELS = (
    "log_return",
    "log_volume",
    "vol_21",
    "ma_dev_21",
    "drawdown_126",
    "log_close",
    "observed",
)

DBS_SCALARS = (
    "dbs_ret_1m",
    "dbs_ret_3m",
    "dbs_ret_6m",
    "dbs_vol_21",
    "dbs_vol_63",
    "dbs_vol_126",
    "dbs_drawdown_126",
    "dbs_volume_rel_21",
    "dbs_n_obs_63",
    "dbs_n_obs_126",
    "dbs_gap_max_days_63",
    "dbs_asof_age_days",
)

# Trailing windows expressed in valid trading observations, not calendar days.
DBS_RET_WINDOWS = {"dbs_ret_1m": 21, "dbs_ret_3m": 63, "dbs_ret_6m": 126}
DBS_VOL_WINDOWS = {"dbs_vol_21": 21, "dbs_vol_63": 63, "dbs_vol_126": 126}

# ---------------------------------------------------------------- COE
COE_CATEGORIES = ("a", "b", "c", "e")     # fixed order, index 0..3
COE_LOOKBACK_MONTHS = 12
COE_MIN_ROUNDS_COMPLETE = 2

COE_CHANNELS = (
    "premium",           # mean of QUOTA_PREMIUM across the month's auctions
    "quota",             # sum of QUOTA
    "bids",              # sum of BIDS_RECEIVED
    "bids_quota_ratio",  # sum(bids) / sum(quota), masked when quota == 0
    "auction_count",     # rounds present
    "observed",          # 1 = at least one auction
)

COE_SCALARS = (
    "coe_premium_latest",
    "coe_premium_chg_1m",
    "coe_premium_chg_3m",
    "coe_ratio_latest",
    "coe_ratio_chg_1m",
    "coe_auctions_12m",
    "coe_n_months_12",
    "coe_incomplete_latest",
    "coe_premium_rel_12m",
)

MONTH_ABBREVIATIONS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

# ---------------------------------------------------------------- arms
ROLE_V21_SCALARS = "v21_scalars"
ROLE_DBS_SCALARS = "dbs_scalars"
ROLE_COE_SCALARS = "coe_scalars"
ROLE_DBS_SEQUENCE = "dbs_sequence"
ROLE_COE_SEQUENCE = "coe_sequence"

ARM_INPUTS: dict[str, tuple[str, ...]] = {
    "macro_none": (ROLE_V21_SCALARS,),
    "macro_dbs": (ROLE_V21_SCALARS, ROLE_DBS_SCALARS, ROLE_DBS_SEQUENCE),
    "macro_coe": (ROLE_V21_SCALARS, ROLE_COE_SCALARS, ROLE_COE_SEQUENCE),
    "macro_both": (ROLE_V21_SCALARS, ROLE_DBS_SCALARS, ROLE_COE_SCALARS,
                   ROLE_DBS_SEQUENCE, ROLE_COE_SEQUENCE),
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
    spec["inner_cutoff_index"] = spec["cutoff_index"] - 12
    if spec["val_start_index"] != spec["cutoff_index"] + 1:
        raise ValueError(f"{split_id}: val_start must follow the cutoff")
    spec["n_validation_months"] = spec["val_end_index"] - spec["cutoff_index"]
    return spec


def dbs_scalar_names() -> list[str]:
    return list(DBS_SCALARS)


def coe_scalar_names() -> list[str]:
    """Flat names in (category, scalar) order."""
    return [f"coe_{category}_{name}" for category in COE_CATEGORIES for name in COE_SCALARS]


def arm_scalar_roles(arm: str) -> tuple[str, ...]:
    if arm not in ARM_INPUTS:
        raise KeyError(f"unknown arm {arm!r}")
    return ARM_INPUTS[arm]


def arm_uses_dbs_sequence(arm: str) -> bool:
    return ROLE_DBS_SEQUENCE in ARM_INPUTS[arm]


def arm_uses_coe_sequence(arm: str) -> bool:
    return ROLE_COE_SEQUENCE in ARM_INPUTS[arm]


def contract_identity() -> dict:
    return {
        "contract_version": CONTRACT_VERSION,
        "splits": {key: split_spec(key) for key in SPLITS},
        "policies": list(POLICIES),
        "default_policy": DEFAULT_POLICY,
        "policy_description": dict(POLICY_DESCRIPTION),
        "dbs": {
            "symbol": DBS_SYMBOL,
            "lookback_rows": DBS_LOOKBACK_ROWS,
            "history_rows": DBS_HISTORY_ROWS,
            "channels": list(DBS_CHANNELS),
            "scalars": list(DBS_SCALARS),
            "return_windows_valid_obs": dict(DBS_RET_WINDOWS),
            "vol_windows_valid_obs": dict(DBS_VOL_WINDOWS),
        },
        "coe": {
            "categories": list(COE_CATEGORIES),
            "lookback_months": COE_LOOKBACK_MONTHS,
            "min_rounds_complete": COE_MIN_ROUNDS_COMPLETE,
            "channels": list(COE_CHANNELS),
            "scalars": list(COE_SCALARS),
        },
        "arms": {arm: list(roles) for arm, roles in ARM_INPUTS.items()},
    }
