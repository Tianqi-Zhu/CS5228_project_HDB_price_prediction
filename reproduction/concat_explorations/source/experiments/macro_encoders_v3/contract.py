"""Locating the real v3 artifacts.  No invented paths, no guessed schema.

Everything is read through ``manifest.json``: the manifest is the authority for
where a split/stage's ``macro`` and ``features`` arrays live, and for which policy
and contract id produced them.  This module never constructs an artifact path by
convention, so a producer-side rename cannot silently point the encoder at the
wrong file.

Units and axes are preserved as exported:

* DBS: ``(n_windows, 63, 7)`` -- trading rows, oldest first, left-padded.
* COE: ``(n_windows, 4, 12, 6)`` -- (category, month, channel), oldest month last.

The COE tensor is *reshaped* to months x category-channels when encoding, but the
categories are never concatenated along the time axis: month stays the recurrent
axis and the four categories become 24 parallel channels of that month.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

CONTRACT_VERSION = "macro-encoders-v3.0.0"

PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = PACKAGE_DIR.parents[1]

V3_REPORT_ROOT = ROOT / "reports" / "fusion_contract_v3"
REPORT_DIR = ROOT / "reports" / "macro_encoders_v3"

# Channel layout, copied from the frozen v3 contract (see CONTRACT.md there).
DBS_SYMBOL = "D05.SI"
DBS_LOOKBACK_ROWS = 63
DBS_CHANNELS = (
    "log_return", "log_volume", "vol_21", "ma_dev_21",
    "drawdown_126", "log_close", "observed",
)
DBS_SCALARS = (
    "dbs_ret_1m", "dbs_ret_3m", "dbs_ret_6m",
    "dbs_vol_21", "dbs_vol_63", "dbs_vol_126",
    "dbs_drawdown_126", "dbs_volume_rel_21",
    "dbs_n_obs_63", "dbs_n_obs_126", "dbs_gap_max_days_63", "dbs_asof_age_days",
)

COE_CATEGORIES = ("a", "b", "c", "e")
COE_LOOKBACK_MONTHS = 12
COE_CHANNELS = (
    "premium", "quota", "bids", "bids_quota_ratio", "auction_count", "observed",
)
COE_SCALARS = (
    "coe_premium_latest", "coe_premium_chg_1m", "coe_premium_chg_3m",
    "coe_ratio_latest", "coe_ratio_chg_1m", "coe_auctions_12m",
    "coe_n_months_12", "coe_incomplete_latest", "coe_premium_rel_12m",
)

# Months x (category*channel) for the COE recurrent axis.
COE_FLAT_CHANNELS = len(COE_CATEGORIES) * len(COE_CHANNELS)   # 24

HIDDEN_DIM = 32
TOKEN_DIM = 48
DEFAULT_TOKEN_DIM = TOKEN_DIM


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latest_v3_contract_dir() -> Path:
    """Newest exported v3 contract directory (by manifest mtime)."""
    candidates = sorted(V3_REPORT_ROOT.glob("fusion-contract-v3*/manifest.json"),
                        key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            f"no v3 macro contract found under {V3_REPORT_ROOT}; "
            "run experiments.fusion_contract_v3.exporter first")
    return candidates[-1].parent


def load_manifest(contract_dir: Path | None = None) -> dict:
    directory = Path(contract_dir) if contract_dir else latest_v3_contract_dir()
    path = directory / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    manifest = json.loads(path.read_text())
    for key in ("contract_id", "policy", "source_hashes", "splits"):
        if key not in manifest:
            raise ValueError(f"manifest missing {key!r}: {path}")
    manifest["_directory"] = str(directory)
    return manifest


def stage_entry(manifest: dict, split_id: str, stage: str) -> dict:
    """Resolve one stage's recorded artifacts.  Raises for unknown keys."""
    try:
        entry = manifest["splits"][split_id]["stages"][stage]
    except KeyError as error:
        raise KeyError(
            f"split/stage not present in this contract: {split_id}/{stage}; "
            f"available: {sorted(manifest['splits'])}") from error
    for key in ("artifacts",):
        if key not in entry:
            raise ValueError(f"{split_id}/{stage}: manifest entry has no {key!r}")
    for name in ("macro", "features", "samples", "encoding"):
        if name not in entry["artifacts"]:
            raise ValueError(f"{split_id}/{stage}: no {name!r} artifact recorded")
    return entry


def resolve(relative: str) -> Path:
    """Manifest paths are recorded relative to the project root."""
    return ROOT / relative
