"""Loading and split helpers for the shared validation contract."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frame(path: Path | None = None) -> pd.DataFrame:
    """Read the spatial feature CSV, preserving original row order.

    ``SOURCE_ROW_ID`` is checked to be unique and to equal the physical row
    order, so that exported predictions can be joined back unambiguously.
    """
    path = Path(path or config.SOURCE)
    frame = pd.read_csv(path)
    required = set(config.SPATIAL_BASELINE) | {
        config.TARGET, config.ID_COL, config.DATE_COL, config.MONTH_INDEX_COL,
        "BLOCK_CLEAN", "STREET_CLEAN", "TOWN_CLEAN", "SUBZONE",
        "PLANNING_AREA", "REGION", "FLAT_TYPE_CLEAN", "FLOOR_AREA_SQM",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"source is missing columns: {sorted(missing)}")

    if not frame[config.ID_COL].is_unique:
        raise ValueError("SOURCE_ROW_ID is not unique")
    if not np.array_equal(frame[config.ID_COL].to_numpy(), np.arange(len(frame))):
        raise ValueError("SOURCE_ROW_ID does not match physical row order")

    dates = pd.to_datetime(frame[config.DATE_COL], format="%Y-%m", errors="coerce")
    if dates.isna().any():
        raise ValueError("unparseable RENT_APPROVAL_DATE values")
    derived = (dates.dt.year - config.FIRST_MONTH[0]) * 12 + dates.dt.month - config.FIRST_MONTH[1]
    if not np.array_equal(derived.to_numpy(), frame[config.MONTH_INDEX_COL].to_numpy()):
        raise ValueError("APPROVAL_MONTH_INDEX disagrees with RENT_APPROVAL_DATE")
    if frame[config.TARGET].isna().any():
        raise ValueError("missing target in labelled source")

    frame = frame.copy()
    frame[config.DATE_COL] = dates.dt.strftime("%Y-%m")
    frame["BLOCK_ID"] = block_key(frame)
    return frame


def block_key(frame: pd.DataFrame) -> pd.Series:
    """Block identity required by the plan: town + street + block."""
    parts = [frame["TOWN_CLEAN"].fillna("unknown").astype(str),
             frame["STREET_CLEAN"].fillna("unknown").astype(str),
             frame["BLOCK_CLEAN"].fillna("unknown").astype(str)]
    return parts[0].str.cat(parts[1:], sep="|")


def split_mask(frame: pd.DataFrame, split: dict) -> tuple[np.ndarray, np.ndarray]:
    """Boolean (train, validation) masks using an explicit validation window.

    Months are kept wholly on one side: month index <= cutoff trains,
    ``cutoff+1 .. val_end`` validates.  Rows after ``val_end`` belong to
    **neither** side and are excluded, so a split never silently extends to the
    end of the data file.
    """
    month = frame[config.MONTH_INDEX_COL].to_numpy()
    cutoff = int(split["cutoff_index"])
    val_start = int(split["val_start_index"])
    val_end = int(split["val_end_index"])
    if val_start != cutoff + 1:
        raise ValueError(f"split {split['id']}: validation must start immediately after the cutoff")
    train = month <= cutoff
    valid = (month >= val_start) & (month <= val_end)
    if train.sum() == 0 or valid.sum() == 0:
        raise ValueError(f"empty side for split {split['id']}")
    if month[train].max() >= month[valid].min():
        raise ValueError(f"overlapping months for split {split['id']}")
    if month[valid].max() > val_end:
        raise ValueError(f"split {split['id']}: validation extends past val_end")
    return train, valid


def ignored_mask(frame: pd.DataFrame, split: dict) -> np.ndarray:
    """Rows deliberately excluded from both sides of ``split``."""
    month = frame[config.MONTH_INDEX_COL].to_numpy()
    return month > int(split["val_end_index"])


def horizon(frame: pd.DataFrame, split: dict) -> np.ndarray:
    return frame[config.MONTH_INDEX_COL].to_numpy() - split["cutoff_index"]


def split_summary(frame: pd.DataFrame, split: dict) -> dict:
    """Realised boundaries plus a hard assertion that they match the spec."""
    month = frame[config.MONTH_INDEX_COL].to_numpy()
    train, valid = split_mask(frame, split)
    ignored = ignored_mask(frame, split)
    realised_train = (int(month[train].min()), int(month[train].max()))
    realised_valid = (int(month[valid].min()), int(month[valid].max()))
    expected_valid = (int(split["val_start_index"]), int(split["val_end_index"]))
    if realised_train[1] != int(split["cutoff_index"]):
        raise AssertionError(f"{split['id']}: training does not end at the cutoff")
    if realised_valid != expected_valid:
        raise AssertionError(
            f"{split['id']}: validation window {realised_valid} != declared {expected_valid}")
    n_months = realised_valid[1] - realised_valid[0] + 1
    if n_months != int(split["n_validation_months_expected"]):
        raise AssertionError(f"{split['id']}: {n_months} validation months != "
                             f"{split['n_validation_months_expected']}")
    return {
        "split_id": split["id"],
        "cutoff_month": config.index_to_month(split["cutoff_index"]),
        "cutoff_index": int(split["cutoff_index"]),
        "val_start_month": config.index_to_month(expected_valid[0]),
        "val_end_month": config.index_to_month(expected_valid[1]),
        "train_months": [config.index_to_month(realised_train[0]), config.index_to_month(realised_train[1])],
        "validation_months": [config.index_to_month(realised_valid[0]), config.index_to_month(realised_valid[1])],
        "n_train": int(train.sum()),
        "n_validation": int(valid.sum()),
        "n_validation_months": int(n_months),
        "n_excluded_after_val_end": int(ignored.sum()),
        "description": split["description"],
    }
