"""Read-only source checks and feature creation shared by experiments."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.config import VALIDATION_START


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_main(raw: pd.DataFrame) -> pd.DataFrame:
    """Add features to a copy while preserving all source rows and CSV files."""
    frame = raw.copy(deep=True)
    frame["FLAT_TYPE_CLEAN"] = (
        frame["FLAT_TYPE"].fillna("unknown").astype(str).str.lower().str.strip()
        .str.replace(r"^(\d+)\s*[- ]\s*room$", r"\1-room", regex=True)
    )
    frame["FLAT_MODEL_CLEAN"] = frame["FLAT_MODEL"].fillna("unknown").astype(str).str.lower().str.strip()
    frame["TOWN_CLEAN"] = frame["TOWN"].fillna("unknown").astype(str).str.lower().str.strip()
    date = pd.to_datetime(frame["RENT_APPROVAL_DATE"], format="%Y-%m", errors="coerce")
    if date.isna().any():
        raise ValueError(f"Invalid RENT_APPROVAL_DATE in {int(date.isna().sum())} rows")
    frame["APPROVAL_MONTH_INDEX"] = (date.dt.year - 2021) * 12 + date.dt.month - 1
    frame["APPROVAL_MONTH_OF_YEAR"] = date.dt.month
    frame["FLAT_AGE_YEARS"] = date.dt.year - frame["LEASE_COMMENCE_DATE"]
    frame.loc[frame["FLAT_AGE_YEARS"] < 0, "FLAT_AGE_YEARS"] = np.nan
    return frame


def temporal_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by approval month; keep every row and prevent month overlap."""
    fitting = frame[frame["RENT_APPROVAL_DATE"] < VALIDATION_START].copy()
    validation = frame[frame["RENT_APPROVAL_DATE"] >= VALIDATION_START].copy()
    if fitting.empty or validation.empty:
        raise ValueError("Temporal split has an empty fitting or validation partition")
    if fitting["RENT_APPROVAL_DATE"].max() >= validation["RENT_APPROVAL_DATE"].min():
        raise AssertionError("Fitting and validation months overlap")
    return fitting, validation


def normalized_address(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized join keys for the HDB block table."""
    result = frame.copy()
    for column in ("TOWN", "BLOCK", "STREET"):
        result[column] = (
            result[column].astype("string").str.casefold().str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )
    return result
