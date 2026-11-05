"""Read-only source checks and feature creation shared by experiments."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from analysis.pipeline.config import DATE, FEATURE_SETS, TARGET, VALIDATION_START
from analysis.pipeline.models import build_transformer


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


def temporal_split(
    frame: pd.DataFrame, validation_start: str = VALIDATION_START,
    fit_window_months: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by approval month; keep every row and prevent month overlap."""
    dates = pd.to_datetime(frame[DATE], format="%Y-%m", errors="raise")
    cutoff = pd.to_datetime(validation_start, format="%Y-%m", errors="raise")
    fit_mask = dates < cutoff
    if fit_window_months is not None:
        if fit_window_months < 1:
            raise ValueError("fit_window_months must be positive")
        fit_mask &= dates >= cutoff - pd.DateOffset(months=fit_window_months)
    fitting = frame.loc[fit_mask].copy()
    validation = frame.loc[dates >= cutoff].copy()
    if fitting.empty or validation.empty:
        raise ValueError("Temporal split has an empty fitting or validation partition")
    if fitting["RENT_APPROVAL_DATE"].max() >= validation["RENT_APPROVAL_DATE"].min():
        raise AssertionError("Fitting and validation months overlap")
    return fitting, validation


def load_main(path: Path, *, require_target: bool = False) -> pd.DataFrame:
    """Read and validate a main CSV without dropping or reordering rows."""
    frame = pd.read_csv(path, dtype={"BLOCK": "string"})
    required = [DATE, "TOWN", "BLOCK", "STREET", "FLAT_TYPE", "FLAT_MODEL",
                "FLOOR_AREA_SQM", "LEASE_COMMENCE_DATE"]
    if require_target:
        required.append(TARGET)
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{path.name}: no rows")
    if require_target:
        target = pd.to_numeric(frame[TARGET], errors="coerce")
        if not np.isfinite(target.to_numpy(dtype=float)).all():
            raise ValueError(f"{path.name}: invalid target values")
    return frame


@contextmanager
def protect_sources(paths):
    """Check source bytes even if a run exits with an exception."""
    paths = [Path(p) for p in paths]
    before = {str(p): sha256(p) for p in paths}
    try:
        yield before
    finally:
        after = {str(p): sha256(p) for p in paths}
        if after != before:
            raise AssertionError("A source CSV changed during the run")


def check_output_path(path: Path) -> Path:
    """Never let a CLI output overwrite original data or repository code."""
    from analysis.pipeline.config import DATA, ROOT
    resolved = Path(path).resolve()
    data_root = DATA.resolve()
    if resolved == data_root or data_root in resolved.parents:
        raise ValueError("Outputs cannot be written inside data/")
    protected = [ROOT / name for name in ("analysis", ".git")]
    if any(resolved == p or p in resolved.parents for p in protected):
        raise ValueError("Outputs cannot overwrite code or Git metadata")
    return resolved


def normalized_address(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized join keys for the HDB block table."""
    result = frame.copy()
    for column in ("TOWN", "BLOCK", "STREET"):
        result[column] = norm_block(result[column]) if column == "BLOCK" else norm_text(result[column])
    return result


def norm_text(series: pd.Series) -> pd.Series:
    return (series.astype("string").str.casefold().str.strip()
            .str.replace(r"\s+", " ", regex=True).replace({"<na>": pd.NA, "": pd.NA}))


def norm_flat_type(series: pd.Series) -> pd.Series:
    """EDA display labels use spaces; historical model labels use hyphens."""
    return norm_text(series).str.replace(r"\s*-\s*", " ", regex=True).str.replace(r"\s+", " ", regex=True)


def norm_block(series: pd.Series) -> pd.Series:
    return norm_text(series).str.replace(r"\s+", "", regex=True)


@dataclass
class PreprocessorConfig:
    """Configuration for deterministic, row-preserving feature creation."""

    target_col: str = TARGET
    date_col: str | None = DATE
    feature_set: str = "C_property_time_town"


class HDBPreprocessor:
    """Clean a data copy and derive A/B/C features without fitting encoders."""

    def __init__(self, config: PreprocessorConfig | None = None):
        self.config = config or PreprocessorConfig()

    def load_data(self, path: str | Path) -> pd.DataFrame:
        path = Path(path)
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        raise ValueError(f"Unsupported file type: {path.suffix}")

    def basic_eda(self, df: pd.DataFrame) -> dict:
        return {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isna().sum().to_dict(),
            "duplicates": int(df.duplicated().sum()),
            "numeric_summary": df.describe(include="number").to_dict(),
        }

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy(deep=True)
        result.columns = [
            re.sub(r"[^\w]+", "_", str(column).strip()).upper()
            for column in result.columns
        ]
        if result.columns.duplicated().any():
            raise ValueError("Column normalization produced duplicate names")
        return result

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy(deep=True)
        date_col = self.config.date_col
        if date_col and date_col.upper() != DATE and date_col.upper() in result:
            result[DATE] = result[date_col.upper()]
        required = {DATE, "FLAT_TYPE", "FLAT_MODEL", "TOWN", "LEASE_COMMENCE_DATE"}
        if not required.issubset(result.columns):
            raise ValueError(f"Missing feature source columns: {sorted(required - set(result.columns))}")
        return prepare_main(result)

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        return self.engineer_features(self.clean(raw))

    def split_features_target(self, df: pd.DataFrame):
        target = self.config.target_col.upper()
        if target not in df:
            raise KeyError(f"Target column '{target}' not found")
        columns = FEATURE_SETS[self.config.feature_set]
        return df[columns].copy(), df[target].copy()

    @staticmethod
    def infer_feature_types(X: pd.DataFrame):
        numeric = X.select_dtypes(include="number").columns.tolist()
        categorical = X.select_dtypes(
            include=["object", "string", "category", "bool"]
        ).columns.tolist()
        return numeric, categorical

    def build_transformer(self, X: pd.DataFrame) -> ColumnTransformer:
        numeric, categorical = self.infer_feature_types(X)
        return build_transformer(numeric, categorical, scale_numeric=True)

    def prepare(self, df: pd.DataFrame):
        X, y = self.split_features_target(self.transform(df))
        return X, y, self.build_transformer(X)
