"""Feature-matrix assembly, LightGBM/CatBoost fitting and metric computation."""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import config
import data
import features
import history

try:  # LightGBM needs libomp on macOS; imports are reported, not hidden.
    import lightgbm as lgb
    from lightgbm import LGBMRegressor
except Exception as error:  # pragma: no cover - environment dependent
    lgb = None
    LGBMRegressor = None
    LIGHTGBM_IMPORT_ERROR = repr(error)
else:
    LIGHTGBM_IMPORT_ERROR = None

try:
    import catboost
    from catboost import CatBoostRegressor
except Exception as error:  # pragma: no cover
    catboost = None
    CatBoostRegressor = None
    CATBOOST_IMPORT_ERROR = repr(error)
else:
    CATBOOST_IMPORT_ERROR = None


CATEGORICAL_COLUMNS = set(config.CAT_FEATURES) | {
    "SUBZONE_ID", "PLANNING_AREA_ID", "REGION_ID", "STREET_ID", "BLOCK_ID",
}


def build_feature_frame(
    frame: pd.DataFrame,
    split: dict,
    groups: tuple[str, ...],
    smoothing: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """All features for ``groups`` on one split, plus a construction manifest."""
    pieces: list[pd.DataFrame] = []
    manifest: dict = {"groups": list(groups), "split": split["id"]}

    static, static_manifest = features.build_static_features(frame, split, groups)
    if static.shape[1]:
        pieces.append(static)
    manifest["static"] = static_manifest

    history_groups = tuple(group for group in groups if group in history.HISTORY_REQUESTS)
    if history_groups:
        block, history_manifest = history.build_history_features(frame, split, history_groups, smoothing)
        pieces.append(block)
        manifest["history"] = history_manifest

    window_groups = tuple(group for group in groups if group in {"win_rent", "win_psm"})
    if window_groups:
        block, window_manifest = history.build_window_features(frame, split, window_groups)
        pieces.append(block)
        manifest["windows"] = window_manifest

    if pieces:
        extra = pd.concat(pieces, axis=1)
    else:
        extra = pd.DataFrame(index=frame.index)
    manifest["n_extra_columns"] = int(extra.shape[1])
    manifest["extra_columns"] = list(extra.columns)
    return extra, manifest


def feature_columns(groups: tuple[str, ...]) -> list[str]:
    names = list(config.SPATIAL_BASELINE)
    names += features.group_feature_names(groups)
    # Preserve first occurrence order, drop duplicates defensively.
    seen: set[str] = set()
    ordered = []
    for name in names:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def prepare_matrix(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Train-only median imputation; train-only categorical vocabularies."""
    train_x = train[columns].copy()
    valid_x = valid[columns].copy()
    medians: dict[str, float] = {}
    levels: dict[str, list] = {}
    for column in columns:
        if column in CATEGORICAL_COLUMNS:
            train_x[column] = train_x[column].fillna("__missing__").astype(str)
            valid_x[column] = valid_x[column].where(valid_x[column].notna(), "__missing__").astype(str)
            categories = pd.CategoricalDtype(categories=sorted(train_x[column].unique()))
            levels[column] = list(categories.categories)
            train_x[column] = train_x[column].astype(categories)
            valid_x[column] = pd.Categorical(
                valid_x[column].where(valid_x[column].isin(categories.categories)),
                categories=categories.categories,
            )
        else:
            train_x[column] = pd.to_numeric(train_x[column], errors="coerce")
            valid_x[column] = pd.to_numeric(valid_x[column], errors="coerce")
            median = train_x[column].median()
            if not np.isfinite(median):
                raise ValueError(f"no finite training median for {column}")
            medians[column] = float(median)
            train_x[column] = train_x[column].fillna(median)
            valid_x[column] = valid_x[column].fillna(median)
    state = {
        "numeric_training_medians": medians,
        "categorical_training_levels": levels,
        "categorical_columns": [c for c in columns if c in CATEGORICAL_COLUMNS],
    }
    return train_x, valid_x, state


def score(actual: np.ndarray, predicted: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    return {
        "rmse_sgd": float(np.sqrt(np.mean(np.square(error)))),
        "mae_sgd": float(np.mean(np.abs(error))),
        "bias_sgd": float(np.mean(error)),
        "n_rows": int(len(error)),
    }


def horizon_band(horizon: int) -> str:
    for low, high in config.BANDS:
        if low <= horizon <= high:
            return config.band_label(low, high)
    return "unassigned"


def monthly_metrics(
    frame: pd.DataFrame,
    mask: np.ndarray,
    split: dict,
    actual: np.ndarray,
    predicted: np.ndarray,
    model: str,
    seed: int,
) -> list[dict]:
    rows = []
    months = frame.loc[mask, config.MONTH_INDEX_COL].to_numpy()
    dates = frame.loc[mask, config.DATE_COL].to_numpy()
    for month_index in sorted(set(months.tolist())):
        in_month = months == month_index
        rows.append({
            "model": model, "seed": seed, "split_id": split["id"],
            "month_index": int(month_index),
            "month": dates[in_month][0],
            "horizon_months": int(month_index - split["cutoff_index"]),
            **score(actual[in_month], predicted[in_month]),
        })
    return rows


def band_metrics(
    frame: pd.DataFrame,
    mask: np.ndarray,
    split: dict,
    actual: np.ndarray,
    predicted: np.ndarray,
    model: str,
    seed: int,
) -> list[dict]:
    months = frame.loc[mask, config.MONTH_INDEX_COL].to_numpy()
    horizons = months - split["cutoff_index"]
    rows = []
    observed = set()
    for low, high in config.BANDS:
        band = config.band_label(low, high)
        in_band = (horizons >= low) & (horizons <= high)
        if in_band.any():
            observed.add(band)
            rows.append({
                "model": model, "seed": seed, "split_id": split["id"],
                "horizon_band": band, "band_low": low, "band_high": high,
                **score(actual[in_band], predicted[in_band]),
            })
        else:
            rows.append({
                "model": model, "seed": seed, "split_id": split["id"],
                "horizon_band": band, "band_low": low, "band_high": high,
                "rmse_sgd": np.nan, "mae_sgd": np.nan, "bias_sgd": np.nan, "n_rows": 0,
            })
    return rows


@dataclass
class RunResult:
    model: str
    split_id: str
    seed: int
    groups: list[str]
    columns: list[str]
    metrics: dict
    monthly: list[dict]
    bands: list[dict]
    predictions: pd.DataFrame
    seconds: float
    preprocessing: dict
    extra: dict = field(default_factory=dict)


def run_lightgbm(
    frame: pd.DataFrame,
    split: dict,
    groups: tuple[str, ...],
    model_name: str,
    seed: int = config.SEED,
    params: dict | None = None,
    feature_frame: pd.DataFrame | None = None,
) -> RunResult:
    if LGBMRegressor is None:
        raise RuntimeError(f"LightGBM unavailable: {LIGHTGBM_IMPORT_ERROR}")
    params = dict(params or config.LIGHTGBM_PARAMS)
    params["random_state"] = seed
    train_mask, valid_mask = data.split_mask(frame, split)
    if feature_frame is None:
        feature_frame, _ = build_feature_frame(frame, split, groups)

    working = pd.concat([frame[config.SPATIAL_BASELINE], feature_frame], axis=1)
    columns = feature_columns(groups)
    missing = [column for column in columns if column not in working.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    train_x, valid_x, state = prepare_matrix(
        working.loc[train_mask, columns], working.loc[valid_mask, columns], columns)
    y_train = frame.loc[train_mask, config.TARGET].to_numpy(dtype=float)
    y_valid = frame.loc[valid_mask, config.TARGET].to_numpy(dtype=float)

    categoricals = [column for column in columns if column in CATEGORICAL_COLUMNS]
    model = LGBMRegressor(**params)
    started = time.perf_counter()
    model.fit(train_x, y_train, categorical_feature=categoricals)
    seconds = time.perf_counter() - started

    predicted = np.asarray(model.predict(valid_x), dtype=float)
    if not np.isfinite(predicted).all():
        raise ValueError(f"{model_name} produced non-finite predictions")
    return _package(model_name, split, frame, train_mask, valid_mask, groups, columns,
                    seed, y_valid, predicted, seconds, state, params)


def run_catboost(
    frame: pd.DataFrame,
    split: dict,
    groups: tuple[str, ...],
    model_name: str,
    seed: int = config.SEED,
    params: dict | None = None,
    feature_frame: pd.DataFrame | None = None,
) -> RunResult:
    if CatBoostRegressor is None:
        raise RuntimeError(f"CatBoost unavailable: {CATBOOST_IMPORT_ERROR}")
    params = dict(params or config.CATBOOST_PARAMS)
    params["random_seed"] = seed
    train_mask, valid_mask = data.split_mask(frame, split)
    if feature_frame is None:
        feature_frame, _ = build_feature_frame(frame, split, groups)

    working = pd.concat([frame[config.SPATIAL_BASELINE], feature_frame], axis=1)
    columns = feature_columns(groups)
    train_x = working.loc[train_mask, columns].copy()
    valid_x = working.loc[valid_mask, columns].copy()
    categoricals = [column for column in columns if column in CATEGORICAL_COLUMNS]
    medians = {}
    for column in columns:
        if column in CATEGORICAL_COLUMNS:
            train_x[column] = train_x[column].fillna("__missing__").astype(str)
            valid_x[column] = valid_x[column].fillna("__missing__").astype(str)
        else:
            train_x[column] = pd.to_numeric(train_x[column], errors="coerce")
            valid_x[column] = pd.to_numeric(valid_x[column], errors="coerce")
            median = train_x[column].median()
            if not np.isfinite(median):
                raise ValueError(f"no finite training median for {column}")
            medians[column] = float(median)
            train_x[column] = train_x[column].fillna(median)
            valid_x[column] = valid_x[column].fillna(median)
    y_train = frame.loc[train_mask, config.TARGET].to_numpy(dtype=float)
    y_valid = frame.loc[valid_mask, config.TARGET].to_numpy(dtype=float)

    model = CatBoostRegressor(**params)
    started = time.perf_counter()
    model.fit(train_x, y_train, cat_features=categoricals)
    seconds = time.perf_counter() - started
    predicted = np.asarray(model.predict(valid_x), dtype=float)
    if not np.isfinite(predicted).all():
        raise ValueError(f"{model_name} produced non-finite predictions")
    state = {"numeric_training_medians": medians,
             "categorical_columns": categoricals,
             "categorical_handling": "catboost native ordered target statistics"}
    return _package(model_name, split, frame, train_mask, valid_mask, groups, columns,
                    seed, y_valid, predicted, seconds, state, params)


def _package(
    model_name: str,
    split: dict,
    frame: pd.DataFrame,
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
    groups: tuple[str, ...],
    columns: list[str],
    seed: int,
    actual: np.ndarray,
    predicted: np.ndarray,
    seconds: float,
    state: dict,
    params: dict,
) -> RunResult:
    valid_rows = frame.loc[valid_mask]
    horizon = valid_rows[config.MONTH_INDEX_COL].to_numpy() - split["cutoff_index"]
    predictions = pd.DataFrame({
        "SOURCE_ROW_ID": valid_rows[config.ID_COL].to_numpy(),
        "split_id": split["id"],
        "cutoff_month": split["cutoff_month"],
        "horizon_months": horizon,
        config.DATE_COL: valid_rows[config.DATE_COL].to_numpy(),
        "actual_rent_sgd": actual,
        "predicted_rent_sgd": predicted,
        "model": model_name,
        "seed": seed,
    })
    overall = score(actual, predicted)
    metrics = {
        "model": model_name, "split_id": split["id"], "seed": seed,
        **overall,
        "n_train": int(train_mask.sum()),
        "n_features": len(columns),
        "feature_groups": ",".join(groups),
        "seconds": round(seconds, 2),
        "parameters": json.dumps(params, sort_keys=True),
    }
    monthly = monthly_metrics(frame, valid_mask, split, actual, predicted, model_name, seed)
    bands = band_metrics(frame, valid_mask, split, actual, predicted, model_name, seed)
    return RunResult(
        model=model_name, split_id=split["id"], seed=seed, groups=list(groups),
        columns=columns, metrics=metrics, monthly=monthly, bands=bands,
        predictions=predictions, seconds=seconds, preprocessing=state,
        extra={"n_validation": int(valid_mask.sum())},
    )


def environment_manifest() -> dict:
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "lightgbm": getattr(lgb, "__version__", None) if lgb is not None else None,
        "lightgbm_import_error": LIGHTGBM_IMPORT_ERROR,
        "catboost": getattr(catboost, "__version__", None) if catboost is not None else None,
        "catboost_import_error": CATBOOST_IMPORT_ERROR,
        "threads_per_fit": config.LIGHTGBM_PARAMS["n_jobs"],
        "cpu_count": __import__("os").cpu_count(),
    }
    return payload


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n")
