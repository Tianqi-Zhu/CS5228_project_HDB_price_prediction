"""Rental surfaces -> PCA -> factor forecasts -> CatBoost hybrids.

All market features used for fitting rows are one-month-ahead forecasts from
earlier labels. Validation forecasts are made once at each fold origin and are
never updated with validation rents.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import catboost
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from analysis.config import AUX, DATA, REPORTS
from analysis.data import sha256
from analysis.evaluation import metrics
from analysis.experiments.coarse_rental_surface import CELL, GRID_SIZE, assign_cells
from analysis.experiments.dynamic_market_index import (
    FEATURES, REFERENCE_CATEGORICAL, REFERENCE_NUMERIC, TREE_SETTINGS,
    reference_model,
)
from analysis.experiments.spatiotemporal_hybrid import (
    FOLDS, MONTH, TARGET, historical_features, model_inputs, spatial_frame,
)


OUT = REPORTS / "factored_rental_surface"
TYPES = ("3-room", "4-room", "5-room", "executive")
LATE_FOLD = ("2023-12", "2025-04")
PCA_SIZES = (3, 5, 8)
FACTOR_METHODS = ("persistence", "ar1", "state_space")
HYBRID_COMPONENTS = 5
SMOOTH_KM = 1.0
SMOOTH_PRIOR_COUNT = 10.0
MARKET_COLUMNS = ["FORECAST_LOCAL_MARKET_RENT", "MARKET_CONFIDENCE"]


@dataclass
class SurfacePanel:
    rows: pd.DataFrame
    months: pd.PeriodIndex
    cells: pd.DataFrame
    values: np.ndarray  # month × (type × cell), rent in SGD
    raw_sum: np.ndarray  # month × type × cell, residual SGD
    raw_count: np.ndarray  # month × type × cell
    effective_count: np.ndarray  # month × type × cell
    base_by_type: dict[str, float]


def _distance_km(coords: np.ndarray) -> np.ndarray:
    radians = np.radians(coords)
    lat, lon = radians[:, 0][:, None], radians[:, 1][:, None]
    a = np.sin((lat - lat.T) / 2) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin((lon - lon.T) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def make_panel(frame: pd.DataFrame, blocks: pd.DataFrame) -> SurfacePanel:
    """Build HDB-supported, composition-adjusted monthly rental maps."""
    bounds = (float(blocks.LATITUDE.min()), float(blocks.LATITUDE.max()),
              float(blocks.LONGITUDE.min()), float(blocks.LONGITUDE.max()))
    block_grid = assign_cells(blocks, bounds)
    cells = (block_grid.groupby(CELL)[["LATITUDE", "LONGITUDE"]].median()
             .reset_index().sort_values(CELL).reset_index(drop=True))
    cell_index = pd.Series(np.arange(len(cells)), index=cells[CELL].to_numpy())
    rows = assign_cells(frame, bounds)
    rows["_CELL_INDEX"] = rows[CELL].map(cell_index).astype(int)
    reference = reference_model(rows)
    ref_columns = REFERENCE_CATEGORICAL + REFERENCE_NUMERIC
    rows["_RESIDUAL"] = rows[TARGET].to_numpy(float) - reference.predict(rows[ref_columns])
    first_year = rows[rows[MONTH] < pd.Period("2022-01", freq="M")]
    base_by_type = first_year.groupby("FLAT_TYPE_CLEAN")[TARGET].median().to_dict()
    base_by_type["_fallback"] = float(first_year[TARGET].median())
    kernel = np.exp(-0.5 * (_distance_km(cells[["LATITUDE", "LONGITUDE"]].to_numpy(float)) / SMOOTH_KM) ** 2)
    kernel[kernel < 0.01] = 0.0
    months = pd.period_range(rows[MONTH].min(), rows[MONTH].max(), freq="M")
    shape = (len(months), len(TYPES), len(cells))
    sums = np.zeros(shape)
    counts = np.zeros(shape)
    effective = np.zeros(shape)
    surface = np.zeros(shape)
    for m_index, month in enumerate(months):
        month_rows = rows[rows[MONTH] == month]
        month_fallback = float(month_rows["_RESIDUAL"].mean())
        for t_index, kind in enumerate(TYPES):
            part = month_rows[month_rows["FLAT_TYPE_CLEAN"] == kind]
            index = part["_CELL_INDEX"].to_numpy(int)
            residual = part["_RESIDUAL"].to_numpy(float)
            sums[m_index, t_index] = np.bincount(index, weights=residual, minlength=len(cells))
            counts[m_index, t_index] = np.bincount(index, minlength=len(cells))
            parent = float(residual.mean()) if len(residual) else month_fallback
            effective[m_index, t_index] = kernel @ counts[m_index, t_index]
            smooth = ((kernel @ sums[m_index, t_index] + SMOOTH_PRIOR_COUNT * parent) /
                      (effective[m_index, t_index] + SMOOTH_PRIOR_COUNT))
            surface[m_index, t_index] = base_by_type[kind] + smooth
    return SurfacePanel(rows, months, cells, surface.reshape(len(months), -1),
                        sums, counts, effective, base_by_type)


def _state_space_forecast(z: np.ndarray, horizons: np.ndarray) -> np.ndarray:
    """Independent damped local-linear-trend Kalman filters for PCA factors."""
    answer = np.empty((len(horizons), z.shape[1]))
    transition = np.array([[1.0, 1.0], [0.0, 0.7]])
    observation = np.array([1.0, 0.0])
    for factor in range(z.shape[1]):
        series = z[:, factor]
        scale = max(float(np.var(np.diff(series))), 1e-4)
        q = np.diag([0.20 * scale, 0.025 * scale])
        r = 0.25 * scale
        state = np.array([series[0], 0.0])
        covariance = np.diag([max(float(np.var(series)), scale), scale])
        for observation_value in series:
            state = transition @ state
            covariance = transition @ covariance @ transition.T + q
            innovation = observation_value - observation @ state
            denominator = float(observation @ covariance @ observation + r)
            gain = covariance @ observation / denominator
            state += gain * innovation
            covariance = (np.eye(2) - np.outer(gain, observation)) @ covariance
        for pos, horizon in enumerate(horizons):
            answer[pos, factor] = (np.linalg.matrix_power(transition, int(horizon)) @ state)[0]
    return answer


def forecast_factors(z: np.ndarray, horizons: np.ndarray, method: str) -> np.ndarray:
    """Forecast fitted PCA scores without seeing target months."""
    horizons = np.asarray(horizons, int)
    if len(z) < 3 or (horizons < 1).any():
        raise ValueError("Need factor history and positive forecast horizons")
    if method == "persistence":
        return np.repeat(z[-1:, :], len(horizons), axis=0)
    if method == "state_space":
        return _state_space_forecast(z, horizons)
    if method != "ar1":
        raise ValueError(method)
    model = Ridge(alpha=max(10.0, len(z) / 2.0))
    model.fit(z[:-1], z[1:])
    state = z[-1].copy()
    result = []
    for _ in range(int(horizons.max())):
        state = model.predict(state[None, :])[0]
        result.append(state.copy())
    return np.asarray(result)[horizons - 1]


def forecast_surfaces(panel: SurfacePanel, origin_index: int, horizons: np.ndarray,
                      components: int, method: str) -> tuple[np.ndarray, float, float]:
    """Fit PCA through the origin and decode future factor forecasts."""
    if origin_index < components + 2 or origin_index + int(np.max(horizons)) > len(panel.months):
        raise ValueError("Invalid PCA fitting window or forecast horizon")
    fitting = panel.values[:origin_index]
    pca = PCA(n_components=components, svd_solver="full")
    factors = pca.fit_transform(fitting)
    reconstructed = pca.inverse_transform(factors)
    latent = forecast_factors(factors, horizons, method)
    future = pca.inverse_transform(latent)
    return future, float(pca.explained_variance_ratio_.sum()), float(np.sqrt(np.mean((fitting - reconstructed) ** 2)))


def market_for_rows(panel: SurfacePanel, rows: pd.DataFrame,
                    forecast: np.ndarray, origin_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Look up forecasts at transactions; use 2021 type medians for rare types."""
    positions = rows[MONTH].astype("int64").to_numpy() - panel.months[origin_index].ordinal
    if (positions < 0).any() or (positions >= len(forecast)).any():
        raise AssertionError("A row lies outside the surface forecast horizon")
    cells = rows["_CELL_INDEX"].to_numpy(int)
    kinds = rows["FLAT_TYPE_CLEAN"].to_numpy()
    type_index = {kind: index for index, kind in enumerate(TYPES)}
    output = np.empty(len(rows))
    confidence = np.zeros(len(rows))
    for pos, kind in enumerate(kinds):
        index = type_index.get(kind)
        if index is None:
            # Rare 1- and 2-room flats retain a type-specific anchor and the
            # mean forecast market movement across the common types.
            output[pos] = panel.base_by_type.get(kind, panel.base_by_type["_fallback"]) + (
                forecast[positions[pos]].mean() - panel.values[0].mean()
            )
        else:
            output[pos] = forecast[positions[pos], index * len(panel.cells) + cells[pos]]
            confidence[pos] = np.log1p(panel.effective_count[origin_index - 1, index, cells[pos]])
    return output, confidence


def out_of_time_market(panel: SurfacePanel, components: int, method: str) -> pd.DataFrame:
    """One-step forecast feature for each fitting month after the 2021 reference."""
    output = pd.DataFrame(index=panel.rows.index,
                          columns=MARKET_COLUMNS, dtype=float)
    for month_index in range(12, len(panel.months)):
        prediction, _, _ = forecast_surfaces(panel, month_index, np.array([1]), components, method)
        mask = panel.rows[MONTH].eq(panel.months[month_index])
        market, confidence = market_for_rows(panel, panel.rows.loc[mask], prediction, month_index)
        output.loc[mask, MARKET_COLUMNS[0]] = market
        output.loc[mask, MARKET_COLUMNS[1]] = confidence
    return output


def fit_catboost(x_fit: pd.DataFrame, target: np.ndarray,
                 x_future: pd.DataFrame) -> tuple[np.ndarray, float]:
    train, future = x_fit.copy(), x_future.copy()
    categories = [column for column in train if train[column].dtype.name == "category"]
    for column in categories:
        train[column] = train[column].astype("string").fillna("__missing__").astype(str)
        future[column] = future[column].astype("string").fillna("__missing__").astype(str)
    start = time.perf_counter()
    model = catboost.CatBoostRegressor(**TREE_SETTINGS["catboost"])
    model.fit(train, target, cat_features=categories)
    return np.asarray(model.predict(future), float), time.perf_counter() - start


def evaluate() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train_path = DATA / "train.csv"
    blocks = pd.read_csv(AUX / "sg-hdb-block.csv")
    panel = make_panel(spatial_frame(pd.read_csv(train_path)), blocks)
    oof = out_of_time_market(panel, HYBRID_COMPONENTS, "state_space")
    saved = pd.read_csv(REPORTS / "dynamic_market_index" / "validation_predictions.csv")
    surface_scores, fold_scores, pred_rows, timings = [], [], [], []
    for start_text, end_text in [*FOLDS, LATE_FOLD]:
        start, end = pd.Period(start_text, freq="M"), pd.Period(end_text, freq="M")
        origin = int(np.flatnonzero(panel.months == start)[0])
        horizons = np.arange(1, end.ordinal - start.ordinal + 1)
        for size in PCA_SIZES:
            for method in FACTOR_METHODS:
                forecasts, variance, reconstruction = forecast_surfaces(panel, origin, horizons, size, method)
                for offset, month in enumerate(panel.months[origin:origin + len(horizons)]):
                    raw = panel.raw_count[origin + offset]
                    mean = np.divide(panel.raw_sum[origin + offset], raw,
                                     out=np.full_like(raw, np.nan), where=raw > 0)
                    anchor = np.array([panel.base_by_type[kind] for kind in TYPES])[:, None]
                    observed = anchor + mean
                    valid = raw >= 5
                    if valid.any():
                        error = forecasts[offset].reshape(raw.shape)[valid] - observed[valid]
                        surface_scores.append({
                            "validation_start": start_text, "month": str(month), "components": size,
                            "forecaster": method, "observed_cells": int(valid.sum()),
                            "weighted_surface_rmse_sgd": float(np.sqrt(np.average(error ** 2, weights=raw[valid]))),
                            "pca_explained_variance": variance, "reconstruction_rmse_sgd": reconstruction,
                        })
        fitting = panel.rows[panel.rows[MONTH] < start].copy()
        validation = panel.rows[(panel.rows[MONTH] >= start) & (panel.rows[MONTH] < end)].copy()
        training = fitting[fitting[MONTH] >= pd.Period("2022-01", freq="M")].copy()
        training_market = oof.loc[training.index]
        if training_market[MARKET_COLUMNS].isna().any().any():
            raise AssertionError("Missing one-step out-of-time training market feature")
        forecast, _, _ = forecast_surfaces(panel, origin, horizons, HYBRID_COMPONENTS, "state_space")
        market, confidence = market_for_rows(panel, validation, forecast, origin)
        train_hist, val_hist = historical_features(fitting, validation)
        training = training.join(train_hist.loc[training.index]).join(training_market)
        validation = validation.join(val_hist)
        validation[MARKET_COLUMNS[0]] = market
        validation[MARKET_COLUMNS[1]] = confidence
        x_fit, x_val = model_inputs(training, validation, FEATURES + MARKET_COLUMNS)
        subset_pred, sec = fit_catboost(x_fit[FEATURES], training[TARGET].to_numpy(float), x_val[FEATURES])
        timings.append({"validation_start": start_text, "model": "catboost_2022plus", "seconds": sec})
        direct_pred, sec = fit_catboost(x_fit, training[TARGET].to_numpy(float), x_val)
        timings.append({"validation_start": start_text, "model": "direct_hybrid", "seconds": sec})
        residual_pred, sec = fit_catboost(
            x_fit.drop(columns=MARKET_COLUMNS[0]),
            training[TARGET].to_numpy(float) - training[MARKET_COLUMNS[0]].to_numpy(float),
            x_val.drop(columns=MARKET_COLUMNS[0]),
        )
        timings.append({"validation_start": start_text, "model": "residual_hybrid", "seconds": sec})
        residual_pred += market
        base = saved[saved["validation_start"].eq(start_text)].set_index("row_id").loc[validation.index]
        if not np.allclose(base[TARGET], validation[TARGET]):
            raise AssertionError("Saved full-history baseline is misaligned")
        outcomes = {"full_history_catboost": base["catboost"].to_numpy(float),
                    "catboost_2022plus": subset_pred, "direct_hybrid": direct_pred,
                    "residual_hybrid": residual_pred}
        for label, values in outcomes.items():
            fold_scores.append({"validation_start": start_text, "model": label,
                                "n": len(validation), **metrics(validation[TARGET], values)})
        pred_rows.append(pd.DataFrame({"validation_start": start_text, "row_id": validation.index,
                                       "month": validation[MONTH].astype(str).to_numpy(),
                                       "actual": validation[TARGET].to_numpy(float),
                                       "market": market, "confidence": confidence,
                                       **outcomes}))
        print(f"{start_text}: full {fold_scores[-4]['rmse_sgd']:.2f}; subset {fold_scores[-3]['rmse_sgd']:.2f}; "
              f"direct {fold_scores[-2]['rmse_sgd']:.2f}; residual {fold_scores[-1]['rmse_sgd']:.2f}", flush=True)
    predictions = pd.concat(pred_rows, ignore_index=True)
    disjoint = predictions[predictions["validation_start"].isin([x for x, _ in FOLDS])]
    names = ("full_history_catboost", "catboost_2022plus", "direct_hybrid", "residual_hybrid")
    pooled = pd.DataFrame([{"model": name, "n": len(disjoint),
                            **metrics(disjoint["actual"], disjoint[name])} for name in names])
    pd.DataFrame(surface_scores).to_csv(OUT / "surface_forecast_scores.csv", index=False, float_format="%.4f")
    pd.DataFrame(fold_scores).to_csv(OUT / "hybrid_fold_scores.csv", index=False, float_format="%.4f")
    pooled.to_csv(OUT / "hybrid_pooled_scores.csv", index=False, float_format="%.4f")
    pd.DataFrame(timings).to_csv(OUT / "fit_times.csv", index=False, float_format="%.2f")
    predictions.to_csv(OUT / "hybrid_predictions.csv", index=False, float_format="%.4f")
    manifest = {
        "source_sha256": {"train.csv": sha256(train_path)},
        "grid": GRID_SIZE, "supported_cells": len(panel.cells), "flat_type_channels": TYPES,
        "smoothing_bandwidth_km": SMOOTH_KM, "smoothing_prior_count": SMOOTH_PRIOR_COUNT,
        "pca_sizes": PCA_SIZES, "factor_forecasters": FACTOR_METHODS,
        "hybrid_pca_size": HYBRID_COMPONENTS, "hybrid_forecaster": "state_space",
        "oof_rule": "2022 onward, one-step feature from PCA and state-space fit on strictly earlier months",
        "validation_rule": "all horizons forecast once at the fitting cutoff; no validation-label feedback",
        "hybrid_training_rule": "2022 onward only because the 2021 composition reference needs all 2021 labels",
    }
    (OUT / "results.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(pooled.to_string(index=False), flush=True)


if __name__ == "__main__":
    evaluate()
