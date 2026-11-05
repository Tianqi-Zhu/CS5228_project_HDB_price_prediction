"""Origin-aligned rent histories and a causally fitted local forecast baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.experiments.spatiotemporal_hybrid import MONTH, TARGET, HISTORY_FEATURES, historical_features
from analysis.experiments.recent_history import RECENT_FEATURES, recent_history_features
from analysis.experiments.tabpfn_adaptation_features import (
    STD_FEATURES, TREND_FEATURES, prior_standard_deviations,
)


AREA, TYPE = "PLANNING_AREA", "FLAT_TYPE_CLEAN"
MAX_HORIZON = 10
HISTORY_COLUMNS = HISTORY_FEATURES + RECENT_FEATURES + STD_FEATURES + TREND_FEATURES
FEATURES = [
    "TEMP_FLAT_RECENT3_MEAN", "TEMP_FLAT_RECENT12_MEAN", "TEMP_FLAT_DAMPED_FORECAST",
    "TEMP_FORECAST_HORIZON", "TEMP_FLAT_HISTORY_COUNT_3M", "TEMP_FLAT_LAST_OBSERVATION_AGE",
    "TEMP_LOCAL_FORECAST", "TEMP_LOCAL_WEIGHT", "TEMP_FORECAST_RMSE",
]


def episode_origins(months, row_ids, first_month, validation_origin=None):
    first = pd.Period(first_month, "M")
    rows = []
    for month, row_id in zip(pd.PeriodIndex(months, freq="M"), row_ids, strict=True):
        if validation_origin is not None:
            origin = pd.Period(validation_origin, "M") - 1
        else:
            available = month.ordinal - first.ordinal
            if available == 0:
                rows.append((row_id, month, pd.NaT))
                continue
            if available < 0:
                raise ValueError("Transaction precedes the rental history")
            eligible = available - 11
            maximum = min(MAX_HORIZON, eligible if eligible > 0 else available)
            horizon = 1 + ((int(row_id) * 2654435761 + 42) % maximum)
            origin = month - horizon
        if not 1 <= month.ordinal - origin.ordinal <= MAX_HORIZON:
            raise ValueError("Invalid forecast origin/horizon")
        rows.append((row_id, month, origin))
    result = pd.DataFrame(rows, columns=["row_id", "month", "forecast_origin"]).set_index("row_id")
    result["forecast_origin"] = pd.PeriodIndex(result.forecast_origin, freq="M")
    return result


def origin_history(past, query):
    """Rebuild every target-derived input using one common information cutoff."""
    if past.empty or query.empty or past[MONTH].max() >= query[MONTH].min():
        raise ValueError("History labels must precede all query months")
    blind = query.drop(columns=TARGET, errors="ignore")
    _, expanding = historical_features(past, blind)
    _, recent = recent_history_features(past, blind)
    result = expanding.join(recent).join(prior_standard_deviations(past, blind, frozen=True))
    result["AREA_RECENT_MINUS_LONG"] = result.RECENT_AREA_FLAT_3M_MEAN - result.RECENT_AREA_FLAT_12M_MEAN
    result["BLOCK_RECENT_MINUS_LONG"] = result.RECENT_BLOCK_FLAT_3M_SHRUNK_MEAN - result.RECENT_BLOCK_FLAT_12M_SHRUNK_MEAN
    result["BLOCK_MINUS_AREA_RECENT"] = result.RECENT_BLOCK_FLAT_3M_SHRUNK_MEAN - result.RECENT_AREA_FLAT_3M_MEAN
    result["AREA_LAST_MINUS_RECENT"] = result.HIST_AREA_FLAT_LAST_MEAN - result.RECENT_AREA_FLAT_3M_MEAN
    return result[HISTORY_COLUMNS]


def divide(total, count):
    return np.divide(total, count, out=np.full(np.shape(total), np.nan, dtype=float), where=count > 0)


class LocalTemporalModel:
    """Pooled damping and local shrinkage, fitted on earlier rolling forecasts.

    Selection uses the last twelve observed target months, weights group means
    by transaction count, and gives each available forecast horizon equal weight.
    Candidates within 1% of the best error use the more conservative setting.
    """

    def __init__(self, frame):
        self.first, self.last = frame[MONTH].min(), frame[MONTH].max()
        self.months = pd.period_range(self.first, self.last, freq="M")
        self.kinds = sorted(frame[TYPE].astype(str).unique())
        self.groups = sorted(set(zip(frame[AREA].astype(str), frame[TYPE].astype(str))))
        self.kind_index = {kind: i for i, kind in enumerate(self.kinds)}
        self.group_index = {group: i for i, group in enumerate(self.groups)}
        self.parent = np.asarray([self.kind_index[kind] for _, kind in self.groups])
        self.gsum, self.gcount = self.arrays(frame, [TYPE], self.kinds)
        self.lsum, self.lcount = self.arrays(frame, [AREA, TYPE], pd.MultiIndex.from_tuples(self.groups))
        self.gprefix = (np.vstack([np.zeros(len(self.kinds)), self.gsum.cumsum(axis=0)]),
                        np.vstack([np.zeros(len(self.kinds)), self.gcount.cumsum(axis=0)]))
        self.lprefix = (np.vstack([np.zeros(len(self.groups)), self.lsum.cumsum(axis=0)]),
                        np.vstack([np.zeros(len(self.groups)), self.lcount.cumsum(axis=0)]))
        times = np.arange(len(self.months))[:, None]
        self.last_seen = np.maximum.accumulate(np.where(self.gcount > 0, times, -1), axis=0)
        self.states, self.parameters = {}, {}

    def arrays(self, frame, keys, columns):
        panel = frame.groupby([MONTH, *keys], observed=True)[TARGET].agg(total="sum", count="size")
        total = panel.total.unstack(keys).reindex(index=self.months, columns=columns).fillna(0).to_numpy(float)
        count = panel["count"].unstack(keys).reindex(index=self.months, columns=columns).fillna(0).to_numpy(float)
        return total, count

    @staticmethod
    def window(prefix, origin, width, lag=0):
        sums, counts = prefix
        end = max(0, origin + 1 - lag)
        start = max(0, end - width)
        count = counts[end] - counts[start]
        return divide(sums[end] - sums[start], count), count

    def state(self, origin):
        if origin not in self.states:
            short, count = self.window(self.gprefix, origin, 3)
            long, _ = self.window(self.gprefix, origin, 12)
            all_mean, _ = self.window(self.gprefix, origin, origin + 1)
            short = np.where(np.isfinite(short), short, np.where(np.isfinite(long), long, all_mean))
            long = np.where(np.isfinite(long), long, all_mean)
            previous, _ = self.window(self.gprefix, origin, 3, lag=3)
            slope = np.clip((short - np.where(np.isfinite(previous), previous, short)) / 3, -100, 100)
            local, local_n = self.window(self.lprefix, origin, 3)
            premium = np.where(local_n > 0, local - short[self.parent], 0.0)
            age = np.where(self.last_seen[origin] >= 0, origin - self.last_seen[origin], np.nan)
            self.states[origin] = {"short": short, "long": long, "count": count, "slope": slope,
                                   "premium": premium, "local_count": local_n, "age": age}
        return self.states[origin]

    @staticmethod
    def gain(phi, horizon):
        return 0.0 if phi is None else (1 - phi ** horizon) / (1 - phi)

    def global_prediction(self, origin, horizon, phi):
        state = self.state(origin)
        return state["short"] + self.gain(phi, horizon) * state["slope"]

    def local_prediction(self, origin, horizon, phi, shrink, decay):
        state = self.state(origin)
        weight = (np.zeros(len(self.groups)) if shrink is None
                  else state["local_count"] / (state["local_count"] + shrink))
        return self.global_prediction(origin, horizon, phi)[self.parent] + weight * state["premium"] * decay ** (horizon - 1)

    @staticmethod
    def loss(records, predict, target, count):
        error_sums, weights = np.zeros(MAX_HORIZON), np.zeros(MAX_HORIZON)
        for origin, horizon, month in records:
            prediction = predict(origin, horizon)
            actual = divide(target[month], count[month])
            valid = np.isfinite(prediction) & np.isfinite(actual) & (count[month] > 0)
            error_sums[horizon - 1] += np.sum((prediction[valid] - actual[valid]) ** 2 * count[month, valid])
            weights[horizon - 1] += np.sum(count[month, valid])
        per_horizon = divide(error_sums, weights)
        return float(np.sqrt(np.nanmean(per_horizon))), np.sqrt(per_horizon)

    def fit_at(self, origin):
        index = pd.Period(origin, "M").ordinal - self.first.ordinal
        if not 0 <= index < len(self.months):
            raise ValueError("Origin is outside the available rental panel")
        if index in self.parameters:
            return self.parameters[index]
        records = [(past, h, past + h)
                   for past in range(11, index)
                   for h in range(1, min(MAX_HORIZON, index - past) + 1)
                   if past + h >= index - 11]
        audit = {"origin": str(origin), "last_tuning_target_month": str(origin) if records else None,
                 "rolling_forecast_pairs": len(records)}
        if not records:
            params = {"phi": None, "shrink": 30.0, "decay": 0.9,
                      "rmse_by_horizon": [None] * MAX_HORIZON, "audit": audit}
        else:
            global_scores = []
            for phi in (None, 0.5, 0.8, 0.95):
                score, _ = self.loss(records, lambda o, h: self.global_prediction(o, h, phi), self.gsum, self.gcount)
                global_scores.append({"phi": phi, "rmse": score})
            threshold = min(row["rmse"] for row in global_scores) * 1.01
            phi = next(row["phi"] for row in global_scores if row["rmse"] <= threshold)
            local_scores = []
            for shrink in (None, 100.0, 30.0, 10.0):
                for decay in ((1.0,) if shrink is None else (0.9, 1.0)):
                    score, errors = self.loss(records, lambda o, h: self.local_prediction(o, h, phi, shrink, decay),
                                              self.lsum, self.lcount)
                    local_scores.append({"shrink": shrink, "decay": decay, "rmse": score,
                                         "errors": [float(value) if np.isfinite(value) else None for value in errors]})
            threshold = min(row["rmse"] for row in local_scores) * 1.01
            selected = next(row for row in local_scores if row["rmse"] <= threshold)
            audit.update({"global_candidates": global_scores,
                          "local_candidates": [{k: v for k, v in row.items() if k != "errors"} for row in local_scores]})
            params = {"phi": phi, "shrink": selected["shrink"], "decay": selected["decay"],
                      "rmse_by_horizon": selected["errors"], "audit": audit}
        self.parameters[index] = params
        return params

    def features(self, query, origin):
        origin = pd.Period(origin, "M")
        index = origin.ordinal - self.first.ordinal
        params, state = self.fit_at(origin), self.state(index)
        rows = []
        for month, area, kind in query[[MONTH, AREA, TYPE]].itertuples(index=False, name=None):
            horizon = month.ordinal - origin.ordinal
            if not 1 <= horizon <= MAX_HORIZON:
                raise ValueError("Query is outside the fitted horizon policy")
            g = self.kind_index.get(str(kind))
            local_index = self.group_index.get((str(area), str(kind)))
            short = state["short"][g] if g is not None else np.nan
            global_value = self.global_prediction(index, horizon, params["phi"])[g] if g is not None else np.nan
            weight = 0.0
            if local_index is not None and params["shrink"] is not None:
                n = state["local_count"][local_index]
                weight = n / (n + params["shrink"])
            local_value = global_value + (weight * state["premium"][local_index] * params["decay"] ** (horizon - 1)
                                          if local_index is not None else 0.0)
            error = params["rmse_by_horizon"][horizon - 1]
            known = [v for v in params["rmse_by_horizon"] if v is not None]
            if error is None and known:
                error = float(np.mean(known))
            rows.append([short, state["long"][g] if g is not None else np.nan,
                         max(1.0, global_value) if np.isfinite(global_value) else np.nan, float(horizon),
                         state["count"][g] if g is not None else 0.0,
                         state["age"][g] + horizon if g is not None else np.nan,
                         max(1.0, local_value) if np.isfinite(local_value) else np.nan,
                         weight, error if error is not None else np.nan])
        return pd.DataFrame(rows, index=query.index, columns=FEATURES)
