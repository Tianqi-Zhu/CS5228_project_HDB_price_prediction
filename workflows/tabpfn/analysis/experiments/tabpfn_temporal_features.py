"""Causal, multi-horizon flat-type forecast inputs for transaction-level TabPFN.

Each historical transaction receives forecasts from a deterministic earlier
origin. The origin selection uses row ID and month only. Temporal fitting,
tuning, and feature calculation use labels no later than that origin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.experiments.flat_type_economic_forecast import (
    ALPHAS, forecast, history_features, training_examples,
)


MIN_MODEL_MONTHS = 18
MAX_HORIZON = 17
FEATURES = [
    "TEMP_FLAT_RECENT3_MEAN", "TEMP_FLAT_RECENT12_MEAN",
    "TEMP_FLAT_DAMPED_FORECAST", "TEMP_FLAT_MODEL_FORECAST",
    "TEMP_FLAT_MODEL_DELTA", "TEMP_FORECAST_HORIZON",
    "TEMP_FLAT_HISTORY_COUNT_3M", "TEMP_FLAT_LAST_OBSERVATION_AGE",
]


def transaction_origins(months, row_ids, first_month, *, validation_origin=None):
    periods = pd.PeriodIndex(months, freq="M")
    ids = np.asarray(row_ids, dtype=np.int64)
    if len(periods) != len(ids) or len(np.unique(ids)) != len(ids):
        raise ValueError("Transaction months and unique row IDs must align")
    first = pd.Period(first_month, "M")
    rows = []
    for month, row_id in zip(periods, ids, strict=True):
        if validation_origin is not None:
            origin = pd.Period(validation_origin, "M") - 1
            horizon = month.ordinal - origin.ordinal
            if not 1 <= horizon <= MAX_HORIZON:
                raise ValueError("Validation month is outside the supported forecast horizon")
        else:
            available = month.ordinal - first.ordinal
            if available < 0:
                raise ValueError("Transaction precedes the rental panel")
            if available == 0:
                rows.append({"row_id": row_id, "month": month, "forecast_origin": pd.NaT,
                             "forecast_horizon": np.nan})
                continue
            # After model warmup, sample only origins with enough model history.
            model_history_origins = available - MIN_MODEL_MONTHS + 1
            maximum = min(MAX_HORIZON, model_history_origins if model_history_origins > 0 else available)
            horizon = 1 + ((int(row_id) * 2654435761 + 42) % maximum)
            origin = month - horizon
        if origin >= month:
            raise AssertionError("Temporal origin must precede its transaction month")
        rows.append({"row_id": row_id, "month": month, "forecast_origin": origin,
                     "forecast_horizon": float(horizon)})
    return pd.DataFrame(rows).set_index("row_id")


class CausalForecastBank:
    def __init__(self, panel):
        self.panel = panel.copy()
        self.first = panel.month.min()
        self.coe = pd.Series(dtype=float)  # Economic features are disabled.
        self.examples = {}

    def training(self, origin):
        if origin not in self.examples:
            self.examples[origin] = training_examples(self.panel, self.coe, origin, economic=False)
        return self.examples[origin]

    def alpha(self, cutoff):
        origins = [cutoff - 12, cutoff - 6]
        origins = [origin for origin in origins
                   if origin.ordinal - self.first.ordinal + 1 >= MIN_MODEL_MONTHS]
        if not origins:
            return 1000.0, []
        losses = {alpha: [[], []] for alpha in ALPHAS}
        for origin in origins:
            examples = self.training(origin)
            # Only months observable at this cutoff are used to tune alpha.
            for alpha in ALPHAS:
                prediction, _ = forecast(self.panel, self.coe, origin, range(1, 7),
                                         economic=False, alpha=alpha, examples=examples)
                observed = prediction.merge(self.panel.loc[self.panel.month <= cutoff],
                                             on=["month", "flat_type"], how="inner", validate="one_to_one")
                if (observed.month > cutoff).any():
                    raise AssertionError("Tuning used outcomes after the origin")
                losses[alpha][0].extend((observed.predicted - observed["mean"]) ** 2)
                losses[alpha][1].extend(observed["count"])
        scores = [{"alpha": alpha, "weighted_mean_rmse_sgd": float(np.sqrt(np.average(errors, weights=weights)))}
                  for alpha, (errors, weights) in losses.items()]
        return min(scores, key=lambda row: row["weighted_mean_rmse_sgd"])["alpha"], scores

    def at_origin(self, origin, max_horizon=MAX_HORIZON):
        origin = pd.Period(origin, "M")
        past = self.panel.loc[self.panel.month <= origin]
        if past.empty:
            raise ValueError("Forecast origin precedes every rental observation")
        kinds = sorted(past.flat_type.unique())
        months = origin.ordinal - self.first.ordinal + 1
        predicted, diagnostic = {}, {"origin": str(origin), "history_months": months,
                                     "last_available_rental_month": str(past.month.max()),
                                     "model_available": months >= MIN_MODEL_MONTHS}
        if months >= MIN_MODEL_MONTHS:
            alpha, tuning = self.alpha(origin)
            result, fitted = forecast(self.panel, self.coe, origin, range(1, max_horizon + 1),
                                      economic=False, alpha=alpha, examples=self.training(origin))
            predicted = result.set_index(["month", "flat_type"]).predicted.to_dict()
            diagnostic.update({**fitted, "tuning": tuning})
            if pd.Period(fitted["last_target_month"], "M") > origin:
                raise AssertionError("Temporal model learned from labels after its origin")
        records = []
        for kind in kinds:
            recent = past.loc[past.flat_type.eq(kind) & (past.month >= origin - 2)]
            count = float(recent["count"].sum())
            for horizon in range(1, max_horizon + 1):
                features, anchor = history_features(self.panel, self.coe, kind, origin, horizon, economic=False)
                month = origin + horizon
                learned = predicted.get((month, kind), np.nan)
                records.append({
                    "forecast_origin": origin, "month": month, "flat_type": kind,
                    "TEMP_FLAT_RECENT3_MEAN": anchor,
                    "TEMP_FLAT_RECENT12_MEAN": anchor - features["short_minus_long"],
                    "TEMP_FLAT_DAMPED_FORECAST": max(1.0, anchor + features["slope_x_horizon"]),
                    "TEMP_FLAT_MODEL_FORECAST": learned,
                    "TEMP_FLAT_MODEL_DELTA": learned - anchor,
                    "TEMP_FORECAST_HORIZON": float(horizon),
                    "TEMP_FLAT_HISTORY_COUNT_3M": count,
                    "TEMP_FLAT_LAST_OBSERVATION_AGE": features["last_observation_age"] + horizon,
                })
        return pd.DataFrame(records), diagnostic


def attach_temporal_features(x, assignments, bank):
    if not x.index.equals(assignments.index):
        raise ValueError("Temporal assignments do not align with source row IDs")
    keys = assignments.reset_index().copy()
    keys["flat_type"] = x.FLAT_TYPE_CLEAN.astype("string").to_numpy()
    joined = keys.merge(bank, on=["forecast_origin", "month", "flat_type"], how="left", validate="many_to_one")
    joined = joined.set_index("row_id").reindex(x.index)
    # Horizon remains known even if a rare flat type has no history at the origin.
    joined["TEMP_FORECAST_HORIZON"] = assignments.forecast_horizon
    added = joined[FEATURES].astype(float)
    if np.isinf(added.to_numpy()).any():
        raise AssertionError("Temporal features contain infinities")
    result = x.join(added)
    pd.testing.assert_frame_equal(result[x.columns], x)
    return result, joined[["month", "forecast_origin", "flat_type", *FEATURES]]
