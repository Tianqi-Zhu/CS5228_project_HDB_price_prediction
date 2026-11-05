"""Leakage and recursive-forecast checks for the spatiotemporal experiment."""

import unittest

import numpy as np
import pandas as pd

from analysis.experiments.baselines.spatiotemporal_hybrid import (
    MONTH, TARGET, forecast_residuals, historical_features,
)


class SpatiotemporalHybridTests(unittest.TestCase):
    def test_history_is_prior_only_and_frozen_at_prediction_origin(self):
        fitting = pd.DataFrame({
            "TOWN": ["a"] * 3, "BLOCK": ["1"] * 3, "STREET": ["road"] * 3,
            "FLAT_TYPE_CLEAN": ["3-room"] * 3, "FLAT_MODEL_CLEAN": ["m"] * 3,
            "FLOOR_AREA_SQM": [70] * 3, "LEASE_COMMENCE_DATE": [2000] * 3,
            "PLANNING_AREA": ["central"] * 3,
            MONTH: pd.PeriodIndex(["2021-01", "2021-02", "2021-03"], freq="M"),
            TARGET: [1000.0, 2000.0, 3000.0],
        })
        prediction = fitting.iloc[[0, 1]].drop(columns=TARGET).copy()
        prediction[MONTH] = pd.PeriodIndex(["2021-04", "2021-06"], freq="M")
        fit_history, pred_history = historical_features(fitting, prediction)
        self.assertTrue(np.isnan(fit_history.loc[0, "HIST_SIGNATURE_MEAN"]))
        self.assertEqual(fit_history.loc[1, "HIST_SIGNATURE_MEAN"], 1000.0)
        self.assertEqual(fit_history.loc[2, "HIST_SIGNATURE_MEAN"], 1500.0)
        self.assertEqual(pred_history["HIST_SIGNATURE_MEAN"].tolist(), [2000.0, 2000.0])
        self.assertEqual(pred_history["HIST_SIGNATURE_RECENCY_MONTHS"].tolist(), [1, 3])

    def test_recursive_forecast_returns_every_group_and_month(self):
        months = pd.period_range("2021-01", periods=18, freq="M")
        rows, residuals = [], []
        for month_index, month in enumerate(months):
            for area_index, area in enumerate(("a", "b")):
                rows.append({
                    "PLANNING_AREA": area, "FLAT_TYPE_CLEAN": "3-room", MONTH: month,
                    "LATITUDE": 1.30 + area_index * .02,
                    "LONGITUDE": 103.80 + area_index * .02,
                })
                residuals.append(10 + month_index + area_index)
        frame = pd.DataFrame(rows)
        future = [pd.Period("2022-07", freq="M"), pd.Period("2022-08", freq="M")]
        forecast = forecast_residuals(frame, np.asarray(residuals), future)
        self.assertEqual(len(forecast.values), 4)
        self.assertEqual(forecast.groups, 2)
        self.assertTrue(np.isfinite(forecast.values["correction"]).all())
        self.assertEqual(set(forecast.values[MONTH]), set(future))


if __name__ == "__main__":
    unittest.main()
