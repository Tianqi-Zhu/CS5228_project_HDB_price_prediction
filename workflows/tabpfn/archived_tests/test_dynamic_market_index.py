"""Checks for fixed-origin market-index construction and forecast alignment."""

import unittest

import numpy as np
import pandas as pd

from analysis.experiments.dynamic_market_index import (
    AREA, MONTH, TYPE, attach_forecast, forecast_indices, observed_indices,
    reference_model,
)
from analysis.experiments.recent_history import recent_history_features
from analysis.experiments.coarse_rental_surface import CELL, forecast_local_change
from analysis.experiments.factored_rental_surface import SurfacePanel, forecast_surfaces


class DynamicMarketIndexTests(unittest.TestCase):
    def test_2021_reference_ignores_later_targets(self):
        rows = pd.DataFrame({
            MONTH: pd.PeriodIndex(["2021-01", "2021-02", "2022-01"], freq="M"),
            TYPE: ["3-room"] * 3, "FLAT_MODEL_CLEAN": ["improved"] * 3,
            "TOWN_CLEAN": ["a"] * 3, "FLOOR_AREA_SQM": [70.0] * 3,
            "FLAT_AGE_YEARS": [20.0] * 3, "MONTHLY_RENT": [1000, 1200, 3000],
        })
        columns = [TYPE, "FLAT_MODEL_CLEAN", "TOWN_CLEAN", "FLOOR_AREA_SQM", "FLAT_AGE_YEARS"]
        first = reference_model(rows).predict(rows[columns])
        rows.loc[2, "MONTHLY_RENT"] = 100000
        second = reference_model(rows).predict(rows[columns])
        np.testing.assert_allclose(first, second)

    def test_fixed_origin_forecast_ignores_future_labels_and_preserves_row_order(self):
        fit = pd.DataFrame({
            MONTH: pd.PeriodIndex(["2021-01", "2021-02", "2021-03", "2021-03"], freq="M"),
            AREA: ["a", "a", "a", "b"], TYPE: ["3-room"] * 4,
        })
        future = pd.DataFrame({
            MONTH: pd.PeriodIndex(["2021-05", "2021-04", "2021-04"], freq="M"),
            AREA: ["b", "a", "b"], TYPE: ["3-room"] * 3,
            "MONTHLY_RENT": [10, 20, 30],
        })
        flat, group = observed_indices(fit, np.array([0., 100., 200., 300.]))
        first = forecast_indices(fit, future, flat, group, "damped", "area_flat_type")
        future["MONTHLY_RENT"] = [999999, 888888, 777777]
        second = forecast_indices(fit, future, flat, group, "damped", "area_flat_type")
        np.testing.assert_allclose(
            attach_forecast(future, first, "delta"),
            attach_forecast(future, second, "delta"),
        )
        persistence = forecast_indices(fit, future, flat, group, "persistence", "area_flat_type")
        np.testing.assert_allclose(attach_forecast(future, persistence, "delta"), 0.)
        illegal = future.copy()
        illegal.loc[0, MONTH] = pd.Period("2021-03", freq="M")
        with self.assertRaises(AssertionError):
            forecast_indices(fit, illegal, flat, group, "damped", "area_flat_type")

    def test_recent_history_excludes_same_month_and_freezes_at_origin(self):
        fit = pd.DataFrame({
            MONTH: pd.PeriodIndex(["2021-01", "2021-02", "2021-02", "2021-04"], freq="M"),
            AREA: ["a"] * 4, TYPE: ["3-room"] * 4,
            "TOWN": ["a"] * 4, "BLOCK": ["1"] * 4, "STREET": ["road"] * 4,
            "MONTHLY_RENT": [1000., 2000., 3000., 4000.],
        })
        future = fit.iloc[[0, 0]].drop(columns="MONTHLY_RENT").copy()
        future[MONTH] = pd.PeriodIndex(["2021-05", "2021-07"], freq="M")
        past, prediction = recent_history_features(fit, future)
        col = "RECENT_BLOCK_FLAT_3M_MEAN"
        self.assertTrue(np.isnan(past.loc[0, col]))
        self.assertEqual(past.loc[1, col], 1000.)
        self.assertEqual(past.loc[2, col], 1000.)
        self.assertEqual(past.loc[3, col], 2000.)
        self.assertEqual(prediction[col].tolist(), [3000., 3000.])

    def test_local_surface_forecast_uses_only_fitting_residuals(self):
        months = pd.period_range("2021-01", periods=6, freq="M")
        fit = pd.DataFrame({
            MONTH: pd.PeriodIndex(np.repeat(months.astype(str), 2), freq="M"),
            CELL: [7, 8] * 6, TYPE: ["3-room"] * 12,
        })
        residuals = np.array([value for i in range(6) for value in (float(i), 0.)])
        future = pd.DataFrame({
            MONTH: pd.PeriodIndex(["2021-07", "2021-08"], freq="M"),
            CELL: [7, 7], TYPE: ["3-room", "3-room"],
            "MONTHLY_RENT": [1000., 2000.],
        })
        first, covered, _ = forecast_local_change(fit, future, residuals)
        self.assertTrue(covered.all())
        self.assertGreater(first[0], 0.)
        future["MONTHLY_RENT"] = [99999., 88888.]
        second, _, _ = forecast_local_change(fit, future, residuals)
        np.testing.assert_allclose(first, second)
        illegal = future.copy()
        illegal.loc[0, MONTH] = months[-1]
        with self.assertRaises(AssertionError):
            forecast_local_change(fit, illegal, residuals)

    def test_pca_forecast_is_unchanged_when_future_surfaces_change(self):
        months = pd.period_range("2021-01", periods=16, freq="M")
        values = np.arange(16 * 6, dtype=float).reshape(16, 6)
        panel = SurfacePanel(pd.DataFrame(), months, pd.DataFrame(), values.copy(),
                             np.empty((0,)), np.empty((0,)), np.empty((0,)), {})
        first, _, _ = forecast_surfaces(panel, 12, np.array([1, 2]), 3, "state_space")
        panel.values[12:] = 1e9
        second, _, _ = forecast_surfaces(panel, 12, np.array([1, 2]), 3, "state_space")
        np.testing.assert_allclose(first, second)


if __name__ == "__main__":
    unittest.main()
