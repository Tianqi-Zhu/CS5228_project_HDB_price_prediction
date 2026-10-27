"""Validate the explicit long-horizon feature policy used for Kaggle test."""

import unittest

import numpy as np
import pandas as pd

from analysis.train_tabpfn_fast_submission import capped_temporal_features


class TemporalCapTest(unittest.TestCase):
    def test_month_ten_matches_model_and_later_months_keep_actual_horizon(self):
        origin = pd.Period("2025-03", "M")
        query = pd.DataFrame({"_MONTH": pd.PeriodIndex(["2025-04", "2026-01", "2026-07"], freq="M")})

        class Model:
            def features(self, projected, model_origin):
                horizon = projected._MONTH.astype("int64").to_numpy() - model_origin.ordinal
                self.seen = horizon
                return pd.DataFrame({"TEMP_LOCAL_FORECAST": 3000 + horizon,
                                     "TEMP_FORECAST_HORIZON": horizon.astype(float),
                                     "TEMP_FLAT_LAST_OBSERVATION_AGE": 2 + horizon}, index=projected.index)

        model = Model()
        result = capped_temporal_features(model, query, origin)
        np.testing.assert_array_equal(model.seen, [1, 10, 10])
        np.testing.assert_array_equal(result.TEMP_FORECAST_HORIZON, [1, 10, 16])
        np.testing.assert_array_equal(result.TEMP_LOCAL_FORECAST, [3001, 3010, 3010])
        np.testing.assert_array_equal(result.TEMP_FLAT_LAST_OBSERVATION_AGE, [3, 12, 18])

    def test_rejects_same_month_query(self):
        class Model:
            def features(self, projected, model_origin):
                raise AssertionError("Should not be called")

        query = pd.DataFrame({"_MONTH": pd.PeriodIndex(["2025-03"], freq="M")})
        with self.assertRaises(ValueError):
            capped_temporal_features(Model(), query, pd.Period("2025-03", "M"))


if __name__ == "__main__":
    unittest.main()
