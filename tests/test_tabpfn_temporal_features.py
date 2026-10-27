"""Leakage and alignment tests for temporal TabPFN inputs."""

import unittest

import numpy as np
import pandas as pd

from analysis.experiments.tabpfn_temporal_features import (
    CausalForecastBank, FEATURES, attach_temporal_features, transaction_origins,
)
from tests.test_flat_type_economic_forecast import synthetic


class TemporalFeatureTests(unittest.TestCase):
    def test_historical_origins_are_earlier_and_independent_of_row_order(self):
        months = ["2021-01", "2022-08", "2023-03", "2023-03"]
        ids = [10, 99, 12, 2]
        result = transaction_origins(months, ids, "2021-01")
        perm = [3, 1, 0, 2]
        other = transaction_origins([months[i] for i in perm], [ids[i] for i in perm], "2021-01")
        pd.testing.assert_frame_equal(result.sort_index(), other.sort_index())
        valid = result.dropna(subset="forecast_origin")
        self.assertTrue((valid.forecast_origin < valid.month).all())
        self.assertTrue(pd.isna(result.loc[10, "forecast_origin"]))

    def test_validation_uses_one_fixed_cutoff(self):
        result = transaction_origins(["2024-06", "2025-03"], [1, 2], "2021-01", validation_origin="2024-06")
        self.assertEqual(result.forecast_origin.astype(str).tolist(), ["2024-05", "2024-05"])
        self.assertEqual(result.forecast_horizon.tolist(), [1.0, 10.0])

    def test_future_labels_cannot_change_fitting_or_tuning(self):
        panel, _ = synthetic()
        cutoff = pd.Period("2022-12", "M")
        before, audit = CausalForecastBank(panel).at_origin(cutoff, max_horizon=3)
        panel.loc[panel.month > cutoff, ["mean", "rent_sum"]] = 1e9
        after, _ = CausalForecastBank(panel).at_origin(cutoff, max_horizon=3)
        pd.testing.assert_frame_equal(before, after)
        self.assertLessEqual(pd.Period(audit["last_target_month"], "M"), cutoff)

    def test_join_preserves_original_values_row_order_and_unseen_groups(self):
        panel, _ = synthetic()
        origin = pd.Period("2021-12", "M")
        bank, _ = CausalForecastBank(panel).at_origin(origin, max_horizon=1)
        x = pd.DataFrame({"FLAT_TYPE_CLEAN": pd.Categorical(["4-room", "unknown", "3-room"]),
                          "FLOOR_AREA_SQM": [95.0, 90.0, 70.0]}, index=pd.Index([42, 2, 90], name="row_id"))
        assignments = transaction_origins(["2022-01"] * 3, x.index, "2021-01", validation_origin="2022-01")
        result, _ = attach_temporal_features(x, assignments, bank)
        pd.testing.assert_frame_equal(result[x.columns], x)
        self.assertTrue(np.isnan(result.loc[2, "TEMP_FLAT_MODEL_FORECAST"]))
        self.assertEqual(result.loc[2, "TEMP_FORECAST_HORIZON"], 1.0)
        self.assertEqual(list(result.columns[-len(FEATURES):]), FEATURES)


if __name__ == "__main__":
    unittest.main()
