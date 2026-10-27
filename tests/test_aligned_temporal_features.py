"""Information-cutoff and local-baseline tests for aligned temporal inputs."""

import unittest

import numpy as np
import pandas as pd

from analysis.experiments.aligned_temporal_features import (
    FEATURES, LocalTemporalModel, episode_origins, origin_history,
)
from analysis.experiments.spatiotemporal_hybrid import MONTH, TARGET


def panel():
    rows = []
    for i, month in enumerate(pd.period_range("2021-01", periods=36, freq="M")):
        for area, premium in (("a", 0), ("b", 300)):
            for kind, level in (("3-room", 0), ("4-room", 500)):
                for unit in range(2):
                    rows.append({MONTH: month, TARGET: 1000 + 20 * i + premium + level + 100 * unit,
                                 "PLANNING_AREA": area, "FLAT_TYPE_CLEAN": kind, "TOWN": area,
                                 "BLOCK": "1", "STREET": "road", "FLAT_MODEL_CLEAN": "improved",
                                 "FLOOR_AREA_SQM": 70 if kind == "3-room" else 90,
                                 "LEASE_COMMENCE_DATE": 1980})
    return pd.DataFrame(rows)


class AlignedTemporalTests(unittest.TestCase):
    def test_all_histories_are_frozen_at_the_selected_origin(self):
        data = panel()
        cutoff = pd.Period("2022-06", "M")
        past = data[data[MONTH] <= cutoff]
        query = data[(data[MONTH] > cutoff) & (data[MONTH] <= cutoff + 2)].copy()
        first = origin_history(past, query)
        query[TARGET] = 1e9
        second = origin_history(past, query)
        pd.testing.assert_frame_equal(first, second)
        np.testing.assert_allclose(first.RECENT_BLOCK_FLAT_3M_COUNT, 6)
        np.testing.assert_allclose(first.HIST_BLOCK_FLAT_COUNT, 36)
        self.assertEqual(sorted(first.HIST_BLOCK_FLAT_RECENCY_MONTHS.unique()), [1.0, 2.0])

    def test_forecast_and_tuning_cannot_see_future_rents(self):
        data = panel()
        cutoff = pd.Period("2022-12", "M")
        query = data[data[MONTH] > cutoff].iloc[:24]
        first_model = LocalTemporalModel(data)
        first = first_model.features(query, cutoff)
        changed = data.copy()
        changed.loc[changed[MONTH] > cutoff, TARGET] = 1e9
        second_model = LocalTemporalModel(changed)
        second = second_model.features(query, cutoff)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first_model.fit_at(cutoff), second_model.fit_at(cutoff))
        self.assertEqual(first_model.fit_at(cutoff)["audit"]["last_tuning_target_month"], str(cutoff))

    def test_new_area_uses_flat_type_forecast_without_a_local_premium(self):
        data = panel()
        cutoff = pd.Period("2022-12", "M")
        query = data[data[MONTH] == cutoff + 1].iloc[:1].copy()
        query["PLANNING_AREA"] = "unseen"
        result = LocalTemporalModel(data).features(query, cutoff)
        self.assertEqual(result.TEMP_LOCAL_WEIGHT.iloc[0], 0)
        self.assertEqual(result.TEMP_LOCAL_FORECAST.iloc[0], result.TEMP_FLAT_DAMPED_FORECAST.iloc[0])
        self.assertEqual(list(result.columns), FEATURES)

    def test_context_origins_cover_the_validation_horizon_without_overlap(self):
        months = ["2021-01", "2022-06", "2023-03", "2024-05"]
        first = episode_origins(months, [40, 2, 10, 99], "2021-01")
        valid = first.dropna(subset="forecast_origin")
        horizons = valid.month.astype("int64") - valid.forecast_origin.astype("int64")
        self.assertTrue(horizons.between(1, 10).all())
        query = episode_origins(["2024-06", "2025-03"], [1, 2], "2021-01", "2024-06")
        self.assertEqual(query.forecast_origin.astype(str).tolist(), ["2024-05", "2024-05"])


if __name__ == "__main__":
    unittest.main()
