"""Focused checks for the temporal rental benchmark."""

import unittest

import numpy as np
import pandas as pd

from benchmark_rent_models import BASE_FEATURES, SPATIAL_FEATURES, score, split_by_month


class BenchmarkTests(unittest.TestCase):
    def test_time_split_keeps_months_disjoint(self):
        frame = pd.DataFrame({
            "RENT_APPROVAL_DATE": ["2024-04", "2024-03", "2025-03", "2021-01"],
            "MONTHLY_RENT": [4, 3, 5, 1],
        })
        fit, valid = split_by_month(frame, "2024-04")
        self.assertEqual(fit["RENT_APPROVAL_DATE"].tolist(), ["2024-03", "2021-01"])
        self.assertEqual(valid["RENT_APPROVAL_DATE"].tolist(), ["2024-04", "2025-03"])

    def test_score_uses_original_rent_units(self):
        result = score(np.array([1000, 2000]), np.array([1100, 1800]))
        self.assertAlmostEqual(result["rmse_sgd"], np.sqrt(25000))
        self.assertEqual(result["mae_sgd"], 150)

    def test_features_exclude_target_and_identifiers(self):
        used = set(BASE_FEATURES + SPATIAL_FEATURES)
        self.assertFalse(used & {"MONTHLY_RENT", "SOURCE_ROW_ID", "FEE", "BLOCK", "STREET", "NEAREST_PLANNED_MRT_KM"})


if __name__ == "__main__":
    unittest.main()
