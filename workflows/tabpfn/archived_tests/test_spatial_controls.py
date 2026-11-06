"""Fast checks for the integrated spatial-control studies."""

import unittest

import numpy as np
import pandas as pd

from analysis.studies.spatial.mrt_fixed_effects import fit
from analysis.studies.spatial.mrt_matching import LABELS, compare, prepare


class SpatialControlTests(unittest.TestCase):
    def test_distance_boundaries_are_explicit(self):
        frame = pd.DataFrame({
            "NEAREST_OPEN_MRT_KM": [0, .3, .300001, 2, 2.000001],
            "FLOOR_AREA_SQM": [50] * 5,
            "BUILDING_AGE_YEARS": [10] * 5,
            "TOWN_CLEAN": ["a"] * 5,
            "BLOCK_CLEAN": ["1"] * 5,
            "STREET_CLEAN": ["x"] * 5,
        })
        self.assertEqual(
            prepare(frame).distance_bin.astype(str).tolist(),
            [LABELS[0], LABELS[0], LABELS[1], LABELS[4], LABELS[5]],
        )

    def test_fixed_effect_fit_recovers_slopes(self):
        rng = np.random.default_rng(7)
        n = 300
        group_a = rng.integers(0, 6, n)
        group_b = rng.integers(0, 3, n)
        x = rng.normal(size=(n, 2))
        y = x @ np.array([2.0, -1.0]) + group_a * 3 + group_b * 5
        beta, covariance, info = fit(y, x, [group_a, group_b], np.arange(n) // 3)
        np.testing.assert_allclose(beta, [2.0, -1.0], atol=1e-9)
        self.assertEqual(covariance.shape, (2, 2))
        self.assertEqual(info["slope_rank"], 2)

    def test_overlap_weighting_removes_composition_shift(self):
        rows = []
        covariates = ["FLOOR_AREA_SQM", "BUILDING_AGE_YEARS", "MAX_FLOOR", "DISTANCE_TO_CBD_KM"]
        for town, base, counts in [("a", 1000, (90, 10)), ("b", 2000, (10, 90))]:
            for distance, count in zip(LABELS[:2], counts):
                for _ in range(count):
                    rows.append({"town": town, "distance_bin": distance,
                                 "MONTHLY_RENT": base + 100 * (distance == LABELS[1]),
                                 "building": town + distance, **{column: 1.0 for column in covariates}})
        result, _ = compare(pd.DataFrame(rows), ["town"], LABELS[1])
        self.assertAlmostEqual(result["raw_difference_sgd"], 900)
        self.assertAlmostEqual(result["standardized_difference_sgd"], 100)


if __name__ == "__main__":
    unittest.main()
