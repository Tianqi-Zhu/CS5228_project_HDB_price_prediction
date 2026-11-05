"""Check graph comparables obey space, type, and forecast cutoffs."""

import unittest

import numpy as np
import pandas as pd

from analysis.experiments.local_comparable_graph import MONTH, TARGET, local_features


class LocalComparableGraphTests(unittest.TestCase):
    def test_excludes_own_block_and_freezes_future_history(self):
        blocks = pd.MultiIndex.from_tuples([
            ("town", "a", "road"), ("town", "b", "road"), ("town", "c", "road")
        ], names=["TOWN", "BLOCK", "STREET"])
        edges = [[(1, 1.0)], [(0, 1.0)], []]
        fitting = pd.DataFrame({
            "TOWN": ["town"] * 5, "BLOCK": ["a", "b", "c", "a", "b"],
            "STREET": ["road"] * 5, "FLAT_TYPE_CLEAN": ["3-room"] * 5,
            "FLOOR_AREA_SQM": [70.0] * 5,
            MONTH: pd.PeriodIndex(["2021-01"] * 3 + ["2021-02"] * 2, freq="M"),
            TARGET: [1000.0, 2000.0, 9000.0, 1100.0, 3000.0],
            "_OOF_RESIDUAL": [np.nan, 20.0, 999.0, np.nan, 40.0],
        })
        prediction = pd.DataFrame({
            "TOWN": ["town"] * 5, "BLOCK": ["a"] * 5,
            "STREET": ["road"] * 5,
            "FLAT_TYPE_CLEAN": ["3-room"] * 3 + ["4-room", "3-room"],
            "FLOOR_AREA_SQM": [70.0] * 4 + [120.0],
            MONTH: pd.PeriodIndex(["2021-02", "2021-03", "2021-05", "2021-03", "2021-03"], freq="M"),
        })
        fit, future = local_features(fitting, prediction, blocks, edges)
        self.assertTrue(np.isnan(fit.loc[0, "LOCAL_RENT_MEAN_3M"]))
        self.assertEqual(fit.loc[3, "LOCAL_RENT_MEAN_3M"], 2000.0)
        self.assertEqual(future["LOCAL_RENT_MEAN_3M"].iloc[:3].tolist(), [2000.0, 2500.0, 2500.0])
        self.assertEqual(future["LOCAL_RENT_COUNT_3M"].tolist(), [1.0, 2.0, 2.0, 0.0, 0.0])
        self.assertEqual(future["LOCAL_RESIDUAL_MEAN_3M"].iloc[:3].tolist(), [20.0, 30.0, 30.0])


if __name__ == "__main__":
    unittest.main()
