import unittest

import numpy as np
import pandas as pd

from analysis.experiments.flat_type_inference import flat_type_partitions, assemble_predictions


class FlatTypeInferenceTests(unittest.TestCase):
    def inputs(self):
        train = pd.DataFrame({"FLAT_TYPE_CLEAN": pd.Categorical(["3-room", "4-room", "3-room"]), "x": [1, 2, 3]}, index=[0, 2, 5])
        query = pd.DataFrame({"FLAT_TYPE_CLEAN": pd.Categorical(["4-room", "3-room", "4-room"]), "x": [4, 5, 6]}, index=[8, 9, 10])
        return train, query

    def test_routes_all_rows_without_cross_type_context(self):
        train, query = self.inputs()
        result = flat_type_partitions(train, query)
        np.testing.assert_array_equal(result["3-room"][0], [0, 2])
        np.testing.assert_array_equal(result["3-room"][1], [1])
        np.testing.assert_array_equal(result["4-room"][0], [1])
        np.testing.assert_array_equal(result["4-room"][1], [0, 2])
        for kind, (a, q) in result.items():
            self.assertTrue(train.iloc[a].FLAT_TYPE_CLEAN.eq(kind).all())
            self.assertTrue(query.iloc[q].FLAT_TYPE_CLEAN.eq(kind).all())

    def test_rejects_unseen_query_type_and_overlapping_ids(self):
        train, query = self.inputs()
        query.FLAT_TYPE_CLEAN = ["executive", "3-room", "4-room"]
        with self.assertRaisesRegex(ValueError, "No matching context"):
            flat_type_partitions(train, query)
        query.index = [0, 9, 10]
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            flat_type_partitions(train, query)

    def test_combines_in_original_order_and_rejects_duplicate_or_missing_rows(self):
        _, query = self.inputs()
        frames = [pd.DataFrame({"row_id": [9], "flat_type": ["3-room"], "predicted": [100.]}),
                  pd.DataFrame({"row_id": [8, 10], "flat_type": ["4-room", "4-room"], "predicted": [200., 300.]})]
        combined = assemble_predictions(query, frames)
        self.assertEqual(list(combined.row_id), [8, 9, 10])
        self.assertEqual(list(combined.predicted), [200., 100., 300.])
        with self.assertRaisesRegex(ValueError, "every query exactly once"):
            assemble_predictions(query, frames[:1])
        with self.assertRaisesRegex(ValueError, "every query exactly once"):
            assemble_predictions(query, frames + frames[:1])


if __name__ == "__main__":
    unittest.main()
