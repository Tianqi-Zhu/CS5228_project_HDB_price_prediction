import unittest

import pandas as pd

from analysis.compare_half_median_ensemble import filter_context
from analysis.filter_rent_below_half_median import FLAG


class HalfMedianTrainingFilterTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame([
            dict(RENT_APPROVAL_DATE="2024-01", TOWN="bedok", FLAT_TYPE="4 room", MONTHLY_RENT=1000),
            dict(RENT_APPROVAL_DATE="2024-02", TOWN=" Bedok ", FLAT_TYPE="4-room", MONTHLY_RENT=3000),
            dict(RENT_APPROVAL_DATE="2024-05", TOWN="bedok", FLAT_TYPE="4-room", MONTHLY_RENT=3000),
            dict(RENT_APPROVAL_DATE="2024-06", TOWN="bedok", FLAT_TYPE="4-room", MONTHLY_RENT=100),
        ], index=[8, 20, 35, 40])

    def test_heldout_labels_cannot_change_training_filter(self):
        raw = self.frame()
        initial = filter_context(raw, [8, 20, 35])
        raw.loc[40, "MONTHLY_RENT"] = 1000000
        changed = filter_context(raw, [8, 20, 35])
        pd.testing.assert_frame_equal(initial, changed)
        self.assertEqual(initial[FLAG].tolist(), [True, False, False])
        self.assertEqual(initial.GROUP_MEDIAN_RENT.tolist(), [3000, 3000, 3000])

    def test_original_source_ids_preserved_and_exact_boundary_retained(self):
        raw = self.frame()
        raw.loc[8, "MONTHLY_RENT"] = 1500
        original = raw.copy(deep=True)
        audit = filter_context(raw, [8, 20, 35])
        self.assertEqual(audit.SOURCE_CSV_LINE.tolist(), [10, 22, 37])
        self.assertFalse(audit[FLAG].any())
        self.assertTrue(audit.loc[8, "EQUAL_TO_HALF_MEDIAN"])
        pd.testing.assert_frame_equal(raw, original)

    def test_reject_validation_ids_in_training_pool(self):
        with self.assertRaisesRegex(ValueError, "Validation rows"):
            filter_context(self.frame(), [8, 20, 35, 40])


if __name__ == "__main__":
    unittest.main()
