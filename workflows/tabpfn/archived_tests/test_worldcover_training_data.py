import unittest

import numpy as np
import pandas as pd

from analysis.experiments.worldcover_training_data import append_tree_features, TREE_COLUMNS


class TrainingDataTests(unittest.TestCase):
    def inputs(self):
        original = pd.DataFrame({f"x{i}": [float(i), float(i+1)] for i in range(76)}, index=[17, 42])
        original["category"] = pd.Categorical(["a", "b"], categories=["a", "b", "unseen"])
        added = pd.DataFrame([[.2, .4, 0.], [np.nan, np.nan, 1.]], index=original.index, columns=TREE_COLUMNS)
        return original, added

    def test_preserves_categories_row_ids_and_native_missingness(self):
        original, added = self.inputs()
        result = append_tree_features(original, added)
        self.assertEqual(result.shape, (2, 80))
        pd.testing.assert_frame_equal(result[original.columns], original)
        pd.testing.assert_frame_equal(result[TREE_COLUMNS], added)
        self.assertEqual(list(result.category.cat.categories), ["a", "b", "unseen"])

    def test_rejects_row_misalignment_and_duplicate_ids(self):
        original, added = self.inputs()
        with self.assertRaisesRegex(ValueError, "identical row IDs"):
            append_tree_features(original, added.iloc[::-1])
        with self.assertRaisesRegex(ValueError, "Duplicate row IDs"):
            append_tree_features(pd.concat([original, original]), pd.concat([added, added]))

    def test_rejects_invalid_fractions_and_false_missingness_flag(self):
        original, added = self.inputs()
        for value in [-.1, 1.1, np.inf]:
            changed = added.copy()
            changed.iloc[0, 0] = value
            with self.assertRaises(ValueError):
                append_tree_features(original, changed)
        added.loc[42, "TREE_COVER_UNAVAILABLE"] = 0.
        with self.assertRaisesRegex(ValueError, "disagrees"):
            append_tree_features(original, added)


if __name__ == "__main__":
    unittest.main()
