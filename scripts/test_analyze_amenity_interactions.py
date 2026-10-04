"""Focused checks for joint-cell interaction contrasts."""
import unittest

import numpy as np

from analyze_amenity_interactions import cell_index, interaction_vector, distance_3bin


class InteractionTests(unittest.TestCase):
    def test_bins_are_right_closed(self):
        actual = distance_3bin(np.array([0, .5, .500001, 1, 1.000001])).astype(str).tolist()
        self.assertEqual(actual, ["0–0.5", "0–0.5", "0.5–1", "0.5–1", "1+"])

    def test_interaction_uses_all_four_cells(self):
        # Coefficients are row-major for the eight non-reference joint cells.
        beta = np.array([10, 20, 30, 44, 50, 60, 70, 80], dtype=float)
        vector = interaction_vector(1, 1)
        self.assertEqual(cell_index(0, 0), None)
        self.assertEqual(cell_index(1, 1), 3)
        self.assertAlmostEqual(vector @ beta, 44 - 30 - 10)
        covariance = np.diag(np.arange(1, 9, dtype=float))
        self.assertAlmostEqual(vector @ covariance @ vector, 4 + 3 + 1)

    def test_reference_row_and_column_have_no_interaction(self):
        self.assertTrue(np.array_equal(interaction_vector(0, 2), np.zeros(8)))
        self.assertTrue(np.array_equal(interaction_vector(2, 0), np.zeros(8)))


if __name__ == "__main__":
    unittest.main()
