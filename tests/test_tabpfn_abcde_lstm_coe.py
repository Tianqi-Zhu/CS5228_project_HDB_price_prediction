"""Check six-member optimization, including solutions using the new member."""
import unittest

import numpy as np

from analysis.compare_tabpfn_abcde_lstm_coe import fit_simplex


class LSTMCOEEnsembleTests(unittest.TestCase):
    def test_all_six_members_can_be_required(self):
        weights = fit_simplex(np.ones(6), 6 * np.eye(6), np.ones(6))
        np.testing.assert_allclose(weights, np.ones(6) / 6, atol=1e-10)

    def test_sixth_member_can_be_the_only_useful_predictor(self):
        actual = np.arange(10, dtype=float)
        predictions = np.column_stack([actual + i for i in range(1, 6)] + [actual])
        weights = fit_simplex(actual, predictions, np.ones(10))
        np.testing.assert_allclose(weights, [0, 0, 0, 0, 0, 1], atol=1e-12, rtol=0)

    def test_six_member_calibration_cannot_lose_to_five(self):
        rng = np.random.default_rng(42)
        actual = rng.normal(size=90)
        predictions = rng.normal(size=(90, 6))
        population = rng.uniform(.1, 3, size=90)
        five = fit_simplex(actual, predictions[:, :5], population)
        six = fit_simplex(actual, predictions, population)
        self.assertLessEqual(
            np.average((predictions @ six - actual)**2, weights=population),
            np.average((predictions[:, :5] @ five - actual)**2, weights=population) + 1e-10,
        )


if __name__ == "__main__":
    unittest.main()
