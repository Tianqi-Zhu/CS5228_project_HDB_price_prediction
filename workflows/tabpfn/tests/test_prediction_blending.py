import unittest

import numpy as np

from analysis.experiments.prediction_blending import blend, convex_weight


class PredictionBlendingTests(unittest.TestCase):
    def test_recovers_known_convex_mixture(self):
        first, second = np.array([0., 2., 8.]), np.array([5., 1., 3.])
        actual = .3 * first + .7 * second
        self.assertAlmostEqual(convex_weight(actual, first, second), .3)
        np.testing.assert_allclose(blend(first, second, .3), actual)

    def test_projects_to_endpoints(self):
        self.assertEqual(convex_weight([3., 3.], [2., 2.], [1., 1.]), 1.)
        self.assertEqual(convex_weight([0., 0.], [2., 2.], [1., 1.]), 0.)

    def test_identical_predictions_are_well_defined(self):
        self.assertEqual(convex_weight([1., 2.], [3., 4.], [3., 4.]), .5)

    def test_sample_weights_match_repeated_observations(self):
        actual, first, second = [np.array(v) for v in ([1., 4.], [0., 5.], [2., 1.])]
        weighted = convex_weight(actual, first, second, [2., 5.])
        repeated = convex_weight(np.repeat(actual, [2, 5]), np.repeat(first, [2, 5]), np.repeat(second, [2, 5]))
        self.assertAlmostEqual(weighted, repeated)

    def test_fitted_weight_minimizes_calibration_loss(self):
        actual, first, second = [np.array(v) for v in ([1., 7., 3.], [2., 5., 2.], [0., 8., 6.])]
        best = convex_weight(actual, first, second)
        loss = np.mean((blend(first, second, best) - actual) ** 2)
        for candidate in np.linspace(0, 1, 101):
            self.assertLessEqual(loss, np.mean((blend(first, second, candidate) - actual) ** 2) + 1e-12)

    def test_invalid_values_are_rejected(self):
        for values in (([], [], []), ([1.], [1., 2.], [1.]), ([np.nan], [1.], [2.])):
            with self.assertRaises(ValueError):
                convex_weight(*values)
        with self.assertRaises(ValueError):
            convex_weight([1.], [2.], [3.], [-1.])
        with self.assertRaises(ValueError):
            blend([1.], [2.], 1.1)


if __name__ == "__main__":
    unittest.main()
