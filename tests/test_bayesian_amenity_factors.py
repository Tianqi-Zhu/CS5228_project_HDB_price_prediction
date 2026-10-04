"""Conjugate posterior and amenity representation checks."""

import unittest

import numpy as np
from scipy import sparse

from analysis.experiments.bayesian_amenity_factors import (
    Design, fit_posterior, spline_basis,
)


class BayesianAmenityTests(unittest.TestCase):
    def test_normal_prior_shrinks_mean_and_predictive_interval_contains_it(self):
        x = sparse.csr_matrix([[0.0], [1.0], [1.0]])
        design = Design(x, sparse.csr_matrix([[1.0]]), np.array([[3.0]]), {"intercept": slice(0, 1)})
        posterior = fit_posterior(design, np.array([100.0, 200.0, 300.0]))
        # Without the prior, the fitted correction is 50; the prior shrinks it to 20.
        self.assertAlmostEqual(posterior.predict(design.prediction)[0], 220.0)
        lower, upper = posterior.interval(design.prediction)
        self.assertLess(lower[0], 220.0)
        self.assertGreater(upper[0], 220.0)

    def test_distance_spline_is_linear_between_knots(self):
        basis = spline_basis(np.array([.25, .50, 1.00]), (.30, .60))
        self.assertEqual(basis.shape, (3, 3))
        np.testing.assert_allclose(basis[0], [.25, 0.0, 0.0])
        np.testing.assert_allclose(basis[1], [.50, .20, 0.0])
        np.testing.assert_allclose(basis[2], [1.0, .70, .40])


if __name__ == "__main__":
    unittest.main()
