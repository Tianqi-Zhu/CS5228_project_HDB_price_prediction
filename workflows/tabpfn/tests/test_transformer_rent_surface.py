"""Support, interpolation and coordinate invariants for the rental surface."""
import unittest

import numpy as np

from analysis.plot_transformer_rent_surface import interpolation_matrix, metric_xy


class RentSurfaceTests(unittest.TestCase):
    def test_local_constant_field_and_far_unsupported_location(self):
        nodes = np.array([[0, 0], [.1, 0], [0, .1]])
        grid = np.array([[.03, .03], [5., 5.]])
        matrix, supported, nearest, effective = interpolation_matrix(nodes, grid)
        np.testing.assert_array_equal(supported, [True, False])
        self.assertAlmostEqual(float((matrix@np.full(3, 1234.))[0]), 1234.)
        self.assertGreater(nearest[1], .5)
        self.assertGreater(effective[0], 2)
        self.assertEqual(matrix.getrow(1).nnz, 0)

    def test_interpolation_stays_within_neighbor_prediction_range(self):
        nodes = np.array([[0, 0], [.1, 0], [0, .1], [.1, .1]])
        grid = np.array([[.02, .04], [.08, .09]])
        matrix, supported, *_ = interpolation_matrix(nodes, grid)
        result = matrix@np.array([2000., 3000., 4000., 5000.])
        self.assertTrue(supported.all())
        self.assertTrue(((result >= 2000) & (result <= 5000)).all())
        np.testing.assert_allclose(np.asarray(matrix.sum(axis=1)).ravel(), 1.)

    def test_all_local_anchors_contribute_without_fixed_neighbor_truncation(self):
        angle = np.linspace(0, 2*np.pi, 40, endpoint=False)
        nodes = .3*np.column_stack([np.cos(angle), np.sin(angle)])
        matrix, supported, _, effective = interpolation_matrix(nodes, np.array([[0., 0.]]))
        self.assertEqual(matrix.nnz, 40)
        self.assertTrue(supported[0])
        self.assertAlmostEqual(effective[0], 40)

    def test_coordinate_units_and_origin(self):
        xy = metric_xy([103.85, 103.85], [1.35, 1.35+1/111.2])
        np.testing.assert_allclose(xy, [[0., 0.], [0., 1.]], atol=1e-12)

    def test_invalid_geometry_does_not_silently_extrapolate(self):
        with self.assertRaises(ValueError):
            interpolation_matrix(np.zeros((3, 2)), np.zeros((1, 2)), bandwidth=2., radius=1.)


if __name__ == "__main__":
    unittest.main()
