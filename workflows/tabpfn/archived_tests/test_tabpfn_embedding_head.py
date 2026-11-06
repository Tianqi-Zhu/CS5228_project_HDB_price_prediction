"""Focused checks for frozen-embedding head inputs and selection math."""

import numpy as np
import pandas as pd
import unittest

from analysis.experiments.tabpfn_embedding_head import (HeadFeatures, numeric_features,
                                                        population_weights, shrink_choice)


class EmbeddingHeadTest(unittest.TestCase):
    def test_repeated_forecast_origins_share_one_month_population(self):
        months = ["2023-12"] * 4 + ["2024-01"] * 2
        origins = ["2023-05"] * 2 + ["2023-08"] * 2 + ["2023-08"] * 2
        result = population_weights(months, origins, [10] * 4 + [20] * 2)
        np.testing.assert_array_equal(result, [5] * 4 + [20] * 2)
        self.assertEqual(sum(result[:4]), 20)
        self.assertEqual(sum(result[4:]), 40)


    def test_numeric_features_use_query_values_and_origin(self):
        from analysis.experiments.tabpfn_embedding_head import NUMERIC

        frame = pd.DataFrame({name: [1., 2.] for name in NUMERIC})
        x = numeric_features(frame, ["2023-12", "2024-01"], ["2023-08", "2023-08"], [2800, 3000])
        self.assertEqual(x.shape, (2, len(NUMERIC) + 4))
        np.testing.assert_array_equal(x[:, len(NUMERIC)], [2800, 3000])
        self.assertFalse(np.allclose(x[0, -2:], x[1, -2:]))


    def test_preprocessor_uses_only_training_rows(self):
        numeric = np.array([[0.], [2.], [1000.]])
        latent = np.array([[1., 2.], [3., 4.], [999., 999.]])
        prep = HeadFeatures("raw").fit(numeric[:2], latent[:2])
        transformed = prep.transform(numeric, latent)
        np.testing.assert_allclose(transformed[:2].mean(axis=0), 0, atol=1e-7)
        self.assertGreater(np.max(np.abs(transformed[2])), 100)


    def test_shrinkage_can_select_no_head_without_claiming_gain(self):
        actual = np.array([1., 2.])
        base = np.array([1., 2.])
        gamma, curve = shrink_choice(actual, base, np.array([100., -100.]), [1., 1.])
        self.assertEqual(gamma, 0)
        self.assertEqual(curve[gamma], 0)
        self.assertGreater(curve[1], 0)


if __name__ == "__main__":
    unittest.main()
