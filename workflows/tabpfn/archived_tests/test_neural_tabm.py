"""Preprocessing and blending checks that do not require PyTorch."""

import unittest

import numpy as np
import pandas as pd

from analysis.experiments.neural_tabm import (
    CATEGORICAL_FEATURES, NUMERIC_FEATURES, NeuralPreprocessor, TabMStyleRegressor,
    TrainSettings, select_blend_weight, torch,
)
from analysis.experiments.baselines.spatiotemporal_hybrid import TARGET


class NeuralTabMTests(unittest.TestCase):
    def _frame(self) -> pd.DataFrame:
        rows = 3
        frame = pd.DataFrame(index=range(rows))
        for column in CATEGORICAL_FEATURES:
            frame[column] = ["a", "b", "a"]
        for position, column in enumerate(NUMERIC_FEATURES):
            frame[column] = [float(position), np.nan, float(position + 2)]
        frame[TARGET] = [1000.0, 2000.0, 3000.0]
        return frame

    def test_preprocessor_maps_unknown_categories_and_imputes_numeric(self):
        fitting = self._frame()
        preprocessor = NeuralPreprocessor.fit(fitting)
        prediction = fitting.iloc[[0]].copy()
        prediction.loc[0, CATEGORICAL_FEATURES[0]] = "unseen"
        numeric, categorical = preprocessor.transform(prediction)
        self.assertEqual(categorical[0, 0], 0)
        self.assertTrue(np.isfinite(numeric).all())
        self.assertEqual(numeric.dtype, np.float32)

    def test_blend_weight_prefers_exact_tree_prediction(self):
        actual = np.array([1000.0, 2000.0, 3000.0])
        tree = actual.copy()
        neural = np.array([1500.0, 1500.0, 1500.0])
        weight, score = select_blend_weight(actual, tree, neural)
        self.assertEqual(weight, 1.0)
        self.assertEqual(score, 0.0)

    @unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
    def test_tabm_members_produce_one_prediction_each(self):
        settings = TrainSettings(epochs=1, ensemble_size=3, hidden_dim=16, depth=2)
        model = TabMStyleRegressor([4, 5], n_numeric=6, settings=settings)
        numeric = torch.zeros((7, 6), dtype=torch.float32)
        categorical = torch.ones((7, 2), dtype=torch.int64)
        self.assertEqual(tuple(model(numeric, categorical).shape), (7, 3))


if __name__ == "__main__":
    unittest.main()
