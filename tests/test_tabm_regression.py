"""Leakage-safe preprocessing, TabM member loss, and serialization checks."""

import io
import unittest

import numpy as np
import pandas as pd
import torch

from analysis.experiments.tabm_regression import TabMPreprocessor, make_model, member_mse, predict


class TabMTests(unittest.TestCase):
    def test_preprocessing_uses_fit_only_and_unknown_categories(self):
        fit = pd.DataFrame({"a": [1., 3., np.nan, 5.], "b": [7., 7., 7., 7.],
                            "c": pd.Categorical(["x", "y", "x", "y"])})
        query = pd.DataFrame({"a": [1000., np.nan], "b": [100., 7.],
                              "c": pd.Categorical(["new", None])})
        pre = TabMPreprocessor().fit(fit)
        median, mean = pre.median.copy(), pre.mean.copy()
        numeric, category = pre.transform(query)
        np.testing.assert_array_equal(pre.median, median)
        np.testing.assert_array_equal(pre.mean, mean)
        np.testing.assert_array_equal(category, [[0], [0]])
        self.assertEqual(pre.cardinalities, [3])
        self.assertTrue(np.isfinite(numeric).all())
        with self.assertRaises(ValueError):
            pre.transform(query[["c", "b", "a"]])

    def test_members_trained_individually(self):
        values = torch.tensor([[[1.], [-1.]]], requires_grad=True)
        loss = member_mse(values, torch.zeros(1))
        self.assertEqual(loss.item(), 1.)
        loss.backward()
        self.assertTrue(torch.isfinite(values.grad).all())
        self.assertFalse(torch.equal(values.grad, torch.zeros_like(values.grad)))

    def test_official_model_roundtrip_and_residual_reconstruction(self):
        torch.manual_seed(42)
        num = np.random.default_rng(42).normal(size=(64, 3)).astype(np.float32)
        cat = torch.zeros((64, 1), dtype=torch.long)
        cfg = {"embedding": "ple", "k": 2, "lr": .002}
        model, bins = make_model(cfg, num, [2])
        expected = predict(model, torch.as_tensor(num), cat, 10., 100.)
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        clone, _ = make_model(cfg, num, [2], bins=bins)
        stream.seek(0)
        clone.load_state_dict(torch.load(stream, weights_only=True))
        actual = predict(clone, torch.as_tensor(num), cat, 10., 100.)
        np.testing.assert_array_equal(expected, actual)
        baseline = np.full(64, 3000.)
        np.testing.assert_allclose((actual+baseline)-baseline, actual)


if __name__ == "__main__":
    unittest.main()
