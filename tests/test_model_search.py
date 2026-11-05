"""Training-only preprocessing and official architecture smoke checks."""

import unittest
from pathlib import Path
import tempfile
import json

import numpy as np
import pandas as pd
import torch

from analysis.pipeline.neural_models import NeuralFrameEncoder, build_network, enable_search_dependencies


class ModelSearchTests(unittest.TestCase):
    def sample(self):
        return pd.DataFrame({"kind": ["a", "b", "a", "b"], "area": [60., 80., np.nan, 100.],
            "APPROVAL_MONTH_INDEX": [0., 1., 2., 3.], "HIST_AREA_FLAT_COUNT": [0., 2., 10., 20.]})

    def test_unknown_category_and_future_time_use_training_statistics(self):
        frame = self.sample()
        encoder = NeuralFrameEncoder(("kind",), quantile=True).fit(frame)
        prediction = frame.iloc[[0]].copy()
        prediction["kind"] = "unknown"
        prediction["APPROVAL_MONTH_INDEX"] = 100
        cats, nums = encoder.transform(prediction)
        self.assertEqual(cats[0, 0], 0)
        self.assertEqual(encoder.cardinalities, [3])
        self.assertTrue(np.isfinite(nums).all())
        time_position = encoder.numeric_.index("APPROVAL_MONTH_INDEX")
        self.assertGreater(nums[0, time_position], encoder.transform(frame)[1][:, time_position].max())

    def test_missing_indicator_and_inputs_are_preserved(self):
        frame = self.sample()
        original = frame.copy(deep=True)
        encoder = NeuralFrameEncoder(("kind",), quantile=False).fit(frame)
        _, nums = encoder.transform(frame)
        self.assertEqual(nums.shape, (4, 4))
        self.assertEqual(nums[:, -1].tolist(), [0., 0., 1., 0.])
        pd.testing.assert_frame_equal(frame, original)

    def test_embedding_mlp_and_official_ft_have_finite_regression_outputs(self):
        frame = self.sample()
        encoder = NeuralFrameEncoder(("kind",), quantile=False).fit(frame)
        cats, nums = encoder.transform(frame)
        enable_search_dependencies()
        for family in ("embedding_mlp", "embedding_resnet", "ft_transformer"):
            with self.subTest(family=family):
                model = build_network(family, encoder, nums.shape[1], {"width": 32, "layers": 1})
                output = model(torch.from_numpy(nums), torch.from_numpy(cats)).reshape(-1)
                self.assertEqual(tuple(output.shape), (4,))
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()

    def test_rare_categories_share_unknown_code(self):
        frame = self.sample()
        frame.loc[0, "kind"] = "rare"
        encoder = NeuralFrameEncoder(("kind",), min_category_count=2).fit(frame)
        cats, _ = encoder.transform(frame)
        self.assertEqual(cats[:, 0].tolist(), [0, 1, 0, 1])
        self.assertEqual(encoder.cardinalities, [2])

    def test_fixed_tree_refit_ignores_outer_targets(self):
        from analysis.experiments.model_search import fit_tree
        frame = pd.DataFrame({"area": np.arange(40, dtype=float),
            "kind": pd.Categorical(["a", "b"] * 20)})
        stage = {"x_fit": frame, "x_val": frame.iloc[:5].copy(),
            "y_fit": np.arange(40, dtype=float) * 20 + 2000, "y_val": np.zeros(5)}
        params = {"num_leaves": 4, "min_child_samples": 2, "reg_lambda": 3.}
        _, before, _ = fit_tree("lightgbm", stage, ("kind",), params, fixed_iterations=5)
        stage["y_val"] = np.full(5, 1e9)
        _, after, _ = fit_tree("lightgbm", stage, ("kind",), params, fixed_iterations=5)
        np.testing.assert_array_equal(before, after)

    def test_ensemble_weights_ignore_outer_targets(self):
        from analysis.experiments.model_search import blend
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, inner, outer in (("a", [10., 20.], [60., 70.]), ("b", [20., 30.], [50., 60.])):
                directory = root / "2024-04" / name
                (directory / "trial_0").mkdir(parents=True)
                metadata = {"family": name, "scores": {"rmse_sgd": 10.}, "selection": {
                    "trial_index": 0, "inner_scores": {"rmse_sgd": 0. if name == "a" else 10.}}}
                (directory / "results.json").write_text(json.dumps(metadata), encoding="utf-8")
                pd.DataFrame({"source_row": [1, 2], "approval_month": ["2023-10"] * 2,
                    "actual": [10., 20.], "prediction": inner}).to_csv(directory / "trial_0" / "predictions.csv", index=False)
                pd.DataFrame({"source_row": [3, 4], "approval_month": ["2024-04"] * 2,
                    "actual": [50., 60.], "prediction": outer}).to_csv(directory / "predictions.csv", index=False)
            result = blend("2024-04", root)
            self.assertEqual(result["weights"], {"a": 1., "b": 0.})
            self.assertEqual(result["scores"]["rmse_sgd"], 10.)


if __name__ == "__main__":
    unittest.main()
