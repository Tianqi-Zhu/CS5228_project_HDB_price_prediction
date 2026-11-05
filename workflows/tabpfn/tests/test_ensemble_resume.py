"""Regression check for persisted prediction precision during calibration."""

from pathlib import Path
import shutil
import unittest
from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
import pandas as pd

import analysis.compare_tabpfn_catboost_ensemble as experiment


class EnsembleResumeTests(unittest.TestCase):
    def test_fresh_and_resumed_residual_predictions_are_identical(self):
        cache = (Path(__file__).resolve().parents[1] / ".cache").resolve()
        root = cache / f"ensemble_resume_test_{uuid4().hex}"
        root.mkdir(parents=True)

        def cleanup():
            if not root.resolve().is_relative_to(cache):
                raise ValueError("Test cleanup escaped the workspace cache")
            shutil.rmtree(root)

        self.addCleanup(cleanup)
        output = root / "report"
        output.mkdir()
        x = pd.DataFrame({"feature": [1., 2.]}, index=[10, 11])
        q = pd.DataFrame({"feature": [3.]}, index=[12])
        bundle = {"X_train": x, "X_query": q, "y_train": np.array([10., 11.]),
                  "y_query": np.array([22.]), "train_baseline": np.array([9., 9.]),
                  "query_baseline": np.array([20.]), "train_month": ["2021-01", "2021-01"],
                  "query_month": ["2021-02"]}
        model = MagicMock()
        model.predict.return_value = np.array([1.123456789012])
        model.save_model.side_effect = lambda path: Path(path).write_text("test model")
        with patch.object(experiment, "ROOT", root), patch.object(experiment, "OUT", output), \
                patch.object(experiment, "CatBoostRegressor", return_value=model) as constructor:
            fresh = experiment.fit_catboost(bundle, {}, "cb77residual", "test")
            resumed = experiment.fit_catboost(bundle, {}, "cb77residual", "test")
        np.testing.assert_array_equal(fresh, resumed)
        np.testing.assert_array_equal(fresh, [21.12345679])
        np.testing.assert_array_equal(model.fit.call_args.args[1], [1., 2.])
        self.assertEqual(constructor.call_count, 1)


if __name__ == "__main__":
    unittest.main()
