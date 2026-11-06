"""Default reports never accumulate model binaries or prepared feature caches."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from analysis.pipeline.config import CACHE, EXPERIMENT_REPORTS, MODELS
from analysis.pipeline.paths import feature_cache_path, model_artifact_dir


class ProjectPathTests(unittest.TestCase):
    def test_missing_cache_is_rebuilt_before_refit_and_weights_stay_outside_reports(self):
        from analysis.experiments import model_search
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            stages = {"x_fit": pd.DataFrame({"area": [1., 2., 3.]}),
                      "y_fit": np.array([1000., 1500., 2000.]),
                      "x_val": pd.DataFrame({"area": [4., 5.]}),
                      "y_val": np.array([2500., 3000.]), "actual": np.array([2500., 3000.]),
                      "offset": np.zeros(2), "source_row": np.array([7, 8]),
                      "approval_month": np.array(["2024-04", "2024-05"]), "n_fit": 3}
            def rebuild(fold, out):
                cache = feature_cache_path(out, fold, "compact")
                cache.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump({"categorical": [], "source_sha256": {}, "spec": {},
                             "inner": stages, "outer": stages}, cache)
            def fit(family, stage, categorical, settings, fixed_iterations=None):
                return {"model": "path-check"}, np.array([2400., 2900.]), 2
            with patch.dict(model_search.TRIALS, {"lightgbm": [("compact", {})]}), \
                 patch.object(model_search, "prepare_fold", side_effect=rebuild) as prepare, \
                 patch.object(model_search, "fit_tree", side_effect=fit):
                model_search.run_family("lightgbm", out=output)
            prepare.assert_called_once_with("2024-04", output)
            folder = output / "2024-04/lightgbm"
            self.assertTrue((folder / "results.json").is_file())
            self.assertTrue((folder / "artifacts/model.joblib").is_file())
            self.assertFalse((folder / "model.joblib").exists())

    def test_default_model_storage_mirrors_experiment_names(self):
        report = EXPERIMENT_REPORTS / "comparison_model_search/2024-04/lightgbm"
        self.assertEqual(model_artifact_dir(report), MODELS / "comparison_model_search/2024-04/lightgbm")
        self.assertEqual(model_artifact_dir(EXPERIMENT_REPORTS / "baseline_deep_models/model_a"),
                         MODELS / "baseline_deep_models/model_a")

    def test_default_prepared_cache_is_disposable_and_separate_from_results(self):
        path = feature_cache_path(EXPERIMENT_REPORTS / "comparison_model_search", "2024-04", "compact")
        self.assertEqual(path, CACHE / "comparison_model_search/2024-04/compact/prepared.joblib")
        self.assertFalse(path.is_relative_to(EXPERIMENT_REPORTS))

    def test_custom_experiment_remains_self_contained(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "run"
            self.assertEqual(model_artifact_dir(report), report / "artifacts")
            self.assertEqual(feature_cache_path(report, "2024-04", "compact"),
                             report / ".cache/2024-04/compact/prepared.joblib")


if __name__ == "__main__":
    unittest.main()
