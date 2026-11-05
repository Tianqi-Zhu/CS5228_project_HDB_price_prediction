"""Protect residual reconstruction and restart behavior without making API calls."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import joblib
import numpy as np
import pandas as pd

from analysis.tabpfn_api_77 import ROOT, load_inputs, predict_batches, save_results, write_json


class FakeModel:
    submissions = 0
    interrupt = True

    def submit_predict(self, x, output_type):
        self.size = len(x)
        FakeModel.submissions += 1

    def save_model(self):
        return {"size": self.size}

    @classmethod
    def load_model(cls, record):
        obj = cls()
        obj.size = record["size"]
        return obj

    def wait_for_predict(self):
        if FakeModel.interrupt:
            FakeModel.interrupt = False
            raise TimeoutError("simulated interrupted wait")
        return np.full(self.size, 25.0)

    def get_timings(self):
        return {"fit": None, "predict": None}


class Api77Tests(unittest.TestCase):
    def temporary_directory(self):
        root = ROOT / ".cache/test_tabpfn_api_77"
        root.mkdir(parents=True, exist_ok=True)
        return TemporaryDirectory(dir=root)

    def test_resumes_accepted_prediction_without_resubmitting_or_double_baseline(self):
        FakeModel.submissions, FakeModel.interrupt = 0, True
        data = dict(q=pd.DataFrame({"feature": [1, 2, 3]}, index=[8, 4, 9]),
                    baseline=np.array([1000., 2000., 3000.]))
        client = SimpleNamespace(TabPFNRegressor=FakeModel)
        with self.temporary_directory() as directory:
            cache = Path(directory)
            with self.assertRaises(TimeoutError):
                predict_batches(client, FakeModel(), data, cache, 2)
            values = predict_batches(client, FakeModel(), data, cache, 2)
            np.testing.assert_array_equal(values, [1025., 2025., 3025.])
            self.assertEqual(FakeModel.submissions, 2)
            # Simulate a crash after committing progress but before removing the job.
            write_json(cache / "pending_prediction.json", dict(offset=2, size=1, model={"size": 1}))
            again = predict_batches(client, FakeModel(), data, cache, 2)
            np.testing.assert_array_equal(again, values)
            self.assertEqual(FakeModel.submissions, 2)

    def test_loader_preserves_categories_missing_values_and_target_alignment(self):
        x = pd.DataFrame(np.zeros((2, 77)), columns=[f"f{i}" for i in range(77)], index=[9, 3])
        x["f0"] = pd.Categorical(["B", "A"], categories=["A", "B", "C"])
        x.loc[9, "f1"] = np.nan
        q = x.copy()
        q.index = [12, 7]
        b = dict(X_train=x, X_query=q, y_train=[1100., 2400.],
                 train_baseline=[1000., 2000.], query_baseline=[3000., 4000.],
                 y_query=[3100., 4200.], train_month=["2024-04", "2024-05"],
                 query_month=["2024-06", "2024-07"])
        with self.temporary_directory() as directory:
            path = Path(directory) / "features.joblib"
            joblib.dump(b, path)
            data = load_inputs(path, "validation")
            pd.testing.assert_frame_equal(data["x"], x)
            pd.testing.assert_frame_equal(data["q"], q)
            np.testing.assert_array_equal(data["target"], [100., 400.])
            b["train_month"] = ["2024-07", "2024-05"]
            joblib.dump(b, path)
            with self.assertRaisesRegex(ValueError, "Validation must follow"):
                load_inputs(path, "validation")

    def test_metrics_use_reconstructed_rents_and_keep_query_order(self):
        data = dict(q=pd.DataFrame(index=[9, 1]), month=["2024-06", "2024-07"],
                    actual=np.array([1000., 2000.]))
        with self.temporary_directory() as directory:
            out = Path(directory)
            save_results(data, np.array([1100., 1900.]), SimpleNamespace(query_rows=None), out)
            result = pd.read_csv(out / "predictions.csv")
            self.assertEqual(result.row_id.tolist(), [9, 1])
            self.assertEqual(pd.read_json(out / "metrics.json", typ="series").rmse_sgd, 100.)


if __name__ == "__main__":
    unittest.main()
