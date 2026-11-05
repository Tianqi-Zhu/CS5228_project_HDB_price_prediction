import unittest

import numpy as np
import pandas as pd
import torch

from analysis.evaluate_transformer_hierarchical_calibration import correction, fit
from analysis.evaluate_transformer_tabpfn_distillation import student_features
from analysis.evaluate_transformer_forced_decay import predict as decayed_predict
from analysis.experiments.conditioned_rent_correction import RentCorrection
from analysis.experiments.rental_context_attention import ComparableAttention, retrieve
from analysis.run_transformer_rent_innovations import innovations
from analysis.run_transformer_horizon_weighting import weights


class RankedExperimentTests(unittest.TestCase):
    def test_stable_head_is_exact_zero_correction(self):
        model = RentCorrection(conditioned=True, stabilized=True).eval()
        with torch.no_grad():
            value = model(torch.randn(7, 128), torch.randn(7, 12, 34), torch.randn(7, 4))
        self.assertTrue(torch.equal(value, torch.zeros_like(value)))

    def test_property_conditioning_changes_prediction(self):
        model = RentCorrection(conditioned=True, stabilized=False).eval()
        seq, context = torch.randn(1, 12, 34), torch.randn(1, 4)
        with torch.no_grad():
            a = model(torch.randn(1, 128), seq, context)
            b = model(torch.randn(1, 128), seq, context)
        self.assertGreater(float((a-b).abs().max()), 1e-8)

    def test_hierarchical_calibration_shrinks_and_falls_back(self):
        frame = pd.DataFrame({
            "month": ["2023-01"] * 4, "weight": [1.] * 4,
            "actual": [110., 110., 90., 90.],
            "predicted": [100.] * 4,
            "TOWN_CLEAN": ["A", "A", "B", "B"],
        })
        model = fit(frame, hierarchy=(("TOWN_CLEAN",),), shrink=2., half_life=0.,
                    latest_month="2023-02")
        observed = correction(model, frame)
        self.assertAlmostEqual(observed[0], 5.)
        self.assertAlmostEqual(observed[2], -5.)
        unknown = frame.iloc[:1].copy()
        unknown["TOWN_CLEAN"] = "new"
        self.assertAlmostEqual(correction(model, unknown)[0], 0.)
        self.assertTrue(np.isfinite(observed).all())

    def test_comparables_hide_future_rents(self):
        bank = pd.DataFrame({
            "LATITUDE": [1.35, 1.35], "LONGITUDE": [103.85, 103.85],
            "_MONTH": pd.PeriodIndex(["2023-01", "2023-03"], freq="M"),
            "FLAT_TYPE_CLEAN": ["4-room", "4-room"],
            "TOWN": ["a", "a"], "STREET": ["x", "x"], "BLOCK": ["1", "2"],
            "FLOOR_AREA_SQM": [95., 95.], "MONTHLY_RENT": [3000., 9000.],
        })
        query = bank.iloc[:1].copy()
        query["_MONTH"] = pd.PeriodIndex(["2023-04"], freq="M")
        tokens, present = retrieve(bank, query, pd.PeriodIndex(["2023-02"], freq="M"),
                                   np.array([3000.]), slots=2, search=2)
        self.assertEqual(int(present.sum()), 1)
        self.assertAlmostEqual(float(tokens[0, 0, 0]), 0.)

    def test_context_head_has_safe_empty_support(self):
        model = ComparableAttention().eval()
        with torch.no_grad():
            value = model(torch.randn(2, 128), torch.randn(2, 4),
                          torch.zeros(2, 32, 7), torch.zeros(2, 32, dtype=torch.bool))
        self.assertTrue(torch.isfinite(value).all())
        self.assertTrue(torch.equal(value, torch.zeros_like(value)))

    def test_innovations_use_only_preceding_changes(self):
        values = np.zeros((1, 12, 34), np.float32)
        values[0, 1:5, 1] = [2., 4., 6., 10.]
        values[0, 1:5, 18] = 1.
        values[0, :, 0] = 99.
        values[0, :, 17] = 1.
        transformed = innovations(values)
        self.assertAlmostEqual(float(transformed[0, 4, 1]), 6.)
        self.assertTrue(np.all(transformed[0, :, 0] == 0))
        later = values.copy()
        later[0, 8, 1] = 500.
        later[0, 8, 18] = 1.
        self.assertAlmostEqual(float(innovations(later)[0, 4, 1]), 6.)

    def test_horizon_weights_are_normalized_and_emphasize_long(self):
        horizon = np.array([1, 2, 3, 12])
        values = weights(horizon, "long2").numpy()
        self.assertAlmostEqual(float(values.mean()), 1.)
        self.assertGreater(values[-1], values[0])
        self.assertTrue(np.isfinite(weights(horizon, "balanced").numpy()).all())

    def test_distillation_student_cannot_read_teacher_or_label(self):
        frame = pd.DataFrame({
            "predicted": [3000.], "teacher": [3200.], "actual": [3500.],
            "FLOOR_AREA_SQM": [90.], "FLAT_AGE_YEARS": [20.],
            "LATITUDE": [1.35], "LONGITUDE": [103.85],
            "DISTANCE_TO_CBD_KM": [8.], "TEMP_FORECAST_HORIZON": [3.],
            "FLAT_TYPE_CLEAN": ["4-room"],
        })
        baseline = student_features(frame)
        changed = frame.copy()
        changed["teacher"] = 10000.
        changed["actual"] = 1.
        np.testing.assert_array_equal(student_features(changed), baseline)

    def test_forced_decay_uses_only_forecast_and_horizon(self):
        frame = pd.DataFrame({"predicted": [3000., 3000.],
                              "temporal": [3200., 3200.],
                              "horizon": [1, 4], "actual": [1., 2.]})
        result = decayed_predict(frame, .5, 3.)
        self.assertAlmostEqual(result[0], 3100.)
        self.assertAlmostEqual(result[1], 3000.+100.*np.exp(-1.))
        frame["actual"] = [10000., 10000.]
        np.testing.assert_allclose(decayed_predict(frame, .5, 3.), result)


if __name__ == "__main__":
    unittest.main()
