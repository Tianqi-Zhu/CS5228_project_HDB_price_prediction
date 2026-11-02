import unittest
import numpy as np
import pandas as pd

from analysis.experiments.temporal_feature_views import LEVEL_COLUMNS, VARIANTS, temporal_view, simplex_weights
from analysis.experiments.feature_diversified_ensemble import optimize_weights


class TemporalViewTests(unittest.TestCase):
    def fixture(self):
        months = pd.period_range("2021-01", periods=12, freq="M")
        x = pd.DataFrame({c: np.arange(12, dtype=float) + 100 for c in LEVEL_COLUMNS}, index=range(90, 102))
        x["APPROVAL_MONTH_INDEX"] = np.arange(12)
        x["APPROVAL_MONTH_OF_YEAR"] = pd.Categorical(months.month, categories=range(1, 13))
        x["TEMP_FORECAST_HORIZON"] = np.arange(1, 13, dtype=float)
        x["AREA_RECENT_MINUS_LONG"] = 15.
        x["RECENT_BLOCK_FLAT_3M_COUNT"] = 12.
        x["IDENTITY"] = pd.Categorical(["a", "b"] * 6)
        x.loc[90, LEVEL_COLUMNS] = np.nan
        x.loc[90, "TEMP_FORECAST_HORIZON"] = np.nan
        return x, months, np.arange(12, dtype=float) + 80

    def test_row_local_transforms_do_not_consume_other_rows_or_targets(self):
        x, months, b = self.fixture()
        for variant in VARIANTS:
            full = temporal_view(x, months, b, variant)
            part = temporal_view(x.iloc[:5], months[:5], b[:5], variant)
            pd.testing.assert_frame_equal(full.iloc[:5], part)
            poisoned = temporal_view(x.assign(MONTHLY_RENT=1e15), months, b, variant)
            pd.testing.assert_frame_equal(full, poisoned.drop(columns="MONTHLY_RENT"))
            pd.testing.assert_series_equal(full.IDENTITY, x.IDENTITY)

    def test_relative_coordinates_preserve_missingness_and_are_level_shift_invariant(self):
        x, months, b = self.fixture()
        z = temporal_view(x, months, b, "relative_history")
        shifted = x.copy(); shifted[LEVEL_COLUMNS] += 2500
        other = temporal_view(shifted, months, b + 2500, "relative_history")
        pd.testing.assert_frame_equal(z, other)
        for c in LEVEL_COLUMNS:
            np.testing.assert_array_equal(z[c + "_REL_BASE"].isna(), x[c].isna())
        self.assertEqual(z.shape, x.shape)

    def test_horizon_interactions_preserve_originals_and_warmup(self):
        x, months, b = self.fixture(); z = temporal_view(x, months, b, "horizon_dynamics")
        pd.testing.assert_frame_equal(z[x.columns], x)
        self.assertEqual(z.shape[1], x.shape[1] + 6)
        self.assertTrue(z.filter(like="VIEW_").loc[90].isna().all())
        self.assertAlmostEqual(z.loc[93, "VIEW_BLOCK_SUPPORT_DECAY"], 3.)
        self.assertFalse(np.isinf(z.select_dtypes("number").to_numpy()).any())

    def test_solver_matches_prior_three_member_and_supports_five(self):
        rng = np.random.default_rng(57); p = rng.normal(size=(200, 3)); y = rng.normal(size=200)
        w, _ = simplex_weights(y, p); prior, _ = optimize_weights(y, p)
        np.testing.assert_allclose(w, prior, atol=1e-10)
        p = np.eye(5) * 100 + 1000; y = np.full(5, 1020.)
        w, _ = simplex_weights(y, p)
        np.testing.assert_allclose(w, np.full(5, .2), atol=1e-10)
        duplicate = np.repeat(p[:, :1], 5, axis=1)
        w, _ = simplex_weights(y, duplicate)
        np.testing.assert_array_equal(w, [1, 0, 0, 0, 0])

    def test_invalid_baseline_and_horizon_rejected(self):
        x, months, b = self.fixture()
        with self.assertRaises(ValueError): temporal_view(x, months, b[:-1], "relative_history")
        x.loc[91, "TEMP_FORECAST_HORIZON"] = 0
        with self.assertRaises(ValueError): temporal_view(x, months, b, "horizon_dynamics")

    def test_five_member_weighted_solution_matches_independent_slsqp(self):
        from scipy.optimize import minimize
        rng = np.random.default_rng(209)
        p = rng.normal(size=(150, 5)); y = p @ np.array([.4, .3, 0., .2, .1]) + rng.normal(size=150) * .2
        v = rng.uniform(.1, 2., len(p)); v /= v.sum()
        w, info = simplex_weights(y, p, v)
        reference = minimize(lambda z: np.sum(v * (p @ z - y)**2), np.full(5, .2),
                             method="SLSQP", bounds=[(0., 1.)] * 5,
                             constraints={"type": "eq", "fun": lambda z: z.sum() - 1},
                             options={"ftol": 1e-13, "maxiter": 1000})
        self.assertTrue(reference.success)
        self.assertAlmostEqual(info["weighted_mse_sgd2"], reference.fun, places=10)
        self.assertTrue((w >= 0).all()); self.assertAlmostEqual(w.sum(), 1.)


if __name__ == "__main__": unittest.main()
