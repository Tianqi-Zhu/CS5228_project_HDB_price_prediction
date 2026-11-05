"""Information-cutoff and aggregation checks for industry-stock experiments."""

import unittest

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis.experiments.stock_sector_signals import (
    SECTORS, adjusted_rent_index, attach_stock_features, candidate_features,
    monthly_forecasts, ridge_forecast, select_monthly, stock_panel,
)


class StockSectorSignalsTests(unittest.TestCase):
    def stock_fixture(self):
        symbols = sorted(set(sum(SECTORS.values(), [])))
        return pd.DataFrame([
            {"SYMBOL": s, "DATE": str(m.to_timestamp() + pd.Timedelta(days=20)),
             "ADJUSTED_CLOSE": 100 * (1.02 ** i), "NAME": s}
            for s in symbols for i, m in enumerate(pd.period_range("2021-01", "2025-04", freq="M"))
        ])

    def test_returns_are_not_price_weighted_and_future_is_excluded(self):
        raw = self.stock_fixture()
        raw.loc[raw.SYMBOL == "D05.SI", "ADJUSTED_CLOSE"] *= 1000
        panel, _ = stock_panel(raw, pd.Period("2025-03", "M"))
        self.assertAlmostEqual(panel.loc["2021-02", "banks__return_1m"], .02)
        changed = raw.copy()
        changed.loc[changed.DATE > "2025-03-31", "ADJUSTED_CLOSE"] *= 1000
        pd.testing.assert_frame_equal(panel, stock_panel(changed, pd.Period("2025-03", "M"))[0])

    def test_missing_month_is_not_bridged(self):
        raw = self.stock_fixture()
        raw = raw[~((raw.SYMBOL == "D05.SI") & raw.DATE.str.startswith("2021-03"))]
        panel, coverage = stock_panel(raw, pd.Period("2025-03", "M"))
        # Two of three banks is below the 80% coverage rule in March AND April.
        self.assertTrue(pd.isna(panel.loc["2021-03", "banks__return_1m"]))
        self.assertTrue(pd.isna(panel.loc["2021-04", "banks__return_1m"]))
        self.assertEqual(coverage.query("basket == 'banks' and month == '2021-04'").observed_returns.iloc[0], 2)

    def test_calendar_lag_and_fixed_origin_cap(self):
        idx = pd.period_range("2024-01", "2024-07", freq="M")
        features = pd.DataFrame({"signal": np.arange(len(idx))}, index=idx)
        rows = pd.DataFrame({"RENT_APPROVAL_DATE": ["2024-03", "2024-05", "2024-07"]})
        self.assertEqual(attach_stock_features(rows, features, ["signal"]).signal.tolist(), [1, 3, 5])
        fixed = attach_stock_features(rows, features, ["signal"], pd.Period("2024-02", "M"))
        self.assertEqual(fixed.signal.tolist(), [1, 1, 1])

    def test_delayed_features_use_calendar_months(self):
        panel, _ = stock_panel(self.stock_fixture(), pd.Period("2025-03", "M"))
        f, _ = candidate_features(panel)
        self.assertEqual(f.loc["2024-06", "banks__momentum_3m__delay5"], panel.loc["2024-01", "banks__momentum_3m"])

    def test_numpy_ridge_matches_training_only_sklearn_pipeline(self):
        rng = np.random.default_rng(1)
        x, y, q = rng.normal(size=(15, 4)), rng.normal(size=15), rng.normal(size=4)
        x[0, 0], q[1] = np.nan, np.nan
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10))
        model.fit(x, y)
        self.assertAlmostEqual(ridge_forecast(x, y, q)[0], model.predict(q.reshape(1, -1))[0])

    def test_direct_horizons_never_train_on_unmatured_labels(self):
        idx = pd.period_range("2021-01", "2025-03", freq="M")
        rent = pd.DataFrame({"adjusted_rent": 2000 + np.arange(len(idx)) * 12 + np.sin(np.arange(len(idx)))}, index=idx)
        signals = pd.DataFrame({"stock": np.sin(np.arange(len(idx)))}, index=idx)
        variants = {"rent_ar": [], "sector": ["stock"]}
        before = monthly_forecasts(rent, signals, variants)
        self.assertTrue((before.max_train_target <= before.origin).all())
        changed_rent, changed_signals = rent.copy(), signals.copy()
        changed_rent.loc["2024-06":, "adjusted_rent"] += 50000
        changed_signals.loc["2024-06":, "stock"] += 50000
        after = monthly_forecasts(changed_rent, changed_signals, variants)
        keys = ["origin", "target_month", "horizon", "variant", "prediction"]
        pd.testing.assert_frame_equal(before.loc[before.origin <= "2024-05", keys].reset_index(drop=True),
                                      after.loc[after.origin <= "2024-05", keys].reset_index(drop=True))

    def test_property_adjustment_fits_reference_year_only(self):
        idx = pd.period_range("2021-01", "2025-03", freq="M")
        raw = pd.DataFrame([{"RENT_APPROVAL_DATE": str(m), "TOWN": "a", "FLAT_TYPE": "3 room",
                             "FLAT_MODEL": "model a", "FLOOR_AREA_SQM": 70., "LEASE_COMMENCE_DATE": 2000,
                             "MONTHLY_RENT": 2000 + i * 20} for i, m in enumerate(idx)])
        before = adjusted_rent_index(raw)
        raw.loc[raw.RENT_APPROVAL_DATE >= "2022-01", "MONTHLY_RENT"] += 10000
        after = adjusted_rent_index(raw)
        pd.testing.assert_frame_equal(before.loc[:"2021-12"], after.loc[:"2021-12"])

    def test_outer_scores_do_not_select_sector(self):
        rows = []
        for phase in ("development", "outer"):
            for h in (1, 3, 6):
                for variant, error in (("persistence", 10), ("rent_ar", 9), ("momentum_3m", 11),
                                       ("broad_sg_lag1", 8), ("banks_lag1", 7), ("developers_lag1", 12)):
                    rows.append({"phase": phase, "horizon": h, "variant": variant, "rmse_sgd": error})
        scores = pd.DataFrame(rows)
        before, _ = select_monthly(scores)
        scores.loc[(scores.phase == "outer") & (scores.variant == "developers_lag1"), "rmse_sgd"] = 0
        after, _ = select_monthly(scores)
        self.assertEqual(before, after)
        self.assertEqual(before["industry"], "banks_lag1")


if __name__ == "__main__":
    unittest.main()
