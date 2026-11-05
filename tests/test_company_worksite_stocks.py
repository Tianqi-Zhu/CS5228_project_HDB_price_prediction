"""Guard calendar availability, spatial support and dependent test correction."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from analysis.experiments.company_worksite_stocks import (
    dependent_fdr, forward_predictions, hac_association, local_panel,
    property_residuals, regression_design, rental_coordinates, stock_returns,
)


class CompanyWorksiteStockTests(unittest.TestCase):
    def design_fixture(self):
        months = pd.period_range("2021-01", "2025-03", freq="M")
        rng = np.random.default_rng(42)
        panel = pd.DataFrame({"month": months.astype(str), "premium_change": rng.normal(size=len(months)),
                              "national_change": rng.normal(size=len(months))})
        returns = pd.DataFrame({"A.SI": rng.normal(size=len(months)),
                                "B.SI": rng.normal(size=len(months)),
                                "C": rng.normal(size=len(months))}, index=months)
        return panel, returns

    def test_company_and_control_baskets_use_only_prior_months(self):
        panel, returns = self.design_fixture()
        d = regression_design(panel, returns, "A.SI", ["A.SI", "B.SI", "C"], 3)
        self.assertEqual(d.loc["2024-06", "company_return"], returns.loc["2024-03", "A.SI"])
        self.assertEqual(d.loc["2024-06", "broad_return_lag1"], returns.loc["2024-05", "B.SI"])
        self.assertEqual(d.loc["2024-06", "peer_return_lag1"], returns.loc["2024-05", ["B.SI", "C"]].mean())
        changed = returns.copy()
        changed.loc["2024-06":] += 10000
        after = regression_design(panel, changed, "A.SI", ["A.SI", "B.SI", "C"], 3)
        pd.testing.assert_frame_equal(d.loc[:"2024-06"], after.loc[:"2024-06"])

    def test_future_rents_and_stocks_do_not_change_earlier_predictions(self):
        panel, returns = self.design_fixture()
        d = regression_design(panel, returns, "A.SI", ["A.SI", "B.SI", "C"], 1)
        before = forward_predictions(d)
        panel.loc[panel.month >= "2024-06", ["premium_change", "national_change"]] += 10000
        returns.loc["2024-06":] += 10000
        after = forward_predictions(regression_design(panel, returns, "A.SI", ["A.SI", "B.SI", "C"], 1))
        columns = ["month", "persistence", "rent_market_control", "company_stock", "last_training_month"]
        pd.testing.assert_frame_equal(before.loc[before.month <= "2024-06", columns].reset_index(drop=True),
                                      after.loc[after.month <= "2024-06", columns].reset_index(drop=True))
        self.assertTrue((before.last_training_month < before.month).all())

    def test_stock_prices_do_not_bridge_missing_month_or_use_future(self):
        rows = pd.DataFrame({"SYMBOL": ["A.SI"] * 4,
                             "DATE": ["2021-01-20", "2021-03-20", "2021-04-20", "2021-05-20"],
                             "ADJUSTED_CLOSE": [100, 110, 121, 999]})
        result = stock_returns(rows, ["A.SI"], "2021-04")
        self.assertEqual(str(result.index.max()), "2021-04")
        self.assertTrue(pd.isna(result.loc["2021-03", "A.SI"]))
        self.assertAlmostEqual(result.loc["2021-04", "A.SI"], 100 * np.log(1.1))

    def test_dependent_fdr_handles_order_and_missing_values(self):
        p = pd.Series([.2, .01, np.nan, .04], index=["c", "a", "missing", "b"])
        q = dependent_fdr(p)
        self.assertAlmostEqual(q["a"], .055)
        self.assertAlmostEqual(q["b"], .11)
        self.assertAlmostEqual(q["c"], .2 * (1 + .5 + 1 / 3))
        self.assertTrue(pd.isna(q["missing"]))

    def test_hac_partial_slope_matches_least_squares_and_exact_stock_move(self):
        rng = np.random.default_rng(2)
        x, stock = rng.normal(size=30), rng.normal(size=30)
        y = 1 + .7 * x + .4 * stock + rng.normal(size=30) * .2
        d = pd.DataFrame({"control": x, "company_return": stock, "response": y, "phase": "development"})
        result = hac_association(d)
        expected = np.linalg.lstsq(np.column_stack([np.ones(30), x, stock]), y, rcond=None)[0][-1]
        self.assertAlmostEqual(result["coefficient"], expected)
        self.assertAlmostEqual(result["effect_per_10pct_stock"], expected * 100 * np.log1p(.1))
        self.assertLess(result["effect_ci_low"], result["effect_ci_high"])

    def test_fixed_property_adjustment_never_uses_post2021_labels(self):
        months = pd.period_range("2021-01", "2025-03", freq="M")
        raw = pd.DataFrame([{"RENT_APPROVAL_DATE": str(m), "TOWN": "a", "BLOCK": "1", "STREET": "a road",
                             "FLAT_TYPE": "3 room", "FLAT_MODEL": "model a", "FLOOR_AREA_SQM": 70.,
                             "LEASE_COMMENCE_DATE": 2000, "MONTHLY_RENT": 2000 + i * 10} for i, m in enumerate(months)])
        blocks = pd.DataFrame({"TOWN": ["a"], "BLOCK": ["1"], "STREET": ["a road"],
                               "LATITUDE": [1.35], "LONGITUDE": [103.8]})
        frame = rental_coordinates(raw, blocks)
        before = property_residuals(frame)
        frame.loc[frame.RENT_APPROVAL_DATE >= "2022-01", "MONTHLY_RENT"] *= 2
        after = property_residuals(frame)
        np.testing.assert_allclose(before[:12], after[:12])
        np.testing.assert_allclose(after[12:] - before[12:], np.log(2))

    def test_sparse_month_is_missing_and_calendar_difference_does_not_bridge(self):
        # One near and one ring building, 20 rentals per month, except February.
        rows = []
        for month in ("2022-01", "2022-02", "2022-03", "2022-04"):
            for region, longitude in (("near", 103.8), ("ring", 103.84)):
                count = 19 if month == "2022-02" and region == "near" else 20
                rows.extend({"_month": pd.Period(month, "M"), "LONGITUDE": longitude, "LATITUDE": 1.35,
                             "address_key": region} for _ in range(count))
        frame = pd.DataFrame(rows)
        sites = pd.DataFrame({"site_id": ["a"], "company": ["A"], "symbol": ["A.SI"],
                              "longitude": [103.8], "latitude": [1.35]})
        with patch("analysis.experiments.company_worksite_stocks.property_residuals", return_value=np.zeros(len(frame))):
            panel, _, _ = local_panel(frame, sites)
        d = panel[panel.radius_m == 2000].set_index("month")
        self.assertTrue(pd.isna(d.loc["2022-02", "premium_logpoints"]))
        self.assertTrue(pd.isna(d.loc["2022-03", "premium_change"]))
        self.assertEqual(d.loc["2022-04", "premium_change"], 0.)

    def test_ambiguous_addresses_fail_before_aggregating_targets(self):
        raw = pd.DataFrame({"RENT_APPROVAL_DATE": ["2021-01"], "TOWN": ["a"], "BLOCK": ["1"], "STREET": ["a road"],
                            "FLAT_TYPE": ["3 room"], "FLAT_MODEL": ["model a"], "FLOOR_AREA_SQM": [70.],
                            "LEASE_COMMENCE_DATE": [2000], "MONTHLY_RENT": [2000]})
        blocks = pd.DataFrame({"TOWN": ["a", "a"], "BLOCK": ["1", "1"], "STREET": ["a road", "a road"],
                               "LATITUDE": [1.35, 1.36], "LONGITUDE": [103.8, 103.8]})
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            rental_coordinates(raw, blocks)


if __name__ == "__main__":
    unittest.main()
