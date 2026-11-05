"""Category pooling, sparse calendars, global correction and joint comparisons."""

import unittest

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from analysis.experiments.financial_sector_rent_categories import (
    analysis_rent, category_panels, correct_category_tests, slope_equality,
)
from analysis.experiments.company_worksite_stocks import dependent_fdr


class FinancialSectorRentCategoryTests(unittest.TestCase):
    def fixture(self):
        config = {"dimensions": {"room_type": "FLAT_TYPE_CLEAN", "flat_model": "FLAT_MODEL_CLEAN"},
                  "reference_end": "2021-12", "minimum_reference_rentals_per_month": 2,
                  "minimum_analysis_rentals_per_month": 2}
        rows = []
        for i, month in enumerate(pd.period_range("2021-01", "2024-12", freq="M")):
            for room, base, count in [("3-room", 2000, 2), ("4-room", 3000, 2), ("1-room", 1000, 1)]:
                for j in range(count):
                    rows.append({"RENT_APPROVAL_DATE": str(month), "TOWN": "town", "BLOCK": room,
                                 "STREET": "street", "FLAT_TYPE": room.replace("-", " ") if j else room,
                                 "FLAT_MODEL": "standard", "FLOOR_AREA_SQM": 70 + 20 * j,
                                 "LEASE_COMMENCE_DATE": 1980, "MONTHLY_RENT": base + i * 10 + j * 100})
        return pd.DataFrame(rows), config

    def test_spelling_variants_pool_and_means_reconcile_to_source(self):
        raw, config = self.fixture()
        with threadpool_limits(limits=1):
            panel, audit = category_panels(raw, config)
        rooms = panel[panel.dimension == "room_type"]
        self.assertEqual(set(rooms.category), {"1-room", "3-room", "4-room"})
        p = rooms[(rooms.category == "3-room") & (rooms.month == "2021-01")].iloc[0]
        self.assertEqual(p.n, 2)
        self.assertEqual(p.mean_rent, 2050)
        self.assertFalse(audit[(audit.dimension == "room_type") & (audit.category == "1-room")].eligible.iloc[0])
        weighted = (rooms.n * rooms.mean_rent).groupby(rooms.month).sum() / rooms.groupby("month").n.sum()
        actual = raw.groupby("RENT_APPROVAL_DATE").MONTHLY_RENT.mean()
        np.testing.assert_allclose(weighted, actual)

    def test_later_outcomes_do_not_change_earlier_means_adjustments_or_eligibility(self):
        raw, config = self.fixture()
        with threadpool_limits(limits=1):
            before, before_audit = category_panels(raw, config)
            raw.loc[raw.RENT_APPROVAL_DATE >= "2024-06", "MONTHLY_RENT"] *= 5
            after, after_audit = category_panels(raw, config)
        pd.testing.assert_frame_equal(before[before.month < "2024-06"], after[after.month < "2024-06"])
        pd.testing.assert_frame_equal(before_audit, after_audit)

    def test_thin_or_missing_month_does_not_bridge_growth(self):
        raw, config = self.fixture()
        missing = (raw.RENT_APPROVAL_DATE == "2023-02") & raw.FLAT_TYPE.str.startswith("3")
        raw = raw[~missing]
        thin = raw[(raw.RENT_APPROVAL_DATE == "2023-03") & raw.FLAT_TYPE.str.startswith("3")].index[0]
        raw = raw.drop(index=thin)
        with threadpool_limits(limits=1):
            panel, _ = category_panels(raw, config)
        p = analysis_rent(panel[(panel.dimension == "room_type") & (panel.category == "3-room")])
        self.assertTrue(p.eligible.all())  # Eligibility was fixed from 2021.
        self.assertTrue(pd.isna(p.loc["2023-02", "mean_rent"]))
        self.assertTrue(pd.isna(p.loc["2023-03", "mean_rent"]))
        growth = np.log(p.mean_rent / p.mean_rent.shift(1))
        self.assertTrue(pd.isna(growth.loc["2023-04"]))

    def test_multiple_testing_correction_spans_category_dimensions(self):
        rows = [{"dimension": dimension, "category": category, "period": "development",
                 "controls": "full_controls", "price_field": "ADJUSTED_CLOSE", "target": "mean_rent",
                 "signal": "banks_equal", "lag": 1, "p_value": p, "q_old_local": 0.0}
                for dimension, category, p in [("room_type", "3-room", .01), ("room_type", "4-room", .04),
                                                ("flat_model", "standard", .03)]]
        source = pd.DataFrame(rows)
        out = correct_category_tests(source)
        np.testing.assert_allclose(out.q_global_all_lags, dependent_fdr(source.p_value))
        self.assertNotIn("q_old_local", out)
        self.assertTrue(pd.isna(out.loc[2, "q_primary_room_bank_lag1"]))
        np.testing.assert_allclose(out.loc[:1, "q_primary_room_bank_lag1"], dependent_fdr(source.loc[:1, "p_value"]))

    def test_synchronized_equality_recovers_identical_category_slopes(self):
        rng = np.random.default_rng(123)
        x = rng.normal(size=29)
        y = .5 * x + rng.normal(size=29)
        d = pd.DataFrame({"response": y, "financial_return": x}, index=pd.period_range("2022-01", periods=29, freq="M"))
        other = d.copy()
        other.response += 10  # Category-level intercept difference, same slope.
        test, contrasts = slope_equality({"3-room": d, "4-room": other}, repeats=100)
        self.assertGreater(test["equality_p_value"], .9)
        self.assertAlmostEqual(contrasts[0]["difference_per_10pct_stock"], 0, places=10)

    def test_synchronized_equality_detects_a_known_slope_difference(self):
        rng = np.random.default_rng(456)
        x = rng.normal(size=29)
        d = pd.DataFrame({"response": rng.normal(size=29), "financial_return": x},
                         index=pd.period_range("2022-01", periods=29, freq="M"))
        other = d.copy()
        other.response += 2 * x
        test, contrasts = slope_equality({"3-room": d, "4-room": other}, repeats=100)
        self.assertLess(test["equality_p_value"], .05)
        expected = -2 * 100 * np.log1p(.1)
        self.assertAlmostEqual(contrasts[0]["difference_per_10pct_stock"], expected)


if __name__ == "__main__":
    unittest.main()
