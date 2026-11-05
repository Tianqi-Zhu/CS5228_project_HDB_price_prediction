"""Checks for the E plus engineered-economic-feature comparison."""

import unittest

import pandas as pd

from analysis.eda.economic_signals import FOLDS, attach_prior_month, monthly_market
from analysis.experiments.auxiliary_ablation import FEATURE_STAGES
from analysis.experiments.economic_feature_ablation import (
    COE_ENGINEERED, E_FEATURES, SG_STOCK_ENGINEERED, VARIANTS, score_deltas,
)


class EconomicFeatureAblationTests(unittest.TestCase):
    def test_E_and_engineered_groups_are_separate(self):
        self.assertEqual(E_FEATURES, FEATURE_STAGES["E_plus_amenities"])
        self.assertEqual(VARIANTS["E_plus_coe"], COE_ENGINEERED)
        self.assertEqual(VARIANTS["E_plus_sg_stocks"], SG_STOCK_ENGINEERED)
        self.assertEqual(
            VARIANTS["E_plus_both"], COE_ENGINEERED + SG_STOCK_ENGINEERED
        )
        self.assertIn("COE_C_BID_PRESSURE_LAG1", COE_ENGINEERED)
        self.assertIn("SG_STOCK_VOL_3M_LAG1", SG_STOCK_ENGINEERED)
        self.assertFalse(
            any(name.startswith("STOCK_MEDIAN") for name in SG_STOCK_ENGINEERED)
        )

    def test_prior_month_market_values_only(self):
        coe = pd.DataFrame({
            "YEAR": [2021, 2021], "MONTH": ["jan", "feb"],
            "CATEGORY": ["c", "c"], "QUOTA_PREMIUM": ["$40,000", "$44,000"],
            "QUOTA": ["100", "100"], "BIDS_RECEIVED": ["150", "200"],
        })
        stocks = pd.DataFrame({
            "SYMBOL": ["X.SI", "X.SI", "Y", "Y"],
            "DATE": ["2021-01-29", "2021-02-26", "2021-01-29", "2021-02-26"],
            "ADJUSTED_CLOSE": [100, 110, 100, 50],
        })
        market = monthly_market(coe, stocks)
        rent = pd.DataFrame({"RENT_APPROVAL_DATE": ["2021-02", "2021-03"]})
        attached = attach_prior_month(rent, market)
        self.assertEqual(attached.loc[0, "COE_C_PREMIUM_LAG1"], 40000)
        self.assertEqual(attached.loc[1, "COE_C_PREMIUM_LAG1"], 44000)
        self.assertEqual(attached.loc[0, "COE_C_BID_PRESSURE_LAG1"], 1.5)
        self.assertEqual(attached.loc[1, "COE_C_BID_PRESSURE_LAG1"], 2.0)
        self.assertAlmostEqual(attached.loc[1, "SG_STOCK_RETURN_LAG1"], 0.1)

    def test_delta_uses_same_fold_E(self):
        rows = []
        for i, (start, _) in enumerate(FOLDS):
            rows.extend([
                {"validation_start": start, "variant": "E_amenities",
                 "rmse_sgd": 600.0 - 50 * i},
                {"validation_start": start, "variant": "E_plus_both",
                 "rmse_sgd": 598.0 - 45 * i},
            ])
        scored = score_deltas(pd.DataFrame(rows))
        self.assertEqual(scored.loc[1, "delta_rmse_vs_E_sgd"], -2.0)
        self.assertEqual(scored.loc[3, "delta_rmse_vs_E_sgd"], 3.0)


if __name__ == "__main__":
    unittest.main()
