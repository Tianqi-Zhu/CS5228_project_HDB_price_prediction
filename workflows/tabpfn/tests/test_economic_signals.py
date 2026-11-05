"""Aggregation and calendar-availability checks for economic EDA."""

import unittest

import pandas as pd

from analysis.eda.economic_signals import attach_prior_month, monthly_market


class EconomicSignalTests(unittest.TestCase):
    def test_quota_weighted_bid_pressure_and_monthly_lag(self):
        coe = pd.DataFrame({
            "YEAR": [2021, 2021, 2021], "MONTH": ["jan", "jan", "mar"],
            "CATEGORY": ["a", "a", "a"], "QUOTA_PREMIUM": ["$40,000", "$50,000", "$60,000"],
            "QUOTA": ["100", "200", "100"], "BIDS_RECEIVED": ["120", "240", "150"],
        })
        stocks = pd.DataFrame({
            "SYMBOL": ["X.SI", "X.SI", "X.SI"],
            "DATE": ["2021-01-28", "2021-02-26", "2021-03-30"],
            "ADJUSTED_CLOSE": [100, 110, 121],
        })
        market = monthly_market(coe, stocks)
        self.assertEqual(market.loc["2021-01", "COE_A_PREMIUM"], 45000)
        self.assertAlmostEqual(market.loc["2021-01", "COE_A_BID_PRESSURE"], 1.2)
        self.assertTrue(pd.isna(market.loc["2021-02", "COE_A_PREMIUM"]))
        rentals = pd.DataFrame({"RENT_APPROVAL_DATE": ["2021-02", "2021-03", "2021-04"]})
        result = attach_prior_month(rentals, market)
        self.assertEqual(result.loc[0, "COE_A_PREMIUM_LAG1"], 45000)
        self.assertTrue(pd.isna(result.loc[1, "COE_A_PREMIUM_LAG1"]))
        self.assertEqual(result.loc[2, "COE_A_PREMIUM_LAG1"], 60000)
        self.assertAlmostEqual(result.loc[2, "SG_STOCK_RETURN_LAG1"], .1)


if __name__ == "__main__":
    unittest.main()
