"""Small invariants for auxiliary feature generation."""

import unittest

import pandas as pd

from analysis.data import add_economic_features, add_spatial_features


class AuxiliaryFeatureTests(unittest.TestCase):
    def test_coe_lags_follow_calendar_months_even_with_missing_target_months(self):
        rentals = pd.DataFrame({"RENT_APPROVAL_DATE": ["2021-02", "2021-04"]})
        coe = pd.DataFrame({
            "YEAR": [2021, 2021, 2021],
            "MONTH": ["jan", "feb", "mar"],
            "CATEGORY": ["a", "a", "a"],
            "QUOTA_PREMIUM": ["$10,000", "$20,000", "$30,000"],
        })
        result = add_economic_features(rentals, coe=coe)
        self.assertEqual(result["COE_A_MEDIAN_LAG1"].tolist(), [10000, 30000])
        self.assertEqual(result.loc[1, "COE_A_MEDIAN_LAG3"], 10000)
        self.assertNotIn("COE_A_MEDIAN", result.columns)

    def test_station_codes_at_same_location_count_once(self):
        rentals = pd.DataFrame({"RENT_APPROVAL_DATE": ["2024-01"], "TOWN": ["Foo"],
                                "BLOCK": ["1"], "STREET": ["Main Road"]})
        blocks = pd.DataFrame({"TOWN": ["foo"], "BLOCK": ["1"], "STREET": ["main road"],
                               "POSTAL_CODE": ["NIL"], "LATITUDE": [1.30], "LONGITUDE": [103.80],
                               "MAX_FLOOR": [-1], "YEAR_COMPLETED": [1990],
                               "SUBZONE": ["foo"], "PLANNING_AREA": ["foo"], "REGION": ["central"]})
        mrt = pd.DataFrame({"STATUS": ["open", "open"], "LATITUDE": [1.30, 1.30],
                            "LONGITUDE": [103.80, 103.80]})
        malls = pd.DataFrame({"LATITUDE": [1.30], "LONGITUDE": [103.80]})
        schools = pd.DataFrame({"MAINLEVEL_CODE": ["primary"], "LATITUDE": [1.30],
                                "LONGITUDE": [103.80]})
        result = add_spatial_features(rentals, blocks, mrt, malls, schools)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, "MRT_COUNT_1.0KM"], 1)
        self.assertEqual(result.loc[0, "AMENITY_DENSITY_1KM"], 3)
        self.assertTrue(pd.isna(result.loc[0, "MAX_FLOOR"]))
        self.assertTrue(pd.isna(result.loc[0, "POSTAL_CODE"]))

    def test_stock_feature_uses_month_end_close_returns(self):
        rentals = pd.DataFrame({"RENT_APPROVAL_DATE": ["2021-03"]})
        stocks = pd.DataFrame({
            "SYMBOL": ["AAA", "AAA", "AAA", "AAA"],
            "DATE": ["2021-01-04", "2021-01-29", "2021-02-01", "2021-02-26"],
            "ADJUSTED_CLOSE": [10.0, 20.0, 30.0, 40.0],
        })
        result = add_economic_features(rentals, stocks=stocks)
        self.assertAlmostEqual(result.loc[0, "STOCK_MEDIAN_RETURN_LAG1"], 1.0)


if __name__ == "__main__":
    unittest.main()
