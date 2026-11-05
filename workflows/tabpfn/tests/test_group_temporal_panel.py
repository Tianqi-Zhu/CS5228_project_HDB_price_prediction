"""Check geographic join and monthly group lag alignment."""

import unittest

import pandas as pd

from analysis.eda.group_temporal_panel import add_planning_area, monthly_panel


class GroupTemporalPanelTests(unittest.TestCase):
    def test_monthly_group_rent_does_not_skip_empty_month(self):
        train = pd.DataFrame({
            "RENT_APPROVAL_DATE": ["2021-01", "2021-01", "2021-03"],
            "TOWN": ["A", "A", "A"], "BLOCK": ["1", "1", "1"],
            "STREET": ["Road", "Road", "Road"],
            "FLAT_TYPE": ["3-room", "3-room", "3-room"],
            "FLAT_MODEL": ["M", "M", "M"], "FLOOR_AREA_SQM": [70, 70, 70],
            "LEASE_COMMENCE_DATE": [2000, 2000, 2000], "MONTHLY_RENT": [1000, 2000, 3000],
        })
        blocks = pd.DataFrame({"TOWN": ["a"], "BLOCK": ["1"], "STREET": ["road"],
                               "PLANNING_AREA": ["central"]})
        frame = add_planning_area(train, blocks)
        self.assertEqual(frame["PLANNING_AREA"].tolist(), ["central"] * 3)
        market = pd.DataFrame({"COE_A_GROWTH": [.1, .2, .3]},
                              index=pd.period_range("2021-01", periods=3, freq="M"))
        panel = monthly_panel(frame, market)
        area = panel[panel["grain"] == "planning_area"].set_index("RENT_APPROVAL_DATE")
        self.assertEqual(area.loc["2021-01", "mean_rent"], 1500)
        self.assertEqual(area.loc["2021-01", "median_rent"], 1500)
        self.assertEqual(area.loc["2021-02", "n"], 0)
        self.assertTrue(pd.isna(area.loc["2021-03", "mean_lag1"]))
        self.assertEqual(area.loc["2021-02", "COE_A_GROWTH_LAG1"], .1)
        self.assertEqual(area.loc["2021-03", "COE_A_GROWTH_LAG1"], .2)


if __name__ == "__main__":
    unittest.main()
