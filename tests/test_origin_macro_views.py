import unittest
import numpy as np
import pandas as pd

from analysis.experiments.origin_macro_views import VARIANTS, macro_view


class OriginMacroTests(unittest.TestCase):
    def fixture(self):
        dates = pd.period_range("2021-01", periods=24, freq="M")
        market = pd.DataFrame(index=dates)
        for c in ("A", "C"):
            market[f"COE_{c}_PREMIUM"] = np.arange(24) * 1000 + 40000
            market[f"COE_{c}_BID_PRESSURE"] = 1.5
        for c in ("SG_STOCK_RETURN", "SG_STOCK_RETURN_3M", "SG_STOCK_VOL_3M"):
            market[c] = np.arange(24) * .001
        market["SG_STOCK_COVERAGE"] = 20
        x = pd.DataFrame({"TEMP_FORECAST_HORIZON": [1., 6., 10.], "OTHER": [2., 3., 4.]}, index=[5, 9, 15])
        months = pd.PeriodIndex(["2022-01", "2022-06", "2022-10"], freq="M")
        origins = pd.PeriodIndex(["2021-12"] * 3, freq="M")
        return x, months, origins, market

    def test_origin_freeze_ignores_future_and_buffer_month_observations(self):
        x, months, origins, market = self.fixture()
        poisoned = market.copy(); poisoned.loc[poisoned.index >= "2021-12"] = 1e12
        for variant in VARIANTS:
            a = macro_view(x, months, origins, market, variant)
            b = macro_view(x, months, origins, poisoned, variant)
            pd.testing.assert_frame_equal(a, b)
            pd.testing.assert_frame_equal(a[x.columns], x)
            for col in a.filter(like="MACRO_"):
                if not col.endswith("_X_H"):
                    self.assertEqual(a[col].nunique(), 1)

    def test_unknown_origin_stays_missing_and_never_uses_latest_market(self):
        x, months, origins, market = self.fixture()
        origins = pd.PeriodIndex([pd.NaT, "2021-12", "2021-12"], freq="M")
        for variant in VARIANTS:
            a = macro_view(x, months, origins, market, variant)
            new = a.filter(like="MACRO_")
            flags = [c for c in new if c.endswith("MISSING")]
            self.assertTrue(new.drop(columns=flags).iloc[0].isna().all())
            self.assertEqual(new[flags].iloc[0, 0], 1.)

    def test_prefix_replay_and_horizon_mismatch(self):
        x, months, origins, market = self.fixture()
        for variant in VARIANTS:
            a = macro_view(x, months, origins, market, variant)
            b = macro_view(x.iloc[:1], months[:1], origins[:1], market.loc[:"2021-11"], variant)
            pd.testing.assert_frame_equal(a.iloc[:1], b)
        x.loc[5, "TEMP_FORECAST_HORIZON"] = 3.
        with self.assertRaises(AssertionError): macro_view(x, months, origins, market, "coe_origin")


if __name__ == "__main__": unittest.main()
