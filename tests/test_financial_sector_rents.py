"""Dedicated finance basket arithmetic, calendar and future-information checks."""

import unittest

import numpy as np
import pandas as pd

from analysis.experiments.financial_sector_rents import (
    build_financial_signals, block_correlation_interval, forecast_experiment,
    regression_inputs,
)


class FinancialSectorRentTests(unittest.TestCase):
    def fixture(self):
        config = {"banks": ["D05.SI", "O39.SI", "U11.SI"],
                  "financial_services": ["D05.SI", "O39.SI", "U11.SI", "S68.SI", "AIY.SI"],
                  "nonbank_finance": ["S68.SI", "AIY.SI"], "price_fields": ["ADJUSTED_CLOSE", "CLOSE"]}
        weights = pd.DataFrame({"symbol": config["banks"], "fair_value_sgd": [1, 2, 3]})
        rows = []
        dates = ["2021-01-29", "2021-02-26", "2021-03-31"]
        for symbol in config["financial_services"] + ["OTHER.SI"]:
            for i, date in enumerate(dates):
                price = 100 * (1.1 ** i) if symbol != "OTHER.SI" else 100 * (1.02 ** i)
                rows.append({"SYMBOL": symbol, "DATE": date, "ADJUSTED_CLOSE": price, "CLOSE": price})
        return pd.DataFrame(rows), weights, config

    def test_nonfinancial_comparator_excludes_every_financial_name(self):
        stocks, weights, config = self.fixture()
        levels, _, diag = build_financial_signals(stocks, weights, config, "2021-03")
        self.assertEqual(diag["ADJUSTED_CLOSE"]["nonfinancial_members"], ["OTHER.SI"])
        self.assertAlmostEqual(levels["ADJUSTED_CLOSE"].loc["2021-02", "banks_equal"], 110.)
        self.assertAlmostEqual(levels["ADJUSTED_CLOSE"].loc["2021-02", "nonfinancial_market"], 102.)

    def test_financial_indices_are_invariant_to_price_units_and_weights_sum_to_one(self):
        stocks, weights, config = self.fixture()
        before, audit, _ = build_financial_signals(stocks, weights, config, "2021-03")
        stocks.loc[stocks.SYMBOL == "D05.SI", ["ADJUSTED_CLOSE", "CLOSE"]] *= 1000
        after, _, _ = build_financial_signals(stocks, weights, config, "2021-03")
        for field in config["price_fields"]:
            pd.testing.assert_frame_equal(before[field], after[field])
        np.testing.assert_allclose(audit.groupby(["price_field", "month"]).end_month_weight.sum(), 1)

    def test_future_stock_observations_do_not_change_historical_indices(self):
        stocks, weights, config = self.fixture()
        before, _, _ = build_financial_signals(stocks, weights, config, "2021-03")
        extra = stocks.iloc[[0]].copy()
        extra["DATE"] = "2026-01-31"
        extra[["ADJUSTED_CLOSE", "CLOSE"]] = 999999
        after, _, _ = build_financial_signals(pd.concat([stocks, extra]), weights, config, "2021-03")
        pd.testing.assert_frame_equal(before["ADJUSTED_CLOSE"], after["ADJUSTED_CLOSE"])

    def test_missing_constituent_month_is_not_silently_bridged(self):
        stocks, weights, config = self.fixture()
        stocks = stocks[~((stocks.SYMBOL == "D05.SI") & (stocks.DATE == "2021-02-26"))]
        with self.assertRaisesRegex(ValueError, "Missing/nonpositive"):
            build_financial_signals(stocks, weights, config, "2021-03")

    def forecast_fixture(self):
        dates = pd.period_range("2021-01", "2025-03", freq="M")
        t = np.arange(len(dates))
        levels = pd.DataFrame({name: 100 * np.exp(.002 * t + .01 * np.sin(t + j))
                              for j, name in enumerate(["banks_equal", "banks_cap_seeded", "finance_equal", "nonfinancial_market"])}, index=dates)
        controls = pd.DataFrame({"SP500": np.sin(t), "DEXSIUS": np.cos(t)}, index=dates)
        rent = pd.DataFrame({"mean_rent": 2000 + 20 * t + 10 * np.sin(t),
                             "adjusted_mean_rent": 2000 + 20 * t + 5 * np.sin(t)}, index=dates)
        return rent, levels, controls

    def test_regression_finance_and_macro_signals_share_exact_calendar_lag(self):
        rent, levels, controls = self.forecast_fixture()
        d = regression_inputs(rent, levels, controls, "banks_equal", 6, "mean_rent")
        expected = 100 * np.log(levels.loc["2023-12", "banks_equal"] / levels.loc["2023-11", "banks_equal"])
        self.assertAlmostEqual(d.loc["2024-06", "financial_return"], expected)
        self.assertEqual(d.loc["2024-06", "US_market"], controls.loc["2023-12", "SP500"])

    def test_future_labels_and_prices_do_not_change_preceding_forecasts(self):
        rent, levels, controls = self.forecast_fixture()
        before = forecast_experiment(rent, levels, controls)[0]
        rent.loc["2024-06":] *= 1.5
        levels.loc["2024-06":] *= 2
        controls.loc["2024-06":] += 100
        after = forecast_experiment(rent, levels, controls)[0]
        columns = ["target", "month", "model", "predicted_growth", "predicted_mean_rent"]
        pd.testing.assert_frame_equal(before.loc[before.month <= "2024-06", columns].reset_index(drop=True),
                                      after.loc[after.month <= "2024-06", columns].reset_index(drop=True))
        self.assertTrue((before.last_training_target < before.month).all())

    def test_annual_growth_interval_requires_two_annual_blocks(self):
        x = np.arange(20, dtype=float)
        self.assertTrue(np.isnan(block_correlation_interval(x, x, window=12)[0]))
        low, high = block_correlation_interval(np.arange(40.), np.arange(40.), window=12, repeats=100)
        self.assertAlmostEqual(low, 1.)
        self.assertAlmostEqual(high, 1.)


if __name__ == "__main__":
    unittest.main()
