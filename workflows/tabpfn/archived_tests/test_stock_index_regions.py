"""Index arithmetic, regional inference and calendar leakage checks."""

import unittest

import numpy as np
import pandas as pd

from analysis.experiments.stock_index_regions import (
    INDEX_NAMES, REGIONS, TARGETS, construct_indices, correlation_ci,
    forward_experiment, heterogeneous_slopes, regional_rent_panel,
)


class StockIndexRegionTests(unittest.TestCase):
    def stocks(self):
        names = [f"X{i}.SI" for i in range(10)]
        weights = pd.DataFrame({"symbol": names, "fair_value_sgd": np.arange(1, 11),
                                "source_name": names, "nav_weight_pct": np.arange(1, 11)})
        rows = []
        for i, symbol in enumerate(names):
            prices = [100., 200. if i == 0 else 100., 150. if i == 1 else (200. if i == 0 else 100.)]
            rows.extend({"SYMBOL": symbol, "DATE": date, "ADJUSTED_CLOSE": value}
                        for date, value in zip(["2021-01-29", "2021-02-26", "2021-03-31"], prices))
        return pd.DataFrame(rows), weights

    def test_value_weighted_index_drifts_and_reconciles_prior_weights(self):
        prices, weights = self.stocks()
        levels, _, audit, _ = construct_indices(prices, weights, "2021-03")
        w0 = 1 / 55
        self.assertAlmostEqual(levels.loc["2021-02", "cap_seeded_22"], 100 * (1 + w0))
        pivot = audit.pivot(index="month", columns="SYMBOL", values="end_month_weight")
        self.assertAlmostEqual(pivot.loc["2021-02", "X0.SI"], 2 / 56)
        # X1 rises 50% in March: its February-ending weight is the return weight.
        march_return = levels.loc["2021-03", "cap_seeded_22"] / levels.loc["2021-02", "cap_seeded_22"] - 1
        self.assertAlmostEqual(march_return, pivot.loc["2021-02", "X1.SI"] * .5)

    def test_index_is_invariant_to_individual_price_units(self):
        prices, weights = self.stocks()
        before = construct_indices(prices, weights, "2021-03")[0]
        prices.loc[prices.SYMBOL == "X3.SI", "ADJUSTED_CLOSE"] *= 10000
        after = construct_indices(prices, weights, "2021-03")[0]
        pd.testing.assert_frame_equal(before, after)

    def test_late_constituents_and_future_prices_do_not_change_index(self):
        prices, weights = self.stocks()
        before, _, _, diagnostics = construct_indices(prices, weights, "2021-03")
        added = pd.DataFrame({"SYMBOL": ["LATE.SI", "LATE.SI", "X0.SI"],
                              "DATE": ["2021-02-26", "2021-03-31", "2021-04-30"],
                              "ADJUSTED_CLOSE": [100., 100000., 9999999.]})
        after, _, _, diagnostics = construct_indices(pd.concat([prices, added]), weights, "2021-03")
        pd.testing.assert_frame_equal(before, after)
        self.assertEqual(diagnostics["excluded_late_listings"], ["LATE.SI"])

    def test_missing_price_is_not_silently_reweighted(self):
        prices, weights = self.stocks()
        prices = prices[~((prices.SYMBOL == "X0.SI") & (prices.DATE == "2021-02-26"))]
        with self.assertRaisesRegex(ValueError, "Missing constituent prices"):
            construct_indices(prices, weights, "2021-03")

    def test_equal_regional_slopes_are_not_false_independent_observations(self):
        rng = np.random.default_rng(2)
        control, stock = rng.normal(size=40), rng.normal(size=40)
        dates = pd.period_range("2022-01", periods=40, freq="M")
        d = pd.DataFrame({"control": control, "index_return": stock,
                          "response": .7 * control + .2 * stock + rng.normal(size=40), "phase": "development"}, index=dates)
        test, pairs = heterogeneous_slopes({r: d.copy() for r in REGIONS}, repeats=200)
        self.assertEqual(test["equality_p_value"], 1.)
        self.assertTrue(all(p["effect_difference_per_10pct_stock"] == 0 for p in pairs))

    def test_regional_equality_detects_different_slopes(self):
        rng = np.random.default_rng(3)
        dates = pd.period_range("2022-01", periods=40, freq="M")
        control, stock = rng.normal(size=40), rng.normal(size=40)
        designs = {r: pd.DataFrame({"control": control, "index_return": stock,
                                    "response": 2 + control + i * stock, "phase": "development"}, index=dates)
                   for i, r in enumerate(REGIONS)}
        test, _ = heterogeneous_slopes(designs, repeats=200)
        self.assertLess(test["equality_p_value"], .01)

    def test_correlation_interval_resamples_paired_months(self):
        x = np.arange(30, dtype=float)
        low, high = correlation_ci(x, 2 * x + 5, repeats=100)
        self.assertAlmostEqual(low, 1.)
        self.assertAlmostEqual(high, 1.)

    def test_adjusted_means_preserve_region_reference_levels_without_future_labels(self):
        dates = pd.period_range("2021-01", "2022-12", freq="M")
        rows, blocks = [], []
        for i, region in enumerate(REGIONS):
            blocks.append({"TOWN": f"town{i}", "BLOCK": "1", "STREET": "road", "REGION": region,
                           "LATITUDE": 1.35, "LONGITUDE": 103.8 + i * .01})
            for j, month in enumerate(dates):
                rows.extend({"TOWN": f"town{i}", "BLOCK": "1", "STREET": "road", "RENT_APPROVAL_DATE": str(month),
                             "FLAT_TYPE": "3 room", "FLAT_MODEL": "model a", "FLOOR_AREA_SQM": 70.,
                             "LEASE_COMMENCE_DATE": 2000, "MONTHLY_RENT": 2000 + 100 * i + 10 * j} for _ in range(20))
        raw, blocks = pd.DataFrame(rows), pd.DataFrame(blocks)
        before = regional_rent_panel(raw, blocks)
        reference = before[before.month <= "2021-12"].groupby("region")[list(TARGETS)].mean()
        np.testing.assert_allclose(reference.mean_rent, reference.adjusted_mean_rent)
        self.assertGreater(reference.loc["west region", "adjusted_mean_rent"], reference.loc["central region", "adjusted_mean_rent"])
        raw.loc[raw.RENT_APPROVAL_DATE >= "2022-01", "MONTHLY_RENT"] *= 2
        after = regional_rent_panel(raw, blocks)
        pd.testing.assert_frame_equal(before[before.month <= "2021-12"].reset_index(drop=True),
                                      after[after.month <= "2021-12"].reset_index(drop=True))

    def test_future_stock_and_rent_changes_do_not_change_earlier_forecasts(self):
        dates = pd.period_range("2021-01", "2025-03", freq="M")
        t = np.arange(len(dates))
        levels = pd.DataFrame({name: 100 * np.exp(.003 * t + .01 * np.sin(t + i)) for i, name in enumerate(INDEX_NAMES)}, index=dates)
        parts = []
        for i, region in enumerate(["Singapore", *REGIONS]):
            p = pd.DataFrame({"region": region, "month": dates.astype(str)})
            for target in TARGETS:
                p[target] = 2200 + 20 * t + 20 * np.sin(t + i)
                p[f"{target}_growth_1m"] = 100 * np.log(p[target] / p[target].shift(1))
            parts.append(p)
        panel = pd.concat(parts, ignore_index=True)
        before = forward_experiment(panel, levels)[0]
        changed = panel.copy()
        for target in TARGETS:
            changed.loc[changed.month >= "2024-06", target] *= 1.2
            changed[f"{target}_growth_1m"] = changed.groupby("region", sort=False)[target].transform(lambda x: 100 * np.log(x / x.shift(1)))
        levels.loc["2024-06":] *= 2
        after = forward_experiment(changed, levels)[0]
        columns = ["region", "target", "month", "model", "predicted_growth", "predicted_mean_rent"]
        pd.testing.assert_frame_equal(before.loc[before.month <= "2024-06", columns].reset_index(drop=True),
                                      after.loc[after.month <= "2024-06", columns].reset_index(drop=True))
        self.assertTrue((before.last_training_target < before.month).all())


if __name__ == "__main__":
    unittest.main()
