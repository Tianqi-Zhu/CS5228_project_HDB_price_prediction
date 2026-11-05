"""Availability, currency, matching and paired-error checks for the follow-up."""

import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from analysis.config import ROOT
from analysis.experiments.company_worksite_followup import (
    build_panels, expanded_design, match_placebos, paired_month_interval,
    stock_features, weighted_score,
)
from analysis.experiments.worksite_tabpfn_correction import local_ridge


class CompanyWorksiteFollowupTests(unittest.TestCase):
    def feature_fixture(self):
        months = pd.period_range("2021-01", "2025-03", freq="M")
        returns = pd.DataFrame({"KLIC": np.arange(len(months), dtype=float), "B.SI": 1., "C.SI": 2.}, index=months)
        controls = pd.DataFrame({"SP500": 3., "NASDAQSOX": 4., "DEXSIUS": 5.}, index=months)
        sites = pd.DataFrame({"site_id": ["kulicke_soffa_serangoon"], "longitude": [103.8], "latitude": [1.35]})
        hypotheses = [{"site_id": "kulicke_soffa_serangoon", "symbol": "KLIC", "lag": 6, "radius_m": 1000}]
        rows = pd.DataFrame({"RENT_APPROVAL_DATE": ["2024-06", "2024-11", "2024-12", "2025-03"],
                             "LONGITUDE": [103.8] * 4, "LATITUDE": [1.35] * 4, "MONTHLY_RENT": [2000] * 4})
        return rows, sites, returns, controls, hypotheses

    def test_fixed_origin_uses_exact_known_lag_and_never_realized_future_stock(self):
        rows, sites, returns, controls, hypotheses = self.feature_fixture()
        result, _, audit = stock_features(rows, sites, returns, controls, hypotheses, pd.Period("2024-05", "M"))
        self.assertEqual(result.WS_KLIC_RETURN.tolist(), [returns.loc["2023-12", "KLIC"] + 5,
                                                        returns.loc["2024-05", "KLIC"] + 5, 0, 0])
        self.assertEqual(result.WS_KLIC_UNAVAILABLE.tolist(), [0, 0, 1, 1])
        self.assertEqual(audit[0]["latest_used_stock_month"], "2024-05")
        returns.loc["2024-06":] += 10000
        controls.loc["2024-06":] += 10000
        after = stock_features(rows, sites, returns, controls, hypotheses, pd.Period("2024-05", "M"))[0]
        pd.testing.assert_frame_equal(result, after)

    def test_local_company_terms_are_separate_from_local_sector_controls(self):
        rows, sites, returns, controls, hypotheses = self.feature_fixture()
        features, groups, _ = stock_features(rows, sites, returns, controls, hypotheses)
        self.assertEqual(groups["local"], ["WS_KLIC_LOCAL_COMPANY"])
        self.assertEqual(len(groups["sector"]), 3)
        np.testing.assert_allclose(features.WS_KLIC_LOCAL_COMPANY, features.WS_KLIC_RETURN)
        rows.MONTHLY_RENT += 100000
        pd.testing.assert_frame_equal(features, stock_features(rows, sites, returns, controls, hypotheses)[0])

    def test_sector_and_currency_controls_use_the_company_signal_month(self):
        rows, sites, returns, controls, hypotheses = self.feature_fixture()
        local = pd.DataFrame({"month": returns.index.astype(str), "premium_change": .1,
                              "national_change": .2})
        design = expanded_design(local, returns, controls, "KLIC", ["KLIC", "B.SI", "C.SI"], 6, True)
        self.assertEqual(design.loc["2024-06", "company_return"], returns.loc["2023-12", "KLIC"] + 5)
        self.assertEqual(design.loc["2024-06", "NASDAQSOX_at_signal_month"], 4.)
        returns.loc["2024-06":] += 1000
        controls.loc["2024-06":] += 1000
        after = expanded_design(local, returns, controls, "KLIC", ["KLIC", "B.SI", "C.SI"], 6, True)
        pd.testing.assert_frame_equal(design.loc[:"2024-06"], after.loc[:"2024-06"])

    def test_placebo_matching_ignores_future_rent_values(self):
        rows = []
        points = [(1.35, 103.8)] + [(1.25 + .025 * i, 103.61 + .025 * j) for i in range(8) for j in range(12)]
        for i, (lat, lon) in enumerate(points):
            for month in ("2021-01", "2021-06", "2021-12", "2023-01"):
                rows.extend({"address_key": f"b{i}", "LATITUDE": lat, "LONGITUDE": lon,
                             "RENT_APPROVAL_DATE": month, "MONTHLY_RENT": 2000 + i,
                             "FLOOR_AREA_SQM": 80., "LEASE_COMMENCE_DATE": 2000,
                             "FLAT_TYPE_CLEAN": "4-room"} for _ in range(20))
        frame = pd.DataFrame(rows)
        sites = pd.DataFrame({"site_id": ["kulicke_soffa_serangoon"], "latitude": [1.35], "longitude": [103.8]})
        config = {"primary_hypothesis": {"site_id": "kulicke_soffa_serangoon", "radius_m": 1000},
                  "placebo_count": 4, "placebo_min_distance_to_focal_m": 6000, "placebo_min_separation_m": 1200}
        before, _ = match_placebos(frame, sites, config)
        frame.loc[frame.RENT_APPROVAL_DATE > "2021-12", "MONTHLY_RENT"] += 100000
        after, _ = match_placebos(frame, sites, config)
        pd.testing.assert_frame_equal(before, after)
        self.assertTrue((before.distance_from_focal_km >= 6).all())

    def test_vectorized_panels_keep_sparse_months_missing(self):
        months = ["2022-01", "2022-02", "2022-03", "2022-04"]
        rows = []
        for month in months:
            for name, lon in (("near", 103.805), ("ring", 103.84)):
                n = 19 if name == "near" and month == "2022-02" else 20
                rows.extend({"address_key": name, "_month": pd.Period(month, "M"),
                             "LATITUDE": 1.35, "LONGITUDE": lon} for _ in range(n))
        frame = pd.DataFrame(rows)
        centers = pd.DataFrame({"site_id": ["kulicke_soffa_serangoon"], "company": ["K&S"],
                                "symbol": ["KLIC"], "latitude": [1.35], "longitude": [103.8]})
        config = {"primary_hypothesis": {"site_id": "kulicke_soffa_serangoon", "radius_m": 1000},
                  "secondary_hypotheses": [], "min_cell_rentals": 20}
        with patch("analysis.experiments.company_worksite_followup.property_residuals", return_value=np.zeros(len(frame))):
            panel = build_panels(frame, centers, config)
        p = panel[panel.band == "frozen"].set_index("month")
        self.assertTrue(pd.isna(p.loc["2022-02", "premium_logpoints"]))
        self.assertTrue(pd.isna(p.loc["2022-03", "premium_change"]))
        self.assertEqual(p.loc["2022-04", "premium_change"], 0.)

    def test_paired_bootstrap_preserves_rows_and_month_weights(self):
        records = []
        for i, month in enumerate(pd.period_range("2024-06", periods=10, freq="M")):
            for row in range(2):
                for variant, error in (("base", 10), ("candidate", 8)):
                    records.append({"row_id": 2 * i + row, "month": str(month), "weight": row + 1,
                                    "variant": variant, "actual": 2000., "prediction": 2000. + error})
        interval = paired_month_interval(pd.DataFrame(records), "base", "candidate", repeats=100)
        self.assertAlmostEqual(interval["ci_low"], -2.)
        self.assertAlmostEqual(interval["ci_high"], -2.)

    def test_weighted_error_uses_population_weights(self):
        score = weighted_score([0., 0.], [1., 3.], [3., 1.])
        self.assertAlmostEqual(score["rmse_sgd"], np.sqrt(3))
        self.assertAlmostEqual(score["mae_sgd"], 1.5)

    def test_frozen_primary_hypothesis_matches_prior_earlier_selection(self):
        config = json.loads((ROOT / "analysis/configs/company_worksite_followup.json").read_text())
        self.assertEqual(config["primary_hypothesis"], {"site_id": "kulicke_soffa_serangoon", "symbol": "KLIC", "radius_m": 1000, "lag": 6})

    def test_local_head_cannot_change_predictions_outside_spatial_support(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(50, 3))
        y = 30 + x[:, 0] * 10
        query = np.vstack([np.zeros(3), np.ones(3)])
        pred, _ = local_ridge(x, y, np.ones(50), query, alpha=10)
        self.assertEqual(pred[0], 0.)
        self.assertNotEqual(pred[1], 0.)

    def test_local_head_scaling_preserves_units_and_future_queries_do_not_fit_coefficients(self):
        rng = np.random.default_rng(43)
        x, y = rng.normal(size=(50, 2)), rng.normal(size=50)
        q = np.array([[1., 2.], [3., 4.]])
        w = np.linspace(1, 2, 50)
        before, coefficient = local_ridge(x, y, w, q)
        scaled_x, scaled_q = x.copy(), q.copy()
        scaled_x[:, 0] *= 10000
        scaled_q[:, 0] *= 10000
        after, _ = local_ridge(scaled_x, y, w, scaled_q)
        np.testing.assert_allclose(before, after)
        altered_query = np.vstack([q, np.full((1, 2), 100000)])
        _, after_coefficient = local_ridge(x, y, w, altered_query)
        np.testing.assert_allclose(coefficient, after_coefficient)


if __name__ == "__main__":
    unittest.main()
