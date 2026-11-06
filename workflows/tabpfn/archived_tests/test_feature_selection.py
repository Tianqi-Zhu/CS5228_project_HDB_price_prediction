"""Temporal information boundaries for the canonical feature candidates."""

import unittest
from unittest.mock import patch
import pickle

import numpy as np
import pandas as pd

from analysis.pipeline.features import (
    FEATURE_CANDIDATES, HISTORY, HISTORY_POLICIES, EnhancedHistoryEncoder, FeatureSpec,
    HistoricalRentEncoder, HistoryPolicy, MarketContextEncoder, RentalFeatureEngineer, model_inputs,
)


def rentals():
    return pd.DataFrame({
        "RENT_APPROVAL_DATE": ["2021-01", "2021-02", "2021-02", "2021-03"],
        "TOWN_KEY": ["a"] * 4, "BLOCK_KEY": ["1"] * 4, "STREET_KEY": ["road"] * 4,
        "FLAT_TYPE_CLEAN": ["3-room"] * 4, "FLAT_MODEL_CLEAN": ["m"] * 4,
        "FLOOR_AREA_SQM": [70.0] * 4, "LEASE_COMMENCE_DATE": [2000] * 4,
        "PLANNING_AREA_CLEAN": ["central"] * 4,
        "MONTHLY_RENT": [1000.0, 2000.0, 4000.0, 9000.0],
    }, index=[7, 3, 15, 2])


class FeatureSelectionTests(unittest.TestCase):
    def test_own_month_and_all_later_labels_are_excluded(self):
        frame = rentals()
        before = HistoricalRentEncoder().fit_transform(frame)
        changed = frame.copy()
        changed.loc[changed["RENT_APPROVAL_DATE"] >= "2021-02", "MONTHLY_RENT"] *= 100
        after = HistoricalRentEncoder().fit_transform(changed)
        pd.testing.assert_frame_equal(before.loc[[7, 3, 15]], after.loc[[7, 3, 15]])
        self.assertTrue(np.isnan(before.loc[7, "HIST_SIGNATURE_MEAN"]))
        self.assertEqual(before.loc[7, "HIST_SIGNATURE_COUNT"], 0)
        self.assertEqual(before.loc[3, "HIST_SIGNATURE_MEAN"], 1000)
        self.assertEqual(before.loc[15, "HIST_SIGNATURE_MEAN"], 1000)
        self.assertAlmostEqual(before.loc[2, "HIST_SIGNATURE_MEAN"], 7000 / 3)
        self.assertEqual(before.loc[2, "HIST_SIGNATURE_LAST_MEAN"], 3000)

    def test_overlap_month_excludes_that_month_and_future_queries_are_frozen(self):
        frame = rentals()
        prediction = frame.iloc[[0, 0, 0]].drop(columns="MONTHLY_RENT").reset_index(drop=True)
        prediction["RENT_APPROVAL_DATE"] = ["2021-06", "2021-03", "2021-04"]
        result = HistoricalRentEncoder().fit(frame).transform(prediction)
        np.testing.assert_allclose(result["HIST_SIGNATURE_MEAN"], [4000, 7000 / 3, 4000])
        self.assertEqual(result["HIST_SIGNATURE_RECENCY_MONTHS"].tolist(), [3, 1, 1])
        prediction["MONTHLY_RENT"] = [1e9] * 3
        pd.testing.assert_frame_equal(result, HistoricalRentEncoder().fit(frame).transform(prediction))

    def test_unseen_property_can_use_broader_area_history(self):
        frame = rentals()
        prediction = frame.iloc[[0]].drop(columns="MONTHLY_RENT").copy()
        prediction["RENT_APPROVAL_DATE"] = "2021-04"
        prediction["BLOCK_KEY"] = "new"
        result = HistoricalRentEncoder().fit(frame).transform(prediction)
        self.assertEqual(result.iloc[0]["HIST_SIGNATURE_COUNT"], 0)
        self.assertTrue(np.isnan(result.iloc[0]["HIST_SIGNATURE_MEAN"]))
        self.assertEqual(result.iloc[0]["HIST_AREA_FLAT_MEAN"], 4000)

    def test_vocabularies_are_training_only_and_rows_preserved(self):
        frame = rentals()
        prediction = frame.iloc[[0]].drop(columns="MONTHLY_RENT").copy()
        prediction["RENT_APPROVAL_DATE"] = "2021-04"
        prediction["FLAT_TYPE_CLEAN"] = "new-type"
        spec = FeatureSpec(("FLAT_TYPE_CLEAN", "FLOOR_AREA_SQM"), ("FLAT_TYPE_CLEAN",), True)
        x_fit, x_pred = model_inputs(frame, prediction, spec)
        self.assertEqual(x_fit.index.tolist(), frame.index.tolist())
        self.assertEqual(x_pred.index.tolist(), prediction.index.tolist())
        self.assertTrue(pd.isna(x_pred.iloc[0]["FLAT_TYPE_CLEAN"]))
        self.assertNotIn("MONTHLY_RENT", x_fit)
        self.assertEqual(list(x_fit), list(spec.columns + HISTORY))

    def test_schemas_have_no_duplicates_constants_or_economic_features(self):
        for name, spec in FEATURE_CANDIDATES.items():
            with self.subTest(name=name):
                self.assertEqual(len(spec.all_columns), len(set(spec.all_columns)))
                self.assertTrue(set(spec.categorical).issubset(spec.columns))
                self.assertFalse({"FEE", "FURNISHED", "MONTHLY_RENT"}.intersection(spec.all_columns))
                self.assertFalse(any(column.startswith(("COE_", "STOCK_", "SG_STOCK_")) for column in spec.all_columns))

    def test_persisted_engineer_uses_fitting_y_and_ignores_prediction_targets(self):
        frame = rentals()
        spec = FeatureSpec(("FLAT_TYPE_CLEAN", "FLOOR_AREA_SQM"), ("FLAT_TYPE_CLEAN",), True)
        with patch.dict(FEATURE_CANDIDATES, {"test": spec}), patch(
            "analysis.pipeline.features.build_feature_frame", side_effect=lambda x: x.copy(),
        ):
            engineer = RentalFeatureEngineer("test")
            x_fit = engineer.fit_transform(frame.drop(columns="MONTHLY_RENT"), frame["MONTHLY_RENT"])
            restored = pickle.loads(pickle.dumps(engineer))
            prediction = frame.iloc[[0, 1]].copy()
            prediction["RENT_APPROVAL_DATE"] = ["2021-04", "2021-05"]
            prediction["MONTHLY_RENT"] = 1e9
            predicted_features = restored.transform(prediction)
            self.assertEqual(predicted_features["HIST_SIGNATURE_MEAN"].tolist(), [4000, 4000])
            self.assertEqual(list(x_fit), list(restored.get_feature_names_out()))
            pd.testing.assert_frame_equal(engineer.transform(prediction), predicted_features)

    def test_enhanced_policies_exclude_own_and_later_months(self):
        frame = rentals()
        changed = frame.copy()
        changed.loc[changed["RENT_APPROVAL_DATE"] >= "2021-02", "MONTHLY_RENT"] *= 100
        for name, policy in HISTORY_POLICIES.items():
            with self.subTest(policy=name):
                before = EnhancedHistoryEncoder(policy).fit(frame).transform(frame)
                after = EnhancedHistoryEncoder(policy).fit(changed).transform(changed)
                pd.testing.assert_frame_equal(before.loc[[7, 3, 15]], after.loc[[7, 3, 15]])

    def test_recent_window_stays_frozen_in_long_forecast(self):
        frame = rentals()
        prediction = frame.iloc[[0, 0]].drop(columns="MONTHLY_RENT").reset_index(drop=True)
        prediction["RENT_APPROVAL_DATE"] = ["2021-04", "2023-04"]
        encoder = EnhancedHistoryEncoder(HistoryPolicy("W", window_months=1)).fit(frame)
        result = encoder.transform(prediction)
        self.assertEqual(result["W_SIGNATURE_MEAN"].tolist(), [9000, 9000])
        self.assertEqual(result["W_SIGNATURE_COUNT"].tolist(), [1, 1])
        self.assertEqual(result["W_SIGNATURE_RECENCY_MONTHS"].tolist(), [1, 25])

    def test_decay_increases_weight_of_recent_records(self):
        frame = rentals()
        prediction = frame.iloc[[0]].drop(columns="MONTHLY_RENT").copy()
        prediction["RENT_APPROVAL_DATE"] = "2021-04"
        encoder = EnhancedHistoryEncoder(HistoryPolicy("D", half_life_months=1)).fit(frame)
        result = encoder.transform(prediction)
        expected = (1000 * .125 + 6000 * .25 + 9000 * .5) / (.125 + 2 * .25 + .5)
        self.assertAlmostEqual(result.iloc[0]["D_SIGNATURE_MEAN"], expected)

    def test_market_context_is_causal_and_damps_future_growth(self):
        frame = rentals()
        changed = frame.copy()
        changed.loc[changed["RENT_APPROVAL_DATE"] >= "2021-02", "MONTHLY_RENT"] *= 100
        before = MarketContextEncoder().fit(frame).transform(frame)
        after = MarketContextEncoder().fit(changed).transform(changed)
        pd.testing.assert_frame_equal(before.loc[[7, 3, 15]], after.loc[[7, 3, 15]])
        prediction = frame.iloc[[0, 0]].drop(columns="MONTHLY_RENT").reset_index(drop=True)
        prediction["RENT_APPROVAL_DATE"] = ["2021-04", "2023-04"]
        result = MarketContextEncoder().fit(frame).transform(prediction)
        self.assertEqual(result["MARKET_TYPE_LAST"].tolist(), [9000, 9000])
        self.assertLess(result["MARKET_TYPE_PROJECTED"].iloc[1], 9000 * np.exp(.02 / .15))

    def test_combined_history_pipeline_matches_experiment_inputs(self):
        frame = rentals()
        prediction = frame.iloc[[0, 1]].drop(columns="MONTHLY_RENT").copy()
        prediction["RENT_APPROVAL_DATE"] = ["2021-04", "2021-05"]
        candidate = FeatureSpec(("FLAT_TYPE_CLEAN", "FLOOR_AREA_SQM"), ("FLAT_TYPE_CLEAN",), True,
            history_policies=("standard", "premium_values"), history_grains=("BLOCK_FLAT", "AREA_FLAT"),
            market_context=True)
        expected_fit, expected_pred = model_inputs(frame, prediction, candidate)
        with patch.dict(FEATURE_CANDIDATES, {"combined_test": candidate}), patch(
            "analysis.pipeline.features.build_feature_frame", side_effect=lambda x: x.copy(),
        ):
            engineer = RentalFeatureEngineer("combined_test")
            actual_fit = engineer.fit_transform(frame.drop(columns="MONTHLY_RENT"), frame["MONTHLY_RENT"])
            actual_pred = engineer.transform(prediction)
        pd.testing.assert_frame_equal(actual_fit, expected_fit)
        pd.testing.assert_frame_equal(actual_pred, expected_pred)
        self.assertEqual(list(actual_fit), list(candidate.all_columns))


if __name__ == "__main__":
    unittest.main()
