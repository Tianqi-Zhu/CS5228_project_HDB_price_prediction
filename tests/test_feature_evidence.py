"""Nonadditive deletion definitions and paired row-weighted month sensitivity."""

import unittest

import numpy as np
import pandas as pd

from analysis.experiments.feature_evidence import (
    GROUPS, JOINT_GROUPS, SPEC, ablations, monthly_squared_errors,
    paired_month_interval, retained_columns,
)


class FeatureEvidenceTests(unittest.TestCase):
    def test_final_submission_checks_schema_ids_and_finite_prediction(self):
        from analysis.experiments.canonical_final import validate_submission
        template = pd.DataFrame({"Id": [0, 1], "Predicted": [99., 99.]})
        result = validate_submission(template, [2100., 3200.])
        self.assertEqual(result.Predicted.tolist(), [2100., 3200.])
        self.assertEqual(template.Predicted.tolist(), [99., 99.])
        for prediction in ([1.], [1., np.nan]):
            with self.assertRaises(ValueError):
                validate_submission(template, prediction)
        with self.assertRaises(ValueError):
            validate_submission(template.iloc[::-1], [2100., 3200.])

    def test_common_model_selection_filters_features_before_ranking(self):
        from analysis.experiments.common_feature_models import choose_compact, source_digests
        trials = [{"profile": "premium", "inner_scores": {"rmse_sgd": 1}},
                  {"profile": "compact", "inner_scores": {"rmse_sgd": 5}},
                  {"profile": "compact", "inner_scores": {"rmse_sgd": 4}}]
        self.assertEqual(choose_compact(trials), trials[2])
        with self.assertRaises(ValueError):
            choose_compact(trials[:1])
        self.assertEqual(source_digests({r"C:\old\data\train.csv": "abc", r"C:\old\data\auxiliary\sg-hdb-block.csv": "def"}),
                         source_digests({"/new/project/data/train.csv": "abc", "/new/project/data/auxiliary/sg-hdb-block.csv": "def"}))

    def test_groups_partition_the_default_schema_and_single_deletions_are_exact(self):
        columns = [c for group in GROUPS.values() for c in group]
        self.assertEqual(set(columns), set(SPEC.all_columns))
        self.assertEqual(len(columns), 27)
        definitions = ablations()
        self.assertEqual(len(definitions), 39)
        for c in SPEC.all_columns:
            remaining = retained_columns(SPEC.all_columns, definitions[f"drop__{c}"]["removed"])
            self.assertEqual(len(remaining), 26)
            self.assertNotIn(c, remaining)
        self.assertEqual(len(JOINT_GROUPS["all_history"]), 8)

    def test_monthly_sse_and_interval_have_correct_deletion_sign_and_row_weighting(self):
        actual = np.zeros(4)
        baseline = np.ones(4)
        candidate = np.full(4, 3.)
        monthly = monthly_squared_errors(["2024-04", "2024-04", "2024-04", "2024-05"],
                                          actual, baseline, candidate)
        self.assertEqual(monthly.n.tolist(), [3, 1])
        np.testing.assert_allclose(monthly.candidate_sse, [27, 9])
        interval = paired_month_interval(monthly, repeats=200)
        self.assertAlmostEqual(interval["month_block_p025"], 2.)
        self.assertAlmostEqual(interval["month_block_p975"], 2.)
        identical = monthly_squared_errors(["a", "a", "b", "c"], actual, baseline, baseline)
        self.assertEqual(paired_month_interval(identical), {"month_block_p025": 0., "month_block_p975": 0.})

    def test_deletion_and_resampling_reject_invalid_definitions(self):
        with self.assertRaises(ValueError):
            retained_columns(["a", "b"], ["c"])
        with self.assertRaises(ValueError):
            retained_columns(["a"], ["a"])
        with self.assertRaises(ValueError):
            paired_month_interval(pd.DataFrame({"n": [0, 1], "baseline_sse": [1, 1], "candidate_sse": [1, 1]}))


if __name__ == "__main__":
    unittest.main()
