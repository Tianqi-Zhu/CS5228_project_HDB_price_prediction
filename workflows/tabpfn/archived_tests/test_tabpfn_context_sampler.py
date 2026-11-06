"""Leakage and determinism checks for TabPFN context retrieval."""

import unittest

import numpy as np
import pandas as pd

from analysis.experiments.tabpfn_context_sampler import (
    SamplerConfig, metadata_from_features, select_context,
)
from analysis.experiments.tabpfn_sampled_finetuning import (
    EpisodeConfig, build_episode_positions, fixed_support_query_split,
)


class ContextSamplerTests(unittest.TestCase):
    def setUp(self):
        count = 72
        ids = np.arange(count)
        self.features = pd.DataFrame({
            "LATITUDE": 1.25 + (ids % 9) * .008,
            "LONGITUDE": 103.75 + (ids % 8) * .009,
            "FLOOR_AREA_SQM": 55 + (ids % 5) * 15,
            "BUILDING_AGE_FROM_COMPLETION": 5 + (ids % 17),
            "FLAT_TYPE_CLEAN": np.where(ids % 2, "4 room", "3 room"),
            "TOWN_CLEAN": np.where(ids % 3, "town a", "town b"),
        }, index=ids)
        self.features.loc[2, "BUILDING_AGE_FROM_COMPLETION"] = np.nan
        months = np.array(["2021-01"] * 24 + ["2021-02"] * 24 + ["2021-03"] * 24)
        groups = ids.copy()
        groups[5] = groups[50]
        self.metadata = metadata_from_features(self.features, months, duplicate_groups=groups)
        self.queries = np.array([50, 51, 64])
        self.eligible = np.arange(50)
        self.config = SamplerConfig(context_size=12, nearest_candidates_per_query=24)

    def test_historical_policy_and_duplicate_exclusion(self):
        result = select_context(self.queries, self.eligible, self.metadata,
                                strategy="S0", information_policy="historical_only",
                                config=self.config, seed=42)
        self.assertEqual(len(result.row_ids), 47)
        self.assertNotIn(5, result.row_ids)
        self.assertTrue((self.metadata.month_ordinal[self.metadata.positions(result.row_ids)]
                         < self.metadata.month_ordinal[self.metadata.positions(self.queries)].min()).all())
        self.assertEqual(result.audit["cutoff_month"], "2021-03")

    def test_sampled_strategies_are_deterministic_and_unique(self):
        for strategy in ("S1", "S2", "S3", "S4"):
            with self.subTest(strategy=strategy):
                a = select_context(self.queries, self.eligible, self.metadata,
                                   strategy=strategy, information_policy="historical_only",
                                   config=self.config, seed=9)
                b = select_context(self.queries, self.eligible, self.metadata,
                                   strategy=strategy, information_policy="historical_only",
                                   config=self.config, seed=9)
                np.testing.assert_array_equal(a.row_ids, b.row_ids)
                self.assertEqual(len(a.row_ids), self.config.context_size)
                self.assertEqual(len(np.unique(a.row_ids)), len(a.row_ids))
                self.assertNotIn(5, a.row_ids)
                self.assertEqual(sum(a.audit["component_counts"].values()), len(a.row_ids))

    def test_earliest_query_month_controls_mixed_batch(self):
        queries = np.array([26, 50])
        result = select_context(queries, np.arange(26), self.metadata,
                                strategy="S4", information_policy="historical_only",
                                config=self.config, seed=42)
        self.assertEqual(result.audit["cutoff_month"], "2021-02")
        self.assertTrue((result.row_ids < 24).all())
        self.assertEqual(len(result.row_ids), 12)

    def test_available_pool_excludes_query_and_duplicate_group(self):
        result = select_context(self.queries, np.arange(72), self.metadata,
                                strategy="S1", information_policy="available_labeled_pool",
                                config=SamplerConfig(context_size=70), seed=5)
        self.assertEqual(len(result.row_ids), 68)
        self.assertFalse(np.isin(result.row_ids, self.queries).any())
        self.assertNotIn(5, result.row_ids)

    def test_local_shortlist_refills_from_eligible_history(self):
        config = SamplerConfig(context_size=12, nearest_candidates_per_query=1)
        result = select_context(self.queries, self.eligible, self.metadata,
                                strategy="S3", information_policy="historical_only",
                                config=config, seed=42)
        self.assertEqual(len(result.row_ids), 12)
        self.assertGreater(result.audit["component_counts"]["fallback_uniform"], 0)
        self.assertTrue(np.isin(result.row_ids, self.eligible).all())

    def test_does_not_change_model_features(self):
        before = self.features.copy(deep=True)
        select_context(self.queries, self.eligible, self.metadata,
                       strategy="S3", information_policy="historical_only",
                       config=self.config, seed=9)
        pd.testing.assert_frame_equal(self.features, before)

    def test_chronological_episodes_share_queries_across_strategies(self):
        episode = EpisodeConfig(support_rows=8, query_rows=4, episodes_per_epoch=3,
                                first_query_month="2021-03")
        config = SamplerConfig(context_size=8, nearest_candidates_per_query=16)
        results = {}
        for strategy in ("S1", "S2", "S3", "S4"):
            positions, audits = build_episode_positions(
                self.metadata, strategy=strategy, sampler_config=config,
                episode_config=episode, epoch_seed=42)
            self.assertEqual(len(positions), 3)
            self.assertTrue(all(len(part) == 12 for part in positions))
            self.assertTrue(all((self.metadata.month_ordinal[part[:8]]
                                 < self.metadata.month_ordinal[part[8:]].min()).all()
                                for part in positions))
            results[strategy] = [row["query_ids_sha256"] for row in audits]
        self.assertTrue(all(result == results["S1"] for result in results.values()))

    def test_fixed_support_query_split_preserves_order(self):
        X = np.arange(12).reshape(6, 2)
        y = np.arange(6)
        context, query, y_context, y_query = fixed_support_query_split(X, y, support_rows=4)
        np.testing.assert_array_equal(context, X[:4])
        np.testing.assert_array_equal(query, X[4:])
        np.testing.assert_array_equal(y_context, y[:4])
        np.testing.assert_array_equal(y_query, y[4:])


if __name__ == "__main__":
    unittest.main()
