import unittest
import numpy as np
import pandas as pd
from shapely.geometry import Point, box

from analysis.experiments.local_spatial_features import (
    metric_xy, commercial_fields, canonical_buildings, park_fields, polygon_assignment, regional_features,
)
from analysis.experiments.local_market_fields import historical_market_features
from analysis.experiments.satellite_feature_import import AE_BANDS, TrainingPCA, join_satellite, validate_export
from analysis.compare_local_economy_landscape import matrix_pair, group_columns
from analysis.experiments.regional_income_features import income_features


class SpatialTests(unittest.TestCase):
    def test_metres_counts_and_normalized_kde(self):
        xy = metric_xy([103.85, 103.851], [1.3, 1.3])
        self.assertAlmostEqual(np.linalg.norm(xy[1] - xy[0]), 111.3, delta=.3)
        fields = commercial_fields([[0, 0]], [[0, 0], [300, 0], [300, 0]], [100, 300], [200], "P")
        self.assertEqual(fields.P_COUNT_100M.iloc[0], 1)
        self.assertEqual(fields.P_COUNT_300M.iloc[0], 2)
        expected = (1 + np.exp(-.5 * (300 / 200)**2)) / (2 * np.pi * .2**2)
        self.assertAlmostEqual(fields.P_KDE_200M_KM2.iloc[0], expected)
        pd.testing.assert_frame_equal(fields, commercial_fields([[0, 0]], [[300, 0], [0, 0]], [100, 300], [200], "P"))
        with self.assertRaises(ValueError):
            commercial_fields([[0, 0]], [], [100], [200], "P")

    def test_canonical_exact_address_and_coordinates(self):
        blocks = pd.DataFrame({"TOWN": ["a", "a"], "BLOCK": ["1", "2"], "STREET": ["Road", "Road"], "LATITUDE": [1.3, 1.3], "LONGITUDE": [103.85, 103.851]})
        transactions = blocks.iloc[[1, 0, 1]].drop(columns=["LATITUDE", "LONGITUDE"]).reset_index(drop=True)
        buildings, links = canonical_buildings(blocks, transactions)
        self.assertEqual(len(buildings), 2)
        self.assertEqual(links.building_id.iloc[0], links.building_id.iloc[2])
        self.assertNotEqual(links.building_id.iloc[0], links.building_id.iloc[1])
        with self.assertRaises(ValueError):
            canonical_buildings(pd.concat([blocks, blocks]), transactions)

    def test_polygon_support_and_no_overlap_double_count(self):
        polygon = box(-1000, -1000, 1000, 1000)
        fields = park_fields([[0, 0], [2000, 0]], [polygon, polygon], [100, 300, 500])
        self.assertAlmostEqual(fields.PARK_FRACTION_500M.iloc[0], 1)
        self.assertAlmostEqual(fields.PARK_FRACTION_500M.iloc[1], 0)
        self.assertAlmostEqual(fields.PARK_NEAREST_M.iloc[1], 1000)
        buildings = pd.DataFrame({"x_m": [0, 2000], "y_m": [0, 0]})
        joined = polygon_assignment(buildings, [polygon], [{"PLN_AREA_N": "AREA", "SUBZONE_N": "SUB"}])
        self.assertEqual(joined.polygon_planning_area.iloc[0], "area")
        self.assertEqual(joined.polygon_match_count.iloc[1], 0)

    def test_census_publication_and_suppression(self):
        census = pd.DataFrame({"region_label": ["a"], "is_planning_area": [True], "CENSUS2020_POPULATION": [100], "CENSUS2020_AGE65_SHARE": [.2]})
        f = pd.DataFrame({"PLANNING_AREA": ["a", "a", "unknown"]})
        result = regional_features(f, ["2021-06", "2021-07", "2022-01"], census)
        self.assertTrue(np.isnan(result.CENSUS2020_POPULATION.iloc[0]))
        self.assertEqual(result.CENSUS2020_POPULATION.iloc[1], 100)
        self.assertEqual(result.CENSUS2020_UNAVAILABLE.iloc[2], 1)

    def test_income_bands_keep_regional_support_and_publication_cutoff(self):
        source = {"Number": "a", "Total": 100, "NoEmployedPerson": 20,
                  "Below_1_000": 2, "1_000_1_999": 3, "2_000_2_999": 5}
        for band in ("10_000_10_999", "11_000_11_999", "12_000_12_999", "13_000_13_999", "14_000_14_999", "15_000_17_499", "17_500_19_999", "20_000andOver"):
            source[band] = 5
        result = income_features(pd.Series(["a", "a", "unpublished"]), ["2022-07", "2022-08", "2024-05"], pd.DataFrame([source]))
        self.assertEqual(result.CENSUS_INCOME_UNAVAILABLE.iloc[0], 1)
        self.assertAlmostEqual(result.CENSUS_INCOME_BELOW3000_SHARE.iloc[1], .1)
        self.assertAlmostEqual(result.CENSUS_INCOME_GE10000_SHARE.iloc[1], .4)
        self.assertAlmostEqual(result.CENSUS_INCOME_NO_EMPLOYED_SHARE.iloc[1], .2)
        self.assertEqual(result.CENSUS_INCOME_UNAVAILABLE.iloc[2], 1)

    def test_histories_are_origin_safe_under_target_mutation(self):
        cfg = {"history_radii_m": [500, 1000], "history_windows_months": [6, 12], "history_publication_lag_months": 1,
               "history_shrinkage_observations": 20, "history_continuous_sigma_m": 500, "history_temporal_half_life_months": 6}
        buildings = pd.DataFrame({"x_m": [0., 300.], "y_m": [0., 0.]}, index=["a", "b"])
        past = pd.DataFrame({"building_id": ["a", "b", "a"], "FLAT_TYPE_CLEAN": ["3"] * 3,
                             "_MONTH": pd.PeriodIndex(["2021-01", "2021-02", "2021-03"], freq="M"),
                             "MONTHLY_RENT": [1000., 2000., 99999.], "FLOOR_AREA_SQM": [60.] * 3})
        q = past.iloc[[2]].copy()
        # March origin -> only January/February available after 1-month lag.
        q["_MONTH"] = pd.Period("2021-04", "M")
        a = historical_market_features(past, q, ["2021-03"], buildings, cfg)
        self.assertAlmostEqual(a.LOCAL_500M_6M_MEAN.iloc[0], 1500.)
        self.assertEqual(a.LOCAL_500M_6M_COUNT.iloc[0], 2)
        changed = past.copy()
        changed.loc[2, "MONTHLY_RENT"] = -1e9
        future = past.iloc[[0]].copy()
        future["_MONTH"] = pd.Period("2022-01", "M")
        future["MONTHLY_RENT"] = 1e12
        changed = pd.concat([changed, future], ignore_index=True)
        q["MONTHLY_RENT"] = -1e12
        b = historical_market_features(changed, q, ["2021-03"], buildings, cfg)
        pd.testing.assert_frame_equal(a, b)
        empty = historical_market_features(past, q, [pd.NaT], buildings, cfg)
        self.assertTrue(empty.LOCAL_KERNEL_MEAN.isna().all())
        self.assertTrue((empty.LOCAL_HISTORY_UNAVAILABLE == 1).all())

    def test_own_month_is_never_included(self):
        # The method rejects an origin which reaches its own target month.
        q = pd.DataFrame({"_MONTH": pd.PeriodIndex(["2021-03"], freq="M")})
        with self.assertRaises(ValueError):
            historical_market_features(pd.DataFrame(), q, ["2021-03"], pd.DataFrame(), {"history_publication_lag_months": 1})

    def test_feature_rows_and_original_inputs_preserved(self):
        original = pd.DataFrame({"cat": pd.Categorical(["a", "b"]), "x": [1, 2]}, index=[4, 8])
        extra = pd.DataFrame({"new": [3, 4]}, index=[4, 8])
        entry = {"bundle": {"X_train": original, "X_query": original}, "added": {"train": extra, "query": extra}}
        x, q = matrix_pair(entry, ["new"])
        pd.testing.assert_frame_equal(x[original.columns], original)
        self.assertEqual(list(q.index), [4, 8])
        with self.assertRaises(ValueError):
            matrix_pair(entry, ["x"])
        groups = group_columns(["CENSUS2020_POPULATION", "CENSUS2020_DENSITY_RETRO_KM2"])
        self.assertNotIn("CENSUS2020_DENSITY_RETRO_KM2", groups["regional"])


class SatelliteTests(unittest.TestCase):
    def export(self):
        return pd.DataFrame([{**{c: .1 for c in AE_BANDS}, "building_id": "a", "observation_year": 2020, "radius_m": 100,
                              "observation_end": "2020-12-31", "available_at": "2025-07-30", "availability_source_url": "https://deepmind.google/",
                              "valid_pixel_fraction": 1., "valid_observation_count": 1}])

    def test_annual_completion_and_actual_publication(self):
        exported = self.export()
        queries = pd.DataFrame({"building_id": ["a", "a"], "forecast_origin": pd.PeriodIndex(["2024-05", "2025-08"], freq="M")})
        strict = join_satellite(exported, queries, "alphaearth", 100)
        self.assertEqual(strict.alphaearth_100m_unavailable.iloc[0], 1)
        self.assertEqual(strict.alphaearth_100m_unavailable.iloc[1], 0)
        retrospective = join_satellite(exported, queries, "alphaearth", 100, retrospective=True)
        self.assertEqual(retrospective.alphaearth_100m_unavailable.iloc[0], 0)
        same_year = queries.iloc[[0]].copy()
        same_year["forecast_origin"] = pd.Period("2020-12", "M")
        self.assertEqual(join_satellite(exported, same_year, "alphaearth", 100, retrospective=True).alphaearth_100m_unavailable.iloc[0], 1)
        exported["available_at"] = "2021-01-01"
        with self.assertRaises(ValueError):
            validate_export(exported, "alphaearth")

    def test_pca_fits_only_training_and_keeps_missing_missing(self):
        rng = np.random.default_rng(42)
        train = rng.normal(size=(40, 64))
        pca = TrainingPCA(8).fit(train)
        saved = pca.model_.components_.copy()
        pca.transform(np.full((5, 64), 999999.))
        np.testing.assert_array_equal(saved, pca.model_.components_)
        np.testing.assert_allclose(pca.mean_, train.mean(axis=0))
        values, flag = pca.transform(np.full((1, 64), np.nan))
        self.assertTrue(np.isnan(values).all())
        self.assertEqual(flag[0], 1)


if __name__ == "__main__":
    unittest.main()
