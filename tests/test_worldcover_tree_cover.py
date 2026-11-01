import unittest

import numpy as np
import pandas as pd
from rasterio.transform import from_origin

from analysis.experiments.worldcover_tree_cover import buffer_statistics, origin_features


class TreeCoverTests(unittest.TestCase):
    def test_tree_class_and_metric_circle(self):
        raster = np.full((100, 100), 50, dtype="uint8")
        raster[:, :50] = 10
        result = buffer_statistics(raster, from_origin(-500, 500, 10, 10), [[0, 0]], [100, 300])
        self.assertEqual(result.TREE_CLASS_FRACTION_100M.iloc[0], .5)
        self.assertEqual(result.TREE_CLASS_FRACTION_300M.iloc[0], .5)
        self.assertEqual(result.WORLDCOVER_VALID_FRACTION_300M.iloc[0], 1)
        mangroves = buffer_statistics(np.full_like(raster, 95), from_origin(-500, 500, 10, 10), [[0, 0]], [100])
        self.assertEqual(mangroves.TREE_CLASS_FRACTION_100M.iloc[0], 0)

    def test_nodata_and_outside_are_missing_instead_of_zero_trees(self):
        raster = np.full((100, 100), 10, dtype="uint8")
        raster[:, :50] = 0
        result = buffer_statistics(raster, from_origin(-500, 500, 10, 10), [[0, 0], [2000, 2000]], [100])
        self.assertTrue(result.TREE_CLASS_FRACTION_100M.isna().all())
        self.assertEqual(result.WORLDCOVER_VALID_FRACTION_100M.iloc[0], .5)
        self.assertEqual(result.WORLDCOVER_VALID_FRACTION_100M.iloc[1], 0)

    def test_publication_boundaries_and_missing_buildings(self):
        cfg = {"radii_m": [100, 300], "products": [
            {"year": 2020, "published_at": "2021-10-20", "available_at": "2021-11-01"},
            {"year": 2021, "published_at": "2022-10-28", "available_at": "2022-11-01"}]}
        products = {year: pd.DataFrame({"building_id": ["a"], "TREE_CLASS_FRACTION_100M": [value], "TREE_CLASS_FRACTION_300M": [value]}) for year, value in [(2020, .2), (2021, .7)]}
        ids = pd.Series(["a", "a", "a", "a", "a", "unknown"], index=[1, 3, 5, 7, 9, 11])
        origins = ["2021-10", "2021-11", "2022-10", "2022-11", pd.NaT, "2024-05"]
        result, years = origin_features(ids, origins, products, cfg)
        self.assertEqual(list(result.index), list(ids.index))
        self.assertTrue(result.loc[[1, 9, 11], "TREE_CLASS_FRACTION_100M"].isna().all())
        np.testing.assert_allclose(result.loc[[3, 5, 7], "TREE_CLASS_FRACTION_100M"], [.2, .2, .7])
        self.assertEqual(years.loc[5], 2020)
        self.assertEqual(years.loc[7], 2021)
        self.assertEqual(result.loc[11, "TREE_COVER_UNAVAILABLE"], 1)


if __name__ == "__main__":
    unittest.main()
