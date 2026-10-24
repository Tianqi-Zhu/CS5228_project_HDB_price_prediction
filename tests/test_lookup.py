"""Check weighting, ambiguity, row alignment, and the complete local export."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from public_rent_lookup.run import (
    build_statistics, canonical_keys, known_label_presence, lookup, make_submission, run, sha256,
)


def local_rows():
    return pd.DataFrame({
        "RENT_APPROVAL_DATE": ["2025-03", "2025-03"],
        "TOWN": ["central", "bedok"], "BLOCK": ["1", "2"],
        "STREET": ["New Market Road", "Bedok North Street 3"],
        "FLAT_TYPE": ["3-room", "4 room"],
    }, index=[7, 3])


def public_rows():
    return pd.DataFrame({
        "rent_approval_date": ["2025-03"] * 4,
        "town": ["BEDOK", "CENTRAL AREA", "CENTRAL AREA", "CENTRAL AREA"],
        "block": ["2", "1", "1", "1"],
        "street_name": ["BEDOK NTH ST 3", "NEW MKT RD", "NEW MKT RD", "NEW MKT RD"],
        "flat_type": ["4-ROOM", "3-ROOM", "3-ROOM", "3-ROOM"],
        "monthly_rent": [4000, 2000, 2000, 5000],
    })


class PublicLookupTests(unittest.TestCase):
    def test_record_weighted_mean_and_population_variance(self):
        joined = lookup(local_rows(), build_statistics(public_rows()))
        np.testing.assert_allclose(joined.public_mean, [3000, 4000])
        np.testing.assert_allclose(joined.public_variance, [2000000, 0])
        self.assertEqual(joined.public_unique_rents.tolist(), [2, 1])

    def test_matching_preserves_order_and_duplicate_local_rows(self):
        local = pd.concat([local_rows().iloc[[1]], local_rows()])
        original = local.copy(deep=True)
        joined = lookup(local, build_statistics(public_rows()))
        self.assertEqual(joined.source_row.tolist(), [0, 1, 2])
        self.assertEqual(joined.public_mean.tolist(), [4000, 3000, 4000])
        pd.testing.assert_frame_equal(local, original)

    def test_no_fuzzy_or_month_fallback(self):
        local = local_rows()
        local.loc[7, "RENT_APPROVAL_DATE"] = "2025-04"
        with self.assertRaisesRegex(ValueError, "1 unmatched"):
            lookup(local, build_statistics(public_rows()))

    def test_known_targets_only_affect_audit_not_predictions(self):
        local = local_rows().assign(MONTHLY_RENT=[2000, 4000])
        changed = local.assign(MONTHLY_RENT=[9999, 9999])
        stats = build_statistics(public_rows())
        np.testing.assert_array_equal(lookup(local, stats).public_mean,
                                      lookup(changed, stats).public_mean)
        self.assertEqual(known_label_presence(local, public_rows()), 2)
        self.assertEqual(known_label_presence(changed, public_rows()), 0)

    def test_missing_keys_and_invalid_public_rents_fail(self):
        local = local_rows()
        local.loc[7, "BLOCK"] = None
        with self.assertRaisesRegex(ValueError, "missing or empty"):
            canonical_keys(local)
        public = public_rows()
        public.loc[0, "monthly_rent"] = np.nan
        with self.assertRaisesRegex(ValueError, "finite, positive"):
            build_statistics(public)

    def test_template_values_are_ignored_and_reordered_ids_fail(self):
        template = pd.DataFrame({"Id": [0, 1], "Predicted": [99999, -99]})
        result = make_submission(template, np.array([3000., 4000.]))
        self.assertEqual(result.Predicted.tolist(), [3000, 4000])
        self.assertEqual(template.Predicted.tolist(), [99999, -99])
        with self.assertRaisesRegex(ValueError, "original order"):
            make_submission(template.iloc[::-1], np.array([3000., 4000.]))

    def test_end_to_end_exports_and_preserves_source_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            (base / "data").mkdir()
            (base / "source").mkdir()
            paths = {"public": base / "source/public.csv", "train": base / "data/train.csv",
                     "test": base / "data/test.csv", "template": base / "data/example.csv"}
            local = local_rows().reset_index(drop=True)
            public_rows().to_csv(paths["public"], index=False)
            local.assign(MONTHLY_RENT=[2000, 4000]).to_csv(paths["train"], index=False)
            local.to_csv(paths["test"], index=False)
            pd.DataFrame({"Id": [0, 1], "Predicted": [0, 0]}).to_csv(paths["template"], index=False)
            before = {name: sha256(path) for name, path in paths.items()}
            summary = run(paths["public"], paths["train"], paths["test"],
                          paths["template"], base / "outputs")
            self.assertEqual(before, {name: sha256(path) for name, path in paths.items()})
            saved = pd.read_csv(base / "outputs/submission.csv")
            self.assertEqual(saved.Predicted.tolist(), [3000, 4000])
            self.assertIsNone(summary["actual_kaggle_test_rmse"])
            self.assertEqual(summary["test"]["single_rent_rows"], 1)
            self.assertAlmostEqual(summary["known_label_reconstruction"]["all_train"]["rmse_sgd"],
                                   np.sqrt(500000))
            metadata = json.loads((base / "outputs/summary.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["outputs"]["submission.csv"]["rows"], 2)


if __name__ == "__main__":
    unittest.main()
