"""Exact canonical joining for the public-source audit."""

import unittest

import pandas as pd

from analysis.studies.source_overlap import canonical_keys


class SourceOverlapTests(unittest.TestCase):
    def test_public_and_competition_spellings_match_without_labels(self):
        local = pd.DataFrame({"RENT_APPROVAL_DATE": ["2021-01", "2024-04", "2024-05"],
            "TOWN": ["Punggol", "central", "bedok"], "BLOCK": ["604a", "1", "2"],
            "STREET": ["Punggol Road", "New Market Road", "Bedok North Street 3"],
            "FLAT_TYPE": ["4 room", "3-room", "5 room"]})
        official = pd.DataFrame({"rent_approval_date": ["2021-01", "2024-04", "2024-05"],
            "town": ["PUNGGOL", "CENTRAL AREA", "BEDOK"], "block": ["604A", "1", "2"],
            "street_name": ["PUNGGOL RD", "NEW MKT RD", "BEDOK NTH ST 3"],
            "flat_type": ["4-ROOM", "3-ROOM", "5-ROOM"]})
        pd.testing.assert_frame_equal(canonical_keys(local), canonical_keys(official, public=True))

    def test_street_and_month_remain_part_of_the_key(self):
        frame = pd.DataFrame({"RENT_APPROVAL_DATE": ["2021-01", "2021-02", "2021-01"],
            "TOWN": ["a"] * 3, "BLOCK": ["1"] * 3,
            "STREET": ["First Road", "First Road", "Second Road"], "FLAT_TYPE": ["4 room"] * 3})
        keys = canonical_keys(frame)
        self.assertEqual(len(keys.drop_duplicates()), 3)

    def test_normalization_preserves_source_rows_and_input(self):
        frame = pd.DataFrame({"RENT_APPROVAL_DATE": ["2021-01", "2021-01"],
            "TOWN": ["a", "a"], "BLOCK": ["1A", "1a"],
            "STREET": ["First Road", "first rd"], "FLAT_TYPE": ["4 room", "4-room"]}, index=[5, 2])
        original = frame.copy(deep=True)
        result = canonical_keys(frame)
        self.assertEqual(result.index.tolist(), [5, 2])
        self.assertEqual(len(result), 2)
        pd.testing.assert_frame_equal(frame, original)


if __name__ == "__main__":
    unittest.main()
