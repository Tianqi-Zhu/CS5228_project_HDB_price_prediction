"""Meaningful regression checks for cleaning decisions; no external packages."""
import copy
import unittest

from data_quality import baseline_eligible, clean_rows, iqr_bounds, remove_outliers


class CleaningTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(RENT_APPROVAL_DATE='2021-01', TOWN='punggol', BLOCK='604a',
                        STREET='punggol road', FLAT_TYPE='4-room', FLAT_MODEL='model a',
                        FLOOR_AREA_SQM='93.0', FURNISHED='yes', LEASE_COMMENCE_DATE='2012',
                        FEE='0.0', MONTHLY_RENT='1800')

    def test_normalization_and_idempotence(self):
        raw = dict(self.row, STREET='  Punggol  Road ', FLAT_TYPE='4 room', BLOCK='0604A')
        original = copy.deepcopy(raw)
        rows, _ = clean_rows([raw])
        self.assertEqual(rows[0]['STREET'], 'punggol road')
        self.assertEqual(rows[0]['FLAT_TYPE'], '4-room')
        self.assertEqual(rows[0]['BLOCK'], '0604a')
        self.assertEqual(clean_rows(rows)[0], rows)
        self.assertEqual(raw, original)

    def test_missing_markers_and_real_zero(self):
        for marker in ('', '  ', 'NA', ' NaN ', 'null', '?'):
            row = clean_rows([dict(self.row, MONTHLY_RENT=marker)])[0][0]
            self.assertEqual(row['MONTHLY_RENT'], '')
            self.assertFalse(baseline_eligible(row))
            self.assertEqual(row['FEE'], '0.0')

    def test_invalid_target_never_imputed(self):
        for value in ('inf', '-inf', 'abc', '-1', '0'):
            row = clean_rows([dict(self.row, MONTHLY_RENT=value)])[0][0]
            self.assertFalse(baseline_eligible(row))

    def test_impossible_dates(self):
        for date in ('2021-13', '2021-00', '2021-1', '0000-01'):
            row = clean_rows([dict(self.row, RENT_APPROVAL_DATE=date)])[0][0]
            self.assertEqual(row['RENT_APPROVAL_DATE'], '')
            self.assertFalse(baseline_eligible(row))

    def test_future_lease_masks_feature_without_dropping_label(self):
        rows, issues = clean_rows([dict(self.row, LEASE_COMMENCE_DATE='2022')])
        self.assertEqual(rows[0]['LEASE_COMMENCE_DATE'], '')
        self.assertEqual(rows[0]['MONTHLY_RENT'], '1800')
        self.assertTrue(baseline_eligible(rows[0]))
        self.assertEqual(issues[0]['csv_row'], 2)
        self.assertEqual(issues[0]['issue'], 'temporal_inconsistency')

    def test_cell_cleaning_precedes_row_filtering(self):
        raw = [self.row, dict(self.row), dict(self.row, MONTHLY_RENT='7600')]
        clean, _ = clean_rows(raw)
        self.assertEqual(clean, raw)

    def test_outlier_union_and_inclusive_boundaries(self):
        bounds = {'MONTHLY_RENT': (450, 4850), 'FLOOR_AREA_SQM': (5, 173)}
        keep = [dict(self.row, MONTHLY_RENT='450', FLOOR_AREA_SQM='5'),
                dict(self.row, MONTHLY_RENT='4850', FLOOR_AREA_SQM='173')]
        drop = [dict(self.row, MONTHLY_RENT='449'),
                dict(self.row, MONTHLY_RENT='4851'),
                dict(self.row, FLOOR_AREA_SQM='4'),
                dict(self.row, FLOOR_AREA_SQM='174'),
                dict(self.row, MONTHLY_RENT='7600', FLOOR_AREA_SQM='215')]
        self.assertEqual(remove_outliers(keep + drop, bounds), keep)

    def test_iqr_fitted_on_supplied_training_partition_only(self):
        rows = [dict(self.row, MONTHLY_RENT=str(v)) for v in (1000, 2000, 3000, 4000, 10000)]
        self.assertEqual(iqr_bounds(rows)['MONTHLY_RENT'], (-1000, 7000))
        self.assertEqual(len(remove_outliers(rows, iqr_bounds(rows))), 4)


if __name__ == '__main__':
    unittest.main()
