"""Tests for leakage-sensitive preprocessing and unknown-category inference."""
import unittest

import numpy as np

from data_quality import clean_rows
from train_relu import features, prepare_estimator, prepare_training, predict


class NeuralTests(unittest.TestCase):
    def setUp(self):
        self.base = dict(RENT_APPROVAL_DATE='2024-01', TOWN='punggol', BLOCK='604a',
                         STREET='punggol road', FLAT_TYPE='4-room', FLAT_MODEL='model a',
                         FLOOR_AREA_SQM='90', FURNISHED='yes', LEASE_COMMENCE_DATE='2012',
                         FEE='0.0', MONTHLY_RENT='2500')

    def test_target_is_not_an_input(self):
        a = features([self.base])
        b = features([dict(self.base, MONTHLY_RENT='99999')])
        self.assertTrue(a.equals(b))
        self.assertNotIn('MONTHLY_RENT', a.columns)

    def test_future_lease_becomes_imputable_missing_feature(self):
        rows, _ = clean_rows([dict(self.base, LEASE_COMMENCE_DATE='2025')])
        frame = features(rows)
        self.assertTrue(np.isnan(frame.loc[0, 'lease_year']))
        self.assertTrue(np.isnan(frame.loc[0, 'lease_age']))
        self.assertEqual(frame.loc[0, 'lease_missing'], 1)

    def test_scaling_and_categories_are_fitted_on_training_only(self):
        rows = [dict(self.base, FLOOR_AREA_SQM='80', MONTHLY_RENT='2000'),
                dict(self.base, FLOOR_AREA_SQM='100', MONTHLY_RENT='3000')]
        bundle, x, y = prepare_estimator(rows)
        scaler = bundle['preprocessing'].named_transformers_['numeric'].named_steps['scale']
        imputer = bundle['preprocessing'].named_transformers_['numeric'].named_steps['impute']
        self.assertEqual(scaler.mean_[0], 90)
        self.assertEqual(imputer.statistics_[0], 90)
        self.assertEqual(bundle['target_mean'], 2500)
        self.assertEqual(bundle['target_std'], 500)
        bundle['estimator'].set_params(batch_size=2)
        bundle['estimator'].partial_fit(x, y)
        unseen = dict(self.base, TOWN='unseen town', STREET='new street', FLOOR_AREA_SQM='', MONTHLY_RENT='99999')
        result = predict(bundle, [unseen])
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(scaler.mean_[0], 90)
        self.assertEqual(bundle['estimator'].out_activation_, 'identity')

    def test_outliers_filtered_without_altering_input(self):
        rows = [dict(self.base, MONTHLY_RENT=str(v)) for v in (2000, 2100, 2200, 2300, 2400, 20000)]
        retained, report = prepare_training(rows)
        self.assertEqual(report['iqr_removed'], 1)
        self.assertEqual(len(retained), 5)
        self.assertEqual(rows[-1]['MONTHLY_RENT'], '20000')


if __name__ == '__main__':
    unittest.main()
