import unittest
import numpy as np
import pandas as pd
from analysis.experiments.historical_risk_correction import historical_risk_features, predict_correction, select_candidate


class HistoricalRiskCorrectionTests(unittest.TestCase):
    def frames(self):
        f = pd.DataFrame({'RENT_APPROVAL_DATE': ['2023-01', '2023-02', '2023-03'],
                          'TOWN': 'a', 'BLOCK': '1', 'STREET': 'road', 'FLAT_TYPE_CLEAN': '5-room',
                          'FLAT_MODEL_CLEAN': 'm', 'FLOOR_AREA_SQM': 110., 'LEASE_COMMENCE_DATE': 2002,
                          'FLAT_AGE_YEARS': 21., 'LATITUDE': 1.3, 'LONGITUDE': 103.8,
                          'MONTHLY_RENT': [1000., 2000., 3000.]})
        q = f.iloc[[0, 0]].copy(); q.index=[7, 8]; q.RENT_APPROVAL_DATE=['2023-04', '2023-06']
        return f, q

    def test_future_targets_cannot_change_features(self):
        f, q = self.frames()
        first = historical_risk_features(f, q, np.array([3000., 3000.]), np.array([20., 20.]), np.array([0., 0.]))
        q.MONTHLY_RENT = [1e9, -1e9]
        second = historical_risk_features(f, q, np.array([3000., 3000.]), np.array([20., 20.]), np.array([0., 0.]))
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first.SIGNATURE_ALL_count.iloc[0], np.log1p(3))
        self.assertEqual(first.SIGNATURE_ALL_mean_minus_prediction.iloc[0], 0)
        with self.assertRaises(ValueError):
            historical_risk_features(f, f, np.zeros(3), np.zeros(3), np.zeros(3))

    def test_same_observed_inputs_keep_same_correction(self):
        f, q = self.frames();q.RENT_APPROVAL_DATE='2023-04'
        features = historical_risk_features(f, q, np.array([3000., 3000.]), np.array([20., 20.]), np.array([0., 0.]))
        pd.testing.assert_series_equal(features.iloc[0], features.iloc[1], check_names=False)
        spec = {'kind':'history_blend','weight':.5,'cap':600}
        np.testing.assert_array_equal(predict_correction(spec, None, features), [0, 0])

    def test_selection_uses_declared_minimum_gain_and_identity_ties(self):
        self.assertEqual(select_candidate([{'name':'identity','rmse_sgd':500}, {'name':'ridge','rmse_sgd':499.5}]),'identity')
        self.assertEqual(select_candidate([{'name':'identity','rmse_sgd':500}, {'name':'ridge','rmse_sgd':495}]),'ridge')

    def test_correction_is_bounded(self):
        class Model:
            def predict(self, x):return np.array([-5000., 5000.])
        np.testing.assert_array_equal(predict_correction({'kind':'ridge','cap':600}, Model(), pd.DataFrame(index=[0,1])),[-600,600])

    def test_raw_history_ignores_query_targets_and_tail_policy_only_lowers(self):
        from analysis.experiments.recent_tail_correction import raw_history_delta, policy_correction
        f,q=self.frames();p=np.array([3000.,3000.])
        first=raw_history_delta(f,q,p);q.MONTHLY_RENT=999999.
        np.testing.assert_array_equal(first,raw_history_delta(f,q,p))
        features=pd.DataFrame({'SIGNATURE_12M_low_share':[.2,.2],'BLOCK_TYPE_12M_low_share':[0.,0.],
                               'SIGNATURE_12M_q10_minus_prediction':[-500.,500.],'BLOCK_TYPE_12M_q10_minus_prediction':[0.,0.]})
        np.testing.assert_array_equal(policy_correction({'kind':'low_tail','weight':1},features,first),[-100.,0.])


if __name__ == '__main__':
    unittest.main()
