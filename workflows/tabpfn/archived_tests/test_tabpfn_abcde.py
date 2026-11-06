"""Five-member nesting, weighted solutions and selection chronology checks."""
import unittest
import numpy as np
import pandas as pd
from analysis.experiments.temporal_feature_views import simplex_weights
from analysis.experiments.feature_diversified_ensemble import combine_predictions


class ABCDETests(unittest.TestCase):
    def test_five_member_calibration_optimum_cannot_lose_to_three_member_subsets(self):
        rng=np.random.default_rng(41);y=rng.normal(size=80);p=rng.normal(size=(80,5));v=rng.uniform(.1,2,size=80)
        w,info=simplex_weights(y,p,v)
        for subset in [[2,3,4],[1,3,4]]:
            z,small=simplex_weights(y,p[:,subset],v)
            self.assertLessEqual(info['weighted_mse_sgd2'],small['weighted_mse_sgd2']+1e-10)
        self.assertAlmostEqual(w.sum(),1.)
        self.assertTrue((w>=0).all())

    def test_extra_identical_models_do_not_improve_predictions(self):
        y=np.arange(12.);p=np.repeat((y+2)[:,None],5,axis=1)
        w,_=simplex_weights(y,p)
        np.testing.assert_array_equal(w,[1,0,0,0,0])
        np.testing.assert_array_equal(combine_predictions(p,w),p[:,0])

    def test_all_five_members_can_be_required(self):
        p=np.eye(5)*50;y=np.full(5,10.)
        w,info=simplex_weights(y,p)
        np.testing.assert_allclose(w,np.full(5,.2),atol=1e-10)
        self.assertEqual(info['active_members'],5)

    def test_future_labels_do_not_change_weights_fit_on_past_rows(self):
        rng=np.random.default_rng(4);p=rng.normal(size=(30,5));y=rng.normal(size=30)
        months=pd.period_range('2022-01',periods=30,freq='M')
        earlier=months<pd.Period('2024-01','M')
        w,_=simplex_weights(y[earlier],p[earlier])
        y[~earlier]*=99999;p[~earlier]*=99999
        z,_=simplex_weights(y[earlier],p[earlier])
        np.testing.assert_array_equal(w,z)


if __name__=='__main__':unittest.main()
