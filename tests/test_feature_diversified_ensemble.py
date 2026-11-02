import unittest
import numpy as np
from analysis.experiments.feature_diversified_ensemble import optimize_weights,combine_predictions


class FeatureEnsembleTests(unittest.TestCase):
    def test_sequential_two_view_means_match_six_view_average(self):
        views=np.random.default_rng(42).normal(size=(100,6))
        grouped=np.column_stack([views[:,i:i+2].mean(axis=1) for i in [0,2,4]])
        np.testing.assert_allclose(combine_predictions(grouped,np.full(3,1/3)),views.mean(axis=1),atol=1e-15)

    def test_opposite_errors_cancel(self):
        y=np.array([1000.,2000.,3000.]); p=np.column_stack([y+10,y-10,y+100])
        w,info=optimize_weights(y,p)
        np.testing.assert_allclose(combine_predictions(p,w),y,atol=1e-8)
        self.assertAlmostEqual(w.sum(),1); self.assertLess(info['weighted_rmse_sgd'],1e-5)
        self.assertTrue(np.isfinite(info['weighted_rmse_sgd']))

    def test_all_three_members_can_be_optimal(self):
        p=np.eye(3)*30+1000; y=np.full(3,1010.)
        w,_=optimize_weights(y,p); np.testing.assert_allclose(w,np.full(3,1/3),atol=1e-8)

    def test_sample_weights_change_solution(self):
        y=np.zeros(2); p=np.array([[0.,10.],[10.,0.]])
        w,_=optimize_weights(y,p,[9,1]); np.testing.assert_allclose(w,[.9,.1],atol=1e-8)

    def test_baseline_vertex_and_identical_predictor_tie_are_exact(self):
        p=np.array([[1234.567890123456,1400.,1500.],[1500.,1600.,1700.]])
        w,_=optimize_weights(p[:,0],p); np.testing.assert_array_equal(w,[1.,0.,0.])
        np.testing.assert_array_equal(combine_predictions(p,w),p[:,0])
        identical=np.repeat(p[:,:1],3,axis=1); w,_=optimize_weights([0,0],identical)
        np.testing.assert_array_equal(w,[1.,0.,0.])

    def test_permuted_members_keep_optimal_prediction(self):
        rng=np.random.default_rng(42); p=rng.normal(size=(100,3)); y=rng.normal(size=100)
        w,_=optimize_weights(y,p); z,_=optimize_weights(y,p[:,[2,0,1]])
        np.testing.assert_allclose(combine_predictions(p,w),combine_predictions(p[:,[2,0,1]],z),atol=1e-10)

    def test_simplex_solution_beats_coarse_grid(self):
        rng=np.random.default_rng(5); y=rng.normal(size=40); p=rng.normal(size=(40,3)); w,info=optimize_weights(y,p)
        for a in np.linspace(0,1,11):
            for b in np.linspace(0,1-a,11):
                self.assertLessEqual(info['weighted_mse_sgd2'],np.mean((p@np.array([a,b,1-a-b])-y)**2)+1e-10)

    def test_invalid_inputs_are_rejected(self):
        p=np.ones((3,3))
        for weights in [[-.1,.1,1],[.2,.2,.2],[1,0],[np.nan,0,1]]:
            with self.assertRaises(ValueError): combine_predictions(p,weights)
        with self.assertRaises(ValueError): optimize_weights([1,2],p)
        with self.assertRaises(ValueError): optimize_weights([1,2,3],p,[1,0,1])
        p[0,0]=np.nan
        with self.assertRaises(ValueError): optimize_weights([1,2,3],p)


if __name__=='__main__': unittest.main()
