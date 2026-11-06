import unittest
import numpy as np
import pandas as pd
from scipy import sparse
from analysis.experiments.hierarchical_latent_residual import HierarchicalResidual,DynamicHierarchicalResidual,latest_available_residuals,apply_correction


def cases():
    return pd.DataFrame(dict(row_id=range(8),month=['2023-01']*4+['2023-02']*4,
        origin=['2023-01']*8,building=['a','a','b','c']*2,submarket=['x','x','x','y']*2,
        residual=[100.,120.,80.,-200.,150.,170.,90.,-180.],weight=[1.,2.,1.,1.]*2))


class HierarchicalTests(unittest.TestCase):
    def test_dynamic_map_matches_small_dense_reference(self):
        data=cases(); model=DynamicHierarchicalResidual(5,8,.5,10).fit(data,'2023-03')
        x,r,w,_=model._system(data,'2023-03')
        a=np.vstack([x.toarray()*np.sqrt(w)[:,None],r.toarray()])
        y=np.r_[data.residual.to_numpy()*np.sqrt(w),np.zeros(r.shape[0])]
        expected=np.linalg.lstsq(a,y,rcond=None)[0]
        np.testing.assert_allclose(model.coef_,expected,atol=1e-6,rtol=0)
        self.assertTrue(sparse.issparse(r))

    def test_dynamic_forecasts_decay_by_actual_calendar_horizon(self):
        model=DynamicHierarchicalResidual(100,100,.5,2).fit(cases(),'2023-03')
        query=pd.DataFrame(dict(month=['2023-03','2023-04','2024-02'],building=['a']*3,submarket=['x']*3,actual=[1.,2.,3.]))
        before=model.states_.copy(); c=model.predict_components(query)
        self.assertNotEqual(c.dynamic_effect.iloc[0],0.)
        self.assertAlmostEqual(c.dynamic_effect.iloc[1]/c.dynamic_effect.iloc[0],.5)
        self.assertAlmostEqual(c.dynamic_effect.iloc[2]/c.dynamic_effect.iloc[0],.5**11)
        query.actual=1e12
        pd.testing.assert_frame_equal(c,model.predict_components(query))
        np.testing.assert_array_equal(before,model.states_)

    def test_missing_calendar_month_has_its_own_prior_state(self):
        data=cases(); data.loc[data.month=='2023-02','month']='2023-03'
        model=DynamicHierarchicalResidual().fit(data,'2023-04')
        self.assertEqual(model.states_.shape[1],3)
        self.assertEqual(model.state_last_month_,'2023-03')
        with self.assertRaises(ValueError): model.predict_components(pd.DataFrame(dict(month=['2023-03'],building=['a'],submarket=['x'])))

    def test_zero_persistence_and_unseen_group_dynamic_fallback(self):
        model=DynamicHierarchicalResidual(rho=0).fit(cases(),'2023-03')
        q=pd.DataFrame(dict(month=['2023-03','2023-10'],building=['a','new'],submarket=['x','unseen']))
        np.testing.assert_array_equal(model.predict_components(q).dynamic_effect,[0.,0.])
        with self.assertRaises(ValueError): DynamicHierarchicalResidual(rho=1.)

    def test_sparse_map_matches_small_dense_reference(self):
        data=cases(); model=HierarchicalResidual(5,8).fit(data,'2023-03')
        x,r,w,_=model._system(data,'2023-03')
        a=np.vstack([x.toarray()*np.sqrt(w)[:,None],r.toarray()])
        y=np.r_[data.residual.to_numpy()*np.sqrt(w),np.zeros(r.shape[0])]
        expected=np.linalg.lstsq(a,y,rcond=None)[0]
        np.testing.assert_allclose(model.coef_,expected,atol=1e-6,rtol=0)
        self.assertTrue(sparse.issparse(x)); self.assertLess(x.nnz,len(data)*4)
        self.assertAlmostEqual(model.coef_[1:1+model.n_buildings_].sum(),0.,places=5)

    def test_unseen_building_and_submarket_fallback(self):
        model=HierarchicalResidual(5,1).fit(cases(),'2023-03')
        q=pd.DataFrame(dict(month=['2023-03']*3,building=['new','a','new'],submarket=['x','unseen','unseen']))
        c=model.predict_components(q)
        self.assertEqual(c.building_effect.iloc[0],0.)
        self.assertNotEqual(c.submarket_effect.iloc[0],0.)
        np.testing.assert_array_equal(c.total_correction.iloc[1:],[model.coef_[0]]*2)

    def test_stronger_priors_shrink_building_effects(self):
        weak=HierarchicalResidual(1,1000).fit(cases(),'2023-03')
        strong=HierarchicalResidual(1000,1000).fit(cases(),'2023-03')
        self.assertLess(np.linalg.norm(strong.coef_[1:4]),np.linalg.norm(weak.coef_[1:4]))

    def test_fitting_rejects_future_targets_and_duplicate_case_ids(self):
        with self.assertRaises(ValueError): HierarchicalResidual().fit(cases(),'2023-02')
        data=cases(); data.loc[1,'row_id']=0
        with self.assertRaises(ValueError): HierarchicalResidual().fit(data,'2023-03')

    def test_latest_eligible_forecast_does_not_use_future_labels(self):
        data=cases(); newer=data.iloc[[4]].copy(); newer['origin']='2023-02'
        data=pd.concat([data,newer],ignore_index=True)
        past=latest_available_residuals(data,'2023-02')
        self.assertEqual(len(past),4); self.assertTrue((past.month<'2023-02').all())
        all_data=latest_available_residuals(data,'2023-03')
        self.assertEqual(len(all_data),8); self.assertEqual(all_data.loc[all_data.row_id==4,'origin'].iloc[0],'2023-02')
        self.assertAlmostEqual(all_data.fit_weight.sum(),8.)

    def test_zero_correction_and_query_label_mutation(self):
        model=HierarchicalResidual().fit(cases(),'2023-03')
        q=pd.DataFrame(dict(month=['2023-03'],building=['a'],submarket=['x'],actual=[1000.]))
        first=model.predict_components(q); q.actual=-1e12
        pd.testing.assert_frame_equal(first,model.predict_components(q))
        baseline=np.array([1234.567890123456])
        np.testing.assert_array_equal(apply_correction(baseline,first.total_correction.to_numpy(),0),baseline)


if __name__=='__main__': unittest.main()
