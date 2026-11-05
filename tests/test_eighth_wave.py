"""Chronological meta fitting and sparse-history ablation boundaries."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from analysis.experiments.eighth_wave_meta import fit,forecast,MetaFeatures,pre_origin_pool
from analysis.experiments.third_wave_training import inputs


class EighthWaveTests(unittest.TestCase):
    def test_sample_weight_rescaling_does_not_change_regularized_fit(self):
        x=pd.DataFrame({'PRED_BASE':np.linspace(2,4,30),'D_00':np.linspace(-1,1,30)**2})
        target=np.sin(np.arange(30))*.1+x.D_00*.3
        weights=np.linspace(1,3,30)
        spec={'family':'ridge','features':'predictions','alpha':10.}
        a=fit(spec,x,target,weights)
        b=fit(spec,x,target,weights/weights.sum())
        np.testing.assert_allclose(a.predict(x),b.predict(x),atol=1e-12)

    def test_meta_pool_excludes_all_query_month_labels(self):
        def item(months,values):
            return {'month':np.array(months),'x':pd.DataFrame({'v':values}),
                'actual':np.array(values,dtype=float),'baseline':np.zeros(len(values))}
        early=item(['2023-10','2024-03'],[1,2])
        annual=item(['2024-04','2024-05','2024-06','2025-03'],[3,4,999,999])
        x,y,w,months=pre_origin_pool(early,annual,'2024-06')
        self.assertEqual(x.v.tolist(),[1,2,3,4])
        self.assertTrue(all(m<'2024-06' for m in months))
        np.testing.assert_allclose(y,[.001,.002,.003,.004])
        self.assertAlmostEqual(w[:2].sum(),.5)
        self.assertAlmostEqual(w[2:].sum(),.5)

    def test_type_interactions_do_not_learn_query_categories(self):
        training=pd.DataFrame({'PRED_BASE':[3.],'D_00':[.1],
            'FLAT_TYPE_CLEAN':['A'],'SUPPORT24':[1.],'SUPPORT36':[0.]})
        transform=MetaFeatures('typed').fit(training)
        query=training.copy();query.FLAT_TYPE_CLEAN='unseen'
        result=transform.transform(query)
        self.assertEqual(transform.types_,['A'])
        self.assertEqual(result.D_00_TYPE0.iloc[0],0.)

    def test_ablation_preserves_targets_and_every_other_input(self):
        frame=pd.DataFrame({'AREA':[1.,2.],'ANN_BLOCK_36_R1_N':[1.,2.],
            'ANN_BLOCK_36_R1_RATIO':[.8,.9],'ANN_BLOCK_24_R1_N':[2.,3.]})
        query=frame.iloc[:1].copy()
        base=np.array([100.,200.]);offset=np.array([150.])
        bundle={'ann_fit':None,'ann_val':None,'extra_fit':None,'extra_val':None}
        with patch('analysis.experiments.third_wave_training.select_inputs',return_value=(frame.copy(),query.copy())), \
             patch('analysis.experiments.third_wave_training.adjusted_baseline',side_effect=[(base,{}),(offset,{})]):
            x,q,a,b=inputs(bundle,{'drop_ann36':True,'target':'residual'})
        self.assertEqual(list(x),['AREA','ANN_BLOCK_24_R1_N'])
        np.testing.assert_array_equal(a,base);np.testing.assert_array_equal(b,offset)
        pd.testing.assert_frame_equal(q,query[['AREA','ANN_BLOCK_24_R1_N']])


if __name__=='__main__':unittest.main()
