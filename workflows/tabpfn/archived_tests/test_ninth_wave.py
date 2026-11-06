"""Blend constraints, temporal fitting pools and group fallback checks."""
import unittest
import numpy as np
import pandas as pd
from analysis.experiments.ninth_wave_blending import solve,fit,predict,fitting_pool


class NinthWaveTests(unittest.TestCase):
    def item(self):
        n=1200
        x=np.linspace(1000,4000,n)
        return {'matrix':np.column_stack([x-100,x+100]),'actual':x,
            'names':['low','high'],'prior':np.array([.5,.5]),'row':np.arange(n),
            'month':np.where(np.arange(n)<600,'2023-10','2023-11'),
            'context':pd.DataFrame({'support24':['yes']*n,'type':['known']*n,
                'disagreement':np.linspace(0,100,n)})}

    def test_convex_constraints_and_weight_scaling(self):
        item=self.item();spec={'objective':'mse','ridge':.003}
        a=solve(item['matrix'],item['actual'],item['month'],np.ones(1200),item['prior'],spec)
        b=solve(item['matrix'],item['actual'],item['month'],np.ones(1200)/1200,item['prior'],spec)
        np.testing.assert_allclose(a,b,atol=1e-10)
        self.assertAlmostEqual(a.sum(),1.)
        self.assertGreaterEqual(a.min(),0.);self.assertLessEqual(a.max(),.6)

    def test_month_centered_objective_ignores_common_month_level_error(self):
        item=self.item();spec={'objective':'month_centered','ridge':.003}
        weights=np.ones(1200)
        a=solve(item['matrix'],item['actual'],item['month'],weights,item['prior'],spec)
        changed=item['actual']+np.where(item['month']=='2023-10',1000,-500)
        b=solve(item['matrix'],changed,item['month'],weights,item['prior'],spec)
        np.testing.assert_allclose(a,b,atol=1e-9)

    def test_group_fallback_and_predictions_do_not_read_query_labels(self):
        item=self.item();spec={'objective':'mse','ridge':.003,'group_ridge':.03,'gate':'type'}
        model=fit(item,spec)
        query=item.copy();query['context']=item['context'].copy();query['context']['type']='unseen'
        a=predict(model,query)
        np.testing.assert_allclose(a,item['matrix']@model['global_weights'])
        query['actual']=np.full(1200,1.e10);query['month']=np.full(1200,'2099-01')
        np.testing.assert_array_equal(a,predict(model,query))

    def test_disagreement_threshold_is_learned_on_fitting_context(self):
        item=self.item();spec={'objective':'mse','ridge':.003,'group_ridge':.03,'gate':'disagreement'}
        model=fit(item,spec)
        self.assertAlmostEqual(model['threshold'],50.)
        self.assertEqual(set(model['group_weights']),{'low','high'})

    def test_pre_origin_pool_filters_later_labels_and_balances_origins(self):
        first=self.item();annual=self.item()
        annual['month']=np.where(np.arange(1200)<600,'2024-05','2024-06')
        annual['actual']=np.where(np.arange(1200)<600,3000,1.e10)
        pooled=fitting_pool(first,annual,'2024-06')
        self.assertEqual(len(pooled['actual']),1800)
        self.assertLess(max(pooled['month']),'2024-06')
        self.assertLess(pooled['actual'].max(),1.e10)
        self.assertAlmostEqual(pooled['weights'][:1200].sum(),.5)
        self.assertAlmostEqual(pooled['weights'][1200:].sum(),.5)
        self.assertEqual(len(fitting_pool(first,annual,'2024-06',False)['actual']),1200)


if __name__=='__main__':unittest.main()
