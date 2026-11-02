import unittest
import numpy as np
import pandas as pd
from analysis.experiments.spatial_temporal_residual import (
    market_features, check_partition, corrected, choose_strength, residual_features, BASIC,
)


def history():
    rows=[]
    for i,month in enumerate(pd.period_range('2022-01','2022-12',freq='M')):
        for town in ['a','b']:
            for kind in ['3-room','4-room']:
                for j in range(10):
                    rows.append(dict(_MONTH=month,TOWN_CLEAN=town,FLAT_TYPE_CLEAN=kind,
                                     MONTHLY_RENT=1000+100*i+200*(town=='b')+300*(kind=='4-room')))
    return pd.DataFrame(rows)


class ResidualTests(unittest.TestCase):
    def test_fixed_batch_history_ignores_current_future_and_query_labels(self):
        p=history(); q=p.iloc[:2].copy(); q.index=[9001,9002]
        q['_MONTH']=pd.PeriodIndex(['2022-07','2022-10'],freq='M')
        q['TOWN_CLEAN']='a'; q['FLAT_TYPE_CLEAN']='3-room'
        first,_=market_features(p,q,'2022-07')
        p.loc[p._MONTH>=pd.Period('2022-07'), 'MONTHLY_RENT']=1e12
        q['MONTHLY_RENT']=-1e12
        second,_=market_features(p,q,'2022-07')
        pd.testing.assert_frame_equal(first,second)
        np.testing.assert_array_equal(first.LOCAL_COUNT_6M,[60,60])
        np.testing.assert_array_equal(first.LOCAL_HIST_MEDIAN,[1250,1250])
        self.assertAlmostEqual(first.LOCAL_GROWTH_3M.iloc[0],(1400-1100)/1100)
        pd.testing.assert_series_equal(first.iloc[0],first.iloc[1],check_names=False)

    def test_fallback_and_unobserved_group_zero_count(self):
        p=history(); q=p.iloc[:2].copy(); q['_MONTH']=pd.Period('2023-01')
        q.loc[q.index[0],'TOWN_CLEAN']='new town'
        q.loc[q.index[1],['TOWN_CLEAN','FLAT_TYPE_CLEAN']]=['new town','unseen type']
        result,audit=market_features(p,q,'2023-01')
        np.testing.assert_array_equal(audit.median_fallback_level,[1,2])
        np.testing.assert_array_equal(result.LOCAL_COUNT_6M,[0,0])
        self.assertTrue(np.isfinite(result.LOCAL_HIST_MEDIAN).all())

    def test_sparse_growth_remains_missing(self):
        p=history().iloc[:2]; q=p.iloc[:1].copy(); q['_MONTH']=pd.Period('2023-01')
        result,_=market_features(p,q,'2023-01')
        self.assertTrue(np.isnan(result.LOCAL_GROWTH_3M.iloc[0]))

    def test_query_context_overlap_duplicate_and_time_leak_rejected(self):
        with self.assertRaises(ValueError): check_partition(['2022-01'],['2022-02'],[1],[1],'2022-02')
        with self.assertRaises(ValueError): check_partition(['2022-02'],['2022-02'],[1],[2],'2022-02')
        with self.assertRaises(ValueError): check_partition(['2022-01'],['2022-02']*2,[1],[2,2],'2022-02')
        check_partition(['2022-01'],['2022-02','2022-03'],[1],[2,3],'2022-02')

    def test_lambda_zero_exact_and_sgd_scale(self):
        base=np.array([1234.567890123456,4000.]); delta=np.array([100.,-200.])
        np.testing.assert_array_equal(corrected(base,delta,0),base)
        np.testing.assert_array_equal(corrected(base,delta,.5),base+np.array([50.,-100.]))

    def test_strength_selection_uses_supplied_validation_only(self):
        actual=np.array([1000.,2000.]); base=actual+100
        strength,_=choose_strength(actual,base,np.array([-200.,-200.]))
        self.assertEqual(strength,.5)
        strength,_=choose_strength(actual,base,np.array([200.,200.]))
        self.assertEqual(strength,0)

    def test_schema_and_row_ids_preserved(self):
        x=pd.DataFrame({c:[1.,2.] for c in BASIC if c!='TABPFN_RENT'},index=[7,3])
        x['FLAT_TYPE_CLEAN']=pd.Categorical(['3-room','4-room']); x['TOWN_CLEAN']=['a','b']
        result=residual_features(x,np.array([1234.,2345.]))
        self.assertEqual(list(result),BASIC); self.assertEqual(result.index.tolist(),[7,3])
        np.testing.assert_array_equal(result.TABPFN_RENT,[1234.,2345.])
        m=pd.DataFrame(index=[3,7],columns=['LOCAL_HIST_MEDIAN','LOCAL_GROWTH_3M','LOCAL_COUNT_6M'])
        with self.assertRaises(ValueError): residual_features(x,[1234.,2345.],m)


if __name__=='__main__': unittest.main()
