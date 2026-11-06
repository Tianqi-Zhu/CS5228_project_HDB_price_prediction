import unittest
import numpy as np,pandas as pd
from analysis.experiments.auxiliary_ablation import BASE,BLOCK,AMENITIES,CATEGORICAL
from analysis.experiments.embedding_rent_search import prepare


class RawTimeFeatureTests(unittest.TestCase):
    def test_future_horizon_remains_distinguishable(self):
        data={c:['a']*20 if c in CATEGORICAL else np.ones(20) for c in BASE+BLOCK+AMENITIES}
        data['STREET_ID']=['s']*20;data['BUILDING_ID']=['b']*20
        f=pd.DataFrame(data);f.APPROVAL_MONTH_INDEX=np.arange(20)
        v=f.iloc[:2].copy();v.APPROVAL_MONTH_INDEX=[20,25]
        (_,n,_,nv),meta=prepare(f,v,False,True,True)
        self.assertGreater(float(nv[1,-1]),float(nv[0,-1]))
        self.assertIsNotNone(meta['raw_time_scaler'])
        # Changing future input values must not change fitting encodings.
        v.APPROVAL_MONTH_INDEX=[200,250]
        (_,other,_,_),_=prepare(f,v,False,True,True)
        np.testing.assert_array_equal(n,other)


if __name__=='__main__':unittest.main()
