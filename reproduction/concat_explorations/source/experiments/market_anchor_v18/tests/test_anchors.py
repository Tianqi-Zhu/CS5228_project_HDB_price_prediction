import numpy as np
import pandas as pd
from experiments.market_anchor_v18.anchors import market_anchors


def data():
    return pd.DataFrame({'MONTH_INDEX':[0,1,2,3], 'MONTHLY_RENT':[1000.,2000.,3000.,4000.], 'TOWN_CLEAN':['a']*4, 'FLAT_TYPE_CLEAN':['3']*4})


def test_future_and_validation_labels_are_unavailable():
    f=data(); q=pd.DataFrame({'origin':[-1,1,1], 'horizon':[1,1,2], 'town':['a','a','z'],'flat_type':['3','3','unknown']})
    a=market_anchors(f,q); f.loc[f.MONTH_INDEX>1,'MONTHLY_RENT']=999999.; b=market_anchors(f,q)
    for name in a:
        for key in a[name]: np.testing.assert_array_equal(a[name][key],b[name][key])
    assert a['market_level']['anchor_sgd'][0]==0 and a['market_level']['missing'][0]
    assert np.isfinite(a['market_trend']['anchor_sgd']).all()


def test_hand_calculation_hierarchy_and_no_trend():
    f=pd.DataFrame({'MONTH_INDEX':[0,0,0], 'MONTHLY_RENT':[1000.,3000.,3000.], 'TOWN_CLEAN':['a','b','b'], 'FLAT_TYPE_CLEAN':['3']*3})
    q=pd.DataFrame({'origin':[0,0], 'horizon':[1,2], 'town':['a','new'], 'flat_type':['3','3']})
    a=market_anchors(f,q); x=a['market_level']
    expected=(1000+10*(7000/3))/11
    np.testing.assert_allclose(x['anchor_sgd'],[expected,7000/3])
    np.testing.assert_allclose(a['market_trend']['anchor_sgd'],x['anchor_sgd'])
    assert list(x['count'])==[1,0] and list(x['missing'])==[False,False]


def test_target_month_is_always_after_origin():
    f=data(); q=pd.DataFrame({'origin':[2], 'horizon':[0], 'town':['a'],'flat_type':['3']})
    import pytest
    with pytest.raises(ValueError): market_anchors(f,q)


def test_frozen_row_contract_and_reconstruction():
    from experiments.market_anchor_v18.contract import load_anchor
    from experiments.spatiotemporal_v17.contract import load_bank
    b=load_bank('F0',variant='causal_no_mrt')
    for name in ('original','refreshed_12m','market_level','market_trend'):
        a=load_anchor(name)
        for part in ('train','val'):
            np.testing.assert_array_equal(a[part]['row_id'],b[part]['row_id'])
            np.testing.assert_allclose(a[part]['anchor_sgd']+(b[part]['y_sgd']-a[part]['anchor_sgd']),b[part]['y_sgd'],atol=.001)
            assert np.isfinite(a[part]['anchor_sgd']).all()


def test_real_source_validation_label_perturbation():
    from experiments.fusion_contract_v2 import export
    from experiments.spatiotemporal_v17.contract import load_bank
    b=load_bank('F0',variant='causal_no_mrt')['val']; f=export.load_source()
    q=pd.DataFrame({'origin':b['origin'][:16], 'horizon':b['horizon'][:16], 'town':b['town'][:16], 'flat_type':b['flat_type'][:16]})
    a=market_anchors(f,q); f.loc[f.MONTH_INDEX>38,'MONTHLY_RENT']+=1000000
    changed=market_anchors(f,q)
    for name in a:
        for k in a[name]: np.testing.assert_array_equal(a[name][k],changed[name][k])
