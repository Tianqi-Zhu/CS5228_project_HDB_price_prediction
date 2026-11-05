import numpy as np
import pandas as pd
from experiments.spatiotemporal_v17.features import temporal_features, fit_categories


def frame():
    return pd.DataFrame({'MONTH_INDEX':[0,1,2,3], 'MONTHLY_RENT':[1000.,2000.,3000.,4000.], 'FLOOR_AREA_SQM':[50.]*4, 'TOWN_CLEAN':['a']*4, 'FLAT_TYPE_CLEAN':['3']*4, 'BLOCK_ID':['b']*4})


def test_history_is_causal_and_cold_start_finite():
    f=frame(); origins=np.array([-1,0,1,2]); x,seq,mask,names=temporal_features(f, origins, 2)
    changed=f.copy(); changed.loc[changed.MONTH_INDEX>1,'MONTHLY_RENT']=99999
    y,seq2,mask2,_=temporal_features(changed,origins,2)
    np.testing.assert_array_equal(x[:3],y[:3]); np.testing.assert_array_equal(seq[:3],seq2[:3])
    assert np.isfinite(x).all() and not mask[0].any()
    assert x[0,names.index('town_ft_3_missing')]==1


def test_categories_fit_only():
    a=pd.DataFrame({'k':['a','b','c']}); b=a.copy(); b.loc[2,'k']='z'
    x,v=fit_categories(a,[0,1],['k']); y,w=fit_categories(b,[0,1],['k'])
    assert v==w and x[2,0]==y[2,0]==0


def test_self_and_future_labels_never_enter_history():
    f=frame(); origin=f.MONTH_INDEX.to_numpy()-1
    x,seq,mask,_=temporal_features(f,origin,2)
    for i in range(len(f)):
        changed=f.copy(); changed.loc[i,'MONTHLY_RENT']+=100000
        z,s,_,_=temporal_features(changed,origin,2)
        np.testing.assert_array_equal(x[i],z[i]); np.testing.assert_array_equal(seq[i],s[i])


def test_saved_screen_contract():
    from experiments.spatiotemporal_v17.contract import load_bank
    b=load_bank('F1'); t,v=b['train'],b['val']
    assert t['x_num'].shape==(30000,56) and v['x_num'].shape==(5000,56)
    assert not np.intersect1d(t['row_id'],v['row_id']).size
    assert np.all(v['origin']==38) and np.all(t['month']>t['origin'])
    assert np.isfinite(t['x_num']).all() and np.isfinite(v['x_num']).all()
    np.testing.assert_allclose((v['residual']+v['anchor_ksgd'])*1000,v['y_sgd'],atol=.001)


def test_no_mrt_banks_are_separate_and_aligned():
    from experiments.spatiotemporal_v17.contract import load_bank
    old=load_bank('F1'); new=load_bank('F1',variant='causal_no_mrt')
    assert old['train']['x_num'].shape[1]==56 and new['train']['x_num'].shape[1]==52
    assert not any('MRT' in n for n in new['numeric_names'])
    np.testing.assert_array_equal(old['train']['row_id'],new['train']['row_id'])
    assert new['manifest']['variant']=='causal_no_mrt'


def test_full_build_refuses_existing_artifacts(tmp_path):
    from experiments.spatiotemporal_v17.full_builder import check_destination
    import pytest
    (tmp_path/'manifest.json').write_text('{}')
    with pytest.raises(FileExistsError): check_destination(tmp_path)


def test_support_trend_never_calls_missing_recent_a_crash():
    from experiments.spatiotemporal_v17.support_variants import support_trends
    seq=np.zeros((2,12,4),dtype=np.float32); mask=np.zeros_like(seq,dtype=bool)
    seq[0,0,0]=3; seq[0,0,1]=np.log1p(2); mask[0,0,:2]=True
    seq[0,0,2]=4; seq[0,0,3]=np.log1p(2); mask[0,0,2:]=True
    trend, flags=support_trends(seq,mask)
    assert np.isfinite(trend).all() and np.isfinite(flags).all()
    assert trend[0,0]==0 and trend[0,1]==0
    assert flags[0,0]==1 and flags[0,2]==1
    assert trend[1,0]==trend[1,1]==0 and flags[1,1]==flags[1,3]==1


def test_support_trend_respects_future_boundary():
    from experiments.spatiotemporal_v17.support_variants import support_trends
    f=frame(); origins=np.array([-1,0,1,2]); _,s,m,_=temporal_features(f,origins,2)
    f.loc[f.MONTH_INDEX>1,'MONTHLY_RENT']=99999
    _,s2,m2,_=temporal_features(f,origins,2)
    a,b=support_trends(s,m); c,d=support_trends(s2,m2)
    np.testing.assert_array_equal(a[:3],c[:3]); np.testing.assert_array_equal(b[:3],d[:3])
