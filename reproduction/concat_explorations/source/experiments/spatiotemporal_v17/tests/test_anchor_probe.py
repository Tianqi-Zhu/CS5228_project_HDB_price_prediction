import numpy as np
import pandas as pd
from experiments.spatiotemporal_v17.anchor_probe import fit_calibrators, predict_calibrators


def test_validation_labels_never_fit_and_oov_falls_back():
    train=pd.DataFrame({'anchor':[1000.,1000.,1000.,1000.], 'target':[1100.,1200.,1300.,1400.], 'horizon':[1.,2.,3.,4.], 'town':['a']*4,'flat_type':['3']*4})
    val=pd.DataFrame({'anchor':[1000.,2000.], 'target':[10.,20.], 'horizon':[2.,3.], 'town':['never','a'], 'flat_type':['3','3']})
    state=fit_calibrators(train); p=predict_calibrators(state,val)
    val.target=[100000.,-100000.]; q=predict_calibrators(state,val)
    for name in p: np.testing.assert_allclose(p[name],q[name])
    assert p['group_horizon'][0]==p['horizon'][0]
    np.testing.assert_allclose(p['anchor'],val.anchor)


def test_horizon_fit_uses_training_only_and_reconstructs():
    train=pd.DataFrame({'anchor':[1000.,1100.,1200.], 'target':[1100.,1300.,1500.], 'horizon':[1.,2.,3.], 'town':['a']*3,'flat_type':['3']*3})
    state=fit_calibrators(train); pred=predict_calibrators(state,train)
    assert np.isfinite(np.concatenate(list(pred.values()))).all()
    np.testing.assert_allclose(pred['train_mean'],train.anchor+(train.target-train.anchor).mean())


def test_refreshed_anchor_shrinks_to_town_and_falls_back():
    from experiments.spatiotemporal_v17.anchor_probe import refreshed_anchor
    s=np.zeros((3,12,4),dtype=np.float32); m=np.zeros_like(s,dtype=bool)
    s[0,-1,:]=[3,np.log1p(4),4,np.log1p(2)]; m[0,-1,:]=True
    s[1,-1,:2]=[2,np.log1p(3)]; m[1,-1,:2]=True
    a=refreshed_anchor(s,m,np.array([1500.,1600.,1700.]),6)
    np.testing.assert_allclose(a,[1000*(2*4+10*3)/12,2000,1700],rtol=1e-6)


def test_fallback_history_ignores_future_labels():
    from experiments.spatiotemporal_v17.anchor_probe import refreshed_anchor
    from experiments.spatiotemporal_v17.features import temporal_features
    f=pd.DataFrame({'MONTH_INDEX':[0,1,2], 'MONTHLY_RENT':[1000.,2000.,3000.], 'FLOOR_AREA_SQM':[50.]*3, 'TOWN_CLEAN':['a']*3, 'FLAT_TYPE_CLEAN':['3']*3, 'BLOCK_ID':['b']*3})
    origin=np.array([-1,0,1]); _,seq,mask,_=temporal_features(f,origin,2)
    f.loc[2,'MONTHLY_RENT']=99999.; _,seq2,mask2,_=temporal_features(f,origin,2)
    np.testing.assert_array_equal(refreshed_anchor(seq,mask,np.zeros(3),12),refreshed_anchor(seq2,mask2,np.zeros(3),12))
