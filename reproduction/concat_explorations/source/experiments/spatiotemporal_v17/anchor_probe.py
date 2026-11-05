"""Preregistered train-only residual calibration probes, SGD units.

Horizon ridge penalty 1000 and group prior 20 are fixed before validation;
there is no validation tuning. The anchor remains the frozen causal anchor.
"""
import hashlib,json,os,time
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.fusion_contract_v2 import export
from .contract import load_bank,ROOT

HORIZON_RIDGE=1000.0
GROUP_PRIOR=20.0
OUT=ROOT/'reports/spatiotemporal_v17/anchor_probe'


def fit_calibrators(train):
    y=train['target'].to_numpy(dtype=float)-train['anchor'].to_numpy(dtype=float)
    h=train['horizon'].to_numpy(dtype=float); mean=float(y.mean()); hm=float(h.mean()); hc=h-hm
    slope=float(np.dot(hc,y-mean)/(np.dot(hc,hc)+HORIZON_RIDGE))
    residual=y-(mean+slope*hc)
    tmp=pd.DataFrame({'town':train['town'].astype(str).to_numpy(),'flat_type':train['flat_type'].astype(str).to_numpy(),'r':residual})
    group=tmp.groupby(['town','flat_type'],sort=True).r.agg(['sum','count'])
    effects={(str(t),str(f)):float(row['sum']/(row['count']+GROUP_PRIOR)) for (t,f),row in group.iterrows()}
    counts={(str(t),str(f)):int(row['count']) for (t,f),row in group.iterrows()}
    return {'mean_residual':mean,'horizon_mean':hm,'horizon_slope':slope,'group_effects':effects,'group_counts':counts,'ridge_penalty':HORIZON_RIDGE,'group_prior':GROUP_PRIOR,'n_train':len(train)}


def predict_calibrators(state,rows):
    anchor=rows['anchor'].to_numpy(dtype=float); h=rows['horizon'].to_numpy(dtype=float)
    linear=anchor+state['mean_residual']+state['horizon_slope']*(h-state['horizon_mean'])
    effects=np.array([state['group_effects'].get((str(t),str(f)),0.) for t,f in zip(rows['town'],rows['flat_type'])])
    return {'anchor':anchor.copy(),'train_mean':anchor+state['mean_residual'],'horizon':linear,'group_horizon':linear+effects}


def refreshed_anchor(sequence, mask, original_anchor, window):
    """Count-weighted rent in kSGD; shrink block to town with fixed prior 10."""
    seq=np.asarray(sequence,dtype=float); mask=np.asarray(mask,dtype=bool)
    if seq.ndim!=3 or seq.shape[1:]!=(12,4) or mask.shape!=seq.shape or window not in (6,12): raise ValueError('invalid sequence')
    def aggregate(j):
        count=np.where(mask[:,-window:,j],np.expm1(seq[:,-window:,j+1]),0.)
        n=count.sum(axis=1); total=(count*seq[:,-window:,j]).sum(axis=1)*1000
        return n,total
    tn,ts=aggregate(0); bn,bs=aggregate(2)
    original=np.asarray(original_anchor,dtype=float)
    town=np.divide(ts,tn,out=original.copy(),where=tn>0)
    return (bs+10*town)/(bn+10)


def _hash(x): return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def _metrics(y,p):
    e=p-y; return {'n':len(y),'rmse_sgd':float(np.sqrt(np.mean(e*e))),'bias_sgd':float(e.mean())}


def run():
    start=time.monotonic(); b=load_bank('F0',variant='causal_no_mrt'); src=export.load_source(); populations={}
    for part in ('train','val'):
        x=b[part]; raw=src.iloc[x['row_id']]
        assert np.array_equal(raw.SOURCE_ROW_ID.to_numpy(),x['row_id'])
        assert np.array_equal(raw.MONTH_INDEX.to_numpy(),x['month'])
        assert np.allclose(raw.MONTHLY_RENT.to_numpy(dtype=float),x['y_sgd'])
        assert np.array_equal(raw.TOWN_CLEAN.fillna('__missing__').astype(str).to_numpy(),x['town'])
        assert np.array_equal(raw.FLAT_TYPE_CLEAN.fillna('__missing__').astype(str).to_numpy(),x['flat_type'])
        assert np.array_equal(x['horizon'],x['month']-x['origin'])
        assert np.allclose(x['y_sgd'],(x['residual']+x['anchor_ksgd'])*1000,atol=.001)
        populations[part]=pd.DataFrame({'row_id':x['row_id'],'target':x['y_sgd'].astype(float),'anchor':x['anchor_ksgd'].astype(float)*1000,'horizon':x['horizon'],'origin':x['origin'],'month':x['month'],'town':x['town'],'flat_type':x['flat_type'],'floor_area_sqm':pd.to_numeric(raw.FLOOR_AREA_SQM,errors='coerce').to_numpy()})
    train,val=populations['train'],populations['val']; state=fit_calibrators(train); train_pred=predict_calibrators(state,train); val_pred=predict_calibrators(state,val)
    known=train.anchor.to_numpy()!=0
    known_mean=float((train.target.to_numpy()[known]-train.anchor.to_numpy()[known]).mean())
    train_pred['train_mean_known_anchor_only']=train.anchor.to_numpy()+known_mean
    val_pred['train_mean_known_anchor_only']=val.anchor.to_numpy()+known_mean
    refreshed_info={}
    for window in (6,12):
        key=f'refreshed_{window}m'; rtr=refreshed_anchor(b['train']['rent_sequence'],b['train']['sequence_mask'],train.anchor.to_numpy(),window)
        rv=refreshed_anchor(b['val']['rent_sequence'],b['val']['sequence_mask'],val.anchor.to_numpy(),window)
        correction=float((train.target.to_numpy()-rtr).mean())
        train_pred[key]=rtr; val_pred[key]=rv
        train_pred[key+'_train_mean']=rtr+correction; val_pred[key+'_train_mean']=rv+correction
        refreshed_info[key]={'mean_training_residual_sgd':correction,'window_months':window,'prior':10,'units':'monthly mean kSGD and log1p count; output SGD'}
        if window==12:
            known_correction=float((train.target.to_numpy()[known]-rtr[known]).mean())
            train_pred[key+'_known_anchor_mean']=rtr+known_correction; val_pred[key+'_known_anchor_mean']=rv+known_correction
            refreshed_info[key]['known_anchor_training_residual_mean_sgd']=known_correction

    f0=pd.read_csv(ROOT/'reports/spatiotemporal_v17/models/causal_no_mrt/F0/predictions.csv'); reference=val[['row_id','target']].merge(f0[['row_id','y','pred']],on='row_id',validate='one_to_one'); assert len(reference)==len(val) and np.allclose(reference.target,reference.y)
    OUT.mkdir(parents=True,exist_ok=True); metrics={}; group_rows=[]
    for name,pred in val_pred.items():
        y=val.target.to_numpy(); yt=train.target.to_numpy(); base=reference.pred.to_numpy(); err=pred-y
        result={'validation':_metrics(y,pred),'train':_metrics(yt,train_pred[name]),'paired_mse_delta_vs_f0_sgd2':float(np.mean(err**2-(base-y)**2))}
        metrics[name]=result
        output=val[['row_id','target','anchor','month','horizon','origin','town','flat_type']].copy(); output['prediction']=pred; output['f0_prediction']=base
        output.to_csv(OUT/f'{name}_predictions.csv',index=False)
        for col in ('month','horizon','town','flat_type'):
            for key,group in output.groupby(col,sort=True):
                e=group.prediction-group.target; ref=group.f0_prediction-group.target
                group_rows.append({'model':name,'group':col,'value':str(key),'n':len(group),'rmse_sgd':float(np.sqrt(np.mean(e**2))),'bias_sgd':float(e.mean()),'paired_mse_delta_vs_f0_sgd2':float(np.mean(e**2-ref**2))})
    pd.DataFrame(group_rows).to_csv(OUT/'group_metrics.csv',index=False)
    serial={k:v for k,v in state.items() if k not in ('group_effects','group_counts')}; serial['groups']=[{'town':t,'flat_type':f,'effect_sgd':effect,'count':state['group_counts'][(t,f)]} for (t,f),effect in state['group_effects'].items()]
    manifest={'protocol':'predeclared fixed training-only estimators; screen development validation; no 150k refit','source_bank_sha256':b['manifest']['array_sha256'],'train_row_ids_sha256':_hash(train.row_id.to_numpy()),'validation_row_ids_sha256':_hash(val.row_id.to_numpy()),'fit_targets_sha256':_hash(train.target.to_numpy()),'fit_anchor_sha256':_hash(train.anchor.to_numpy()),'horizon_ridge_penalty':HORIZON_RIDGE,'group_pseudocount':GROUP_PRIOR,'source_alignment':'row_id, month, target, town, flat_type and horizon checked; area retrieved but not used','f0_reference':'reports/spatiotemporal_v17/models/causal_no_mrt/F0/predictions.csv','metrics':metrics,'coefficients':serial,'known_anchor_only_mean_residual_sgd':known_mean,'known_anchor_training_rows':int(known.sum()),'cold_start_training_rows':int((~known).sum()),'refreshed_anchor_parameters':refreshed_info,'runtime_seconds':time.monotonic()-start,'caveats':'Comparative development selection only; frozen anchors include all legal past training history; no validation labels fit any calibration.'}
    (OUT/'results.json').write_text(json.dumps(manifest,indent=2)); print(json.dumps({'metrics':metrics,'mean_residual':state['mean_residual'],'horizon_slope':state['horizon_slope'],'runtime_seconds':manifest['runtime_seconds']},indent=2))

if __name__=='__main__': run()
