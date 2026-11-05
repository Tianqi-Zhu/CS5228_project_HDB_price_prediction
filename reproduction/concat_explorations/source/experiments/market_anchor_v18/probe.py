"""Evaluate fixed v18 anchors on frozen screen rows; no validation fitting."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.spatiotemporal_v17.contract import load_bank
from .contract import OUT,ROOT,load_anchor

DEST=ROOT/'reports/market_anchor_v18/probes'

def run():
    start=time.monotonic(); base=load_bank('F0',variant='causal_no_mrt'); reference=pd.read_csv(ROOT/'reports/spatiotemporal_v17/models/causal_no_mrt/F0/predictions.csv'); results={}; groups=[]
    for name in ('market_level','market_trend'):
        bank=load_anchor(name)
        for part in ('train','val'):
            assert np.array_equal(base[part]['row_id'],bank[part]['row_id'])
            assert np.all(base[part]['month']>base[part]['origin'])
        train,val=base['train'],base['val']; a=bank['train']['anchor_sgd'].astype(float); av=bank['val']['anchor_sgd'].astype(float); yt=train['y_sgd'].astype(float); y=val['y_sgd'].astype(float)
        mean=float(np.mean(yt-a)); ordered=pd.DataFrame({'row_id':val['row_id'],'y':y,'anchor':av,'month':val['month'],'horizon':val['horizon'],'town':val['town'],'flat_type':val['flat_type']})
        ordered=ordered.merge(reference[['row_id','pred']],on='row_id',validate='one_to_one').rename(columns={'pred':'F0'}); assert len(ordered)==5000
        for variant,offset in (('anchor',0.),('train_mean',mean)):
            pred=av+offset; e=pred-y; ref=ordered.F0.to_numpy()-y
            key=name+'_'+variant; results[key]={'train_rmse_sgd':float(np.sqrt(np.mean((a+offset-yt)**2))),'val_rmse_sgd':float(np.sqrt(np.mean(e**2))),'val_bias_sgd':float(e.mean()),'train_mean_residual_sgd':mean,'paired_mse_delta_vs_f0_sgd2':float(np.mean(e**2-ref**2))}
            output=ordered.copy(); output['prediction']=pred; output.to_csv(DEST/f'{key}_predictions.csv',index=False)
            for col in ('month','horizon','town','flat_type'):
                for v,frame in output.groupby(col,sort=True):
                    err=frame.prediction-frame.y; f0=frame.F0-frame.y
                    groups.append({'probe':key,'group':col,'value':str(v),'n':len(frame),'rmse_sgd':float(np.sqrt(np.mean(err**2))),'bias_sgd':float(err.mean()),'paired_mse_delta_vs_f0_sgd2':float(np.mean(err**2-f0**2))})
    pd.DataFrame(groups).to_csv(DEST/'group_metrics.csv',index=False)
    prior=ROOT/'reports/spatiotemporal_v17/anchor_probe/results.json'
    manifest={'models':results,'previous_v17_probe_sha256':hashlib.sha256(prior.read_bytes()).hexdigest(),'v17_anchor_prior_results':str(prior.relative_to(ROOT)),'protocol':'fixed origin-asof anchor and fit-only mean residual, frozen v17 screen; not unseen test','seconds':time.monotonic()-start}
    (DEST/'results.json').write_text(json.dumps(manifest,indent=2)); print(json.dumps(manifest,indent=2))
def cold_group_probe():
    """Supervised cold/horizon/group correction, not an as-of anchor feature."""
    b=load_bank('F0',variant='causal_no_mrt'); a=load_anchor('original'); t,v=b['train'],b['val']
    at=a['train']['anchor_sgd'].astype(float); av=a['val']['anchor_sgd'].astype(float)
    yt=t['y_sgd'].astype(float); y=v['y_sgd'].astype(float); hm=float(t['horizon'].mean())
    def design(p,anchor): return np.column_stack([np.ones(len(anchor)),p['horizon']-hm,(anchor==0).astype(float)])
    x=design(t,at); z=design(v,av); coeff=np.linalg.solve(x.T@x+np.diag([0.,1000.,1000.]),x.T@(yt-at))
    frame=pd.DataFrame({'town':t['town'],'flat':t['flat_type'],'resid':yt-at-x@coeff})
    grouped=frame.groupby(['town','flat']).resid.agg(['sum','count']); effects=(grouped['sum']/(grouped['count']+20)).to_dict()
    def values(pop): return np.array([effects.get((town,flat),0.) for town,flat in zip(pop['town'],pop['flat_type'])])
    train=at+x@coeff+values(t); pred=av+z@coeff+values(v)
    output=pd.DataFrame({'row_id':v['row_id'],'y':y,'prediction':pred,'horizon':v['horizon'],'month':v['month'],'town':v['town'],'flat_type':v['flat_type']})
    output.to_csv(DEST/'original_cold_group_horizon_predictions.csv',index=False)
    result={'scope':'supervised fitted correction evaluated directly; never used as historical anchor input','train_rmse_sgd':float(np.sqrt(np.mean((train-yt)**2))),'val_rmse_sgd':float(np.sqrt(np.mean((pred-y)**2))),'val_bias_sgd':float(np.mean(pred-y)),'coefficients':{'intercept':float(coeff[0]),'horizon_centered':float(coeff[1]),'cold_start':float(coeff[2]),'horizon_mean':hm},'penalties':{'horizon_ridge':1000,'cold_ridge':1000,'group_pseudocount':20},'group_effects':[{'town':town,'flat_type':flat,'effect_sgd':float(effect),'count':int(grouped.loc[(town,flat),'count'])} for (town,flat),effect in effects.items()]}
    (DEST/'original_cold_group_horizon.json').write_text(json.dumps(result,indent=2)); return result


if __name__=='__main__':
    run(); cold_group_probe()
