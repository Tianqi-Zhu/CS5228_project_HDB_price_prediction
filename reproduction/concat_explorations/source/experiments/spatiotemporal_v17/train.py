"""Auditable static-FT exploratory screen; no full-label refit."""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, resource, time
from pathlib import Path
import numpy as np
import torch
from experiments.feature_transformer_v4.models import FeatureTransformer
from experiments.spatiotemporal_v17.models import TemporalFT
from experiments.training_schedule_v15 import WarmupCosine
from experiments.concat_warmup_v15.pipeline import atomic_save, atomic_json, fingerprint, validate_resume

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/spatiotemporal_v17/models/causal_no_mrt'
BATCH=512

def validate_bank(bank, ntrain=30000, nval=5000):
    a,b=bank['train'],bank['val']
    if len(a['row_id'])!=ntrain or len(b['row_id'])!=nval: raise ValueError('unexpected split size')
    if np.intersect1d(a['row_id'],b['row_id']).size: raise ValueError('split overlap')
    for p in (a,b):
        n=len(p['row_id'])
        if len(np.unique(p['row_id']))!=n: raise ValueError('duplicate row IDs')
        if any(len(p[k])!=n for k in ('x_num','x_cat','y_sgd','residual','anchor_ksgd','origin','horizon','month','town','flat_type')): raise ValueError('row mismatch')
        if p['x_num'].shape[1]!=len(bank['numeric_names']) or p['x_cat'].shape[1]!=len(bank['cardinalities']): raise ValueError('feature shape')
        if not np.isfinite(p['x_num']).all() or not np.isfinite(p['y_sgd']).all(): raise ValueError('nonfinite values')
        if not np.allclose(p['y_sgd'],1000*(p['residual']+p['anchor_ksgd']),atol=.01): raise ValueError('target reconstruction')
        for i,c in enumerate(bank['cardinalities']):
            if p['x_cat'][:,i].min()<0 or p['x_cat'][:,i].max()>=c: raise ValueError('category range')
    if np.any(b['origin'] != 38): raise ValueError('validation origin')

def metrics(p,pred):
    err=(np.asarray(pred,dtype=np.float64)-p['y_sgd'])**2
    if len(err)!=len(p['row_id']) or not np.isfinite(err).all(): raise ValueError('incomplete predictions')
    result={'n':len(err),'rmse_sgd':float(np.sqrt(err.mean()))}
    for col in ('month','horizon','town','flat_type'):
        result['by_'+col]={str(v):{'n':int(np.sum(p[col]==v)),'rmse_sgd':float(np.sqrt(err[p[col]==v].mean()))} for v in np.unique(p[col])}
    if 'sequence_mask' in p:
        cold=~np.asarray(p['sequence_mask']).reshape(len(err),-1).any(axis=1)
        result['cold_start']={'n':int(cold.sum()),'rmse_sgd':float(np.sqrt(err[cold].mean())) if cold.any() else None}
        result['history_coverage_fraction']=float((~cold).mean())
    return result

def residual_stats(predicted,actual):
    predicted=np.asarray(predicted,dtype=np.float64)*1000; actual=np.asarray(actual,dtype=np.float64)*1000
    corr=float(np.corrcoef(predicted,actual)[0,1]) if np.std(predicted)>1e-10 and np.std(actual)>1e-10 else None
    return float(np.mean(predicted)),float(np.std(predicted)),corr

def batch(p,ids):
    result={'numeric':torch.as_tensor(p['x_num'][ids],dtype=torch.float32),'categorical':torch.as_tensor(p['x_cat'][ids],dtype=torch.long)}
    if 'rent_sequence' in p:
        result['rent_sequence']=torch.as_tensor(p['rent_sequence'][ids],dtype=torch.float32)
        result['sequence_mask']=torch.as_tensor(p['sequence_mask'][ids],dtype=torch.bool)
    return result

def predict(model,p):
    model.eval(); output=[]
    with torch.no_grad():
        for i in range(0,len(p['row_id']),BATCH): output.append(model(batch(p,slice(i,i+BATCH))).numpy())
    return np.concatenate(output)

def run(bank_id, max_epochs=6, expected=(30000,5000), output=None, architecture='static', lr=.0003, diagnostics=False):
    if architecture not in ('static','concat','cross'): raise ValueError('unknown architecture')
    from experiments.spatiotemporal_v17.contract import load_bank
    torch.set_num_threads(2); bank=load_bank(bank_id,split='orig',stage='screen',variant='causal_no_mrt'); validate_bank(bank,*expected)
    fit,val=bank['train'],bank['val']
    probe_ix=np.sort(np.random.default_rng(17042).choice(len(fit['row_id']),size=min(5000,len(fit['row_id'])),replace=False))
    probe={k:v[probe_ix] for k,v in fit.items()} if diagnostics else None
    path=Path(output or (OUT/bank_id if architecture=='static' else OUT/'temporal'/f'{bank_id}_{architecture}')); path.mkdir(parents=True,exist_ok=True)
    config=dict(bank=bank_id,architecture=architecture,variant='causal_no_mrt',seed=42,epochs=max_epochs,schedule_epochs=40,batch=512,lr=lr,diagnostics=diagnostics,train_probe_seed=17042,train_probe_rows=len(probe_ix) if diagnostics else 0,train_probe_row_id_sha256=fingerprint(fit['row_id'][probe_ix]) if diagnostics else None,wd=.0001,clip=1.,warmup_epochs=1,dropout=.05,macro=False,manifest=bank['manifest'],numeric_names=bank['numeric_names'],categorical_names=bank['categorical_names'],cardinalities=bank['cardinalities'])
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'experiments/feature_transformer_v4/models.py',ROOT/'experiments/training_schedule_v15.py',ROOT/'experiments/spatiotemporal_v17/models.py']}
    identity_fields=('row_id','x_num','x_cat','residual','y_sgd','anchor_ksgd','origin','horizon') + (('rent_sequence','sequence_mask') if architecture!='static' else ())
    identity=fingerprint(config,sources,*[p[k] for p in (fit,val) for k in identity_fields]); config.update(identity=identity,source_hashes=sources)
    if (path/'config.json').exists() and json.loads((path/'config.json').read_text()).get('identity') != identity: raise RuntimeError('existing config identity mismatch')
    atomic_json(config,path/'config.json'); random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if architecture=='static':
        model=FeatureTransformer(bank['numeric_names'],tuple(bank['cardinalities']),d_model=48,layers=3,heads=4,ff_multiplier=2,dropout=.05,pooling='cls')
    else:
        model=TemporalFT(bank['numeric_names'],bank['cardinalities'],len(bank['sequence_names']),mode=architecture,d_model=48,layers=3,heads=4,ff_multiplier=2,dropout=.05)
    config['parameter_count']=sum(p.numel() for p in model.parameters()); atomic_json(config,path/'config.json')
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.0001); steps=math.ceil(len(fit['row_id'])/BATCH)
    sched=WarmupCosine(opt,total_steps=steps*40,warmup_steps=steps,min_lr_ratio=.01,start_ratio=.01)
    epoch,offset,best,history=1,0,float('inf'),[]; grad_norms=[]; used_lrs=[]; progress=path/'last.pt'
    if progress.exists():
        s=torch.load(progress,map_location='cpu',weights_only=False)
        if s['identity']!=identity: raise RuntimeError('resume identity mismatch')
        if s.get('complete'): return
        model.load_state_dict(s['model']); opt.load_state_dict(s['optimizer']); sched.load_state_dict(s['scheduler']); torch.set_rng_state(s['torch_rng']); np.random.set_state(s['numpy_rng']); random.setstate(s['python_rng'])
        epoch,offset,best,history=(s[k] for k in ('epoch','offset','best','history')); grad_norms=s.get('grad_norms',[]); used_lrs=s.get('used_lrs',[]); validate_resume(epoch,offset,len(fit['row_id']),BATCH,sched.updates_completed)
    def save(complete=False):
        atomic_save(dict(identity=identity,model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),epoch=epoch,offset=offset,best=best,history=history,grad_norms=grad_norms,used_lrs=used_lrs,complete=complete),progress)
    while epoch<=max_epochs:
        start=time.monotonic(); model.train(); order=np.random.default_rng(42+epoch).permutation(len(fit['row_id']))
        while offset<len(order):
            ix=order[offset:offset+BATCH]; opt.zero_grad(set_to_none=True); loss=(model(batch(fit,ix))-torch.as_tensor(fit['residual'][ix],dtype=torch.float32)).square().mean()
            if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
            loss.backward(); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True); grad_norms.append(float(norm)); used_lrs.append(float(opt.param_groups[0]['lr'])); opt.step(); sched.step(); offset+=len(ix); save()
            if math.ceil(offset/BATCH)%10==0 or offset==len(order):
                info=dict(status='training',bank=bank_id,epoch=epoch,rows_seen=offset,loss=float(loss.detach()),lr=opt.param_groups[0]['lr']); atomic_json(info,path/'progress.json'); print(json.dumps(info),flush=True)
        residual=predict(model,val); pred=1000*(residual+val['anchor_ksgd']); m=metrics(val,pred); elapsed=time.monotonic()-start
        record=dict(epoch=epoch,rmse_sgd=m['rmse_sgd'],elapsed_seconds=elapsed)
        if diagnostics:
            train_residual=predict(model,probe); train_pred=1000*(train_residual+probe['anchor_ksgd']); train_err=train_pred-probe['y_sgd']; val_err=pred-val['y_sgd']
            record.update(train_eval_rmse_sgd=float(np.sqrt(np.mean(train_err.astype(np.float64)**2))),train_prediction_mean_sgd=float(np.mean(train_pred)),train_prediction_bias_sgd=float(np.mean(train_err)),train_prediction_std_sgd=float(np.std(train_pred)),val_prediction_mean_sgd=float(np.mean(pred)),val_prediction_bias_sgd=float(np.mean(val_err)),val_prediction_std_sgd=float(np.std(pred)),grad_clip_fraction=float(np.mean(np.asarray(grad_norms)>1.)),preclip_grad_norm_mean=float(np.mean(grad_norms)),preclip_grad_norm_max=float(np.max(grad_norms)),learning_rate_min=min(used_lrs),learning_rate_max=max(used_lrs),by_month=m['by_month'],by_horizon=m['by_horizon'])
            for label,rs,actual in (('train',train_residual,probe['residual']),('val',residual,val['residual'])):
                avg,sd,corr=residual_stats(rs,actual); record.update({label+'_residual_prediction_mean_sgd':avg,label+'_residual_prediction_std_sgd':sd,label+'_residual_correlation':corr})
            record['elapsed_seconds']=time.monotonic()-start
        history.append(record)
        if m['rmse_sgd']<best:
            best=m['rmse_sgd']; atomic_save(dict(identity=identity,epoch=epoch,model=model.state_dict(),residual=residual),path/'best.pt')
        print(json.dumps(history[-1]),flush=True); epoch+=1; offset=0; grad_norms=[]; used_lrs=[]; save()
    ck=torch.load(path/'best.pt',map_location='cpu',weights_only=False); model.load_state_dict(ck['model']); residual=predict(model,val)
    if not np.allclose(residual,ck['residual'],rtol=1e-6,atol=1e-6): raise RuntimeError('reload mismatch')
    pred=1000*(residual+val['anchor_ksgd']); m=metrics(val,pred)
    import csv
    tmp=path/'predictions.csv.tmp'
    with tmp.open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['row_id','y','pred','month','horizon','town','flat_type','origin'])
        w.writerows(zip(val['row_id'],val['y_sgd'],pred,val['month'],val['horizon'],val['town'],val['flat_type'],val['origin']))
    os.replace(tmp,path/'predictions.csv')
    with (path/'predictions.csv').open(newline='') as f:
        rows=list(csv.DictReader(f))
    saved_rmse=float(np.sqrt(np.mean([(float(r['y'])-float(r['pred']))**2 for r in rows])))
    if len(rows)!=len(val['row_id']) or not np.isclose(saved_rmse,m['rmse_sgd'],rtol=1e-7,atol=1e-4): raise RuntimeError('saved prediction metric mismatch')
    m.update(saved_prediction_rmse_sgd=saved_rmse,training_seconds=sum(x['elapsed_seconds'] for x in history),history=history,best_epoch=ck['epoch'],identity=identity,reload_verified=True,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,stage='exploratory screen'); atomic_json(m,path/'result.json'); save(True); atomic_json(dict(status='complete',bank=bank_id),path/'progress.json')

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('bank',choices=['F0','F1','F2','F3','F4','F5','F4_support','F3_no_block']); parser.add_argument('--architecture',choices=['static','concat','cross'],default='static'); parser.add_argument('--output',type=Path); parser.add_argument('--lr',type=float,default=.0003); parser.add_argument('--diagnostics',action='store_true'); args=parser.parse_args(); run(args.bank,architecture=args.architecture,output=args.output,lr=args.lr,diagnostics=args.diagnostics)
