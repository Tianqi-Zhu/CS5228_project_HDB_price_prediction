"""Matched ridge and LightGBM controls for causal anchors."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import numpy as np
from experiments.market_anchor_v18.residual import build_inputs,reconstruct
LGB_PARAMS={'n_estimators':300,'learning_rate':.05,'num_leaves':31,'min_child_samples':30,'reg_lambda':3.,'n_jobs':2,'random_state':42,'verbosity':-1}
RIDGE_PARAMS={'alpha':100.0,'fit_intercept':True}

def fit_predict(bank,anchors,learner,mode):
    if mode not in ('direct','residual'): raise ValueError('invalid mode')
    x,y,state=build_inputs(bank,anchors)
    if learner=='ridge':
        from sklearn.linear_model import Ridge
        model=Ridge(**RIDGE_PARAMS)
    elif learner=='lightgbm':
        from lightgbm import LGBMRegressor
        model=LGBMRegressor(**LGB_PARAMS)
    else: raise ValueError('invalid learner')
    model.fit(x['train'],y['train'][mode])
    out={'model':model,'preprocessing':state,'model_parameters':RIDGE_PARAMS if learner=='ridge' else LGB_PARAMS}
    for split in ('train','val'):
        pred=model.predict(x[split]); out[split+'_model_output']=pred;out[split+'_prediction']=reconstruct(pred,anchors[split]['anchor_sgd'],mode)
        if not np.isfinite(out[split+'_prediction']).all(): raise RuntimeError('nonfinite model predictions')
    return out

def run(bank,anchors,learner,mode,path):
    import csv,hashlib,json,os,pickle,time
    from pathlib import Path
    from experiments.market_anchor_v18.evaluate import evaluate
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    h=hashlib.sha256()
    sources={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (Path(__file__),Path(__file__).with_name('residual.py'),Path(__file__).with_name('evaluate.py'))}
    for f in sources: h.update(Path(f).read_bytes())
    h.update(json.dumps([learner,mode],sort_keys=True).encode())
    for split in ('train','val'):
        for group in (bank[split],anchors[split]):
            for k,v in sorted(group.items()):
                if isinstance(v,np.ndarray): h.update(k.encode()); h.update(str(v.dtype).encode()); h.update(str(v.shape).encode()); h.update(np.ascontiguousarray(v).tobytes())
    identity=h.hexdigest(); config_path=path/'config.json'
    if config_path.exists() and json.loads(config_path.read_text())['identity']!=identity: raise RuntimeError('run identity mismatch')
    if (path/'result.json').exists():
        existing=json.loads((path/'result.json').read_text())
        if existing.get('complete') and existing.get('identity')==identity: return existing
        raise RuntimeError('existing result incomplete or mismatched')
    def write_json(dst,value):
        tmp=dst.with_suffix(dst.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');os.replace(tmp,dst)
    write_json(config_path,{'identity':identity,'learner':learner,'mode':mode,'parameters':RIDGE_PARAMS if learner=='ridge' else LGB_PARAMS,'split':'v17 frozen orig/screen; development selection','seed':42,'input_manifest':bank.get('manifest',{}),'anchor_manifest':anchors.get('manifest',{}),'source_hashes':sources})
    start=time.monotonic(); fitted=fit_predict(bank,anchors,learner,mode); elapsed=time.monotonic()-start
    model_path=path/'model.pkl'; temp=model_path.with_suffix('.pkl.tmp')
    with temp.open('wb') as f:pickle.dump({'model':fitted['model'],'preprocessing':fitted['preprocessing'],'identity':identity},f)
    os.replace(temp,model_path)
    with model_path.open('rb') as f: loaded=pickle.load(f)
    x,_,_=build_inputs(bank,anchors)
    verified=reconstruct(loaded['model'].predict(x['val']),anchors['val']['anchor_sgd'],mode)
    if not np.allclose(verified,fitted['val_prediction'],rtol=1e-9,atol=1e-6): raise RuntimeError('model reload mismatch')
    for split in ('train','val'):
        p=bank[split];a=anchors[split];pred=fitted[split+'_prediction']; tmp=path/(split+'_predictions.csv.tmp')
        with tmp.open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['row_id','y','pred','anchor_sgd','premium_sgd','month','horizon','town','flat_type','count','level','missing'])
            w.writerows(zip(p['row_id'],p['y_sgd'],pred,a['anchor_sgd'],pred-a['anchor_sgd'],p['month'],p['horizon'],p['town'],p['flat_type'],a['count'],a['level'],a['missing']))
        os.replace(tmp,path/('predictions.csv' if split=='val' else 'train_predictions.csv'))
    correction=float(np.mean(bank['train']['y_sgd']-anchors['train']['anchor_sgd']))
    val=bank['val'];anchor=anchors['val'];result={'complete':True,'reload_verified':True,'identity':identity,'elapsed_seconds':elapsed,'train':evaluate(bank['train'],anchors['train'],fitted['train_prediction']),'val':evaluate(val,anchor,fitted['val_prediction'],baseline=anchor['anchor_sgd']+correction),'anchor_only':evaluate(val,anchor,anchor['anchor_sgd']),'anchor_plus_train_mean':evaluate(val,anchor,anchor['anchor_sgd']+correction),'train_mean_correction_sgd':correction,'preprocessing':fitted['preprocessing']}
    baseline_path=Path(__file__).resolve().parents[2]/'reports/spatiotemporal_v17/models/causal_no_mrt/F0/predictions.csv'
    if len(val['row_id'])==5000 and baseline_path.exists():
        from experiments.market_anchor_v18.evaluate import paired
        with baseline_path.open(newline='') as f: baseline=list(csv.DictReader(f))
        candidate={'row_id':np.asarray(val['row_id']).astype(str),'y':np.asarray(val['y_sgd'],dtype=float),'pred':np.asarray(fitted['val_prediction'])}
        ref={'row_id':np.array([r['row_id'] for r in baseline]),'y':np.array([float(r['y']) for r in baseline]),'pred':np.array([float(r['pred']) for r in baseline])}
        delta=paired(candidate,ref); by_month={}
        for month in np.unique(val['month']):
            ix=val['month']==month;by_month[str(month)]=float(np.mean((candidate['pred'][ix]-candidate['y'][ix])**2-(ref['pred'][ix]-ref['y'][ix])**2))
        result['paired_vs_v17_static_f0']={'mean_squared_error_delta':delta,'by_month':by_month,'reference':'v17 F0 static 3e-4, best checkpoint; exploratory screen'}
    write_json(path/'result.json',result);return result
