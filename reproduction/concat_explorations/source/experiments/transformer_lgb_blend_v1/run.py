"""Fixed orig/A FT and LightGBM blend; no validation fitting."""
import hashlib, json, os, time, traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

OUT=Path('reports/transformer_lgb_blend_v1')
BASE=Path('reports/diagnostics_phase2_c/frozen')
FT=Path('reports/feature_transformer_v4/runs/v4-search-mps-s42/trial6/inner_predictions.npz')
SUMMARY=FT.parent/'summary.json'
CONFIG=dict(n_estimators=300,learning_rate=.05,num_leaves=31,min_child_samples=30,reg_lambda=3,n_jobs=2,random_state=42,verbosity=-1)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()
def dump(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def status(state,**kw): dump('STATUS.json',dict(status=state,provider='Terra',session='terra_ft_lgb_blend',pid=os.getpid(),updated_at=datetime.now(timezone.utc).isoformat(),**kw))
def align(hold,ft):
    ids=hold['row_id']; other=ft['row_id']
    if len(np.unique(ids)) != len(ids) or len(np.unique(other)) != len(other): raise ValueError('duplicate row ID')
    order=np.argsort(other); idx=np.searchsorted(other[order],ids)
    if len(ids)!=len(other) or np.any(idx>=len(other)) or not np.array_equal(other[order][idx],ids): raise ValueError('ID set mismatch')
    ix=order[idx]
    for key in ('actual_sgd','horizon'):
        if not np.array_equal(hold[key],ft[key][ix]): raise ValueError(key+' mismatch')
    return ix
def to_sgd(residual,anchor): return (np.asarray(residual,dtype=np.float64)+anchor)*1000

def score(y,p):
    e=p-y
    return dict(n=len(e),rmse_sgd=float(np.sqrt(np.mean(e**2))),mae_sgd=float(np.mean(np.abs(e))),bias_sgd=float(np.mean(e)))
def all_metrics(df):
    result={}
    for name in ('ft','lgb','blend'):
        p=df[name+'_prediction'].to_numpy(); y=df.actual_sgd.to_numpy(); h=df.horizon.to_numpy()
        result[name]={'overall':score(y,p),'horizon':{f'{lo}-{lo+2}':score(y[(h>=lo)&(h<=lo+2)],p[(h>=lo)&(h<=lo+2)]) for lo in (1,4,7,10)}}
    result['residual_correlation']=float(np.corrcoef(df.ft_prediction-df.actual_sgd,df.lgb_prediction-df.actual_sgd)[0,1])
    result['blend_minus_ft_rmse_sgd']=result['blend']['overall']['rmse_sgd']-result['ft']['overall']['rmse_sgd']
    result['blend_minus_lgb_rmse_sgd']=result['blend']['overall']['rmse_sgd']-result['lgb']['overall']['rmse_sgd']
    return result

def main():
    status('RUNNING',started_at=datetime.now(timezone.utc).isoformat())
    inputs=[BASE/'afit.npz',BASE/'ahold.npz',BASE/'afit_columns.json',BASE/'ahold_columns.json',FT,SUMMARY,Path('experiments/diagnostics_phase2_c/protocol.py'),Path('docs/experiments/PROTOCOL_v7.md'),Path(__file__)]
    cols=json.loads((BASE/'afit_columns.json').read_text()); assert cols==json.loads((BASE/'ahold_columns.json').read_text())
    assert len(cols['numeric_names'])==56 and len(cols['categorical_names'])==4
    manifest=dict(registered_at=datetime.now(timezone.utc).isoformat(),hypothesis='Fixed equal blend may reduce orig/A RMSE through complementary prediction errors.',baseline='Existing v4 trial6 FT on orig/A',change='50% FT + 50% fixed-configuration LightGBM, no weight fitting',evaluation='Development-only orig/A; improvement is candidate, requires early/A and seed confirmation',fit_end='2023-03',holdout='2023-04..2024-03',seed=42,weights={'ft':.5,'lgb':.5},lightgbm_config=CONFIG,feature_order=cols['numeric_names']+cols['categorical_names'],frozen_code_matrix_order=cols['codes'],categorical_names=cols['categorical_names'],source_and_input_sha256={str(p):sha(p) for p in inputs},environment=dict(python=__import__('sys').version,numpy=np.__version__,pandas=pd.__version__,lightgbm=lgb.__version__,cpu_threads=2),budget='one CPU LightGBM fit, 300 estimators; no FT retraining or weight search',output=str(OUT),early_window='No identified matching-contract FT and LGB pair; no new FT training',origin_provenance='FT NPZ has no origin field; paired unique row ID inherits frozen holdout origin; independent FT origin verification unavailable.')
    dump('manifest.json',manifest)
    start=time.monotonic()
    with np.load(BASE/'afit.npz',allow_pickle=False) as f, np.load(BASE/'ahold.npz',allow_pickle=False) as h, np.load(FT,allow_pickle=False) as t:
        assert len(f['row_id'])==52310 and len(h['row_id'])==35882
        assert len(np.unique(f['row_id']))==len(f['row_id']) and not np.intersect1d(f['row_id'],h['row_id']).size
        ix=align(h,t)
        assert np.all((h['horizon']>=1)&(h['horizon']<=12)) and np.all(h['origin']==26)
        for z in (f,h):
            assert np.allclose(to_sgd(z['target'],z['anchor_ksgd']),z['actual_sgd'],rtol=0,atol=.001)
        def frame(z):
            d=pd.DataFrame(z['numeric'],columns=cols['numeric_names'])
            for j,name in enumerate(cols['categorical_names']): d[name]=pd.Categorical(z['categorical'][:,j],categories=range(cols['vocabularies'][name]))
            return d
        model=lgb.LGBMRegressor(**CONFIG)
        model.fit(frame(f),f['target'],categorical_feature=cols['categorical_names'])
        assert model.booster_.feature_name() == manifest['feature_order']
        lp=to_sgd(model.predict(frame(h),num_threads=2),h['anchor_ksgd'])
        model.booster_.save_model(str(OUT/'lightgbm_model.txt'))
        fp=t['predicted_sgd'][ix]
        df=pd.DataFrame(dict(row_id=h['row_id'],origin=h['origin'],horizon=h['horizon'],actual_sgd=h['actual_sgd'],ft_prediction=fp,lgb_prediction=lp,blend_prediction=.5*(fp+lp)))
        df.to_csv(OUT/'predictions.csv',index=False,float_format='%.17g')
    reread=pd.read_csv(OUT/'predictions.csv',float_precision='round_trip')
    metrics=all_metrics(reread); metrics.update(fit_rows=52310,holdout_rows=35882,elapsed_seconds=time.monotonic()-start)
    assert abs(metrics['ft']['overall']['rmse_sgd']-581.0164302081428)<1e-9
    dump('metrics.json',metrics)
    lines=['# Transformer + LightGBM fixed blend: orig/A', '', 'Development-only candidate; orig/A fit through 2023-03 (52,310 rows), holdout 2023-04–2024-03 (35,882 rows). Both models use the same frozen inputs and residual target. FT is the existing selected v4 trial6 checkpoint prediction; LightGBM is a single fixed fit. FT training was previously selected on A holdout; results are not an independent test.', '', '| Model | RMSE SGD | MAE SGD | Bias SGD |', '|---|---:|---:|---:|']
    for n in ('ft','lgb','blend'):
        s=metrics[n]['overall']; lines.append(f"| {n} | {s['rmse_sgd']:.6f} | {s['mae_sgd']:.6f} | {s['bias_sgd']:.6f} |")
    lines+=['','Bias = prediction − actual. Horizon metrics (RMSE / MAE / bias in SGD):','','| Horizon | FT | LGB | Blend |','|---|---:|---:|---:|']
    for k in ('1-3','4-6','7-9','10-12'):
        cells=[]
        for n in ('ft','lgb','blend'):
            s=metrics[n]['horizon'][k]; cells.append(f"{s['rmse_sgd']:.4f} / {s['mae_sgd']:.4f} / {s['bias_sgd']:.4f} (n={s['n']})")
        lines.append('| '+k+' | '+' | '.join(cells)+' |')
    lines += ['',f"Residual correlation: {metrics['residual_correlation']:.8f}. Blend − FT RMSE: {metrics['blend_minus_ft_rmse_sgd']:+.6f} SGD; blend − LGB: {metrics['blend_minus_lgb_rmse_sgd']:+.6f} SGD.", '', 'Weights fixed at 0.5 before scoring. No validation-derived calibration or weight search. Frozen numeric transforms and categories are sourced from the established fit-only contract; historical feature timing is inherited from that contract, not independently reconstructed here. ID, actual and horizon are verified exactly; the FT file does not contain origin, so origin is supplied by its uniquely matched frozen row rather than independently verified. Predictions.csv is the scoring source and can independently reproduce metrics.json. Saved LightGBM model uses the documented 56 numeric and four categorical feature order.', '', 'No verified same-contract early/A FT and LightGBM pair was found; no second-window result or two-window ranking is claimed. No training-side OOF pair was established, so no learned weight is reported. Complete the second development window and paired seed confirmation before drawing a robust conclusion. Historical B results are not comparable to this window; B labels were not loaded.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    status('COMPLETED',elapsed_seconds=metrics['elapsed_seconds'])
    print(json.dumps(metrics,indent=2))
if __name__=='__main__':
    try: main()
    except Exception as e:
        status('FAILED',reason=str(e),traceback=traceback.format_exc()); raise
