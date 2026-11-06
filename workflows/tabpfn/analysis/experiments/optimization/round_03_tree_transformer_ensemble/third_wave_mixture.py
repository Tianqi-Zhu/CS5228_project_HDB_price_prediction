"""Compare gated low/regular-rent experts with direct squared-error models."""
from __future__ import annotations
import argparse
import gc
import json
import time
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier,LGBMRegressor,early_stopping
from analysis.experiments.frontier_search import score,write_json
from analysis.experiments.third_wave_training import OUT,ART,make_data,inputs,adjusted_baseline

SPECS=[{'name':f'gate{int(threshold*100)}_lgb{leaves}','threshold':threshold,'leaves':leaves,
    'adjusted':True,'training':'all'} for threshold in (.7,.85,1.) for leaves in (15,31)]


def fit(spec,bundle,iterations=None):
    x,q,_,_=inputs(bundle,spec)
    baseline,_=adjusted_baseline(bundle['ann_fit'],bundle['extra_fit'],x)
    offset,_=adjusted_baseline(bundle['ann_val'],bundle['extra_val'],q)
    regime=(bundle['y_fit']<baseline*spec['threshold']).astype(int)
    query_regime=None if iterations else (bundle['y_val']<offset*spec['threshold']).astype(int)
    common={'learning_rate':.03,'num_leaves':spec['leaves'],'min_child_samples':70,
        'reg_lambda':20.,'random_state':42,'n_jobs':4,'verbosity':-1}
    gate=LGBMClassifier(n_estimators=iterations['gate'] if iterations else 1600,**common)
    gate.fit(x,regime,**({} if iterations else {'eval_set':[(q,query_regime)],
        'eval_metric':'binary_logloss','callbacks':[early_stopping(100,verbose=False)]}))
    counts={'gate':iterations['gate'] if iterations else int(gate.best_iteration_)}
    probability=gate.predict_proba(q)[:,1]
    experts=[]
    values=[]
    for state in (0,1):
        train=regime==state
        key=f'expert{state}'
        model=LGBMRegressor(n_estimators=iterations[key] if iterations else 1600,**common)
        options={}
        if not iterations:
            chosen=query_regime==state
            options={'eval_set':[(q.loc[chosen],(bundle['y_val']-offset)[chosen])],
                'callbacks':[early_stopping(100,verbose=False)]}
        model.fit(x.loc[train],(bundle['y_fit']-baseline)[train],**options)
        counts[key]=iterations[key] if iterations else int(model.best_iteration_)
        experts.append(model)
        values.append(model.predict(q)+offset)
    prediction=(1-probability)*values[0]+probability*values[1]
    return {'gate':gate,'experts':experts,'spec':spec},prediction,counts


def run(stage,names=None):
    write_json(OUT/'mixture_plan.json',{'specs':SPECS,'selection':['2023-10','2024-03'],
        'hypothesis':'A low-rent probability model plus conditional rent experts may represent the broad rent distribution.',
        'notes':'Regime is a supervised target only. Outer predictions use predicted probabilities and no actual query regime or rent.'})
    bundle=make_data(stage)
    for spec in SPECS:
        if names and spec['name'] not in names:
            continue
        destination=OUT/'mixture'/stage/spec['name']
        if (destination/'results.json').exists():
            continue
        iterations=None if stage=='screen' else json.loads((OUT/'mixture/screen'/spec['name']/'results.json').read_text())['iterations']
        started=time.perf_counter()
        print('MIXTURE',stage,spec['name'],flush=True)
        payload,pred,counts=fit(spec,bundle,iterations)
        artifact=ART/'mixture'/stage/spec['name']
        artifact.mkdir(parents=True,exist_ok=True)
        joblib.dump(payload,artifact/'model.joblib',compress=1)
        pd.DataFrame({'source_row':bundle['row_val'],'month':bundle['date_val'],
            'actual':bundle['y_val'],'prediction':pred}).to_csv(artifact/'predictions.csv',index=False)
        result={'spec':spec,'stage':stage,'iterations':counts,'query_rows':len(pred),
            **score(bundle['y_val'],pred),'seconds':time.perf_counter()-started,
            'source_sha256':bundle['hashes']}
        write_json(destination/'results.json',result)
        print(f"MIXTURE RESULT {stage} {spec['name']}: RMSE={result['rmse_sgd']:.4f}",flush=True)
        del payload
        gc.collect()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['screen','outer','june'])
    parser.add_argument('--names',nargs='+')
    args=parser.parse_args()
    run(args.stage,args.names)
