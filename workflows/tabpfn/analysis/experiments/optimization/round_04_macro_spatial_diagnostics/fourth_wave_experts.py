"""Property-type experts with fixed-origin historical information only."""
from __future__ import annotations
import argparse
import gc
import json
import time
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor,early_stopping
from analysis.experiments.frontier_search import score,write_json
from analysis.experiments.third_wave_training import make_data,inputs
from analysis.experiments.fourth_wave_diagnostics import OUT,ART

SPECS=[{'name':f'{family}_{target}','family':family,'target':target,'adjusted':True,'training':'all'}
    for family in ('lgb15','lgb31','cat6','cat8') for target in ('raw','residual')]


def fit(spec,bundle,iterations=None):
    x,q,base,offset=inputs(bundle,spec)
    kind=x.FLAT_TYPE_CLEAN.astype('string').to_numpy(str)
    query_kind=q.FLAT_TYPE_CLEAN.astype('string').to_numpy(str)
    target=bundle['y_fit']-base
    validation=bundle['y_val']-offset
    # Every query retains a forecast, including rare or unsupported categories.
    pred=bundle['fallback'].copy()
    models={};counts={}
    for key in sorted(np.unique(kind)):
        selected=kind==key
        query=query_kind==key
        if selected.sum()<500 or not query.any():
            continue
        count=None if iterations is None else iterations[key]
        xx,qq=x.loc[selected].copy(),q.loc[query].copy()
        if spec['family'].startswith('lgb'):
            leaves=int(spec['family'][3:])
            model=LGBMRegressor(n_estimators=count or 2000,learning_rate=.03,num_leaves=leaves,
                min_child_samples=35,reg_lambda=15.,random_state=42,n_jobs=4,verbosity=-1)
            model.fit(xx,target[selected],**({} if count else {'eval_set':[(qq,validation[query])],
                'callbacks':[early_stopping(120,verbose=False)]}))
            count=count or int(model.best_iteration_)
        else:
            from catboost import CatBoostRegressor
            cats=list(xx.select_dtypes(include=['category','object','string']).columns)
            for c in cats:
                xx[c]=xx[c].astype('string').fillna('__MISSING__').astype(str)
                qq[c]=qq[c].astype('string').fillna('__MISSING__').astype(str)
            xx,qq=xx.fillna(-999.),qq.fillna(-999.)
            model=CatBoostRegressor(iterations=count or 2200,learning_rate=.04,depth=int(spec['family'][3:]),
                l2_leaf_reg=15.,random_strength=.4,random_seed=42,task_type='GPU',devices='0',
                gpu_ram_part=.55,border_count=128,allow_writing_files=False,verbose=False)
            model.fit(xx,target[selected],cat_features=cats,**({} if count else {
                'eval_set':(qq,validation[query]),'early_stopping_rounds':120}))
            count=count or int(model.get_best_iteration())+1
        pred[query]=model.predict(qq)+offset[query]
        counts[key]=int(count);models[key]=model
        print('EXPERT',key,'fit',selected.sum(),'rounds',count,flush=True)
    return {'models':models,'spec':spec},pred,counts


def run(stage,names=None):
    write_json(OUT/'experts_plan.json',{'specs':SPECS,'selection':['2023-10','2024-03'],
        'notes':'Separate flat-type models with all validation rows retained. Rare/unseen types retain the previously locked second-wave prediction. Each type selects rounds only on the earlier screen and freezes them at later origins.'})
    bundle=make_data(stage)
    for spec in SPECS:
        name=spec['name']
        if names and name not in names:
            continue
        path=OUT/'experts'/stage/name
        if (path/'results.json').exists():
            continue
        counts=None if stage=='screen' else json.loads((OUT/'experts/screen'/name/'results.json').read_text())['iterations']
        started=time.perf_counter()
        print('EXPERTS',stage,name,flush=True)
        payload,pred,counts=fit(spec,bundle,counts)
        artifact=ART/'experts'/stage/name
        artifact.mkdir(parents=True,exist_ok=True)
        joblib.dump(payload,artifact/'model.joblib',compress=1)
        pd.DataFrame({'source_row':bundle['row_val'],'month':bundle['date_val'],
            'actual':bundle['y_val'],'prediction':pred}).to_csv(artifact/'predictions.csv',index=False)
        result={'spec':spec,'stage':stage,'iterations':counts,'query_rows':len(pred),
            **score(bundle['y_val'],pred),'seconds':time.perf_counter()-started,'source_sha256':bundle['hashes']}
        write_json(path/'results.json',result)
        print('EXPERTS RESULT',stage,name,result['rmse_sgd'],flush=True)
        del payload
        gc.collect()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['screen','outer','june'])
    parser.add_argument('--names',nargs='+')
    args=parser.parse_args()
    run(args.stage,args.names)
