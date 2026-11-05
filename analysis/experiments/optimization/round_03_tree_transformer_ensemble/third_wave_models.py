"""Controlled model-family comparison using the same historical feature set."""
from __future__ import annotations
import argparse
import gc
import json
import sys
import time
import joblib
import numpy as np
import pandas as pd
from analysis.experiments.frontier_search import ROOT,score,write_json
from analysis.experiments.third_wave_training import OUT,ART,make_data,inputs

SPECS=[{'name':f'xgb{depth}', 'family':'xgb','depth':depth,'training':'all'} for depth in (3,5,7,9)]
SPECS += [{'name':f'xgb{depth}_residual','family':'xgb','depth':depth,
    'training':'all','adjusted':True,'target':'residual'} for depth in (3,5)]
SPECS += [{'name':family,'family':family,'training':'all','adjusted':True,
    'target':'residual'} for family in ('embedding_mlp','embedding_resnet','ft_transformer')]
SPECS += [{'name':'catgpu8_control','family':'cat','depth':8,'training':'all'}]


def fit(spec,bundle,iterations=None):
    x,q,base,offset=inputs(bundle,spec)
    y=bundle['y_fit']-base
    target=bundle['y_val']-offset
    if spec['family']=='xgb':
        sys.path.insert(0,str(ROOT/'.cache/model-deps'))
        from xgboost import XGBRegressor
        model=XGBRegressor(n_estimators=iterations or 3000,max_depth=spec['depth'],
            learning_rate=.03,min_child_weight=20,reg_lambda=20.,reg_alpha=.5,
            subsample=.85,colsample_bytree=.9,random_state=42,n_jobs=4,
            tree_method='hist',device='cuda',enable_categorical=True,
            **({} if iterations else {'early_stopping_rounds':120}))
        model.fit(x,y,**({} if iterations else {'eval_set':[(q,target)],'verbose':False}))
        count=iterations or model.best_iteration+1
        prediction=model.predict(q)+offset
        payload={'model':model,'spec':spec,'columns':list(x)}
    elif spec['family']=='cat':
        from catboost import CatBoostRegressor
        categorical=list(x.select_dtypes(include=['category','string','object']).columns)
        for c in categorical:
            x[c]=x[c].astype('string').fillna('__MISSING__').astype(str)
            q[c]=q[c].astype('string').fillna('__MISSING__').astype(str)
        x,q=x.fillna(-999.),q.fillna(-999.)
        model=CatBoostRegressor(iterations=iterations or 2200,learning_rate=.04,depth=spec['depth'],
            l2_leaf_reg=10.,random_seed=42,thread_count=4,loss_function='RMSE',
            task_type='GPU',devices='0',gpu_ram_part=.55,border_count=128,
            random_strength=.4,allow_writing_files=False,verbose=False)
        model.fit(x,y,cat_features=categorical,**({} if iterations else {
            'eval_set':(q,target),'early_stopping_rounds':120}))
        count=iterations or int(model.get_best_iteration())+1
        prediction=model.predict(q)+offset
        payload={'model':model,'spec':spec,'columns':list(x)}
    else:
        from analysis.pipeline.neural_models import fit_neural,predict_neural
        # A feature with no variation in the fitting pool has no learned effect.
        # Keep this decision independent of query values and query labels.
        columns=[c for c in x if x[c].nunique(dropna=False)>1 and x[c].notna().any()]
        x,q=x[columns],q[columns]
        categorical=list(x.select_dtypes(include=['category','string','object']).columns)
        settings={'width':96 if spec['family']=='ft_transformer' else 192,
            'layers':2,'dropout':.15,'lr':3e-4 if spec['family']=='ft_transformer' else 1e-3,
            'max_epochs':35,'patience':6,'batch_size':256 if spec['family']=='ft_transformer' else 1024,
            'seed':spec.get('seed',42),'weight_decay':1e-3}
        fitted=fit_neural(spec['family'],x,y,categorical,settings,
            x_eval=None if iterations else q,y_eval=None if iterations else target,
            fixed_epochs=iterations)
        prediction=predict_neural(fitted,q)+offset
        count=fitted['epochs']
        model=fitted['model'].to('cpu')
        payload={'spec':spec,'settings':settings,'columns':columns,
            'fitted':fitted,'categorical':categorical}
        fitted['device']='cpu'
    return payload,prediction,int(count),len(x),x.shape[1]


def run(stage,names=None):
    write_json(OUT/'models_plan.json',{'specs':SPECS,'selection':['2023-10','2024-03'],
        'notes':'All models use the same pre-origin historical pool and all validation rows. Outer iteration counts are frozen from the earlier chronological screen.'})
    bundle=make_data(stage)
    for spec in SPECS:
        if names and spec['name'] not in names:
            continue
        destination=OUT/'models'/stage/spec['name']
        if (destination/'results.json').exists():
            continue
        iterations=None if stage=='screen' else json.loads((OUT/'models/screen'/spec['name']/'results.json').read_text())['iterations']
        started=time.perf_counter()
        print('MODEL',stage,spec['name'],flush=True)
        payload,pred,count,n_fit,n_features=fit(spec,bundle,iterations)
        artifact=ART/'models'/stage/spec['name']
        artifact.mkdir(parents=True,exist_ok=True)
        joblib.dump(payload,artifact/'model.joblib',compress=1)
        pd.DataFrame({'source_row':bundle['row_val'],'month':bundle['date_val'],
            'actual':bundle['y_val'],'prediction':pred}).to_csv(artifact/'predictions.csv',index=False)
        record={'spec':spec,'stage':stage,'iterations':count,'supervised_rows':n_fit,
            'features':n_features,'query_rows':len(pred),**score(bundle['y_val'],pred),
            'seconds':time.perf_counter()-started,'source_sha256':bundle['hashes']}
        write_json(destination/'results.json',record)
        print(f"MODEL RESULT {stage} {spec['name']}: RMSE={record['rmse_sgd']:.4f}, rounds={count}",flush=True)
        del payload
        gc.collect()
        if spec['family'] in ('embedding_mlp','embedding_resnet','ft_transformer'):
            import torch
            torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['screen','outer','june'])
    parser.add_argument('--names',nargs='+')
    args=parser.parse_args()
    run(args.stage,args.names)
