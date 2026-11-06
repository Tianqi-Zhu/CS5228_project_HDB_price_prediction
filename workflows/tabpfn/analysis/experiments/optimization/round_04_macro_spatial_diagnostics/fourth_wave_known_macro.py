"""Competition-supplied macro covariates, including the query-month series.

This is a separately labeled information scenario: future-month macro values
are in the supplied auxiliary files. Rental labels remain frozen at origin.
It is not a claim of availability in deployment before those months happen.
"""
from __future__ import annotations
import argparse
import gc
import json
import time
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor,early_stopping
from analysis.pipeline.auxiliary import _monthly_macro_features,MACRO_NUMERIC
from analysis.experiments.frontier_search import frame_data,score,write_json
from analysis.experiments.second_wave_anniversary import select_inputs
from analysis.experiments.third_wave_training import make_data
from analysis.experiments.fourth_wave_diagnostics import OUT,ART


def macro_table():
    table=_monthly_macro_features().set_index('RENT_APPROVAL_DATE').sort_index()
    # Reindex so shifts represent calendar months, never variable row gaps.
    months=pd.period_range(table.index.min(),table.index.max(),freq='M').astype(str)
    table=table.reindex(months)
    original=table[MACRO_NUMERIC].copy()
    result=original.add_prefix('GIVEN_')
    for lag in (3,6,12):
        result=pd.concat([result,original.shift(lag).add_prefix(f'GIVEN_LAG{lag}_')],axis=1)
    levels=[c for c in MACRO_NUMERIC if 'LEVEL' in c or 'PREMIUM' in c]
    change=original[levels]/original[levels].shift(12).replace(0,np.nan)-1
    result=pd.concat([result,change.add_prefix('GIVEN_CHANGE12_')],axis=1)
    return result.replace([np.inf,-np.inf],np.nan)


def macro_data(stage):
    bundle=make_data(stage)
    raw,_=frame_data()
    source=raw.loc[raw.RENT_APPROVAL_DATE<bundle['start']]
    query=raw.loc[(raw.RENT_APPROVAL_DATE>=bundle['start'])&(raw.RENT_APPROVAL_DATE<bundle['end'])]
    table=macro_table()
    bundle['macro_fit']=pd.DataFrame(table.reindex(source.RENT_APPROVAL_DATE).to_numpy(np.float32),
        index=source.index,columns=table.columns)
    bundle['macro_val']=pd.DataFrame(table.reindex(query.RENT_APPROVAL_DATE).to_numpy(np.float32),
        index=query.index,columns=table.columns)
    return bundle


def inputs(bundle,spec):
    x,q=select_inputs(bundle,'window')
    columns=list(bundle['macro_fit']) if spec['kind']!='current' else ['GIVEN_'+c for c in MACRO_NUMERIC]
    x=pd.concat([x,bundle['macro_fit'][columns]],axis=1)
    q=pd.concat([q,bundle['macro_val'][columns]],axis=1)
    if spec['kind']=='no_time':
        x=x.drop(columns='APPROVAL_MONTH_INDEX')
        q=q.drop(columns='APPROVAL_MONTH_INDEX')
    return x,q


SPECS=[{'name':f'{kind}_{family}{size}','kind':kind,'family':family,
    'leaves' if family=='lgb' else 'depth':size}
    for kind in ('current','lags','no_time') for family,size in [('lgb',15),('lgb',31),('cat',6),('cat',8)]]


def fit(spec,bundle,iterations=None):
    x,q=inputs(bundle,spec)
    if spec['family']=='lgb':
        model=LGBMRegressor(n_estimators=iterations or 2200,learning_rate=.03,
            num_leaves=spec['leaves'],min_child_samples=50,reg_lambda=15.,
            random_state=42,n_jobs=4,verbosity=-1)
        model.fit(x,bundle['y_fit'],**({} if iterations else {'eval_set':[(q,bundle['y_val'])],
            'callbacks':[early_stopping(120,verbose=False)]}))
        count=iterations or int(model.best_iteration_)
    else:
        from catboost import CatBoostRegressor
        cats=list(x.select_dtypes(include=['category','object','string']).columns)
        for c in cats:
            x[c]=x[c].astype('string').fillna('__MISSING__').astype(str)
            q[c]=q[c].astype('string').fillna('__MISSING__').astype(str)
        x,q=x.fillna(-999.),q.fillna(-999.)
        model=CatBoostRegressor(iterations=iterations or 2200,learning_rate=.04,depth=spec['depth'],
            l2_leaf_reg=15.,random_strength=.4,random_seed=42,task_type='GPU',devices='0',
            gpu_ram_part=.55,border_count=128,allow_writing_files=False,verbose=False)
        model.fit(x,bundle['y_fit'],cat_features=cats,**({} if iterations else {
            'eval_set':(q,bundle['y_val']),'early_stopping_rounds':120}))
        count=iterations or int(model.get_best_iteration())+1
    return {'model':model,'spec':spec},model.predict(q),int(count),x.shape[1]


def run(stage,names=None):
    write_json(OUT/'known_macro_plan.json',{'specs':SPECS,'selection':['2023-10','2024-03'],
        'scenario':'Competition-supplied observed macro covariates through query horizon; rental history remains frozen at forecast origin.',
        'notes':'Only supplied COE and stock files; no external or query rental targets. Treat as a distinct information condition from unknown future macro forecasts.'})
    bundle=macro_data(stage)
    for spec in SPECS:
        name=spec['name']
        if names and name not in names:
            continue
        path=OUT/'known_macro'/stage/name
        if (path/'results.json').exists():
            continue
        count=None if stage=='screen' else json.loads((OUT/'known_macro/screen'/name/'results.json').read_text())['iterations']
        started=time.perf_counter()
        print('GIVEN MACRO',stage,name,flush=True)
        payload,pred,count,features=fit(spec,bundle,count)
        artifact=ART/'known_macro'/stage/name
        artifact.mkdir(parents=True,exist_ok=True)
        joblib.dump(payload,artifact/'model.joblib',compress=1)
        pd.DataFrame({'source_row':bundle['row_val'],'month':bundle['date_val'],
            'actual':bundle['y_val'],'prediction':pred}).to_csv(artifact/'predictions.csv',index=False)
        result={'spec':spec,'stage':stage,'iterations':count,'features':features,'query_rows':len(pred),
            **score(bundle['y_val'],pred),'seconds':time.perf_counter()-started,'source_sha256':bundle['hashes'],
            'scenario':'given_query_month_macro'}
        write_json(path/'results.json',result)
        print('GIVEN MACRO RESULT',stage,name,result['rmse_sgd'],flush=True)
        del payload
        gc.collect()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['screen','outer','june'])
    parser.add_argument('--names',nargs='+')
    args=parser.parse_args()
    run(args.stage,args.names)
