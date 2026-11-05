"""Label-free query batch counts and historical calendar-cohort support.

Predictions are made for a supplied batch of property records. The number of
records in each query month/address/type is observable without their rents.
Training batches use the analogous fitting-month records. No query target is
read by the feature builder; the batch-dependent inference contract is explicit.
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
from analysis.experiments.frontier_search import frame_data,score,write_json
from analysis.experiments.second_wave_anniversary import select_inputs
from analysis.experiments.third_wave_training import make_data
from analysis.experiments.fourth_wave_diagnostics import OUT,ART


def batch_features(query,ann):
    result={}
    date='RENT_APPROVAL_DATE'
    for name,fields in [('BLOCK',['ADDRESS_KEY','FLAT_TYPE_CLEAN']),('BUILDING',['ADDRESS_KEY']),
        ('STREET',['STREET_CLEAN','FLAT_TYPE_CLEAN']),('AREA',['PLANNING_AREA_CLEAN','FLAT_TYPE_CLEAN']),
        ('TYPE',['FLAT_TYPE_CLEAN'])]:
        counts=query.groupby([date,*fields],dropna=False)[date].transform('size').to_numpy(float)
        result[f'BATCH_{name}_COUNT']=counts
    n=result['BATCH_BLOCK_COUNT']
    result['BATCH_TYPE_SHARE']=n/result['BATCH_TYPE_COUNT']
    result['BATCH_STREET_SHARE']=n/result['BATCH_STREET_COUNT']
    result['BATCH_BUILDING_TYPE_SHARE']=n/result['BATCH_BUILDING_COUNT']
    for lag in (12,24,36):
        for radius in (0,1):
            prefix=f'ANN_BLOCK_{lag}_R{radius}'
            historical=ann[prefix+'_N'].to_numpy(float)
            result[f'FLOW_{lag}_R{radius}_RELATIVE_COUNT']=historical/n
            result[f'FLOW_{lag}_R{radius}_COMMON_COUNT']=np.minimum(n,historical)
            result[f'FLOW_{lag}_R{radius}_CAPPED_SHARE']=np.minimum(1.,historical/n)
            result[f'FLOW_{lag}_R{radius}_GAP']=n-historical
            result[f'FLOW_{lag}_R{radius}_EXCESS']=np.maximum(0,n-historical)
    return pd.DataFrame(result,index=query.index,dtype=np.float32)


def flow_data(stage):
    bundle=make_data(stage)
    raw,_=frame_data()
    source=raw.loc[raw.RENT_APPROVAL_DATE<bundle['start']]
    query=raw.loc[(raw.RENT_APPROVAL_DATE>=bundle['start'])&(raw.RENT_APPROVAL_DATE<bundle['end'])]
    bundle['flow_fit']=batch_features(source,bundle['ann_fit'])
    bundle['flow_val']=batch_features(query,bundle['ann_val'])
    return bundle


def inputs(bundle,spec):
    x,q=select_inputs(bundle,'window')
    columns=[c for c in bundle['flow_fit'] if spec['kind']=='cohort' or c.startswith('BATCH_')]
    return pd.concat([x,bundle['flow_fit'][columns]],axis=1),pd.concat([q,bundle['flow_val'][columns]],axis=1)


SPECS=[{'name':f'{kind}_{family}{size}','kind':kind,'family':family,
    'leaves' if family=='lgb' else 'depth':size}
    for kind in ('counts','cohort') for family,size in [('lgb',15),('lgb',31),('cat',6),('cat',8)]]


def fit(spec,bundle,iterations=None):
    x,q=inputs(bundle,spec)
    if spec['family']=='lgb':
        model=LGBMRegressor(n_estimators=iterations or 2200,learning_rate=.03,
            num_leaves=spec['leaves'],min_child_samples=40,reg_lambda=10.,
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
    write_json(OUT/'flows_plan.json',{'specs':SPECS,'selection':['2023-10','2024-03'],
        'notes':'Counts of unlabeled records in the supplied prediction batch; no current rents. Historical cohort counts divided by current observable record counts. Requires batch inference; incomplete batches may alter forecasts. All validation records retained.'})
    bundle=flow_data(stage)
    for spec in SPECS:
        name=spec['name']
        if names and name not in names:
            continue
        path=OUT/'flows'/stage/name
        if (path/'results.json').exists():
            continue
        count=None if stage=='screen' else json.loads((OUT/'flows/screen'/name/'results.json').read_text())['iterations']
        started=time.perf_counter()
        print('FLOW',stage,name,flush=True)
        payload,pred,count,features=fit(spec,bundle,count)
        artifact=ART/'flows'/stage/name
        artifact.mkdir(parents=True,exist_ok=True)
        joblib.dump(payload,artifact/'model.joblib',compress=1)
        pd.DataFrame({'source_row':bundle['row_val'],'month':bundle['date_val'],
            'actual':bundle['y_val'],'prediction':pred}).to_csv(artifact/'predictions.csv',index=False)
        result={'spec':spec,'stage':stage,'iterations':count,'features':features,'query_rows':len(pred),
            **score(bundle['y_val'],pred),'seconds':time.perf_counter()-started,'source_sha256':bundle['hashes']}
        write_json(path/'results.json',result)
        print('FLOW RESULT',stage,name,result['rmse_sgd'],flush=True)
        del payload
        gc.collect()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['screen','outer','june'])
    parser.add_argument('--names',nargs='+')
    args=parser.parse_args()
    run(args.stage,args.names)
