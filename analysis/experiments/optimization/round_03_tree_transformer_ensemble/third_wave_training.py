"""Match supervised training support to available anniversary histories.

All earlier labels remain in the history pool even when their supervised rows
are downweighted or excluded. Every validation record is retained.
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
from analysis.experiments.frontier_search import ROOT,frame_data,month_number,score,write_json
from analysis.experiments.second_wave_anniversary import data,select_inputs

OUT=ROOT/'output/experiment_runs/round_03_tree_transformer_ensemble'
ART=ROOT/'output/models/round_03_tree_transformer_ensemble'
SECOND=ROOT/'output/experiment_runs/round_02_anniversary_history'
SECOND_ART=ROOT/'output/models/round_02_anniversary_history'


def reference_predictions(stage,rows,actual):
    if stage=='screen':
        members=json.loads((SECOND/'ensemble_selection.json').read_text())['members']
        result=np.zeros(len(rows))
        for item in members.values():
            p=pd.read_csv(ROOT/item['screen_path']).sort_values('source_row')
            np.testing.assert_array_equal(p.source_row,rows)
            np.testing.assert_array_equal(p.actual,actual)
            result+=item['weight']*p.prediction.to_numpy()
        return result
    p=pd.read_csv(SECOND_ART/'ensemble'/stage/'predictions.csv').sort_values('source_row')
    np.testing.assert_array_equal(p.source_row,rows)
    np.testing.assert_array_equal(p.actual,actual)
    return p.prediction.to_numpy()


def make_data(stage):
    start,end=('2023-10','2024-04') if stage=='screen' else (
        ('2024-04','2025-04') if stage=='outer' else ('2024-06','2025-04'))
    bundle=data(start,end)
    frame,_=frame_data()
    source=frame.loc[frame.RENT_APPROVAL_DATE<start]
    bundle['fit_month']=month_number(source.RENT_APPROVAL_DATE)
    bundle['origin']=int(month_number(pd.Series([start]))[0])
    bundle['fallback']=reference_predictions(stage,bundle['row_val'],bundle['y_val'])
    bundle['start'],bundle['end']=start,end
    return bundle


def training_mask(bundle,spec):
    months=bundle['fit_month']
    policy=spec['training']
    selected=np.ones(len(months),bool)
    if policy=='last12':
        selected &= months>=bundle['origin']-12
    elif policy=='last24':
        selected &= months>=bundle['origin']-24
    elif policy=='history24':
        selected &= months>=months.min()+24
    elif policy=='supported24':
        selected &= bundle['ann_fit'].ANN_BLOCK_24_R1_N.to_numpy()>0
    elif policy not in ('all','decay12'):
        raise ValueError(policy)
    return selected


def adjusted_baseline(ann,extra,raw):
    prior=extra.BLOCK_RATIO_MEAN.to_numpy(float)
    prior=np.where(np.isfinite(prior),prior,extra.AREA_RATIO_MEAN.to_numpy(float))
    prior=np.where(np.isfinite(prior),prior,1.)
    n=ann.ANN_BLOCK_24_R1_N.to_numpy(float)
    value=ann.ANN_BLOCK_24_R1_RATIO.to_numpy(float)
    ratio=(np.nan_to_num(value)*n+2.*prior)/(n+2.)
    level=extra.MARKET_3M.to_numpy(float)
    # Label-independent cold-start fallback, relevant only to the first month.
    level=np.where(np.isfinite(level),level,raw.FLOOR_AREA_SQM.to_numpy(float)*30.)
    return ratio*level,{'COHORT24_ADJUSTED_RENT':ratio*level,
        'COHORT24_RATIO_VS_BLOCK':value-prior,
        'COHORT24_SHRUNK_RATIO':ratio,'PAST_TYPE_LEVEL':level,
        'COHORT24_RELATIVE_SUPPORT':n/(extra.BLOCK_N.to_numpy(float)+1.)}


def inputs(bundle,spec):
    x,q=select_inputs(bundle,'window')
    base,extra=adjusted_baseline(bundle['ann_fit'],bundle['extra_fit'],x)
    offset,query_extra=adjusted_baseline(bundle['ann_val'],bundle['extra_val'],q)
    if spec.get('adjusted'):
        for column in extra:
            x[column]=extra[column].astype(np.float32)
            q[column]=query_extra[column].astype(np.float32)
    if spec.get('target')!='residual':
        base,offset=np.zeros(len(x)),np.zeros(len(q))
    if spec.get('drop_ann36'):
        remove=[c for c in x if c.startswith('ANN_BLOCK_36_')]
        x,q=x.drop(columns=remove),q.drop(columns=remove)
    return x,q,base,offset


SPECS=[]
for family in ('lgb15','lgb31','catgpu6'):
    for policy in ('all','last24','last12','history24','decay12','supported24'):
        SPECS.append({'name':f'{family}_{policy}','family':family,'training':policy,'target':'raw'})
for family in ('lgb15','catgpu6'):
    for policy in ('all','last12','history24'):
        SPECS.append({'name':f'{family}_adjusted_{policy}','family':family,'training':policy,
            'target':'residual','adjusted':True})


def fit(spec,bundle,iterations=None):
    x,q,base,offset=inputs(bundle,spec)
    selected=training_mask(bundle,spec)
    query_selected=np.ones(len(q),bool)
    if spec['training']=='supported24':
        query_selected=bundle['ann_val'].ANN_BLOCK_24_R1_N.to_numpy()>0
    if selected.sum()<500 or query_selected.sum()<100:
        raise ValueError('Insufficient supported data')
    weights=np.ones(len(x))
    if spec['training']=='decay12':
        weights=np.exp2(-(bundle['origin']-bundle['fit_month'])/12.)
    target=bundle['y_fit']-base
    validation=bundle['y_val']-offset
    family=spec['family']
    if family.startswith('lgb'):
        leaves=int(family[3:])
        model=LGBMRegressor(n_estimators=iterations or 2200,learning_rate=.03,
            num_leaves=leaves,min_child_samples=40,reg_lambda=10.,
            n_jobs=4,random_state=42,verbosity=-1)
        options={} if iterations else {'eval_set':[(q.loc[query_selected],validation[query_selected])],
            'callbacks':[early_stopping(120,verbose=False)]}
        model.fit(x.loc[selected],target[selected],sample_weight=weights[selected],**options)
        count=iterations or int(model.best_iteration_)
    else:
        from catboost import CatBoostRegressor,Pool
        cats=list(x.select_dtypes(include=['category','object','string']).columns)
        for column in cats:
            x[column]=x[column].astype('string').fillna('__MISSING__').astype(str)
            q[column]=q[column].astype('string').fillna('__MISSING__').astype(str)
        x,q=x.fillna(-999.),q.fillna(-999.)
        model=CatBoostRegressor(iterations=iterations or 2200,learning_rate=.04,depth=6,
            l2_leaf_reg=10.,random_strength=.4,random_seed=42,task_type='GPU',devices='0',
            gpu_ram_part=.6,border_count=128,allow_writing_files=False,verbose=False)
        pool=Pool(x.loc[selected],target[selected],cat_features=cats,weight=weights[selected])
        options={} if iterations else {'eval_set':Pool(q.loc[query_selected],validation[query_selected],cat_features=cats),
            'early_stopping_rounds':120}
        model.fit(pool,**options)
        count=iterations or int(model.get_best_iteration())+1
    predicted=np.asarray(model.predict(q),float)+offset
    if spec['training']=='supported24':
        predicted=np.where(query_selected,predicted,bundle['fallback'])
    return model,predicted,count,int(selected.sum()),int(query_selected.sum())


def run(stage,names=None):
    write_json(OUT/'training_plan.json',{'specs':SPECS,'selection':['2023-10','2024-03'],
        'outer':['2024-04','2025-03'],'june':['2024-06','2025-03'],
        'notes':'All past records remain in the historical pool. Supervised masks use dates/available history only. Unsupported query rows retain the previously locked second-wave forecast. GPU CatBoost is recorded as a different engine; its bitwise reproducibility is not assumed.'})
    bundle=make_data(stage)
    rows=[]
    for spec in SPECS:
        if names and spec['name'] not in names:
            continue
        destination=OUT/'training'/stage/spec['name']
        if (destination/'results.json').exists():
            continue
        iterations=None if stage=='screen' else json.loads((OUT/'training/screen'/spec['name']/'results.json').read_text())['iterations']
        print('THIRD WAVE',stage,spec['name'],flush=True)
        started=time.perf_counter()
        model,prediction,count,n_fit,n_query=fit(spec,bundle,iterations)
        artifact=ART/'training'/stage/spec['name']
        artifact.mkdir(parents=True,exist_ok=True)
        joblib.dump({'model':model,'spec':spec},artifact/'model.joblib',compress=1)
        pd.DataFrame({'source_row':bundle['row_val'],'month':bundle['date_val'],
            'actual':bundle['y_val'],'prediction':prediction}).to_csv(artifact/'predictions.csv',index=False)
        record={'spec':spec,'stage':stage,'iterations':count,'supervised_rows':n_fit,
            'historical_rows':len(bundle['y_fit']),'specialized_query_rows':n_query,
            'query_rows':len(prediction),**score(bundle['y_val'],prediction),
            'seconds':time.perf_counter()-started,'source_sha256':bundle['hashes']}
        write_json(destination/'results.json',record)
        print(f"THIRD RESULT {stage} {spec['name']}: RMSE={record['rmse_sgd']:.4f}, fitting={n_fit}, rounds={count}",flush=True)
        del model
        gc.collect()
    for path in (OUT/'training').glob('*/*/results.json'):
        value=json.loads(path.read_text())
        rows.append({k:v for k,v in value.items() if k not in ('spec','source_sha256')}|{'name':value['spec']['name']})
    if rows:
        pd.DataFrame(rows).sort_values(['stage','rmse_sgd']).to_csv(OUT/'training_metrics.csv',index=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['screen','outer','june'])
    parser.add_argument('--names',nargs='+')
    args=parser.parse_args()
    run(args.stage,args.names)
