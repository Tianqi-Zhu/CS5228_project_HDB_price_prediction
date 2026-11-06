"""Focused refinements of the discovered anniversary signal."""
from __future__ import annotations
import argparse
import gc
import json
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor,early_stopping
from analysis.experiments.second_wave_anniversary import data,select_inputs
from analysis.experiments.frontier_search import ROOT,frame_data,score,write_json

OUT=ROOT/'output/experiment_runs/round_02_anniversary_history/cohort_extensions'
ART=ROOT/'output/models/round_02_anniversary_history/cohort_extensions'
SPECS=[{'name':f'{kind}_{"ratio" if normalized else "raw"}_{family}',
    'kind':kind,'normalized':normalized,'family':family,'leaves':15,'depth':6}
    for kind in ('short','cross') for normalized in (False,True) for family in ('lgb','cat')]


def make_data(start,end):
    bundle=data(start,end)
    source,_=frame_data()
    bundle['raw_fit']=source.loc[source.RENT_APPROVAL_DATE<start]
    bundle['raw_val']=source.loc[(source.RENT_APPROVAL_DATE>=start)&(source.RENT_APPROVAL_DATE<end)]
    return bundle


def features(bundle,spec):
    kind=spec['kind']
    x,q=select_inputs(bundle,'window' if kind in ('short','cross') else kind)
    if kind in ('short','cross'):
        remove=[c for c in x if c.startswith('ANN_BLOCK_36_')]
        x,q=x.drop(columns=remove),q.drop(columns=remove)
    if kind=='cross':
        for frame,source in ((x,bundle['raw_fit']),(q,bundle['raw_val'])):
            frame['ADDRESS_KEY']=source.ADDRESS_KEY.to_numpy(str)
            month=pd.to_datetime(source.RENT_APPROVAL_DATE).dt.month.astype(str)
            frame['ADDRESS_TYPE_MONTH']=(source.ADDRESS_KEY.astype(str)+'|'+
                source.FLAT_TYPE_CLEAN.astype(str)+'|'+month).to_numpy(str)
        for c in ('ADDRESS_KEY','ADDRESS_TYPE_MONTH'):
            levels=x[c].unique()
            x[c]=pd.Categorical(x[c],categories=levels)
            q[c]=pd.Categorical(q[c].where(q[c].isin(levels)),categories=levels)
    if spec.get('normalized'):
        remove=[c for c in x if c=='APPROVAL_MONTH_INDEX' or
            (c.startswith('HIST_') and c.endswith('_MEAN')) or
            (c.startswith('ANN_') and c.endswith('_RENT'))]
        x,q=x.drop(columns=remove),q.drop(columns=remove)
    return x,q


def fit(spec,bundle,iterations=None):
    x,q=features(bundle,spec)
    target=bundle['y_fit'].copy()
    query_target=bundle['y_val'].copy()
    level=np.ones(len(q))
    weights=None
    query_weights=None
    if spec.get('normalized'):
        source=bundle['raw_fit']
        reference=source.groupby(['RENT_APPROVAL_DATE','FLAT_TYPE_CLEAN']).MONTHLY_RENT.transform('mean').to_numpy(float)
        target/=reference
        level=bundle['extra_val'].MARKET_3M.to_numpy(float)
        query_target/=level
        weights=(reference/3000.)**2
        query_weights=(level/3000.)**2
    if spec['family']=='lgb':
        model=LGBMRegressor(n_estimators=iterations or 1800,learning_rate=.03,
            num_leaves=spec['leaves'],min_child_samples=40,reg_lambda=10.,
            random_state=42,n_jobs=4,verbosity=-1)
        args={} if iterations else {'eval_set':[(q,query_target)],'callbacks':[early_stopping(120,verbose=False)]}
        if query_weights is not None and not iterations:
            args['eval_sample_weight']=[query_weights]
        model.fit(x,target,sample_weight=weights,**args)
        count=iterations or int(model.best_iteration_)
    else:
        from catboost import CatBoostRegressor,Pool
        categories=list(x.select_dtypes(include=['category','object','string']).columns)
        for c in categories:
            x[c]=x[c].astype('string').fillna('__MISSING__').astype(str)
            q[c]=q[c].astype('string').fillna('__MISSING__').astype(str)
        x,q=x.fillna(-999.),q.fillna(-999.)
        model=CatBoostRegressor(iterations=iterations or 1800,learning_rate=.045,depth=6,
            l2_leaf_reg=15.,random_seed=42,thread_count=4,loss_function='RMSE',
            verbose=False,allow_writing_files=False)
        if weights is None:
            model.fit(x,target,cat_features=categories,**({} if iterations else {
                'eval_set':(q,query_target),'early_stopping_rounds':100}))
        else:
            pool=Pool(x,target,cat_features=categories,weight=weights)
            query_pool=Pool(q,query_target,cat_features=categories,weight=query_weights)
            model.fit(pool,**({} if iterations else {'eval_set':query_pool,'early_stopping_rounds':100}))
        count=iterations or int(model.get_best_iteration())+1
    return model,np.asarray(model.predict(q))*level,count,x.shape[1]


def run(stage,names=None):
    write_json(OUT/'plan.json',{'trials':SPECS,
        'note':'Added after discovery of the anniversary signal. Earlier selection only; outer periods are reused development data. Short removes sparsely supported 36-month fields; cross adds address and address-type-calendar-month categories; ratio uses observed fitting-month means only for training targets.'})
    start,end=('2023-10','2024-04') if stage=='screen' else (
        ('2024-04','2025-04') if stage=='outer' else ('2024-06','2025-04'))
    bundle=make_data(start,end)
    for spec in SPECS:
        if names and spec['name'] not in names:
            continue
        destination=OUT/stage/spec['name']
        if (destination/'results.json').exists():
            continue
        count=None if stage=='screen' else json.loads((OUT/'screen'/spec['name']/'results.json').read_text())['iterations']
        model,prediction,count,n_features=fit(spec,bundle,count)
        directory=ART/stage/spec['name']
        directory.mkdir(parents=True,exist_ok=True)
        joblib.dump({'model':model,'spec':spec},directory/'model.joblib',compress=1)
        pd.DataFrame({'source_row':bundle['row_val'],'month':bundle['date_val'],
            'actual':bundle['y_val'],'prediction':prediction}).to_csv(directory/'predictions.csv',index=False)
        record={'name':spec['name'],'spec':spec,'stage':stage,'iterations':count,'features':n_features,
            'rows':len(prediction),**score(bundle['y_val'],prediction),'source_sha256':bundle['hashes']}
        write_json(destination/'results.json',record)
        print('COHORT EXTENSION',stage,spec['name'],record['rmse_sgd'],flush=True)
        del model
        gc.collect()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['screen','outer','june'])
    parser.add_argument('--names',nargs='+')
    args=parser.parse_args()
    run(args.stage,args.names)
