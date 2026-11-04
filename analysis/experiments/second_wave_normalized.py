"""Separate observed training-month market levels from property-level pricing.

Training targets may be centered/scaled by the *training* month's observed
flat-type mean. Query levels are always forecast from strictly earlier months.
Neither a query label nor its month's observed market mean enters prediction.
"""
from __future__ import annotations
import argparse
import gc
import json
import time
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping
from analysis.experiments.frontier_search import ROOT, frame_data, score, write_json
from analysis.experiments.frontier_compact import prepare_compact, ADDITIONS

OUT=ROOT/'reports/experiments/second_wave/normalized_market'
ART=ROOT/'output/models/second_wave/normalized_market'
SPECS=[]
for features in ('relative','distribution'):
    for target in ('ratio','difference'):
        for forecast in ('MARKET_LAST','MARKET_3M','MARKET_FORECAST'):
            SPECS.append({'name':f'{features}_{target}_{forecast.lower()}', 'family':'lgb',
                'features':features,'target':target,'forecast':forecast,'leaves':31,'normalizer':'mean'})
for features in ('compact','relative'):
    for forecast in ('MARKET_LAST','MARKET_3M','MARKET_FORECAST'):
        SPECS.append({'name':f'{features}_ratio15_{forecast.lower()}','family':'lgb',
            'features':features,'target':'ratio','forecast':forecast,'leaves':15,'normalizer':'mean'})
for target in ('ratio','difference'):
    SPECS.append({'name':f'median_{target}','family':'lgb','features':'relative',
        'target':target,'forecast':'MARKET_MEDIAN','leaves':31,'normalizer':'median'})
    for forecast in ('MARKET_LAST','MARKET_3M'):
        SPECS.append({'name':f'cat_{target}_{forecast.lower()}','family':'cat','features':'relative',
            'target':target,'forecast':forecast,'depth':6,'normalizer':'mean'})


def make_data(start,end):
    bundle=prepare_compact(start,end)
    frame,hashes=frame_data()
    fitting=frame.loc[frame.RENT_APPROVAL_DATE<start]
    np.testing.assert_array_equal(fitting.MONTHLY_RENT,bundle['y_fit'])
    reference={method:fitting.groupby(['RENT_APPROVAL_DATE','FLAT_TYPE_CLEAN']).MONTHLY_RENT.transform(method).to_numpy(float)
        for method in ('mean','median')}
    bundle['reference']=reference
    return bundle


def select_inputs(bundle,spec):
    x,q=bundle['x_fit'].copy(),bundle['x_val'].copy()
    # Remove stale currency levels and absolute time to encourage stationary premiums.
    remove=[c for c in x if c=='APPROVAL_MONTH_INDEX' or (c.startswith('HIST_') and c.endswith('_MEAN'))]
    x,q=x.drop(columns=remove),q.drop(columns=remove)
    if spec['features']!='compact':
        for column in ADDITIONS['relative_history']:
            x[column]=bundle['extra_fit'][column].to_numpy(np.float32)
            q[column]=bundle['extra_val'][column].to_numpy(np.float32)
    if spec['features']=='distribution':
        for column in ADDITIONS['all_shape']:
            if 'RENT_SD' not in column:
                x[column]=bundle['extra_fit'][column].to_numpy(np.float32)
                q[column]=bundle['extra_val'][column].to_numpy(np.float32)
    reference=bundle['reference'][spec['normalizer']]
    level=bundle['extra_val'][spec['forecast']].to_numpy(float)
    y=bundle['y_fit']/reference if spec['target']=='ratio' else bundle['y_fit']-reference
    validation=bundle['y_val']/level if spec['target']=='ratio' else bundle['y_val']-level
    return x,q,y,validation,reference,level


def fit(spec,bundle,iterations=None):
    x,q,y,validation,reference,level=select_inputs(bundle,spec)
    categories=list(x.select_dtypes(include=['category','object','string']).columns)
    weights=(reference/3000.)**2 if spec['target']=='ratio' else np.ones(len(y))
    query_weights=(level/3000.)**2 if spec['target']=='ratio' else np.ones(len(q))
    if spec['family']=='lgb':
        model=LGBMRegressor(n_estimators=iterations or 1800,learning_rate=.035,
            num_leaves=spec['leaves'],min_child_samples=60,reg_lambda=15.,
            random_state=42,n_jobs=4,verbosity=-1)
        args={} if iterations else {'eval_set':[(q,validation)],'eval_sample_weight':[query_weights],
            'callbacks':[early_stopping(100,verbose=False)]}
        model.fit(x,y,sample_weight=weights,**args)
        count=iterations or int(model.best_iteration_)
    else:
        from catboost import CatBoostRegressor,Pool
        for column in categories:
            x[column]=x[column].astype('string').fillna('__MISSING__').astype(str)
            q[column]=q[column].astype('string').fillna('__MISSING__').astype(str)
        x,q=x.fillna(-999.),q.fillna(-999.)
        training=Pool(x,y,cat_features=categories,weight=weights)
        testing=Pool(q,validation,cat_features=categories,weight=query_weights)
        model=CatBoostRegressor(iterations=iterations or 1800,depth=spec['depth'],
            learning_rate=.045,l2_leaf_reg=15.,random_seed=42,thread_count=4,
            loss_function='RMSE',verbose=False,allow_writing_files=False)
        model.fit(training,**({} if iterations else {'eval_set':testing,'early_stopping_rounds':100}))
        count=iterations or int(model.get_best_iteration())+1
    prediction=model.predict(q)
    prediction=prediction*level if spec['target']=='ratio' else prediction+level
    return model,prediction,count


def run(stage,names=None):
    write_json(OUT/'plan.json',{'trials':SPECS,'selection':['2023-10','2024-03'],
        'outer':['2024-04','2025-03'],'matched':['2024-06','2025-03'],
        'note':'Training-month mean/median is used only to transform training labels. Forecast query levels use prior labels. Ratio fit/evaluation weights maintain the SGD squared-error objective.'})
    start,end=('2023-10','2024-04') if stage=='screen' else (
        ('2024-04','2025-04') if stage=='outer' else ('2024-06','2025-04'))
    bundle=make_data(start,end)
    for spec in SPECS:
        if names and spec['name'] not in names:
            continue
        destination=OUT/stage/spec['name']
        if (destination/'results.json').exists():
            continue
        begun=time.perf_counter()
        iterations=None if stage=='screen' else json.loads((OUT/'screen'/spec['name']/'results.json').read_text())['iterations']
        model,prediction,count=fit(spec,bundle,iterations)
        artifact=ART/stage/spec['name']
        artifact.mkdir(parents=True,exist_ok=True)
        joblib.dump({'model':model,'spec':spec},artifact/'model.joblib',compress=1)
        pd.DataFrame({'source_row':bundle['row_val'],'month':bundle['date_val'],
            'actual':bundle['y_val'],'prediction':prediction}).to_csv(artifact/'predictions.csv',index=False)
        record={'name':spec['name'],'spec':spec,'stage':stage,'iterations':count,
            'rows':len(prediction),'features':len(model.feature_name_ if spec['family']=='lgb' else model.feature_names_),
            **score(bundle['y_val'],prediction),'seconds':time.perf_counter()-begun,'source_sha256':bundle['hashes']}
        write_json(destination/'results.json',record)
        print(f"NORMALIZED {stage} {spec['name']}: RMSE={record['rmse_sgd']:.4f}, iterations={count}, seconds={record['seconds']:.1f}",flush=True)
        del model
        gc.collect()
    rows=[]
    for path in OUT.glob('*/*/results.json'):
        value=json.loads(path.read_text())
        rows.append({k:v for k,v in value.items() if k not in ('spec','source_sha256')})
    if rows:
        pd.DataFrame(rows).sort_values(['stage','rmse_sgd']).to_csv(OUT/'metrics.csv',index=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['screen','outer','june'])
    parser.add_argument('--names',nargs='+')
    args=parser.parse_args()
    run(args.stage,args.names)
