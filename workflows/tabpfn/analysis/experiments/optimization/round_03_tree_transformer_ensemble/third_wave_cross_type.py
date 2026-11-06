"""Share normalized historical rents across flat types in the same building."""
from __future__ import annotations
import argparse
import gc
import json
import time
import joblib
import numpy as np
import pandas as pd
from analysis.experiments.frontier_search import frame_data,DistributionHistory,month_number,score,write_json
from analysis.experiments.second_wave_anniversary import select_inputs
from analysis.experiments.third_wave_features import KernelCohorts
from analysis.experiments.third_wave_training import OUT,ART,make_data


def normalized_source(source):
    frame=source.copy()
    level=frame.groupby(['RENT_APPROVAL_DATE','FLAT_TYPE_CLEAN']).MONTHLY_RENT.transform('mean')
    frame['MONTHLY_RENT']=1000*frame.MONTHLY_RENT/level
    frame['FLAT_TYPE_CLEAN']='ALL_TYPES'
    return frame


def cross_data(stage):
    bundle=make_data(stage)
    cache=ART/'cross_type_cache'/f'{stage}.joblib'
    if cache.exists():
        saved=joblib.load(cache)
        if saved['hashes']==bundle['hashes']:
            bundle.update(saved['features'])
            return bundle
    raw,_=frame_data()
    source=raw.loc[raw.RENT_APPROVAL_DATE<bundle['start']]
    query=raw.loc[(raw.RENT_APPROVAL_DATE>=bundle['start'])&(raw.RENT_APPROVAL_DATE<bundle['end'])].copy()
    source=normalized_source(source)
    query['FLAT_TYPE_CLEAN']='ALL_TYPES'
    query['MONTHLY_RENT']=0. # No query targets enter either encoder.
    history=DistributionHistory().fit(source)
    fit_extra=history.transform(source,month_number(source.RENT_APPROVAL_DATE))
    val_extra=history.transform(query,np.full(len(query),bundle['origin']))
    kernel=KernelCohorts().fit(source)
    features={'cross_fit':fit_extra,'cross_val':val_extra,
        'cross_ann_fit':kernel.transform(source),'cross_ann_val':kernel.transform(query)}
    cache.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump({'hashes':bundle['hashes'],'features':features},cache,compress=1)
    bundle.update(features)
    return bundle


def inputs(bundle,spec):
    x,q=select_inputs(bundle,'window')
    history=[f'{grain}_{stat}' for grain in ('BLOCK','STREET') for stat in (
        'RATIO_MEAN','LAST_RATIO','RATIO_6M','RATIO_12M','N','N_6M','N_12M','RATIO_SD','LOW_SHARE')]
    cohorts=[c for c in bundle['cross_ann_fit'] if c.startswith('KERNEL_BLOCK_') and
        any(f'_{lag}_' in c for lag in (12,24,36)) and not c.endswith('_RENT')]
    for frame,extra,ann,old in [(x,bundle['cross_fit'],bundle['cross_ann_fit'],bundle['ann_fit']),
        (q,bundle['cross_val'],bundle['cross_ann_val'],bundle['ann_val'])]:
        if spec['kind'] in ('history','both'):
            for c in history:
                frame['CROSS_TYPE_'+c]=extra[c].to_numpy(np.float32)
        if spec['kind'] in ('cohort24','both','allcohorts'):
            for c in cohorts:
                if spec['kind']!='allcohorts' and '_24_' not in c:
                    continue
                frame['CROSS_TYPE_'+c]=ann[c].to_numpy(np.float32)
            frame['CROSS_TYPE_24_MINUS_SAME_TYPE']=(ann.KERNEL_BLOCK_24_RATIO.to_numpy()-
                old.ANN_BLOCK_24_R1_RATIO.to_numpy()).astype(np.float32)
    return x,q


SPECS=[{'name':f'{kind}_{family}{size}','kind':kind,'family':family,
    'leaves' if family=='lgb' else 'depth':size}
    for kind in ('history','cohort24','both','allcohorts')
    for family,size in [('lgb',15),('lgb',31),('cat',6),('cat',8)]]


def fit(spec,bundle,iterations=None):
    # Reuse the identical tree fitting parameters while substituting inputs.
    from lightgbm import LGBMRegressor,early_stopping
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
        categories=list(x.select_dtypes(include=['category','object','string']).columns)
        for c in categories:
            x[c]=x[c].astype('string').fillna('__MISSING__').astype(str)
            q[c]=q[c].astype('string').fillna('__MISSING__').astype(str)
        x,q=x.fillna(-999.),q.fillna(-999.)
        model=CatBoostRegressor(iterations=iterations or 2200,learning_rate=.04,depth=spec['depth'],
            l2_leaf_reg=10.,random_seed=42,thread_count=4,loss_function='RMSE',task_type='GPU',
            devices='0',gpu_ram_part=.55,border_count=128,random_strength=.4,
            allow_writing_files=False,verbose=False)
        model.fit(x,bundle['y_fit'],cat_features=categories,**({} if iterations else {
            'eval_set':(q,bundle['y_val']),'early_stopping_rounds':120}))
        count=iterations or int(model.get_best_iteration())+1
    return {'model':model,'spec':spec},model.predict(q),int(count),x.shape[1]


def run(stage,names=None):
    write_json(OUT/'cross_type_plan.json',{'specs':SPECS,'selection':['2023-10','2024-03'],
        'hypothesis':'Building rent premiums may be partially shared across flat types; normalize each observed source month by its own flat-type rent mean.',
        'information_boundary':'The source normalization is computed on fitting labels only and can only be retrieved in an earlier historical month. Query rent is zeroed before encoding.'})
    bundle=cross_data(stage)
    for spec in SPECS:
        if names and spec['name'] not in names:
            continue
        destination=OUT/'cross_type'/stage/spec['name']
        if (destination/'results.json').exists():
            continue
        iterations=None if stage=='screen' else json.loads((OUT/'cross_type/screen'/spec['name']/'results.json').read_text())['iterations']
        print('CROSS TYPE',stage,spec['name'],flush=True)
        started=time.perf_counter()
        payload,pred,count,n_features=fit(spec,bundle,iterations)
        artifact=ART/'cross_type'/stage/spec['name']
        artifact.mkdir(parents=True,exist_ok=True)
        joblib.dump(payload,artifact/'model.joblib',compress=1)
        pd.DataFrame({'source_row':bundle['row_val'],'month':bundle['date_val'],
            'actual':bundle['y_val'],'prediction':pred}).to_csv(artifact/'predictions.csv',index=False)
        result={'spec':spec,'stage':stage,'iterations':count,'features':n_features,'query_rows':len(pred),
            **score(bundle['y_val'],pred),'seconds':time.perf_counter()-started,'source_sha256':bundle['hashes']}
        write_json(destination/'results.json',result)
        print(f"CROSS TYPE RESULT {stage} {spec['name']}: RMSE={result['rmse_sgd']:.4f}",flush=True)
        del payload
        gc.collect()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['screen','outer','june'])
    parser.add_argument('--names',nargs='+')
    args=parser.parse_args()
    run(args.stage,args.names)
