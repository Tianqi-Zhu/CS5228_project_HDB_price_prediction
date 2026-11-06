"""Train calendar-cohort models at simulated frozen information cutoffs."""
from __future__ import annotations
import argparse
import gc
import json
import time
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor,early_stopping
from analysis.experiments.frontier_search import ROOT,frame_data,month_number,DistributionHistory,score,write_json
from analysis.experiments.frontier_compact import ADDITIONS
from analysis.experiments.second_wave_anniversary import AnniversaryHistory,select_inputs
from analysis.experiments.third_wave_training import make_data
from analysis.experiments.fourth_wave_diagnostics import OUT,ART


class CutoffCohorts(AnniversaryHistory):
    def transform(self,query,cutoff=None):
        from analysis.experiments.frontier_search import keys
        months=month_number(query.RENT_APPROVAL_DATE)
        cutoff=months if cutoff is None else np.asarray(cutoff)
        if np.any(cutoff>months):
            raise ValueError('Information cutoff follows query month')
        identifiers=keys(query,('ADDRESS_KEY','FLAT_TYPE_CLEAN'))
        columns={}
        for lag in (12,24,36):
            total=np.zeros((len(query),5))
            for delta in (-1,0,1):
                historical=months-lag+delta
                index=pd.MultiIndex.from_arrays([identifiers,historical],names=['key','month'])
                found=self.tables['BLOCK'].reindex(index).fillna(0).to_numpy(float)
                total+=found*(historical<cutoff)[:,None]
            n,rent,ratio,ratio2,low=total.T
            divisor=np.where(n>0,n,np.nan)
            prefix=f'ANN_BLOCK_{lag}_R1'
            columns[prefix+'_N']=n
            columns[prefix+'_RENT']=rent/divisor
            columns[prefix+'_RATIO']=ratio/divisor
            columns[prefix+'_SD']=np.sqrt(np.maximum(0,ratio2/divisor-(ratio/divisor)**2))
            columns[prefix+'_LOW']=low/divisor
        return pd.DataFrame(columns,index=query.index)


def build_rows(raw,static,cutoff,distribution,cohorts):
    extra=distribution.transform(raw,cutoff,neighbors=False)
    x=static[[c for c in static if not c.startswith('HIST_')]].copy()
    for grain,source in [('BLOCK_FLAT','BLOCK'),('AREA_FLAT','AREA')]:
        for suffix,field in [('MEAN','RENT_MEAN'),('COUNT','N'),('LAST_MEAN','LAST_MEAN'),('RECENCY_MONTHS','RECENCY')]:
            x[f'HIST_{grain}_{suffix}']=extra[f'{source}_{field}'].to_numpy(np.float32)
    for c in ADDITIONS['relative_history']:
        x[c]=extra[c].to_numpy(np.float32)
    ann=cohorts.transform(raw,cutoff)
    x=pd.concat([x,ann.astype(np.float32)],axis=1)
    for c in ('INFORMATION_AGE','FORECAST_HORIZON','MARKET_LAST','MARKET_3M','MARKET_12M',
        'MARKET_LOG_SLOPE','MARKET_FORECAST','MARKET_COUNT_3M'):
        x[c]=extra[c].to_numpy(np.float32)
    level=extra.MARKET_3M.to_numpy(float)
    level=np.where(np.isfinite(level),level,raw.FLOOR_AREA_SQM.to_numpy(float)*30.)
    return x,level


def augmented_data(stage):
    bundle=make_data(stage)
    cache=ART/'horizon_cache'/f'{stage}.joblib'
    if cache.exists():
        saved=joblib.load(cache)
        if saved['hashes']==bundle['hashes']:
            bundle.update(saved['features'])
            return bundle
    raw,_=frame_data()
    source=raw.loc[raw.RENT_APPROVAL_DATE<bundle['start']]
    query=raw.loc[(raw.RENT_APPROVAL_DATE>=bundle['start'])&(raw.RENT_APPROVAL_DATE<bundle['end'])]
    history=DistributionHistory().fit(source)
    cohorts=CutoffCohorts().fit(source)
    months=month_number(source.RENT_APPROVAL_DATE)
    reference=source.groupby(['RENT_APPROVAL_DATE','FLAT_TYPE_CLEAN']).MONTHLY_RENT.transform('mean').to_numpy(float)
    views={}
    for delay in (0,3,6,9,12):
        selected=np.ones(len(source),bool) if delay==0 else months-delay>=months.min()+3
        positions=np.flatnonzero(selected)
        print('HORIZON FEATURES',stage,'delay',delay,'rows',len(positions),flush=True)
        x,level=build_rows(source.loc[selected],bundle['x_fit'].loc[selected],months[selected]-delay,history,cohorts)
        if delay==0:
            before,_=select_inputs(bundle,'window')
            for column in before.select_dtypes(exclude=['category','object','string']).columns:
                np.testing.assert_allclose(before[column],x[column],equal_nan=True,atol=1e-3,rtol=1e-5)
        views[delay]={'x':x,'y':bundle['y_fit'][selected],'level':level,
            'reference':reference[selected],'source_row':source.index.to_numpy()[selected]}
    q,level=build_rows(query,bundle['x_val'],np.full(len(query),bundle['origin']),history,cohorts)
    features={'views':views,'horizon_query':q,'query_level':level}
    cache.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump({'hashes':bundle['hashes'],'features':features},cache,compress=1)
    bundle.update(features)
    return bundle


SPECS=[{'name':f'{view}_{target}_{family}','view':view,'target':target,'family':family}
    for view in ('one','multi','long') for target in ('raw','residual','ratio')
    for family in ('lgb15','cat6')]


def inputs(bundle,spec):
    delays={'one':(0,),'multi':(0,3,6,9),'long':(0,6,12)}[spec['view']]
    views=[bundle['views'][delay] for delay in delays]
    x=pd.concat([v['x'] for v in views],ignore_index=True)
    q=bundle['horizon_query'].copy()
    y=np.concatenate([v['y'] for v in views])
    ref=np.concatenate([v['reference'] for v in views])
    level=np.concatenate([v['level'] for v in views])
    # Each original row has a total weight of one across available snapshots.
    ids=np.concatenate([v['source_row'] for v in views])
    _,inverse,count=np.unique(ids,return_inverse=True,return_counts=True)
    weights=1./count[inverse]
    offset=bundle['query_level'].copy()
    validation=bundle['y_val'].copy()
    query_weights=None
    if spec['target']=='residual':
        y=y-level
        validation=validation-offset
    elif spec['target']=='ratio':
        y=y/ref
        validation=validation/offset
        weights*=(ref/3000.)**2
        query_weights=(offset/3000.)**2
    else:
        offset[:]=0.
    return x,q,y,validation,weights,query_weights,offset


def fit(spec,bundle,iterations=None):
    x,q,y,target,weights,query_weights,offset=inputs(bundle,spec)
    if spec['family'].startswith('lgb'):
        model=LGBMRegressor(n_estimators=iterations or 2000,learning_rate=.03,num_leaves=15,
            min_child_samples=60,reg_lambda=15.,random_state=42,n_jobs=4,verbosity=-1)
        options={} if iterations else {'eval_set':[(q,target)],'callbacks':[early_stopping(120,verbose=False)]}
        if query_weights is not None and not iterations:
            options['eval_sample_weight']=[query_weights]
        model.fit(x,y,sample_weight=weights,**options)
        count=iterations or int(model.best_iteration_)
    else:
        from catboost import CatBoostRegressor,Pool
        cats=list(x.select_dtypes(include=['category','object','string']).columns)
        for c in cats:
            x[c]=x[c].astype('string').fillna('__MISSING__').astype(str)
            q[c]=q[c].astype('string').fillna('__MISSING__').astype(str)
        x,q=x.fillna(-999.),q.fillna(-999.)
        model=CatBoostRegressor(iterations=iterations or 2200,learning_rate=.04,depth=6,
            l2_leaf_reg=15.,random_strength=.4,random_seed=42,task_type='GPU',devices='0',
            gpu_ram_part=.55,border_count=128,allow_writing_files=False,verbose=False)
        train=Pool(x,y,cat_features=cats,weight=weights)
        options={} if iterations else {'eval_set':Pool(q,target,cat_features=cats,weight=query_weights),
            'early_stopping_rounds':120}
        model.fit(train,**options)
        count=iterations or int(model.get_best_iteration())+1
    pred=np.asarray(model.predict(q),float)
    pred=pred*offset if spec['target']=='ratio' else pred+offset
    return {'model':model,'spec':spec},pred,int(count),x.shape


def run(stage,names=None):
    write_json(OUT/'horizon_plan.json',{'specs':SPECS,'selection':['2023-10','2024-03'],
        'snapshot_delays':[0,3,6,9,12],
        'notes':'Histories freeze at row month minus delay. No historical source record on/after that cutoff may contribute. Each unique supervised row retains unit total weight; all queries retained.'})
    bundle=augmented_data(stage)
    for spec in SPECS:
        name=spec['name']
        if names and name not in names:
            continue
        path=OUT/'horizons'/stage/name
        if (path/'results.json').exists():
            continue
        count=None if stage=='screen' else json.loads((OUT/'horizons/screen'/name/'results.json').read_text())['iterations']
        started=time.perf_counter()
        print('HORIZON',stage,name,flush=True)
        payload,pred,count,shape=fit(spec,bundle,count)
        artifact=ART/'horizons'/stage/name
        artifact.mkdir(parents=True,exist_ok=True)
        joblib.dump(payload,artifact/'model.joblib',compress=1)
        pd.DataFrame({'source_row':bundle['row_val'],'month':bundle['date_val'],
            'actual':bundle['y_val'],'prediction':pred}).to_csv(artifact/'predictions.csv',index=False)
        result={'spec':spec,'stage':stage,'iterations':count,'fit_rows_after_augmentation':shape[0],
            'features':shape[1],'query_rows':len(pred),**score(bundle['y_val'],pred),
            'seconds':time.perf_counter()-started,'source_sha256':bundle['hashes']}
        write_json(path/'results.json',result)
        print('HORIZON RESULT',stage,name,result['rmse_sgd'],flush=True)
        del payload
        gc.collect()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['screen','outer','june'])
    parser.add_argument('--names',nargs='+')
    args=parser.parse_args()
    run(args.stage,args.names)
