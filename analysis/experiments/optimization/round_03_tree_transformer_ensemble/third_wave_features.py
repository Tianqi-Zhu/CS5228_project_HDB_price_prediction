"""Causal weighted calendar-cohort features from train.csv only.

The fitting pool is frozen at the forecast origin. Lag windows are strictly
earlier than each query month, including for fitting rows. Local normalization
uses only the already observed source month, never the query month's labels.
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
from analysis.experiments.frontier_search import frame_data,month_number,keys,score,write_json
from analysis.experiments.second_wave_anniversary import select_inputs
from analysis.experiments.third_wave_training import OUT,ART,make_data


class KernelCohorts:
    def fit(self,source):
        y=source.MONTHLY_RENT.to_numpy(float)
        global_level=source.groupby(['RENT_APPROVAL_DATE','FLAT_TYPE_CLEAN']).MONTHLY_RENT.transform('mean').to_numpy(float)
        local=source.groupby(['RENT_APPROVAL_DATE','PLANNING_AREA_CLEAN','FLAT_TYPE_CLEAN']).MONTHLY_RENT
        local_level=(local.transform('sum').to_numpy()+20*global_level)/(local.transform('count').to_numpy()+20)
        ratio=y/global_level
        premium=y/local_level
        self.tables={}
        for grain,fields in [('BLOCK',('ADDRESS_KEY','FLAT_TYPE_CLEAN')),
            ('STREET',('STREET_CLEAN','FLAT_TYPE_CLEAN'))]:
            raw=pd.DataFrame({'key':keys(source,fields),'month':month_number(source.RENT_APPROVAL_DATE),
                'n':1.,'rent':y,'ratio':ratio,'ratio2':ratio**2,'local':premium,
                'local2':premium**2,'low':(ratio<.5).astype(float),
                'midlow':(ratio<.75).astype(float),'high':(ratio>1.25).astype(float),
                'minimum':ratio,'maximum':ratio})
            self.tables[grain]=raw.groupby(['key','month']).agg({
                **{c:'sum' for c in ('n','rent','ratio','ratio2','local','local2','low','midlow','high')},
                'minimum':'min','maximum':'max'})
        return self

    def transform(self,query):
        months=month_number(query.RENT_APPROVAL_DATE)
        result={}
        for grain,fields in [('BLOCK',('ADDRESS_KEY','FLAT_TYPE_CLEAN')),
            ('STREET',('STREET_CLEAN','FLAT_TYPE_CLEAN'))]:
            identifiers=keys(query,fields)
            for lag in ((12,18,24,30,36) if grain=='BLOCK' else (24,)):
                sums=np.zeros((len(query),9))
                square_n=np.zeros(len(query))
                lo=np.full(len(query),np.inf)
                hi=np.full(len(query),-np.inf)
                for delta in range(-3,4):
                    historical_month=months-lag+delta
                    if np.any(historical_month>=months):
                        raise AssertionError('History window reaches query month')
                    index=pd.MultiIndex.from_arrays([identifiers,historical_month],names=['key','month'])
                    found=self.tables[grain].reindex(index).to_numpy(float)
                    weight=np.exp(-.5*(delta/1.5)**2)
                    sums+=np.nan_to_num(found[:,:9])*weight
                    square_n+=np.nan_to_num(found[:,0])*weight**2
                    lo=np.minimum(lo,np.where(np.isfinite(found[:,9]),found[:,9],np.inf))
                    hi=np.maximum(hi,np.where(np.isfinite(found[:,10]),found[:,10],-np.inf))
                n,rent,r,r2,p,p2,low,midlow,high=sums.T
                denominator=np.where(n>0,n,np.nan)
                prefix=f'KERNEL_{grain}_{lag}'
                values={'N':n,'EFFECTIVE_N':n**2/np.where(square_n>0,square_n,np.nan),
                    'RENT':rent/denominator,'RATIO':r/denominator,
                    'SD':np.sqrt(np.maximum(0,r2/denominator-(r/denominator)**2)),
                    'LOCAL':p/denominator,'LOCAL_SD':np.sqrt(np.maximum(0,p2/denominator-(p/denominator)**2)),
                    'LOW':low/denominator,'MIDLOW':midlow/denominator,'HIGH':high/denominator,
                    'MIN':np.where(np.isfinite(lo),lo,np.nan),'MAX':np.where(np.isfinite(hi),hi,np.nan)}
                for name,value in values.items():
                    result[prefix+'_'+name]=value.astype(np.float32)
        return pd.DataFrame(result,index=query.index)


def feature_data(stage):
    bundle=make_data(stage)
    cache=ART/'feature_cache'/f'{stage}.joblib'
    if cache.exists():
        saved=joblib.load(cache)
        if saved['hashes']==bundle['hashes']:
            bundle.update(saved['features'])
            return bundle
    raw,_=frame_data()
    source=raw.loc[raw.RENT_APPROVAL_DATE<bundle['start']]
    query=raw.loc[(raw.RENT_APPROVAL_DATE>=bundle['start'])&(raw.RENT_APPROVAL_DATE<bundle['end'])]
    history=KernelCohorts().fit(source)
    features={'kernel_fit':history.transform(source),'kernel_val':history.transform(query)}
    cache.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump({'hashes':bundle['hashes'],'features':features},cache,compress=1)
    bundle.update(features)
    return bundle


def inputs(bundle,spec):
    x,q=select_inputs(bundle,'window')
    kind=spec['kind']
    if kind=='replace':
        old=[c for c in x if c.startswith('ANN_')]
        x,q=x.drop(columns=old),q.drop(columns=old)
    additions=[]
    for c in bundle['kernel_fit']:
        if kind in ('wide24','tails24') and not c.startswith('KERNEL_BLOCK_24_'):
            continue
        if kind=='tails24' and not c.endswith(('_LOW','_MIDLOW','_HIGH','_MIN','_MAX','_LOCAL','_LOCAL_SD')):
            continue
        if kind in ('replace','hybrid') and (c.startswith('KERNEL_STREET') or '_18_' in c or '_30_' in c):
            continue
        additions.append(c)
    x=pd.concat([x,bundle['kernel_fit'][additions]],axis=1).copy()
    q=pd.concat([q,bundle['kernel_val'][additions]],axis=1).copy()
    # Express the rent cohort on the current *historically observed* market scale.
    for suffix,frame,extra,kernel in [('fit',x,bundle['extra_fit'],bundle['kernel_fit']),
        ('val',q,bundle['extra_val'],bundle['kernel_val'])]:
        n=kernel.KERNEL_BLOCK_24_EFFECTIVE_N.to_numpy(float)
        prior=extra.BLOCK_RATIO_MEAN.to_numpy(float)
        prior=np.where(np.isfinite(prior),prior,extra.AREA_RATIO_MEAN.to_numpy(float))
        prior=np.where(np.isfinite(prior),prior,1.)
        ratio=kernel.KERNEL_BLOCK_24_RATIO.to_numpy(float)
        n=np.nan_to_num(n)
        shrunk=(np.nan_to_num(ratio)*n+2*prior)/(n+2)
        frame['KERNEL24_SHRUNK']=shrunk.astype(np.float32)
        frame['KERNEL24_ADJUSTED_RENT']=(shrunk*extra.MARKET_3M.to_numpy(float)).astype(np.float32)
        frame['KERNEL24_LOCAL_RENT']=(kernel.KERNEL_BLOCK_24_LOCAL.to_numpy(float)*
            extra.AREA_RATIO_6M.to_numpy(float)*extra.MARKET_3M.to_numpy(float)).astype(np.float32)
    return x,q


SPECS=[{'name':f'{kind}_lgb{leaves}','kind':kind,'family':'lgb','leaves':leaves}
    for kind in ('wide24','tails24','replace','hybrid','full') for leaves in (15,31)]
SPECS += [{'name':f'{kind}_cat{depth}','kind':kind,'family':'cat','depth':depth}
    for kind in ('wide24','tails24','replace','hybrid','full') for depth in (6,8)]


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
        categories=list(x.select_dtypes(include=['category','string','object']).columns)
        for c in categories:
            x[c]=x[c].astype('string').fillna('__MISSING__').astype(str)
            q[c]=q[c].astype('string').fillna('__MISSING__').astype(str)
        x,q=x.fillna(-999.),q.fillna(-999.)
        model=CatBoostRegressor(iterations=iterations or 2200,learning_rate=.04,depth=spec['depth'],
            l2_leaf_reg=10.,random_seed=42,thread_count=4,loss_function='RMSE',
            task_type='GPU',devices='0',gpu_ram_part=.55,border_count=128,
            random_strength=.4,allow_writing_files=False,verbose=False)
        model.fit(x,bundle['y_fit'],cat_features=categories,**({} if iterations else {
            'eval_set':(q,bundle['y_val']),'early_stopping_rounds':120}))
        count=iterations or int(model.get_best_iteration())+1
    return {'model':model,'spec':spec},model.predict(q),int(count),x.shape[1]


def run(stage,names=None):
    write_json(OUT/'features_plan.json',{'specs':SPECS,'selection':['2023-10','2024-03'],
        'hypotheses':['Wider calendar windows can reduce sparse cohort noise.',
            'Local price normalization can separate area price and within-area rent differences.',
            'Historical tail frequencies and effective sample size can describe mixture/reliability.'],
        'data':'train.csv labels; existing static auxiliary features; no external rental targets'})
    bundle=feature_data(stage)
    for spec in SPECS:
        if names and spec['name'] not in names:
            continue
        destination=OUT/'features'/stage/spec['name']
        if (destination/'results.json').exists():
            continue
        iterations=None if stage=='screen' else json.loads((OUT/'features/screen'/spec['name']/'results.json').read_text())['iterations']
        started=time.perf_counter()
        print('FEATURE',stage,spec['name'],flush=True)
        payload,pred,count,n_features=fit(spec,bundle,iterations)
        artifact=ART/'features'/stage/spec['name']
        artifact.mkdir(parents=True,exist_ok=True)
        joblib.dump(payload,artifact/'model.joblib',compress=1)
        pd.DataFrame({'source_row':bundle['row_val'],'month':bundle['date_val'],
            'actual':bundle['y_val'],'prediction':pred}).to_csv(artifact/'predictions.csv',index=False)
        record={'spec':spec,'stage':stage,'iterations':count,'features':n_features,
            'query_rows':len(pred),**score(bundle['y_val'],pred),
            'seconds':time.perf_counter()-started,'source_sha256':bundle['hashes']}
        write_json(destination/'results.json',record)
        print(f"FEATURE RESULT {stage} {spec['name']}: RMSE={record['rmse_sgd']:.4f}, rounds={count}",flush=True)
        del payload
        gc.collect()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['screen','outer','june'])
    parser.add_argument('--names',nargs='+')
    args=parser.parse_args()
    run(args.stage,args.names)
