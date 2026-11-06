"""Calendar-matched spatial comparable rents from earlier observed records."""
from __future__ import annotations
import argparse
import gc
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from lightgbm import LGBMRegressor,early_stopping
from analysis.experiments.frontier_search import frame_data,month_number,score,write_json
from analysis.experiments.second_wave_anniversary import select_inputs
from analysis.experiments.third_wave_training import make_data
from analysis.experiments.fourth_wave_diagnostics import OUT,ART


def coordinates(frame):
    return np.column_stack([frame.LATITUDE.to_numpy(float)*111.,frame.LONGITUDE.to_numpy(float)*111.,
        frame.FLOOR_AREA_SQM.to_numpy(float)/20.,frame.LEASE_COMMENCE_DATE.to_numpy(float)/20.])


class CalendarComparables:
    def fit(self,source):
        self.source=source.copy()
        self.months=month_number(source.RENT_APPROVAL_DATE)
        self.coordinates=coordinates(source)
        self.types=source.FLAT_TYPE_CLEAN.to_numpy(str)
        self.addresses=source.ADDRESS_KEY.to_numpy(str)
        level=source.groupby(['RENT_APPROVAL_DATE','FLAT_TYPE_CLEAN']).MONTHLY_RENT.transform('mean').to_numpy(float)
        self.ratio=source.MONTHLY_RENT.to_numpy(float)/level
        return self

    def transform(self,query,cutoff,exclude_same_block=True):
        cutoff=np.asarray(cutoff)
        months=month_number(query.RENT_APPROVAL_DATE)
        if np.any(cutoff>months):
            raise ValueError('Cutoff follows target month')
        kinds=query.FLAT_TYPE_CLEAN.to_numpy(str)
        address=query.ADDRESS_KEY.to_numpy(str)
        coords=coordinates(query)
        result={}
        for lag in (12,24,36):
            arrays={k:np.full((len(query),6),np.nan) for k in (8,32,96)}
            for kind,month,cut in set(zip(kinds,months,cutoff)):
                selected=(kinds==kind)&(months==month)&(cutoff==cut)&np.isfinite(coords).all(axis=1)
                available=(self.types==kind)&(np.abs(self.months-(month-lag))<=1)&(self.months<cut)
                available&=np.isfinite(self.coordinates).all(axis=1)
                source_rows=np.flatnonzero(available)
                positions=np.flatnonzero(selected)
                if not len(source_rows) or not len(positions):
                    continue
                tree=NearestNeighbors(n_neighbors=min(160,len(source_rows)),algorithm='kd_tree',n_jobs=1)
                tree.fit(self.coordinates[source_rows])
                distance,neighbors=tree.kneighbors(coords[positions])
                neighbors=source_rows[neighbors]
                if exclude_same_block:
                    distance=np.where(self.addresses[neighbors]==address[positions,None],np.inf,distance)
                    order=np.argsort(distance,axis=1,kind='stable')
                    distance=np.take_along_axis(distance,order,axis=1)
                    neighbors=np.take_along_axis(neighbors,order,axis=1)
                for k in arrays:
                    d=distance[:,:k]
                    rows=neighbors[:,:k]
                    finite=np.isfinite(d)
                    weight=np.where(finite,1/(.15+d)**2,0.)
                    total=weight.sum(axis=1)
                    divisor=np.where(total>0,total,np.nan)
                    weight=weight/divisor[:,None]
                    ratio=self.ratio[rows]
                    mean=(weight*ratio).sum(axis=1)
                    sd=np.sqrt((weight*(ratio-mean[:,None])**2).sum(axis=1))
                    low=(weight*(ratio<.65)).sum(axis=1)
                    effective=1/np.square(weight).sum(axis=1)
                    distance_mean=(weight*np.where(finite,d,0.)).sum(axis=1)
                    arrays[k][positions]=np.column_stack([mean,sd,low,effective,distance_mean,finite.sum(axis=1)])
            for k,values in arrays.items():
                for index,name in enumerate(('RATIO','SD','LOW','EFFECTIVE_N','DISTANCE','N')):
                    result[f'SPATIAL_COHORT_{lag}_K{k}_{name}']=values[:,index].astype(np.float32)
        return pd.DataFrame(result,index=query.index)


def spatial_data(stage):
    bundle=make_data(stage)
    cache=ART/'spatial_cache'/f'{stage}.joblib'
    if cache.exists():
        saved=joblib.load(cache)
        if saved['hashes']==bundle['hashes']:
            bundle.update(saved['features'])
            return bundle
    raw,_=frame_data()
    source=raw.loc[raw.RENT_APPROVAL_DATE<bundle['start']]
    query=raw.loc[(raw.RENT_APPROVAL_DATE>=bundle['start'])&(raw.RENT_APPROVAL_DATE<bundle['end'])]
    history=CalendarComparables().fit(source)
    print('SPATIAL COHORTS',stage,'training',flush=True)
    x=history.transform(source,month_number(source.RENT_APPROVAL_DATE))
    print('SPATIAL COHORTS',stage,'query',flush=True)
    q=history.transform(query,np.full(len(query),bundle['origin']))
    features={'spatial_fit':x,'spatial_val':q}
    cache.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump({'hashes':bundle['hashes'],'features':features},cache,compress=1)
    bundle.update(features)
    return bundle


def inputs(bundle,spec):
    x,q=select_inputs(bundle,'window')
    columns=[c for c in bundle['spatial_fit'] if (spec['kind']=='all' or
        (c.startswith('SPATIAL_COHORT_24_') and (spec['kind']=='year24' or '_K32_' in c)))]
    x=pd.concat([x,bundle['spatial_fit'][columns]],axis=1)
    q=pd.concat([q,bundle['spatial_val'][columns]],axis=1)
    return x,q


SPECS=[{'name':f'{kind}_{family}{size}','kind':kind,'family':family,
    'leaves' if family=='lgb' else 'depth':size}
    for kind in ('compact24','year24','all') for family,size in [('lgb',15),('lgb',31),('cat',6),('cat',8)]]


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


def run(stage,names=None,prepare_only=False):
    write_json(OUT/'spatial_plan.json',{'specs':SPECS,'selection':['2023-10','2024-03'],
        'notes':'Other-building comparable rents matched to 12/24/36-month historical windows, radius one month. Coordinates/area/lease-age matching; same-block rows excluded. No target-month source rents.'})
    bundle=spatial_data(stage)
    if prepare_only:
        return
    for spec in SPECS:
        name=spec['name']
        if names and name not in names:
            continue
        path=OUT/'spatial'/stage/name
        if (path/'results.json').exists():
            continue
        count=None if stage=='screen' else json.loads((OUT/'spatial/screen'/name/'results.json').read_text())['iterations']
        print('SPATIAL',stage,name,flush=True)
        started=time.perf_counter()
        payload,pred,count,features=fit(spec,bundle,count)
        artifact=ART/'spatial'/stage/name
        artifact.mkdir(parents=True,exist_ok=True)
        joblib.dump(payload,artifact/'model.joblib',compress=1)
        pd.DataFrame({'source_row':bundle['row_val'],'month':bundle['date_val'],
            'actual':bundle['y_val'],'prediction':pred}).to_csv(artifact/'predictions.csv',index=False)
        result={'spec':spec,'stage':stage,'iterations':count,'features':features,'query_rows':len(pred),
            **score(bundle['y_val'],pred),'seconds':time.perf_counter()-started,'source_sha256':bundle['hashes']}
        write_json(path/'results.json',result)
        print('SPATIAL RESULT',stage,name,result['rmse_sgd'],flush=True)
        del payload
        gc.collect()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['screen','outer','june'])
    parser.add_argument('--names',nargs='+')
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    run(args.stage,args.names,args.prepare_only)
