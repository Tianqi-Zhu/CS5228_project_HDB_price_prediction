"""Test whether prior one/two/three-year rental cohorts help identify renewals.

These are historical calendar cohorts, not identified tenants or apartments.
Queries can only match observed fitting-month records, including long horizons.
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
from analysis.experiments.frontier_search import ROOT,frame_data,month_number,keys,score,write_json
from analysis.experiments.frontier_compact import prepare_compact,ADDITIONS

OUT=ROOT/'output/experiment_runs/round_02_anniversary_history/anniversary'
ART=ROOT/'output/models/round_02_anniversary_history/anniversary'
GRAINS={'BLOCK':('ADDRESS_KEY','FLAT_TYPE_CLEAN'),
    'STREET':('STREET_CLEAN','FLAT_TYPE_CLEAN'),
    'AREA':('PLANNING_AREA_CLEAN','FLAT_TYPE_CLEAN')}


class AnniversaryHistory:
    def fit(self,source):
        months=month_number(source.RENT_APPROVAL_DATE)
        y=source.MONTHLY_RENT.to_numpy(float)
        reference=source.groupby(['RENT_APPROVAL_DATE','FLAT_TYPE_CLEAN']).MONTHLY_RENT.transform('mean').to_numpy(float)
        ratio=y/reference
        self.tables={}
        for grain,columns in GRAINS.items():
            frame=pd.DataFrame({'key':keys(source,columns),'month':months,
                'n':1.,'rent':y,'ratio':ratio,'ratio2':ratio**2,'low':(ratio<.65).astype(float)})
            self.tables[grain]=frame.groupby(['key','month']).sum()
        return self

    def transform(self,query):
        months=month_number(query.RENT_APPROVAL_DATE)
        columns={}
        for grain,fields in GRAINS.items():
            identifiers=keys(query,fields)
            for lag in ((12,24,36) if grain=='BLOCK' else (24,)):
                for radius in ((0,1) if grain=='BLOCK' else (1,)):
                    totals=np.zeros((len(query),5))
                    for delta in range(-radius,radius+1):
                        source_month=months-lag+delta
                        if np.any(source_month>=months):
                            raise AssertionError('Nonhistorical anniversary lookup')
                        index=pd.MultiIndex.from_arrays([identifiers,source_month],names=['key','month'])
                        totals+=self.tables[grain].reindex(index).fillna(0).to_numpy(float)
                    n,rent,ratio,ratio2,low=totals.T
                    divisor=np.where(n>0,n,np.nan)
                    prefix=f'ANN_{grain}_{lag}_R{radius}'
                    columns[prefix+'_N']=n
                    columns[prefix+'_RENT']=rent/divisor
                    columns[prefix+'_RATIO']=ratio/divisor
                    columns[prefix+'_SD']=np.sqrt(np.maximum(0,ratio2/divisor-(ratio/divisor)**2))
                    columns[prefix+'_LOW']=low/divisor
        return pd.DataFrame(columns,index=query.index)


def data(start,end):
    bundle=prepare_compact(start,end)
    source,_=frame_data()
    fit=source.loc[source.RENT_APPROVAL_DATE<start]
    query=source.loc[(source.RENT_APPROVAL_DATE>=start)&(source.RENT_APPROVAL_DATE<end)]
    history=AnniversaryHistory().fit(fit)
    bundle['ann_fit'],bundle['ann_val']=history.transform(fit),history.transform(query)
    return bundle


def select_inputs(bundle,kind):
    x,q=bundle['x_fit'].copy(),bundle['x_val'].copy()
    for c in ADDITIONS['relative_history']:
        x[c]=bundle['extra_fit'][c].to_numpy(np.float32)
        q[c]=bundle['extra_val'][c].to_numpy(np.float32)
    columns=[c for c in bundle['ann_fit'] if (kind=='broad' or
        (c.startswith('ANN_BLOCK') and (('_R0_' in c) if kind=='exact' else ('_R1_' in c))))]
    for c in columns:
        x[c]=bundle['ann_fit'][c].to_numpy(np.float32)
        q[c]=bundle['ann_val'][c].to_numpy(np.float32)
    return x,q


SPECS=[{'name':f'{kind}_lgb{leaves}','kind':kind,'family':'lgb','leaves':leaves}
    for kind in ('exact','window','broad') for leaves in (15,31,63)]
SPECS += [{'name':f'{kind}_cat6','kind':kind,'family':'cat','depth':6} for kind in ('window','broad')]


def run(stage,names=None):
    write_json(OUT/'plan.json',{'trials':SPECS,'selection':['2023-10','2024-03'],
        'outer':['2024-04','2025-03'],'hypothesis':'Prior 12/24/36-month cohorts may capture recurring rental levels. No unit or tenant identity is assumed.'})
    start,end=('2023-10','2024-04') if stage=='screen' else (
        ('2024-04','2025-04') if stage=='outer' else ('2024-06','2025-04'))
    bundle=data(start,end)
    for spec in SPECS:
        if names and spec['name'] not in names:
            continue
        destination=OUT/stage/spec['name']
        if (destination/'results.json').exists():
            continue
        started=time.perf_counter()
        x,q=select_inputs(bundle,spec['kind'])
        iterations=None if stage=='screen' else json.loads((OUT/'screen'/spec['name']/'results.json').read_text())['iterations']
        if spec['family']=='lgb':
            model=LGBMRegressor(n_estimators=iterations or 1800,learning_rate=.03,
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
            model=CatBoostRegressor(iterations=iterations or 1800,learning_rate=.045,depth=6,
                l2_leaf_reg=15.,random_seed=42,thread_count=4,loss_function='RMSE',
                verbose=False,allow_writing_files=False)
            model.fit(x,bundle['y_fit'],cat_features=categories,**({} if iterations else {
                'eval_set':(q,bundle['y_val']),'early_stopping_rounds':100}))
            count=iterations or int(model.get_best_iteration())+1
        predicted=model.predict(q)
        artifact=ART/stage/spec['name']
        artifact.mkdir(parents=True,exist_ok=True)
        joblib.dump({'model':model,'spec':spec},artifact/'model.joblib',compress=1)
        pd.DataFrame({'source_row':bundle['row_val'],'month':bundle['date_val'],'actual':bundle['y_val'],
            'prediction':predicted}).to_csv(artifact/'predictions.csv',index=False)
        record={'name':spec['name'],'spec':spec,'stage':stage,'iterations':count,'features':x.shape[1],
            'rows':len(predicted),**score(bundle['y_val'],predicted),'seconds':time.perf_counter()-started,'source_sha256':bundle['hashes']}
        write_json(destination/'results.json',record)
        print(f"ANNIVERSARY {stage} {spec['name']}: RMSE={record['rmse_sgd']:.4f}",flush=True)
        del model,x,q
        gc.collect()
    rows=[]
    for path in OUT.glob('*/*/results.json'):
        r=json.loads(path.read_text())
        rows.append({k:v for k,v in r.items() if k not in ('spec','source_sha256')})
    if rows:
        pd.DataFrame(rows).sort_values(['stage','rmse_sgd']).to_csv(OUT/'metrics.csv',index=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['screen','outer','june'])
    parser.add_argument('--names',nargs='+')
    args=parser.parse_args()
    run(args.stage,args.names)
