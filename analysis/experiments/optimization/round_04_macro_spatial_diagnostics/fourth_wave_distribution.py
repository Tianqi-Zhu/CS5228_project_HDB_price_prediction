"""Predict rent-band probabilities and integrate their conditional means."""
from __future__ import annotations
import argparse
import gc
import json
import time
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier,early_stopping
from analysis.experiments.frontier_search import frame_data,score,write_json
from analysis.experiments.second_wave_anniversary import select_inputs
from analysis.experiments.third_wave_training import make_data,adjusted_baseline
from analysis.experiments.fourth_wave_diagnostics import OUT,ART

SPECS=[{'name':f'{mode}_lgb{leaves}','mode':mode,'leaves':leaves}
    for mode in ('ratio','raw') for leaves in (15,31)]
SPECS += [{'name':f'{mode}_lgb{leaves}','mode':mode,'leaves':leaves}
    for mode in ('local_ratio','coarse_ratio') for leaves in (15,31)]


def distribution_data(stage):
    bundle=make_data(stage)
    raw,_=frame_data()
    source=raw.loc[raw.RENT_APPROVAL_DATE<bundle['start']]
    bundle['reference']=source.groupby(['RENT_APPROVAL_DATE','FLAT_TYPE_CLEAN']).MONTHLY_RENT.transform('mean').to_numpy(float)
    return bundle


def fit(spec,bundle,iterations=None):
    x,q=select_inputs(bundle,'window')
    if spec['mode'] in ('ratio','coarse_ratio'):
        target=bundle['y_fit']/bundle['reference']
        scale=bundle['extra_val'].MARKET_3M.to_numpy(float)
        boundaries=np.arange(.4,1.71,.2 if spec['mode']=='coarse_ratio' else .1)
        weights=(bundle['reference']/3000.)**2
    elif spec['mode']=='local_ratio':
        reference,_=adjusted_baseline(bundle['ann_fit'],bundle['extra_fit'],x)
        scale,_=adjusted_baseline(bundle['ann_val'],bundle['extra_val'],q)
        target=bundle['y_fit']/reference
        boundaries=np.arange(.4,1.71,.1)
        weights=(reference/3000.)**2
    else:
        target=bundle['y_fit'].copy()
        scale=np.ones(len(q))
        boundaries=np.arange(1200.,5100.,300.)
        weights=np.ones(len(target))
    labels=np.digitize(target,boundaries)
    # Fitting-only means preserve expected rent inside the tail bins too.
    classes=np.unique(labels)
    centers=np.array([np.average(target[labels==k],weights=weights[labels==k]) for k in classes])
    mapping={k:i for i,k in enumerate(classes)}
    labels=np.array([mapping[k] for k in labels])
    query_labels=np.clip(np.digitize(bundle['y_val']/scale,boundaries),classes.min(),classes.max())
    query_labels=np.array([mapping.get(k,0) for k in query_labels])
    def rent_rmse(_labels,probabilities):
        prediction=probabilities@centers*scale
        return 'rent_rmse_sgd',score(bundle['y_val'],prediction)['rmse_sgd'],False
    model=LGBMClassifier(n_estimators=iterations or 1800,learning_rate=.035,
        num_leaves=spec['leaves'],min_child_samples=60,reg_lambda=20.,
        random_state=42,n_jobs=4,verbosity=-1,objective='multiclass',metric='None')
    model.fit(x,labels,sample_weight=weights,**({} if iterations else {
        'eval_set':[(q,query_labels)],'eval_metric':rent_rmse,'callbacks':[early_stopping(100,verbose=False)]}))
    count=iterations or int(model.best_iteration_)
    prediction=model.predict_proba(q)@centers*scale
    return {'model':model,'spec':spec,'centers':centers,'boundaries':boundaries},prediction,int(count)


def run(stage,names=None):
    write_json(OUT/'distribution_plan.json',{'specs':SPECS,'selection':['2023-10','2024-03'],
        'notes':'Fit rent-band probabilities, integrate fitting-only within-bin means; train flat-type/month normalization only uses fitting labels. Earlier stopping uses rent RMSE, not classification accuracy. All queries retained.'})
    bundle=distribution_data(stage)
    for spec in SPECS:
        name=spec['name']
        if names and name not in names:
            continue
        path=OUT/'distribution'/stage/name
        if (path/'results.json').exists():
            continue
        count=None if stage=='screen' else json.loads((OUT/'distribution/screen'/name/'results.json').read_text())['iterations']
        started=time.perf_counter()
        print('DISTRIBUTION',stage,name,flush=True)
        payload,pred,count=fit(spec,bundle,count)
        artifact=ART/'distribution'/stage/name
        artifact.mkdir(parents=True,exist_ok=True)
        joblib.dump(payload,artifact/'model.joblib',compress=1)
        pd.DataFrame({'source_row':bundle['row_val'],'month':bundle['date_val'],
            'actual':bundle['y_val'],'prediction':pred}).to_csv(artifact/'predictions.csv',index=False)
        result={'spec':spec,'stage':stage,'iterations':count,'query_rows':len(pred),
            **score(bundle['y_val'],pred),'seconds':time.perf_counter()-started,'source_sha256':bundle['hashes']}
        write_json(path/'results.json',result)
        print('DISTRIBUTION RESULT',stage,name,result['rmse_sgd'],flush=True)
        del payload
        gc.collect()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['screen','outer','june'])
    parser.add_argument('--names',nargs='+')
    args=parser.parse_args()
    run(args.stage,args.names)
