"""CPU asymmetric-tree structures alongside the neural GPU experiments."""
from __future__ import annotations
import gc
import json
import time
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from analysis.experiments.frontier_search import ROOT,score,write_json
from analysis.experiments.third_wave_training import make_data
from analysis.experiments.second_wave_anniversary import select_inputs
from analysis.experiments.tenth_wave_structures import OUT,ART

SPECS=[{'name':'cat_depthwise6','grow_policy':'Depthwise','depth':6,'min_data_in_leaf':20},
    {'name':'cat_lossguide64','grow_policy':'Lossguide','depth':8,'max_leaves':64,'min_data_in_leaf':20}]


def inputs(bundle):
    x,q=select_inputs(bundle,'window')
    categories=list(x.select_dtypes(include=['category','object','string']).columns)
    for column in categories:
        x[column]=x[column].astype('string').fillna('__MISSING__').astype(str)
        q[column]=q[column].astype('string').fillna('__MISSING__').astype(str)
    return x.fillna(-999.),q.fillna(-999.),categories


def run():
    write_json(OUT/'tree_plan.json',{'specs':SPECS,'iterations':833,'learning_rate':.045,
        'l2_leaf_reg':15.,'random_seed':42,'thread_count':4,'target':'raw',
        'features':'Same 62 window history inputs as saved symmetric CPU CatBoost control; different from neural adjusted/no36 inputs.',
        'note':'Additional tree structural exploration during GPU neural evaluation. Early screen chooses growth rule; fixed 833 rounds in every origin, no later-label stopping.',
        'source':'https://catboost.ai/docs/en/references/training-parameters/common#grow_policy'})
    for stage in ('screen','outer','june'):
        specs=SPECS if stage=='screen' else [json.loads((OUT/'tree_selection.json').read_text(encoding='utf-8'))['spec']]
        bundle=make_data(stage);x,q,categories=inputs(bundle)
        for spec in specs:
            path=OUT/'trees'/stage/spec['name']/'results.json'
            if path.exists():continue
            started=time.perf_counter()
            parameters={'iterations':833,'learning_rate':.045,'l2_leaf_reg':15.,'random_seed':42,
                'thread_count':4,'loss_function':'RMSE','allow_writing_files':False,'verbose':False,
                **{key:value for key,value in spec.items() if key!='name'}}
            model=CatBoostRegressor(**parameters);model.fit(x,bundle['y_fit'],cat_features=categories)
            prediction=model.predict(q)
            directory=ART/'trees'/stage/spec['name'];directory.mkdir(parents=True,exist_ok=True)
            joblib.dump({'model':model,'spec':spec,'parameters':parameters,'categories':categories},directory/'model.joblib',compress=1)
            replay=joblib.load(directory/'model.joblib')['model'].predict(q)
            np.testing.assert_allclose(replay,prediction,atol=1e-6,rtol=0)
            pd.DataFrame({'source_row':bundle['row_val'],'month':bundle['date_val'],'actual':bundle['y_val'],
                'prediction':prediction}).to_csv(directory/'predictions.csv',index=False)
            record={'stage':stage,'name':spec['name'],'spec':spec,'iterations':833,'features':len(x.columns),
                'fit_rows':len(x),'rows':len(q),**score(bundle['y_val'],prediction),
                'seconds':time.perf_counter()-started,'source_sha256':bundle['hashes']}
            write_json(path,record);print('TREE STRUCTURE',stage,spec['name'],record['rmse_sgd'],flush=True)
            del model;gc.collect()
        if stage=='screen':
            candidates=[json.loads((OUT/'trees'/stage/spec['name']/'results.json').read_text(encoding='utf-8')) for spec in SPECS]
            path=OUT/'tree_selection.json'
            if not path.exists():write_json(path,min(candidates,key=lambda r:r['rmse_sgd']))


if __name__=='__main__':run()
