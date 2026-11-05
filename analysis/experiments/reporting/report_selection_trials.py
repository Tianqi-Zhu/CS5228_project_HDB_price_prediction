"""Fill report comparisons and run a bounded, pre-origin optimization screen."""
from __future__ import annotations
import argparse
import gc
import json
import time
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from analysis.experiments.frontier_search import ROOT,score,write_json
from analysis.experiments.second_wave_anniversary import select_inputs,data as feature_data
from analysis.experiments.third_wave_training import make_data

OUT=ROOT/'output/experiment_runs/00_report_and_final'
ART=ROOT/'output/models/00_report_and_final'
OPT=[{'name':f'cat_depth{depth}_l2{l2}','family':'cat','depth':depth,'l2':l2,
    'gpu':True} for depth,l2 in [(4,10),(5,10),(7,10),(9,10),(6,40),(8,40)]]
OPT += [{'name':f'cat_ordered{depth}','family':'cat','depth':depth,'l2':20.,
    'gpu':False,'boosting':'Ordered'} for depth in (4,6)]
OPT += [{'name':f'lgb_extra{leaves}','family':'lgb','leaves':leaves} for leaves in (15,31)]


def save(kind,stage,name,bundle,pred,payload,record):
    destination=ART/kind/stage/name
    destination.mkdir(parents=True,exist_ok=True)
    joblib.dump(payload,destination/'model.joblib',compress=1)
    pd.DataFrame({'source_row':bundle['row_val'],'month':bundle['date_val'],
        'actual':bundle['y_val'],'prediction':pred}).to_csv(destination/'predictions.csv',index=False)
    write_json(OUT/kind/stage/name/'results.json',record|{'stage':stage,'name':name,
        'rows':len(pred),'source_sha256':bundle['hashes'],**score(bundle['y_val'],pred)})
    print(kind,stage,name,score(bundle['y_val'],pred)['rmse_sgd'],flush=True)


def select_features(bundle,level):
    x,q=select_inputs(bundle,'window')
    if level==62:
        return x,q
    columns=list(bundle['x_fit']) if level==27 else [c for c in x if not c.startswith('ANN_')]
    if level==19:
        columns=[c for c in bundle['x_fit'] if not c.startswith('HIST_')]
    x,q=x[columns].copy(),q[columns].copy()
    assert x.shape[1]==level
    return x,q


def ladder(stage):
    # A feature-only comparison does not consume the old ensemble's fallback
    # predictions. Rebuild its identical feature pool directly from raw data.
    start,end={'screen':('2023-10','2024-04'),'outer':('2024-04','2025-04'),'june':('2024-06','2025-04')}[stage]
    bundle=feature_data(start,end)
    # Chosen in the previous chronological screen; held constant across feature
    # sets. This isolates feature effects from differing stopping times.
    for family,count in [('lgb',353),('cat',535)]:
        for level in (19,27,47,62):
            name=f'{family}_f{level}'
            if (OUT/'ladder'/stage/name/'results.json').exists():
                continue
            started=time.perf_counter()
            x,q=select_features(bundle,level)
            if family=='lgb':
                model=LGBMRegressor(n_estimators=count,learning_rate=.03,num_leaves=15,
                    min_child_samples=40,reg_lambda=10.,random_state=42,n_jobs=4,verbosity=-1)
                model.fit(x,bundle['y_fit'])
            else:
                from catboost import CatBoostRegressor
                cats=list(x.select_dtypes(include=['category','object','string']).columns)
                for c in cats:
                    x[c]=x[c].astype('string').fillna('__MISSING__').astype(str)
                    q[c]=q[c].astype('string').fillna('__MISSING__').astype(str)
                x,q=x.fillna(-999.),q.fillna(-999.)
                model=CatBoostRegressor(iterations=count,learning_rate=.04,depth=6,
                    l2_leaf_reg=10.,random_strength=.4,random_seed=42,task_type='GPU',
                    devices='0',gpu_ram_part=.55,border_count=128,allow_writing_files=False,verbose=False)
                model.fit(x,bundle['y_fit'],cat_features=cats)
            save('ladder',stage,name,bundle,model.predict(q),{'model':model,'level':level,'family':family},
                {'features':level,'iterations':count,'family':family,'seconds':time.perf_counter()-started,
                    'notes':'Nested features, identical rows and identical hyperparameters/rounds within each model. No stage labels used for stopping.'})
            del model,x,q
            gc.collect()


def ridge():
    selection_path=OUT/'ridge_selection.json'
    if not selection_path.exists():
        bundle=make_data('screen')
        records=[]
        for alpha in (1.,100.,10000.):
            x,q=bundle['x_fit'].copy(),bundle['x_val'].copy()
            cats=list(x.select_dtypes(include=['category','object','string']).columns)
            nums=[c for c in x if c not in cats]
            for c in cats:
                x[c]=x[c].astype('string').fillna('__MISSING__').astype(str)
                q[c]=q[c].astype('string').fillna('__MISSING__').astype(str)
            preprocessing=ColumnTransformer([
                ('numeric',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),nums),
                ('category',OneHotEncoder(handle_unknown='ignore'),cats)])
            model=make_pipeline(preprocessing,Ridge(alpha=alpha,solver='lsqr'))
            model.fit(x,bundle['y_fit'])
            pred=model.predict(q)
            name=f'ridge_{int(alpha)}'
            record={'alpha':alpha,'features':27,'selection_months':['2023-10','2024-03']}
            save('ridge','screen',name,bundle,pred,{'model':model},record)
            records.append(record|score(bundle['y_val'],pred))
        selected=min(records,key=lambda r:r['rmse_sgd'])
        write_json(selection_path,selected|{'trials':records})
    selected=json.loads(selection_path.read_text())
    for stage in ('outer','june'):
        name='ridge27'
        if (OUT/'ridge'/stage/name/'results.json').exists():
            continue
        bundle=make_data(stage)
        x,q=bundle['x_fit'].copy(),bundle['x_val'].copy()
        cats=list(x.select_dtypes(include=['category','object','string']).columns)
        nums=[c for c in x if c not in cats]
        for c in cats:
            x[c]=x[c].astype('string').fillna('__MISSING__').astype(str)
            q[c]=q[c].astype('string').fillna('__MISSING__').astype(str)
        preprocessing=ColumnTransformer([
            ('numeric',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),nums),
            ('category',OneHotEncoder(handle_unknown='ignore'),cats)])
        model=make_pipeline(preprocessing,Ridge(alpha=selected['alpha'],solver='lsqr'))
        model.fit(x,bundle['y_fit'])
        save('ridge',stage,name,bundle,model.predict(q),{'model':model},
            {'alpha':selected['alpha'],'features':27,'selection_months':['2023-10','2024-03']})


def optimize(stage,names=None):
    write_json(OUT/'optimization_plan.json',{'specs':OPT,'selection':['2023-10','2024-03'],
        'features':62,'data':'train.csv + supplied static auxiliary',
        'notes':'Bounded architecture/regularization exploration. Fixed iteration refits at later origins; all validation rows retained.'})
    bundle=make_data(stage)
    for spec in OPT:
        name=spec['name']
        if names and name not in names:
            continue
        if (OUT/'optimization'/stage/name/'results.json').exists():
            continue
        count=None if stage=='screen' else json.loads((OUT/'optimization/screen'/name/'results.json').read_text())['iterations']
        x,q=select_inputs(bundle,'window')
        started=time.perf_counter()
        print('OPTIMIZE',stage,name,flush=True)
        if spec['family']=='cat':
            from catboost import CatBoostRegressor
            cats=list(x.select_dtypes(include=['category','object','string']).columns)
            for c in cats:
                x[c]=x[c].astype('string').fillna('__MISSING__').astype(str)
                q[c]=q[c].astype('string').fillna('__MISSING__').astype(str)
            x,q=x.fillna(-999.),q.fillna(-999.)
            model=CatBoostRegressor(iterations=count or 2000,learning_rate=.04,depth=spec['depth'],
                l2_leaf_reg=spec['l2'],random_strength=.4,random_seed=42,thread_count=4,
                task_type='GPU' if spec['gpu'] else 'CPU',
                boosting_type=spec.get('boosting','Plain'),
                **({'devices':'0','gpu_ram_part':.55,'border_count':128} if spec['gpu'] else {}),
                allow_writing_files=False,verbose=False)
            model.fit(x,bundle['y_fit'],cat_features=cats,**({} if count else {
                'eval_set':(q,bundle['y_val']),'early_stopping_rounds':120}))
            count=count or int(model.get_best_iteration())+1
        else:
            from lightgbm import early_stopping
            model=LGBMRegressor(n_estimators=count or 2200,learning_rate=.025,
                num_leaves=spec['leaves'],min_child_samples=60,reg_lambda=25.,
                extra_trees=True,colsample_bytree=.8,subsample=.85,subsample_freq=1,
                random_state=42,n_jobs=4,verbosity=-1)
            model.fit(x,bundle['y_fit'],**({} if count else {'eval_set':[(q,bundle['y_val'])],
                'callbacks':[early_stopping(120,verbose=False)]}))
            count=count or int(model.best_iteration_)
        save('optimization',stage,name,bundle,model.predict(q),{'model':model,'spec':spec},
            {'spec':spec,'features':62,'iterations':int(count),'seconds':time.perf_counter()-started})
        del model,x,q
        gc.collect()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task',choices=['ridge','ladder','optimize'])
    parser.add_argument('--stage',choices=['screen','outer','june'],default='screen')
    parser.add_argument('--names',nargs='+')
    args=parser.parse_args()
    if args.task=='ridge':
        ridge()
    elif args.task=='ladder':
        ladder(args.stage)
    else:
        optimize(args.stage,args.names)
