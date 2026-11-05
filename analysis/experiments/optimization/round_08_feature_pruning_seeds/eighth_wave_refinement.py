"""Fixed seed averaging and sparse-history ablations of trained components."""
from __future__ import annotations
import argparse
import gc
import json
import time
import joblib
import numpy as np
import pandas as pd
from analysis.experiments.frontier_search import ROOT,score,write_json
from analysis.experiments.third_wave_training import make_data
from analysis.experiments.second_wave_anniversary import select_inputs

OUT=ROOT/'output/experiment_runs/round_08_feature_pruning_seeds'
ART=ROOT/'output/models/round_08_feature_pruning_seeds'
SEEDS=(43,44,45)


def save(stage,name,bundle,payload,prediction,started,recipe):
    destination=ART/stage/name;destination.mkdir(parents=True,exist_ok=True)
    joblib.dump(payload,destination/'model.joblib',compress=1)
    pd.DataFrame({'source_row':bundle['row_val'],'month':bundle['date_val'],
        'actual':bundle['y_val'],'prediction':prediction}).to_csv(destination/'predictions.csv',index=False)
    record={'stage':stage,'name':name,'recipe':recipe,'rows':len(prediction),
        'fit_rows':len(bundle['y_fit']),**score(bundle['y_val'],prediction),
        'seconds':time.perf_counter()-started,'source_sha256':bundle['hashes']}
    write_json(OUT/stage/name/'results.json',record)
    print('REFINEMENT',stage,name,record['rmse_sgd'],flush=True)
    return record


def bag():
    from catboost import CatBoostRegressor
    write_json(OUT/'bag_plan.json',{'new_seeds':SEEDS,'original_seed':42,
        'iterations':833,'learning_rate':.045,'depth':6,'l2_leaf_reg':15.,
        'engine':'CPU','new_component':'equal mean of seeds 42,43,44,45',
        'replacement':'keep original ensemble weights; replace its CatBoost leaf by the seed mean',
        'selection':'Choose half/full/no replacement on earlier screen only, before later fitting.',
        'note':'Repeated development windows; no later-label stopping or seed selection.'})
    for stage in ('screen','outer','june'):
        bundle=make_data(stage)
        x,q=select_inputs(bundle,'window')
        categories=list(x.select_dtypes(include=['category','object','string']).columns)
        for column in categories:
            x[column]=x[column].astype('string').fillna('__MISSING__').astype(str)
            q[column]=q[column].astype('string').fillna('__MISSING__').astype(str)
        x,q=x.fillna(-999.),q.fillna(-999.)
        original=joblib.load(ROOT/'output/models/round_02_anniversary_history/anniversary'/stage/'window_cat6/model.joblib')['model']
        # Inherit the saved recipe and change only the seed. No eval_set.
        parameters=original.get_params()|{'iterations':833,'thread_count':4}
        del original
        for seed in SEEDS:
            name=f'cat_seed{seed}'
            if (OUT/stage/name/'results.json').exists():continue
            started=time.perf_counter()
            model=CatBoostRegressor(**(parameters|{'random_seed':seed}))
            model.fit(x,bundle['y_fit'],cat_features=categories)
            prediction=model.predict(q)
            save(stage,name,bundle,{'model':model,'columns':list(x),'categories':categories,
                'parameters':parameters|{'random_seed':seed}},prediction,started,
                {'seed':seed,'iterations':833,'features':len(x.columns),'engine':'CPU'})
            del model;gc.collect()
        if stage=='screen':
            select_bag(bundle)


def read_prediction(path,bundle):
    table=pd.read_csv(path).sort_values('source_row')
    np.testing.assert_array_equal(table.source_row,bundle['row_val'])
    np.testing.assert_array_equal(table.actual,bundle['y_val'])
    return table.prediction.to_numpy(float)


def cat_delta(stage,bundle):
    old=read_prediction(ROOT/'output/models/round_02_anniversary_history/anniversary'/stage/'window_cat6/predictions.csv',bundle)
    new=np.mean([old]+[read_prediction(ART/stage/f'cat_seed{seed}/predictions.csv',bundle) for seed in SEEDS],axis=0)
    return new-old


def select_bag(bundle):
    from analysis.experiments.fourth_wave_ensemble import old_best
    best=json.loads((ROOT/'output/experiment_runs/00_report_and_final/best_ensemble.json').read_text(encoding='utf-8'))
    weight=next(m['weight'] for m in best['leaf_members'] if m['artifact_name']=='window_cat6')
    baseline=old_best('screen',bundle);delta=cat_delta('screen',bundle)
    results=[{'replacement_fraction':fraction,**score(bundle['y_val'],baseline+fraction*weight*delta)}
        for fraction in (0.,.5,1.)]
    path=OUT/'bag_selection.json'
    if not path.exists():
        write_json(path,{'selected':min(results,key=lambda r:r['rmse_sgd']),
            'candidates':results,'cat_leaf_weight':weight,'locked_before_later_fits':True})


def deep():
    import torch
    from analysis.experiments import sixth_wave_deep as sixth
    from analysis.experiments.third_wave_models import fit as fit_ft
    from analysis.experiments.third_wave_training import inputs
    sixth.dependencies()
    # The early source has no variable ANN36 fields. Its old encoder already
    # removes all five, so dropping them now yields exactly the old early input.
    early=make_data('screen')
    x,_,_,_=inputs(early,{'adjusted':True,'target':'residual'})
    removed=[c for c in x if c.startswith('ANN_BLOCK_36_')]
    assert len(removed)==5 and all(x[c].nunique(dropna=False)<=1 or not x[c].notna().any() for c in removed)
    write_json(OUT/'deep_plan.json',{'intervention':'Remove five ANN_BLOCK_36 fields before neural encoding.',
        'removed':removed,'early_equivalence_verified':True,
        'early_predictions':'Reuse original screen predictions: original fit-only schema already drops ANN36.',
        'FT_epochs':3,'TabM_seeds':[42,43,44],'TabM_epochs':[18,15,15],
        'later_fixed_epochs':True,'notes':'Early constant-column equivalence is checked, not assumed. Later validation targets are only scored.'})
    for stage in ('outer','june'):
        bundle=make_data(stage)
        specs=[({'name':'ft_no36','family':'ft_transformer','training':'all','adjusted':True,
            'target':'residual','drop_ann36':True},3)]
        base=next(s for s in sixth.SPECS if s['name']=='tabm_ple_residual_lr0.0005')
        specs += [(base|{'name':f'tabm_no36_seed{seed}','seed':seed,'drop_ann36':True},epochs)
            for seed,epochs in zip((42,43,44),(18,15,15))]
        for spec,epochs in specs:
            if (OUT/stage/spec['name']/'results.json').exists():continue
            sixth.seed(spec.get('seed',42));started=time.perf_counter()
            if spec['family']=='ft_transformer':
                payload,prediction,_,_,_=fit_ft(spec,bundle,epochs)
            else:
                payload,prediction,_=sixth.fit_network(spec,bundle,epochs)
            save(stage,spec['name'],bundle,payload,prediction,started,
                {'spec':spec,'epochs':epochs,'features':len(payload['columns'])})
            del payload;gc.collect();torch.cuda.empty_cache()


def ft_bag():
    import torch
    from analysis.experiments.sixth_wave_deep import seed
    from analysis.experiments.third_wave_models import fit
    write_json(OUT/'ft_bag_plan.json',{'new_seeds':[43,44,45],'epochs':3,
        'intervention':'FT no36, equal average of seeds 42,43,44,45',
        'replacement':'keep original FT leaf weight; no seed selection',
        'note':'Additional exploration after seeing seed42 no36 results. All periods are reused development data.'})
    for stage in ('screen','outer','june'):
        bundle=make_data(stage)
        for seed_value in (43,44,45):
            name=f'ft_no36_seed{seed_value}'
            if (OUT/stage/name/'results.json').exists():continue
            seed(seed_value);started=time.perf_counter()
            spec={'name':name,'family':'ft_transformer','training':'all','adjusted':True,
                'target':'residual','drop_ann36':True,'seed':seed_value}
            payload,prediction,_,_,_=fit(spec,bundle,3)
            save(stage,name,bundle,payload,prediction,started,{'spec':spec,'epochs':3,
                'features':len(payload['columns'])})
            del payload;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind',choices=['bag','deep','ft_bag'])
    args=parser.parse_args()
    {'bag':bag,'deep':deep,'ft_bag':ft_bag}[args.kind]()
