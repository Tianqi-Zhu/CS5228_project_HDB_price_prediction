"""Choose an MLP training half-life on earlier folds; evaluate and export.

Cleaned fitting rows, three seeds, fixed original architecture and 3 epochs.
Validation metrics are unweighted and retain every validation row. The annual
outer window has been inspected in earlier work and remains development data.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
import torch

from analysis.pipeline.config import ROOT,DATA,TARGET
from analysis.pipeline.data import check_output_path,protect_sources,sha256
from analysis.pipeline.auxiliary import build_auxiliary_features
from analysis.pipeline.evaluation import metrics
from analysis.experiments.cleaned_neural_ensemble import _normalise,training_cleaning_audit
from analysis.experiments.embedding_rent_search import fit_model
from analysis.experiments.recency_weighting import weight_summary
from analysis.experiments.mlp_fullfit_submission import restore_predictions,run as fullfit,write_json

FOLDS=[('early_1','2023-04','2023-10'),('early_2','2023-10','2024-04')]


def evaluate(frame,start,end,half_life,spec,epochs,seeds,directory,save_models=False):
    directory.mkdir(parents=True)
    original=frame.loc[frame.RENT_APPROVAL_DATE<start]
    audit=training_cleaning_audit(original,'obvious_low')
    fitting=original.loc[~audit['remove']].copy()
    validation=frame.loc[(frame.RENT_APPROVAL_DATE>=start)&(frame.RENT_APPROVAL_DATE<end)].copy()
    blind=validation.drop(columns=TARGET)
    parts=[];members=[];started=time.perf_counter()
    for seed in seeds:
        config=dict(spec,seed=seed)
        if half_life is not None:config['half_life']=half_life
        else:config.pop('half_life',None)
        model,meta,_,_,_,prediction,_=fit_model(fitting,blind,config,epochs=epochs)
        if not np.isfinite(prediction).all():raise ValueError('Nonfinite predictions')
        member={'seed':seed,'weight_summary':meta['training_weight_summary']}
        if save_models:
            target=directory/f'seed{seed}';target.mkdir()
            torch.save(model.state_dict(),target/'model.pt');joblib.dump(meta,target/'preprocessor.joblib')
            replay=restore_predictions(target,fitting,blind)
            np.testing.assert_allclose(replay,prediction,atol=1e-6,rtol=0)
            member['reload_max_difference']=float(np.max(abs(replay-prediction)))
        members.append(member);parts.append(prediction)
    values=np.column_stack(parts);prediction=values.mean(1)
    result={'half_life_months':half_life,'start':start,'end_exclusive':end,
            'fitting_rows':len(fitting),'removed_rows':int(audit['remove'].sum()),'rows':len(validation),
            'epochs':epochs,'seeds':seeds,'members':members,
            'weight_summary':weight_summary(fitting.RENT_APPROVAL_DATE,half_life),
            **metrics(validation[TARGET].to_numpy(),prediction),'seconds':time.perf_counter()-started}
    write_json(directory/'results.json',result)
    predictions=pd.DataFrame({'source_row':validation.index,'month':validation.RENT_APPROVAL_DATE.to_numpy(),
                              'actual_sgd':validation[TARGET].to_numpy(),'prediction_sgd':prediction,
                              'ensemble_std_sgd':values.std(1)})
    for i,seed in enumerate(seeds):predictions[f'seed{seed}_prediction_sgd']=values[:,i]
    predictions.to_csv(directory/'predictions.csv',index=False)
    print(f'RESULT start={start}, half_life={half_life}, RMSE={result["rmse_sgd"]:.4f}, MAE={result["mae_sgd"]:.4f}',flush=True)
    return result


def run(out,half_lives):
    out=check_output_path(out)
    if out.exists() and any(out.iterdir()):raise ValueError('Use an empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    baseline=ROOT/'reports/experiments/mlp_summary/uncertainty'
    original=json.loads((baseline/'protocol.json').read_text())
    spec=dict(original['specs']['mlp']);epochs=int(json.loads((baseline/'mlp_selected.json').read_text())['epochs'])
    seeds=original['seeds']
    sources=[DATA/'train.csv',DATA/'test.csv',DATA/'example-submission.csv',*sorted((DATA/'auxiliary').glob('*.csv'))]
    with protect_sources(sources):
        protocol={'half_lives_months':[None,*half_lives],'selection_folds':FOLDS,
                  'outer':['2024-04','2025-03'],'cleaning_policy':'obvious_low','spec':spec,
                  'epochs':epochs,'seeds':seeds,'selection_metric':'Row-pooled unweighted RMSE of the two non-overlapping earlier validation windows.',
                  'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in sources},
                  'limits':['All local windows have been examined in earlier project work; this is development evidence.',
                            'Only training loss weights change. History aggregates and numeric preprocessing keep their original definitions.',
                            'The auxiliary price-band loss receives the same sample weights as regression.',
                            'Best positive half-life is exported for comparison even if uniform weighting wins local selection.',
                            'No new Kaggle submission or test score is obtained automatically.']}
        write_json(out/'protocol.json',protocol)
        for name in ['mlp_recency_experiment.py','recency_weighting.py','embedding_rent_search.py','mlp_fullfit_submission.py','adjusted_rent_history.py']:
            source=ROOT/'analysis/experiments'/name
            target=out/'source'/name;target.parent.mkdir(exist_ok=True)
            target.write_bytes(source.read_bytes())
        frame=_normalise(build_auxiliary_features(pd.read_csv(DATA/'train.csv')))
        rows=[]
        for fold,start,end in FOLDS:
            for h in [None,*half_lives]:
                name='uniform' if h is None else f'h{h:g}'
                print('TRAIN_FOLD',fold,name,flush=True)
                result=evaluate(frame,start,end,h,spec,epochs,seeds,out/'selection'/fold/name)
                result['fold']=fold;rows.append(result)
                pd.DataFrame([{k:v for k,v in r.items() if k not in ['members','weight_summary']} for r in rows]).to_csv(out/'selection_fold_metrics.csv',index=False)
        pooled=[]
        for h in [None,*half_lives]:
            parts=[r for r in rows if r['half_life_months']==h]
            n=np.array([r['rows'] for r in parts])
            pooled.append({'half_life_months':h,'rows':int(n.sum()),
                           'rmse_sgd':float(np.sqrt(np.average([r['rmse_sgd']**2 for r in parts],weights=n))),
                           'mae_sgd':float(np.average([r['mae_sgd'] for r in parts],weights=n)),
                           'bias_sgd':float(np.average([r['bias_sgd'] for r in parts],weights=n))})
        weighted=min([p for p in pooled if p['half_life_months'] is not None],key=lambda p:p['rmse_sgd'])
        best=min(pooled,key=lambda p:p['rmse_sgd'])
        h=weighted['half_life_months']
        selection={'pooled_metrics':pooled,'best_overall_half_life_months':best['half_life_months'],
                   'best_weighted_half_life_months':h,'locked_before_outer_evaluation':True}
        write_json(out/'selection.json',selection)
        pd.DataFrame(pooled).to_csv(out/'pooled_selection_metrics.csv',index=False)
        print('LOCKED_SELECTION',json.dumps(selection),flush=True)
        outer=[]
        for half_life in [None,h]:
            name='uniform' if half_life is None else f'h{half_life:g}'
            outer.append(evaluate(frame,'2024-04','2025-04',half_life,spec,epochs,seeds,out/'outer'/name,save_models=True))
        pd.DataFrame([{k:v for k,v in r.items() if k not in ['members','weight_summary']} for r in outer]).to_csv(out/'outer_comparison.csv',index=False)
        write_json(out/'evaluation_summary.json',{'selected_weighted_half_life_months':h,'earlier_selection':selection,
                   'outer_baseline':outer[0],'outer_weighted':outer[1],
                   'outer_rmse_reduction_sgd':outer[0]['rmse_sgd']-outer[1]['rmse_sgd'],
                   'note':'Later annual development comparison; no hidden test labels.'})
        fullfit(out/'fullfit',baseline,h)
        write_json(out/'completion.json',{'completed':True,'weighted_half_life_months':h,
                   'selection_model_fits':len(FOLDS)*(len(half_lives)+1)*len(seeds),
                   'outer_model_fits':2*len(seeds),'fullfit_model_fits':2*len(seeds),
                   'source_hashes_unchanged':all(sha256(ROOT/p)==h for p,h in protocol['source_sha256'].items()),
                   'kaggle_score':None,'kaggle_upload_performed':False})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--half-lives',type=float,nargs='+',default=[6,12,24,36])
    p.add_argument('--threads',type=int,default=2)
    a=p.parse_args()
    if not a.half_lives or any(not np.isfinite(h) or h<=0 for h in a.half_lives) or len(set(a.half_lives))!=len(a.half_lives):
        p.error('Half-lives must be distinct finite positive month counts')
    torch.set_num_threads(a.threads);run(a.output_dir,a.half_lives)


if __name__=='__main__':main()
