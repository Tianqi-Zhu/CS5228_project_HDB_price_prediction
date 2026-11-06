"""Refit the existing MLP ensemble on all supplied labels, then export test CSVs.

No Kaggle upload, test-label evaluation, hyperparameter selection, or reuse of
the old prediction-interval calibrator. Historical query features stay prior
to each query month, including March 2025's train/test month overlap.
"""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

from analysis.pipeline.config import DATA, ROOT, TARGET
from analysis.pipeline.data import check_output_path, protect_sources, sha256
from analysis.pipeline.auxiliary import build_auxiliary_features
from analysis.experiments.adjusted_rent_history import adjusted_history, adjusted_reference
from analysis.experiments.cleaned_neural_ensemble import _normalise, training_cleaning_audit
from analysis.experiments.deep_learning import build_network, ExperimentConfig
from analysis.experiments.embedding_rent_search import fit_model, predict
from analysis.experiments.neural_ensemble_uncertainty import ensemble_summary
from analysis.experiments.recency_weighting import recency_weights,weight_summary


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def restore_predictions(directory, fitting, query):
    """Reconstruct the model and transform queries with its saved preprocessor."""
    meta=joblib.load(directory/'preprocessor.joblib')
    spec=meta['config']
    query=query.drop(columns=TARGET,errors='ignore').copy()
    if spec.get('adjusted_history'):
        _,history=adjusted_history(fitting,query)
        query=query.join(history)
    cats=np.column_stack([pd.Index(meta['vocab'][c]).get_indexer(query[c])+1
                          for c in meta['categories']])
    imputed=meta['imputer'].transform(query[meta['numeric']])
    nums=meta['scaler'].transform(imputed).astype('float32')
    if meta['raw_time_scaler'] is not None:
        column=meta['numeric'].index('APPROVAL_MONTH_INDEX')
        nums=np.column_stack([nums,meta['raw_time_scaler'].transform(imputed[:,[column]])]).astype('float32')
    config=ExperimentConfig(architecture=spec['architecture'],width=spec.get('width',128),
                            blocks=spec['blocks'],dropout=spec.get('dropout',.1))
    model=build_network([len(meta['vocab'][c])+1 for c in meta['categories']],nums.shape[1],config)
    if spec.get('aux_weight'):
        model.output[-1]=nn.Linear(model.output[-1].in_features,4)
    model.load_state_dict(torch.load(directory/'model.pt',map_location='cpu',weights_only=True))
    offset=adjusted_reference(query) if spec.get('residual_reference') else 0.
    return predict(model,torch.as_tensor(cats),torch.as_tensor(nums))*meta['target_std']+meta['target_mean']+offset


def validate_submission(sample, prediction):
    values=np.asarray(prediction,float)
    if list(sample.columns)!=['Id','Predicted'] or len(sample)!=50000:
        raise ValueError('Expected original 50,000-row Id,Predicted template')
    if not np.array_equal(sample.Id.to_numpy(),np.arange(len(sample))):
        raise ValueError('Template ID order differs')
    if values.shape!=(len(sample),) or not np.isfinite(values).all() or (values<=0).any():
        raise ValueError('Invalid or incomplete rent predictions')
    result=sample.copy();result['Predicted']=values
    return result


def run(out, baseline,half_life_months=None):
    out=check_output_path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new empty output directory')
    original=json.loads((baseline/'protocol.json').read_text())
    spec=dict(original['specs']['mlp'])
    if half_life_months is not None:
        recency_weights(['2025-03'],half_life_months)
        spec['half_life']=float(half_life_months)
    epochs=int(json.loads((baseline/'mlp_selected.json').read_text())['epochs'])
    seeds=original['seeds']
    if spec.get('history') or not spec.get('adjusted_history'):
        raise ValueError('This exporter expects the existing adjusted-history MLP')
    out.mkdir(parents=True,exist_ok=True)
    sources=[DATA/'train.csv',DATA/'test.csv',DATA/'example-submission.csv',*sorted((DATA/'auxiliary').glob('*.csv'))]
    with protect_sources(sources):
        hashes={str(p.relative_to(ROOT)):sha256(p) for p in sources}
        code=[Path(__file__),Path(__file__).resolve().parents[2]/'analysis/experiments/adjusted_rent_history.py',
              Path(__file__).resolve().parents[2]/'analysis/experiments/embedding_rent_search.py',
              Path(__file__).resolve().parents[2]/'analysis/experiments/cleaned_neural_ensemble.py',
              Path(__file__).resolve().parents[2]/'analysis/experiments/spatiotemporal_hybrid.py',
              Path(__file__).resolve().parents[2]/'analysis/experiments/deep_learning.py',Path(__file__).resolve().parents[2]/'analysis/pipeline/auxiliary.py',
              Path(__file__).resolve().parents[2]/'analysis/experiments/recency_weighting.py']
        protocol={'spec':spec,'epochs':epochs,'seeds':seeds,'policies':['uncleaned','obvious_low'],
                  'fitting_months':['2021-01','2025-03'],'source_sha256':hashes,
                  'code_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in code},
                  'versions':{p:version(p) for p in ['numpy','pandas','scikit-learn','torch']},
                  'selection':'Keep the earlier selected architecture, optimizer, 3 epochs and three seeds; optional training half-life is supplied explicitly.',
                  'recency_half_life_months':half_life_months,
                  'recency_weighting':'Mean-one 2 ** (-months_before_last_fitting_month / half_life); weights apply to regression and auxiliary classification losses. None means uniform.',
                  'information_boundary':'Test labels absent; each test history uses only fitting months strictly before that query month.',
                  'supervised_overlap_note':'Full-label model fitting includes March 2025; March 2025 test histories exclude that month. This overlap subset is not wholly ex ante.',
                  'uncertainty':'Seed standard deviation is model disagreement, not a calibrated prediction interval.',
                  'time_encoding':'Original quantile preprocessing retained to isolate the full-data refit; no raw-time change.',
                  'test_rmse_sgd':None,'kaggle_upload_performed':False}
        write_json(out/'protocol.json',protocol)
        for source in code:
            saved=out/'source'/source.relative_to(ROOT);saved.parent.mkdir(parents=True,exist_ok=True)
            saved.write_bytes(source.read_bytes())
        raw=pd.read_csv(DATA/'train.csv')
        raw_test=pd.read_csv(DATA/'test.csv')
        if len(raw)!=150000 or TARGET not in raw or TARGET in raw_test:
            raise ValueError('Unexpected training/test schema')
        frame=_normalise(build_auxiliary_features(raw))
        test=_normalise(build_auxiliary_features(raw_test))
        if not frame.index.equals(raw.index) or not test.index.equals(raw_test.index):
            raise ValueError('Auxiliary joins changed source order')
        test.index=np.arange(len(frame),len(frame)+len(test))
        sample=pd.read_csv(DATA/'example-submission.csv')
        query=test.drop(columns=TARGET,errors='ignore')
        summaries=[]
        for policy in protocol['policies']:
            directory=out/policy;directory.mkdir()
            if policy=='uncleaned':
                fitting=frame.copy();removed=pd.DataFrame()
            else:
                audit=training_cleaning_audit(frame,policy)
                fitting=frame.loc[~audit['remove']].copy()
                removed=raw.loc[audit.index[audit['remove']]].copy()
                removed.insert(0,'SOURCE_ROW',removed.index)
                removed.insert(1,'ORIGINAL_CSV_LINE',removed.index+2)
                removed=removed.join(audit.loc[removed.index])
                removed.to_csv(directory/'removed_training_rows.csv',index=False)
            pd.DataFrame({'SOURCE_ROW':fitting.index,'ORIGINAL_CSV_LINE':fitting.index+2}).to_csv(
                directory/'retained_training_rows.csv',index=False)
            row_weights=recency_weights(fitting.RENT_APPROVAL_DATE,half_life_months)
            pd.DataFrame({'month':fitting.RENT_APPROVAL_DATE,'weight':row_weights}).groupby('month').agg(
                rows=('weight','size'),weight=('weight','mean')).to_csv(directory/'monthly_training_weights.csv')
            print(f'POLICY {policy}: fitting={len(fitting):,}, removed={len(removed):,}, months={fitting.RENT_APPROVAL_DATE.min()}..{fitting.RENT_APPROVAL_DATE.max()}',flush=True)
            parts=[];members=[]
            for seed in seeds:
                started=time.perf_counter();member=directory/f'seed{seed}';member.mkdir()
                print(f'TRAIN {policy} seed={seed}, epochs={epochs}',flush=True)
                model,meta,_,_,_,prediction,train_prediction=fit_model(fitting,query,dict(spec,seed=seed),epochs=epochs)
                if not np.isfinite(prediction).all():raise ValueError('Nonfinite inference')
                torch.save(model.state_dict(),member/'model.pt')
                joblib.dump(meta,member/'preprocessor.joblib')
                replay=restore_predictions(member,fitting,query)
                np.testing.assert_allclose(replay,prediction,rtol=0,atol=1e-6)
                pd.DataFrame({'Id':sample.Id,'prediction_sgd':prediction}).to_csv(member/'predictions.csv',index=False)
                summary={'seed':seed,'epochs':epochs,'fitting_rows':len(fitting),
                         'train_rmse_sgd':float(np.sqrt(np.mean((train_prediction-fitting[TARGET].to_numpy())**2))),
                         'reload_max_difference':float(np.abs(replay-prediction).max()),'seconds':time.perf_counter()-started}
                write_json(member/'summary.json',summary);members.append(summary);parts.append(prediction)
                print('MEMBER_COMPLETE',policy,json.dumps(summary),flush=True)
            values=np.column_stack(parts);mean,std=ensemble_summary(values)
            details=pd.DataFrame({'Id':sample.Id,'month':test.RENT_APPROVAL_DATE.to_numpy(),
                                  'prediction_sgd':mean,'ensemble_std_sgd':std})
            for i,seed in enumerate(seeds):details[f'seed{seed}_prediction_sgd']=values[:,i]
            details.to_csv(directory/'prediction_details.csv',index=False)
            prefix='submission_mlp_fullfit_' if half_life_months is None else f'submission_mlp_timeweighted_h{half_life_months:g}_'
            name=prefix+('uncleaned' if policy=='uncleaned' else 'cleaned')+'.csv'
            path=out/name
            validate_submission(sample,mean).to_csv(path,index=False,float_format='%.17g')
            saved=pd.read_csv(path)
            np.testing.assert_array_equal(saved.Id,sample.Id)
            np.testing.assert_allclose(saved.Predicted,mean,atol=1e-9,rtol=0)
            summary={'policy':policy,'fitting_rows':len(fitting),'removed_rows':len(removed),
                     'months':[fitting.RENT_APPROVAL_DATE.min(),fitting.RENT_APPROVAL_DATE.max()],
                     'test_rows':len(test),'overlapping_test_month_rows':int((test.RENT_APPROVAL_DATE<=fitting.RENT_APPROVAL_DATE.max()).sum()),
                     'epochs':epochs,'seeds':seeds,'members':members,'submission':name,'submission_sha256':sha256(path),
                     'training_weight_summary':weight_summary(fitting.RENT_APPROVAL_DATE,half_life_months),
                     'prediction_min_sgd':float(mean.min()),'prediction_max_sgd':float(mean.max()),
                     'test_rmse_sgd':None,'note':'All supplied labels offered to fitting; cleaning policy may remove rows. No independent validation of this full-data fit.'}
            write_json(directory/'summary.json',summary);summaries.append(summary)
            print('SUBMISSION_COMPLETE',name,flush=True)
        write_json(out/'summary.json',{'completed':True,'models':summaries,'source_hashes_unchanged':all(sha256(ROOT/p)==h for p,h in hashes.items()),
                                    'test_rmse_sgd':None,'kaggle_upload_performed':False})
    return summaries


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--baseline-dir',type=Path,default=ROOT/'reproduction/mlp/config/interval_baseline')
    parser.add_argument('--half-life-months',type=float,help='Positive exponential training-weight half-life; omitted means uniform')
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args();torch.set_num_threads(args.threads)
    run(args.output_dir,args.baseline_dir,args.half_life_months)


if __name__=='__main__':main()
