"""Rebuild the same77 inputs from deduplicated labels; evaluate frozen TabPFN."""
from __future__ import annotations
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from analysis.config import ROOT,DATA,REPORTS
from analysis.data import prepare_main,normalized_address,sha256
from analysis.evaluation import metrics
from analysis.compare_local_economy_landscape import immutable,write_json,hashlib_array,frozen_predict
from analysis.compare_tabpfn_v3_5_fast_80_20 import CHECKPOINT
from analysis.experiments.aligned_temporal_features import HISTORY_COLUMNS,FEATURES,origin_history,LocalTemporalModel
from analysis.experiments.canonical_rental_dedup import canonical_training_rows
from analysis.summarize_local_economy_landscape import cluster_bootstrap,table

NAME='tabpfn_deduplicated_77_80_20'
OUT=REPORTS/NAME; CACHE=ROOT/'.cache'/NAME
PROCESSED=DATA/'processed'/NAME
BASE=ROOT/'.cache/tabpfn_v3_features/calendar_80_20_categorical.joblib'
REFERENCE=REPORTS/'tabpfn_v3_5_fast_calendar_80_20/predictions_outer_categorical.csv'
SUBMISSION=REPORTS/'tabpfn_diagscale_mse_final_submission/submission_tabpfn_diagscale_mse_150k.csv'
DERIVED=HISTORY_COLUMNS+FEATURES


def audit():
    for p in [OUT,CACHE,PROCESSED]: p.mkdir(parents=True,exist_ok=True)
    source=json.loads((REPORTS/'tabpfn_v3_5_fast_calendar_80_20/settings.json').read_text())
    manifest=json.loads((REPORTS/'tabpfn_v3_5_fast_calendar_80_20/feature_manifest_categorical.json').read_text())
    assert sha256(BASE)==manifest['sha256'] and sha256(CHECKPOINT)==source['checkpoint_sha256']
    for p,h in source['source_sha256'].items(): assert sha256(ROOT/p)==h
    baseline_predictions_hash=json.loads((REPORTS/'tabpfn_local_economy_landscape_80_20/settings.json').read_text())['baseline_predictions_sha256']
    assert sha256(REFERENCE)==baseline_predictions_hash
    files=[BASE,REFERENCE,CHECKPOINT,DATA/'train.csv',DATA/'test.csv',SUBMISSION,
           ROOT/'.cache/tabpfn_diagscale_mse_final_submission/adapter.pt']
    code=[Path(__file__),ROOT/'analysis/experiments/canonical_rental_dedup.py',
          ROOT/'analysis/experiments/aligned_temporal_features.py',ROOT/'analysis/experiments/spatiotemporal_hybrid.py',
          ROOT/'analysis/experiments/recent_history.py',ROOT/'analysis/experiments/tabpfn_adaptation_features.py',
          ROOT/'analysis/compare_local_economy_landscape.py']
    immutable(OUT/'settings.json',dict(duplicate_identity='All source fields including approval month and raw SGD rent, after existing type/model/address canonicalization; retain first source row',
        split='Original chronological cutoff June2024; original28995 validation rows preserved for primary comparison',
        context_history='All37 history inputs and9 temporal inputs/baselines rebuilt using deduplicated permitted history; original pseudo-origins and stable row IDs retained',
        static_features='Original31 static/calendar/categorical inputs unchanged; existingMRT source issue not mixed into this ablation',
        model='Frozen pretrained TabPFN-3.5-Fast; no DiagScale adaptation or CatBoost correction',
        inference=dict(n_estimators=1,seed=42,context_subsampling=False,batch_size=200,preprocessing='native fitted on retained context'),
        source_sha256={p.relative_to(ROOT).as_posix():sha256(p) for p in files},
        code_sha256={p.relative_to(ROOT).as_posix():sha256(p) for p in code},
        packages={p:version(p) for p in ['tabpfn','torch','numpy','pandas']}))
    return joblib.load(BASE)


def check_hashes():
    settings=json.loads((OUT/'settings.json').read_text())
    for p,h in settings['source_sha256'].items(): assert sha256(ROOT/p)==h
    for p,h in settings['code_sha256'].items(): assert sha256(ROOT/p)==h


def prepare():
    base=audit(); path=CACHE/'features.joblib'
    if path.exists():
        assert sha256(path)==json.loads((OUT/'prepared_manifest.json').read_text())['sha256']; return
    raw=pd.read_csv(DATA/'train.csv'); cleaned,mapping=canonical_training_rows(raw)
    assert len(raw)==150000 and len(cleaned)==149359 and len(mapping)==641
    cleaned.to_csv(PROCESSED/'train_canonical_deduplicated.csv',index=True,index_label='row_id',float_format='%.17g')
    mapping.to_csv(PROCESSED/'removed_duplicate_mapping.csv',index=False,float_format='%.17g')
    frame=normalized_address(prepare_main(raw)); frame['_MONTH']=pd.PeriodIndex(frame.RENT_APPROVAL_DATE,freq='M')
    static_all=pd.concat([base['X_train'],base['X_query']]).sort_index()
    frame['PLANNING_AREA']=static_all.loc[frame.index,'PLANNING_AREA'].astype(str)
    # Independently replay the inherited construction without removing records.
    # This detects a changed join/dtype/origin convention before any new scoring.
    original_fitting=frame.loc[base['X_train'].index]
    control_temporal=LocalTemporalModel(original_fitting)
    all_assignments=pd.concat([base['train_temporal_origins'],base['query_temporal_origins']])
    replay=[]
    for origin in [pd.Period('2021-06'),pd.Period('2022-12'),pd.Period('2024-05')]:
        ids=all_assignments.index[all_assignments.forecast_origin.eq(origin)][:40]
        past=original_fitting.loc[original_fitting._MONTH<=origin]
        blind=frame.loc[ids].drop(columns='MONTHLY_RENT')
        reconstructed=pd.concat([origin_history(past,blind),control_temporal.features(blind,origin)],axis=1)[DERIVED]
        expected=static_all.loc[ids,DERIVED]
        np.testing.assert_allclose(reconstructed,expected,rtol=0,atol=1e-7,equal_nan=True)
        difference=np.abs(reconstructed.to_numpy()-expected.to_numpy())
        replay.append(dict(origin=str(origin),rows=len(ids),maximum_absolute_difference=float(np.nanmax(difference))))
    write_json(OUT/'baseline_feature_replay.json',dict(verified=True,samples=replay,changed_labels=False))
    keep=base['X_train'].index.isin(cleaned.index); x=base['X_train'].loc[keep].copy(); q=base['X_query'].copy()
    fitting=frame.loc[x.index]; assert fitting._MONTH.max()==pd.Period('2024-05')
    assert len(x)==120761 and len(q)==28995 and fitting.index.isin(cleaned.index).all()
    original_origins=base['train_temporal_origins'].loc[x.index].copy()
    assignments={'train':original_origins,'query':base['query_temporal_origins'].copy()}
    temporal=LocalTemporalModel(fitting)
    matrices={'train':x,'query':q}
    origins=sorted(set(original_origins.forecast_origin.dropna()).union(assignments['query'].forecast_origin.dropna()))
    for i,origin in enumerate(origins,1):
        past=fitting.loc[fitting._MONTH<=origin]
        temporal.fit_at(origin)
        for split,matrix in matrices.items():
            ids=assignments[split].index[assignments[split].forecast_origin.eq(origin)]
            if not len(ids): continue
            blind=frame.loc[ids].drop(columns='MONTHLY_RENT')
            assert past._MONTH.max()<blind._MONTH.min()
            matrix.loc[ids,HISTORY_COLUMNS]=origin_history(past,blind).reindex(ids).to_numpy()
            matrix.loc[ids,FEATURES]=temporal.features(blind,origin).reindex(ids).to_numpy()
        if i%5==0 or i==len(origins): print('REBUILT',i,'/',len(origins),'origins through',origin,flush=True)
    static=[c for c in base['X_train'] if c not in DERIVED]
    pd.testing.assert_frame_equal(x[static],base['X_train'].loc[x.index,static])
    pd.testing.assert_frame_equal(q[static],base['X_query'][static])
    assert list(x)==list(q)==list(base['X_train']) and len(x.columns)==77
    bundle=dict(X_train=x,X_query=q,y_train=raw.loc[x.index,'MONTHLY_RENT'].to_numpy(float),y_query=np.asarray(base['y_query']),
        train_month=np.asarray(base['train_month'])[keep],query_month=np.asarray(base['query_month']),
        train_temporal_origins=assignments['train'],query_temporal_origins=assignments['query'],
        train_baseline=np.where(np.isfinite(x.TEMP_LOCAL_FORECAST),x.TEMP_LOCAL_FORECAST,0.),
        query_baseline=q.TEMP_LOCAL_FORECAST.to_numpy(float),unique_validation_ids=q.index[q.index.isin(cleaned.index)].to_numpy())
    assert np.isfinite(bundle['train_baseline']).all() and np.isfinite(bundle['query_baseline']).all()
    np.testing.assert_array_equal(bundle['y_train'],np.asarray(base['y_train'])[keep])
    pd.testing.assert_frame_equal(assignments['train'],base['train_temporal_origins'].loc[x.index])
    changes=[]
    for split,matrix in matrices.items():
        before=base['X_'+split].loc[matrix.index]
        equal=(matrix[DERIVED].eq(before[DERIVED])|(matrix[DERIVED].isna()&before[DERIVED].isna()))
        changes.extend(dict(split=split,feature=c,changed_rows=int((~equal[c]).sum())) for c in DERIVED)
        matrix.to_csv(PROCESSED/f'{split}_features_77.csv',index=True,index_label='row_id',float_format='%.17g')
        target=pd.DataFrame(dict(row_id=matrix.index,month=bundle[f'{split}_month'],
            MONTHLY_RENT=bundle[f'y_{split}'],additive_baseline_sgd=bundle[f'{split}_baseline'],
            residual_target_sgd=bundle[f'y_{split}']-bundle[f'{split}_baseline']))
        target.to_csv(PROCESSED/f'{split}_targets.csv',index=False,float_format='%.17g')
    pd.DataFrame(changes).to_csv(OUT/'feature_changes.csv',index=False)
    joblib.dump(bundle,path,compress=3)
    write_json(OUT/'prepared_manifest.json',dict(sha256=sha256(path),full_source_rows=150000,full_unique_rows=len(cleaned),
        total_removed_rows=len(mapping),original_context_rows=121005,context_rows=len(x),removed_context_rows=121005-len(x),
        validation_rows=len(q),unique_validation_rows=len(bundle['unique_validation_ids']),static_columns_unchanged=static,
        retained_context_ids_sha256=hashlib_array(x.index),query_ids_sha256=hashlib_array(q.index),
        exports={p.name:sha256(p) for p in PROCESSED.glob('*.csv')}))
    write_json(PROCESSED/'schema.json',dict(feature_columns=list(x),categorical_columns=list(x.select_dtypes(include='category')),
        target='MONTHLY_RENT',metadata_column='row_id; preserve this original ID, not a model input',
        feature_count=77,missing_numeric='Intentional NaNs retained; native TabPFN handling',
        processed_raw='train_canonical_deduplicated.csv retains original11 record fields with normalized category/address strings, plus source row_id',
        partition_exports='train_features_77.csv =120761 fitting rows; query_features_77.csv =original28995 evaluation rows; aligned targets in separate files',
        reconstruction='fit residual_target_sgd, add query additive_baseline_sgd once',
        duplicate_rule_includes_rent=True,train_cutoff_exclusive='2024-06',unique_validation_ids=bundle['unique_validation_ids'].tolist()))
    print('PREPARED',len(x),'context,',len(q),'validation; saved',str(PROCESSED),flush=True)


def infer():
    import torch
    torch.set_num_threads(1); check_hashes()
    manifest=json.loads((OUT/'prepared_manifest.json').read_text()); assert sha256(CACHE/'features.joblib')==manifest['sha256']
    b=joblib.load(CACHE/'features.joblib'); x,q=b['X_train'],b['X_query']
    immutable(OUT/'inference_settings.json',dict(prepared_sha256=manifest['sha256'],checkpoint_sha256=sha256(CHECKPOINT),
        context_ids_sha256=hashlib_array(x.index),query_ids_sha256=hashlib_array(q.index),batch=200,features=list(x)))
    if (OUT/'predictions.csv').exists():
        timing=json.loads((OUT/'timing.json').read_text()); assert sha256(OUT/'predictions.csv')==timing['prediction_sha256']; return
    print('INFERENCE',len(x),'deduplicated context;',len(q),'complete queries;77 features',flush=True)
    residual,timing=frozen_predict(x,b['y_train']-b['train_baseline'],q,CACHE/'partial.npy')
    prediction=residual+b['query_baseline']
    result=pd.DataFrame(dict(row_id=q.index,month=b['query_month'],actual=b['y_query'],predicted=prediction))
    result.to_csv(OUT/'predictions.csv',index=False,float_format='%.17g')
    write_json(OUT/'timing.json',dict(**timing,**metrics(result.actual,result.predicted),prediction_sha256=sha256(OUT/'predictions.csv')))
    print(metrics(result.actual,result.predicted),flush=True)


def summarize():
    check_hashes(); b=joblib.load(CACHE/'features.joblib')
    manifest=json.loads((OUT/'prepared_manifest.json').read_text()); assert sha256(CACHE/'features.joblib')==manifest['sha256']
    for name,h in manifest['exports'].items(): assert sha256(PROCESSED/name)==h
    reference=pd.read_csv(REFERENCE,float_precision='round_trip'); pred=pd.read_csv(OUT/'predictions.csv',float_precision='round_trip')
    np.testing.assert_array_equal(pred.row_id,reference.row_id); np.testing.assert_array_equal(pred.actual,reference.actual)
    assert pred.row_id.is_unique and len(pred)==28995 and np.isfinite(pred.predicted).all()
    score=[]; monthly=[]
    unique=pred.row_id.isin(b['unique_validation_ids']).to_numpy()
    for population,mask in [('original_complete_validation',np.ones(len(pred),bool)),('unique_validation_sensitivity',unique)]:
        baseline=metrics(pred.actual[mask],reference.predicted[mask])
        for name,values in [('original_77',reference.predicted),('deduplicated_rebuilt_77',pred.predicted)]:
            m=metrics(pred.actual[mask],values[mask]); score.append(dict(population=population,model=name,n=int(mask.sum()),**m,
                delta_rmse_sgd=m['rmse_sgd']-baseline['rmse_sgd'],rmse_improvement_pct=100*(baseline['rmse_sgd']-m['rmse_sgd'])/baseline['rmse_sgd']))
    for month,ids in pred.groupby('month').groups.items():
        for name,p in [('original_77',reference),('deduplicated_rebuilt_77',pred)]: monthly.append(dict(model=name,month=month,n=len(ids),**metrics(p.actual.iloc[ids],p.predicted.iloc[ids])))
    pd.DataFrame(score).to_csv(OUT/'scores.csv',index=False); pd.DataFrame(monthly).to_csv(OUT/'monthly_scores.csv',index=False)
    ci=cluster_bootstrap(pred.actual.to_numpy(),reference.predicted.to_numpy(),pred.predicted.to_numpy(),pred.month,n=5000)
    write_json(OUT/'month_cluster_bootstrap.json',ci)
    main_scores=pd.DataFrame(score).query("population == 'original_complete_validation'")
    delta=main_scores.iloc[1].delta_rmse_sgd
    write_json(OUT/'verification.json',dict(source_hashes_unchanged=True,processed_export_hashes_verified=True,
        checkpoint_unchanged=True,kaggle_submission_unchanged=True,all77_feature_names_preserved=True,
        static_features_unchanged=True,history_and_baselines_rebuilt_from_unique_context=True,
        unchanged_data_feature_replay_verified=True,
        stable_row_ids_and_pseudo_origins_preserved=True,validation_ids_and_labels_unchanged=True,
        inference_parameters_unchanged=True,full_retained_context_used=True,validation_rows=28995,
        retained_context_rows=120761,predictions_finite=True))
    report=['# Deduplicated rental records: frozen TabPFN77-feature experiment','',
        f'The fully rebuilt deduplicated-history experiment changes RMSE by **S${delta:+.3f}** against the original frozen77-feature baseline on the identical28,995 validation observations.','',
        table(main_scores),'','## Duplicate rule and saved data','',
        'This is a duplicate-row removal experiment, with all77 model columns retained. Case/whitespace in town/block/street and existing flat-type/model aliases are canonicalized. A duplicate must match every original record attribute, including approval month, floor area, lease year, rent, furnished status and fee. Keep the first original row ID; different prices or dates are not collapsed. Source data lacks unit/tenancy identifiers, so this is a sensitivity policy, not proof that the removed rows were erroneous contracts.','',
        'The complete labelled file changes from150,000 to149,359 rows (641 removed). Within the original January2021-May2024 fitting period,244 rows are removed:121,005 becomes120,761. The June2024-March2025 validation population remains all28,995 rows for a directly comparable primary score. No random repartition or boundary shift is performed. A secondary table filters both models to the same28,598 unique validation rows without using it to select a model.','',
        f'Processed records and model-ready CSVs are saved under `{PROCESSED.relative_to(ROOT).as_posix()}/`:','',
        '- train_canonical_deduplicated.csv:149,359 normalized records, all original11 fields plus stable source row_id.\n- removed_duplicate_mapping.csv:641 removed IDs and their retained counterpart.\n- train_features_77.csv/query_features_77.csv:77 model columns plus metadata row_id.\n- train_targets.csv/query_targets.csv:aligned raw SGD rent, residual target, month and additive baseline.\n- schema.json:feature list, categorical columns, loading and reconstruction policy.','',
        '## Feature rebuilding and inference','',
        'Retained context rows keep their original deterministic historical forecast origins. Every37 target-history input and9 temporal input is recalculated using only deduplicated fitting records available by that origin. The global/local temporal model is refitted and its existing earlier-only tuning rule reapplied. Validation history is frozen at May2024; no validation outcomes enter history construction or inference. Warmup rows without earlier data retain the original empty-history representation and zero additive baseline. Simply dropping TabPFN context rows while leaving duplicate-influenced histories was deliberately avoided.','',
        'The other31 inputs, categorical dtypes, original spatial source, month encoding and every feature name remain unchanged. In particular, the known future-MRT source issue is preserved in both arms to isolate this duplicate policy. This is frozen pretrained TabPFN-3.5-Fast inference, one estimator, seed42,200-row batches, every retained context row and native preprocessing. No DiagScale parameter optimization, CatBoost correction or new hyperparameter search is performed. Fit target is raw rent minus the recomputed temporal baseline; add the recomputed query baseline once.','',
        '## Unique-validation sensitivity','',table(pd.DataFrame(score).query("population == 'unique_validation_sensitivity'")),'',
        '## Uncertainty and limits','',
        f'Paired month-cluster bootstrap uses5000 resamples of the ten validation months. Candidate-minus-control RMSE interval: [{ci["rmse_delta_interval_95"][0]:+.3f}, {ci["rmse_delta_interval_95"][1]:+.3f}] SGD. This is descriptive development-period uncertainty; the holdout has been repeatedly inspected and is not a fresh competition-test estimate.','',
        'This primary ablation combines removing duplicate context rows with removing their contribution to histories and temporal forecasts. Feature changes are recorded in feature_changes.csv. It does not isolate those two mechanisms separately. The original MSE-only DiagScale checkpoint and Kaggle CSV are unchanged; this result should not be presented as a refitted MSE-only DiagScale result.','',
        '## Reproduction','',
        'Run `.venv/Scripts/python.exe -m unittest tests.test_canonical_rental_dedup tests.test_aligned_temporal_features`, then `.venv/Scripts/python.exe analysis/compare_tabpfn_deduplicated_77_80_20.py --stage all`. Prepared features and resumable inference batches live under ignored .cache/. No current Kaggle submission or raw source file is overwritten.']
    (OUT/'report.md').write_text('\n'.join(report)+'\n',encoding='utf8')
    write_json(OUT/'status.json',dict(status='complete',report='report.md',processed_data=PROCESSED.relative_to(ROOT).as_posix()))
    print(pd.DataFrame(score).to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--stage',choices=['prepare','infer','summarize','all'],default='all'); a=parser.parse_args()
    for k,v in dict(OMP_NUM_THREADS='1',TABPFN_NO_BROWSER='1',TABPFN_DISABLE_TELEMETRY='1',TABPFN_MAX_BATCHED_ESTIMATOR_ROWS='0').items(): os.environ.setdefault(k,v)
    if a.stage in ['prepare','all']: prepare()
    if a.stage in ['infer','all']:
        if a.stage=='all':
            with (OUT/'inference.log').open('w',encoding='utf8') as log:
                subprocess.run([sys.executable,str(Path(__file__)),'--stage','infer'],stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
        else: infer()
    if a.stage in ['summarize','all']: summarize()


if __name__=='__main__': main()
