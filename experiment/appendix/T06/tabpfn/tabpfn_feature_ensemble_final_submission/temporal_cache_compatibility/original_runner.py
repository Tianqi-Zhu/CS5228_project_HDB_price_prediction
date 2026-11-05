"""Full-data frozen feature-diversified TabPFN context and Kaggle inference."""
from __future__ import annotations
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from analysis.config import ROOT,DATA,REPORTS
from analysis.data import prepare_main,normalized_address,sha256
from analysis.compare_local_economy_landscape import immutable,write_json,hashlib_array
from analysis.compare_tabpfn_feature_ensemble import check_sources as check_experiment,OUT as EXPERIMENT,MEMBERS
from analysis.compare_tabpfn_hierarchical_latent import weight_digest
from analysis.compare_tabpfn_v3_5_fast_80_20 import CHECKPOINT
from analysis.train_tabpfn_fast_submission import FEATURE_PATH
from analysis.experiments.aligned_temporal_features import episode_origins
from analysis.experiments.calendar_encoding import encoded_features
from analysis.experiments.local_market_fields import historical_market_features
from analysis.experiments.feature_diversified_ensemble import combine_predictions

OUT=REPORTS/'tabpfn_feature_ensemble_final_submission'
CACHE=ROOT/'.cache/tabpfn_feature_ensemble_final_submission'
SUBMISSION=OUT/'submission_tabpfn_optimized_ensemble_150k.csv'
LOCAL=REPORTS/'tabpfn_local_economy_landscape_80_20'
BATCH=200


def full_base():
    m=json.loads((REPORTS/'tabpfn_fast_final_submission/feature_manifest.json').read_text())
    assert sha256(FEATURE_PATH)==m['sha256']
    prior=json.loads((REPORTS/'tabpfn_fast_final_submission/settings.json').read_text())
    for key in ['source_sha256','code_sha256']:
        for p,h in prior[key].items(): assert sha256(ROOT/p)==h,p
    b=joblib.load(FEATURE_PATH); train=pd.read_csv(DATA/'train.csv'); test=pd.read_csv(DATA/'test.csv')
    assert b['X_train'].shape==(150000,77) and b['X_query'].shape==(50000,77)
    assert list(b['X_train'])==list(b['X_query']) and 'MONTHLY_RENT' not in test
    np.testing.assert_array_equal(b['X_train'].index,train.index)
    np.testing.assert_array_equal(b['X_query'].index,test.index)
    np.testing.assert_array_equal(b['y_train'],train.MONTHLY_RENT)
    np.testing.assert_array_equal(b['train_month'],train.RENT_APPROVAL_DATE)
    np.testing.assert_array_equal(b['query_month'],test.RENT_APPROVAL_DATE)
    origin=pd.PeriodIndex(b['query_origin'],freq='M'); query_month=pd.PeriodIndex(b['query_month'],freq='M')
    assert (origin.asi8<query_month.asi8).all()
    np.testing.assert_array_equal(b['X_query'].TEMP_FORECAST_HORIZON,query_month.asi8-origin.asi8)
    np.testing.assert_array_equal(np.asarray(b['query_origin'])[query_month==pd.Period('2025-03','M')],np.full(sum(query_month==pd.Period('2025-03','M')),'2025-02'))
    assert set(np.asarray(b['query_origin'])[query_month>pd.Period('2025-03','M')])=={'2025-03'}
    assert np.isfinite(b['y_train']).all() and np.isfinite(b['train_baseline']).all() and np.isfinite(b['query_baseline']).all()
    return b


def verify_sources():
    s=json.loads((OUT/'settings.json').read_text())
    for key in ['source_sha256','code_sha256']:
        for p,h in s[key].items(): assert sha256(ROOT/p)==h,p
    return s


def load_member(member):
    if member=='baseline': return full_base()
    path=CACHE/f'features_{member}.joblib'; m=json.loads((OUT/f'features_{member}.json').read_text())
    assert sha256(path)==m['sha256']
    return joblib.load(path)


def source_rows(raw,links,offset):
    frame=normalized_address(prepare_main(raw)); frame['_MONTH']=pd.PeriodIndex(frame.RENT_APPROVAL_DATE,freq='M')
    linked=links.loc[frame.index+offset]
    keys=(frame.TOWN+'|'+frame.BLOCK+'|'+frame.STREET).to_numpy()
    np.testing.assert_array_equal(keys,linked.address_key.to_numpy())
    frame['building_id']=linked.building_id.to_numpy()
    return frame


def prepare():
    OUT.mkdir(parents=True,exist_ok=True); CACHE.mkdir(parents=True,exist_ok=True)
    check_experiment(); b=full_base()
    selection=json.loads((EXPERIMENT/'selection.json').read_text()); weights=selection['models']['optimized']['weights']
    assert selection['member_order']==MEMBERS and np.isclose(sum(weights),1) and min(weights)>=0
    spatial_names=json.loads((EXPERIMENT/'features_2024-06_spatial.json').read_text())['added']
    assert len(spatial_names)==12
    sources=[DATA/'train.csv',DATA/'test.csv',DATA/'example-submission.csv',CHECKPOINT,FEATURE_PATH,
        EXPERIMENT/'selection.json',EXPERIMENT/'settings.json',EXPERIMENT/'final_verification.json',
        EXPERIMENT/'features_2024-06_spatial.json',EXPERIMENT/'features_2024-06_temporal.json',
        ROOT/'.cache/tabpfn_feature_diversified_ensemble_80_20/features_2024-06_spatial.joblib',
        LOCAL/'canonical_buildings.csv',LOCAL/'transaction_building_links.csv',ROOT/'analysis/configs/local_economy_landscape.json',
        REPORTS/'tabpfn_fast_final_submission/settings.json',REPORTS/'tabpfn_fast_final_submission/feature_manifest.json',
        REPORTS/'tabpfn_diagscale_mse_final_submission/submission_tabpfn_diagscale_mse_150k.csv',
        REPORTS/'tabpfn_fast_final_submission/submission_tabpfn_fast_150k.csv']
    code=[Path(__file__),ROOT/'analysis/experiments/feature_diversified_ensemble.py',ROOT/'analysis/experiments/calendar_encoding.py',
        ROOT/'analysis/experiments/local_market_fields.py',ROOT/'analysis/experiments/aligned_temporal_features.py',
        ROOT/'analysis/train_tabpfn_fast_submission.py']
    immutable(OUT/'settings.json',dict(model='Frozen TabPFN-3.5-Fast external feature-diversified optimized averaging',
        members=MEMBERS,weights=weights,weight_selection='Fixed earlier chronological calibration; no full-data/test weight tuning',
        source_development_rmse_sgd=455.1718734783953,feature_counts=dict(baseline=77,spatial=89,temporal=84),
        spatial_columns=spatial_names,context_rows=150000,test_rows=50000,seed=42,n_estimators_per_member=2,batch_size=BATCH,
        full_context=True,subsampling=False,gradients_prohibited=True,no_adapters=True,
        estimator_batching='Native defaults, row/cell environment overrides unset; matched to validated optimized ensemble',
        allocator_policy='Clear unused CUDA cache before each200-row batch; retain live fitted context',
        target='Raw SGD rent minus original train baseline; add original test baseline exactly once before convex averaging',
        query_policy='March2025 baseline history throughFeb2025; later test months throughMarch2025. Spatial fields keep additional one-month lag.',
        long_horizon='Inherited baseline local forecast plateaus after horizon10; actual calendar,horizon and history age retained',
        no_validation_rows_reserved=True,no_test_labels=True,original_duplicates_retained=True,
        source_sha256={p.relative_to(ROOT).as_posix():sha256(p) for p in sources},
        code_sha256={p.relative_to(ROOT).as_posix():sha256(p) for p in code},
        versions={p:version(p) for p in ['tabpfn','torch','numpy','pandas']}))
    verify_sources()
    if (OUT/'prepared_complete.json').exists():
        for member in MEMBERS[1:]: load_member(member)
        print('PREPARED full-data features already verified',flush=True); return
    cfg=json.loads((ROOT/'analysis/configs/local_economy_landscape.json').read_text())
    links=pd.read_csv(LOCAL/'transaction_building_links.csv',index_col=0)
    buildings=pd.read_csv(LOCAL/'canonical_buildings.csv').set_index('building_id',drop=False); assert buildings.index.is_unique
    train=source_rows(pd.read_csv(DATA/'train.csv'),links,0)
    test=source_rows(pd.read_csv(DATA/'test.csv'),links,len(train))
    np.testing.assert_array_equal(links.loc[train.index,'split'],np.full(len(train),'train'))
    np.testing.assert_array_equal(links.loc[test.index+len(train),'split'],np.full(len(test),'test'))
    assignments=episode_origins(b['train_month'],train.index,train._MONTH.min())
    origins={'train':assignments.forecast_origin,'query':pd.Series(pd.PeriodIndex(b['query_origin'],freq='M'),index=test.index)}
    assert (pd.PeriodIndex(b['query_origin'],freq='M').asi8<pd.PeriodIndex(test._MONTH,freq='M').asi8).all()
    extra={}
    for split,frame in [('train',train),('query',test)]:
        print('BUILD spatial fields',split,len(frame),'all fitting labels filtered at each row origin',flush=True)
        extra[split]=historical_market_features(train,frame.drop(columns='MONTHLY_RENT',errors='ignore'),origins[split],buildings,cfg)[spatial_names]
        assert extra[split].index.equals(frame.index) and not np.isinf(extra[split].to_numpy(float)).any()
    # Independent numerical replay over all pre-June2024 fitting rows. Later
    # labels are present in the full data but must not affect earlier fields.
    earlier=joblib.load(ROOT/'.cache/tabpfn_feature_diversified_ensemble_80_20/features_2024-06_spatial.joblib')
    expected=earlier['X_train'][spatial_names]
    np.testing.assert_allclose(extra['train'].loc[expected.index],expected,rtol=0,atol=1e-8,equal_nan=True)
    write_json(OUT/'spatial_context_replay.json',dict(verified=True,rows=len(expected),features=12,
        maximum_difference=float(np.nanmax(np.abs(extra['train'].loc[expected.index].to_numpy()-expected.to_numpy()))),
        later_training_labels_do_not_change_earlier_context_features=True))
    coverage=[]
    for member in MEMBERS[1:]:
        matrices={}
        for split in ['train','query']:
            base=b['X_'+split]
            if member=='spatial':
                z=base.join(extra[split]); pd.testing.assert_frame_equal(z[base.columns],base)
            else:
                z=encoded_features(base,b[split+'_month'],'cyclical',calendar=True)
                original=[c for c in base if c!='APPROVAL_MONTH_OF_YEAR']; pd.testing.assert_frame_equal(z[original],base[original])
            assert z.shape==(150000 if split=='train' else 50000,89 if member=='spatial' else 84)
            expected_names=json.loads((EXPERIMENT/f'features_2024-06_{member}.json').read_text())['feature_names']
            assert list(z)==expected_names
            for c in z:
                if c not in base: coverage.append(dict(member=member,split=split,feature=c,missing=int(z[c].isna().sum()),rows=len(z)))
            matrices['X_'+split]=z
        path=CACHE/f'features_{member}.joblib'; joblib.dump({**b,**matrices},path,compress=3)
        write_json(OUT/f'features_{member}.json',dict(sha256=sha256(path),feature_names=list(matrices['X_train']),
            context_rows=150000,test_rows=50000,train_ids_sha256=hashlib_array(train.index),test_ids_sha256=hashlib_array(test.index),
            original_targets_and_residual_baselines_unchanged=True))
        print('SAVED',member,'full-data feature bundle',flush=True)
    pd.DataFrame(coverage).to_csv(OUT/'new_feature_coverage.csv',index=False)
    write_json(OUT/'prepared_complete.json',dict(verified=True,context_rows=150000,test_rows=50000,weights=weights,
        test_months=sorted(set(b['query_month'])),maximum_forecast_horizon=int(b['X_query'].TEMP_FORECAST_HORIZON.max()),
        test_rows_beyond10_months=int((b['X_query'].TEMP_FORECAST_HORIZON>10).sum()),
        march_test_spatial_latest_label_month='2025-01',later_test_spatial_latest_label_month='2025-02'))


def predict(member):
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    from tabpfn.settings import settings
    from unittest.mock import patch
    verify_sources()
    assert settings.tabpfn.max_batched_estimator_rows is None and settings.tabpfn.max_batched_estimator_cells is None
    path=OUT/f'predictions_{member}.csv'; meta_path=OUT/f'timing_{member}.json'
    if path.exists() and meta_path.exists():
        assert sha256(path)==json.loads(meta_path.read_text())['predictions_sha256']; print('REUSED',member,flush=True); return
    b=load_member(member); x,q=b['X_train'],b['X_query']
    assert x.shape==(150000,{'baseline':77,'spatial':89,'temporal':84}[member]) and len(q)==50000
    torch.set_num_threads(1); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model=TabPFNRegressor.create_default_for_version(ModelVersion.V3_5_FAST,model_path=str(CHECKPOINT),device='cuda',n_estimators=2,random_state=42,
        fit_mode='fit_with_cache',keep_cache_on_device=True,n_preprocessing_jobs=1,inference_config={'SUBSAMPLE_SAMPLES':None},
        categorical_features_indices=[i for i,c in enumerate(x) if isinstance(x[c].dtype,pd.CategoricalDtype)])
    progress=CACHE/f'partial_{member}.npy'
    values=np.load(progress) if progress.exists() else np.full(50000,np.nan)
    assert values.shape==(50000,)
    def forbidden(*a,**kw): raise AssertionError('TabPFN must remain frozen; gradients prohibited')
    started=time.perf_counter()
    with patch('torch.Tensor.backward',forbidden),patch('torch.autograd.backward',forbidden),patch('torch.autograd.grad',forbidden):
        model.fit(x,np.asarray(b['y_train'])-b['train_baseline'])
        assert type(model) is TabPFNRegressor and model.n_train_samples_==150000 and model.n_estimators_==2 and model.inference_config_.SUBSAMPLE_SAMPLES is None
        indices=model.ensemble_preprocessor_.subsample_row_indices
        if indices is not None: assert all(len(i)==150000 and len(np.unique(i))==150000 for i in indices)
        for m in model.models_:
            for parameter in m.parameters(): parameter.requires_grad_(False)
        fitted=time.perf_counter(); before=weight_digest(model); versions=[p._version for m in model.models_ for p in m.parameters()]
        prediction_started=time.perf_counter(); replay_error=None; done_before=int(np.isfinite(values).sum())
        if np.isfinite(values[:BATCH]).all():
            torch.cuda.empty_cache(); replay=np.asarray(model.predict(q.iloc[:BATCH],output_type='mean'),float)+b['query_baseline'][:BATCH]
            replay_error=float(np.max(np.abs(replay-values[:BATCH]))); np.testing.assert_allclose(replay,values[:BATCH],rtol=0,atol=1e-5)
        for i in range(0,50000,BATCH):
            stop=min(i+BATCH,50000)
            if np.isfinite(values[i:stop]).all(): continue
            torch.cuda.empty_cache()
            values[i:stop]=np.asarray(model.predict(q.iloc[i:stop],output_type='mean'),float)+b['query_baseline'][i:stop]
            assert np.isfinite(values[i:stop]).all()
            tmp=progress.with_suffix('.part.npy'); np.save(tmp,values); tmp.replace(progress)
            if i%2000==0 or stop==50000: print('PREDICT',member,stop,'/50000',flush=True)
        predicted=time.perf_counter(); after=weight_digest(model)
        assert before==after and versions==[p._version for m in model.models_ for p in m.parameters()]
    frame=pd.DataFrame(dict(Id=q.index,month=b['query_month'],Predicted=values))
    frame.to_csv(path,index=False,float_format='%.17g')
    write_json(meta_path,dict(member=member,fit_seconds=fitted-started,predict_seconds=predicted-prediction_started,
        peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,context_rows=150000,test_rows=50000,features=len(x.columns),
        n_estimators=2,seed=42,batch_size=200,subsampling=False,native_default_estimator_batching=True,
        frozen_digest_before=before,frozen_digest_after=after,parameters_unchanged=True,no_adapters=True,gradients_prohibited=True,
        training_context_ids_sha256=hashlib_array(x.index),test_ids_sha256=hashlib_array(q.index),
        predictions_sha256=sha256(path),resumed_rows=done_before,resume_replay_error_sgd=replay_error))
    print('FINISHED',member,flush=True)


def export():
    s=verify_sources(); b=full_base(); sample=pd.read_csv(DATA/'example-submission.csv')
    assert sample.shape==(50000,2) and list(sample)==['Id','Predicted'] and sample.Id.is_unique
    np.testing.assert_array_equal(sample.Id,b['X_query'].index)
    vectors=[]; timing=[]; result=pd.DataFrame(dict(Id=sample.Id,month=b['query_month']))
    for member in MEMBERS:
        path=OUT/f'predictions_{member}.csv'; meta=json.loads((OUT/f'timing_{member}.json').read_text())
        assert sha256(path)==meta['predictions_sha256'] and meta['parameters_unchanged'] and meta['frozen_digest_before']==meta['frozen_digest_after']
        f=pd.read_csv(path,float_precision='round_trip'); assert len(f)==50000 and f.Id.is_unique
        np.testing.assert_array_equal(f.Id,sample.Id); np.testing.assert_array_equal(f.month,b['query_month'])
        assert np.isfinite(f.Predicted).all(); vectors.append(f.Predicted.to_numpy()); result[member]=f.Predicted
        timing.append(meta)
    p=np.column_stack(vectors); weights=np.asarray(s['weights']); values=combine_predictions(p,weights)
    np.testing.assert_allclose(values,np.sum(p*weights[None,:],axis=1),atol=1e-10,rtol=0)
    assert np.isfinite(values).all() and (values>0).all()
    result['optimized']=values; result.to_csv(OUT/'predictions_members.csv',index=False,float_format='%.17g')
    submission=pd.DataFrame(dict(Id=sample.Id.astype(int),Predicted=values))
    if SUBMISSION.exists():
        old=pd.read_csv(SUBMISSION); np.testing.assert_array_equal(old.Id,submission.Id); np.testing.assert_allclose(old.Predicted,values,atol=1e-6,rtol=0)
    else:
        tmp=SUBMISSION.with_suffix('.part.csv'); submission.to_csv(tmp,index=False,float_format='%.6f'); tmp.replace(SUBMISSION)
    saved=pd.read_csv(SUBMISSION); assert saved.shape==sample.shape and list(saved)==list(sample)
    np.testing.assert_array_equal(saved.Id,sample.Id); assert saved.Id.is_unique and np.isfinite(saved.Predicted).all() and (saved.Predicted>0).all()
    assert float(np.max(np.abs(saved.Predicted-values)))<=1e-6
    pd.DataFrame(timing).to_csv(OUT/'timings.csv',index=False)
    summary=pd.DataFrame(dict(month=b['query_month'],predicted=values,horizon=b['X_query'].TEMP_FORECAST_HORIZON))
    summary.groupby('month').agg(rows=('predicted','size'),mean=('predicted','mean'),minimum=('predicted','min'),maximum=('predicted','max'),horizon=('horizon','max')).to_csv(OUT/'test_month_summary.csv')
    verify_sources()
    write_json(OUT/'verification.json',dict(submission=SUBMISSION.relative_to(ROOT).as_posix(),submission_sha256=sha256(SUBMISSION),rows=50000,columns=list(saved),
        exact_sample_row_order=True,all_predictions_positive_and_finite=True,all_three_models_use150000_context_rows=True,
        weights_unchanged_from_validated_optimized_model=True,weights=s['weights'],feature_counts=s['feature_counts'],
        frozen_backbone_and_source_files_unchanged=True,old_submissions_unchanged=True,baseline_added_once=True,
        native_default_estimator_batching=True,test_targets_used=False,test_rmse_available=False,
        prediction_min_sgd=float(values.min()),prediction_max_sgd=float(values.max()),prediction_mean_sgd=float(values.mean()),
        maximum_csv_rounding_error_sgd=float(np.max(np.abs(saved.Predicted-values)))))
    info=json.loads((OUT/'prepared_complete.json').read_text())
    report=f'''# Optimized feature-diversified TabPFN: full-data submission

Kaggle file: [{SUBMISSION.name}]({SUBMISSION.name}) — 50,000 rows, exactly `Id,Predicted`, in sample-submission/test order. All rents are finite and positive.

Three plain frozen TabPFN-3.5-Fast regressors prepare inference context from **all150,000 labeled transactions** with77 baseline,89 spatial and84 temporal features. Each uses two internal estimators,seed42,200-row batches,native-default estimator batching,full context and no subsampling. No DiagScale,LoRA,adapter or neural-weight training is used. Each predicts raw SGD minus the unchanged temporal baseline, adding that baseline once before convex averaging. Parameters are hashed before/after inference and all gradient calls prohibited.

Fixed earlier-validation weights, in baseline/spatial/temporal order: **{s['weights']}**. They are not reoptimized on full-data training errors or hidden test targets. Source development RMSE455.171873 concerns the previous80/20 backtest, not this full-data refit or the unlabeled Kaggle test. No test score can be measured locally.

The existing full-data77-feature cache preserves original transaction rows and causal pseudo-origins. All12 spatial fields are rebuilt from permitted labels, then replayed against every121,005 old development-context row. New spatial fields preserve their additional one-month publication lag. March2025 test baseline histories endFebruary2025; its added spatial histories endJanuary2025. Later test baseline histories endMarch2025, and its added spatial histories endFebruary2025. All150,000 provided labels, includingMarch2025, remain in every model's labeled context as requested; only feature-history cutoffs differ for overlapping March test records.

The test spansMarch2025-July2026. The inherited local forecasting model was validated only through10 months: its forecast level/uncertainty plateau beyond10 while actual month,horizon and history age remain accurate. There are{info['test_rows_beyond10_months']:,} test rows beyond10 months and maximum forecast horizon{info['maximum_forecast_horizon']}. No new long-horizon assumption is tuned here. Existing static MRT/geography availability limitations are inherited from the validated feature pipeline.

Saved artifacts: settings.json/source hashes,feature manifests,spatial_context_replay.json,per-member predictions and timing/frozen-weight digests,predictions_members.csv,test_month_summary.csv and verification.json. Both prior frozen and DiagScale Kaggle submissions remain unchanged. Reproduce with `.venv/Scripts/python.exe analysis/train_tabpfn_feature_ensemble_submission.py --stage all`. The50,000 test rows and original IDs are checked after CSV reload; predictions are written to six decimal places. This command saves the upload file locally and does not upload it to Kaggle.
'''
    (OUT/'report.md').write_text(report,encoding='utf-8')
    print('READY',SUBMISSION,'rows50000',flush=True)


def child(member):
    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','TABPFN_NO_BROWSER':'1','TABPFN_DISABLE_TELEMETRY':'1'}
    for name in ['TABPFN_MAX_BATCHED_ESTIMATOR_ROWS','TABPFN_MAX_BATCHED_ESTIMATOR_CELLS']: env.pop(name,None)
    log=OUT/f'inference_{member}.log'
    with log.open('a',encoding='utf-8') as stream:
        r=subprocess.run([sys.executable,'-u',str(Path(__file__)),'--stage','one','--member',member],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
    if r.returncode: raise RuntimeError(f'{member} failed; inspect {log}')
    print('COMPLETED',member,flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--stage',choices=['prepare','one','predict','export','all'],default='all'); p.add_argument('--member',choices=MEMBERS); args=p.parse_args()
    if args.stage in ['prepare','all']: prepare()
    if args.stage=='one': predict(args.member)
    if args.stage in ['predict','all']:
        for member in MEMBERS: child(member)
    if args.stage in ['export','all']: export()


if __name__=='__main__': main()
