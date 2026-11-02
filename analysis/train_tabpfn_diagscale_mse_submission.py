"""Refit the selected MSE-only DiagScale recipe on all labels and export Kaggle CSV."""
from __future__ import annotations
import argparse
import gc
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
from analysis.data import sha256
from analysis.compare_local_economy_landscape import immutable,write_json,hashlib_array
from analysis.compare_tabpfn_v3_5_fast_80_20 import CHECKPOINT
from analysis.train_tabpfn_fast_submission import FEATURE_PATH

OUT=REPORTS/'tabpfn_diagscale_mse_final_submission'
CACHE=ROOT/'.cache/tabpfn_diagscale_mse_final_submission'
SUBMISSION=OUT/'submission_tabpfn_diagscale_mse_150k.csv'
ADAPTER=CACHE/'adapter.pt'
SOURCE=REPORTS/'tabpfn_components_80_20/candidates/diag_mse_only/calibration.json'
OLD_SUBMISSION=REPORTS/'tabpfn_fast_final_submission/submission_tabpfn_fast_150k.csv'
CFG=dict(component='diagscale',lr=.01,crps=0)
BATCH=200


def checked_bundle():
    manifest=json.loads((REPORTS/'tabpfn_fast_final_submission/feature_manifest.json').read_text())
    assert sha256(FEATURE_PATH)==manifest['sha256']
    original=json.loads((REPORTS/'tabpfn_fast_final_submission/settings.json').read_text())
    for path,h in original['source_sha256'].items(): assert sha256(ROOT/path)==h
    for path,h in original['code_sha256'].items(): assert sha256(ROOT/path)==h
    b=joblib.load(FEATURE_PATH); x,q=b['X_train'],b['X_query']
    assert x.shape==(150000,77) and q.shape==(50000,77) and list(x)==list(q)
    raw=pd.read_csv(DATA/'train.csv'); test=pd.read_csv(DATA/'test.csv')
    assert 'MONTHLY_RENT' not in test
    np.testing.assert_array_equal(b['y_train'],raw.MONTHLY_RENT)
    np.testing.assert_array_equal(x.index,raw.index)
    np.testing.assert_array_equal(q.index,test.index)
    np.testing.assert_array_equal(b['train_month'],raw.RENT_APPROVAL_DATE)
    np.testing.assert_array_equal(b['query_month'],test.RENT_APPROVAL_DATE)
    assert np.all(pd.PeriodIndex(b['query_origin'],freq='M').asi8<pd.PeriodIndex(b['query_month'],freq='M').asi8)
    y=np.asarray(b['y_train'],float)-np.asarray(b['train_baseline'],float)
    np.testing.assert_allclose(y+b['train_baseline'],b['y_train'],rtol=0,atol=1e-10)
    assert np.isfinite(y).all() and np.isfinite(b['query_baseline']).all()
    from analysis.check_diagscale_categorical_alignment_80_20 import category_positions
    assert category_positions(x)==category_positions(q) and len(category_positions(x))==9
    return b,manifest


def prepare():
    OUT.mkdir(parents=True,exist_ok=True); CACHE.mkdir(parents=True,exist_ok=True)
    b,manifest=checked_bundle(); source=json.loads(SOURCE.read_text())
    assert source['config']==CFG and source['selected_epoch']==0
    prior_adapter=ROOT/'.cache/tabpfn_components_80_20/candidates/diag_mse_only/adapter.pt'
    assert sha256(prior_adapter)==source['adapter_sha256']
    import torch
    files=[DATA/'train.csv',DATA/'test.csv',DATA/'example-submission.csv',FEATURE_PATH,CHECKPOINT,
           SOURCE,prior_adapter,OLD_SUBMISSION,REPORTS/'tabpfn_components_80_20/outer/diag_mse_only/predictions.csv']
    code=[Path(__file__),ROOT/'analysis/experiments/tabpfn_component_adaptation.py',
          ROOT/'analysis/compare_tabpfn_components_80_20.py',ROOT/'analysis/check_diagscale_categorical_alignment_80_20.py']
    settings=dict(model='TabPFN-3.5-Fast with MSE-only DiagScale',configuration=CFG,epochs=source['selected_epoch']+1,
        learning_rate=.01,weight_decay=.01,training_task_rows=1024,training_query_fraction=.2,
        training_views=2,inference_views=1,trainable_parameters=8576,seed=42,batch_size=BATCH,
        training_pool_rows=150000,test_rows=50000,features=77,validation_split_ratio=0,early_stopping=False,
        initialization='original pretrained checkpoint, new adapter trained on full labelled data; no warm-start from development adapter',
        target='raw rent minus train_baseline; add query_baseline exactly once',
        query_policy=manifest,subsampling=False,test_labels_available=False,
        source_validation_rmse_sgd=455.0710283636764,
        source_sha256={p.relative_to(ROOT).as_posix():sha256(p) for p in files},
        code_sha256={p.relative_to(ROOT).as_posix():sha256(p) for p in code},
        gpu=torch.cuda.get_device_name(),versions={p:version(p) for p in ['tabpfn','torch','numpy','pandas']})
    immutable(OUT/'settings.json',settings)
    write_json(OUT/'status.json',dict(status='prepared',training_rows=len(b['X_train']),test_rows=len(b['X_query'])))


def verify_sources():
    settings=json.loads((OUT/'settings.json').read_text())
    for path,h in settings['source_sha256'].items(): assert sha256(ROOT/path)==h
    for path,h in settings['code_sha256'].items(): assert sha256(ROOT/path)==h
    return settings


def train():
    import torch
    from tabpfn.constants import ModelVersion
    from analysis.check_diagscale_categorical_alignment_80_20 import category_positions
    from analysis.experiments.tabpfn_component_adaptation import FinetunedComponentRegressor,frozen_digest
    verify_sources()
    if ADAPTER.exists():
        record=json.loads((OUT/'training.json').read_text()); assert sha256(ADAPTER)==record['adapter_sha256']
        print('REUSE full-data adapter',flush=True); return
    b,_=checked_bundle(); x=b['X_train']; y=np.asarray(b['y_train'])-b['train_baseline']
    class FullDataMSE(FinetunedComponentRegressor):
        def _evaluate_model(self,*args,**kwargs):
            raise AssertionError('Full-data refit must not reserve or select using test targets')
    torch.set_num_threads(1); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model=FullDataMSE(model_version=ModelVersion.V3_5_FAST,device='cuda',epochs=1,learning_rate=.01,
        weight_decay=.01,random_state=42,validation_split_ratio=0,early_stopping=False,
        n_finetune_ctx_plus_query_samples=1024,finetune_ctx_query_split_ratio=.2,
        n_inference_subsample_samples=None,n_estimators_finetune=2,n_estimators_validation=1,
        n_estimators_final_inference=1,use_activation_checkpointing=True,
        ce_loss_weight=0,crps_loss_weight=0,mse_loss_weight=1,
        extra_regressor_kwargs=dict(model_path=str(CHECKPOINT),categorical_features_indices=category_positions(x),
            auto_scale_n_estimators=False,inference_config={'SUBSAMPLE_SAMPLES':None},n_preprocessing_jobs=1))
    model.component_config=CFG
    started=time.perf_counter(); print('TRAIN one MSE-only epoch on all 150,000 rows',flush=True)
    model.fit(x,y)
    architecture=model.finetuned_estimator_.model_
    digest=frozen_digest(architecture)
    assert digest==json.loads(SOURCE.read_text())['frozen_digest']
    state={n:p.detach().float().cpu().clone() for n,p in architecture.named_parameters() if p.requires_grad}
    assert sum(p.numel() for p in state.values())==8576 and all(n.endswith('.delta') for n in state)
    norm=float(sum(p.square().sum().item() for p in state.values())**.5)
    assert norm>0 and all(torch.isfinite(p).all() for p in state.values())
    temporary=ADAPTER.with_suffix('.part.pt'); torch.save(state,temporary); temporary.replace(ADAPTER)
    write_json(OUT/'training.json',dict(training_pool_rows=len(x),epochs=1,validation_rows_reserved=0,
        fitting_row_ids_sha256=hashlib_array(x.index),adapter_sha256=sha256(ADAPTER),
        frozen_parameters_unchanged=True,frozen_digest=digest,trainable_parameters=8576,delta_l2=norm,
        fit_seconds=time.perf_counter()-started,peak_vram_gib=torch.cuda.max_memory_allocated()/1024**3,
        test_targets_used=False))
    print('SAVED',str(ADAPTER),flush=True)


def estimator(b):
    from analysis.check_diagscale_categorical_alignment_80_20 import category_positions
    from analysis.compare_tabpfn_components_80_20 import make_estimator
    model=make_estimator(CFG,ADAPTER,category_positions(b['X_train']),views=1)
    model.fit(b['X_train'],np.asarray(b['y_train'])-b['train_baseline'])
    assert model.n_train_samples_==150000 and model.inference_config_.SUBSAMPLE_SAMPLES is None
    indices=model.ensemble_preprocessor_.subsample_row_indices
    if indices is not None: assert all(len(i)==150000 and len(np.unique(i))==150000 for i in indices)
    return model


def predict():
    import torch
    verify_sources(); b,_=checked_bundle(); q=b['X_query']
    training=json.loads((OUT/'training.json').read_text()); assert sha256(ADAPTER)==training['adapter_sha256']
    immutable(OUT/'inference_settings.json',dict(adapter_sha256=sha256(ADAPTER),features_sha256=sha256(FEATURE_PATH),
        checkpoint_sha256=sha256(CHECKPOINT),context_ids_sha256=hashlib_array(b['X_train'].index),
        test_ids_sha256=hashlib_array(q.index),batch_size=BATCH,inference_views=1,context_rows=150000))
    path=CACHE/'predictions.npy'; timing_path=OUT/'inference_timing.json'
    if path.exists() and timing_path.exists():
        values=np.load(path); assert len(values)==50000 and np.isfinite(values).all(); print('REUSE complete inference',flush=True); return
    values=np.load(path) if path.exists() else np.full(len(q),np.nan)
    assert values.shape==(50000,)
    torch.set_num_threads(1); torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter(); model=estimator(b); fitted=time.perf_counter()
    signature=[p._version for m in model.models_ for p in m.parameters()]
    print('INFERENCE all 150,000 context rows, 50,000 test rows',flush=True)
    for i in range(0,len(q),BATCH):
        stop=min(i+BATCH,len(q))
        if np.isfinite(values[i:stop]).all(): continue
        values[i:stop]=np.asarray(model.predict(q.iloc[i:stop],output_type='mean'),float)+b['query_baseline'][i:stop]
        assert np.isfinite(values[i:stop]).all()
        temporary=path.with_suffix('.part.npy'); np.save(temporary,values); temporary.replace(path)
        if i%4000==0 or stop==len(q): print('PREDICT',stop,'/',len(q),flush=True)
        write_json(OUT/'status.json',dict(status='inference',completed_rows=stop,total_rows=len(q)))
    assert signature==[p._version for m in model.models_ for p in m.parameters()]
    assert (values>0).all()
    write_json(timing_path,dict(context_rows=150000,test_rows=50000,features=77,batch_size=BATCH,
        fit_seconds=fitted-started,predict_seconds=time.perf_counter()-fitted,
        peak_vram_gib=torch.cuda.max_memory_allocated()/1024**3,parameter_versions_unchanged=True,subsampling=False))


def export():
    verify_sources(); b,manifest=checked_bundle(); values=np.load(CACHE/'predictions.npy')
    sample=pd.read_csv(DATA/'example-submission.csv')
    assert list(sample)==['Id','Predicted'] and len(sample)==50000 and sample.Id.is_unique
    np.testing.assert_array_equal(sample.Id,b['X_query'].index)
    assert values.shape==(50000,) and np.isfinite(values).all() and (values>0).all()
    submission=pd.DataFrame(dict(Id=sample.Id.astype(int),Predicted=values))
    if SUBMISSION.exists():
        existing=pd.read_csv(SUBMISSION); np.testing.assert_allclose(existing.Predicted,values,rtol=0,atol=1e-6)
    else:
        temporary=SUBMISSION.with_suffix('.part.csv'); submission.to_csv(temporary,index=False,float_format='%.6f'); temporary.replace(SUBMISSION)
    saved=pd.read_csv(SUBMISSION)
    assert list(saved)==list(sample) and saved.shape==sample.shape and saved.Id.is_unique
    np.testing.assert_array_equal(saved.Id,sample.Id)
    assert np.isfinite(saved.Predicted).all() and (saved.Predicted>0).all()
    assert np.max(np.abs(saved.Predicted-values))<=.000001
    monthly=pd.DataFrame(dict(month=b['query_month'],predicted=values,horizon=b['X_query'].TEMP_FORECAST_HORIZON))
    monthly.groupby('month').agg(rows=('predicted','size'),mean=('predicted','mean'),minimum=('predicted','min'),
        maximum=('predicted','max'),max_horizon=('horizon','max')).to_csv(OUT/'test_month_summary.csv',float_format='%.6f')
    write_json(OUT/'verification.json',dict(submission=SUBMISSION.relative_to(ROOT).as_posix(),sha256=sha256(SUBMISSION),
        rows=50000,columns=list(saved),row_order_matches_sample=True,predictions_positive_and_finite=True,
        context_rows=150000,refit_pool_rows=150000,features=77,baseline_added_once=True,
        prediction_min_sgd=float(values.min()),prediction_max_sgd=float(values.max()),prediction_mean_sgd=float(values.mean()),
        old_submission_unchanged=True,source_artifacts_unchanged=True,test_labels_used=False,
        adapter_sha256=sha256(ADAPTER),features_sha256=sha256(FEATURE_PATH)))
    write_json(OUT/'status.json',dict(status='exported',submission=SUBMISSION.name,rows=50000))
    report=(f'# MSE-only DiagScale full-data Kaggle submission\n\n'
        f'Ready-to-upload file: `{SUBMISSION.name}` (50,000 rows, `Id,Predicted`).\n\n'
        'A fresh DiagScale adapter is fitted on all 150,000 labelled rows for the previously selected one epoch. '
        'The original pretrained TabPFN backbone is frozen; 8,576 attention-scaling parameters use MSE only, '
        'learning rate0.01, weight decay0.01, seed42, 1,024-row tasks, 20% supervised queries and two training views. '
        'No rows are reserved for validation and no test targets or new tuning select the epoch. '
        'This is a full-data refit of the configuration measured at S$455.071 development RMSE; '
        'that score is not a measured score for the newly refitted model or the unlabelled competition test.\n\n'
        'Test inference uses the saved new adapter, the existing77-feature full-data cache, all150,000 context rows, '
        'one deployment view, native preprocessing and200-row query batches without context subsampling. '
        'Targets are raw rent minus the original training temporal baseline; query baselines are added exactly once. '
        'The CatBoost residual corrections and additional amenity features are omitted.\n\n'
        'The established test feature policy is preserved: March2025 test histories stop at February2025, '
        'later test histories stop at March2025. All labelled March rows remain available as model context. '
        f'For {manifest["beyond_ten_month_rows"]:,} test rows beyond the ten-month validated temporal forecast range, '
        'the forecast level stays at month10 while actual horizon, age and calendar month remain accurate.\n\n'
        f'CSV predictions range from S${values.min():,.2f} to S${values.max():,.2f}; '
        'schema,50,000 unique sample IDs, order, positivity, finiteness and CSV rounding are checked. '
        'Original development adapters, checkpoints, reports and the frozen-model submission remain unchanged. '
        'settings.json,training.json,inference_timing.json,verification.json andtest_month_summary.csv record the provenance.\n\n'
        'Reproduce/resume: `.venv/Scripts/python.exe analysis/train_tabpfn_diagscale_mse_submission.py --stage all`. '
        'Training and inference use separate fresh processes; the adapter and predictions are cached under '
        '`.cache/tabpfn_diagscale_mse_final_submission/`. This script saves the CSV locally and does not upload it.\n')
    (OUT/'report.md').write_text(report,encoding='utf8')
    print('READY',str(SUBMISSION),flush=True)


def replay():
    import torch
    verify_sources(); b,_=checked_bundle(); values=np.load(CACHE/'predictions.npy')
    torch.set_num_threads(1); model=estimator(b); largest=0.
    for start in [0,49800]:
        prediction=np.asarray(model.predict(b['X_query'].iloc[start:start+BATCH],output_type='mean'),float)+b['query_baseline'][start:start+BATCH]
        largest=max(largest,float(np.max(np.abs(prediction-values[start:start+BATCH]))))
        np.testing.assert_allclose(prediction,values[start:start+BATCH],atol=1e-5,rtol=0)
    write_json(OUT/'adapter_replay.json',dict(independent_reload=True,full_context_rows=150000,
        replay_rows=400,max_absolute_difference_sgd=largest,verified=True))
    write_json(OUT/'status.json',dict(status='complete',submission=SUBMISSION.name,rows=50000,adapter_replay_verified=True))
    print('REPLAY VERIFIED',largest,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['prepare','train','predict','export','replay','all'],default='all'); args=parser.parse_args()
    for key,value in dict(OMP_NUM_THREADS='1',TABPFN_NO_BROWSER='1',TABPFN_DISABLE_TELEMETRY='1',TABPFN_MAX_BATCHED_ESTIMATOR_ROWS='0').items(): os.environ.setdefault(key,value)
    if args.stage=='all':
        prepare()
        for stage in ['train','predict','export','replay']:
            with (OUT/f'{stage}.log').open('w',encoding='utf8') as log:
                subprocess.run([sys.executable,str(Path(__file__)),'--stage',stage],stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
            print('COMPLETED',stage,flush=True)
    else:
        globals()[args.stage]()


if __name__=='__main__': main()
