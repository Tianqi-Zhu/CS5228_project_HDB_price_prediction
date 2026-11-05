"""Five expanding-window adapters and lightweight CatBoost residual corrections."""
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from analysis.config import ROOT, DATA, REPORTS
from analysis.data import sha256
from analysis.evaluation import metrics
from analysis.compare_local_economy_landscape import immutable, write_json, hashlib_array
from analysis.compare_tabpfn_v3_5_fast_80_20 import CHECKPOINT
from analysis.experiments.spatial_temporal_residual import (
    historical_bundle, check_partition, market_features, residual_features, corrected, choose_strength, BASIC, MARKET, CATEGORICAL,
)

OUT = REPORTS/'tabpfn_spatial_temporal_residual_80_20'
CACHE = ROOT/'.cache/tabpfn_spatial_temporal_residual_80_20'
BASE = ROOT/'.cache/tabpfn_v3_features/calendar_80_20_categorical.joblib'
ADAPTER = ROOT/'.cache/tabpfn_components_80_20/candidates/diag_mse_only/adapter.pt'
REFERENCE = REPORTS/'tabpfn_components_80_20/outer/diag_mse_only/predictions.csv'
FOLDS = [('2022-09','2023-01'), ('2023-01','2023-05'), ('2023-05','2023-09'),
         ('2023-09','2023-12'), ('2023-12','2024-06')]
CFG = dict(component='diagscale', lr=.01, crps=0)
CB = dict(iterations=800, depth=5, learning_rate=.04, loss_function='RMSE', l2_leaf_reg=5,
          random_seed=42, verbose=False, thread_count=-1, allow_writing_files=False)


def save_frame(frame, path):
    frame.to_csv(path, index=False, float_format='%.17g')


def audit():
    OUT.mkdir(exist_ok=True); CACHE.mkdir(exist_ok=True)
    feature_manifest=json.loads((REPORTS/'tabpfn_v3_5_fast_calendar_80_20/feature_manifest_categorical.json').read_text())
    adapter_manifest=json.loads((REPORTS/'tabpfn_components_80_20/candidates/diag_mse_only/calibration.json').read_text())
    assert sha256(BASE)==feature_manifest['sha256']
    assert sha256(ADAPTER)==adapter_manifest['adapter_sha256']
    base=joblib.load(BASE); reference=pd.read_csv(REFERENCE, float_precision='round_trip')
    np.testing.assert_array_equal(reference.row_id,base['X_query'].index)
    np.testing.assert_array_equal(reference.actual,base['y_query'])
    assert abs(metrics(reference.actual,reference.predicted)['rmse_sgd']-455.071)<.001
    preservation=REPORTS/'tabpfn_local_economy_landscape_80_20/submission_preservation.json'
    preserved=json.loads(preservation.read_text()); assert sha256(ROOT/preserved['path'])==preserved['sha256']
    frozen=[BASE,ADAPTER,CHECKPOINT,REFERENCE,DATA/'train.csv',DATA/'test.csv',ROOT/preserved['path'],
            REPORTS/'tabpfn_components_80_20/report.md', REPORTS/'tabpfn_components_80_20/plan.json',
            ROOT/'analysis/experiments/aligned_temporal_features.py', ROOT/'analysis/experiments/spatiotemporal_hybrid.py',
            ROOT/'analysis/experiments/tabpfn_component_adaptation.py']
    code=[Path(__file__),ROOT/'analysis/experiments/spatial_temporal_residual.py']
    settings=dict(baseline='MSE-only DiagScale, numerically lowest existing declared 80/20 result',
        baseline_selection_caveat='No unique global winner selected by one earlier-only rule; README retains older frozen control. Numerical-best adapted model chosen before this residual study; outer is a repeatedly inspected development holdout.',
        folds=FOLDS, adapter_recipe=dict(**CFG,epochs=1,train_views=2,deployment_views=1,task_rows=1024,query_fraction=.2,weight_decay=.01),
        residual_early_stop=['2023-09','2023-12'],residual_validation=['2023-12','2024-06'],outer=['2024-06','2025-04'],
        strengths=[0,.25,.5,.75,1],catboost=CB,features=dict(E1=BASIC,E2=BASIC+MARKET),seed=42,
        market_policy='Fixed batch origin; only labels strictly before origin. Town x flat history median minimum20; flat-type then global fallback. Growth adjacent 3-month medians minimum10 at same fallback level. Six-month same-group count.',
        adapter_selection='Fixed existing one-epoch recipe; no OOF query targets used for gradient updates, early stopping or checkpoint selection. Existing full/pre-Dec2023 adapters not reused for earlier OOF.',
        final_test_available=False,development_evaluation=True,
        source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in frozen},
        implementation_sha256={str(p.relative_to(ROOT)):sha256(p) for p in code},
        versions={p:version(p) for p in ['tabpfn','torch','catboost','numpy','pandas']})
    immutable(OUT/'settings.json',settings)
    write_json(OUT/'baseline_verification.json',dict(**metrics(reference.actual,reference.predicted),rows=len(reference),
        checkpoint=CHECKPOINT.relative_to(ROOT).as_posix(),adapter=ADAPTER.relative_to(ROOT).as_posix(),features=77,
        target='raw rent minus original train_baseline; original query_baseline added once',competition_test_labels=False))
    return base, reference


def frame_from_base(base):
    # Reuse existing prepared spatial/property inputs instead of geocoding again.
    from analysis.data import prepare_main
    raw=prepare_main(pd.read_csv(DATA/'train.csv'))
    raw.index.name='row_id'
    full=pd.concat([base['X_train'],base['X_query']]).sort_index()
    for c in ['LATITUDE','LONGITUDE','PLANNING_AREA','BLOCK_ID','STREET_ID']:
        raw[c]=full.loc[raw.index,c].astype('string') if isinstance(full[c].dtype,pd.CategoricalDtype) else full.loc[raw.index,c]
    raw['_MONTH']=pd.PeriodIndex(raw.RENT_APPROVAL_DATE,freq='M')
    np.testing.assert_array_equal(raw.loc[base['X_train'].index,'MONTHLY_RENT'],base['y_train'])
    return raw


def prepare():
    base,_=audit(); frame=frame_from_base(base); rows=[]
    for start,stop in FOLDS:
        dest=CACHE/f'features_{start}.joblib'; manifest=OUT/f'features_{start}.json'
        if dest.exists():
            assert sha256(dest)==json.loads(manifest.read_text())['sha256']; continue
        b=historical_bundle(base,frame,start,stop)
        q=frame.loc[b['X_query'].index].drop(columns='MONTHLY_RENT')
        market,fallback=market_features(frame.loc[b['X_train'].index],q,start)
        b['market']=market; joblib.dump(b,dest,compress=3)
        fallback['row_id']=fallback.index; save_frame(fallback,OUT/f'market_fallback_{start}.csv')
        record=dict(origin=start,end_exclusive=stop,context_rows=len(b['X_train']),query_rows=len(q),
                    context_max_month=str(pd.PeriodIndex(b['train_month'],freq='M').max()),query_min_month=start,
                    query_max_month=str(pd.PeriodIndex(b['query_month'],freq='M').max()),
                    context_ids_sha256=hashlib_array(b['X_train'].index),query_ids_sha256=hashlib_array(q.index),
                    sha256=sha256(dest),fold_target_labels_in_context=False,market_last_allowed_month=str(pd.Period(start,'M')-1))
        immutable(manifest,record); print('PREPARED',record,flush=True)
    for start,_ in FOLDS: rows.append(json.loads((OUT/f'features_{start}.json').read_text()))
    save_frame(pd.DataFrame(rows),OUT/'fold_boundaries.csv')
    b=dict(bundle=base,market=market_features(frame.loc[base['X_train'].index],frame.loc[base['X_query'].index].drop(columns='MONTHLY_RENT'),'2024-06')[0])
    dest=CACHE/'outer_features.joblib'
    if not dest.exists():
        joblib.dump(b,dest,compress=3); write_json(OUT/'outer_features.json',dict(sha256=sha256(dest)))


def one_fold(origin):
    import torch
    from tabpfn.constants import ModelVersion
    from analysis.experiments.tabpfn_component_adaptation import FinetunedComponentRegressor,frozen_digest
    from analysis.compare_tabpfn_components_80_20 import make_estimator
    from analysis.check_diagscale_categorical_alignment_80_20 import category_positions
    torch.set_num_threads(1); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    manifest=json.loads((OUT/f'features_{origin}.json').read_text()); dest=CACHE/f'features_{origin}.joblib'
    assert sha256(dest)==manifest['sha256']; b=joblib.load(dest)
    x,q=b['X_train'],b['X_query']; y=np.asarray(b['y_train'])-b['train_baseline']
    check_partition(b['train_month'],b['query_month'],x.index,q.index,origin)
    path=OUT/f'oof_predictions_{origin}.csv'; record_path=OUT/f'oof_provenance_{origin}.json'
    if path.exists() and record_path.exists():
        record=json.loads(record_path.read_text()); assert sha256(path)==record['prediction_sha256']; print('REUSE',origin,flush=True); return
    adapter=CACHE/f'adapter_{origin}.pt'; training_path=OUT/f'adapter_{origin}.json'
    if not adapter.exists():
        class NoQueryEvaluation(FinetunedComponentRegressor):
            def _evaluate_model(self,*args,**kwargs):
                raise AssertionError('OOF query labels must never select adapter weights')
        model=NoQueryEvaluation(model_version=ModelVersion.V3_5_FAST,device='cuda',epochs=1,learning_rate=.01,
            weight_decay=.01,random_state=42,validation_split_ratio=0,early_stopping=False,
            n_finetune_ctx_plus_query_samples=1024,finetune_ctx_query_split_ratio=.2,
            n_inference_subsample_samples=None,n_estimators_finetune=2,n_estimators_validation=1,
            n_estimators_final_inference=1,use_activation_checkpointing=True,ce_loss_weight=0,crps_loss_weight=0,mse_loss_weight=1,
            extra_regressor_kwargs=dict(model_path=str(CHECKPOINT),categorical_features_indices=category_positions(x),
                auto_scale_n_estimators=False,inference_config={'SUBSAMPLE_SAMPLES':None},n_preprocessing_jobs=1))
        model.component_config=CFG
        start=time.perf_counter(); print('TRAIN fold-specific one-epoch adapter',origin,len(x),flush=True)
        model.fit(x,y)
        architecture=model.finetuned_estimator_.model_
        digest=frozen_digest(architecture)
        expected=json.loads((REPORTS/'tabpfn_components_80_20/candidates/diag_mse_only/calibration.json').read_text())['frozen_digest']
        assert digest==expected
        state={n:p.detach().float().cpu().clone() for n,p in architecture.named_parameters() if p.requires_grad}
        assert sum(t.numel() for t in state.values())==8576
        torch.save(state,adapter)
        write_json(training_path,dict(fit_rows=len(x),epochs=1,validation_targets_used=False,checkpoint_selection=False,
            frozen_digest=digest,frozen_parameters_unchanged=True,adapter_sha256=sha256(adapter),fit_seconds=time.perf_counter()-start))
        del model,architecture,state; gc.collect(); torch.cuda.empty_cache()
    training=json.loads(training_path.read_text()); assert sha256(adapter)==training['adapter_sha256']
    predictor=make_estimator(CFG,adapter,category_positions(x),views=1)
    started=time.perf_counter(); predictor.fit(x,y)
    assert predictor.n_train_samples_==len(x) and predictor.inference_config_.SUBSAMPLE_SAMPLES is None
    signature=[p._version for m in predictor.models_ for p in m.parameters()]
    partial=CACHE/f'partial_{origin}.npy'
    predictions=np.load(partial) if partial.exists() else np.full(len(q),np.nan)
    assert len(predictions)==len(q)
    for i in range(0,len(q),200):
        stop=min(i+200,len(q))
        if np.isfinite(predictions[i:stop]).all(): continue
        predictions[i:stop]=np.asarray(predictor.predict(q.iloc[i:stop],output_type='mean'),float)+b['query_baseline'][i:stop]
        np.save(partial,predictions)
        if i%4000==0 or stop==len(q): print('PREDICT',origin,stop,'/',len(q),flush=True)
    assert signature==[p._version for m in predictor.models_ for p in m.parameters()]
    assert np.isfinite(predictions).all()
    result=pd.DataFrame(dict(row_id=q.index,month=b['query_month'],origin=origin,actual=b['y_query'],predicted=predictions))
    result['residual']=result.actual-result.predicted; save_frame(result,path)
    record=dict(**manifest,**training,prediction_seconds=time.perf_counter()-started,prediction_sha256=sha256(path),
        prediction_parameter_versions_unchanged=True,**metrics(result.actual,result.predicted))
    write_json(record_path,record); print('OOF COMPLETE',origin,metrics(result.actual,result.predicted),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--stage',choices=['prepare','fold','develop','evaluate','all'],default='all'); parser.add_argument('--origin')
    a=parser.parse_args()
    for key,value in dict(OMP_NUM_THREADS='1',TABPFN_NO_BROWSER='1',TABPFN_DISABLE_TELEMETRY='1',TABPFN_MAX_BATCHED_ESTIMATOR_ROWS='0').items(): os.environ.setdefault(key,value)
    if a.stage in ['prepare','all']: prepare()
    if a.stage=='fold': one_fold(a.origin)
    if a.stage=='all':
        for origin,_ in FOLDS:
            with (OUT/f'fold_{origin}.log').open('w',encoding='utf8') as log:
                subprocess.run([sys.executable,str(Path(__file__)),'--stage','fold','--origin',origin],stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
            print('FINISHED FOLD',origin,flush=True)
    if a.stage in ['develop','all']:
        from analysis.evaluate_tabpfn_spatial_temporal_residual import develop
        develop()
    if a.stage in ['evaluate','all']:
        from analysis.evaluate_tabpfn_spatial_temporal_residual import evaluate
        evaluate()


if __name__=='__main__': main()
