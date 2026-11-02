"""Frozen two-view TabPFN plus small hierarchical Gaussian residual models."""
from __future__ import annotations
import argparse
import hashlib
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
from analysis.compare_tabpfn_v3_5_fast_80_20 import CHECKPOINT
from analysis.experiments.calendar_encoding import encoded_features
from analysis.evaluation import metrics

OUT=REPORTS/'tabpfn_hierarchical_latent_80_20'
CACHE=ROOT/'.cache/tabpfn_hierarchical_latent_80_20'
BASE=ROOT/'.cache/tabpfn_v3_features/calendar_80_20_categorical.joblib'
REFERENCE=REPORTS/'tabpfn_lora_qk_lr00003_80_20/predictions_2views.csv'
HISTORICAL=['2022-05','2022-09','2023-01','2023-05','2023-07','2023-08','2024-02']
OUTERS=['2022-09','2023-07','2024-06']


def feature_source(origin):
    if origin=='2023-07':
        return ROOT/'.cache/tabpfn_v3_features/calendar_inner_2023-07.joblib',REPORTS/'tabpfn_v3_5_fast_calendar_80_20/inner_features_2023-07.json'
    return ROOT/f'.cache/tabpfn_fast_embedding_head/features_{origin}.joblib',REPORTS/f'tabpfn_fast_embedding_head_80_20/features_{origin}.json'


def load_bundle(origin):
    if origin=='2024-06': return joblib.load(BASE)
    path,manifest=feature_source(origin)
    assert sha256(path)==json.loads(manifest.read_text())['sha256']
    b=joblib.load(path)
    if origin=='2023-07':
        b['X_train']=encoded_features(b['X_train'],b['train_month'],'categorical')
        b['X_query']=encoded_features(b['X_query'],b['query_month'],'categorical')
    return b


def raw_frame():
    r=normalized_address(prepare_main(pd.read_csv(DATA/'train.csv')))
    r['month']=r.RENT_APPROVAL_DATE.astype(str)
    r['building']=r.TOWN+'|'+r.STREET+'|'+r.BLOCK
    r['submarket']=r.TOWN_CLEAN+'|'+r.FLAT_TYPE_CLEAN
    return r


def prepare():
    OUT.mkdir(parents=True,exist_ok=True); CACHE.mkdir(parents=True,exist_ok=True)
    original=json.loads((REPORTS/'tabpfn_v3_5_fast_calendar_80_20/settings.json').read_text())
    manifest=json.loads((REPORTS/'tabpfn_v3_5_fast_calendar_80_20/feature_manifest_categorical.json').read_text())
    assert sha256(BASE)==manifest['sha256'] and sha256(CHECKPOINT)==original['checkpoint_sha256']
    for path,h in original['source_sha256'].items(): assert sha256(ROOT/path)==h
    b=joblib.load(BASE); ref=pd.read_csv(REFERENCE,float_precision='round_trip')
    np.testing.assert_array_equal(ref.row_id,b['X_query'].index); np.testing.assert_array_equal(ref.actual,b['y_query'])
    score=metrics(ref.actual,ref.frozen)
    assert abs(score['rmse_sgd']-455.3959272763438)<1e-7
    sources=[BASE,REFERENCE,CHECKPOINT,DATA/'train.csv',DATA/'test.csv',
        REPORTS/'tabpfn_diagscale_mse_final_submission/submission_tabpfn_diagscale_mse_150k.csv']
    for origin in HISTORICAL: sources.extend(feature_source(origin))
    code=[Path(__file__),ROOT/'analysis/experiments/aligned_temporal_features.py',
          ROOT/'analysis/experiments/calendar_encoding.py',ROOT/'analysis/compare_local_economy_landscape.py']
    settings=dict(baseline='Original77-feature categorical-month non-deduplicated full-context frozen TabPFN, two preprocessing estimators',
        feature_names=list(b['X_train']),categorical_positions=[i for i,c in enumerate(b['X_train']) if isinstance(b['X_train'][c].dtype,pd.CategoricalDtype)],
        baseline_metrics=score,baseline_selection='Lowest observed valid frozen setting: two views455.396 versus one455.814; same77 feature family. Added geography/economy/trees/quality and deduplication did not establish a better retained configuration. All adapted results excluded.',
        inference=dict(model_version='V3_5_FAST',n_estimators=2,random_state=42,batch_size=200,
            fit_mode='fit_with_cache',keep_cache_on_device=True,subsample_samples=None,n_preprocessing_jobs=1),
        historical_origins=HISTORICAL,outer_origins=OUTERS,
        residual_pool='At each fitting origin, keep the latest eligible forecast per physical source row; only labels strictly before origin. Normalize inverse-sampling weights to sum to actual distinct residual count.',
        historical_query_sampling='Reuse fixed label-independent200 queries/month and supplied month-population/sample weights; retain complete historical contexts. Earlier outer tests use same sampled rows; main80/20 has every28995 row.',
        nested_protocol='For each outerT, tune on historical validation origins beforeT, using only labels beforeT; each inner model fits only residuals with target month before its validation origin. Cold start with no earlier inner validation selects alpha0.',
        coarse_grid=dict(block_penalty=[25.,100.],group_penalty=[25.,100.],rho=[0.,.5,.9],state_penalty=[25.,100.],alpha=[0.,.25,.5,.75,1.]),
        dynamic_gate='At least8 historical months and abs monthly-mean residual lag1 autocorrelation>=0.1; evaluated before dynamic fitting',
        source_sha256={p.relative_to(ROOT).as_posix():sha256(p) for p in sources},
        code_sha256={p.relative_to(ROOT).as_posix():sha256(p) for p in code},
        packages={p:version(p) for p in ['tabpfn','torch','numpy','pandas','scipy']},
        adapted_model_or_trainable_adapter_used=False,external_new_features=False,development_holdout_previously_examined=True)
    immutable(OUT/'settings.json',settings)
    write_json(OUT/'baseline_specification.json',dict(**settings['inference'],features=settings['feature_names'],
        excluded='DiagScale, LoRA, new economic/environmental/quality features, deduplicated data, sampled context',
        target='original raw rent minus context temporal baseline; add original query baseline once',
        histories='Original37 label-derived histories and9 temporal inputs obey original pseudo-origins; forecast query features fixed at origin minus1month',
        spatial_limitation='Inherited static MRT/geography snapshots; future-MRT source issue is held fixed across all models, not certified point-in-time geography',
        frozen_reference_column='frozen column only; trained columns never used',**score))
    frame=raw_frame()
    for split in ['train','query']:
        x=b['X_'+split]; known=x.BLOCK_ID.notna()
        np.testing.assert_array_equal(x.loc[known,'BLOCK_ID'].astype(str),frame.loc[x.index[known],'building'])
    rows=[]
    for origin in HISTORICAL:
        h=load_bundle(origin); x,q=h['X_train'],h['X_query']
        assert list(x)==list(q)==list(b['X_train']) and len(x.columns)==77
        assert set(x.index).isdisjoint(q.index)
        assert frame.loc[x.index,'month'].max()<origin and frame.loc[q.index,'month'].min()>=origin
        np.testing.assert_array_equal(h['y_query'],frame.loc[q.index,'MONTHLY_RENT'])
        rows.append(dict(origin=origin,context_rows=len(x),query_rows=len(q),first_month=min(h['query_month']),last_month=max(h['query_month']),
            represented_rows=float(np.sum(h['query_weights'])),context_ids_sha256=hashlib_array(x.index),query_ids_sha256=hashlib_array(q.index)))
    pd.DataFrame(rows).to_csv(OUT/'historical_fold_specification.csv',index=False)
    print('BASELINE',score,'two estimators; historical contexts/rows verified',flush=True)


def check_sources():
    s=json.loads((OUT/'settings.json').read_text())
    for p,h in s['source_sha256'].items(): assert sha256(ROOT/p)==h
    for p,h in s['code_sha256'].items(): assert sha256(ROOT/p)==h


def weight_digest(model):
    digest=hashlib.sha256()
    for k,m in enumerate(model.models_):
        for name,p in m.named_parameters():
            if 'lora' in name.lower() or name.endswith('.delta'): raise AssertionError('Trainable adapter found in frozen baseline')
            digest.update(f'{k}:{name}'.encode()); digest.update(p.detach().cpu().contiguous().view(__import__('torch').uint8).numpy().tobytes())
    return digest.hexdigest()


def predict_plain(b,tag,limit=None):
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    from unittest.mock import patch
    torch.set_num_threads(1); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    q=b['X_query'] if limit is None else b['X_query'].iloc[:limit]
    progress=CACHE/f'partial_{tag}.npy'
    values=np.load(progress) if progress.exists() else np.full(len(q),np.nan)
    assert values.shape==(len(q),)
    model=TabPFNRegressor.create_default_for_version(ModelVersion.V3_5_FAST,model_path=str(CHECKPOINT),device='cuda',
        n_estimators=2,random_state=42,fit_mode='fit_with_cache',keep_cache_on_device=True,
        categorical_features_indices=[i for i,c in enumerate(b['X_train']) if isinstance(b['X_train'][c].dtype,pd.CategoricalDtype)],
        inference_config={'SUBSAMPLE_SAMPLES':None},n_preprocessing_jobs=1)
    started=time.perf_counter()
    def forbidden(*args,**kwargs): raise AssertionError('Gradient-based training is prohibited')
    with patch('torch.Tensor.backward',forbidden),patch('torch.autograd.backward',forbidden),patch('torch.autograd.grad',forbidden):
        model.fit(b['X_train'],np.asarray(b['y_train'])-b['train_baseline'])
        assert type(model) is TabPFNRegressor and model.n_train_samples_==len(b['X_train'])
        assert model.inference_config_.SUBSAMPLE_SAMPLES is None and model.n_estimators_==2
        indices=model.ensemble_preprocessor_.subsample_row_indices
        if indices is not None: assert all(len(a)==len(b['X_train']) and len(np.unique(a))==len(a) for a in indices)
        for m in model.models_:
            for parameter in m.parameters(): parameter.requires_grad_(False)
        before=weight_digest(model); signature=[p._version for m in model.models_ for p in m.parameters()]
        fitted=time.perf_counter()
        for i in range(0,len(q),200):
            stop=min(i+200,len(q))
            if np.isfinite(values[i:stop]).all(): continue
            values[i:stop]=np.asarray(model.predict(q.iloc[i:stop],output_type='mean'),float)+b['query_baseline'][i:stop]
            np.save(progress,values)
            if i%1000==0 or stop==len(q): print('PREDICT',tag,stop,'/',len(q),flush=True)
        assert signature==[p._version for m in model.models_ for p in m.parameters()]
        after=weight_digest(model); assert before==after and np.isfinite(values).all()
    timing=dict(fit_seconds=fitted-started,predict_seconds=time.perf_counter()-fitted,context_rows=len(b['X_train']),query_rows=len(q),
        peak_vram_gib=torch.cuda.max_memory_allocated()/1024**3,parameter_digest_before=before,parameter_digest_after=after,
        parameters_unchanged=True,gradients_prohibited=True,no_adapters=True,n_estimators=2,subsampling=False)
    return values,timing


def probe():
    check_sources(); b=load_bundle('2024-06'); reference=pd.read_csv(REFERENCE,float_precision='round_trip')
    path=OUT/'baseline_replay.json'
    if path.exists(): assert json.loads(path.read_text())['verified']; return
    prediction,timing=predict_plain(b,'baseline_replay',limit=400)
    error=float(np.max(np.abs(prediction-reference.frozen.to_numpy()[:400])))
    np.testing.assert_allclose(prediction,reference.frozen.to_numpy()[:400],rtol=0,atol=1e-5)
    write_json(path,dict(verified=True,plain_unmodified_tabpfn=True,rows=400,max_absolute_difference_sgd=error,
        cached_complete_rows=len(reference),cached_complete_metrics=metrics(reference.actual,reference.frozen),**timing))
    print('BASELINE REPLAY VERIFIED',error,flush=True)


def historical(origin):
    check_sources(); assert json.loads((OUT/'baseline_replay.json').read_text())['verified']
    b=load_bundle(origin); path=OUT/f'historical_predictions_{origin}.csv'; meta=OUT/f'historical_provenance_{origin}.json'
    if path.exists() and meta.exists():
        assert sha256(path)==json.loads(meta.read_text())['predictions_sha256']; print('REUSE',origin,flush=True); return
    frame=raw_frame(); x,q=b['X_train'],b['X_query']
    assert frame.loc[x.index,'month'].max()<origin and set(x.index).isdisjoint(q.index)
    prediction,timing=predict_plain(b,origin)
    f=frame.loc[q.index,['building','submarket','TOWN_CLEAN','FLAT_TYPE_CLEAN']].copy()
    f.insert(0,'row_id',f.index); f['origin']=origin; f['month']=b['query_month']; f['actual']=b['y_query']; f['E0']=prediction
    f['residual']=f.actual-f.E0; f['weight']=b['query_weights']; f['horizon']=pd.PeriodIndex(f.month,freq='M').asi8-pd.Period(origin,'M').ordinal+1
    f=f.rename(columns={'TOWN_CLEAN':'town','FLAT_TYPE_CLEAN':'flat_type'})
    f.to_csv(path,index=False,float_format='%.17g')
    write_json(meta,dict(**timing,origin=origin,context_max_month=frame.loc[x.index,'month'].max(),target_min_month=min(b['query_month']),
        target_max_month=max(b['query_month']),context_query_disjoint=True,target_labels_in_context=False,
        context_ids_sha256=hashlib_array(x.index),query_ids_sha256=hashlib_array(q.index),predictions_sha256=sha256(path)))
    print('HISTORICAL COMPLETE',origin,metrics(f.actual,f.E0),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--stage',choices=['prepare','probe','historical','one','latent','all'],default='all'); p.add_argument('--origin'); a=p.parse_args()
    for k,v in dict(OMP_NUM_THREADS='1',TABPFN_NO_BROWSER='1',TABPFN_DISABLE_TELEMETRY='1',TABPFN_MAX_BATCHED_ESTIMATOR_ROWS='0').items(): os.environ.setdefault(k,v)
    if a.stage in ['prepare','all']: prepare()
    if a.stage=='probe': probe()
    if a.stage in ['historical','all']:
        jobs=[('probe',None)]+[('one',o) for o in HISTORICAL]
        for stage,origin in jobs:
            cmd=[sys.executable,str(Path(__file__)),'--stage',stage]
            if origin: cmd+=['--origin',origin]
            with (OUT/f'{origin or "baseline_probe"}.log').open('w',encoding='utf8') as log:
                subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
            print('COMPLETED',origin or 'baseline_probe',flush=True)
    if a.stage=='one': historical(a.origin)
    if a.stage in ['latent','all']:
        from analysis.evaluate_tabpfn_hierarchical_latent import run
        run()


if __name__=='__main__': main()
