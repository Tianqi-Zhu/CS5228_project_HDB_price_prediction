"""Frozen spatial/seasonal feature diversity versus matched six-view inference."""
from __future__ import annotations
import argparse
import json
from importlib.metadata import version
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from analysis.config import ROOT,DATA,AUX,REPORTS
from analysis.data import sha256
from analysis.compare_local_economy_landscape import immutable,write_json,hashlib_array,group_columns
from analysis.compare_tabpfn_hierarchical_latent import load_bundle,raw_frame,weight_digest,check_sources as check_baseline_sources
from analysis.compare_tabpfn_v3_5_fast_80_20 import CHECKPOINT
from analysis.experiments.calendar_encoding import encoded_features
from analysis.experiments.local_market_fields import historical_market_features
from analysis.experiments.feature_diversified_ensemble import optimize_weights,combine_predictions

OUT=REPORTS/'tabpfn_feature_diversified_ensemble_80_20'
CACHE=ROOT/'.cache/tabpfn_feature_diversified_ensemble_80_20'
PRIOR=REPORTS/'tabpfn_hierarchical_latent_80_20'
LOCAL=REPORTS/'tabpfn_local_economy_landscape_80_20'
LOCAL_CACHE=ROOT/'.cache/tabpfn_local_economy_landscape_80_20'
ORIGINS=['2022-09','2023-07','2024-06']
MEMBERS=['baseline','spatial','temporal']


def check_sources():
    settings=json.loads((OUT/'settings.json').read_text())
    for k in ['source_sha256','code_sha256']:
        for p,h in settings[k].items(): assert sha256(ROOT/p)==h,p


def load_prepared(origin,member):
    path=CACHE/f'features_{origin}_{member}.joblib'
    m=json.loads((OUT/f'features_{origin}_{member}.json').read_text())
    assert sha256(path)==m['sha256']
    return joblib.load(path)


def prepare():
    OUT.mkdir(parents=True,exist_ok=True); CACHE.mkdir(parents=True,exist_ok=True)
    check_baseline_sources()
    prepared_path=LOCAL_CACHE/'prepared.joblib'
    assert sha256(prepared_path)==json.loads((LOCAL/'prepared_manifest.json').read_text())['sha256']
    sources=[prepared_path,LOCAL/'prepared_manifest.json',LOCAL/'canonical_buildings.csv',LOCAL/'transaction_building_links.csv',
        ROOT/'analysis/configs/local_economy_landscape.json',DATA/'train.csv',DATA/'test.csv',AUX/'sg-hdb-block.csv',CHECKPOINT,
        PRIOR/'settings.json',PRIOR/'baseline_predictions.csv',PRIOR/'baseline_replay.json',
        REPORTS/'tabpfn_diagscale_mse_final_submission/submission_tabpfn_diagscale_mse_150k.csv']
    for o in ORIGINS[:-1]: sources.extend([PRIOR/f'historical_predictions_{o}.csv',PRIOR/f'historical_provenance_{o}.json'])
    main=load_bundle('2024-06')
    source_settings=json.loads((PRIOR/'settings.json').read_text())
    baseline_sources=[ROOT/p for p in source_settings['source_sha256'] if '.joblib' in p or 'inner_features_' in p]
    sources.extend(baseline_sources)
    code=[Path(__file__),ROOT/'analysis/experiments/feature_diversified_ensemble.py',ROOT/'analysis/experiments/calendar_encoding.py',
        ROOT/'analysis/experiments/local_market_fields.py',ROOT/'analysis/compare_tabpfn_hierarchical_latent.py',
        ROOT/'analysis/experiments/local_spatial_features.py']
    immutable(OUT/'settings.json',dict(objective='Three frozen TabPFN feature representations; external convex ensembling vs internal estimator count',
        origins=ORIGINS,members=MEMBERS,baseline='Retained77 original categorical-month residual features; plain two-estimator V3.5FAST',
        spatial='Original77 plus existing12 market_smoothed fields; no new external inputs or duplicate amenity counts',
        temporal='Original trend/history/residual baseline retained; categorical month replaced by sin/cos plus6 existing recurring calendar windows,84 features',
        selection_basis='Existing earlier local-market screening favors smoothed over raw fields; prior controlled calendar screening favors cyclical+calendar. Broader dynamic20-feature replacement and raw neighborhood means failed; do not repeat broad searches.',
        feature_counts=dict(baseline=77,spatial=89,temporal=84),inference=dict(checkpoint='V3_5_FAST',estimators_per_member=2,standard_control_estimators=6,
            seed=42,device='cuda',batch_size=200,full_context=True,subsampling=False,fit_mode='fit_with_cache',keep_cache_on_device=True,n_preprocessing_jobs=1),
        target='raw SGD minus original local temporal baseline; add original query baseline once for every member/control',
        weights='Exact nonnegative sum-to-one MSE optimization on earlier out-of-time SGD predictions, no intercept; no outer evaluation labels',
        nested_evaluation='September2022 optimized cold-start baseline; July2023 weights learned from September2022; June2024 weights learned from both earlier disjoint ten-month windows',
        earlier_query_sampling='Original200 query rows/month, seed42, label-independent; month population/sample weights. Context and main28995 queries are not sampled',
        final_test_status='All available labeled months previously examined. Untouched labeled chronological test unavailable; report development backtests, not final test claims',
        historical_availability='New spatial fields use original conservative extra one-month publication lag and original pseudo-origins. Calendar additions are deterministic recurring proxies, not exact school dates. Inherited static MRT timing issue remains fixed.',
        no_finetuning=True,new_external_datasets=False,source_sha256={p.relative_to(ROOT).as_posix():sha256(p) for p in sources},
        code_sha256={p.relative_to(ROOT).as_posix():sha256(p) for p in code},packages={p:version(p) for p in ['tabpfn','torch','numpy','pandas','scipy']}))
    check_sources()
    if (OUT/'prepared_complete.json').exists():
        for origin in ORIGINS:
            for member in MEMBERS[1:]: load_prepared(origin,member)
        print('PREPARED features already verified',flush=True); return
    prepared=joblib.load(prepared_path); spatial_names=group_columns(prepared['outer']['added']['train'].columns)['market_smoothed']
    assert len(spatial_names)==12
    cfg=json.loads((ROOT/'analysis/configs/local_economy_landscape.json').read_text())
    raw=raw_frame(); raw['_MONTH']=pd.PeriodIndex(raw.month,freq='M')
    links=pd.read_csv(LOCAL/'transaction_building_links.csv',index_col=0)
    buildings=pd.read_csv(LOCAL/'canonical_buildings.csv').set_index('building_id',drop=False)
    assert buildings.index.is_unique
    raw['building_id']=links.loc[raw.index,'building_id']
    coverage=[]
    for origin in ORIGINS:
        b=load_bundle(origin); stage='outer' if origin=='2024-06' else f'inner_{origin}'
        x,q=b['X_train'],b['X_query']
        assert set(x.index).isdisjoint(q.index) and raw.loc[x.index,'month'].max()<origin<=raw.loc[q.index,'month'].min()
        assert len(x.columns)==len(q.columns)==77
        np.testing.assert_array_equal(raw.loc[q.index,'MONTHLY_RENT'],b['y_query'])
        before=prepared[stage]
        pd.testing.assert_frame_equal(before['bundle']['X_train'],x)
        extra_train=before['added']['train'][spatial_names]
        if origin=='2024-06': extra_query=before['added']['query'][spatial_names]
        else:
            origins=pd.Series(pd.Period(origin,'M')-1,index=q.index)
            extra_query=historical_market_features(raw.loc[x.index],raw.loc[q.index].drop(columns='MONTHLY_RENT'),origins,buildings,cfg)[spatial_names]
            cached=before['added']['query'][spatial_names]
            assert set(cached.index).issubset(q.index)
            np.testing.assert_allclose(extra_query.loc[cached.index],cached,rtol=0,atol=1e-8,equal_nan=True)
        frames={'spatial':(x.join(extra_train),q.join(extra_query)),'temporal':tuple(encoded_features(z,raw.loc[z.index,'month'],'cyclical',calendar=True) for z in [x,q])}
        for member,(new_x,new_q) in frames.items():
            if member=='spatial':
                pd.testing.assert_frame_equal(new_x[x.columns],x); pd.testing.assert_frame_equal(new_q[q.columns],q)
            else:
                original=[c for c in x if c!='APPROVAL_MONTH_OF_YEAR']
                pd.testing.assert_frame_equal(new_x[original],x[original]); pd.testing.assert_frame_equal(new_q[original],q[original])
            assert len(new_x.columns)==len(new_q.columns)=={'spatial':89,'temporal':84}[member]
            path=CACHE/f'features_{origin}_{member}.joblib'
            joblib.dump({**b,'X_train':new_x,'X_query':new_q},path,compress=3)
            write_json(OUT/f'features_{origin}_{member}.json',dict(sha256=sha256(path),origin=origin,member=member,
                feature_names=list(new_x),categorical_positions=[i for i,c in enumerate(new_x) if isinstance(new_x[c].dtype,pd.CategoricalDtype)],
                context_rows=len(new_x),query_rows=len(new_q),context_ids_sha256=hashlib_array(new_x.index),query_ids_sha256=hashlib_array(new_q.index),
                added=[c for c in new_x if c not in x],removed=[c for c in x if c not in new_x],source_labels_and_baselines_unchanged=True))
            for split,z in [('train',new_x),('query',new_q)]:
                coverage.extend(dict(origin=origin,member=member,split=split,feature=c,missing=int(z[c].isna().sum()),rows=len(z)) for c in z if c not in x)
        reuse_baseline(origin,b)
        print('PREPARED',origin,len(x),len(q),'spatial89 temporal84; cached spatial replay passed',flush=True)
    pd.DataFrame(coverage).to_csv(OUT/'new_feature_coverage.csv',index=False)
    write_json(OUT/'prepared_complete.json',dict(verified=True,spatial_columns=spatial_names,all_original_targets_and_residual_baselines_preserved=True))


def reuse_baseline(origin,b):
    source=PRIOR/'baseline_predictions.csv' if origin=='2024-06' else PRIOR/f'historical_predictions_{origin}.csv'
    meta=PRIOR/'baseline_replay.json' if origin=='2024-06' else PRIOR/f'historical_provenance_{origin}.json'
    m=json.loads(meta.read_text()); key='baseline_predictions_sha256' if origin=='2024-06' else 'predictions_sha256'
    assert sha256(source)==m[key] and m['n_estimators']==2 and m['parameters_unchanged'] and m['no_adapters']
    f=pd.read_csv(source,float_precision='round_trip'); np.testing.assert_array_equal(f.row_id,b['X_query'].index); np.testing.assert_array_equal(f.actual,b['y_query'])
    result=pd.DataFrame(dict(row_id=f.row_id,origin=origin,month=f.month,actual=f.actual,
        weight=np.ones(len(f)) if origin=='2024-06' else b['query_weights'],predicted=f.E0))
    path=OUT/f'predictions_{origin}_baseline.csv'; result.to_csv(path,index=False,float_format='%.17g')
    write_json(OUT/f'timing_{origin}_baseline.json',dict(**m,prediction_sha256=sha256(path),cached_prediction_reused=True,reuse_source=source.relative_to(ROOT).as_posix()))


def infer(origin,member):
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    from unittest.mock import patch
    check_sources(); torch.set_num_threads(1)
    output=OUT/f'predictions_{origin}_{member}.csv'; timing_path=OUT/f'timing_{origin}_{member}.json'
    if output.exists() and timing_path.exists():
        assert sha256(output)==json.loads(timing_path.read_text())['prediction_sha256']; print('REUSED',origin,member,flush=True); return
    b=load_bundle(origin) if member=='standard6' else load_prepared(origin,member)
    x,q=b['X_train'],b['X_query']; count=6 if member=='standard6' else 2
    progress=CACHE/f'partial_{origin}_{member}.npy'
    values=np.load(progress) if progress.exists() else np.full(len(q),np.nan)
    assert values.shape==(len(q),)
    torch.cuda.reset_peak_memory_stats()
    model=TabPFNRegressor.create_default_for_version(ModelVersion.V3_5_FAST,model_path=str(CHECKPOINT),device='cuda',n_estimators=count,
        random_state=42,fit_mode='fit_with_cache',keep_cache_on_device=True,n_preprocessing_jobs=1,
        categorical_features_indices=[i for i,c in enumerate(x) if isinstance(x[c].dtype,pd.CategoricalDtype)],inference_config={'SUBSAMPLE_SAMPLES':None})
    def forbidden(*args,**kwargs): raise AssertionError('Fine-tuning/autograd is prohibited')
    started=time.perf_counter()
    with patch('torch.Tensor.backward',forbidden),patch('torch.autograd.backward',forbidden),patch('torch.autograd.grad',forbidden):
        model.fit(x,np.asarray(b['y_train'])-b['train_baseline'])
        assert type(model) is TabPFNRegressor and model.n_train_samples_==len(x) and model.n_estimators_==count
        assert model.inference_config_.SUBSAMPLE_SAMPLES is None
        rows=model.ensemble_preprocessor_.subsample_row_indices
        if rows is not None: assert all(len(a)==len(x) and len(np.unique(a))==len(x) for a in rows)
        for m in model.models_:
            for p in m.parameters(): p.requires_grad_(False)
        fitted=time.perf_counter(); before=weight_digest(model); versions=[p._version for m in model.models_ for p in m.parameters()]
        prediction_started=time.perf_counter()
        for i in range(0,len(q),200):
            stop=min(i+200,len(q))
            if np.isfinite(values[i:stop]).all(): continue
            values[i:stop]=np.asarray(model.predict(q.iloc[i:stop],output_type='mean'),float)+b['query_baseline'][i:stop]
            np.save(progress,values)
            if i%1000==0 or stop==len(q): print('PREDICT',origin,member,stop,'/',len(q),flush=True)
        predicted=time.perf_counter()
        assert versions==[p._version for m in model.models_ for p in m.parameters()]
        after=weight_digest(model); assert before==after and np.isfinite(values).all()
    result=pd.DataFrame(dict(row_id=q.index,origin=origin,month=b['query_month'],actual=b['y_query'],
        weight=np.ones(len(q)) if origin=='2024-06' else b['query_weights'],predicted=values))
    result.to_csv(output,index=False,float_format='%.17g')
    write_json(timing_path,dict(origin=origin,member=member,n_estimators=count,features=len(x.columns),context_rows=len(x),query_rows=len(q),
        fit_seconds=fitted-started,predict_seconds=predicted-prediction_started,peak_vram_gib=torch.cuda.max_memory_allocated()/1024**3,
        weights_digest_before=before,weights_digest_after=after,parameters_unchanged=True,gradients_prohibited=True,no_adapters=True,subsampling=False,
        context_ids_sha256=hashlib_array(x.index),query_ids_sha256=hashlib_array(q.index),prediction_sha256=sha256(output),cached_prediction_reused=False))
    print('FINISHED',origin,member,flush=True)


def aligned(origin):
    frames=[]
    for member in MEMBERS:
        path=OUT/f'predictions_{origin}_{member}.csv'; meta=json.loads((OUT/f'timing_{origin}_{member}.json').read_text())
        assert sha256(path)==meta['prediction_sha256'] and meta['parameters_unchanged'] and meta['no_adapters'] and meta['n_estimators']==2
        f=pd.read_csv(path,float_precision='round_trip')
        if frames:
            for c in ['row_id','origin','month','actual','weight']: np.testing.assert_array_equal(f[c],frames[0][c])
        assert f.row_id.is_unique and np.isfinite(f.predicted).all(); frames.append(f)
    result=frames[0].drop(columns='predicted').copy()
    for member,f in zip(MEMBERS,frames): result[member]=f.predicted
    return result


def weight_candidates(f):
    p=f[MEMBERS].to_numpy(); y=f.actual.to_numpy(); weight=f.weight.to_numpy()
    w,info=optimize_weights(y,p,weight)
    result={'optimized':dict(weights=w.tolist(),**info)}
    for label,indices in [('opt_baseline_spatial',[0,1]),('opt_baseline_temporal',[0,2]),('opt_spatial_temporal',[1,2])]:
        z,info=optimize_weights(y,p[:,indices],weight); full=np.zeros(3); full[indices]=z
        result[label]=dict(weights=full.tolist(),**info)
    return result


def select():
    check_sources(); frames=[aligned(o) for o in ORIGINS[:-1]]; f=pd.concat(frames,ignore_index=True)
    assert f.month.max()<'2024-06' and not f.row_id.duplicated().any()
    selection=dict(origin='2024-06',member_order=MEMBERS,selection_case_rows=len(f),maximum_selection_label_month=f.month.max(),
        source_prediction_sha256={p.name:sha256(p) for o in ORIGINS[:-1] for m in MEMBERS for p in [OUT/f'predictions_{o}_{m}.csv']},
        evaluation_labels_used=False,models=weight_candidates(f),selection_ids_sha256=hashlib_array(f.row_id))
    immutable(OUT/'selection.json',selection)
    # Temporal transfer diagnostics; the earliest window has no preceding calibration.
    nested={ORIGINS[0]:dict(cold_start=True,models={k:dict(weights=[1.,0.,0.]) for k in selection['models']}),
        ORIGINS[1]:dict(cold_start=False,maximum_selection_label_month=frames[0].month.max(),models=weight_candidates(frames[0]))}
    assert nested[ORIGINS[1]]['maximum_selection_label_month']<ORIGINS[1]
    immutable(OUT/'nested_selection.json',nested)
    print('LOCKED WEIGHTS',selection['models'],flush=True)


def child(origin,member):
    log=OUT/f'inference_{origin}_{member}.log'
    with log.open('a',encoding='utf-8') as stream:
        result=subprocess.run([sys.executable,'-u',str(Path(__file__)),'--stage','one','--origin',origin,'--member',member],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
            env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
    if result.returncode: raise RuntimeError(f'Inference failed; inspect {log}')
    print('COMPLETED',origin,member,flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--stage',choices=['prepare','earlier','select','outer','one','summarize','all'],default='all')
    p.add_argument('--origin',choices=ORIGINS); p.add_argument('--member',choices=['spatial','temporal','standard6']); args=p.parse_args()
    if args.stage in ['prepare','all']: prepare()
    if args.stage in ['earlier','all']:
        for origin in ORIGINS[:-1]:
            for member in ['spatial','temporal','standard6']: child(origin,member)
    if args.stage in ['select','all']: select()
    if args.stage in ['outer','all']:
        assert (OUT/'selection.json').exists()
        for member in ['spatial','temporal','standard6']: child('2024-06',member)
    if args.stage=='one': infer(args.origin,args.member)
    if args.stage in ['summarize','all']:
        from analysis.summarize_tabpfn_feature_ensemble import summarize
        summarize()


if __name__=='__main__': main()
