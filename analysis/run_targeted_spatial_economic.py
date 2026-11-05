"""Execute targeted causal feature ablations against the frozen three-member ensemble."""
from __future__ import annotations

import argparse
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
from analysis.data import prepare_main,sha256
from analysis.compare_local_economy_landscape import write_json,immutable
from analysis.compare_tabpfn_hierarchical_latent import load_bundle
from analysis.experiments.feature_diversified_ensemble import optimize_weights
from analysis.experiments.targeted_spatial_economic import hierarchy_features,row_cutoffs,residual_smoother
import analysis.compare_tabpfn_feature_ensemble as prior

OUT=REPORTS/'targeted_spatial_economic'
CACHE=ROOT/'.cache/targeted_spatial_economic'
REFERENCE=REPORTS/'tabpfn_feature_diversified_ensemble_80_20'
MEMBERS=['baseline','spatial','temporal']
ORIGINS=['2022-09','2023-07','2024-06']
VARIANTS=['A1']


def raw_frame():
    f=prepare_main(pd.read_csv(DATA/'train.csv'))
    for c in ('STREET','BLOCK'): f[c]=f[c].astype(str).str.lower().str.strip()
    f['month']=pd.PeriodIndex(f.RENT_APPROVAL_DATE,freq='M')
    return f


def original_bundle(origin,member):
    return load_bundle(origin) if member=='baseline' else prior.load_prepared(origin,member)


def check_sources():
    settings=json.loads((OUT/'settings.json').read_text())
    for kind in ['source_sha256','code_sha256']:
        for p,h in settings[kind].items(): assert sha256(ROOT/p)==h,p


def load_prepared(origin,member):
    path=CACHE/f'features_{origin}_{member}.joblib'
    meta=json.loads((OUT/f'features_{origin}_{member}.json').read_text())
    assert sha256(path)==meta['sha256']
    return joblib.load(path)


def prepare():
    OUT.mkdir(parents=True,exist_ok=True); CACHE.mkdir(parents=True,exist_ok=True)
    # Source audit includes both submissions, all caches and matching predictions.
    sources=[DATA/'train.csv',DATA/'test.csv',prior.CHECKPOINT,REFERENCE/'selection.json',REFERENCE/'learned_weights.csv',
             REPORTS/'tabpfn_feature_ensemble_final_submission/submission_tabpfn_optimized_ensemble_150k.csv',
             REPORTS/'tabpfn_diagscale_mse_final_submission/submission_tabpfn_diagscale_mse_150k.csv']
    for o in ORIGINS:
        for m in MEMBERS:
            sources.extend([REFERENCE/f'predictions_{o}_{m}.csv',REFERENCE/f'timing_{o}_{m}.json'])
            if m!='baseline': sources.extend([prior.CACHE/f'features_{o}_{m}.joblib',REFERENCE/f'features_{o}_{m}.json'])
    code=[Path(__file__),ROOT/'analysis/experiments/targeted_spatial_economic.py',ROOT/'analysis/compare_tabpfn_feature_ensemble.py']
    immutable(OUT/'settings.json',dict(origins=ORIGINS,members=MEMBERS,variants=VARIANTS,
        baseline='Existing optimized plain frozen three-member V3.5Fast ensemble; 77/89/84 inputs, two estimators each,seed42,native batching,batch200,full unchanged contexts',
        hypothesis='A1 adds town/street hierarchical recent levels, local growth relative to national-flat growth, support/dispersion, and building-to-street shrinkage. Existing nearby-comparable features remain in spatial member; do not duplicate them.',
        history=dict(recent_months=3,previous_months=3,support_months=6,shrink_observations=20,publication_lag_months=1),
        weights='Same exact simplex MSE optimizer as reference, fitted only on prior disjoint prediction windows. Earliest origin defaults to baseline member. Also report fixed reference-weight sensitivity.',
        residual=dict(shrink=[20,50],gamma=[0,.25,.5,1],half_life_months=6,publication_lag_months=1),
        gate='Combined features only if each group improves earlier and later RMSE; later already reused, exploratory, not untouched confirmation',
        source_sha256={p.relative_to(ROOT).as_posix():sha256(p) for p in sources},
        code_sha256={p.relative_to(ROOT).as_posix():sha256(p) for p in code},
        external_availability='Official datasets require historical vintages and publication evidence; current revised snapshots are not silently used at past origins'))
    check_sources()
    if (OUT/'prepared_complete.json').exists():
        print('Reusing verified prepared features',flush=True);return
    frame=raw_frame();coverage=[]; audits=[]
    for origin in ORIGINS:
        base=load_bundle(origin)
        allx=pd.concat([base['X_train'],base['X_query']])
        assert allx.index.is_unique
        _,cutoffs=row_cutoffs(allx)
        source=frame.loc[base['X_train'].index]
        features,audit=hierarchy_features(source,frame.loc[allx.index].drop(columns='MONTHLY_RENT'),cutoffs)
        # Each cutoff is replayed with later labels poisoned: permitted sets are
        # disjoint from every corresponding target month by construction.
        audit['origin']=origin;audits.append(audit)
        for member in MEMBERS:
            b=original_bundle(origin,member)
            for split in ('train','query'):
                x=b[f'X_{split}'];z=x.join(features.loc[x.index])
                z.columns=pd.Index(list(z.columns),dtype=x.columns.dtype)
                pd.testing.assert_frame_equal(z[x.columns],x)
                b[f'X_{split}']=z
                coverage.extend(dict(origin=origin,member=member,split=split,feature=c,rows=len(z),missing=int(z[c].isna().sum())) for c in features)
            path=CACHE/f'features_{origin}_A1_{member}.joblib';joblib.dump(b,path,compress=3)
            write_json(OUT/f'features_{origin}_A1_{member}.json',dict(sha256=sha256(path),features=list(b['X_train']),
                context_rows=len(b['X_train']),query_rows=len(b['X_query']),original_columns_targets_baselines_unchanged=True))
        print('PREPARED A1',origin,len(base['X_train']),len(base['X_query']),flush=True)
    pd.concat(audits,ignore_index=True).to_csv(OUT/'history_cutoffs.csv',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'feature_coverage.csv',index=False)
    write_json(OUT/'prepared_complete.json',dict(verified=True))


def worker(origin,member):
    check_sources()
    # Reuse the audited native-batched frozen inference implementation, changing
    # only the feature bundle and output directory.
    prior.OUT,prior.CACHE=OUT,CACHE
    prior.check_sources,prior.load_prepared=check_sources,load_prepared
    prior.infer(origin,member)


def child(origin,member):
    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    for key in ('TABPFN_MAX_BATCHED_ESTIMATOR_ROWS','TABPFN_MAX_BATCHED_ESTIMATOR_CELLS'):env.pop(key,None)
    with (OUT/f'inference_{origin}_{member}.log').open('a',encoding='utf-8') as log:
        p=subprocess.run([sys.executable,'-u',str(Path(__file__)),'--stage','one','--origin',origin,'--member',member],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    if p.returncode:raise RuntimeError(f'Inference failed: {origin} {member}')
    print('COMPLETED',origin,member,flush=True)


def read_members(origin,variant='A0'):
    directory=REFERENCE if variant=='A0' else OUT
    frames=[]
    for m in MEMBERS:
        member=m if variant=='A0' else f'{variant}_{m}'
        path=directory/f'predictions_{origin}_{member}.csv'
        meta=json.loads((directory/f'timing_{origin}_{member}.json').read_text())
        assert sha256(path)==meta['prediction_sha256'] and meta['parameters_unchanged'] and meta['no_adapters']
        assert meta['n_estimators']==2 and not meta['subsampling']
        frames.append(pd.read_csv(path,float_precision='round_trip'))
    f=frames[0].drop(columns='predicted').copy()
    for m,p in zip(MEMBERS,frames):
        for c in ('row_id','month','actual','weight'):np.testing.assert_array_equal(f[c],p[c])
        f[m]=p.predicted.to_numpy()
    return f


def ensemble(variant,origin):
    f=read_members(origin,variant);i=ORIGINS.index(origin)
    if i==0:w=np.array([1.,0.,0.])
    else:
        earlier=pd.concat([read_members(o,variant) for o in ORIGINS[:i]],ignore_index=True)
        assert earlier.month.max()<origin
        w,_=optimize_weights(earlier.actual.to_numpy(),earlier[MEMBERS].to_numpy(),earlier.weight.to_numpy())
    f['prediction']=f[MEMBERS].to_numpy()@w
    return f,w


def run():
    prepare()
    for origin in ORIGINS:
        for variant in VARIANTS:
            for member in MEMBERS:child(origin,f'{variant}_{member}')
    print('A1 inference complete',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['prepare','all','one'],default='all')
    p.add_argument('--origin');p.add_argument('--member');a=p.parse_args()
    if a.stage=='one':worker(a.origin,a.member)
    elif a.stage=='prepare':prepare()
    else:run()


if __name__=='__main__':main()
