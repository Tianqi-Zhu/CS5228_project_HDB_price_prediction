"""Earlier-weighted A/B/C/D/E comparison with genuinely historical A adapters."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
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
from analysis.compare_tabpfn_hierarchical_latent import load_bundle
from analysis.experiments.temporal_feature_views import simplex_weights
from analysis.experiments.feature_diversified_ensemble import combine_predictions
from analysis.experiments.stock_index_regions import block_positions

OUT=REPORTS/'tabpfn_abcde_ensemble'
CACHE=ROOT/'.cache/tabpfn_abcde_ensemble'
FROZEN=REPORTS/'tabpfn_feature_diversified_ensemble_80_20'
RELATIVE=REPORTS/'tabpfn_temporal_view_ensembles_80_20'
OOF=REPORTS/'tabpfn_spatial_temporal_residual_80_20'
OOF_CACHE=ROOT/'.cache/tabpfn_spatial_temporal_residual_80_20'
MAIN_A=REPORTS/'tabpfn_components_80_20/outer/diag_mse_only/predictions.csv'
ORIGINS=['2022-09','2023-07','2024-06']
MEMBERS=list('ABCDE')
GROUPS=['A','BDE','CDE','AC','ABDE','ACDE','ABCDE']


def prediction_path(origin,member):
    if member=='A':return MAIN_A if origin=='2024-06' else OUT/f'oof_predictions_{origin}.csv'
    if member=='C':return RELATIVE/f'predictions_{origin}_relative_history.csv'
    name={'B':'baseline','D':'temporal','E':'spatial'}[member]
    return FROZEN/f'predictions_{origin}_{name}.csv'


def check_sources():
    settings=json.loads((OUT/'settings.json').read_text())
    for group in ['source_sha256','code_sha256']:
        for name,digest in settings[group].items():assert sha256(ROOT/name)==digest,name


def prepare():
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    sources=[DATA/'train.csv',MAIN_A,FROZEN/'selection.json',FROZEN/'learned_weights.csv',
             OOF_CACHE/'adapter_2022-09.pt',OOF/'adapter_2022-09.json',OOF/'features_2022-09.json',
             REPORTS/'tabpfn_components_80_20/candidates/diag_mse_only/calibration.json',
             REPORTS/'tabpfn_feature_ensemble_final_submission/submission_tabpfn_optimized_ensemble_150k.csv',
             REPORTS/'tabpfn_diagscale_mse_final_submission/submission_tabpfn_diagscale_mse_150k.csv']
    for origin in ORIGINS:
        for member in MEMBERS[1:]:sources.append(prediction_path(origin,member))
    code=[Path(__file__),ROOT/'analysis/experiments/temporal_feature_views.py',ROOT/'analysis/compare_tabpfn_spatial_temporal_residual.py',
          ROOT/'analysis/compare_tabpfn_components_80_20.py',ROOT/'analysis/experiments/tabpfn_component_adaptation.py']
    immutable(OUT/'settings.json',dict(members=dict(A='77-feature MSE-only DiagScale, one inference view',B='77-feature frozen Fast, two views',
        C='77-feature relative-history frozen Fast, two views',D='84-feature seasonal frozen Fast, two views',E='89-feature nearby-rent frozen Fast, two views'),
        origins=ORIGINS,groups=GROUPS,optimizer='Existing exact active-face convex MSE simplex, nonnegative sum-one weights, no intercept',
        selection='July2023 nested-transfer RMSE with weights from September2022 only; ties favor fewer active models. Refit each group on both earlier disjoint windows for June2024 evaluation. No new main-period fitted/oracle weights.',
        first_origin='Equal weights, no earlier calibration; reported as cold start only. Individual September predictions calibrate July weights.',
        primary_comparison='ABCDE versus matched CDE and BDE; retain original locked BDE as deployment reference; report A and focused AC/ABDE/ACDE as declared candidates',
        A_recipe=dict(epochs=1,lr=.01,weight_decay=.01,seed=42,task_rows=1024,query_fraction=.2,training_views=2,inference_views=1,crps=0,mse=1),
        A_provenance='Reuse September2022 adapter only after exact complete context-frame agreement; newly fit July2023 adapter using only rows through June2023, with no query evaluation or checkpoint selection. Main uses unchanged preDecember2023 selected adapter.',
        cost='Reuse every frozen prediction and main A prediction. Only historical A inference and one historical adapter fit are new.',
        evaluation='All available labeled periods already examined. Existing main scores/error correlations informed hypotheses; this is retrospective development, not independent validation.',
        pre_analysis='The five-member simplex contains CDE and BDE, so optimized calibration loss cannot increase. Added A has strongest standalone RMSE; A/C error correlation is lower than A/B. More weights can overfit shared errors. Prefer zero/sparse weights if unsupported.',
        promotion='Only consider a separate local candidate CSV if earlier-selected family beats both A and locked BDE on later RMSE, improves nested transfer over both, and main block upper95 difference versus both is below zero. Otherwise preserve current candidates and report.',
        source_sha256={p.relative_to(ROOT).as_posix():sha256(p) for p in sources},
        code_sha256={p.relative_to(ROOT).as_posix():sha256(p) for p in code}))
    check_sources()
    (OUT/'pre_analysis.txt').write_text('Before fitting new weights: ABCDE can reproduce CDE/BDE by zeroing A/B or A/C. This guarantees only weakly lower optimized calibration MSE, not lower later risk. Existing observed A RMSE455.071, B455.396,C455.370,D455.804,E456.482; A/C signed-error rho~0.9934 versus A/B~0.9965. The targeted candidates are AC,ACDE,ABDE,ABCDE and reference CDE/BDE/A. Existing inspected main diagnostics inform this retrospective hypothesis; weight selection remains earlier-only.\n')
    raw=pd.read_csv(DATA/'train.csv')
    for origin in ORIGINS[:-1]:
        path=CACHE/f'features_{origin}.joblib'
        if path.exists():assert sha256(path)==json.loads((OUT/f'features_{origin}.json').read_text())['sha256'];continue
        b=load_bundle(origin)
        b['train_month']=raw.loc[b['X_train'].index,'RENT_APPROVAL_DATE'].to_numpy()
        assert max(b['train_month'])<origin<=min(b['query_month'])
        if origin=='2022-09':
            old=joblib.load(OOF_CACHE/'features_2022-09.joblib')
            pd.testing.assert_frame_equal(old['X_train'],b['X_train'])
            np.testing.assert_array_equal(old['y_train'],b['y_train'])
            np.testing.assert_array_equal(old['train_baseline'],b['train_baseline'])
            source=OOF_CACHE/'adapter_2022-09.pt';meta=json.loads((OOF/'adapter_2022-09.json').read_text())
            assert sha256(source)==meta['adapter_sha256'] and not meta['validation_targets_used']
            assert not meta['checkpoint_selection'] and meta['epochs']==1
            shutil.copy2(source,CACHE/'adapter_2022-09.pt')
            write_json(OUT/'adapter_2022-09.json',dict(**meta,reused=True,source=str(source.relative_to(ROOT)),context_exact_replay=True))
        joblib.dump(b,path,compress=3)
        write_json(OUT/f'features_{origin}.json',dict(origin=origin,end_exclusive=str(pd.Period(max(b['query_month']),'M')+1),
            context_rows=len(b['X_train']),query_rows=len(b['X_query']),context_max_month=max(b['train_month']),
            query_min_month=min(b['query_month']),query_max_month=max(b['query_month']),
            context_ids_sha256=hashlib_array(b['X_train'].index),query_ids_sha256=hashlib_array(b['X_query'].index),
            sha256=sha256(path),fold_target_labels_in_context=False,market_last_allowed_month=str(pd.Period(origin,'M')-1)))
        print('PREPARED A',origin,len(b['X_train']),len(b['X_query']),flush=True)


def worker(origin):
    check_sources()
    import analysis.compare_tabpfn_spatial_temporal_residual as prior
    prior.OUT,prior.CACHE=OUT,CACHE
    prior.one_fold(origin)


def load_predictions(origin):
    frames=[]
    for member in MEMBERS:
        p=prediction_path(origin,member)
        if member=='A' and origin!='2024-06':
            provenance=json.loads((OUT/f'oof_provenance_{origin}.json').read_text())
            assert sha256(p)==provenance['prediction_sha256'] and provenance['context_max_month']<origin
            assert not provenance['validation_targets_used'] and not provenance['fold_target_labels_in_context']
        f=pd.read_csv(p,float_precision='round_trip')
        assert len(f)==(28995 if origin=='2024-06' else 2000) and f.row_id.is_unique
        if frames:
            for c in ['row_id','month','actual']:np.testing.assert_array_equal(f[c],frames[0][c])
        frames.append(f)
    base=frames[1][['row_id','origin','month','actual','weight']].copy()
    for name,f in zip(MEMBERS,frames):base[name]=f.predicted.to_numpy()
    return base


def score(f,prediction):
    error=np.asarray(prediction)-f.actual.to_numpy();w=f.weight.to_numpy()
    return dict(rmse_sgd=float(np.sqrt(np.average(error**2,weights=w))),mae_sgd=float(np.average(abs(error),weights=w)),bias_sgd=float(np.average(error,weights=w)))


def fit_group(f,group):
    indices=[MEMBERS.index(m) for m in group]
    w,info=simplex_weights(f.actual.to_numpy(),f[list(group)].to_numpy(),f.weight.to_numpy())
    full=np.zeros(5);full[indices]=w
    return full,info


def selection():
    check_sources();early,later=[load_predictions(o) for o in ORIGINS[:-1]]
    rows=[]
    for group in GROUPS:
        w,info=fit_group(early,group)
        rows.append(dict(group=group,weights=w.tolist(),**score(later,later[MEMBERS].to_numpy()@w),active_members=info['active_members']))
    best=min(rows,key=lambda r:(r['rmse_sgd'],r['active_members'],GROUPS.index(r['group'])))
    pool=pd.concat([early,later],ignore_index=True)
    main_weights={g:fit_group(pool,g)[0].tolist() for g in GROUPS}
    value=dict(selected=best['group'],nested_results=rows,main_weights=main_weights,
        weight_fit_max_month=pool.month.max(),family_selection_targets=[later.month.min(),later.month.max()],
        nested_weight_fit_max_month=early.month.max(),main_labels_used=False,
        input_predictions={str(prediction_path(o,m).relative_to(ROOT)):sha256(prediction_path(o,m)) for o in ORIGINS[:-1] for m in MEMBERS})
    immutable(OUT/'selection.json',value)
    print('LOCKED SELECTION',best['group'],{r['group']:round(r['rmse_sgd'],6) for r in rows},flush=True)
    return value


def interval(f,candidate,reference):
    stats=[]
    for month,p in f.groupby('month',sort=True):
        w=p.weight.to_numpy();y=p.actual.to_numpy()
        stats.append([w.sum(),np.sum(w*(p[candidate]-y)**2),np.sum(w*(p[reference]-y)**2)])
    a=np.asarray(stats);pos=block_positions(len(a),repeats=5000,block=3)
    b=a[pos].sum(axis=1);delta=np.sqrt(b[:,1]/b[:,0])-np.sqrt(b[:,2]/b[:,0])
    low,high=np.quantile(delta,[.025,.975])
    return dict(ci_low=float(low),ci_high=float(high),months=len(a),block_months=3)


def evaluate():
    choice=selection();check_sources();weights=[];allrows=[];scores=[];monthly=[];subgroups=[];comparisons=[]
    raw=pd.read_csv(DATA/'train.csv')
    from analysis.data import prepare_main
    raw=prepare_main(raw)
    cached_weights=pd.read_csv(FROZEN/'learned_weights.csv')
    for origin in ORIGINS:
        f=load_predictions(origin)
        for group in GROUPS:
            if origin==ORIGINS[0]:
                w=np.array([1/len(group) if m in group else 0 for m in MEMBERS])
            elif origin==ORIGINS[1]:w=np.asarray(next(r['weights'] for r in choice['nested_results'] if r['group']==group))
            else:w=np.asarray(choice['main_weights'][group])
            f[group if group!='A' else 'A_single']=combine_predictions(f[MEMBERS].to_numpy(),w)
            weights.append(dict(origin=origin,group=group,**dict(zip(MEMBERS,w)),cold_start=origin==ORIGINS[0]))
        z=cached_weights[(cached_weights.origin==origin)&(cached_weights.model=='optimized')].iloc[0]
        f['BDE_existing']=f[['B','D','E']].to_numpy()@np.array([z.baseline_weight,z.temporal_weight,z.spatial_weight])
        np.testing.assert_allclose(f.BDE_existing,f.BDE,atol=1e-8) if origin!=ORIGINS[0] else None
        f['selected']=f['A_single' if choice['selected']=='A' else choice['selected']]
        for group in ['CDE','BDE','ABCDE']:
            f[group+'_equal']=f[list(group)].mean(axis=1)
        names=['A_single',*GROUPS[1:],'B','C','D','E','BDE_existing','selected','CDE_equal','BDE_equal','ABCDE_equal']
        for name in names:
            s=score(f,f[name]);base=score(f,f.BDE_existing)
            scores.append(dict(origin=origin,model=name,rows=len(f),**s,delta_vs_existing_BDE=s['rmse_sgd']-base['rmse_sgd']))
            for month,p in f.groupby('month',sort=True):
                m,b=score(p,p[name]),score(p,p.BDE_existing)
                monthly.append(dict(origin=origin,month=month,model=name,rows=len(p),**m,delta_vs_existing_BDE=m['rmse_sgd']-b['rmse_sgd']))
            if origin==ORIGINS[-1]:
                for dimension in ['TOWN_CLEAN','FLAT_TYPE_CLEAN']:
                    labels=raw.loc[f.row_id,dimension].to_numpy()
                    for label in sorted(set(labels)):
                        p=f.loc[labels==label];m,b=score(p,p[name]),score(p,p.BDE_existing)
                        subgroups.append(dict(dimension=dimension,category=label,model=name,rows=len(p),**m,delta_vs_existing_BDE=m['rmse_sgd']-b['rmse_sgd']))
        if origin==ORIGINS[-1]:
            for candidate,reference in [('ABCDE','CDE'),('ABCDE','BDE'),('ACDE','CDE'),('ABDE','BDE'),('selected','BDE_existing'),('selected','A'),('CDE','BDE'),('AC','A')]:
                comparisons.append(dict(candidate=candidate,reference=reference,delta_rmse=score(f,f[candidate])['rmse_sgd']-score(f,f[reference])['rmse_sgd'],**interval(f,candidate,reference)))
            error=f[MEMBERS].sub(f.actual,axis=0)
            error.corr().to_csv(OUT/'error_correlations.csv')
            (error.to_numpy().T@error.to_numpy()/len(error)).tofile(CACHE/'main_error_second_moments.bin')
            pd.DataFrame(error.to_numpy().T@error.to_numpy()/len(error),index=MEMBERS,columns=MEMBERS).to_csv(OUT/'error_second_moments.csv')
        allrows.append(f)
    scores=pd.DataFrame(scores);weights=pd.DataFrame(weights);monthly=pd.DataFrame(monthly);comparisons=pd.DataFrame(comparisons)
    pd.concat(allrows,ignore_index=True).to_csv(OUT/'predictions_all_origins.csv',index=False,float_format='%.17g')
    scores.to_csv(OUT/'scores.csv',index=False);weights.to_csv(OUT/'weights.csv',index=False)
    monthly.to_csv(OUT/'monthly_scores.csv',index=False);pd.DataFrame(subgroups).to_csv(OUT/'subgroup_scores.csv',index=False)
    comparisons.to_csv(OUT/'paired_comparisons.csv',index=False)
    from analysis.summarize_tabpfn_abcde import report
    report(choice,scores,weights,comparisons,monthly)
    check_sources()
    write_json(OUT/'verification.json',dict(verified=True,selected=choice['selected'],full_main_rows=28995,
        historical_cases=4000,historical_target_windows_disjoint=True,original_BDE_reproduced=True,
        no_main_labels_used_for_weights=True,no_earlier_A_adapter_uses_future_labels=True,sources_unchanged=True))
    print(scores[scores.origin=='2024-06'][['model','rmse_sgd','delta_vs_existing_BDE']].to_string(index=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['all','prepare','one','select','evaluate'],default='all');p.add_argument('--origin');a=p.parse_args()
    if a.stage=='one':worker(a.origin);return
    if a.stage=='select':selection();return
    if a.stage=='evaluate':evaluate();return
    prepare()
    if a.stage=='prepare':return
    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    for key in ['TABPFN_MAX_BATCHED_ESTIMATOR_ROWS','TABPFN_MAX_BATCHED_ESTIMATOR_CELLS']:env.pop(key,None)
    for origin in ORIGINS[:-1]:
        with (OUT/f'inference_A_{origin}.log').open('a',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,'-u',str(Path(__file__)),'--stage','one','--origin',origin],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'Historical A failed: {origin}')
        print('COMPLETED A',origin,flush=True)
    evaluate()


if __name__=='__main__':main()
