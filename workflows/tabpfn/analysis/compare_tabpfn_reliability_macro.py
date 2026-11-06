"""Nested reliability and forecast-derived COE inputs, with fixed residual targets."""
from __future__ import annotations
import argparse
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
import torch
from threadpoolctl import threadpool_limits

from analysis.config import ROOT, DATA, AUX, REPORTS
from analysis.data import sha256
from analysis.compare_local_economy_landscape import immutable, write_json
from analysis.compare_tabpfn_hierarchical_latent import load_bundle, feature_source, BASE
from analysis.compare_tabpfn_v3_5_fast_80_20 import CHECKPOINT
from analysis import compare_tabpfn_feature_ensemble as prior
from analysis.experiments.reliability_macro_features import VARIANTS, add_forecast, forecast_table, dates_and_origins
from analysis.experiments.temporal_sequence_ablation import build_panel, table
from analysis.experiments.hierarchical_latent_residual import weighted_metrics

OUT = REPORTS / 'tabpfn_reliability_macro_80_20'
CACHE = ROOT / '.cache/tabpfn_reliability_macro_80_20'
REFERENCE = REPORTS / 'tabpfn_feature_diversified_ensemble_80_20'
ORIGINS = ['2022-09', '2023-07', '2024-06']
MEMBERS = ['baseline', *VARIANTS]
ADAPTER = ROOT / '.cache/tabpfn_components_80_20/candidates/diag_mse_only/adapter.pt'


def verify_sources():
    settings = json.loads((OUT / 'settings.json').read_text())
    for group in ('source_sha256', 'code_sha256'):
        for path, digest in settings[group].items():
            assert sha256(ROOT / path) == digest, path
    return settings


def load_prepared(origin, member):
    path = CACHE / f'features_{origin}_{member}.joblib'
    manifest = json.loads((OUT / f'features_{origin}_{member}.json').read_text())
    assert sha256(path) == manifest['sha256']
    return joblib.load(path)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    sources = [DATA/'train.csv', AUX/'sg-hdb-block.csv', AUX/'sg-coe-prices.csv', AUX/'sg-stock-prices.csv',
               ROOT/'analysis/configs/sti_etf_weights_2020_12.csv', CHECKPOINT, BASE,
               ADAPTER, REPORTS/'tabpfn_components_80_20/outer/diag_mse_only/predictions.csv',
               REPORTS/'tabpfn_components_80_20/candidates/diag_mse_only/calibration.json']
    sources += [REPORTS/'tabpfn_diagscale_mse_final_submission/submission_tabpfn_diagscale_mse_150k.csv',
                REPORTS/'tabpfn_feature_ensemble_final_submission/submission_tabpfn_optimized_ensemble_150k.csv']
    for origin in ORIGINS[:-1]:
        sources.extend(feature_source(origin))
    for origin in ORIGINS:
        sources += [REFERENCE/f'predictions_{origin}_baseline.csv', REFERENCE/f'timing_{origin}_baseline.json']
    code = [Path(__file__), ROOT/'analysis/experiments/reliability_macro_features.py',
            ROOT/'analysis/experiments/temporal_sequence_ablation.py', ROOT/'analysis/experiments/stock_index_regions.py',
            ROOT/'analysis/compare_tabpfn_feature_ensemble.py', ROOT/'analysis/compare_tabpfn_hierarchical_latent.py',
            ROOT/'analysis/experiments/aligned_temporal_features.py', ROOT/'EXPERIMENT_PLAN_TABPFN_RELIABILITY_MACRO.md']
    immutable(OUT/'settings.json', dict(origins=ORIGINS, members=MEMBERS,
        feature_counts=dict(baseline=77,reliability=82,rent_forecast=84,coe_increment=85),
        input_ablation='Nested additions; original77 features, labels and residual baselines unchanged',
        inference='Plain frozen V3.5Fast,seed42,two estimators,native batching,batch200,full unchanged contexts',
        macro='Matched12-month8-hidden LSTMs,seed42,100epochs;direct10heads;3 mature monthly labels per head;nativeNaN warmup',
        macro_information='Original per-row feature origin minus1month;both rent andCOE stop there;growth plateaus after head10',
        selection='Minimum pooled earlier weighted MSE including baseline;lock before later inference;all variants reported',
        diagscale_confirmation='Only if added features win earlier;fixed preDec2023 MSE adapter,one inference view,no retraining',
        long_horizon='Separate cell-mean temporal-baseline study,not a TabPFN target/baseline refit',
        validation_status='Repeatedly examined development period;no untouched labeled test',
        source_sha256={p.relative_to(ROOT).as_posix():sha256(p) for p in sources},
        code_sha256={p.relative_to(ROOT).as_posix():sha256(p) for p in code},
        packages={p:version(p) for p in ('pandas','numpy','torch','tabpfn')}))
    verify_sources()
    if (OUT/'prepared_complete.json').exists():
        for o in ORIGINS:
            for m in VARIANTS: load_prepared(o,m)
        print('PREPARED caches verified',flush=True); return
    panel,_ = build_panel(pd.read_csv(DATA/'train.csv'),pd.read_csv(AUX/'sg-stock-prices.csv'),
                          pd.read_csv(AUX/'sg-coe-prices.csv'),pd.read_csv(ROOT/'analysis/configs/sti_etf_weights_2020_12.csv'),
                          pd.read_csv(AUX/'sg-hdb-block.csv'))
    panel = panel.loc[:'2024-04']
    panel.to_csv(OUT/'macro_panel.csv',index_label='month')
    forecasts = forecast_table(panel)
    forecasts.to_csv(OUT/'national_forecasts.csv',index=False)
    audits = []
    # Refit with every later source observation poisoned at three row cutoffs.
    for position in (19,29,39):
        cut=panel.index[position]; poisoned=panel.copy();poisoned.iloc[position+1:]*=123.
        replay=forecast_table(poisoned,cutoffs=[cut])
        expected=forecasts[forecasts.cutoff.eq(str(cut))].reset_index(drop=True)
        pd.testing.assert_frame_equal(expected,replay)
        audits.append(dict(cutoff=str(cut),future_rents_and_coe_poisoned=True,forecast_replay_exact=True))
    coverage=[]
    for origin in ORIGINS:
        b=load_bundle(origin)
        ref,meta=read_prediction(origin,'baseline')
        np.testing.assert_array_equal(ref.row_id,b['X_query'].index)
        np.testing.assert_array_equal(ref.actual,b['y_query'])
        for member in VARIANTS:
            frames={}
            for split in ('train','query'):
                x=b[f'X_{split}']; z=add_forecast(x,forecasts,member)
                pd.testing.assert_frame_equal(x,z[x.columns])
                frames[f'X_{split}']=z
                assert z.shape[1]==dict(reliability=82,rent_forecast=84,coe_increment=85)[member]
                for c in z:
                    if c not in x:
                        coverage.append(dict(origin=origin,member=member,split=split,feature=c,
                                             missing=int(z[c].isna().sum()),rows=len(z)))
            path=CACHE/f'features_{origin}_{member}.joblib'
            joblib.dump({**b,**frames},path,compress=3)
            write_json(OUT/f'features_{origin}_{member}.json',dict(sha256=sha256(path),
                feature_names=list(frames['X_train']),context_rows=len(b['X_train']),query_rows=len(b['X_query']),
                original_columns_targets_baselines_ids_unchanged=True))
        print('PREPARED',origin,len(b['X_train']),len(b['X_query']),flush=True)
    pd.DataFrame(coverage).to_csv(OUT/'feature_coverage.csv',index=False)
    write_json(OUT/'causality_audit.json',dict(verified=True,checks=audits,reference_adjustment_fitted_2021_only=True,
                                            no_macro_forecast_uses_post_April2024_labels=True))
    write_json(OUT/'prepared_complete.json',dict(verified=True,source_hashes_verified=True))


def read_prediction(origin,member):
    directory=REFERENCE if member=='baseline' else OUT
    path=directory/f'predictions_{origin}_{member}.csv'
    meta=json.loads((directory/f'timing_{origin}_{member}.json').read_text())
    assert sha256(path)==meta['prediction_sha256'] and meta['parameters_unchanged']
    assert meta['n_estimators']==2 and meta['no_adapters'] and not meta['subsampling']
    return pd.read_csv(path,float_precision='round_trip'),meta


def aligned(origin):
    base,_=read_prediction(origin,'baseline'); frame=base.drop(columns='predicted').copy()
    for member in MEMBERS:
        other,_=read_prediction(origin,member)
        for c in ('row_id','origin','month','actual','weight'):
            np.testing.assert_array_equal(frame[c],other[c])
        frame[member]=other.predicted.to_numpy()
    return frame


def selection():
    frames=[aligned(o) for o in ORIGINS[:-1]]
    pool=pd.concat(frames,ignore_index=True)
    scores={m:weighted_metrics(pool.actual,pool[m],pool.weight)['rmse_sgd'] for m in MEMBERS}
    best=min(MEMBERS,key=lambda m:(scores[m],MEMBERS.index(m)))
    nested=min(MEMBERS,key=lambda m:(weighted_metrics(frames[0].actual,frames[0][m],frames[0].weight)['rmse_sgd'],MEMBERS.index(m)))
    immutable(OUT/'selection.json',dict(selected=best,earlier_rmse=scores,nested_July2023_selected=nested,
        maximum_selection_target_month=pool.month.max(),query_rows=len(pool),later_labels_used=False))
    print('LOCKED EARLIER SELECTION',best,scores,flush=True)


def worker(origin,member):
    verify_sources()
    if member=='diag_confirm': return diag_worker()
    prior.OUT,prior.CACHE=OUT,CACHE
    prior.check_sources,prior.load_prepared=verify_sources,load_prepared
    prior.infer(origin,member)


def child(origin,member):
    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    for key in ('TABPFN_MAX_BATCHED_ESTIMATOR_ROWS','TABPFN_MAX_BATCHED_ESTIMATOR_CELLS'):
        env.pop(key,None)
    path=OUT/f'inference_{origin}_{member}.log'
    with path.open('a',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,'-u',str(Path(__file__)),'--stage','one','--origin',origin,'--member',member],
                              cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError(f'Worker failed; see {path}')
    print('COMPLETED',origin,member,flush=True)


def diag_worker():
    from analysis.compare_tabpfn_components_80_20 import make_estimator
    choice=json.loads((OUT/'selection.json').read_text())['selected']
    assert choice!='baseline'
    output=OUT/'diagscale_confirmation.csv'
    if output.exists() and (OUT/'diagscale_confirmation.json').exists():
        assert sha256(output)==json.loads((OUT/'diagscale_confirmation.json').read_text())['prediction_sha256'];return
    b=load_prepared('2024-06',choice); x,q=b['X_train'],b['X_query']
    torch.set_num_threads(1);torch.cuda.reset_peak_memory_stats()
    model=make_estimator(dict(component='diagscale',lr=.01,crps=0),ADAPTER,
                         [i for i,c in enumerate(x) if isinstance(x[c].dtype,pd.CategoricalDtype)],views=1)
    start=time.perf_counter()
    model.fit(x,np.asarray(b['y_train'])-b['train_baseline'])
    signatures=[p._version for m in model.models_ for p in m.parameters()]
    values=[]
    for i in range(0,len(q),200):
        torch.cuda.empty_cache()
        values.extend(np.asarray(model.predict(q.iloc[i:i+200],output_type='mean'),float)+b['query_baseline'][i:i+200])
        if i%2000==0: print('DIAGSCALE',i+min(200,len(q)-i),'/',len(q),flush=True)
    assert signatures==[p._version for m in model.models_ for p in m.parameters()]
    base=pd.read_csv(REPORTS/'tabpfn_components_80_20/outer/diag_mse_only/predictions.csv')
    np.testing.assert_array_equal(base.row_id,q.index);np.testing.assert_array_equal(base.actual,b['y_query'])
    result=base.rename(columns={'predicted':'baseline'}).assign(candidate=np.asarray(values))
    result.to_csv(output,index=False,float_format='%.17g')
    write_json(OUT/'diagscale_confirmation.json',dict(selected=choice,adapter_sha256=sha256(ADAPTER),
        adapter_fixed=True,adapter_retrained_on_new_features=False,parameters_unchanged=True,views=1,
        features=len(x.columns),context_rows=len(x),seconds=time.perf_counter()-start,
        baseline_metrics=weighted_metrics(result.actual,result.baseline),candidate_metrics=weighted_metrics(result.actual,result.candidate),
        prediction_sha256=sha256(output)))


def long_horizon():
    from analysis.experiments.spatiotemporal_hybrid import spatial_frame,MONTH
    from analysis.experiments.aligned_temporal_features import LocalTemporalModel,AREA,TYPE
    path=OUT/'long_horizon_predictions.csv'
    if path.exists() and (OUT/'long_horizon_scores.csv').exists(): return
    frame=spatial_frame(pd.read_csv(DATA/'train.csv')); rows=[]
    for start in ('2022-09','2023-01','2023-07','2023-12'):
        origin=pd.Period(start,'M')-1
        past=frame[frame[MONTH]<=origin]
        model=LocalTemporalModel(past); params=model.fit_at(origin)
        position=len(model.months)-1
        future=frame[(frame[MONTH]>origin)&(frame[MONTH]<=origin+16)]
        cells=future.groupby([MONTH,AREA,TYPE],observed=True).MONTHLY_RENT.agg(['mean','size'])
        for (month,area,kind),cell in cells.iterrows():
            g=model.kind_index.get(str(kind));l=model.group_index.get((str(area),str(kind)))
            if g is None: continue
            horizon=month.ordinal-origin.ordinal
            for policy in ('plateau10','damped16','persistence'):
                h=min(horizon,10) if policy=='plateau10' else horizon
                phi=None if policy=='persistence' else params['phi']
                decay=1. if policy=='persistence' else params['decay']
                if policy=='persistence': h=1
                prediction=(model.local_prediction(position,h,phi,params['shrink'],decay)[l] if l is not None else
                            model.global_prediction(position,h,phi)[g])
                rows.append(dict(origin=start,month=str(month),horizon=horizon,band='11-16' if horizon>10 else '1-10',
                                 phase='development' if str(month)<'2024-06' else 'later_reused',
                                 area=area,flat_type=kind,n=int(cell['size']),actual=float(cell['mean']),
                                 policy=policy,predicted=max(1.,float(prediction))))
        print('LONG HORIZON',start,'through',str(origin+16),flush=True)
    predictions=pd.DataFrame(rows);predictions.to_csv(path,index=False)
    rows=[]
    for keys,p in predictions.groupby(['phase','band','policy']):
        rows.append(dict(zip(['phase','band','policy'],keys),cells=len(p),unique_target_months=p.month.nunique(),
                         rental_case_weights=int(p.n.sum()),**weighted_metrics(p.actual,p.predicted,p.n)))
    scores=pd.DataFrame(rows);scores.to_csv(OUT/'long_horizon_scores.csv',index=False)
    eligible=scores[scores.phase.eq('development')&scores.band.eq('11-16')].sort_values('rmse_sgd')
    write_json(OUT/'long_horizon_selection.json',dict(policy=eligible.policy.iloc[0],
                objective='development11-16-month cell-mean RMSE',selection_targets_before='2024-06',
                independent_TabPFN_model_refit=False))


def interval(frame,candidate,reference):
    temp=frame.assign(c=(frame[candidate]-frame.actual)**2*frame.weight,
                      r=(frame[reference]-frame.actual)**2*frame.weight)
    grouped=temp.groupby('month')[['c','r','weight']].sum()
    n=len(grouped);rng=np.random.default_rng(42);diff=[]
    for _ in range(2000):
        starts=rng.integers(0,n-3+1,size=int(np.ceil(n/3)))
        indices=np.concatenate([np.arange(s,s+3) for s in starts])[:n]
        p=grouped.iloc[indices]
        diff.append(np.sqrt(p.c.sum()/p.weight.sum())-np.sqrt(p.r.sum()/p.weight.sum()))
    return np.quantile(diff,[.025,.975]).tolist()


def report():
    verify_sources();rows=[]; contrasts=[];monthly=[]
    for origin in ORIGINS:
        f=aligned(origin)
        for member in MEMBERS:
            score=weighted_metrics(f.actual,f[member],f.weight)
            base=weighted_metrics(f.actual,f.baseline,f.weight)['rmse_sgd']
            rows.append(dict(origin=origin,member=member,features=dict(baseline=77,reliability=82,rent_forecast=84,coe_increment=85)[member],
                             rows=len(f),**score,delta_vs_baseline_sgd=score['rmse_sgd']-base))
            for month,p in f.groupby('month'):
                monthly.append(dict(origin=origin,month=month,member=member,**weighted_metrics(p.actual,p[member],p.weight)))
        for candidate,reference in [('reliability','baseline'),('rent_forecast','reliability'),('coe_increment','rent_forecast'),
                                    ('rent_forecast','baseline'),('coe_increment','baseline')]:
            low,high=interval(f,candidate,reference)
            delta=weighted_metrics(f.actual,f[candidate],f.weight)['rmse_sgd']-weighted_metrics(f.actual,f[reference],f.weight)['rmse_sgd']
            contrasts.append(dict(origin=origin,candidate=candidate,reference=reference,delta_rmse_sgd=delta,ci_low=low,ci_high=high))
    scores=pd.DataFrame(rows);scores.to_csv(OUT/'scores.csv',index=False)
    pd.DataFrame(monthly).to_csv(OUT/'monthly_scores.csv',index=False)
    contrasts=pd.DataFrame(contrasts);contrasts.to_csv(OUT/'adjacent_feature_comparisons.csv',index=False)
    chosen=json.loads((OUT/'selection.json').read_text());longchoice=json.loads((OUT/'long_horizon_selection.json').read_text())
    later=scores[scores.origin.eq('2024-06')]
    longscores=pd.read_csv(OUT/'long_horizon_scores.csv')
    lines=['# TabPFN: reliability and forecast-derived COE input ablation','',
           f'Earlier-only input selection chooses **{chosen["selected"]}**. All added-feature variants are reported below. '
           'This is a matched frozen TabPFN study; the previously measured MSE-only DiagScale control uses a different adapter/view setting. '
           'The already examined later period is development evidence, not a new test.','',
           '## Main individual-rental results','',table(later[['member','features','rows','rmse_sgd','mae_sgd','bias_sgd','delta_vs_baseline_sgd']]),'',
           '## Earlier chronology and selection','',table(scores[scores.origin.ne('2024-06')][['origin','member','rmse_sgd','delta_vs_baseline_sgd']]),'',
           'Pooled earlier weighted RMSE: '+json.dumps(chosen['earlier_rmse'])+'. '
           f'The July2023 nested choice, using only September2022 predictions, is {chosen["nested_July2023_selected"]}. '
           'No later target selected a feature set or a macro architecture/window.','',
           '## Incremental feature effects','',table(contrasts[contrasts.origin.eq('2024-06')]),'',
           'Intervals use paired contiguous three-month blocks and retain population sampling weights. They condition on fitted '
           'models and omit fitting/selection uncertainty; the ten later months are a small dependent sample.','',
           '## Feature definitions and availability','',
           '- reliability82: original77 plus missing signature/block history, block3m count<2, block12m count<5, and unknown BLOCK_ID flags.',
           '- rent_forecast84: reliability82 plus predicted national rent-only log growth and a macro-unavailable indicator.',
           '- coe_increment85: rent_forecast84 plus (rent+COE forecast growth minus matched rent-only forecast growth).',
           '- Both macro LSTMs have12-month windows,8 hidden units,100 fixed epochs,seed42 and ten direct heads. '
           'Heads require3 mature monthly labels; early unsupported forecasts remain NaN. All neural loss targets stop at their source cutoff.',
           '- Macro data stop at each original per-row feature origin minus one month. The extra buffer can make the macro '
           'forecast distance eleven months; the growth feature then stays at head10. Actual calendar/horizon and all original inputs stay unchanged.',
           '- The national index uses the old2021-only property adjustment, and is never exposed as a forecast before2021 ends. '
           'There are only51 monthly rent observations in total. Source snapshots do not reproduce exact historical release vintages.',
           '- All original contexts, native categorical/missing handling, residual targets and additive rental baselines are unchanged. '
           'Two internal estimators,seed42,full context,native batching and batch200 match the audited frozen control. '
           'Control predictions are reused only with hashes, IDs, labels and settings checked.','',
           '## DiagScale transfer confirmation','']
    diag=OUT/'diagscale_confirmation.json'
    if diag.exists():
        d=json.loads(diag.read_text());lines += [table(pd.DataFrame([dict(model='original77',**d['baseline_metrics']),
                                                                   dict(model=d['selected'],**d['candidate_metrics'])])), '',
              'The existing preDecember2023 MSE-only adapter is held fixed with one inference view. It is not retrained '
              'for the appended inputs; this measures feature transfer to that adapter, not an optimized new adapter.','']
    else:
        lines += ['No added input set won earlier selection, so the prespecified DiagScale transfer gate did not trigger. '
                  'Its existing S$455.071 development result remains unchanged.','']
    lines += ['## Separate long-horizon baseline experiment','',
              f'Earlier 11–16-month cell-mean errors select **{longchoice["policy"]}**.','',table(longscores),'',
              'These are planning-area/flat-type cell-mean forecasts, weighted by rental counts, from four historical origins. '
              'Their SGD RMSE is not individual-rental TabPFN RMSE. Forecast cases overlap; rental_case_weights can count '
              'the same observed rental under different origins. Native month10 plateau is compared with continued '
              'damped trend/local-premium decay and persistence of the latest shrunken local level. Only targets beforeJune2024 '
              'select a policy. This does not refit TabPFN residual targets or change the submission baseline.','',
              '## Reproduce and artifacts','',
              '`.venv/Scripts/python.exe analysis/compare_tabpfn_reliability_macro.py --stage all`','',
              'settings.json records source/code hashes and settings; features_*.json and feature_coverage.csv describe matrices; '
              'national_forecasts.csv records every causal macro cutoff/head; causality_audit.json records future-value poisoning; '
              'predictions_*.csv and timing_*.json record fixed-model inference. Stage one resumes completed hash-checked runs. '
              'Existing reports, raw CSVs, adapters and both final submission files remain unchanged.','']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    verify_sources()
    write_json(OUT/'verification.json',dict(verified=True,original77_columns_labels_baselines_preserved=True,
               original_submissions_unchanged=True,query_ids_and_labels_aligned=True,future_macro_poisoning_replayed=True,
               earlier_only_selection=True,all_context_rows_retained=True,test_labels_used=False))
    print('MAIN RESULTS',later[['member','rmse_sgd','delta_vs_baseline_sgd']].to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',default='all',choices=['all','prepare','one','select','long','report'])
    parser.add_argument('--origin',default='2024-06');parser.add_argument('--member',default='reliability')
    args=parser.parse_args()
    torch.set_num_threads(1);torch.backends.mkldnn.enabled=False;torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=1):
        if args.stage in ('all','prepare'):prepare()
        if args.stage=='one':worker(args.origin,args.member)
        if args.stage=='all':
            for origin in ORIGINS[:-1]:
                for member in VARIANTS:child(origin,member)
            selection()
            for member in VARIANTS:child('2024-06',member)
            if json.loads((OUT/'selection.json').read_text())['selected']!='baseline':child('2024-06','diag_confirm')
            long_horizon();report()
        if args.stage=='select':selection()
        if args.stage=='long':long_horizon()
        if args.stage=='report':report()


if __name__=='__main__':main()
