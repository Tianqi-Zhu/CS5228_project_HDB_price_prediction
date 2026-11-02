"""Artifacts, paired clustered uncertainty and report for the frozen latent study."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analysis.compare_tabpfn_hierarchical_latent import OUT,CACHE,ROOT,OUTERS,HISTORICAL,check_sources,load_bundle
from analysis.data import sha256
from analysis.compare_local_economy_landscape import write_json
from analysis.experiments.hierarchical_latent_residual import weighted_metrics,apply_correction

MODELS=['E0','E1','E2','E1_alpha1','E2_alpha1','selected_prediction']


def table(frame,columns=None):
    f=frame if columns is None else frame[columns]
    def fmt(v):
        if isinstance(v,(float,np.floating)): return f'{v:.3f}' if np.isfinite(v) else 'not available'
        return str(v).replace('|',' / ')
    return '| '+' | '.join(map(str,f.columns))+' |\n| '+' | '.join(['---']*len(f.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in f.itertuples(index=False,name=None))


def scores(f,label):
    base=weighted_metrics(f.actual,f.E0,f.weight); result=[]
    for name in MODELS:
        m=weighted_metrics(f.actual,f[name],f.weight)
        result.append(dict(origin=label,model=name,n=len(f),represented_rows=float(f.weight.sum()),**m,
            rmse_improvement_pct=100*(base['rmse_sgd']-m['rmse_sgd'])/base['rmse_sgd'],
            mae_improvement_pct=100*(base['mae_sgd']-m['mae_sgd'])/base['mae_sgd']))
    return result


def clustered_bootstrap(f):
    """Paired cluster bootstrap; aggregate weighted SSE, never average fold RMSE."""
    rows=[]; rng=np.random.default_rng(42)
    for field in ['month','building']:
        groups=f.groupby(field,sort=True).indices
        indices=list(groups.values()); weight=np.array([f.weight.iloc[i].sum() for i in indices])
        sq={name:np.array([np.sum(f.weight.iloc[i].to_numpy()*(f[name].iloc[i]-f.actual.iloc[i]).to_numpy()**2) for i in indices]) for name in ['E0','E1','E2']}
        # Aggregate 5,000 independent draws over the cluster distribution.
        counts=rng.multinomial(len(indices),np.full(len(indices),1/len(indices)),size=5000)
        denom=counts@weight
        rmse={name:np.sqrt((counts@sse)/denom) for name,sse in sq.items()}
        for model,reference in [('E1','E0'),('E2','E0'),('E2','E1')]:
            differences=rmse[model]-rmse[reference]
            actual=weighted_metrics(f.actual,f[model],f.weight)['rmse_sgd']-weighted_metrics(f.actual,f[reference],f.weight)['rmse_sgd']
            rows.append(dict(origin='2024-06',cluster=field,clusters=len(indices),replicates=5000,model=model,reference=reference,
                rmse_difference_sgd=actual,lower95_sgd=float(np.quantile(differences,.025)),upper95_sgd=float(np.quantile(differences,.975)),
                probability_difference_negative=float(np.mean(differences<0))))
    return pd.DataFrame(rows)


def subgroup_scores(f):
    f=f.copy()
    f['residual_support']=pd.cut(f.E1_residual_building_count,[-1,0,2,9,np.inf],labels=['unseen in residual fit','1-2 residuals','3-9 residuals','10+ residuals'])
    f['context_support']=pd.cut(f.context_building_count,[-1,0,5,19,np.inf],labels=['unseen in TabPFN context','1-5 context rows','6-19 context rows','20+ context rows'])
    f['horizon_band']=pd.cut(f.horizon,[0,3,5,6,10],labels=['1-3 months','4-5 months','6 months','7-10 months'])
    rows=[]
    for field in ['town','flat_type','residual_support','context_support','horizon','horizon_band']:
        for group,g in f.groupby(field,observed=True):
            base=weighted_metrics(g.actual,g.E0,g.weight)['rmse_sgd']
            for model in ['E0','E1','E2']:
                m=weighted_metrics(g.actual,g[model],g.weight)
                rows.append(dict(dimension=field,group=str(group),model=model,n=len(g),**m,rmse_difference_sgd=m['rmse_sgd']-base))
    return pd.DataFrame(rows)


def verify_artifacts(predictions,selections):
    check_sources(); assertions=[]
    for origin in OUTERS:
        q=predictions.loc[predictions.origin==origin].reset_index(drop=True)
        pool=pd.read_csv(OUT/'origins'/origin/'fitting_residuals.csv',float_precision='round_trip')
        assert pool.row_id.is_unique and pool.month.max()<origin and set(pool.row_id).isdisjoint(q.row_id)
        assert q.row_id.is_unique and np.isfinite(q[MODELS].to_numpy()).all()
        s=selections[origin]; assert s['outer_targets_used'] is False
        timings=pd.read_csv(OUT/'origins'/origin/'timings.csv')
        for name in ['E1','E2']:
            modelpath=CACHE/f'{origin}_{name}.joblib'
            assert sha256(modelpath)==timings.set_index('model').loc[name,'model_sha256']
            model=joblib.load(modelpath)
            blind=q.drop(columns=['actual','residual'],errors='ignore')
            original=model.predict_components(blind).total_correction.to_numpy()
            poisoned=model.predict_components(blind.assign(actual=1e12,residual=-1e12)).total_correction.to_numpy()
            np.testing.assert_array_equal(original,poisoned)
            np.testing.assert_array_equal(apply_correction(q.E0,original,0),q.E0)
            np.testing.assert_allclose(apply_correction(q.E0,original,s[name]['alpha']),q[name],atol=1e-10,rtol=0)
            assert model.last_fitting_month_<origin
        b=load_bundle(origin)
        assert b['X_query'].shape[1]==77
        np.testing.assert_array_equal(q.row_id,b['X_query'].index)
        np.testing.assert_array_equal(q.actual,b['y_query'])
        np.testing.assert_allclose(q.E0+q.residual,q.actual,atol=1e-10,rtol=0)
        assertions.append(dict(origin=origin,fit_rows=len(pool),evaluation_rows=len(q),fit_targets_strictly_past=True,
            context_evaluation_disjoint=True,row_alignment=True,finite_predictions=True,alpha_zero_exact=True,
            loaded_model_replay=True,query_target_poison_invariant=True,selection_excludes_outer_targets=True))
    write_json(OUT/'correctness_audit.json',dict(assertions=assertions,source_hashes_unchanged=True,
        baseline_and_historical_parameter_hashes_unchanged=True,autograd_calls_prohibited=True,
        geography_availability_caveat='Original static MRT snapshot not point-in-time certified; held fixed, no new external inputs',
        feature_tests='See tests.log for pseudo-origin and fixed-origin future-target mutation tests'))


def figures(results,main,subgroups,persistence):
    directory=OUT/'figures'; directory.mkdir(exist_ok=True)
    plt.rcParams.update({'figure.dpi':130,'font.size':10})
    def save(fig,name):
        fig.tight_layout(); fig.savefig(directory/name); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4)); pivot=results.loc[results.model.isin(['E0','E1','E2']) & results.origin.ne('pooled')].pivot(index='origin',columns='model',values='rmse_sgd')
    pivot[['E0','E1','E2']].plot.bar(ax=ax,rot=0); ax.set_ylabel('Out-of-time RMSE (SGD)'); ax.set_xlabel('Fixed forecast origin'); save(fig,'rmse_comparison.png')
    for field in ['building','submarket']:
        f=pd.read_csv(OUT/f'{field}_persistence.csv')
        fig,ax=plt.subplots(figsize=(5,4)); ax.scatter(f.effect_early,f.effect_late,s=12,alpha=.5)
        p=next(p for p in persistence if p['group']==field)
        ax.axhline(0,color='grey',lw=.5); ax.axvline(0,color='grey',lw=.5)
        ax.set(xlabel='May 2022-Apr 2023 effect (SGD)',ylabel='May 2023-May 2024 effect (SGD)',title=f"{field.title()} persistence: r={p['pearson']:.3f}, n={len(f)}")
        save(fig,f'{field}_persistence.png')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for model in ['E1','E2']:
        correction=(main[model]-main.E0).to_numpy(); axes[0].hist(correction,bins=50,alpha=.5,label=model)
    for column,label in [('E1_building_effect','Building'),('E1_submarket_effect','Submarket'),('E2_dynamic_effect','Dynamic')]:
        axes[1].hist(main[column],bins=50,alpha=.5,label=label)
    axes[0].set(xlabel='Applied correction (SGD)',title='Validation-selected alpha'); axes[1].set(xlabel='Unscaled component (SGD)',title='Estimated components'); [a.legend() for a in axes]
    save(fig,'latent_corrections.png')
    fig,ax=plt.subplots(figsize=(8,4)); h=subgroups.loc[subgroups.dimension=='horizon'].copy(); h['horizon']=h.group.astype(int)
    for name in ['E0','E1','E2']:
        s=h.loc[h.model==name].sort_values('horizon'); ax.plot(s.horizon,s.rmse_sgd,marker='o',label=name)
    ax.set(xlabel='Calendar months ahead (Jun 2024 = 1)',ylabel='RMSE (SGD)',xticks=range(1,11)); ax.legend(); save(fig,'horizon_performance.png')
    path=OUT/'origins/2024-06/dynamic_states_and_forecasts.csv'
    if path.exists():
        states=pd.read_csv(path); groups=pd.read_csv(OUT/'origins/2024-06/submarket_effects.csv').nlargest(4,'residual_count').submarket
        fig,axes=plt.subplots(2,2,figsize=(11,6),sharex=True)
        for group,ax in zip(groups,axes.ravel()):
            f=states.loc[states.submarket==group]; t=pd.to_datetime(f.month)
            past=f.fitted_historical.astype(bool); ax.plot(t.loc[past],f.loc[past,'state'],label='Historical MAP state'); ax.plot(t.loc[~past],f.loc[~past,'state'],'--',label='AR forecast')
            ax.axvline(pd.Timestamp('2024-06-01'),color='grey',lw=.7); ax.set_title(group); ax.set_ylabel('State (SGD)'); ax.tick_params(axis='x',rotation=30)
        axes[0,0].legend(fontsize=8); save(fig,'dynamic_state_forecasts.png')


def summarize():
    predictions=pd.read_csv(OUT/'predictions_all_origins.csv',float_precision='round_trip')
    selections={o:json.loads((OUT/'origins'/o/'selection.json').read_text()) for o in OUTERS}
    verify_artifacts(predictions,selections)
    results=pd.DataFrame([r for origin,f in predictions.groupby('origin') for r in scores(f,origin)]+scores(predictions,'pooled'))
    timings=pd.concat([pd.read_csv(OUT/'origins'/o/'timings.csv') for o in OUTERS],ignore_index=True)
    results=results.merge(timings[['origin','model','fit_seconds','predict_overhead_seconds']],how='left',on=['origin','model'])
    for origin in OUTERS:
        path=OUT/'baseline_replay.json' if origin=='2024-06' else OUT/f'historical_provenance_{origin}.json'
        t=json.loads(path.read_text()); mask=(results.origin==origin)&(results.model=='E0')
        results.loc[mask,'fit_seconds']=t['fit_seconds']; results.loc[mask,'predict_overhead_seconds']=t['predict_seconds']
    results.to_csv(OUT/'comparison.csv',index=False)
    main=predictions.loc[predictions.origin=='2024-06'].reset_index(drop=True)
    ci=clustered_bootstrap(main); ci.to_csv(OUT/'paired_cluster_bootstrap.csv',index=False)
    subgroups=subgroup_scores(main); subgroups.to_csv(OUT/'subgroup_scores.csv',index=False)
    persistence=json.loads((OUT/'persistence_diagnostics.json').read_text())
    figures(results,main,subgroups,persistence)
    settings=json.loads((OUT/'settings.json').read_text()); baseline=json.loads((OUT/'baseline_replay.json').read_text())
    feature_rows=[]
    for i,name in enumerate(settings['feature_names']):
        family='Temporal forecast' if name.startswith('TEMP_') else 'Causal rental history' if name.startswith(('HIST_','RECENT_')) or name in ['AREA_RECENT_MINUS_LONG','BLOCK_RECENT_MINUS_LONG','BLOCK_MINUS_AREA_RECENT','AREA_LAST_MINUS_RECENT'] else 'Static/calendar/identity'
        feature_rows.append(dict(position=i,name=name,family=family,categorical=i in settings['categorical_positions']))
    feature_frame=pd.DataFrame(feature_rows); feature_frame.to_csv(OUT/'feature_catalog.csv',index=False)
    (OUT/'feature_specification.md').write_text('# Frozen feature specification\n\n'+table(feature_frame)+'\n\nOriginal histories: property signature, block x flat type and planning-area x flat type means/counts/latest means/recency; 3/6/12-month recent means/counts/standard deviations; block means shrunk toward area; four historical trends. Nine temporal inputs use the original causally fitted local forecasting baseline and fixed forecast horizon. Context rows use deterministic pseudo-origins 1-10 months back; evaluation rows use only labels before the fixed origin. No future evaluation labels update these inputs.\n\nMissing numeric values retain native NaN; categories and unknown values preserve the cached training categories, including categorical calendar month. No new scaling, imputation, feature selection or target transform is fitted. Fit target is raw SGD minus the original temporal baseline; add the query temporal baseline exactly once. Group keys are auxiliary residual metadata. Static coordinates/building attributes and amenity snapshots are inherited. The previously discovered future-MRT snapshot issue is held fixed: this experiment does not certify historical external-data availability. No new external data, environment, quality or census features are introduced.\n',encoding='utf-8')
    selected=[]
    for origin,s in selections.items():
        for model in ['E1','E2']:
            v=s[model]; selected.append(dict(origin=origin,model=model,alpha=v['alpha'],**v['config'],
                dynamic_enabled=model=='E2' and s['dynamic_gate']['enabled'],inner_rmse_sgd=v.get('cv_rmse_sgd'),
                baseline_inner_rmse_sgd=v.get('baseline_cv_rmse_sgd'),residual_fit_rows=s['residual_fit_rows'],selected_family=s['selected_family']))
    selection_frame=pd.DataFrame(selected); selection_frame.to_csv(OUT/'selected_configurations.csv',index=False)
    skill=selection_frame.copy(); skill['inner_residual_forecast_R2']=1-(skill.inner_rmse_sgd/skill.baseline_inner_rmse_sgd)**2
    skill.to_csv(OUT/'inner_forecast_skill.csv',index=False)
    mainresults=results.loc[results.origin=='2024-06'].set_index('model')
    chosen=selections['2024-06']['selected_family']
    assessed=results.loc[(results.model==chosen)&results.origin.isin(['2023-07','2024-06','pooled'])]
    stable=chosen!='E0' and (assessed.rmse_improvement_pct>0).all()
    recommendation=chosen if stable else 'E0'
    decision=dict(pre_outer_selected_family=chosen,recommended_model=recommendation,
        rule='Retain earlier-selected correction only if RMSE improves in both non-cold origins and pooled evaluation; otherwise retain E0',
        correction_consistently_improves=bool(stable),no_kaggle_submission_changed=True)
    write_json(OUT/'decision.json',decision)
    horizon=subgroups.loc[(subgroups.dimension=='horizon_band')&subgroups.model.isin(['E0','E1','E2'])]
    support=subgroups.loc[(subgroups.dimension=='residual_support')&subgroups.model.isin(['E0','E1','E2'])]
    flat=subgroups.loc[(subgroups.dimension=='flat_type')&subgroups.model.isin(['E0','E1','E2'])]
    pool_count=pd.read_csv(OUT/'origins/2024-06/fitting_residuals.csv').shape[0]
    gate=json.loads((OUT/'diagnostics/2024-06/temporal_evidence.json').read_text())
    total_tabpfn=sum(json.loads((OUT/f'historical_provenance_{o}.json').read_text())[k] for o in HISTORICAL for k in ['fit_seconds','predict_seconds'])+baseline['fit_seconds']+baseline['predict_seconds']
    forecast_text='E2 improves historical reconstruction but not necessarily forecasting; reconstruction scores below are descriptive.'
    for origin in OUTERS:
        t=timings.loc[timings.origin==origin].set_index('model'); rr=results.loc[results.origin==origin].set_index('model')
        if t.loc['E2','training_reconstruction_rmse_sgd']<t.loc['E1','training_reconstruction_rmse_sgd'] and rr.loc['E2','rmse_sgd']>rr.loc['E1','rmse_sgd']:
            forecast_text+=f" At {origin}, E2 reconstructs training residuals better ({t.loc['E2','training_reconstruction_rmse_sgd']:.3f} vs {t.loc['E1','training_reconstruction_rmse_sgd']:.3f}) but forecasts worse ({rr.loc['E2','rmse_sgd']:.3f} vs {rr.loc['E1','rmse_sgd']:.3f})."
    town_dif=subgroups.loc[(subgroups.dimension=='town')&(subgroups.model=='E1')]
    town_gains=int((town_dif.rmse_difference_sgd<0).sum())
    report=f'''# Hierarchical latent residuals on frozen TabPFN

Recommended model: **{recommendation}**. The family chosen before reading the main outer scores was **{chosen}**. The decision rule requires improvement at both non-cold outer origins and on pooled predictions; a tiny improvement in this repeatedly used development holdout is not evidence of a general improvement.

## Main 80/20 comparison

All 28,995 evaluation transactions, June 2024-March 2025; context has 121,005 transactions through May 2024. Prices and errors are SGD. Alpha is selected on earlier folds; alpha=1 is separately reported.

{table(results.loc[results.origin=='2024-06'],['model','rmse_sgd','mae_sgd','bias_sgd','rmse_improvement_pct','mae_improvement_pct','fit_seconds','predict_overhead_seconds'])}

E1 adds global/building/town x flat-type effects; E2 adds regularized monthly submarket AR(1) states. `selected_prediction` is the family selected on earlier validation, not a new ensemble. Latent inference times are additional CPU costs over E0; E0's fit is context preparation, without optimization. Alpha=0 reproduces E0 bit for bit. No TabPFN parameters changed, no gradients or adapters were used, and no final Kaggle output was overwritten.

## Frozen baseline and provenance

The retained original **77 features** contain 31 static/calendar/identity variables, 37 causal rental-history variables and 9 temporal variables. The exact names, categories and temporal provenance are in [feature_specification.md](feature_specification.md) and [feature_catalog.csv](feature_catalog.csv). Historical features already encode signature, block/type and area/type histories with 3/6/12-month counts, shrinkage, dispersion and trend. Extra economy/landscape/amenity-quality features and deduplicated data did not establish a stronger retained frozen baseline. They are excluded. Adapted MSE-only DiagScale's S$455.071 is outside this experiment's frozen-weight constraints.

TabPFN {settings['packages']['tabpfn']}, V3.5 FAST checkpoint `tabpfn-v3.5-fast-20260909.safetensors`, SHA256 `{settings['source_sha256']['.cache/tabpfn/tabpfn-v3.5-fast-20260909.safetensors']}`. Use the plain TabPFNRegressor class, 2 preprocessing estimators, seed42, full expanding historical context (no cap or subsampling), batch200, fit_with_cache, GPU cache, one preprocessing job, RTX4080. Settings and feature/checkpoint/source hashes are in [settings.json](settings.json).

Earlier December2023-May2024 calibration (600 fixed sampled rows) selected 2 estimators: RMSE520.473 versus521.864 for1. One-view replay matched its saved reference within {baseline['one_view_reference_replay_error']:.2g} SGD; two independent bare-model 400-row runs agreed exactly. The previous two-view 'frozen' artifact used a zero-initialized adapter wrapper and failed strict plain-model equivalence (maximum full-output difference {baseline['cached_wrapper_control_max_difference_sgd']:.6f} SGD). It was investigated and not reused. The complete plain-model E0 was recomputed: **RMSE{baseline['native_complete_metrics']['rmse_sgd']:.6f}, MAE{baseline['native_complete_metrics']['mae_sgd']:.6f}**, versus455.395927/327.793699 in that prior control. Small aggregate agreement does not establish row-wise identity; all E0/E1/E2 comparisons here share the new bare-model predictions.

## Chronological protocol

Earlier established fixed-origin backtests at September2022 and July2023 retain their original label-independent200 queries/month and month-population sampling weights; the main June2024 window evaluates every query. Each window spans10 target months. Earlier-window metrics estimate their represented population, rather than scoring every transaction. Their target-month windows do not overlap one another or the main window. Pooled RMSE is sqrt(sum(weight * squared error)/sum(weight)), never a mean of fold RMSEs. The main holdout and feature family were examined during prior development: **these are development backtests, not an untouched final test**.

Seven cached feature bundles were validated and used for fresh, compatible bare-two-view historical predictions. All context rows are retained (49,389 to109,008 historical context rows); the outer main context is121,005. Small early contexts may yield a different residual distribution, so results need cautious interpretation. There are12,800 historical forecast cases before eligibility filtering; overlapping forecasts of a physical source row are reduced to its latest available forecast at each fit origin. Both its label month and its forecast origin must precede the fit origin. Weights are normalized to the number of distinct residual cases, avoiding artificial sample-size inflation. The main latent fit uses {pool_count:,} distinct historical residual cases. This de-duplication concerns cached forecasts, not removing real transactions from the original dataset.

Each inner forecast fits latent effects only to residual labels available before its own origin; its validation labels must precede the corresponding outer origin. Small grids: block/group penalties25 or100, alpha0/.25/.5/.75/1, then shared rho0/.5/.9 and state penalty25/100. E2 uses the earlier-selected static penalties. E0/E1/E2 family ties within0.1% prefer E0 thenE1 thenE2. The earliest September2022 origin has no complete earlier inner validation: alpha0 is selected prospectively, so equality there is a cold-start fallback, not evidence for the corrections. Selection metadata is written before outer metrics and includes label cutoffs and ID hashes.

{table(selection_frame)}

## All fixed-origin results

{table(results.loc[results.model.isin(['E0','E1','E2'])],['origin','model','n','rmse_sgd','mae_sgd','bias_sgd','rmse_improvement_pct'])}

Unattenuated alpha=1 scores for all windows are included in [comparison.csv](comparison.csv). [inner_forecast_skill.csv](inner_forecast_skill.csv) reports actual inner forecast skill against zero correction, rather than in-sample residual fit.

## Residual evidence and estimation

Buildings are normalized town|street|block; block number alone is not used. No individual rental-unit identifiers are available. Submarkets are town x flat type. Monthly historical residual summaries, building/submarket counts and variances, and horizon diagnostics are in `diagnostics/`. Non-overlapping target periods May2022-April2023 and May2023-May2024 assess persistence. Building estimates use separate shrunken fits; submarket residual means have fitted building effects and the global intercept removed. These are descriptive checks, not validation scores.

{table(pd.DataFrame(persistence),['group','common_groups','minimum_cases_per_period','pearson','spearman'])}

The main historical pool contains {gate['months']} monthly means with lag1 correlation {gate['lag1_monthly_mean_acf']:.3f}. This justifies testing dynamics, not assuming forecast improvement. A per-origin gate requires at least8 observed months and absolute monthly lag1 correlation>=0.1; earlier origins cannot borrow later evidence. If the gate is disabled E2 is explicitly the E1 fallback. Fits minimize weighted squared residuals plus positive Gaussian building/group penalties; the global intercept is unpenalized. Rare groups shrink more, unseen buildings get zero building effect, and unseen submarkets fall back to the global effect. Shared AR prior is lambda_d*((1-rho^2)*d_first^2 + sum(d_t-rho*d_previous)^2), including missing calendar months. Only historical states through origin-1 are fitted; future values are rho^h times the final historical state. Even rho0 can change the joint historical static fit, while its future dynamic contribution is zero.

Sparse augmented least squares uses [SciPy LSMR](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsmr.html), tolerances1e-10 and checked convergence. No dense observation-by-parameter matrix or covariance is built. These are Gaussian MAP point estimates, not full posterior uncertainty estimates. {forecast_text}

{table(timings,['origin','model','fit_seconds','predict_overhead_seconds','solver_iterations','system_columns','system_nnz','training_reconstruction_rmse_sgd'])}

Fresh primary E0 plus seven historical runs took {total_tabpfn:.1f}s of recorded fit/inference time, excluding calibration probes, hashing and feature-cache loading. Small-grid tuning times are recorded per selection JSON; inference overhead excludes file serialization. Saved models live in `.cache/tabpfn_hierarchical_latent_80_20`.

## Horizons and subgroups

Horizon1 means the first target month after the last available label month; June2024=1 and March2025=10. The 6-month row is **exactly h=6**, not horizons1-6. Existing forecasts stop at10, so **12-month RMSE is unavailable**. No performance claim is made for the longer unlabeled Kaggle forecast.

{table(horizon,['group','model','n','rmse_sgd','rmse_difference_sgd'])}

Residual support refers to the number of sampled chronological residual observations for a building, which differs from full TabPFN training-context support. A building unseen in the residual pool can already have many rental records in TabPFN's context. Full-context rare/unseen groups, all towns and all flat types are in [subgroup_scores.csv](subgroup_scores.csv). E1 improves {town_gains}/{len(town_dif)} town RMSEs; subgroup comparisons are descriptive and were not used for tuning.

{table(support,['group','model','n','rmse_sgd','rmse_difference_sgd'])}

{table(flat,['group','model','n','rmse_sgd','rmse_difference_sgd'])}

## Paired uncertainty

Main-window paired cluster bootstrap,5,000 replicates,seed42. Values are model RMSE minus reference RMSE (negative favors model). Months and buildings are resampled separately; neither calculation assumes independent transactions. Only10 time clusters limit temporal inference; building resampling does not capture all shared market shocks. Intervals are descriptive after repeated model development, not confirmatory confidence in a new Kaggle score.

{table(ci,['cluster','clusters','model','reference','rmse_difference_sgd','lower95_sgd','upper95_sgd'])}

## Correctness and limitations

[correctness_audit.json](correctness_audit.json) verifies source hashes, model serialization/replay, exact alpha0, row alignment, finite SGD predictions, disjoint residual-fit/evaluation IDs and strict fitting-label cutoffs. Poisoning evaluation targets leaves latent predictions unchanged. All eight primary/history inference runs have matching before/after full parameter digests; backward/autograd calls were prohibited. Existing feature builders and pseudo-origins are preserved; target-causality tests are recorded in `tests.log`. Baseline and historical residual target values are converted back to SGD once before correction and metrics.

The original static MRT snapshot includes sites whose availability does not match every historical date. It is inherited and held identical across E0/E1/E2; this is **not a certification of fully point-in-time external geography**. No new external data is introduced. Historical residual queries cover only200 transactions/month, limiting building-effect coverage. Different expanding context sizes and overlapping inner forecast cases can reduce effective information. Inner validation is chronological but some historical targets occur in more than one forecast task; final latent fitting counts each physical source row once. Three outer origins, a cold-start earliest window and a repeatedly used main holdout limit certainty. No labeled Kaggle test is available. No new submission is generated by this evaluation.

## Diagnostic figures

![RMSE comparison](figures/rmse_comparison.png)
![Building persistence](figures/building_persistence.png)
![Submarket persistence](figures/submarket_persistence.png)
![Corrections](figures/latent_corrections.png)
![Horizon performance](figures/horizon_performance.png)
![Dynamic state forecasts](figures/dynamic_state_forecasts.png)

## Reproduce and artifacts

From the repository root: `.venv/Scripts/python.exe analysis/compare_tabpfn_hierarchical_latent.py --stage prepare`, then `--stage historical`, then `--stage latent`. Cached predictions avoid repeated GPU inference. `settings.json` and `latent_plan.json` lock source/code hashes; changed code fails provenance checks. Predictions with transaction row IDs, origins, actuals, all correction components and selected/unattenuated outputs are in `predictions_all_origins.csv` and `origins/*/predictions.csv`. Source row_id is the original training CSV index. Selection grids, fit pools, timings and serialized models are retained. Existing baseline experiments, pretrained weights and Kaggle submissions are unchanged.
'''
    (OUT/'report.md').write_text(report,encoding='utf-8')
    print(results[['origin','model','rmse_sgd','mae_sgd']].to_string(index=False),flush=True)
    print('DECISION',decision,flush=True)


if __name__=='__main__': summarize()
