"""Chronological evaluation, complementarity, uncertainty and cost reporting."""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analysis.compare_tabpfn_feature_ensemble import OUT,CACHE,ROOT,PRIOR,ORIGINS,MEMBERS,aligned,check_sources,load_prepared,load_bundle
from analysis.compare_local_economy_landscape import immutable,write_json
from analysis.data import sha256
from analysis.experiments.feature_diversified_ensemble import combine_predictions
from analysis.experiments.hierarchical_latent_residual import weighted_metrics
from analysis.summarize_tabpfn_hierarchical_latent import table

MODEL_NAMES=['baseline','spatial','temporal','equal','optimized','opt_baseline_spatial','opt_baseline_temporal','opt_spatial_temporal','standard6']


def lock_plan():
    immutable(OUT/'reporting_plan.json',dict(code_sha256=sha256(Path(__file__)),bootstrap_replicates=3000,bootstrap_seed=42,
        clusters=['month','building'],primary_model='optimized',primary_reference='baseline',compute_reference='standard6',
        advancement_rule='Candidate for independent confirmation only if optimized beats baseline by>=0.1%, improves July2023 nested-transfer RMSE, month and building paired95% upper limits<0, and does not lose to standard6. No automatic deployment or Kaggle submission.',
        untouched_labeled_test_available=False,no_outer_weight_optimization=True))


def materialize(origin,selection,nested):
    f=aligned(origin); p=f[MEMBERS].to_numpy()
    schemes=selection['models'] if origin=='2024-06' else nested[origin]['models']
    f['equal']=combine_predictions(p,np.full(3,1/3))
    weights=[]
    for name,s in schemes.items():
        f[name]=combine_predictions(p,s['weights'])
        weights.append(dict(origin=origin,model=name,baseline_weight=s['weights'][0],spatial_weight=s['weights'][1],temporal_weight=s['weights'][2],
            active_members=int(np.count_nonzero(s['weights'])),fit_label_max=selection['maximum_selection_label_month'] if origin=='2024-06' else nested[origin].get('maximum_selection_label_month','none; cold-start baseline'),
            cold_start=origin==ORIGINS[0]))
    control_path=OUT/f'predictions_{origin}_standard6.csv'
    meta=json.loads((OUT/f'timing_{origin}_standard6.json').read_text())
    assert sha256(control_path)==meta['prediction_sha256'] and meta['n_estimators']==6 and meta['parameters_unchanged'] and meta['no_adapters']
    control=pd.read_csv(control_path,float_precision='round_trip')
    for c in ['row_id','origin','month','actual','weight']: np.testing.assert_array_equal(f[c],control[c])
    f['standard6']=control.predicted
    assert np.isfinite(f[MODEL_NAMES].to_numpy()).all()
    return f,weights


def cluster_uncertainty(f):
    pairs=[('equal','baseline'),('optimized','baseline'),('optimized','equal'),('standard6','baseline'),('optimized','standard6'),
        ('opt_baseline_spatial','baseline'),('opt_baseline_temporal','baseline')]
    rows=[]; rng=np.random.default_rng(42); models=sorted({m for pair in pairs for m in pair})
    for column in ['month','building']:
        groups=list(f.groupby(column,sort=True).indices.values()); n=len(groups)
        denominator=np.array([f.weight.iloc[i].sum() for i in groups])
        sums=np.column_stack([np.array([np.sum(f.weight.iloc[i].to_numpy()*(f[m].iloc[i]-f.actual.iloc[i]).to_numpy()**2) for i in groups]) for m in models])
        counts=rng.multinomial(n,np.full(n,1/n),size=3000)
        errors=np.sqrt((counts@sums)/(counts@denominator)[:,None])
        for model,reference in pairs:
            d=errors[:,models.index(model)]-errors[:,models.index(reference)]
            point=weighted_metrics(f.actual,f[model],f.weight)['rmse_sgd']-weighted_metrics(f.actual,f[reference],f.weight)['rmse_sgd']
            rows.append(dict(cluster=column,clusters=n,model=model,reference=reference,replicates=3000,rmse_difference_sgd=point,
                lower95_sgd=float(np.quantile(d,.025)),upper95_sgd=float(np.quantile(d,.975)),probability_negative=float(np.mean(d<0))))
    return pd.DataFrame(rows)


def complementarity(f,origin):
    errors=f[MEMBERS].to_numpy()-f.actual.to_numpy()[:,None]
    w=f.weight.to_numpy(); w=w/w.sum(); means=np.sum(w[:,None]*errors,axis=0)
    centered=errors-means; covariance=centered.T@(w[:,None]*centered)
    denom=np.sqrt(np.outer(np.diag(covariance),np.diag(covariance))); corr=covariance/denom
    second=errors.T@(w[:,None]*errors)
    pd.DataFrame(corr,index=MEMBERS,columns=MEMBERS).to_csv(OUT/f'error_correlations_{origin}.csv')
    pd.DataFrame(second,index=MEMBERS,columns=MEMBERS).to_csv(OUT/f'error_second_moments_{origin}.csv')
    rows=[]
    for i in range(3):
        for j in range(i+1,3):
            rows.append(dict(origin=origin,first=MEMBERS[i],second=MEMBERS[j],error_correlation=float(corr[i,j]),
                rms_prediction_difference_sgd=float(np.sqrt(np.sum(w*(f[MEMBERS[i]]-f[MEMBERS[j]])**2))),
                first_bias_sgd=float(means[i]),second_bias_sgd=float(means[j]),
                pair_equal_rmse_sgd=float(np.sqrt(np.sum(w*((errors[:,i]+errors[:,j])/2)**2)))))
    equal=np.full(3,1/3)
    actual_mse=weighted_metrics(f.actual,f.equal,f.weight)['rmse_sgd']**2
    np.testing.assert_allclose(equal@second@equal,actual_mse,rtol=1e-12,atol=1e-7)
    return rows,dict(origin=origin,mean_individual_mse_sgd2=float(np.mean(np.diag(second))),equal_mse_sgd2=float(actual_mse),
        equal_diversity_gain_sgd2=float(np.mean(np.diag(second))-actual_mse),
        interpretation='Diversity gain is against average member MSE, not proof of beating the best individual model')


def cost_table(origin,schemes):
    base=[]
    for m in MEMBERS+['standard6']:
        t=json.loads((OUT/f'timing_{origin}_{m}.json').read_text())
        base.append(dict(origin=origin,model=m,fit_seconds=t['fit_seconds'],predict_seconds=t['predict_seconds'],
            peak_allocated_vram_gib=t['peak_vram_gib'],internal_estimators=t['n_estimators'],external_members=t.get('external_models',1)))
    byname={r['model']:r for r in base}
    for name,w in [('equal',np.full(3,1/3))]+[(n,s['weights']) for n,s in schemes.items()]:
        active=[byname[m] for m,value in zip(MEMBERS,w) if value>0]
        base.append(dict(origin=origin,model=name,fit_seconds=sum(t['fit_seconds'] for t in active),predict_seconds=sum(t['predict_seconds'] for t in active),
            peak_allocated_vram_gib=max(t['peak_allocated_vram_gib'] for t in active),internal_estimators=2*len(active),external_members=len(active)))
    return base


def grouped_results(f):
    f=f.copy(); f['horizon_band']=pd.cut(f.horizon,[0,3,5,6,10],labels=['1-3 months','4-5 months','6 months','7-10 months'])
    rows=[]
    for dimension in ['month','horizon','horizon_band','town','flat_type']:
        for label,g in f.groupby(dimension,observed=True):
            base=weighted_metrics(g.actual,g.baseline,g.weight)['rmse_sgd']
            for model in ['baseline','spatial','temporal','equal','optimized','standard6']:
                result=weighted_metrics(g.actual,g[model],g.weight)
                rows.append(dict(dimension=dimension,group=str(label),model=model,n=len(g),**result,rmse_difference_sgd=result['rmse_sgd']-base))
    return pd.DataFrame(rows)


def figures(results,main,correlations,weights,groups,cost):
    directory=OUT/'figures'; directory.mkdir(exist_ok=True); plt.rcParams.update({'figure.dpi':140,'font.size':10})
    def save(fig,name): fig.tight_layout(); fig.savefig(directory/name); plt.close(fig)
    chosen=['baseline','spatial','temporal','equal','optimized','standard6']
    r=results.loc[results.origin=='2024-06'].set_index('model').loc[chosen]
    fig,ax=plt.subplots(figsize=(9,4)); ax.bar(r.index,r.rmse_sgd,color=['#426580','#658c7d','#c68e46','#9690bb','#4b8eac','#969696']); ax.set(ylabel='RMSE (SGD)',title='Full 80/20 development evaluation'); ax.tick_params(axis='x',rotation=20); save(fig,'rmse_comparison.png')
    fig,ax=plt.subplots(figsize=(5,4)); im=ax.imshow(correlations,vmin=.9,vmax=1,cmap='Blues'); ax.set_xticks(range(3),MEMBERS); ax.set_yticks(range(3),MEMBERS)
    for i in range(3):
        for j in range(3): ax.text(j,i,f'{correlations[i,j]:.4f}',ha='center',va='center',color='black')
    fig.colorbar(im,ax=ax,label='Weighted prediction-error correlation'); save(fig,'error_correlations.png')
    fig,ax=plt.subplots(figsize=(8,4)); w=weights.loc[weights.model=='optimized'].set_index('origin')[['baseline_weight','spatial_weight','temporal_weight']]
    w.columns=MEMBERS; w.plot.bar(stacked=True,ax=ax,rot=0); ax.set(ylabel='Earlier-validation weight',xlabel='Forecast origin',ylim=(0,1)); save(fig,'learned_weights.png')
    fig,ax=plt.subplots(figsize=(8,4)); h=groups.loc[groups.dimension=='horizon'].copy(); h['h']=h.group.astype(int)
    for name in chosen:
        z=h.loc[h.model==name].sort_values('h'); ax.plot(z.h,z.rmse_sgd,marker='.',label=name)
    ax.set(xlabel='Calendar months ahead (Jun2024=1)',ylabel='RMSE (SGD)',xticks=range(1,11)); ax.legend(fontsize=8); save(fig,'horizon_performance.png')
    fig,ax=plt.subplots(figsize=(7,4)); c=cost.loc[cost.origin=='2024-06'].set_index('model')
    offsets={'equal':(-10,18),'optimized':(-10,-16),'standard6':(-10,0)}
    for name in chosen:
        ax.scatter(c.loc[name,'predict_seconds'],r.loc[name,'rmse_sgd'],s=45)
        ax.annotate(name,(c.loc[name,'predict_seconds'],r.loc[name,'rmse_sgd']),xytext=offsets.get(name,(6,4)),
            textcoords='offset points',fontsize=8,ha='right' if name in offsets else 'left')
    ax.set(xlabel='Sequential prediction seconds (active members)',ylabel='RMSE (SGD)')
    ax.set_xlim(left=0,right=float(c.loc[chosen,'predict_seconds'].max())*1.15)
    ax.margins(y=.12); save(fig,'cost_vs_rmse.png')


def summarize():
    check_sources(); plan=json.loads((OUT/'reporting_plan.json').read_text()); assert sha256(Path(__file__))==plan['code_sha256']
    selection=json.loads((OUT/'selection.json').read_text()); nested=json.loads((OUT/'nested_selection.json').read_text())
    for filename,h in selection['source_prediction_sha256'].items(): assert sha256(OUT/filename)==h
    raw=pd.read_csv(ROOT/'data/train.csv'); prepared=load_bundle('2024-06')
    from analysis.compare_tabpfn_hierarchical_latent import raw_frame
    metadata=raw_frame()
    predictions=[]; weights=[]; scores=[]; costs=[]; pairs=[]; diversity=[]
    for origin in ORIGINS:
        f,w=materialize(origin,selection,nested); weights.extend(w)
        b=load_bundle(origin); np.testing.assert_array_equal(f.row_id,b['X_query'].index); np.testing.assert_array_equal(f.actual,b['y_query'])
        assert metadata.loc[b['X_train'].index,'month'].max()<origin<=f.month.min() and set(b['X_train'].index).isdisjoint(f.row_id)
        f['horizon']=pd.PeriodIndex(f.month,freq='M').asi8-pd.Period(origin,'M').ordinal+1
        f['building']=metadata.loc[f.row_id,'building'].to_numpy(); f['town']=metadata.loc[f.row_id,'TOWN_CLEAN'].to_numpy(); f['flat_type']=metadata.loc[f.row_id,'FLAT_TYPE_CLEAN'].to_numpy()
        for member in MEMBERS[1:]:
            aug=load_prepared(origin,member)
            np.testing.assert_array_equal(aug['y_train'],b['y_train']); np.testing.assert_array_equal(aug['y_query'],b['y_query'])
            for key in ['train_baseline','query_baseline']: np.testing.assert_array_equal(aug[key],b[key])
        schemes=selection['models'] if origin=='2024-06' else nested[origin]['models']
        costs.extend(cost_table(origin,schemes))
        for model in MODEL_NAMES:
            m=weighted_metrics(f.actual,f[model],f.weight); base=weighted_metrics(f.actual,f.baseline,f.weight)
            scores.append(dict(origin=origin,model=model,n=len(f),**m,rmse_difference_sgd=m['rmse_sgd']-base['rmse_sgd'],
                rmse_improvement_pct=100*(base['rmse_sgd']-m['rmse_sgd'])/base['rmse_sgd'],mae_improvement_pct=100*(base['mae_sgd']-m['mae_sgd'])/base['mae_sgd']))
        pairs_for_origin,div=complementarity(f,origin); pairs.extend(pairs_for_origin); diversity.append(div); predictions.append(f)
    allp=pd.concat(predictions,ignore_index=True); allp.to_csv(OUT/'predictions_all_origins.csv',index=False,float_format='%.17g')
    scores=pd.DataFrame(scores); cost=pd.DataFrame(costs); scores=scores.merge(cost,on=['origin','model'],how='left'); scores.to_csv(OUT/'comparison.csv',index=False)
    weights=pd.DataFrame(weights); weights.to_csv(OUT/'learned_weights.csv',index=False)
    pairframe=pd.DataFrame(pairs); pairframe.to_csv(OUT/'complementarity.csv',index=False); write_json(OUT/'diversity_identity.json',diversity)
    main=predictions[-1]; ci=cluster_uncertainty(main); ci.to_csv(OUT/'paired_cluster_bootstrap.csv',index=False)
    groups=grouped_results(main); groups.to_csv(OUT/'subgroup_scores.csv',index=False)
    cost.to_csv(OUT/'costs.csv',index=False)
    correlation=pd.read_csv(OUT/'error_correlations_2024-06.csv',index_col=0).to_numpy(); figures(scores,main,correlation,weights,groups,cost)
    r=scores.loc[scores.origin=='2024-06'].set_index('model'); prior=scores.loc[scores.origin=='2023-07'].set_index('model')
    ci_primary=ci.loc[(ci.model=='optimized')&(ci.reference=='baseline')]
    qualifies=bool(r.loc['optimized','rmse_improvement_pct']>=.1 and prior.loc['optimized','rmse_improvement_pct']>0 and (ci_primary.upper95_sgd<0).all() and r.loc['optimized','rmse_sgd']<=r.loc['standard6','rmse_sgd'])
    decision=dict(candidate_for_independent_confirmation=qualifies,production_submission_unchanged=True,untouched_test_available=False,
        best_frozen_reference_rmse_sgd=float(r.loc['baseline','rmse_sgd']),optimized_rmse_sgd=float(r.loc['optimized','rmse_sgd']),standard6_rmse_sgd=float(r.loc['standard6','rmse_sgd']),
        deployment_recommendation='Require independent chronological confirmation before deployment' if qualifies else 'Retain existing model; specified feature-diversified ensemble lacks sufficient stable evidence')
    write_json(OUT/'decision.json',decision)
    tests=(OUT/'tests.log').read_text(); assert 'Ran 24 tests' in tests and tests.strip().endswith('OK')
    runmeta=[json.loads((OUT/f'timing_{o}_{m}.json').read_text()) for o in ORIGINS for m in MEMBERS+['standard6']]
    assert all(t['parameters_unchanged'] and t['gradients_prohibited'] and t['no_adapters'] for t in runmeta)
    check_sources()
    write_json(OUT/'verification.json',dict(source_checkpoint_and_submission_hashes_unchanged=True,baseline_reused_from_verified_plain_model=True,
        tests_passed=24,contexts_exclude_future_labels=True,all_member_rows_labels_baselines_identical=True,all_predictions_finite=True,
        all_model_parameters_unchanged=True,autograd_prohibited=True,all_context_rows_retained=True,weights_nonnegative_sum_to_one=True,
        outer_weight_optimization=False,selection_label_max=selection['maximum_selection_label_month'],no_untouched_final_test_claim=True,
        prediction_sha256=sha256(OUT/'predictions_all_origins.csv')))
    nominal=cost.loc[(cost.origin=='2024-06')&(cost.model=='equal')].iloc[0]
    active=selection['models']['optimized']['weights']; maxcorr=float(correlation[np.triu_indices(3,1)].max()); mincorr=float(correlation[np.triu_indices(3,1)].min())
    gain=r.loc['baseline','rmse_sgd']-r.loc['optimized','rmse_sgd']; equal_gain=r.loc['equal','rmse_sgd']-r.loc['optimized','rmse_sgd']
    weights_text=', '.join(f'{m}={w:.4f}' for m,w in zip(MEMBERS,active))
    if np.count_nonzero(active)==1: membership='The optimizer selected a single member. This is feature selection, not evidence that combining errors helps.'
    else: membership=f'The optimized ensemble retains {np.count_nonzero(active)} members; zero-weight members can be omitted at deployment.'
    answer='The optimized feature-diversified ensemble improves main development RMSE.' if gain>0 else 'The optimized feature-diversified ensemble does not improve main development RMSE.'
    report=f'''# Spatial-temporal feature-diversified frozen TabPFN

{answer} Optimized minus baseline RMSE is {-gain:+.3f} SGD ({r.loc['optimized','rmse_improvement_pct']:+.3f}% improvement). Optimized averaging changes RMSE by {-equal_gain:+.3f} SGD relative to equal averaging. **{decision['deployment_recommendation']}.** This is a repeatedly examined development backtest; no untouched labeled chronological test is available.

## Main 80/20 results

All121,005 pre-June2024 context rows and28,995 June2024-March2025 evaluation rows. RMSE,MAE,bias are original SGD; negative RMSE difference is better. Costs sum sequential inference for active members and exclude feature preparation, hashing and serialization. Internal-estimator count is total active TabPFN views.

{table(scores.loc[scores.origin=='2024-06'],['model','rmse_sgd','mae_sgd','bias_sgd','rmse_difference_sgd','rmse_improvement_pct','predict_seconds','peak_allocated_vram_gib','internal_estimators'])}

Main optimized weights: **{weights_text}**. {membership} No intercept, target calibration or post-hoc clipping is used. Only earlier out-of-time validation targets determine weights; there is no optimization on the main holdout or Kaggle test. The four optimized rows include the prespecified pair ablations, rather than outer-selected alternatives.

## Representation choices

All members use the same pretrained V3.5FAST checkpoint, seed42, plain TabPFNRegressor, full expanding context, no subsampling, fit_with_cache, GPU caches, one preprocessing job, batch200 and two internal preprocessing estimators. All retain the same original raw labels and residual target: y minus the local temporal baseline, adding that baseline once after prediction. No DiagScale,LoRA,adapter training or neural optimization occurs. TabPFN's internal estimator ensemble differs from this external ensemble across feature matrices.

Baseline: retained **77** categorical-month inputs with31 static/calendar/identity,37 causal histories and9 temporal inputs. Every final baseline,feature and seed model uses the plain two-view class with **native-default estimator batching**, without environment overrides on row/cell budgets. The main E0 was freshly computed and all earlier E0 predictions regenerated with that same execution policy. No trained-adapter predictions or parameters are consumed. Existing adapted DiagScale455.071 is outside the frozen-weight comparison.

A final strict replay exposed a hidden execution mismatch in the initially reused baseline: the prior hierarchical experiment set TABPFN_MAX_BATCHED_ESTIMATOR_ROWS=0, running estimators separately, whereas new feature/seed models used native-default batching. This changes individual prices by up to0.554555 SGD even though aggregate baseline RMSE changes only from455.396319 to455.395927. A controlled400-row plain-model probe with the rows0 override reproduces the old sequential output, isolating the cause; see estimator_batching_isolation.json. The current full plain baseline also agrees with the earlier two-view frozen-column reference within5e-9 SGD; that is verification only, not reuse of an adapter model. Provisional mixed-execution outputs and decisions are archived in batching_reconciliation/. Final weights are recomputed by the unchanged earlier-only optimizer from the corrected earlier baseline predictions. Feature definitions,folds,hyperparameters and decision thresholds stay fixed; no outer targets enter this correction.

Spatial: **89** inputs, adding the existing market_smoothed12 columns: same-flat-type500/1000m6/12-month shrunken rent means and counts, plus Gaussian neighborhood rent/support/rent-per-sqm and a missing-history flag. Existing screening favored these over raw neighborhood means (608.353 vs624.615 pooled earlier RMSE in the one-view60-per-month study); they later worsened standalone main RMSE. That negative result does not rule out complementary prediction errors. Here they are evaluated with matched two-view inference. No extra MRT,mall,school,park,census or amenity-quality features are appended. The fields include own-building **earlier** labels; they never include the current observation or future labels. Coordinates are inherited existing geography, not a new dataset.

Temporal: **84** inputs, keeping all rental histories, nine trend/forecast variables and the residual baseline; replaces the categorical month with sine/cosine and adds six existing recurring January school-opening and January/August university-intake/lead-window proxies. Prior controlled earlier folds preferred this calendar representation (599.762 vs602.587 categorical-month one-view RMSE), while broader20-feature dynamic replacements worsened main performance and created reference warmup gaps. Thus this focused seasonal representation tests diversity without changing the target baseline or rebuilding an unpromising complex dynamics model. Calendar proxies are hypotheses, not precise historical school calendars; they are deterministic and complete in every context month. This result does not test every possible market-dynamics representation.

These three definitions were locked before new ensemble scoring. Complete per-fold feature names, new/removed columns, categories and row-ID hashes are in features_*.json; new-feature coverage is in new_feature_coverage.csv. Spatial cached training features are reused only after exact original context-frame agreement. New earlier200-per-month spatial queries reproduce all corresponding prior600-query cached values within1e-8. Source and code hashes are fixed in settings.json.

## Causality and validation

Earlier fixed-origin windows: September2022-June2023 and July2023-April2024; each uses the original200 label-independent queries/month and month-population/sample weights (2,000 cases per window). All60,079/87,937 context rows are retained. The main June2024 origin fits weights on those4,000 earlier predictions. Both earlier label windows end before June2024; their physical source row IDs and target months are disjoint. The shared original pseudo-origins govern context histories. Query rental histories remain fixed at the forecasting cutoff throughout each10-month horizon.

New spatial fields preserve the existing additional one-month publication lag: labels<=feature_origin-1. For a query forecast beginning at T, feature_origin=T-1, so the spatial field's latest rental month is T-2; baseline histories retain their original T-1 cutoff. This conservative policy is kept identical to the earlier implementation, not silently shifted. Query targets are dropped before spatial feature construction. Early missing history retains native NaN,zero support and explicit missing-history flag. Categorical/unknown-value handling and native numeric missingness remain unchanged; no full-data scaling or imputation is added.

For genuine chronological transfer of the weights, September2022 has no earlier ensemble calibration and defaults to baseline-only; July2023 learns weights only on the preceding September2022 window (labels throughJune2023). Main weights use both earlier windows. Pooled training-weight RMSE is a calibration criterion, not out-of-sample performance of the weight optimizer. All metrics use original-scale aligned predictions. Weighted pooled objectives use summed squared errors, never averaged fold RMSEs. Weight fitting enumerates the faces of a nonnegative simplex, giving the convex quadratic MSE optimum for<=3 members, with exact ties favoring the baseline. No exhaustive feature/weight hyperparameter search is used.

{table(weights,['origin','model','baseline_weight','spatial_weight','temporal_weight','active_members','fit_label_max','cold_start'])}

## Chronological transfer and ablations

{table(scores.loc[scores.model.isin(['baseline','spatial','temporal','equal','optimized','standard6'])],['origin','model','n','rmse_sgd','mae_sgd','bias_sgd','rmse_improvement_pct'])}

The optimized pair ablations in comparison.csv answer whether the spatial or temporal member contributes beyond the baseline without selecting weights on the main targets. Compare optimized_baseline_spatial,optimized_baseline_temporal and the three-member optimized result under the same earlier training labels. The earliest optimized row is explicitly a cold-start baseline fallback. No new feature variant was selected after observing these scores.

## Complementarity

Main weighted signed-error correlations range **{mincorr:.6f}-{maxcorr:.6f}**. High correlations limit the available error cancellation, even if averaging reduces noise. Pairwise prediction differences and uncentered error second moments are saved in complementarity.csv,error_correlations_*.csv and error_second_moments_*.csv. An exact check verifies ensemble MSE=w'Qw in SGD. The equal-weight diversity identity compares the ensemble with average individual member MSE; a positive diversity gain alone does not establish improvement over the strongest member. Different biases can also affect averaging; bias is reported directly.

{table(pairframe.loc[pairframe.origin=='2024-06'],['first','second','error_correlation','rms_prediction_difference_sgd','first_bias_sgd','second_bias_sgd','pair_equal_rmse_sgd'])}

## Does diversity beat comparable inference compute?

A standard **six-view same-feature TabPFN ensemble** provides the nominal compute control: three sequential two-estimator77-feature models with seeds42/43/44, averaged equally. It uses the same checkpoint,target/context,batch and GPU cache settings, with no feature diversity. Feature-diversified members instead all use seed42, isolating a comparison with ordinary seed/preprocessing diversity at the same total six-view budget. This control is not one six-estimator TabPFN object. It matches total views, not exact wall-clock cost; preprocessing,84/89 versus77 columns and execution can differ. A zero-weight external member may be skipped, reducing optimized deployment cost below the nominal three-member ensemble.

The original monolithic six-estimator pilot finished its two earlier folds, but at87,937 context rows it allocated19.54GiB and left less than0.5GiB available hostRAM on this16GiB machine. It was therefore not scaled to the main121,005-row context. Pilot results/configurations are preserved in monolithic_six_view_pilot/ and excluded from the uniform sequential control. No feature-member predictions or learned combination weights changed. The sequential control is evaluated uniformly at all three origins; memory is reclaimed between processes.

Nominal three-member sequential prediction time: **{nominal.predict_seconds:.1f}s**, versus{r.loc['baseline','predict_seconds']:.1f}s for baseline and{r.loc['standard6','predict_seconds']:.1f}s for standard6. The optimized active-member sum is{r.loc['optimized','predict_seconds']:.1f}s. Feature diversity is only supported beyond extra internal views if it compares favorably with standard6; mere improvement over baseline2 is insufficient for that claim. All jobs run sequentially on one RTX4080 in separate processes. Memory is peak PyTorch allocation, not total driver reservation; external-ensemble peak is the maximum across sequential active members, not their sum.

All main-window cost timers use the final native-default estimator execution and memory policy: release unused CUDA allocator cache between batches while retaining the live fitted context. This avoids memory paging without changing the fitted model. The current native-default main baseline has a complete measured fit/predict run under the same policy; the old sequential-estimator baseline timer is not mixed into these costs. All1,600 spatial predictions computed before the allocator change reproduce exactly; see allocator_prediction_replay.json. Timing still reflects single-run wall-clock measurements, not a controlled hardware microbenchmark.

{table(cost.loc[cost.origin=='2024-06'],['model','fit_seconds','predict_seconds','peak_allocated_vram_gib','internal_estimators','external_members'])}

## Paired uncertainty

3,000 paired bootstrap replicates,seed42, clustering separately by target month or building. Negative differences favor the model; these are descriptive development intervals, not confirmatory significance. Ten adjacent month clusters are few; building resampling does not capture all common market shocks. Weight-estimation uncertainty and repeated feature development are not removed by these intervals.

{table(ci,['cluster','clusters','model','reference','rmse_difference_sgd','lower95_sgd','upper95_sgd'])}

## Horizon and subgroup checks

Horizon1 isJune2024, horizon10 isMarch2025. Existing evaluation covers10 months, so12/17-month accuracy is unavailable; no Kaggle test labels are used. Month,town,flat-type and exact-horizon tables are in subgroup_scores.csv. The6-month row means exactly horizon6.

{table(groups.loc[groups.dimension=='horizon_band'],['group','model','n','rmse_sgd','rmse_difference_sgd'])}

## Limitations and correctness

An untouched labeled chronological test **cannot be supplied from this repository**: all available labeled periods were previously used in development, and the supplied Kaggle test is unlabelled. The final requested untouched-test claim therefore remains unfulfilled. The work provides locked-weight chronological development evaluations only; independent future labeled data or a new competition score is required for confirmation. Existing full-history DiagScale Kaggle output remains unchanged.

The inherited static amenity/MRT snapshot includes sites whose historical opening availability was not enforced. It remains identical across all compared models; no claim of fully point-in-time external geography is made. The spatial additions require no new external sources; deterministic calendar windows have complete historical availability. Earlier queries are sampled (context/main are full), and only two older market regimes estimate weights. Although weights are chosen without outer labels, the designs were informed by repeatedly examined past experiments. High error correlation and regime changes can make old optimal weights transfer poorly. This focused study covers one spatial and one seasonal temporal representation; it cannot rule out all feature-diversified ensembles.

All24 tests pass: exact convex/simplex boundary and interior solutions, sample-weight behavior, finite inputs, member permutation, coarse-grid dominance, exact baseline-vertex reconstruction, equivalence of three two-view means to a six-view mean, causal rental features under future/own-label mutation, calendar consistency and missingness. Runtime checks enforce identical member row IDs,labels,months,weights and residual baselines, pre-origin context cutoffs, finite predictions, unchanged parameter hashes and prohibited autograd. See tests.log and verification.json. The6-view control and all new two-view members use the plain pretrained class, no adapters and no context subsampling. All original source/checkpoint/submission hashes remain unchanged.

An independent numerical audit replays all12 new spatial fields at sampled query rows in every horizon month plus two context pseudo-origins per fold. Changing unavailable fitting labels and query targets has no effect. See [independent_feature_audit.json](independent_feature_audit.json). This validates the additional rental fields, while the inherited static MRT availability caveat remains.

## Figures

![RMSE](figures/rmse_comparison.png)
![Error correlations](figures/error_correlations.png)
![Weights](figures/learned_weights.png)
![Horizons](figures/horizon_performance.png)
![Cost versus error](figures/cost_vs_rmse.png)

## Reproduce

Run `.venv/Scripts/python.exe analysis/compare_tabpfn_feature_ensemble.py --stage prepare`, then `--stage earlier`, then `--stage select`. Lock reporting with `.venv/Scripts/python.exe -c "from analysis.summarize_tabpfn_feature_ensemble import lock_plan; lock_plan()"`, then run `--stage outer` and `--stage summarize`. Existing completed inference outputs are reused only after hash verification. An incomplete new-memory-policy job replays its saved first200 rows and recomputes the complete query for comparable full timing. Settings/selection guards prevent silently overwriting experiments with different code or weights. Optional independent feature audit: `.venv/Scripts/python.exe analysis/audit_tabpfn_feature_ensemble.py`. Outputs include comparison.csv,learned_weights.csv,complementarity.csv,costs.csv,subgroup_scores.csv,paired_cluster_bootstrap.csv and row-aligned predictions_all_origins.csv. No test submission is generated.
'''
    (OUT/'report.md').write_text(report,encoding='utf-8')
    print(scores.loc[scores.origin=='2024-06',['model','rmse_sgd','mae_sgd','rmse_difference_sgd']].to_string(index=False),flush=True)
    print('DECISION',decision,flush=True)


if __name__=='__main__': summarize()
