"""Independent feature replay and descriptive diagnostics; never reselect models."""
from __future__ import annotations
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analysis.compare_tabpfn_hierarchical_latent import OUT,ROOT,HISTORICAL,OUTERS,raw_frame,load_bundle,check_sources
from analysis.experiments.aligned_temporal_features import HISTORY_COLUMNS,FEATURES,origin_history,LocalTemporalModel
from analysis.compare_local_economy_landscape import immutable,write_json
from analysis.summarize_tabpfn_hierarchical_latent import table,clustered_bootstrap,subgroup_scores
from analysis.data import sha256


def audit_features():
    main=load_bundle('2024-06'); frame=raw_frame()
    static=pd.concat([main['X_train'],main['X_query']]).sort_index()
    frame['_MONTH']=pd.PeriodIndex(frame.month,freq='M')
    frame['PLANNING_AREA']=static.loc[frame.index,'PLANNING_AREA'].astype(str)
    replay=[]; derived=HISTORY_COLUMNS+FEATURES
    for origin in HISTORICAL+['2024-06']:
        b=load_bundle(origin); past=frame.loc[b['X_train'].index]; q=b['X_query']
        temporal=LocalTemporalModel(past)
        # Replay 2 rows in every query month, not only the first forecast month.
        ids=frame.loc[q.index].groupby('month',sort=True).head(2).index
        cutoff=pd.Period(origin,'M')-1
        blind=frame.loc[ids].drop(columns='MONTHLY_RENT')
        assert past._MONTH.max()<=cutoff<blind._MONTH.min()
        x=pd.concat([origin_history(past,blind),temporal.features(blind,cutoff)],axis=1)[derived]
        np.testing.assert_allclose(x,q.loc[ids,derived],rtol=0,atol=1e-7,equal_nan=True)
        for split in ['train','query']:
            np.testing.assert_allclose(b[f'{split}_baseline'],b[f'X_{split}'].TEMP_LOCAL_FORECAST.fillna(0).to_numpy(),rtol=0,atol=1e-10)
        # Historical contexts are subsets of the original causal pseudo-origin
        # context. Those small bundles deliberately omit assignment metadata.
        assignments=b.get('train_temporal_origins',main['train_temporal_origins'].loc[b['X_train'].index])
        origins=sorted(assignments.forecast_origin.dropna().unique())
        context_checks=[]
        for cutoff2 in [origins[min(5,len(origins)-1)],origins[-1]]:
            train_ids=assignments.index[assignments.forecast_origin.eq(cutoff2)][:4]
            history=past.loc[past._MONTH<=cutoff2]
            target=frame.loc[train_ids].drop(columns='MONTHLY_RENT')
            assert history._MONTH.max()<=cutoff2<target._MONTH.min()
            x2=pd.concat([origin_history(history,target),temporal.features(target,cutoff2)],axis=1)[derived]
            np.testing.assert_allclose(x2,b['X_train'].loc[train_ids,derived],rtol=0,atol=1e-7,equal_nan=True)
            context_checks.append(dict(pseudo_origin=str(cutoff2),rows=len(train_ids)))
        replay.append(dict(forecast_origin=origin,query_rows=len(ids),replayed_history_and_temporal_columns=len(derived),
            maximum_absolute_difference=float(np.nanmax(np.abs(x.to_numpy()-q.loc[ids,derived].to_numpy()))),
            query_cutoff=str(cutoff),context_pseudo_origin_checks=context_checks,baseline_reconstruction_verified=True))
        print('FEATURE REPLAY',origin,'passed',flush=True)
    write_json(OUT/'feature_replay_audit.json',dict(verified=True,tolerance_sgd=1e-7,folds=replay,
        limitation='Independent numerical replay of existing label-derived inputs; does not certify inherited static amenity date availability'))


def diagnostics():
    predictions=pd.read_csv(OUT/'predictions_all_origins.csv',float_precision='round_trip')
    f=predictions.loc[predictions.origin=='2024-06'].reset_index(drop=True).copy()
    f['E1']=f.E1_alpha1; f['E2']=f.E2_alpha1
    subgroups=subgroup_scores(f); subgroups.to_csv(OUT/'subgroup_scores_alpha1.csv',index=False)
    ci=clustered_bootstrap(f); ci.to_csv(OUT/'paired_cluster_bootstrap_alpha1.csv',index=False)
    positive=[]
    for origin in OUTERS[1:]:
        grid=pd.read_csv(OUT/'origins'/origin/'hyperparameter_scores.csv')
        for model in ['E1','E2']:
            for condition in ['positive_alpha','alpha1']:
                eligible=grid.loc[(grid.model==model)&((grid.alpha>0) if condition=='positive_alpha' else grid.alpha.eq(1))]
                best=eligible.loc[eligible.cv_rmse_sgd.idxmin()]
                positive.append(dict(origin=origin,model=model,condition=condition,alpha=float(best.alpha),
                    cv_rmse_sgd=float(best.cv_rmse_sgd),baseline_cv_rmse_sgd=float(best.baseline_cv_rmse_sgd),
                    config=best.config,inner_forecast_R2=1-(best.cv_rmse_sgd/best.baseline_cv_rmse_sgd)**2))
    positive=pd.DataFrame(positive); positive.to_csv(OUT/'inner_nonzero_correction_skill.csv',index=False)
    diagnostics=[]
    for origin,g in predictions.groupby('origin'):
        pool=pd.read_csv(OUT/'origins'/origin/'fitting_residuals.csv')
        diagnostics.append(dict(origin=origin,residual_fit_rows=len(pool),
            weighted_historical_residual_mean=float(np.average(pool.residual,weights=pool.fit_weight)),
            evaluation_residual_mean=float(np.average(g.actual-g.E0,weights=g.weight)),
            mean_E1_alpha1_correction=float(np.average(g.E1_total_correction,weights=g.weight)),
            mean_E2_alpha1_correction=float(np.average(g.E2_total_correction,weights=g.weight))))
    shifts=pd.DataFrame(diagnostics); shifts.to_csv(OUT/'residual_distribution_shift.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    horizon=subgroups.loc[subgroups.dimension=='horizon'].copy(); horizon['h']=horizon.group.astype(int)
    for name in ['E0','E1','E2']:
        g=horizon.loc[horizon.model==name].sort_values('h'); axes[0].plot(g.h,g.rmse_sgd,marker='o',label=name if name=='E0' else name+' alpha=1')
    axes[0].set(xlabel='Months ahead (Jun 2024 = 1)',ylabel='RMSE (SGD)',title='Diagnostic only: unattenuated corrections'); axes[0].legend(fontsize=8)
    monthly=pd.read_csv(OUT/'diagnostics/2024-06/monthly_residuals.csv')
    axes[1].plot(pd.to_datetime(monthly.month),monthly.mean_residual,marker='.',label='Historical OOF residual')
    actual=f.groupby('month').residual.mean(); axes[1].plot(pd.to_datetime(actual.index),actual.values,marker='.',label='Outer residual (diagnostic only)')
    axes[1].axhline(0,color='grey',lw=.7); axes[1].axvline(pd.Timestamp('2024-06-01'),color='grey',lw=.7)
    axes[1].set(ylabel='Actual minus E0 (SGD)',title='Residual mean changes across forecast periods'); axes[1].tick_params(axis='x',rotation=30); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT/'figures/unattenuated_horizon_and_residual_shift.png',dpi=140); plt.close(fig)
    report=OUT/'report.md'; original=report.read_text(encoding='utf-8').split('\n## Additional interpretation and independent replay\n')[0]
    support=subgroups.loc[subgroups.dimension.eq('residual_support')]
    horizon_bands=subgroups.loc[subgroups.dimension.eq('horizon_band')]
    text=f'''\n## Additional interpretation and independent replay

**Neither E1 nor E2 supplies validated extra predictive signal in this implementation.** Both choose alpha0 at each non-cold origin. Their reported selected-model horizon/subgroup differences and bootstrap intervals are therefore exactly zero by construction; those zeros are not independent evidence of model equivalence. The alpha1 diagnostics below show what the rejected corrections would have done.

The 313 recurring buildings have across-period effect correlation -0.103, and the 95 common submarkets have correlation -0.318 after removing fitted building/global effects. These residual premiums are not positively persistent in these periods. Strong monthly historical autocorrelation (0.881) coexists with poor out-of-time correction skill. The residual-level shift below is consistent with changes in market regime, forecast horizons and expanding TabPFN context; this experiment cannot identify which caused it.

{table(shifts)}

All main E1/E2 penalty and persistence candidates tie when alpha=0, so reported lambda100 and rho0 are **deterministic tie-break values, not identified optimal effect variances or evidence of zero underlying market persistence**. Conditional on a nonzero correction, the best earlier-validation results still fail to beat E0:

{table(positive,['origin','model','condition','alpha','cv_rmse_sgd','baseline_cv_rmse_sgd','inner_forecast_R2'])}

E2 lowers main historical residual reconstruction RMSE from598.974 to592.800; its alpha1 future RMSE464.246 remains worse than E0's455.396. E2's small alpha1 improvement over E1's464.650 does not justify deploying either model.

The main selected E2 has rho0, so its future dynamic-state forecasts are identically zero; the plotted historical states still come from a regularized joint fit. Other rho candidates were tested only through the documented earlier validation, not selected retrospectively on the outer targets.

### Unattenuated correction uncertainty and subgroup diagnosis

Paired RMSE differences below use alpha1, which was rejected by earlier validation. They are diagnostic, not a proposed deployment setting. Negative E2-E1 means E2 is better than E1; it does not imply that either beats E0. The selected alpha0 bootstrap table earlier is necessarily degenerate.

{table(ci,['cluster','clusters','model','reference','rmse_difference_sgd','lower95_sgd','upper95_sgd'])}

{table(horizon_bands,['group','model','n','rmse_sgd','rmse_difference_sgd'])}

{table(support,['group','model','n','rmse_sgd','rmse_difference_sgd'])}

![Unattenuated horizon errors and residual mean shifts](figures/unattenuated_horizon_and_residual_shift.png)

All flat-type, town, horizon and rare/unseen groups for rejected alpha1 corrections are also saved in [subgroup_scores_alpha1.csv](subgroup_scores_alpha1.csv). No subgroup was used to alter selection. Sparse building coverage and nonpersistent effects limit conclusions about denser, more recent residual collections; this result concerns the specified small Gaussian models and these folds, not every possible latent model.

### Independent feature and artifact checks

In addition to22 passing unit tests, an independent numerical replay reconstructs all46 target-derived inputs from permitted raw labels for2 rows in every target month in all8 inference windows, plus two context pseudo-origins per window. All reproduce the cached inputs within1e-7. All additive baselines match TEMP_LOCAL_FORECAST, with the original zero fallback for missing early context forecasts. See [feature_replay_audit.json](feature_replay_audit.json). This checks historical rent causality and reconstruction, while the inherited static MRT timing caveat remains.

Core experimental code and selection results remain unchanged. [supplementary_audit_manifest.json](supplementary_audit_manifest.json) records the independent postprocessing code and the original plan hash. To regenerate summary artifacts from the completed, locked fits, run `.venv/Scripts/python.exe -m analysis.summarize_tabpfn_hierarchical_latent`, followed by `.venv/Scripts/python.exe analysis/audit_tabpfn_hierarchical_latent.py`. The latent-fitting command is for a fresh fit; immutable selection files intentionally guard against overwriting earlier decisions with a new run.
'''
    report.write_text(original+text,encoding='utf-8')


def main():
    check_sources(); plan=json.loads((OUT/'latent_plan.json').read_text())
    for p,h in plan['code_sha256'].items(): assert sha256(ROOT/p)==h
    immutable(OUT/'supplementary_audit_manifest.json',dict(code_sha256=sha256(Path(__file__)),
        original_plan_sha256=sha256(OUT/'latent_plan.json'),no_model_reselection=True,no_new_inference=True))
    audit_features(); diagnostics(); check_sources()
    print('INDEPENDENT AUDIT COMPLETE; original selections and source hashes unchanged',flush=True)


if __name__=='__main__': main()
