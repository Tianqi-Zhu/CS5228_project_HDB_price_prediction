"""Chronological residual diagnosis, selection, frozen evaluation and reporting."""
from __future__ import annotations
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from analysis.compare_tabpfn_spatial_temporal_residual import (
    OUT,CACHE,BASE,REFERENCE,ADAPTER,CHECKPOINT,FOLDS,CFG,CB,save_frame,audit,
)
from analysis.config import ROOT,DATA,REPORTS
from analysis.data import sha256
from analysis.evaluation import metrics
from analysis.compare_local_economy_landscape import immutable,write_json,hashlib_array
from analysis.experiments.spatial_temporal_residual import (
    residual_features,corrected,choose_strength,BASIC,MARKET,CATEGORICAL,
)
from analysis.summarize_local_economy_landscape import cluster_bootstrap,table


def extended_metrics(y,pred):
    m=metrics(y,pred); y=np.asarray(y,float)
    m['r2']=float(1-np.sum((y-pred)**2)/np.sum((y-y.mean())**2))
    return m


def oof_data():
    frames=[]; xs=[]
    for start,_ in FOLDS:
        path=OUT/f'oof_predictions_{start}.csv'; provenance=json.loads((OUT/f'oof_provenance_{start}.json').read_text())
        assert sha256(path)==provenance['prediction_sha256']
        assert provenance['validation_targets_used'] is False and provenance['checkpoint_selection'] is False
        b=joblib.load(CACHE/f'features_{start}.joblib')
        assert sha256(CACHE/f'features_{start}.joblib')==provenance['sha256']
        f=pd.read_csv(path,float_precision='round_trip')
        np.testing.assert_array_equal(f.row_id,b['X_query'].index)
        np.testing.assert_array_equal(f.actual,b['y_query'])
        assert set(f.row_id).isdisjoint(b['X_train'].index)
        assert pd.PeriodIndex(b['train_month'],freq='M').max()<pd.Period(start,'M')
        assert pd.PeriodIndex(f.month,freq='M').min()>=pd.Period(start,'M')
        np.testing.assert_allclose(f.residual,f.actual-f.predicted,atol=1e-10,rtol=0)
        f=f.set_index('row_id',drop=False)
        x=residual_features(b['X_query'],f.predicted.to_numpy(),b['market'])
        frames.append(f); xs.append(x)
    f=pd.concat(frames); x=pd.concat(xs)
    assert f.index.equals(x.index) and f.index.is_unique and np.isfinite(f.predicted).all()
    assert f.month.max()<'2024-06'
    return f,x


def diagnostics(f,x):
    joined=f.copy(); joined['town']=x.TOWN_CLEAN
    summaries={}
    for key in ['month','town']:
        rows=[]
        for name,g in joined.groupby(key):
            rows.append(dict(group=name,n=len(g),mean_residual=float(g.residual.mean()),
                residual_std=float(g.residual.std()),residual_p10=float(g.residual.quantile(.1)),
                residual_p50=float(g.residual.median()),residual_p90=float(g.residual.quantile(.9)),
                **extended_metrics(g.actual,g.predicted.to_numpy())))
        summaries[key]=pd.DataFrame(rows)
        save_frame(summaries[key],OUT/f'residual_diagnostics_{key}.csv')
    acf={str(lag):float(summaries['month'].mean_residual.autocorr(lag)) for lag in [1,2,3]}
    write_json(OUT/'residual_distribution.json',dict(n=len(f),mean=float(f.residual.mean()),std=float(f.residual.std()),
        quantiles={str(k):float(v) for k,v in f.residual.quantile([0,.01,.1,.25,.5,.75,.9,.99,1]).items()},
        monthly_mean_autocorrelation=acf,unit='SGD; residual=actual-predicted',data='OOF only, all before June 2024'))
    save_frame(f,OUT/'oof_predictions.csv')
    fig,axes=plt.subplots(1,3,figsize=(16,5.5),constrained_layout=True)
    axes[0].hist(f.residual,bins=80,color='#0d9488'); axes[0].set_xlabel('Actual minus TabPFN rent (SGD)'); axes[0].set_title('Out-of-sample residuals')
    a=summaries['month']; axes[1].plot(a.group,a.mean_residual,marker='o',color='#0d9488'); axes[1].axhline(0,color='gray',ls='--'); axes[1].tick_params(axis='x',rotation=75,labelsize=7); axes[1].set_title('Mean residual by month'); axes[1].set_ylabel('SGD')
    a=summaries['town'].sort_values('mean_residual'); axes[2].barh(a.group,a.mean_residual,color='#0d9488'); axes[2].axvline(0,color='gray',ls='--'); axes[2].tick_params(axis='y',labelsize=7); axes[2].set_title('Mean residual by town'); axes[2].set_xlabel('SGD')
    fig.savefig(OUT/'residual_diagnostics.png',dpi=160); plt.close(fig)
    return summaries


def develop():
    audit()
    immutable(OUT/'evaluation_plan.json',dict(code_sha256=sha256(Path(__file__)),
        early_stopping_months=['2023-09','2023-12'],strength_selection_months=['2023-12','2024-06'],
        strengths=[0,.25,.5,.75,1],outer_labels_used=False,variants=['E1','E2']))
    f,x=oof_data(); diagnostics(f,x)
    fit=(f.month<'2023-09').to_numpy(); stopping=((f.month>='2023-09')&(f.month<'2023-12')).to_numpy()
    training=(f.month<'2023-12').to_numpy(); validation=(f.month>='2023-12').to_numpy()
    assert fit.any() and stopping.any() and validation.any()
    assert f.loc[training,'month'].max()<f.loc[validation,'month'].min()
    selected={}; curves=[]; validations=[]; importance=[]; durations=[]
    residual=f.residual.to_numpy(); actual=f.actual.to_numpy(); base=f.predicted.to_numpy()
    for name,columns in [('E1',BASIC),('E2',BASIC+MARKET)]:
        model_path=CACHE/f'{name}_validation.cbm'; final_path=CACHE/f'{name}_final.cbm'; meta_path=OUT/f'{name}_selection.json'
        if meta_path.exists():
            record=json.loads(meta_path.read_text()); assert sha256(model_path)==record['validation_model_sha256']; assert sha256(final_path)==record['final_model_sha256']
            model=CatBoostRegressor(); model.load_model(str(model_path)); delta=model.predict(x.loc[validation,columns])
        else:
            start=time.perf_counter()
            probe=CatBoostRegressor(**CB)
            probe.fit(x.loc[fit,columns],residual[fit],cat_features=CATEGORICAL,
                eval_set=(x.loc[stopping,columns],residual[stopping]),early_stopping_rounds=75,use_best_model=True)
            iterations=int(probe.tree_count_)
            model=CatBoostRegressor(**{**CB,'iterations':iterations})
            model.fit(x.loc[training,columns],residual[training],cat_features=CATEGORICAL)
            delta=np.asarray(model.predict(x.loc[validation,columns]),float)
            strength,curve=choose_strength(actual[validation],base[validation],delta)
            model.save_model(str(model_path))
            # Freeze tree count and strength before refitting on all pre-outer OOF labels.
            final=CatBoostRegressor(**{**CB,'iterations':iterations})
            final.fit(x[columns],residual,cat_features=CATEGORICAL); final.save_model(str(final_path))
            record=dict(model=name,columns=columns,categorical=CATEGORICAL,iterations=iterations,strength=strength,
                early_stop_fit_rows=int(fit.sum()),early_stop_rows=int(stopping.sum()),residual_training_rows=int(training.sum()),
                validation_rows=int(validation.sum()),final_refit_rows=len(f),validation_model_sha256=sha256(model_path),
                final_model_sha256=sha256(final_path),seconds=time.perf_counter()-start,
                validation_train_ids_sha256=hashlib_array(f.index[training]),validation_ids_sha256=hashlib_array(f.index[validation]),
                final_train_ids_sha256=hashlib_array(f.index),outer_targets_used=False,correction_sign='actual minus base in SGD',
                curve=[dict(strength=s,rmse_sgd=v) for s,v in curve])
            immutable(meta_path,record)
        selected[name]=record
        for point in record['curve']: curves.append(dict(model=name,**point))
        pred=corrected(base[validation],delta,record['strength'])
        np.testing.assert_array_equal(corrected(base[validation],delta,0),base[validation])
        score=extended_metrics(actual[validation],pred)
        zero_skill=float(1-np.sum((residual[validation]-delta)**2)/np.sum(residual[validation]**2))
        ordinary=float(1-np.sum((residual[validation]-delta)**2)/np.sum((residual[validation]-residual[validation].mean())**2))
        validations.append(dict(model=name,strength=record['strength'],**score,residual_zero_baseline_r2=zero_skill,
            residual_ordinary_r2=ordinary,selected_correction_zero_baseline_r2=float(1-np.sum((actual[validation]-pred)**2)/np.sum(residual[validation]**2))))
        out=f.loc[validation,['row_id','month','actual','predicted','residual']].copy()
        out['predicted_residual']=delta; out['corrected']=pred; save_frame(out,OUT/f'validation_predictions_{name}.csv')
        final=CatBoostRegressor(); final.load_model(str(final_path))
        importance.extend(dict(model=name,feature=c,importance=float(v)) for c,v in zip(columns,final.get_feature_importance()))
        durations.append(dict(model=name,iterations=record['iterations'],fit_predict_seconds=record['seconds'],features=len(columns)))
        print('VALIDATED',name,validations[-1],flush=True)
    baseline=extended_metrics(actual[validation],base[validation])
    validations.insert(0,dict(model='B0',strength=0,**baseline,residual_zero_baseline_r2=0,residual_ordinary_r2=np.nan,selected_correction_zero_baseline_r2=0))
    preferred=min(validations,key=lambda row:(row['rmse_sgd'],['B0','E1','E2'].index(row['model'])))['model']
    immutable(OUT/'selection.json',dict(selected=preferred,models=selected,criterion='minimum December2023-May2024 RMSE; ties prefer simpler model',
        baseline_validation=baseline,outer_labels_used=False,validation_row_ids_sha256=hashlib_array(f.index[validation]),
        final_residual_train_last_month=f.month.max(),oof_rows=len(f)))
    save_frame(pd.DataFrame(validations),OUT/'validation_results.csv'); save_frame(pd.DataFrame(curves),OUT/'validation_strength_curves.csv')
    save_frame(pd.DataFrame(importance),OUT/'feature_importance.csv'); save_frame(pd.DataFrame(durations),OUT/'catboost_timings.csv')
    schema=dict(E1=BASIC,E2=BASIC+MARKET,categorical=CATEGORICAL,numeric_missing='CatBoost native NaN',categorical_missing='__missing__',units='rent and residual SGD; growth dimensionless; history count transactions')
    immutable(OUT/'feature_schema.json',schema)
    write_json(OUT/'development_complete.json',dict(selection_sha256=sha256(OUT/'selection.json'),outer_labels_used=False))


def evaluate():
    # The selection artifact must exist before this function can score outer labels.
    selection=json.loads((OUT/'selection.json').read_text())
    assert not selection['outer_labels_used']
    assert sha256(Path(__file__))==json.loads((OUT/'evaluation_plan.json').read_text())['code_sha256']
    settings=json.loads((OUT/'settings.json').read_text())
    for path,h in settings['source_sha256'].items(): assert sha256(ROOT/path)==h
    for path,h in settings['implementation_sha256'].items(): assert sha256(ROOT/path)==h
    base=joblib.load(BASE); ref=pd.read_csv(REFERENCE,float_precision='round_trip')
    outer_path=CACHE/'outer_features.joblib'; assert sha256(outer_path)==json.loads((OUT/'outer_features.json').read_text())['sha256']
    data=joblib.load(outer_path); b=data['bundle']
    for k in ['X_train','X_query']: pd.testing.assert_frame_equal(b[k],base[k])
    for k in ['y_train','y_query','train_baseline','query_baseline']: np.testing.assert_array_equal(b[k],base[k])
    q=b['X_query']; np.testing.assert_array_equal(q.index,ref.row_id); np.testing.assert_array_equal(b['y_query'],ref.actual)
    x=residual_features(q,ref.predicted.to_numpy(),data['market'])
    f,_=oof_data(); assert set(f.row_id).isdisjoint(ref.row_id)
    assert f.month.max()<ref.month.min()
    predictions={'B0':ref.predicted.to_numpy()}; raw={}; times=[]
    for name in ['E1','E2']:
        record=selection['models'][name]; model=CatBoostRegressor(); path=CACHE/f'{name}_final.cbm'
        assert sha256(path)==record['final_model_sha256']; model.load_model(str(path))
        assert model.feature_names_==record['columns']==list(x[record['columns']])
        start=time.perf_counter(); delta=np.asarray(model.predict(x[record['columns']]),float)
        raw[name]=delta; predictions[name]=corrected(predictions['B0'],delta,record['strength'])
        np.testing.assert_array_equal(corrected(predictions['B0'],delta,0),predictions['B0'])
        times.append(dict(model=name,predict_seconds=time.perf_counter()-start))
    scores=[]; monthly=[]; grouped=[]; intervals={}; comparison=ref[['row_id','month','actual']].copy()
    actual=ref.actual.to_numpy(); control=extended_metrics(actual,predictions['B0'])
    links=pd.read_csv(REPORTS/'tabpfn_local_economy_landscape_80_20/transaction_building_links.csv',index_col='row_id')
    building_ids=links.loc[ref.row_id,'building_id'].to_numpy()
    for name,pred in predictions.items():
        m=extended_metrics(actual,pred)
        scores.append(dict(model=name,n=len(q),strength=0 if name=='B0' else selection['models'][name]['strength'],**m,
            delta_rmse_sgd=m['rmse_sgd']-control['rmse_sgd'],rmse_improvement_pct=100*(control['rmse_sgd']-m['rmse_sgd'])/control['rmse_sgd'],
            mae_improvement_pct=100*(control['mae_sgd']-m['mae_sgd'])/control['mae_sgd']))
        comparison[name]=pred
        if name!='B0':
            comparison[f'{name}_predicted_residual']=raw[name]
            intervals[name]=dict(month=cluster_bootstrap(actual,predictions['B0'],pred,ref.month,n=5000),
                                 building=cluster_bootstrap(actual,predictions['B0'],pred,building_ids,n=5000))
        for month,idx in ref.groupby('month').groups.items(): monthly.append(dict(model=name,month=month,n=len(idx),**extended_metrics(actual[idx],pred[idx])))
        for c in ['TOWN_CLEAN','FLAT_TYPE_CLEAN']:
            for group,ids in x.groupby(c).groups.items():
                idx=q.index.get_indexer(ids); grouped.append(dict(model=name,category=c,group=group,n=len(idx),**extended_metrics(actual[idx],pred[idx])))
    save_frame(comparison,OUT/'predictions.csv'); save_frame(pd.DataFrame(scores),OUT/'development_holdout_results.csv')
    save_frame(pd.DataFrame(monthly),OUT/'monthly_scores.csv'); save_frame(pd.DataFrame(grouped),OUT/'grouped_scores.csv')
    save_frame(pd.DataFrame(times),OUT/'inference_timings.csv'); write_json(OUT/'cluster_bootstrap.json',intervals)
    validation=pd.read_csv(OUT/'validation_results.csv')
    model_comparison=pd.DataFrame(scores).merge(validation[['model','rmse_sgd']].rename(columns={'rmse_sgd':'validation_rmse_sgd'}),on='model')
    save_frame(model_comparison,OUT/'model_comparison.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    for name in predictions:
        part=pd.DataFrame(monthly).query('model == @name'); axes[0].plot(part.month,part.rmse_sgd,marker='o',label=name)
    axes[0].tick_params(axis='x',rotation=60); axes[0].set_ylabel('RMSE (SGD)'); axes[0].legend(); axes[0].set_title('Existing 80/20 development holdout')
    imp=pd.read_csv(OUT/'feature_importance.csv').query("model == 'E2'").sort_values('importance')
    axes[1].barh(imp.feature,imp.importance,color='#0d9488'); axes[1].set_xlabel('CatBoost prediction-value-change importance'); axes[1].tick_params(axis='y',labelsize=8); axes[1].set_title('E2 feature importance (association)')
    fig.savefig(OUT/'comparison.png',dpi=160); plt.close(fig)
    oof_provenance=[json.loads((OUT/f'oof_provenance_{start}.json').read_text()) for start,_ in FOLDS]
    verification=dict(oof_targets_excluded_from_adapter_training=True,oof_query_labels_not_used_for_checkpoint_selection=True,
        outer_rows_excluded_from_residual_training=True,market_features_fixed_at_batch_origin=True,schemas_consistent=True,
        predictions_and_targets_in_sgd=True,row_ids_aligned_and_unique=True,predictions_complete_and_finite=True,
        lambda_zero_bit_exact=True,original_artifacts_unchanged=True,outer_targets_not_used_for_residual_selection=True,
        oof_rows=len(f),outer_rows=len(ref),fold_provenance=oof_provenance,
        final_test_metrics_available=False,selection_sha256=sha256(OUT/'selection.json'),
        prediction_sha256=sha256(OUT/'predictions.csv'))
    write_json(OUT/'verification.json',verification)
    report(model_comparison,validation,selection,intervals)
    write_json(OUT/'status.json',dict(status='complete',selected_by_earlier_validation=selection['selected'],report='report.md',competition_submission_unchanged=True))
    print(model_comparison.to_string(index=False),flush=True)


def report(comparison,validation,selection,intervals):
    chosen=selection['selected']; row=comparison.set_index('model').loc[chosen]
    text=['# Spatial-temporal CatBoost correction of TabPFN residuals','',
        f"Earlier chronological validation selects **{chosen}**. On the existing 80/20 development holdout, this frozen choice scores **S${row.rmse_sgd:.3f} RMSE**, a change of **S${row.delta_rmse_sgd:+.3f}** from the unchanged MSE-only DiagScale baseline.",'',
        '## Scope and baseline ambiguity','',
        'The numerical-best existing declared TabPFN result is MSE-only DiagScale (S$455.071). Its adapter was trained before December 2023 and its epoch selected on 600 weighted December 2023-May 2024 queries. The repository also retains older frozen/integer controls in its README; no single global earlier-only rule selected a winner across every later architectural family. This study explicitly targets the existing numerical-best adapted model, fixed before fitting residual corrections. It does not infer a new universal winner from its outer scores.','',
        'The primary 77-feature bundle, TabPFN-3.5 Fast pretrained checkpoint, original selected DiagScale adapter, one deployment view, seed42, native preprocessing and residual reconstruction are preserved. All 121,005 January 2021-May 2024 context rows and 28,995 June 2024-March 2025 development queries are identical to the reference. Baseline predictions are reused and independently rescored; no parameter or existing prediction file is modified.','',
        '**There is no labelled competition test set.** The 50,000 competition rows remain unpredicted by this correction experiment, and their RMSE/MAE cannot be computed. The original 80/20 holdout has been repeatedly inspected in previous work. Its role here is a frozen development-period confirmation, not an untouched final-test claim.','',
        '## Leakage-safe residual generation','',
        table(pd.read_csv(OUT/'fold_boundaries.csv')[['origin','end_exclusive','context_rows','query_rows','context_max_month']]),'',
        'Each fold starts from the original pretrained backbone and fits a new adapter using only pre-origin rows. No existing adapter trained on later rental labels is used for historical residuals. The previously selected MSE-only recipe is fixed to one epoch (epoch0 in the original trace), two training views, lr0.01, weight decay0.01, 1024-row tasks and 20% task queries. OOF query labels never enter gradient fitting, early stopping, or checkpoint selection. All remaining backbone parameters are hash-checked against the original frozen tensor digest. Original historical context inputs are reused; every target-derived query history and temporal forecast is rebuilt at that fold origin using existing utilities. Query windows are complete and disjoint; no sampling or duplicate residual observations is used.','',
        'Residuals are actual SGD rent minus out-of-sample TabPFN SGD rent. Target reconstruction adds the query temporal baseline exactly once. The fixed historical recipe was developed in prior project work and is not itself a newly prospective hyperparameter choice. Fold-specific one-epoch recipes preserve the algorithm but are not byte-identical copies of the final adapter selected by its original calibration. The final baseline retains its original pre-December-2023 learned adapter and all pre-June-2024 inference context.','',
        '## Residual diagnostics before final evaluation','',
        'Diagnostics use only out-of-sample residuals before June 2024. residual_diagnostics_month.csv and residual_diagnostics_town.csv contain counts, mean residuals, spread, quantiles and prediction errors. residual_distribution.json reports pooled distribution and autocorrelation of monthly mean residuals (not arbitrarily ordered transaction rows). See residual_diagnostics.png. Spatial and temporal correlations are descriptive, not causal.','',
        '## Residual features and CatBoost selection','',
        f"E1 ({len(BASIC)} columns): "+', '.join(BASIC)+'.','',
        f"E2 ({len(BASIC+MARKET)} columns): E1 plus "+', '.join(MARKET)+'.','',
        'E1 includes forecast horizon because simultaneous future batches have stale fixed-origin information. Calendar year is represented by the existing continuous month index; it is not duplicated as a second equivalent year field. Town and flat type are native CatBoost categories. Numeric missing values retain native NaN; no label-informed imputation or scaling is fitted. No new amenity datasets or proximity features are introduced.','',
        'E2 median uses all strictly pre-origin town x flat-type rents with at least20 observations, otherwise flat-type then global historical median. Growth compares medians from the latest three observed months and the preceding three months, with at least10 observations in each window at the same grouping level; flat-type/global fallbacks have the same paired-window requirement. Insufficient growth is missing. The six-month transaction count always refers to the exact town x flat-type group, even when the median falls back. All future queries in one batch share the same origin; no intermediate query labels update histories. These are recent historical growth signals, not realised future market growth.','',
        'CatBoost uses CPU, depth5, lr0.04, L2=5, seed42 and an800-tree cap. September2022-August2023 residuals fit the early-stopping probe; September-November2023 selects its tree count (patience75). Each variant then refits on all September2022-November2023 OOF residuals. December2023-May2024 selects lambda from {0,.25,.5,.75,1} and the preferred variant, with ties favouring simpler models. After choices are frozen, final CatBoost models refit on all pre-June2024 OOF residuals, retaining those tree counts and strengths. No outer targets enter an eval_set or selection. [CatBoost fit semantics](https://catboost.ai/docs/en/concepts/python-reference_catboostregressor_fit).','',
        table(validation),'',
        'Residual zero-baseline R2 = 1-SSE(residual prediction)/sum(actual residual squared), with the unshrunk residual prediction. Ordinary residual R2 uses residual deviations from their validation mean. The selected-correction skill applies the selected lambda and can be zero even if the raw residual model has negative skill. Full lambda curves are in validation_strength_curves.csv; per-variant tree counts, model hashes, strengths and row counts are in E1_selection.json/E2_selection.json.','',
        '## Frozen 80/20 development confirmation','',table(comparison),'',
        'The table uses development-holdout RMSE/MAE rather than inventing unavailable competition-test scores. Improvements are relative to B0. Both E1/E2 are scored with their earlier-selected strength; the primary model remains the variant selected before outer scoring. Zero strength returns an exact copy of B0.','',
        'Paired uncertainty intervals below are candidate-minus-B0 RMSE, with5000 month-cluster and5000 building-cluster resamples, seed42. They describe this development period and do not adjust for repeated project-wide holdout reuse.','',
        table(pd.DataFrame([dict(model=name,cluster=cluster,lower_95=record['rmse_delta_interval_95'][0],upper_95=record['rmse_delta_interval_95'][1],clusters=record['clusters']) for name,groups in intervals.items() for cluster,record in groups.items()])),'',
        '## Interpretation and limitations','',
        'Positive validation skill is evidence of predictability at that earlier origin; it does not guarantee the same residual bias or conditional error pattern in the next market regime. E1 versus E2 tests the incremental utility of three historical market features under fixed conservative CatBoost settings. Comparing them is predictive, not causal attribution. The expanding residual windows have three/four-month horizons, residual validation has six, and the outer period has ten; long-horizon extrapolation and changing context size remain limitations. Town-level histories can already be encoded indirectly by TabPFN, leaving little incremental signal. Sparse groups and changing rental-market bias can make an apparently learnable correction unreliable.','',
        'No existing best model or submission is replaced. A positive outer development result would still require another forecast origin or genuinely future labelled data before claiming test generalization. Verification covers checkpoint/source preservation, OOF chronology, absence of outer training IDs, feature schema, SGD units, prediction alignment, complete coverage and exact lambda-zero replay.','',
        '## Artifacts and reproduction','',
        'settings.json and feature_schema.json define the experiment; fold_boundaries.csv and oof_provenance_*.json document contexts/adapters. Binary fold adapters, query feature bundles and E1/E2 CatBoost models are under .cache/tabpfn_spatial_temporal_residual_80_20/. Row predictions, diagnostics, curves, importances, figures, metrics and verification.json are in this report directory.','',
        'Run `.venv/Scripts/python.exe -m unittest tests.test_spatial_temporal_residual tests.test_aligned_temporal_features`, then `.venv/Scripts/python.exe analysis/compare_tabpfn_spatial_temporal_residual.py --stage all`. Completed fold predictions and CatBoost models are hash-checked and reused. The baseline checkpoint, adapter, sources, original report and submission hashes must remain unchanged.']
    (OUT/'report.md').write_text('\n'.join(text)+'\n',encoding='utf8')
