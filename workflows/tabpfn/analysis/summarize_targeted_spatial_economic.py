"""Chronological residual correction and paired reports for targeted ablations."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis.compare_local_economy_landscape import write_json,immutable
from analysis.data import sha256
from analysis.experiments.stock_index_regions import block_positions
from analysis.experiments.financial_sector_rents import table
import analysis.run_targeted_spatial_economic as core


def score(f,prediction):
    error=np.asarray(prediction,float)-f.actual.to_numpy(float);w=f.weight.to_numpy(float)
    return dict(rmse_sgd=float(np.sqrt(np.average(error**2,weights=w))),
                mae_sgd=float(np.average(abs(error),weights=w)),bias_sgd=float(np.average(error,weights=w)))


def residual_history():
    raw=core.raw_frame();parts=[]
    for origin in core.ORIGINS[:-1]:
        f,w=core.ensemble('A0',origin)
        part=raw.loc[f.row_id,['month','TOWN_CLEAN','FLAT_TYPE_CLEAN','STREET','BLOCK']].copy()
        part['residual']=f.actual.to_numpy()-f.prediction.to_numpy()
        part['weight']=f.weight.to_numpy();part['origin']=origin
        part['context_last_month']=pd.Period(origin,'M')-1
        part['row_id']=f.row_id.to_numpy()
        assert (part.month>part.context_last_month).all()
        parts.append(part)
    result=pd.concat(parts)
    assert result.row_id.is_unique
    result.to_csv(core.OUT/'A2_residual_history.csv',index=False)
    return result


def residual_experiment():
    started=time.perf_counter();core.check_sources();history=residual_history();raw=core.raw_frame()
    # Use September2022-origin errors to generate July2023 forecasts. Strength
    # tuning uses only July-December2023 labels; January-April2024 is separate.
    july,_=core.ensemble('A0','2023-07');query=raw.loc[july.row_id].drop(columns='MONTHLY_RENT')
    cutoff=pd.Period('2023-05','M');calibrate=july.month<='2023-12'
    residuals={};grid=[]
    for shrink in (20,50):
        residuals[shrink],audit=core.residual_smoother(history,query,cutoff,shrink=shrink)
        for gamma in (0.,.25,.5,1.):
            candidate=july.prediction.to_numpy()+gamma*residuals[shrink]
            grid.append(dict(shrink=shrink,gamma=gamma,**score(july.loc[calibrate],candidate[calibrate])))
    grid=pd.DataFrame(grid).sort_values(['rmse_sgd','gamma','shrink'])
    chosen=grid.iloc[0];gamma=float(chosen.gamma);shrink=int(chosen.shrink)
    selection=dict(gamma=gamma,shrink=shrink,half_life_months=6,
        calibration_targets=['2023-07','2023-12'],confirmation_targets=['2024-01','2024-04'],
        maximum_selection_label_month='2023-12',main_evaluation_labels_used=False,
        code_sha256=sha256(Path(__file__)),
        source_prediction_hashes={f'{o}_{m}':sha256(core.REFERENCE/f'predictions_{o}_{m}.csv') for o in core.ORIGINS[:-1] for m in core.MEMBERS})
    immutable(core.OUT/'A2_selection.json',selection)
    grid.to_csv(core.OUT/'A2_calibration_grid.csv',index=False)
    # Keep the tuning and later forecast datasets visibly separate.
    confirmation=july.loc[~calibrate].copy()
    confirmation['A2']=july.prediction.to_numpy()[~calibrate]+gamma*residuals[shrink][~calibrate]
    confirmation.to_csv(core.OUT/'A2_early_confirmation.csv',index=False)
    audits=[];rows=[]
    for origin in core.ORIGINS:
        f,_=core.ensemble('A0',origin)
        cutoff=pd.Period(origin,'M')-2
        query=raw.loc[f.row_id].drop(columns='MONTHLY_RENT')
        estimates,audit=core.residual_smoother(history,query,cutoff,shrink=shrink)
        audit.update(origin=origin,cutoff=str(cutoff),query_min_month=f.month.min(),query_max_month=f.month.max())
        audits.append(audit)
        # July2023 fitted strength is a calibration diagnostic, not OOS selection.
        f['A2']=f.prediction.to_numpy()+gamma*estimates
        f['historical_residual']=estimates;f['origin']=origin
        f['selection_status']='pre-calibration cold start' if origin=='2022-09' else 'calibration and early confirmation' if origin=='2023-07' else 'later reused evaluation'
        rows.append(f)
    pd.concat(rows,ignore_index=True).to_csv(core.OUT/'A2_predictions.csv',index=False,float_format='%.17g')
    write_json(core.OUT/'A2_availability_audit.json',dict(selection=selection,origins=audits,seconds=time.perf_counter()-started,
        new_TabPFN_inference_calls=0,query_targets_used_as_history=False,
        confirmation=score(confirmation,confirmation.A2),confirmation_baseline=score(confirmation,confirmation.prediction)))
    print('A2 LOCKED',selection['gamma'],selection['shrink'],'confirmation',score(confirmation,confirmation.A2),flush=True)


def bootstrap(f,candidate):
    rows=[]
    for month,p in f.groupby('month',sort=True):
        w=p.weight.to_numpy(float);y=p.actual.to_numpy(float)
        rows.append([w.sum(),np.sum(w*(p.A0-y)**2),np.sum(w*(p[candidate]-y)**2),
                     np.sum(w*abs(p.A0-y)),np.sum(w*abs(p[candidate]-y))])
    a=np.asarray(rows);n=len(a);block=min(3,n)
    positions=block_positions(n,repeats=3000,block=block)
    draws=a[positions].sum(axis=1)
    delta=np.sqrt(draws[:,2]/draws[:,0])-np.sqrt(draws[:,1]/draws[:,0])
    mae=(draws[:,4]-draws[:,3])/draws[:,0]
    return dict(months=n,block_months=block,rmse_delta_ci_low=float(np.quantile(delta,.025)),rmse_delta_ci_high=float(np.quantile(delta,.975)),
                mae_delta_ci_low=float(np.quantile(mae,.025)),mae_delta_ci_high=float(np.quantile(mae,.975)))


def availability_report():
    statuses=[dict(experiment='A1',status='completed',reason='11 new hierarchy fields appended to each original ensemble member; existing nearby comparables retained'),
              dict(experiment='C1',status='not run: historical vintages unavailable',reason='Official current quarterly-median dataset extends to 2005 but is a 2026-updated snapshot. Public metadata and update-history UI provide ingestion timestamps, not historical values. An original 2020-Q1 Annex D link returned404 and a 2021-Q4 release confirmed release timing but did not restore a full historical median panel. Cannot certify original values at earlier feature origins; publication delay alone cannot fix revisions.'),
              dict(experiment='B1',status='completed with archive gaps',reason='Official archived 2019,2021,2022,2023 town stock/completion tables; conservative explicit availability, no backfill, age/missingness exposed. Older snapshots persist through documented gaps.'),
              dict(experiment='B2',status='not run: historical vintages unavailable',reason='Official MOM page and linked workbook are a current historical snapshot; no complete original EP/S Pass vintage series and original release schedule was established. Reporting-frequency lag cannot certify unrevised historical values. No current figures are backdated or made up.'),
              dict(experiment='A2',status='completed',reason='Reuse 4,000 disjoint frozen-ensemble forward-origin predictions; correction sees only matured earlier residuals; gamma/shrink selected July-Dec2023, checked Jan-Apr2024'),
              dict(experiment='A3/B3/other combinations',status='conditional',reason='Run only if component groups independently pass earlier and later RMSE gates; otherwise do not combine')]
    pd.DataFrame(statuses).to_csv(core.OUT/'experiment_status.csv',index=False)
    write_json(core.OUT/'external_data_review.json',dict(sources=[
        dict(name='HDB quarterly median rents',url='https://data.gov.sg/datasets/d_23000a00c52996c55106084ed0339566/view',metadata_file='sources/rent_metadata.json',status='not admitted: current values lack original historical vintages'),
        dict(name='Data.gov.sg dataset metadata API',url='https://guide.data.gov.sg/developer-guide/dataset-apis/get-dataset-metadata',status='metadata does not provide historical median values'),
        dict(name='Data.gov.sg update-history client',url='https://data.gov.sg/_next/static/chunks/8108-dd90070dba50b08b.js',local_file='sources/8108-dd90070dba50b08b.js',status='UI explicitly describes ingestion timestamps; no vintage-value download shown'),
        dict(name='HDB2020Q1 original median annex',url='https://www.hdb.gov.sg/cs/infoweb/-/media/doc/PressReleases/corppr24042020annex-d.ashx',status='404 at retrieval'),
        dict(name='HDB2021Q4 original release',url='https://www.hdb.gov.sg/hdb-pulse/news/2022/release-of-4th-quarter-2021-public-housing-data',status='release date27Jan2022 confirms quarterly timing; a full original-value panel was not reconstructed'),
        dict(name='MOM foreign-workforce numbers',url='https://www.mom.gov.sg/foreign-workforce-numbers',status='current revised snapshot; original pass-stock vintages not established')],
        rule='No external historical values are used merely because their reference period precedes the forecast date. B1 alone has admitted original annual-report vintages.'))
    return statuses


def summarize():
    core.check_sources();raw=core.raw_frame();residual_experiment()
    variants=['A1','B1'];allrows=[];weights=[];scores=[];monthly=[];groups=[];costs=[]
    residual=pd.read_csv(core.OUT/'A2_predictions.csv',float_precision='round_trip')
    a2meta=json.loads((core.OUT/'A2_availability_audit.json').read_text())
    for origin in core.ORIGINS:
        base,bw=core.ensemble('A0',origin)
        f=base[['row_id','origin','month','actual','weight']].copy();f['A0']=base.prediction
        weights.append(dict(origin=origin,variant='A0',weights=bw.tolist()))
        for variant in variants:
            p,w=core.ensemble(variant,origin)
            for c in ('row_id','month','actual','weight'):np.testing.assert_array_equal(p[c],base[c])
            f[variant]=p.prediction
            f[f'{variant}_fixed_weights']=p[core.MEMBERS].to_numpy()@bw
            weights.append(dict(origin=origin,variant=variant,weights=w.tolist()))
        rr=residual[residual.origin==origin]
        np.testing.assert_array_equal(rr.row_id,f.row_id);np.testing.assert_array_equal(rr.actual,f.actual)
        f['A2']=rr.A2.to_numpy()
        baseline_score=score(f,f.A0)
        assert len(f)==(28995 if origin=='2024-06' else 2000)
        if origin=='2024-06':assert abs(baseline_score['rmse_sgd']-455.1718734783953)<1e-8
        candidates=['A0','A1','A1_fixed_weights','B1','B1_fixed_weights','A2']
        for candidate in candidates:
            metrics=score(f,f[candidate]);delta=metrics['rmse_sgd']-baseline_score['rmse_sgd']
            scores.append(dict(origin=origin,variant=candidate,rows=len(f),**metrics,
                rmse_delta_sgd=delta,rmse_improvement_pct=-100*delta/baseline_score['rmse_sgd'],
                mae_delta_sgd=metrics['mae_sgd']-baseline_score['mae_sgd'],
                mae_improvement_pct=100*(baseline_score['mae_sgd']-metrics['mae_sgd'])/baseline_score['mae_sgd'],
                **bootstrap(f,candidate)))
            for month,part in f.groupby('month',sort=True):
                m,b=score(part,part[candidate]),score(part,part.A0)
                monthly.append(dict(origin=origin,month=month,variant=candidate,rows=len(part),**m,
                    rmse_delta_sgd=m['rmse_sgd']-b['rmse_sgd'],
                    paired_mse_delta_sgd2=float(np.average((part[candidate]-part.actual)**2-(part.A0-part.actual)**2,weights=part.weight)),
                    paired_mae_delta_sgd=m['mae_sgd']-b['mae_sgd']))
            for column in ['TOWN_CLEAN','FLAT_TYPE_CLEAN']:
                labels=raw.loc[f.row_id,column].to_numpy()
                for label in sorted(set(labels)):
                    part=f.loc[labels==label];m,b=score(part,part[candidate]),score(part,part.A0)
                    groups.append(dict(origin=origin,dimension=column,category=label,variant=candidate,rows=len(part),**m,rmse_delta_sgd=m['rmse_sgd']-b['rmse_sgd']))
        for variant in ['A0','A1','B1']:
            directory=core.REFERENCE if variant=='A0' else core.OUT
            metas=[json.loads((directory/f"timing_{origin}_{m if variant=='A0' else variant+'_'+m}.json").read_text()) for m in core.MEMBERS]
            costs.append(dict(origin=origin,variant=variant,fit_seconds=sum(m['fit_seconds'] for m in metas),
                predict_seconds=sum(m['predict_seconds'] for m in metas),peak_vram_gib=max(m['peak_vram_gib'] for m in metas),
                internal_estimators=6,cached_baseline=variant=='A0'))
        allrows.append(f)
    scores=pd.DataFrame(scores);monthly=pd.DataFrame(monthly);costs=pd.DataFrame(costs)
    pd.concat(allrows,ignore_index=True).to_csv(core.OUT/'ensemble_predictions.csv',index=False,float_format='%.17g')
    scores.to_csv(core.OUT/'ablation_scores.csv',index=False);monthly.to_csv(core.OUT/'monthly_scores.csv',index=False)
    pd.DataFrame(groups).to_csv(core.OUT/'subgroup_scores.csv',index=False);costs.to_csv(core.OUT/'inference_costs.csv',index=False)
    write_json(core.OUT/'ensemble_weights.json',dict(weights=weights,main_reference_weights_reproduced=True))
    statuses=availability_report()
    main=scores[scores.origin=='2024-06']
    # A2's July window includes calibration, so use separate untouched-within-
    # development January-April confirmation for its earlier consistency gate.
    a2early=a2meta['confirmation']['rmse_sgd']<a2meta['confirmation_baseline']['rmse_sgd']
    gates={v:bool((scores[(scores.variant==v)&(scores.origin!='2024-06')].rmse_delta_sgd<0).all()
                  and main[main.variant==v].rmse_delta_sgd.iloc[0]<0) for v in variants}
    gates['A2']=bool(a2early and main[main.variant=='A2'].rmse_delta_sgd.iloc[0]<0)
    write_json(core.OUT/'combination_gate.json',dict(individual_consistency=gates,
        A3_eligible=bool(gates['A1'] and gates['A2']),B3_eligible=False,
        reason='Combine only independently consistent gains; B2 unavailable',no_outer_selected_feature_search=True))
    plot_results(monthly,main)
    write_report(main,scores,monthly,costs,statuses,gates,a2meta)
    core.check_sources()
    write_json(core.OUT/'verification.json',dict(baseline_rmse_reproduced=455.1718734783953,
        aligned_rows_all_origins=32995,unchanged_submissions=True,source_hashes_verified=True,
        all_original_inputs_targets_baselines_and_contexts_preserved=True,new_neural_weight_training=False,
        prediction_files_hashed=True,bootstrap_unit='paired contiguous calendar-month blocks',
        no_untouched_labeled_holdout=True,reporting_code_sha256=sha256(Path(__file__))))
    print('Saved',core.OUT/'report.md',flush=True)
    print(main[['variant','rmse_sgd','rmse_delta_sgd','mae_sgd']].to_string(index=False),flush=True)


def plot_results(monthly,main):
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    x=monthly[(monthly.origin=='2024-06')&monthly.variant.isin(['A1','B1','A2'])]
    for variant,p in x.groupby('variant'):ax.plot(pd.to_datetime(p.month),p.rmse_delta_sgd,marker='o',label=variant)
    ax.axhline(0,color='black',lw=1);ax.legend();ax.set(ylabel='RMSE difference from optimized ensemble (SGD)',title='Monthly changes in prediction error (negative is better)')
    fig.savefig(core.OUT/'01_monthly_ablation_differences.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    p=main[main.variant.isin(['A1','B1','A2'])]
    y=p.rmse_delta_sgd.to_numpy();low=np.maximum(y-p.rmse_delta_ci_low.to_numpy(),0);high=np.maximum(p.rmse_delta_ci_high.to_numpy()-y,0)
    ax.errorbar(y,np.arange(len(p)),xerr=np.vstack([low,high]),fmt='o',capsize=4)
    ax.set_yticks(np.arange(len(p)),p.variant);ax.axvline(0,color='black',lw=1)
    ax.set(xlabel='RMSE difference from optimized ensemble (SGD)',title='Paired three-month-block intervals, reused later period')
    fig.savefig(core.OUT/'02_ablation_uncertainty.png',dpi=160);plt.close(fig)


def write_report(main,scores,monthly,costs,statuses,gates,a2meta):
    selection=a2meta['selection']
    lines=['# Targeted spatial-temporal and economic experiments','',
        'The reference is the existing optimized **frozen three-member TabPFN ensemble**, not the weaker single-member control and not DiagScale. Its cached June2024–March2025 RMSE reproduces **S$455.171873**. All 28,995 evaluation rentals and all original training rows, targets, temporal baselines, feature representations and native inference settings are preserved.','',
        '## Main ablations','',table(main,['variant','rows','rmse_sgd','rmse_delta_sgd','rmse_improvement_pct','mae_sgd','mae_delta_sgd','mae_improvement_pct','rmse_delta_ci_low','rmse_delta_ci_high']),'',
        'Negative RMSE/MAE differences are better; positive improvement percentages are better. A1_fixed_weights and B1_fixed_weights retain the reference ensemble weights exactly. Other A1/B1 rows reapply the established earlier-only convex weighting method to their added-feature predictions. Neither method tunes weights on the later labels.','',
        '## Earlier chronological results','',table(scores[scores.origin!='2024-06'],['origin','variant','rows','rmse_sgd','rmse_delta_sgd','mae_sgd']),'',
        'The September2022–June2023 and July2023–April2024 windows contain 200 label-independent queries per month and population sampling weights, with all60,079/87,937 available context rows. The first window uses the baseline member as the established cold-start ensemble fallback; July weights use only September-window labels. June2024 weights use only the two earlier disjoint windows. Main context is121,005 rows, with28,995 complete later queries. These periods have all been examined in earlier project work.','',
        '## A1: hierarchical local rental dynamics','',
        'The reference already contains37 historical rental fields, nine local temporal fields, and nearby same-type500/1000m and Gaussian comparable histories in its spatial member. Those are reused. Eleven additions test missing town/street hierarchy and relative growth: town/type and street/type shrunken3-month level,6-month count and dispersion, and3-month growth relative to national/type growth; building/type level and relative growth shrunken toward the street; and a no-history indicator.','',
        'At every pseudo-origin, stats use only the preceding permitted six-month history. Recent three months are compared with the previous three months. Pooled national -> national/type -> town/type -> street/type -> building/type sufficient statistics borrow20 observations from their parent, including second moments for dispersion. Local growth is100log(recent/previous) minus national/type growth. Existing room counts/model/area and nearby comparables remain in the original members. No new hyperparameters are selected.','',
        'Each existing row has its original forecast origin reconstructed from its month and TEMP_FORECAST_HORIZON. Added features stop one extra month before that origin, preserving the spatial member’s conservative publication buffer. Main query features therefore stop April2024 and remain fixed over the ten-month forecast. Context rows use their individual earlier pseudo-origins; the first unsupported rows retain NaN,zero support and explicit missingness. No current/query labels or future rental months enter the additions. history_cutoffs.csv and targeted future-label poisoning tests verify this.','',
        '## B1: town housing completions','',
        'One substantive supply feature is annual completed HDB residential units / reported town HDB stock, accompanied by observation age and unavailability. It is extracted from archived HDB2018/19,2020/21,2021/22,2022/23 tables. Numeric extraction is checked against every printed national stock/completion total and selected source tables are visually verified. This is new information beyond building age and static geography.','',
        'Availability is conservative and explicit: 2019 report fromJanuary2020; the retrieved2021 SGX filing only fromJanuary2023 because its verified broadcast isNovember2022;2022 report fromJanuary2023 after the October2022 release;2023 report fromJanuary2024 after the October2023 release. The newer2022 report supersedes2021 when both become available. FY2019/20 and an earlier verified2021 circulation were not reconstructed, so the older2019 snapshot remains in use through the gap, with its age exposed. That is stale known information, not interpolation or future backfill. See housing_supply_sources.json for URLs, timing evidence, PDF hashes and totals.','',
        'Completions include home-ownership/public-rental housing and do not directly measure newly rentable open-market units. Minimum-occupation rules and owner-occupier moves can delay or change their relation to rental supply. This experiment tests a broad supply proxy, not the number of available rental listings. Zero-stock/unknown cells retain missing values and an explicit flag; no query is removed.','',
        '## C1 and B2: not admitted to retrospective inference','',
        'C1: official quarterly HDB medians extend to2005, but the current retrieved series was updated inOctober2026. Public metadata and its See history component record ingestion timestamps, not historical-value vintages. A historical2020Q1 Annex D link returned404. A dated2021Q4 release verifies publication lag but is insufficient to reconstruct an entire original-value panel. Using the latest series with an arbitrary quarter lag would not establish absence of retrospective revisions. Long-term-cycle inputs are therefore not fabricated or tested under a false point-in-time claim.','',
        'B2: MOM’s current foreign-workforce page/workbook reports historical EP/S Pass levels, but the original series and original publication schedule at every earlier forecast cutoff were not established. The page notes revisions at industry level; that does not prove the pass totals were revised, nor does it certify they were unchanged. Current historical values are not silently backdated. No workforce-demand performance claim is made. external_data_review.json records the inspected sources and precise limitation.','',
        '## A2: cross-fitted historical residual correction','',
        f"Reused4,000 disjoint earlier out-of-time queries from the reference ensemble. Every source residual is actual minus a prediction whose context predates that target month; the ensemble weights themselves also use only earlier windows. The smoother uses exponential six-month decay and global -> flat type -> town/type -> street/type -> building/type shrinkage, with sample weights normalized to observed effective support. The small search selects gamma={selection['gamma']} and shrink={selection['shrink']} using July–December2023 targets only, with January–April2024 reserved for an earlier confirmation check. No new TabPFN inference is needed.",'',
        'July2023 smoothing sees only residuals throughMay2023. MainJune2024 smoothing sees only residuals throughApril2024; later validation residuals never update it. Training/query cross-fitting alone would not suffice, so both provenance and maturity are audited. The aggregate July2023 A2 row includes tuning labels and is a calibration diagnostic. The separate January–April2024 check is the earlier out-of-time correction result.','',
        f"Earlier confirmation baseline RMSE: S${a2meta['confirmation_baseline']['rmse_sgd']:.3f}; A2: S${a2meta['confirmation']['rmse_sgd']:.3f}. A2 CPU work took{a2meta['seconds']:.2f}s in this run and reused the existing ensemble predictions; no additional GPU inference cost.",'',
        '## Uncertainty and subgroups','',
        'Main intervals jointly resample paired contiguous three-month blocks, retaining all source rows and their population weights within a month. There are ten later calendar months, not28,995 independent time observations. Intervals condition on fitted models and omit fitting/feature-search uncertainty. They are exploratory, without fresh confirmation after repeated reuse of this holdout. monthly_scores.csv records each month’s RMSE,MAE and paired squared/absolute-error differences; subgroup_scores.csv reports all towns and flat types, including small groups whose estimates are unstable.','',
        '![](01_monthly_ablation_differences.png)','',
        '![](02_ablation_uncertainty.png)','',
        '## Inference cost and frozen-model verification','',table(costs[costs.origin=='2024-06'],['variant','fit_seconds','predict_seconds','peak_vram_gib','internal_estimators','cached_baseline']),'',
        'Each added-feature variant has three sequential plain V3.5Fast models, two internal estimators per member,seed42,200-row batches,full unchanged context,native estimator batching and no adapter/subsampling. All gradient calls are prohibited; before/after parameter digests and versions match. Peak memory is the maximum across sequential members, not their sum. The table uses measured inference timings; feature preparation/acquisition/hashing were not separately timed. Baseline times are prior audited runs on the same RTX4080, not simultaneous benchmarks. No paid API inference is used.','',
        '## Decision and execution status','',table(pd.DataFrame(statuses),['experiment','status','reason']),'',
        f"Consistency gates (earlier and later RMSE): {json.dumps(gates)}. Only independently consistent components qualify for combinations. A2 uses its separate early confirmation rather than its calibration rows for that gate. No additional broad feature/parameter search is performed to rescue a negative result.",'',
        'Retain the existing optimized frozen ensemble unless a component passes the documented consistency gate and its benefit is meaningful relative to uncertainty and cost. The source submissions and all prior reports/models remain untouched. This report has no Kaggle test score because test rents are unavailable.','',
        '## Reproduction','',
        '1. `.venv/Scripts/python.exe analysis/acquire_targeted_economic_sources.py` (downloads original documents without replacing existing files; validates hashes when a manifest exists).',
        '2. `.venv/Scripts/python.exe analysis/run_targeted_spatial_economic.py --stage all`',
        '3. `.venv/Scripts/python.exe analysis/run_targeted_housing_supply.py --stage all`',
        '4. `.venv/Scripts/python.exe analysis/summarize_targeted_spatial_economic.py --stage all`',
        '5. `.venv/Scripts/python.exe -m unittest tests.test_targeted_spatial_economic`','',
        'Prepared bundles live in.cache/targeted_spatial_economic/ with hashes and full feature lists in this directory. settings.json and B1_settings.json freeze inputs/code. Individual member predictions/timings,ensemble_predictions.csv,ensemble_weights.json,A2_selection.json,ablation_scores.csv,monthly_scores.csv,subgroup_scores.csv and verification.json retain complete results.','',
        'Official sources: [HDB quarterly medians](https://data.gov.sg/datasets/d_23000a00c52996c55106084ed0339566/view), [MOM workforce numbers](https://www.mom.gov.sg/foreign-workforce-numbers), [HDB2022 annual-report release](https://www.hdb.gov.sg/hdb-pulse/news/2022/hdb-annual-report-fy2021). Original annual PDFs and availability citations are recorded in housing_supply_sources.json.','']
    (core.OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['a2','all'],default='all');a=p.parse_args()
    if a.stage=='a2':residual_experiment()
    else:summarize()


if __name__=='__main__':main()
