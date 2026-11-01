"""Verify and summarize completed amenity-quality experiments without retuning."""
from __future__ import annotations
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis.config import ROOT,DATA
from analysis.data import sha256
from analysis.evaluation import metrics
from analysis.compare_amenity_quality import paths
from analysis.summarize_local_economy_landscape import cluster_bootstrap,table
from analysis.compare_local_economy_landscape import write_json
from analysis.compare_tabpfn_v3_5_fast_80_20 import CHECKPOINT
from analysis.experiments.amenity_quality_features import historical_latest

LABELS={'baseline':'Baseline (77)', 'schools_dated':'Schools (82)', 'malls_dated':'Malls (84)',
        'transport_canonical_dated':'MRT / buses (86)', 'combined_canonical_dated':'Combined dated (98)',
        'combined_canonical_static_weights':'Combined retrospective (98)'}

def main():
    cfg,out,cache=paths(); settings=json.loads((out/'settings.json').read_text()); prep=json.loads((out/'prepared_manifest.json').read_text())
    assert sha256(ROOT/cfg['baseline'])==cfg['baseline_sha256']
    assert sha256(CHECKPOINT)==settings['checkpoint_sha256']
    assert sha256(cache/'prepared.joblib')==prep['sha256']
    assert sha256(cache/'building_features.joblib')==prep['building_cache_sha256']
    for source,h in settings['source_sha256'].items(): assert sha256(cache/'sources'/source)==h
    for code,h in settings['code_sha256'].items(): assert sha256(ROOT/code)==h
    refinement=json.loads((out/'bus_name_refinement.json').read_text())
    assert refinement['parent_prepared_sha256']==prep['sha256']
    assert sha256(cache/'prepared_bus_aliases.joblib')==refinement['sha256']
    assert sha256(ROOT/'analysis/refine_amenity_bus_names.py')==refinement['code_sha256']
    for name,h in refinement['canonical_file_sha256'].items(): assert sha256(cache/'canonical_bus_names'/name)==h
    preservation=json.loads((ROOT/'reports/tabpfn_local_economy_landscape_80_20/submission_preservation.json').read_text())
    assert sha256(ROOT/preservation['path'])==preservation['sha256']
    base=joblib.load(ROOT/cfg['baseline']); data=joblib.load(cache/'prepared_bus_aliases.joblib'); b=data['outer']['bundle']
    for k in ['X_train','X_query']: pd.testing.assert_frame_equal(b[k],base[k])
    for k in ['y_train','y_query','train_baseline','query_baseline']: np.testing.assert_array_equal(b[k],base[k])
    reference=pd.read_csv(ROOT/cfg['baseline_predictions']); assert sha256(ROOT/cfg['baseline_predictions'])==settings['baseline_predictions_sha256']
    np.testing.assert_array_equal(reference.row_id,b['X_query'].index); np.testing.assert_array_equal(reference.actual,b['y_query'])
    models={'baseline':reference}; timings=[]
    for path in sorted(out.glob('predictions_outer_*.csv')):
        if 'smoke' in path.name: continue
        name=path.stem.removeprefix('predictions_outer_'); f=pd.read_csv(path)
        np.testing.assert_array_equal(f.row_id,reference.row_id); np.testing.assert_array_equal(f.actual,reference.actual); np.testing.assert_array_equal(f.month,reference.month)
        assert f.row_id.is_unique and len(f)==28995 and np.isfinite(f.predicted).all()
        timing=json.loads((out/f'timing_outer_{name}.json').read_text())
        job=json.loads((out/f'job_outer_{name}.json').read_text())
        assert job['checkpoint_sha256']==settings['checkpoint_sha256']
        assert job['prepared_sha256']==(refinement['sha256'] if '_canonical_' in name else prep['sha256'])
        assert timing['context_rows']==121005 and not timing['subsampling'] and timing['parameter_versions_unchanged']
        models[name]=f; timings.append(dict(model=name,**timing))
    actual=reference.actual.to_numpy(); control=reference.predicted.to_numpy(); base_rmse=metrics(actual,control)['rmse_sgd']
    raw=pd.read_csv(DATA/'train.csv'); query=raw.loc[reference.row_id].set_axis(reference.index)
    links=pd.read_csv(ROOT/'reports/tabpfn_local_economy_landscape_80_20/transaction_building_links.csv',index_col='row_id')
    query['building_id']=links.loc[reference.row_id,'building_id'].to_numpy()
    scores=[]; monthly=[]; geographic=[]; bootstrap={}; joined=reference.rename(columns={'predicted':'baseline'})
    for name,f in models.items():
        pred=f.predicted.to_numpy(); m=metrics(actual,pred)
        scores.append(dict(model=name,n=len(f),**m,delta_rmse_sgd=m['rmse_sgd']-base_rmse,relative_rmse_improvement_pct=100*(base_rmse-m['rmse_sgd'])/base_rmse,
                           r2=1-np.sum((pred-actual)**2)/np.sum((actual-actual.mean())**2)))
        for month,idx in f.groupby('month').groups.items(): monthly.append(dict(model=name,month=month,n=len(idx),**metrics(actual[idx],pred[idx])))
        for col in ['TOWN','FLAT_TYPE']:
            for value,idx in query.groupby(col).groups.items(): geographic.append(dict(model=name,category=col,group=value,n=len(idx),**metrics(actual[idx],pred[idx])))
        if name!='baseline':
            bootstrap[name]=dict(building=cluster_bootstrap(actual,control,pred,query.building_id),month=cluster_bootstrap(actual,control,pred,reference.month))
            joined[name]=pred
    scores=pd.DataFrame(scores)
    order={name:i for i,name in enumerate(LABELS)}
    scores=scores.sort_values('model',key=lambda s:s.map(order)).reset_index(drop=True)
    scores.to_csv(out/'scores.csv',index=False)
    pd.DataFrame(monthly).to_csv(out/'monthly_scores.csv',index=False); pd.DataFrame(geographic).to_csv(out/'grouped_scores.csv',index=False)
    pd.DataFrame(timings).to_csv(out/'timings.csv',index=False); joined.to_csv(out/'predictions_comparison.csv',index=False,float_format='%.10f')
    write_json(out/'cluster_bootstrap.json',bootstrap)
    inner_rows=[]
    for group in settings['groups']:
        variant=f'{group}_canonical' if group in ['transport','combined'] else group
        files=[out/f'predictions_{stage}_{variant}_dated.csv' for stage in cfg['inner_stages']]
        if not all(f.exists() for f in files): continue
        frames=[pd.read_csv(f) for f in files]; error=np.concatenate([(f.predicted-f.actual).to_numpy() for f in frames]); weights=np.concatenate([f.weight for f in frames])
        inner_rows.append(dict(model=group,query_rows=len(error),rmse_sgd=float(np.sqrt(np.average(error**2,weights=weights))),mae_sgd=float(np.average(np.abs(error),weights=weights))))
    inner=pd.DataFrame(inner_rows); inner.to_csv(out/'inner_scores.csv',index=False)
    coverage=pd.read_csv(out/'feature_coverage.csv'); outer_coverage=coverage.loc[(coverage['stage']=='outer')&(coverage['mode']=='dated'),['split','feature','missing_fraction']]
    source_stats=pd.read_csv(out/'mall_records.csv'); school_stats=pd.read_csv(out/'school_name_match_audit.csv')
    school_records=pd.read_csv(out/'school_demand_records.csv')
    latest=historical_latest(school_records,'school_key','2024-05-31').copy()
    latest['percentile_of_latest_records']=latest.log_demand.rank(method='average',pct=True)
    latest['above_median_access_weight']=np.maximum(latest.percentile_of_latest_records-.5,0)*2
    school_stats.merge(latest,on='school_key',how='left').to_csv(out/'school_model_weights_2024-05.csv',index=False)
    station_weights=pd.read_csv(out/'station_weight_assignments_canonical_2024-05.csv')
    taps=pd.read_csv(out/'station_weight_assignments_2024-05.csv')[['station_key','daily_station_entry_exit_taps_2023']]
    station_weights.merge(taps,on='station_key',how='left',validate='one_to_one').to_csv(out/'station_model_weights_2024-05.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),constrained_layout=True)
    for pos,row in enumerate(scores.itertuples(index=False)):
        lo,hi=bootstrap[row.model]['building']['rmse_delta_interval_95'] if row.model in bootstrap else (0,0)
        axes[0].plot([lo,hi],[pos,pos],color='#0d9488',linewidth=2); axes[0].scatter([row.delta_rmse_sgd],[pos],color='#64748b' if row.model=='baseline' else '#0d9488')
    axes[0].set_yticks(range(len(scores)),scores.model.map(lambda n:LABELS.get(n,n))); axes[0].invert_yaxis(); axes[0].axvline(0,color='#b91c1c',linestyle='--')
    axes[0].set_xlabel('RMSE change (SGD; negative improves)'); axes[0].set_title('Paired building-cluster bootstrap 95% intervals',fontsize=10)
    monthly=pd.DataFrame(monthly)
    for name in models:
        if 'static_weights' in name: continue
        part=monthly.loc[monthly.model==name]; axes[1].plot(part.month,part.rmse_sgd,marker='o',markersize=3,label=LABELS.get(name,name))
    axes[1].tick_params(axis='x',rotation=60); axes[1].set_ylabel('Monthly RMSE (SGD)'); axes[1].legend(fontsize=7)
    fig.savefig(out/'comparison.png',dpi=160); plt.close(fig)
    recommendation='Full holdout inference is pending.'
    if 'combined_canonical_dated' in models:
        d=scores.set_index('model').loc['combined_canonical_dated','delta_rmse_sgd']; lo,hi=bootstrap['combined_canonical_dated']['building']['rmse_delta_interval_95']
        recommendation=f'The combined dated-source augmentation changes RMSE by S${d:+.3f}, with paired building-cluster 95% interval [{lo:+.3f}, {hi:+.3f}]. '
        recommendation+=('Retain the 77-feature control; no robust gain is established.' if hi>=0 else 'The development-period gain warrants confirmation on independent future data; no submission is replaced automatically.')
    report=['# Frozen TabPFN amenity-quality augmentation','',recommendation,'',
      'This experiment adds source-backed school demand, mall scale/activity and MRT connectivity/passenger measurements to all 77 existing baseline features. No TabPFN training, adapters, blending or context subsampling is used. The same checkpoint, single estimator, seed 42, residual targets, reconstruction and 200-row query batches are retained.','',
      '## Evaluation and experiment definitions','',
      'Chronological 80/20: 121,005 context rows (January 2021–May 2024) and the identical 28,995 June 2024–March 2025 development queries. The 50,000 unlabelled competition rows and existing submission are unchanged. Two earlier ten-month folds retain every context row and use the existing fixed label-free 600-query samples per fold, weighted to monthly populations. The four feature blocks and full evaluations were defined before new outer scoring. No outer targets set source values or weights.','',
      '| Group | Added dimensions | Total input columns |\n| --- | ---: | ---: |\n| Baseline | 0 | 77 |\n| Schools | 5 | 82 |\n| Malls | 7 | 84 |\n| Transport | 9 | 86 |\n| Combined | 21 | 98 |','',
      'These are logical input-column counts; the SDK can remove constant/all-missing columns in a particular fold.','',
      'Actual May 2024 per-location inputs are saved in school_model_weights_2024-05.csv, mall_weight_assignments_2024-05.csv and station_model_weights_2024-05.csv. The older school_weight_assignments.csv gives within-year reference ranks, not the mixed-latest-record ranks used by the current model. Per-property features apply the documented distance kernels to these source values.','',
      '## Sources and measurements','',
      f'Schools: {len(school_stats)} existing primary-school locations joined to 729 historical school-year records: 186 schools in 2020 and 181 in each of 2021–2023. {int(school_stats.matched_2023.sum())} of the 182 sites match the 2023 archive. Source: [SG Schooling historical tables](https://sgschooling.com/year/2020/all), tabulating MOE registration facts. This is a secondary archive, not an official ranking; historical versions are not independently preserved here. P1 demand is a proxy for parental demand, not a teaching-quality assessment or a secondary-school league table.','',
      'The school block contains nearest-school log((Phase-2C applicants+1)/(vacancies+1)), maximum latest-available demand percentile within 1 km, distance-weighted above-median demand access, source coverage within 1 km, and nearest-source unknown indicator. Same-year figures enter conservatively in October after Phase 2C. Percentiles rank the latest available record per archived school at each cutoff. Schools missing from a newer archive can retain earlier records, so this is not strict within-registration-year normalization; the 2022 reserved-seat change can still affect comparisons for such records. Distance decay is exp(-d/500 m). Primary admissions source columns are parsed separately across old/new phase structures and tested against sample rows.','',
      f'Malls: published measurements for {source_stats.mall.nunique()} of the 113 existing mall locations, extracted from CMT/CICT and FCT 2019–2023 owner reports plus Link REIT\'s dated Jurong Point disclosure. [CICT reports](https://investor.cict.com.sg/ar.html), [FCT reports](https://fct.frasersproperty.com/ar.html), [Link disclosure](https://www.linkreit.com/en/media/news-releases/link-enters-singapore-through-the-acquisition-of-jurong-point-and-swing-by-thomson-plaza/). Source page numbers, units, property scope and extraction method are in mall_records.csv. Representative pages were rendered and visually checked, exposing and excluding grouped Bugis+/Bukit Panjang Plaza revenue.','',
      'Mall features: nearest reported retail NLA, distance-weighted reported NLA, owner gross revenue and annual footfall within 2 km, NLA coverage, revenue coverage and nearest record-publication age. Sums use exp(-d/750 m) and log1p; NLA is converted to square metres. Gross property revenue is not tenant retail sales or net property income. Mixed-use office/warehouse area, partial Northpoint City wings, combined-property revenue and Waterway Point/NEX stakeholder-share revenue are excluded. Unknown observations are not assigned arbitrary tiers or zero quality. A weighted sum is missing when malls exist nearby but none has that measurement; actual absence of any mall within the buffer has zero access. The 2019 retail-component areas for Funan/IMM/Raffles City are retained as dated measurements with newer mixed-use revenue omitted. Jurong Point NLA is approximate as disclosed. Mall levels are not used because physical, retail and carpark storey definitions differ.','',
      'Transport: an operational-date-filtered MRT graph, distinct operating MRT-line count, normalized harmonic graph centrality, graph hops to Raffles Place, same-name/nearby bus services and station entry/exit volume. [LTA TEL dates](https://www.lta.gov.sg/content/ltagov/en/upcoming_projects/rail_expansion/thomson_east_coast_line.html/1000). The inherited station table contains future stations marked open. Its original 77 features are preserved for control; only the new graph filters those records by verified operating dates. Historical CE1/CE2 branch links are used for source labels CC34/CC33, and the post-2026 Circle Line closure is not invented for 2021–2024. LRT topology is incomplete and excluded from the graph. This block combines an operating-station distance correction with quality/connectivity measures; without a distance-only ablation, any gain cannot be attributed solely to crowd volume or network centrality.','',
      'Bus connections use dated public [BusRouter SG / cheeaun archives](https://github.com/cheeaun/sgbusdata) derived from LTA. Same-name matching uses normalized station/stop names, stop/interchange descriptors and a 500 m distance guard; a separate 400 m proximity count covers connections without a matching name. Counts are unions of unique service IDs across stops and both directions, not frequencies or passenger counts. The audit lists matched stop names/codes and distances. Archive schema changes (additional road-name field) are handled explicitly. Underlying data attribution remains LTA/OneMap/OSM; the repository\'s software license is not treated as a blanket data license.','',
      'Ridership is from the public LTA-derived [station passenger archive](https://www.kaggle.com/datasets/yorkyong/singapore-passenger-volume-by-train-stations), containing August–October 2023 hourly entry/exit totals, uploaded 6 December 2023. Monthly totals are divided by calendar days and averaged across the three months. Values are available from January 2024 conservatively; missing-source indicators are retained. Daily tap volume is an activity proxy, not train occupancy, platform density or observed waiting time. Only 2.52% of the main fitting rows have dated ridership coverage, versus full holdout coverage, limiting what this experiment can establish about crowd-volume usefulness. The official LTA API key was not configured; no private key was requested or exposed.','',
      '## Information policy and limitations','',
      'Dated-source runs use the latest observation available at each context pseudo-origin and the fixed May 2024 query origin. Label-free January 2021 warmup rows without a target-history origin use December 2020 for external data. FCT fiscal reports enter in January of the next year; CICT calendar-year reports enter in May of the next year, conservatively after publication. There is no future-source backfill in dated mode. Underlying geocoding/location snapshots are inherited; a perfectly archived historical geography is not claimed.','',
      'The separately labelled combined_canonical_static_weights sensitivity fixes external measurements at May 2024 for all rows while retaining the contemporaneous operational rail graph. This explicitly retrospective representation tests source-coverage asymmetry; it is not evidence of point-in-time performance and must not replace the dated result. Historical-school archive revisions and incomplete mall reporting remain limitations in both modes.','',
      'Before full transport/combined scoring, a source audit exposed abbreviated stop names such as S\'goon, Botanic Gdns and 6th Ave. A separate reproducible correction expands these published abbreviations and changes only the same-name bus-service column, with all other 20 added columns asserted identical. Nineteen of 134 station counts change in May 2024, including Serangoon (37 services). The raw source files, original prepared artifact and initial earlier-fold diagnostic outputs are preserved. Primary transport and combined results below use canonical names; station_bus_match_audit_canonical_2024-05.csv and bus_stop_abbreviation_audit.csv contain the evidence. Remaining zero same-name matches are retained alongside the independent nearby-stop count.','',
      '## Earlier diagnostic results','',table(inner) if len(inner) else 'Pending.','',
      '## Complete 80/20 results','',table(scores),'',
      'Intervals in cluster_bootstrap.json use 5,000 paired resamples separately by building and month. This holdout has been inspected in previous project work; intervals are descriptive and not adjusted for repeated development/model selection. Grouped scores by town and flat type, all monthly results and per-transaction errors are saved alongside the report.','',
      '## Coverage and computation','',table(outer_coverage),'',
      table(pd.DataFrame(timings)[['model','added_dimensions','fit_seconds','predict_seconds','peak_vram_gib','context_rows']]) if timings else 'Outer inference pending.','',
      'GPU figures are peak PyTorch-allocated memory, not total driver reservation. All runs retain the same full context; none changes the original labels or residual baseline. Source hashes, parsed records, coordinate joins, source-quality limits and match audits are stored separately from model outputs.','',
      'Validation: 25 tests pass across the new source/feature cases and existing spatial/temporal safeguards (test_results.txt). Baseline replay agrees within 1e-5 SGD; all 77 control columns, targets, residual baselines and holdout row IDs are checked, and checkpoint/source/code/submission hashes are verified.','',
      'Reproduce: install requirements-amenity-quality.txt; run analysis/acquire_amenity_quality.py --stage all and --stage followups; then analysis/compare_amenity_quality.py --stage prepare, --stage smoke and --stage inner; run analysis/refine_amenity_bus_names.py --stage all for corrected earlier diagnostics and full outer evaluations; finally run analysis/summarize_amenity_quality.py. Each stage is a separate invocation. Source URLs are in source_manifest.json and bibliography.json. Existing baseline/submission outputs are preserved.']
    (out/'report.md').write_text('\n'.join(report)+'\n',encoding='utf8')
    replay=json.loads((out/'baseline_replay.json').read_text()); assert replay['verified'] and replay['max_abs_difference']<1e-5
    write_json(out/'verification.json',dict(source_hashes_verified=True,code_hashes_verified=True,checkpoint_unchanged=True,all_77_baseline_columns_unchanged=True,
                 targets_and_residual_baselines_unchanged=True,evaluation_rows_identical=True,predictions_finite=True,baseline_replay=replay,
                 canonical_bus_names_verified=True,refinement_sha256=refinement['sha256'],submission_preservation=preservation,
                 prediction_sha256={p.name:sha256(p) for p in out.glob('predictions_outer_*.csv')},completed_models=list(models)))
    expected={'schools_dated','malls_dated','transport_canonical_dated','combined_canonical_dated','combined_canonical_static_weights'}
    if expected<=set(models): write_json(out/'status.json',dict(status='complete',completed_models=list(models),report='report.md',limitations='source coverage and static sensitivity policies documented; no submission replaced'))
    print(scores.to_string(index=False),flush=True)


if __name__=='__main__': main()
