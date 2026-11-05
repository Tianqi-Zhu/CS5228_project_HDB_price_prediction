"""Replay source/cache integrity, unchanged baseline inputs and feature cutoffs."""
from __future__ import annotations
import gc
from importlib.metadata import version
import json
import platform
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from analysis.data import sha256
from analysis.compare_local_economy_landscape import write_json
import analysis.run_targeted_spatial_economic as core
from analysis.run_targeted_housing_supply import check_sources,supply_features


def main():
    check_sources();cases=[]
    frame=core.raw_frame();v=pd.read_csv(core.OUT/'housing_supply_vintages.csv')
    assert v.groupby('year').completed.sum().to_dict()=={2019:15748,2021:8234,2022:13627,2023:23782}
    supply_audit=[]
    for origin in core.ORIGINS:
        base,bw=core.ensemble('A0',origin)
        original_weights=pd.read_csv(core.REFERENCE/'learned_weights.csv')
        w=original_weights[(original_weights.origin==origin)&(original_weights.model=='optimized')].iloc[0]
        np.testing.assert_allclose(bw,[w.baseline_weight,w.spatial_weight,w.temporal_weight],atol=1e-12,rtol=0)
        for member in core.MEMBERS:
            old=core.original_bundle(origin,member)
            assert old['X_train'].index.is_unique and old['X_query'].index.is_unique
            assert set(old['X_train'].index).isdisjoint(old['X_query'].index)
            assert frame.loc[old['X_train'].index,'month'].max()<pd.Period(origin,'M')
            for variant in ['A1','B1']:
                b=core.load_prepared(origin,f'{variant}_{member}')
                for split in ['train','query']:
                    x,q=old[f'X_{split}'],b[f'X_{split}']
                    pd.testing.assert_frame_equal(x,q[list(x)])
                    np.testing.assert_array_equal(old[f'y_{split}'],b[f'y_{split}'])
                    np.testing.assert_array_equal(old[f'{split}_baseline'],b[f'{split}_baseline'])
                    months,cutoffs=core.row_cutoffs(x)
                    valid=cutoffs.notna().to_numpy()
                    assert (pd.PeriodIndex(cutoffs[valid],freq='M').asi8<months[valid].asi8).all()
                    if variant=='B1':
                        expected=supply_features(frame.loc[x.index],cutoffs,v)
                        np.testing.assert_allclose(q[list(expected)],expected,equal_nan=True,rtol=0,atol=0)
                        supply_audit.append(dict(origin=origin,member=member,split=split,rows=len(q),
                            unsupported=int(q.SUPPLY_UNAVAILABLE.sum()),
                            minimum_known_age_months=float(q.SUPPLY_OBSERVATION_AGE_MONTHS.min())))
                path=core.OUT/f'predictions_{origin}_{variant}_{member}.csv'
                meta=json.loads((core.OUT/f'timing_{origin}_{variant}_{member}.json').read_text())
                original_meta=json.loads((core.REFERENCE/f'timing_{origin}_{member}.json').read_text())
                assert sha256(path)==meta['prediction_sha256']
                assert meta['weights_digest_before']==meta['weights_digest_after']
                assert meta['weights_digest_before']==original_meta['weights_digest_before']
                assert meta['parameters_unchanged'] and meta['gradients_prohibited'] and meta['no_adapters']
                assert meta['n_estimators']==2 and meta['random_state']==42 and not meta['subsampling']
                assert meta['native_default_estimator_batching']
                cases.append(dict(origin=origin,member=member,variant=variant,features=len(b['X_train'].columns),
                                  context_rows=len(b['X_train']),query_rows=len(b['X_query']),verified=True))
                del b;gc.collect()
            del old;gc.collect()
    history=pd.read_csv(core.OUT/'A2_residual_history.csv')
    assert history.row_id.is_unique and len(history)==4000
    assert (history.context_last_month<history.month).all()
    selection=json.loads((core.OUT/'A2_selection.json').read_text())
    assert selection['maximum_selection_label_month']=='2023-12'
    assert not selection['main_evaluation_labels_used']
    for c in ['month','context_last_month']:history[c]=pd.PeriodIndex(history[c],freq='M')
    residual_predictions=pd.read_csv(core.OUT/'A2_predictions.csv',float_precision='round_trip')
    for origin in core.ORIGINS:
        base,_=core.ensemble('A0',origin)
        query=frame.loc[base.row_id].drop(columns='MONTHLY_RENT')
        r,_=core.residual_smoother(history,query,pd.Period(origin,'M')-2,shrink=selection['shrink'])
        expected=base.prediction.to_numpy()+selection['gamma']*r
        actual=residual_predictions[residual_predictions.origin==origin]
        np.testing.assert_array_equal(actual.row_id,base.row_id)
        np.testing.assert_allclose(actual.A2,expected,rtol=0,atol=1e-8)
    replay=pd.read_csv(core.OUT/'history_cutoffs.csv')
    populated=replay.source_rentals>0
    assert (replay.loc[populated,'latest_source_month']<=replay.loc[populated,'cutoff']).all()
    assert (replay.cutoff<replay.target_min_month).all()
    pd.DataFrame(cases).to_csv(core.OUT/'baseline_feature_equivalence_audit.csv',index=False)
    pd.DataFrame(supply_audit).to_csv(core.OUT/'supply_availability_coverage.csv',index=False)
    write_json(core.OUT/'independent_audit.json',dict(verified=True,inference_cases=len(cases),
        original_inputs_targets_baselines_rows_and_categories_unchanged=True,
        baseline_ensemble_weights_match_existing_all_origins=True,
        all_appended_feature_cutoffs_precede_target_months=True,
        source_submission_checkpoint_and_prediction_hashes_verified=True,
        archived_supply_totals_and_feature_replay_verified=True,
        historical_residual_rows=4000,in_sample_or_later_residuals_used=False,
        audit_code_sha256=sha256(Path(__file__))))
    write_json(core.OUT/'environment.json',dict(python=platform.python_version(),platform=platform.platform(),
        gpu='NVIDIA GeForce RTX4080 (16GiB)',packages={p:version(p) for p in ['numpy','pandas','scipy','torch','tabpfn','PyMuPDF']}))
    # Finalize reporting/provenance only; no feature, prediction or selection
    # artifact changes. The retrieved official CNA video is the verified timing
    # citation, replacing a provisional auxiliary link in the source notes.
    sources_path=core.OUT/'housing_supply_sources.json'
    source_notes=json.loads(sources_path.read_text())
    source_notes['timing_sources']=[u for u in source_notes['timing_sources'] if '/watch/hdb-records-' not in u]
    official_video='https://www.youtube.com/watch?v=goZ1m5R9r0M'
    if official_video not in source_notes['timing_sources']:source_notes['timing_sources'].append(official_video)
    write_json(sources_path,source_notes)
    status_path=core.OUT/'experiment_status.csv';statuses=pd.read_csv(status_path)
    row=statuses.index[statuses.experiment=='A3/B3/other combinations'][0]
    line=lambda r:'| '+' | '.join(str(r[k]) for k in ['experiment','status','reason'])+' |'
    old_line=line(statuses.loc[row])
    statuses.loc[row,'status']='not run: component gates failed'
    statuses.loc[row,'reason']='A1, B1 and A2 fail the earlier/later consistency gate; B2 has no admitted historical vintages'
    statuses.to_csv(status_path,index=False)
    report_path=core.OUT/'report.md';report=report_path.read_text(encoding='utf-8').replace(old_line,line(statuses.loc[row]))
    decision='**Decision: retain the existing optimized frozen ensemble.** All three completed additions worsen later RMSE; no combination is promoted. Quarterly-cycle and workforce-demand hypotheses remain untested because admissible historical vintages were not established.'
    if decision not in report:report=report.replace('\n\n','\n\n'+decision+'\n\n',1)
    if '## Independent verification' not in report:
        report+='\n## Independent verification\n\nAll18 model/origin cases pass exact original-input/target/baseline/row checks, original ensemble-weight replay, source/prediction hashes and pretrained-weight digest checks. Added supply fields and residual corrections are replayed from their permitted historical cutoffs.\n\nThe inherited static amenity/geography dating limitations of the frozen baseline remain; this is a matched incremental experiment, not a claim that the entire original pipeline has reconstructed historical data vintages. The FY2018/19 supply pilot used an incorrect column order, was invalidated and quarantined, and all accepted B1 inference was rerun with the printed15,748 completion total verified.\n\nAfter the report-generation command, finalize checks and reporting with `.venv/Scripts/python.exe analysis/audit_targeted_spatial_economic.py`. This last step also records package versions and corrects the source-timing citation to the [official CNA video published31October2023](https://www.youtube.com/watch?v=goZ1m5R9r0M).\n'
    report_path.write_text(report,encoding='utf-8')
    print('Independent audit passed:',len(cases),'model/origin cases; all inputs, labels, baseline weights, feature cutoffs, frozen parameters and source hashes verified',flush=True)


if __name__=='__main__':main()
