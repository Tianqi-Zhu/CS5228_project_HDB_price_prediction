"""Independent row/weight/provenance checks for the ABCDE ensemble experiment."""
from __future__ import annotations
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from analysis.data import sha256
from analysis.compare_local_economy_landscape import write_json
from analysis.experiments.feature_diversified_ensemble import combine_predictions
import analysis.compare_tabpfn_abcde as run


def main():
    run.check_sources()
    settings=json.loads((run.OUT/'settings.json').read_text())
    choice=json.loads((run.OUT/'selection.json').read_text())
    assert not choice['main_labels_used'] and choice['nested_weight_fit_max_month']=='2023-06' and choice['weight_fit_max_month']=='2024-04'
    for name,digest in choice['input_predictions'].items():assert sha256(run.ROOT/name)==digest
    predictions=pd.read_csv(run.OUT/'predictions_all_origins.csv',float_precision='round_trip')
    weights=pd.read_csv(run.OUT/'weights.csv')
    assert len(predictions)==32995
    assert not predictions.duplicated(['origin','row_id']).any()
    assert np.isfinite(predictions[run.MEMBERS]).all().all()
    for origin in run.ORIGINS:
        actual=predictions[predictions.origin==origin].copy()
        for letter in run.MEMBERS:
            source=pd.read_csv(run.prediction_path(origin,letter),float_precision='round_trip')
            for c in ['row_id','month','actual']:np.testing.assert_array_equal(source[c],actual[c])
            np.testing.assert_array_equal(source.predicted,actual[letter])
            if 'weight' in source:np.testing.assert_array_equal(source.weight,actual.weight)
            if 'origin' in source:assert source.origin.eq(origin).all()
        for group in run.GROUPS:
            row=weights[(weights.origin==origin)&(weights.group==group)].iloc[0]
            w=row[run.MEMBERS].to_numpy(float)
            assert np.isclose(w.sum(),1) and (w>=0).all()
            assert all(w[run.MEMBERS.index(m)]==0 for m in run.MEMBERS if m not in group)
            expected=combine_predictions(actual[run.MEMBERS].to_numpy(),w)
            np.testing.assert_allclose(expected,actual['A_single' if group=='A' else group],rtol=0,atol=1e-8)
        if origin!='2024-06':
            provenance=json.loads((run.OUT/f'oof_provenance_{origin}.json').read_text())
            assert provenance['context_max_month']<actual.month.min()
            assert not provenance['validation_targets_used'] and not provenance['checkpoint_selection']
            assert provenance['frozen_parameters_unchanged'] and provenance['prediction_parameter_versions_unchanged']
            assert sha256(run.CACHE/f'adapter_{origin}.pt')==provenance['adapter_sha256']
    # Check that earlier-only family selection uses the declared nested scores,
    # rather than the much better observed later ABDE/equal-weight scores.
    selected=min(choice['nested_results'],key=lambda r:(r['rmse_sgd'],r['active_members'],run.GROUPS.index(r['group'])))['group']
    assert selected==choice['selected']=='CDE'
    scores=pd.read_csv(run.OUT/'scores.csv')
    for _,row in scores.iterrows():
        f=predictions[predictions.origin==row.origin]
        y=f.actual.to_numpy();p=f[row.model].to_numpy();w=f.weight.to_numpy()
        np.testing.assert_allclose(row.rmse_sgd,np.sqrt(np.average((p-y)**2,weights=w)),rtol=0,atol=1e-9)
        np.testing.assert_allclose(row.mae_sgd,np.average(abs(p-y),weights=w),rtol=0,atol=1e-9)
    original=pd.read_csv(run.FROZEN/'learned_weights.csv')
    row=original[(original.origin=='2024-06')&(original.model=='optimized')].iloc[0]
    later=predictions[predictions.origin=='2024-06']
    expected=later[['B','D','E']].to_numpy()@np.array([row.baseline_weight,row.temporal_weight,row.spatial_weight])
    np.testing.assert_allclose(expected,later.BDE_existing,rtol=0,atol=1e-8)
    report=run.OUT/'report.md'
    text=report.read_text(encoding='utf-8')
    if 'Independent audit:' not in text:
        text+='\nIndependent audit: `.venv/Scripts/python.exe analysis/audit_tabpfn_abcde.py` verifies all32,995 saved cases,source predictions,population weights,convex weights and score reconstruction; both historical A adapters predate their targets. Existing BDE and both protected submissions reproduce their recorded hashes.\n'
        report.write_text(text,encoding='utf-8')
    write_json(run.OUT/'independent_audit.json',dict(verified=True,rows=32995,later_rows=28995,
        member_predictions_and_population_weights_exact=True,original_BDE_reproduced=True,
        historical_A_query_labels_excluded_from_adapter_fit_and_selection=True,
        all_scores_recomputed=True,weights_earlier_only=True,selected=selected,
        no_main_period_oracle_weights=True,submissions_unchanged=True,
        reporting_code_sha256=sha256(run.ROOT/'analysis/summarize_tabpfn_abcde.py'),audit_code_sha256=sha256(Path(__file__))))
    print('ABCDE independent audit passed:',len(predictions),'aligned cases; selected',selected,flush=True)


if __name__=='__main__':main()
