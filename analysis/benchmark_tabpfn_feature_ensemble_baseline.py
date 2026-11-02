"""Rebenchmark cached E0 under the final allocator policy; never change E0."""
from __future__ import annotations
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from analysis.compare_tabpfn_feature_ensemble import OUT,CACHE,infer,load_bundle,check_sources
from analysis.compare_local_economy_landscape import immutable,write_json
from analysis.data import sha256


def main():
    check_sources(); name='baseline_rebenchmark'; origin='2024-06'
    immutable(OUT/'baseline_benchmark_plan.json',dict(code_sha256=sha256(Path(__file__)),
        purpose='Full E0 timing under same final allocator policy as new members; predictions and locked weights remain unchanged',
        seed=42,n_estimators=2,batch_size=200,tolerance_sgd=1e-5))
    path=CACHE/f'features_{origin}_{name}.joblib'; manifest=OUT/f'features_{origin}_{name}.json'
    if not path.exists():
        joblib.dump(load_bundle(origin),path,compress=3)
        write_json(manifest,dict(sha256=sha256(path),member=name,features=77,seed=42))
    infer(origin,name)
    old=pd.read_csv(OUT/f'predictions_{origin}_baseline.csv',float_precision='round_trip')
    fresh=pd.read_csv(OUT/f'predictions_{origin}_{name}.csv',float_precision='round_trip')
    for c in ['row_id','month','actual','weight']: np.testing.assert_array_equal(old[c],fresh[c])
    difference=float(np.max(np.abs(old.predicted-fresh.predicted)))
    np.testing.assert_allclose(old.predicted,fresh.predicted,atol=1e-5,rtol=0)
    timing_path=OUT/f'timing_{origin}_baseline.json'; original=json.loads(timing_path.read_text())
    immutable(OUT/'timing_main_baseline_original_cached.json',original)
    fresh_meta=json.loads((OUT/f'timing_{origin}_{name}.json').read_text())
    for key in ['fit_seconds','predict_seconds','peak_vram_gib']: original[key]=fresh_meta[key]
    original['timing_rebenchmarked_under_final_allocator_policy']=True
    original['full_prediction_replay_max_difference_sgd']=difference
    write_json(timing_path,original)
    control_path=OUT/f'timing_{origin}_standard6.json'; control=json.loads(control_path.read_text())
    parts=[json.loads((OUT/f'timing_{origin}_{m}.json').read_text()) for m in ['baseline','seed43','seed44']]
    for field in ['fit_seconds','predict_seconds']: control[field]=sum(t[field] for t in parts)
    control['peak_vram_gib']=max(t['peak_vram_gib'] for t in parts)
    control['baseline_timing_rebenchmarked']=True; write_json(control_path,control)
    write_json(OUT/'baseline_benchmark_replay.json',dict(verified=True,rows=len(old),maximum_difference_sgd=difference,
        baseline_predictions_unchanged=True,feature_ensemble_weights_unchanged=True,fit_seconds=fresh_meta['fit_seconds'],predict_seconds=fresh_meta['predict_seconds']))
    check_sources(); print('FULL BASELINE TIMING REPLAY PASSED',difference,flush=True)


if __name__=='__main__': main()
