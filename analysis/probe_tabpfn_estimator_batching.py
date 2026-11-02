"""Isolate the old sequential-estimator environment override on plain TabPFN."""
import os
os.environ['TABPFN_MAX_BATCHED_ESTIMATOR_ROWS']='0'
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from unittest.mock import patch
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
from tabpfn.settings import settings
from analysis.compare_tabpfn_feature_ensemble import OUT,PRIOR,load_bundle,weight_digest,CHECKPOINT
from analysis.compare_local_economy_landscape import write_json


def main():
    torch.set_num_threads(1); b=load_bundle('2024-06'); x=b['X_train']; q=b['X_query'].iloc[:400]
    assert settings.tabpfn.max_batched_estimator_rows==0
    model=TabPFNRegressor.create_default_for_version(ModelVersion.V3_5_FAST,model_path=str(CHECKPOINT),device='cuda',n_estimators=2,
        random_state=42,fit_mode='fit_with_cache',keep_cache_on_device=True,n_preprocessing_jobs=1,
        categorical_features_indices=[i for i,c in enumerate(x) if isinstance(x[c].dtype,pd.CategoricalDtype)],inference_config={'SUBSAMPLE_SAMPLES':None})
    def forbidden(*a,**k): raise AssertionError('No training allowed')
    with patch('torch.Tensor.backward',forbidden),patch('torch.autograd.backward',forbidden),patch('torch.autograd.grad',forbidden):
        model.fit(x,b['y_train']-b['train_baseline'])
        for m in model.models_:
            for p in m.parameters(): p.requires_grad_(False)
        before=weight_digest(model); parts=[]
        for i in [0,200]:
            torch.cuda.empty_cache(); parts.append(model.predict(q.iloc[i:i+200],output_type='mean')+b['query_baseline'][i:i+200])
        predicted=np.concatenate(parts); assert before==weight_digest(model)
    sequential=pd.read_csv(PRIOR/'baseline_predictions.csv',float_precision='round_trip').E0.to_numpy()[:400]
    native=pd.read_csv(OUT/'predictions_2024-06_baseline.csv',float_precision='round_trip').predicted.to_numpy()[:400]
    err=float(np.max(np.abs(predicted-sequential))); np.testing.assert_allclose(predicted,sequential,atol=1e-5,rtol=0)
    write_json(OUT/'estimator_batching_isolation.json',dict(verified=True,rows=400,sequential_override=0,
        sequential_probe_vs_original_sequential_max_error_sgd=err,sequential_vs_current_native_max_difference_sgd=float(np.max(np.abs(predicted-native))),
        checkpoint_weights_unchanged=True,only_estimator_batching_override_changed=True,
        conclusion='Original sequential cached baseline reproduced by plain class with rows0 override. All final feature/seed/baseline comparisons use matched native-default batching.'))
    print('BATCHING ISOLATED; sequential replay max error',err,flush=True)


if __name__=='__main__': main()
