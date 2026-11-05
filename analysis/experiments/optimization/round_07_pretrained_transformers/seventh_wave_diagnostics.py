"""Describe error bias and source/query history support; no calibrated predictions."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from analysis.experiments.frontier_search import ROOT,write_json,score
from analysis.experiments.third_wave_training import make_data
from analysis.experiments.seventh_wave_transformers import OUT,context_indices
from analysis.experiments.second_wave_report import matched_file


def main():
    support=[];bias=[]
    for stage in ('screen','outer','june'):
        bundle=make_data(stage)
        sampled=context_indices(bundle,{'family':'nori15','context':2048})
        for lag in (12,24,36):
            name=f'ANN_BLOCK_{lag}_R1_N'
            source=bundle['ann_fit'][name].to_numpy(float)>0
            query=bundle['ann_val'][name].to_numpy(float)>0
            support.append({'stage':stage,'lag_months':lag,'source_rows':len(source),'query_rows':len(query),
                'source_supported_fraction':float(source.mean()),'query_supported_fraction':float(query.mean()),
                'nori2048_context_supported_fraction':float(source[sampled].mean())})
        if stage=='screen':
            continue
        candidates=[('history_ensemble',ROOT/f'output/models/round_03_tree_transformer_ensemble/ablations/{stage}/models/predictions.csv'),
            ('old_ft_transformer',ROOT/f'output/models/round_03_tree_transformer_ensemble/models/{stage}/ft_transformer/predictions.csv')]
        selection=json.loads((ROOT/'output/experiment_runs/round_06_deep_models/ensemble_selection.json').read_text())
        for family,name in selection['representatives'].items():
            candidates.append((family,ROOT/f'output/models/round_06_deep_models/{stage}/{name}/predictions.csv'))
        for name,path in candidates:
            frame=matched_file(path)
            np.testing.assert_array_equal(frame.source_row,bundle['row_val'])
            error=frame.prediction.to_numpy(float)-frame.actual.to_numpy(float)
            mean=float(error.mean());variance=float(np.mean((error-mean)**2));mse=float(np.mean(error**2))
            np.testing.assert_allclose(mse,variance+mean**2,atol=1e-8,rtol=0)
            bias.append({'stage':stage,'model':name,**score(frame.actual,frame.prediction),
                'squared_bias':mean**2,'centered_error_variance':variance,
                'bias_fraction_of_mse':mean**2/mse,'centered_error_std_diagnostic_only':float(np.sqrt(variance))})
    pd.DataFrame(support).to_csv(OUT/'history_support_shift.csv',index=False)
    pd.DataFrame(bias).to_csv(OUT/'bias_variance_diagnostic.csv',index=False)
    summary={'support':support,'error_decomposition':bias,
        'warning':'Centered error standard deviation uses validation labels only to describe errors. It is not a valid calibrated RMSE, submitted prediction, or target-achievement score.',
        'causal_limit':'History support differences, finite contexts, and mean bias are observations; they do not by themselves identify a unique causal explanation or prove one architecture inferior.'}
    write_json(OUT/'diagnostics.json',summary)
    print(json.dumps({'june_bias':[row for row in bias if row['stage']=='june'],'support':support},indent=2),flush=True)


if __name__=='__main__':
    main()
