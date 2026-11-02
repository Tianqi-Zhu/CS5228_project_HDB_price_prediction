"""Independently replay saved corrections and add data-dependent conclusions."""
from __future__ import annotations
import json
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from analysis.compare_tabpfn_spatial_temporal_residual import OUT,CACHE,ROOT,REFERENCE,FOLDS,save_frame
from analysis.data import sha256
from analysis.evaluation import metrics
from analysis.experiments.spatial_temporal_residual import residual_features,corrected
from analysis.compare_local_economy_landscape import write_json
from analysis.summarize_local_economy_landscape import table


def main():
    selection=json.loads((OUT/'selection.json').read_text())
    baseline=pd.read_csv(REFERENCE,float_precision='round_trip')
    saved=pd.read_csv(OUT/'predictions.csv',float_precision='round_trip')
    np.testing.assert_array_equal(saved.row_id,baseline.row_id)
    np.testing.assert_array_equal(saved.B0,baseline.predicted)
    data=joblib.load(CACHE/'outer_features.joblib')
    x=residual_features(data['bundle']['X_query'],baseline.predicted.to_numpy(),data['market'])
    scores=pd.read_csv(OUT/'development_holdout_results.csv').set_index('model')
    split_usage=[]
    for name in ['E1','E2']:
        record=selection['models'][name]
        path=CACHE/f'{name}_final.cbm'; assert sha256(path)==record['final_model_sha256']
        model=CatBoostRegressor(); model.load_model(str(path))
        json_path=CACHE/f'{name}_final_structure.json'
        model.save_model(str(json_path),format='json')
        structure=json.loads(json_path.read_text())
        float_names={f['feature_index']:f['feature_id'] for f in structure['features_info']['float_features']}
        for tree_index,tree in enumerate(structure['oblivious_trees']):
            for split in tree.get('splits') or []:
                split_usage.append(dict(model=name,tree=tree_index,split_type=split['split_type'],
                    feature=float_names[split['float_feature_index']] if split['split_type']=='FloatFeature' else str(split)))
        delta=model.predict(x[record['columns']])
        np.testing.assert_array_equal(delta,saved[f'{name}_predicted_residual'])
        np.testing.assert_array_equal(corrected(saved.B0,delta,record['strength']),saved[name])
        np.testing.assert_array_equal(corrected(saved.B0,delta,0),saved.B0)
        score=metrics(saved.actual,saved[name])
        for metric,value in score.items(): assert abs(value-scores.loc[name,metric])<1e-10
    for start,_ in FOLDS:
        record=json.loads((OUT/f'oof_provenance_{start}.json').read_text())
        assert sha256(CACHE/f'adapter_{start}.pt')==record['adapter_sha256']
        assert sha256(CACHE/f'features_{start}.joblib')==record['sha256']
        assert sha256(OUT/f'oof_predictions_{start}.csv')==record['prediction_sha256']
    settings=json.loads((OUT/'settings.json').read_text())
    for path,value in settings['source_sha256'].items(): assert sha256(ROOT/path)==value
    intervals=json.loads((OUT/'cluster_bootstrap.json').read_text())
    validations=pd.read_csv(OUT/'validation_results.csv').set_index('model')
    chosen=selection['selected']; selected=scores.loc[chosen]
    e2_skill=validations.loc['E2','selected_correction_zero_baseline_r2']
    lo,hi=intervals['E2']['month']['rmse_delta_interval_95']
    summary=pd.read_csv(OUT/'model_comparison.csv')[['model','validation_rmse_sgd','strength','rmse_sgd','mae_sgd','delta_rmse_sgd']]
    summary=summary.rename(columns={'rmse_sgd':'development_holdout_rmse_sgd','mae_sgd':'development_holdout_mae_sgd'})
    save_frame(summary,OUT/'comparison_summary.csv')
    save_frame(pd.DataFrame(split_usage),OUT/'selected_tree_splits.csv')
    market_names={'LOCAL_HIST_MEDIAN','LOCAL_GROWTH_3M','LOCAL_COUNT_6M'}
    market_splits=[r for r in split_usage if r['model']=='E2' and r['feature'] in market_names]
    tests=(OUT/'test_results.txt').read_text(encoding='utf8')
    count=int(re.search(r'Ran (\d+) tests',tests).group(1)); assert tests.rstrip().endswith('OK')
    importance=pd.read_csv(OUT/'feature_importance.csv')
    market=importance.loc[(importance.model=='E2')&importance.feature.isin(['LOCAL_HIST_MEDIAN','LOCAL_GROWTH_3M','LOCAL_COUNT_6M'])]
    block=['<!-- result-summary-start -->','## Observed conclusions','',table(summary),'',
        f'1. Residual predictability is weak: E2 unshrunk validation zero-baseline R2 is {validations.loc["E2","residual_zero_baseline_r2"]:.6f}; lambda=0.5 produces skill {e2_skill:.6f} (about {e2_skill*100:.3f}% reduction in squared error). Both early-stopping probes select only four trees, rather than using the800-tree cap. This is the selected conservative result, not an execution failure.','',
        f'2. Basic correction does not improve the later development period: E1 increases RMSE by S${scores.loc["E1","delta_rmse_sgd"]:.3f} and MAE by S${scores.loc["E1","mae_sgd"]-scores.loc["B0","mae_sgd"]:.3f}.','',
        f'3. Spatial-temporal features improve validation RMSE over E1 by only S${validations.loc["E1","rmse_sgd"]-validations.loc["E2","rmse_sgd"]:.3f}; E2 is S${scores.loc["E2","rmse_sgd"]-scores.loc["E1","rmse_sgd"]:.3f} worse than E1 on the later holdout. Their final-model importance sum is {market.importance.sum():.3f}%, with {len(market_splits)} selected splits on these three features (selected_tree_splits.csv). The small validation difference therefore provides no direct evidence that the model learned useful historical-market effects; adding candidate columns can alter tree fitting even when those columns receive no final splits.','',
        f'4. The earlier-selected {chosen} does not confirm a gain: RMSE changes by S${selected.delta_rmse_sgd:+.3f}. E2 month-cluster descriptive95% interval is [{lo:+.3f}, {hi:+.3f}] SGD. Its underprediction bias increases from S${scores.loc["B0","bias_sgd"]:.3f} to S${scores.loc["E2","bias_sgd"]:.3f}; the learned downward adjustment transfers poorly to the later period.','',
        '5. Retain B0. The tested correction adds complexity without a supported performance benefit. Generalization to the unlabelled competition test cannot be assessed; neither models nor existing submission are replaced.','',
        f'Validation: {count} tests passed. Independent saved-model replay reproduces every E1/E2 holdout residual and corrected prediction bit for bit, and lambda0 reproduces B0 exactly. Fold adapter/feature/prediction hashes and all protected source hashes are verified. See summary_verification.json.','',
        'To regenerate this concise conclusion after the main runner, execute `.venv/Scripts/python.exe analysis/summarize_tabpfn_spatial_temporal_residual.py`.','<!-- result-summary-end -->','']
    path=OUT/'report.md'; text=path.read_text(encoding='utf8')
    text=re.sub(r'<!-- result-summary-start -->.*?<!-- result-summary-end -->\n*','',text,flags=re.S)
    insertion=text.index('## Scope and baseline ambiguity')
    path.write_text(text[:insertion]+'\n'.join(block)+'\n'+text[insertion:],encoding='utf8')
    write_json(OUT/'summary_verification.json',dict(saved_model_replay_bit_exact=True,lambda_zero_bit_exact=True,
        all_fold_artifact_hashes_verified=True,all_protected_source_hashes_verified=True,tests_passed=count,
        e2_market_feature_split_count=len(market_splits),
        summarizer_sha256=sha256(Path(__file__)),report_sha256=sha256(path)))
    print(summary.to_string(index=False))


if __name__=='__main__': main()
