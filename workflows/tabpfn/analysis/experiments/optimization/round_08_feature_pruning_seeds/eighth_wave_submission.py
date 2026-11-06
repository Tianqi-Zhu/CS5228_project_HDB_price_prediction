"""Export the fixed FT no36 replacement after a real full-data refit."""
from __future__ import annotations
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from analysis.experiments.frontier_search import ROOT,write_json
from analysis.experiments.best_ensemble_submission import prepare,predict_saved,OUT as OLD
from analysis.experiments.canonical_final import validate_submission
from analysis.pipeline.data import sha256


def run(variant='single'):
    from analysis.experiments.sixth_wave_deep import seed,dependencies
    from analysis.experiments.third_wave_models import fit
    dependencies()
    destination=ROOT/'output/submissions/components'/('round_08_ft_seed_average' if variant=='mean' else 'round_08_ft_single_seed')
    destination.mkdir(parents=True,exist_ok=True)
    bundle,hashes=prepare()
    recipe={'name':'ft_no36','family':'ft_transformer','training':'all',
        'adjusted':True,'target':'residual','drop_ann36':True}
    predictions=[];seeds=(42,43,44,45) if variant=='mean' else (42,)
    for seed_value in seeds:
        checkpoint=destination/f'ft_seed{seed_value}.joblib'
        if checkpoint.exists():
            payload=joblib.load(checkpoint)
            assert payload['source_sha256']==hashes and payload['iterations']==3
            part=predict_saved(payload,bundle,'models')
        else:
            seed(seed_value)
            payload,part,count,_,_=fit(recipe|{'seed':seed_value},bundle,3)
            payload.update(source_sha256=hashes,iterations=count)
            joblib.dump(payload,checkpoint,compress=1)
            replay=predict_saved(joblib.load(checkpoint),bundle,'models')
            np.testing.assert_allclose(part,replay,atol=1e-6,rtol=0)
        predictions.append(part)
        del payload
        import gc,torch
        gc.collect();torch.cuda.empty_cache()
        print('FULL FT REFIT',seed_value,'complete',flush=True)
    prediction=np.mean(predictions,axis=0)
    np.save(destination/'ft_predictions.npy',prediction)
    old=pd.read_csv(OLD/'submission.csv')
    old_macro=pd.read_csv(OLD/'submission_given_macro.csv')
    specification=json.loads((ROOT/'output/experiment_runs/00_report_and_final/best_ensemble.json').read_text(encoding='utf-8'))
    ft_member=next(m for m in specification['leaf_members'] if m['artifact_name']=='ft_transformer')
    old_ft=np.load(OLD/'models'/ft_member['name']/'predictions.npy')
    weight=ft_member['weight']
    delta=weight*(prediction-old_ft)
    macro_weight=json.loads((ROOT/'output/experiment_runs/round_04_macro_spatial_diagnostics/given_macro_selection.json').read_text(encoding='utf-8'))['versions']['conservative']['weights']['previous_best']
    outputs=[];template=pd.read_csv(ROOT/'data/example-submission.csv')
    local=pd.read_csv(ROOT/'output/experiment_runs/round_08_feature_pruning_seeds/comparison.csv').set_index('name')
    suffix='_mean' if variant=='mean' else ''
    local_history='ft_seedmean_replacement' if variant=='mean' else 'ft_replacement'
    local_macro='ft_seedmean_given_macro' if variant=='mean' else 'ft_replacement_given_macro'
    for name,values,local_name in [(f'kaggle_history_ft_no36{suffix}.csv',old.Predicted.to_numpy()+delta,local_history),
        (f'kaggle_macro_ft_no36{suffix}.csv',old_macro.Predicted.to_numpy()+macro_weight*delta,local_macro)]:
        result=validate_submission(template,values)
        path=destination/name;result.to_csv(path,index=False)
        reread=pd.read_csv(path)
        np.testing.assert_array_equal(reread.Id,np.arange(50000))
        np.testing.assert_allclose(reread.Predicted,values,atol=1e-8,rtol=0)
        outputs.append({'path':str(path.relative_to(ROOT)).replace('\\','/'),'sha256':sha256(path),
            'local_recent_rmse':float(local.loc[local_name,'june']),
            'local_annual_rmse':float(local.loc[local_name,'outer'])})
    for path,digest in hashes.items():assert sha256(ROOT/path)==digest
    summary={'training_rows':len(bundle['y_fit']),'test_rows':len(prediction),'recipe':recipe,
        'epochs':3,'seeds':seeds,'FT_weight':weight,'source_sha256':hashes,'outputs':outputs,
        'actual_kaggle_score':None,'uploaded':False,
        'note':'Local scores are chronological development refits. Full-data test predictions have no observed RMSE. Original submissions are preserved.'}
    write_json(destination/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=['single','mean'],default='single')
    run(parser.parse_args().variant)
