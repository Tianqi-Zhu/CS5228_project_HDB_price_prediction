"""Refit every base learner from train/auxiliary with frozen recipes/weights.

Requires the training Conda environment and an NVIDIA CUDA GPU. There are no
cached trained checkpoints in this branch. Fixed fusion weights were learned
in the recorded historical OOT experiments; this entry point refits base
models, not the entire development search.
"""
from pathlib import Path
import csv
import gc
import hashlib
import json
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent/"runtime"))
import joblib
import numpy as np
import pandas as pd
import torch
from reproduce import HERE,compose
from analysis.experiments.best_ensemble_submission import prepare
from analysis.experiments.sixth_wave_deep import dependencies,seed


def clean():gc.collect();torch.cuda.empty_cache()


def main():
    if not torch.cuda.is_available():raise RuntimeError('Full training needs a CUDA GPU; use reproduce.py for CPU replay.')
    dependencies();torch.set_num_threads(4)
    recipe=json.loads((HERE/'training_recipes.json').read_text(encoding='utf-8'))
    started=time.perf_counter();bundle,hashes=prepare()
    normalize=lambda items:{name.replace('\\','/'):value for name,value in items.items()}
    assert normalize(hashes)==normalize(recipe['source_sha256']),'Training inputs differ from the recorded competition files.'
    out=ROOT/'output/review_f1_trained';out.mkdir(parents=True,exist_ok=True)
    names=[];predictions=[];models=[]
    def save(name,payload,prediction,rounds):
        directory=out/'models'/name;directory.mkdir(parents=True,exist_ok=True)
        joblib.dump(payload,directory/'model.joblib',compress=1)
        np.save(directory/'predictions.npy',prediction)
        models.append({'name':name,'rounds_or_epochs':int(rounds),'training_rows':len(bundle['y_fit'])})
        print('TRAINED',name,rounds,flush=True)
    for item in recipe['history_members']:
        if item['kind']=='normalized_market':
            from analysis.experiments.second_wave_normalized import fit
            model,prediction,count=fit(item['spec'],bundle,item['iterations'])
            payload={'model':model,'spec':item['spec']}
        elif item['kind']=='anniversary':
            from analysis.experiments.second_wave_cohort_extensions import fit
            model,prediction,count,_=fit(item['spec'],bundle,item['iterations'])
            payload={'model':model,'spec':item['spec']}
        else:
            from analysis.experiments.third_wave_models import fit
            payload,prediction,count,_,_=fit(item['spec'],bundle,item['iterations'])
        names.append(item['name']);predictions.append(np.asarray(prediction,dtype=float))
        save(item['name'],payload,prediction,count)
        del payload
        if 'model' in locals():del model
        clean()
    seed_predictions=[]
    from analysis.experiments.third_wave_models import fit as fit_ft
    for seed_value in recipe['no36']['seeds']:
        seed(seed_value)
        spec=recipe['no36']['spec']|{'seed':seed_value}
        payload,prediction,count,_,_=fit_ft(spec,bundle,recipe['no36']['epochs'])
        save(f'ft_no36_seed{seed_value}',payload,prediction,count)
        seed_predictions.append(prediction);del payload;clean()
    names.append('ft_no36_seedmean');predictions.append(np.mean(seed_predictions,axis=0))
    from analysis.experiments.tenth_wave_structures import fit as fit_plr
    payload,prediction,count=fit_plr(recipe['plr']['spec'],bundle,recipe['plr']['epochs'])
    save('ft_plr_128_3',payload,prediction,count)
    names.append('tenth/ft_plr_128_3');predictions.append(prediction);del payload;clean()
    generated=compose(np.column_stack(predictions),names,
        bundle['ann_val'].ANN_BLOCK_24_R1_N.to_numpy()>0,bundle['ann_val'].ANN_BLOCK_36_R1_N.to_numpy()>0)
    records=[]
    for name,prediction in generated.items():
        assert prediction.shape==(50000,) and np.isfinite(prediction).all()
        path=out/name
        with path.open('w',newline='',encoding='utf-8') as stream:
            writer=csv.writer(stream,lineterminator='\n');writer.writerow(['Id','Predicted'])
            writer.writerows((i,float(value)) for i,value in enumerate(prediction))
        records.append({'file':str(path.relative_to(ROOT)).replace('\\','/'),'rows':50000,
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    report={'mode':'Fresh base-model training with frozen selected hyperparameters and learned fusion weights',
        'trained_model_count':len(models),'models':models,'outputs':records,'seconds':time.perf_counter()-started,
        'source_sha256':hashes,'actual_kaggle_scores_for_these_new_files':None,
        'note':'The screenshot scores belong to the original F1 submission. GPU refits can vary; fresh outputs have no observed Kaggle score.'}
    (out/'training_run.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
