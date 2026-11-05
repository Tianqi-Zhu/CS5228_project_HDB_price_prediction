"""Reload all freshly trained checkpoints and compare with their saved forecasts."""
from pathlib import Path
import gc
import json
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent/"runtime"))
import joblib
import numpy as np
import pandas as pd
import torch
from analysis.experiments.sixth_wave_deep import dependencies
from analysis.experiments.best_ensemble_submission import prepare,predict_saved
from analysis.experiments.tenth_wave_structures import predict_saved as predict_plr
from analysis.experiments.fourth_wave_known_macro import macro_table


def main():
    dependencies();torch.set_num_threads(4)
    out=ROOT/'output/review_three_submissions_trained'
    record=json.loads((out/'training_run.json').read_text(encoding='utf-8'))
    assert record['trained_model_count']==16
    bundle,hashes=prepare()
    normalize=lambda items:{name.replace('\\','/'):value for name,value in items.items()}
    assert normalize(hashes)==normalize(record['source_sha256'])
    table=macro_table()
    for suffix in ('fit','val'):
        raw=bundle['raw_'+suffix]
        bundle['macro_'+suffix]=pd.DataFrame(table.reindex(raw.RENT_APPROVAL_DATE).to_numpy(np.float32),index=raw.index,columns=table.columns)
    checks=[]
    for item in record['models']:
        name=item['name'];directory=out/'models'/name;payload=joblib.load(directory/'model.joblib')
        if name=='ft_plr_128_3':prediction=predict_plr(payload,bundle)
        else:
            kind='normalized_market' if '/normalized_market/' in name else 'anniversary' if '/anniversary/' in name else 'known_macro' if name.startswith('given_macro/') else 'models'
            prediction=predict_saved(payload,bundle,kind)
        expected=np.load(directory/'predictions.npy')
        np.testing.assert_allclose(prediction,expected,atol=1e-5,rtol=0)
        checks.append({'name':name,'rows':len(expected),'max_abs_difference':float(np.max(np.abs(prediction-expected)))})
        print('RELOADED',name,checks[-1]['max_abs_difference'],flush=True)
        del payload;gc.collect();torch.cuda.empty_cache()
    for item in record['outputs']:
        frame=pd.read_csv(ROOT/item['file'])
        assert len(frame)==50000 and list(frame)==['Id','Predicted']
        np.testing.assert_array_equal(frame.Id,np.arange(50000));assert np.isfinite(frame.Predicted).all()
    result={'model_checks':checks,'model_count':len(checks),'output_csv_count':3,'all_passed':True}
    (out/'checkpoint_replay.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
