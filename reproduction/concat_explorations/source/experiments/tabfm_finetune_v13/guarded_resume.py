"""Offline guard and recovery entry point for the immutable registered screen.

Run only after the training process has stopped. A torn best/resume transaction
is rejected instead of silently using inconsistent state.
"""
import hashlib, json, traceback
from pathlib import Path
from experiments.tabfm_finetune_v13 import runner
from experiments.tabfm_finetune_v13.data import load_banks
import numpy as np
import torch

def canonical_metadata(value):
    return json.loads(json.dumps(value,sort_keys=True))

def file_sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def pre_register_banks():
    metadata={}
    for name in runner.CONFIG['banks']:
        _,_,data=load_banks(expanded=name!='base',stock=name=='stock')
        metadata[name]=canonical_metadata(data)
    path=runner.OUT/'all_banks_metadata.json'
    if path.exists() and json.loads(path.read_text())!=metadata: raise ValueError('all-bank metadata mismatch')
    if not path.exists(): runner.atomic_json(path,metadata)
    return metadata

def check_sources():
    reg=json.loads((runner.OUT/'registration.json').read_text())
    for key,path in [('runner_sha256','experiments/tabfm_finetune_v13/runner.py'),('core_sha256','experiments/tabfm_finetune_v13/core.py'),('data_sha256','experiments/tabfm_finetune_v13/data.py'),('runtime_model_sha256','reports/tabfm_v10/runtime/source/tabfm/src/pytorch/model.py')]:
        if file_sha(runner.ROOT/path)!=reg[key]: raise ValueError('source mismatch '+path)
    for path,digest in reg['supporting_source_sha256'].items():
        if file_sha(runner.ROOT/path)!=digest: raise ValueError('source mismatch '+path)
    weight=runner.ROOT/'reports/tabfm_v10/weights/regression/model.safetensors'
    if weight.stat().st_size!=reg['weights_size']: raise ValueError('weight size mismatch')
    with weight.open('rb') as f: h=hashlib.sha256(f.read(1048576)).hexdigest()
    if h!=reg['weights_first_mib_sha256']: raise ValueError('weight header mismatch')
    guard=json.loads((runner.ROOT/'reports/tabfm_finetune_v13/root_original_guard.json').read_text())
    for path,digest in guard.items():
        if file_sha(runner.ROOT/path)!=digest: raise ValueError('original guard mismatch '+path)

def check_best_transaction(path):
    resume=path/'resume.pt'
    if not resume.exists(): return
    state=torch.load(resume,map_location='cpu',weights_only=False)
    p=np.load(path/'best_predictions.npz'); score=float(np.sqrt(np.mean((p['predicted_sgd']-p['actual_sgd'])**2)))
    if not np.isclose(score,state['best_rmse_sgd'],rtol=0,atol=1e-9): raise ValueError('torn best/resume transaction; do not resume automatically')
    if not np.isclose(min(x['rmse_sgd'] for x in state['curve']),score,rtol=0,atol=1e-9): raise ValueError('best/curve mismatch')
    if not (path/'best_parameters.pt').exists(): raise ValueError('best parameter state missing')

def main():
    torch.set_num_threads(2); check_sources(); pre_register_banks()
    from experiments.tabfm_finetune_v13.replay import replay
    for bank in runner.CONFIG['banks']:
        check_best_transaction(runner.OUT/bank)
        if (runner.OUT/bank/'resume.pt').exists(): replay(bank)
    def canonical_load(*args,**kwargs):
        fit,hold,meta=load_banks(*args,**kwargs)
        return fit,hold,canonical_metadata(meta)
    runner.load_banks=canonical_load
    try:
        for bank in runner.CONFIG['banks']:
            runner.atomic_json(runner.OUT/'queue_status.json',{'status':'running','bank':bank,'final_verification':'pending','resumed':True})
            runner.run_bank(bank)
        runner.atomic_json(runner.OUT/'queue_status.json',{'status':'training_complete','final_verification':'pending'})
    except BaseException as exc:
        runner.atomic_json(runner.OUT/'queue_status.json',{'status':'FAILED','error':repr(exc),'traceback':traceback.format_exc(),'final_verification':'pending'}); raise

if __name__=='__main__': main()
