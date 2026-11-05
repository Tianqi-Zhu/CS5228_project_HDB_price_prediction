"""Reconstruct best cached-prefix monitor predictions from saved adapter state."""
import json, sys
from pathlib import Path
from experiments.tabfm_finetune_v13.runner import ROOT, OUT, materialize, episodes, select, predict, preprocess
from experiments.tabfm_finetune_v13.data import load_banks
from experiments.tabfm_finetune_v13.core import prepare_lora_head, cached_forward
import numpy as np
import torch
from tabfm import tabfm_v1_0_0_pytorch

def replay(name):
    torch.set_num_threads(2); fit,hold,meta=load_banks(expanded=name!='base',stock=name=='stock')
    _,ids=select(fit,meta); lookup={int(v):i for i,v in enumerate(hold.row_id)}; idx=np.array([lookup[int(i)] for i in ids]); eps=episodes(fit,hold,idx,44)
    actual={int(k):float(v) for k,v in zip(hold.row_id,hold.actual_sgd)}
    for ep in eps: ep['query_actual_sgd']=np.array([actual[int(k)] for k in ep['query_row_id']])
    model=tabfm_v1_0_0_pytorch.load(model_type='regression',checkpoint_path=str(ROOT/'reports/tabfm_v10/weights/regression'),device='cpu',dtype=torch.bfloat16,use_cache=False); prepare_lora_head(model,4)
    params=dict(model.named_parameters()); state=torch.load(OUT/name/'best_parameters.pt',map_location='cpu',weights_only=True)
    with torch.no_grad():
        for n,v in state.items(): params[n].copy_(v)
    import hashlib
    fingerprint=hashlib.sha256((OUT/'registration.json').read_bytes()+json.dumps(meta,sort_keys=True).encode()).hexdigest()
    cache=materialize(model,eps,OUT/name,'validation',fingerprint); metric,p,a,ids=predict(model,cache)
    ep=eps[0]; x=torch.from_numpy(preprocess(ep))[None]; y=torch.from_numpy(ep['y'])[None]; mask=torch.from_numpy(ep['categorical_mask'])[None]; ts=torch.tensor([ep['train_size']]); d=torch.tensor([x.shape[-1]])
    with torch.no_grad():
        raw=model(x,y,ts,cat_mask=mask,d=d); cached=cached_forward(model,cache[0][0])
    if not torch.equal(raw,cached): raise ValueError('raw versus cache replay mismatch')
    saved=np.load(OUT/name/'best_predictions.npz'); assert np.array_equal(ids,saved['row_id']) and np.array_equal(p,saved['predicted_sgd']) and np.array_equal(a,saved['actual_sgd'])
    return metric

if __name__=='__main__': print(json.dumps(replay(sys.argv[1]),indent=2))
