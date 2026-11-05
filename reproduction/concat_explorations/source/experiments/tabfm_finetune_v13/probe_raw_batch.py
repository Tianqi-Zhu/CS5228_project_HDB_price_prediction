"""Measure real selection-stock frozen prefix on context64/query32,64,128."""
import json,resource,time,numpy as np
from experiments.tabfm_finetune_v13.runner import ROOT,torch
from experiments.tabfm_finetune_v13.data_full import prepare_phase,episode
from experiments.tabfm_finetune_v13.pipeline_raw import inputs
from experiments.tabfm_finetune_v13.core import prepare_lora_head,frozen_prefix
from tabfm import tabfm_v1_0_0_pytorch

torch.set_num_threads(2); torch.manual_seed(42)
f,v,_,_=prepare_phase('selection','stock'); m=tabfm_v1_0_0_pytorch.load(model_type='regression',checkpoint_path=str(ROOT/'reports/tabfm_v10/weights/regression'),device='cpu',dtype=torch.bfloat16,use_cache=False); prepare_lora_head(m,4); m.icl_predictor.tf_icl.blocks[-1].float()
origin=v.origin[0]; context=np.flatnonzero(f.target_month<=origin)[:64]; results=[]
for size in (32,64,128):
    ep=episode(f,v,np.arange(size),context); x,y,n,c,d=inputs(ep); t=time.perf_counter(); cap=frozen_prefix(m,x,y,n,c,d); elapsed=time.perf_counter()-t
    results.append({'context':64,'queries':size,'features':188,'prefix_seconds':elapsed,'seconds_per_query':elapsed/size,'peak_rss_bytes_macos':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}); del cap
    print(results[-1],flush=True)
(ROOT/'reports/tabfm_finetune_v13/probe_raw_batch.json').write_text(json.dumps(results,indent=2)+'\n')
