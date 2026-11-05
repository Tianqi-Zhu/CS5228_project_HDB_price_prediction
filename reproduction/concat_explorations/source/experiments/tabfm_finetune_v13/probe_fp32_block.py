"""Time actual cached last-block training in BF16 versus frozen FP32 base."""
import json,time
from experiments.tabfm_finetune_v13.runner import ROOT,OUT,torch
from experiments.tabfm_finetune_v13.core import prepare_lora_head,cached_forward
from tabfm import tabfm_v1_0_0_pytorch

def run():
    torch.set_num_threads(2); torch.manual_seed(42)
    m=tabfm_v1_0_0_pytorch.load(model_type='regression',checkpoint_path=str(ROOT/'reports/tabfm_v10/weights/regression'),device='cpu',dtype=torch.bfloat16,use_cache=False); prepare_lora_head(m,4)
    path=OUT/'base/prefix_cache/train/0000.pt'; saved=torch.load(path,map_location='cpu',weights_only=False); cap=saved['capture']
    results={}
    for dtype in ('bfloat16','float32'):
        if dtype=='float32':
            m.icl_predictor.tf_icl.blocks[-1].float(); cap={'args':tuple(v.float() if torch.is_tensor(v) and v.is_floating_point() else v for v in cap['args']),'kwargs':cap['kwargs']}
        params=[p for p in m.parameters() if p.requires_grad]; optim=torch.optim.AdamW(params,lr=1e-4); times=[]
        for step in range(4):
            t=time.perf_counter(); optim.zero_grad(set_to_none=True); p=cached_forward(m,cap); loss=p[:,64:].square().mean(); loss.backward(); optim.step(); dt=time.perf_counter()-t
            if step: times.append(dt)
        results[dtype]={'warmup_excluded_seconds':times,'mean_seconds':sum(times)/len(times)}
    path=ROOT/'reports/tabfm_finetune_v13/probe_fp32_block.json'; path.write_text(json.dumps(results,indent=2)+'\n'); print(json.dumps(results,indent=2),flush=True)
if __name__=='__main__': run()
