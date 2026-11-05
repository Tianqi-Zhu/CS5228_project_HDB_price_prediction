"""Real checkpoint zero-context, masked-query and FP32 last block probe."""
import json,time
from experiments.tabfm_finetune_v13.runner import ROOT,torch
from experiments.tabfm_finetune_v13.core import prepare_lora_head,frozen_prefix,cached_forward
from experiments.tabfm_finetune_v13.full_runner import cast_cap
from tabfm import tabfm_v1_0_0_pytorch

torch.manual_seed(42); torch.set_num_threads(2)
m=tabfm_v1_0_0_pytorch.load(model_type='regression',checkpoint_path=str(ROOT/'reports/tabfm_v10/weights/regression'),device='cpu',dtype=torch.bfloat16,use_cache=False); prepare_lora_head(m,4); block=m.icl_predictor.tf_icl.blocks[-1]; block.float()
x=torch.randn(1,2,188); x[:,:,-4:]=torch.randint(0,3,(1,2,4)).float(); y=torch.zeros(1,2); n=torch.tensor([0]); cats=torch.zeros(1,188,dtype=torch.bool); cats[:,-4:]=True; d=torch.tensor([188]); target=torch.tensor([[.2,-.4]])
t=time.perf_counter(); cap=frozen_prefix(m,x,y,n,cats,d); prefix=time.perf_counter()-t
handle=block.register_forward_pre_hook(lambda m,args: tuple(a.float() if torch.is_tensor(a) and a.is_floating_point() else a for a in args))
try:
    with torch.no_grad(): raw=m(x,y,n,cat_mask=cats,d=d); cached=cached_forward(m,cast_cap(cap))
finally: handle.remove()
assert torch.equal(raw,cached) and torch.isfinite(raw).all()
params=[p for p in m.parameters() if p.requires_grad]; opt=torch.optim.AdamW(params,lr=1e-4); out=cached_forward(m,cast_cap(cap)); loss=(out[:,:,0]-target).square().mean(); loss.backward()
head=[p for p in m.icl_predictor.decoder.parameters()]; assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head); assert any(p.grad.abs().sum()>0 for p in head)
old=[p.detach().clone() for p in head]; opt.step(); assert any(not torch.equal(a,b) for a,b in zip(old,head))
result={'context':0,'query':2,'features':188,'query_labels_masked':True,'forward_finite':True,'fp32_block_raw_cache_exact':True,'head_gradients_finite_nonzero':True,'head_weights_updated':True,'loss':loss.item(),'prefix_seconds':prefix}
(ROOT/'reports/tabfm_finetune_v13/probe_empty_context.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2),flush=True)
