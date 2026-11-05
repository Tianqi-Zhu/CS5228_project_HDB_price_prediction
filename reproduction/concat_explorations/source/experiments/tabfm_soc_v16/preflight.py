"""Mandatory real-checkpoint local-device preflight, no external downloads."""
import argparse,copy,json,time,traceback
from pathlib import Path
import numpy as np
import torch
from experiments.tabfm_soc_v16 import pipeline as p
from experiments.tabfm_soc_v16.runtime import resolve_device,run_lock,capture_rng,restore_rng,move_tree,device_identity
from experiments.tabfm_finetune_v13.core import cached_forward

def sync():
    if p.DEVICE.type=='cuda': torch.cuda.synchronize(p.DEVICE)

def run(episodes):
    if episodes<2: raise ValueError('preflight requires at least two episodes')
    p.IN_PREFLIGHT=True; torch.set_num_threads(2)
    train,_,metadata,fingerprint=p.setup('selection'); model=p.model_init(); params={n:v for n,v in model.named_parameters() if v.requires_grad}
    optim=torch.optim.AdamW(params.values(),lr=p.CONFIG['learning_rates'][0],weight_decay=p.CONFIG['weight_decay']); schedule=p.make_schedule(optim,len(train))
    cold=next(e for e in train if e['train_size']==0); ordinary=next(e for e in train if e['train_size']>0)
    nonempty=[e for e in train if e['train_size']>0]
    selected=[cold,ordinary]+[nonempty[i] for i in np.linspace(0,len(nonempty)-1,episodes-2,dtype=int)]
    block=model.icl_predictor.tf_icl.blocks[-1]
    handle=block.register_forward_pre_hook(lambda mod,args:tuple(v.float() if torch.is_tensor(v) and v.is_floating_point() else v for v in args))
    try:
        x,y,n,c,d=p.inputs(ordinary); sync(); started=time.perf_counter()
        with torch.no_grad(): raw=model(x,y,n,cat_mask=c,d=d)
        sync(); direct_seconds=time.perf_counter()-started
        if not torch.isfinite(raw).all(): raise ValueError('nonfinite raw forward')
    finally: handle.remove()
    before={n:v.detach().clone() for n,v in params.items()}; prefix_times=[]; update_times=[]; total=0
    for i,ep in enumerate(selected):
        x,y,n,c,d=p.inputs(ep); sync(); start=time.perf_counter(); cap=p.frozen_prefix(model,x,y,n,c,d); sync(); prefix_times.append(time.perf_counter()-start)
        if i<2: p.raw_parity_own(model,ep,cap)
        sync(); start=time.perf_counter(); optim.zero_grad(set_to_none=True); prediction=cached_forward(model,p.cast_cap(cap))[0,ep['train_size']:,0].float(); target=torch.from_numpy((ep['query_targets']-ep['target_mean'])/ep['target_std']).float().to(p.DEVICE); loss=(prediction-target).square().mean()
        if not torch.isfinite(loss): raise ValueError('nonfinite preflight loss')
        loss.backward(); grads=[v.grad for v in params.values() if v.grad is not None]
        if not grads or any(not torch.isfinite(g).all() for g in grads): raise ValueError('nonfinite gradients')
        if not any(v.grad is not None and v.grad.abs().sum()>0 for n,v in params.items() if 'decoder' in n): raise ValueError('head gradient missing')
        torch.nn.utils.clip_grad_norm_(params.values(),p.CONFIG['gradient_clip'],error_if_nonfinite=True); optim.step(); schedule.step(); sync(); update_times.append(time.perf_counter()-start)
        if i<2: p.raw_parity_own(model,ep,cap)
        total+=len(ep['query_row_id'])
    changed=[n for n,v in params.items() if not torch.equal(v,before[n])]
    if not any('decoder' in n for n in changed) or not all(any(n.endswith(part) for n in changed) for part in ('q_proj.b','v_proj.b')): raise ValueError('query/value adapter or head did not update')
    rng=capture_rng(p.DEVICE); expected=torch.randn(4,device=p.DEVICE); restore_rng(rng,p.DEVICE); actual=torch.randn(4,device=p.DEVICE)
    if not torch.equal(expected,actual): raise ValueError('RNG replay failed')
    opt_cpu=move_tree(optim.state_dict(),torch.device('cpu')); sched_state=schedule.state_dict()
    checkpoint=p.OUT/'resume_preflight.pt'; p.save_atomic(checkpoint,{'parameters':{n:v.detach().clone() for n,v in params.items()},'optimizer':opt_cpu,'scheduler':sched_state,'rng':capture_rng(p.DEVICE)})
    restored=torch.load(checkpoint,map_location='cpu',weights_only=False)
    optim.load_state_dict(copy.deepcopy(restored['optimizer'])); schedule.load_state_dict(restored['scheduler']); restore_rng(restored['rng'],p.DEVICE)
    if schedule.state_dict()!=sched_state or any(not torch.equal(params[n],v.to(p.DEVICE)) for n,v in restored['parameters'].items()): raise ValueError('preflight checkpoint replay mismatch')
    x,y,n,c,d=p.inputs(ordinary); cap=p.frozen_prefix(model,x,y,n,c,d)
    target=torch.from_numpy((ordinary['query_targets']-ordinary['target_mean'])/ordinary['target_std']).float().to(p.DEVICE)
    def advance():
        optim.zero_grad(set_to_none=True); output=cached_forward(model,p.cast_cap(cap))[0,ordinary['train_size']:,0].float(); ((output-target).square().mean()).backward()
        torch.nn.utils.clip_grad_norm_(params.values(),p.CONFIG['gradient_clip'],error_if_nonfinite=True); optim.step(); schedule.step()
        return {n:v.detach().clone() for n,v in params.items()},torch.randn(4,device=p.DEVICE)
    first,draw1=advance()
    with torch.no_grad():
        for n,v in params.items(): v.copy_(restored['parameters'][n])
    optim.load_state_dict(copy.deepcopy(restored['optimizer'])); schedule.load_state_dict(restored['scheduler']); restore_rng(restored['rng'],p.DEVICE)
    second,draw2=advance()
    if any(not torch.equal(first[n],second[n]) for n in first) or not torch.equal(draw1,draw2): raise ValueError('optimizer/scheduler/RNG next-step replay failed')
    report={'status':'passed','fingerprint':fingerprint,'device':device_identity(p.DEVICE),'episodes':episodes,'queries':total,'direct_raw_seconds_one_context_episode':direct_seconds,'prefix_seconds':prefix_times,'cached_update_seconds':update_times,'peak_allocated_bytes':torch.cuda.max_memory_allocated(p.DEVICE) if p.DEVICE.type=='cuda' else None,'peak_reserved_bytes':torch.cuda.max_memory_reserved(p.DEVICE) if p.DEVICE.type=='cuda' else None,'cold_start_finite':True,'raw_cache_before_after_updates':'exact','adapter_head_updates':'passed','optimizer_scheduler_rng_checkpoint':'passed','changed_parameter_names':changed}
    p.runner.atomic_json(p.OUT/'preflight.json',report); print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--device',choices=['cuda','cpu'],required=True); parser.add_argument('--output-dir',required=True); parser.add_argument('--checkpoint-dir',required=True); parser.add_argument('--episodes',type=int,default=20); args=parser.parse_args()
    p.OUT=Path(args.output_dir).resolve(); p.CHECKPOINT=Path(args.checkpoint_dir).resolve()
    with run_lock(p.OUT):
        p.runner.atomic_json(p.OUT/'preflight.json',{'status':'running'})
        try:
            p.DEVICE=resolve_device(args.device)
            run(args.episodes)
        except Exception as e:
            p.runner.atomic_json(p.OUT/'preflight.json',{'status':'FAILED','error':repr(e),'traceback':traceback.format_exc()}); raise
