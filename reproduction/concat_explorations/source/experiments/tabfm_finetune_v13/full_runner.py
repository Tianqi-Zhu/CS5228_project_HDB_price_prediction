"""Full causally eligible fit + full holdout LoRA training on local CPU.

All execution and cache artifacts live in reports/tabfm_finetune_v13/full.
First epoch creates prefix caches lazily, so optimizer updates begin immediately.
"""
import hashlib,json,os,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from experiments.tabfm_finetune_v13 import runner
from experiments.tabfm_finetune_v13.data import load_banks
from experiments.tabfm_finetune_v13.core import prepare_lora_head,frozen_prefix,cached_forward
from tabfm import tabfm_v1_0_0_pytorch

ROOT=runner.ROOT; OUT=ROOT/'reports/tabfm_finetune_v13/full'
CONFIG={'scope':'all 49146 causally eligible fit queries, full 35882 holdout','bank':'stock188','seed':42,'rank':4,'context':64,'query_batch':32,'cpu_threads':2,'last_block_dtype':'float32','frozen_prefix_dtype':'bfloat16','optimizer':'AdamW','lr':1e-4,'weight_decay':1e-4,'clip':1.,'max_full_epochs':40,'patience_full_validation_epochs':5,'loss':'per-episode mean normalized residual MSE; episodes weighted equally','numeric_preprocessing':'context-only mean/std, floor 1e-6, clip [-10,10]','validation':'all 35882 holdout rows after each full training epoch','context_policy':'seeded random legal fit labels, exclude all query ids','cache':'lazy per-episode frozen prefix','checkpoint_every_steps':10}

def atomic_json(path,value): runner.atomic_json(path,value)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()
def save_atomic(path,state):
    temp=path.with_suffix('.tmp'); torch.save(state,temp); os.replace(temp,path)
def cast_cap(cap):
    return {'args':tuple(x.float() if torch.is_tensor(x) and x.is_floating_point() else x for x in cap['args']),'kwargs':cap['kwargs']}
def to_args(ep):
    x=torch.from_numpy(runner.preprocess(ep))[None]; y=torch.from_numpy(ep['y'])[None]
    return x,y,torch.tensor([ep['train_size']]),torch.from_numpy(ep['categorical_mask'])[None],torch.tensor([x.shape[-1]])
def get_cache(model,ep,path,fingerprint):
    h=hashlib.sha256(fingerprint.encode())
    for field in ('x','y','categorical_mask','query_row_id','context_row_id'):
        a=np.ascontiguousarray(ep[field]); h.update(a.tobytes()); h.update(str(a.shape).encode())
    digest=h.hexdigest()
    if path.exists():
        data=torch.load(path,map_location='cpu',weights_only=False)
        if data['fingerprint']!=digest: raise ValueError('cached episode fingerprint mismatch')
        return data['capture']
    x,y,n,c,d=to_args(ep); cap=frozen_prefix(model,x,y,n,c,d)
    path.parent.mkdir(parents=True,exist_ok=True); save_atomic(path,{'fingerprint':digest,'capture':cap}); return cap

def initialize():
    torch.set_num_threads(2); torch.manual_seed(CONFIG['seed'])
    fit,hold,meta=load_banks(expanded=True,stock=True)
    idx=np.flatnonzero(fit.origin>fit.origin.min()); assert len(idx)==49146 and len(hold.row_id)==35882
    val_idx=np.arange(len(hold.row_id)); train=runner.episodes(fit,fit,idx,43); valid=runner.episodes(fit,hold,val_idx,44)
    assert sum(len(x['query_row_id']) for x in train)==len(idx)
    assert len(np.unique(np.concatenate([x['query_row_id'] for x in train])))==len(idx)
    for episodes,pop in ((train,fit),(valid,hold)):
        lookup={int(k):float(v) for k,v in zip(pop.row_id,pop.actual_sgd)}
        for ep in episodes: ep['query_actual_sgd']=np.array([lookup[int(k)] for k in ep['query_row_id']])
    OUT.mkdir(parents=True,exist_ok=True)
    weight=ROOT/'reports/tabfm_v10/weights/regression/model.safetensors'
    paths=['experiments/tabfm_finetune_v13/full_runner.py','experiments/tabfm_finetune_v13/runner.py','experiments/tabfm_finetune_v13/core.py','experiments/tabfm_finetune_v13/data.py','reports/tabfm_v10/runtime/source/tabfm/src/pytorch/model.py','reports/tabfm_v10/runtime/source/tabfm/src/pytorch/tabfm_v1_0_0.py','experiments/tabfm_v10/benchmark.py','experiments/macro_encoders_v3/contract.py','experiments/macro_encoders_v3/data.py','experiments/macro_encoders_v3/scaling.py']
    full_weight_sha=sha(weight)
    expected=json.loads((ROOT/'reports/tabfm_finetune_v13/root_base_weight.json').read_text())['sha256']
    if full_weight_sha!=expected: raise ValueError('full checkpoint SHA mismatch')
    reg={'config':CONFIG,'sources':{p:sha(ROOT/p) for p in paths},'metadata':meta,'torch_version':torch.__version__,'numpy_version':np.__version__,'original_guard_sha256':sha(ROOT/'reports/tabfm_finetune_v13/root_original_guard.json'),'weight_size':weight.stat().st_size,'weight_sha256':full_weight_sha,'sampling':'every eligible fit row; all holdout rows','fit_row_ids_sha256':hashlib.sha256(fit.row_id[idx].tobytes()).hexdigest(),'holdout_row_ids_sha256':hashlib.sha256(hold.row_id[val_idx].tobytes()).hexdigest(),'fit_origin_range':[int(fit.origin[idx].min()),int(fit.origin[idx].max())],'holdout_origin_range':[int(hold.origin.min()),int(hold.origin.max())],'last_block_conversion':'frozen BF16 values cast exactly to FP32; compared only against same FP32 baseline'}
    encoded=json.dumps(reg,sort_keys=True,default=str).encode(); fingerprint=hashlib.sha256(encoded).hexdigest()
    regpath=OUT/'registration.json'
    if regpath.exists():
        old=json.loads(regpath.read_text())
        if old['fingerprint']!=fingerprint: raise ValueError('full training fingerprint differs')
    else: atomic_json(regpath,{'fingerprint':fingerprint,'registration':reg})
    selection=OUT/'selection.npz'
    if not selection.exists(): np.savez_compressed(selection,train_row_id=fit.row_id[idx],train_origin=fit.origin[idx],train_horizon=fit.horizon[idx],holdout_row_id=hold.row_id[val_idx],holdout_origin=hold.origin[val_idx],holdout_horizon=hold.horizon[val_idx],train_context=np.stack([e['context_row_id'] for e in train]),validation_context=np.stack([e['context_row_id'] for e in valid]))
    model=tabfm_v1_0_0_pytorch.load(model_type='regression',checkpoint_path=str(ROOT/'reports/tabfm_v10/weights/regression'),device='cpu',dtype=torch.bfloat16,use_cache=False)
    prepare_lora_head(model,4); model.icl_predictor.tf_icl.blocks[-1].float()
    return fit,hold,train,valid,model,fingerprint

def raw_parity(model,ep,cap):
    block=model.icl_predictor.tf_icl.blocks[-1]
    def hook(module,args): return tuple(v.float() if torch.is_tensor(v) and v.is_floating_point() else v for v in args)
    handle=block.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            x,y,n,c,d=to_args(ep); raw=model(x,y,n,cat_mask=c,d=d); cached=cached_forward(model,cast_cap(cap))
            if not torch.equal(raw,cached): raise ValueError('FP32 last-block raw/cache mismatch')
    finally: handle.remove()

def predictions(model,eps,fingerprint,params,baseline):
    """Full validation: baseline and candidate share each cached prefix and IDs."""
    current={n:p.detach().clone() for n,p in params.items()}; predictions=[]; frozen=[]; actual=[]; ids=[]
    try:
        for i,ep in enumerate(eps):
            cap=cast_cap(get_cache(model,ep,OUT/'cache'/'validation'/f'{i:05}.pt',fingerprint))
            with torch.no_grad():
                out=cached_forward(model,cap)[0,ep['train_size']:,0].float().numpy()
                for n,p in params.items(): p.copy_(baseline[n])
                old=cached_forward(model,cap)[0,ep['train_size']:,0].float().numpy()
                for n,p in params.items(): p.copy_(current[n])
            predictions.extend(((out*ep['target_std']+ep['target_mean']+ep['anchor_ksgd'])*1000).tolist()); frozen.extend(((old*ep['target_std']+ep['target_mean']+ep['anchor_ksgd'])*1000).tolist()); actual.extend(ep['query_actual_sgd'].tolist()); ids.extend(ep['query_row_id'].tolist())
            if i%10==0: atomic_json(OUT/'progress.json',{'stage':'full_validation','episodes_completed':i+1,'episodes_total':len(eps)})
    finally:
        with torch.no_grad():
            for n,p in params.items(): p.copy_(current[n])
    return np.array(predictions),np.array(frozen),np.array(actual),np.array(ids)
def rmse(p,a): return float(np.sqrt(np.mean((p-a)**2)))

def main():
    fit,hold,train,valid,model,fingerprint=initialize(); params={n:p for n,p in model.named_parameters() if p.requires_grad}; opt=torch.optim.AdamW(params.values(),lr=CONFIG['lr'],weight_decay=CONFIG['weight_decay'])
    initial={n:p.detach().clone() for n,p in params.items()}; epoch=1; start=0; best=float('inf'); bad=0; curve=[]; bestpack=None
    resume=OUT/'resume.pt'
    if resume.exists():
        state=torch.load(resume,map_location='cpu',weights_only=False)
        if state['fingerprint']!=fingerprint: raise ValueError('resume fingerprint differs')
        with torch.no_grad():
            for n,p in params.items(): p.copy_(state['params'][n])
        opt.load_state_dict(state['optimizer']); initial=state['initial']; epoch=state['epoch']; start=state['next_step']; best=state['best']; bad=state['bad']; curve=state['curve']; bestpack=state['bestpack']; torch.set_rng_state(state['rng'])
    def checkpoint(ep,step):
        save_atomic(resume,{'fingerprint':fingerprint,'params':{n:p.detach().clone() for n,p in params.items()},'initial':initial,'optimizer':opt.state_dict(),'epoch':ep,'next_step':step,'best':best,'bad':bad,'curve':curve,'bestpack':bestpack,'rng':torch.get_rng_state()})
    if (epoch>CONFIG['max_full_epochs'] or bad>=CONFIG['patience_full_validation_epochs']) and bestpack is not None:
        save_atomic(OUT/'best_parameters.pt',bestpack['params'])
        np.savez_compressed(OUT/'best_predictions.npz',row_id=bestpack['row_id'],predicted_sgd=bestpack['predicted_sgd'],actual_sgd=bestpack['actual_sgd'])
        np.savez_compressed(OUT/'baseline_predictions.npz',row_id=bestpack['row_id'],predicted_sgd=bestpack['baseline_predictions'],actual_sgd=bestpack['actual_sgd'])
        atomic_json(OUT/'curve.json',curve); atomic_json(OUT/'status.json',{'status':'training_complete','full_epochs_completed':epoch-1,'best_rmse_sgd':best,'final_verification':'pending'}); return
    checkpoint(epoch,start); atomic_json(OUT/'status.json',{'status':'running','stage':'training','epoch':epoch,'next_step':start,'final_verification':'pending'})
    for epoch in range(epoch,CONFIG['max_full_epochs']+1):
        perm=np.random.default_rng(CONFIG['seed']+epoch).permutation(len(train))
        for step in range(start,len(perm)):
            i=int(perm[step]); ep=train[i]; t=time.perf_counter(); cap=get_cache(model,ep,OUT/'cache'/'train'/f'{i:05}.pt',fingerprint)
            if epoch==1 and step==0: raw_parity(model,ep,cap)
            opt.zero_grad(set_to_none=True); out=cached_forward(model,cast_cap(cap))[0,ep['train_size']:,0].float(); target=torch.from_numpy((ep['query_targets']-ep['target_mean'])/ep['target_std']).float(); loss=(out-target).square().mean()
            if not torch.isfinite(loss): raise ValueError('nonfinite training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(params.values(),CONFIG['clip'],error_if_nonfinite=True); opt.step()
            if epoch==1 and step==0:
                if not any(p.detach().abs().sum()>0 for n,p in params.items() if n.endswith('.b')): raise ValueError('LoRA did not update')
                raw_parity(model,ep,cap)
            if step%CONFIG['checkpoint_every_steps']==0 or step==len(perm)-1: checkpoint(epoch,step+1)
            atomic_json(OUT/'progress.json',{'stage':'training','epoch':epoch,'steps_completed':step+1,'steps_total':len(perm),'query_rows_in_step':len(ep['query_row_id']),'last_seconds':time.perf_counter()-t,'loss':loss.item()})
            print('train',epoch,step+1,'/',len(perm),'seconds',time.perf_counter()-t,flush=True)
        start=0; p,b,a,ids=predictions(model,valid,fingerprint,params,initial); assert len(a)==35882 and len(np.unique(ids))==35882
        base_score=rmse(b,a); score=rmse(p,a)
        if not curve:
            best=base_score; bestpack={'epoch':0,'params':initial,'row_id':ids,'predicted_sgd':b,'actual_sgd':a,'baseline_predictions':b}; curve.append({'epoch':0,'rmse_sgd':base_score})
        curve.append({'epoch':epoch,'rmse_sgd':score}); print('full_validation',curve[-1],flush=True)
        if score<best:
            best=score; bad=0; bestpack={'epoch':epoch,'params':{n:p.detach().clone() for n,p in params.items()},'row_id':ids,'predicted_sgd':p,'actual_sgd':a,'baseline_predictions':b}
        else: bad+=1
        checkpoint(epoch+1,0)  # authoritative atomic commit includes best parameters and predictions
        save_atomic(OUT/'best_parameters.pt',bestpack['params'])
        np.savez_compressed(OUT/'best_predictions.npz',row_id=bestpack['row_id'],predicted_sgd=bestpack['predicted_sgd'],actual_sgd=bestpack['actual_sgd'])
        np.savez_compressed(OUT/'baseline_predictions.npz',row_id=bestpack['row_id'],predicted_sgd=bestpack['baseline_predictions'],actual_sgd=bestpack['actual_sgd'])
        atomic_json(OUT/'curve.json',curve)
        if bad>=CONFIG['patience_full_validation_epochs']: break
    atomic_json(OUT/'status.json',{'status':'training_complete','full_epochs_completed':epoch,'best_rmse_sgd':best,'final_verification':'pending'})

if __name__=='__main__':
    try: main()
    except BaseException as exc:
        OUT.mkdir(parents=True,exist_ok=True); atomic_json(OUT/'status.json',{'status':'FAILED','error':repr(exc),'traceback':traceback.format_exc(),'final_verification':'pending'}); raise
