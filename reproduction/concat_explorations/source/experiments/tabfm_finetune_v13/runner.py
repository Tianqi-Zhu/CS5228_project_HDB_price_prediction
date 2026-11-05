"""Registered fixed-subset, frozen-prefix LoRA screen; local CPU only."""
import hashlib, json, os, sys, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
for sub in ('dependencies','source'): sys.path.insert(0,str(ROOT/'reports/tabfm_v10/runtime'/sub))
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from tabfm import tabfm_v1_0_0_pytorch
from experiments.tabfm_finetune_v13.core import prepare_lora_head, frozen_prefix, cached_forward
from experiments.tabfm_finetune_v13.data import load_banks, episode

OUT=ROOT/'reports/tabfm_finetune_v13/screen'
CONFIG={'seed':42,'train_queries':1024,'monitor_queries':512,'context':64,'query_batch':32,'rank':4,'lora':'last ICL block q and v','optimizer':'AdamW','lr':0.0001,'weight_decay':0.0001,'clip':1.,'max_subset_epochs':40,'patience':5,'loss':'mean squared normalized residual, context-only mean/std','selection':'even allocation across eligible origins, random within origin','baseline':'epoch 0 pretrained head and zero LoRA','numeric_preprocessing':'context-only mean/std, std floor 1e-6, clip [-10,10]','categories':'raw fixed integer codes','banks':['base','macro','stock'],'validation':'fixed v10 512 row monitor, not full holdout','threads':2}

def atomic_json(path, obj):
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj,indent=2,default=str)+'\n'); os.replace(tmp,path)

def preprocess(ep):
    x=ep['x'].copy(); n=int((~ep['categorical_mask']).sum()); ctx=x[:ep['train_size'],:n]
    mu=ctx.mean(axis=0, dtype=np.float64); sd=ctx.std(axis=0, dtype=np.float64); sd[sd<1e-6]=1
    x[:,:n]=np.clip((x[:,:n]-mu)/sd,-10,10)
    assert np.isfinite(x).all()
    return x

def select(fit, meta):
    rng=np.random.default_rng(CONFIG['seed']); eligible=np.flatnonzero(fit.origin>np.min(fit.origin)); origins=np.unique(fit.origin[eligible]); chosen=[]
    for i,origin in enumerate(origins):
        ids=eligible[fit.origin[eligible]==origin]; n=CONFIG['train_queries']//len(origins)+(i<CONFIG['train_queries']%len(origins)); chosen.extend(rng.choice(ids,size=min(len(ids),n),replace=False).tolist())
    hold_ids=np.asarray(meta['fixed_validation_row_id']); return np.array(chosen), hold_ids

def episodes(fit, pop, indices, seed):
    rng=np.random.default_rng(seed); results=[]
    for origin in np.unique(pop.origin[indices]):
        idx=indices[pop.origin[indices]==origin]
        for start in range(0,len(idx),CONFIG['query_batch']):
            query=idx[start:start+CONFIG['query_batch']]
            candidates=np.flatnonzero((fit.target_month<=origin) & ~np.isin(fit.row_id,pop.row_id[query]))
            if len(candidates)<CONFIG['context']: raise ValueError('insufficient legal context')
            context=np.sort(rng.choice(candidates,CONFIG['context'],replace=False))
            results.append(episode(fit,pop,query,context))
    return results

def materialize(model, eps, out, stage, fingerprint):
    result=[]
    directory=out/'prefix_cache'/stage; directory.mkdir(parents=True,exist_ok=True)
    for index,ep in enumerate(eps):
        key=hashlib.sha256(fingerprint.encode())
        for field in ('x','y','categorical_mask','query_row_id','context_row_id'):
            a=np.ascontiguousarray(ep[field]); key.update(a.tobytes()); key.update(str(a.shape).encode())
        digest=key.hexdigest(); path=directory/f'{index:04}.pt'; started=time.perf_counter()
        if path.exists():
            saved=torch.load(path,map_location='cpu',weights_only=False)
            if saved['fingerprint']!=digest: raise ValueError(f'cache fingerprint mismatch: {path}')
            cap=saved['capture']; action='reused'
        else:
            x=torch.from_numpy(preprocess(ep))[None]; y=torch.from_numpy(ep['y'])[None]
            mask=torch.from_numpy(ep['categorical_mask'])[None]; ts=torch.tensor([ep['train_size']]); d=torch.tensor([x.shape[-1]])
            cap=frozen_prefix(model,x,y,ts,mask,d)
            tmp=path.with_suffix('.tmp'); torch.save({'fingerprint':digest,'capture':cap},tmp); os.replace(tmp,path); action='created'
        result.append((cap,ep))
        atomic_json(out/'progress.json',{'stage':'cache_'+stage,'episodes_completed':index+1,'episodes_total':len(eps),'last_action':action,'last_seconds':time.perf_counter()-started})
        print(stage,index+1,'/',len(eps),action,flush=True)
    return result

def predict(model, cached):
    predictions=[]; actual=[]; ids=[]
    with torch.no_grad():
        for cap,ep in cached:
            p=cached_forward(model,cap)[0,ep['train_size']:,0].float().numpy()
            predictions.extend(((p*ep['target_std']+ep['target_mean']+ep['anchor_ksgd'])*1000).tolist())
            actual.extend(ep['query_actual_sgd'].tolist()); ids.extend(ep['query_row_id'].tolist())
    p=np.array(predictions); a=np.array(actual)
    return {'rmse_sgd':float(np.sqrt(np.mean((p-a)**2))),'mae_sgd':float(np.mean(abs(p-a))),'bias_sgd':float(np.mean(p-a)),'n':len(p)},p,a,np.array(ids)

def run_bank(name):
    out=OUT/name
    if (out/'status.json').exists():
        print(name, 'already complete', flush=True); return
    fit,hold,metadata=load_banks(expanded=name!='base',stock=name=='stock')
    train_idx,hold_ids=select(fit,metadata); lookup={int(v):i for i,v in enumerate(hold.row_id)}; val_idx=np.array([lookup[int(i)] for i in hold_ids])
    assert len(train_idx)==1024 and len(val_idx)==512
    train_eps=episodes(fit,fit,train_idx,43); val_eps=episodes(fit,hold,val_idx,44)
    for eps,pop in ((train_eps,fit),(val_eps,hold)):
        actual={int(k):float(v) for k,v in zip(pop.row_id,pop.actual_sgd)}
        for ep in eps: ep['query_actual_sgd']=np.array([actual[int(k)] for k in ep['query_row_id']],dtype=np.float64)
    out.mkdir(parents=True,exist_ok=True)
    metapath=out/'data_metadata.json'
    if metapath.exists() and json.loads(metapath.read_text())!=metadata: raise ValueError('resume data metadata mismatch')
    if not metapath.exists(): atomic_json(metapath,metadata)
    np.savez_compressed(out/'selection.npz',train_row_id=fit.row_id[train_idx],validation_row_id=hold.row_id[val_idx],train_context=np.stack([e['context_row_id'] for e in train_eps]),validation_context=np.stack([e['context_row_id'] for e in val_eps]))
    torch.manual_seed(42); model=tabfm_v1_0_0_pytorch.load(model_type='regression',checkpoint_path=str(ROOT/'reports/tabfm_v10/weights/regression'),device='cpu',dtype=torch.bfloat16,use_cache=False); prepare_lora_head(model,4)
    fingerprint=hashlib.sha256((OUT/'registration.json').read_bytes()+json.dumps(metadata,sort_keys=True).encode()).hexdigest()
    t=time.perf_counter(); train_cache=materialize(model,train_eps,out,'train',fingerprint); val_cache=materialize(model,val_eps,out,'validation',fingerprint)
    atomic_json(out/'cache_timing.json',{'seconds':time.perf_counter()-t,'train_episodes':len(train_cache),'validation_episodes':len(val_cache)})
    params={n:p for n,p in model.named_parameters() if p.requires_grad}; opt=torch.optim.AdamW(params.values(),lr=CONFIG['lr'],weight_decay=CONFIG['weight_decay'])
    baseline,p,a,ids=predict(model,val_cache); best=baseline['rmse_sgd']; bad=0; curve=[{'epoch':0,**baseline}]
    if not (out/'baseline_predictions.npz').exists(): np.savez_compressed(out/'baseline_predictions.npz',row_id=ids,predicted_sgd=p,actual_sgd=a)
    if not (out/'best_parameters.pt').exists():
        torch.save({n:p.detach().clone() for n,p in params.items()},out/'best_parameters.pt')
        np.savez_compressed(out/'best_predictions.npz',row_id=ids,predicted_sgd=p,actual_sgd=a)
    def save(epoch):
        state={'epoch':epoch,'parameters':{n:p.detach().clone() for n,p in params.items()},'optimizer':opt.state_dict(),'torch_rng':torch.get_rng_state(),'best_rmse_sgd':best,'bad_epochs':bad,'curve':curve,'config':CONFIG}
        torch.save(state,out/'resume.tmp'); os.replace(out/'resume.tmp',out/'resume.pt')
    first_epoch=1
    if (out/'resume.pt').exists():
        state=torch.load(out/'resume.pt',map_location='cpu',weights_only=False)
        if state['config']!=CONFIG: raise ValueError('resume config differs')
        if set(state['parameters'])!=set(params): raise ValueError('resume parameter schema differs')
        with torch.no_grad():
            for n,p in params.items(): p.copy_(state['parameters'][n])
        opt.load_state_dict(state['optimizer']); torch.set_rng_state(state['torch_rng'])
        best=state['best_rmse_sgd']; bad=state['bad_epochs']; curve=state['curve']; first_epoch=state['epoch']+1
    else:
        save(0); atomic_json(out/'curve.json',curve)
    epoch=first_epoch-1
    if bad>=CONFIG['patience']: first_epoch=CONFIG['max_subset_epochs']+1
    for epoch in range(first_epoch,CONFIG['max_subset_epochs']+1):
        start=time.perf_counter(); losses=[]
        rng=np.random.default_rng(CONFIG['seed']+epoch)
        for step,i in enumerate(rng.permutation(len(train_cache))):
            cap,ep=train_cache[i]; step_start=time.perf_counter()
            opt.zero_grad(set_to_none=True); output=cached_forward(model,cap)[0,ep['train_size']:,0].float()
            target=torch.from_numpy((ep['query_targets']-ep['target_mean'])/ep['target_std']).float(); loss=(output-target).square().mean(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params.values(),CONFIG['clip'],error_if_nonfinite=True); opt.step(); losses.append(loss.item())
            if step < 4: print('representative_step',epoch,step,'rows',len(ep['query_row_id']),'seconds',time.perf_counter()-step_start,flush=True)
            if step%5==0: atomic_json(out/'progress.json',{'stage':'training','epoch':epoch,'step':step+1,'steps_total':len(train_cache)})
        metrics,p,a,ids=predict(model,val_cache); curve.append({'epoch':epoch,'train_loss_mean':float(np.mean(losses)),'seconds':time.perf_counter()-start,**metrics})
        if metrics['rmse_sgd']<best:
            best=metrics['rmse_sgd']; bad=0; torch.save({n:p.detach().clone() for n,p in params.items()},out/'best_parameters.pt'); np.savez_compressed(out/'best_predictions.npz',row_id=ids,predicted_sgd=p,actual_sgd=a)
        else: bad+=1
        save(epoch); atomic_json(out/'curve.json',curve)
        print(name,curve[-1],flush=True)
        if bad>=CONFIG['patience']: break
    atomic_json(out/'status.json',{'status':'training_complete','best_rmse_sgd':best,'pretrained_selected':best==baseline['rmse_sgd'],'epochs_completed':epoch,'monitor_n':512,'final_verification':'pending'})

if __name__=='__main__':
    torch.set_num_threads(2); OUT.mkdir(parents=True,exist_ok=True)
    weight=ROOT/'reports/tabfm_v10/weights/regression/model.safetensors'
    with weight.open('rb') as f: header_digest=hashlib.sha256(f.read(1048576)).hexdigest()
    registration={'config':CONFIG,'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'core_sha256':hashlib.sha256((ROOT/'experiments/tabfm_finetune_v13/core.py').read_bytes()).hexdigest(),'data_sha256':hashlib.sha256((ROOT/'experiments/tabfm_finetune_v13/data.py').read_bytes()).hexdigest(),'runtime_model_sha256':hashlib.sha256((ROOT/'reports/tabfm_v10/runtime/source/tabfm/src/pytorch/model.py').read_bytes()).hexdigest(),'weights_size':weight.stat().st_size,'weights_first_mib_sha256':header_digest}
    source_paths=['experiments/tabfm_v10/benchmark.py','experiments/macro_encoders_v3/contract.py','experiments/macro_encoders_v3/data.py','experiments/macro_encoders_v3/scaling.py','reports/tabfm_v10/runtime/source/tabfm/src/pytorch/tabfm_v1_0_0.py']
    registration['supporting_source_sha256']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_paths}
    register=OUT/'registration.json'
    if register.exists() and json.loads(register.read_text())!=registration: raise ValueError('registered code or configuration changed')
    if not register.exists(): atomic_json(register,registration)
    try:
        for name in CONFIG['banks']:
            atomic_json(OUT/'queue_status.json',{'status':'running','bank':name,'final_verification':'pending'})
            run_bank(name)
        atomic_json(OUT/'queue_status.json',{'status':'training_complete','final_verification':'pending'})
    except BaseException as exc:
        atomic_json(OUT/'queue_status.json',{'status':'FAILED','error':repr(exc),'traceback':traceback.format_exc(),'final_verification':'pending'})
        raise
