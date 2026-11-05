"""Chronological raw-data LoRA selection with warmup; no final refit.

Caches are keyed by frozen-prefix inputs and phase; LR and adapter weights do not
enter the prefix. Only the final ICL block (FP32 q/v LoRA) and head train.
"""
import hashlib,json,os,time,traceback
from pathlib import Path
import numpy as np
import torch
from experiments.tabfm_finetune_v13 import runner
from experiments.training_schedule_v15 import WarmupCosine
from experiments.tabfm_finetune_v13.core import prepare_lora_head,frozen_prefix,cached_forward
from experiments.tabfm_finetune_v13.full_runner import cast_cap,raw_parity,save_atomic,sha
from tabfm import tabfm_v1_0_0_pytorch
ROOT=runner.ROOT; OUT=ROOT/'reports/tabfm_warmup_v15'
CONFIG={'seed':42,'stock_features':188,'context':64,'query_batch':32,'rank':4,'learning_rates':[1e-4,3e-5,1e-5],'weight_decay':1e-4,'optimizer':'AdamW','gradient_clip':1.,'max_selection_epochs':40,'early_stopping_patience':5,'validation':'complete chronological raw 2024-04 through 2025-03','final_refit':'not authorized until user compares both models and chooses','first_origin':'zero context allowed; normalized target mean 0 std 1','loss':'query-only normalized residual mean squared loss; equal weight per episode','threads':2,'base_dtype':'bfloat16','last_block_dtype':'float32','checkpoint_steps':10,'schedule':'linear warmup then cosine decay','warmup_epochs':1,'start_ratio':0.01,'min_lr_ratio':0.01}

def build_episodes(fit,pop,indices,seed,episode_fn):
    rng=np.random.default_rng(seed); result=[]
    for origin in np.unique(pop.origin[indices]):
        ids=indices[pop.origin[indices]==origin]
        for j in range(0,len(ids),CONFIG['query_batch']):
            query=ids[j:j+CONFIG['query_batch']]
            available=np.flatnonzero((fit.target_month<=origin)&~np.isin(fit.row_id,pop.row_id[query]))
            count=min(CONFIG['context'],len(available)); context=np.sort(rng.choice(available,count,replace=False)) if count else np.array([],dtype=int)
            result.append(episode_fn(fit,pop,query,context))
    return result

def preprocess(ep):
    x=ep['x'].copy(); count=ep['train_size']; numeric=int((~ep['categorical_mask']).sum())
    if count:
        context=x[:count,:numeric]; mu=context.mean(axis=0,dtype=np.float64); sd=context.std(axis=0,dtype=np.float64); sd[sd<1e-6]=1
    else: mu=np.zeros(numeric); sd=np.ones(numeric)
    x[:,:numeric]=np.clip((x[:,:numeric]-mu)/sd,-10,10)
    if not np.isfinite(x).all(): raise ValueError('nonfinite feature input')
    return x

def inputs(ep):
    x=torch.from_numpy(preprocess(ep))[None]; return x,torch.from_numpy(ep['y'])[None],torch.tensor([ep['train_size']]),torch.from_numpy(ep['categorical_mask'])[None],torch.tensor([x.shape[-1]])

def raw_parity_own(model,ep,cap):
    block=model.icl_predictor.tf_icl.blocks[-1]
    handle=block.register_forward_pre_hook(lambda mod,args:tuple(v.float() if torch.is_tensor(v) and v.is_floating_point() else v for v in args))
    try:
        x,y,n,c,d=inputs(ep)
        with torch.no_grad(): raw=model(x,y,n,cat_mask=c,d=d); cached=cached_forward(model,cast_cap(cap))
        if not torch.equal(raw,cached): raise ValueError('raw/cache parity mismatch')
    finally: handle.remove()

def cache(model,ep,phase,kind,index,prefix_hash):
    digest=hashlib.sha256(prefix_hash.encode())
    for name in ('x','y','categorical_mask','query_row_id','context_row_id'):
        value=np.ascontiguousarray(ep[name]); digest.update(value.tobytes()); digest.update(str(value.shape).encode())
    fingerprint=digest.hexdigest(); path=OUT/'cache'/phase/kind/f'{index:05}.pt'
    if path.exists():
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state['fingerprint']!=fingerprint: raise ValueError('prefix cache identity mismatch')
        return state['capture']
    x,y,n,c,d=inputs(ep); capture=frozen_prefix(model,x,y,n,c,d)
    path.parent.mkdir(parents=True,exist_ok=True); save_atomic(path,{'fingerprint':fingerprint,'capture':capture}); return capture

def model_init():
    torch.manual_seed(CONFIG['seed']); m=tabfm_v1_0_0_pytorch.load(model_type='regression',checkpoint_path=str(ROOT/'reports/tabfm_v10/weights/regression'),device='cpu',dtype=torch.bfloat16,use_cache=False)
    prepare_lora_head(m,CONFIG['rank']); m.icl_predictor.tf_icl.blocks[-1].float(); return m

def setup(phase):
    from experiments.tabfm_finetune_v13.data_full import prepare_phase,episode
    fit,hold,meta,state=prepare_phase(phase,feature_bank='stock')
    expected=114842 if phase=='selection' else 150000
    if len(fit.row_id)!=expected or (phase=='selection' and len(hold.row_id)!=35158): raise ValueError('raw split count mismatch')
    train=build_episodes(fit,fit,np.arange(len(fit.row_id)),43,episode)
    valid=build_episodes(fit,hold,np.arange(len(hold.row_id)),44,episode) if hold is not None else []
    for eps,pop in ((train,fit),(valid,hold)):
        if pop is None: continue
        lookup={int(k):float(v) for k,v in zip(pop.row_id,pop.actual_sgd)}
        for ep in eps: ep['query_actual_sgd']=np.array([lookup[int(k)] for k in ep['query_row_id']])
        ids=np.concatenate([e['query_row_id'] for e in eps]); assert len(ids)==len(pop.row_id) and len(np.unique(ids))==len(ids)
    OUT.mkdir(parents=True,exist_ok=True); out=OUT/phase; out.mkdir(parents=True,exist_ok=True)
    # Metadata includes scaler state, source identities and feature order; omit LR.
    weight=ROOT/'reports/tabfm_v10/weights/regression/model.safetensors'; weight_sha=sha(weight)
    if weight_sha!=json.loads((ROOT/'reports/tabfm_finetune_v13/root_base_weight.json').read_text())['sha256']: raise ValueError('original checkpoint SHA mismatch')
    source=['experiments/tabfm_warmup_v15/pipeline.py','experiments/training_schedule_v15.py','experiments/tabfm_finetune_v13/data_full.py','experiments/tabfm_finetune_v13/core.py','reports/tabfm_v10/runtime/source/tabfm/src/pytorch/model.py']
    reg={'phase':phase,'config':CONFIG,'metadata':meta,'source_sha256':{p:sha(ROOT/p) for p in source},'weight_sha256':weight_sha,'torch':str(torch.__version__),'numpy':np.__version__,'train_ids_sha256':hashlib.sha256(fit.row_id.tobytes()).hexdigest(),'validation_ids_sha256':hashlib.sha256(hold.row_id.tobytes()).hexdigest() if hold is not None else None}
    fingerprint=hashlib.sha256(json.dumps(reg,sort_keys=True,default=str).encode()).hexdigest(); path=out/'registration.json'
    if path.exists() and json.loads(path.read_text())['fingerprint']!=fingerprint: raise ValueError('phase fingerprint differs')
    if not path.exists(): runner.atomic_json(path,{'fingerprint':fingerprint,'registration':reg})
    state_path=out/'preprocessing_state.pt'
    if not state_path.exists(): save_atomic(state_path,state)
    sel=out/'selection.npz'
    def packed(eps):
        lengths=np.array([len(e['context_row_id']) for e in eps],dtype=np.int64)
        values=np.concatenate([e['context_row_id'] for e in eps]) if eps else np.array([],dtype=np.int64)
        return values,np.concatenate(([0],np.cumsum(lengths)))
    if not sel.exists():
        tc,to=packed(train); vc,vo=packed(valid)
        np.savez_compressed(sel,train_row_id=fit.row_id,train_origin=fit.origin,train_horizon=fit.horizon,validation_row_id=hold.row_id if hold is not None else np.array([],dtype=int),validation_origin=hold.origin if hold is not None else np.array([],dtype=int),validation_horizon=hold.horizon if hold is not None else np.array([],dtype=int),train_context=tc,train_context_offsets=to,validation_context=vc,validation_context_offsets=vo)
    return train,valid,meta,fingerprint

def predict(model,params,initial,episodes,phase,fp):
    current={n:p.detach().clone() for n,p in params.items()}; pred=[]; base=[]; actual=[]; ids=[]
    try:
        for i,ep in enumerate(episodes):
            c=cast_cap(cache(model,ep,phase,'validation',i,fp))
            with torch.no_grad():
                a=cached_forward(model,c)[0,ep['train_size']:,0].float().numpy()
                for n,p in params.items(): p.copy_(initial[n])
                b=cached_forward(model,c)[0,ep['train_size']:,0].float().numpy()
                for n,p in params.items(): p.copy_(current[n])
            pred.extend(((a*ep['target_std']+ep['target_mean']+ep['anchor_ksgd'])*1000).tolist()); base.extend(((b*ep['target_std']+ep['target_mean']+ep['anchor_ksgd'])*1000).tolist()); actual.extend(ep['query_actual_sgd'].tolist()); ids.extend(ep['query_row_id'].tolist())
            if i%10==0: runner.atomic_json(OUT/'progress.json',{'stage':'validation','phase':phase,'episodes_completed':i+1,'episodes_total':len(episodes)})
    finally:
        with torch.no_grad():
            for n,p in params.items(): p.copy_(current[n])
    return tuple(np.asarray(x) for x in (pred,base,actual,ids))

def rmse(pred,actual): return float(np.sqrt(np.mean((pred-actual)**2)))

def pause_after_step(limit, completed, checkpoint, epoch, next_step):
    if limit is None or completed < limit: return False
    checkpoint(epoch,next_step)
    return True

def validate_schedule_position(state,epoch,next_step,steps_per_epoch):
    if state['updates_completed'] != (epoch-1)*steps_per_epoch+next_step:
        raise ValueError('schedule position does not match checkpoint epoch and step')

def make_schedule(optimizer,steps_per_epoch):
    return WarmupCosine(optimizer,total_steps=steps_per_epoch*CONFIG['max_selection_epochs'],warmup_steps=steps_per_epoch,min_lr_ratio=CONFIG['min_lr_ratio'],start_ratio=CONFIG['start_ratio'])

def train_run(phase,name,lr,train,valid,fp,max_epochs,early_stop,max_new_steps=None):
    out=OUT/phase/name; out.mkdir(parents=True,exist_ok=True)
    complete=out/'status.json'
    if complete.exists() and json.loads(complete.read_text()).get('status')=='training_complete': return json.loads(complete.read_text())
    m=model_init(); params={n:p for n,p in m.named_parameters() if p.requires_grad}; optim=torch.optim.AdamW(params.values(),lr=lr,weight_decay=CONFIG['weight_decay'])
    schedule=make_schedule(optim,len(train))
    initial={n:p.detach().clone() for n,p in params.items()}; epoch=1; start=0; best=float('inf'); bad=0; bestpack=None; curve=[]
    resume=out/'resume.pt'
    if resume.exists():
        s=torch.load(resume,map_location='cpu',weights_only=False)
        if s['fingerprint']!=fp or s['lr']!=lr: raise ValueError('resume fingerprint or learning rate mismatch')
        with torch.no_grad():
            for n,p in params.items(): p.copy_(s['params'][n])
        validate_schedule_position(s['scheduler'],s['epoch'],s['next_step'],len(train))
        optim.load_state_dict(s['optimizer']); schedule.load_state_dict(s['scheduler']); initial=s['initial']; epoch=s['epoch']; start=s['next_step']; best=s['best']; bad=s['bad']; curve=s['curve']; bestpack=s['bestpack']; torch.set_rng_state(s['rng'])
    def commit(ep,step):
        validate_schedule_position(schedule.state_dict(),ep,step,len(train))
        save_atomic(resume,{'fingerprint':fp,'lr':lr,'params':{n:p.detach().clone() for n,p in params.items()},'optimizer':optim.state_dict(),'scheduler':schedule.state_dict(),'initial':initial,'epoch':ep,'next_step':step,'best':best,'bad':bad,'curve':curve,'bestpack':bestpack,'rng':torch.get_rng_state()})
    def export():
        if bestpack is not None:
            save_atomic(out/'best_parameters.pt',bestpack['params'])
            np.savez_compressed(out/'best_predictions.npz',row_id=bestpack['row_id'],predicted_sgd=bestpack['prediction'],actual_sgd=bestpack['actual'],baseline_sgd=bestpack['baseline'])
            runner.atomic_json(out/'curve.json',curve)
    if epoch>max_epochs or (early_stop and bad>=CONFIG['early_stopping_patience']):
        if not valid: save_atomic(out/'final_parameters.pt',{n:p.detach().clone() for n,p in params.items()})
        export(); result={'status':'training_complete','epochs_completed':epoch-1,'best_epoch':bestpack['epoch'] if bestpack else None,'best_rmse_sgd':best if bestpack else None,'final_verification':'pending'}; runner.atomic_json(complete,result); return result
    commit(epoch,start); new_steps=0
    for epoch in range(epoch,max_epochs+1):
        order=np.random.default_rng(CONFIG['seed']+epoch).permutation(len(train))
        for step in range(start,len(order)):
            index=int(order[step]); ep=train[index]; tick=time.perf_counter(); cap=cache(m,ep,phase,'train',index,fp)
            if epoch==1 and step==0: raw_parity_own(m,ep,cap)
            optim.zero_grad(set_to_none=True); outp=cached_forward(m,cast_cap(cap))[0,ep['train_size']:,0].float(); target=torch.from_numpy((ep['query_targets']-ep['target_mean'])/ep['target_std']).float(); loss=(outp-target).square().mean()
            if not torch.isfinite(loss): raise ValueError('nonfinite query loss')
            loss.backward(); gradnorm=torch.nn.utils.clip_grad_norm_(params.values(),CONFIG['gradient_clip'],error_if_nonfinite=True); used_lr=optim.param_groups[0]['lr']; optim.step(); schedule.step()
            if epoch==1 and step==0: raw_parity_own(m,ep,cap)
            new_steps+=1
            if pause_after_step(max_new_steps,new_steps,commit,epoch,step+1):
                runner.atomic_json(OUT/'progress.json',{'phase':phase,'run':name,'stage':'paused_for_queue','epoch':epoch,'next_step':step+1,'episodes':len(order),'new_optimizer_steps':new_steps,'used_learning_rate':used_lr,'gradient_norm':float(gradnorm)})
                return {'status':'paused_for_queue','epoch':epoch,'next_step':step+1,'new_optimizer_steps':new_steps,'used_learning_rate':used_lr,'gradient_norm':float(gradnorm)}
            if step%CONFIG['checkpoint_steps']==0 or step==len(order)-1: commit(epoch,step+1)
            if step%5==0: runner.atomic_json(OUT/'progress.json',{'phase':phase,'run':name,'stage':'training','epoch':epoch,'step':step+1,'episodes':len(order),'queries_in_episode':len(ep['query_row_id']),'last_seconds':time.perf_counter()-tick,'loss':float(loss),'used_learning_rate':used_lr,'gradient_norm':float(gradnorm)})
            print(phase,name,'epoch',epoch,'step',step+1,'/',len(order),'seconds',time.perf_counter()-tick,'lr',used_lr,'gradient_norm',float(gradnorm),flush=True)
        start=0
        if valid:
            p,b,a,ids=predict(m,params,initial,valid,phase,fp)
            if len(ids)!=35158 or len(np.unique(ids))!=35158: raise ValueError('validation coverage mismatch')
            if not curve:
                best=rmse(b,a); bestpack={'epoch':0,'params':initial,'row_id':ids,'prediction':b,'actual':a,'baseline':b}; curve.append({'epoch':0,'rmse_sgd':best})
            score=rmse(p,a); curve.append({'epoch':epoch,'rmse_sgd':score})
            if score<best: best=score; bad=0; bestpack={'epoch':epoch,'params':{n:p.detach().clone() for n,p in params.items()},'row_id':ids,'prediction':p,'actual':a,'baseline':b}
            else: bad+=1
            print('validation',phase,name,curve[-1],flush=True)
        commit(epoch+1,0); export()
        if early_stop and bad>=CONFIG['early_stopping_patience']: break
    if not valid: save_atomic(out/'final_parameters.pt',{n:p.detach().clone() for n,p in params.items()})
    result={'status':'training_complete','epochs_completed':epoch,'best_epoch':bestpack['epoch'] if bestpack else None,'best_rmse_sgd':best if bestpack else None,'final_verification':'pending'}; runner.atomic_json(complete,result); return result

def main(max_new_steps=None):
    torch.set_num_threads(CONFIG['threads']); OUT.mkdir(parents=True,exist_ok=True)
    train,valid,meta,fp=setup('selection'); results=[]
    for lr in CONFIG['learning_rates']:
        name=f'lr_{lr:.0e}'; runner.atomic_json(OUT/'status.json',{'status':'running','phase':'selection','run':name,'final_verification':'pending'})
        result=train_run('selection',name,lr,train,valid,fp,CONFIG['max_selection_epochs'],True,max_new_steps=max_new_steps)
        if result['status']=='paused_for_queue':
            runner.atomic_json(OUT/'status.json',{'status':'paused_for_queue','phase':'selection','run':name,**result,'final_verification':'pending'}); return
        results.append({'lr':lr,**result})
    chosen=min(results,key=lambda r:r['best_rmse_sgd']); runner.atomic_json(OUT/'selection_result.json',{'selected_lr':chosen['lr'],'selected_epoch':chosen['best_epoch'],'validation_rmse_sgd':chosen['best_rmse_sgd'],'all_candidates':results,'population':'35158 chronological validation rows','pretrained_epoch_zero_eligible':True})
    runner.atomic_json(OUT/'status.json',{'status':'training_complete','phase':'selection','final_verification':'pending','note':'Final refit awaits user decision after comparing both models.'})

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--selection-only',action='store_true',help='Run the authorized chronological selection stage (default).')
    parser.add_argument('--max-new-steps',type=int,default=None,help='Pause after this many newly executed optimizer steps; resume without this option.')
    args=parser.parse_args()
    if args.max_new_steps is not None and args.max_new_steps<1: parser.error('--max-new-steps must be positive')
    try: main(args.max_new_steps)
    except BaseException as e:
        OUT.mkdir(parents=True,exist_ok=True); runner.atomic_json(OUT/'status.json',{'status':'FAILED','error':repr(e),'traceback':traceback.format_exc(),'final_verification':'pending'}); raise
