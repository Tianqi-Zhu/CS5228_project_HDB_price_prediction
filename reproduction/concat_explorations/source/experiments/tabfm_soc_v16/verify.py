"""Post-queue verification of raw chronological LoRA selection; no refit.

Rebuilds only data and the selected final-block predictions using existing
fingerprinted validation caches. An absent or mismatched cache fails closed.
"""
import argparse, hashlib, json, time, traceback
from pathlib import Path
import numpy as np
import torch
from experiments.tabfm_soc_v16 import pipeline as pipe
from experiments.tabfm_soc_v16.runtime import resolve_device,run_lock
from experiments.tabfm_finetune_v13.data_full import prepare_phase, episode
from experiments.tabfm_finetune_v13.core import cached_forward
cast_cap=pipe.cast_cap
file_sha=pipe.sha

OUT=pipe.OUT

def metrics(pred,actual):
    pred=np.asarray(pred,dtype=float); actual=np.asarray(actual,dtype=float)
    if pred.shape!=actual.shape or not np.isfinite(pred).all() or not np.isfinite(actual).all() or not len(pred): raise ValueError('invalid prediction or actual array')
    err=pred-actual
    return {'n':len(err),'rmse_sgd':float(np.sqrt(np.mean(err**2))),'mae_sgd':float(np.mean(abs(err))),'bias_sgd':float(np.mean(err))}

def check_identity(ids,actual,hold):
    ids=np.asarray(ids)
    if len(ids)!=len(hold.row_id) or len(np.unique(ids))!=len(ids): raise ValueError('validation row count or uniqueness differs')
    lookup={int(k):i for i,k in enumerate(hold.row_id)}
    try: idx=np.array([lookup[int(k)] for k in ids])
    except KeyError as exc: raise ValueError('unknown validation row id') from exc
    if not np.array_equal(np.asarray(actual),hold.actual_sgd[idx].astype(float)): raise ValueError('validation actual labels differ')
    return idx

def verify_sources(reg):
    for name,digest in reg['source_sha256'].items():
        if file_sha(pipe.ROOT/name)!=digest: raise ValueError('registered source changed: '+name)
    weight=pipe.CHECKPOINT/'model.safetensors'
    if file_sha(weight)!=reg['weight_sha256']: raise ValueError('original checkpoint changed')
    if file_sha(pipe.CHECKPOINT/'config.json')!=reg['checkpoint_config_sha256']: raise ValueError('checkpoint configuration changed')
    manifest=pipe.ROOT/'DEPLOY_MANIFEST.json'
    if (file_sha(manifest) if manifest.is_file() else None)!=reg['deployment_manifest_sha256']: raise ValueError('deployment manifest changed')
    if reg['device']!=pipe.device_identity(pipe.DEVICE): raise ValueError('verification device differs from training')

def run():
    torch.set_num_threads(pipe.CONFIG['threads']); status=json.loads((OUT/'status.json').read_text())
    if status.get('status')!='training_complete' or status.get('phase')!='selection': raise ValueError('selection queue not complete')
    registration=json.loads((OUT/'selection/registration.json').read_text()); reg=registration['registration']; fingerprint=registration['fingerprint']
    calc=hashlib.sha256(json.dumps(reg,sort_keys=True,default=str).encode()).hexdigest()
    if fingerprint!=calc: raise ValueError('registration fingerprint mismatch')
    verify_sources(reg)
    fit,hold,metadata,state=prepare_phase('selection',feature_bank='stock')
    if json.dumps(metadata,sort_keys=True,default=str)!=json.dumps(reg['metadata'],sort_keys=True,default=str) or len(fit.row_id)!=114842 or len(hold.row_id)!=35158: raise ValueError('data population/metadata mismatch')
    saved=torch.load(OUT/'selection/preprocessing_state.pt',map_location='cpu',weights_only=False)
    if json.dumps(saved,sort_keys=True,default=str)!=json.dumps(state,sort_keys=True,default=str): raise ValueError('feature state mismatch')
    selection=np.load(OUT/'selection/selection.npz',allow_pickle=False)
    if not np.array_equal(selection['train_row_id'],fit.row_id) or not np.array_equal(selection['validation_row_id'],hold.row_id): raise ValueError('saved population IDs mismatch')
    valid=pipe.build_episodes(fit,hold,np.arange(len(hold.row_id)),44,episode)
    offsets=selection['validation_context_offsets']; contexts=selection['validation_context']
    if len(offsets)!=len(valid)+1: raise ValueError('context episode count mismatch')
    for i,ep in enumerate(valid):
        if not np.array_equal(ep['context_row_id'],contexts[offsets[i]:offsets[i+1]]): raise ValueError('context identity mismatch')
    actual_by_id={int(k):float(v) for k,v in zip(hold.row_id,hold.actual_sgd)}
    for ep in valid: ep['query_actual_sgd']=np.array([actual_by_id[int(k)] for k in ep['query_row_id']])
    summaries=[]; baseline_reference=None; chosen_result=json.loads((OUT/'selection_result.json').read_text())
    for lr in pipe.CONFIG['learning_rates']:
        directory=OUT/'selection'/f'lr_{lr:.0e}'; candidate=json.loads((directory/'status.json').read_text())
        if candidate['status']!='training_complete': raise ValueError('candidate incomplete')
        state=torch.load(directory/'resume.pt',map_location='cpu',weights_only=False); pack=state['bestpack']
        pipe.validate_schedule_position(state['scheduler'],state['epoch'],state['next_step'],len(selection['train_context_offsets'])-1)
        if state['fingerprint']!=fingerprint or state['lr']!=lr or pack is None: raise ValueError('candidate state fingerprint mismatch')
        parameter_file=torch.load(directory/'best_parameters.pt',map_location='cpu',weights_only=True)
        if set(parameter_file)!=set(pack['params']) or any(not torch.equal(parameter_file[n],pack['params'][n]) for n in parameter_file): raise ValueError('best parameter artifact differs from atomic checkpoint')
        values=np.load(directory/'best_predictions.npz',allow_pickle=False)
        for name,key in [('row_id','row_id'),('predicted_sgd','prediction'),('actual_sgd','actual'),('baseline_sgd','baseline')]:
            if not np.array_equal(values[name],pack[key]): raise ValueError('candidate prediction differs from atomic best state')
        ids=values['row_id']; idx=check_identity(ids,values['actual_sgd'],hold)
        scores=metrics(values['predicted_sgd'],values['actual_sgd']); baseline=metrics(values['baseline_sgd'],values['actual_sgd'])
        minimum=min(item['rmse_sgd'] for item in state['curve'])
        if not np.isclose(scores['rmse_sgd'],minimum,rtol=0,atol=1e-9) or not np.isclose(candidate['best_rmse_sgd'],minimum,rtol=0,atol=1e-9): raise ValueError('candidate best score mismatch')
        if pack['epoch']!=candidate['best_epoch'] or not any(x['epoch']==pack['epoch'] and np.isclose(x['rmse_sgd'],minimum,rtol=0,atol=1e-9) for x in state['curve']): raise ValueError('candidate best epoch mismatch')
        if baseline_reference is None: baseline_reference=(ids.copy(),values['baseline_sgd'].copy())
        elif not np.array_equal(ids,baseline_reference[0]) or not np.array_equal(values['baseline_sgd'],baseline_reference[1]): raise ValueError('baseline pairing mismatch')
        groups={str(h):metrics(values['predicted_sgd'][hold.horizon[idx]==h],values['actual_sgd'][hold.horizon[idx]==h]) for h in np.unique(hold.horizon[idx])}
        summaries.append({'lr':lr,'best_epoch':pack['epoch'],'metrics':scores,'baseline':baseline,'horizon_groups':groups})
    chosen=min(summaries,key=lambda x:x['metrics']['rmse_sgd'])
    if chosen_result['selected_lr']!=chosen['lr'] or chosen_result['selected_epoch']!=chosen['best_epoch'] or not np.isclose(chosen_result['validation_rmse_sgd'],chosen['metrics']['rmse_sgd'],rtol=0,atol=1e-9): raise ValueError('selected candidate mismatch')
    directory=OUT/'selection'/f"lr_{chosen['lr']:.0e}"; saved=torch.load(directory/'resume.pt',map_location='cpu',weights_only=False); best=saved['bestpack']; model=pipe.model_init(); parameters={n:p for n,p in model.named_parameters() if p.requires_grad}
    if set(parameters)!=set(best['params']): raise ValueError('selected parameter schema mismatch')
    with torch.no_grad():
        for n,p in parameters.items(): p.copy_(best['params'][n])
    predictions=[]; identifiers=[]
    for i,ep in enumerate(valid):
        cache_path=OUT/'cache'/'selection'/'validation'/f'{i:05}.pt'
        if not cache_path.is_file(): raise ValueError('missing validation cache: '+str(cache_path))
        cap=pipe.cache(model,ep,'selection','validation',i,fingerprint)
        if i==0: pipe.raw_parity_own(model,ep,cap)
        with torch.no_grad(): value=cached_forward(model,cast_cap(cap))[0,ep['train_size']:,0].float().cpu().numpy()
        predictions.extend(((value*ep['target_std']+ep['target_mean']+ep['anchor_ksgd'])*1000).tolist()); identifiers.extend(ep['query_row_id'].tolist())
    if not np.array_equal(np.asarray(identifiers),best['row_id']) or not np.array_equal(np.asarray(predictions),best['prediction']): raise ValueError('selected full cached replay mismatch')
    report={'scope':'raw chronological selection: 114842 training rows, 35158 validation rows; no final refit','selected':chosen,'candidates':summaries,'paired_pretrained_baseline':summaries[0]['baseline'],'verification':{'selected_full_cached_replay':'exact','selected_raw_episode_replay':'exact','data_context_source_and_original_checkpoint_guards':'passed'},'final_verification':'passed'}
    pipe.runner.atomic_json(OUT/'verification_report.json',report)
    status['final_verification']='passed'; pipe.runner.atomic_json(OUT/'status.json',status)
    pipe.runner.atomic_json(OUT/'verifier_status.json',{'status':'passed'})
    print(json.dumps(report,indent=2),flush=True)

def wait_ready():
    while True:
        path=OUT/'status.json'
        state=json.loads(path.read_text()) if path.exists() else {}
        if state.get('status')=='FAILED': raise RuntimeError('selection failed: '+str(state.get('error')))
        if state.get('status')=='training_complete' and state.get('phase')=='selection': return
        time.sleep(30)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--wait',action='store_true'); parser.add_argument('--selection-only',action='store_true'); parser.add_argument('--device',choices=['cpu','cuda'],required=True); parser.add_argument('--output-dir',required=True); parser.add_argument('--checkpoint-dir',required=True); args=parser.parse_args()
    pipe.OUT=Path(args.output_dir).resolve(); pipe.CHECKPOINT=Path(args.checkpoint_dir).resolve(); pipe.DEVICE=resolve_device(args.device); globals()['OUT']=pipe.OUT
    try:
        if args.wait: wait_ready()
        with run_lock(OUT):
            try: run()
            except Exception as exc:
                pipe.runner.atomic_json(OUT/'verifier_status.json',{'status':'FAILED','error':repr(exc),'traceback':traceback.format_exc()}); raise
    except Exception:
        raise
